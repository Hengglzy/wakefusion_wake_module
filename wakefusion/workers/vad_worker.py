"""
VAD (Voice Activity Detection) 工作线程
使用webrtcvad进行语音活动检测
"""

import asyncio
import numpy as np
import time
from typing import Optional, Callable, List
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from wakefusion.types import AudioFrame, EventType, BaseEvent
from wakefusion.logging import get_logger
from wakefusion.metrics import get_metrics, record_latency


logger = get_logger("vad_worker")
metrics = get_metrics()


class SpeechState(str, Enum):
    """语音状态"""
    SILENCE = "SILENCE"
    SPEECH = "SPEECH"


@dataclass
class VADResult:
    """VAD检测结果"""
    state: SpeechState
    ts: float
    confidence: float
    duration_ms: float


class VADWorker:
    """VAD工作线程"""

    def __init__(
        self,
        aggressiveness: int = 2,
        speech_start_ms: int = 120,
        speech_end_ms: int = 500,
        frame_ms: int = 20,
        sample_rate: int = 16000,
        event_callback: Optional[Callable[[BaseEvent], None]] = None
    ):
        """
        初始化VAD工作线程

        Args:
            aggressiveness: VAD激进程度 (0-3, 越高越激进)
            speech_start_ms: 语音起始阈值（毫秒）
            speech_end_ms: 语音结束阈值（毫秒）
            frame_ms: 帧长（毫秒）
            sample_rate: 采样率
            event_callback: 事件回调函数
        """
        self.aggressiveness = aggressiveness
        self.speech_start_ms = speech_start_ms
        self.speech_end_ms = speech_end_ms
        self.frame_ms = frame_ms
        self.sample_rate = sample_rate
        self.event_callback = event_callback

        # webrtcvad实例
        self.vad: Optional[any] = None

        # 状态
        self.is_running = False
        self.current_state = SpeechState.SILENCE
        self.speech_start_ts: Optional[float] = None
        self.speech_end_ts: Optional[float] = None
        self.silence_start_ts: Optional[float] = None

        # 统计
        self.speech_segments = 0
        self.total_frames = 0
        self.speech_frames = 0

        # Session
        self.session_id = self._generate_session_id()

        logger.info(
            "VADWorker initialized",
            extra={
                "aggressiveness": aggressiveness,
                "speech_start_ms": speech_start_ms,
                "speech_end_ms": speech_end_ms
            }
        )

    def _generate_session_id(self) -> str:
        """生成会话ID"""
        return f"vad-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    def start(self):
        """启动VAD工作线程"""
        try:
            import webrtcvad

            self.vad = webrtcvad.Vad(self.aggressiveness)
            self.is_running = True

            logger.info(
                "VADWorker started",
                extra={"aggressiveness": self.aggressiveness}
            )

        except ImportError:
            logger.error("webrtcvad not installed. Please install: pip install webrtcvad")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize VAD: {e}")
            raise

    def stop(self):
        """停止VAD工作线程"""
        self.is_running = False
        logger.info("VADWorker stopped")

    def process_frame(self, frame: AudioFrame) -> Optional[VADResult]:
        """
        处理音频帧

        Args:
            frame: 音频帧

        Returns:
            VADResult: 检测结果，如果状态未变化则返回None
        """
        if not self.is_running or not self.vad:
            logger.warning("VADWorker not running")
            return None

        start_time = time.perf_counter()

        try:
            # webrtcvad期望特定采样率
            # 支持的采样率: 8000, 16000, 32000, 48000
            if frame.sample_rate not in [8000, 16000, 32000, 48000]:
                logger.warning(f"Unsupported sample rate: {frame.sample_rate}")
                return None

            # 将PCM数据转换为bytes（webrtcvad期望的格式）
            # 将int16数组转换为bytes
            pcm_bytes = frame.pcm16.tobytes()

            # 运行VAD
            is_speech = self.vad.is_speech(pcm_bytes, frame.sample_rate)

            self.total_frames += 1
            if is_speech:
                self.speech_frames += 1

            # 状态机
            result = self._update_state(is_speech, frame.ts)

            # 记录延迟
            latency_ms = (time.perf_counter() - start_time) * 1000
            record_latency("vad.inference_latency_ms", latency_ms)

            return result

        except Exception as e:
            logger.error(f"Error in VAD prediction: {e}")
            metrics.increment_counter("vad.errors")
            return None

    def _update_state(self, is_speech: bool, ts: float) -> Optional[VADResult]:
        """
        更新VAD状态机

        Args:
            is_speech: 是否检测到语音
            ts: 当前时间戳

        Returns:
            VADResult: 状态变化结果，如果状态未变化则返回None
        """
        current_time_ms = ts * 1000
        result = None

        if is_speech:
            if self.current_state == SpeechState.SILENCE:
                # 从静音切换到语音
                if self.speech_start_ts is None:
                    self.speech_start_ts = current_time_ms
                elif current_time_ms - self.speech_start_ts >= self.speech_start_ms:
                    # 确认语音开始
                    self.current_state = SpeechState.SPEECH
                    self.speech_end_ts = None
                    self.silence_start_ts = None

                    logger.info("Speech start detected")

                    # 触发事件
                    if self.event_callback:
                        event = BaseEvent(
                            type=EventType.SPEECH_START,
                            ts=ts,
                            session_id=self.session_id,
                            priority=60
                        )
                        self.event_callback(event)

                    metrics.increment_counter("vad.speech_start_count")
                    self.speech_segments += 1

                    result = VADResult(
                        state=SpeechState.SPEECH,
                        ts=ts,
                        confidence=1.0,
                        duration_ms=0
                    )
            else:
                # 持续语音
                self.speech_end_ts = None
                self.silence_start_ts = None

        else:  # not is_speech
            if self.current_state == SpeechState.SPEECH:
                # 从语音切换到静音
                if self.silence_start_ts is None:
                    self.silence_start_ts = current_time_ms
                elif current_time_ms - self.silence_start_ts >= self.speech_end_ms:
                    # 确认语音结束
                    self.current_state = SpeechState.SILENCE
                    self.speech_start_ts = None

                    duration_ms = current_time_ms - self.speech_end_ts if self.speech_end_ts else 0

                    logger.info(f"Speech end detected (duration={duration_ms:.0f}ms)")

                    # 触发事件
                    if self.event_callback:
                        event = BaseEvent(
                            type=EventType.SPEECH_END,
                            ts=ts,
                            session_id=self.session_id,
                            priority=60
                        )
                        self.event_callback(event)

                    metrics.increment_counter("vad.speech_end_count")
                    metrics.record("vad.speech_duration_ms", duration_ms)

                    result = VADResult(
                        state=SpeechState.SILENCE,
                        ts=ts,
                        confidence=1.0,
                        duration_ms=duration_ms
                    )
            else:
                # 持续静音
                pass

        return result

    def get_stats(self) -> dict:
        """获取统计信息"""
        speech_ratio = self.speech_frames / self.total_frames if self.total_frames > 0 else 0.0

        return {
            "current_state": self.current_state,
            "total_frames": self.total_frames,
            "speech_frames": self.speech_frames,
            "speech_ratio": speech_ratio,
            "speech_segments": self.speech_segments,
            "is_running": self.is_running
        }


class AsyncVADWorker:
    """异步VAD工作线程（包装器）"""

    def __init__(self, worker: VADWorker):
        """
        初始化异步VAD工作线程

        Args:
            worker: 底层VAD工作线程
        """
        self.worker = worker
        self.queue: asyncio.Queue[AudioFrame] = asyncio.Queue(maxsize=50)
        self.result_callbacks: list[Callable[[VADResult], None]] = []

    def start(self):
        """启动工作线程"""
        self.worker.start()

    def stop(self):
        """停止工作线程"""
        self.worker.stop()

    def add_result_callback(self, callback: Callable[[VADResult], None]):
        """添加结果回调"""
        self.result_callbacks.append(callback)

    async def process(self):
        """异步处理音频帧"""
        while True:
            frame = await self.queue.get()

            # 在线程池中处理（避免阻塞事件循环）
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.worker.process_frame,
                frame
            )

            if result:
                for callback in self.result_callbacks:
                    callback(result)

    def submit_frame(self, frame: AudioFrame):
        """提交音频帧（非阻塞）"""
        try:
            self.queue.put_nowait(frame)
        except asyncio.QueueFull:
            logger.warning("VAD frame queue full, dropping frame")
            metrics.increment_counter("vad.queue_overflows")
