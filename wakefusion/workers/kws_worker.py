"""
KWS (Keyword Spotting) 工作线程
使用openWakeWord进行唤醒词检测
"""

import asyncio
import numpy as np
import time
from typing import Optional, Callable
from dataclasses import dataclass
from datetime import datetime

from wakefusion.types import AudioFrame, EventType, KWSHitPayload, BaseEvent
from wakefusion.logging import get_logger
from wakefusion.metrics import get_metrics, record_latency


logger = get_logger("kws_worker")
metrics = get_metrics()


@dataclass
class KWSResult:
    """KWS检测结果"""
    keyword: str
    confidence: float
    ts: float
    pre_roll_start_ts: float
    pre_roll_end_ts: float


class KWSWorker:
    """KWS工作线程"""

    def __init__(
        self,
        keyword: str = "hey_assistant",
        threshold: float = 0.55,
        cooldown_ms: int = 1200,
        model_path: Optional[str] = None,
        event_callback: Optional[Callable[[BaseEvent], None]] = None
    ):
        """
        初始化KWS工作线程

        Args:
            keyword: 唤醒词
            threshold: 检测阈值
            cooldown_ms: 冷却时长（毫秒）
            model_path: 模型路径（如果为None，使用openWakeWord默认模型）
            event_callback: 事件回调函数
        """
        self.keyword = keyword
        self.threshold = threshold
        self.cooldown_ms = cooldown_ms
        self.model_path = model_path
        self.event_callback = event_callback

        # openWakeWord模型
        self.model: Optional[any] = None

        # 状态
        self.is_running = False
        self.last_detection_ts: Optional[float] = None
        self.session_id = self._generate_session_id()

        # 统计
        self.total_predictions = 0
        self.detections = 0
        self.false_positives = 0  # 将被决策引擎过滤的

        logger.info(
            "KWSWorker initialized",
            extra={
                "keyword": keyword,
                "threshold": threshold,
                "cooldown_ms": cooldown_ms
            }
        )

    def _generate_session_id(self) -> str:
        """生成会话ID"""
        return f"kws-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    def start(self):
        """启动KWS工作线程"""
        try:
            # 动态导入openWakeWord
            from openwakeword.model import Model

            # 初始化模型
            if self.model_path:
                logger.info(f"Loading custom model from: {self.model_path}")
                self.model = Model(wakeword_models=[self.model_path])
            else:
                logger.info("Loading default openWakeWord models")
                self.model = Model()

            self.is_running = True

            logger.info(
                "KWSWorker started",
                extra={
                    "keyword": self.keyword,
                    "threshold": self.threshold
                }
            )

        except ImportError:
            logger.error("openWakeWord not installed. Please install: pip install openwakeword")
            raise
        except Exception as e:
            logger.error(f"Failed to initialize KWS model: {e}")
            raise

    def stop(self):
        """停止KWS工作线程"""
        self.is_running = False
        logger.info("KWSWorker stopped")

    def process_frame(self, frame: AudioFrame) -> Optional[KWSResult]:
        """
        处理音频帧

        Args:
            frame: 音频帧

        Returns:
            KWSResult: 检测结果，如果未检测到则返回None
        """
        if not self.is_running or not self.model:
            logger.warning("KWSWorker not running")
            return None

        start_time = time.perf_counter()

        try:
            # 检查冷却期
            if self.last_detection_ts:
                elapsed_ms = (frame.ts - self.last_detection_ts) * 1000
                if elapsed_ms < self.cooldown_ms:
                    logger.debug(f"In cooldown period: {elapsed_ms:.0f}ms / {self.cooldown_ms}ms")
                    return None

            # openWakeWord期望16kHz 16-bit PCM
            # 确保音频帧格式正确
            if frame.sample_rate != 16000:
                logger.warning(f"Unexpected sample rate: {frame.sample_rate}, expected 16000")
                return None

            # 运行预测
            prediction = self.model.predict(frame.pcm16)

            self.total_predictions += 1

            # 检查是否有唤醒词匹配
            if self.keyword in prediction:
                confidence = prediction[self.keyword]

                logger.debug(
                    f"KWS prediction: {self.keyword} = {confidence:.3f}",
                    extra={"keyword": self.keyword, "confidence": confidence}
                )

                # 检查是否超过阈值
                if confidence >= self.threshold:
                    self.detections += 1
                    self.last_detection_ts = frame.ts

                    # 计算pre-roll时间范围
                    pre_roll_start_ts = frame.ts - self.cooldown_ms / 1000.0
                    pre_roll_end_ts = frame.ts

                    result = KWSResult(
                        keyword=self.keyword,
                        confidence=confidence,
                        ts=frame.ts,
                        pre_roll_start_ts=pre_roll_start_ts,
                        pre_roll_end_ts=pre_roll_end_ts
                    )

                    logger.info(
                        f"KWS detected: {self.keyword} (confidence={confidence:.3f})",
                        extra={
                            "keyword": self.keyword,
                            "confidence": confidence,
                            "pre_roll_ms": self.cooldown_ms
                        }
                    )

                    # 触发事件
                    if self.event_callback:
                        event = BaseEvent(
                            type=EventType.KWS_HIT,
                            ts=frame.ts,
                            session_id=self.session_id,
                            priority=80,
                            **{
                                "payload": KWSHitPayload(
                                    keyword=self.keyword,
                                    confidence=confidence,
                                    pre_roll_ms=self.cooldown_ms,
                                    audio_start_ts=pre_roll_start_ts,
                                    audio_end_ts=pre_roll_end_ts
                                ).model_dump()
                            }
                        )
                        self.event_callback(event)

                    # 记录指标
                    metrics.increment_counter("kws.hits")
                    metrics.record("kws.confidence", confidence)

                    return result

            # 记录延迟
            latency_ms = (time.perf_counter() - start_time) * 1000
            record_latency("kws.inference_latency_ms", latency_ms)

            return None

        except Exception as e:
            logger.error(f"Error in KWS prediction: {e}")
            metrics.increment_counter("kws.errors")
            return None

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "total_predictions": self.total_predictions,
            "detections": self.detections,
            "detection_rate": self.detections / self.total_predictions if self.total_predictions > 0 else 0.0,
            "last_detection_ts": self.last_detection_ts,
            "is_running": self.is_running
        }


class AsyncKWSWorker:
    """异步KWS工作线程（包装器）"""

    def __init__(self, worker: KWSWorker):
        """
        初始化异步KWS工作线程

        Args:
            worker: 底层KWS工作线程
        """
        self.worker = worker
        self.queue: asyncio.Queue[AudioFrame] = asyncio.Queue(maxsize=50)
        self.result_callbacks: list[Callable[[KWSResult], None]] = []

    def start(self):
        """启动工作线程"""
        self.worker.start()

    def stop(self):
        """停止工作线程"""
        self.worker.stop()

    def add_result_callback(self, callback: Callable[[KWSResult], None]):
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
            logger.warning("KWS frame queue full, dropping frame")
            metrics.increment_counter("kws.queue_overflows")
