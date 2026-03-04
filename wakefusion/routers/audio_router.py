"""
音频路由器
负责下采样、Ring Buffer管理和音频帧分发
"""

import asyncio
import numpy as np
from collections import deque
from typing import Optional, List, Callable
from dataclasses import dataclass
import time

from wakefusion.types import AudioFrameRaw, AudioFrame
from wakefusion.logging import get_logger
from wakefusion.metrics import get_metrics, record_latency


logger = get_logger("audio_router")
metrics = get_metrics()


@dataclass
class RingBufferStats:
    """Ring Buffer统计信息"""
    size: int
    capacity: int
    duration_sec: float
    oldest_ts: float
    newest_ts: float


class AudioRouter:
    """音频路由器"""

    def __init__(
        self,
        capture_sample_rate: int = 48000,
        work_sample_rate: int = 16000,
        frame_ms: int = 20,
        ring_buffer_sec: float = 2.0,
        rnnoise_enabled: bool = False
    ):
        """
        初始化音频路由器

        Args:
            capture_sample_rate: 采集采样率
            work_sample_rate: 工作采样率（下采样目标）
            frame_ms: 帧长（毫秒）
            ring_buffer_sec: Ring buffer长度（秒）
            rnnoise_enabled: 是否启用 RNNoise 降噪
        """
        self.capture_sample_rate = capture_sample_rate
        self.work_sample_rate = work_sample_rate
        self.frame_ms = frame_ms

        # 计算帧大小
        self.capture_frame_size = int(capture_sample_rate * frame_ms / 1000)
        self.work_frame_size = int(work_sample_rate * frame_ms / 1000)

        # 计算下采样比例
        self.resample_ratio = work_sample_rate / capture_sample_rate

        # Ring Buffer
        self.ring_buffer: deque[AudioFrame] = deque()
        self.ring_buffer_capacity = int(ring_buffer_sec * 1000 / frame_ms)

        # 订阅者列表
        self.subscribers: List[Callable[[AudioFrame], None]] = []

        # 统计信息
        self.total_frames = 0
        self.dropped_frames = 0
        self.last_frame_time: Optional[float] = None

        # 初始化 RNNoise 服务（独立模块，可选）
        self.rnnoise_service = None
        if rnnoise_enabled:
            try:
                from wakefusion.services.rnnoise_service import RNNoiseService
                self.rnnoise_service = RNNoiseService(
                    enabled=True,
                    sample_rate=capture_sample_rate
                )
                if self.rnnoise_service.is_available():
                    logger.info("RNNoise service integrated into AudioRouter")
                else:
                    logger.warning("RNNoise service requested but not available, using passthrough")
            except Exception as e:
                logger.warning(f"Failed to initialize RNNoise service: {e}, using passthrough")

        logger.info(
            "AudioRouter initialized",
            extra={
                "capture_sample_rate": capture_sample_rate,
                "work_sample_rate": work_sample_rate,
                "frame_ms": frame_ms,
                "ring_buffer_sec": ring_buffer_sec,
                "ring_buffer_capacity": self.ring_buffer_capacity,
                "rnnoise_enabled": rnnoise_enabled and (self.rnnoise_service is not None and self.rnnoise_service.is_available())
            }
        )

    def subscribe(self, callback: Callable[[AudioFrame], None]):
        """
        订阅音频帧

        Args:
            callback: 回调函数
        """
        self.subscribers.append(callback)
        logger.debug(f"New subscriber, total: {len(self.subscribers)}")

    def unsubscribe(self, callback: Callable[[AudioFrame], None]):
        """
        取消订阅

        Args:
            callback: 回调函数
        """
        if callback in self.subscribers:
            self.subscribers.remove(callback)
            logger.debug(f"Subscriber removed, total: {len(self.subscribers)}")

    def process_raw_frame(self, raw_frame: AudioFrameRaw):
        """
        处理原始音频帧（从驱动调用）

        Args:
            raw_frame: 原始音频帧
        """
        start_time = time.perf_counter()

        try:
            # 1. RNNoise 降噪（在 48kHz 下处理，如果启用）
            pcm16_raw = raw_frame.pcm16
            if self.rnnoise_service is not None and self.rnnoise_service.is_available():
                pcm16_raw = self.rnnoise_service.process(pcm16_raw)
            
            # 2. 下采样到工作采样率
            pcm16 = self._resample(pcm16_raw)

            # 3. 创建工作音频帧
            frame = AudioFrame(
                ts=raw_frame.ts,
                pcm16=pcm16,
                sample_rate=self.work_sample_rate
            )

            # 添加到Ring Buffer
            self._add_to_ring_buffer(frame)

            # 通知订阅者
            for callback in self.subscribers:
                try:
                    callback(frame)
                except Exception as e:
                    logger.error(f"Error in subscriber callback: {e}")

            # 统计
            self.total_frames += 1
            if self.last_frame_time:
                gap = frame.ts - self.last_frame_time
                if gap > self.frame_ms / 1000 * 1.5:  # 允许50%的抖动
                    logger.warning(f"Large frame gap: {gap*1000:.1f}ms")
                    metrics.increment_counter("audio.frame_gaps")

            self.last_frame_time = frame.ts

            # 记录延迟
            latency_ms = (time.perf_counter() - start_time) * 1000
            record_latency("audio.router_latency_ms", latency_ms)

        except Exception as e:
            logger.error(f"Error processing audio frame: {e}")
            metrics.increment_counter("audio.processing_errors")

    def _resample(self, pcm16: np.ndarray) -> np.ndarray:
        """
        下采样音频数据

        Args:
            pcm16: 原始PCM数据

        Returns:
            np.ndarray: 下采样后的PCM数据
        """
        # 简单的下采样：取样本点
        # TODO: 可以使用librosa进行更高质量的重采样

        # 计算目标长度
        target_length = int(len(pcm16) * self.resample_ratio)

        # 使用线性插值进行下采样
        indices = np.linspace(0, len(pcm16) - 1, target_length)
        resampled = np.interp(indices, np.arange(len(pcm16)), pcm16.astype(np.float32))
        resampled = resampled.astype(np.int16)

        return resampled

    def _add_to_ring_buffer(self, frame: AudioFrame):
        """
        添加音频帧到Ring Buffer

        Args:
            frame: 音频帧
        """
        self.ring_buffer.append(frame)

        # 如果超过容量，删除最旧的帧
        if len(self.ring_buffer) > self.ring_buffer_capacity:
            self.ring_buffer.popleft()
            self.dropped_frames += 1
            metrics.increment_counter("audio.ring_buffer_overflows")

    def get_ring_buffer_stats(self) -> RingBufferStats:
        """
        获取Ring Buffer统计信息

        Returns:
            RingBufferStats: 统计信息
        """
        if not self.ring_buffer:
            return RingBufferStats(
                size=0,
                capacity=self.ring_buffer_capacity,
                duration_sec=0.0,
                oldest_ts=0.0,
                newest_ts=0.0
            )

        oldest_frame = self.ring_buffer[0]
        newest_frame = self.ring_buffer[-1]

        return RingBufferStats(
            size=len(self.ring_buffer),
            capacity=self.ring_buffer_capacity,
            duration_sec=(newest_frame.ts - oldest_frame.ts),
            oldest_ts=oldest_frame.ts,
            newest_ts=newest_frame.ts
        )

    def fetch_audio_segment(
        self,
        start_ts: float,
        end_ts: Optional[float] = None,
        max_duration_ms: int = 2000
    ) -> Optional[np.ndarray]:
        """
        从Ring Buffer中获取音频片段

        Args:
            start_ts: 起始时间戳
            end_ts: 结束时间戳（如果为None，则使用max_duration_ms）
            max_duration_ms: 最大时长（毫秒）

        Returns:
            np.ndarray: 音频数据，如果未找到则返回None
        """
        if not self.ring_buffer:
            return None

        # 确定结束时间戳
        if end_ts is None:
            end_ts = start_ts + max_duration_ms / 1000.0

        # 收集符合条件的帧
        frames = []
        for frame in self.ring_buffer:
            if start_ts <= frame.ts <= end_ts:
                frames.append(frame)

        if not frames:
            logger.warning(f"No frames found in range [{start_ts:.3f}, {end_ts:.3f}]")
            return None

        # 合并帧
        audio_data = np.concatenate([frame.pcm16 for frame in frames])
        return audio_data

    def get_recent_frames(self, duration_ms: int) -> List[AudioFrame]:
        """
        获取最近的N帧

        Args:
            duration_ms: 时长（毫秒）

        Returns:
            List[AudioFrame]: 音频帧列表
        """
        frame_count = int(duration_ms / self.frame_ms)
        return list(self.ring_buffer)[-frame_count:]

    def clear(self):
        """清空Ring Buffer"""
        self.ring_buffer.clear()
        logger.info("Ring buffer cleared")

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "total_frames": self.total_frames,
            "dropped_frames": self.dropped_frames,
            "drop_rate": self.dropped_frames / self.total_frames if self.total_frames > 0 else 0.0,
            "subscribers": len(self.subscribers),
            "ring_buffer": self.get_ring_buffer_stats().__dict__
        }


class AsyncAudioRouter:
    """异步音频路由器（包装器）"""

    def __init__(self, router: AudioRouter):
        """
        初始化异步音频路由器

        Args:
            router: 底层音频路由器
        """
        self.router = router
        self.queue: asyncio.Queue[AudioFrame] = asyncio.Queue(maxsize=100)

        # 订阅底层路由器的事件
        self.router.subscribe(self._on_audio_frame)

    def _on_audio_frame(self, frame: AudioFrame):
        """
        音频帧回调（从底层路由器调用）

        Args:
            frame: 音频帧
        """
        try:
            # 非阻塞地放入队列
            self.queue.put_nowait(frame)
        except asyncio.QueueFull:
            logger.warning("Audio frame queue full, dropping frame")
            metrics.increment_counter("audio.queue_overflows")

    async def get_frame(self) -> AudioFrame:
        """
        异步获取音频帧

        Returns:
            AudioFrame: 音频帧
        """
        return await self.queue.get()

    async def get_frame_timeout(self, timeout_ms: int = 100) -> Optional[AudioFrame]:
        """
        异步获取音频帧（带超时）

        Args:
            timeout_ms: 超时时间（毫秒）

        Returns:
            AudioFrame: 音频帧，如果超时则返回None
        """
        try:
            return await asyncio.wait_for(self.queue.get(), timeout_ms / 1000.0)
        except asyncio.TimeoutError:
            return None
