"""
视觉路由器
负责视觉帧缓存和时间戳检索
"""

import asyncio
import numpy as np
from collections import deque
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import time

from wakefusion.types import VisionFrame
from wakefusion.logging import get_logger
from wakefusion.metrics import get_metrics


logger = get_logger("vision_router")
metrics = get_metrics()


@dataclass
class VisionCacheStats:
    """视觉缓存统计"""
    size: int
    capacity: int
    duration_ms: float
    oldest_ts: float
    newest_ts: float
    presence_count: int


class VisionRouter:
    """视觉路由器"""

    def __init__(
        self,
        cache_ms: int = 600,
        target_fps: int = 15
    ):
        """
        初始化视觉路由器

        Args:
            cache_ms: 缓存时长（毫秒）
            target_fps: 目标帧率（降频以减少计算量）
        """
        self.cache_ms = cache_ms
        self.target_fps = target_fps

        # 计算缓存容量
        self.frame_interval_ms = 1000 / target_fps
        self.cache_capacity = int(cache_ms / self.frame_interval_ms)

        # 帧缓存（使用deque自动管理容量）
        self.frame_cache: deque[VisionFrame] = deque(maxlen=self.cache_capacity)

        # 统计
        self.total_frames = 0
        self.dropped_frames = 0
        self.last_frame_time: Optional[float] = None

        logger.info(
            "VisionRouter initialized",
            extra={
                "cache_ms": cache_ms,
                "target_fps": target_fps,
                "cache_capacity": self.cache_capacity
            }
        )

    def process_frame(self, frame: VisionFrame):
        """
        处理视觉帧

        Args:
            frame: 视觉帧
        """
        # 检查是否需要跳帧（降频）
        if self.last_frame_time:
            elapsed_ms = (frame.ts - self.last_frame_time) * 1000
            if elapsed_ms < self.frame_interval_ms:
                # 跳过此帧
                self.dropped_frames += 1
                return

        # 添加到缓存
        self.frame_cache.append(frame)
        self.total_frames += 1
        self.last_frame_time = frame.ts

        logger.debug(
            f"Vision frame cached: presence={frame.presence}",
            extra={
                "presence": frame.presence,
                "distance_m": frame.distance_m,
                "confidence": frame.confidence
            }
        )

        # 记录指标
        metrics.record("vision.frame_queue_size", len(self.frame_cache))
        metrics.increment_counter("vision.frames_processed")

    def get_recent_frames(self, duration_ms: int = None) -> List[VisionFrame]:
        """
        获取最近的N帧

        Args:
            duration_ms: 时长（毫秒），如果为None则使用cache_ms

        Returns:
            List[VisionFrame]: 视觉帧列表
        """
        if duration_ms is None:
            duration_ms = self.cache_ms

        current_time = time.time()
        cutoff_time = current_time - duration_ms / 1000.0

        # 过滤符合条件的帧
        recent_frames = [
            frame for frame in self.frame_cache
            if frame.ts >= cutoff_time
        ]

        return recent_frames

    def get_latest_frame(self) -> Optional[VisionFrame]:
        """
        获取最新的一帧

        Returns:
            VisionFrame: 最新帧，如果缓存为空则返回None
        """
        if not self.frame_cache:
            return None

        return self.frame_cache[-1]

    def get_frame_at_time(self, ts: float, max_age_ms: int = 300) -> Optional[VisionFrame]:
        """
        获取指定时间戳附近的帧

        Args:
            ts: 目标时间戳
            max_age_ms: 最大年龄（毫秒）

        Returns:
            VisionFrame: 视觉帧，如果未找到则返回None
        """
        if not self.frame_cache:
            return None

        # 查找最接近的帧
        best_frame = None
        best_diff = float('inf')

        for frame in self.frame_cache:
            diff = abs(frame.ts - ts)
            if diff < best_diff and diff <= max_age_ms / 1000.0:
                best_frame = frame
                best_diff = diff

        return best_frame

    def get_cache_stats(self) -> VisionCacheStats:
        """
        获取缓存统计信息

        Returns:
            VisionCacheStats: 统计信息
        """
        if not self.frame_cache:
            return VisionCacheStats(
                size=0,
                capacity=self.cache_capacity,
                duration_ms=0.0,
                oldest_ts=0.0,
                newest_ts=0.0,
                presence_count=0
            )

        oldest_frame = self.frame_cache[0]
        newest_frame = self.frame_cache[-1]

        presence_count = sum(1 for f in self.frame_cache if f.presence)

        return VisionCacheStats(
            size=len(self.frame_cache),
            capacity=self.cache_capacity,
            duration_ms=(newest_frame.ts - oldest_frame.ts) * 1000,
            oldest_ts=oldest_frame.ts,
            newest_ts=newest_frame.ts,
            presence_count=presence_count
        )

    def get_presence_summary(self, max_age_ms: int = 500) -> Dict[str, Any]:
        """
        获取presence检测摘要

        Args:
            max_age_ms: 最大年龄（毫秒）

        Returns:
            Dict: presence摘要
        """
        recent_frames = self.get_recent_frames(max_age_ms)

        if not recent_frames:
            return {
                "has_presence": False,
                "presence_count": 0,
                "avg_confidence": 0.0,
                "avg_distance_m": None
            }

        presence_frames = [f for f in recent_frames if f.presence]

        return {
            "has_presence": len(presence_frames) > 0,
            "presence_count": len(presence_frames),
            "avg_confidence": (
                sum(f.confidence for f in presence_frames) / len(presence_frames)
                if presence_frames else 0.0
            ),
            "avg_distance_m": (
                sum(f.distance_m or 0 for f in presence_frames) / len(presence_frames)
                if presence_frames else None
            ),
            "latest_distance_m": (
                presence_frames[-1].distance_m
                if presence_frames else None
            )
        }

    def clear(self):
        """清空缓存"""
        self.frame_cache.clear()
        logger.info("Vision frame cache cleared")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        return {
            "total_frames": self.total_frames,
            "dropped_frames": self.dropped_frames,
            "drop_rate": (
                self.dropped_frames / self.total_frames
                if self.total_frames > 0 else 0.0
            ),
            "cache": self.get_cache_stats().__dict__,
            "presence": self.get_presence_summary()
        }


class AsyncVisionRouter:
    """异步视觉路由器（包装器）"""

    def __init__(self, router: VisionRouter):
        """
        初始化异步视觉路由器

        Args:
            router: 底层视觉路由器
        """
        self.router = router
        self.queue: asyncio.Queue[VisionFrame] = asyncio.Queue(maxsize=30)

    def process_frame(self, frame: VisionFrame):
        """处理视觉帧（非阻塞）"""
        try:
            self.queue.put_nowait(frame)
        except asyncio.QueueFull:
            logger.warning("Vision frame queue full, dropping frame")
            metrics.increment_counter("vision.queue_overflows")

    async def get_frame(self) -> VisionFrame:
        """异步获取视觉帧"""
        frame = await self.queue.get()
        self.router.process_frame(frame)
        return frame

    async def get_frame_timeout(self, timeout_ms: int = 200) -> Optional[VisionFrame]:
        """异步获取视觉帧（带超时）"""
        try:
            frame = await asyncio.wait_for(
                self.queue.get(),
                timeout_ms / 1000.0
            )
            self.router.process_frame(frame)
            return frame
        except asyncio.TimeoutError:
            return None
