"""
决策引擎
多模态融合决策逻辑：KWS + VAD + Vision → WAKE_CONFIRMED/BARGE_IN
"""

import asyncio
import time
from typing import Optional, Dict, Any, List
from dataclasses import dataclass
from enum import Enum
from datetime import datetime

from wakefusion.types import (
    BaseEvent, EventType,
    KWSHitPayload, WakeConfirmedPayload, HealthPayload
)
from wakefusion.logging import get_logger
from wakefusion.metrics import get_metrics, increment_counter


logger = get_logger("decision_engine")
metrics = get_metrics()


class FusionState(str, Enum):
    """融合状态"""
    IDLE = "IDLE"
    KWS_DETECTED = "KWS_DETECTED"
    WAITING_VISION = "WAITING_VISION"
    CONFIRMED = "CONFIRMED"
    REJECTED = "REJECTED"


@dataclass
class VisionGateResult:
    """视觉门控结果"""
    valid: bool
    presence: bool
    distance_m: Optional[float]
    confidence: float
    ts: float


class DecisionEngine:
    """决策引擎"""

    def __init__(
        self,
        kws_threshold: float = 0.55,
        probation_enabled: bool = True,
        probation_ms: int = 1000,
        barge_in_enabled: bool = True,
        event_callback: Optional[callable] = None
    ):
        """
        初始化决策引擎

        Args:
            kws_threshold: KWS置信度阈值
            probation_enabled: 是否启用降级策略
            probation_ms: 降级窗口时长（毫秒）
            barge_in_enabled: 是否启用打断
            event_callback: 事件回调函数
        """
        self.kws_threshold = kws_threshold
        self.probation_enabled = probation_enabled
        self.probation_ms = probation_ms
        self.barge_in_enabled = barge_in_enabled
        self.event_callback = event_callback

        # 状态（SystemState已删除，使用字符串常量）
        self.system_state = "IDLE"  # 可能的值: "IDLE", "LISTENING", "SPEAKING", "PROCESSING"
        self.fusion_state = FusionState.IDLE
        self.current_kws: Optional[Dict[str, Any]] = None
        self.current_vad_state: Optional[str] = None
        self.vision_cache: List[VisionGateResult] = []

        # Session
        self.session_id = self._generate_session_id()

        # 统计
        self.total_kws_hits = 0
        self.wake_confirmed_count = 0
        self.wake_rejected_count = 0
        self.barge_in_count = 0

        logger.info(
            "DecisionEngine initialized",
            extra={
                "kws_threshold": kws_threshold,
                "probation_enabled": probation_enabled,
                "barge_in_enabled": barge_in_enabled
            }
        )

    def _generate_session_id(self) -> str:
        """生成会话ID"""
        return f"fusion-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    def set_system_state(self, state: str):
        """
        设置系统状态

        Args:
            state: 系统状态（字符串，可能的值: "IDLE", "LISTENING", "SPEAKING", "PROCESSING"）
        """
        old_state = self.system_state
        self.system_state = state

        logger.info(
            f"System state changed: {old_state} → {state}",
            extra={"old_state": old_state, "new_state": state}
        )

    def update_vision_cache(self, result: VisionGateResult):
        """
        更新视觉缓存

        Args:
            result: 视觉门控结果
        """
        self.vision_cache.append(result)

        # 只保留最近的500ms
        current_time = time.time()
        self.vision_cache = [
            r for r in self.vision_cache
            if (current_time - r.ts) * 1000 < 500
        ]

    def get_recent_vision(self, max_age_ms: int = 300) -> Optional[VisionGateResult]:
        """
        获取最近的视觉结果

        Args:
            max_age_ms: 最大年龄（毫秒）

        Returns:
            VisionGateResult: 视觉结果，如果未找到则返回None
        """
        if not self.vision_cache:
            return None

        current_time = time.time()
        for result in reversed(self.vision_cache):
            age_ms = (current_time - result.ts) * 1000
            if age_ms <= max_age_ms:
                return result

        return None

    def process_kws_hit(self, event: BaseEvent) -> Optional[BaseEvent]:
        """
        处理KWS命中事件

        Args:
            event: KWS_HIT事件

        Returns:
            BaseEvent: 融合后的事件（WAKE_CONFIRMED或WAKE_REJECTED）
        """
        payload = KWSHitPayload(**event.payload)
        self.total_kws_hits += 1

        logger.info(
            f"KWS hit: {payload.keyword} (confidence={payload.confidence:.3f})",
            extra={
                "keyword": payload.keyword,
                "confidence": payload.confidence
            }
        )

        # 检查是否应该进入打断模式
        if self.system_state == "SPEAKING" and self.barge_in_enabled:
            return self._handle_barge_in(payload)

        # 检查视觉门控
        vision_result = self.get_recent_vision(max_age_ms=300)

        if vision_result and vision_result.valid:
            # 视觉门控通过
            return self._confirm_wake(payload, vision_result)
        else:
            # 视觉门控失败或未启用
            if self.probation_enabled:
                return self._probation_wake(payload, vision_result)
            else:
                return self._reject_wake(payload, reason="vision_gate_failed")

    def _handle_barge_in(self, payload: KWSHitPayload) -> BaseEvent:
        """处理打断"""
        self.barge_in_count += 1

        logger.info(
            f"Barge-in detected: {payload.keyword}",
            extra={
                "keyword": payload.keyword,
                "confidence": payload.confidence
            }
        )

        increment_counter("fusion.barge_in_count")

        event = BaseEvent(
            type=EventType.BARGE_IN,
            ts=time.time(),
            session_id=self.session_id,
            priority=100,  # 最高优先级
            **{
                "payload": payload.model_dump()
            }
        )

        if self.event_callback:
            self.event_callback(event)

        return event

    def _confirm_wake(
        self,
        payload: KWSHitPayload,
        vision_result: VisionGateResult
    ) -> BaseEvent:
        """确认唤醒"""
        self.wake_confirmed_count += 1

        logger.info(
            f"Wake confirmed: {payload.keyword}",
            extra={
                "keyword": payload.keyword,
                "kws_confidence": payload.confidence,
                "vision_confidence": vision_result.confidence,
                "distance_m": vision_result.distance_m
            }
        )

        increment_counter("fusion.wake_confirmed_count")

        wake_payload = WakeConfirmedPayload(
            keyword=payload.keyword,
            confidence=payload.confidence,
            pre_roll_ms=payload.pre_roll_ms,
            vision_gate=True,
            vision_confidence=vision_result.confidence,
            distance_m=vision_result.distance_m
        )

        event = BaseEvent(
            type=EventType.WAKE_CONFIRMED,
            ts=time.time(),
            session_id=self.session_id,
            priority=90,
            **{
                "payload": wake_payload.model_dump()
            }
        )

        if self.event_callback:
            self.event_callback(event)

        return event

    def _probation_wake(
        self,
        payload: KWSHitPayload,
        vision_result: Optional[VisionGateResult]
    ) -> BaseEvent:
        """降级唤醒（视觉验证失败但仍接受）"""
        self.wake_confirmed_count += 1

        vision_confidence = vision_result.confidence if vision_result else 0.0
        distance_m = vision_result.distance_m if vision_result else None

        logger.info(
            f"Wake confirmed (probation): {payload.keyword}",
            extra={
                "keyword": payload.keyword,
                "kws_confidence": payload.confidence,
                "vision_confidence": vision_confidence
            }
        )

        increment_counter("fusion.wake_probation_count")

        wake_payload = WakeConfirmedPayload(
            keyword=payload.keyword,
            confidence=payload.confidence * 0.8,  # 降低置信度
            pre_roll_ms=payload.pre_roll_ms,
            vision_gate=False,
            vision_confidence=vision_confidence,
            distance_m=distance_m
        )

        event = BaseEvent(
            type=EventType.WAKE_CONFIRMED,
            ts=time.time(),
            session_id=self.session_id,
            priority=80,  # 较低优先级
            **{
                "payload": wake_payload.model_dump()
            }
        )

        if self.event_callback:
            self.event_callback(event)

        return event

    def _reject_wake(self, payload: KWSHitPayload, reason: str) -> Optional[BaseEvent]:
        """拒绝唤醒"""
        self.wake_rejected_count += 1

        logger.info(
            f"Wake rejected: {payload.keyword} (reason={reason})",
            extra={
                "keyword": payload.keyword,
                "confidence": payload.confidence,
                "reason": reason
            }
        )

        increment_counter("fusion.wake_rejected_count")

        # 可选：发送REJECTED事件用于调试
        event = BaseEvent(
            type=EventType.WAKE_REJECTED,
            ts=time.time(),
            session_id=self.session_id,
            priority=50,
            **{
                "payload": {
                    "keyword": payload.keyword,
                    "confidence": payload.confidence,
                    "reason": reason
                }
            }
        )

        if self.event_callback:
            self.event_callback(event)

        return event

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_decisions = self.wake_confirmed_count + self.wake_rejected_count

        return {
            "system_state": self.system_state,
            "fusion_state": self.fusion_state,
            "total_kws_hits": self.total_kws_hits,
            "wake_confirmed_count": self.wake_confirmed_count,
            "wake_rejected_count": self.wake_rejected_count,
            "barge_in_count": self.barge_in_count,
            "confirmation_rate": (
                self.wake_confirmed_count / total_decisions
                if total_decisions > 0 else 0.0
            ),
            "vision_cache_size": len(self.vision_cache)
        }
