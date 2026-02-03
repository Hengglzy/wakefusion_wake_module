"""
人脸门控工作线程
负责presence检测、深度门控和人脸验证
"""

import asyncio
import numpy as np
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
import time

from wakefusion.types import VisionFrame, EventType, BaseEvent
from wakefusion.decision import VisionGateResult
from wakefusion.logging import get_logger
from wakefusion.metrics import get_metrics, record_latency


logger = get_logger("face_gate")
metrics = get_metrics()


@dataclass
class FaceGateConfig:
    """人脸门控配置"""
    distance_m_max: float = 4.0  # 最大检测距离（米）
    distance_m_min: float = 0.5  # 最小检测距离（米）
    face_conf_min: float = 0.55  # 最小人脸置信度
    depth_confidence_threshold: int = 100  # 深度置信度阈值
    enable_face_detection: bool = False  # Phase 1暂不启用人脸检测
    enable_depth_gate: bool = True  # 启用深度门控


class FaceGateWorker:
    """人脸门控工作线程"""

    def __init__(
        self,
        config: FaceGateConfig = None,
        event_callback: Optional[callable] = None
    ):
        """
        初始化人脸门控工作线程

        Args:
            config: 门控配置
            event_callback: 事件回调函数
        """
        self.config = config or FaceGateConfig()
        self.event_callback = event_callback

        # MediaPipe人脸检测（可选）
        self.face_detector = None
        if self.config.enable_face_detection:
            try:
                import mediapipe as mp
                mp_face_detection = mp.solutions.face_detection
                self.face_detector = mp_face_detection.FaceDetection(
                    min_detection_confidence=self.config.face_conf_min
                )
                logger.info("MediaPipe face detection initialized")
            except ImportError:
                logger.warning("mediapipe not installed, face detection disabled")
                self.config.enable_face_detection = False

        # 状态
        self.is_running = False

        # 统计
        self.total_frames = 0
        self.valid_user_count = 0
        self.rejected_count = 0

        logger.info(
            "FaceGateWorker initialized",
            extra={
                "distance_m_max": self.config.distance_m_max,
                "distance_m_min": self.config.distance_m_min,
                "face_conf_min": self.config.face_conf_min,
                "enable_face_detection": self.config.enable_face_detection,
                "enable_depth_gate": self.config.enable_depth_gate
            }
        )

    def start(self):
        """启动人脸门控工作线程"""
        self.is_running = True
        logger.info("FaceGateWorker started")

    def stop(self):
        """停止人脸门控工作线程"""
        self.is_running = False
        logger.info("FaceGateWorker stopped")

    def process_frame(self, frame: VisionFrame) -> Optional[VisionGateResult]:
        """
        处理视觉帧

        Args:
            frame: 视觉帧

        Returns:
            VisionGateResult: 门控结果
        """
        if not self.is_running:
            return None

        start_time = time.perf_counter()

        try:
            self.total_frames += 1

            # 1. Presence检测（基于深度数据）
            presence = self._detect_presence(frame)

            # 2. 人脸检测（可选）
            faces = []
            if self.config.enable_face_detection and self.face_detector and frame.rgb is not None:
                faces = self._detect_faces(frame.rgb)

            # 3. 深度门控
            distance_m = None
            if self.config.enable_depth_gate and frame.depth is not None:
                distance_m = self._estimate_distance(frame.depth)

                # 更新帧的距离信息
                frame.distance_m = distance_m

            # 4. 综合判断
            is_valid = self._validate_gate(frame, presence, distance_m, faces)

            # 5. 计算置信度
            confidence = self._calculate_confidence(frame, presence, distance_m, faces)

            # 更新帧的presence和confidence
            frame.presence = presence
            frame.confidence = confidence
            frame.faces = faces

            # 创建门控结果
            result = VisionGateResult(
                valid=is_valid,
                presence=presence,
                distance_m=distance_m,
                confidence=confidence,
                ts=frame.ts
            )

            # 统计
            if is_valid:
                self.valid_user_count += 1
            else:
                self.rejected_count += 1

            # 记录延迟
            latency_ms = (time.perf_counter() - start_time) * 1000
            record_latency("face_gate.inference_latency_ms", latency_ms)

            # 触发PRESENCE事件
            if self.event_callback and presence:
                event = BaseEvent(
                    type=EventType.PRESENCE,
                    ts=frame.ts,
                    session_id=f"vision-{int(frame.ts)}",
                    priority=50,
                    **{
                        "payload": {
                            "presence": True,
                            "distance_m": distance_m,
                            "confidence": confidence
                        }
                    }
                )
                self.event_callback(event)

            return result

        except Exception as e:
            logger.error(f"Error in face gate processing: {e}")
            metrics.increment_counter("face_gate.errors")
            return None

    def _detect_presence(self, frame: VisionFrame) -> bool:
        """
        检测是否有人（基于深度数据）

        Args:
            frame: 视觉帧

        Returns:
            bool: 是否检测到人
        """
        if frame.depth is None:
            return False

        try:
            # 深度数据有效性检查
            # 将深度值转换为米（Femto Bolt的单位通常是毫米）
            depth_m = frame.depth.astype(np.float32) / 1000.0

            # 过滤有效深度范围
            valid_mask = (
                (depth_m >= self.config.distance_m_min) &
                (depth_m <= self.config.distance_m_max)
            )

            # 计算有效像素比例
            valid_ratio = np.sum(valid_mask) / valid_mask.size

            # 如果有效深度像素超过阈值，认为有人
            presence_threshold = 0.01  # 1%的像素
            has_presence = valid_ratio > presence_threshold

            return has_presence

        except Exception as e:
            logger.error(f"Error in presence detection: {e}")
            return False

    def _detect_faces(self, rgb: np.ndarray) -> List[Dict[str, Any]]:
        """
        检测人脸

        Args:
            rgb: RGB图像

        Returns:
            List[Dict]: 人脸列表
        """
        if not self.face_detector:
            return []

        try:
            # MediaPipe期望RGB图像
            rgb_uint8 = rgb.astype(np.uint8)

            # 运行人脸检测
            results = self.face_detector.process(rgb_uint8)

            faces = []
            if results.detections:
                for detection in results.detections:
                    bbox = detection.location_data.relative_bounding_box

                    face_info = {
                        "xmin": bbox.xmin,
                        "ymin": bbox.ymin,
                        "width": bbox.width,
                        "height": bbox.height,
                        "confidence": detection.score[0]
                    }

                    faces.append(face_info)

            return faces

        except Exception as e:
            logger.error(f"Error in face detection: {e}")
            return []

    def _estimate_distance(self, depth: np.ndarray) -> Optional[float]:
        """
        估计用户距离（基于深度数据）

        Args:
            depth: 深度图

        Returns:
            float: 距离（米），如果无法估计则返回None
        """
        try:
            # 转换为米
            depth_m = depth.astype(np.float32) / 1000.0

            # 过滤有效范围
            valid_mask = (
                (depth_m >= self.config.distance_m_min) &
                (depth_m <= self.config.distance_m_max)
            )

            if not np.any(valid_mask):
                return None

            # 计算中心区域的平均深度（更可靠）
            h, w = depth.shape
            center_h, center_w = h // 2, w // 2
            half_size = min(h, w) // 4

            center_depth = depth_m[
                center_h - half_size:center_h + half_size,
                center_w - half_size:center_w + half_size
            ]

            # 使用中位数（更抗干扰）
            median_depth = np.median(center_depth[valid_mask[
                center_h - half_size:center_h + half_size,
                center_w - half_size:center_w + half_size
            ]])

            return float(median_depth)

        except Exception as e:
            logger.error(f"Error in distance estimation: {e}")
            return None

    def _validate_gate(
        self,
        frame: VisionFrame,
        presence: bool,
        distance_m: Optional[float],
        faces: List[Dict[str, Any]]
    ) -> bool:
        """
        验证门控条件

        Args:
            frame: 视觉帧
            presence: 是否检测到人
            distance_m: 距离（米）
            faces: 人脸列表

        Returns:
            bool: 是否通过门控
        """
        # 1. 必须有presence
        if not presence:
            return False

        # 2. 距离门控
        if distance_m is not None:
            if distance_m > self.config.distance_m_max:
                logger.debug(f"Distance too far: {distance_m:.2f}m > {self.config.distance_m_max}m")
                return False

            if distance_m < self.config.distance_m_min:
                logger.debug(f"Distance too close: {distance_m:.2f}m < {self.config.distance_m_min}m")
                return False

        # 3. 人脸门控（可选）
        if self.config.enable_face_detection and faces:
            # 检查是否有人脸且置信度足够
            has_valid_face = any(
                f['confidence'] >= self.config.face_conf_min
                for f in faces
            )

            if not has_valid_face:
                return False

        return True

    def _calculate_confidence(
        self,
        frame: VisionFrame,
        presence: bool,
        distance_m: Optional[float],
        faces: List[Dict[str, Any]]
    ) -> float:
        """
        计算门控置信度

        Args:
            frame: 视觉帧
            presence: 是否检测到人
            distance_m: 距离（米）
            faces: 人脸列表

        Returns:
            float: 置信度 (0-1)
        """
        confidence = 0.0

        # 1. Presence置信度（基于深度质量）
        if presence and frame.depth is not None:
            # 计算有效深度比例
            depth_m = frame.depth.astype(np.float32) / 1000.0
            valid_ratio = np.sum(
                (depth_m >= self.config.distance_m_min) &
                (depth_m <= self.config.distance_m_max)
            ) / depth_m.size

            confidence += valid_ratio * 0.5

        # 2. 距离置信度（距离越近置信度越高）
        if distance_m is not None:
            # 归一化距离 (0-4m -> 1-0)
            distance_score = max(0, 1 - distance_m / self.config.distance_m_max)
            confidence += distance_score * 0.3

        # 3. 人脸置信度
        if self.config.enable_face_detection and faces:
            face_conf = max(f['confidence'] for f in faces)
            confidence += face_conf * 0.2

        return min(1.0, confidence)

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        total_decisions = self.valid_user_count + self.rejected_count

        return {
            "total_frames": self.total_frames,
            "valid_user_count": self.valid_user_count,
            "rejected_count": self.rejected_count,
            "pass_rate": (
                self.valid_user_count / total_decisions
                if total_decisions > 0 else 0.0
            ),
            "is_running": self.is_running,
            "enable_face_detection": self.config.enable_face_detection,
            "enable_depth_gate": self.config.enable_depth_gate
        }


class AsyncFaceGateWorker:
    """异步人脸门控工作线程（包装器）"""

    def __init__(self, worker: FaceGateWorker):
        """
        初始化异步人脸门控工作线程

        Args:
            worker: 底层人脸门控工作线程
        """
        self.worker = worker
        self.queue: asyncio.Queue[VisionFrame] = asyncio.Queue(maxsize=30)

    def start(self):
        """启动工作线程"""
        self.worker.start()

    def stop(self):
        """停止工作线程"""
        self.worker.stop()

    async def process(self):
        """异步处理视觉帧"""
        while True:
            frame = await self.queue.get()

            # 在线程池中处理
            loop = asyncio.get_event_loop()
            result = await loop.run_in_executor(
                None,
                self.worker.process_frame,
                frame
            )

            # TODO: 处理结果（例如发送到决策引擎）

    def submit_frame(self, frame: VisionFrame):
        """提交视觉帧（非阻塞）"""
        try:
            self.queue.put_nowait(frame)
        except asyncio.QueueFull:
            logger.warning("Face gate frame queue full, dropping frame")
            metrics.increment_counter("face_gate.queue_overflows")
