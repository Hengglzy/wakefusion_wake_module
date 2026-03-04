"""
人脸门控工作线程
负责presence检测、深度门控和人脸验证
"""

import asyncio
import numpy as np
import logging
from typing import Optional, List, Dict, Any
from dataclasses import dataclass
from collections import deque
import time
import cv2
import os
import socket
import json
import select

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
    enable_face_detection: bool = True  # 启用人脸检测（通过 UDP 接收）
    udp_port: int = 9999  # UDP 接收端口
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

        # 日志级别统一为 INFO，禁用 DEBUG；同时将 camera_driver 的日志降为 ERROR，防止 AlignFilter 等细节刷屏
        logger.logger.setLevel(logging.INFO)
        logging.getLogger("camera_driver").setLevel(logging.ERROR)

        # UDP Socket（用于接收远程视觉服务数据）
        self.udp_socket = None
        if self.config.enable_face_detection:
            try:
                # 初始化非阻塞 UDP Socket
                self.udp_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
                self.udp_socket.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
                self.udp_socket.bind(("127.0.0.1", self.config.udp_port))
                self.udp_socket.setblocking(False)  # 非阻塞模式
                logger.debug(f"UDP socket initialized on port {self.config.udp_port}")
            except Exception as e:
                logger.error(f"Failed to initialize UDP socket: {e}, face detection disabled")
                self.config.enable_face_detection = False
                self.udp_socket = None

        # 从 UDP 接收的数据
        self.faces: List[Dict[str, Any]] = []  # 人脸列表（归一化坐标）
        self.current_gesture: Optional[str] = None  # 当前手势名称
        self.hand_center: Optional[tuple] = None  # 手部中心坐标（归一化）
        self.hand_distance_m: Optional[float] = None  # 手部距离（米）
        self.udp_distance_m: Optional[float] = None  # 服务端计算的全局距离（米）

        # 距离历史（用于时间平滑，增加到15以获得更平稳的平均值）
        self.distance_history = deque(maxlen=15)

        # 迟滞状态（用于消除闪烁）
        self.last_valid_state = False
        self.face_missing_frames = 0  # 人脸消失帧数计数

        # 状态
        self.is_running = False

        # 统计
        self.total_frames = 0
        self.valid_user_count = 0
        self.rejected_count = 0

        logger.debug(
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
        logger.debug("FaceGateWorker started")

    def stop(self):
        """停止人脸门控工作线程"""
        self.is_running = False
        if self.udp_socket:
            self.udp_socket.close()
            self.udp_socket = None
        logger.debug("FaceGateWorker stopped")

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

            # 1. 从 UDP 接收视觉服务数据（非阻塞）
            if self.config.enable_face_detection and self.udp_socket:
                self._receive_udp_data(frame)
            
            # 使用接收到的 faces 数据
            faces = self.faces.copy() if self.faces else []

            # 2. Presence检测
            presence = self._detect_presence(frame)

            # 3. 距离门控（优先用服务端距离；本地有 depth 时再现场计算）
            distance_m = None
            if self.config.enable_depth_gate:
                # 3.0 优先使用 UDP 服务端距离（工业模式）
                raw_distance = self.udp_distance_m
                if raw_distance is None and faces and isinstance(faces[0], dict):
                    raw_distance = faces[0].get("distance_m")

                # 3.1 若没有服务端距离但本地有 depth，则现场计算并补全每张脸/手距离
                if raw_distance is None and frame.depth is not None:
                    # 为人脸计算距离
                    for face in faces:
                        face_distance = self._estimate_distance_for_face(frame.depth, face)
                        face['distance_m'] = face_distance if face_distance is not None else None

                    # 为手部计算距离（如果存在手部中心坐标）
                    if self.hand_center is not None:
                        hand_distance = self._estimate_distance_for_hand(frame.depth, self.hand_center)
                        self.hand_distance_m = hand_distance if hand_distance is not None else None

                    # 使用第一张人脸的距离作为全局距离；否则回退中心区域
                    if faces and faces[0].get('distance_m') is not None:
                        raw_distance = faces[0].get('distance_m')
                    else:
                        raw_distance = self._estimate_distance(frame.depth, None)

                # 3.2 无论来源如何，对 raw_distance 做时间平滑
                if raw_distance is not None:
                    try:
                        raw_distance_f = float(raw_distance)
                        self.distance_history.append(raw_distance_f)
                        distance_m = float(np.mean(self.distance_history)) if len(self.distance_history) > 0 else raw_distance_f
                        frame.distance_m = distance_m
                    except Exception:
                        distance_m = None
                        frame.distance_m = None

            # 4. 综合判断
            is_valid = self._validate_gate(frame, presence, distance_m, faces)

            # 5. 计算置信度
            confidence = self._calculate_confidence(frame, presence, distance_m, faces)

            # 更新帧的presence和confidence
            frame.presence = presence
            frame.confidence = confidence
            frame.faces = faces
            
            # 更新手势和手部信息（用于 GUI 显示）
            frame.gesture = self.current_gesture
            frame.hand_center = self.hand_center
            frame.hand_distance_m = self.hand_distance_m

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

    def _receive_udp_data(self, frame: VisionFrame):
        """非阻塞读取 UDP 数据（JSON），更新 faces/gesture/hand_center/距离字段。"""
        if not self.udp_socket:
            return
        try:
            while True:
                data, _addr = self.udp_socket.recvfrom(65535)
                try:
                    parsed = json.loads(data.decode("utf-8"))
                except Exception:
                    continue
                if not isinstance(parsed, dict):
                    continue

                # faces：兼容 vision_service 的 x/y/w/h，也兼容旧的 xmin/ymin/width/height
                faces_in = parsed.get("faces", [])
                faces_out: List[Dict[str, Any]] = []
                if isinstance(faces_in, list):
                    for f in faces_in:
                        if not isinstance(f, dict):
                            continue
                        x = f.get("xmin", f.get("x", 0.0))
                        y = f.get("ymin", f.get("y", 0.0))
                        w = f.get("width", f.get("w", 0.0))
                        h = f.get("height", f.get("h", 0.0))
                        out = dict(f)
                        out["xmin"] = float(x)
                        out["ymin"] = float(y)
                        out["width"] = float(w)
                        out["height"] = float(h)
                        # 也保留 x/y/w/h，方便 GUI/兼容
                        out["x"] = float(x)
                        out["y"] = float(y)
                        out["w"] = float(w)
                        out["h"] = float(h)
                        if out.get("confidence") is not None:
                            try:
                                out["confidence"] = float(out["confidence"])
                            except Exception:
                                out["confidence"] = 0.0
                        if out.get("distance_m") is not None:
                            try:
                                out["distance_m"] = float(out["distance_m"])
                            except Exception:
                                out["distance_m"] = None
                        faces_out.append(out)
                self.faces = faces_out

                # 多手数据解析：hands 列表
                # 健壮性：在 _receive_udp_data 中对 hands 列表按置信度排序，并对 x, y 坐标增加 float 强制转换及 None 值保护
                hands_in = parsed.get("hands", [])
                hands_out: List[Dict[str, Any]] = []
                if isinstance(hands_in, list):
                    # 先收集所有手部数据，并添加置信度字段（用于排序）
                    hands_with_confidence = []
                    for h in hands_in:
                        if not isinstance(h, dict):
                            continue
                        out = dict(h)
                        # 确保关键字段存在且类型正确
                        if "index" not in out:
                            continue
                        try:
                            out["index"] = int(out["index"])
                            out["gesture"] = str(out.get("gesture")) if out.get("gesture") is not None else None
                            # 健壮性：增加对 None 值的类型转换保护，确保 x, y 始终为 float
                            x_val = out.get("x")
                            y_val = out.get("y")
                            if x_val is None or y_val is None:
                                continue  # 跳过无效的手部数据
                            out["x"] = float(x_val)
                            out["y"] = float(y_val)
                            if out.get("distance_m") is not None:
                                out["distance_m"] = float(out["distance_m"])
                            else:
                                out["distance_m"] = None
                            # 添加置信度字段（用于排序，如果没有则默认为 0.5）
                            confidence = out.get("confidence", 0.5)
                            try:
                                out["confidence"] = float(confidence) if confidence is not None else 0.5
                            except (ValueError, TypeError):
                                out["confidence"] = 0.5
                            hands_with_confidence.append(out)
                        except Exception:
                            continue
                    # 按置信度排序（从高到低）
                    hands_with_confidence.sort(key=lambda x: x.get("confidence", 0.5), reverse=True)
                    hands_out = hands_with_confidence
                
                # 向后兼容：保留第一只手的数据
                self.current_gesture = hands_out[0].get("gesture") if hands_out else parsed.get("gesture", None)
                if hands_out:
                    self.hand_center = (hands_out[0]["x"], hands_out[0]["y"])
                    self.hand_distance_m = hands_out[0].get("distance_m")
                else:
                    hc = parsed.get("hand_center", None)
                    if isinstance(hc, (list, tuple)) and len(hc) >= 2:
                        try:
                            self.hand_center = (float(hc[0]), float(hc[1]))
                        except Exception:
                            self.hand_center = None
                    else:
                        self.hand_center = None
                    hd = parsed.get("hand_distance_m", None)
                    if hd is not None:
                        try:
                            self.hand_distance_m = float(hd)
                        except Exception:
                            self.hand_distance_m = None

                dm = parsed.get("distance_m", None)
                if dm is not None:
                    try:
                        self.udp_distance_m = float(dm)
                    except Exception:
                        self.udp_distance_m = None
                
                # 更新 VisionFrame.hands（用于业务逻辑和 GUI 显示）
                frame.hands = hands_out

        except BlockingIOError:
            return
        except Exception as e:
            logger.warning(f"Error receiving UDP data: {e}")

    def _detect_presence(self, frame: VisionFrame) -> bool:
        """
        检测是否有人（基于深度数据）

        Args:
            frame: 视觉帧

        Returns:
            bool: 是否检测到人
        """
        # 方案B：GUI 进程不带 depth，仅从 UDP 看到 faces/hand_center
        if frame.depth is None:
            return bool(self.faces) or (self.hand_center is not None)

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

    def _estimate_distance_for_face(self, depth: np.ndarray, face: Dict[str, Any]) -> Optional[float]:
        """
        为单张人脸估计距离（基于深度数据，支持空间滤波和人脸对焦）

        Args:
            depth: 深度图（16位，单位：毫米）
            face: 单张人脸信息字典，包含 xmin, ymin, width, height

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

            h, w = depth.shape

            # 将归一化坐标转换为绝对像素坐标
            # face 格式：xmin/ymin/width/height 或 x/y/w/h（都是归一化 0.0-1.0）
            xmin_rel = face.get("xmin", face.get("x", 0.0))
            ymin_rel = face.get("ymin", face.get("y", 0.0))
            width_rel = face.get("width", face.get("w", 0.0))
            height_rel = face.get("height", face.get("h", 0.0))

            face_xmin = int(float(xmin_rel) * w)
            face_ymin = int(float(ymin_rel) * h)
            face_xmax = int((float(xmin_rel) + float(width_rel)) * w)
            face_ymax = int((float(ymin_rel) + float(height_rel)) * h)
            
            # 确保坐标在有效范围内
            face_xmin = max(0, face_xmin)
            face_ymin = max(0, face_ymin)
            face_xmax = min(w, face_xmax)
            face_ymax = min(h, face_ymax)
            
            # 计算人脸框中心 40% 区域（缩小采样范围以避免捕捉到背景或头发）
            # 由于深度已对齐到 RGB，坐标可以直接对应
            face_center_x = (face_xmin + face_xmax) // 2
            face_center_y = (face_ymin + face_ymax) // 2
            face_width = face_xmax - face_xmin
            face_height = face_ymax - face_ymin
            
            # 中心 40% 区域（40% 的一半 = 20%）
            half_width = int(face_width * 0.2)
            half_height = int(face_height * 0.2)
            
            target_xmin = max(0, face_center_x - half_width)
            target_ymin = max(0, face_center_y - half_height)
            target_xmax = min(w, face_center_x + half_width)
            target_ymax = min(h, face_center_y + half_height)
            
            # 提取人脸中心区域的深度（原始数据）
            target_depth_raw = depth_m[target_ymin:target_ymax, target_xmin:target_xmax]
            target_mask = valid_mask[target_ymin:target_ymax, target_xmin:target_xmax]

            # 统计过滤：使用中位数计算距离（np.median 本身就是极强的空间滤波器，无需重复滤波）
            if not np.any(target_mask):
                return None

            median_depth = np.median(target_depth_raw[target_mask])

            return float(median_depth)

        except Exception as e:
            logger.error(f"Error in distance estimation for face: {e}")
            return None

    def _estimate_distance(self, depth: np.ndarray, faces: List[Dict[str, Any]] = None) -> Optional[float]:
        """
        估计用户距离（基于深度数据，支持空间滤波和人脸对焦）
        注意：此方法用于无人脸或中心区域测距，单张人脸测距请使用 _estimate_distance_for_face

        Args:
            depth: 深度图（16位，单位：毫米）
            faces: 人脸列表（可选，用于动态对焦，但此方法只使用第一张人脸）

        Returns:
            float: 距离（米），如果无法估计则返回None
        """
        try:
            # 转换为米
            depth_m = depth.astype(np.float32) / 1000.0

            # 过滤有效范围（先过滤，再对采样区域应用双边滤波）
            valid_mask = (
                (depth_m >= self.config.distance_m_min) &
                (depth_m <= self.config.distance_m_max)
            )

            if not np.any(valid_mask):
                return None

            h, w = depth.shape

            # 目标对焦：如果检测到人脸，使用人脸框中心 40% 区域；否则使用中心区域
            if faces and len(faces) > 0:
                # 使用第一张人脸（置信度最高）
                return self._estimate_distance_for_face(depth, faces[0])
            else:
                # 使用中心区域
                center_h, center_w = h // 2, w // 2
                half_size = min(h, w) // 4

                target_depth_raw = depth_m[
                    center_h - half_size:center_h + half_size,
                    center_w - half_size:center_w + half_size
                ]
                target_mask = valid_mask[
                    center_h - half_size:center_h + half_size,
                    center_w - half_size:center_w + half_size
                ]

                # 对采样区域应用双边滤波（降噪，减少随机跳动）
                if target_depth_raw.size > 0:
                    target_depth_filtered = cv2.bilateralFilter(
                        target_depth_raw.astype(np.float32), 5, 75, 75
                    )
                else:
                    target_depth_filtered = target_depth_raw

                # 统计过滤：使用中位数计算距离（更抗干扰）
                if not np.any(target_mask):
                    return None

                median_depth = np.median(target_depth_filtered[target_mask])

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
        验证门控条件（带迟滞逻辑，消除闪烁）

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
            self.face_missing_frames = 0
            self.last_valid_state = False
            return False

        # 2. 人脸门控（如果启用）
        has_valid_face = False
        if self.config.enable_face_detection:
            # 检查是否有人脸且置信度足够
            has_valid_face = any(
                f['confidence'] >= self.config.face_conf_min
                for f in faces
            )

            if not has_valid_face:
                self.face_missing_frames += 1
            else:
                self.face_missing_frames = 0

        # 2.5 多手 OK 手势强制唤醒联动：
        # 联动策略升级：修改 _validate_gate。遍历 frame.hands 列表，只要检测到 "ok" 且距离 < 4m，无视人脸状态直接设 READY
        hands = frame.hands if hasattr(frame, 'hands') and isinstance(frame.hands, list) else []
        for hand in hands:
            if isinstance(hand, dict):
                gesture = hand.get("gesture")
                hand_dist = hand.get("distance_m")
                if gesture == "ok" and hand_dist is not None:
                    try:
                        if float(hand_dist) < 4.0:
                            self.last_valid_state = True
                            self.face_missing_frames = 0  # 重置人脸消失计数
                            # 唤醒锁定机制：一旦由 OK 手势触发 READY，强制锁定该状态 15 帧（约 0.5 秒），防止手部细微晃动导致绿框瞬间变黄
                            if not hasattr(self, '_wake_lock_frames'):
                                self._wake_lock_frames = 0
                            self._wake_lock_frames = 15  # 保持 15 帧的唤醒锁定（约 0.5 秒）
                            return True  # 直接返回，无视其他条件
                    except Exception:
                        continue
        
        # 唤醒锁定：即使 OK 手势消失，也保持 15 帧的唤醒状态
        if hasattr(self, '_wake_lock_frames') and self._wake_lock_frames > 0:
            self._wake_lock_frames -= 1
            self.last_valid_state = True
            self.face_missing_frames = 0
            return True

        # 3. 状态平滑：即使 UDP 丢包 1 帧，也通过 last_valid_state 保持 3 帧的 Wake Ready 状态
        # 如果之前是 READY 状态，即使当前帧条件不满足，也保持最多 3 帧
        if self.last_valid_state:
            if not hasattr(self, '_ready_persistence_frames'):
                self._ready_persistence_frames = 0
            self._ready_persistence_frames += 1
        else:
            if hasattr(self, '_ready_persistence_frames'):
                self._ready_persistence_frames = 0
        
        # 3. 迟滞逻辑（进入 3.8m，离开 4.2m）
        enter_threshold = 3.8  # 进入 READY 的阈值
        exit_threshold = 4.2   # 离开 READY 的阈值
        
        if distance_m is None:
            # 距离无效，保持当前状态或返回 False
            if self.last_valid_state and self.face_missing_frames < 5:
                # 如果之前是 READY 且人脸消失少于 5 帧，保持状态
                return True
            # 状态平滑：即使 UDP 丢包，也保持最多 3 帧的 READY 状态
            if hasattr(self, '_ready_persistence_frames') and self._ready_persistence_frames > 0 and self._ready_persistence_frames <= 3:
                return True
            self.last_valid_state = False
            return False
        
        # 检查距离是否过近
        if distance_m < self.config.distance_m_min:
            logger.debug(f"Distance too close: {distance_m:.2f}m < {self.config.distance_m_min}m")
            self.last_valid_state = False
            return False

        # 迟滞判定
        if self.last_valid_state:
            # 当前是 READY 状态：只有距离 > 4.2m 或人脸消失超过 5 帧才切回 STANDBY
            if distance_m > exit_threshold:
                logger.debug(f"Distance too far (exit): {distance_m:.2f}m > {exit_threshold}m")
                self.last_valid_state = False
                if hasattr(self, '_ready_persistence_frames'):
                    self._ready_persistence_frames = 0
                return False
            if self.config.enable_face_detection and self.face_missing_frames >= 5:
                logger.debug(f"Face missing for {self.face_missing_frames} frames")
                self.last_valid_state = False
                if hasattr(self, '_ready_persistence_frames'):
                    self._ready_persistence_frames = 0
                return False
            # 保持 READY 状态
            return True
        else:
            # 当前是 STANDBY 状态：只有距离 < 3.8m 且有人脸才切到 READY
            if distance_m < enter_threshold:
                if not self.config.enable_face_detection or has_valid_face:
                    logger.debug(f"Entering READY: {distance_m:.2f}m < {enter_threshold}m")
                    self.last_valid_state = True
                    return True
            # 保持 STANDBY 状态
            return False

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

        # 3. 人脸置信度（如果启用）
        if self.config.enable_face_detection and faces:
            face_conf = max(f['confidence'] for f in faces) if faces else 0.0
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
