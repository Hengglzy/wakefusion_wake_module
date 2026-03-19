"""
视觉桥接服务
在独立进程中运行 MediaPipe，计算结果通过 UDP 发出
"""

# 强力导入（显式物理路径），绕过包解析差异
import mediapipe.python.solutions.face_detection as mp_face

# MediaPipe Tasks（Gesture Recognizer）
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

import cv2
import numpy as np
import socket
import json
import time
import threading
import queue
import logging
import os
from pathlib import Path
from typing import Optional, List, Dict, Any
from collections import deque
import zmq
from wakefusion.config import get_config
from wakefusion.workers.lip_sync_detector import LipSyncDetector

# 日志级别设置（MediaPipe 设为 WARNING）
logging.getLogger("mediapipe").setLevel(logging.WARNING)

# VisionService 专用日志器（文本格式由 wakefusion.logging 统一配置）
vision_logger = logging.getLogger("vision_service")
vision_logger.setLevel(logging.INFO)

# 相机与人脸门控日志：仅保留 ERROR 级别，避免 AlignFilter / 设备搜索等细节刷屏
# 注意：camera_driver 使用 wakefusion.logging.get_logger，需要直接设置其级别
try:
    from wakefusion.logging import get_logger
    camera_logger = get_logger("camera_driver")
    camera_logger.logger.setLevel(logging.ERROR)
    face_gate_logger = get_logger("face_gate")
    face_gate_logger.logger.setLevel(logging.ERROR)
except Exception:
    # 如果导入失败，回退到标准日志库设置
    logging.getLogger("camera_driver").setLevel(logging.ERROR)
    logging.getLogger("face_gate").setLevel(logging.ERROR)
# OpenCV 日志级别设置（兼容不同版本）
try:
    if hasattr(cv2, 'setLogLevel'):
        cv2.setLogLevel(cv2.LOG_LEVEL_WARNING)
    elif hasattr(cv2.utils, 'setLogLevel'):
        cv2.utils.setLogLevel(cv2.utils.LOG_LEVEL_WARNING)
except Exception:
    # 如果 OpenCV 版本不支持，忽略（不影响功能）
    pass


class VisionService:
    """视觉处理引擎 - MediaPipe 人脸和手势检测"""
    
    def __init__(
        self,
        config_path: Optional[str] = None,
        target_fps: int = 15,
        jpeg_quality: int = 55,  # 默认55，在画质和性能之间取得平衡（降低约8%文件大小，画质几乎无影响）
    ):
        """
        初始化视觉服务
        
        Args:
            config_path: 配置文件路径（可选）
            target_fps: 目标帧率
            jpeg_quality: JPEG 压缩质量（0-100）
        """
        # 加载配置
        self.config = get_config(config_path)
        self.vision_wake_config = self.config.vision_wake
        self.zmq_config = self.config.zmq
        
        self.target_fps = target_fps
        self.frame_time = 1.0 / target_fps
        self.jpeg_quality = int(jpeg_quality)
        
        # 性能优化：跳帧处理计数器（降低CPU负担）
        self._process_frame_counter = 0  # MediaPipe处理跳帧计数器
        self._depth_send_counter = 0      # 深度图发送跳帧计数器
        
        # 初始化 ZMQ Context 和 PUB Socket
        self.zmq_context = zmq.Context()
        self._zmq_pub_socket = self.zmq_context.socket(zmq.PUB)
        vision_pub_port = self.zmq_config.vision_pub_port
        self._zmq_pub_socket.bind(f"tcp://127.0.0.1:{vision_pub_port}")
        vision_logger.info(f"ZMQ PUB Socket bound to tcp://127.0.0.1:{vision_pub_port}")

        # 视觉唤醒状态机已移除，业务逻辑移至core_server
        
        # 保留图像发送功能（UDP图像端口可保留，或后续讨论）
        self.udp_host = "127.0.0.1"  # UDP 目标地址（GUI 接收地址）
        self.udp_image_port = 10000
        self.udp_depth_port = 10001
        self.udp_image_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_depth_socket = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.udp_image_socket.setblocking(False)
        self.udp_depth_socket.setblocking(False)

        # 图像异步发送队列（避免 JPEG 压缩阻塞推理）
        # item: (magic, port, bgr_frame)
        self._img_queue: "queue.Queue[tuple[bytes, int, np.ndarray]]" = queue.Queue(maxsize=4)
        self._img_thread_stop = threading.Event()
        self._img_thread = threading.Thread(target=self._image_sender_loop, daemon=True)
        self._img_thread.start()

        # 分包协议参数（UDP 单包上限约 65KB，这里保守设置）
        self._rgb_magic = b"IMG1"  # RGB 视频流
        self._dpt_magic = b"DPT1"  # 深度彩色流
        self._rgb_frame_id = 0
        self._dpt_frame_id = 0
        self._max_payload = 60000  # bytes (header 外的 payload)

        # MediaPipe 初始化（按指令：强制显式路径 + 固定参数）
        # 人脸检测器（模型选择 1，针对近距离）
        self.face_detector = mp_face.FaceDetection(
            model_selection=1,
            min_detection_confidence=0.5,
        )

        # Gesture Recognizer（LIVE_STREAM 异步回调）
        # 注意：LIVE_STREAM 模式的结果是异步回调返回，因此需要线程安全缓存
        self._gesture_lock = threading.Lock()
        self._latest_gesture_data: Dict[str, Any] = {
            # 最近一次 GestureRecognizer 结果的毫秒时间戳
            "timestamp_ms": None,
            # hands: List[{"builtin_gesture": Optional[str], "landmarks": List[{"x","y","z"}]}]
            # 即使还没有任何结果返回，也保证 hands 至少是一个空列表，避免 None 带来的解析问题
            "hands": [],
        }

        # 采用"内存缓冲区"方案彻底修复 GestureRecognizer 初始化失败的问题
        # 直接在 Python 层读取文件，避开 MediaPipe 的路径 Bug
        self.recognizer = None
        try:
            with open("gesture_recognizer.task", "rb") as f:
                model_buffer = f.read()
            
            base_options = python.BaseOptions(model_asset_buffer=model_buffer)
            # 不同 MediaPipe 版本的 Options 参数可能略有差异，这里做一次兼容性兜底
            try:
                options = vision.GestureRecognizerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.LIVE_STREAM,
                    num_hands=4,
                    result_callback=self._on_gesture_result,
                )
            except TypeError:
                options = vision.GestureRecognizerOptions(
                    base_options=base_options,
                    running_mode=vision.RunningMode.LIVE_STREAM,
                    result_callback=self._on_gesture_result,
                )
            
            self.recognizer = vision.GestureRecognizer.create_from_options(options)
            vision_logger.info("[GestureRecognizer] 使用 Buffer 模式初始化成功")
        except FileNotFoundError:
            vision_logger.error("请确保 gesture_recognizer.task 文件位于项目根目录下")
            self.recognizer = None
        except Exception as e:
            vision_logger.exception(f"[GestureRecognizer] 初始化失败: {e}")
            self.recognizer = None
        
        # 手势识别相关：状态管理（基于质心的轨迹历史）
        # palm_center_history: Dict[int, deque] - track_id -> deque(maxlen=30) of (ts, x, y)
        # 用于挥手（waving）“简谐运动”判定和轨迹平滑
        self.palm_center_history: Dict[int, deque] = {}
        
        # 基于质心的手部追踪器（Centroid Tracker）
        # hand_tracks: track_id -> {
        #   "id": int,
        #   "x": float, "y": float,               # 当前质心（归一化）
        #   "vx": float, "vy": float,             # 速度向量（单位：每秒归一化坐标）
        #   "gesture": Optional[str],             # 当前手势
        #   "last_ts": float,                     # 最近一次被真实检测更新的时间戳
        #   "missing_frames": int,                # 连续丢失检测的帧数（用于惯性追踪与清理）
        # }
        self.hand_tracks: Dict[int, Dict[str, Any]] = {}
        self._next_hand_id: int = 0
        # 惯性追踪：维持最多 6 帧的缓存（检测丢失时仍保留手势框）
        self._inertia_max_frames = 6
        # 上一次轨迹更新时间（用于插值预测）
        self._last_tracks_ts: float = time.time()
        # 最近一帧的人脸检测结果（用于跳帧时复用）
        self._last_faces: List[Dict[str, Any]] = []
        # 🌟 任务3：EMA滤波存储 - 存储每个face的上一帧frontal_percent（key: face_id）
        self._face_frontal_history: Dict[int, float] = {}
        
        # 初始化唇动检测器
        self.lip_detector = LipSyncDetector()
        
        vision_logger.info(
            f"VisionService initialized: ZMQ PUB tcp://127.0.0.1:{vision_pub_port}, "
            f"IMG UDP 127.0.0.1:{self.udp_image_port}, DEPTH UDP 127.0.0.1:{self.udp_depth_port}, "
            f"FPS={target_fps}, JPEG={self.jpeg_quality}, Max Hands=4"
        )

    def _next_frame_id(self, magic: bytes) -> int:
        if magic == self._rgb_magic:
            fid = self._rgb_frame_id & 0xFFFFFFFF
            self._rgb_frame_id += 1
            return fid
        fid = self._dpt_frame_id & 0xFFFFFFFF
        self._dpt_frame_id += 1
        return fid

    def _pack_and_send_image(self, magic: bytes, port: int, bgr_frame: np.ndarray):
        """
        将 BGR 帧压缩为 JPEG 并分包发送（UDP）。
        包格式（小端）：
          magic(4) + frame_id(u32) + chunk_idx(u16) + total_chunks(u16) + payload
        """
        # JPEG 压缩（BGR）
        ok, enc = cv2.imencode(
            ".jpg",
            bgr_frame,
            [int(cv2.IMWRITE_JPEG_QUALITY), self.jpeg_quality],
        )
        if not ok:
            return
        data = enc.tobytes()

        frame_id = self._next_frame_id(magic)

        total_chunks = (len(data) + self._max_payload - 1) // self._max_payload
        if total_chunks <= 0:
            return
        if total_chunks > 65535:
            # 极端情况：直接丢弃
            return

        for idx in range(total_chunks):
            start = idx * self._max_payload
            end = min(len(data), start + self._max_payload)
            payload = data[start:end]

            header = (
                magic
                + int(frame_id).to_bytes(4, "little", signed=False)
                + int(idx).to_bytes(2, "little", signed=False)
                + int(total_chunks).to_bytes(2, "little", signed=False)
            )
            pkt = header + payload
            try:
                if magic == self._rgb_magic:
                    self.udp_image_socket.sendto(pkt, (self.udp_host, port))
                else:
                    self.udp_depth_socket.sendto(pkt, (self.udp_host, port))
            except (BlockingIOError, OSError):
                # 发送缓冲满或临时错误：丢包即可（实时视频允许）
                break

    def _image_sender_loop(self):
        """后台线程：从队列取帧，做 JPEG 压缩 + UDP 发送。"""
        while not self._img_thread_stop.is_set():
            try:
                magic, port, bgr_frame = self._img_queue.get(timeout=0.1)
            except queue.Empty:
                continue
            try:
                self._pack_and_send_image(magic, port, bgr_frame)
            except Exception:
                # 工业模式：不让线程崩掉
                pass
            finally:
                try:
                    self._img_queue.task_done()
                except Exception:
                    pass

    def _on_gesture_result(self, result, output_image, timestamp_ms: int):
        """
        GestureRecognizer LIVE_STREAM 模式的异步回调。
        将 result.gestures 与 result.hand_landmarks 写入线程安全缓存 self._latest_gesture_data。
        """
        # 始终构造一个合法的 hands 列表，即使没有检测到任何手势，也不会返回 None
        hands: List[Dict[str, Any]] = []
        try:
            gestures = getattr(result, "gestures", None)
            hand_landmarks = getattr(result, "hand_landmarks", None)
            # 以 hand_landmarks 为准确定手数量（gestures 有时可能为空）
            num_hands = len(hand_landmarks) if hand_landmarks is not None else (len(gestures) if gestures else 0)

            for i in range(num_hands):
                builtin_gesture = None
                try:
                    if gestures and i < len(gestures) and gestures[i]:
                        builtin_gesture = getattr(gestures[i][0], "category_name", None)
                except Exception:
                    builtin_gesture = None

                lm_dicts: List[Dict[str, float]] = []
                try:
                    if hand_landmarks and i < len(hand_landmarks) and hand_landmarks[i] is not None:
                        lm_list = hand_landmarks[i]
                        # 兼容不同返回类型：可能是 list，也可能是带 landmark 属性的对象
                        if hasattr(lm_list, "landmark"):
                            lm_iter = lm_list.landmark
                        else:
                            lm_iter = lm_list
                        for lm in lm_iter:
                            lm_dicts.append(
                                {
                                    "x": float(getattr(lm, "x", 0.0)),
                                    "y": float(getattr(lm, "y", 0.0)),
                                    "z": float(getattr(lm, "z", 0.0)),
                                }
                            )
                except Exception:
                    lm_dicts = []

                hands.append(
                    {
                        "builtin_gesture": str(builtin_gesture) if builtin_gesture else None,
                        "landmarks": lm_dicts,
                    }
                )
        except Exception:
            # 回调里不抛异常，避免中断 recognizer 内部线程
            hands = []

        # 用锁保护回调结果更新，保证主线程读取时的一致性
        # 仅当新结果的时间戳更新（更大）时才刷新缓存，避免乱序结果“回滚”状态
        try:
            new_ts = int(timestamp_ms) if timestamp_ms is not None else None
        except (TypeError, ValueError):
            new_ts = None
        with self._gesture_lock:
            old_ts = None
            if isinstance(self._latest_gesture_data, dict):
                old_ts = self._latest_gesture_data.get("timestamp_ms")
            if old_ts is None or (new_ts is not None and new_ts >= old_ts):
                self._latest_gesture_data = {
                    "timestamp_ms": new_ts,
                    "hands": hands if isinstance(hands, list) else [],
                }
    
    def _detect_gesture(
        self,
        landmarks: Optional[List[Dict[str, float]]],
        builtin_gesture: Optional[str] = None,
    ) -> Optional[str]:
        """
        手势检测（优先级：waving > ok > 内置手势）。

        注意：
        - waving 在 track 级别基于轨迹判定（见 _is_waving_track），不在这里直接返回
        - ok 保留自定义逻辑：点4与点8距离 / (点0到点9距离) 的比例判定
        - thumbs_up / fist 等由 MediaPipe Tasks 内置 GestureRecognizer 输出
        """
        if not landmarks or len(landmarks) < 10:
            return self._map_builtin_gesture(builtin_gesture)

        def _xy(idx: int) -> tuple[float, float]:
            p = landmarks[idx]
            return float(p.get("x", 0.0)), float(p.get("y", 0.0))

        # 自定义 OK：点4与点8比例 < 0.25（对距离尺度鲁棒）
        x4, y4 = _xy(4)
        x8, y8 = _xy(8)
        x0, y0 = _xy(0)
        x9, y9 = _xy(9)

        ok_distance = float(np.sqrt((x8 - x4) ** 2 + (y8 - y4) ** 2))
        palm_scale = float(np.sqrt((x0 - x9) ** 2 + (y0 - y9) ** 2))
        if palm_scale > 1e-3:
            ok_ratio = ok_distance / palm_scale
            if ok_ratio < 0.25:
                return "ok"

        return self._map_builtin_gesture(builtin_gesture)

    def _map_builtin_gesture(self, builtin_gesture: Optional[str]) -> Optional[str]:
        """将 MediaPipe Tasks 的手势名映射到本项目的手势字段。"""
        if builtin_gesture is None:
            return None
        name = str(builtin_gesture).strip()
        if not name or name.lower() == "none":
            return None

        mapping = {
            "Closed_Fist": "fist",
            "Open_Palm": "open_palm",
            "Pointing_Up": "pointing_up",
            "Thumb_Down": "thumb_down",
            "Thumb_Up": "thumbs_up",
            "Victory": "victory",
            "ILoveYou": "i_love_you",
        }
        return mapping.get(name, name)
    
    def _extract_hand_center(self, landmarks: Optional[List[Dict[str, float]]]) -> Optional[tuple]:
        """
        提取手部中心坐标（归一化）
        
        Args:
            landmarks: 21 个手部关键点（归一化），来自 GestureRecognizer 回调缓存
            
        Returns:
            (x, y) 归一化坐标，或 None
        """
        if not landmarks:
            return None
        
        # 使用手腕位置作为手部中心
        wrist = landmarks[0]
        return (float(wrist.get("x", 0.0)), float(wrist.get("y", 0.0)))

    def _update_palm_history(self, track_id: int, ts: float, x: float, y: float):
        """
        更新指定 track 的掌心轨迹历史（用于挥手判定），保留最近约 0.5 秒、最多 30 帧。
        """
        if track_id not in self.palm_center_history:
            self.palm_center_history[track_id] = deque(maxlen=30)
        history = self.palm_center_history[track_id]
        history.append((ts, float(x), float(y)))
        # 仅保留最近 0.5 秒的轨迹
        while history and (ts - float(history[0][0])) > 0.5:
            history.popleft()

    def _is_waving_track(self, track_id: int) -> bool:
        """
        基于掌心轨迹的“简谐运动”挥手判定（仅使用稳定的 track_id，不依赖 MediaPipe index）。

        判定要点：
        - 在最近 0.5 秒内，X 轴方向至少发生约 2.5 次完整切换（≈5 次方向反转）
        - X 轴总位移跨度足够大（避免微小抖动）
        - 正/负方向的位移量大致均衡（简单的能量对称性判断）
        - Y 轴随 X 摆动呈现轻微“圆弧感”（|corr(|x-mean_x|, y)| 较大且 y 有一定幅度）
        """
        history = self.palm_center_history.get(track_id)
        if not history or len(history) < 8:
            return False

        xs = [float(x) for _ts, x, _y in history]
        ys = [float(y) for _ts, _x, y in history]
        if len(xs) < 2:
            return False

        max_x = max(xs)
        min_x = min(xs)
        x_span = max_x - min_x
        if x_span < 0.12:
            # 摆动幅度过小，不认为是挥手
            return False

        # 方向反转统计（基于 X 轴速度符号）
        direction_changes = 0
        prev_sign = 0
        eps = 0.005  # 噪声阈值：过滤微小抖动
        sum_pos = 0.0
        sum_neg = 0.0
        for i in range(1, len(xs)):
            dx = xs[i] - xs[i - 1]
            if abs(dx) < eps:
                continue
            if dx > 0:
                sum_pos += dx
                sign = 1
            else:
                sum_neg += -dx
                sign = -1
            if prev_sign != 0 and sign != prev_sign:
                direction_changes += 1
            prev_sign = sign

        # 至少约 2.5 个完整往复 ≈ 5 次方向反转
        if direction_changes < 5:
            return False

        # 正负方向能量需大致均衡，避免“只往一个方向甩”的误判
        if sum_pos <= 0.02 or sum_neg <= 0.02:
            return False
        ratio = sum_pos / max(sum_neg, 1e-6)
        if not (0.5 <= ratio <= 2.0):
            return False

        # 圆弧运动校验：Y 随 X 摆动呈现轻微二次曲线特征
        y_span = max(ys) - min(ys)
        if y_span < 0.01:
            # 几乎没有 Y 变化，更像纯水平平移，排除
            return False

        mean_x = float(np.mean(xs))
        x_dev = [abs(x - mean_x) for x in xs]
        # 计算 |x-mean_x| 与 y 的相关性，反映“靠两侧时 Y 更高或更低”的弧线感
        try:
            if len(x_dev) >= 3:
                corr_mat = np.corrcoef(x_dev, ys)
                arc_score = float(corr_mat[0, 1])
            else:
                arc_score = 0.0
        except Exception:
            arc_score = 0.0

        if abs(arc_score) < 0.3:
            return False

        return True
    
    def _update_hand_tracks(self, detections: List[Dict[str, Any]], ts: float) -> None:
        """
        使用质心距离将当前帧的检测结果与已有 track 进行匹配，更新 hand_tracks。
        - 仅当新检测点与历史质心的欧氏距离 < 0.2 时视为同一只手
        - 否则在“最多 4 手”的上限内新建 track，并复用 0-3 的编号
        - 未匹配到的旧 track 通过 missing_frames 实现惯性追踪与超时删除
        """
        # 记录当前仍然存活但本帧尚未匹配到的 track_id 集合
        unmatched_tracks = set(self.hand_tracks.keys())

        # 为每个检测分配最近的 track（先粗配，后按距离排序避免一对多）
        assign_det_to_track: Dict[int, int] = {}
        det_min_dist: Dict[int, float] = {}

        for det_idx, det in enumerate(detections):
            cx = float(det["x"])
            cy = float(det["y"])
            best_id = None
            best_dist2 = None
            for track_id in unmatched_tracks:
                t = self.hand_tracks[track_id]
                dx = cx - float(t.get("x", 0.0))
                dy = cy - float(t.get("y", 0.0))
                dist2 = dx * dx + dy * dy
                if best_dist2 is None or dist2 < best_dist2:
                    best_dist2 = dist2
                    best_id = track_id
            if best_id is not None and best_dist2 is not None:
                det_min_dist[det_idx] = float(np.sqrt(best_dist2))
                assign_det_to_track[det_idx] = best_id

        # 距离阈值过滤 + 按距离从小到大精配，避免一个 track 被多次分配
        used_tracks: set[int] = set()
        final_assign: Dict[int, int] = {}
        # 阈值适当放宽到 0.2，以抵御 MediaPipe 自身的小抖动，减少“错误新建 track”
        threshold = 0.2
        for det_idx, dist in sorted(det_min_dist.items(), key=lambda kv: kv[1]):
            if dist > threshold:
                continue
            track_id = assign_det_to_track[det_idx]
            if track_id in used_tracks:
                continue
            final_assign[det_idx] = track_id
            used_tracks.add(track_id)
            if track_id in unmatched_tracks:
                unmatched_tracks.remove(track_id)

        # 更新已匹配 track（加入位置平滑，降低抖动）
        for det_idx, track_id in final_assign.items():
            det = detections[det_idx]
            cx = float(det["x"])
            cy = float(det["y"])
            gesture_name = det.get("gesture")

            t = self.hand_tracks[track_id]
            prev_x = float(t.get("x", cx))
            prev_y = float(t.get("y", cy))
            prev_ts = float(t.get("last_ts", ts))
            dt = max(1e-3, ts - prev_ts)

            vx = (cx - prev_x) / dt
            vy = (cy - prev_y) / dt

            # 位置平滑：新位置 = 60% 当前检测 + 40% 之前位置，缓解关键点抖动
            alpha = 0.6
            smooth_x = alpha * cx + (1.0 - alpha) * prev_x
            smooth_y = alpha * cy + (1.0 - alpha) * prev_y

            t["x"] = smooth_x
            t["y"] = smooth_y
            t["vx"] = vx
            t["vy"] = vy
            t["gesture"] = str(gesture_name) if gesture_name is not None else None
            t["last_ts"] = ts
            t["missing_frames"] = 0

            # 更新挥手轨迹历史，并根据轨迹决定是否判定为 waving
            self._update_palm_history(track_id, ts, smooth_x, smooth_y)
            if self._is_waving_track(track_id):
                t["gesture"] = "waving"

        # 为未匹配到任何已有 track 的检测创建新 track（最多 4 个，复用 0-3 编号）
        all_det_indices = set(range(len(detections)))
        unmatched_dets = all_det_indices - set(final_assign.keys())
        for det_idx in unmatched_dets:
            det = detections[det_idx]
            cx = float(det["x"])
            cy = float(det["y"])
            gesture_name = det.get("gesture")

            # 控制最大 track 数量为 4，并复用编号 0-3，避免编号无限增长
            max_tracks = 4
            existing_ids = set(self.hand_tracks.keys())
            if len(existing_ids) >= max_tracks:
                # 已达到最大可跟踪手数，上限外的检测忽略，避免“多余框”
                continue

            # 选择一个最小可用的 id（0-3 中的空位）
            track_id = None
            for candidate_id in range(max_tracks):
                if candidate_id not in existing_ids:
                    track_id = candidate_id
                    break
            if track_id is None:
                continue

            self.hand_tracks[track_id] = {
                "id": int(track_id),
                "x": cx,
                "y": cy,
                "vx": 0.0,
                "vy": 0.0,
                "gesture": str(gesture_name) if gesture_name is not None else None,
                "last_ts": ts,
                "missing_frames": 0,
            }

            # 初始化该 track 的掌心轨迹，并根据轨迹判断是否为挥手
            self._update_palm_history(track_id, ts, cx, cy)
            if self._is_waving_track(track_id):
                self.hand_tracks[track_id]["gesture"] = "waving"

        # 未被匹配到的旧 track：只更新 missing_frames，必要时删除（惯性结束）
        to_delete: List[int] = []
        for track_id in unmatched_tracks:
            t = self.hand_tracks.get(track_id)
            if not t:
                continue
            t["missing_frames"] = int(t.get("missing_frames", 0)) + 1
            if t["missing_frames"] >= self._inertia_max_frames:
                to_delete.append(track_id)

        for track_id in to_delete:
            self.hand_tracks.pop(track_id, None)
            self.palm_center_history.pop(track_id, None)

        # 对于没有任何检测结果但仍然存在的 track，也要做一次基于轨迹的挥手更新
        for track_id, t in self.hand_tracks.items():
            if t.get("gesture") != "waving" and self._is_waving_track(track_id):
                t["gesture"] = "waving"

        # 更新全局轨迹时间戳
        self._last_tracks_ts = ts

    def _build_result_from_tracks(self, faces: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        根据当前 hand_tracks 构建统一的结果字典：
        - hands 列表：[{index(track_id), gesture, x, y, distance_m(None)}]
        - gesture / hand_center：兼容旧字段（取第一只手）
        """
        hands: List[Dict[str, Any]] = []
        # 按 track_id 排序，最多输出 4 只手
        # 过滤掉已丢失超过阈值的 track（手离开屏幕后应该及时消失）
        # 保留短暂丢失（missing_frames < 3）的 track，以便在短暂丢失检测时紫色框仍然保持稳定可见
        for track_id in sorted(self.hand_tracks.keys()):
            t = self.hand_tracks[track_id]
            missing_frames = int(t.get("missing_frames", 0))
            # 如果手已经丢失超过 3 帧（约 0.1 秒），不再显示该 track
            if missing_frames >= 3:
                continue
            hand_data = {
                "index": int(track_id),
                "gesture": t.get("gesture"),
                "x": float(t.get("x", 0.0)),
                "y": float(t.get("y", 0.0)),
                "distance_m": None,
            }
            hands.append(hand_data)
            if len(hands) >= 4:
                break

        gesture = hands[0]["gesture"] if hands and hands[0].get("gesture") else None
        hand_center = [hands[0]["x"], hands[0]["y"]] if hands else None

        result: Dict[str, Any] = {
            "faces": faces,
            "hands": hands,
            "gesture": gesture,
            "hand_center": hand_center,
        }
        return result

    def process_frame(self, rgb_frame: np.ndarray) -> Dict[str, Any]:
        """
        处理一帧图像（多手支持），并通过质心追踪器输出稳定的手部 track_id。
        
        Args:
            rgb_frame: RGB 图像 (H, W, 3)
            
        Returns:
            检测结果字典（包含 hands 列表）
        """
        # 转换为 RGB（MediaPipe 需要 RGB）
        rgb_image = cv2.cvtColor(rgb_frame, cv2.COLOR_BGR2RGB) if rgb_frame.shape[2] == 3 else rgb_frame
        now_ts = time.time()
        
        # 唇动检测（判断是否在说话）
        is_talking = self.lip_detector.process_frame(rgb_image)
        
        # 人脸检测
        face_results = self.face_detector.process(rgb_image)
        faces: List[Dict[str, Any]] = []
        if face_results.detections:
            h, w = rgb_frame.shape[:2]
            for idx, detection in enumerate(face_results.detections):
                bbox = detection.location_data.relative_bounding_box
                if bbox.width <= 0 or bbox.height <= 0:
                    continue
                    
                # 1. 提取置信度
                confidence = float(detection.score[0])
                
                # 🌟 2. 新增：提取关键点计算正面百分比（任务3：数学平滑与容错优化）
                keypoints = detection.location_data.relative_keypoints
                raw_frontal_percent = 0.0
                if len(keypoints) >= 3:
                    right_eye = keypoints[0]  # 右眼
                    left_eye = keypoints[1]   # 左眼
                    nose_tip = keypoints[2]   # 鼻尖
                    
                    # 🌟 任务3：使用两眼距离作为分母防除零（支持更远距离识别）
                    eye_dist = abs(right_eye.x - left_eye.x)
                    
                    if eye_dist > 0.001:
                        # 计算两眼中心点
                        eye_center_x = (right_eye.x + left_eye.x) / 2.0
                        # 计算鼻子偏离中心的程度（yaw_ratio）
                        offset = abs(nose_tip.x - eye_center_x)
                        yaw_ratio = offset / (eye_dist / 2.0)
                        
                        # 🌟 任务3：限制 yaw_ratio 的极端值在 -1.5 到 +1.5 之间
                        yaw_ratio = max(-1.5, min(1.5, yaw_ratio))
                        
                        # 计算正面分数
                        frontal_score = max(0.0, 1.0 - abs(yaw_ratio))
                        raw_frontal_percent = round(frontal_score * 100, 1)
                    else:
                        raw_frontal_percent = 0.0
                
                # 🌟 任务3：EMA滤波（在追踪逻辑中）
                face_id = idx + 1
                # EMA 平滑滤波：60%历史数据 + 40%最新数据，消除摄像头微抖动
                smoothed_frontal = 0.6 * self._face_frontal_history.get(face_id, raw_frontal_percent) + 0.4 * raw_frontal_percent
                frontal_percent = round(smoothed_frontal, 1)
                
                # 更新历史记录
                self._face_frontal_history[face_id] = frontal_percent
                
                # 3. 估算距离
                k_factor = 0.5 
                distance = round(k_factor / np.sqrt(bbox.height), 2)
                distance = min(max(distance, 0.3), 10.0)
                
                # 4. 构建结构化数据 (加入 frontal_percent)
                faces.append({
                    "id": idx + 1,
                    "x": float(bbox.xmin),
                    "y": float(bbox.ymin),
                    "w": float(bbox.width),
                    "h": float(bbox.height),
                    "confidence": round(confidence, 3),
                    "frontal_percent": frontal_percent,  # 🌟 真正的人脸正面度
                    "distance": distance,
                    "bbox": [
                        round(bbox.xmin, 3),
                        round(bbox.ymin, 3),
                        round(bbox.width, 3),
                        round(bbox.height, 3)
                    ]
                })

        # 记录最近一帧的人脸结果，供跳帧插值时复用
        self._last_faces = faces
        
        # 🌟 任务3：清理消失的人脸的历史记录（防止内存泄漏）
        current_face_ids = {face.get("id") for face in faces}
        disappeared_ids = set(self._face_frontal_history.keys()) - current_face_ids
        for face_id in disappeared_ids:
            del self._face_frontal_history[face_id]

        # 手部检测（MediaPipe Tasks GestureRecognizer，LIVE_STREAM 异步）
        # 如果 recognizer 尚未初始化成功，安全降级为"仅人脸检测"，hands 为空
        if self.recognizer is None:
            result = self._build_result_from_tracks(faces)
            result["is_talking"] = is_talking
            return result

        # 1) 推送当前帧到 recognizer（异步回调更新 self._latest_gesture_data）
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)
            # 使用 time.time_ns() 生成毫秒级时间戳，保证单调递增且精度更高
            ts_ms = int(time.time_ns() // 1_000_000)
            self.recognizer.recognize_async(mp_image, ts_ms)
        except Exception:
            # 异步推理失败不影响主循环（本帧仅输出 faces）
            result = self._build_result_from_tracks(faces)
            result["is_talking"] = is_talking
            return result

        # 2) 从异步缓存读取最近一次结果（可能滞后一帧；若尚未返回则为空列表）
        with self._gesture_lock:
            latest = self._latest_gesture_data if isinstance(self._latest_gesture_data, dict) else None

        if not isinstance(latest, dict):
            latest = {"timestamp_ms": None, "hands": []}

        hands_data = latest.get("hands")
        if hands_data is None or not isinstance(hands_data, list):
            hands_data = []

        detections: List[Dict[str, Any]] = []
        if hands_data:
            for hand in hands_data:
                if not isinstance(hand, dict):
                    continue
                landmarks = hand.get("landmarks")
                builtin_gesture = hand.get("builtin_gesture")
                gesture_name = self._detect_gesture(landmarks, builtin_gesture)
                hand_center = self._extract_hand_center(landmarks)
                if hand_center is None:
                    continue
                cx, cy = hand_center
                detections.append({"x": float(cx), "y": float(cy), "gesture": gesture_name})

        # 基于质心的手部追踪与挥手判定
        # 特殊处理：若当前帧未检测到任何手，但已有历史 track，则保持上一帧状态，
        # 不立即调用 _update_hand_tracks 增加 missing_frames，从而避免紫色框瞬间消失。
        if not detections and self.hand_tracks:
            return self._build_result_from_tracks(faces)

        self._update_hand_tracks(detections, now_ts)

        # 构建结果（距离字段由 run() 基于 Depth 补全）
        result = self._build_result_from_tracks(faces)
        # 添加唇动检测结果
        result["is_talking"] = is_talking
        return result
    
    def send_result(self, result: Dict[str, Any]):
        """
        通过 ZMQ PUB 发送检测结果（纯传感器数据，不包含业务逻辑）
        
        Args:
            result: 检测结果字典
        """
        try:
            is_talking = result.get("is_talking", False)
            # 🌟 调试：记录 is_talking 状态变化（仅在状态改变时打印）
            if not hasattr(self, '_last_sent_is_talking'):
                self._last_sent_is_talking = None
            if self._last_sent_is_talking != is_talking:
                vision_logger.info(f"👄 [VisionService] 发送 is_talking 状态: {self._last_sent_is_talking} → {is_talking}")
                self._last_sent_is_talking = is_talking
            
            # 只发送原始数据，不包含wake字段（业务逻辑在core_server中处理）
            zmq_data = {
                "faces": result.get("faces", []),
                "hands": result.get("hands", []),
                "is_talking": is_talking,
                "timestamp": time.time()
            }
            # 通过ZMQ PUB发送
            self._zmq_pub_socket.send_json(zmq_data)
        except Exception as e:
            vision_logger.error(f"Error sending ZMQ data: {e}")

    def send_frame_image_async(self, bgr_frame: np.ndarray):
        """
        异步发送当前帧图像（JPEG over UDP）。
        为了实时性：队列满则丢弃旧帧/新帧都可以，这里选择丢弃新帧。
        """
        if bgr_frame is None:
            return
        try:
            # 只保留最新：若满，直接丢弃
            self._img_queue.put_nowait((self._rgb_magic, self.udp_image_port, bgr_frame))
        except queue.Full:
            return

    def send_depth_image_async(self, depth_bgr: np.ndarray):
        """异步发送上色后的深度图（BGR JPEG over UDP）。"""
        if depth_bgr is None:
            return
        try:
            self._img_queue.put_nowait((self._dpt_magic, self.udp_depth_port, depth_bgr))
        except queue.Full:
            return

    @staticmethod
    def depth_to_colormap(depth_mm: np.ndarray, min_depth_mm: int = 500, max_depth_mm: int = 4000) -> np.ndarray:
        """服务端渲染深度伪彩（BGR）：500-4000mm 裁剪 + JET + 无效点黑。"""
        depth_clipped = np.clip(depth_mm, min_depth_mm, max_depth_mm)
        depth_normalized = (
            (depth_clipped - min_depth_mm) / float(max_depth_mm - min_depth_mm) * 255.0
        ).astype(np.uint8)
        depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
        depth_colormap[depth_mm == 0] = [0, 0, 0]
        return depth_colormap

    @staticmethod
    def _sample_depth_m(depth_mm: np.ndarray, x_norm: float, y_norm: float, box_px: int = 50) -> Optional[float]:
        """在深度图上按归一化坐标采样中位数距离（米）。"""
        if depth_mm is None:
            return None
        h, w = depth_mm.shape[:2]
        x = int(float(x_norm) * w)
        y = int(float(y_norm) * h)
        x = max(0, min(w - 1, x))
        y = max(0, min(h - 1, y))
        half = max(1, int(box_px // 2))
        x0 = max(0, x - half)
        y0 = max(0, y - half)
        x1 = min(w, x + half)
        y1 = min(h, y + half)
        roi = depth_mm[y0:y1, x0:x1]
        if roi.size == 0:
            return None
        valid = (roi >= 500) & (roi <= 4000) & (roi > 0)
        if not np.any(valid):
            return None
        median_mm = float(np.median(roi[valid]))
        return median_mm / 1000.0
    
    def run(self, camera_index: int = 0):
        """
        运行视觉服务（主循环）
        
        Args:
            camera_index: 相机索引
        """
        # 采集 RGB+Depth：使用 Gemini330Driver（避免 GUI 进程抢占设备）
        # 先设置日志级别（在创建 driver 之前，避免初始化时的 DEBUG 日志）
        try:
            from wakefusion.logging import get_logger
            camera_logger = get_logger("camera_driver")
            camera_logger.logger.setLevel(logging.ERROR)
            face_gate_logger = get_logger("face_gate")
            face_gate_logger.logger.setLevel(logging.ERROR)
        except Exception:
            # 如果导入失败，回退到标准日志库设置
            logging.getLogger("camera_driver").setLevel(logging.ERROR)
            logging.getLogger("face_gate").setLevel(logging.ERROR)
        
        try:
            from wakefusion.drivers import Gemini330Driver, CameraConfig
        except Exception as e:
            print(f"[ERROR] 无法导入 Gemini330Driver/CameraConfig：{e}")
            print("[ERROR] 当前版本的 vision_service.py 需要 Orbbec 驱动以采集 Depth。")
            return

        driver = Gemini330Driver(
            config=CameraConfig(
                rgb_width=640,
                rgb_height=480,
                rgb_fps=self.target_fps,
                depth_width=640,
                depth_height=480,
                depth_fps=self.target_fps,
                enable_rgb=True,
                enable_depth=True,
            ),
            callback=None,
        )

        print("启动相机（Gemini330Driver）...")
        driver.start()
        
        # 创建退出控制窗口（改善退出体验）
        cv2.namedWindow("VisionServiceControl", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("VisionServiceControl", 200, 100)
        # 在窗口内绘制提示文字（红色加粗，居中显示）
        control_img = np.zeros((100, 200, 3), dtype=np.uint8)
        # 使用醒目的红色文字 "CLICK HERE & PRESS Q"，居中显示
        text = "CLICK HERE & PRESS Q"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 2
        (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
        text_x = (200 - text_width) // 2
        text_y = (100 + text_height) // 2
        cv2.putText(control_img, text, (text_x, text_y), font, font_scale, (0, 0, 255), thickness, cv2.LINE_AA)
        cv2.imshow("VisionServiceControl", control_img)
        
        print(f"VisionService started, capturing from Gemini330Driver, camera_index={camera_index}")
        print("Press 'q' in VisionServiceControl window to quit")
        
        last_time = time.time()
        
        try:
            while True:
                frame_data = driver.capture_frame()
                if frame_data is None or frame_data.rgb is None:
                    time.sleep(0.005)
                    continue
                
                # 控制帧率
                current_time = time.time()
                elapsed = current_time - last_time
                if elapsed < self.frame_time:
                    time.sleep(self.frame_time - elapsed)
                last_time = time.time()
                
                rgb_rgb = frame_data.rgb  # Gemini330Driver 输出为 RGB
                depth_mm = frame_data.depth  # uint16 深度（mm）

                # GUI/发送：转 BGR（OpenCV/JPEG）
                bgr = cv2.cvtColor(rgb_rgb, cv2.COLOR_RGB2BGR)

                # 性能优化：MediaPipe 处理半速
                # - 偶数帧：运行 MediaPipe，更新质心追踪器与静态手势判定
                # - 奇数帧：不再做速度外推，仅复用上一帧检测结果，避免框抖动与"乱飞"
                # 注意：即使跳帧，也要更新 track 的 missing_frames，确保手离开后及时消失
                self._process_frame_counter += 1
                if self._process_frame_counter % 2 == 0 or not hasattr(self, "_last_result"):
                    # 处理帧（MediaPipe）
                    result = self.process_frame(bgr)
                    self._last_result = result
                else:
                    # 跳帧：复用上一帧的检测结果，不要去破坏原本的 track 状态
                    if hasattr(self, "_last_result") and self._last_result:
                        faces = self._last_result.get("faces", [])
                        # 直接基于现有的 track 重建结果，不用传空数组去清理
                        result = self._build_result_from_tracks(faces)
                        result["faces"] = faces
                        result["hand_distance_m"] = self._last_result.get("hand_distance_m")
                        result["distance_m"] = self._last_result.get("distance_m")
                        result["presence"] = self._last_result.get("presence", False)
                        result["confidence"] = self._last_result.get("confidence", 0.0)
                        # 继承唇动状态
                        result["is_talking"] = self._last_result.get("is_talking", False)
                        
                        self._last_result = result
                    else:
                        result = self.process_frame(bgr)
                        self._last_result = result

                # 深度采样：补全手部距离/人脸距离（供 face_gate 业务逻辑使用）
                hands = result.get("hands", []) if isinstance(result.get("hands", []), list) else []
                if depth_mm is not None and hands:
                    for hand in hands:
                        try:
                            hd = self._sample_depth_m(depth_mm, float(hand["x"]), float(hand["y"]), box_px=50)
                            hand["distance_m"] = float(hd) if hd is not None else None
                        except Exception:
                            hand["distance_m"] = None
                
                # 向后兼容：第一只手的距离
                hand_distance_m = hands[0].get("distance_m") if hands and hands[0].get("distance_m") is not None else None
                result["hand_distance_m"] = float(hand_distance_m) if hand_distance_m is not None else None

                faces = result.get("faces", []) if isinstance(result.get("faces", []), list) else []
                global_distance_m = None
                if depth_mm is not None and faces:
                    for f in faces:
                        try:
                            cx = float(f["x"]) + float(f["w"]) / 2.0
                            cy = float(f["y"]) + float(f["h"]) / 2.0
                            fd = self._sample_depth_m(depth_mm, cx, cy, box_px=60)
                            f["distance_m"] = float(fd) if fd is not None else None
                        except Exception:
                            f["distance_m"] = None
                    if faces and faces[0].get("distance_m") is not None:
                        global_distance_m = float(faces[0]["distance_m"])

                result["distance_m"] = float(global_distance_m) if global_distance_m is not None else None
                
                # 视觉唤醒状态机逻辑已移除，业务逻辑移至core_server
                
                # 发送结果（通过ZMQ PUB，纯传感器数据）
                self.send_result(result)
                
                # 发送 RGB 图像（JPEG，异步）- 每帧都发送以保持流畅
                self.send_frame_image_async(bgr)

                # 性能优化：深度图发送跳帧（每2帧发送一次，减少约50%的深度图处理负担）
                # 深度图主要用于可视化，降低频率不影响核心检测功能
                self._depth_send_counter += 1
                if depth_mm is not None and self._depth_send_counter % 2 == 0:
                    color_depth_bgr = self.depth_to_colormap(depth_mm)
                    self.send_depth_image_async(color_depth_bgr)
                
                # 'q' 键退出检测
                # 更新退出控制窗口（保持窗口可见）
                cv2.imshow("VisionServiceControl", control_img)
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
        
        except KeyboardInterrupt:
            print("\nVisionService stopped by user")
        finally:
            try:
                driver.stop()
            except Exception:
                pass
            cv2.destroyAllWindows()
            # 关闭ZMQ socket
            try:
                self._zmq_pub_socket.close()
                self.zmq_context.term()
            except Exception:
                pass
            try:
                self._img_thread_stop.set()
                # 给线程一点时间退出（不 join 也可，daemon=True）
                self.udp_image_socket.close()
                self.udp_depth_socket.close()
            except Exception:
                pass
            # 安全关闭 MediaPipe 检测器
            if self.face_detector is not None:
                try:
                    self.face_detector.close()
                except Exception as e:
                    print(f"[WARNING] 关闭 face_detector 时出错: {e}")
            if self.recognizer is not None:
                try:
                    self.recognizer.close()
                except Exception as e:
                    print(f"[WARNING] 关闭 GestureRecognizer 时出错: {e}")
            if self.lip_detector is not None:
                try:
                    self.lip_detector.close()
                except Exception as e:
                    print(f"[WARNING] 关闭 LipSyncDetector 时出错: {e}")
            print("VisionService cleaned up")


def main():
    """主函数"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Vision Service with MediaPipe")
    parser.add_argument("--host", type=str, default="127.0.0.1", help="UDP host")
    parser.add_argument("--port", type=int, default=9999, help="UDP port")
    parser.add_argument("--img-port", type=int, default=10000, help="UDP image port (JPEG)")
    parser.add_argument("--depth-port", type=int, default=10001, help="UDP depth image port (JPEG)")
    parser.add_argument("--fps", type=int, default=30, help="Target FPS")
    parser.add_argument("--camera", type=int, default=0, help="Camera index")
    parser.add_argument("--jpeg-quality", type=int, default=55, help="JPEG quality (0-100), default 55 for better performance")
    
    args = parser.parse_args()
    
    service = VisionService(
        target_fps=args.fps,
        jpeg_quality=args.jpeg_quality
    )
    
    service.run(camera_index=args.camera)


if __name__ == "__main__":
    main()
