"""
WakeFusion 可视化测试脚本
使用 OpenCV 创建实时 GUI 监控界面，显示三层可视化看板

功能：
- 主视图：RGB 彩色画面
- 画中画：伪彩色深度图（Depth Heatmap）
- 层1 - 硬件监控：实时 FPS、设备状态
- 层2 - 感知数据：准星、距离显示
- 层3 - 业务逻辑：唤醒状态边框

按键控制：
- q: 退出程序
- c: 触发基准环境校准（预留）
"""

import cv2
import numpy as np
import time
import socket
import json
import zmq
from collections import deque
from typing import Optional, Dict, Any, Tuple

# 导入 WakeFusion 组件（业务门控）
from wakefusion.workers import FaceGateWorker, FaceGateConfig
from wakefusion.types import VisionFrame
from wakefusion.config import get_config


# ============================================================================
# 配置参数
# ============================================================================

# UDP 配置（方案B：GUI 仅接收，不占用摄像头）
UDP_HOST = "127.0.0.1"
UDP_JSON_PORT = 9999     # FaceGate / 旧版视觉 JSON（坐标/手势）
UDP_IMG_PORT = 10000     # vision_service.py 发送 JPEG 图像（分包）
UDP_DEPTH_PORT = 10001   # vision_service.py 发送深度彩色图（分包 JPEG）

# ZMQ 配置：从全局配置中读取 Vision PUB 端口，避免端口冲突/硬编码
_CONFIG = get_config()
VISION_PUB_PORT = _CONFIG.zmq.vision_pub_port

# 显示配置
WINDOW_NAME = "WakeFusion Vision Monitor"
PIP_SIZE = (320, 200)        # 画中画尺寸
PIP_MARGIN = 10              # 画中画边距
FPS_HISTORY_SIZE = 10        # FPS 平滑计算历史帧数

# 颜色定义（BGR 格式）
COLOR_GREEN = (0, 255, 0)
COLOR_RED = (0, 0, 255)
COLOR_YELLOW = (0, 255, 255)
COLOR_CYAN = (255, 255, 0)  # 青色（用于人脸框）
COLOR_MAGENTA = (255, 0, 255)  # 紫红（用于手势框）
COLOR_WHITE = (255, 255, 255)
COLOR_GRAY = (128, 128, 128)
COLOR_BLACK = (0, 0, 0)


# ============================================================================
# 可视化辅助函数
# ============================================================================

def draw_transparent_rect(frame: np.ndarray, x: int, y: int, w: int, h: int, 
                          color: tuple, alpha: float = 0.6):
    """
    绘制半透明矩形
    
    Args:
        frame: 图像帧
        x, y: 左上角坐标
        w, h: 宽度和高度
        color: BGR 颜色
        alpha: 透明度 (0-1)
    """
    overlay = frame.copy()
    cv2.rectangle(overlay, (x, y), (x + w, y + h), color, -1)
    cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)


def draw_crosshair(frame: np.ndarray, center_x: int, center_y: int, 
                   size: int = 40, color: tuple = COLOR_WHITE, thickness: int = 2):
    """
    绘制十字准星
    
    Args:
        frame: 图像帧
        center_x, center_y: 中心坐标
        size: 准星大小
        color: BGR 颜色
        thickness: 线宽
    """
    half = size // 2
    gap = 8  # 中心空隙
    
    # 水平线（左右两段）
    cv2.line(frame, (center_x - half, center_y), (center_x - gap, center_y), color, thickness)
    cv2.line(frame, (center_x + gap, center_y), (center_x + half, center_y), color, thickness)
    
    # 垂直线（上下两段）
    cv2.line(frame, (center_x, center_y - half), (center_x, center_y - gap), color, thickness)
    cv2.line(frame, (center_x, center_y + gap), (center_x, center_y + half), color, thickness)
    
    # 中心小圆
    cv2.circle(frame, (center_x, center_y), 3, color, -1)


def depth_to_colormap(depth: np.ndarray, min_depth_mm: int = 500, 
                      max_depth_mm: int = 4000) -> np.ndarray:
    """
    将深度图转换为伪彩色图（Viewer 级质量）
    
    Args:
        depth: 16位深度图（单位：毫米）
        min_depth_mm: 最小深度值（默认 500mm）
        max_depth_mm: 最大深度值（默认 4000mm）
        
    Returns:
        BGR 伪彩色图像
    """
    # 裁剪到有效范围（确保细节像 Viewer 一样丰富）
    depth_clipped = np.clip(depth, min_depth_mm, max_depth_mm)
    
    # 归一化到 0-255（8位）
    # 固定映射逻辑：depth_normalized = ((depth_clipped - 500) / (4000 - 500) * 255)
    # 这样 0.5m 对应 0 (Blue)，4.0m 对应 255 (Red)
    depth_normalized = ((depth_clipped - min_depth_mm) / (max_depth_mm - min_depth_mm) * 255).astype(np.uint8)
    
    # 应用 Jet 色图（Viewer 同款）
    # 色彩区间：0.5m(深蓝) -> 1.5m(青/绿) -> 2.5m(黄) -> 4.0m(红)
    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
    
    # 将无效点强制设为黑色
    depth_colormap[depth == 0] = [0, 0, 0]
    
    return depth_colormap


def put_text_with_bg(frame: np.ndarray, text: str, pos: tuple, 
                     font_scale: float = 0.6, color: tuple = COLOR_WHITE,
                     bg_color: tuple = COLOR_BLACK, thickness: int = 1):
    """
    绘制带背景的文字
    
    Args:
        frame: 图像帧
        text: 文字内容
        pos: 文字位置 (x, y)
        font_scale: 字体大小
        color: 文字颜色
        bg_color: 背景颜色
        thickness: 字体粗细
    """
    font = cv2.FONT_HERSHEY_SIMPLEX
    (text_width, text_height), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    
    x, y = pos
    # 绘制背景矩形
    cv2.rectangle(frame, (x - 2, y - text_height - 2), 
                  (x + text_width + 2, y + baseline + 2), bg_color, -1)
    # 绘制文字
    cv2.putText(frame, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)


def render_face_box(frame: np.ndarray, faces: list, distance_m: Optional[float] = None):
    """
    绘制人脸检测框（青色追踪框）和距离信息
    
    Args:
        frame: 图像帧（BGR 格式）
        faces: 人脸列表，每个元素为字典，包含：
            - xmin, ymin: 左上角相对坐标 (0-1)
            - width, height: 宽高相对值 (0-1)
            - confidence: 置信度 (0-1)
            - distance_m: 距离（米），如果存在则显示各自的距离
        distance_m: 全局距离（米），如果 faces 中没有 distance_m 则使用此值（向后兼容）
    """
    if not faces:
        return
    
    h, w = frame.shape[:2]
    
    for face in faces:
        # 兼容两种字段命名：
        # - 旧：xmin/ymin/width/height
        # - 新（vision_service）：x/y/w/h
        xmin_rel = face.get("xmin", face.get("x", 0.0))
        ymin_rel = face.get("ymin", face.get("y", 0.0))
        w_rel = face.get("width", face.get("w", 0.0))
        h_rel = face.get("height", face.get("h", 0.0))

        # 将相对坐标转换为绝对像素坐标
        xmin = int(float(xmin_rel) * w)
        ymin = int(float(ymin_rel) * h)
        xmax = int((float(xmin_rel) + float(w_rel)) * w)
        ymax = int((float(ymin_rel) + float(h_rel)) * h)
        
        # 绘制青色矩形框
        cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), COLOR_CYAN, 2)
        
        # 显示距离标签（优先使用每张人脸各自的距离，否则使用全局距离）
        face_distance = face.get('distance_m', distance_m)
        if face_distance is not None:
            distance_text = f"Dist: {face_distance:.2f}m"
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.7
            thickness = 2
            (text_width, text_height), baseline = cv2.getTextSize(
                distance_text, font, font_scale, thickness
            )
            
            # 计算标签位置（框的正上方，居中）
            face_center_x = (xmin + xmax) // 2
            label_x = face_center_x - text_width // 2
            label_y = max(ymin - 15, text_height + 10)
            label_w = text_width + 10
            label_h = text_height + baseline + 10
            
            # 绘制半透明背景
            draw_transparent_rect(frame, label_x - 5, label_y - label_h, label_w, label_h, COLOR_BLACK, 0.7)
            
            # 绘制距离文字（青色，加粗）
            cv2.putText(
                frame,
                distance_text,
                (label_x, label_y - 5),
                font,
                font_scale,
                COLOR_CYAN,
                thickness,
                cv2.LINE_AA
            )
        
        # 显示置信度（在框的顶部，如果距离已显示则放在距离下方）
        conf_percent = int(float(face.get("confidence", 0.0)) * 100)
        conf_text = f"Face: {conf_percent}%"
        
        # 计算文字尺寸
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        thickness = 1
        (text_width, text_height), baseline = cv2.getTextSize(
            conf_text, font, font_scale, thickness
        )
        
        # 绘制文字背景
        text_x = xmin
        if face_distance is not None:
            # 如果有距离，置信度显示在距离下方
            text_y = ymin + text_height + 25
        else:
            text_y = max(ymin - 5, text_height + 5)
        
        cv2.rectangle(
            frame,
            (text_x - 2, text_y - text_height - 2),
            (text_x + text_width + 2, text_y + baseline + 2),
            COLOR_BLACK,
            -1
        )
        
        # 绘制文字
        cv2.putText(
            frame,
            conf_text,
            (text_x, text_y),
            font,
            font_scale,
            COLOR_CYAN,
            thickness,
            cv2.LINE_AA
        )


def render_hand_gesture(
    frame: np.ndarray,
    gesture: Optional[str],
    hand_center: Optional[tuple],
    hand_distance_m: Optional[float],
    hand_index: int = 0
):
    """
    渲染手势可视化（紫色框和标签，支持多手）
    
    Args:
        frame: 图像帧
        gesture: 手势名称（thumbs_up, ok, waving, fist）
        hand_center: 手部中心坐标（归一化，0.0-1.0）
        hand_distance_m: 手部距离（米）
        hand_index: 手部索引（0-3，用于多手显示）
    """
    if hand_center is None:
        return
    
    h, w = frame.shape[:2]
    
    # 将归一化坐标转换为像素坐标，先对 hand_center 做合法性检查（限制在 0-1 范围内）
    try:
        x_norm = float(hand_center[0])
        y_norm = float(hand_center[1])
    except (TypeError, ValueError, IndexError):
        return

    # 如果 x 或 y 不在 0-1 之间，将其修正到边缘
    x_norm = max(0.0, min(1.0, x_norm))
    y_norm = max(0.0, min(1.0, y_norm))

    hand_x = int(x_norm * w)
    hand_y = int(y_norm * h)
    
    # 确保坐标在有效范围内
    hand_x = max(0, min(w - 1, hand_x))
    hand_y = max(0, min(h - 1, hand_y))
    
    # 绘制手部区域框（紫色，100x100 像素区域）
    box_size = 100
    half_size = box_size // 2
    
    xmin = max(0, hand_x - half_size)
    ymin = max(0, hand_y - half_size)
    xmax = min(w, hand_x + half_size)
    ymax = min(h, hand_y + half_size)
    
    # 显示手势标签和距离（在紫色框顶部，格式：Hand [ID]: [Gesture] | [Dist]m）
    # 增强手势反馈：
    #   - 'ok'：紫色框加粗并显示 "🔥 OK - READY"
    #   - 'waving'：在紫色框上方显示动态字符 "WAVING..."（提示挥手检测）
    #   - 其他手势：显示规范化名称，例如 "Thumbs Up"
    if gesture == "ok":
        gesture_text = "🔥 OK - READY"
        box_thickness = 4  # 加粗紫色框
    elif gesture == "waving":
        gesture_text = "WAVING..."  # 特殊动画字符提示挥手
        box_thickness = 3
    elif gesture:
        gesture_text = gesture.replace("_", " ").title()  # "thumbs_up" -> "Thumbs Up"
        box_thickness = 2
    else:
        gesture_text = "Tracking Hand..."  # 若无手势名则显示 "Tracking Hand..."
        box_thickness = 2
    
    # 绘制紫色矩形框（根据手势类型调整粗细）
    cv2.rectangle(frame, (xmin, ymin), (xmax, ymax), COLOR_MAGENTA, box_thickness)
    
    label_parts = [f"Hand {hand_index}", gesture_text]
    if hand_distance_m is not None:
        label_parts.append(f"{hand_distance_m:.2f}m")
    label_text = " | ".join(label_parts)
    
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.6
    thickness = 2
    (text_width, text_height), baseline = cv2.getTextSize(
        label_text, font, font_scale, thickness
    )
    
    # 计算标签位置（框的正上方，居中）
    box_center_x = (xmin + xmax) // 2
    label_x = box_center_x - text_width // 2
    label_y = max(ymin - 15, text_height + 10)
    label_w = text_width + 10
    label_h = text_height + baseline + 10
    
    # 绘制半透明背景
    draw_transparent_rect(frame, label_x - 5, label_y - label_h, label_w, label_h, COLOR_BLACK, 0.7)
    
    # 绘制手势文字（紫色，加粗）
    cv2.putText(
        frame,
        label_text,
        (label_x, label_y - 5),
        font,
        font_scale,
        COLOR_MAGENTA,
        thickness,
        cv2.LINE_AA
    )
    
    # 渲染消隐：如果手势名为 None，在紫色框中心显示 "Tracking..." 提示
    if gesture is None:
        tracking_text = "Tracking..."
        (tracking_width, tracking_height), _ = cv2.getTextSize(
            tracking_text, font, font_scale, thickness
        )
        tracking_x = box_center_x - tracking_width // 2
        tracking_y = (ymin + ymax) // 2
        # 绘制半透明背景
        draw_transparent_rect(frame, tracking_x - 5, tracking_y - tracking_height - 5, 
                             tracking_width + 10, tracking_height + 10, COLOR_BLACK, 0.7)
        # 绘制 "Tracking..." 文字（黄色，居中）
        cv2.putText(
            frame,
            tracking_text,
            (tracking_x, tracking_y),
            font,
            font_scale,
            (0, 255, 255),  # 黄色
            thickness,
            cv2.LINE_AA,
        )
    
    # 在手部中心位置绘制一个紫色圆点（标记手部位置）
    cv2.circle(frame, (hand_x, hand_y), 5, COLOR_MAGENTA, -1)


# ============================================================================
# UDP 接收（方案B：Producer-Consumer）
# ============================================================================

class UdpVisionReceiver:
    """
    接收 vision_service.py 发送的数据：
    - RGB JPEG（10000）：分包图像
    - 深度彩色 JPEG（10001）：分包图像
    """

    _RGB_MAGIC = b"IMG1"
    _DPT_MAGIC = b"DPT1"
    _IMG_HEADER_LEN = 12  # magic(4) + frame_id(u32) + chunk_idx(u16) + total(u16)

    def __init__(self, host: str = UDP_HOST, img_port: int = UDP_IMG_PORT, depth_port: int = UDP_DEPTH_PORT):
        self.host = host
        self.img_port = int(img_port)
        self.depth_port = int(depth_port)

        self.sock_img = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_img.bind((self.host, self.img_port))
        self.sock_img.setblocking(False)

        self.sock_depth = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        self.sock_depth.bind((self.host, self.depth_port))
        self.sock_depth.setblocking(False)

        # 图像重组缓冲：frame_id -> {total:int, chunks:dict[int,bytes], ts:float}
        self._rgb_frames: Dict[int, Dict[str, Any]] = {}
        self._dpt_frames: Dict[int, Dict[str, Any]] = {}
        self._last_rgb_bgr: Optional[np.ndarray] = None
        self._last_depth_bgr: Optional[np.ndarray] = None

        # 记录最近一次成功解码图像的时间，用于检测 VisionService 是否已退出
        now = time.time()
        self._last_rgb_update_ts: float = now
        self._last_depth_update_ts: float = now

    def close(self):
        try:
            self.sock_img.close()
        except Exception:
            pass
        try:
            self.sock_depth.close()
        except Exception:
            pass

    def _poll_stream(
        self,
        sock: socket.socket,
        expected_magic: bytes,
        frames_buf: Dict[int, Dict[str, Any]],
        last_img: Optional[np.ndarray],
        is_rgb: bool,
        max_packets: int = 200,
        expire_sec: float = 0.5
    ) -> Optional[np.ndarray]:
        now = time.time()
        for _ in range(max_packets):
            try:
                pkt, _addr = sock.recvfrom(65535)
            except BlockingIOError:
                break
            except Exception:
                break
            if len(pkt) < self._IMG_HEADER_LEN:
                continue
            if pkt[:4] != expected_magic:
                continue

            frame_id = int.from_bytes(pkt[4:8], "little", signed=False)
            chunk_idx = int.from_bytes(pkt[8:10], "little", signed=False)
            total = int.from_bytes(pkt[10:12], "little", signed=False)
            payload = pkt[12:]

            if total <= 0 or chunk_idx >= total:
                continue

            buf = frames_buf.get(frame_id)
            if buf is None:
                buf = {"total": total, "chunks": {}, "ts": now}
                frames_buf[frame_id] = buf
            if buf["total"] != total:
                frames_buf.pop(frame_id, None)
                continue

            buf["chunks"][chunk_idx] = payload
            buf["ts"] = now

            if len(buf["chunks"]) == total:
                data_bytes = b"".join(buf["chunks"][i] for i in range(total))
                arr = np.frombuffer(data_bytes, dtype=np.uint8)
                img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
                if img is not None:
                    last_img = img
                    # 更新最近一次图像更新时间戳
                    if is_rgb:
                        self._last_rgb_update_ts = now
                    else:
                        self._last_depth_update_ts = now
                frames_buf.pop(frame_id, None)

        if frames_buf:
            expired = [fid for fid, b in frames_buf.items() if now - float(b.get("ts", now)) > expire_sec]
            for fid in expired:
                frames_buf.pop(fid, None)

        return last_img

    def poll(self) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        """非阻塞轮询：返回（最新 RGB BGR, 最新 Depth-Color BGR）。"""
        self._last_rgb_bgr = self._poll_stream(
            self.sock_img, self._RGB_MAGIC, self._rgb_frames, self._last_rgb_bgr, True
        )
        self._last_depth_bgr = self._poll_stream(
            self.sock_depth, self._DPT_MAGIC, self._dpt_frames, self._last_depth_bgr, False
        )
        return self._last_rgb_bgr, self._last_depth_bgr


# ============================================================================
# 三层可视化看板
# ============================================================================

def render_layer1_hardware(
    frame: np.ndarray,
    fps: float,
    device_status: str,
    is_valid: bool = False,
    is_talking: bool = False,
):
    """
    层1 - 硬件监控（右上角，简洁样式）
    显示 FPS、Wake 状态和唇动检测状态
    
    Args:
        frame: 图像帧
        fps: 当前帧率
        device_status: 设备状态字符串
        is_valid: 是否处于唤醒就绪状态
        is_talking: 唇动检测结果（来自 vision_service 的 is_talking）
    """
    h, w = frame.shape[:2]
    
    # 右上角半透明背景框（更简洁）
    box_w, box_h = 200, 80
    box_x, box_y = w - box_w - 10, 10
    draw_transparent_rect(frame, box_x, box_y, box_w, box_h, COLOR_BLACK, 0.7)
    
    # FPS 显示
    fps_color = COLOR_GREEN if fps >= 15 else COLOR_YELLOW if fps >= 10 else COLOR_RED
    fps_text = f"FPS: {fps:.1f}"
    cv2.putText(frame, fps_text, (box_x + 10, box_y + 25), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, fps_color, 2, cv2.LINE_AA)
    
    # Wake Status（简洁显示）
    if is_valid:
        status_text = "Wake: READY"
        status_color = COLOR_GREEN
    else:
        status_text = "Wake: STANDBY"
        status_color = COLOR_YELLOW
    
    cv2.putText(frame, status_text, (box_x + 10, box_y + 50), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, status_color, 1, cv2.LINE_AA)

    # 唇动检测状态
    lip_text = "Lip: TALKING" if is_talking else "Lip: SILENT"
    lip_color = COLOR_GREEN if is_talking else COLOR_GRAY
    cv2.putText(frame, lip_text, (box_x + 10, box_y + 70),
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, lip_color, 1, cv2.LINE_AA)


def render_layer2_perception(frame: np.ndarray, has_presence: bool, 
                             distance_m: Optional[float], confidence: float):
    """
    层2 - 感知数据（中心区域）
    显示准星和距离信息
    
    Args:
        frame: 图像帧
        has_presence: 是否检测到人
        distance_m: 距离（米）
        confidence: 置信度
    """
    h, w = frame.shape[:2]
    center_x, center_y = w // 2, h // 2
    
    if has_presence and distance_m is not None:
        # 检测到人 - 显示绿色准星和距离
        crosshair_color = COLOR_GREEN
        draw_crosshair(frame, center_x, center_y, size=60, color=crosshair_color, thickness=2)
        
        # 显示距离
        distance_text = f"{distance_m:.2f}m"
        put_text_with_bg(frame, distance_text, (center_x + 40, center_y - 10), 
                         font_scale=1.0, color=COLOR_GREEN, thickness=2)
        
        # 显示置信度（小字）
        conf_text = f"Conf: {confidence:.0%}"
        put_text_with_bg(frame, conf_text, (center_x + 40, center_y + 20), 
                         font_scale=0.5, color=COLOR_WHITE)
    else:
        # 未检测到人 - 显示灰色准星和扫描状态
        draw_crosshair(frame, center_x, center_y, size=60, color=COLOR_GRAY, thickness=1)
        put_text_with_bg(frame, "Scanning...", (center_x + 40, center_y), 
                         font_scale=0.7, color=COLOR_GRAY)


def render_layer3_business(frame: np.ndarray, is_valid: bool, 
                           has_presence: bool, distance_m: Optional[float],
                           faces: Optional[list] = None):
    """
    层3 - 业务逻辑（全局边框）
    显示唤醒状态
    
    Args:
        frame: 图像帧
        is_valid: 是否在唤醒区（基于深度门控和人脸检测结果）
        has_presence: 是否检测到人
        distance_m: 距离（米）
        faces: 人脸列表（可选）
    """
    h, w = frame.shape[:2]
    border_thickness = 8
    
    has_face = faces is not None and len(faces) > 0
    
    if is_valid:
        # 绿色状态 (Ready)：有人 + 距离 < 4m + 有正脸检测结果
        border_color = COLOR_GREEN
        status_text = "WAKE READY"
        status_color = COLOR_GREEN
    elif has_presence and distance_m is not None and distance_m <= 4.0 and not has_face:
        # 黄色状态 (Warning)：有人 + 距离 < 4m + 无正脸检测结果
        border_color = COLOR_YELLOW
        status_text = "LOOK AT CAMERA (Face Required)"
        status_color = COLOR_YELLOW
    elif has_presence and distance_m is not None and distance_m > 4.0:
        # 红色状态 (Standby)：距离过远
        border_color = COLOR_RED
        status_text = "STANDBY (Too Far)"
        status_color = COLOR_RED
    else:
        # 无人或距离不合适 - 不绘制边框
        return
    
    # 绘制边框
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), border_color, border_thickness)
    
    # 顶部居中显示状态
    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 1.2
    thickness = 3
    (text_width, text_height), _ = cv2.getTextSize(status_text, font, font_scale, thickness)
    
    text_x = (w - text_width) // 2
    text_y = 50
    
    # 背景
    draw_transparent_rect(frame, text_x - 15, text_y - text_height - 10, 
                          text_width + 30, text_height + 20, COLOR_BLACK, 0.7)
    
    # 文字
    cv2.putText(frame, status_text, (text_x, text_y), font, font_scale, 
                status_color, thickness, cv2.LINE_AA)


def render_pip_depth(frame: np.ndarray, color_depth: Optional[np.ndarray] = None):
    """
    渲染画中画深度图（直接显示驱动返回的上色深度帧）
    
    Args:
        frame: 主图像帧（BGR 格式）
        color_depth: 上色后的深度图（BGR 格式），由驱动直接提供
    """
    if color_depth is None:
        return
    
    h, w = frame.shape[:2]
    pip_w, pip_h = PIP_SIZE
    pip_x = w - pip_w - PIP_MARGIN
    pip_y = h - pip_h - PIP_MARGIN
    
    # 直接缩放到画中画尺寸（驱动已经提供了完美的上色画面，无需再做 clip 或 applyColorMap）
    depth_resized = cv2.resize(color_depth, (pip_w, pip_h))
    
    # 添加边框
    cv2.rectangle(depth_resized, (0, 0), (pip_w - 1, pip_h - 1), COLOR_WHITE, 2)
    
    # 添加标签
    cv2.putText(depth_resized, "Depth", (10, 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)
    
    # 叠加到主帧
    frame[pip_y:pip_y + pip_h, pip_x:pip_x + pip_w] = depth_resized


# ============================================================================
# 主程序
# ============================================================================

def main():
    """主程序入口"""
    # WakeFusion Vision Monitor
    # 控制键: q - 退出程序, c - 触发基准环境校准（预留）
    # 正在初始化（方案B：UDP 接收模式，不占用摄像头）...
    receiver = UdpVisionReceiver(host=UDP_HOST, img_port=UDP_IMG_PORT, depth_port=UDP_DEPTH_PORT)

    # ZMQ 订阅：直接从 vision_service 的 PUB 通道获取原始感知结果（含唇动 is_talking）
    zmq_ctx = zmq.Context()
    vision_sub = zmq_ctx.socket(zmq.SUB)
    vision_sub.connect(f"tcp://127.0.0.1:{VISION_PUB_PORT}")
    # 订阅全部主题（vision_service 发送的是纯 JSON）
    vision_sub.setsockopt_string(zmq.SUBSCRIBE, "")
    vision_sub.setsockopt(zmq.RCVTIMEO, 0)  # 非阻塞

    latest_is_talking = False

    # FaceGate（业务逻辑：从 UDP JSON 读取 faces/gesture/距离，输出 valid 状态）
    face_gate = FaceGateWorker(
        config=FaceGateConfig(
            distance_m_max=4.0,
            distance_m_min=0.5,
            enable_face_detection=True,
            udp_port=UDP_JSON_PORT,
            face_conf_min=0.55,
            enable_depth_gate=True,
        ),
        event_callback=None,
    )
    face_gate.start()
    
    # FPS 计算
    fps_history = deque(maxlen=FPS_HISTORY_SIZE)
    last_frame_time = time.time()
    
    # 帧计数
    total_frames = 0
    
    try:
        # 创建窗口
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(WINDOW_NAME, 1280, 800)
        
        # 等待 vision_service.py 图像流（UDP）...

        # 性能优化：15FPS 跳帧显示（如果检测到卡顿）
        # 通过跳帧显示来保证后端算法的实时性
        last_render_time = time.time()
        frame_skip_counter = 0
        
        # 主循环（仅接收 + 渲染，不做本地推理）
        # 如果 VisionService 已通过 VisionServiceControl 窗口按 'q' 退出，
        # 将在若干秒内检测到图像停止更新并自动关闭 GUI。
        while True:
            # 先尝试从 ZMQ 读取最新的视觉感知结果（含唇动 is_talking）
            try:
                while True:
                    msg = vision_sub.recv_json(flags=zmq.NOBLOCK)
                    # vision_service.py send_result 中的字段：
                    # {"wake": bool, "faces": [...], "hands": [...], "is_talking": bool, "timestamp": float}
                    latest_is_talking = bool(msg.get("is_talking", False))
            except zmq.Again:
                pass

            img_bgr, depth_bgr = receiver.poll()

            # 若长时间未收到新的 RGB 帧，认为 VisionService 已退出，自动结束 GUI
            now_ts = time.time()
            if img_bgr is not None and hasattr(receiver, "_last_rgb_update_ts"):
                if now_ts - float(receiver._last_rgb_update_ts) > 3.0:
                    # 连接中断超过 3 秒，干净退出所有后台组件
                    # 不再在控制台打印，改为在画面上给出退出提示
                    warning_frame = np.zeros((800, 1280, 3), dtype=np.uint8)
                    cv2.putText(
                        warning_frame,
                        "VisionService stopped (no new frames), closing GUI...",
                        (60, 380),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.8,
                        COLOR_YELLOW,
                        2,
                        cv2.LINE_AA,
                    )
                    cv2.imshow(WINDOW_NAME, warning_frame)
                    cv2.waitKey(1500)
                    break
            
            # 性能优化：15FPS 跳帧显示
            current_time = time.time()
            elapsed = current_time - last_render_time
            target_fps = 15.0  # 15FPS
            target_interval = 1.0 / target_fps
            
            # 如果帧间隔太短（卡顿），跳过渲染
            if elapsed < target_interval:
                frame_skip_counter += 1
                continue
            
            last_render_time = current_time
            
            total_frames += 1
            
            # 计算 FPS
            frame_interval = current_time - last_frame_time
            last_frame_time = current_time
            
            if frame_interval > 0:
                fps_history.append(1.0 / frame_interval)
            current_fps = np.mean(fps_history) if fps_history else 0
            
            # 准备显示帧
            if img_bgr is not None:
                display_frame = img_bgr.copy()
            else:
                display_frame = np.zeros((800, 1280, 3), dtype=np.uint8)
                cv2.putText(
                    display_frame,
                    "Connecting to Vision Service...",
                    (300, 360),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    1.1,
                    COLOR_YELLOW,
                    2,
                    cv2.LINE_AA,
                )
                params = f"JSON:{UDP_HOST}:{UDP_JSON_PORT}  RGB:{UDP_HOST}:{UDP_IMG_PORT}  DEPTH:{UDP_HOST}:{UDP_DEPTH_PORT}"
                cv2.putText(
                    display_frame,
                    params,
                    (170, 420),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    COLOR_WHITE,
                    1,
                    cv2.LINE_AA,
                )
            
            # 确保帧尺寸正确
            if display_frame.shape[:2] != (800, 1280):
                display_frame = cv2.resize(display_frame, (1280, 800))
            
            # 性能优化：15FPS 跳帧显示（如果检测到卡顿）
            # 通过跳帧显示来保证后端算法的实时性
            if not hasattr(main, '_last_render_time'):
                main._last_render_time = time.time()
                main._frame_skip_counter = 0
            
            current_time = time.time()
            elapsed = current_time - main._last_render_time
            target_fps = 15.0  # 15FPS
            target_interval = 1.0 / target_fps
            
            # 如果帧间隔太短（卡顿），跳过渲染
            if elapsed < target_interval:
                main._frame_skip_counter += 1
                continue
            
            main._last_render_time = current_time
            
            # FaceGate 处理（从 UDP JSON 获取人脸/手势/距离，输出 valid）
            frame_data = VisionFrame(ts=time.time())
            gate_result = face_gate.process_frame(frame_data)

            has_presence = gate_result.presence if gate_result else False
            distance_m = gate_result.distance_m if gate_result else None
            confidence = gate_result.confidence if gate_result else 0.0
            is_valid = gate_result.valid if gate_result else False

            faces = frame_data.faces if frame_data.faces else []
            # 严谨判定：确保 hands 存在且为列表且非空
            hands = []
            if hasattr(frame_data, 'hands') and isinstance(frame_data.hands, list) and len(frame_data.hands) > 0:
                hands = frame_data.hands
            
            # ============ 渲染三层看板 ============
            
            # 层3 - 业务逻辑边框（先绘制，作为底层）
            render_layer3_business(display_frame, is_valid, has_presence, distance_m, faces)
            
            # 交互提示：在主循环的 display_frame 左上角，绘制一条常驻的文字提示
            cv2.putText(display_frame, "GUI Focused: Press Q to Exit All", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2, cv2.LINE_AA)
            
            # 动态距离显示逻辑
            if faces:
                # 如果检测到人脸，在人脸框上显示距离，不再显示中心准星
                render_face_box(display_frame, faces, distance_m)
            
            # 多手循环渲染（每只手独立显示）
            for hand in hands:
                if isinstance(hand, dict):
                    hand_index = hand.get("index", 0)
                    hand_gesture = hand.get("gesture")
                    hand_x = hand.get("x")
                    hand_y = hand.get("y")
                    hand_dist = hand.get("distance_m")
                    if hand_x is not None and hand_y is not None:
                        render_hand_gesture(
                            display_frame,
                            hand_gesture,
                            (float(hand_x), float(hand_y)),
                            hand_dist,
                            hand_index
                        )
            
            if not faces:
                # 如果没有检测到人脸，在屏幕中心显示 "Scanning..."
                h, w = display_frame.shape[:2]
                center_x, center_y = w // 2, h // 2
                scanning_text = "Scanning..."
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 1.0
                thickness = 2
                (text_width, text_height), baseline = cv2.getTextSize(
                    scanning_text, font, font_scale, thickness
                )
                text_x = center_x - text_width // 2
                text_y = center_y
                
                # 绘制文字背景
                cv2.rectangle(
                    display_frame,
                    (text_x - 5, text_y - text_height - 5),
                    (text_x + text_width + 5, text_y + baseline + 5),
                    COLOR_BLACK,
                    -1
                )
                
                # 绘制灰色文字
                cv2.putText(
                    display_frame,
                    scanning_text,
                    (text_x, text_y),
                    font,
                    font_scale,
                    (128, 128, 128),  # 灰色
                    thickness,
                    cv2.LINE_AA
                )
            
            # 层1 - 硬件监控（右上角，简洁样式）+ 唇动检测状态
            render_layer1_hardware(display_frame, current_fps, "UDP", is_valid, latest_is_talking)

            # 深度画中画（右下角）
            if depth_bgr is not None:
                render_pip_depth(display_frame, depth_bgr)
            
            # 显示帧数（左下角调试信息）
            debug_text = f"Frame: {total_frames}"
            cv2.putText(display_frame, debug_text, (10, display_frame.shape[0] - 10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, COLOR_WHITE, 1, cv2.LINE_AA)
            
            # 显示画面
            cv2.imshow(WINDOW_NAME, display_frame)
            
            # 键盘控制
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                # 收到退出指令
                break
            elif key == ord('c'):
                # [校准] 触发基准环境校准（预留功能）
                # 当前距离和置信度信息已通过 GUI 显示
                pass
    
    except KeyboardInterrupt:
        # 收到中断信号...
        pass
    
    except Exception as e:
        # 错误已通过日志系统记录
        import traceback
        traceback.print_exc()
    
    finally:
        # 清理资源
        # 正在清理资源...
        
        cv2.destroyAllWindows()
        receiver.close()
        face_gate.stop()
        try:
            vision_sub.close()
            zmq_ctx.term()
        except Exception:
            pass
        
        # 总帧数和退出信息已通过日志系统记录


if __name__ == "__main__":
    main()
