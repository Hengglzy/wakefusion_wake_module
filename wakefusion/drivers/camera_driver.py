"""
Orbbec Gemini 330 系列深度相机驱动（支持 335/336）
负责RGB + Depth数据采集和自动重连
"""

import os
import sys

# 添加 pyorbbecsdk DLL 搜索路径（Windows）
dll_path = r"D:\tools\cursor_project\Orbbec Gemini 335336\Orbbec Gemini 335336 SDK\bin"
if os.path.exists(dll_path):
    os.add_dll_directory(dll_path)

import asyncio
import numpy as np
import time
import cv2
from typing import Optional, Callable
from dataclasses import dataclass
from enum import Enum

from wakefusion.types import VisionFrame
from wakefusion.logging import get_logger
from wakefusion.metrics import get_metrics, record_latency


logger = get_logger("camera_driver")
metrics = get_metrics()


class CameraState(str, Enum):
    """相机状态"""
    STOPPED = "STOPPED"
    STARTING = "STARTING"
    RUNNING = "RUNNING"
    ERROR = "ERROR"
    RECONNECTING = "RECONNECTING"


@dataclass
class CameraConfig:
    """相机配置（默认为 Gemini 335 硬件参数）"""
    rgb_width: int = 1280
    rgb_height: int = 800
    rgb_fps: int = 30
    depth_width: int = 1280
    depth_height: int = 800
    depth_fps: int = 30
    enable_rgb: bool = True
    enable_depth: bool = True


class Gemini330Driver:
    """Orbbec Gemini 330 系列深度相机驱动（支持 335/336）"""

    def __init__(
        self,
        config: CameraConfig = None,
        callback: Optional[Callable[[VisionFrame], None]] = None
    ):
        """
        初始化相机驱动

        Args:
            config: 相机配置
            callback: 视觉帧回调函数
        """
        self.config = config or CameraConfig()
        self.callback = callback

        # pyorbbecsdk 实例
        self.pipeline: Optional[any] = None
        self.device: Optional[any] = None

        # SDK 原生滤镜（Viewer 级数据质量）
        self.align_filter: Optional[any] = None  # 核心：解决对齐不准
        self.colorizer: Optional[any] = None      # 核心：实现 Viewer 同款上色
        self.spatial_filter: Optional[any] = None # 降噪

        # 深度备份（用于消除空帧闪烁）
        self.last_valid_depth_array: Optional[np.ndarray] = None
        
        # 对齐性能优化：跟踪上一帧时间，用于判断剧烈运动
        self.last_frame_timestamp: Optional[float] = None
        self.last_aligned_depth_backup: Optional[any] = None  # 对齐后的深度帧备份

        # 状态
        self.state = CameraState.STOPPED
        self.reconnect_interval = 2.0  # 重连间隔（秒）
        self.max_reconnect_attempts = 5

        # 统计
        self.total_frames = 0
        self.dropped_frames = 0
        self.last_frame_time: Optional[float] = None

        logger.info(
            "Gemini330Driver initialized",
            extra={
                "rgb_resolution": f"{self.config.rgb_width}x{self.config.rgb_height}",
                "rgb_fps": self.config.rgb_fps,
                "depth_resolution": f"{self.config.depth_width}x{self.config.depth_height}",
                "depth_fps": self.config.depth_fps
            }
        )

    def _find_best_profile(self, profile_list, target_width: int, target_height: int, 
                           target_format, target_fps: int):
        """
        从 Profile 列表中查找最佳匹配的配置
        
        Args:
            profile_list: 流配置列表
            target_width: 目标宽度
            target_height: 目标高度
            target_format: 目标格式（如 ob.OBFormat.Y16）
            target_fps: 目标帧率
            
        Returns:
            最佳匹配的 profile，如果未找到则返回 None
        """
        try:
            # 优先使用 get_video_stream_profile 方法直接匹配
            try:
                direct_profile = profile_list.get_video_stream_profile(
                    target_width,
                    target_height,
                    target_format,
                    target_fps
                )
                if direct_profile:
                    logger.debug(f"  ✓ Direct match found: {target_width}x{target_height}@{target_fps}fps")
                    return direct_profile
            except Exception as e:
                logger.debug(f"Direct profile match failed: {e}, falling back to iteration")
            
            # 如果直接匹配失败，遍历所有 Profile 查找最佳匹配
            best_profile = None
            best_score = float('inf')
            
            count = profile_list.get_count()
            logger.debug(f"Found {count} available profiles, searching for best match...")
            
            for i in range(count):
                # 使用索引操作符获取 profile（更通用的方式）
                profile = profile_list[i]
                video_profile = profile.as_video_stream_profile()
                
                if video_profile is None:
                    continue
                
                # 获取 Profile 属性
                width = video_profile.get_width()
                height = video_profile.get_height()
                fps = video_profile.get_fps()
                fmt = video_profile.get_format()
                
                # 日志降噪：将 _find_best_profile 中所有记录 Profile 详情的 info 日志改为 debug
                logger.debug(f"  Profile {i}: {width}x{height}@{fps}fps, format={fmt}")
                
                # 检查格式是否匹配
                if fmt != target_format:
                    continue
                
                # 计算匹配分数（越小越好）
                # 优先匹配分辨率，其次匹配帧率
                resolution_diff = abs(width - target_width) + abs(height - target_height)
                fps_diff = abs(fps - target_fps)
                score = resolution_diff * 100 + fps_diff
                
                # 精确匹配时分数为 0
                if width == target_width and height == target_height and fps == target_fps:
                    logger.debug(f"  ✓ Exact match found: {width}x{height}@{fps}fps")
                    return video_profile
                
                # 记录最佳匹配
                if score < best_score:
                    best_score = score
                    best_profile = video_profile
                    
            if best_profile:
                w = best_profile.get_width()
                h = best_profile.get_height()
                f = best_profile.get_fps()
                logger.debug(f"  ✓ Best match found: {w}x{h}@{f}fps (target was {target_width}x{target_height}@{target_fps}fps)")
                
            return best_profile
                    
        except Exception as e:
            logger.error(f"Error finding profile: {e}")
            return None

    def start(self):
        """启动相机采集（适配 pyorbbecsdk v2 API）"""
        if self.state != CameraState.STOPPED:
            logger.warning(f"Camera already in state: {self.state}")
            return

        self.state = CameraState.STARTING

        try:
            # 动态导入pyorbbecsdk
            import pyorbbecsdk as ob

            logger.info("Initializing pyorbbecsdk pipeline...")

            # 创建pipeline
            self.pipeline = ob.Pipeline()

            # 获取设备
            self.device = self.pipeline.get_device()
            device_name = self.device.get_device_info().get_name()

            logger.debug(f"Camera device found: {device_name}")

            # 创建配置对象（pyorbbecsdk v2 API）
            config = ob.Config()
            streams_configured = 0

            # 配置RGB流（Color Sensor）
            if self.config.enable_rgb:
                try:
                    logger.debug("Configuring Color stream...")
                    # 获取Color传感器的Profile列表
                    color_profile_list = self.pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
                    
                    # 查找最佳匹配的Profile（使用 MJPEG 格式节省 USB 带宽）
                    color_profile = self._find_best_profile(
                        color_profile_list,
                        self.config.rgb_width,
                        self.config.rgb_height,
                        ob.OBFormat.MJPG,
                        self.config.rgb_fps
                    )
                    
                    if color_profile:
                        config.enable_stream(color_profile)
                        streams_configured += 1
                        logger.debug(
                            f"RGB stream enabled: {color_profile.get_width()}x{color_profile.get_height()}@{color_profile.get_fps()}fps"
                        )
                    else:
                        logger.warning(f"No matching RGB profile found, trying default...")
                        # 尝试获取默认配置
                        try:
                            default_profile = color_profile_list.get_default_video_stream_profile()
                            if default_profile:
                                config.enable_stream(default_profile)
                                streams_configured += 1
                                logger.debug(f"RGB stream enabled (default): {default_profile.get_width()}x{default_profile.get_height()}@{default_profile.get_fps()}fps")
                        except Exception as e2:
                            logger.warning(f"Failed to get default RGB profile: {e2}")
                        
                except Exception as e:
                    logger.warning(f"Failed to configure RGB stream: {e}")

            # 配置Depth流（Depth Sensor）
            if self.config.enable_depth:
                try:
                    logger.debug("Configuring Depth stream...")
                    # 获取Depth传感器的Profile列表
                    depth_profile_list = self.pipeline.get_stream_profile_list(ob.OBSensorType.DEPTH_SENSOR)
                    
                    # 查找最佳匹配的Profile
                    depth_profile = self._find_best_profile(
                        depth_profile_list,
                        self.config.depth_width,
                        self.config.depth_height,
                        ob.OBFormat.Y16,
                        self.config.depth_fps
                    )
                    
                    if depth_profile:
                        config.enable_stream(depth_profile)
                        streams_configured += 1
                        logger.debug(
                            f"Depth stream enabled: {depth_profile.get_width()}x{depth_profile.get_height()}@{depth_profile.get_fps()}fps"
                        )
                    else:
                        logger.warning(f"No matching Depth profile found, trying default...")
                        # 尝试获取默认配置
                        try:
                            default_profile = depth_profile_list.get_default_video_stream_profile()
                            if default_profile:
                                config.enable_stream(default_profile)
                                streams_configured += 1
                                logger.debug(f"Depth stream enabled (default): {default_profile.get_width()}x{default_profile.get_height()}@{default_profile.get_fps()}fps")
                        except Exception as e2:
                            logger.warning(f"Failed to get default Depth profile: {e2}")
                        
                except Exception as e:
                    logger.warning(f"Failed to configure Depth stream: {e}")

            # 检查是否至少有一个流被配置
            if streams_configured == 0:
                raise RuntimeError("No streams could be configured. Check camera compatibility.")

            # 启动pipeline（传入配置对象）
            self.pipeline.start(config)

            # 初始化 SDK 原生滤镜（Viewer 级数据质量）
            # 注意：某些版本的 pyorbbecsdk 可能没有 Colorizer 和 SpatialFilter
            try:
                # 对齐滤镜：将深度对齐到 RGB 流（核心：解决对齐不准）
                self.align_filter = ob.AlignFilter(align_to_stream=ob.OBStreamType.COLOR_STREAM)
                logger.info("AlignFilter initialized successfully")
            except Exception as e:
                logger.error(f"Failed to initialize AlignFilter: {e}")
                self.align_filter = None
            
            # Colorizer 和 SpatialFilter 可能不存在，使用 OpenCV 作为替代
            self.colorizer = None  # SDK 的 Colorizer 不存在，将使用 OpenCV 上色
            self.spatial_filter = None  # SDK 的 SpatialFilter 不存在，将使用 OpenCV 滤波
            logger.info("Using OpenCV for depth colorization (SDK Colorizer not available)")

            self.state = CameraState.RUNNING

            logger.info(
                f"Camera started successfully with {streams_configured} stream(s)",
                extra={
                    "device_name": device_name,
                    "state": self.state,
                    "streams_configured": streams_configured
                }
            )

        except ImportError:
            logger.error("pyorbbecsdk not installed. Please install: pip install pyorbbecsdk")
            self.state = CameraState.ERROR
            raise
        except Exception as e:
            logger.error(f"Failed to start camera: {e}")
            self.state = CameraState.ERROR
            raise

    def stop(self):
        """停止相机采集"""
        if self.state == CameraState.STOPPED:
            return

        logger.info("Stopping camera...")

        if self.pipeline:
            try:
                self.pipeline.stop()
            except Exception as e:
                logger.error(f"Error stopping pipeline: {e}")

        self.pipeline = None
        self.device = None
        self.state = CameraState.STOPPED

        logger.debug("Camera stopped")

    def capture_frame(self) -> Optional[VisionFrame]:
        """
        捕获一帧数据（阻塞调用）

        Returns:
            VisionFrame: 视觉帧，如果失败则返回None
        """
        if self.state != CameraState.RUNNING:
            return None

        start_time = time.perf_counter()

        try:
            # 动态导入 pyorbbecsdk 以获取格式常量
            import pyorbbecsdk as ob
            
            # 等待帧（1秒超时，防止高分辨率模式下初始化缓慢导致丢帧）
            frameset = self.pipeline.wait_for_frames(1000)

            if not frameset:
                logger.warning("Timeout waiting for frames")
                metrics.increment_counter("camera.frame_timeout")
                return None

            # SDK 原生滤波和对齐处理（Viewer 级数据质量）
            color_depth_frame = None
            aligned_depth_frame_obj = None
            if frameset:
                try:
                    # 1. 对齐：将深度对齐到 RGB 流（核心：修复坐标漂移，确保 RGB 与 Depth 像素一一对应）
                    # 注意：AlignFilter.process() 处理的是单个 Frame，返回的也是 Frame
                    # 性能优化：如果发生剧烈运动（通过时间间隔判断），跳过该帧的 AlignFilter 处理
                    current_timestamp = time.perf_counter()
                    should_skip_align = False
                    if self.last_frame_timestamp is not None:
                        time_interval = current_timestamp - self.last_frame_timestamp
                        # 如果时间间隔过短（< 0.01s，即 > 100fps），可能是剧烈运动导致，跳过对齐
                        if time_interval < 0.01:
                            should_skip_align = True
                            if self.last_aligned_depth_backup is not None:
                                aligned_depth_frame_obj = self.last_aligned_depth_backup
                            else:
                                aligned_depth_frame_obj = frameset.get_depth_frame()
                    self.last_frame_timestamp = current_timestamp
                    
                    if self.align_filter and self.config.enable_depth and not should_skip_align:
                        # 检查 frameset 是否包含深度帧（AlignFilter 需要同时看到 RGB 和 Depth 帧）
                        depth_frame_for_align = frameset.get_depth_frame()
                        if depth_frame_for_align:
                            try:
                                # 帧对齐容错：在 capture_frame 中，如果 AlignFilter 处理时间过长或失败，直接回退使用上一帧成功的对齐备份
                                # 注意：AlignFilter.process() 需要接收整个 frameset（包含 RGB 和 Depth），而不是单独的深度帧
                                # 新版本 pyorbbecsdk: process() 返回 Frame，需要先转换为 FrameSet
                                align_start_time = time.perf_counter()
                                aligned_raw = self.align_filter.process(frameset)
                                align_elapsed = time.perf_counter() - align_start_time
                                
                                # 从对齐后的 frameset 中提取深度帧
                                aligned_depth_frame_obj = None
                                if aligned_raw is not None:
                                    # 新版本 SDK: 需要先转换为 FrameSet
                                    aligned_frameset = aligned_raw.as_frame_set() if hasattr(aligned_raw, 'as_frame_set') else aligned_raw
                                    if aligned_frameset is not None:
                                        aligned_depth_frame_obj = aligned_frameset.get_depth_frame()
                                
                                # 如果对齐处理时间过长（> 0.05s），回退使用上一帧备份
                                if align_elapsed > 0.05 and self.last_aligned_depth_backup is not None:
                                    aligned_depth_frame_obj = self.last_aligned_depth_backup
                                    logger.debug(f"AlignFilter processing too slow ({align_elapsed*1000:.1f}ms), using backup")
                                
                                # 检查对齐是否成功并验证尺寸匹配
                                if aligned_depth_frame_obj:
                                    # 验证对齐后的深度帧尺寸是否与 RGB 帧匹配（确保像素一一对应）
                                    aligned_depth_width = aligned_depth_frame_obj.get_width()
                                    aligned_depth_height = aligned_depth_frame_obj.get_height()
                                    
                                    # 获取 RGB 帧尺寸（用于验证）
                                    color_frame_check = frameset.get_color_frame()
                                    if color_frame_check:
                                        rgb_width = color_frame_check.get_width()
                                        rgb_height = color_frame_check.get_height()
                                        
                                        # 检查尺寸是否匹配（对齐后深度图应该与 RGB 图尺寸一致）
                                        if aligned_depth_width != rgb_width or aligned_depth_height != rgb_height:
                                            logger.debug(f"Align mismatch: depth={aligned_depth_width}x{aligned_depth_height}, "
                                                        f"RGB={rgb_width}x{rgb_height} - 坐标无法一一对应")
                                            aligned_depth_frame_obj = None  # 尺寸不匹配视为失败
                                        else:
                                            # 验证深度数据是否有效
                                            depth_data_check = aligned_depth_frame_obj.get_data()
                                            depth_array_check = np.frombuffer(depth_data_check, dtype=np.uint16)
                                            if np.all(depth_array_check == 0):
                                                logger.debug("Align filter may have failed: depth frame is all zeros")
                                                aligned_depth_frame_obj = None  # 全0视为失败
                                            elif self.total_frames == 0:
                                                # 首次成功时打印确认信息
                                                valid_count = np.sum(depth_array_check > 0)
                                                logger.info(f"✓ Align successful: depth={aligned_depth_width}x{aligned_depth_height} "
                                                           f"matches RGB={rgb_width}x{rgb_height}, "
                                                           f"valid_pixels={valid_count}/{depth_array_check.size} "
                                                           f"({100*valid_count/depth_array_check.size:.1f}%) - 坐标已对齐")
                                    else:
                                        # 没有 RGB 帧，无法验证，但继续使用对齐后的深度帧
                                        if self.total_frames == 0:
                                            logger.debug(f"Align completed: depth={aligned_depth_width}x{aligned_depth_height} "
                                                        f"(no RGB frame for verification)")
                                else:
                                    logger.debug("Align filter returned None")
                                    # 若本次对齐失败，优先回退到上一帧成功的对齐结果
                                    if self.last_aligned_depth_backup is not None:
                                        aligned_depth_frame_obj = self.last_aligned_depth_backup
                                        logger.debug("Using last aligned depth backup as placeholder")
                                    else:
                                        aligned_depth_frame_obj = None
                            except Exception as e:
                                logger.debug(f"Align filter process failed: {e}")
                                import traceback
                                logger.debug(traceback.format_exc())
                                aligned_depth_frame_obj = None
                            
                            # 如果对齐成功，备份对齐后的深度帧（用于剧烈运动时跳过对齐）
                            if aligned_depth_frame_obj is not None:
                                self.last_aligned_depth_backup = aligned_depth_frame_obj
                    
                    # 2. 上色：使用 OpenCV 将深度图转换为彩色图（SDK Colorizer 不可用时的替代方案）
                    if self.config.enable_depth:
                        # 使用对齐后的深度帧（如果对齐成功），否则使用原始深度帧
                        depth_frame_obj = aligned_depth_frame_obj if aligned_depth_frame_obj else frameset.get_depth_frame()
                        if depth_frame_obj:
                            try:
                                # 获取深度数据
                                depth_width = depth_frame_obj.get_width()
                                depth_height = depth_frame_obj.get_height()
                                depth_data = depth_frame_obj.get_data()
                                depth_array = np.frombuffer(depth_data, dtype=np.uint16).reshape((depth_height, depth_width))
                                
                                # 固定深度范围：0.5m (500mm) 到 4m (4000mm)
                                depth_min = 500   # 0.5m
                                depth_max = 4000  # 4.0m
                                
                                # 裁剪深度值到固定范围
                                depth_clipped = np.clip(depth_array, depth_min, depth_max)
                                
                                # 归一化到 0-255
                                # 公式：normalized = (depth - 500) / (4000 - 500) * 255
                                depth_normalized = ((depth_clipped - depth_min) / (depth_max - depth_min) * 255).astype(np.uint8)
                                
                                # 应用 Jet 色图（Viewer 同款）
                                # Jet 色图颜色对应（固定范围 0.5m-4.0m）：
                                # - 蓝色（Blue）：500mm (0.5m) - 最近
                                # - 青色（Cyan）：约 1000-1500mm (1.0-1.5m)
                                # - 绿色（Green）：约 1500-2500mm (1.5-2.5m)
                                # - 黄色（Yellow）：约 2500-3500mm (2.5-3.5m)
                                # - 红色（Red）：4000mm (4.0m) - 最远
                                color_depth_frame = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                                
                                # 保存有效的深度数组作为备份（用于消除空帧闪烁）
                                self.last_valid_depth_array = depth_array.copy()
                                
                            except Exception as e:
                                if self.total_frames < 5:
                                    logger.error(f"Error in depth colorization: {e}")
                                    import traceback
                                    logger.debug(traceback.format_exc())
                                color_depth_frame = None
                except Exception as e:
                    logger.warning(f"Error processing frameset with filters: {e}, using raw frameset")
                    import traceback
                    logger.debug(traceback.format_exc())

            current_ts = time.time()

            # 获取RGB帧
            rgb_frame = None
            if self.config.enable_rgb:
                color_frame = frameset.get_color_frame()
                if color_frame:
                    # 获取帧格式和尺寸
                    frame_format = color_frame.get_format()
                    width = color_frame.get_width()
                    height = color_frame.get_height()
                    
                    # 获取原始数据
                    raw_data = np.frombuffer(color_frame.get_data(), dtype=np.uint8)
                    
                    # 根据格式解码
                    if frame_format == ob.OBFormat.MJPG:
                        # MJPEG 格式：使用 cv2 解码为 BGR，然后转为 RGB（确保传给 MediaPipe 的是 RGB）
                        data = color_frame.get_data()
                        bgr_frame = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
                        if bgr_frame is not None:
                            rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                        else:
                            logger.warning("Failed to decode MJPEG frame")
                            rgb_frame = None
                    elif frame_format == ob.OBFormat.RGB:
                        # RGB 格式：直接重塑
                        rgb_frame = raw_data.reshape((height, width, 3))
                    elif frame_format == ob.OBFormat.BGR:
                        # BGR 格式：重塑并转换
                        bgr_frame = raw_data.reshape((height, width, 3))
                        rgb_frame = cv2.cvtColor(bgr_frame, cv2.COLOR_BGR2RGB)
                    elif frame_format == ob.OBFormat.RGBA:
                        # RGBA 格式：重塑并移除 Alpha
                        rgba_frame = raw_data.reshape((height, width, 4))
                        rgb_frame = rgba_frame[:, :, :3]
                    elif frame_format == ob.OBFormat.BGRA:
                        # BGRA 格式：重塑、转换并移除 Alpha
                        bgra_frame = raw_data.reshape((height, width, 4))
                        rgb_frame = cv2.cvtColor(bgra_frame, cv2.COLOR_BGRA2RGB)
                    else:
                        # 未知格式：尝试直接重塑为 RGB
                        logger.warning(f"Unknown color format: {frame_format}, trying RGB reshape")
                        try:
                            rgb_frame = raw_data.reshape((height, width, 3))
                        except ValueError:
                            logger.error(f"Cannot reshape data to ({height}, {width}, 3)")

            # 获取Depth帧（使用对齐后的深度帧，带备份机制）
            depth_frame = None
            if self.config.enable_depth:
                # 使用对齐后的深度帧（如果对齐成功），否则使用原始深度帧
                depth = aligned_depth_frame_obj if aligned_depth_frame_obj is not None else (frameset.get_depth_frame() if frameset else None)
                if depth:
                    width = depth.get_width()
                    height = depth.get_height()
                    depth_data = depth.get_data()
                    # 明确指定形状，防止出现一维数组
                    depth_frame = np.frombuffer(depth_data, dtype=np.uint16).reshape((height, width))
                    
                    # 如果深度帧有效，更新备份
                    if depth_frame is not None and np.any(depth_frame > 0):
                        self.last_valid_depth_array = depth_frame.copy()
                elif self.last_valid_depth_array is not None:
                    # 如果 AlignFilter 返回 None，使用上一帧的深度备份（消除空帧闪烁）
                    depth_frame = self.last_valid_depth_array.copy()

            # 创建VisionFrame
            vision_frame = VisionFrame(
                ts=current_ts,
                rgb=rgb_frame,
                depth=depth_frame,  # 原始深度数据，用于距离计算
                color_depth=color_depth_frame,  # 上色后的深度图（BGR格式），用于显示
                presence=False,  # 将由FaceGate检测
                faces=[],
                distance_m=None,
                confidence=0.0
            )

            # 统计
            self.total_frames += 1
            if self.last_frame_time:
                gap = current_ts - self.last_frame_time
                expected_gap = 1.0 / self.config.rgb_fps
                if gap > expected_gap * 1.5:
                    logger.warning(f"Large frame gap: {gap*1000:.1f}ms")
                    metrics.increment_counter("camera.frame_gaps")

            self.last_frame_time = current_ts

            # 记录延迟
            latency_ms = (time.perf_counter() - start_time) * 1000
            record_latency("camera.capture_latency_ms", latency_ms)

            # 回调
            if self.callback:
                self.callback(vision_frame)

            return vision_frame

        except Exception as e:
            logger.error(f"Error capturing frame: {e}")
            metrics.increment_counter("camera.capture_errors")
            return None

    async def run_with_reconnect(self):
        """
        运行相机并自动重连

        当检测到断连时，自动尝试重连
        """
        reconnect_attempts = 0

        while self.state in [CameraState.RUNNING, CameraState.RECONNECTING]:
            try:
                # 等待一小段时间
                await asyncio.sleep(0.1)

                # 捕获帧
                frame = await asyncio.get_event_loop().run_in_executor(
                    None,
                    self.capture_frame
                )

                if frame:
                    reconnect_attempts = 0  # 重置重连计数

                    # 更新设备状态指标
                    metrics.set_gauge("camera.device_connected", 1.0)
                    metrics.set_gauge("camera.fps", self.config.rgb_fps)

                elif reconnect_attempts > self.max_reconnect_attempts:
                    # 检测到断连
                    logger.warning(f"Camera disconnected, reconnecting... (attempt {reconnect_attempts})")
                    metrics.increment_counter("camera.reconnect_count")

                    self.state = CameraState.RECONNECTING

                    # 尝试重连
                    self.stop()
                    await asyncio.sleep(self.reconnect_interval)
                    self.start()

                    reconnect_attempts = 0

            except Exception as e:
                logger.error(f"Error in camera loop: {e}")
                reconnect_attempts += 1

                if reconnect_attempts > self.max_reconnect_attempts:
                    logger.critical("Max reconnection errors reached, stopping")
                    self.stop()
                    break

    def get_device_status(self) -> dict:
        """获取设备状态"""
        return {
            "state": self.state,
            "device_name": self.device.get_device_info().get_name() if self.device else "None",
            "total_frames": self.total_frames,
            "dropped_frames": self.dropped_frames,
            "rgb_enabled": self.config.enable_rgb,
            "depth_enabled": self.config.enable_depth,
            "rgb_resolution": f"{self.config.rgb_width}x{self.config.rgb_height}",
            "depth_resolution": f"{self.config.depth_width}x{self.config.depth_height}"
        }


# 向后兼容别名（保持与原有代码的兼容性）
FemtoBoltDriver = Gemini330Driver


async def test_camera_driver():
    """测试相机驱动"""

    def on_vision_frame(frame: VisionFrame):
        print(f"[{frame.ts:.3f}] Vision frame:")
        if frame.rgb is not None:
            print(f"  RGB: {frame.rgb.shape}")
        if frame.depth is not None:
            print(f"  Depth: {frame.depth.shape}, range={frame.depth.min()}-{frame.depth.max()}")

    driver = Gemini330Driver(
        config=CameraConfig(
            rgb_width=1280,
            rgb_height=800,
            rgb_fps=15,  # 降低帧率以减少CPU负载
            enable_depth=True
        ),
        callback=on_vision_frame
    )

    capture_task = None
    try:
        driver.start()
        print("Starting capture loop...")
        
        # 启动后台采集循环
        capture_task = asyncio.create_task(driver.run_with_reconnect())
        
        print("Capturing for 10 seconds...")
        await asyncio.sleep(10)
        
        print("Done!")

    except KeyboardInterrupt:
        print("\nInterrupted by user")
    except Exception as e:
        print(f"Error: {e}")
    finally:
        # 取消采集任务
        if capture_task and not capture_task.done():
            capture_task.cancel()
            try:
                await capture_task
            except asyncio.CancelledError:
                pass
        
        # 停止驱动
        driver.stop()
        print(f"Total frames captured: {driver.total_frames}")


if __name__ == "__main__":
    asyncio.run(test_camera_driver())
