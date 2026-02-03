"""
Femto Bolt 深度相机驱动
负责RGB + Depth数据采集和自动重连
"""

import asyncio
import numpy as np
import time
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
    """相机配置"""
    rgb_width: int = 640
    rgb_height: int = 480
    rgb_fps: int = 30
    depth_width: int = 640
    depth_height: int = 480
    depth_fps: int = 30
    enable_rgb: bool = True
    enable_depth: bool = True


class FemtoBoltDriver:
    """Femto Bolt深度相机驱动"""

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

        # 状态
        self.state = CameraState.STOPPED
        self.reconnect_interval = 2.0  # 重连间隔（秒）
        self.max_reconnect_attempts = 5

        # 统计
        self.total_frames = 0
        self.dropped_frames = 0
        self.last_frame_time: Optional[float] = None

        logger.info(
            "FemtoBoltDriver initialized",
            extra={
                "rgb_resolution": f"{self.config.rgb_width}x{self.config.rgb_height}",
                "rgb_fps": self.config.rgb_fps,
                "depth_resolution": f"{self.config.depth_width}x{self.config.depth_height}",
                "depth_fps": self.config.depth_fps
            }
        )

    def start(self):
        """启动相机采集"""
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
            device_name = self.device.get_device_name()

            logger.info(f"Camera device found: {device_name}")

            # 配置RGB流
            if self.config.enable_rgb:
                rgb_config = self.pipeline.get_stream_config(ob.VideoMode.RGB_VIDEO)
                rgb_config.set_width(self.config.rgb_width)
                rgb_config.set_height(self.config.rgb_height)
                rgb_config.set_fps(self.config.rgb_fps)
                rgb_config.set_format(ob.OBFormat.RGB)

                logger.info(
                    "RGB stream configured",
                    extra={
                        "width": self.config.rgb_width,
                        "height": self.config.rgb_height,
                        "fps": self.config.rgb_fps
                    }
                )

            # 配置Depth流
            if self.config.enable_depth:
                depth_config = self.pipeline.get_stream_config(ob.VideoMode.DEPTH_VIDEO)
                depth_config.set_width(self.config.depth_width)
                depth_config.set_height(self.config.depth_height)
                depth_config.set_fps(self.config.depth_fps)
                depth_config.set_format(ob.OBFormat.Y16)

                logger.info(
                    "Depth stream configured",
                    extra={
                        "width": self.config.depth_width,
                        "height": self.config.depth_height,
                        "fps": self.config.depth_fps
                    }
                )

            # 启动pipeline
            self.pipeline.start()

            self.state = CameraState.RUNNING

            logger.info(
                "Camera started successfully",
                extra={
                    "device_name": device_name,
                    "state": self.state
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

        logger.info("Camera stopped")

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
            # 等待帧
            frameset = self.pipeline.wait_for_frames(100)  # 100ms超时

            if not frameset:
                logger.warning("Timeout waiting for frames")
                metrics.increment_counter("camera.frame_timeout")
                return None

            current_ts = time.time()

            # 获取RGB帧
            rgb_frame = None
            if self.config.enable_rgb:
                color_frame = frameset.get_color_frame()
                if color_frame:
                    rgb_data = color_frame.get_data()
                    rgb_frame = np.array(rgb_data, dtype=np.uint8)
                    # 转换为RGB (如果需要)
                    if rgb_frame.shape[-1] == 4:  # RGBA
                        rgb_frame = rgb_frame[:, :, :3]  # 移除Alpha通道

            # 获取Depth帧
            depth_frame = None
            if self.config.enable_depth:
                depth = frameset.get_depth_frame()
                if depth:
                    depth_data = depth.get_data()
                    depth_frame = np.array(depth_data, dtype=np.uint16)

            # 创建VisionFrame
            vision_frame = VisionFrame(
                ts=current_ts,
                rgb=rgb_frame,
                depth=depth_frame,
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
            "device_name": self.device.get_device_name() if self.device else "None",
            "total_frames": self.total_frames,
            "dropped_frames": self.dropped_frames,
            "rgb_enabled": self.config.enable_rgb,
            "depth_enabled": self.config.enable_depth,
            "rgb_resolution": f"{self.config.rgb_width}x{self.config.rgb_height}",
            "depth_resolution": f"{self.config.depth_width}x{self.config.depth_height}"
        }


async def test_camera_driver():
    """测试相机驱动"""
    import sys

    def on_vision_frame(frame: VisionFrame):
        print(f"[{frame.ts:.3f}] Vision frame:")
        if frame.rgb is not None:
            print(f"  RGB: {frame.rgb.shape}")
        if frame.depth is not None:
            print(f"  Depth: {frame.depth.shape}, range={frame.depth.min()}-{frame.depth.max()}")

    driver = FemtoBoltDriver(
        config=CameraConfig(
            rgb_width=640,
            rgb_height=480,
            rgb_fps=15,  # 降低帧率以减少CPU负载
            enable_depth=True
        ),
        callback=on_vision_frame
    )

    try:
        driver.start()
        print("Capturing for 10 seconds...")
        await asyncio.sleep(10)
        driver.stop()
        print("Done!")

    except KeyboardInterrupt:
        driver.stop()
    except Exception as e:
        print(f"Error: {e}")
        driver.stop()


if __name__ == "__main__":
    asyncio.run(test_camera_driver())
