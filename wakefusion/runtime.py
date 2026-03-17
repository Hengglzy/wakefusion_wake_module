"""
主运行时
WakeFusion系统入口，负责启动和管理所有组件
"""

import asyncio
import signal
import sys
from pathlib import Path

from wakefusion.config import get_config
from wakefusion.logging import get_logger
from wakefusion.metrics import get_metrics, SystemMetrics, set_gauge

from wakefusion.drivers import XVF3800Driver, FemtoBoltDriver, CameraConfig
from wakefusion.routers import AudioRouter, VisionRouter
from wakefusion.workers import KWSWorker, VADWorker, FaceGateWorker, FaceGateConfig, MatchboxNetKWSWorker, MatchboxNetConfig
from wakefusion.decision import DecisionEngine, VisionGateResult
from wakefusion.io import WSEventPublisher, HealthServer

from wakefusion.types import (
    BaseEvent, EventType, AudioFrameRaw, AudioFrame, VisionFrame
)


logger = get_logger("runtime")
metrics = get_metrics()


class WakeFusionRuntime:
    """WakeFusion运行时"""

    def __init__(self, config_path: str = None):
        """
        初始化运行时

        Args:
            config_path: 配置文件路径
        """
        # 加载配置
        self.config = get_config(config_path)

        # 音频组件
        self.audio_driver: XVF3800Driver = None
        self.audio_router: AudioRouter = None
        self.kws_worker: KWSWorker = None
        self.vad_worker: VADWorker = None

        # 视觉组件
        self.camera_driver: FemtoBoltDriver = None
        self.vision_router: VisionRouter = None
        self.face_gate: FaceGateWorker = None

        # 决策和输出
        self.decision_engine: DecisionEngine = None
        self.ws_publisher: WSEventPublisher = None
        self.health_server: HealthServer = None

        # 状态
        self.is_running = False
        self.shutdown_event = asyncio.Event()

        logger.info("WakeFusionRuntime initialized")

    async def start(self):
        """启动运行时"""
        logger.info("Starting WakeFusion runtime...")

        try:
            # 1. 启动WebSocket发布器
            self.ws_publisher = WSEventPublisher(
                host="0.0.0.0",
                port=self.config.runtime.websocket_port
            )
            await self.ws_publisher.start()

            # 2. 启动健康检查服务
            self.health_server = HealthServer(
                host="0.0.0.0",
                port=self.config.runtime.health_port
            )
            await self.health_server.start()

            # 3. 初始化决策引擎
            self.decision_engine = DecisionEngine(
                kws_threshold=self.config.kws.threshold,
                probation_enabled=self.config.fusion.probation_enabled,
                barge_in_enabled=self.config.fusion.barge_in_enabled,
                event_callback=self._on_decision_event
            )

            # 4. 初始化音频路由器（集成 RNNoise 服务）
            self.audio_router = AudioRouter(
                capture_sample_rate=self.config.audio.capture_sample_rate,
                work_sample_rate=self.config.audio.work_sample_rate,
                frame_ms=self.config.audio.frame_ms,
                ring_buffer_sec=self.config.audio.ring_buffer_sec,
                rnnoise_enabled=self.config.audio.rnnoise_enabled
            )

            # 5. 初始化KWS工作线程
            kws_engine = getattr(self.config.kws, 'engine', 'matchboxnet').lower()

            if kws_engine == 'matchboxnet':
                # 使用 MatchboxNet (推荐)
                model_name = getattr(self.config.kws, 'model_name', 'commandrecognition_en_matchboxnet3x1x64_v1')
                device = getattr(self.config.kws, 'device', 'cpu')

                logger.info(f"Initializing MatchboxNet KWS with model: {model_name}")

                self.kws_worker = MatchboxNetKWSWorker(
                    config=MatchboxNetConfig(
                        model_name=model_name,
                        threshold=self.config.kws.threshold,
                        cooldown_ms=self.config.kws.cooldown_ms,
                        device=device
                    ),
                    event_callback=self._on_kws_event
                )
            else:
                # 使用 openWakeWord (备选)
                logger.info(f"Initializing openWakeWord KWS with keyword: {self.config.kws.keyword}")
                self.kws_worker = KWSWorker(
                    keyword=self.config.kws.keyword,
                    threshold=self.config.kws.threshold,
                    cooldown_ms=self.config.kws.cooldown_ms,
                    event_callback=self._on_kws_event
                )

            self.kws_worker.start()

            # 6. 初始化VAD工作线程
            self.vad_worker = VADWorker(
                aggressiveness=2,
                speech_start_ms=self.config.vad.speech_start_ms,
                speech_end_ms=self.config.vad.speech_end_ms,
                frame_ms=self.config.audio.frame_ms,
                sample_rate=self.config.audio.work_sample_rate,
                event_callback=self._on_vad_event
            )
            self.vad_worker.start()

            # 7. 订阅音频路由器
            self.audio_router.subscribe(self._on_audio_frame)

            # 8. 初始化并启动音频驱动
            self.audio_driver = XVF3800Driver(
                device_match=self.config.audio.device_match,
                sample_rate=self.config.audio.capture_sample_rate,
                channels=self.config.audio.channels,
                frame_ms=self.config.audio.frame_ms,
                callback=self._on_raw_audio_frame
            )
            self.audio_driver.start()

            # ============ Phase 2: 视觉组件 ============
            if self.config.vision.enabled:
                logger.info("Initializing vision components...")

                # 初始化视觉路由器
                self.vision_router = VisionRouter(
                    cache_ms=self.config.vision.cache_ms,
                    target_fps=15
                )

                # 初始化FaceGate
                self.face_gate = FaceGateWorker(
                    config=FaceGateConfig(
                        distance_m_max=self.config.vision.distance_m_max,
                        face_conf_min=self.config.vision.face_conf_min,
                        enable_face_detection=False,  # Phase 2暂不启用
                        enable_depth_gate=True
                    ),
                    event_callback=self._on_vision_event
                )
                self.face_gate.start()

                # 初始化并启动相机驱动
                self.camera_driver = FemtoBoltDriver(
                    config=CameraConfig(
                        rgb_width=640,
                        rgb_height=480,
                        rgb_fps=15,
                        enable_rgb=False,  # Phase 2只需要深度
                        enable_depth=True
                    ),
                    callback=self._on_vision_frame
                )
                self.camera_driver.start()

                # 启动相机自动重连任务
                asyncio.create_task(self.camera_driver.run_with_reconnect())

                logger.info("Vision components started successfully")

            # 9. 注册健康检查回调
            self.health_server.register_component("audio", self._get_audio_status)
            self.health_server.register_component("kws", self.kws_worker.get_stats)
            self.health_server.register_component("vad", self.vad_worker.get_stats)
            self.health_server.register_component("fusion", self.decision_engine.get_stats)
            self.health_server.register_component("ws", self.ws_publisher.get_stats)

            if self.config.vision.enabled:
                self.health_server.register_component("camera", self.camera_driver.get_device_status)
                self.health_server.register_component("vision", self.vision_router.get_stats)
                self.health_server.register_component("face_gate", self.face_gate.get_stats)

            # 10. 启动健康检查报告任务
            asyncio.create_task(self._health_report_loop())

            self.is_running = True

            logger.info(
                "WakeFusion runtime started successfully",
                extra={
                    "websocket_url": f"ws://0.0.0.0:{self.config.runtime.websocket_port}",
                    "health_url": f"http://0.0.0.0:{self.config.runtime.health_port}/health"
                }
            )

        except Exception as e:
            logger.error(f"Failed to start runtime: {e}")
            await self.stop()
            raise

    async def stop(self):
        """停止运行时"""
        if not self.is_running:
            return

        logger.info("Stopping WakeFusion runtime...")

        self.is_running = False
        self.shutdown_event.set()

        # 按相反顺序停止组件
        if self.camera_driver:
            self.camera_driver.stop()

        if self.face_gate:
            self.face_gate.stop()

        if self.audio_driver:
            self.audio_driver.stop()

        if self.kws_worker:
            self.kws_worker.stop()

        if self.vad_worker:
            self.vad_worker.stop()

        if self.ws_publisher:
            await self.ws_publisher.stop()

        if self.health_server:
            await self.health_server.stop()

        logger.info("WakeFusion runtime stopped")

    async def run(self):
        """运行运行时（阻塞直到shutdown）"""
        await self.start()

        # 等待shutdown信号
        await self.shutdown_event.wait()

        await self.stop()

    def _on_raw_audio_frame(self, frame: AudioFrameRaw):
        """
        原始音频帧回调（从音频驱动调用）

        Args:
            frame: 原始音频帧
        """
        # 路由到音频路由器
        self.audio_router.process_raw_frame(frame)

    def _on_audio_frame(self, frame: AudioFrame):
        """
        音频帧回调（从音频路由器调用）

        Args:
            frame: 音频帧
        """
        # 提交给KWS和VAD工作线程
        if self.config.kws.enabled:
            self.kws_worker.process_frame(frame)

        if self.config.vad.enabled:
            self.vad_worker.process_frame(frame)

    def _on_kws_event(self, event: BaseEvent):
        """
        KWS事件回调

        Args:
            event: KWS事件
        """
        # 提交给决策引擎
        if event.type == EventType.KWS_HIT:
            result = self.decision_engine.process_kws_hit(event)

            # 发布KWS_HIT事件
            asyncio.create_task(self.ws_publisher.publish(event))

    def _on_vad_event(self, event: BaseEvent):
        """
        VAD事件回调

        Args:
            event: VAD事件
        """
        # 发布VAD事件
        asyncio.create_task(self.ws_publisher.publish(event))

    def _on_vision_frame(self, frame: VisionFrame):
        """
        视觉帧回调（从相机驱动调用）

        Args:
            frame: 视觉帧
        """
        # 路由到视觉路由器
        self.vision_router.process_frame(frame)

        # 提交给FaceGate
        if self.config.vision.enabled and self.face_gate:
            result = self.face_gate.process_frame(frame)

            # 更新决策引擎的视觉缓存
            if result:
                self.decision_engine.update_vision_cache(result)

    def _on_vision_event(self, event: BaseEvent):
        """
        视觉事件回调

        Args:
            event: 视觉事件
        """
        # 发布视觉事件
        asyncio.create_task(self.ws_publisher.publish(event))

    def _on_decision_event(self, event: BaseEvent):
        """
        决策事件回调

        Args:
            event: 决策事件
        """
        # 发布决策事件
        asyncio.create_task(self.ws_publisher.publish(event))

    async def _health_report_loop(self):
        """健康检查报告循环"""
        while self.is_running:
            try:
                await asyncio.sleep(self.config.runtime.health_interval_sec)

                # 更新系统指标
                set_gauge("system.cpu_percent", SystemMetrics.get_cpu_percent())
                set_gauge("system.memory_mb", SystemMetrics.get_memory_mb())

                # 发布健康事件
                health_payload = {
                    "audio_fps": metrics.get_metric("audio.fps").value if metrics.get_metric("audio.fps") else 0,
                    "audio_latency_ms": metrics.get_metric("audio.router_latency_ms").avg if metrics.get_metric("audio.router_latency_ms") else 0,
                    "kws_hit_count": self.kws_worker.detections if self.kws_worker else 0,
                    "vad_speech_segments": self.vad_worker.speech_segments if self.vad_worker else 0,
                    "device_status": self.audio_driver.get_device_status() if self.audio_driver else {},
                    "cpu_percent": SystemMetrics.get_cpu_percent(),
                    "memory_mb": SystemMetrics.get_memory_mb()
                }

                # 添加视觉指标
                if self.config.vision.enabled:
                    if self.camera_driver:
                        health_payload["camera_fps"] = self.config.vision.get("target_fps", 15)
                    if self.face_gate:
                        health_payload["face_gate_valid_count"] = self.face_gate.valid_user_count

                # TODO: 发布HEALTH事件

            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in health report loop: {e}")

    def _get_audio_status(self) -> dict:
        """获取音频状态"""
        if self.audio_driver and self.audio_router:
            return {
                "driver": self.audio_driver.get_device_status(),
                "router": self.audio_router.get_stats()
            }
        return {}


async def main():
    """主函数"""
    # 查找配置文件
    config_paths = [
        "config/config.yaml",
        "config.yaml",
        "../config/config.yaml"
    ]

    config_path = None
    for path in config_paths:
        if Path(path).exists():
            config_path = path
            break

    # 创建运行时
    runtime = WakeFusionRuntime(config_path)

    # 注册信号处理
    def signal_handler():
        logger.info("Received shutdown signal")
        runtime.shutdown_event.set()

    loop = asyncio.get_event_loop()
    loop.add_signal_handler(signal.SIGINT, signal_handler)
    loop.add_signal_handler(signal.SIGTERM, signal_handler)

    # 运行
    try:
        await runtime.run()
    except Exception as e:
        logger.error(f"Runtime error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    # 设置日志级别
    import logging
    logging.basicConfig(level=logging.INFO)

    # 运行
    asyncio.run(main())
