"""
健康检查服务
提供HTTP健康检查端点
"""

import asyncio
import json
from aiohttp import web
import psutil
from typing import Dict, Any

from wakefusion.logging import get_logger
from wakefusion.metrics import get_metrics, SystemMetrics


logger = get_logger("health_server")
metrics = get_metrics()


class HealthServer:
    """健康检查服务器"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8080):
        """
        初始化健康检查服务器

        Args:
            host: 监听地址
            port: 监听端口
        """
        self.host = host
        self.port = port
        self.app: web.Application = None
        self.runner: web.AppRunner = None
        self.site: web.TCPSite = None
        self.is_running = False

        # 组件状态回调
        self.component_callbacks: Dict[str, callable] = {}

        logger.info(
            "HealthServer initialized",
            extra={"host": host, "port": port}
        )

    def register_component(self, name: str, callback: callable):
        """
        注册组件状态回调

        Args:
            name: 组件名称
            callback: 状态回调函数，返回dict
        """
        self.component_callbacks[name] = callback

    async def start(self):
        """启动健康检查服务器"""
        self.app = web.Application()
        self.app.router.add_get("/health", self._handle_health)
        self.app.router.add_get("/metrics", self._handle_metrics)

        self.runner = web.AppRunner(self.app)
        await self.runner.setup()

        self.site = web.TCPSite(self.runner, self.host, self.port)
        await self.site.start()

        self.is_running = True

        logger.info(
            "Health server started",
            extra={
                "host": self.host,
                "port": self.port,
                "health_url": f"http://{self.host}:{self.port}/health",
                "metrics_url": f"http://{self.host}:{self.port}/metrics"
            }
        )

    async def stop(self):
        """停止健康检查服务器"""
        self.is_running = False

        if self.site:
            await self.site.stop()
            self.site = None

        if self.runner:
            await self.runner.cleanup()
            self.runner = None

        logger.info("Health server stopped")

    async def _handle_health(self, request: web.Request) -> web.Response:
        """
        处理健康检查请求

        Args:
            request: HTTP请求

        Returns:
            web.Response: JSON响应
        """
        try:
            # 获取系统指标
            health_status = {
                "status": "healthy",
                "timestamp": asyncio.get_event_loop().time(),
                "system": {
                    "cpu_percent": SystemMetrics.get_cpu_percent(),
                    "memory_mb": SystemMetrics.get_memory_mb(),
                    "thread_count": SystemMetrics.get_thread_count()
                },
                "components": {}
            }

            # 获取各组件状态
            for name, callback in self.component_callbacks.items():
                try:
                    health_status["components"][name] = callback()
                except Exception as e:
                    logger.error(f"Error getting component status for {name}: {e}")
                    health_status["components"][name] = {"status": "error", "error": str(e)}

            return web.json_response(health_status)

        except Exception as e:
            logger.error(f"Error in health check: {e}")
            return web.json_response(
                {"status": "error", "error": str(e)},
                status=500
            )

    async def _handle_metrics(self, request: web.Request) -> web.Response:
        """
        处理指标请求

        Args:
            request: HTTP请求

        Returns:
            web.Response: JSON响应
        """
        try:
            all_metrics = metrics.get_all()

            return web.json_response(all_metrics)

        except Exception as e:
            logger.error(f"Error getting metrics: {e}")
            return web.json_response(
                {"error": str(e)},
                status=500
            )
