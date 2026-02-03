"""
WebSocket事件发布器
负责将内部事件发布到外部系统
"""

import asyncio
import json
import time
from typing import Optional, Set
import websockets
from websockets.server import WebSocketServerProtocol

from wakefusion.types import BaseEvent, EventType
from wakefusion.logging import get_logger
from wakefusion.metrics import get_metrics, increment_counter, set_gauge


logger = get_logger("publisher_ws")
metrics = get_metrics()


class WSEventPublisher:
    """WebSocket事件发布器"""

    def __init__(self, host: str = "0.0.0.0", port: int = 8765):
        """
        初始化WebSocket发布器

        Args:
            host: 监听地址
            port: 监听端口
        """
        self.host = host
        self.port = port
        self.server: Optional[websockets.WebSocketServer] = None
        self.clients: Set[WebSocketServerProtocol] = set()
        self.is_running = False

        # 事件队列
        self.event_queue: asyncio.Queue[BaseEvent] = asyncio.Queue()

        logger.info(
            "WSEventPublisher initialized",
            extra={"host": host, "port": port}
        )

    async def start(self):
        """启动WebSocket服务器"""
        self.is_running = True

        # 启动事件处理任务
        asyncio.create_task(self._event_dispatcher())

        # 启动WebSocket服务器
        self.server = await websockets.serve(
            self._handle_client,
            self.host,
            self.port,
            ping_interval=10,
            ping_timeout=5
        )

        logger.info(
            "WebSocket server started",
            extra={
                "host": self.host,
                "port": self.port,
                "url": f"ws://{self.host}:{self.port}"
            }
        )

    async def stop(self):
        """停止WebSocket服务器"""
        self.is_running = False

        # 关闭所有客户端连接
        for client in self.clients:
            await client.close()

        self.clients.clear()

        # 关闭服务器
        if self.server:
            self.server.close()
            await self.server.wait_closed()
            self.server = None

        logger.info("WebSocket server stopped")

    async def _handle_client(
        self,
        websocket: WebSocketServerProtocol,
        path: str
    ):
        """
        处理客户端连接

        Args:
            websocket: WebSocket连接
            path: 请求路径
        """
        client_addr = f"{websocket.remote_address[0]}:{websocket.remote_address[1]}"
        logger.info(f"Client connected: {client_addr}")

        self.clients.add(websocket)
        increment_counter("ws.clients_connected")
        set_gauge("ws.active_clients", len(self.clients))

        try:
            # 保持连接并处理控制命令
            async for message in websocket:
                try:
                    await self._handle_control_message(websocket, message)
                except Exception as e:
                    logger.error(f"Error handling control message: {e}")

        except websockets.exceptions.ConnectionClosed:
            logger.info(f"Client disconnected: {client_addr}")
        finally:
            self.clients.discard(websocket)
            increment_counter("ws.clients_disconnected")
            set_gauge("ws.active_clients", len(self.clients))

    async def _handle_control_message(
        self,
        websocket: WebSocketServerProtocol,
        message: str
    ):
        """
        处理控制消息

        Args:
            websocket: WebSocket连接
            message: 消息内容
        """
        try:
            data = json.loads(message)
            msg_type = data.get("type")

            logger.debug(f"Control message: {msg_type}", extra={"message": data})

            # TODO: 实现控制命令处理
            # 例如: SET_SYSTEM_STATE, SET_POLICY等

        except json.JSONDecodeError as e:
            logger.error(f"Invalid JSON message: {e}")

    async def _event_dispatcher(self):
        """事件分发器（从队列中获取事件并广播给所有客户端）"""
        while self.is_running:
            try:
                # 从队列获取事件
                event = await asyncio.wait_for(
                    self.event_queue.get(),
                    timeout=1.0
                )

                # 转换为JSON
                event_json = self._serialize_event(event)

                # 广播给所有客户端
                if self.clients:
                    await asyncio.gather(
                        *[
                            self._send_to_client(client, event_json)
                            for client in self.clients
                        ],
                        return_exceptions=True
                    )

                    increment_counter("ws.events_published")

            except asyncio.TimeoutError:
                continue
            except Exception as e:
                logger.error(f"Error in event dispatcher: {e}")

    async def _send_to_client(
        self,
        client: WebSocketServerProtocol,
        message: str
    ):
        """
        发送消息给客户端

        Args:
            client: WebSocket客户端
            message: 消息内容
        """
        try:
            await client.send(message)
        except Exception as e:
            logger.warning(f"Failed to send to client: {e}")
            increment_counter("ws.send_errors")

    def _serialize_event(self, event: BaseEvent) -> str:
        """
        序列化事件为JSON

        Args:
            event: 事件对象

        Returns:
            str: JSON字符串
        """
        event_dict = {
            "type": event.type,
            "ts": event.ts,
            "session_id": event.session_id,
            "priority": event.priority,
            "payload": event.payload
        }

        return json.dumps(event_dict, ensure_ascii=False)

    async def publish(self, event: BaseEvent):
        """
        发布事件（非阻塞）

        Args:
            event: 事件对象
        """
        try:
            self.event_queue.put_nowait(event)
        except asyncio.QueueFull:
            logger.warning("Event queue full, dropping event")
            increment_counter("ws.queue_overflows")

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "is_running": self.is_running,
            "active_clients": len(self.clients),
            "event_queue_size": self.event_queue.qsize()
        }
