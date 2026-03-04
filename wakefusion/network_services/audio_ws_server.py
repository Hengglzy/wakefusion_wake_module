"""
音频 WebSocket 服务器 (Audio Gateway)
功能：接收底层 UDP 10002 数据转 WebSocket -> 接收 HTTP API 请求转 UDP 10003 指令
启动：python -m wakefusion.network_services.audio_ws_server
"""
import asyncio
import json
import socket
import logging
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from contextlib import asynccontextmanager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger("audio_ws_server")

UDP_LISTEN_PORT = 10002
UDP_CTRL_TARGET = ("127.0.0.1", 10003)

connected_clients = set()

class UDPDataReceiver(asyncio.DatagramProtocol):
    """UDP 异步监听协议"""
    def datagram_received(self, data, addr):
        msg = data.decode('utf-8')
        # 广播给所有连上的 WebSocket 客户端
        for client in connected_clients:
            asyncio.create_task(client.send_text(msg))

@asynccontextmanager
async def lifespan(app: FastAPI):
    """现代 FastAPI 生命周期管理，替代旧的 on_event"""
    logger.info("启动 UDP 监听端口: 10002")
    loop = asyncio.get_event_loop()
    transport, protocol = await loop.create_datagram_endpoint(
        lambda: UDPDataReceiver(),
        local_addr=('127.0.0.1', UDP_LISTEN_PORT)
    )
    yield
    transport.close()

app = FastAPI(title="Audio WebSocket Server", lifespan=lifespan)

# 允许跨域（前端 HTML 调用 API 时需要）
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["*"], allow_headers=["*"])

@app.websocket("/ws/audio")
async def websocket_audio(websocket: WebSocket):
    await websocket.accept()
    connected_clients.add(websocket)
    logger.info(f"客户端已连接，当前连接数: {len(connected_clients)}")
    try:
        while True:
            await websocket.receive_text() # 保持连接心跳
    except WebSocketDisconnect:
        connected_clients.remove(websocket)
        logger.info(f"客户端断开，当前连接数: {len(connected_clients)}")

@app.get("/api/stop")
async def stop_audio_stream():
    """主控 API：打断音频流"""
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        payload = json.dumps({"command": "stop"}).encode('utf-8')
        sock.sendto(payload, UDP_CTRL_TARGET)
        sock.close()
        logger.info("已向底层引擎发送停止指令！")
        return {"status": "success", "message": "已成功拦截音频流"}
    except Exception as e:
        return {"status": "error", "message": str(e)}

if __name__ == "__main__":
    import uvicorn
    # 🌟 注意：这里使用 8001 端口，绝不和视觉冲突
    uvicorn.run(app, host="0.0.0.0", port=8001)
