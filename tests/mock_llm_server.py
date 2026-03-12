import asyncio
import json
import logging
import websockets
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("mock_llm_server")

# 存储连接的客户端
core_clients = set()
ui_clients = set()

async def core_handler(websocket):
    """处理来自 Core Server 的连接 (Client Mode)"""
    logger.info(f"🔔 Core Server 已连接: {websocket.remote_address}")
    core_clients.add(websocket)
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                logger.info(f"📥 收到 Core 消息: {data.get('type')} - {data.get('text', '')[:50]}")
                
                # 转发给所有 UI 客户端，并进行协议转换
                if ui_clients:
                    # Core -> UI 协议转换
                    # Core: {"type": "asr", "stage": "final", "text": "...", "traceId": "...", "confidence": 0.95}
                    # UI: {"type": "asr_result", "text": "...", "is_final": ..., "confidence": ..., "timestamp": ...}
                    if data.get("type") == "asr":
                        ui_msg = {
                            "type": "asr_result",
                            "text": data.get("text", ""),
                            "is_final": data.get("stage") == "final",
                            "confidence": data.get("confidence", 1.0),
                            "timestamp": data.get("timestamp", datetime.now().timestamp())
                        }
                        # 通知 UI
                        await asyncio.gather(*[client.send(json.dumps(ui_msg)) for client in ui_clients])
                    else:
                        # 其他消息直接转发
                        await asyncio.gather(*[client.send(message) for client in ui_clients])
            except json.JSONDecodeError:
                logger.warning(f"⚠️ 收到无效 JSON 消息: {message}")
            except Exception as e:
                logger.error(f"❌ 处理 Core 消息异常: {e}")
    finally:
        core_clients.remove(websocket)
        logger.info(f"📴 Core Server 断开连接: {websocket.remote_address}")

async def ui_handler(websocket):
    """处理来自 Mock LLM.html UI 的连接"""
    logger.info(f"🎨 UI 客户端已连接: {websocket.remote_address}")
    ui_clients.add(websocket)
    try:
        async for message in websocket:
            try:
                data = json.loads(message)
                logger.info(f"📤 收到 UI 消息: {data.get('type')} - {data.get('text', '')[:50]}")
                
                # 转发给所有 Core 客户端，并进行协议转换
                if core_clients:
                    # UI -> Core 协议转换
                    # UI: {"type": "tts_request", "text": "...", "is_final": ...}
                    # Core: {"type": "route", "text": "...", "isFinal": ...}
                    if data.get("type") == "tts_request":
                        core_msg = {
                            "type": "route",
                            "text": data.get("text", ""),
                            "isFinal": data.get("is_final", False)
                        }
                        await asyncio.gather(*[client.send(json.dumps(core_msg)) for client in core_clients])
                    elif data.get("type") == "stop_synthesis":
                        core_msg = {"type": "stop_tts"}
                        await asyncio.gather(*[client.send(json.dumps(core_msg)) for client in core_clients])
                    else:
                        # 其他消息直接转发
                        await asyncio.gather(*[client.send(message) for client in core_clients])
            except json.JSONDecodeError:
                logger.warning(f"⚠️ 收到无效 JSON 消息: {message}")
            except Exception as e:
                logger.error(f"❌ 处理 UI 消息异常: {e}")
    finally:
        ui_clients.remove(websocket)
        logger.info(f"📴 UI 客户端断开连接: {websocket.remote_address}")

async def main():
    # 命令行参数配置（可选，这里直接写死）
    core_port = 8080
    ui_port = 8765
    
    # 启动两个服务
    async with websockets.serve(core_handler, "127.0.0.1", core_port):
        async with websockets.serve(ui_handler, "127.0.0.1", ui_port):
            logger.info(f"🚀 Mock LLM Relay Server 已启动!")
            logger.info(f"   1. 等待 Core Server 连接 (Client Mode): ws://127.0.0.1:{core_port}")
            logger.info(f"   2. 等待 Mock LLM.html UI 连接: ws://127.0.0.1:{ui_port}")
            logger.info(f"   请在浏览器中打开 tests/Mock LLM.html 并点击连接")
            
            # 保持运行
            await asyncio.Future()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 服务已停止")
