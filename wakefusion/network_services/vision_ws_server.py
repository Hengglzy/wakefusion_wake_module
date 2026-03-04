"""
视觉 WebSocket 服务器（UDP 监听版）
作为 UDP 消费者，从 VisionService 的 UDP 输出中读取数据并提供 WebSocket 推送服务

架构说明：
- 本服务不导入或启动 VisionService
- 仅作为 UDP 消费者，监听 VisionService 发送的 UDP 数据
- 将接收到的数据通过 WebSocket 推送给客户端

使用方法:
    python -m wakefusion.network_services.vision_ws_server
"""

import asyncio
import json
import logging
import socket
from typing import Dict, Any, Optional, List
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
import uvicorn

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("vision_ws_server")

# 全局变量：存储最新的视觉检测数据
latest_vision_data: Optional[Dict[str, Any]] = None
_udp_lock = asyncio.Lock()  # 保护 latest_vision_data 的并发访问

# UDP 监听配置
UDP_HOST = "127.0.0.1"
UDP_PORT = 9999

# 创建 FastAPI 应用
app = FastAPI(title="Vision WebSocket Server (UDP Consumer)")


async def listen_udp_data():
    """
    后台 UDP 监听任务
    持续接收 VisionService 发送的 UDP 数据并更新全局变量
    """
    global latest_vision_data
    
    logger.info(f"启动 UDP 监听: {UDP_HOST}:{UDP_PORT}")
    
    # 创建 UDP Socket
    sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
    sock.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    sock.setblocking(False)  # 非阻塞模式
    
    try:
        sock.bind((UDP_HOST, UDP_PORT))
        logger.info(f"UDP Socket 已绑定到 {UDP_HOST}:{UDP_PORT}")
    except OSError as e:
        logger.error(f"UDP Socket 绑定失败: {e}")
        logger.error("请确保 VisionService 未占用该端口，或修改 UDP_PORT")
        return
    
    # 创建事件循环用于接收数据
    loop = asyncio.get_event_loop()
    
    while True:
        try:
            # 非阻塞接收数据
            try:
                data, addr = sock.recvfrom(65535)  # 最大 UDP 包大小
            except BlockingIOError:
                # 没有数据可读，等待一小段时间
                await asyncio.sleep(0.01)
                continue
            
            # 解析 JSON 数据
            try:
                json_str = data.decode('utf-8')
                vision_data = json.loads(json_str)
                
                # 更新全局变量（使用锁保护）
                async with _udp_lock:
                    latest_vision_data = vision_data
                
                logger.debug(f"收到 UDP 数据: {addr}, 数据大小: {len(data)} bytes")
                
            except json.JSONDecodeError as e:
                logger.warning(f"UDP 数据 JSON 解析失败: {e}, 数据: {data[:100]}")
                continue
            except UnicodeDecodeError as e:
                logger.warning(f"UDP 数据解码失败: {e}")
                continue
            except Exception as e:
                logger.error(f"处理 UDP 数据时出错: {e}", exc_info=True)
                continue
                
        except Exception as e:
            logger.error(f"UDP 监听循环出错: {e}", exc_info=True)
            await asyncio.sleep(0.1)
    
    # 清理（理论上不会执行到这里）
    sock.close()
    logger.info("UDP Socket 已关闭")


@app.websocket("/ws/vision")
async def websocket_vision(websocket: WebSocket):
    """
    WebSocket 路由：推送视觉检测结果
    
    推送频率：1Hz（每秒一次）
    数据格式：{"status": "detected", "people": [{"distance": 2.5, "score": 0.95}]}
    """
    await websocket.accept()
    logger.info(f"WebSocket 客户端已连接: {websocket.client}")
    
    try:
        while True:
            # 1Hz 频率：每秒推送一次
            await asyncio.sleep(1.0)
            
            # 读取最新的视觉数据（使用锁保护）
            async with _udp_lock:
                current_data = latest_vision_data
            
            if current_data is None:
                # 还没有收到任何 UDP 数据，跳过本次发送
                continue
            
            # 提取 faces 数组
            faces = current_data.get("faces", [])
            if not isinstance(faces, list):
                continue
            
            # 过滤逻辑：只保留 distance <= 4.0 且 score > 0.5 的人脸
            filtered_people = []
            for face in faces:
                if not isinstance(face, dict):
                    continue
                
                # 获取距离（米）
                distance_m = face.get("distance_m")
                if distance_m is None:
                    continue
                
                # 获取置信度（score）
                # 注意：VisionService 可能使用 "confidence" 字段，这里兼容两种命名
                score = face.get("score") or face.get("confidence", 0.0)
                
                try:
                    distance = float(distance_m)
                    score_float = float(score)
                    
                    # 过滤条件：distance <= 4.0 且 score > 0.5
                    if distance <= 4.0 and score_float > 0.5:
                        filtered_people.append({
                            "distance": distance,
                            "score": score_float
                        })
                except (ValueError, TypeError):
                    continue
            
            # 如果有符合条件的人物，发送数据
            if filtered_people:
                message = {
                    "status": "detected",
                    "people": filtered_people
                }
                await websocket.send_json(message)
                logger.debug(f"已推送检测结果: {len(filtered_people)} 个人物")
            # 如果数组为空，跳过本次发送（不发送空消息）
            
    except WebSocketDisconnect:
        logger.info(f"WebSocket 客户端已断开: {websocket.client}")
    except Exception as e:
        logger.error(f"WebSocket 处理出错: {e}", exc_info=True)
    finally:
        logger.info(f"WebSocket 连接已关闭: {websocket.client}")


@app.on_event("startup")
async def startup_event():
    """应用启动时启动 UDP 监听任务"""
    logger.info("正在启动 Vision WebSocket 服务器（UDP 消费者模式）...")
    logger.info(f"等待 VisionService 的 UDP 数据: {UDP_HOST}:{UDP_PORT}")
    
    # 启动后台 UDP 监听任务
    asyncio.create_task(listen_udp_data())
    
    logger.info("Vision WebSocket 服务器已启动")


@app.on_event("shutdown")
async def shutdown_event():
    """应用关闭时清理资源"""
    logger.info("正在关闭 Vision WebSocket 服务器...")
    logger.info("Vision WebSocket 服务器已关闭")


@app.get("/")
async def root():
    """根路径，返回服务信息"""
    return {
        "service": "Vision WebSocket Server (UDP Consumer)",
        "status": "running",
        "websocket_endpoint": "/ws/vision",
        "udp_listening": f"{UDP_HOST}:{UDP_PORT}",
        "data_available": latest_vision_data is not None
    }


@app.get("/health")
async def health():
    """健康检查端点"""
    return {
        "status": "healthy",
        "udp_listening": f"{UDP_HOST}:{UDP_PORT}",
        "data_available": latest_vision_data is not None,
        "latest_data_keys": list(latest_vision_data.keys()) if latest_vision_data else []
    }


def main():
    """主函数：启动 WebSocket 服务器"""
    logger.info("=" * 60)
    logger.info("🚀 Vision WebSocket 服务器（UDP 消费者模式）")
    logger.info("=" * 60)
    logger.info("WebSocket 端点: ws://0.0.0.0:8000/ws/vision")
    logger.info(f"UDP 监听: {UDP_HOST}:{UDP_PORT}")
    logger.info("健康检查: http://0.0.0.0:8000/health")
    logger.info("=" * 60)
    logger.info("⚠️  注意：请确保 VisionService 正在运行并发送 UDP 数据")
    logger.info("=" * 60)
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info"
    )


if __name__ == "__main__":
    main()
