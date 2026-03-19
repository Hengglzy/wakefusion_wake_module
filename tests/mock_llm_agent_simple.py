"""
简化的Mock LLM Agent测试脚本
用于在没有真实服务端的情况下测试WakeFusion客户端链路

功能：
1. 接收ASR识别结果（audio_start/audio_end）
2. 接收二进制音频数据流
3. 自动回复（发送route消息）
4. 发送数字人动作指令（unity_command）
5. 发送模拟的音频流（二进制帧，16kHz PCM提示音）
6. 发送TTS_DONE消息
7. 支持interrupt打断机制
8. 支持stop强制休眠机制
9. 支持audio_cancel垃圾回收机制
10. 支持键盘监听（输入'q'发送interrupt，输入's'发送stop）

使用方法：
1. 启动所有WakeFusion服务（视觉、音频、Core Server）
2. 运行此脚本：python tests/mock_llm_agent_simple.py
3. 进行唤醒和对话测试
4. 输入'q'并按回车可发送interrupt指令
5. 输入's'并按回车可发送stop指令（强制休眠）
"""

import asyncio
import json
import logging
import websockets
import numpy as np
import time
import threading
import sys
from datetime import datetime

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger("MockAgent")

# 配置参数
WS_HOST = "127.0.0.1"  # 监听地址（本地测试使用127.0.0.1）
WS_PORT = 8080  # 端口号，必须与config.yaml中的llm_agent.host端口一致
SAMPLE_RATE = 16000  # 音频采样率（Hz），必须与config.yaml中的audio_playback.sample_rate一致
CHUNK_DURATION_MS = 100  # 音频块时长（毫秒），流式发送的块大小


class MockAgent:
    """模拟LLM Agent服务器"""
    
    def __init__(self):
        self.current_trace_id = None
        self.is_speaking = False
        self.response_task = None
        self.connected_clients = set()  # 存储所有连接的客户端
        self.loop = None  # 存储事件循环
    
    def parse_query_params(self, path: str) -> dict:
        """解析查询参数"""
        params = {}
        if "?" in path:
            query_string = path.split("?")[1]
            for param in query_string.split("&"):
                if "=" in param:
                    key, value = param.split("=", 1)
                    params[key] = value
        return params
    
    async def handler(self, websocket, path):
        """处理来自Core Server的WebSocket连接"""
        # 解析查询参数
        query_params = self.parse_query_params(path)
        device_id = query_params.get("deviceId", "unknown")
        token = query_params.get("token", "")
        
        # 添加到连接列表
        self.connected_clients.add(websocket)
        
        logger.info("=" * 60)
        logger.info("✅ 客户端 (Core Server) 已连接到模拟服务端!")
        logger.info(f"   设备ID: {device_id}")
        logger.info(f"   路径: {path}")
        logger.info("=" * 60)
        
        try:
            async for message in websocket:
                if isinstance(message, str):
                    # 处理客户端发来的文本/JSON控制指令
                    try:
                        data = json.loads(message)
                        msg_type = data.get("type", "")
                        
                        if msg_type == "audio_start":
                            # 用户开始说话
                            self.current_trace_id = data.get("trace_id")
                            logger.info(f"🎙️ 用户开始说话 (TraceID: {self.current_trace_id})")
                            
                            # 核心打断逻辑：如果AI正在说话，用户又发来了新的 audio_start，说明用户插话了！
                            if self.is_speaking and self.response_task:
                                logger.warning("🚨 检测到用户插话！立即终止当前的回答任务。")
                                self.response_task.cancel()
                                # 向客户端下发 interrupt 指令，要求客户端闭嘴
                                await websocket.send(json.dumps({
                                    "type": "interrupt",
                                    "traceId": self.current_trace_id
                                }))
                                self.is_speaking = False
                        
                        elif msg_type == "audio_cancel":
                            reason = data.get("reason", "unknown")
                            logger.warning(f"🗑️ 收到客户端 audio_cancel 指令，清空并丢弃当前音频缓存 (原因: {reason})")
                            # 重置服务端状态，清除痕迹，不触发任何大模型思考和回复
                            if self.response_task:
                                self.response_task.cancel()
                            self.is_speaking = False
                            self.current_trace_id = None
                        
                        elif msg_type == "audio_end":
                            # 用户说话结束
                            trace_id = data.get("trace_id")
                            reason = data.get("reason", "normal")
                            logger.info(f"🛑 用户说话结束 (TraceID: {trace_id}, 原因: {reason})，开始思考并回复...")
                            
                            # 触发AI回复的异步任务
                            self.response_task = asyncio.create_task(
                                self.send_response(websocket, trace_id)
                            )
                        
                        elif msg_type == "client_interrupt":
                            # 收到客户端的物理硬打断
                            reason = data.get("reason", "unknown")
                            logger.warning(f"🔪 收到客户端的物理硬打断 (原因: {reason})")
                            if self.response_task:
                                self.response_task.cancel()
                            self.is_speaking = False
                        
                        elif msg_type == "device_state":
                            # 收到设备状态上报
                            state = data.get("state", "")
                            state_icons = {
                                "idle": "💤",
                                "listening": "👂",
                                "thinking": "🤔",
                                "speaking": "🗣️"
                            }
                            icon = state_icons.get(state, "❓")
                            logger.info(f"{icon} 设备状态: {state.upper()}")
                        
                        elif msg_type == "timeout_exit":
                            # 收到超时退出信号
                            logger.info(f"⏰ 收到超时退出信号 (设备ID: {data.get('device_id', 'unknown')})")
                            if self.response_task:
                                self.response_task.cancel()
                            self.is_speaking = False
                        
                        else:
                            logger.debug(f"📨 收到其他消息: {msg_type}")
                    
                    except json.JSONDecodeError:
                        logger.warning(f"⚠️ 收到无效JSON消息: {message[:100]}")
                    except Exception as e:
                        logger.error(f"❌ 处理消息异常: {e}", exc_info=True)
                
                elif isinstance(message, bytes):
                    # 收到客户端传来的二进制声音数据
                    logger.debug(f"📦 收到音频数据 (长度: {len(message)} bytes)")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info("❌ 客户端断开连接")
            if self.response_task:
                self.response_task.cancel()
            self.is_speaking = False
        except Exception as e:
            logger.error(f"❌ 连接异常: {e}", exc_info=True)
        finally:
            # 清理状态
            self.connected_clients.discard(websocket)
            if self.response_task:
                self.response_task.cancel()
            self.is_speaking = False
            self.current_trace_id = None
    
    async def send_response(self, websocket, trace_id):
        """模拟大模型的思考和流式返回"""
        try:
            self.is_speaking = True
            
            # 1. 延时 0.5s
            await asyncio.sleep(0.5)
            
            # 2. 🌟 发送 ASR 结果（模拟）：用户语音转文字（附带时间戳证明动态性）
            current_time = datetime.now().strftime("%H:%M:%S")
            asr_text = f"你好小康，今天天气怎么样啊？({current_time})"
            logger.info(f"📤 发送ASR结果: {asr_text}")
            await websocket.send(json.dumps({
                "type": "asr",
                "text": asr_text
            }))
            # 兼容旧版协议
            await websocket.send(json.dumps({
                "type": "asr_result",
                "text": asr_text,
                "is_final": True,
                "confidence": 0.95,
                "timestamp": time.time()
            }))
            
            # 3. 延时 0.5s
            await asyncio.sleep(0.5)
            
            # 4. 🌟 发送 AI 回复文本（附带时间戳证明动态性）
            ai_text = f"我是小康，现在是 {current_time}，很高兴为你服务！"
            logger.info(f"📤 发送AI回复: {ai_text}")
            
            # 发送给 core_server，让其转发给 UI（新版协议）
            await websocket.send(json.dumps({
                "type": "text",
                "text": ai_text
            }))
            
            # 发送给 core_server，让其转发给 UI（兼容旧版）
            await websocket.send(json.dumps({
                "type": "route",
                "text": ai_text,
                "isFinal": True
            }))
            
            # 兼容 app.html 的直接识别
            await websocket.send(json.dumps({
                "type": "chat_reply",
                "text": ai_text
            }))
            
            # 5. 发送动作指令 (让 Unity 数字人做动作)
            action_command = "shuohua4"
            logger.info(f"📤 下发数字人动作指令: {action_command}")
            await websocket.send(json.dumps({
                "type": "unity_command",
                "payload": {
                    "type": "character.action",
                    "data": {
                        "assistant": "1",
                        "content": action_command
                    }
                }
            }))
            
            # 6. 生成并发送二进制音频 (模拟 TTS 发音)
            duration_sec = 2.5  # 声音持续 2.5 秒
            frequency = 440.0  # 440Hz (A4音符)
            volume = 0.3  # 音量（0.0-1.0）
            
            logger.info(f"📤 下发二进制音频流 (模拟 {duration_sec} 秒的提示音，{frequency}Hz)...")
            
            # 生成正弦波音频数据
            num_samples = int(SAMPLE_RATE * duration_sec)
            t = np.linspace(0, duration_sec, num_samples, False)
            # 生成 440Hz 的正弦波，转换为 int16 格式
            audio_data = np.sin(frequency * 2 * np.pi * t) * volume
            audio_int16 = (audio_data * 32767).astype(np.int16)
            audio_bytes = audio_int16.tobytes()
            
            # 将音频切分成小块，流式发给客户端（完全模拟真实网络传输）
            chunk_size = int(SAMPLE_RATE * CHUNK_DURATION_MS / 1000 * 2)  # 0.1秒的 int16 字节数
            num_chunks = len(audio_bytes) // chunk_size
            if len(audio_bytes) % chunk_size != 0:
                num_chunks += 1
            
            for i in range(0, len(audio_bytes), chunk_size):
                chunk = audio_bytes[i:i + chunk_size]
                await websocket.send(chunk)
                # 控制发送速率（模拟真实TTS流式输出）
                await asyncio.sleep(CHUNK_DURATION_MS / 1000.0)
            
            logger.info(f"✅ 音频发送完毕 (共 {num_chunks} 个块)")
            
            # 7. 音频发完了，发送结束信号
            await websocket.send(json.dumps({
                "type": "TTS_DONE"
            }))
            logger.info("✅ 已发送TTS_DONE消息")
        
        except asyncio.CancelledError:
            logger.warning("⚠️ 回复任务被中断 (Task Cancelled)")
        except Exception as e:
            logger.error(f"❌ 发送回复时发生错误: {e}", exc_info=True)
        finally:
            self.is_speaking = False
    
    async def broadcast_interrupt(self):
        """向所有连接的客户端广播interrupt指令"""
        if not self.connected_clients:
            return
        
        interrupt_msg = json.dumps({
            "type": "interrupt",
            "reason": "server_mock_interrupt"
        })
        
        # 向所有客户端发送interrupt
        disconnected = set()
        for client in self.connected_clients:
            try:
                await client.send(interrupt_msg)
                logger.info(f"📤 已向客户端广播 interrupt 指令")
            except Exception as e:
                logger.warning(f"⚠️ 向客户端发送interrupt失败: {e}")
                disconnected.add(client)
        
        # 清理断开的连接
        self.connected_clients -= disconnected
    
    async def broadcast_stop(self):
        """向所有连接的客户端广播stop指令"""
        if not self.connected_clients:
            return
        
        stop_msg = json.dumps({
            "type": "stop",
            "reason": "server_stopped_by_admin"
        })
        
        # 向所有客户端发送stop
        disconnected = set()
        for client in self.connected_clients:
            try:
                await client.send(stop_msg)
                logger.info(f"📤 已向客户端广播 stop 指令")
            except Exception as e:
                logger.warning(f"⚠️ 向客户端发送stop失败: {e}")
                disconnected.add(client)
        
        # 清理断开的连接
        self.connected_clients -= disconnected


def keyboard_listener(agent, loop):
    """键盘监听线程：读取stdin，输入'q'发送interrupt，输入's'发送stop"""
    logger.info("⌨️ 键盘监听线程已启动（输入'q'发送interrupt，输入's'发送stop指令）")
    while True:
        try:
            line = sys.stdin.readline().strip().lower()
            if line == 'q':
                logger.info("⌨️ 检测到'q'输入，发送interrupt指令...")
                # 使用线程安全的方式调用异步函数
                asyncio.run_coroutine_threadsafe(
                    agent.broadcast_interrupt(),
                    loop
                )
            elif line == 's':
                logger.info("⌨️ 检测到's'输入，发送stop指令（强制休眠）...")
                # 使用线程安全的方式调用异步函数
                asyncio.run_coroutine_threadsafe(
                    agent.broadcast_stop(),
                    loop
                )
            elif line:
                logger.info(f"⌨️ 收到输入: {line}（输入'q'发送interrupt，输入's'发送stop）")
        except Exception as e:
            logger.error(f"❌ 键盘监听异常: {e}")
            break


async def main():
    """启动Mock LLM Agent服务器"""
    logger.info("=" * 60)
    logger.info("🚀 Mock LLM Agent 测试服务器")
    logger.info("=" * 60)
    logger.info(f"监听地址: ws://{WS_HOST}:{WS_PORT}/api/voice/ws")
    logger.info(f"完整连接地址: ws://{WS_HOST}:{WS_PORT}/api/voice/ws?deviceId=<deviceId>&token=<token>")
    logger.info(f"音频配置: {SAMPLE_RATE}Hz, 单声道, {CHUNK_DURATION_MS}ms/块")
    logger.info("")
    logger.info("功能说明:")
    logger.info("  1. 接收 audio_start/audio_end 消息")
    logger.info("  2. 接收二进制音频数据流")
    logger.info("  3. 自动回复文本（'我是小康，很高兴为你服务...'）")
    logger.info("  4. 发送数字人动作指令（shuohua4）")
    logger.info("  5. 发送模拟音频流（440Hz正弦波，2.5秒）")
    logger.info("  6. 支持 interrupt 打断机制")
    logger.info("  7. 支持 stop 强制休眠机制")
    logger.info("  8. 支持 audio_cancel 垃圾回收机制")
    logger.info("  9. 支持键盘监听（输入'q'发送interrupt，输入's'发送stop）")
    logger.info("")
    logger.info("配置说明:")
    logger.info(f"  - 确保config/config.yaml中的llm_agent.host配置为: {WS_HOST}:{WS_PORT}")
    logger.info(f"  - 确保audio_playback.sample_rate配置为: {SAMPLE_RATE}Hz")
    logger.info("")
    logger.info("测试步骤:")
    logger.info("  1. 确保所有WakeFusion服务已启动")
    logger.info("  2. 进行唤醒和对话测试")
    logger.info("  3. 观察日志输出和音频播放")
    logger.info("  4. 输入'q'并按回车可发送interrupt指令")
    logger.info("  5. 输入's'并按回车可发送stop指令（强制休眠）")
    logger.info("=" * 60)
    logger.info("")
    
    agent = MockAgent()
    agent.loop = asyncio.get_event_loop()
    
    # 🌟 启动键盘监听线程
    kb_thread = threading.Thread(
        target=keyboard_listener,
        args=(agent, agent.loop),
        daemon=True
    )
    kb_thread.start()
    
    # 启动WebSocket服务器
    async def handler(websocket):
        # 获取路径（兼容不同版本的websockets库）
        try:
            if hasattr(websocket, 'path'):
                path = websocket.path
            elif hasattr(websocket, 'request') and hasattr(websocket.request, 'path'):
                path = websocket.request.path
            elif hasattr(websocket, 'raw_path'):
                path = websocket.raw_path.decode('utf-8') if isinstance(websocket.raw_path, bytes) else websocket.raw_path
            else:
                path = "/api/voice/ws"
                logger.warning("⚠️ 无法获取WebSocket路径，使用默认路径")
        except Exception as e:
            logger.warning(f"⚠️ 获取路径失败: {e}，使用默认路径")
            path = "/api/voice/ws"
        
        # 检查路径是否匹配
        if path.startswith("/api/voice/ws"):
            await agent.handler(websocket, path)
        else:
            logger.warning(f"⚠️ 收到不匹配的路径请求: {path}")
            try:
                await websocket.close(code=1008, reason="Path not found")
            except:
                pass
    
    async with websockets.serve(handler, WS_HOST, WS_PORT):
        logger.info(f"✅ 服务器已启动，等待Core Server连接...")
        logger.info(f"   按 Ctrl+C 停止服务器")
        logger.info("")
        
        # 保持运行
        await asyncio.Future()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("")
        logger.info("🛑 服务器已停止")
    except Exception as e:
        logger.error(f"❌ 服务器异常: {e}", exc_info=True)
