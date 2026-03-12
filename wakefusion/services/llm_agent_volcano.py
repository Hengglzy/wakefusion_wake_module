"""
LLM Agent 服务 - 火山引擎豆包模型
基于统一WebSocket协议，接收ASR结果，调用火山引擎API生成回答，流式返回TTS文本

通信协议：
  - WebSocket Server：接收Core Server的连接（ws://127.0.0.1:8080/api/voice/ws）
  - 调用火山引擎API：https://ark.cn-beijing.volces.com/api/v3/responses
"""
import asyncio
import json
import logging
import time
import websockets
import websockets.exceptions
from typing import Optional, Dict, Set
from urllib.parse import parse_qs
import aiohttp

# 配置日志
logger = logging.getLogger("llm_agent_volcano")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)


class VolcanoLLMAgent:
    """火山引擎LLM Agent服务"""
    
    def __init__(self, config):
        """
        初始化LLM Agent
        
        Args:
            config: 应用配置对象
        """
        self.config = config
        self.llm_config = config.llm_agent
        
        # 火山引擎API配置（从配置文件读取，如果没有则使用默认值）
        self.volcano_api_url = getattr(self.llm_config, 'volcano_api_url', None) or "https://ark.cn-beijing.volces.com/api/v3/responses"
        self.volcano_api_key = getattr(self.llm_config, 'volcano_api_key', None) or "13855083-26c0-423a-9720-f0d02f9e7668"
        self.volcano_model = getattr(self.llm_config, 'volcano_model', None) or "doubao-seed-2-0-mini-260215"
        
        # WebSocket连接管理
        self.connected_clients: Dict[str, any] = {}  # deviceId -> websocket
        self.device_sessions: Dict[str, Dict] = {}  # deviceId -> session info
        
        # 当前活跃请求（按deviceId管理）
        self.active_requests: Dict[str, Optional[str]] = {}  # deviceId -> traceId
        
        # HTTP客户端（用于调用火山引擎API）
        self.http_session: Optional[aiohttp.ClientSession] = None
        
        logger.info("🌋 火山引擎LLM Agent初始化完成")
        logger.info(f"  API地址: {self.volcano_api_url}")
        logger.info(f"  模型: {self.volcano_model}")
    
    async def _init_http_session(self):
        """初始化HTTP客户端"""
        if self.http_session is None:
            self.http_session = aiohttp.ClientSession()
    
    async def _close_http_session(self):
        """关闭HTTP客户端"""
        if self.http_session:
            await self.http_session.close()
            self.http_session = None
    
    async def _call_volcano_api(self, text: str, device_id: str) -> str:
        """
        调用火山引擎API生成回答
        
        Args:
            text: 用户输入的文本（ASR识别结果）
            device_id: 设备ID（用于会话隔离）
        
        Returns:
            LLM生成的回答文本
        """
        try:
            await self._init_http_session()
            
            # 构建请求体（参考curl命令格式）
            payload = {
                "model": self.volcano_model,
                "input": [
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": text
                            }
                        ]
                    }
                ]
            }
            
            headers = {
                "Authorization": f"Bearer {self.volcano_api_key}",
                "Content-Type": "application/json"
            }
            
            logger.info(f"🌋 调用火山引擎API: {text[:50]}...")
            logger.debug(f"📤 请求URL: {self.volcano_api_url}")
            logger.debug(f"📤 请求Payload: {json.dumps(payload, ensure_ascii=False, indent=2)}")
            start_time = time.time()
            
            async with self.http_session.post(
                self.volcano_api_url,
                json=payload,
                headers=headers,
                timeout=aiohttp.ClientTimeout(total=30)
            ) as response:
                elapsed = time.time() - start_time
                logger.info(f"📥 收到API响应: HTTP {response.status} (耗时: {elapsed:.2f}s)")
                
                if response.status != 200:
                    error_text = await response.text()
                    logger.error(f"❌ 火山引擎API错误: {response.status}")
                    logger.error(f"   错误详情: {error_text[:500]}")
                    return "抱歉，我暂时无法回答这个问题。"
                
                # 获取原始响应文本（用于调试）
                response_text = await response.text()
                logger.debug(f"📥 原始响应文本: {response_text[:500]}...")
                
                try:
                    result = json.loads(response_text)
                except json.JSONDecodeError as e:
                    logger.error(f"❌ 响应不是有效的JSON: {e}")
                    logger.error(f"   响应内容: {response_text[:500]}")
                    return "抱歉，服务返回了无效的数据格式。"
                
                logger.debug(f"📥 解析后的JSON结构: {json.dumps(result, ensure_ascii=False, indent=2)[:1000]}...")
                
                # 解析响应（根据火山引擎API的实际响应格式）
                # 尝试多种可能的响应格式
                answer = None
                
                # 格式1: {"output": [{"choices": [{"message": {"content": "..."}}]}]}
                if "output" in result and isinstance(result["output"], list) and len(result["output"]) > 0:
                    output_item = result["output"][0]
                    logger.debug(f"🔍 找到output数组，第一项: {json.dumps(output_item, ensure_ascii=False)[:200]}...")
                    if "choices" in output_item and isinstance(output_item["choices"], list) and len(output_item["choices"]) > 0:
                        choice = output_item["choices"][0]
                        logger.debug(f"🔍 找到choices数组，第一项: {json.dumps(choice, ensure_ascii=False)[:200]}...")
                        if "message" in choice and "content" in choice["message"]:
                            answer = choice["message"]["content"]
                            logger.debug(f"✅ 从output[0].choices[0].message.content提取答案")
                        elif "text" in choice:
                            answer = choice["text"]
                            logger.debug(f"✅ 从output[0].choices[0].text提取答案")
                    elif "text" in output_item:
                        answer = output_item["text"]
                        logger.debug(f"✅ 从output[0].text提取答案")
                
                # 格式2: {"choices": [{"message": {"content": "..."}}]}
                if not answer and "choices" in result and isinstance(result["choices"], list) and len(result["choices"]) > 0:
                    choice = result["choices"][0]
                    logger.debug(f"🔍 找到choices数组（顶层），第一项: {json.dumps(choice, ensure_ascii=False)[:200]}...")
                    if "message" in choice and "content" in choice["message"]:
                        answer = choice["message"]["content"]
                        logger.debug(f"✅ 从choices[0].message.content提取答案")
                    elif "text" in choice:
                        answer = choice["text"]
                        logger.debug(f"✅ 从choices[0].text提取答案")
                
                # 格式3: {"text": "..."} 或 {"content": "..."}
                if not answer:
                    answer = result.get("text") or result.get("content") or result.get("answer")
                    if answer:
                        logger.debug(f"✅ 从顶层text/content/answer提取答案")
                
                # 格式4: 直接是字符串
                if not answer and isinstance(result, str):
                    answer = result
                    logger.debug(f"✅ 响应本身就是字符串")
                
                # 如果还是找不到，记录完整响应以便调试
                if not answer:
                    logger.warning(f"⚠️ 无法解析火山引擎API响应")
                    logger.warning(f"   响应结构: {json.dumps(result, ensure_ascii=False, indent=2)}")
                    logger.warning(f"   响应键: {list(result.keys()) if isinstance(result, dict) else 'N/A'}")
                    answer = "抱歉，我暂时无法理解这个问题。"
                
                logger.info(f"✅ 火山引擎API响应: {answer[:50]}... (耗时: {elapsed:.2f}s)")
                return answer if answer else "抱歉，我暂时无法回答这个问题。"
        
        except asyncio.TimeoutError:
            logger.error("❌ 火山引擎API超时（30秒）")
            return "抱歉，响应超时，请稍后再试。"
        except aiohttp.ClientConnectorError as e:
            logger.error(f"❌ 火山引擎API连接失败: {e}")
            logger.error(f"   可能原因：网络连接问题、防火墙阻止、SSL证书问题")
            logger.error(f"   API地址: {self.volcano_api_url}")
            return "抱歉，无法连接到服务，请检查网络连接。"
        except aiohttp.ClientError as e:
            logger.error(f"❌ 火山引擎API客户端错误: {e}")
            logger.error(f"   错误类型: {type(e).__name__}")
            return "抱歉，服务暂时不可用。"
        except Exception as e:
            logger.error(f"❌ 调用火山引擎API异常: {e}", exc_info=True)
            logger.error(f"   异常类型: {type(e).__name__}")
            return "抱歉，服务暂时不可用。"
    
    async def _handle_asr_message(self, websocket, data: dict, device_id: str):
        """处理ASR识别结果"""
        stage = data.get("stage")
        text = data.get("text", "").strip()
        trace_id = data.get("traceId")
        
        if not text:
            return
        
        # 只处理final结果，触发LLM回答
        if stage == "final":
            logger.info(f"📥 [设备: {device_id}] 收到ASR最终结果: {text} (traceId: {trace_id})")
            
            # 取消之前的请求（如果有）
            old_trace_id = self.active_requests.get(device_id)
            if old_trace_id and old_trace_id != trace_id:
                logger.info(f"🛑 取消旧请求: {old_trace_id}")
                await self._send_stop_tts(websocket, old_trace_id)
            
            # 记录当前活跃请求
            self.active_requests[device_id] = trace_id
            
            # 发送meta消息（确认请求已受理）
            await self._send_meta(websocket, trace_id, device_id)
            
            # 调用火山引擎API生成回答
            try:
                answer = await self._call_volcano_api(text, device_id)
                
                # 检查是否被中断
                if self.active_requests.get(device_id) != trace_id:
                    logger.warning(f"⚠️ 请求已被中断，丢弃回答: {trace_id}")
                    return
                
                # 流式发送回答文本（按标点符号切分）
                await self._send_streaming_text(websocket, answer, trace_id)
                
                # 发送final消息
                await self._send_final(websocket, trace_id, "ok")
                
            except Exception as e:
                logger.error(f"❌ 处理ASR消息异常: {e}")
                await self._send_error(websocket, trace_id, "assistant_stream_failed", str(e))
        else:
            # partial结果，只记录日志
            logger.debug(f"📥 [设备: {device_id}] 收到ASR中间结果: {text[:30]}...")
    
    async def _handle_interrupt_message(self, websocket, data: dict, device_id: str):
        """处理中断消息"""
        trace_id = data.get("traceId")
        reason = data.get("reason", "unknown")
        
        logger.info(f"🛑 [设备: {device_id}] 收到中断请求: traceId={trace_id}, reason={reason}")
        
        # 取消当前活跃请求
        if self.active_requests.get(device_id) == trace_id:
            self.active_requests[device_id] = None
        
        # 发送停止TTS信号
        await self._send_stop_tts(websocket, trace_id)
    
    async def _handle_device_state_message(self, data: dict, device_id: str):
        """处理设备状态消息"""
        state = data.get("state")
        logger.debug(f"📊 [设备: {device_id}] 设备状态: {state}")
    
    async def _handle_ping_message(self, websocket, data: dict, device_id: str):
        """处理ping消息"""
        await websocket.send(json.dumps({
            "type": "pong",
            "deviceId": device_id,
            "timestamp": time.time()
        }))
    
    async def _send_meta(self, websocket, trace_id: str, device_id: str):
        """发送meta消息（确认请求已受理）"""
        message = {
            "type": "meta",
            "traceId": trace_id,
            "deviceId": device_id,
            "sessionKey": f"voice:{device_id}",
            "agentId": "main",
            "tenantId": "default",
            "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S.000Z", time.gmtime())
        }
        await websocket.send(json.dumps(message))
        logger.debug(f"📤 [设备: {device_id}] 已发送meta消息: {trace_id}")
    
    async def _send_streaming_text(self, websocket, text: str, trace_id: str):
        """
        流式发送文本（按标点符号切分，模拟流式输出）
        
        Args:
            websocket: WebSocket连接
            text: 完整回答文本
            trace_id: 请求标识
        """
        import re
        
        # 按标点符号切分文本（模拟流式输出）
        # 切分点：句号、问号、感叹号、逗号
        pattern = re.compile(r'([。！？，,])')
        parts = pattern.split(text)
        
        current_chunk = ""
        for i, part in enumerate(parts):
            current_chunk += part
            
            # 如果遇到句号、问号、感叹号，或者累积了一定长度，发送一个chunk
            if part in ['。', '！', '？', '.', '!', '?'] or len(current_chunk) >= 20:
                if current_chunk.strip():
                    message = {
                        "type": "route",
                        "text": current_chunk,
                        "isFinal": False,
                        "traceId": trace_id
                    }
                    await websocket.send(json.dumps(message, ensure_ascii=False))
                    logger.debug(f"📤 流式文本块: {current_chunk[:30]}...")
                    current_chunk = ""
                    
                    # 小延迟，模拟流式输出
                    await asyncio.sleep(0.1)
        
        # 发送最后一块
        if current_chunk.strip():
            message = {
                "type": "route",
                "text": current_chunk,
                "isFinal": True,
                "traceId": trace_id
            }
            await websocket.send(json.dumps(message, ensure_ascii=False))
            logger.debug(f"📤 流式文本块（最终）: {current_chunk[:30]}...")
        else:
            # 如果没有剩余文本，发送空的final标记
            message = {
                "type": "route",
                "text": "",
                "isFinal": True,
                "traceId": trace_id
            }
            await websocket.send(json.dumps(message, ensure_ascii=False))
    
    async def _send_final(self, websocket, trace_id: str, status: str = "ok"):
        """发送final消息"""
        message = {
            "type": "final",
            "traceId": trace_id,
            "status": status,
            "route": "fast"
        }
        await websocket.send(json.dumps(message))
        logger.debug(f"📤 已发送final消息: {trace_id}")
    
    async def _send_stop_tts(self, websocket, trace_id: str):
        """发送停止TTS信号"""
        message = {
            "type": "stop_tts",
            "traceId": trace_id,
            "reason": "interrupt"
        }
        await websocket.send(json.dumps(message))
        logger.debug(f"📤 已发送stop_tts消息: {trace_id}")
    
    async def _send_error(self, websocket, trace_id: str, code: str, message: str):
        """发送错误消息"""
        error_msg = {
            "type": "error",
            "traceId": trace_id,
            "code": code,
            "message": message
        }
        await websocket.send(json.dumps(error_msg))
        logger.error(f"📤 已发送错误消息: {code} - {message}")
    
    async def _handle_websocket_connection(self, websocket):
        """处理WebSocket连接"""
        # 从websocket对象获取path（新版本websockets库）
        # 尝试多种方式获取path
        path = '/'
        if hasattr(websocket, 'path'):
            path = websocket.path
        elif hasattr(websocket, 'request') and hasattr(websocket.request, 'path'):
            path = websocket.request.path
        elif hasattr(websocket, 'raw_path'):
            path = websocket.raw_path.decode('utf-8') if isinstance(websocket.raw_path, bytes) else websocket.raw_path
        
        # 解析连接参数
        query_params = parse_qs(path.split('?')[1] if '?' in path else '')
        device_id = query_params.get('deviceId', [None])[0]
        token = query_params.get('token', [None])[0]
        
        if not device_id:
            logger.error("❌ 缺少deviceId参数，拒绝连接")
            await websocket.close(code=4000, reason="Missing deviceId")
            return
        
        if not token:
            logger.warning(f"⚠️ 设备 {device_id} 缺少token参数，但允许连接（开发模式）")
        
        logger.info(f"🔔 设备已连接: {device_id} ({websocket.remote_address})")
        
        # 如果该设备已有连接，关闭旧连接
        if device_id in self.connected_clients:
            old_ws = self.connected_clients[device_id]
            try:
                await old_ws.close(code=1000, reason="New connection from same device")
            except:
                pass
        
        # 记录新连接
        self.connected_clients[device_id] = websocket
        self.device_sessions[device_id] = {
            "connected_at": time.time(),
            "trace_id": None
        }
        self.active_requests[device_id] = None
        
        try:
            # 接收消息循环
            async for message in websocket:
                try:
                    data = json.loads(message)
                    msg_type = data.get("type")
                    
                    if msg_type == "asr":
                        await self._handle_asr_message(websocket, data, device_id)
                    elif msg_type == "interrupt":
                        await self._handle_interrupt_message(websocket, data, device_id)
                    elif msg_type == "device_state":
                        await self._handle_device_state_message(data, device_id)
                    elif msg_type == "ping":
                        await self._handle_ping_message(websocket, data, device_id)
                    else:
                        logger.warning(f"⚠️ 未知消息类型: {msg_type}")
                
                except json.JSONDecodeError:
                    logger.warning(f"⚠️ 无效的JSON消息: {message}")
                except Exception as e:
                    logger.error(f"❌ 处理消息异常: {e}")
        
        except websockets.exceptions.ConnectionClosed:
            logger.info(f"📴 设备断开连接: {device_id}")
        except Exception as e:
            logger.error(f"❌ WebSocket连接异常: {e}")
        finally:
            # 清理连接
            if device_id in self.connected_clients:
                del self.connected_clients[device_id]
            if device_id in self.device_sessions:
                del self.device_sessions[device_id]
            if device_id in self.active_requests:
                del self.active_requests[device_id]
    
    async def start(self, host: str = "127.0.0.1", port: int = 8080):
        """启动LLM Agent服务"""
        logger.info(f"🚀 启动火山引擎LLM Agent服务...")
        logger.info(f"  WebSocket地址: ws://{host}:{port}/api/voice/ws")
        
        async with websockets.serve(
            self._handle_websocket_connection,
            host,
            port,
            ping_interval=None  # 禁用自动ping，使用应用层ping
        ):
            logger.info(f"✅ 火山引擎LLM Agent服务已启动")
            logger.info(f"   等待设备连接...")
            
            # 保持运行
            await asyncio.Future()
    
    async def stop(self):
        """停止服务"""
        logger.info("🛑 正在停止火山引擎LLM Agent服务...")
        
        # 关闭所有连接
        for device_id, websocket in list(self.connected_clients.items()):
            try:
                await websocket.close(code=1000, reason="Server shutdown")
            except:
                pass
        
        # 关闭HTTP客户端
        await self._close_http_session()
        
        logger.info("✅ 服务已停止")


async def main():
    """主函数"""
    import sys
    from pathlib import Path
    from wakefusion.config import get_config
    
    # 自动定位配置文件
    # 从 wakefusion/services/llm_agent_volcano.py 到项目根目录
    current_file = Path(__file__).resolve()
    project_root = current_file.parents[2]  # wakefusion/services -> wakefusion -> project_root
    config_path = project_root / "config" / "config.yaml"
    
    if not config_path.exists():
        logger.error(f"❌ 配置文件不存在: {config_path}")
        logger.error(f"   当前文件: {current_file}")
        logger.error(f"   项目根目录: {project_root}")
        sys.exit(1)
    
    logger.info(f"📄 自动定位配置文件: {config_path}")
    config = get_config(str(config_path))
    
    # 创建LLM Agent
    agent = VolcanoLLMAgent(config)
    
    # 获取配置中的端口
    llm_config = config.llm_agent
    host = llm_config.host.split(':')[0] if ':' in llm_config.host else "127.0.0.1"
    port = int(llm_config.host.split(':')[1]) if ':' in llm_config.host else 8080
    
    try:
        await agent.start(host=host, port=port)
    except KeyboardInterrupt:
        logger.info("🛑 收到中断信号")
    finally:
        await agent.stop()


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("🛑 服务已停止")
