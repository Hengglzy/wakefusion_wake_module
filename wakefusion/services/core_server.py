"""
核心决策模块 (Core Server)
整合视觉和音频输入，实现多模态唤醒和状态管理
"""
import zmq
import json
import time
import threading
import queue
import logging
import uuid
import asyncio
from enum import Enum
from typing import Optional
from wakefusion.config import get_config
from wakefusion.types import SystemState

# 配置日志
logger = logging.getLogger("core_server")
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
logger.addHandler(handler)


class CoreServer:
    """核心决策服务器"""
    
    def __init__(self, config_path: Optional[str] = None):
        """
        初始化核心服务器
        
        Args:
            config_path: 配置文件路径（可选）
        """
        # 加载配置
        self.config = get_config(config_path)
        self.zmq_config = self.config.zmq
        self.vision_wake_config = self.config.vision_wake
        self.audio_threshold_config = self.config.audio_threshold
        self.conversation_config = self.config.conversation
        self.llm_agent_config = self.config.llm_agent
        
        # 初始化ZMQ Context
        self.zmq_context = zmq.Context()
        
        # ZMQ SUB Sockets（订阅视觉和音频数据）
        self.vision_sub_socket = self.zmq_context.socket(zmq.SUB)
        self.vision_sub_socket.connect(f"tcp://127.0.0.1:{self.zmq_config.vision_pub_port}")
        self.vision_sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        
        self.audio_sub_socket = self.zmq_context.socket(zmq.SUB)
        self.audio_sub_socket.connect(f"tcp://127.0.0.1:{self.zmq_config.audio_pub_port}")
        self.audio_sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        
        # ZMQ REQ Socket（控制音频模块）
        self.audio_req_socket = self.zmq_context.socket(zmq.REQ)
        self.audio_req_socket.connect(f"tcp://127.0.0.1:{self.zmq_config.audio_ctrl_port}")
        self.audio_req_socket.setsockopt(zmq.REQ_RELAXED, 1)
        self.audio_req_socket.setsockopt(zmq.REQ_CORRELATE, 1)
        self.audio_req_socket.setsockopt(zmq.RCVTIMEO, self.zmq_config.req_rep_timeout_ms)
        
        # ZMQ PUSH Socket（转发音频到ASR）- 全局长连接防积压
        self._asr_push_socket = self.zmq_context.socket(zmq.PUSH)
        self._asr_push_socket.setsockopt(zmq.SNDHWM, 50)  # 最多堆积50个音频块，超过主动丢弃
        self._asr_push_socket.connect(f"tcp://127.0.0.1:{self.zmq_config.asr_pull_port}")
        
        # ZMQ PULL Socket（接收ASR识别结果）
        self._asr_result_pull_socket = self.zmq_context.socket(zmq.PULL)
        self._asr_result_pull_socket.setsockopt(zmq.RCVHWM, 50)  # 接收端限制积压
        self._asr_result_pull_socket.bind(f"tcp://127.0.0.1:{self.zmq_config.asr_result_push_port}")
        
        # ZMQ PUSH Socket（发送TTS合成文本给TTS模块）
        self._tts_text_push_socket = self.zmq_context.socket(zmq.PUSH)
        self._tts_text_push_socket.setsockopt(zmq.SNDHWM, 50)  # 发送端限制积压
        self._tts_text_push_socket.connect(f"tcp://127.0.0.1:{self.zmq_config.tts_text_pull_port}")
        
        # VAD 防抖参数（展厅抗噪核心）
        self._vad_speech_count: int = 0
        self._vad_silence_count: int = 0
        
        # ZMQ PULL Socket（接收TTS音频）
        self._tts_pull_socket: Optional[zmq.Socket] = None
        
        # ZMQ PUB Socket（发送停止信号给TTS）
        self._tts_stop_pub_socket: Optional[zmq.Socket] = None
        
        # ZMQ REP Socket（接收LLM控制指令）
        self._control_rep_socket: Optional[zmq.Socket] = None
        
        # 音频播放队列（可清空）
        self.audio_playback_queue = queue.Queue()
        self._playback_active = False
        self._playback_thread: Optional[threading.Thread] = None
        
        # 硬件冷却期（硬打断后）
        self._cooldown_until: float = 0.0
        
        # 🌟 新增：用于屏蔽麦克风瞬间开启时的物理电流声或尾音回声
        self._ignore_audio_until: float = 0.0
        
        # Poller用于同时监听多个socket
        self.poller = zmq.Poller()
        self.poller.register(self.vision_sub_socket, zmq.POLLIN)
        self.poller.register(self.audio_sub_socket, zmq.POLLIN)
        self.poller.register(self._asr_result_pull_socket, zmq.POLLIN)
        
        # 状态管理
        self.current_state = SystemState.IDLE
        
        # 视觉状态缓存
        self._latest_vision_wake: bool = False
        self._latest_vision_is_talking: bool = False
        self._last_vision_timestamp: float = 0.0
        
        # VAD超时管理
        self._last_vad_time: float = 0.0
        self._last_lip_active_time: float = time.time()  # 唇动计时器
        self.current_silence_timeout: float = self.conversation_config.vad_silence_timeout_default_sec
        self._vad_timeout_thread: Optional[threading.Thread] = None
        self._vad_timeout_should_exit: bool = False
        
        # 🌟 保底机制：30秒交流上限时间（防止视觉受损和极度噪音环境下数据无限制传输）
        self._listening_start_time: float = 0.0  # 进入LISTENING状态的时间戳
        self._max_listening_duration: float = 30.0  # 最大监听时长（秒）
        
        # 对话轮次和唤醒路径管理
        self._conversation_round: int = 0  # 对话轮次：0=首次唤醒，1+=持续对话
        self._wake_path: str = "unknown"  # 唤醒路径："visual" 或 "audio"
        
        # 宏微观双重超时管理
        self._user_has_spoken: bool = False  # 用户是否已开口（用于微观超时判断）
        
        # PROCESSING超时管理
        self._processing_start_time: float = 0.0
        self._processing_timeout_sec: float = self.config.runtime.processing_timeout_sec
        self._processing_timeout_thread: Optional[threading.Thread] = None
        self._processing_timeout_should_exit: bool = False
        
        # traceId管理（每次进入LISTENING时生成）
        self._current_trace_id: Optional[str] = None
        
        # WebSocket Client（统一网关）
        self._ws_client = None
        self._ws_connected = False
        self._ws_thread: Optional[threading.Thread] = None
        self._ws_loop: Optional[asyncio.AbstractEventLoop] = None
        
        # 初始化TTS相关sockets
        self._init_tts_sockets()
        
        # 初始化控制socket
        self._init_control_socket()
        
        # 初始化WebSocket Client
        self._init_websocket_client()
        
        # 启动音频播放线程
        self._start_playback_thread()
        
        # 启动ASR结果接收线程
        self._start_asr_result_receiver()
        
        logger.info("Core Server initialized")
        logger.info(f"  Vision SUB: tcp://127.0.0.1:{self.zmq_config.vision_pub_port}")
        logger.info(f"  Audio SUB: tcp://127.0.0.1:{self.zmq_config.audio_pub_port}")
        logger.info(f"  Audio REQ: tcp://127.0.0.1:{self.zmq_config.audio_ctrl_port}")
        logger.info(f"  ASR Result PULL: tcp://127.0.0.1:{self.zmq_config.asr_result_push_port}")
        logger.info(f"  TTS Text PUSH: tcp://127.0.0.1:{self.zmq_config.tts_text_pull_port}")
        logger.info(f"  TTS PULL: tcp://127.0.0.1:{self.zmq_config.tts_push_port}")
        logger.info(f"  TTS STOP PUB: tcp://127.0.0.1:{self.zmq_config.tts_stop_pub_port}")
        logger.info(f"  Control REP: tcp://127.0.0.1:{self.zmq_config.core_control_rep_port}")
        logger.info(f"  LLM Agent: {self.llm_agent_config.host} (deviceId: {self.llm_agent_config.device_id})")
        logger.info(f"  Initial timeout: {self.current_silence_timeout}s")
    
    def _send_threshold_command(self, threshold: float):
        """向音频模块发送阈值调整指令"""
        try:
            self.audio_req_socket.send_json({"command": "set_threshold", "value": threshold})
            reply = self.audio_req_socket.recv_json()
            if reply.get("status") == "ok":
                logger.info(f"✅ 音频阈值已更新: {threshold:.2f}")
                return True
            else:
                logger.warning(f"⚠️ 音频阈值更新失败: {reply}")
                return False
        except zmq.Again:
            logger.error("❌ 音频阈值更新超时")
            return False
        except Exception as e:
            logger.error(f"❌ 音频阈值更新异常: {e}")
            return False
    
    def _init_tts_sockets(self):
        """初始化TTS相关sockets"""
        # ZMQ PULL Socket（接收TTS音频）
        # 注意：使用 connect 而不是 bind，因为 TTS 模块已经在 5559 端口 bind
        self._tts_pull_socket = self.zmq_context.socket(zmq.PULL)
        self._tts_pull_socket.connect(f"tcp://127.0.0.1:{self.zmq_config.tts_push_port}")
        self.poller.register(self._tts_pull_socket, zmq.POLLIN)
        
        # ZMQ PUB Socket（发送停止信号给TTS）
        self._tts_stop_pub_socket = self.zmq_context.socket(zmq.PUB)
        self._tts_stop_pub_socket.bind(f"tcp://127.0.0.1:{self.zmq_config.tts_stop_pub_port}")
        
        # ZMQ PUSH Socket（发送文本给TTS）
        self._tts_text_push_socket = self.zmq_context.socket(zmq.PUSH)
        # 注意：使用 connect 因为 TTS 模块 bind 了这个端口
        self._tts_text_push_socket.connect(f"tcp://127.0.0.1:{self.zmq_config.tts_text_pull_port}")
    
    def _init_control_socket(self):
        """初始化控制socket（接收LLM指令）"""
        self._control_rep_socket = self.zmq_context.socket(zmq.REP)
        self._control_rep_socket.bind(f"tcp://127.0.0.1:{self.zmq_config.core_control_rep_port}")
        self._control_rep_socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms超时，非阻塞
        self.poller.register(self._control_rep_socket, zmq.POLLIN)
    
    def _init_websocket_client(self):
        """初始化WebSocket Client（统一网关）"""
        try:
            import websockets
            self._websockets_module = websockets
        except ImportError:
            logger.error("❌ websockets未安装，无法连接LLM Agent。请运行: pip install websockets")
            return
        
        # 启动WebSocket Client线程
        self._ws_thread = threading.Thread(target=self._websocket_client_worker, daemon=True)
        self._ws_thread.start()
        logger.info("🌐 WebSocket Client线程已启动")
    
    def _websocket_client_worker(self):
        """WebSocket Client工作线程（异步事件循环）"""
        import websockets
        import websockets.exceptions
        
        # 创建新的事件循环（在独立线程中）
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self._ws_loop = loop
        
        # 构建WebSocket URL
        protocol = "wss" if self.llm_agent_config.use_ssl else "ws"
        host = self.llm_agent_config.host
        device_id = self.llm_agent_config.device_id
        token = self.llm_agent_config.token
        url = f"{protocol}://{host}/api/voice/ws?deviceId={device_id}&token={token}"
        
        reconnect_interval = self.llm_agent_config.reconnect_interval_sec
        ping_interval = self.llm_agent_config.ping_interval_sec
        
        async def client_main():
            """WebSocket客户端主循环"""
            while True:
                try:
                    logger.info(f"🔌 正在连接LLM Agent: {url}")
                    async with websockets.connect(url) as websocket:
                        self._ws_client = websocket
                        self._ws_connected = True
                        logger.info("✅ WebSocket已连接")
                        
                        # 发送初始设备状态
                        self._report_device_state("idle")
                        
                        # 启动ping任务
                        ping_task = asyncio.create_task(self._ping_worker(websocket, ping_interval))
                        
                        try:
                            # 接收消息循环
                            async for message in websocket:
                                try:
                                    data = json.loads(message)
                                    await self._handle_websocket_message(data)
                                except json.JSONDecodeError:
                                    logger.warning(f"⚠️ 无效的JSON消息: {message}")
                                except Exception as e:
                                    logger.error(f"❌ 处理WebSocket消息失败: {e}")
                        except websockets.exceptions.ConnectionClosed:
                            logger.warning("⚠️ WebSocket连接已关闭")
                        finally:
                            ping_task.cancel()
                            self._ws_connected = False
                            self._ws_client = None
                
                except Exception as e:
                    logger.error(f"❌ WebSocket连接失败: {e}")
                    self._ws_connected = False
                    self._ws_client = None
                
                # 重连前等待
                logger.info(f"⏳ {reconnect_interval}秒后重连...")
                await asyncio.sleep(reconnect_interval)
        
        # 运行事件循环
        loop.run_until_complete(client_main())
    
    async def _ping_worker(self, websocket, interval: float):
        """Ping保活任务"""
        try:
            while True:
                await asyncio.sleep(interval)
                # 🌟 修复：直接尝试发送ping，如果连接已关闭会抛出异常
                try:
                    await websocket.send(json.dumps({"type": "ping"}))
                except websockets.exceptions.ConnectionClosed:
                    # 连接已关闭，退出循环
                    break
                except AttributeError:
                    # websocket对象可能已被销毁，退出循环
                    break
        except asyncio.CancelledError:
            # 任务被取消（正常情况，当连接关闭时）
            pass
        except Exception as e:
            # 其他异常，记录但不中断主循环
            logger.debug(f"Ping任务异常（已忽略）: {e}")
    
    async def _handle_websocket_message(self, data: dict):
        """处理从LLM Agent接收的WebSocket消息"""
        msg_type = data.get("type")
        
        if msg_type == "route":
            # TTS合成请求
            text = data.get("text", "")
            is_final = data.get("isFinal", False)
            
            # 🌟 新增：如果当前是PROCESSING状态，收到第一个TTS文本时上报thinking状态
            if self.current_state == SystemState.PROCESSING and text:
                self._report_device_state("thinking")
            
            if text or is_final:
                # 通过ZMQ发送给TTS模块
                self._send_tts_text(text, is_final)
        elif msg_type == "stop_tts":
            # 停止TTS合成
            self._send_stop_tts()
        elif msg_type == "pong":
            # Ping响应
            pass
        elif msg_type == "error":
            # 错误消息
            error_msg = data.get("message", "未知错误")
            logger.error(f"❌ LLM Agent错误: {error_msg}")
        elif msg_type == "warning":
            # 警告消息
            warn_msg = data.get("message", "未知警告")
            logger.warning(f"⚠️ LLM Agent警告: {warn_msg}")
    
    def _send_tts_text(self, text: str, is_final: bool = False):
        """通过ZMQ发送TTS合成文本给TTS模块"""
        try:
            message = {
                "type": "route",
                "text": text,
                "isFinal": is_final
            }
            message_json = json.dumps(message, ensure_ascii=False)
            self._tts_text_push_socket.send_string(message_json, zmq.NOBLOCK)
            # 🌟 修复：使用info级别以便调试
            if text:
                logger.info(f"📤 TTS文本已发送: {text[:50]}... (isFinal={is_final})")
            else:
                logger.info(f"📤 TTS空文本已发送 (isFinal={is_final})")
        except zmq.Again:
            logger.warning("⚠️ TTS文本队列已满，丢弃消息")
        except Exception as e:
            logger.error(f"❌ 发送TTS文本失败: {e}")
    
    def _send_stop_tts(self):
        """发送停止TTS合成信号"""
        try:
            message = {"type": "stop_tts"}
            message_json = json.dumps(message, ensure_ascii=False)
            self._tts_text_push_socket.send_string(message_json, zmq.NOBLOCK)
            logger.info("🛑 已发送停止TTS信号")
        except Exception as e:
            logger.error(f"❌ 发送停止TTS信号失败: {e}")
    
    def _start_asr_result_receiver(self):
        """启动ASR识别结果接收线程"""
        def receiver():
            logger.info("📥 ASR结果接收线程已启动")
            while True:
                try:
                    # 从ZMQ接收ASR识别结果
                    message = self._asr_result_pull_socket.recv_string(zmq.NOBLOCK)
                    try:
                        data = json.loads(message)
                        # 处理ASR结果（按照unified-voice-ws-protocol.md格式）
                        self._handle_asr_result(data)
                    except json.JSONDecodeError:
                        logger.warning(f"⚠️ 无效的JSON消息: {message}")
                    except Exception as e:
                        logger.error(f"❌ 处理ASR结果失败: {e}")
                except zmq.Again:
                    time.sleep(0.01)
                    continue
                except Exception as e:
                    logger.error(f"❌ ASR结果接收线程出错: {e}")
                    time.sleep(0.1)
        
        thread = threading.Thread(target=receiver, daemon=True)
        thread.start()
        logger.info("📥 ASR结果接收线程已启动")
    
    def _handle_asr_result(self, data: dict):
        """处理ASR识别结果，转发给LLM Agent"""
        msg_type = data.get("type")
        if msg_type != "asr":
            return
        
        stage = data.get("stage")  # "partial" 或 "final"
        text = data.get("text", "")
        trace_id = data.get("traceId")
        confidence = data.get("confidence", 0.0)
        timestamp = data.get("timestamp", time.time())
        
        # 更新当前traceId（如果ASR发送了新的traceId）
        if trace_id:
            self._current_trace_id = trace_id
        
        # 构建消息（按照unified-voice-ws-protocol.md格式）
        message = {
            "type": "asr",
            "stage": stage,
            "text": text,
            "traceId": self._current_trace_id or trace_id,
            "deviceId": self.llm_agent_config.device_id,
            "timestamp": timestamp,
            "confidence": confidence
        }
        
        # 通过WebSocket发送给LLM Agent
        self._send_websocket_message(message)
        
        # 如果是final结果，进入thinking状态
        if stage == "final":
            self._report_device_state("thinking")
    
    def _send_websocket_message(self, message: dict):
        """通过WebSocket发送消息给LLM Agent（线程安全）"""
        if not self._ws_connected or not self._ws_client or not self._ws_loop:
            logger.debug("⚠️ WebSocket未连接，跳过消息发送")
            return
        
        try:
            message_json = json.dumps(message, ensure_ascii=False)
            # 使用线程安全的方式发送（从ZMQ线程调用WebSocket线程的event loop）
            asyncio.run_coroutine_threadsafe(
                self._ws_client.send(message_json),
                self._ws_loop
            )
        except Exception as e:
            logger.error(f"❌ 发送WebSocket消息失败: {e}")
    
    def _report_device_state(self, state: str):
        """上报设备状态（idle/listening/thinking/speaking）"""
        message = {
            "type": "device_state",
            "state": state,
            "deviceId": self.llm_agent_config.device_id,
            "timestamp": time.time()
        }
        self._send_websocket_message(message)
        logger.debug(f"📊 设备状态已上报: {state}")
    
    def _send_interrupt(self, trace_id: Optional[str] = None, reason: str = "unknown"):
        """发送interrupt消息给LLM Agent"""
        message = {
            "type": "interrupt",
            "traceId": trace_id or self._current_trace_id,
            "deviceId": self.llm_agent_config.device_id,
            "reason": reason,
            "timestamp": time.time()
        }
        self._send_websocket_message(message)
        logger.info(f"🛑 已发送interrupt消息 (traceId={trace_id or self._current_trace_id}, reason={reason})")
    
    def _start_playback_thread(self):
        """启动音频播放线程"""
        self._playback_active = True
        self._current_playback_sample_rate = 24000  # 默认采样率
        self._playback_thread = threading.Thread(target=self._audio_playback_worker, daemon=True)
        self._playback_thread.start()
        logger.info("🎵 音频播放线程已启动")
    
    def _audio_playback_worker(self):
        """音频播放工作线程（通过XVF3800扬声器接口播放）"""
        try:
            import sounddevice as sd
            import numpy as np
        except ImportError:
            logger.error("❌ sounddevice未安装，无法播放音频。请运行: pip install sounddevice")
            return
        
        is_playing = False  # 当前是否正在播放音频块
        playback_started = False  # 是否已经开始播放（用于状态转换）
        
        # 尝试查找XVF3800输出设备（可选）
        output_device = None
        try:
            devices = sd.query_devices()
            for i, device in enumerate(devices):
                if 'xvf3800' in device['name'].lower() or 'xv3800' in device['name'].lower():
                    if device['max_output_channels'] > 0:
                        output_device = i
                        logger.info(f"🎵 找到XVF3800输出设备: {device['name']} (ID: {i})")
                        break
        except Exception as e:
            logger.debug(f"查找音频设备时出错（将使用默认设备）: {e}")
        
        while self._playback_active:
            try:
                # 尝试从队列获取音频块（带超时，以便检查停止标志）
                audio_data = self.audio_playback_queue.get(timeout=0.1)
                
                # 🌟 新增真正的结束标志校验：彻底抛弃依据队列空超时来猜算是否结束
                if audio_data == b"END_OF_TTS":
                    if self.current_state == SystemState.SPEAKING:
                        logger.info("🎵 收到了明确的 TTS 结束标签，并且音频全部播放完毕，安全进入 LISTENING 状态")
                        playback_started = False
                        self._transition_from_speaking()
                    continue
                
                # 转换为numpy数组（int16格式）
                audio_array = np.frombuffer(audio_data, dtype=np.int16)
                
                # 实际播放音频（阻塞，直到播放完成）
                is_playing = True
                playback_started = True
                
                # 🌟 使用从metadata中读取的采样率（如果已设置）
                sample_rate = getattr(self, '_current_playback_sample_rate', 24000)
                
                if output_device is not None:
                    sd.play(audio_array, samplerate=sample_rate, device=output_device)
                else:
                    sd.play(audio_array, samplerate=sample_rate)
                
                sd.wait()  # 等待播放完成（阻塞）
                is_playing = False
                
                sample_rate = getattr(self, '_current_playback_sample_rate', 24000)
                logger.debug(f"✅ 播放完成: {len(audio_array)} 采样点 (采样率: {sample_rate}Hz)")
                
                # 不再在此处基于队列为空推测结束！
                # 必须等队列消费到 b"END_OF_TTS" 标记才结束
                
            except queue.Empty:
                # 队列为空，只需要静静等待即可，TTS 合成中间的停顿不再导致状态早退跳回！
                continue
            except Exception as e:
                logger.error(f"❌ 音频播放异常: {e}")
                is_playing = False
                # 发生异常且不再播放音频时，只有在队列为空的情况下才结束，防止卡死
                if self.audio_playback_queue.empty() and self.current_state == SystemState.SPEAKING:
                    playback_started = False
                    self._transition_from_speaking()
    
    def clear_playback_queue(self):
        """清空播放队列（硬打断）"""
        count = 0
        while not self.audio_playback_queue.empty():
            try:
                self.audio_playback_queue.get_nowait()
                count += 1
            except queue.Empty:
                break
        if count > 0:
            logger.info(f"已清空播放队列: {count} 个音频块")
    
    def _handle_hard_cutoff(self):
        """硬打断处理：停止播报，清空队列，发送停止信号"""
        logger.warning("🚨 硬打断：停止播报，清空队列，发送停止信号")
        
        # 1. 发送interrupt消息给LLM Agent
        self._send_interrupt(reason="barge-in")
        
        # 2. 清空播报队列
        self.clear_playback_queue()
        
        # 3. 向TTS发送停止信号（ZMQ PUB）
        if self._tts_stop_pub_socket:
            try:
                self._tts_stop_pub_socket.send_string("STOP_SYNTHESIS", zmq.NOBLOCK)
                logger.info("✅ 已发送STOP_SYNTHESIS信号给TTS模块")
            except zmq.Again:
                logger.warning("⚠️ 发送停止信号失败（缓冲区满）")
            except Exception as e:
                logger.error(f"❌ 发送停止信号异常: {e}")
        
        # 3. 启动1.5秒硬件冷却期
        self._cooldown_until = time.time() + 1.5
        logger.info("✅ 硬打断完成，启动1.5秒冷却期")
    
    def _is_in_cooldown(self) -> bool:
        """检查是否在冷却期内"""
        return time.time() < self._cooldown_until
    
    def _send_reset_cooldown_command(self):
        """向音频模块发送重置冷却期指令（允许立即再次唤醒）"""
        try:
            self.audio_req_socket.send_json({"command": "reset_cooldown"})
            reply = self.audio_req_socket.recv_json()
            if reply.get("status") == "ok":
                logger.info("✅ 音频冷却期已重置")
                return True
            else:
                logger.warning(f"⚠️ 音频冷却期重置失败: {reply}")
                return False
        except zmq.Again:
            logger.error("❌ 音频冷却期重置超时")
            return False
        except Exception as e:
            logger.error(f"❌ 音频冷却期重置异常: {e}")
            return False
    
    def _send_start_streaming_command(self):
        """向音频模块发送强制推流指令（用于连续对话免唤醒）"""
        try:
            self.audio_req_socket.send_json({"command": "start_streaming"})
            reply = self.audio_req_socket.recv_json()
            if reply.get("status") == "ok":
                logger.info("✅ 已通知音频底层开启持续拾音")
                return True
            else:
                logger.warning(f"⚠️ 开启推流指令失败: {reply}")
                return False
        except zmq.Again:
            logger.error("❌ 开启推流指令超时")
            return False
        except Exception as e:
            logger.error(f"❌ 发送开启推流指令异常: {e}")
            return False
    
    def _send_stop_streaming_command(self):
        """向音频模块发送停止推流指令（进入PROCESSING状态时停止音频推流）"""
        try:
            self.audio_req_socket.send_json({"command": "stop_streaming"}, zmq.NOBLOCK)
            try:
                reply = self.audio_req_socket.recv_json(zmq.NOBLOCK)
                if reply.get("status") == "ok":
                    logger.info("✅ 已通知音频底层停止推流")
                    return True
                else:
                    logger.warning(f"⚠️ 音频推流停止失败: {reply}")
                    return False
            except zmq.Again:
                # 非阻塞接收，如果没有回复也不影响
                logger.debug("音频推流停止命令已发送（无回复）")
                return True
        except zmq.Again:
            logger.warning("⚠️ 音频推流停止命令发送失败（缓冲区满）")
            return False
        except Exception as e:
            logger.error(f"❌ 音频推流停止异常: {e}")
            return False
    
    def _transition_to_idle(self, use_abort: bool = False, skip_marker: bool = False):
        """转换到IDLE状态
        
        Args:
            use_abort: 如果为True，使用ABORT_SPEECH标记（用于微信等废弃场景）
            skip_marker: 如果为True，跳过发送标记（用于已经在外部发送标记的情况）
        """
        logger.info("状态转换: -> IDLE")
        prev_state = self.current_state
        
        # 停止PROCESSING超时检查器
        self._processing_timeout_should_exit = True
        
        # 🌟 修复：如果从LISTENING/PROCESSING状态转换过来，需要停止音频推流
        if self.current_state in [SystemState.LISTENING, SystemState.PROCESSING]:
            self._send_stop_streaming_command()
        
        # 🌟 修复：根据前一个状态决定发送END_OF_SPEECH还是ABORT_SPEECH
        # 只有LISTENING状态下才有音频在ASR中处理，需要发送END_OF_SPEECH进行最终识别
        # 其他状态（如VISUAL_WAKE）下没有音频在ASR中，应该使用ABORT_SPEECH清空cache
        if not skip_marker:
            if prev_state == SystemState.LISTENING:
                # LISTENING状态下，根据use_abort参数决定
                if use_abort:
                    self._send_abort_marker()
                else:
                    self._send_end_marker()
            else:
                # 非LISTENING状态下（如VISUAL_WAKE、PROCESSING、SPEAKING），使用ABORT清空cache
                # 避免ASR对残留cache进行识别，导致误识别出"嗯"等填充词
                self._send_abort_marker()
        
        # 发送恢复阈值指令
        self._send_threshold_command(self.audio_threshold_config.default)
        # 重置冷却期，允许立即再次唤醒
        self._send_reset_cooldown_command()
        # 重置对话轮次和唤醒路径
        self._conversation_round = 0
        self._wake_path = "unknown"
        # 清空traceId
        self._current_trace_id = None
        self.current_state = SystemState.IDLE
        
        # 上报设备状态
        self._report_device_state("idle")
    
    def _transition_to_visual_wake(self, use_abort: bool = False):
        """转换到VISUAL_WAKE状态"""
        logger.info("状态转换: -> VISUAL_WAKE")
        if use_abort:
            self._send_abort_marker()
        else:
            self._send_end_marker()
        # 不恢复阈值（保持0.4）
        self.current_state = SystemState.VISUAL_WAKE
    
    def _transition_to_listening(self, wake_path: str = "unknown"):
        """转换到LISTENING状态"""
        prev_state = self.current_state
        logger.info(f"状态转换: {prev_state} -> LISTENING (路径: {wake_path}, 轮次: {self._conversation_round})")
        
        # 如果从PROCESSING/SPEAKING状态转换过来，触发硬打断
        if self.current_state in [SystemState.PROCESSING, SystemState.SPEAKING]:
            self._handle_hard_cutoff()
        
        # 🌟 生成新的traceId（每次进入LISTENING时生成）
        self._current_trace_id = str(uuid.uuid4())
        logger.info(f"🆕 新对话轮次开始，生成traceId: {self._current_trace_id}")
        
        # 记录唤醒路径（仅在首次唤醒时记录）
        if self._conversation_round == 0:
            # 将中文描述转换为标准路径标识
            if "视觉" in wake_path or "visual" in wake_path.lower():
                self._wake_path = "visual"
            elif "纯语音" in wake_path or "audio" in wake_path.lower():
                self._wake_path = "audio"
            else:
                # 根据当前状态推断路径
                if self.current_state == SystemState.VISUAL_WAKE:
                    self._wake_path = "visual"
                else:
                    self._wake_path = "audio"
            logger.info(f"📝 记录唤醒路径: {self._wake_path} (描述: {wake_path}, 轮次: {self._conversation_round})")
        else:
            # 🌟 修复：非首次唤醒时，也输出当前状态信息
            logger.info(f"📝 持续对话模式 (轮次: {self._conversation_round}, 路径: {self._wake_path})")
        
        # 初始化VAD超时管理
        self._user_has_spoken = False
        self._last_vad_time = time.time()
        self._last_lip_active_time = time.time()  # 重置唇动计时器
        self.current_silence_timeout = self.conversation_config.vad_silence_timeout_default_sec
        self._vad_timeout_should_exit = False
        
        # 🌟 保底机制：记录进入LISTENING状态的时间
        self._listening_start_time = time.time()
        logger.info(f"⏱️ [保底机制] 开始计时，30秒交流上限时间已启动")
        
        # 🌟 修复 Bug #2: ASR 丢失唤醒词后的前几个字
        # 我们现在仅仅发送 START_OF_SPEECH 解除拒收状态，ASR_service 已经修改为不再在这个指令下清空文字盆和特征缓存
        if self._asr_push_socket:
            try:
                self._asr_push_socket.send(b"START_OF_SPEECH", zmq.NOBLOCK)
            except:
                pass
        
        # 如果是进入免唤醒持续对话，必须主动通知音频底层拉起流模式
        if wake_path == "持续对话":
            self._send_start_streaming_command()
        
        # 启动VAD超时检查线程
        if self._vad_timeout_thread is None or not self._vad_timeout_thread.is_alive():
            self._vad_timeout_thread = threading.Thread(target=self._vad_timeout_checker, daemon=True)
            self._vad_timeout_thread.start()
        
        self.current_state = SystemState.LISTENING
        
        # 上报设备状态
        self._report_device_state("listening")
    
    def _send_end_marker(self):
        """发送结束标记给ASR"""
        if self._asr_push_socket:
            try:
                self._asr_push_socket.send(b"END_OF_SPEECH", zmq.NOBLOCK)
                logger.info("✅ 已发送END_OF_SPEECH标记给ASR")
            except zmq.Again:
                logger.warning("⚠️ 发送END_OF_SPEECH失败（缓冲区满）")
            except Exception as e:
                logger.error(f"❌ 发送END_OF_SPEECH异常: {e}")
    
    def _send_abort_marker(self):
        """发送强制中止标记给ASR（用于游客离开等废弃场景）"""
        if self._asr_push_socket:
            try:
                self._asr_push_socket.send(b"ABORT_SPEECH", zmq.NOBLOCK)
            except Exception:
                pass
                
        # 🌟 修复：发送高速带外广播信号，让 ASR 瞬间清空积压队列！
        if self._tts_stop_pub_socket:
            try:
                self._tts_stop_pub_socket.send_string("ABORT_ASR", zmq.NOBLOCK)
                logger.info("🛑 已发送 ABORT_ASR 高速带外强杀信号")
            except Exception:
                pass
    
    def _vad_timeout_checker(self):
        """VAD超时检查线程（宏微观双重监控机制）"""
        logger.info("🔍 [VAD检查线程] 已启动，开始监控...")
        check_count = 0
        while self.current_state == SystemState.LISTENING:
            if self._vad_timeout_should_exit:
                logger.info("🔍 [VAD检查线程] 收到退出信号")
                break
            
            time.sleep(self.conversation_config.vad_check_interval_ms / 1000.0)
            check_count += 1
            
            time_since_last_vad = time.time() - self._last_vad_time
            _last_lip_time = getattr(self, '_last_lip_active_time', None)
            if _last_lip_time is None:
                _last_lip_time = time.time()
                self._last_lip_active_time = _last_lip_time
            time_since_lip_closed = time.time() - _last_lip_time
            
            # 🌟 保底机制：30秒交流上限时间（最高优先级检查，防止数据无限制传输）
            time_since_listening_start = time.time() - self._listening_start_time
            if time_since_listening_start >= self._max_listening_duration:
                logger.warning(f"🛑 [保底机制] 30秒交流上限时间已到，强制截断！(已监听{time_since_listening_start:.2f}s)")
                logger.warning(f"   说明：在视觉受损或极度噪音环境下，其他截断机制可能失效，此机制确保数据不会无限制传输")
                self._send_end_marker()
                self._send_stop_streaming_command()
                logger.info("状态转换: LISTENING -> PROCESSING (30秒保底截断)")
                self._transition_to_processing()
                break
            
            if self._user_has_spoken:
                # 🌟 调试信息：打印当前状态（每5次检查打印一次，避免刷屏）
                if check_count % 5 == 0:  # 每5次检查（约1秒）打印一次
                    logger.info(
                        f"🔍 [VAD检查 #{check_count}] time_since_last_vad={time_since_last_vad:.2f}s, "
                        f"time_since_lip_closed={time_since_lip_closed:.2f}s, "
                        f"is_talking={self._latest_vision_is_talking}, "
                        f"time_since_listening_start={time_since_listening_start:.2f}s, "
                        f"_last_lip_active_time={_last_lip_time:.3f}, "
                        f"current_time={time.time():.3f}"
                    )
                
                # 🔪 策略 1：双模态快刀 (音频安静1.5s + 嘴巴闭上1.5s)
                # 🌟 修复：从1.0秒提高到1.5s，避免用户说话时嘴巴稍微闭合就误触发截断
                if time_since_last_vad > 1.5 and time_since_lip_closed > 1.5:
                    logger.info(f"👁️+👂 视觉闭嘴+音频短静音，触发 1.5s 双模态截断！(vad={time_since_last_vad:.2f}s, lip={time_since_lip_closed:.2f}s)")
                    self._send_end_marker()
                    self._send_stop_streaming_command()  # 🌟 修复：通知音频服务停止推流
                    logger.info("状态转换: LISTENING -> PROCESSING (VAD截断)")
                    self._transition_to_processing()
                    break
                
                # 🔪 策略 2：视觉强杀！(环境太吵VAD失效，但嘴巴已经死死闭上 2.0s，并且音频VAD至少也安静了0.5s，强行切断)
                # 🌟 修复：增加音频保底条件，防止纯视觉误判（比如发呆但背景有说话）
                elif time_since_lip_closed > 2.0 and time_since_last_vad > 0.5:
                    logger.info(f"👁️ 视觉强杀！无视环境噪音，判定用户已闭嘴结束说话。(lip_closed={time_since_lip_closed:.2f}s, vad={time_since_last_vad:.2f}s)")
                    self._send_end_marker()
                    self._send_stop_streaming_command()  # 🌟 修复：通知音频服务停止推流
                    logger.info("状态转换: LISTENING -> PROCESSING (视觉强杀)")
                    self._transition_to_processing()
                    break
                
                # 🔪 策略 3：纯音频兜底 (人没在看屏幕时，或者视觉检测不到嘴巴时，只要音频安静 2.5s 就切断)
                elif time_since_last_vad > 2.5:
                    logger.info(f"🔪 VAD音频硬兜底触发（2.5s尾音静音）(vad={time_since_last_vad:.2f}s, lip={time_since_lip_closed:.2f}s)")
                    self._send_end_marker()
                    self._send_stop_streaming_command()  # 🌟 修复：通知音频服务停止推流
                    logger.info("状态转换: LISTENING -> PROCESSING (音频兜底)")
                    self._transition_to_processing()
                    break
            else:
                # ⏳ 宏观容器：发呆超时直接关门
                if time_since_last_vad > self.current_silence_timeout:
                    logger.info(f"⏳ 宏观发呆超时（{self.current_silence_timeout}s无语音），退回IDLE")
                    self._handle_vad_timeout()
                    break
    
    def _handle_vad_timeout(self):
        """处理VAD超时"""
        # 🌟 修复：如果用户根本没开口，直接发 ABORT 强杀，防止 ASR 幻觉出"嗯"
        use_abort = not self._user_has_spoken
        if use_abort:
            logger.info("👻 检测到幽灵唤醒（用户未开口），强制丢弃音频！")
            
        if self._latest_vision_wake:
            self._transition_to_visual_wake(use_abort=use_abort)
        else:
            self._transition_to_idle(use_abort=use_abort)
    
    def _transition_to_processing(self):
        """转换到PROCESSING状态，并启动看门狗"""
        logger.info("状态转换: -> PROCESSING")
        self.current_state = SystemState.PROCESSING
        self._processing_start_time = time.time()
        self._processing_timeout_should_exit = False
        
        # 上报设备状态
        self._report_device_state("thinking")
        
        if self._processing_timeout_thread is None or not self._processing_timeout_thread.is_alive():
            self._processing_timeout_thread = threading.Thread(target=self._processing_timeout_checker, daemon=True)
            self._processing_timeout_thread.start()
            
    def _processing_timeout_checker(self):
        """PROCESSING超时守卫线程：卡死在思考状态时，强制唤醒退回监听/空闲"""
        logger.info(f"⏱️ [PROCESSING守卫] 启动，超时阈值={self._processing_timeout_sec}秒")
        while self.current_state == SystemState.PROCESSING:
            if self._processing_timeout_should_exit:
                break
            
            elapsed = time.time() - self._processing_start_time
            if elapsed > self._processing_timeout_sec:
                logger.error(f"🚨 [处理超时] 核心推理/TTS生成卡住超过 {self._processing_timeout_sec} 秒！")
                logger.error("🛑 触发系统自救协议 -> 强制清退并回到空闲状态")
                self._handle_hard_cutoff()
                self._transition_to_idle(use_abort=True)
                break
                
            time.sleep(1.0)
    
    def _transition_to_speaking(self):
        """转换到SPEAKING状态（TTS开始播放）"""
        logger.info("状态转换: -> SPEAKING")
        self.current_state = SystemState.SPEAKING
        # 🎙️ 关键：播报时让耳朵重置，回去抓取唤醒词（准备随时硬打断）
        self._send_reset_cooldown_command()
        # conversation_round 保持不变（首次=0，持续对话>=1）
        
        # 上报设备状态
        self._report_device_state("speaking")
    
    def _transition_from_speaking(self):
        """从SPEAKING状态退出（TTS播放结束）"""
        logger.info("状态转换: SPEAKING -> LISTENING (TTS结束)")
        
        # 如果这是首次回答结束，标记进入持续对话模式
        if self._conversation_round == 0:
            self._conversation_round = 1
            logger.info("✅ 首次对话完成，进入持续对话模式")
        else:
            # 持续对话模式，递增轮次
            self._conversation_round += 1
            logger.info(f"📊 对话轮次递增: {self._conversation_round}")
        
        # 🌟 核心修复：开启 0.8 秒的绝对物理防抖时间！
        # 屏蔽这段时间内的任何 VAD 波动和硬打断，防止自己录到自己的尾音
        self._ignore_audio_until = time.time() + 0.8
        logger.info(f"🛡️ 激活物理防抖护盾：0.8秒内忽略所有音频输入（直到 {self._ignore_audio_until:.2f}）")
        
        # 转换到 LISTENING 状态（等待用户继续说话）
        self._transition_to_listening("持续对话")
    
    def _handle_visual_cutoff(self, context: str = "监听"):
        """处理视觉斩断（最高优先级打断）"""
        logger.warning(f"🚨 视觉斩断：检测到用户离开，立即切断{context}")
        
        # 发送interrupt消息给LLM Agent（如果当前有traceId）
        if self._current_trace_id:
            self._send_interrupt(reason="visual-cutoff")
        
        # 发送中止标记（废弃场景，不进行ASR结算）
        self._send_abort_marker()
        # 停止VAD超时检查线程（如果正在运行）
        if self._vad_timeout_thread and self._vad_timeout_thread.is_alive():
            # 标记线程应该退出（通过共享标志）
            self._vad_timeout_should_exit = True
        # 状态瞬间回落到IDLE（跳过标记发送，因为已经在上面发送了）
        self._transition_to_idle(skip_marker=True)
    
    def _handle_tts_stop(self):
        """处理TTS停止（停止数字人播报）"""
        # TODO: 当TTS模块实现后，通过ZMQ发送 b"STOP_SYNTHESIS" 信号
        # 这里先预留接口，实际实现时需要通过ZMQ PUSH发送到TTS模块
        logger.warning("🚨 视觉斩断：停止TTS播报")
        # 预留：self.tts_push_socket.send(b"STOP_SYNTHESIS", zmq.NOBLOCK)
        # 状态回落到IDLE
        self._transition_to_idle()
    
    def set_silence_timeout(self, timeout_sec: float):
        """
        外部接口：动态调整VAD静音超时时间
        
        Args:
            timeout_sec: 新的超时时间（秒）
        """
        if timeout_sec < 1.0 or timeout_sec > 60.0:
            logger.warning(f"超时值 {timeout_sec} 超出合理范围 [1.0, 60.0]，已限制")
            timeout_sec = max(1.0, min(60.0, timeout_sec))
        self.current_silence_timeout = timeout_sec
        logger.info(f"✅ VAD静音超时已动态调整为: {timeout_sec}秒")
    
    def run(self):
        """运行核心服务器主循环"""
        logger.info("🚀 Core Server started")
        
        try:
            while True:
                # 使用Poller同时监听视觉和音频数据
                socks = dict(self.poller.poll(timeout=100))  # 100ms超时
                
                # 处理视觉数据
                if self.vision_sub_socket in socks:
                    try:
                        vision_data = self.vision_sub_socket.recv_json(zmq.NOBLOCK)
                        # 🌟 已实现唇动检测功能，不再输出详细日志（避免刷屏）
                        self._process_vision_data(vision_data)
                    except zmq.Again:
                        pass
                    except Exception as e:
                        logger.error(f"处理视觉数据异常: {e}")
                
                # 处理音频数据
                if self.audio_sub_socket in socks:
                    try:
                        # 接收Multipart Message
                        metadata_json, audio_binary = self.audio_sub_socket.recv_multipart(zmq.NOBLOCK)
                        metadata = json.loads(metadata_json.decode('utf-8'))
                        self._process_audio_data(metadata, audio_binary)
                    except zmq.Again:
                        pass
                    except Exception as e:
                        logger.error(f"处理音频数据异常: {e}")
                
                # 处理TTS音频数据
                if self._tts_pull_socket and self._tts_pull_socket in socks:
                    try:
                        # 接收Multipart Message
                        metadata_json, audio_binary = self._tts_pull_socket.recv_multipart(zmq.NOBLOCK)
                        metadata = json.loads(metadata_json.decode('utf-8'))
                        
                        if metadata.get("type") == "tts_end":
                            self.audio_playback_queue.put_nowait(b"END_OF_TTS")
                        else:
                            self._process_tts_audio(metadata, audio_binary)
                    except zmq.Again:
                        pass
                    except Exception as e:
                        logger.error(f"处理TTS音频数据异常: {e}")
                
                # 处理控制指令（LLM下行控制）
                if self._control_rep_socket and self._control_rep_socket in socks:
                    try:
                        request = self._control_rep_socket.recv_json(zmq.NOBLOCK)
                        response = self._handle_control_command(request)
                        self._control_rep_socket.send_json(response)
                    except zmq.Again:
                        pass
                    except Exception as e:
                        logger.error(f"处理控制指令异常: {e}")
                        try:
                            self._control_rep_socket.send_json({"status": "error", "message": str(e)})
                        except:
                            pass
                
                # 检查视觉数据丢失
                if time.time() - self._last_vision_timestamp > 3.0:
                    if self._latest_vision_wake:
                        logger.warning("⚠️ 视觉数据丢失（>3秒），视为wake=false")
                        self._latest_vision_wake = False
                
                # 检查冷却期（忽略VAD和KWS事件）
                if self._is_in_cooldown():
                    continue
                
        except KeyboardInterrupt:
            logger.info("🛑 Core Server stopped by user")
        finally:
            self._cleanup()
    
    def _process_vision_data(self, vision_data: dict):
        """处理视觉数据"""
        vision_wake = vision_data.get("wake", False)
        is_talking = vision_data.get("is_talking", False)
        
        # 更新视觉状态缓存
        self._latest_vision_wake = vision_wake
        prev_is_talking = self._latest_vision_is_talking
        self._latest_vision_is_talking = is_talking
        self._last_vision_timestamp = time.time()
        
        # 🌟 已实现唇动检测功能，不再输出详细日志（避免刷屏）
        # 只在必要时使用debug级别日志
        
        # 🌟 修复：一旦检测到嘴巴动，刷新计时器
        # 关键：无论 is_talking 是 True 还是 False，只要状态发生变化，都要更新时间戳
        if is_talking:
            self._last_lip_active_time = time.time()  # 只要嘴巴动，就刷新秒表
        elif prev_is_talking and not is_talking:
            # 🌟 关键修复：从 talking 变为 silent 时，也要更新时间戳
            # 这样 time_since_lip_closed 才能正确计算从闭嘴开始的时间
            self._last_lip_active_time = time.time()
        
        # ========== LISTENING 状态下的视觉斩断规则（关麦） ==========
        if (self.current_state == SystemState.LISTENING and 
            not vision_wake and 
            self.vision_wake_config.visual_cutoff_enabled):
            # 排除纯语音唤醒的首次对话，其他情况一律斩断（关麦）
            is_pure_audio_first_turn = (self._conversation_round == 0 and self._wake_path == "audio")
            if not is_pure_audio_first_turn:
                # 场景B：首次唤醒且路径为visual，或场景C：持续对话期（round >= 1）
                self._handle_visual_cutoff("监听")
                return  # 跳过后续处理，因为状态已改变
            else:
                # 场景A：纯语音唤醒的首次对话，不触发视觉斩断（修复Bug）
                logger.debug(f"🛡️ 纯语音唤醒首次对话保护：轮次={self._conversation_round}, 路径={self._wake_path}")
        
        # ========== SPEAKING 状态下的视觉斩断规则（停止TTS） ==========
        if (self.current_state == SystemState.SPEAKING and 
            not vision_wake and 
            self.vision_wake_config.visual_cutoff_enabled):
            # 场景E：持续对话期（round >= 1），用户离开立即停止TTS
            if self._conversation_round >= 1:
                self._handle_tts_stop()
                return  # 跳过后续处理，因为状态已改变
            else:
                # 场景D：首次回答（round == 0），不触发视觉斩断，允许完成首次回答
                logger.debug(f"🛡️ 首次回答保护：轮次={self._conversation_round}")
        
        # ========== LISTENING 状态下，视觉检测到用户说话时也标记为已开口 ==========
        # 🌟 修复：即使音频VAD因RMS阈值过高未检测到声音，视觉检测到说话也应标记为已开口
        # 这样视觉强杀逻辑才能正常触发，避免被"聋子麦克风"绑架
        if (self.current_state == SystemState.LISTENING and 
            is_talking and 
            not self._user_has_spoken):
            # 视觉检测到用户说话，即使音频VAD没检测到，也标记为已开口
            self._user_has_spoken = True
            logger.info("👄 [视觉数据] 检测到用户说话（视觉确认），标记为已开口（即使音频VAD未检测到）")
        
        # ========== VISUAL_WAKE 状态下，如果检测到用户开始说话，不自动越权！ ==========
        # 🌟 修复 Bug #1: 视频唤醒只是降低阈值，还是需要音频唤醒！
        # 删除了原来只要 is_talking 就自动跳入 LISTENING 的错误逻辑，
        # 用户即使嘴巴动了，也需要说出唤醒词（降低到 0.6 的阈值）才能进入真实监听。
        if (self.current_state == SystemState.VISUAL_WAKE and 
            is_talking):
            # 仅仅记录日志，不再越权跳转状态
            logger.debug("👄 [VISUAL_WAKE] 检测到用户开始说话，等待音频达到较低阈值(0.6)触发正式唤醒...")
        
        # ========== VISUAL_WAKE 状态下，如果视觉唤醒结束，恢复到 IDLE ==========
        if (self.current_state == SystemState.VISUAL_WAKE and 
            not vision_wake):
            logger.info("👁️ 视觉唤醒结束，恢复到IDLE状态")
            self._transition_to_idle()
            return  # 跳过后续处理，因为状态已改变
        
        # ========== IDLE 状态下，收到视觉唤醒，进入 VISUAL_WAKE（视觉降维打击） ==========
        if (self.current_state == SystemState.IDLE and 
            vision_wake):
            # 收到视觉唤醒，发送降阈指令
            logger.info("👁️ 视觉唤醒检测，发送降阈指令")
            self._send_threshold_command(self.audio_threshold_config.visual_wake)
            self.current_state = SystemState.VISUAL_WAKE
    
    def _process_audio_data(self, metadata: dict, audio_binary: bytes):
        """处理音频数据"""
        current_time = time.time()
        
        # 🛡️ 护盾拦截：如果在无敌时间内，直接丢弃 VAD 和 打断 信号！
        if current_time < getattr(self, '_ignore_audio_until', 0.0):
            # 忽略刚开麦时的物理噪音
            logger.debug(f"🛡️ 物理防抖护盾生效中，忽略音频输入（剩余 {self._ignore_audio_until - current_time:.2f}秒）")
            return
        
        # 🌟 修复声学反馈问题：在SPEAKING状态下，忽略所有音频输入（除了唤醒词）
        # 防止TTS播放的音频被麦克风拾取后误识别为ASR输入
        if self.current_state == SystemState.SPEAKING:
            # 在SPEAKING状态下，只处理唤醒词（用于硬打断），不处理VAD，不转发音频到ASR
            wake_word = metadata.get("wake_word", {})
            if wake_word.get("detected", False):
                confidence = wake_word.get("confidence", 0.0)
                # 🌟 修复：根据唤醒路径选择阈值
                # 视觉唤醒路径使用0.4，纯语音唤醒路径使用0.95
                if self._wake_path == "visual":
                    threshold = self.audio_threshold_config.visual_wake
                else:
                    threshold = self.audio_threshold_config.default
                
                if confidence >= threshold:
                    logger.info(f"🛑 听到唤醒词（置信度={confidence:.2%} >= 阈值{threshold}），触发硬打断！")
                    self._transition_to_listening("唤醒词打断")
            # 其他情况直接返回，不处理VAD，不转发音频
            return
        
        vad = metadata.get("vad", False)
        wake_word = metadata.get("wake_word", {})
        
        # 🌟 修复状态机漏风：只有在真正倾听时，才允许修改 _user_has_spoken！
        # 防止 VISUAL_WAKE 状态下偷偷记录开口事件
        if self.current_state == SystemState.LISTENING:
            # 更新VAD时间戳（展厅防抖机制）
            if vad:
                self._vad_speech_count += 1
                self._vad_silence_count = 0
                
                # 必须连续2次（0.4秒）检测到人声，才认定为真语音，避免瞬间噪音（如咳嗽、碰撞）打断1.5秒计时
                if self._vad_speech_count >= 2:
                    self._last_vad_time = current_time
                    if not self._user_has_spoken:
                        self._user_has_spoken = True
                        logger.info("🗣️ 检测到用户真实开口，切换为1.5秒微观截断模式")
            else:
                self._vad_silence_count += 1
                self._vad_speech_count = 0
                # 真正的静音不需要操作，让 time_since_last_vad 自然累加即可触发超时
        else:
            # 🌟 修复：非 LISTENING 状态下，不更新 VAD 状态变量，防止状态泄漏
            # 但需要重置计数器，避免状态残留
            if not vad:
                self._vad_silence_count += 1
                self._vad_speech_count = 0
        
        # 处理唤醒词检测
        if wake_word.get("detected", False):
            confidence = wake_word.get("confidence", 0.0)
            keyword = wake_word.get("keyword", "unknown")
            
            # 🌟 修复"闹鬼"问题：记录所有收到的唤醒词事件（用于调试）
            logger.info(f"🔔 [CoreServer] 收到唤醒词事件: 关键词={keyword}, 置信度={confidence:.2%}, 当前状态={self.current_state}")
            
            # 路径A：视觉降维打击（在VISUAL_WAKE状态下）
            if (self.current_state == SystemState.VISUAL_WAKE and 
                confidence >= self.audio_threshold_config.visual_wake):
                logger.info(f"✅ [路径A] 视觉降维打击：置信度{confidence:.2%} >= 阈值{self.audio_threshold_config.visual_wake}")
                self._transition_to_listening("视觉降维打击")
            
            # 路径B：纯语音唤醒（在IDLE状态下，突破高阈值）
            elif (self.current_state == SystemState.IDLE and 
                  confidence >= self.audio_threshold_config.default):
                logger.info(f"✅ [路径B] 纯语音唤醒：置信度{confidence:.2%} >= 阈值{self.audio_threshold_config.default}")
                self._transition_to_listening("纯语音唤醒")
            
            # 路径C：唤醒词硬打断（在SPEAKING状态下）
            # 🌟 修复连环车祸：硬打断【只有】在数字人正在说话（SPEAKING）时才允许触发！
            # 绝对不允许在 LISTENING 或 IDLE 时触发打断导致清空队列！
            elif self.current_state == SystemState.SPEAKING:
                if self._wake_path == "visual":
                    threshold = self.audio_threshold_config.visual_wake
                else:
                    threshold = self.audio_threshold_config.default
                
                if confidence >= threshold:
                    logger.info(f"✅ [路径C] 唤醒词硬打断：置信度{confidence:.2%} >= 阈值{threshold}")
                    logger.info("🛑 听到唤醒词，触发硬打断！")
                    self._handle_hard_cutoff()
                    self._transition_to_listening("唤醒词打断")
            
            # 🌟 修复：路径D：在PROCESSING状态下检测到唤醒词，立即进入LISTENING（持续对话）
            elif (self.current_state == SystemState.PROCESSING and 
                  confidence >= self.audio_threshold_config.visual_wake):
                logger.info(f"✅ [路径D] 持续对话：置信度{confidence:.2%} >= 阈值{self.audio_threshold_config.visual_wake}")
                logger.info(f"🔄 [PROCESSING] 检测到唤醒词（置信度={confidence:.2%}），进入持续对话")
                self._transition_to_listening("持续对话")
            
            else:
                # 🌟 修复"闹鬼"问题：记录为什么唤醒词没有被处理
                if self.current_state == SystemState.SPEAKING:
                    # 🌟 修复：根据唤醒路径显示正确的阈值
                    if self._wake_path == "visual":
                        threshold = self.audio_threshold_config.visual_wake
                    else:
                        threshold = self.audio_threshold_config.default
                    logger.warning(f"⚠️ [路径C被跳过] SPEAKING状态，但置信度{confidence:.2%} < 阈值{threshold}，不触发硬打断")
                elif self.current_state == SystemState.PROCESSING:
                    logger.warning(f"⚠️ [路径D被跳过] PROCESSING状态，但置信度{confidence:.2%} < 阈值{self.audio_threshold_config.visual_wake}，不触发持续对话")
                elif self.current_state == SystemState.VISUAL_WAKE:
                    logger.warning(f"⚠️ [路径A被跳过] VISUAL_WAKE状态，但置信度{confidence:.2%} < 阈值{self.audio_threshold_config.visual_wake}")
                elif self.current_state == SystemState.IDLE:
                    logger.warning(f"⚠️ [路径B被跳过] IDLE状态，但置信度{confidence:.2%} < 阈值{self.audio_threshold_config.default}")
                else:
                    logger.warning(f"⚠️ [唤醒词被忽略] 当前状态={self.current_state}，不匹配任何唤醒路径")
        
        # 🌟 修复: 无论 vad 是 True 还是 False，都必须必须将音频流持续转发给 ASR
        # FunASR-online 是基于连续时间线的模型，它在静音期间也需要这些静音数据来结算拼音和断句！
        # 如果因为 VAD 为 False（比如用户声音较轻，或句间停顿）就停止输送音频，ASR 会彻底卡死丢失方向！
        if self.current_state == SystemState.LISTENING:
            if self._asr_push_socket:
                try:
                    # 直接转发完整的连续二进制音频数据
                    self._asr_push_socket.send(audio_binary, zmq.NOBLOCK)
                except zmq.Again:
                    # 发送缓冲区满，丢弃此帧
                    pass
                except Exception as e:
                    logger.error(f"转发音频到ASR异常: {e}")
    
    def _process_tts_audio(self, metadata: dict, audio_binary: bytes):
        """处理TTS音频数据"""
        # 将音频数据加入播放队列
        try:
            self.audio_playback_queue.put_nowait(audio_binary)
            
            # 🌟 从metadata中读取采样率（如果提供）
            sample_rate = metadata.get('sample_rate', 24000)
            if hasattr(self, '_current_playback_sample_rate'):
                if self._current_playback_sample_rate != sample_rate:
                    logger.info(f"🎵 采样率变化: {self._current_playback_sample_rate}Hz → {sample_rate}Hz")
            else:
                logger.info(f"🎵 设置播放采样率: {sample_rate}Hz")
            self._current_playback_sample_rate = sample_rate
            
            # 如果当前不在SPEAKING状态，转换到SPEAKING状态
            if self.current_state != SystemState.SPEAKING:
                self._transition_to_speaking()
        except queue.Full:
            logger.warning("⚠️ 播放队列已满，丢弃音频块")
        except Exception as e:
            logger.error(f"处理TTS音频异常: {e}")
    
    def _handle_control_command(self, request: dict) -> dict:
        """
        处理控制指令（LLM下行控制）
        
        Args:
            request: 控制请求（JSON格式）
        
        Returns:
            dict: 响应（JSON格式）
        """
        command = request.get("command")
        
        if command == "extend_window":
            # 延长免唤醒窗口
            timeout_sec = request.get("value", 15.0)
            self.set_silence_timeout(timeout_sec)
            return {"status": "ok", "new_timeout": timeout_sec}
        
        elif command == "play_video":
            # 播放视频（未来扩展）
            video_id = request.get("video_id", "")
            logger.info(f"收到播放视频指令: {video_id}")
            return {"status": "ok", "message": "视频播放功能待实现"}
        
        elif command == "set_parameter":
            # 设置其他系统参数（未来扩展）
            param_name = request.get("param_name", "")
            param_value = request.get("param_value", "")
            logger.info(f"收到设置参数指令: {param_name} = {param_value}")
            return {"status": "ok", "message": "参数设置功能待实现"}
        
        else:
            return {"status": "error", "message": f"未知命令: {command}"}
    
    def _cleanup(self):
        """清理资源"""
        logger.info("正在清理资源...")
        
        # 停止音频播放线程
        self._playback_active = False
        if self._playback_thread and self._playback_thread.is_alive():
            self._playback_thread.join(timeout=0.5)
            if self._playback_thread.is_alive():
                logger.warning("⚠️ 音频播放线程未在超时内退出，强制继续")
        
        # 停止VAD超时线程
        self._vad_timeout_should_exit = True
        if self._vad_timeout_thread and self._vad_timeout_thread.is_alive():
            self._vad_timeout_thread.join(timeout=0.5)
            if self._vad_timeout_thread.is_alive():
                logger.warning("⚠️ VAD超时线程未在超时内退出，强制继续")
        
        # 关闭sockets（设置LINGER=0避免阻塞）
        # 注意：必须先设置LINGER，再关闭socket，最后关闭context
        try:
            # 设置所有socket的LINGER为0，立即关闭，不等待
            sockets_to_close = []
            if self._asr_push_socket:
                sockets_to_close.append(self._asr_push_socket)
            if self._tts_pull_socket:
                sockets_to_close.append(self._tts_pull_socket)
            if hasattr(self, '_tts_text_push_socket') and self._tts_text_push_socket:
                sockets_to_close.append(self._tts_text_push_socket)
            if self._tts_stop_pub_socket:
                sockets_to_close.append(self._tts_stop_pub_socket)
            if self._control_rep_socket:
                sockets_to_close.append(self._control_rep_socket)
            sockets_to_close.extend([
                self.vision_sub_socket,
                self.audio_sub_socket,
                self.audio_req_socket
            ])
            
            # 先设置LINGER，再关闭
            for sock in sockets_to_close:
                try:
                    sock.setsockopt(zmq.LINGER, 0)  # 立即关闭，不等待
                except:
                    pass
            
            # 然后关闭所有socket
            for sock in sockets_to_close:
                try:
                    sock.close()
                except:
                    pass
            
            # 最后关闭context
            try:
                self.zmq_context.term()
            except:
                pass
                
        except Exception as e:
            logger.error(f"清理资源异常: {e}")
        
        logger.info("✅ 资源清理完成")


def main():
    """主函数"""
    import argparse
    from pathlib import Path
    
    parser = argparse.ArgumentParser(description="Core Server - 核心决策模块")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    
    args = parser.parse_args()
    
    # 🌟 修复：自动搜索配置文件（和 runtime.py / tts_service.py 保持一致）
    config_path = args.config
    if config_path is None:
        # 尝试从项目根目录定位 config.yaml
        project_root = Path(__file__).resolve().parents[2]
        candidate = project_root / "config" / "config.yaml"
        if candidate.exists():
            config_path = str(candidate)
            print(f"自动定位配置文件: {config_path}")
        else:
            # 兜底搜索
            for p in ["config/config.yaml", "config.yaml", "../config/config.yaml"]:
                if Path(p).exists():
                    config_path = p
                    break
    
    server = CoreServer(config_path=config_path)
    server.run()


if __name__ == "__main__":
    main()
