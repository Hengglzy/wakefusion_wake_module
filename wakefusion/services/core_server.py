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
        
        # Poller用于同时监听多个socket
        self.poller = zmq.Poller()
        self.poller.register(self.vision_sub_socket, zmq.POLLIN)
        self.poller.register(self.audio_sub_socket, zmq.POLLIN)
        
        # 状态管理
        self.current_state = SystemState.IDLE
        
        # 视觉状态缓存
        self._latest_vision_wake: bool = False
        self._last_vision_timestamp: float = 0.0
        
        # VAD超时管理
        self._last_vad_time: float = 0.0
        self.current_silence_timeout: float = self.conversation_config.vad_silence_timeout_default_sec
        self._vad_timeout_thread: Optional[threading.Thread] = None
        self._vad_timeout_should_exit: bool = False
        
        # 对话轮次和唤醒路径管理
        self._conversation_round: int = 0  # 对话轮次：0=首次唤醒，1+=持续对话
        self._wake_path: str = "unknown"  # 唤醒路径："visual" 或 "audio"
        
        # 宏微观双重超时管理
        self._user_has_spoken: bool = False  # 用户是否已开口（用于微观超时判断）
        
        # 初始化TTS相关sockets
        self._init_tts_sockets()
        
        # 初始化控制socket
        self._init_control_socket()
        
        # 启动音频播放线程
        self._start_playback_thread()
        
        logger.info("Core Server initialized")
        logger.info(f"  Vision SUB: tcp://127.0.0.1:{self.zmq_config.vision_pub_port}")
        logger.info(f"  Audio SUB: tcp://127.0.0.1:{self.zmq_config.audio_pub_port}")
        logger.info(f"  Audio REQ: tcp://127.0.0.1:{self.zmq_config.audio_ctrl_port}")
        logger.info(f"  TTS PULL: tcp://127.0.0.1:{self.zmq_config.tts_push_port}")
        logger.info(f"  TTS STOP PUB: tcp://127.0.0.1:{self.zmq_config.tts_stop_pub_port}")
        logger.info(f"  Control REP: tcp://127.0.0.1:{self.zmq_config.core_control_rep_port}")
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
    
    def _init_control_socket(self):
        """初始化控制socket（接收LLM指令）"""
        self._control_rep_socket = self.zmq_context.socket(zmq.REP)
        self._control_rep_socket.bind(f"tcp://127.0.0.1:{self.zmq_config.core_control_rep_port}")
        self._control_rep_socket.setsockopt(zmq.RCVTIMEO, 100)  # 100ms超时，非阻塞
        self.poller.register(self._control_rep_socket, zmq.POLLIN)
    
    def _start_playback_thread(self):
        """启动音频播放线程"""
        self._playback_active = True
        self._playback_thread = threading.Thread(target=self._audio_playback_worker, daemon=True)
        self._playback_thread.start()
    
    def _audio_playback_worker(self):
        """音频播放工作线程"""
        while self._playback_active:
            try:
                audio_data = self.audio_playback_queue.get(timeout=0.1)
                # TODO: 实际播放音频（需要音频播放库，如sounddevice或pyaudio）
                # self._play_audio(audio_data)
                logger.debug(f"播放音频块: {len(audio_data)} bytes")
            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"音频播放异常: {e}")
    
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
        
        # 1. 清空播报队列
        self.clear_playback_queue()
        
        # 2. 向TTS发送停止信号（ZMQ PUB）
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
    
    def _transition_to_idle(self):
        """转换到IDLE状态"""
        logger.info("状态转换: -> IDLE")
        # 发送结束标记（告诉ASR这句话说完了）
        self._send_end_marker()
        # 发送恢复阈值指令
        self._send_threshold_command(self.audio_threshold_config.default)
        # 重置冷却期，允许立即再次唤醒
        self._send_reset_cooldown_command()
        # 重置对话轮次和唤醒路径
        self._conversation_round = 0
        self._wake_path = "unknown"
        self.current_state = SystemState.IDLE
    
    def _transition_to_visual_wake(self):
        """转换到VISUAL_WAKE状态"""
        logger.info("状态转换: -> VISUAL_WAKE")
        # 发送结束标记（告诉ASR这句话说完了，即使视觉仍在唤醒状态）
        self._send_end_marker()
        # 不恢复阈值（保持0.4）
        self.current_state = SystemState.VISUAL_WAKE
    
    def _transition_to_listening(self, wake_path: str = "unknown"):
        """转换到LISTENING状态"""
        logger.info(f"状态转换: -> LISTENING (路径: {wake_path})")
        
        # 如果从PROCESSING/SPEAKING状态转换过来，触发硬打断
        if self.current_state in [SystemState.PROCESSING, SystemState.SPEAKING]:
            self._handle_hard_cutoff()
        
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
        
        # 初始化VAD超时管理
        self._user_has_spoken = False
        self._last_vad_time = time.time()
        self.current_silence_timeout = self.conversation_config.vad_silence_timeout_default_sec
        self._vad_timeout_should_exit = False
        
        # 如果是进入免唤醒持续对话，必须主动通知音频底层拉起流模式
        if wake_path == "持续对话":
            self._send_start_streaming_command()
        
        # 启动VAD超时检查线程
        if self._vad_timeout_thread is None or not self._vad_timeout_thread.is_alive():
            self._vad_timeout_thread = threading.Thread(target=self._vad_timeout_checker, daemon=True)
            self._vad_timeout_thread.start()
        
        self.current_state = SystemState.LISTENING
    
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
    
    def _vad_timeout_checker(self):
        """VAD超时检查线程（宏微观双重监控机制）"""
        while self.current_state == SystemState.LISTENING:
            if self._vad_timeout_should_exit:
                break
            
            time.sleep(self.conversation_config.vad_check_interval_ms / 1000.0)
            time_since_last_vad = time.time() - self._last_vad_time
            
            if self._user_has_spoken:
                # 🔪 微观刀刃：只要 1.5 秒没声音就截断
                if time_since_last_vad > 1.5:
                    logger.info("🔪 VAD微观刀刃触发（1.5s尾音静音），判定用户说完")
                    self._send_end_marker()
                    # 重点：进入PROCESSING状态，挂起超时检查，静等大模型思考
                    self.current_state = SystemState.PROCESSING
                    break  # 退出监听状态的心跳检测
            else:
                # ⏳ 宏观容器：发呆超过 8 秒直接关门
                if time_since_last_vad > self.current_silence_timeout:
                    logger.info(f"⏳ 宏观发呆超时（{self.current_silence_timeout}s无语音），退回IDLE")
                    self._handle_vad_timeout()
                    break
    
    def _handle_vad_timeout(self):
        """处理VAD超时"""
        # 检查视觉状态
        if self._latest_vision_wake:
            # 退回到VISUAL_WAKE（发送结束标记）
            self._transition_to_visual_wake()
        else:
            # 退回到IDLE（发送结束标记）
            # 注意：首次对话完成时，conversation_round会在进入PROCESSING/SPEAKING时更新
            self._transition_to_idle()
    
    def _transition_to_speaking(self):
        """转换到SPEAKING状态（TTS开始播放）"""
        logger.info("状态转换: -> SPEAKING")
        self.current_state = SystemState.SPEAKING
        # 🎙️ 关键：播报时让耳朵重置，回去抓取唤醒词（准备随时硬打断）
        self._send_reset_cooldown_command()
        # conversation_round 保持不变（首次=0，持续对话>=1）
    
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
        
        # 转换到 LISTENING 状态（等待用户继续说话）
        self._transition_to_listening("持续对话")
    
    def _handle_visual_cutoff(self, context: str = "监听"):
        """处理视觉斩断（最高优先级打断）"""
        logger.warning(f"🚨 视觉斩断：检测到用户离开，立即切断{context}")
        # 停止VAD超时检查线程（如果正在运行）
        if self._vad_timeout_thread and self._vad_timeout_thread.is_alive():
            # 标记线程应该退出（通过共享标志）
            self._vad_timeout_should_exit = True
        # 状态瞬间回落到IDLE
        self._transition_to_idle()
    
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
        
        # 更新视觉状态缓存
        self._latest_vision_wake = vision_wake
        self._last_vision_timestamp = time.time()
        
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
        vad = metadata.get("vad", False)
        wake_word = metadata.get("wake_word", {})
        
        # 更新VAD时间戳（展厅防抖机制）
        if vad:
            self._vad_speech_count += 1
            self._vad_silence_count = 0
            
            # 必须连续2次（0.4秒）检测到人声，才认定为真语音，避免瞬间噪音（如咳嗽、碰撞）打断1.5秒计时
            if self._vad_speech_count >= 2:
                self._last_vad_time = time.time()
                if not self._user_has_spoken:
                    self._user_has_spoken = True
                    logger.info("🗣️ 检测到用户真实开口，切换为1.5秒微观截断模式")
        else:
            self._vad_silence_count += 1
            self._vad_speech_count = 0
            # 真正的静音不需要操作，让 time_since_last_vad 自然累加即可触发超时
        
        # 处理唤醒词检测
        if wake_word.get("detected", False):
            confidence = wake_word.get("confidence", 0.0)
            
            # 路径A：视觉降维打击（在VISUAL_WAKE状态下）
            if (self.current_state == SystemState.VISUAL_WAKE and 
                confidence >= self.audio_threshold_config.visual_wake):
                self._transition_to_listening("视觉降维打击")
            
            # 路径B：纯语音唤醒（在IDLE状态下，突破高阈值）
            elif (self.current_state == SystemState.IDLE and 
                  confidence >= self.audio_threshold_config.default):
                self._transition_to_listening("纯语音唤醒")
            
            # 路径C：唤醒词硬打断（在SPEAKING状态下）
            elif (self.current_state == SystemState.SPEAKING and 
                  confidence >= self.audio_threshold_config.default):
                logger.info("🛑 听到唤醒词，触发硬打断！")
                self._transition_to_listening("唤醒词打断")
        
        # 在LISTENING状态下，转发音频到ASR
        if self.current_state == SystemState.LISTENING:
            if self._asr_push_socket:
                try:
                    # 直接转发二进制音频数据
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
    
    parser = argparse.ArgumentParser(description="Core Server - 核心决策模块")
    parser.add_argument("--config", type=str, default=None, help="配置文件路径")
    
    args = parser.parse_args()
    
    server = CoreServer(config_path=args.config)
    server.run()


if __name__ == "__main__":
    main()
