"""
ASR服务模块 - FunASR流式识别
接收Core Server的音频数据，进行实时语音识别，通过WebSocket发送识别结果给LLM

通信协议：
  - ZMQ PULL：接收Core Server的音频数据（tcp://127.0.0.1:{asr_pull_port}）
  - WebSocket Server：向LLM发送识别文本（ws://0.0.0.0:{asr_ws_port}）
"""
import json
import logging
import threading
import time
import zmq
import numpy as np
from typing import Optional, Dict, Any
from wakefusion.config import get_config

logger = logging.getLogger(__name__)


class ASRModule:
    """ASR模块：FunASR流式识别"""
    
    def __init__(self, config):
        """
        初始化ASR模块
        
        Args:
            config: 应用配置对象
        """
        self.config = config
        self.asr_config = config.asr
        self.zmq_config = config.zmq
        self.websocket_config = config.websocket
        
        # FunASR模型和缓存
        self.model = None
        self.cache = {}  # 状态缓存，必须在整个一句话期间维护
        self.text_buffer = ""  # 🌟 修复：新增文本缓存盆，接住增量输出的每一段文字
        
        # ZMQ和WebSocket
        self.zmq_context = None
        self.pull_socket = None
        self.ws_server = None
        self.ws_clients = []  # WebSocket客户端连接列表
        self.ws_loop = None  # WebSocket线程的event loop引用
        
        # 运行状态
        self._running = False
        self.discarding = False  # 拒收模式：强杀后丢弃所有音频
        
        # 加载FunASR模型
        self._load_model()
        
        # 初始化ZMQ和WebSocket
        self._init_zmq()
        self._init_websocket()
    
    def _load_model(self):
        """加载FunASR模型（paraformer-zh-online）"""
        try:
            logger.info(f"正在加载FunASR模型: {self.asr_config.model_name}...")
            
            # 导入FunASR AutoModel
            try:
                from funasr import AutoModel
            except ImportError:
                logger.error("FunASR未安装，请运行: pip install funasr")
                raise
            
            # 加载模型（paraformer-zh-online支持流式识别）
            # 如果配置了model_path，使用本地路径；否则使用模型名称自动下载
            if self.asr_config.model_path and self.asr_config.model_path.strip():
                model_path = self.asr_config.model_path.strip()
                logger.info(f"使用本地模型路径: {model_path}")
                self.model = AutoModel(
                    model=model_path,
                    trust_remote_code=True,  # 绕过安全验证
                    disable_update=True  # 加快启动速度
                )
            else:
                # 使用模型名称，FunASR会自动从ModelScope下载
                logger.info(f"使用模型名称自动下载: {self.asr_config.model_name}")
                self.model = AutoModel(
                    model=self.asr_config.model_name,
                    trust_remote_code=True,  # 绕过安全验证
                    disable_update=True  # 加快启动速度
                )
            
            logger.info(f"✅ FunASR模型加载成功: {self.asr_config.model_name}")
            
        except ImportError as e:
            logger.error(f"FunASR导入失败: {e}")
            logger.error("请确保已安装FunASR: pip install funasr modelscope")
            raise
        except Exception as e:
            logger.error(f"FunASR模型加载失败: {e}")
            logger.warning("ASR功能将不可用，但模块会继续运行（跳过推理）")
            self.model = None  # 设置为None，后续会跳过推理
    
    def _init_zmq(self):
        """初始化ZMQ PULL Socket"""
        self.zmq_context = zmq.Context()
        self.pull_socket = self.zmq_context.socket(zmq.PULL)
        self.pull_socket.setsockopt(zmq.RCVHWM, 50)  # 接收端限制积压，宁可丢帧也不能让延迟累积
        self.pull_socket.bind(f"tcp://127.0.0.1:{self.zmq_config.asr_pull_port}")
        logger.info(f"ZMQ PULL Socket已绑定: tcp://127.0.0.1:{self.zmq_config.asr_pull_port}")
        
        # 订阅带外强杀信号
        self.ctrl_sub_socket = self.zmq_context.socket(zmq.SUB)
        self.ctrl_sub_socket.connect(f"tcp://127.0.0.1:{self.zmq_config.tts_stop_pub_port}")
        self.ctrl_sub_socket.setsockopt_string(zmq.SUBSCRIBE, "ABORT_ASR")
        
        # 使用 Poller 监听双通道
        self.poller = zmq.Poller()
        self.poller.register(self.pull_socket, zmq.POLLIN)
        self.poller.register(self.ctrl_sub_socket, zmq.POLLIN)
    
    def _init_websocket(self):
        """初始化WebSocket服务器"""
        try:
            import websockets
            from websockets.server import serve
            
            async def websocket_handler(websocket, path):
                """WebSocket连接处理"""
                logger.info(f"新的WebSocket客户端连接: {websocket.remote_address}")
                self.ws_clients.append(websocket)
                try:
                    async for message in websocket:
                        # 处理客户端消息（如果需要）
                        pass
                except websockets.exceptions.ConnectionClosed:
                    pass
                finally:
                    if websocket in self.ws_clients:
                        self.ws_clients.remove(websocket)
                    logger.info(f"WebSocket客户端断开: {websocket.remote_address}")
            
            # 启动WebSocket服务器（在独立线程中）
            import asyncio
            async def run_ws_server():
                # 保存event loop引用，供其他线程使用
                self.ws_loop = asyncio.get_event_loop()
                async with serve(websocket_handler, "0.0.0.0", self.websocket_config.asr_port):
                    await asyncio.Future()  # 永久运行
            
            self.ws_thread = threading.Thread(
                target=lambda: asyncio.run(run_ws_server()),
                daemon=True
            )
            self.ws_thread.start()
            logger.info(f"WebSocket服务器已启动: ws://0.0.0.0:{self.websocket_config.asr_port}")
        except ImportError:
            logger.warning("websockets未安装，WebSocket功能将不可用")
            self.ws_clients = []
    
    def _send_to_llm(self, text: str, is_final: bool = False, confidence: float = 0.0):
        """
        通过WebSocket向LLM发送识别结果
        
        Args:
            text: 识别文本
            is_final: 是否为最终结果
            confidence: 置信度
        """
        if not self.ws_clients:
            return
        
        message = {
            "type": "asr_result",
            "text": text,
            "is_final": is_final,
            "confidence": confidence,
            "timestamp": time.time()
        }
        
        message_json = json.dumps(message, ensure_ascii=False)
        
        # 向所有连接的客户端广播（线程安全）
        import asyncio
        async def broadcast():
            disconnected = []
            for client in list(self.ws_clients):  # 创建副本避免迭代时修改
                try:
                    await client.send(message_json)
                except Exception as e:
                    logger.warning(f"发送WebSocket消息失败: {e}")
                    disconnected.append(client)
            
            # 清理断开的连接
            for client in disconnected:
                if client in self.ws_clients:
                    self.ws_clients.remove(client)
        
        # 使用线程安全的方式调用（从音频处理线程调用WebSocket线程的event loop）
        if self.ws_loop and self.ws_loop.is_running():
            try:
                asyncio.run_coroutine_threadsafe(broadcast(), self.ws_loop)
            except Exception as e:
                logger.error(f"WebSocket广播失败: {e}")
        else:
            logger.warning("WebSocket event loop 不可用，无法发送消息")
    
    def _process_audio_chunk(self, audio_chunk: bytes, is_final: bool = False):
        """
        处理音频块，进行FunASR推理
        
        Args:
            audio_chunk: 音频数据（二进制PCM，int16格式）
            is_final: 是否为最终音频块
        """
        if self.model is None:
            logger.warning("FunASR模型未加载，跳过推理")
            return
        
        try:
            # 将二进制PCM转换为numpy数组
            audio_array = np.frombuffer(audio_chunk, dtype=np.int16)
            
            # 如果音频块为空且不是最终标记，跳过
            if len(audio_array) == 0 and not is_final:
                return
            
            # FunASR流式推理（使用cache机制）
            # paraformer-zh-online支持流式识别，需要维护cache状态
            if is_final:
                # FunASR 不接受 None，如果是空数组，喂给它一段微小的静音来安全触发结算
                if len(audio_array) == 0:
                    # 🌟 修复：尾音截断问题。FunASR 需要足够的尾部静音来把最后一个字的拼音吐出来
                    # 如果只给 10ms，它可能觉得还没说完就直接结算，丢掉最后一个词
                    # 这里给它补充 ~300ms 的静音数据 (16kHz * 0.3s = 4800 采样点)
                    audio_array = np.zeros(4800, dtype=np.int16)
                
                # 最终结果：设置is_final=True获取完整识别文本
                res = self.model.generate(
                    input=audio_array,
                    cache=self.cache,
                    is_final=True
                )
            else:
                # 中间结果：实时推理，返回partial_text
                res = self.model.generate(
                    input=audio_array,
                    cache=self.cache,
                    is_final=False
                )
            
            # 提取识别文本
            # FunASR返回格式可能是列表、dict或字符串，需要适配
            if isinstance(res, list) and len(res) > 0:
                # 列表格式：[{'key': '...', 'text': '...'}] 或 [{'text': '...'}]
                first_item = res[0]
                if isinstance(first_item, dict):
                    text = first_item.get("text", "")
                    confidence = first_item.get("confidence", 0.95)
                else:
                    text = str(first_item)
                    confidence = 0.95
            elif isinstance(res, dict):
                text = res.get("text", "")
                # 如果有置信度信息，提取
                confidence = res.get("confidence", 0.0)
            elif isinstance(res, str):
                text = res
                confidence = 0.95  # 默认置信度
            else:
                # 其他格式，尝试转换为字符串
                text = str(res) if res else ""
                confidence = 0.95
            
            # 发送识别结果（如果有文本）
            # 🌟 修复：增量拼接，FunASR 会一段段吐出已确认的文字，并把缓存清除
            # 我们必须用一个 "文字拼接盆" 接住所有中间片断，最后一起发给 LLM
            if text and text.strip():
                self.text_buffer += text.strip()
                if not is_final:
                    logger.debug(f"📝 ASR新增中间结果: {text.strip()} (当前累积: {self.text_buffer})")
            
            # 🌟 结算时发送盆里的所有拼接文字
            if is_final:
                final_text = self.text_buffer.strip()
                if final_text:
                    self._send_to_llm(final_text, is_final=True, confidence=confidence)
                    logger.info(f"✅ ASR最终完整识别结果推送: {final_text}")
                else:
                    logger.info("✅ ASR最终识别结果为空")
                
                # 重置状态
                self.cache = {}
                self.text_buffer = ""
                logger.info("ASR已结算完毕，清空cache与文字盆")
        
        except Exception as e:
            logger.error(f"FunASR推理失败: {e}")
            # 如果是最终结果，即使出错也要重置cache
            if is_final:
                self.cache = {}
                self.text_buffer = ""
                logger.warning("ASR推理出错，已强制重置cache")
    
    def process_audio_stream(self):
        """处理音频流（主循环）"""
        logger.info("开始处理音频流...")
        while self._running:
            try:
                socks = dict(self.poller.poll(timeout=100))
                
                # 1. 优先处理带外强杀
                if getattr(self, 'ctrl_sub_socket', None) in socks:
                    ctrl_msg = self.ctrl_sub_socket.recv_string(zmq.NOBLOCK)
                    if "ABORT_ASR" in ctrl_msg:
                        logger.info("🚨 收到带外强杀信号，进入拒收模式并清空队列！")
                        self.discarding = True
                        self.cache = {}
                        self.text_buffer = ""
                        while True:
                            try:
                                self.pull_socket.recv(zmq.NOBLOCK)
                            except zmq.Again:
                                break
                        continue
                
                # 2. 正常处理音频流
                if self.pull_socket in socks:
                    message = self.pull_socket.recv(zmq.NOBLOCK)
                    
                    if message == b"START_OF_SPEECH":
                        self.discarding = False
                        # 🌟 修复 Bug #2: 唤醒时发送 START_OF_SPEECH 只是为了解除拒收模式，绝对不可以在此时清空缓存与文字盆！
                        # 因为此时 ASR_service 的 receive 队列里，可能已经积压了 Audio_service 传来的带有这句话开头的 1S 回捞音频
                    elif message == b"END_OF_SPEECH":
                        if not self.discarding:
                            logger.info("收到END_OF_SPEECH，获取最终识别结果")
                            self._process_audio_chunk(b"", is_final=True)
                    elif message == b"ABORT_SPEECH":
                        self.discarding = True
                        self.cache = {}
                        self.text_buffer = ""
                    else:
                        if not self.discarding:
                            self._process_audio_chunk(message, is_final=False)
                        
            except zmq.Again:
                continue
            except Exception as e:
                logger.error(f"处理音频流时出错: {e}")
    
    def start(self):
        """启动ASR模块"""
        self._running = True
        
        # 启动音频处理线程
        process_thread = threading.Thread(target=self.process_audio_stream, daemon=True)
        process_thread.start()
        
        logger.info("ASR模块已启动")
    
    def stop(self):
        """停止ASR模块"""
        self._running = False
        
        # 关闭ZMQ sockets
        if self.pull_socket:
            self.pull_socket.close()
        if hasattr(self, 'ctrl_sub_socket') and self.ctrl_sub_socket:
            self.ctrl_sub_socket.close()
        if self.zmq_context:
            self.zmq_context.term()
        
        logger.info("ASR模块已停止")


def main():
    """ASR模块主入口"""
    import logging
    from pathlib import Path
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 🌟 修复：显式指定项目根目录下的 config.yaml，确保读取到自定义配置
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "config" / "config.yaml"
    config = get_config(str(config_path))
    
    # 创建ASR模块
    asr_module = ASRModule(config)
    
    try:
        # 启动模块
        asr_module.start()
        
        # 主线程等待
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭...")
    finally:
        asr_module.stop()


if __name__ == "__main__":
    main()
