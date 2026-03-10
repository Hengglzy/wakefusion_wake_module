"""
TTS服务模块 - Qwen3-TTS-12Hz-0.6B-Base
接收LLM的文本消息，进行语音合成，通过ZMQ PUSH发送音频数据给Core Server

通信协议：
  - WebSocket Server：接收LLM的文本消息（ws://0.0.0.0:{tts_ws_port}）
  - ZMQ PUSH：向Core Server发送音频数据（tcp://127.0.0.1:{tts_push_port}）
  - ZMQ SUB：订阅Core Server的停止信号（tcp://127.0.0.1:{tts_stop_pub_port}）
"""
import json
import logging
import re
import threading
import time
import zmq
import numpy as np
from typing import Optional, List
from wakefusion.config import get_config

logger = logging.getLogger(__name__)


class TTSModule:
    """TTS模块：Qwen3-TTS-12Hz-0.6B-Base（Voice Clone）"""
    
    def __init__(self, config):
        """
        初始化TTS模块
        
        Args:
            config: 应用配置对象
        """
        self.config = config
        self.tts_config = config.tts
        self.zmq_config = config.zmq
        self.websocket_config = config.websocket
        
        # 熔断机制（使用threading.Event）
        self._stop_event = threading.Event()
        
        # 字符缓冲区（标点切分）
        self.char_buffer = ""
        
        # Qwen3-TTS模型
        self.tts = None
        
        # 参考音频路径（Voice Clone必需）
        self.ref_audio_path = self.tts_config.ref_audio_path
        
        # ZMQ和WebSocket
        self.zmq_context = None
        self.push_socket = None
        self.stop_sub_socket = None
        self.ws_server = None
        self.ws_client = None  # 单个LLM客户端连接（点对点模式）
        
        # 运行状态
        self._running = False
        
        # 加载Qwen3-TTS模型
        self._load_model()
        
        # 初始化ZMQ
        self._init_zmq()
        
        # 冷启动预热
        self._warmup()
        
        # 启动停止信号监听线程
        self._start_stop_signal_listener()
        
        # 初始化WebSocket
        self._init_websocket()
    
    def _load_model(self):
        """加载Qwen3-TTS模型"""
        try:
            logger.info(f"正在加载Qwen3-TTS模型: {self.tts_config.model_name}...")
            from qwen_tts import Qwen3TTSModel
            
            self.tts = Qwen3TTSModel.from_pretrained(
                "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                device_map="cuda:0"  # 强制GPU
            )
            logger.info("Qwen3-TTS模型加载成功")
        except ImportError:
            logger.error("qwen-tts包未安装，请运行: pip install qwen-tts")
            raise
        except Exception as e:
            logger.error(f"Qwen3-TTS模型加载失败: {e}")
            raise
    
    def _init_zmq(self):
        """初始化ZMQ Sockets"""
        self.zmq_context = zmq.Context()
        
        # ZMQ PUSH Socket（发送音频到Core Server）
        self.push_socket = self.zmq_context.socket(zmq.PUSH)
        self.push_socket.bind(f"tcp://127.0.0.1:{self.zmq_config.tts_push_port}")
        logger.info(f"ZMQ PUSH Socket已绑定: tcp://127.0.0.1:{self.zmq_config.tts_push_port}")
        
        # ZMQ SUB Socket（订阅停止信号）
        self.stop_sub_socket = self.zmq_context.socket(zmq.SUB)
        self.stop_sub_socket.connect(f"tcp://127.0.0.1:{self.zmq_config.tts_stop_pub_port}")
        self.stop_sub_socket.setsockopt_string(zmq.SUBSCRIBE, "STOP_SYNTHESIS")
        logger.info(f"ZMQ SUB Socket已连接: tcp://127.0.0.1:{self.zmq_config.tts_stop_pub_port}")
    
    def _warmup(self):
        """冷启动预热（Warm-up）"""
        if not self.tts_config.warmup_enabled:
            logger.info("冷启动预热已禁用")
            return
        
        logger.info("开始Qwen3-TTS冷启动预热...")
        warmup_text = self.tts_config.warmup_text
        
        try:
            # 执行一次完整的合成流程（使用参考音频）
            # generate_voice_clone 返回 (wavs, sr) 元组，不是生成器
            wavs, sr = self.tts.generate_voice_clone(
                text=warmup_text,
                ref_audio=self.ref_audio_path,
                x_vector_only_mode=True  # 声纹抽取模式，不需要ref_text
            )
            
            # 预热完成，丢弃音频数据
            logger.info("Qwen3-TTS预热完成，模型已就绪")
        except Exception as e:
            logger.error(f"Qwen3-TTS预热失败: {e}")
            raise
    
    def _start_stop_signal_listener(self):
        """启动停止信号监听线程"""
        def listener():
            while self._running:
                try:
                    # 非阻塞接收停止信号
                    message = self.stop_sub_socket.recv_string(zmq.NOBLOCK)
                    if message == "STOP_SYNTHESIS":
                        logger.warning("收到停止信号，设置停止标志位")
                        self._stop_event.set()  # 设置全局停止标志
                except zmq.Again:
                    time.sleep(0.01)
                    continue
                except Exception as e:
                    logger.error(f"停止信号监听出错: {e}")
                    time.sleep(0.1)
        
        thread = threading.Thread(target=listener, daemon=True)
        thread.start()
        logger.info("停止信号监听线程已启动")
    
    def _init_websocket(self):
        """初始化WebSocket服务器"""
        try:
            import websockets
            from websockets.server import serve
            import asyncio
            
            async def websocket_handler(websocket, path):
                """WebSocket连接处理"""
                logger.info(f"新的WebSocket客户端连接: {websocket.remote_address}")
                self.ws_client = websocket
                
                try:
                    async for message in websocket:
                        # 解析消息
                        try:
                            data = json.loads(message)
                            await self._handle_websocket_message(data)
                        except json.JSONDecodeError:
                            logger.warning(f"无效的JSON消息: {message}")
                        except Exception as e:
                            logger.error(f"处理WebSocket消息失败: {e}")
                
                except websockets.exceptions.ConnectionClosed:
                    pass
                finally:
                    self.ws_client = None
                    logger.info(f"WebSocket客户端断开: {websocket.remote_address}")
            
            # 启动WebSocket服务器（在独立线程中）
            async def run_ws_server():
                async with serve(websocket_handler, "0.0.0.0", self.websocket_config.tts_port):
                    await asyncio.Future()  # 永久运行
            
            self.ws_thread = threading.Thread(
                target=lambda: asyncio.run(run_ws_server()),
                daemon=True
            )
            self.ws_thread.start()
            logger.info(f"WebSocket服务器已启动: ws://0.0.0.0:{self.websocket_config.tts_port}")
        
        except ImportError:
            logger.warning("websockets未安装，WebSocket功能将不可用")
            self.ws_client = None
    
    async def _handle_websocket_message(self, data: dict):
        """
        处理WebSocket消息
        
        Args:
            data: 消息数据（JSON格式）
        """
        msg_type = data.get("type")
        
        if msg_type == "tts_request":
            # TTS请求
            text = data.get("text", "")
            is_streaming = data.get("is_streaming", True)
            is_final = data.get("is_final", False)
            
            if text:
                # 处理流式文本（标点切分）
                self.process_streaming_text(text, is_final=is_final)
        
        elif msg_type == "stop_synthesis":
            # 停止合成
            logger.warning("收到WebSocket停止信号")
            self._stop_event.set()
    
    def process_streaming_text(self, text_chunk: str, is_final: bool = False):
        """
        处理流式文本，标点切分
        
        Args:
            text_chunk: 文本块
            is_final: 是否为最后一块文本
        """
        if self._stop_event.is_set():
            return  # 如果已停止，忽略新文本
        
        self.char_buffer += text_chunk
        
        # 标点检测正则
        punctuation_pattern = re.compile(r'[。！？.!?，,]')
        
        while True:
            match = punctuation_pattern.search(self.char_buffer)
            if not match:
                break
            
            # 截取到标点符号的短句
            sentence = self.char_buffer[:match.end()]
            self.char_buffer = self.char_buffer[match.end():]
            
            # 合成并发送（带熔断机制）
            self.synthesize_with_cutoff(sentence)
        
        # 如果是最后一块文本，处理剩余的缓冲区内容
        if is_final and self.char_buffer:
            if len(self.char_buffer) >= self.tts_config.min_sentence_length:
                self.synthesize_with_cutoff(self.char_buffer)
            self.char_buffer = ""
    
    def synthesize_with_cutoff(self, text: str):
        """
        带熔断机制的合成方法
        
        Args:
            text: 要合成的文本
        """
        if not text or len(text.strip()) == 0:
            return
        
        self._stop_event.clear()  # 重置停止标志
        
        try:
            # generate_voice_clone 返回 (wavs, sr) 元组，不是生成器
            # wavs 是音频数组列表，sr 是采样率
            wavs, sr = self.tts.generate_voice_clone(
                text=text,
                ref_audio=self.ref_audio_path,
                x_vector_only_mode=True  # 声纹抽取模式，不需要ref_text
            )
            
            # 检查停止标志（在合成完成后检查）
            if self._stop_event.is_set():
                logger.warning(f"检测到停止标志，丢弃合成结果: {text[:20]}...")
                return  # 直接返回，不发送音频
            
            # 提取音频数据（wavs 是列表，取第一个元素）
            if wavs and len(wavs) > 0:
                audio_chunk = wavs[0]
                # 发送音频块到ZMQ
                self._send_audio_chunk(audio_chunk)
            else:
                logger.warning(f"TTS合成结果为空: {text[:20]}...")
        
        except Exception as e:
            logger.error(f"TTS合成失败: {e}")
        
        finally:
            # 如果被中断，清空字符缓冲区
            if self._stop_event.is_set():
                self.char_buffer = ""
    
    def _send_audio_chunk(self, audio_chunk):
        """
        发送音频块到Core Server（ZMQ PUSH）
        
        Args:
            audio_chunk: 音频数据（numpy数组或torch tensor）
        """
        try:
            # 转换为numpy数组（如果是torch tensor）
            if hasattr(audio_chunk, 'cpu'):
                audio_chunk = audio_chunk.cpu().numpy()
            elif hasattr(audio_chunk, 'numpy'):
                audio_chunk = audio_chunk.numpy()
            
            # 确保是int16格式
            if audio_chunk.dtype != np.int16:
                # 归一化到[-1, 1]范围，然后转换为int16
                if audio_chunk.dtype == np.float32 or audio_chunk.dtype == np.float64:
                    audio_chunk = np.clip(audio_chunk, -1.0, 1.0)
                    audio_chunk = (audio_chunk * 32767).astype(np.int16)
                else:
                    audio_chunk = audio_chunk.astype(np.int16)
            
            # 第一帧：JSON元数据
            metadata = {
                "type": "tts_audio",
                "sample_rate": self.tts_config.sample_rate,
                "channels": 1,
                "timestamp": time.time()
            }
            
            # 第二帧：原始PCM音频数据（int16，二进制）
            # 使用Multipart Message发送
            self.push_socket.send_multipart([
                json.dumps(metadata).encode('utf-8'),
                audio_chunk.tobytes()
            ], zmq.NOBLOCK)
        
        except zmq.Again:
            # 发送缓冲区满，丢弃此帧
            pass
        except Exception as e:
            logger.error(f"发送音频块失败: {e}")
    
    def start(self):
        """启动TTS模块"""
        self._running = True
        logger.info("TTS模块已启动")
    
    def stop(self):
        """停止TTS模块"""
        self._running = False
        self._stop_event.set()  # 设置停止标志
        
        # 关闭ZMQ sockets
        if self.push_socket:
            self.push_socket.close()
        if self.stop_sub_socket:
            self.stop_sub_socket.close()
        if self.zmq_context:
            self.zmq_context.term()
        
        logger.info("TTS模块已停止")


def main():
    """TTS模块主入口"""
    import logging
    
    # 配置日志
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    
    # 加载配置
    config = get_config()
    
    # 创建TTS模块
    tts_module = TTSModule(config)
    
    try:
        # 启动模块
        tts_module.start()
        
        # 主线程等待
        while True:
            time.sleep(1)
    
    except KeyboardInterrupt:
        logger.info("收到中断信号，正在关闭...")
    finally:
        tts_module.stop()


if __name__ == "__main__":
    main()
