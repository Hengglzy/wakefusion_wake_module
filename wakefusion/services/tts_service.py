"""
TTS服务模块 - Qwen3-TTS-12Hz-0.6B-Base
接收Core Server的文本消息，进行语音合成，通过ZMQ PUSH发送音频数据给Core Server

通信协议：
  - ZMQ PULL：接收Core Server的合成文本（tcp://127.0.0.1:{tts_text_pull_port}）
  - ZMQ PUSH：向Core Server发送音频数据（tcp://127.0.0.1:{tts_push_port}）
  - ZMQ SUB：订阅Core Server的停止信号（tcp://127.0.0.1:{tts_stop_pub_port}）
"""
import json
import logging
import os
import re
import threading
import time
import queue
from pathlib import Path
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
        
        # 熔断机制（使用threading.Event）
        self._stop_event = threading.Event()
        
        # 文本处理队列（基于句子的异步队列）
        self.synthesis_queue = queue.Queue()
        self.char_buffer = ""
        # 标点符号切分正则（支持中英文多种标点）
        self.punctuation_pattern = re.compile(self.tts_config.punctuation_pattern)
        
        # 🌟 首句极速模式：首句用逗号切分，后续用句号切分
        self._first_sentence_pattern = re.compile(r'[，,]+')  # 首句切分：逗号
        self._subsequent_sentence_pattern = re.compile(r'[。！？.!?]+')  # 后续切分：句号、问号、感叹号
        self._has_sent_first_sentence = False  # 是否已发送第一句
        
        # 记录是否正在合成
        self._is_synthesizing = False
        self._synthesis_thread: Optional[threading.Thread] = None
        
        # Qwen3-TTS模型
        self.tts = None
        
        # 参考音频路径（Voice Clone必需）
        self.ref_audio_path = self.tts_config.ref_audio_path
        
        # ZMQ
        self.zmq_context = None
        self.text_pull_socket = None  # 接收Core Server的合成文本
        self.push_socket = None  # 推送音频给Core Server
        self.stop_sub_socket = None
        
        # 运行状态
        self._running = False
        
        # 加载Qwen3-TTS模型
        self._load_model()
        
        # 初始化ZMQ
        self._init_zmq()
        
        # 冷启动预热
        self._warmup()
        
    def _start_synthesis_worker(self):
        """启动后台合成工作线程"""
        def worker():
            logger.info("合成工作线程已启动")
            while self._running:
                try:
                    # 获取待合成的文本块，带超时以便检查停止标志
                    text_chunk, is_final_chunk = self.synthesis_queue.get(timeout=0.1)
                        
                    if text_chunk == "END_OF_TTS_SESSION":
                        # 收到结束信号，发送TTS结束标记
                        self._send_tts_end_marker()
                        self.synthesis_queue.task_done()
                        continue
                        
                    self._is_synthesizing = True
                    try:
                        self.synthesize_with_cutoff(text_chunk)
                    except Exception as e:
                        logger.error(f"合成工作线程异常: {e}")
                    finally:
                        self._is_synthesizing = False
                        self.synthesis_queue.task_done()
                        
                except queue.Empty:
                    continue
                except Exception as e:
                    logger.error(f"合成工作线程出错: {e}")
                    time.sleep(0.1)
                    
        self._synthesis_thread = threading.Thread(target=worker, daemon=True)
        self._synthesis_thread.start()
    
    def _load_model(self):
        """加载Qwen3-TTS模型（优先使用本地路径，类似ASR模块）"""
        try:
            logger.info(f"正在加载Qwen3-TTS模型: {self.tts_config.model_name}...")
            from qwen_tts import Qwen3TTSModel
            
            # 加载模型（Qwen3-TTS支持本地路径和HuggingFace模型名称）
            # 如果配置了model_path，使用本地路径；否则使用模型名称自动下载
            if self.tts_config.model_path and self.tts_config.model_path.strip():
                model_path = self.tts_config.model_path.strip()
                logger.info(f"使用本地模型路径: {model_path}")
                
                # 🌟 修复：自动查找 HuggingFace 缓存目录中的 snapshots 子目录
                # HuggingFace 缓存结构：models--Qwen--Qwen3-TTS-12Hz-0.6B-Base/snapshots/<hash>/
                if os.path.isdir(model_path):
                    snapshots_dir = os.path.join(model_path, "snapshots")
                    if os.path.isdir(snapshots_dir):
                        # 查找 snapshots 下的第一个子目录（通常是 hash 命名的目录）
                        snapshot_dirs = [d for d in os.listdir(snapshots_dir) 
                                        if os.path.isdir(os.path.join(snapshots_dir, d))]
                        if snapshot_dirs:
                            # 使用第一个找到的 snapshots 目录
                            actual_model_path = os.path.join(snapshots_dir, snapshot_dirs[0])
                            logger.info(f"自动定位到 snapshots 目录: {actual_model_path}")
                            model_path = actual_model_path
                        else:
                            logger.warning(f"snapshots 目录为空，尝试使用根目录: {model_path}")
                    else:
                        # 如果已经是 snapshots 下的目录，直接使用
                        logger.info(f"使用提供的路径（可能是 snapshots 子目录）: {model_path}")
                
                # 检查路径是否存在，如果存在则强制离线模式
                is_local_dir = os.path.isdir(model_path)
                if not is_local_dir:
                    logger.error(f"模型路径不存在: {model_path}")
                    raise FileNotFoundError(f"模型路径不存在: {model_path}")
                
                logger.info(f"最终使用模型路径: {model_path} (local_files_only={is_local_dir})")
                
                self.tts = Qwen3TTSModel.from_pretrained(
                    model_path,
                    device_map="cuda:0",  # 强制GPU
                    local_files_only=is_local_dir  # 🌟 如果是本地目录，强制离线模式，绝不联网
                )
            else:
                # 使用模型名称，Qwen3-TTS会自动从HuggingFace下载
                logger.info(f"使用模型名称自动下载: {self.tts_config.model_name}")
                self.tts = Qwen3TTSModel.from_pretrained(
                    "Qwen/Qwen3-TTS-12Hz-0.6B-Base",
                    device_map="cuda:0"  # 强制GPU
                )
            
            logger.info("✅ Qwen3-TTS模型加载成功")
        except ImportError:
            logger.error("qwen-tts包未安装，请运行: pip install qwen-tts")
            raise
        except Exception as e:
            logger.error(f"Qwen3-TTS模型加载失败: {e}")
            raise
    
    def _init_zmq(self):
        """初始化ZMQ Sockets"""
        self.zmq_context = zmq.Context()
        
        # ZMQ PULL Socket（接收Core Server的合成文本）
        self.text_pull_socket = self.zmq_context.socket(zmq.PULL)
        self.text_pull_socket.setsockopt(zmq.RCVHWM, 50)  # 接收端限制积压
        self.text_pull_socket.bind(f"tcp://127.0.0.1:{self.zmq_config.tts_text_pull_port}")
        logger.info(f"ZMQ PULL Socket已绑定: tcp://127.0.0.1:{self.zmq_config.tts_text_pull_port}")
        
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
                        logger.warning("收到停止信号，设置停止标志位，并清空合成队列")
                        self._stop_event.set()  # 设置全局停止标志
                        self.char_buffer = ""
                        self._has_sent_first_sentence = False  # 重置首句标志
                        # 清空合成队列
                        while not self.synthesis_queue.empty():
                            try:
                                self.synthesis_queue.get_nowait()
                                self.synthesis_queue.task_done()
                            except queue.Empty:
                                break
                except zmq.Again:
                    time.sleep(0.01)
                    continue
                except Exception as e:
                    logger.error(f"停止信号监听出错: {e}")
                    time.sleep(0.1)
        
        thread = threading.Thread(target=listener, daemon=True)
        thread.start()
        logger.info("停止信号监听线程已启动")
    
    def _start_text_receiver(self):
        """启动文本接收线程（从ZMQ接收Core Server的合成文本）"""
        def receiver():
            logger.info("文本接收线程已启动")
            while self._running:
                try:
                    # 从ZMQ接收文本消息
                    message = self.text_pull_socket.recv_string(zmq.NOBLOCK)
                    try:
                        data = json.loads(message)
                        # 处理消息（按照unified-voice-ws-protocol.md格式）
                        msg_type = data.get("type")
                        if msg_type == "route":
                            # TTS合成请求
                            text = data.get("text", "")
                            is_final = data.get("isFinal", False)
                            if text or is_final:
                                self.process_streaming_text(text, is_final=is_final)
                        elif msg_type == "stop_tts":
                            # 停止合成
                            logger.warning("收到停止合成信号")
                            self._stop_event.set()
                    except json.JSONDecodeError:
                        logger.warning(f"无效的JSON消息: {message}")
                    except Exception as e:
                        logger.error(f"处理文本消息失败: {e}")
                except zmq.Again:
                    time.sleep(0.01)
                    continue
                except Exception as e:
                    logger.error(f"文本接收线程出错: {e}")
                    time.sleep(0.1)
        
        thread = threading.Thread(target=receiver, daemon=True)
        thread.start()
        logger.info("文本接收线程已启动")
    
    def process_streaming_text(self, text_chunk: str, is_final: bool = False):
        """
        处理流式文本，智能标点切分并放入合成队列（动态滑窗优化版）
        """
        self.char_buffer += text_chunk
        
        # 🌟 获取最小句子长度门限
        min_len = getattr(self.tts_config, 'min_sentence_length', 8)
        
        # 🌟 核心修复：使用 while 循环，因为一次推送可能包含多个符合条件的句子
        while True:
            # 强制切分门限，防止无限积攒
            force_cut = len(self.char_buffer) > 40
            
            # 根据是否发送过首句，选择不同的标点正则
            if not self._has_sent_first_sentence:
                pattern = self._first_sentence_pattern  # 包含逗号，追求首句破冰
            else:
                pattern = self._subsequent_sentence_pattern  # 仅限大标点，追求语气连贯
                
            matches = list(pattern.finditer(self.char_buffer))
            
            chunk_to_synth = None
            cut_idx = 0
            
            if matches:
                # 🌟 核心修复：遍历所有标点，像滚雪球一样寻找第一个能让句子长度 >= min_len 的点！
                for m in matches:
                    candidate_idx = m.end()
                    candidate_text = self.char_buffer[:candidate_idx]
                    clean_text = candidate_text.strip(" 。，！？.,!?\n\r\t")
                    
                    if len(clean_text) >= min_len:
                        chunk_to_synth = candidate_text
                        cut_idx = candidate_idx
                        break  # 找到了完美的切分点，跳出 for 循环
                
                # 如果所有标点后的长度都不够，但触发了强切防卡死，就在最后一个标点处切断
                if not chunk_to_synth and force_cut:
                    cut_idx = matches[-1].end()
                    chunk_to_synth = self.char_buffer[:cut_idx]

            # 兜底：全是文字没标点且过长，或收到最终结束标志
            if not chunk_to_synth:
                if force_cut or (is_final and self.char_buffer.strip()):
                    chunk_to_synth = self.char_buffer
                    cut_idx = len(self.char_buffer)

            # 执行切分并推入后台异步合成队列
            if chunk_to_synth:
                # 再次确认是否有实质内容
                clean_chunk = chunk_to_synth.strip(" 。，！？.,!?\n\r\t")
                if clean_chunk:
                    logger.info(f"🚀 [动态切分] 提取合成块: {chunk_to_synth.strip()} (内容长度: {len(clean_chunk)}字符)")
                    self.synthesis_queue.put_nowait((chunk_to_synth.strip(), False))
                    self._has_sent_first_sentence = True
                
                # 从缓冲区移除已处理部分，继续下一轮 while 检查
                self.char_buffer = self.char_buffer[cut_idx:]
            else:
                # 长度不够且未触发强切，退出 while 循环，等待大模型吐出更多文本
                break
                
        # 所有文本结算完毕，并且缓冲区彻底空了，发送整段结束标记
        if is_final and not self.char_buffer.strip():
            self._has_sent_first_sentence = False
            self.char_buffer = "" 
            self.synthesis_queue.put_nowait(("END_OF_TTS_SESSION", True))
    
    def synthesize_with_cutoff(self, text: str):
        """
        带熔断机制的合成方法（后台异步执行，不阻塞文本接收）
        
        🌟 核心优化：利用播放时间掩盖推理时间
        - 第一句合成时，用户等待（但第一句通常很短，合成快）
        - 第一句播放时（1-2秒），后台已经在合成第二句
        - 第一句播放完，第二句音频已经准备好，实现无缝衔接
        
        Args:
            text: 要合成的文本
        """
        if not text or len(text.strip()) == 0:
            return
        
        self._stop_event.clear()  # 重置停止标志
        
        try:
            # 🌟 记录合成开始时间（用于性能分析）
            synthesis_start = time.time()
            
            # generate_voice_clone 返回 (wavs, sr) 元组，不是生成器
            # wavs 是音频数组列表，sr 是采样率
            wavs, sr = self.tts.generate_voice_clone(
                text=text,
                ref_audio=self.ref_audio_path,
                x_vector_only_mode=True  # 声纹抽取模式，不需要ref_text
            )
            
            # 🌟 记录合成耗时
            synthesis_time = time.time() - synthesis_start
            logger.debug(f"⏱️ TTS合成耗时: {synthesis_time:.2f}s, 文本长度: {len(text)}字符, 文本: {text[:30]}...")
            
            # 检查停止标志（在合成完成后检查）
            if self._stop_event.is_set():
                logger.warning(f"检测到停止标志，丢弃合成结果: {text[:20]}...")
                return  # 直接返回，不发送音频
            
            # 提取音频数据（wavs 是列表，取第一个元素）
            if wavs and len(wavs) > 0:
                audio_chunk = wavs[0]
                # 发送音频块到ZMQ
                self._send_audio_chunk(audio_chunk)
                logger.debug(f"✅ 音频已发送: {len(text)}字符 -> {len(audio_chunk)}采样点")
            else:
                logger.warning(f"TTS合成结果为空: {text[:20]}...")
        
        except Exception as e:
            logger.error(f"TTS合成失败: {e}")
        
        finally:
            # 如果被中断，清空字符缓冲区和首句标志
            if self._stop_event.is_set():
                self.char_buffer = ""
                self._has_sent_first_sentence = False
    
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
            
    def _send_tts_end_marker(self):
        """发送整段TTS结束标记"""
        try:
            metadata = {
                "type": "tts_end",
                "timestamp": time.time()
            }
            # 发送空的二进制数据
            empty_audio = b""
            self.push_socket.send_multipart([
                json.dumps(metadata).encode('utf-8'),
                empty_audio
            ], zmq.NOBLOCK)
            logger.info("✅ 已发送整段TTS结束标记给Core Server")
        except Exception as e:
            logger.error(f"发送TTS结束标记异常: {e}")
    
    def start(self):
        """启动TTS模块"""
        self._running = True
        logger.info("TTS模块已启动")
        
        # 启动停止信号监听线程
        self._start_stop_signal_listener()
        
        # 启动文本接收线程
        self._start_text_receiver()
        
        # 启动合成工作线程
        self._start_synthesis_worker()
    
    def stop(self):
        """停止TTS模块"""
        self._running = False
        self._stop_event.set()  # 设置停止标志
        
        # 关闭ZMQ sockets
        if self.text_pull_socket:
            self.text_pull_socket.close()
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
    # 加载配置：显式指定项目根目录下的 config.yaml，确保读取到自定义的 model_path
    project_root = Path(__file__).resolve().parents[2]
    config_path = project_root / "config" / "config.yaml"
    config = get_config(str(config_path))
    
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
