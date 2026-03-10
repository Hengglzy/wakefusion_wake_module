"""
VAD引擎模块 - 独立的语音端点检测引擎
支持多种VAD引擎，使用面向对象设计实现代码级解耦
"""
import os
import torch
import numpy as np
from typing import Optional
import logging

logger = logging.getLogger(__name__)


class SileroVADEngine:
    """独立的 VAD 引擎类，完全与网络和麦克风解耦
    
    使用Silero VAD（深度学习模型）进行智能语音端点检测，
    能够区分人声和环境噪音（风声、衣服摩擦等）。
    
    设计原则：
    - CPU推理：强制使用CPU，避免占用GPU资源
    - 完全解耦：与网络发送、麦克风采集逻辑分离
    - 组合模式：可以被AudioService组合使用，易于替换
    """
    
    def __init__(self, threshold: float = 0.5, sample_rate: int = 16000):
        """
        初始化 Silero VAD 引擎
        
        Args:
            threshold: 语音概率阈值（0.0-1.0），默认0.5
            sample_rate: 音频采样率（默认16kHz）
        """
        self.threshold = threshold
        self.sample_rate = sample_rate
        
        # 音频缓冲区：用于拼接残留音频，保证波形绝对连续
        self._audio_buffer = np.array([], dtype=np.int16)
        
        logger.info(f"正在加载Silero VAD模型（v4，16kHz，CPU推理）...")
        
        # 强制将 PyTorch Hub 缓存路径设为纯英文路径，避免中文路径导致的 errno 42 错误
        hub_cache_dir = "D:/AI_Cache/torch"
        os.makedirs(hub_cache_dir, exist_ok=True)
        torch.hub.set_dir(hub_cache_dir)
        logger.info(f"PyTorch Hub 缓存目录已设置为: {hub_cache_dir}")
        
        # 加载 Silero VAD（v4，16kHz，CPU推理）
        try:
            self.model, self.utils = torch.hub.load(
                repo_or_dir='snakers4/silero-vad',
                model='silero_vad',
                force_reload=False,
                onnx=False,
                trust_repo=True  # 明确添加此参数绕过安全提示
            )
            # 强制CPU推理，避免占用GPU资源
            self.model.to(torch.device('cpu'))
            self.model.eval()
            logger.info(f"Silero VAD模型加载成功（阈值={threshold}，采样率={sample_rate}Hz）")
        except Exception as e:
            logger.error(f"Silero VAD模型加载失败: {e}")
            raise
    
    def is_speech(self, audio_chunk: np.ndarray) -> bool:
        """
        检测音频块是否包含人声
        
        Args:
            audio_chunk: 音频数据（int16格式的numpy数组，长度可变）
        
        Returns:
            bool: True表示检测到人声，False表示静音或噪音
        
        注意：
            - 使用音频缓冲区拼接残留音频，保证波形绝对连续，避免波形断裂产生爆音
            - 此方法不会重置RNN状态，以保持时间连续性
            - 必须完整处理所有音频块，确保RNN状态正确更新
        """
        try:
            # 1. 拼接上次残留的音频，保证波形绝对连续
            self._audio_buffer = np.concatenate((self._audio_buffer, audio_chunk))
            
            window_size = 512  # Silero VAD v4 在 16kHz 下要求的固定窗口大小
            # 2. 计算可以处理的完整窗口数
            valid_len = (len(self._audio_buffer) // window_size) * window_size
            
            if valid_len == 0:
                return False
                
            # 3. 提取整数倍的音频，把除不尽的余数留给下一次
            chunk_to_process = self._audio_buffer[:valid_len]
            self._audio_buffer = self._audio_buffer[valid_len:]
            
            # 4. 转换为tensor并归一化
            audio_tensor = torch.from_numpy(chunk_to_process).float() / 32768.0
            
            has_speech = False
            with torch.no_grad():
                for i in range(0, len(audio_tensor), window_size):
                    chunk = audio_tensor[i : i + window_size]
                    speech_prob = self.model(chunk, self.sample_rate).item()
                    if speech_prob >= self.threshold:
                        has_speech = True  # 只要有任何一帧是语音，就判定为True
                        
            return has_speech
        except Exception as e:
            logger.error(f"Silero VAD推理失败: {e}")
            return False
    
    def update_threshold(self, threshold: float):
        """
        动态更新阈值
        
        Args:
            threshold: 新的语音概率阈值（0.0-1.0）
        """
        if 0.0 <= threshold <= 1.0:
            self.threshold = threshold
            logger.info(f"VAD阈值已更新: {threshold}")
        else:
            logger.warning(f"无效的VAD阈值: {threshold}，应在[0.0, 1.0]范围内")
    
    def get_threshold(self) -> float:
        """获取当前阈值"""
        return self.threshold
    
    def reset_states(self):
        """
        重置RNN状态和残余缓存
        
        注意：正常情况下不需要调用此方法，因为is_speech()会保持状态连续性。
        只有在需要强制重置状态时（如切换音频源、长时间静音后、唤醒成功推流前）才调用。
        """
        self.model.reset_states()
        self._audio_buffer = np.array([], dtype=np.int16)
        logger.debug("VAD RNN状态及音频缝合缓存已重置")