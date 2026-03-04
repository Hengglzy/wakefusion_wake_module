"""
RNNoise 降噪服务
独立的音频降噪模块，可在配置中开启或关闭
支持降级：如果 RNNoise 不可用，自动降级为直通模式
"""

import numpy as np
from typing import Optional
from wakefusion.logging import get_logger
from wakefusion.metrics import get_metrics, record_latency

logger = get_logger("rnnoise_service")
metrics = get_metrics()


class RNNoiseService:
    """
    RNNoise 降噪服务
    
    功能：
    - 对 48kHz 音频进行降噪处理
    - 支持开启/关闭
    - 自动降级：如果 pyrnnoise 不可用，自动降级为直通模式
    """
    
    def __init__(self, enabled: bool = False, sample_rate: int = 48000):
        """
        初始化 RNNoise 服务
        
        Args:
            enabled: 是否启用降噪
            sample_rate: 音频采样率（必须为 48000，RNNoise 要求）
        """
        self.enabled = enabled
        self.sample_rate = sample_rate
        
        # RNNoise 要求采样率为 48000 Hz
        if sample_rate != 48000:
            logger.warning(
                f"RNNoise requires 48kHz, got {sample_rate}Hz. Denoising disabled."
            )
            self.enabled = False
            self.denoiser = None
            self.available = False
            return
        
        # 尝试导入 pyrnnoise
        self.denoiser = None
        self.available = False
        
        if self.enabled:
            try:
                import pyrnnoise
                self.denoiser = pyrnnoise.RNNoise()
                self.available = True
                logger.info("RNNoise service initialized successfully")
            except ImportError:
                logger.warning(
                    "pyrnnoise not available. RNNoise service disabled. "
                    "Install with: pip install pyrnnoise"
                )
                self.available = False
                self.enabled = False
            except Exception as e:
                logger.error(f"Failed to initialize RNNoise: {e}")
                self.available = False
                self.enabled = False
        else:
            logger.info("RNNoise service disabled by configuration")
    
    def process(self, pcm16: np.ndarray) -> np.ndarray:
        """
        处理音频数据（降噪）
        
        Args:
            pcm16: 输入 PCM 数据（int16，48kHz，单声道）
            
        Returns:
            np.ndarray: 降噪后的 PCM 数据（int16）
        """
        # 如果未启用或不可用，直接返回原始数据（直通模式）
        if not self.enabled or not self.available or self.denoiser is None:
            return pcm16
        
        import time
        start_time = time.perf_counter()
        
        try:
            # RNNoise 处理需要 float32 格式（范围 -1.0 到 1.0）
            # 将 int16 转换为 float32
            audio_float = pcm16.astype(np.float32) / 32768.0
            
            # 应用降噪（pyrnnoise 的 process 方法）
            denoised_float = self.denoiser.process(audio_float)
            
            # 转换回 int16
            denoised_int16 = (denoised_float * 32768.0).astype(np.int16)
            
            # 记录延迟
            latency_ms = (time.perf_counter() - start_time) * 1000
            record_latency("rnnoise.processing_latency_ms", latency_ms)
            metrics.increment_counter("rnnoise.frames_processed")
            
            return denoised_int16
            
        except Exception as e:
            logger.error(f"RNNoise processing error: {e}")
            metrics.increment_counter("rnnoise.processing_errors")
            # 降级：返回原始数据
            return pcm16
    
    def is_available(self) -> bool:
        """
        检查 RNNoise 是否可用
        
        Returns:
            bool: 如果可用返回 True
        """
        return self.available and self.denoiser is not None
    
    def enable(self):
        """启用降噪服务"""
        if not self.available:
            logger.warning("Cannot enable RNNoise: service not available")
            return
        self.enabled = True
        logger.info("RNNoise service enabled")
    
    def disable(self):
        """禁用降噪服务（降级为直通模式）"""
        self.enabled = False
        logger.info("RNNoise service disabled (passthrough mode)")
    
    def __del__(self):
        """清理资源"""
        if self.denoiser is not None:
            try:
                # pyrnnoise 可能不需要显式清理，但为了安全起见
                del self.denoiser
            except Exception:
                pass
