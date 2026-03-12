"""
配置管理模块
支持从YAML文件加载配置，支持运行时动态调整
"""

import yaml
from pathlib import Path
from typing import Optional
from wakefusion.types import AppConfig


class ConfigManager:
    """配置管理器"""

    def __init__(self, config_path: Optional[str] = None):
        """
        初始化配置管理器

        Args:
            config_path: 配置文件路径（YAML格式）
        """
        self.config_path = config_path
        self._config: Optional[AppConfig] = None

    def load(self, config_path: Optional[str] = None) -> AppConfig:
        """
        加载配置

        Args:
            config_path: 配置文件路径，如果为None则使用初始化时的路径

        Returns:
            AppConfig: 应用配置对象
        """
        if config_path:
            self.config_path = config_path

        if self.config_path and Path(self.config_path).exists():
            # 加载YAML文件
            with open(self.config_path, 'r', encoding='utf-8') as f:
                config_dict = yaml.safe_load(f) or {}
            
            # 使用 Pydantic 的 model_validate 方法，确保 YAML 中的值能够深度覆盖默认值
            # 如果 YAML 中缺失某个字段，会使用 types.py 中的默认值
            self._config = AppConfig.model_validate(config_dict)
        else:
            # 使用默认配置（完全使用 types.py 中的默认值）
            self._config = AppConfig()

        return self._config

    @property
    def config(self) -> AppConfig:
        """获取当前配置（懒加载）"""
        if self._config is None:
            self._config = self.load()
        return self._config

    def save(self, path: Optional[str] = None):
        """
        保存配置到文件

        Args:
            path: 保存路径，如果为None则使用当前config_path
        """
        save_path = path or self.config_path
        if not save_path:
            raise ValueError("No config path specified")

        with open(save_path, 'w', encoding='utf-8') as f:
            yaml.dump(
                self.config.model_dump(),
                f,
                default_flow_style=False,
                allow_unicode=True
            )

    def update_kws_threshold(self, threshold: float):
        """更新KWS阈值"""
        self._config.kws.threshold = threshold

    def update_vad_threshold(self, speech_start_ms: int, speech_end_ms: int):
        """更新VAD阈值"""
        self._config.vad.speech_start_ms = speech_start_ms
        self._config.vad.speech_end_ms = speech_end_ms

    def update_vision_distance(self, distance_m_max: float):
        """更新视觉最大检测距离"""
        self._config.vision.distance_m_max = distance_m_max

    def get_zmq_config(self):
        """获取ZMQ配置"""
        return self.config.zmq

    def get_vision_wake_config(self):
        """获取视觉唤醒配置"""
        return self.config.vision_wake

    def get_audio_threshold_config(self):
        """获取音频阈值配置"""
        return self.config.audio_threshold

    def get_conversation_config(self):
        """获取持续对话配置"""
        return self.config.conversation

    def get_environments_config(self):
        """获取环境配置"""
        return self.config.environments
    
    def get_asr_config(self):
        """获取ASR配置"""
        return self.config.asr
    
    def get_tts_config(self):
        """获取TTS配置"""
        return self.config.tts
    
    def get_websocket_config(self):
        """获取WebSocket配置"""
        return self.config.websocket
    
    def get_llm_agent_config(self):
        """获取LLM Agent配置"""
        return self.config.llm_agent
    
    def get_vad_config(self):
        """获取VAD配置"""
        return self.config.vad


# 全局配置实例
_global_config_manager: Optional[ConfigManager] = None


def get_config(config_path: Optional[str] = None) -> AppConfig:
    """
    获取全局配置实例

    Args:
        config_path: 配置文件路径

    Returns:
        AppConfig: 应用配置对象
    """
    global _global_config_manager

    if _global_config_manager is None:
        _global_config_manager = ConfigManager(config_path)

    if config_path:
        _global_config_manager.load(config_path)

    return _global_config_manager.config


def set_config_manager(manager: ConfigManager):
    """设置全局配置管理器（用于测试）"""
    global _global_config_manager
    _global_config_manager = manager
