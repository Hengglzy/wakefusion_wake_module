"""
训练配置模型
定义所有训练相关的数据结构
"""

from typing import List, Optional, Dict, Any
from dataclasses import dataclass, field
from pathlib import Path
from pydantic import BaseModel, Field
from enum import Enum


class TTSVoice(str, Enum):
    """TTS音色枚举"""
    XIAOXIAO = "zh-CN-XiaoxiaoNeural"  # 女声
    YUNXI = "zh-CN-YunxiNeural"        # 男声
    YUNYANG = "zh-CN-YunyangNeural"    # 男声
    XIAOYI = "zh-CN-XiaoyiNeural"      # 女声
    YUNJIAN = "zh-CN-YunjianNeural"    # 男声


class AugmentationType(str, Enum):
    """数据增强类型"""
    SPEED = "speed"
    PITCH = "pitch"
    NOISE = "noise"
    VOLUME = "volume"


class TTSConfig(BaseModel):
    """TTS配置"""
    voices: List[TTSVoice] = Field(
        default=[TTSVoice.XIAOXIAO, TTSVoice.YUNXI],
        description="使用的TTS音色列表"
    )
    samples_per_voice: int = Field(
        default=50,
        ge=10, le=200,
        description="每个音色生成的样本数"
    )
    output_format: str = Field(
        default="wav",
        description="输出音频格式"
    )
    sample_rate: int = Field(
        default=16000,
        description="采样率"
    )


class DataAugmentationConfig(BaseModel):
    """数据增强配置"""
    enabled: bool = True
    speed_perturbation: bool = True
    speed_rates: List[float] = Field(default=[0.9, 1.0, 1.1])
    gain_perturbation: bool = True
    gain_range: List[float] = Field(default=[-5, 5])
    noise_injection: bool = True
    noise_snr_range: List[float] = Field(default=[10, 30])


class NeMoTrainingConfig(BaseModel):
    """NeMo训练配置"""
    # 模型架构
    num_blocks: int = Field(default=3, ge=1, le=5)
    num_layers_per_block: int = Field(default=1, ge=1, le=3)
    channels: int = Field(default=64, ge=32, le=128)

    # 训练参数
    batch_size: int = Field(default=32, ge=8, le=128)
    num_epochs: int = Field(default=50, ge=10, le=200)
    learning_rate: float = Field(default=0.001, ge=0.0001, le=0.01)
    weight_decay: float = Field(default=0.0001)

    # 优化器和调度器
    optimizer: str = Field(default="adam")
    scheduler: str = Field(default="CosineAnnealing")

    # 数据
    train_manifest: Optional[str] = None
    val_manifest: Optional[str] = None
    num_workers: int = Field(default=4)

    # Checkpoint
    save_top_k: int = Field(default=3)
    monitor_metric: str = Field(default="val_accuracy")


class ValidationConfig(BaseModel):
    """验证配置"""
    test_samples: int = Field(default=20, ge=5, le=100)
    threshold_range: List[float] = Field(default=[0.3, 0.9])
    threshold_step: float = Field(default=0.05)
    calculate_latency: bool = True
    generate_confusion_matrix: bool = True


class TrainingConfig(BaseModel):
    """训练总配置"""
    # 基础信息
    wake_word: str = Field(..., description="唤醒词文本")
    label_name: str = Field(..., description="模型标签名（英文）")

    # 路径配置
    output_dir: str = Field(default="training_output")
    data_dir: str = Field(default="data")
    checkpoint_dir: str = Field(default="checkpoints")
    model_dir: str = Field(default="models")

    # 子配置
    tts: TTSConfig = Field(default_factory=TTSConfig)
    augmentation: DataAugmentationConfig = Field(default_factory=DataAugmentationConfig)
    training: NeMoTrainingConfig = Field(default_factory=NeMoTrainingConfig)
    validation: ValidationConfig = Field(default_factory=ValidationConfig)

    # 设备配置
    device: str = Field(default="cpu")  # cpu 或 cuda
    mixed_precision: bool = Field(default=False)

    # 负样本（可选）
    negative_samples: List[str] = Field(
        default_factory=list,
        description="负样本列表（相似词、干扰词）"
    )


@dataclass
class TrainingResult:
    """训练结果"""
    success: bool
    wake_word: str
    label_name: str

    # 路径信息
    nemo_model_path: Optional[str] = None
    onnx_model_path: Optional[str] = None
    config_path: Optional[str] = None
    report_path: Optional[str] = None

    # 训练指标
    train_accuracy: float = 0.0
    val_accuracy: float = 0.0
    best_epoch: int = 0

    # 验证指标
    test_accuracy: float = 0.0
    test_recall: float = 0.0
    test_precision: float = 0.0
    avg_latency_ms: float = 0.0
    recommended_threshold: float = 0.5

    # 元数据
    training_time_sec: float = 0.0
    total_samples: int = 0
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """转换为字典"""
        return {
            "success": self.success,
            "wake_word": self.wake_word,
            "label_name": self.label_name,
            "nemo_model_path": self.nemo_model_path,
            "onnx_model_path": self.onnx_model_path,
            "config_path": self.config_path,
            "train_accuracy": self.train_accuracy,
            "val_accuracy": self.val_accuracy,
            "test_accuracy": self.test_accuracy,
            "test_recall": self.test_recall,
            "test_precision": self.test_precision,
            "avg_latency_ms": self.avg_latency_ms,
            "recommended_threshold": self.recommended_threshold,
            "training_time_sec": self.training_time_sec,
            "total_samples": self.total_samples,
            "error_message": self.error_message
        }
