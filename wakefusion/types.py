"""
数据模型定义 - 使用pydantic进行验证
定义所有事件、帧、配置的数据结构
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field


# ============================================================================
# 枚举类型
# ============================================================================

class SystemState(str, Enum):
    """系统状态枚举"""
    IDLE = "IDLE"           # 空闲状态
    LISTENING = "LISTENING"  # 监听状态
    SPEAKING = "SPEAKING"    # 数字人播报状态


class EventType(str, Enum):
    """事件类型枚举"""
    # 音频相关
    SPEECH_START = "SPEECH_START"
    SPEECH_END = "SPEECH_END"
    KWS_HIT = "KWS_HIT"

    # 视觉相关
    PRESENCE = "PRESENCE"
    VALID_USER = "VALID_USER"

    # 决策相关
    WAKE_CONFIRMED = "WAKE_CONFIRMED"
    WAKE_REJECTED = "WAKE_REJECTED"
    WAKE_PROBATION = "WAKE_PROBATION"
    BARGE_IN = "BARGE_IN"

    # 系统相关
    HEALTH = "HEALTH"
    ERROR = "ERROR"


# ============================================================================
# 音频帧模型
# ============================================================================

@dataclass
class AudioFrame:
    """音频帧数据"""
    ts: float                          # 时间戳（单调时钟）
    pcm16: np.ndarray                  # 16-bit PCM数据 (16kHz mono)
    sample_rate: int = 16000           # 采样率
    rms: Optional[float] = None        # RMS能量
    peak: Optional[float] = None       # 峰值

    def __post_init__(self):
        """计算RMS和峰值"""
        if self.rms is None:
            # 避免除零
            if len(self.pcm16) > 0:
                self.rms = np.sqrt(np.mean(self.pcm16.astype(np.float32) ** 2))
            else:
                self.rms = 0.0

        if self.peak is None:
            if len(self.pcm16) > 0:
                self.peak = np.max(np.abs(self.pcm16.astype(np.float32)))
            else:
                self.peak = 0.0


@dataclass
class AudioFrameRaw:
    """原始音频帧（采集采样率）"""
    ts: float                          # 时间戳
    pcm16: np.ndarray                  # 原始PCM数据
    sample_rate: int = 48000           # 原始采样率
    channels: int = 1                  # 声道数


# ============================================================================
# 视觉帧模型
# ============================================================================

@dataclass
class VisionFrame:
    """视觉帧数据"""
    ts: float                          # 时间戳
    rgb: Optional[np.ndarray] = None   # RGB图像 (H,W,3)
    depth: Optional[np.ndarray] = None # 深度图 (H,W)
    presence: bool = False             # 是否检测到人
    faces: List[Dict[str, Any]] = field(default_factory=list)  # 检测到的人脸
    distance_m: Optional[float] = None  # 估计距离（米）
    confidence: float = 0.0            # 检测置信度


# ============================================================================
# 事件模型
# ============================================================================

class BaseEvent(BaseModel):
    """事件基类"""
    type: EventType
    ts: float = Field(default_factory=lambda: datetime.now().timestamp())
    session_id: str
    priority: int = 50                 # 优先级 (0-100, 越高越优先)


class KWSHitPayload(BaseModel):
    """KWS命中事件载荷"""
    keyword: str
    confidence: float                  # 置信度 (0-1)
    pre_roll_ms: int = 800             # 回捞时长
    audio_start_ts: float              # 回捞起始时间戳
    audio_end_ts: float                # 回捞结束时间戳


class WakeConfirmedPayload(BaseModel):
    """唤醒确认事件载荷"""
    keyword: str
    confidence: float
    pre_roll_ms: int
    vision_gate: bool = False          # 视觉门控是否通过
    vision_confidence: float = 0.0     # 视觉置信度
    distance_m: Optional[float] = None  # 用户距离


class HealthPayload(BaseModel):
    """健康检查事件载荷"""
    audio_fps: float
    audio_latency_ms: float
    kws_hit_count: int
    vad_speech_segments: int
    vision_fps: Optional[float] = None
    device_status: Dict[str, str]      # 设备状态
    cpu_percent: float
    memory_mb: float


# ============================================================================
# 唤醒上下文
# ============================================================================

@dataclass
class WakeContext:
    """唤醒上下文（用于回捞音频）"""
    keyword: str
    confidence: float
    start_ts: float                    # 回捞起始时间戳
    end_ts: float                      # 回捞结束时间戳
    pre_roll_ms: int = 800
    vision_verified: bool = False
    vision_distance: Optional[float] = None


# ============================================================================
# 配置模型
# ============================================================================

class AudioConfig(BaseModel):
    """音频配置"""
    device_match: str = "XVF3800"      # 设备匹配名称
    capture_sample_rate: int = 48000   # 采集采样率
    work_sample_rate: int = 16000      # 工作采样率
    frame_ms: int = 20                 # 帧长（毫秒）
    ring_buffer_sec: float = 2.0       # Ring buffer长度（秒）
    pre_roll_ms: int = 800             # Pre-roll时长（毫秒）
    channels: int = 1                  # 声道数


class KWSConfig(BaseModel):
    """KWS配置"""
    enabled: bool = True
    model: str = "openwakeword"        # 模型类型
    keyword: str = "hey_assistant"     # 唤醒词
    threshold: float = 0.55            # 检测阈值
    cooldown_ms: int = 1200            # 冷却时长（毫秒）


class VADConfig(BaseModel):
    """VAD配置"""
    enabled: bool = True
    model: str = "webrtcvad"
    speech_start_ms: int = 120         # 语音起始阈值（毫秒）
    speech_end_ms: int = 500           # 语音结束阈值（毫秒）


class VisionConfig(BaseModel):
    """视觉配置"""
    enabled: bool = False              # Phase 1暂不启用
    gate_on_kws_only: bool = True
    cache_ms: int = 600                # 缓存时长（毫秒）
    distance_m_max: float = 4.0        # 最大检测距离（米）
    face_conf_min: float = 0.55        # 最小人脸置信度


class FusionConfig(BaseModel):
    """融合策略配置"""
    probation_enabled: bool = True     # 是否启用降级策略
    probation_ms: int = 1000           # 降级窗口（毫秒）
    barge_in_enabled: bool = True      # 是否启用打断


class RuntimeConfig(BaseModel):
    """运行时配置"""
    health_interval_sec: int = 2       # 健康检查间隔（秒）
    log_level: str = "INFO"
    websocket_port: int = 8765         # WebSocket端口
    health_port: int = 8080            # 健康检查端口


class AppConfig(BaseModel):
    """应用总配置"""
    audio: AudioConfig = Field(default_factory=AudioConfig)
    kws: KWSConfig = Field(default_factory=KWSConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)


# ============================================================================
# 控制命令
# ============================================================================

class SetSystemStateCommand(BaseModel):
    """设置系统状态命令"""
    state: SystemState


class SetPolicyCommand(BaseModel):
    """动态调整策略命令"""
    kws_threshold: Optional[float] = None
    vad_speech_start_ms: Optional[int] = None
    vision_distance_m_max: Optional[float] = None
