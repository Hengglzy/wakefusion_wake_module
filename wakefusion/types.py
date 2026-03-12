"""
数据模型定义 - 使用pydantic进行验证
定义所有事件、帧、配置的数据结构
"""

from typing import Optional, Dict, Any, List
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
import numpy as np
from pydantic import BaseModel, Field, ConfigDict


# ============================================================================
# 枚举类型
# ============================================================================

class SystemState(str, Enum):
    """系统状态枚举"""
    IDLE = "IDLE"           # 空闲状态
    VISUAL_WAKE = "VISUAL_WAKE"  # 视觉唤醒状态
    LISTENING = "LISTENING"  # 监听状态
    PROCESSING = "PROCESSING"  # 处理状态（未来用）
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
    depth: Optional[np.ndarray] = None # 深度图 (H,W) - 原始16位深度数据，用于距离计算
    color_depth: Optional[np.ndarray] = None  # 上色后的深度图 (H,W,3) - BGR格式，用于显示
    presence: bool = False             # 是否检测到人
    faces: List[Dict[str, Any]] = field(default_factory=list)  # 检测到的人脸
    distance_m: Optional[float] = None  # 估计距离（米）
    confidence: float = 0.0            # 检测置信度
    gesture: Optional[str] = None      # 手势类型（thumbs_up, ok, waving, fist）- 向后兼容，优先使用 hands
    hand_center: Optional[tuple] = None  # 手部中心坐标（归一化，0.0-1.0）- 向后兼容，优先使用 hands
    hand_distance_m: Optional[float] = None  # 手部距离（米）- 向后兼容，优先使用 hands
    hands: List[Dict[str, Any]] = field(default_factory=list)  # 多手列表，每个元素包含：index, gesture, x, y, distance_m


# ============================================================================
# 事件模型
# ============================================================================

class BaseEvent(BaseModel):
    """事件基类"""
    model_config = ConfigDict(extra='allow')  # 允许额外字段（如 payload）
    
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
    rnnoise_enabled: bool = False      # RNNoise降噪开关


class KWSConfig(BaseModel):
    """KWS配置"""
    enabled: bool = True
    model: str = "openwakeword"        # 模型类型
    keyword: str = "hey_assistant"     # 唤醒词
    threshold: float = 0.55            # 检测阈值
    cooldown_ms: int = 1200            # 冷却时长（毫秒）


class VADConfig(BaseModel):
    """VAD配置（支持Silero VAD和WebRTC VAD）"""
    enabled: bool = True
    engine: str = "silero"  # VAD引擎："silero" 或 "webrtcvad"
    model_version: Optional[str] = "v4"  # Silero VAD版本
    model_name: Optional[str] = "silero_vad"  # Silero VAD模型名称（16kHz）
    device: str = "cpu"  # 推理设备：强制CPU，避免占用GPU资源
    threshold: float = 0.5  # 语音概率阈值（0.0-1.0），仅用于Silero VAD
    sample_rate: int = 16000  # 采样率
    # 以下字段保留用于向后兼容（已废弃）
    model: Optional[str] = "webrtcvad"  # 已废弃，由engine替代
    speech_start_ms: Optional[int] = 120  # 语音起始阈值（毫秒）- 已废弃
    speech_end_ms: Optional[int] = 500  # 语音结束阈值（毫秒）- 已废弃


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
    processing_timeout_sec: int = 60   # PROCESSING状态最大等待时间（秒）


class ZMQConfig(BaseModel):
    """ZMQ通信配置"""
    vision_pub_port: int = 5555       # 视觉数据发布端口
    audio_pub_port: int = 5556        # 音频数据发布端口
    audio_ctrl_port: int = 5557       # 音频控制端口（REQ-REP）
    req_rep_timeout_ms: int = 2000    # REQ-REP超时时间（毫秒）
    asr_pull_port: int = 5558         # ASR模块PULL端口（Core Server PUSH音频到此端口）
    asr_result_push_port: int = 5562  # ASR模块PUSH端口（ASR推送识别结果给Core Server）
    tts_text_pull_port: int = 5563   # TTS模块PULL端口（TTS接收Core Server的合成文本）
    tts_push_port: int = 5559         # TTS模块PUSH端口（Core Server PULL从此端口接收音频）
    tts_stop_pub_port: int = 5560     # TTS停止信号PUB端口（Core Server发布，TTS订阅）
    core_control_rep_port: int = 5561  # Core Server控制端口（REP，接收LLM指令）


class VisionWakeConfig(BaseModel):
    """视觉唤醒配置"""
    detection_distance_m: float = 3.0  # 3米内开始检测
    frontal_percent_threshold: int = 75  # 正面率阈值（%）
    distance_range: List[float] = Field(default_factory=lambda: [0.1, 3.5])  # 有效距离区间（米）
    leave_timeout_sec: float = 1.5  # 离开区间持续多久才判定结束（秒）
    leave_check_frames: int = 20  # 连续N帧不在区间内才判定离开
    visual_cutoff_enabled: bool = True  # 启用视觉斩断机制（最高优先级打断）


class AudioThresholdConfig(BaseModel):
    """音频动态阈值配置"""
    default: float = 0.85  # 默认高阈值（无人时）
    visual_wake: float = 0.6  # 视觉唤醒后的低阈值
    change_timeout_ms: int = 100  # 阈值修改指令超时（毫秒）


class ConversationConfig(BaseModel):
    """持续对话配置"""
    vad_silence_timeout_default_sec: float = 3.0  # 默认免唤醒窗口（秒）
    vad_silence_timeout_extended_sec: float = 15.0  # 未来用于疑问句延长的窗口（秒）
    vad_check_interval_ms: int = 200  # VAD检查间隔（毫秒）
    vad_rms_threshold: float = 0.003  # VAD RMS阈值（用于简单VAD检测，低于此值视为静音）


class ASRConfig(BaseModel):
    """ASR配置"""
    enabled: bool = True
    engine: str = "funasr"  # ASR引擎
    model_name: str = "iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"  # FunASR模型名称（在线推理模型，完整repo ID）
    model_path: Optional[str] = "D:/AI_Cache/modelscope/models/iic/speech_paraformer-large_asr_nat-zh-cn-16k-common-vocab8404-online"  # FunASR模型路径（本地路径，避免联网下载）
    sample_rate: int = 16000  # 采样率
    chunk_size_ms: int = 100  # 流式识别块大小（毫秒）
    enable_online_inference: bool = True  # 启用在线推理（真正的流式识别）
    enable_partial_results: bool = True  # 启用中间结果推送（partial_text）
    use_cache: bool = True  # 启用cache机制（必须启用，否则流式识别会崩溃）
    timeout_sec: int = 30  # 超时时间（秒）


class TTSConfig(BaseModel):
    """TTS配置（Qwen3-TTS-12Hz-0.6B-Base）"""
    enabled: bool = True
    engine: str = "qwen3-tts"  # TTS引擎
    model_name: str = "Qwen3-TTS-12Hz-0.6B-Base"
    model_path: Optional[str] = ""  # TTS模型路径（本地路径，避免联网下载，留空则使用model_name自动下载）
    ref_audio_path: str = "D:/tools/cursor_project/wakefusion_wake_module/real_audio/recording_0001.wav"  # Voice Clone参考音频（必需）
    sample_rate: int = 24000  # 采样率（Qwen3-TTS输出为24000Hz）
    speed: float = 1.0  # 默认语速
    chunk_size_ms: int = 20  # 音频块大小（毫秒）
    enable_streaming: bool = True  # 是否启用流式输出
    punctuation_pattern: str = "[。！？.!?，,]"  # 标点切分正则表达式
    min_sentence_length: int = 3  # 最小句子长度（字符数），避免过短片段
    warmup_text: str = "系统初始化"  # 冷启动预热文本
    warmup_enabled: bool = True  # 是否启用冷启动预热


class LLMAgentConfig(BaseModel):
    """LLM Agent配置（统一WebSocket协议）"""
    host: str = "127.0.0.1:8080"  # LLM Agent服务地址（格式：host:port）
    device_id: str = "wakefusion-device-01"  # 设备标识
    token: str = "your-token-here"  # 认证令牌
    use_ssl: bool = False  # 是否使用SSL（true for wss://, false for ws://）
    reconnect_interval_sec: float = 5.0  # 断线重连间隔（秒）
    ping_interval_sec: float = 30.0  # 保活ping间隔（秒）
    
    # 火山引擎API配置（可选，如果使用火山引擎LLM Agent）
    volcano_api_url: Optional[str] = None  # 火山引擎API地址
    volcano_api_key: Optional[str] = None  # 火山引擎API密钥
    volcano_model: Optional[str] = None  # 火山引擎模型名称


class WebSocketConfig(BaseModel):
    """WebSocket配置（已废弃，保留用于向后兼容）"""
    asr_port: int = 8766  # 已废弃：ASR WebSocket端口
    tts_port: int = 8767  # 已废弃：TTS WebSocket端口
    core_control_port: int = 8768  # 已废弃：Core Server控制WebSocket端口


class EnvironmentsConfig(BaseModel):
    """环境配置（用于启动脚本）"""
    vision: str = "wakefusion_vision"  # 视觉模块Conda环境名
    audio: str = "wakefusion"  # 音频模块Conda环境名
    core: str = "wakefusion"  # 核心模块Conda环境名
    asr: str = "wakefusion"  # ASR模块Conda环境名（可与core相同）
    tts: str = "wakefusion"  # TTS模块Conda环境名（可与core相同）


class AppConfig(BaseModel):
    """应用总配置"""
    audio: AudioConfig = Field(default_factory=AudioConfig)
    kws: KWSConfig = Field(default_factory=KWSConfig)
    vad: VADConfig = Field(default_factory=VADConfig)
    vision: VisionConfig = Field(default_factory=VisionConfig)
    fusion: FusionConfig = Field(default_factory=FusionConfig)
    runtime: RuntimeConfig = Field(default_factory=RuntimeConfig)
    zmq: ZMQConfig = Field(default_factory=ZMQConfig)
    vision_wake: VisionWakeConfig = Field(default_factory=VisionWakeConfig)
    audio_threshold: AudioThresholdConfig = Field(default_factory=AudioThresholdConfig)
    conversation: ConversationConfig = Field(default_factory=ConversationConfig)
    asr: ASRConfig = Field(default_factory=ASRConfig)
    tts: TTSConfig = Field(default_factory=TTSConfig)
    llm_agent: LLMAgentConfig = Field(default_factory=LLMAgentConfig)
    websocket: WebSocketConfig = Field(default_factory=WebSocketConfig)
    environments: EnvironmentsConfig = Field(default_factory=EnvironmentsConfig)


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
