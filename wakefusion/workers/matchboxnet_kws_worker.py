"""
MatchboxNet KWS Worker
基于 NVIDIA NeMo 框架的关键词检测
"""

import threading
import queue
import time
import logging
from typing import Optional, Callable, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch

from wakefusion.types import BaseEvent, EventType, AudioFrame
from wakefusion.logging import get_logger
from wakefusion.metrics import get_metrics, record_latency


logger = get_logger("matchboxnet_kws")
metrics = get_metrics()


@dataclass
class MatchboxNetConfig:
    """MatchboxNet配置"""
    model_name: str = "commandrecognition_en_matchboxnet3x1x64_v1"  # 预训练模型名称
    sample_rate: int = 16000  # 工作采样率
    frame_ms: int = 80  # MatchboxNet输入帧长（ms）
    threshold: float = 0.5  # 置信度阈值
    cooldown_ms: int = 1200  # 冷却期（ms）
    device: str = "cpu"  # 推理设备：cpu 或 cuda
    model_path: Optional[str] = None  # 本地模型路径（优先使用）


class MatchboxNetKWSWorker:
    """
    MatchboxNet关键词检测工作线程

    使用NVIDIA NeMo框架的MatchboxNet模型进行关键词检测
    """

    def __init__(
        self,
        config: MatchboxNetConfig,
        event_callback: Optional[Callable[[BaseEvent], None]] = None
    ):
        """
        初始化MatchboxNet KWS Worker

        Args:
            config: MatchboxNet配置
            event_callback: 事件回调函数
        """
        self.config = config
        self.event_callback = event_callback

        # 计算帧大小
        self.frame_size = int(config.sample_rate * config.frame_ms / 1000)

        # 模型和分词器
        self.model = None
        self.labels: List[str] = []

        # 循环缓冲区（1.28秒上下文）- 精确匹配模型训练长度
        # MatchboxNet 模型基于 128 帧 × 10ms = 1.28秒 的固定长度训练
        # 必须精确匹配，否则特征图不完整会被判定为背景噪音
        buffer_length_samples = int(config.sample_rate * 1.28)  # 1.28秒 = 20480 采样点
        self.audio_buffer = np.zeros(buffer_length_samples, dtype=np.float32)
        self.buffer_write_pos = 0  # 循环缓冲区的写入位置
        self.buffer_frames_received = 0  # 已接收的总帧数（用于判断缓冲区是否填满）
        self.step_counter = 0  # 步长计数器（独立于总帧数）
        self.step_size_frames = 2  # 每 2 帧（160ms）触发一次推理
        # 缓冲区最小填充要求：必须完全填满 1.28 秒才能开始推理
        # 1.28秒 = 1280ms / 80ms = 16帧
        self.min_buffer_frames = int(1.28 * config.sample_rate / self.frame_size)

        # 线程控制
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.task_queue: queue.Queue = queue.Queue(maxsize=64)  # 增加队列大小，减少丢帧

        # 状态跟踪
        self.last_detection_time = 0.0
        self.detection_count = 0
        self.processed_frames = 0
        self.dropped_frames = 0
        self.session_id = self._generate_session_id()

        # 延迟统计
        self.latency_samples = []

        logger.info(
            "MatchboxNetKWSWorker initialized",
            extra={
                "model_name": config.model_name,
                "frame_ms": config.frame_ms,
                "threshold": config.threshold,
                "device": config.device
            }
        )

    def _generate_session_id(self) -> str:
        """生成会话ID"""
        from datetime import datetime
        return f"matchboxnet-{datetime.now().strftime('%Y%m%d-%H%M%S')}"

    def load_model(self):
        """加载MatchboxNet模型"""
        try:
            from nemo.collections.asr.models import EncDecClassificationModel
            from nemo.core.classes import ModelPT

            logger.info(f"Loading MatchboxNet model: {self.config.model_name}")

            # 加载模型
            if self.config.model_path and Path(self.config.model_path).exists():
                # 从本地路径加载
                logger.info(f"Loading model from local path: {self.config.model_path}")
                self.model = EncDecClassificationModel.restore_from(
                    restore_path=self.config.model_path
                )
            else:
                # 从NGC加载预训练模型
                logger.info(f"Loading pretrained model from NGC: {self.config.model_name}")
                self.model = EncDecClassificationModel.from_pretrained(
                    model_name=self.config.model_name
                )

            # 设置设备
            device = torch.device(self.config.device)
            self.model = self.model.to(device)
            self.model.eval()

            # 获取标签
            self.labels = None
            
            # 途径1: 尝试从模型属性获取
            if hasattr(self.model, 'labels') and self.model.labels:
                self.labels = self.model.labels
            
            # 途径2: 尝试从模型配置(cfg)获取 (针对自定义训练模型)
            if not self.labels and hasattr(self.model, 'cfg') and hasattr(self.model.cfg, 'labels'):
                self.labels = list(self.model.cfg.labels)
            
            # 途径3: 尝试从解码器词表获取
            if not self.labels and hasattr(self.model, 'decoder') and hasattr(self.model.decoder, 'vocabulary'):
                self.labels = self.model.decoder.vocabulary
            
            # 兜底方案: 如果都失败了，强制指定为 ["xiaokang"]，防止崩溃
            if not self.labels:
                logger.warning("Labels not found in model, using fallback: ['xiaokang']")
                self.labels = ["xiaokang"]
            
            # 确保 labels 是列表格式，并展平嵌套列表
            if not isinstance(self.labels, list):
                if isinstance(self.labels, (tuple, set)):
                    self.labels = list(self.labels)
            else:
                    self.labels = [self.labels]
            
            # 展平嵌套列表（处理 [[...]] 的情况）
            if self.labels and isinstance(self.labels[0], list):
                self.labels = self.labels[0]
            
            logger.info(f"Loaded labels from model: {self.labels}")

            logger.info(
                "MatchboxNet model loaded successfully",
                extra={
                    "labels_count": len(self.labels),
                    "labels": self.labels[:5] if len(self.labels) > 0 else [],  # 只记录前5个标签
                    "device": str(device)
                }
            )

            # 检查模型的预处理配置
            if hasattr(self.model, 'cfg') and hasattr(self.model.cfg, 'preprocessor'):
                preprocessor_cfg = self.model.cfg.preprocessor
                logger.info(f"Model preprocessor config: {preprocessor_cfg}")
                
                # 检查采样率是否匹配
                if hasattr(preprocessor_cfg, 'sample_rate'):
                    expected_sr = preprocessor_cfg.sample_rate
                    if expected_sr != self.config.sample_rate:
                        logger.warning(
                            f"Sample rate mismatch: model expects {expected_sr}Hz, "
                            f"but config is {self.config.sample_rate}Hz"
                        )
            
            # 检查模型的其他关键配置
            if hasattr(self.model, 'cfg'):
                logger.debug(f"Model config keys: {list(self.model.cfg.keys()) if hasattr(self.model.cfg, 'keys') else 'N/A'}")
                
                # 关键修复：检查 timesteps 配置的实际含义
                if hasattr(self.model.cfg, 'timesteps'):
                    timesteps = self.model.cfg.timesteps
                    logger.info(f"Model timesteps config: {timesteps} (type: {type(timesteps)})")
                    # timesteps 通常指的是预处理后的特征时间步，不是原始采样点
                    # NeMo 模型期望原始音频输入，内部会自动预处理
                    if isinstance(timesteps, (int, float)) and timesteps < 1000:
                        logger.info(
                            f"ℹ️  Note: Model timesteps ({timesteps}) refers to internal feature dimensions, "
                            f"not input requirements. NeMo models expect raw audio input."
                        )
                        # 存储这个信息，用于后续的输入验证
                        self._model_timesteps = timesteps
                    else:
                        self._model_timesteps = None
                else:
                    self._model_timesteps = None
                
                # 关键修复：检查 timesteps 配置的实际含义
                if hasattr(self.model.cfg, 'timesteps'):
                    timesteps = self.model.cfg.timesteps
                    logger.info(f"Model timesteps config: {timesteps} (type: {type(timesteps)})")
                    # timesteps 通常指的是预处理后的特征时间步，不是原始采样点
                    # NeMo 模型期望原始音频输入，内部会自动预处理
                    if isinstance(timesteps, (int, float)) and timesteps < 1000:
                        logger.info(
                            f"⚠️  Note: Model timesteps ({timesteps}) likely refers to feature timesteps, "
                            f"not raw audio samples. NeMo models typically expect raw audio input."
                        )
                        # 存储这个信息，用于后续的输入验证
                        self._model_timesteps = timesteps
                    else:
                        self._model_timesteps = None
                else:
                    self._model_timesteps = None

            # 记录指标
            metrics.set_gauge("kws.model_loaded", 1.0)

        except Exception as e:
            logger.error(f"Failed to load MatchboxNet model: {e}")
            raise

    def start(self):
        """启动工作线程"""
        if self.is_running:
            logger.warning("MatchboxNetKWSWorker already running")
            return

        # 加载模型
        self.load_model()
        
        # 重置缓冲区状态
        self.audio_buffer.fill(0.0)
        self.buffer_write_pos = 0
        self.buffer_frames_received = 0
        self.step_counter = 0

        self.is_running = True
        self.thread = threading.Thread(target=self._run_loop, daemon=True)
        self.thread.start()

        logger.info("MatchboxNetKWSWorker started")

    def stop(self):
        """停止工作线程"""
        if not self.is_running:
            return

        self.is_running = False

        # 清空队列并唤醒线程
        while not self.task_queue.empty():
            try:
                self.task_queue.get_nowait()
            except queue.Empty:
                break

        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)

        logger.info("MatchboxNetKWSWorker stopped")

    def process_frame(self, frame: AudioFrame):
        """
        处理音频帧（非阻塞）

        Args:
            frame: 音频帧
        """
        if not self.is_running:
            return

        # 检查采样率
        if frame.sample_rate != self.config.sample_rate:
            logger.warning(
                f"Frame sample rate mismatch: expected {self.config.sample_rate}, got {frame.sample_rate}"
            )
            return

        # 将帧放入队列（非阻塞）
        try:
            self.task_queue.put_nowait(frame)
            self.processed_frames += 1
        except queue.Full:
            self.dropped_frames += 1
            metrics.increment_counter("kws.dropped_frames")

            if self.dropped_frames % 100 == 0:
                logger.warning(
                    f"KWS task queue full, dropped {self.dropped_frames} frames"
                )

    def _run_loop(self):
        """主处理循环（工作线程）"""
        logger.info("MatchboxNetKWSWorker processing loop started")

        while self.is_running:
            try:
                # 从队列获取帧（带超时）
                frame = self.task_queue.get(timeout=0.5)

                # 处理帧
                self._detect_keyword(frame)

            except queue.Empty:
                continue
            except Exception as e:
                logger.error(f"Error in KWS processing loop: {e}")
                metrics.increment_counter("kws.errors")

        logger.info("MatchboxNetKWSWorker processing loop stopped")

    def _detect_keyword(self, frame: AudioFrame):
        """
        检测关键词（使用滑动窗口累积 2.0 秒上下文）

        Args:
            frame: 音频帧
        
        实现：
        - 将当前帧追加到 2.0 秒的滑动窗口缓冲区（关键修复：从1.5秒增加到2.0秒）
        - 每收到 2 帧（160ms）才触发一次推理，减少 CPU 压力
        - 使用整个缓冲区进行推理，提供完整的上下文信息
        - 确保能完整覆盖中文唤醒词"你好小康"（通常需要1.5-2秒）
        """
        start_time = time.perf_counter()

        try:
            # 将当前帧归一化
            frame_audio = frame.pcm16.astype(np.float32) / 32768.0
            frame_len = len(frame_audio)
            
            # 循环缓冲区写入：使用高效的循环索引，避免 np.roll 的开销
            # 将新帧写入到循环缓冲区的当前位置
            buffer_len = len(self.audio_buffer)
            
            # 如果帧长度小于等于缓冲区剩余空间，直接写入
            if self.buffer_write_pos + frame_len <= buffer_len:
                self.audio_buffer[self.buffer_write_pos:self.buffer_write_pos + frame_len] = frame_audio
                self.buffer_write_pos = (self.buffer_write_pos + frame_len) % buffer_len
            else:
                # 帧跨越缓冲区边界，需要分两部分写入
                first_part_len = buffer_len - self.buffer_write_pos
                self.audio_buffer[self.buffer_write_pos:] = frame_audio[:first_part_len]
                self.audio_buffer[:frame_len - first_part_len] = frame_audio[first_part_len:]
                self.buffer_write_pos = frame_len - first_part_len
            
            self.buffer_frames_received += 1
            self.step_counter += 1
            
            # 关键修复：等待缓冲区完全填满到 1.28 秒才开始推理
            # 这样可以避免在缓冲区大部分是0（静音）时进行推理
            if self.buffer_frames_received < self.min_buffer_frames:
                # 缓冲区还未完全填满，跳过本次推理
                logger.debug(
                    f"Buffer warming up: {self.buffer_frames_received}/{self.min_buffer_frames} frames"
                )
                return
            
            # 步长控制：每收到 2 帧（160ms）才触发一次推理
            if self.step_counter < self.step_size_frames:
                # 步长未达到，跳过本次推理
                return
            
            # 重置步长计数器（但保持总帧数和缓冲区内容）
            self.step_counter = 0
            
            # 准备输入：从循环缓冲区中提取完整的 1.28 秒上下文
            # 由于使用循环缓冲区，需要处理循环边界，提取连续的 1.28 秒音频
            buffer_len = len(self.audio_buffer)
            
            if self.buffer_write_pos == 0:
                # 缓冲区已完全填满，没有循环，直接使用整个缓冲区
                audio_data = self.audio_buffer.copy()
            else:
                # 缓冲区有循环，需要重新排列：从 write_pos 开始，到 write_pos 结束（循环）
                # 提取从 write_pos 到末尾 + 从开头到 write_pos 的连续数据
                audio_data = np.concatenate([
                    self.audio_buffer[self.buffer_write_pos:],
                    self.audio_buffer[:self.buffer_write_pos]
                ])
            
            # MatchboxNet期望的输入: [batch, time]
            # 注意：audio_data 已经是归一化到 [-1, 1] 的 float32 数据（从 pcm16 / 32768.0）
            audio_tensor = torch.from_numpy(audio_data)
            
            # 调试信息：检查缓冲区是否有有效音频（非全0）
            buffer_rms = float(np.sqrt(np.mean(audio_data ** 2)))
            if buffer_rms < 0.001:
                logger.debug(f"Buffer appears to be mostly silence (RMS={buffer_rms:.6f})")

            # 关键调试：检查输入长度（仅第一次）
            if not hasattr(self, '_input_debug_printed'):
                logger.info(f"Input audio shape (before batch): {audio_tensor.shape}")
                logger.info(f"Input audio length: {len(audio_data)} samples ({len(audio_data) / self.config.sample_rate:.3f} seconds)")
                logger.info(f"Buffer length: {len(self.audio_buffer)} samples ({len(self.audio_buffer) / self.config.sample_rate:.3f} seconds)")
                # 检查模型期望的输入长度（如果有配置）
                if hasattr(self, '_model_timesteps') and self._model_timesteps:
                    logger.info(f"Model timesteps config: {self._model_timesteps} (likely feature timesteps, not raw samples)")
                    logger.info(
                        f"ℹ️  NeMo EncDecClassificationModel expects raw audio input. "
                        f"The timesteps config refers to internal feature dimensions, not input requirements."
                    )
                self._input_debug_printed = True

            # 添加batch维度
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            # 移动到设备
            device = torch.device(self.config.device)
            audio_tensor = audio_tensor.to(device)
            
            # 关键修复：恢复均值方差标准化（必须与训练时一致）
            # MatchboxNet 训练时使用标准化音频（std=1.0），推理时也必须使用
            # 峰值归一化无法替代标准化，因为模型对信号强度极其敏感
            audio_mean = audio_tensor.mean()
            audio_std = audio_tensor.std()
            if audio_std > 1e-6:  # 避免除零
                audio_tensor = (audio_tensor - audio_mean) / (audio_std + 1e-6)
                logger.debug(f"Audio standardized: mean={audio_mean:.6f}, std={audio_std:.6f}")
            else:
                # 如果标准差太小（接近静音），保持原值
                logger.debug(f"Audio std too small ({audio_std:.6f}), skipping standardization")

            # 调试信息：打印音频统计信息（仅第一帧）
            if not hasattr(self, '_debug_printed'):
                logger.info(f"Audio input stats: mean={audio_tensor.mean():.6f}, "
                          f"std={audio_tensor.std():.6f}, max={audio_tensor.max():.6f}, "
                          f"min={audio_tensor.min():.6f}, rms={torch.sqrt(torch.mean(audio_tensor**2)):.6f}")
                self._debug_printed = True

            # 推理
            with torch.no_grad():
                logits = self.model.forward(input_signal=audio_tensor, input_signal_length=torch.tensor([audio_tensor.shape[1]]))
                
                # 关键调试：检查 logits 的形状和值（仅第一次）
                if not hasattr(self, '_logits_debug_printed'):
                    logits_np = logits.cpu().numpy()
                    logger.info(f"Logits shape: {logits.shape}, dtype: {logits.dtype}")
                    logger.info(f"Logits values (first inference): {logits_np}")
                    logger.info(f"Logits min: {logits.min().item():.6f}, max: {logits.max().item():.6f}, mean: {logits.mean().item():.6f}")
                    logger.info(f"Expected labels count: {len(self.labels)}, logits size: {logits.shape[-1]}")
                    if logits.shape[-1] != len(self.labels):
                        logger.error(f"⚠️  CRITICAL: Logits size ({logits.shape[-1]}) != labels count ({len(self.labels)})")
                    self._logits_debug_printed = True

                # 获取概率分布
                probs = torch.softmax(logits, dim=-1)
                
                # 关键调试：检查 softmax 后的概率分布（仅第一次）
                if not hasattr(self, '_probs_debug_printed'):
                    probs_np = probs.cpu().numpy()
                    logger.info(f"Probs shape: {probs.shape}")
                    logger.info(f"Probs sum: {probs.sum().item():.6f} (should be ~1.0)")
                    logger.info(f"Probs values (first inference): {probs_np}")
                    logger.info(f"Probs per label: {dict(zip(self.labels, probs_np[0]))}")
                    self._probs_debug_printed = True

                # 获取top-k结果
                probs_values, probs_indices = torch.topk(probs[0], k=min(5, len(self.labels)))

            # 转换为numpy
            probs_values = probs_values.cpu().numpy()
            probs_indices = probs_indices.cpu().numpy()
            probs_full = probs[0].cpu().numpy()  # 完整概率分布

            # 关键修复2：增强调试输出 - 每一帧都打印 xiaokang 的实时置信度
            # 检查最高置信度的标签
            top_confidence = float(probs_values[0])
            top_label_idx = int(probs_indices[0])
            top_label = self.labels[top_label_idx]
            
            # 关键修复3：确保正确获取 xiaokang 的置信度（无论它是否是最高分）
            xiaokang_idx = None
            xiaokang_prob = 0.0
            
            # 关键调试：打印所有标签和对应的索引（仅第一次）
            if not hasattr(self, '_labels_debug_printed'):
                logger.info(f"Labels: {self.labels}")
                logger.info(f"Labels count: {len(self.labels)}")
                logger.info(f"Probs full shape: {probs_full.shape}")
                logger.info(f"Probs full length: {len(probs_full)}")
                if len(probs_full) != len(self.labels):
                    logger.error(f"⚠️  CRITICAL: Label count mismatch! Labels: {len(self.labels)}, Probs: {len(probs_full)}")
                # 打印每个标签对应的概率
                logger.info("Label-to-probability mapping:")
                for i, label in enumerate(self.labels):
                    if i < len(probs_full):
                        logger.info(f"  [{i}] {label}: {probs_full[i]:.6f}")
                    else:
                        logger.error(f"  [{i}] {label}: INDEX OUT OF RANGE!")
                self._labels_debug_printed = True
            
            for i, label in enumerate(self.labels):
                if label == "xiaokang":
                    xiaokang_idx = i
                    if i < len(probs_full):
                        xiaokang_prob = float(probs_full[i])
                    else:
                        logger.error(f"⚠️  xiaokang index {i} out of range! Probs length: {len(probs_full)}")
                    break
            
            # 关键修复4：每一帧都打印 xiaokang 的实时置信度（格式：Confidence[xiaokang]: 0.XXXXXX）
            if xiaokang_idx is not None:
                logger.info(f"Confidence[xiaokang]: {xiaokang_prob:.6f} | Top: {top_label} ({top_confidence:.6f})")
            else:
                logger.warning("⚠️  'xiaokang' label not found in model labels!")
            
            # 打印完整概率分布（仅在DEBUG级别，避免刷屏）
            logger.debug("=" * 50)
            logger.debug("Full probability distribution:")
            for i, prob in enumerate(probs_full):
                if prob > 1e-6:  # 只打印非零概率
                    logger.debug(f"  {self.labels[i]}: {prob:.6f} ({prob*100:.2f}%)")
            logger.debug("=" * 50)
            
            # 关键修复5：优化检测条件 - 如果 xiaokang 概率 > 0.05，即使不是最高分也提示
            if xiaokang_idx is not None and xiaokang_prob > 0.05:
                if top_label != "xiaokang":
                    logger.info(
                        f"⚠️  xiaokang detected with probability {xiaokang_prob:.4f} "
                        f"(but top label is '{top_label}' with {top_confidence:.4f})"
                    )
                else:
                    logger.info(f"✅ xiaokang is top label with probability {xiaokang_prob:.4f}")

            # 记录延迟
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.latency_samples.append(latency_ms)
            if len(self.latency_samples) > 100:
                self.latency_samples.pop(0)

            record_latency("kws.inference_latency_ms", latency_ms)

            # 检查是否达到阈值
            current_time = time.time()
            in_cooldown = (current_time - self.last_detection_time) * 1000 < self.config.cooldown_ms

            # 过滤背景噪音：不将 _background_noise_ 视为有效关键词
            is_background_noise = (top_label == "_background_noise_" or 
                                  top_label.startswith("_background") or
                                  "noise" in top_label.lower())
            
            if top_confidence >= self.config.threshold and not in_cooldown and not is_background_noise:
                # 检测到关键词（排除背景噪音）
                self.detection_count += 1
                self.last_detection_time = current_time

                logger.info(
                    f"Keyword detected: {top_label}",
                    extra={
                        "keyword": top_label,
                        "confidence": top_confidence,
                        "latency_ms": latency_ms
                    }
                )

                metrics.increment_counter("kws.detections")
                metrics.set_gauge("kws.last_confidence", top_confidence)

                # 发布事件
                if self.event_callback:
                    event = BaseEvent(
                        type=EventType.KWS_HIT,
                        ts=frame.ts,
                        session_id=self.session_id,
                        priority=80,
                        **{
                            "payload": {
                            "keyword": top_label,
                            "confidence": top_confidence,
                            "inference_latency_ms": latency_ms,
                            "label_idx": top_label_idx
                            }
                        }
                    )
                    self.event_callback(event)
            elif is_background_noise and top_confidence >= self.config.threshold:
                # 记录背景噪音检测（用于调试，但不触发事件）
                logger.debug(
                    f"Background noise detected (filtered): {top_label} (confidence={top_confidence:.3f})",
                    extra={
                        "keyword": top_label,
                        "confidence": top_confidence
                    }
                )

            # 记录指标
            metrics.set_gauge("kws.top_confidence", top_confidence)
            metrics.set_gauge("kws.top_label", top_label_idx)

        except Exception as e:
            logger.error(f"Error in keyword detection: {e}")
            metrics.increment_counter("kws.errors")

    def get_stats(self) -> Dict[str, Any]:
        """获取统计信息"""
        avg_latency = np.mean(self.latency_samples) if self.latency_samples else 0.0

        return {
            "model": self.config.model_name,
            "labels": self.labels,
            "threshold": self.config.threshold,
            "cooldown_ms": self.config.cooldown_ms,
            "detections": self.detection_count,
            "processed_frames": self.processed_frames,
            "dropped_frames": self.dropped_frames,
            "avg_latency_ms": avg_latency,
            "is_running": self.is_running,
            "device": self.config.device
        }

    def set_threshold(self, threshold: float):
        """动态设置阈值"""
        self.config.threshold = threshold
        logger.info(f"KWS threshold updated to {threshold}")

    def set_cooldown(self, cooldown_ms: int):
        """动态设置冷却期"""
        self.config.cooldown_ms = cooldown_ms
        logger.info(f"KWS cooldown updated to {cooldown_ms}ms")


# 便捷函数
def create_matchboxnet_worker(
    model_name: str = "commandrecognition_en_matchboxnet3x1x64_v1",
    threshold: float = 0.5,
    cooldown_ms: int = 1200,
    device: str = "cpu",
    event_callback: Optional[Callable[[BaseEvent], None]] = None
) -> MatchboxNetKWSWorker:
    """
    创建MatchboxNet KWS Worker

    Args:
        model_name: 模型名称或路径
        threshold: 置信度阈值
        cooldown_ms: 冷却期（毫秒）
        device: 推理设备（cpu/cuda）
        event_callback: 事件回调

    Returns:
        MatchboxNetKWSWorker实例
    """
    config = MatchboxNetConfig(
        model_name=model_name,
        threshold=threshold,
        cooldown_ms=cooldown_ms,
        device=device
    )

    return MatchboxNetKWSWorker(
        config=config,
        event_callback=event_callback
    )
