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

        # 线程控制
        self.is_running = False
        self.thread: Optional[threading.Thread] = None
        self.task_queue: queue.Queue = queue.Queue(maxsize=32)

        # 状态跟踪
        self.last_detection_time = 0.0
        self.detection_count = 0
        self.processed_frames = 0
        self.dropped_frames = 0

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
            if hasattr(self.model, 'labels'):
                self.labels = self.model.labels
            else:
                # 默认标签（Google Speech Commands数据集）
                self.labels = [
                    'yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go',
                    'zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine',
                    'bed', 'bird', 'cat', 'dog', 'happy', 'house', 'marvin', 'sheila', 'tree', 'wow'
                ]
                logger.warning("Model has no labels attribute, using default Google Speech Commands labels")

            logger.info(
                "MatchboxNet model loaded successfully",
                extra={
                    "labels_count": len(self.labels),
                    "labels": self.labels[:5],  # 只记录前5个标签
                    "device": str(device)
                }
            )

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
        检测关键词

        Args:
            frame: 音频帧
        """
        start_time = time.perf_counter()

        try:
            # 准备输入
            # MatchboxNet期望的输入: [batch, time]
            audio_tensor = torch.from_numpy(frame.pcm16.astype(np.float32) / 32768.0)

            # 添加batch维度
            if audio_tensor.dim() == 1:
                audio_tensor = audio_tensor.unsqueeze(0)

            # 移动到设备
            device = torch.device(self.config.device)
            audio_tensor = audio_tensor.to(device)

            # 推理
            with torch.no_grad():
                logits = self.model.forward(input_signal=audio_tensor, input_signal_length=torch.tensor([audio_tensor.shape[1]]))

                # 获取概率分布
                probs = torch.softmax(logits, dim=-1)

                # 获取top-k结果
                probs_values, probs_indices = torch.topk(probs[0], k=min(5, len(self.labels)))

            # 转换为numpy
            probs_values = probs_values.cpu().numpy()
            probs_indices = probs_indices.cpu().numpy()

            # 检查最高置信度的标签
            top_confidence = float(probs_values[0])
            top_label_idx = int(probs_indices[0])
            top_label = self.labels[top_label_idx]

            # 记录延迟
            latency_ms = (time.perf_counter() - start_time) * 1000
            self.latency_samples.append(latency_ms)
            if len(self.latency_samples) > 100:
                self.latency_samples.pop(0)

            record_latency("kws.inference_latency_ms", latency_ms)

            # 检查是否达到阈值
            current_time = time.time()
            in_cooldown = (current_time - self.last_detection_time) * 1000 < self.config.cooldown_ms

            if top_confidence >= self.config.threshold and not in_cooldown:
                # 检测到关键词
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
                        payload={
                            "keyword": top_label,
                            "confidence": top_confidence,
                            "inference_latency_ms": latency_ms,
                            "label_idx": top_label_idx
                        }
                    )
                    self.event_callback(event)

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
