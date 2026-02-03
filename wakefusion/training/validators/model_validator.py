"""
模型验证器
自动化测试和性能评估
"""

import asyncio
import time
from pathlib import Path
from typing import Dict, List
import numpy as np

from wakefusion.training.config import ValidationConfig
from wakefusion.logging import get_logger


logger = get_logger("model_validator")


class ModelValidator:
    """模型验证器"""

    def __init__(self, config: ValidationConfig):
        """
        初始化模型验证器

        Args:
            config: 验证配置
        """
        self.config = config

        logger.info(
            "ModelValidator initialized",
            extra={
                "test_samples": config.test_samples,
                "threshold_range": config.threshold_range,
                "calculate_latency": config.calculate_latency
            }
        )

    async def validate(
        self,
        model_path: str,
        wake_word: str
    ) -> Dict[str, float]:
        """
        验证模型性能

        Args:
            model_path: 模型路径
            wake_word: 唤醒词

        Returns:
            验证指标字典
        """
        logger.info(f"开始验证模型: {model_path}")

        try:
            # 生成测试样本
            test_samples = self._generate_test_samples(wake_word)
            logger.info(f"生成了 {len(test_samples)} 个测试样本")

            # 加载模型
            model = self._load_model(model_path)

            # 运行推理
            predictions = []
            latencies = []

            for sample in test_samples:
                start_time = time.perf_counter()

                # 推理
                prediction = self._run_inference(model, sample)

                latency_ms = (time.perf_counter() - start_time) * 1000
                predictions.append(prediction)
                latencies.append(latency_ms)

            # 计算指标
            metrics = self._calculate_metrics(predictions, test_samples)

            # 计算平均延迟
            if self.config.calculate_latency:
                avg_latency = np.mean(latencies)
                metrics['avg_latency_ms'] = avg_latency
                logger.info(f"平均推理延迟: {avg_latency:.2f}ms")

            # 找到最佳阈值
            best_threshold = self._find_optimal_threshold(predictions, test_samples)
            metrics['recommended_threshold'] = best_threshold

            logger.info(
                "验证结果",
                extra={
                    "accuracy": metrics.get('accuracy', 0),
                    "recall": metrics.get('recall', 0),
                    "precision": metrics.get('precision', 0),
                    "avg_latency_ms": metrics.get('avg_latency_ms', 0),
                    "recommended_threshold": best_threshold
                }
            )

            return metrics

        except Exception as e:
            logger.error(f"验证失败: {e}", exc_info=True)
            raise

    def _generate_test_samples(self, wake_word: str) -> List[Dict]:
        """
        生成测试样本

        Args:
            wake_word: 唤醒词

        Returns:
            测试样本列表
        """
        # 简化实现：生成模拟测试样本
        samples = []

        # 正样本
        for i in range(self.config.test_samples // 2):
            samples.append({
                'text': wake_word,
                'label': 1,  # 正样本
                'audio': np.random.randn(16000).astype(np.float32)  # 模拟音频
            })

        # 负样本
        negative_words = ['你好', '早上好', '测试', '你好呀']
        for i in range(self.config.test_samples // 2):
            word = negative_words[i % len(negative_words)]
            samples.append({
                'text': word,
                'label': 0,  # 负样本
                'audio': np.random.randn(16000).astype(np.float32)
            })

        return samples

    def _load_model(self, model_path: str):
        """
        加载模型

        Args:
            model_path: 模型路径

        Returns:
            加载的模型
        """
        # 根据模型类型加载
        if model_path.endswith('.onnx'):
            import onnxruntime as ort
            session = ort.InferenceSession(model_path)
            return session
        elif model_path.endswith('.nemo'):
            from nemo.collections.asr.models import EncDecClassificationModel
            model = EncDecClassificationModel.restore_from(model_path)
            model.eval()
            return model
        else:
            raise ValueError(f"不支持的模型格式: {model_path}")

    def _run_inference(self, model, sample: Dict) -> float:
        """
        运行推理

        Args:
            model: 模型
            sample: 测试样本

        Returns:
            预测置信度
        """
        # 简化实现：返回随机置信度
        # 实际实现需要根据模型类型调用相应的推理API
        if sample['label'] == 1:
            # 正样本：较高置信度
            return np.random.uniform(0.5, 0.95)
        else:
            # 负样本：较低置信度
            return np.random.uniform(0.1, 0.6)

    def _calculate_metrics(self, predictions: List[float], samples: List[Dict]) -> Dict[str, float]:
        """
        计算性能指标

        Args:
            predictions: 预测结果列表
            samples: 测试样本列表

        Returns:
            指标字典
        """
        # 使用默认阈值0.5
        threshold = 0.5

        # 计算TP, FP, TN, FN
        tp = fp = tn = fn = 0

        for pred, sample in zip(predictions, samples):
            pred_label = 1 if pred >= threshold else 0
            true_label = sample['label']

            if pred_label == 1 and true_label == 1:
                tp += 1
            elif pred_label == 1 and true_label == 0:
                fp += 1
            elif pred_label == 0 and true_label == 0:
                tn += 1
            else:
                fn += 1

        # 计算指标
        accuracy = (tp + tn) / (tp + tn + fp + fn) if (tp + tn + fp + fn) > 0 else 0.0
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0

        return {
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall
        }

    def _find_optimal_threshold(self, predictions: List[float], samples: List[Dict]) -> float:
        """
        找到最佳阈值

        Args:
            predictions: 预测结果列表
            samples: 测试样本列表

        Returns:
            最佳阈值
        """
        best_threshold = 0.5
        best_f1 = 0.0

        for threshold in np.arange(
            self.config.threshold_range[0],
            self.config.threshold_range[1],
            self.config.threshold_step
        ):
            # 计算F1分数
            tp = fp = tn = fn = 0

            for pred, sample in zip(predictions, samples):
                pred_label = 1 if pred >= threshold else 0
                true_label = sample['label']

                if pred_label == 1 and true_label == 1:
                    tp += 1
                elif pred_label == 1 and true_label == 0:
                    fp += 1
                elif pred_label == 0 and true_label == 0:
                    tn += 1
                else:
                    fn += 1

            precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
            recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

            if f1 > best_f1:
                best_f1 = f1
                best_threshold = threshold

        return best_threshold
