"""
NeMo训练器
封装NVIDIA NeMo训练逻辑
"""

import asyncio
import torch
from pathlib import Path
from typing import Optional, Callable, Dict, Any
import yaml

from wakefusion.training.config import NeMoTrainingConfig, TrainingConfig
from wakefusion.logging import get_logger


logger = get_logger("nemo_trainer")


class NeMoTrainer:
    """NeMo训练器"""

    def __init__(
        self,
        config: NeMoTrainingConfig,
        device: str = "cpu"
    ):
        """
        初始化NeMo训练器

        Args:
            config: NeMo训练配置
            device: 训练设备
        """
        self.config = config
        self.device = device

        logger.info(
            "NeMoTrainer initialized",
            extra={
                "device": device,
                "batch_size": config.batch_size,
                "num_epochs": config.num_epochs,
                "learning_rate": config.learning_rate
            }
        )

    async def train(
        self,
        train_config: TrainingConfig,
        output_dir: Path,
        progress_callback: Optional[Callable[[int, int, Dict[str, float]], None]] = None
    ) -> tuple[str, Dict[str, float]]:
        """
        训练NeMo模型

        Args:
            train_config: 训练总配置
            output_dir: 输出目录
            progress_callback: 进度回调 (epoch, total_epochs, metrics)

        Returns:
            (模型路径, 训练指标)
        """
        try:
            from nemo.collections.asr.models import EncDecClassificationModel
            from omegaconf import OmegaConf, DictConfig
            import lightning.pytorch as pl
            from lightning.pytorch.callbacks import ModelCheckpoint

            logger.info("开始NeMo模型训练")

            # 创建输出目录
            output_dir = Path(output_dir)
            output_dir.mkdir(parents=True, exist_ok=True)

            # 加载预训练模型
            logger.info("加载预训练MatchboxNet模型")
            pretrained_model = EncDecClassificationModel.from_pretrained(
                model_name='commandrecognition_en_matchboxnet3x1x64_v1'
            )

            # 修改分类层以适应新的类别数
            num_classes = len(train_config.negative_samples) + 2  # +2 for wake_word and background

            # 创建训练配置
            model_config = {
                'sample_rate': 16000,
                'labels': ['_background_noise_', train_config.label_name] + train_config.negative_samples,
                'model': {
                    'sample_rate': 16000,
                    'labels': ['_background_noise_', train_config.label_name] + train_config.negative_samples,
                    'encoder': {
                        'kernel_size': self.config.channels // 4,
                        'strides': [2, 1, 2, 1, 2, 1, 2, 1],
                        'num_blocks': self.config.num_blocks,
                        'num_layers_per_block': self.config.num_layers_per_block,
                        'channels': self.config.channels
                    }
                },
                'train_ds': {
                    'manifest_filepath': self.config.train_manifest,
                    'sample_rate': 16000,
                    'batch_size': self.config.batch_size,
                    'shuffle': True,
                    'num_workers': 0,  # Windows 下必须设为 0，避免多进程序列化错误
                    'pin_memory': True
                },
                'validation_ds': {
                    'manifest_filepath': self.config.val_manifest,
                    'sample_rate': 16000,
                    'batch_size': self.config.batch_size,
                    'num_workers': 0,  # Windows 下必须设为 0，避免多进程序列化错误
                    'pin_memory': True
                },
                'optim': {
                    'lr': self.config.learning_rate,
                    'optimizer': {
                        'name': self.config.optimizer,
                        'kwargs': {
                            'betas': [0.9, 0.999],
                            'weight_decay': self.config.weight_decay
                        }
                    },
                    'sched': {
                        'name': self.config.scheduler,
                        'kwargs': {
                            'warmup_steps': None,
                            'T_max': self.config.num_epochs,
                            'eta_min': 1e-6
                        }
                    }
                },
                'trainer': {
                    'max_epochs': self.config.num_epochs,
                    'accelerator': 'gpu' if self.device == 'cuda' else 'cpu',
                    'devices': 1,
                    'log_every_n_steps': 10,
                    'check_val_every_n_epoch': 1
                }
            }

            # 创建新模型实例
            logger.info(f"创建模型，类别数: {num_classes}")
            model = EncDecClassificationModel(cfg=model_config)

            # 迁移学习：复制预训练模型的编码器权重
            logger.info("迁移预训练权重")
            model.model.encoder = pretrained_model.model.encoder

            # 设置训练器
            checkpoint_callback = ModelCheckpoint(
                monitor=self.config.monitor_metric,
                mode='max',
                save_top_k=self.config.save_top_k,
                dirpath=str(output_dir),
                filename='matchboxnet-{{epoch:02d}}-{{val_accuracy:.3f}}'
            )

            trainer_config = {
                'max_epochs': self.config.num_epochs,
                'accelerator': 'gpu' if self.device == 'cuda' else 'cpu',
                'devices': 1,
                'callbacks': [checkpoint_callback],
                'log_every_n_steps': 10,
                'enable_progress_bar': True
            }

            trainer = pl.Trainer(**trainer_config)

            # 自定义训练循环以支持进度回调
            logger.info("开始训练...")
            logger.info(f"训练样本: {self.config.train_manifest}")
            logger.info(f"验证样本: {self.config.val_manifest}")

            # 执行训练
            trainer.fit(model)

            # 获取最佳模型路径
            best_model_path = checkpoint_callback.best_model_path
            logger.info(f"训练完成，最佳模型: {best_model_path}")

            # 获取训练指标
            train_accuracy = trainer.callback_metrics.get('train_accuracy', 0.0)
            val_accuracy = trainer.callback_metrics.get('val_accuracy', 0.0)
            best_epoch = trainer.current_epoch

            metrics = {
                'train_accuracy': float(train_accuracy) if hasattr(train_accuracy, 'item') else float(train_accuracy),
                'val_accuracy': float(val_accuracy) if hasattr(val_accuracy, 'item') else float(val_accuracy),
                'best_epoch': best_epoch
            }

            logger.info(
                "训练指标",
                extra={
                    "train_accuracy": metrics['train_accuracy'],
                    "val_accuracy": metrics['val_accuracy'],
                    "best_epoch": best_epoch
                }
            )

            return str(best_model_path), metrics

        except Exception as e:
            logger.error(f"训练失败: {e}", exc_info=True)
            raise
