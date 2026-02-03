"""
训练编排器
协调整个自动化训练流程
"""

import asyncio
import time
from pathlib import Path
from typing import Optional, Callable
from datetime import datetime
import yaml

from wakefusion.training.config import TrainingConfig, TrainingResult
from wakefusion.training.generators.tts_generator import TTSDataGenerator
from wakefusion.training.trainers.nemo_trainer import NeMoTrainer
from wakefusion.training.validators.model_validator import ModelValidator
from wakefusion.training.exporters.onnx_exporter import ONNXExporter
from wakefusion.logging import get_logger


logger = get_logger("training_orchestrator")


class TrainingOrchestrator:
    """
    训练编排器

    管理整个自动化训练流程：
    1. TTS数据生成
    2. 数据增强
    3. 创建NeMo清单
    4. NeMo模型训练
    5. ONNX导出
    6. 模型验证
    7. 配置更新
    8. 生成报告
    """

    def __init__(
        self,
        config: TrainingConfig,
        progress_callback: Optional[Callable[[str, float], None]] = None
    ):
        """
        初始化训练编排器

        Args:
            config: 训练配置
            progress_callback: 进度回调函数 (stage, progress 0-1)
        """
        self.config = config
        self.progress_callback = progress_callback

        # 创建输出目录
        self.output_dir = Path(config.output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # 子目录
        self.data_dir = self.output_dir / config.data_dir
        self.checkpoint_dir = self.output_dir / config.checkpoint_dir
        self.model_dir = self.output_dir / config.model_dir
        self.report_dir = self.output_dir / "reports"

        for dir_path in [self.data_dir, self.checkpoint_dir,
                         self.model_dir, self.report_dir]:
            dir_path.mkdir(parents=True, exist_ok=True)

        # 初始化各模块
        self.tts_generator = TTSDataGenerator(config.tts)
        self.nemo_trainer = NeMoTrainer(config.training, config.device)
        self.onnx_exporter = ONNXExporter()
        self.model_validator = ModelValidator(config.validation)

        logger.info(
            "TrainingOrchestrator initialized",
            extra={
                "wake_word": config.wake_word,
                "label_name": config.label_name,
                "output_dir": str(self.output_dir)
            }
        )

    def _update_progress(self, stage: str, progress: float, message: str = ""):
        """更新进度"""
        logger.info(f"[{stage}] {progress*100:.1f}% - {message}")

        if self.progress_callback:
            self.progress_callback(stage, progress)

    async def train(self) -> TrainingResult:
        """
        执行完整训练流程

        Returns:
            TrainingResult: 训练结果
        """
        start_time = time.time()
        result = TrainingResult(
            success=False,
            wake_word=self.config.wake_word,
            label_name=self.config.label_name
        )

        try:
            # ========================================
            # Stage 1: 生成TTS数据
            # ========================================
            logger.info("=" * 70)
            logger.info("Stage 1: 生成TTS数据")
            logger.info("=" * 70)

            self._update_progress("tts_generation", 0.0, "开始生成TTS数据")

            audio_files = await self.tts_generator.generate(
                text=self.config.wake_word,
                output_dir=self.data_dir / "raw",
                negative_samples=self.config.negative_samples
            )

            result.total_samples = len(audio_files)
            self._update_progress("tts_generation", 1.0,
                                f"生成了 {len(audio_files)} 个音频样本")

            # ========================================
            # Stage 2: 数据增强
            # ========================================
            logger.info("=" * 70)
            logger.info("Stage 2: 数据增强")
            logger.info("=" * 70)

            if self.config.augmentation.enabled:
                self._update_progress("augmentation", 0.0, "开始数据增强")

                augmented_files = await self.tts_generator.augment(
                    audio_files=audio_files,
                    output_dir=self.data_dir / "augmented",
                    config=self.config.augmentation
                )

                result.total_samples = len(augmented_files)
                self._update_progress("augmentation", 1.0,
                                    f"增强后共 {len(augmented_files)} 个样本")
            else:
                augmented_files = audio_files

            # ========================================
            # Stage 3: 创建NeMo清单
            # ========================================
            logger.info("=" * 70)
            logger.info("Stage 3: 创建NeMo清单文件")
            logger.info("=" * 70)

            self._update_progress("manifest_creation", 0.0, "创建训练清单")

            train_manifest, val_manifest = self._create_manifests(
                audio_files=augmented_files,
                output_dir=self.data_dir / "manifests"
            )

            self.config.training.train_manifest = str(train_manifest)
            self.config.training.val_manifest = str(val_manifest)

            self._update_progress("manifest_creation", 1.0, "清单创建完成")

            # ========================================
            # Stage 4: NeMo模型训练
            # ========================================
            logger.info("=" * 70)
            logger.info("Stage 4: NeMo模型训练")
            logger.info("=" * 70)

            self._update_progress("model_training", 0.0, "开始训练模型")

            nemo_model_path, training_metrics = await self.nemo_trainer.train(
                train_config=self.config,
                output_dir=self.checkpoint_dir
            )

            result.nemo_model_path = nemo_model_path
            result.train_accuracy = training_metrics.get("train_accuracy", 0.0)
            result.val_accuracy = training_metrics.get("val_accuracy", 0.0)
            result.best_epoch = training_metrics.get("best_epoch", 0)

            self._update_progress("model_training", 1.0, "训练完成")

            # ========================================
            # Stage 5: 导出ONNX
            # ========================================
            logger.info("=" * 70)
            logger.info("Stage 5: 导出ONNX模型")
            logger.info("=" * 70)

            self._update_progress("onnx_export", 0.0, "导出ONNX模型")

            onnx_model_path = await self.onnx_exporter.export(
                nemo_model_path=nemo_model_path,
                output_path=self.model_dir / f"{self.config.label_name}.onnx"
            )

            result.onnx_model_path = onnx_model_path
            self._update_progress("onnx_export", 1.0, "ONNX导出完成")

            # ========================================
            # Stage 6: 模型验证
            # ========================================
            logger.info("=" * 70)
            logger.info("Stage 6: 模型验证")
            logger.info("=" * 70)

            self._update_progress("validation", 0.0, "验证模型性能")

            validation_metrics = await self.model_validator.validate(
                model_path=onnx_model_path,
                wake_word=self.config.wake_word
            )

            result.test_accuracy = validation_metrics.get("accuracy", 0.0)
            result.test_recall = validation_metrics.get("recall", 0.0)
            result.test_precision = validation_metrics.get("precision", 0.0)
            result.avg_latency_ms = validation_metrics.get("avg_latency_ms", 0.0)
            result.recommended_threshold = validation_metrics.get("recommended_threshold", 0.5)

            self._update_progress("validation", 1.0, "验证完成")

            # ========================================
            # Stage 7: 更新配置
            # ========================================
            logger.info("=" * 70)
            logger.info("Stage 7: 更新系统配置")
            logger.info("=" * 70)

            self._update_progress("config_update", 0.0, "更新配置文件")

            config_path = await self._update_system_config(result)

            result.config_path = config_path
            self._update_progress("config_update", 1.0, "配置更新完成")

            # ========================================
            # Stage 8: 生成报告
            # ========================================
            logger.info("=" * 70)
            logger.info("Stage 8: 生成训练报告")
            logger.info("=" * 70)

            self._update_progress("report_generation", 0.0, "生成报告")

            report_path = await self._generate_report(result)

            result.report_path = report_path
            self._update_progress("report_generation", 1.0, "报告生成完成")

            # 标记成功
            result.success = True
            result.training_time_sec = time.time() - start_time

            logger.info("=" * 70)
            logger.info("✅ 训练流程完成！")
            logger.info("=" * 70)
            logger.info(f"训练时长: {result.training_time_sec:.1f}秒")
            logger.info(f"NeMo模型: {result.nemo_model_path}")
            logger.info(f"ONNX模型: {result.onnx_model_path}")
            logger.info(f"配置文件: {result.config_path}")
            logger.info(f"训练报告: {result.report_path}")

        except Exception as e:
            logger.error(f"训练失败: {e}", exc_info=True)
            result.error_message = str(e)
            result.training_time_sec = time.time() - start_time

        return result

    def _create_manifests(
        self,
        audio_files: list,
        output_dir: Path
    ) -> tuple:
        """
        创建NeMo清单文件

        Args:
            audio_files: 音频文件列表
            output_dir: 输出目录

        Returns:
            (train_manifest, val_manifest)
        """
        import random
        import json

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # 打乱顺序
        random.shuffle(audio_files)

        # 划分数据集
        total = len(audio_files)
        train_end = int(total * 0.8)
        val_end = int(total * 0.9)

        train_files = audio_files[:train_end]
        val_files = audio_files[train_end:val_end]

        # 创建训练清单
        train_manifest = output_dir / "train_manifest.json"
        with open(train_manifest, 'w', encoding='utf-8') as f:
            for audio_file in train_files:
                # 从文件名推断标签
                if 'positive' in audio_file.name:
                    label = self.config.label_name
                else:
                    label = '_background_noise_'

                entry = {
                    "audio_filepath": str(audio_file),
                    "label": label,
                    "duration": 1.0  # 简化，实际应该读取文件获取时长
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        # 创建验证清单
        val_manifest = output_dir / "val_manifest.json"
        with open(val_manifest, 'w', encoding='utf-8') as f:
            for audio_file in val_files:
                if 'positive' in audio_file.name:
                    label = self.config.label_name
                else:
                    label = '_background_noise_'

                entry = {
                    "audio_filepath": str(audio_file),
                    "label": label,
                    "duration": 1.0
                }
                f.write(json.dumps(entry, ensure_ascii=False) + '\n')

        logger.info(f"训练清单: {len(train_files)} 个样本")
        logger.info(f"验证清单: {len(val_files)} 个样本")

        return train_manifest, val_manifest

    async def _update_system_config(self, result: TrainingResult) -> str:
        """更新WakeFusion系统配置"""
        config_path = self.model_dir / f"config_{self.config.label_name}.yaml"

        config_data = {
            "audio": {
                "device_match": "XVF3800",
                "capture_sample_rate": 48000,
                "work_sample_rate": 16000,
                "frame_ms": 20,
                "ring_buffer_sec": 2.0,
                "pre_roll_ms": 800,
                "channels": 1
            },
            "kws": {
                "enabled": True,
                "engine": "matchboxnet",
                "model": "matchboxnet",
                "model_name": "local",
                "model_path": str(result.onnx_model_path),
                "device": self.config.device,
                "keyword": self.config.label_name,
                "threshold": result.recommended_threshold,
                "cooldown_ms": 1200
            },
            "vad": {
                "enabled": True,
                "model": "webrtcvad",
                "speech_start_ms": 120,
                "speech_end_ms": 500
            },
            "vision": {
                "enabled": True,
                "gate_on_kws_only": True,
                "cache_ms": 600,
                "target_fps": 15,
                "distance_m_max": 4.0,
                "face_conf_min": 0.55
            },
            "fusion": {
                "probation_enabled": True,
                "probation_ms": 1000,
                "barge_in_enabled": True
            },
            "runtime": {
                "health_interval_sec": 2,
                "log_level": "INFO",
                "websocket_port": 8765,
                "health_port": 8080
            }
        }

        with open(config_path, 'w', encoding='utf-8') as f:
            yaml.dump(config_data, f, allow_unicode=True, default_flow_style=False)

        return str(config_path)

    async def _generate_report(self, result: TrainingResult) -> str:
        """生成训练报告"""
        report_path = self.report_dir / f"training_report_{self.config.label_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"

        report_content = f"""# 训练报告 - {self.config.wake_word}

## 基本信息

- **唤醒词**: {self.config.wake_word}
- **标签名**: {self.config.label_name}
- **训练时间**: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
- **训练时长**: {result.training_time_sec:.1f}秒
- **总样本数**: {result.total_samples}

## 训练配置

### TTS配置
- 音色数量: {len(self.config.tts.voices)}
- 每个音色样本数: {self.config.tts.samples_per_voice}
- 负样本: {', '.join(self.config.negative_samples) if self.config.negative_samples else '无'}

### 数据增强
- 启用: {'是' if self.config.augmentation.enabled else '否'}
- 速度扰动: {'是' if self.config.augmentation.speed_perturbation else '否'}
- 增益扰动: {'是' if self.config.augmentation.gain_perturbation else '否'}
- 噪声注入: {'是' if self.config.augmentation.noise_injection else '否'}

### NeMo配置
- 模型架构: MatchboxNet {self.config.training.num_blocks}x{self.config.training.num_layers_per_block}x{self.config.training.channels}
- Batch Size: {self.config.training.batch_size}
- Epochs: {self.config.training.num_epochs}
- Learning Rate: {self.config.training.learning_rate}

## 训练结果

### 训练指标
- **训练准确率**: {result.train_accuracy*100:.2f}%
- **验证准确率**: {result.val_accuracy*100:.2f}%
- **最佳Epoch**: {result.best_epoch}

### 测试指标
- **测试准确率**: {result.test_accuracy*100:.2f}%
- **召回率**: {result.test_recall*100:.2f}%
- **精确率**: {result.test_precision*100:.2f}%
- **平均延迟**: {result.avg_latency_ms:.2f}ms
- **推荐阈值**: {result.recommended_threshold:.2f}

## 模型文件

- **NeMo模型**: `{result.nemo_model_path}`
- **ONNX模型**: `{result.onnx_model_path}`
- **配置文件**: `{result.config_path}`

## 使用说明

### 1. 配置系统

编辑 `config/config.yaml`:

```yaml
kws:
  enabled: true
  engine: "matchboxnet"
  model_name: "local"
  model_path: "{result.onnx_model_path}"
  keyword: "{self.config.label_name}"
  threshold: {result.recommended_threshold}
```

### 2. 测试模型

```bash
# 使用麦克风测试
python tests/test_matchboxnet_microphone.py

# 运行完整系统
python -m wakefusion.runtime
```

### 3. 性能调优

如果误触发率高：
- 提高 `threshold` (当前: {result.recommended_threshold:.2f})
- 增加 `cooldown_ms`

如果漏检率高：
- 降低 `threshold`
- 重新训练（增加样本数）

## 结论

训练{'成功' if result.success else '失败'}。

{'模型已准备就绪，可以集成到系统使用。' if result.success else f'失败原因: {result.error_message}'}
"""

        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)

        return str(report_path)
