"""
WakeFusion 模型训练脚本 V3 (终极稳定版)
完整流程：加载预训练模型 → 恢复验证 → EarlyStopping → 保存工业级模型

核心改进（相比旧版）：
1. 恢复验证集监控 + EarlyStopping 防止过拟合
2. 猴子补丁绕过 NeMo evaluation_step 的 correct_counts_k 报错
3. 统一 2 秒音频，与推理 BUFFER_DURATION 一致
4. Windows 兼容：num_workers=0
5. ASCII 编码清单，避免 GBK 报错

运行环境：wakefusion 虚拟环境（需要 GPU）
运行命令：python fast_train.py

前置条件：
  1. 已运行 training/tts_positive_generator.py 生成 TTS 正样本
  2. 已运行 training/positive_data_factory.py 数据增强
  3. 已运行 training/tts_negative_generator.py 生成 TTS 负样本
  4. 已运行 training/split_dataset.py 分割数据集
"""
import os
import sys
import json
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import OmegaConf

# 减少无关日志输出
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
torch.set_float32_matmul_precision('medium')

try:
    from nemo.collections.asr.models import EncDecClassificationModel
except ImportError:
    print("❌ 无法导入 NeMo，请确保在 wakefusion 虚拟环境中运行！")
    sys.exit(1)


# ================= 配置区 =================
TRAIN_MANIFEST = os.path.join("custom_dataset", "manifests", "train_manifest.json")
VAL_MANIFEST = os.path.join("custom_dataset", "manifests", "val_manifest.json")
OUTPUT_MODEL = "xiaokang_xvf3800_pro.nemo"

LABELS = ['others', 'xiaokang']  # 标签顺序必须固定
MAX_EPOCHS = 60                   # 最大训练轮数（EarlyStopping 会提前停止）
BATCH_SIZE = 32
LEARNING_RATE = 0.001             # 初始学习率
PATIENCE = 10                     # EarlyStopping 耐心值：连续 10 轮无改善则停止
# ==========================================


# --- 猴子补丁：绕过 NeMo evaluation_step 的 correct_counts_k 报错 ---
def patched_evaluation_step(self, batch, batch_idx, dataloader_idx=0, tag='val'):
    """
    替换 NeMo 原有的 evaluation_step，避免 correct_counts_k 维度冲突。
    只计算 val_loss，不计算准确率指标（避免 NeMo 内部 Bug）。
    """
    if isinstance(batch, (list, tuple)):
        if len(batch) >= 3:
            # NeMo batch 顺序: (audio_signal, audio_signal_length, label)
            signal, signal_len, labels = batch[0], batch[1], batch[2]
        else:
            signal, labels = batch[0], batch[1]
            signal_len = torch.tensor([signal.shape[1]] * signal.shape[0], device=signal.device)
    else:
        signal = batch['audio_signal']
        signal_len = batch['audio_signal_length']
        labels = batch['labels']

    logits = self.forward(input_signal=signal, input_signal_length=signal_len)
    loss = self.loss(logits=logits, labels=labels)
    self.log(f'{tag}_loss', loss, prog_bar=True, sync_dist=True)
    return {'loss': loss}


def patched_multi_evaluation_epoch_end(self, outputs, dataloader_idx=0, tag='val'):
    """替换 NeMo 原有的 multi_evaluation_epoch_end，避免访问不存在的属性"""
    avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    self.log(f'{tag}_loss_epoch', avg_loss, prog_bar=True, sync_dist=True)


def main():
    print("=" * 60)
    print("🚀 WakeFusion 模型训练 V3 (验证+EarlyStopping 版)")
    print("=" * 60)

    # 1. 检查数据清单是否存在
    if not os.path.exists(TRAIN_MANIFEST):
        print(f"❌ 找不到训练集清单: {TRAIN_MANIFEST}")
        print("请先运行 training/split_dataset.py 生成数据清单！")
        return

    if not os.path.exists(VAL_MANIFEST):
        print(f"❌ 找不到验证集清单: {VAL_MANIFEST}")
        print("请先运行 training/split_dataset.py 生成数据清单！")
        return

    # 2. 统计数据集信息
    train_count = sum(1 for _ in open(TRAIN_MANIFEST, 'r', encoding='utf-8'))
    val_count = sum(1 for _ in open(VAL_MANIFEST, 'r', encoding='utf-8'))
    print(f"\n📊 数据集统计:")
    print(f"   训练集: {train_count} 条")
    print(f"   验证集: {val_count} 条")

    # 3. 加载预训练模型
    print("\n📦 正在加载预训练 MatchboxNet 模型...")
    model = EncDecClassificationModel.from_pretrained(
        model_name="commandrecognition_en_matchboxnet3x1x64_v1"
    )

    # 4. 替换分类头
    model.change_labels(new_labels=LABELS)
    print(f"🎯 分类标签已设置为: {LABELS}")

    # 5. 注入猴子补丁（绕过 NeMo 内部 Bug）
    model.evaluation_step = patched_evaluation_step.__get__(model, EncDecClassificationModel)
    model.multi_evaluation_epoch_end = patched_multi_evaluation_epoch_end.__get__(
        model, EncDecClassificationModel
    )
    print("💉 猴子补丁已注入：安全绕过 NeMo 计分板 Bug")

    # 6. 配置训练数据集
    train_config = {
        "manifest_filepath": os.path.abspath(TRAIN_MANIFEST),
        "sample_rate": 16000,
        "labels": LABELS,
        "batch_size": BATCH_SIZE,
        "shuffle": True,
        "num_workers": 0,    # Windows 兼容
        "pin_memory": True,
    }
    model.setup_training_data(train_data_layer_config=OmegaConf.create(train_config))

    # 7. 配置验证数据集
    val_config = {
        "manifest_filepath": os.path.abspath(VAL_MANIFEST),
        "sample_rate": 16000,
        "labels": LABELS,
        "batch_size": BATCH_SIZE,
        "shuffle": False,
        "num_workers": 0,    # Windows 兼容
        "pin_memory": True,
    }
    model.setup_validation_data(val_data_layer_config=OmegaConf.create(val_config))

    # 8. 配置 EarlyStopping
    early_stop_callback = EarlyStopping(
        monitor='val_loss',     # 监控验证集损失
        patience=PATIENCE,      # 连续 N 轮无改善则停止
        mode='min',             # 损失越小越好
        verbose=True,           # 打印停止信息
    )

    # 9. 配置 ModelCheckpoint（保存最佳模型权重）
    ckpt_dir = os.path.join("training_checkpoints", "fast_train")
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='best-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',     # 监控验证集损失
        mode='min',             # 损失越小越好
        save_top_k=1,           # 只保留最佳的 1 个 checkpoint
        verbose=True,
    )

    # 10. 创建训练器
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=MAX_EPOCHS,
        num_sanity_val_steps=0,          # 跳过启动时的验证自检（防 CUDA 粘性错误）
        enable_checkpointing=True,       # 启用 checkpoint
        logger=False,                    # 减少日志输出
        log_every_n_steps=10,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    # 11. 开始训练
    print(f"\n🔥 训练引擎启动！")
    print(f"   最大轮数: {MAX_EPOCHS}")
    print(f"   EarlyStopping 耐心: {PATIENCE} 轮")
    print(f"   批次大小: {BATCH_SIZE}")
    print(f"   验证频率: 每个 epoch 结束后验证一次")
    print(f"   最佳模型保存至: {ckpt_dir}")
    print("=" * 60 + "\n")

    trainer.fit(model)

    # 12. 从最佳 checkpoint 恢复权重，再保存为 .nemo
    best_ckpt = checkpoint_callback.best_model_path
    best_score = checkpoint_callback.best_model_score
    if best_ckpt and os.path.exists(best_ckpt):
        print(f"\n📦 从最佳 checkpoint 恢复权重: {best_ckpt}")
        print(f"   最佳 val_loss: {best_score:.4f}")
        # 加载最佳权重到当前模型
        checkpoint = torch.load(best_ckpt, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        print("\n⚠️ 未找到最佳 checkpoint，使用最后一轮的权重保存")

    model.save_to(OUTPUT_MODEL)

    print("\n" + "=" * 60)
    print(f"🎉 训练完成！")
    print(f"   停止轮次: {trainer.current_epoch}")
    if best_score is not None:
        print(f"   最佳 val_loss: {best_score:.4f}")
    print(f"   模型已保存至: {os.path.abspath(OUTPUT_MODEL)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
