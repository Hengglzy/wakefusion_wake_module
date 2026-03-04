"""
WakeFusion 真机微调脚本 V2 (Fine-tuning with Validation)
用于在已有模型基础上，注入新采集的真实环境数据进行增量微调。

使用场景：
  - 部署后发现某些环境/声音识别不佳
  - 新增了更多真人录音数据
  - 需要适配新的麦克风设备

工作流程：
  1. 把新数据放入 custom_dataset/xiaokang/ 和 custom_dataset/others/
  2. 运行 training/split_dataset.py 重新分割数据
  3. 运行本脚本进行微调

运行环境：wakefusion 虚拟环境（需要 GPU）
运行命令：python training/finetune_xvf3800.py
"""
import os
import sys
import json
import torch
import torch.nn as nn
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, ModelCheckpoint
from omegaconf import OmegaConf

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
torch.set_float32_matmul_precision('medium')

try:
    from nemo.collections.asr.models import EncDecClassificationModel
except ImportError:
    print("❌ 请在 wakefusion 虚拟环境中运行！")
    sys.exit(1)


# ================= 配置区 =================
MODEL_PATH = "xiaokang_xvf3800_pro.nemo"   # 被微调的模型
OUTPUT_MODEL = "xiaokang_xvf3800_pro.nemo"  # 微调后覆盖保存

TRAIN_MANIFEST = os.path.join("custom_dataset", "manifests", "train_manifest.json")
VAL_MANIFEST = os.path.join("custom_dataset", "manifests", "val_manifest.json")

LABELS = ['others', 'xiaokang']
MAX_EPOCHS = 60
BATCH_SIZE = 32
PATIENCE = 10                               # EarlyStopping 耐心值
# ==========================================


# --- 猴子补丁：绕过 NeMo evaluation_step 的 correct_counts_k 报错 ---
def patched_evaluation_step(self, batch, batch_idx, dataloader_idx=0, tag='val'):
    """安全的验证步骤，只计算 loss"""
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
    """安全的验证 epoch 结束回调"""
    avg_loss = torch.stack([x['loss'] for x in outputs]).mean()
    self.log(f'{tag}_loss_epoch', avg_loss, prog_bar=True, sync_dist=True)


def main():
    print("=" * 60)
    print("🔧 WakeFusion 真机微调 V2 (验证+EarlyStopping 版)")
    print("=" * 60)

    # 1. 检查模型文件
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到基础模型: {MODEL_PATH}")
        print("请先运行 fast_train.py 完成初始训练！")
        return

    # 2. 检查数据清单
    if not os.path.exists(TRAIN_MANIFEST):
        print(f"❌ 找不到训练集清单: {TRAIN_MANIFEST}")
        print("请先运行 training/split_dataset.py 生成数据清单！")
        return

    if not os.path.exists(VAL_MANIFEST):
        print(f"❌ 找不到验证集清单: {VAL_MANIFEST}")
        print("请先运行 training/split_dataset.py 生成数据清单！")
        return

    # 3. 统计数据
    train_count = sum(1 for _ in open(TRAIN_MANIFEST, 'r', encoding='utf-8'))
    val_count = sum(1 for _ in open(VAL_MANIFEST, 'r', encoding='utf-8'))
    print(f"\n📊 数据集统计:")
    print(f"   训练集: {train_count} 条")
    print(f"   验证集: {val_count} 条")

    # 4. 加载已有模型
    print(f"\n📦 正在加载已有模型: {MODEL_PATH}")
    model = EncDecClassificationModel.restore_from(MODEL_PATH)

    # 5. 注入猴子补丁
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
        "num_workers": 0,
        "pin_memory": True,
        # 🛠️ 优化点一：开启在线数据增强（防死记硬背）
        "augmentor": {
            "shift": {"prob": 0.5, "min_shift_ms": -5.0, "max_shift_ms": 5.0},
            "white_noise": {"prob": 0.5, "min_level": -90, "max_level": -46}
        }
    }
    model.setup_training_data(train_data_layer_config=OmegaConf.create(train_config))

    # 7. 配置验证数据集
    val_config = {
        "manifest_filepath": os.path.abspath(VAL_MANIFEST),
        "sample_rate": 16000,
        "labels": LABELS,
        "batch_size": BATCH_SIZE,
        "shuffle": False,
        "num_workers": 0,
        "pin_memory": True,
    }
    model.setup_validation_data(val_data_layer_config=OmegaConf.create(val_config))

    # 8. EarlyStopping
    early_stop_callback = EarlyStopping(
        monitor='val_loss',
        patience=PATIENCE,
        mode='min',
        verbose=True,
    )

    # 9. ModelCheckpoint（保存最佳模型权重）
    ckpt_dir = os.path.join("training_checkpoints", "finetune")
    os.makedirs(ckpt_dir, exist_ok=True)
    checkpoint_callback = ModelCheckpoint(
        dirpath=ckpt_dir,
        filename='best-{epoch:02d}-{val_loss:.4f}',
        monitor='val_loss',
        mode='min',
        save_top_k=1,
        verbose=True,
    )

    # 🛠️ 优化点二：配置微调专用的低学习率优化器（极其关键！）
    optim_args = {
        "name": "adamw",
        "lr": 0.0005,  # 微调时学习率要比从头训练时小一点
        "weight_decay": 0.001,
        "sched": {
            "name": "CosineAnnealing",
            "warmup_steps": 20, 
            "min_lr": 1e-6
        }
    }
    model.setup_optimization(optim_config=OmegaConf.create(optim_args))
    print("⚙️ 微调优化器已配置：低学习率 (0.0005) + CosineAnnealing 调度器")

    # 10. 训练器
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1,
        max_epochs=MAX_EPOCHS,
        num_sanity_val_steps=0,
        enable_checkpointing=True,
        logger=False,
        log_every_n_steps=10,
        callbacks=[early_stop_callback, checkpoint_callback],
    )

    # 11. 开始微调
    print(f"\n🔥 微调引擎启动！")
    print(f"   基础模型: {MODEL_PATH}")
    print(f"   最大轮数: {MAX_EPOCHS}")
    print(f"   EarlyStopping 耐心: {PATIENCE} 轮")
    print(f"   最佳模型保存至: {ckpt_dir}")
    print("=" * 60 + "\n")

    trainer.fit(model)

    # 12. 从最佳 checkpoint 恢复权重，再保存为 .nemo
    best_ckpt = checkpoint_callback.best_model_path
    best_score = checkpoint_callback.best_model_score
    if best_ckpt and os.path.exists(best_ckpt):
        print(f"\n📦 从最佳 checkpoint 恢复权重: {best_ckpt}")
        print(f"   最佳 val_loss: {best_score:.4f}")
        checkpoint = torch.load(best_ckpt, map_location='cpu')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
    else:
        print("\n⚠️ 未找到最佳 checkpoint，使用最后一轮的权重保存")

    model.save_to(OUTPUT_MODEL)

    print("\n" + "=" * 60)
    print(f"🎉 微调完成！")
    print(f"   停止轮次: {trainer.current_epoch}")
    if best_score is not None:
        print(f"   最佳 val_loss: {best_score:.4f}")
    print(f"   模型已保存至: {os.path.abspath(OUTPUT_MODEL)}")
    print("=" * 60)
    print("💡 下次启动 audio_service.py 时将自动加载新模型。")


if __name__ == "__main__":
    main()
