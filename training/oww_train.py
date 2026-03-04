"""
OpenWakeWord 唤醒词模型训练脚本 V2 (1D-CNN + Focal Loss)
=========================================================
核心改进（相比 V1 DNN + BCELoss）：

1. 架构升级：DNN (Flatten→Dense) → 1D-CNN
   - DNN 把 16帧×96维 拍平为 1536 维，完全丢失时序信息。
   - CNN 沿时间轴卷积，能学到 "你好→小康" 的先后顺序特征，
     从而区分 "你好小康" 与 "你好你好"、"小康小康" 等干扰音。

2. 损失函数升级：BCELoss → Focal Loss
   - alpha=0.25：正样本权重低于负样本，模型被迫更保守地预测唤醒，
     即减少假阳性（误唤醒）。
   - gamma=2.0：聚焦困难样本（如与唤醒词相似的短语），加强对边界的判别。

模型架构：
  输入: (batch, 16, 96) — 16 帧 × 96 维嵌入向量
  ↓ Transpose → (batch, 96, 16)
  ↓ Conv1d(96→128) + BN + ReLU + Dropout(0.3)
  ↓ Conv1d(128→64) + BN + ReLU + Dropout(0.25)
  ↓ Conv1d(64→32) + BN + ReLU
  ↓ GlobalAvgPool → Flatten → (batch, 32)
  ↓ Linear(32→16) + ReLU + Dropout(0.2)
  ↓ Linear(16→1) + Sigmoid
  输出: (batch, 1) — 唤醒概率 [0, 1]

前置条件：
  - 已运行 python training/oww_prepare_features.py
  - custom_dataset/oww_features/ 中有 features.npy 和 labels.npy

运行环境：wakefusion 虚拟环境
运行命令：python training/oww_train.py
"""
import os
import sys
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader

# ================= 配置区 =================
FEATURES_DIR = "custom_dataset/oww_features"
FEATURES_FILE = os.path.join(FEATURES_DIR, "features.npy")
LABELS_FILE = os.path.join(FEATURES_DIR, "labels.npy")

OUTPUT_MODEL = "xiaokang_oww.onnx"      # 导出的 ONNX 模型路径

N_EMBEDDINGS = 16       # 每个样本的嵌入帧数（由 embed_clips 决定）
EMBEDDING_DIM = 96      # 嵌入向量维度（Google speech_embedding 固定为 96）

SPLIT_RATIO = 0.8       # 80% 训练集，20% 验证集
RANDOM_SEED = 42
MAX_EPOCHS = 150        # 增加上限，配合更大 patience
BATCH_SIZE = 64
LEARNING_RATE = 0.001
PATIENCE = 20           # Early Stopping 耐心值（增大，CNN 收敛更慢）

# Focal Loss 参数
FOCAL_ALPHA = 0.25      # 正样本权重（< 0.5 = 更保守 = 减少误唤醒）
FOCAL_GAMMA = 2.0       # 聚焦系数（聚焦难分类的边界样本）
# ==========================================


class WakeWordDataset(Dataset):
    """PyTorch 数据集：特征 + 标签"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels).unsqueeze(1)

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]


class FocalLoss(nn.Module):
    """
    Focal Loss（Lin et al., 2017）
    ================================
    相比普通 BCELoss 的核心优势：
    1. alpha_t 参数：正样本权重 alpha，负样本权重 (1-alpha)。
       alpha=0.25 → 负样本损失权重为 0.75，远大于正样本的 0.25。
       → 模型被迫更关注"把负样本预测正确"，即减少假阳性（误唤醒）。
    2. (1 - p_t)^gamma 参数：对已经预测正确且置信度高的样本降权，
       专注于预测错误或不确定的困难样本（如 "你好你好" 这类近似音）。

    注：inputs 为 Sigmoid 输出，范围 [0,1]。
    """
    def __init__(self, alpha=0.25, gamma=2.0):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma

    def forward(self, inputs, targets):
        bce = F.binary_cross_entropy(inputs, targets, reduction='none')
        p_t = inputs * targets + (1 - inputs) * (1 - targets)
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        focal_weight = alpha_t * (1 - p_t) ** self.gamma
        return (focal_weight * bce).mean()


class WakeWordCNN(nn.Module):
    """
    1D-CNN 唤醒词分类器（时序感知架构）
    =====================================
    核心改进点：
    - 将 16 帧嵌入视为"时间序列"，沿时间轴做卷积。
    - BatchNorm 提升训练稳定性，防止过拟合。
    - GlobalAvgPool 代替 Flatten，对时间维度做平均池化，
      增强对时序位移的鲁棒性（说话快慢不同影响较小）。

    输入: (batch, n_frames, embedding_dim) = (batch, 16, 96)
    输出: (batch, 1) 概率值 [0, 1]
    """
    def __init__(self, n_frames=16, embedding_dim=96):
        super().__init__()

        self.conv_layers = nn.Sequential(
            # Block 1: (batch, 96, 16) → (batch, 128, 16)
            nn.Conv1d(embedding_dim, 128, kernel_size=3, padding=1),
            nn.BatchNorm1d(128),
            nn.ReLU(),
            nn.Dropout(0.3),

            # Block 2: (batch, 128, 16) → (batch, 64, 16)
            nn.Conv1d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm1d(64),
            nn.ReLU(),
            nn.Dropout(0.25),

            # Block 3: (batch, 64, 16) → (batch, 32, 16)
            nn.Conv1d(64, 32, kernel_size=3, padding=1),
            nn.BatchNorm1d(32),
            nn.ReLU(),
        )

        self.classifier = nn.Sequential(
            nn.AdaptiveAvgPool1d(1),   # (batch, 32, 1) — 全局平均池化
            nn.Flatten(),               # (batch, 32)
            nn.Linear(32, 16),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(16, 1),
            nn.Sigmoid(),
        )

    def forward(self, x):
        # x: (batch, n_frames, embedding_dim) → (batch, embedding_dim, n_frames)
        x = x.transpose(1, 2)
        x = self.conv_layers(x)
        return self.classifier(x)


def train_epoch(model, dataloader, criterion, optimizer, device):
    """训练一个 epoch"""
    model.train()
    total_loss = 0
    correct = 0
    total = 0

    for features, labels in dataloader:
        features, labels = features.to(device), labels.to(device)

        optimizer.zero_grad()
        outputs = model(features)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item() * features.size(0)
        predicted = (outputs > 0.5).float()
        correct += (predicted == labels).sum().item()
        total += labels.size(0)

    return total_loss / total, correct / total


def evaluate(model, dataloader, criterion, device):
    """验证评估"""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    tp = fp = tn = fn = 0

    with torch.no_grad():
        for features, labels in dataloader:
            features, labels = features.to(device), labels.to(device)

            outputs = model(features)
            loss = criterion(outputs, labels)

            total_loss += loss.item() * features.size(0)
            predicted = (outputs > 0.5).float()
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

            tp += ((predicted == 1) & (labels == 1)).sum().item()
            fp += ((predicted == 1) & (labels == 0)).sum().item()
            tn += ((predicted == 0) & (labels == 0)).sum().item()
            fn += ((predicted == 0) & (labels == 1)).sum().item()

    accuracy = correct / total if total > 0 else 0
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0

    return total_loss / total, accuracy, precision, recall, f1


def export_onnx(model, output_path, n_embeddings, embedding_dim):
    """导出为 ONNX 格式（兼容 openWakeWord 推理）"""
    model.eval()
    model.cpu()

    dummy_input = torch.randn(1, n_embeddings, embedding_dim)

    torch.onnx.export(
        model,
        dummy_input,
        output_path,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={
            "input": {0: "batch_size"},
            "output": {0: "batch_size"}
        },
        opset_version=11,
    )


def main():
    print("=" * 65)
    print("🚀 OpenWakeWord 唤醒词模型训练 V2 (1D-CNN + Focal Loss)")
    print("=" * 65)

    # 1. 加载特征
    if not os.path.exists(FEATURES_FILE) or not os.path.exists(LABELS_FILE):
        print(f"❌ 找不到特征文件！")
        print(f"   请先运行: python training/oww_prepare_features.py")
        return

    print("\n📦 加载特征数据...")
    features = np.load(FEATURES_FILE)
    labels = np.load(LABELS_FILE)

    n_pos = int((labels == 1).sum())
    n_neg = int((labels == 0).sum())
    print(f"   总计: {len(labels)} 条 (正样本 {n_pos} + 负样本 {n_neg}，比例 1:{n_neg/n_pos:.1f})")
    print(f"   特征维度: {features.shape}")

    actual_n_embeddings = features.shape[1]
    actual_embedding_dim = features.shape[2]
    print(f"   检测到: {actual_n_embeddings} 帧 × {actual_embedding_dim} 维嵌入")

    # 2. 分割训练集/验证集
    print(f"\n✂️ 分割数据集 ({SPLIT_RATIO:.0%} 训练 / {1 - SPLIT_RATIO:.0%} 验证)...")
    np.random.seed(RANDOM_SEED)
    indices = np.random.permutation(len(labels))

    split_idx = int(len(labels) * SPLIT_RATIO)
    train_indices = indices[:split_idx]
    val_indices = indices[split_idx:]

    train_features, train_labels = features[train_indices], labels[train_indices]
    val_features, val_labels = features[val_indices], labels[val_indices]

    train_pos = int((train_labels == 1).sum())
    train_neg = int((train_labels == 0).sum())
    val_pos = int((val_labels == 1).sum())
    val_neg = int((val_labels == 0).sum())

    print(f"   训练集: {len(train_labels)} 条 (正 {train_pos} + 负 {train_neg})")
    print(f"   验证集: {len(val_labels)} 条 (正 {val_pos} + 负 {val_neg})")

    # 3. 创建 DataLoader
    train_dataset = WakeWordDataset(train_features, train_labels)
    val_dataset = WakeWordDataset(val_features, val_labels)

    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # 4. 创建模型
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = WakeWordCNN(n_frames=actual_n_embeddings, embedding_dim=actual_embedding_dim).to(device)

    criterion = FocalLoss(alpha=FOCAL_ALPHA, gamma=FOCAL_GAMMA)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=0.001)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=MAX_EPOCHS, eta_min=1e-6)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"\n🧠 模型架构: 1D-CNN (时序感知)")
    print(f"   参数总量: {total_params:,} (约 {total_params * 4 / 1024:.1f} KB)")
    print(f"   设备: {device}")
    print(f"   损失函数: Focal Loss (alpha={FOCAL_ALPHA}, gamma={FOCAL_GAMMA})")
    print(f"   学习率: {LEARNING_RATE}  Early Stopping 耐心: {PATIENCE} 轮")

    # 5. 训练循环
    print(f"\n🔥 开始训练 (最大 {MAX_EPOCHS} 轮)")
    print("-" * 80)
    print(f"{'Epoch':>5} | {'Train Loss':>10} | {'Train Acc':>9} | {'Val Loss':>9} | "
          f"{'Val Acc':>8} | {'Prec':>6} | {'Recall':>6} | {'F1':>6} | {'状态'}")
    print("-" * 80)

    best_val_loss = float('inf')
    best_val_f1 = 0
    patience_counter = 0
    best_state_dict = None

    for epoch in range(1, MAX_EPOCHS + 1):
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_prec, val_recall, val_f1 = evaluate(model, val_loader, criterion, device)
        scheduler.step()

        status = ""
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            best_val_f1 = val_f1
            best_state_dict = {k: v.clone() for k, v in model.state_dict().items()}
            patience_counter = 0
            status = "⭐ Best"
        else:
            patience_counter += 1
            if patience_counter >= PATIENCE:
                status = "🛑 Stop"
            elif patience_counter >= PATIENCE - 5:
                status = f"⚠️ {patience_counter}/{PATIENCE}"

        print(f"{epoch:>5} | {train_loss:>10.4f} | {train_acc:>8.1%} | {val_loss:>9.4f} | "
              f"{val_acc:>7.1%} | {val_prec:>5.3f} | {val_recall:>5.3f} | {val_f1:>5.3f} | {status}")

        if patience_counter >= PATIENCE:
            print(f"\n🛑 Early Stopping: 连续 {PATIENCE} 轮无改善，停止训练")
            break

    # 6. 恢复最佳权重
    if best_state_dict:
        model.load_state_dict(best_state_dict)
        print(f"\n📦 已恢复最佳模型权重 (val_loss={best_val_loss:.4f}, F1={best_val_f1:.3f})")

    # 7. 最终评估
    print(f"\n📊 最终验证集评估 (阈值=0.5):")
    val_loss, val_acc, val_prec, val_recall, val_f1 = evaluate(model, val_loader, criterion, device)
    print(f"   准确率 (Accuracy):  {val_acc:.2%}")
    print(f"   精确率 (Precision): {val_prec:.2%}  ← 低精确率 = 高误唤醒")
    print(f"   召回率 (Recall):    {val_recall:.2%}  ← 低召回率 = 难以唤醒")
    print(f"   F1 分数:            {val_f1:.3f}")

    # 8. 导出 ONNX
    print(f"\n📤 导出 ONNX 模型: {OUTPUT_MODEL}")
    export_onnx(model, OUTPUT_MODEL, actual_n_embeddings, actual_embedding_dim)

    try:
        import onnxruntime as ort
        session = ort.InferenceSession(OUTPUT_MODEL)
        inp = session.get_inputs()[0]
        out = session.get_outputs()[0]
        print(f"   ✅ ONNX 验证通过")
        print(f"   输入: {inp.name} shape={inp.shape}")
        print(f"   输出: {out.name} shape={out.shape}")

        test_input = np.random.randn(1, actual_n_embeddings, actual_embedding_dim).astype(np.float32)
        result = session.run(None, {inp.name: test_input})[0]
        print(f"   随机输入测试输出: {result[0][0]:.4f} (应在 0~1 之间)")
    except Exception as e:
        print(f"   ⚠️ ONNX 验证跳过: {e}")

    print(f"\n{'=' * 65}")
    print(f"🎉 训练完成！")
    print(f"   模型: {os.path.abspath(OUTPUT_MODEL)}")
    print(f"   大小: {os.path.getsize(OUTPUT_MODEL) / 1024:.1f} KB")
    print(f"{'=' * 65}")
    print(f"\n💡 下一步:")
    print(f"   python tests/test_compare_models.py  (NeMo vs OWW 对比测试)")


if __name__ == "__main__":
    main()
