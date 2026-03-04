"""
WakeFusion 数据集分割器 (Dataset Splitter)
自动扫描 custom_dataset/xiaokang 和 custom_dataset/others，
按"组"进行 80/20 分割，生成 NeMo 格式的 train_manifest.json 和 val_manifest.json。

核心规则：同一条原始录音的所有增强版本（pitch/speed/aug）必须全部在训练集或全部在验证集，
          防止模型通过识别增强痕迹在验证集上"作弊"。

运行环境：wakefusion 虚拟环境
运行命令：python training/split_dataset.py
"""
import os
import re
import json
import wave
import random
from collections import defaultdict

# ================= 配置区 =================
DATASET_DIR = "custom_dataset"
XIAOKANG_DIR = os.path.join(DATASET_DIR, "xiaokang")
OTHERS_DIR = os.path.join(DATASET_DIR, "others")

OUTPUT_DIR = os.path.join(DATASET_DIR, "manifests")
TRAIN_MANIFEST = os.path.join(OUTPUT_DIR, "train_manifest.json")
VAL_MANIFEST = os.path.join(OUTPUT_DIR, "val_manifest.json")

SPLIT_RATIO = 0.8  # 80% 训练集，20% 验证集
RANDOM_SEED = 42    # 固定种子，保证每次分割结果一致（可修改）
# ==========================================

# 增强后缀的正则匹配（用于提取原始文件的"组名"）
AUG_SUFFIXES = re.compile(
    r'(_pitch_(?:up|down)\d+'
    r'|_speed_(?:fast|slow)'
    r'|_aug_(?:quiet|noisy))'
)


def get_group_key(filename):
    """
    提取文件的"组名"，即去掉所有增强后缀后的基础文件名。
    
    例如：
      real_xiaokang_1772011215.wav                    → real_xiaokang_1772011215
      real_xiaokang_1772011215_pitch_up4.wav           → real_xiaokang_1772011215
      real_xiaokang_1772011215_pitch_up4_aug_quiet.wav → real_xiaokang_1772011215
      tts_pos_XiaoxiaoNeural_normal_abc12345.wav       → tts_pos_XiaoxiaoNeural_normal_abc12345
      tts_pos_XiaoxiaoNeural_normal_abc12345_aug_noisy.wav → tts_pos_XiaoxiaoNeural_normal_abc12345
    """
    base = filename.replace(".wav", "")
    # 反复去除增强后缀，直到没有可去除的
    cleaned = AUG_SUFFIXES.sub("", base)
    while cleaned != base:
        base = cleaned
        cleaned = AUG_SUFFIXES.sub("", base)
    return cleaned


def get_wav_duration(wav_path):
    """获取 WAV 文件的时长（秒）"""
    try:
        with wave.open(wav_path, 'r') as f:
            return f.getnframes() / float(f.getframerate())
    except Exception:
        # 如果 wave 模块无法打开（比如非标准 WAV），回退到 0
        return 0.0


def scan_directory(directory, label):
    """
    扫描目录，返回 {group_key: [file_entry, ...]} 的分组字典
    每个 file_entry = {"audio_filepath": abs_path, "duration": float, "label": str}
    """
    groups = defaultdict(list)

    if not os.path.exists(directory):
        print(f"⚠️ 目录不存在: {directory}")
        return groups

    wav_files = [f for f in os.listdir(directory) if f.endswith('.wav')]

    for filename in wav_files:
        filepath = os.path.abspath(os.path.join(directory, filename))
        duration = get_wav_duration(filepath)

        if duration <= 0:
            print(f"⚠️ 跳过无效文件: {filename}")
            continue

        group_key = get_group_key(filename)
        groups[group_key].append({
            "audio_filepath": filepath,
            "duration": round(duration, 3),
            "label": label
        })

    return groups


def split_groups(groups, ratio, seed):
    """
    按组进行随机分割。
    返回 (train_entries, val_entries)
    """
    random.seed(seed)

    group_keys = list(groups.keys())
    random.shuffle(group_keys)

    split_idx = int(len(group_keys) * ratio)
    train_keys = group_keys[:split_idx]
    val_keys = group_keys[split_idx:]

    train_entries = []
    val_entries = []

    for key in train_keys:
        train_entries.extend(groups[key])
    for key in val_keys:
        val_entries.extend(groups[key])

    return train_entries, val_entries


def write_manifest(entries, filepath):
    """写入 NeMo 格式的 manifest 文件（JSONL 格式，ASCII 编码）"""
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    with open(filepath, 'w', encoding='utf-8') as f:
        for entry in entries:
            f.write(json.dumps(entry, ensure_ascii=True) + '\n')


def main():
    print("=" * 60)
    print("📊 WakeFusion 数据集分割器")
    print(f"分割比例: {SPLIT_RATIO:.0%} 训练 / {1 - SPLIT_RATIO:.0%} 验证")
    print(f"随机种子: {RANDOM_SEED}")
    print("=" * 60)

    # 1. 扫描正样本和负样本目录
    print("\n🔍 正在扫描正样本目录...")
    pos_groups = scan_directory(XIAOKANG_DIR, "xiaokang")
    total_pos = sum(len(v) for v in pos_groups.values())
    print(f"   ├── 发现 {len(pos_groups)} 组原始正样本")
    print(f"   └── 共计 {total_pos} 条文件（含增强版本）")

    print("\n🔍 正在扫描负样本目录...")
    neg_groups = scan_directory(OTHERS_DIR, "others")
    total_neg = sum(len(v) for v in neg_groups.values())
    print(f"   ├── 发现 {len(neg_groups)} 组原始负样本")
    print(f"   └── 共计 {total_neg} 条文件（含增强版本）")

    if total_pos == 0 and total_neg == 0:
        print("\n❌ 没有找到任何数据！请先运行数据生成脚本。")
        return

    # 2. 分别对正/负样本按组分割
    print("\n✂️ 正在按组分割数据集...")
    pos_train, pos_val = split_groups(pos_groups, SPLIT_RATIO, RANDOM_SEED)
    neg_train, neg_val = split_groups(neg_groups, SPLIT_RATIO, RANDOM_SEED + 1)  # 不同种子避免对齐

    train_entries = pos_train + neg_train
    val_entries = pos_val + neg_val

    # 3. 打乱顺序
    random.seed(RANDOM_SEED + 2)
    random.shuffle(train_entries)
    random.shuffle(val_entries)

    # 4. 写入 manifest 文件
    write_manifest(train_entries, TRAIN_MANIFEST)
    write_manifest(val_entries, VAL_MANIFEST)

    # 5. 打印统计信息
    train_pos = len([e for e in train_entries if e["label"] == "xiaokang"])
    train_neg = len([e for e in train_entries if e["label"] == "others"])
    val_pos = len([e for e in val_entries if e["label"] == "xiaokang"])
    val_neg = len([e for e in val_entries if e["label"] == "others"])

    print("\n" + "=" * 60)
    print("📋 分割结果统计")
    print("=" * 60)
    print(f"训练集: {len(train_entries)} 条")
    print(f"   ├── xiaokang (正): {train_pos} 条")
    print(f"   └── others   (负): {train_neg} 条")
    print(f"验证集: {len(val_entries)} 条")
    print(f"   ├── xiaokang (正): {val_pos} 条")
    print(f"   └── others   (负): {val_neg} 条")
    print(f"\n📁 训练集清单: {os.path.abspath(TRAIN_MANIFEST)}")
    print(f"📁 验证集清单: {os.path.abspath(VAL_MANIFEST)}")

    # 6. 数据平衡性检查
    total_pos_all = train_pos + val_pos
    total_neg_all = train_neg + val_neg
    if total_pos_all > 0 and total_neg_all > 0:
        ratio = total_neg_all / total_pos_all
        if ratio > 5:
            print(f"\n⚠️ 警告：正负样本比例为 1:{ratio:.1f}，负样本远多于正样本。")
            print("   建议增加更多正样本，或在训练时使用加权采样。")
        elif ratio < 0.2:
            print(f"\n⚠️ 警告：正负样本比例为 1:{ratio:.1f}，正样本远多于负样本。")
            print("   建议增加更多负样本。")
        else:
            print(f"\n✅ 正负样本比例 1:{ratio:.1f}，数据平衡性良好。")

    print("\n🎉 分割完成！")
    print("💡 下一步：运行 fast_train.py 开始训练模型")


if __name__ == "__main__":
    main()
