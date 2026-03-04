"""
OpenWakeWord 特征提取器 (OWW Feature Extractor)
================================================
将现有 custom_dataset 中的正/负样本音频，通过 Google speech_embedding 预训练模型
转换为 96 维嵌入向量，保存为 .npy 文件供后续训练使用。

处理流程：
  1. 加载音频 → 统一为 16kHz / 2秒 / int16 格式
  2. 通过 melspectrogram ONNX 模型 → 梅尔频谱
  3. 通过 embedding ONNX 模型 → 96 维嵌入向量
  4. 每个 2 秒音频产出 16 个嵌入向量 (16 × 96)

前置条件：
  - pip install openwakeword onnxruntime
  - custom_dataset/xiaokang 和 custom_dataset/others 目录中有音频文件

运行环境：wakefusion 虚拟环境
运行命令：python training/oww_prepare_features.py
"""
import os
import sys
import numpy as np
import librosa

# ================= 配置区 =================
SAMPLE_RATE = 16000
TARGET_DURATION = 2.0       # 统一音频时长（秒），与 NeMo 模型一致
TARGET_SAMPLES = int(SAMPLE_RATE * TARGET_DURATION)

XIAOKANG_DIR = "custom_dataset/xiaokang"
OTHERS_DIR = "custom_dataset/others"
OUTPUT_DIR = "custom_dataset/oww_features"

BATCH_SIZE = 64             # 批量提取特征时的批次大小
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


def pad_or_trim_int16(audio_float, target_len):
    """将 float32 音频统一为目标长度并转为 int16"""
    if len(audio_float) > target_len:
        audio_float = audio_float[:target_len]
    elif len(audio_float) < target_len:
        audio_float = np.pad(audio_float, (0, target_len - len(audio_float)))
    return (audio_float * 32767).astype(np.int16)


def load_audio_batch(directory, label):
    """加载目录下所有 WAV 文件，返回 int16 音频数组和标签"""
    if not os.path.exists(directory):
        print(f"  ⚠️ 目录不存在: {directory}")
        return np.array([]), np.array([])

    wav_files = sorted([f for f in os.listdir(directory) if f.endswith('.wav')])
    if not wav_files:
        print(f"  ⚠️ 目录中没有 WAV 文件: {directory}")
        return np.array([]), np.array([])

    print(f"  📂 {directory}: 发现 {len(wav_files)} 个文件")

    audio_list = []
    skipped = 0

    for idx, filename in enumerate(wav_files):
        filepath = os.path.join(directory, filename)
        try:
            y, _ = librosa.load(filepath, sr=SAMPLE_RATE)
            audio_int16 = pad_or_trim_int16(y, TARGET_SAMPLES)
            audio_list.append(audio_int16)
        except Exception as e:
            skipped += 1

        if (idx + 1) % 100 == 0 or idx + 1 == len(wav_files):
            print(f"     ⏳ 加载进度: {idx + 1}/{len(wav_files)}"
                  + (f" (跳过 {skipped} 个)" if skipped else ""), end='\r')

    print()

    if not audio_list:
        return np.array([]), np.array([])

    audio_batch = np.array(audio_list)     # (N, TARGET_SAMPLES) int16
    labels = np.full(len(audio_list), label, dtype=np.int32)
    return audio_batch, labels


def extract_features_batch(audio_batch, audio_features):
    """
    使用 openWakeWord 的 AudioFeatures 批量提取嵌入特征。
    
    Args:
        audio_batch: (N, samples) int16 音频数组
        audio_features: AudioFeatures 实例
        
    Returns:
        (N, n_frames, 96) float32 嵌入特征数组
    """
    n_total = audio_batch.shape[0]
    all_embeddings = []

    for start in range(0, n_total, BATCH_SIZE):
        end = min(start + BATCH_SIZE, n_total)
        batch = audio_batch[start:end]

        # embed_clips 内部：音频 → 梅尔频谱 → 嵌入向量
        embeddings = audio_features.embed_clips(batch, batch_size=BATCH_SIZE)
        all_embeddings.append(embeddings)

        print(f"     ⏳ 特征提取: {end}/{n_total}", end='\r')

    print()
    return np.concatenate(all_embeddings, axis=0)


def main():
    print("=" * 60)
    print("🔧 OpenWakeWord 特征提取器")
    print("=" * 60)

    # 1. 初始化 openWakeWord 预处理器
    print("\n📦 初始化 openWakeWord 预处理模型...")
    try:
        from openwakeword.utils import AudioFeatures
        audio_features = AudioFeatures(inference_framework="onnx")
        print("   ✅ melspectrogram + embedding 模型加载成功")
    except ImportError:
        print("❌ 未安装 openwakeword，请运行: pip install openwakeword")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        print("   请尝试: python -c \"from openwakeword.utils import download_models; download_models()\"")
        sys.exit(1)

    # 2. 确定嵌入维度
    embedding_shape = audio_features.get_embedding_shape(TARGET_DURATION)
    n_frames, embedding_dim = embedding_shape
    print(f"   📐 每个 {TARGET_DURATION}s 音频 → {n_frames} 帧 × {embedding_dim} 维嵌入")

    # 3. 加载音频
    print(f"\n📊 加载音频数据...")
    pos_audio, pos_labels = load_audio_batch(XIAOKANG_DIR, 1)
    neg_audio, neg_labels = load_audio_batch(OTHERS_DIR, 0)

    if len(pos_audio) == 0 and len(neg_audio) == 0:
        print("❌ 没有找到任何音频数据！请先运行数据生成脚本。")
        return

    # 4. 提取特征
    print(f"\n🔬 提取 speech embedding 特征...")

    features_list = []
    labels_list = []

    if len(pos_audio) > 0:
        print(f"  🎯 正样本 ({len(pos_audio)} 条):")
        pos_features = extract_features_batch(pos_audio, audio_features)
        features_list.append(pos_features)
        labels_list.append(pos_labels)
        print(f"     ✅ 特征维度: {pos_features.shape}")

    if len(neg_audio) > 0:
        print(f"  🛡️ 负样本 ({len(neg_audio)} 条):")
        neg_features = extract_features_batch(neg_audio, audio_features)
        features_list.append(neg_features)
        labels_list.append(neg_labels)
        print(f"     ✅ 特征维度: {neg_features.shape}")

    # 5. 合并并保存
    all_features = np.concatenate(features_list, axis=0)
    all_labels = np.concatenate(labels_list, axis=0)

    features_path = os.path.join(OUTPUT_DIR, "features.npy")
    labels_path = os.path.join(OUTPUT_DIR, "labels.npy")

    np.save(features_path, all_features)
    np.save(labels_path, all_labels)

    # 6. 打印统计
    n_pos = int((all_labels == 1).sum())
    n_neg = int((all_labels == 0).sum())

    print(f"\n{'=' * 60}")
    print(f"✅ 特征提取完成！")
    print(f"   正样本: {n_pos} 条")
    print(f"   负样本: {n_neg} 条")
    print(f"   特征维度: {all_features.shape} ({all_features.shape[0]} 条 × {all_features.shape[1]} 帧 × {all_features.shape[2]} 维)")
    print(f"   保存路径: {os.path.abspath(OUTPUT_DIR)}")
    print(f"{'=' * 60}")
    print(f"\n💡 下一步: python training/oww_train.py")


if __name__ == "__main__":
    main()
