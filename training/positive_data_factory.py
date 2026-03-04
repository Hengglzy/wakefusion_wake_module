"""
WakeFusion 正样本变声工厂 (Positive Data Factory - Voice Transformation Only)
功能：仅对真人录音进行变调/变速裂变，TTS文件保持不变

处理逻辑：
  - 真人录音 → 变调 + 变速（每个文件产生4个变体）
  - TTS样本 → 跳过（已有多声音多风格，无需处理）

识别规则：文件名以 "tts_" 开头的视为TTS生成的样本，跳过处理。

运行环境：wakefusion 虚拟环境
运行命令：python training/positive_data_factory.py
"""
import os
import librosa
import soundfile as sf
import numpy as np

# ================= 配置区 =================
SAMPLE_RATE = 16000
TARGET_DIR = "custom_dataset/xiaokang"
TARGET_DURATION = 2.0  # 统一目标时长 2 秒

PITCH_STEPS = [4, -4]       # 女童声，男低音
SPEED_RATES = [1.2, 0.85]   # 语速快，语速慢
# ==========================================


def get_files_by_condition(directory, exclude_keywords):
    """辅助函数：获取不包含某些关键字的文件"""
    return [f for f in os.listdir(directory)
            if f.endswith('.wav') and not any(k in f for k in exclude_keywords)]


def is_tts_file(filename):
    """判断文件是否为 TTS 生成的样本"""
    return filename.startswith("tts_")


def pad_or_trim(audio, target_len):
    """将音频精确调整为目标长度（采样点数）"""
    if len(audio) > target_len:
        return audio[:target_len]
    elif len(audio) < target_len:
        return np.pad(audio, (0, target_len - len(audio)))
    return audio


def run_factory():
    print("=" * 55)
    print("🏭 WakeFusion 正样本变声工厂已启动")
    print("=" * 55)

    if not os.path.exists(TARGET_DIR):
        print(f"⚠️ 找不到目录: {TARGET_DIR}")
        return

    target_len = int(TARGET_DURATION * SAMPLE_RATE)

    # ---------------------------------------------------------
    # 变声处理 — 仅对纯真人录音（原始录音，不含aug、pitch、speed）
    # TTS 文件 (tts_ 前缀) 跳过此阶段
    # ---------------------------------------------------------
    raw_files = get_files_by_condition(TARGET_DIR, ['pitch', 'speed', 'aug'])
    real_files = [f for f in raw_files if not is_tts_file(f)]
    tts_files = [f for f in raw_files if is_tts_file(f)]

    print(f"🔍 找到 {len(real_files)} 条真人原声 + {len(tts_files)} 条TTS样本")
    print(f"    ├── 真人录音 → 变调 + 变速（每个文件产生4个变体）")
    print(f"    └── TTS样本  → 跳过（已有多声音多风格）")

    fission_count = 0
    for filename in real_files:
        filepath = os.path.join(TARGET_DIR, filename)
        base_name = filename.replace(".wav", "")
        y, sr = librosa.load(filepath, sr=SAMPLE_RATE)

        # 变调
        for step in PITCH_STEPS:
            suffix = f"_pitch_{'up' if step > 0 else 'down'}{abs(step)}"
            out_path = os.path.join(TARGET_DIR, f"{base_name}{suffix}.wav")
            if not os.path.exists(out_path):
                y_shifted = librosa.effects.pitch_shift(y, sr=sr, n_steps=step)
                y_shifted = pad_or_trim(y_shifted, target_len)
                sf.write(out_path, y_shifted, sr)
                fission_count += 1

        # 变速
        for rate in SPEED_RATES:
            suffix = f"_speed_{'fast' if rate > 1.0 else 'slow'}"
            out_path = os.path.join(TARGET_DIR, f"{base_name}{suffix}.wav")
            if not os.path.exists(out_path):
                y_stretched = librosa.effects.time_stretch(y, rate=rate)
                y_stretched = pad_or_trim(y_stretched, target_len)
                sf.write(out_path, y_stretched, sr)
                fission_count += 1

    print(f"✅ 完成！真人录音裂变产出 {fission_count} 条变声数据。")
    print("=" * 55)
    print("🎉 正样本变声处理完毕！")
    print(f"💡 下一步：运行 training/split_dataset.py 分割训练集和验证集")


if __name__ == "__main__":
    run_factory()
