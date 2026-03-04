"""
负样本切片机 V3 (Negative Audio Slicer)
支持两种模式：
  模式 1：单文件切片 — 将一段长音频按 2 秒切割（如背景环境音）
  模式 2：批量误唤醒切片 — 扫描 results 目录下所有 falsedebug_stream_output 前缀的
          误唤醒录音，自动按 2 秒切割后加入 others 负样本目录

工作流（Hard Negative Mining）：
  1. 运行 audio_service + test_ws_audio_saver，收集误唤醒时的录音
  2. 手动将确认为误唤醒的 debug_stream_output_*.wav 重命名为 falsedebug_stream_output_*.wav
  3. 运行本脚本（模式 2），自动切片并加入负样本
  4. 重新运行 split_dataset.py + finetune_xvf3800.py 训练模型

运行环境：wakefusion 虚拟环境
运行命令：python training/negative_slicer.py
"""
import librosa
import soundfile as sf
import os
import uuid

# ================= 配置区 =================
# 模式 1：单文件切片
LONG_AUDIO_FILE = "background_10min.wav"

# 模式 2：批量误唤醒切片
FALSE_WAKE_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "results")
FALSE_WAKE_PREFIX = "falsedebug_stream_output"

# 公共配置
OUTPUT_DIR = "custom_dataset/others"
SAMPLE_RATE = 16000
CHUNK_LEN = 2.0  # 标准模型需要的切片长度(秒)
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)


def slice_single_file(audio_path, output_prefix):
    """将单个音频文件切割为 2 秒片段，返回切片数量"""
    try:
        y, sr = librosa.load(audio_path, sr=SAMPLE_RATE)
    except Exception as e:
        print(f"  ❌ 解析失败: {e}")
        return 0

    duration = len(y) / SAMPLE_RATE
    samples_per_chunk = int(CHUNK_LEN * SAMPLE_RATE)
    total_chunks = len(y) // samples_per_chunk

    if total_chunks == 0:
        print(f"  ⏭️  时长 {duration:.1f}s < {CHUNK_LEN}s，跳过")
        return 0

    unique_id = uuid.uuid4().hex[:6]

    for i in range(total_chunks):
        start = i * samples_per_chunk
        end = start + samples_per_chunk
        chunk = y[start:end]

        out_path = os.path.join(OUTPUT_DIR, f"{output_prefix}_{unique_id}_{i:04d}.wav")
        sf.write(out_path, chunk, SAMPLE_RATE)

    return total_chunks


def mode_single_file():
    """模式 1：单文件切片（原有功能）"""
    print("=" * 55)
    print("🔪 模式 1：单文件切片")
    print("=" * 55)

    if not os.path.exists(LONG_AUDIO_FILE):
        print(f"❌ 找不到素材文件: {LONG_AUDIO_FILE}")
        print("请把你的长录音改名为此名字，或修改脚本中的文件名。")
        return

    print(f"⏳ 正在处理: {LONG_AUDIO_FILE}")
    count = slice_single_file(LONG_AUDIO_FILE, "slice_bg")
    print(f"\n🎉 切片完成！产出 {count} 条 2 秒背景负样本。")


def mode_batch_false_wake():
    """模式 2：批量误唤醒音频切片（Hard Negative Mining）"""
    print("=" * 55)
    print("🔪 模式 2：批量误唤醒音频切片 (Hard Negative Mining)")
    print("=" * 55)

    if not os.path.exists(FALSE_WAKE_DIR):
        print(f"❌ 找不到 results 目录: {FALSE_WAKE_DIR}")
        return

    # 扫描所有 falsedebug_stream_output 前缀的音频文件
    false_wake_files = sorted([
        f for f in os.listdir(FALSE_WAKE_DIR)
        if f.startswith(FALSE_WAKE_PREFIX) and f.endswith('.wav')
    ])

    if not false_wake_files:
        print(f"⚠️  在 {FALSE_WAKE_DIR} 中未找到 {FALSE_WAKE_PREFIX}*.wav 文件")
        print("💡 请先将确认为误唤醒的录音重命名为 falsedebug_stream_output_*.wav")
        return

    print(f"📂 扫描目录: {FALSE_WAKE_DIR}")
    print(f"🔍 找到 {len(false_wake_files)} 个误唤醒录音文件\n")

    total_slices = 0
    processed = 0

    for idx, filename in enumerate(false_wake_files, 1):
        filepath = os.path.join(FALSE_WAKE_DIR, filename)
        print(f"[{idx}/{len(false_wake_files)}] 📄 {filename}")

        count = slice_single_file(filepath, "slice_falsewake")
        if count > 0:
            print(f"  ✅ 切出 {count} 条 2 秒负样本")
            total_slices += count
            processed += 1

    print(f"\n{'=' * 55}")
    print(f"🎉 批量切片完成！")
    print(f"   处理文件数: {processed}/{len(false_wake_files)}")
    print(f"   新增负样本: {total_slices} 条")
    print(f"   输出目录:   {OUTPUT_DIR}")
    print(f"💡 下一步: 运行 training/split_dataset.py 重新分割数据集")
    print(f"{'=' * 55}")


def main():
    print("\n" + "=" * 55)
    print("🔪 WakeFusion 负样本切片机 V3")
    print("=" * 55)
    print("1. 单文件切片（长背景音频 → 2 秒片段）")
    print("2. 批量误唤醒切片（results 目录下的误唤醒录音 → 2 秒负样本）")
    print("q. 退出")
    print("=" * 55)

    choice = input("请选择模式 (1/2/q): ").strip()

    if choice == '1':
        mode_single_file()
    elif choice == '2':
        mode_batch_false_wake()
    elif choice == 'q':
        print("👋 已退出。")
    else:
        print("❌ 无效选择，请输入 1、2 或 q")


if __name__ == "__main__":
    main()
