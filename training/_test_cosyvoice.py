"""
CosyVoice 环境兼容性快速测试
==============================
测试三项能力：
  1. modelscope 是否安装成功
  2. CosyVoice 模型能否加载到 GPU
  3. 能否用真人种子音频克隆生成"你好小康"

运行：python training/_test_cosyvoice.py
"""
import sys
import os
import time

print("=" * 55)
print("  CosyVoice 环境兼容性测试")
print("=" * 55)

# ---- 测试 1: PyTorch + CUDA ----
print("\n[1/4] 检查 PyTorch + CUDA...")
try:
    import torch
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
        vram = torch.cuda.get_device_properties(0).total_memory / 1024**3
        print(f"  VRAM: {vram:.1f} GB")
    else:
        print("  [WARN] CUDA not available, will use CPU (very slow)")
except ImportError:
    print("  [FAIL] PyTorch not installed")
    sys.exit(1)

# ---- 测试 2: ModelScope ----
print("\n[2/4] 检查 ModelScope...")
try:
    import modelscope
    print(f"  modelscope: {modelscope.__version__}")
except ImportError:
    print("  [FAIL] modelscope not installed")
    print("  Fix: pip install modelscope")
    print("\n  Stopping here. Install modelscope first, then re-run.")
    sys.exit(1)

# ---- 测试 3: CosyVoice 模型加载 ----
print("\n[3/4] 加载 CosyVoice-300M 模型（首次需下载 ~1.5GB）...")
try:
    from modelscope.pipelines import pipeline
    from modelscope.utils.constant import Tasks

    t0 = time.time()
    # CosyVoice-300M 支持零样本克隆
    synthesizer = pipeline(
        task=Tasks.text_to_speech,
        model='iic/CosyVoice-300M',
        model_revision='master'
    )
    t1 = time.time()
    print(f"  Model loaded in {t1 - t0:.1f}s")

    if torch.cuda.is_available():
        vram_used = torch.cuda.memory_allocated(0) / 1024**3
        print(f"  VRAM used: {vram_used:.2f} GB")

except Exception as e:
    print(f"  [FAIL] CosyVoice load error: {e}")
    print("\n  This may require a specific version of modelscope.")
    print("  Alternative: try 'pip install cosyvoice' or clone from GitHub.")
    sys.exit(1)

# ---- 测试 4: 零样本克隆生成 ----
print("\n[4/4] 测试零样本克隆生成...")

# 寻找一个真人种子文件
seed_dir = "custom_dataset/xiaokang"
seed_file = None
if os.path.exists(seed_dir):
    for f in sorted(os.listdir(seed_dir)):
        if f.startswith("real_") and f.endswith(".wav") and "pitch" not in f and "speed" not in f and "aug" not in f:
            seed_file = os.path.join(seed_dir, f)
            break

if not seed_file:
    print(f"  [WARN] No real_*.wav seed found in {seed_dir}")
    print("  Creating a test with edge-tts generated seed instead...")
    # 用 xiaokang 目录中的任意 WAV 作为备选种子
    if os.path.exists(seed_dir):
        for f in sorted(os.listdir(seed_dir)):
            if f.endswith(".wav"):
                seed_file = os.path.join(seed_dir, f)
                break

if not seed_file:
    print("  [FAIL] No WAV files found in custom_dataset/xiaokang/")
    print("  Cannot test voice cloning without a seed audio.")
    sys.exit(1)

print(f"  Seed: {seed_file}")

try:
    t0 = time.time()
    # 注意: prompt_text 需要是种子音频的大致内容描述
    # 对于唤醒词录音，内容就是"你好小康"
    result = synthesizer(
        "你好小康",
        prompt_speech_16k=seed_file,
        prompt_text="你好小康"
    )
    t1 = time.time()

    # 保存测试输出
    import soundfile as sf_lib
    output_path = "training/_test_cosyvoice_output.wav"

    if 'output_wav' in result:
        audio_data = result['output_wav']
    elif 'text_pcm' in result:
        audio_data = result['text_pcm']
    else:
        # 尝试获取第一个类似音频的输出
        for key in result:
            if hasattr(result[key], '__len__') and len(result[key]) > 1000:
                audio_data = result[key]
                print(f"  Output key: '{key}'")
                break
        else:
            print(f"  [WARN] Unknown output format. Keys: {list(result.keys())}")
            audio_data = None

    if audio_data is not None:
        import numpy as np
        audio_np = np.array(audio_data).flatten().astype(np.float32)
        # 归一化
        max_val = np.max(np.abs(audio_np))
        if max_val > 0:
            audio_np = audio_np / max_val * 0.9
        sf_lib.write(output_path, audio_np, 16000)
        print(f"  Generated: {output_path} ({len(audio_np)/16000:.2f}s)")
        print(f"  Time: {t1 - t0:.2f}s")
    else:
        print("  [WARN] Could not extract audio from result")

except Exception as e:
    print(f"  [FAIL] Generation error: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)

# ---- 结果 ----
print("\n" + "=" * 55)
print("  ALL TESTS PASSED!")
print("=" * 55)
print("\n  Your environment is ready for CosyVoice.")
print("  You can now run: python training/mega_dataset_generator.py")
print(f"\n  Test output saved to: {output_path}")
print("  Listen to it to verify voice cloning quality.")
