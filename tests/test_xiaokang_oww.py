"""
OpenWakeWord 唤醒词实时测试台 (OWW Real-time Test)
===================================================
使用 openwakeword.Model 加载自训练的 ONNX 模型，进行实时麦克风测试或文件测试。

支持模式：
  1. 实时麦克风测试：持续监听，检测到唤醒词时显示结果
  2. 单次录音测试：录制 2 秒音频并识别
  3. 选择 WAV 文件测试

前置条件：
  - 已运行 python training/oww_train.py 生成 xiaokang_oww.onnx
  - pip install openwakeword sounddevice

运行环境：wakefusion 虚拟环境
运行命令：python tests/test_xiaokang_oww.py
"""
import os
import sys
import numpy as np
import sounddevice as sd
import time

# ================= 配置区 =================
MODEL_PATH = "xiaokang_oww.onnx"
SAMPLE_RATE = 16000
CHUNK_SIZE = 1280           # 80ms，openWakeWord 标准处理块大小
THRESHOLD = 0.5             # 唤醒阈值（可调整）
DURATION = 2.0              # 单次录音时长
DEVICE_ID = 14              # XVF3800 设备 ID（与其他测试脚本一致）
# ==========================================


def load_model():
    """加载 openWakeWord 模型"""
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 找不到模型: {MODEL_PATH}")
        print(f"   请先运行: python training/oww_train.py")
        return None

    print(f"📦 正在加载 OpenWakeWord 模型: {MODEL_PATH}...")
    try:
        from openwakeword.model import Model
        model = Model(
            wakeword_models=[MODEL_PATH],
            inference_framework="onnx"
        )
        print(f"   ✅ 模型加载成功")
        print(f"   模型名称: {list(model.models.keys())}")
        return model
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return None


def get_model_name(model):
    """获取模型的预测键名"""
    names = list(model.models.keys())
    return names[0] if names else "xiaokang_oww"


def test_realtime(model):
    """模式 1：实时麦克风持续监听"""
    model_name = get_model_name(model)
    print(f"\n🎙️ 实时监听模式已启动 (阈值: {THRESHOLD})")
    print(f"   请对着麦克风说 '你好小康'")
    print(f"   按 Ctrl+C 退出")
    print("=" * 50)

    # 重置模型状态
    model.reset()

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=CHUNK_SIZE,
            device=DEVICE_ID,
            dtype='int16'
        ) as stream:
            while True:
                audio_chunk, _ = stream.read(CHUNK_SIZE)
                audio_data = audio_chunk.flatten()

                # 获取预测结果
                predictions = model.predict(audio_data)
                score = predictions.get(model_name, 0.0)

                if score > THRESHOLD:
                    now_str = time.strftime("%H:%M:%S")
                    print(f"⚡ 唤醒词检测！ 置信度: {score:.2%}  🕐 {now_str}")
                elif score > 0.1:
                    print(f"   听... (得分: {score:.2%})   ", end='\r')
                else:
                    print(f"   听... (静默)            ", end='\r')

    except KeyboardInterrupt:
        print(f"\n🛑 监听已停止。")


def test_recording(model):
    """模式 2：单次录音测试"""
    model_name = get_model_name(model)

    print(f"\n🎤 准备录音（{DURATION}秒）...")
    time.sleep(0.5)
    print("🔴 >>> 请说: '你好小康' <<<")

    recording = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        device=DEVICE_ID,
        dtype='int16'
    )
    sd.wait()
    print("🟢 录音结束，正在识别...")

    # 重置模型状态
    model.reset()

    # 逐块送入模型（模拟流式处理）
    audio_data = recording.flatten()
    max_score = 0.0

    for i in range(0, len(audio_data) - CHUNK_SIZE + 1, CHUNK_SIZE):
        chunk = audio_data[i:i + CHUNK_SIZE]
        predictions = model.predict(chunk)
        score = predictions.get(model_name, 0.0)
        max_score = max(max_score, score)

    # 判定结果
    if max_score > THRESHOLD:
        print(f"✨ 【识别成功】 唤醒词得分: {max_score:.2%}")
    else:
        print(f"☁️  【未唤醒】 最高得分: {max_score:.2%} (阈值: {THRESHOLD:.2%})")


def test_file(model):
    """模式 3：选择 WAV 文件测试"""
    model_name = get_model_name(model)

    try:
        from tkinter import Tk, filedialog
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)

        file_path = filedialog.askopenfilename(
            title="选择要测试的 WAV 文件",
            filetypes=[("WAV files", "*.wav")]
        )
        root.destroy()
    except Exception:
        file_path = input("请输入 WAV 文件路径: ").strip().strip('"')

    if not file_path or not os.path.exists(file_path):
        print("取消或文件不存在。")
        return

    print(f"\n📄 测试文件: {os.path.basename(file_path)}")

    # 使用 predict_clip 方法（处理完整音频文件）
    import wave
    try:
        with wave.open(file_path, 'rb') as wf:
            if wf.getframerate() != SAMPLE_RATE:
                print(f"⚠️ 采样率不匹配 (文件: {wf.getframerate()}, 需要: {SAMPLE_RATE})")
                print("   尝试使用 librosa 重采样...")
                import librosa
                audio_float, _ = librosa.load(file_path, sr=SAMPLE_RATE)
                audio_data = (audio_float * 32767).astype(np.int16)
            else:
                audio_data = np.frombuffer(wf.readframes(wf.getnframes()), dtype=np.int16)
    except Exception:
        import librosa
        audio_float, _ = librosa.load(file_path, sr=SAMPLE_RATE)
        audio_data = (audio_float * 32767).astype(np.int16)

    # 重置并逐块处理
    model.reset()
    max_score = 0.0

    for i in range(0, len(audio_data) - CHUNK_SIZE + 1, CHUNK_SIZE):
        chunk = audio_data[i:i + CHUNK_SIZE]
        predictions = model.predict(chunk)
        score = predictions.get(model_name, 0.0)
        max_score = max(max_score, score)

    if max_score > THRESHOLD:
        print(f"🎯 结果: ✅ 唤醒词检测到！ 得分: {max_score:.2%}")
    else:
        print(f"🎯 结果: ❌ 未检测到唤醒词。最高得分: {max_score:.2%}")


def main():
    oww_model = load_model()
    if not oww_model:
        return

    while True:
        print("\n" + "=" * 50)
        print("🚀 小康唤醒词测试台 (OpenWakeWord 版)")
        print(f"   模型: {MODEL_PATH}")
        print(f"   阈值: {THRESHOLD}")
        print("=" * 50)
        print("1. 实时麦克风持续监听")
        print("2. 单次录音测试 (2秒)")
        print("3. 选择 WAV 文件测试")
        print("q. 退出程序")

        choice = input("请输入序号: ").strip()

        if choice == '1':
            test_realtime(oww_model)
        elif choice == '2':
            test_recording(oww_model)
        elif choice == '3':
            test_file(oww_model)
        elif choice.lower() == 'q':
            print("👋 退出。")
            break
        else:
            print("❌ 无效输入。")


if __name__ == "__main__":
    main()
