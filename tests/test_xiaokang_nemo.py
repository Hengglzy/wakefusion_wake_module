import os
import torch
import librosa
import numpy as np
import sounddevice as sd
import time
from tkinter import Tk, filedialog
from nemo.collections.asr.models import EncDecClassificationModel

# ================= 配置区 =================
MODEL_PATH = "xiaokang_xvf3800_pro.nemo"
SAMPLE_RATE = 16000
DURATION = 2  
# 如果有多个麦克风，请在此填入 XVF3800 的设备 ID（默认为 None 使用系统默认）
DEVICE_ID = 14  # 使用 XVF3800 设备，与训练时保持一致 
# ==========================================

def load_model():
    print(f"📦 正在加载模型: {MODEL_PATH}...")
    try:
        model = EncDecClassificationModel.restore_from(MODEL_PATH)
        model.eval()
        if torch.cuda.is_available():
            model = model.cuda()
        return model
    except Exception as e:
        print(f"❌ 加载失败: {e}")
        return None

def run_inference(model, audio_signal):
    audio_tensor = torch.FloatTensor(audio_signal).unsqueeze(0)
    audio_len = torch.LongTensor([len(audio_signal)])
    if torch.cuda.is_available():
        audio_tensor, audio_len = audio_tensor.cuda(), audio_len.cuda()

    with torch.no_grad():
        logits = model.forward(input_signal=audio_tensor, input_signal_length=audio_len)
        probs = torch.softmax(logits, dim=-1)
        labels = model.cfg.labels
        idx = torch.argmax(probs, dim=-1).item()
        conf = probs[0][idx].item()
        return labels[idx], conf

def test_recording(model):
    print(f"\n🎤 使用 XVF3800 准备录音（{DURATION}秒）...")
    time.sleep(0.5)
    print("🔴 >>> 请说: '你好小康' <<<")
    # 指定设备 ID 进行录音
    recording = sd.rec(int(DURATION * SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, device=DEVICE_ID)
    sd.wait()
    print("🟢 录音结束，正在识别...")
    label, conf = run_inference(model, recording.flatten())
    status = "✨ 【识别成功】" if label == "xiaokang" and conf > 0.6 else "☁️  【未唤醒】"
    print(f"{status} 标签: {label} | 置信度: {conf:.2%}")

def select_and_test_file(model):
    """弹出文件资源管理器选择 WAV 文件"""
    root = Tk()
    root.withdraw() # 隐藏 tkinter 主窗口
    root.attributes('-topmost', True) # 将窗口置顶
    
    file_path = filedialog.askopenfilename(
        title="选择要测试的 WAV 文件",
        filetypes=[("WAV files", "*.wav")]
    )
    root.destroy()

    if file_path:
        print(f"\n检查文件: {os.path.basename(file_path)}")
        audio, _ = librosa.load(file_path, sr=SAMPLE_RATE)
        label, conf = run_inference(model, audio)
        print(f"🎯 结果: {label} ({conf:.2%})")
    else:
        print("\n取消选择")

if __name__ == "__main__":
    v_model = load_model()
    if v_model:
        while True:
            print("\n" + "="*40)
            print("🚀 小康唤醒词测试台")
            print("1. 现场录音测试 (XVF3800)")
            print("2. 弹出窗口选择 WAV 文件测试")
            print("q. 退出程序")
            choice = input("请输入序号: ").strip()

            if choice == '1':
                test_recording(v_model)
            elif choice == '2':
                select_and_test_file(v_model)
            elif choice == 'q':
                break
