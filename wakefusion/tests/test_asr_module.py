"""
ASR模块测试脚本
模拟Core Server发送音频数据到ASR模块，验证WebSocket输出和END_OF_SPEECH标记处理
支持选择本地WAV文件进行测试
"""
import json
import time
import zmq
import numpy as np
import soundfile as sf
from tkinter import filedialog
from tkinter import Tk
import sys
from wakefusion.config import get_config


def select_audio_file():
    """弹出文件选择对话框，让用户选择WAV文件"""
    # 创建隐藏的Tkinter根窗口
    root = Tk()
    root.withdraw()  # 隐藏主窗口
    
    # 弹出文件选择对话框
    file_path = filedialog.askopenfilename(
        title="选择WAV音频文件",
        filetypes=[
            ("WAV文件", "*.wav"),
            ("音频文件", "*.wav *.mp3 *.flac *.m4a"),
            ("所有文件", "*.*")
        ]
    )
    
    root.destroy()
    return file_path


def load_and_convert_audio(file_path, target_sample_rate=16000):
    """
    加载音频文件并转换为目标格式
    
    Args:
        file_path: 音频文件路径
        target_sample_rate: 目标采样率（默认16kHz）
    
    Returns:
        audio_int16: int16格式的音频数组
        actual_sample_rate: 实际采样率
    """
    try:
        # 读取音频文件
        audio, sample_rate = sf.read(file_path, dtype='float32')
        
        # 如果是立体声，转换为单声道（取平均值）
        if len(audio.shape) > 1:
            audio = np.mean(audio, axis=1)
        
        # 如果采样率不匹配，需要重采样
        if sample_rate != target_sample_rate:
            print(f"⚠️  检测到采样率 {sample_rate}Hz，需要重采样到 {target_sample_rate}Hz...")
            try:
                import librosa
                audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=target_sample_rate)
                print(f"✅ 重采样完成")
            except ImportError:
                print("❌ 错误：需要 librosa 库进行重采样")
                print("   请运行: pip install librosa")
                sys.exit(1)
        
        # 转换为int16格式
        # 确保音频值在[-1, 1]范围内
        audio = np.clip(audio, -1.0, 1.0)
        audio_int16 = (audio * 32767).astype(np.int16)
        
        return audio_int16, target_sample_rate
    
    except Exception as e:
        print(f"❌ 加载音频文件失败: {e}")
        sys.exit(1)


def test_asr_module():
    """测试ASR模块"""
    config = get_config()
    zmq_config = config.zmq
    
    # 创建ZMQ PUSH Socket（模拟Core Server）
    context = zmq.Context()
    push_socket = context.socket(zmq.PUSH)
    push_socket.connect(f"tcp://127.0.0.1:{zmq_config.asr_pull_port}")
    
    print(f"✅ 已连接到ASR模块: tcp://127.0.0.1:{zmq_config.asr_pull_port}")
    print()
    print("📁 请选择要测试的WAV音频文件...")
    
    # 弹出文件选择对话框
    audio_file = select_audio_file()
    
    if not audio_file:
        print("❌ 未选择文件，测试已取消")
        push_socket.close()
        context.term()
        return
    
    print(f"✅ 已选择文件: {audio_file}")
    print()
    
    # 加载并转换音频
    print("🔄 正在加载音频文件...")
    audio_int16, sample_rate = load_and_convert_audio(audio_file, target_sample_rate=16000)
    duration = len(audio_int16) / sample_rate
    print(f"✅ 音频加载完成: 时长 {duration:.2f}秒, 采样率 {sample_rate}Hz")
    print()
    
    print("📤 开始发送测试音频数据...")
    print("   提示：请确保ASR模块已启动（python -m wakefusion.services.asr_service）")
    print(f"   提示：请确保WebSocket客户端已连接（监听 ws://0.0.0.0:{config.websocket.asr_port}）")
    print()
    
    # 将音频分成多个块发送（模拟流式传输）
    chunk_size = int(sample_rate * 0.1)  # 100ms一块
    chunks = [audio_int16[i:i+chunk_size] for i in range(0, len(audio_int16), chunk_size)]
    
    print(f"📦 发送 {len(chunks)} 个音频块（每块 {chunk_size/sample_rate*1000:.0f}ms）...")
    
    for i, chunk in enumerate(chunks):
        push_socket.send(chunk.tobytes())
        print(f"   块 {i+1}/{len(chunks)} 已发送", end='\r')
        time.sleep(0.1)  # 模拟实时流
    
    print()
    print("📤 发送END_OF_SPEECH标记...")
    push_socket.send(b"END_OF_SPEECH")
    
    print("✅ 测试完成！")
    print("   请检查ASR模块的WebSocket输出，应该能看到识别结果")
    
    # 清理
    push_socket.close()
    context.term()


if __name__ == "__main__":
    test_asr_module()
