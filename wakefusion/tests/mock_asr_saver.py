"""
Mock ASR服务：接收Core Server的音频流并保存为WAV文件
用于测试端到端流程
"""
import zmq
import wave
import time
from pathlib import Path

# 从项目根目录加载配置
project_root = Path(__file__).resolve().parents[2]
config_path = project_root / "config" / "config.yaml"

# 从配置读取端口（简化版，直接使用默认值）
ASR_PULL_PORT = 5558

context = zmq.Context()
socket = context.socket(zmq.PULL)
socket.bind(f"tcp://127.0.0.1:{ASR_PULL_PORT}")
# 设置接收超时，以便能够响应 Ctrl+C
socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1秒超时

output_dir = Path("test_recordings")
output_dir.mkdir(exist_ok=True)

print("Mock ASR 已启动，等待Core Server发送音频...")
print("当数字人被唤醒并开始监听时，音频会流式传输到这里")
print("按 Ctrl+C 退出")
print("-" * 60)

recording_count = 0
wf = None  # Initialize wf variable to None

while True:
    try:
        try:
            audio_chunk = socket.recv()  # 带超时的接收
        except zmq.Again:
            # 超时，继续循环以检查 KeyboardInterrupt
            continue
        
        # Check for end-of-speech or timeout signals
        if audio_chunk == b"END_OF_SPEECH" or audio_chunk == b"TIMEOUT":
            print(f"\n录音 #{recording_count} 结束")
            if wf:
                wf.close()
                wf = None
            continue
        
        # If it's the start of a new recording (first data received)
        if wf is None:
            recording_count += 1
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            wav_path = output_dir / f"recording_{recording_count}_{timestamp}.wav"
            wf = wave.open(str(wav_path), "wb")
            wf.setnchannels(1)
            wf.setsampwidth(2)  # int16 = 2 bytes
            wf.setframerate(16000)
            print(f"\n开始录音 #{recording_count}: {wav_path.name}")
        
        # Write audio data
        wf.writeframes(audio_chunk)
        print(".", end="", flush=True)  # Progress indicator
        
    except KeyboardInterrupt:
        print("\n\n正在停止 Mock ASR...")
        if wf:
            wf.close()
            print(f"已保存录音 #{recording_count}")
        break
    except Exception as e:
        print(f"\n错误: {e}")
        if wf:
            wf.close()
            wf = None

print("Mock ASR 已停止")
socket.close()
context.term()
