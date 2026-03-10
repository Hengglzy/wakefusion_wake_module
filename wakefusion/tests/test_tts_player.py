"""
TTS音频播放测试脚本
接收TTS模块的音频数据并通过系统喇叭播放，用于测试TTS合成效果
"""
import zmq
import json
import numpy as np
import sounddevice as sd


def main():
    """主函数：监听TTS音频并播放"""
    context = zmq.Context()
    # 创建PULL Socket，对接TTS的PUSH Socket
    pull_socket = context.socket(zmq.PULL)
    pull_socket.connect("tcp://127.0.0.1:5559")
    
    # 设置接收超时（1000ms），让 recv_multipart() 定期返回，以便响应 Ctrl+C
    pull_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1秒超时
    
    print("✅ 喇叭已通电，正在监听 TTS 端口 tcp://127.0.0.1:5559 ...")
    print("   提示：请确保TTS模块已启动（python -m wakefusion.services.tts_service）")
    print("   提示：按 Ctrl+C 退出")
    print()
    
    while True:
        try:
            # 接收多部分消息（Metadata + Audio Bytes）
            # 由于设置了超时，如果没有数据会在1秒后抛出 zmq.Again 异常
            parts = pull_socket.recv_multipart()
            if len(parts) == 2:
                metadata = json.loads(parts[0].decode('utf-8'))
                audio_bytes = parts[1]
                
                # 恢复为 NumPy 数组
                audio_data = np.frombuffer(audio_bytes, dtype=np.int16)
                
                # 优先使用metadata中的采样率，如果没有则使用默认值
                # Qwen3-TTS 的真实采样率一般是 24000Hz
                sample_rate = metadata.get('sample_rate', 24000)
                
                print(f"🔊 收到音频包！大小: {len(audio_data)} 采样点 | 采样率: {sample_rate}Hz | 正在播放...")
                
                # 调用系统喇叭播放
                sd.play(audio_data, samplerate=sample_rate)
                sd.wait()  # 等待这句播完
                
        except KeyboardInterrupt:
            print("\n🛑 退出播放测试")
            break
        except zmq.Again:
            # 接收超时，继续循环以检查 KeyboardInterrupt
            continue
        except Exception as e:
            print(f"❌ 播放出错: {e}")
            continue
    
    # 清理
    pull_socket.close()
    context.term()


if __name__ == "__main__":
    main()
