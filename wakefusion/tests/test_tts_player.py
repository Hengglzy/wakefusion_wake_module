"""
TTS音频播放测试脚本
接收TTS模块的音频数据并通过系统喇叭播放，用于测试TTS合成效果
"""
import zmq
import json
import numpy as np
import sounddevice as sd
import queue
import threading


def main():
    """主函数：监听TTS音频并播放"""
    context = zmq.Context()
    # 创建PULL Socket，对接TTS的PUSH Socket
    pull_socket = context.socket(zmq.PULL)
    pull_socket.connect("tcp://127.0.0.1:5559")
    
    # 设置接收超时（1000ms），让 recv_multipart() 定期返回，以便响应 Ctrl+C
    pull_socket.setsockopt(zmq.RCVTIMEO, 1000)  # 1秒超时
    
    # 🌟 修复：使用音频队列和播放线程，确保按顺序播放所有音频块
    audio_queue = queue.Queue()
    playback_active = threading.Event()
    playback_active.set()
    
    def playback_worker():
        """播放线程：从队列中取出音频块并按顺序播放"""
        current_sample_rate = 24000  # 默认采样率
        
        while playback_active.is_set() or not audio_queue.empty():
            try:
                # 从队列中获取音频块（带超时，以便检查停止标志）
                audio_data, sample_rate = audio_queue.get(timeout=0.1)
                current_sample_rate = sample_rate
                
                print(f"🔊 收到音频包！大小: {len(audio_data)} 采样点 | 采样率: {sample_rate}Hz | 正在播放...")
                
                # 调用系统喇叭播放（阻塞，直到播放完成）
                sd.play(audio_data, samplerate=sample_rate)
                sd.wait()  # 等待这句播完
                
                audio_queue.task_done()
                
            except queue.Empty:
                # 队列为空，继续等待
                continue
            except Exception as e:
                print(f"❌ 播放出错: {e}")
                continue
    
    # 启动播放线程
    playback_thread = threading.Thread(target=playback_worker, daemon=True)
    playback_thread.start()
    
    print("✅ 喇叭已通电，正在监听 TTS 端口 tcp://127.0.0.1:5559 ...")
    print("   提示：请确保TTS模块已启动（python -m wakefusion.services.tts_service）")
    print("   提示：按 Ctrl+C 退出")
    print()
    
    try:
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
                    
                    # 🌟 修复：将音频块加入队列，而不是立即播放
                    # 这样即使多个音频块快速到达，也能按顺序播放
                    audio_queue.put((audio_data, sample_rate))
                    
            except KeyboardInterrupt:
                print("\n🛑 退出播放测试")
                break
            except zmq.Again:
                # 接收超时，继续循环以检查 KeyboardInterrupt
                continue
            except Exception as e:
                print(f"❌ 接收出错: {e}")
                continue
    
    finally:
        # 停止播放线程
        playback_active.clear()
        # 等待队列中的音频播放完成
        audio_queue.join()
        # 清理
        pull_socket.close()
        context.term()
        print("✅ 资源清理完成")


if __name__ == "__main__":
    main()
