"""
WebSocket 音频流全自动落盘工具 (Auto Audio Dump)
功能：
1. 听到唤醒自动开始录制。
2. 探测到断流（点击停止按钮后）自动保存为带有时间戳的 WAV 文件。
3. 保存后自动进入下一轮待命。
"""
import os
import time
import asyncio
import websockets
import json
import base64
import wave
import sys

# ================= 配置区 =================
WS_URL = "ws://127.0.0.1:8001/ws/audio"
# 1. 存储地址改为你指定的 results 文件夹
OUTPUT_DIR = r"D:\tools\cursor_project\wakefusion_wake_module\results"
SAMPLE_RATE = 16000

# 确保目标文件夹存在
os.makedirs(OUTPUT_DIR, exist_ok=True)
# ==========================================

async def audio_saver():
    audio_buffer = bytearray()
    is_recording = False
    
    print(f"🔌 正在连接到音频网关: {WS_URL} ...")
    
    try:
        async with websockets.connect(WS_URL) as ws:
            print("🟢 连接成功！请对着麦克风喊出唤醒词。")
            print("🤖 (全自动模式：收到唤醒自动录制，点击网页停止按钮自动保存)")
            
            while True:
                try:
                    # 🌟 核心魔法：最多等 1 秒。如果 1 秒内没收到数据，就会触发 TimeoutError
                    message = await asyncio.wait_for(ws.recv(), timeout=1.0)
                    data = json.loads(message)
                    
                    if data.get("type") == "wake_word_hit":
                        print(f"\n✨ 收到唤醒信号！开始录制音频流 (置信度: {data.get('confidence', 0):.2%})")
                        audio_buffer.clear()
                        is_recording = True
                        
                    elif data.get("type") == "audio_stream" and is_recording:
                        chunk = base64.b64decode(data["data"])
                        audio_buffer.extend(chunk)
                        
                        kb_size = len(audio_buffer) / 1024
                        print(f"🌊 正在接收音频流... 当前大小: {kb_size:.1f} KB", end="\r")

                except asyncio.TimeoutError:
                    # 🌟 当触发超时时，检查是不是正在录音。
                    # 如果是，说明底层断流了（用户点了停止按钮）
                    if is_recording and len(audio_buffer) > 0:
                        # 2. 参照采集脚本，生成带时间戳的文件名
                        filename = f"debug_stream_output_{int(time.time())}.wav"
                        filepath = os.path.join(OUTPUT_DIR, filename)
                        
                        print(f"\n\n💾 探测到推流已停止！正在自动保存音频...")
                        
                        # 写入 WAV 文件
                        with wave.open(filepath, "wb") as wf:
                            wf.setnchannels(1)
                            wf.setsampwidth(2)
                            wf.setframerate(SAMPLE_RATE)
                            wf.writeframes(audio_buffer)
                            
                        print(f"✅ 保存成功: {filepath}")
                        print("🎧 继续潜伏等待下一次唤醒...")
                        
                        # 重置状态，准备下一次抓取！
                        is_recording = False
                        audio_buffer.clear()
                    else:
                        # 如果没有在录音时的超时（平时潜伏的时候），什么都不做，继续等
                        pass

    except websockets.exceptions.ConnectionClosed:
        print("\n🔴 WebSocket 连接已断开。请检查网关是否运行。")
    except asyncio.CancelledError:
        pass
    except KeyboardInterrupt:
        pass

if __name__ == "__main__":
    try:
        asyncio.run(audio_saver())
    except KeyboardInterrupt:
        print("\n👋 收到退出指令，测试工具已关闭。")
        sys.exit(0)
