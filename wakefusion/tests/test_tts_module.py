"""
TTS模块测试脚本
模拟LLM发送文本到TTS模块，验证ZMQ音频输出、流式输出和中断处理
"""
import json
import time
import zmq
import numpy as np
import websockets
import asyncio
from wakefusion.config import get_config


def test_tts_module_zmq():
    """测试TTS模块的ZMQ输出（PULL模式）"""
    config = get_config()
    zmq_config = config.zmq
    
    # 创建ZMQ PULL Socket（模拟Core Server接收TTS音频）
    context = zmq.Context()
    pull_socket = context.socket(zmq.PULL)
    pull_socket.connect(f"tcp://127.0.0.1:{zmq_config.tts_push_port}")
    
    print(f"✅ 已连接到TTS模块: tcp://127.0.0.1:{zmq_config.tts_push_port}")
    print("📥 开始接收TTS音频数据...")
    print("   提示：请确保TTS模块已启动（python -m wakefusion.services.tts_service）")
    print()
    
    # 设置接收超时
    pull_socket.setsockopt(zmq.RCVTIMEO, 5000)  # 5秒超时
    
    audio_chunks_received = 0
    total_samples = 0
    
    try:
        while True:
            try:
                # 接收Multipart Message
                message = pull_socket.recv_multipart()
                
                if len(message) >= 2:
                    # 第一帧：JSON元数据
                    metadata = json.loads(message[0].decode('utf-8'))
                    # 第二帧：PCM音频数据
                    audio_data = message[1]
                    
                    audio_chunks_received += 1
                    samples = len(audio_data) // 2  # int16 = 2 bytes
                    total_samples += samples
                    
                    print(f"📦 收到音频块 {audio_chunks_received}: "
                          f"{samples} 采样点 ({samples/16000*1000:.1f}ms), "
                          f"总计 {total_samples/16000:.2f}秒")
                
            except zmq.Again:
                print("⏱️ 接收超时，停止监听")
                break
            except KeyboardInterrupt:
                print("\n🛑 用户中断")
                break
    
    finally:
        pull_socket.close()
        context.term()
        print(f"\n✅ 测试完成！共收到 {audio_chunks_received} 个音频块，总计 {total_samples/16000:.2f}秒")


async def test_tts_module_websocket():
    """测试TTS模块的WebSocket输入"""
    config = get_config()
    ws_config = config.websocket
    
    print(f"🔌 连接到TTS WebSocket: ws://127.0.0.0:{ws_config.tts_port}")
    print("   提示：请确保TTS模块已启动")
    print()
    
    try:
        async with websockets.connect(f"ws://127.0.0.1:{ws_config.tts_port}") as websocket:
            print("✅ WebSocket连接成功")
            
            # 发送流式文本消息
            test_texts = [
                "你好，",
                "我是",
                "小康。",
                "今天",
                "天气",
                "很好。"
            ]
            
            print("\n📤 发送流式文本消息...")
            for i, text in enumerate(test_texts):
                message = {
                    "type": "tts_request",
                    "text": text,
                    "is_streaming": True,
                    "is_final": (i == len(test_texts) - 1)
                }
                await websocket.send(json.dumps(message, ensure_ascii=False))
                print(f"   消息 {i+1}/{len(test_texts)}: {text}")
                await asyncio.sleep(0.2)  # 模拟LLM流式输出延迟
            
            print("\n✅ 流式文本发送完成")
            print("   请检查TTS模块的ZMQ输出，应该能看到合成的音频数据")
            
            # 等待一段时间让TTS处理
            await asyncio.sleep(2)
            
            # 测试停止信号
            print("\n🛑 发送停止信号...")
            stop_message = {"type": "stop_synthesis"}
            await websocket.send(json.dumps(stop_message))
            print("✅ 停止信号已发送")
    
    except Exception as e:
        print(f"❌ WebSocket测试失败: {e}")


def test_tts_stop_signal():
    """测试TTS模块的停止信号（ZMQ PUB）"""
    config = get_config()
    zmq_config = config.zmq
    
    # 创建ZMQ PUB Socket（模拟Core Server发送停止信号）
    context = zmq.Context()
    pub_socket = context.socket(zmq.PUB)
    pub_socket.bind(f"tcp://127.0.0.1:{zmq_config.tts_stop_pub_port}")
    
    print(f"✅ 停止信号PUB Socket已绑定: tcp://127.0.0.1:{zmq_config.tts_stop_pub_port}")
    print("   提示：请确保TTS模块已启动并订阅此端口")
    print()
    
    input("按回车键发送停止信号...")
    
    pub_socket.send_string("STOP_SYNTHESIS")
    print("📤 停止信号已发送: STOP_SYNTHESIS")
    
    # 清理
    pub_socket.close()
    context.term()


def main():
    """主函数"""
    import sys
    
    if len(sys.argv) > 1:
        test_mode = sys.argv[1]
    else:
        print("请选择测试模式：")
        print("  1. ZMQ输出测试（接收TTS音频）")
        print("  2. WebSocket输入测试（发送文本）")
        print("  3. 停止信号测试（发送STOP_SYNTHESIS）")
        test_mode = input("请输入序号 (1/2/3): ").strip()
    
    if test_mode == "1":
        test_tts_module_zmq()
    elif test_mode == "2":
        asyncio.run(test_tts_module_websocket())
    elif test_mode == "3":
        test_tts_stop_signal()
    else:
        print("无效的测试模式")


if __name__ == "__main__":
    main()
