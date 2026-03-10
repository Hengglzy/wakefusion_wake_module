"""
测试脚本：监听音频模块的ZMQ PUB输出（Multipart Message）
用于验证音频数据流和VAD/KWS事件
"""
import zmq
import json
import numpy as np
import time
from pathlib import Path

# 从项目根目录加载配置
project_root = Path(__file__).resolve().parents[2]
config_path = project_root / "config" / "config.yaml"

# 从配置读取端口（简化版，直接使用默认值）
AUDIO_PUB_PORT = 5556

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect(f"tcp://127.0.0.1:{AUDIO_PUB_PORT}")
socket.setsockopt_string(zmq.SUBSCRIBE, "")

print("正在监听音频模块数据流...")
print("说'你好小康'，应该看到 KWS_HIT 事件")
print("-" * 60)

while True:
    try:
        # 接收Multipart Message：第一帧JSON，第二帧二进制PCM
        metadata_json, audio_binary = socket.recv_multipart(zmq.NOBLOCK)
        metadata = json.loads(metadata_json.decode('utf-8'))
        
        # 检查唤醒词检测
        wake_word = metadata.get("wake_word", {})
        if wake_word.get("detected", False):
            print(f"[KWS] 唤醒词检测: 置信度={wake_word.get('confidence', 0.0):.2%}")
        
        # 可选：打印VAD状态（注释掉以避免刷屏）
        # if metadata.get("vad", False):
        #     print(f"[VAD] 检测到语音")
        
    except zmq.Again:
        time.sleep(0.01)
        continue
    except KeyboardInterrupt:
        print("\n测试结束")
        break

socket.close()
context.term()
