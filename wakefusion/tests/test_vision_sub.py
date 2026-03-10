"""
测试脚本：监听视觉模块的ZMQ PUB输出
用于验证视觉唤醒状态机的功能
"""
import zmq
import json
import time
from pathlib import Path

# 从项目根目录加载配置
project_root = Path(__file__).resolve().parents[2]
config_path = project_root / "config" / "config.yaml"

# 从配置读取端口（简化版，直接使用默认值）
VISION_PUB_PORT = 5555

context = zmq.Context()
socket = context.socket(zmq.SUB)
socket.connect(f"tcp://127.0.0.1:{VISION_PUB_PORT}")
socket.setsockopt_string(zmq.SUBSCRIBE, "")

print("正在监听视觉模块数据...")
print("走到摄像头3米内，正脸看它，应该看到 wake: true")
print("走开3.5米外，1.5秒后应该看到 wake: false")
print("-" * 60)

last_wake_state = None
while True:
    try:
        message = socket.recv_json(zmq.NOBLOCK)
        wake = message.get("wake", False)
        
        if wake != last_wake_state:
            timestamp = time.strftime("%H:%M:%S")
            print(f"[{timestamp}] 视觉唤醒状态变化: {wake}")
            if wake:
                faces = message.get('faces', [])
                face_count = len(faces)
                print(f"  -> 检测到人脸: {face_count} 个")
                # 显示所有人脸的距离和正面率信息
                for i, face in enumerate(faces, 1):
                    distance = face.get('distance', 'N/A')
                    frontal_percent = face.get('frontal_percent', 'N/A')
                    confidence = face.get('confidence', 'N/A')
                    print(f"     人脸{i}: 距离={distance}m, 正面率={frontal_percent}%, 置信度={confidence}")
            last_wake_state = wake
    except zmq.Again:
        time.sleep(0.01)
        continue
    except KeyboardInterrupt:
        print("\n测试结束")
        break

socket.close()
context.term()
