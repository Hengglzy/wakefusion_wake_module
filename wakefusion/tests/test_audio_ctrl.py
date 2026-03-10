"""
测试脚本：发送控制指令到音频模块的ZMQ REP
用于验证动态阈值调整功能
"""
import zmq
import json
import time
from pathlib import Path

# 从项目根目录加载配置
project_root = Path(__file__).resolve().parents[2]
config_path = project_root / "config" / "config.yaml"

# 从配置读取端口（简化版，直接使用默认值）
AUDIO_CTRL_PORT = 5557
REQ_REP_TIMEOUT_MS = 2000

context = zmq.Context()
req_socket = context.socket(zmq.REQ)
req_socket.connect(f"tcp://127.0.0.1:{AUDIO_CTRL_PORT}")
req_socket.setsockopt(zmq.REQ_RELAXED, 1)
req_socket.setsockopt(zmq.REQ_CORRELATE, 1)
req_socket.setsockopt(zmq.RCVTIMEO, REQ_REP_TIMEOUT_MS)

print("测试音频模块控制指令...")
print("-" * 60)

# Test 1: Lower threshold
print("1. 发送降低阈值指令 (0.4)...")
req_socket.send_json({"command": "set_threshold", "value": 0.4})
try:
    reply = req_socket.recv_json()
    print(f"   响应: {reply}")
except zmq.Again:
    print("   ❌ 超时：音频模块无响应")

time.sleep(1)

# Test 2: Restore threshold
print("2. 发送恢复阈值指令 (0.95)...")
req_socket.send_json({"command": "set_threshold", "value": 0.95})
try:
    reply = req_socket.recv_json()
    print(f"   响应: {reply}")
except zmq.Again:
    print("   ❌ 超时：音频模块无响应")

print("\n测试完成")
req_socket.close()
context.term()
