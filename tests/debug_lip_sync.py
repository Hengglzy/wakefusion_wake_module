"""
唇动检测调试脚本
用于排查视觉强杀不触发的问题
"""
import zmq
import json
import time
from wakefusion.config import get_config

def main():
    """监控 vision_service 发送的数据和 core_server 的状态"""
    config = get_config()
    vision_pub_port = config.zmq.vision_pub_port
    
    print(f"🔍 开始监控 vision_service 的 ZMQ PUB (端口 {vision_pub_port})...")
    print("=" * 60)
    
    # 创建 ZMQ SUB socket
    context = zmq.Context()
    sub_socket = context.socket(zmq.SUB)
    sub_socket.connect(f"tcp://127.0.0.1:{vision_pub_port}")
    sub_socket.setsockopt_string(zmq.SUBSCRIBE, "")
    
    last_is_talking = None
    start_time = time.time()
    
    try:
        while True:
            try:
                # 接收数据（阻塞，超时1秒）
                if sub_socket.poll(timeout=1000):
                    data = sub_socket.recv_json(zmq.NOBLOCK)
                    is_talking = data.get("is_talking", False)
                    timestamp = data.get("timestamp", 0)
                    
                    # 只在状态变化时打印
                    if last_is_talking != is_talking:
                        elapsed = time.time() - start_time
                        status = "TALKING" if is_talking else "SILENT"
                        print(f"[{elapsed:6.2f}s] 👄 is_talking 变化: {last_is_talking} → {is_talking} ({status})")
                        last_is_talking = is_talking
                    
                    # 每5秒打印一次当前状态（即使没变化）
                    if int(time.time() - start_time) % 5 == 0 and int(time.time() - start_time) > 0:
                        if int(time.time() - start_time) % 5 == 0:
                            print(f"[{time.time() - start_time:6.2f}s] 📊 当前状态: is_talking={is_talking}, wake={data.get('wake', False)}")
                            time.sleep(0.1)  # 避免重复打印
                else:
                    print("⚠️  超过1秒未收到视觉数据，请检查 vision_service 是否在运行")
                    time.sleep(1)
            except zmq.Again:
                continue
            except KeyboardInterrupt:
                break
    finally:
        sub_socket.close()
        context.term()
        print("\n✅ 监控结束")

if __name__ == "__main__":
    main()
