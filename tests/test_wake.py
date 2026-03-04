"""
WakeFusion 唤醒词测试脚本 (适配 NeMo MatchboxNet)
专门用于测试 'xiaokang.nemo' 模型
"""

import time
import sys
import numpy as np
import pyaudio
from pathlib import Path

# 确保能导入 wakefusion 模块
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from wakefusion.workers import MatchboxNetKWSWorker, MatchboxNetConfig
from wakefusion.types import AudioFrame, BaseEvent

# ================= 配置区域 =================
MODEL_PATH = "xiaokang.nemo"  # 您的模型文件名
WAKEWORD_LABEL = "xiaokang"   # 模型内部训练的标签（通常是拼音）
THRESHOLD = 0.6               # 触发阈值 (0-1)
SAMPLE_RATE = 16000           # NeMo 标准采样率
CHUNK_SIZE = 1280             # 80ms 音频帧 (16000 * 0.08)
# ===========================================

def main():
    print("=" * 60)
    print(f"🎤 WakeFusion MatchboxNet 测试工具")
    print(f"   目标模型: {MODEL_PATH}")
    print("=" * 60)

    # 1. 定义回调函数 (当检测到唤醒词时触发)
    def on_kws_event(event: BaseEvent):
        if event.type.name == "KWS_HIT":
            payload = event.payload
            print(f"\n🚀 [WAKE] 检测到唤醒词: {payload.get('keyword')}")
            print(f"   置信度: {payload.get('confidence'):.2f}")
            print(f"   延迟: {payload.get('inference_latency_ms'):.1f}ms")
            print("-" * 40)

    # 2. 初始化 MatchboxNet Worker
    print(f"\n[1/3] 正在加载 NeMo 模型 ({MODEL_PATH})...")
    
    # 检查文件是否存在
    if not Path(MODEL_PATH).exists():
        # 尝试在上级目录查找
        alt_path = Path(project_root) / MODEL_PATH
        if alt_path.exists():
            final_path = str(alt_path)
        else:
            print(f"❌ 错误: 找不到模型文件 '{MODEL_PATH}'")
            print(f"   请确保它位于项目根目录: {project_root}")
            return
    else:
        final_path = MODEL_PATH

    try:
        # 配置 Worker
        config = MatchboxNetConfig(
            model_name="xiaokang",  # 逻辑名称
            model_path=final_path,  # 物理路径 (关键!)
            threshold=THRESHOLD,
            cooldown_ms=1500,       # 防止重复触发
            device="cpu"            # 默认使用 CPU
        )
        
        worker = MatchboxNetKWSWorker(config, event_callback=on_kws_event)
        worker.start()
        print(f"✅ 模型加载成功! 支持标签: {worker.labels}")
        
    except Exception as e:
        print(f"❌ 模型加载失败: {e}")
        return

    # 3. 初始化音频采集 (使用 PyAudio)
    print("\n[2/3] 初始化麦克风...")
    p = pyaudio.PyAudio()
    
    try:
        stream = p.open(
            format=pyaudio.paInt16,
            channels=1,
            rate=SAMPLE_RATE,
            input=True,
            frames_per_buffer=CHUNK_SIZE
        )
        print("✅ 麦克风已启动")
    except Exception as e:
        print(f"❌ 麦克风启动失败: {e}")
        worker.stop()
        return

    # 4. 开始主循环
    print(f"\n[3/3] 正在监听... 请说 '{WAKEWORD_LABEL}' (中文：你好小康)")
    print("   (按 Ctrl+C 退出)")
    
    try:
        while True:
            # 读取原始音频数据
            data = stream.read(CHUNK_SIZE, exception_on_overflow=False)
            
            # 转换为 numpy 数组
            pcm16 = np.frombuffer(data, dtype=np.int16)
            
            # 封装为 AudioFrame
            frame = AudioFrame(
                ts=time.time(),
                pcm16=pcm16,
                sample_rate=SAMPLE_RATE
                #channels=1
            )
            
            # 发送给 Worker 处理
            worker.process_frame(frame)
            
            # 简单的动态打印证明程序在运行
            sys.stdout.write(".")
            sys.stdout.flush()
            
    except KeyboardInterrupt:
        print("\n\n🛑 测试已停止")
    finally:
        # 清理资源
        worker.stop()
        stream.stop_stream()
        stream.close()
        p.terminate()

if __name__ == "__main__":
    main()