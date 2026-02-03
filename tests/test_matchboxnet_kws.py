"""
MatchboxNet KWS 测试脚本
测试英文预训练模型的关键词检测功能
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from wakefusion.workers import MatchboxNetKWSWorker, MatchboxNetConfig
from wakefusion.types import AudioFrame, BaseEvent
import numpy as np
import time


def create_test_frame(keyword_samples: np.ndarray, sample_rate: int = 16000) -> AudioFrame:
    """
    创建测试音频帧（使用真实音频或噪声）

    Args:
        keyword_samples: 关键词音频样本
        sample_rate: 采样率

    Returns:
        AudioFrame对象
    """
    return AudioFrame(
        ts=time.time(),
        pcm16=keyword_samples,
        sample_rate=sample_rate,
        channels=1
    )


def generate_noise_frame(duration_ms: int = 80, sample_rate: int = 16000) -> np.ndarray:
    """
    生成噪声帧（用于测试）

    Args:
        duration_ms: 时长（毫秒）
        sample_rate: 采样率

    Returns:
        噪声音频样本
    """
    num_samples = int(sample_rate * duration_ms / 1000)
    # 生成低幅值噪声（模拟静音）
    noise = np.random.uniform(-100, 100, num_samples).astype(np.int16)
    return noise


async def test_matchboxnet_kws():
    """测试 MatchboxNet KWS"""

    print("=" * 70)
    print("🎯 MatchboxNet KWS 测试")
    print("=" * 70)
    print()

    # 统计信息
    detected_keywords = []

    def on_kws_event(event: BaseEvent):
        """KWS事件回调"""
        if event.type.value == "KWS_HIT":
            keyword = event.payload.get("keyword", "unknown")
            confidence = event.payload.get("confidence", 0.0)
            latency = event.payload.get("inference_latency_ms", 0.0)

            print(f"✅ 检测到关键词: {keyword}")
            print(f"   置信度: {confidence:.3f}")
            print(f"   延迟: {latency:.1f}ms")
            print()

            detected_keywords.append({
                "keyword": keyword,
                "confidence": confidence,
                "latency": latency,
                "time": time.time()
            })

    # 创建 KWS Worker
    print("📦 初始化 MatchboxNet KWS Worker...")
    print("   模型: commandrecognition_en_matchboxnet3x1x64_v1")
    print("   设备: cpu")
    print("   阈值: 0.5")
    print()

    try:
        kws_worker = MatchboxNetKWSWorker(
            config=MatchboxNetConfig(
                model_name="commandrecognition_en_matchboxnet3x1x64_v1",
                threshold=0.5,
                cooldown_ms=1200,
                device="cpu"
            ),
            event_callback=on_kws_event
        )

        # 启动 Worker
        print("🚀 启动 KWS Worker...")
        kws_worker.start()
        print("✅ KWS Worker 已启动")
        print()

        # 显示支持的关键词
        print("📋 支持的关键词列表:")
        labels = kws_worker.labels
        print(f"   共 {len(labels)} 个关键词")
        print(f"   前10个: {', '.join(labels[:10])}")
        print(f"   后10个: {', '.join(labels[-10:])}")
        print()

        print("=" * 70)
        print("🎤 开始测试（模拟音频流）")
        print("=" * 70)
        print()
        print("说明:")
        print("  - 当前使用随机噪声测试（不会触发关键词）")
        print("  - 如需真实测试，请准备包含关键词的音频文件")
        print("  - 预训练模型支持 Google Speech Commands 的30个关键词")
        print("  - 常见关键词: yes, no, up, down, left, right, on, off, stop, go")
        print()
        print("测试将运行 10 秒...")
        print("按 Ctrl+C 提前结束")
        print()
        print("-" * 70)
        print()

        # 模拟音频流（使用噪声）
        start_time = time.time()
        frame_count = 0

        while time.time() - start_time < 10:
            # 生成噪声帧
            noise_samples = generate_noise_frame(duration_ms=80, sample_rate=16000)
            frame = create_test_frame(noise_samples)

            # 处理帧
            kws_worker.process_frame(frame)
            frame_count += 1

            # 显示进度
            if frame_count % 10 == 0:
                elapsed = time.time() - start_time
                print(f"⏱️  已处理 {frame_count} 帧 ({elapsed:.1f}秒)")

            # 模拟实时帧率（80ms帧长）
            await asyncio.sleep(0.08)

        print()
        print("-" * 70)
        print()

        # 获取统计信息
        stats = kws_worker.get_stats()

        print("📊 测试统计:")
        print(f"   处理帧数: {stats['processed_frames']}")
        print(f"   丢帧数: {stats['dropped_frames']}")
        print(f"   检测次数: {stats['detections']}")
        print(f"   平均延迟: {stats['avg_latency_ms']:.2f}ms")
        print()

        # 显示检测到的关键词
        if detected_keywords:
            print("🔍 检测到的关键词:")
            for kw in detected_keywords:
                print(f"   - {kw['keyword']}: {kw['confidence']:.3f} ({kw['latency']:.1f}ms)")
        else:
            print("ℹ️  未检测到关键词（符合预期，因为使用的是随机噪声）")
            print()
            print("💡 提示:")
            print("   1. 预训练模型需要真实的语音输入才能检测")
            print("   2. 建议使用麦克风实时测试:")
            print("      python tests/test_matchboxnet_microphone.py")
            print("   3. 或者准备包含关键词的音频文件进行测试")

        print()

    except KeyboardInterrupt:
        print()
        print("⏹  测试中断")
    except Exception as e:
        print()
        print(f"❌ 错误: {e}")
        print()
        print("💡 可能的原因:")
        print("   1. NeMo 框架未安装")
        print("   2. 模型下载失败（网络问题）")
        print("   3. 依赖缺失（torch, librosa等）")
        print()
        print("🔧 解决方案:")
        print("   pip install nemo-toolkit[asr]>=1.14.0 torch>=2.0.0 librosa>=0.10.0")
        return False
    finally:
        # 停止 Worker
        if 'kws_worker' in locals():
            kws_worker.stop()
            print("✅ KWS Worker 已停止")

    print()
    print("=" * 70)
    print("✅ 测试完成")
    print("=" * 70)
    print()

    return True


if __name__ == "__main__":
    # 设置日志级别
    import logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )

    # 运行测试
    success = asyncio.run(test_matchboxnet_kws())
    sys.exit(0 if success else 1)
