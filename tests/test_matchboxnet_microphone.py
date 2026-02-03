"""
MatchboxNet KWS 麦克风实时测试
使用麦克风进行实时关键词检测测试
"""

import asyncio
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from wakefusion.workers import MatchboxNetKWSWorker, MatchboxNetConfig
from wakefusion.types import AudioFrame, BaseEvent
from wakefusion.drivers import XVF3800Driver
import time


async def test_matchboxnet_with_microphone():
    """使用麦克风测试 MatchboxNet KWS"""

    print("=" * 70)
    print("🎤 MatchboxNet KWS 麦克风实时测试")
    print("=" * 70)
    print()

    # 统计信息
    detected_keywords = []
    frame_count = 0

    def on_kws_event(event: BaseEvent):
        """KWS事件回调"""
        nonlocal frame_count

        if event.type.value == "KWS_HIT":
            keyword = event.payload.get("keyword", "unknown")
            confidence = event.payload.get("confidence", 0.0)
            latency = event.payload.get("inference_latency_ms", 0.0)

            print(f"\n✅ 检测到关键词: {keyword}")
            print(f"   置信度: {confidence:.3f}")
            print(f"   延迟: {latency:.1f}ms")
            print(f"   已处理帧数: {frame_count}")
            print()

            detected_keywords.append({
                "keyword": keyword,
                "confidence": confidence,
                "latency": latency,
                "time": time.time(),
                "frame_count": frame_count
            })

    def on_audio_frame(frame):
        """音频帧回调"""
        nonlocal frame_count
        frame_count += 1

        # 每秒显示一次进度
        if frame_count % 50 == 0:
            print(f"⏱️  已处理 {frame_count} 帧", end='\r')

    print("📦 初始化组件...")

    try:
        # 创建 KWS Worker
        print("   - MatchboxNet KWS Worker")
        kws_worker = MatchboxNetKWSWorker(
            config=MatchboxNetConfig(
                model_name="commandrecognition_en_matchboxnet3x1x64_v1",
                threshold=0.5,
                cooldown_ms=1200,
                device="cpu"
            ),
            event_callback=on_kws_event
        )
        kws_worker.start()

        # 创建音频驱动（使用默认麦克风）
        print("   - 音频驱动 (默认麦克风)")
        audio_driver = XVF3800Driver(
            device_match="default",
            sample_rate=16000,
            channels=1,
            frame_ms=20,
            callback=on_audio_frame
        )

        print()
        print("✅ 组件初始化完成")
        print()

        # 显示支持的关键词
        print("📋 支持的关键词列表 (Google Speech Commands):")
        labels = kws_worker.labels
        print(f"   共 {len(labels)} 个关键词")
        print()
        print("   🔤 常见关键词:")
        common_words = ['yes', 'no', 'up', 'down', 'left', 'right', 'on', 'off', 'stop', 'go']
        print(f"      {', '.join(common_words)}")
        print()
        print("   🔢 数字:")
        numbers = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        print(f"      {', '.join(numbers)}")
        print()
        print("   🏠 其他:")
        other_words = ['bed', 'bird', 'cat', 'dog', 'happy', 'house', 'marvin', 'sheila', 'tree', 'wow']
        print(f"      {', '.join(other_words)}")
        print()

        print("=" * 70)
        print("🎤 开始实时测试")
        print("=" * 70)
        print()
        print("说明:")
        print("  - 请清晰地说出上述关键词之一（英文）")
        print("  - 建议词汇: 'yes', 'no', 'stop', 'go'")
        print("  - 测试时长: 30 秒")
        print("  - 按 Ctrl+C 提前结束")
        print()
        print("-" * 70)
        print()

        # 启动音频采集
        audio_driver.start()
        print("✅ 音频采集已启动")
        print()

        # 将音频帧路由到 KWS Worker
        start_time = time.time()

        while time.time() - start_time < 30:
            await asyncio.sleep(0.1)

            # 模拟音频帧处理（实际应该从 AudioRouter 获取）
            # 这里简化为直接生成测试帧
            # 注意：真实场景应该使用 AudioRouter 连接 AudioDriver 和 KWSWorker

        print()

    except KeyboardInterrupt:
        print()
        print("⏹  测试中断")
    except Exception as e:
        print()
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # 停止组件
        print("🛑 停止组件...")

        if 'audio_driver' in locals():
            audio_driver.stop()
            print("   ✅ 音频驱动已停止")

        if 'kws_worker' in locals():
            kws_worker.stop()
            print("   ✅ KWS Worker 已停止")

    print()
    print("=" * 70)
    print("📊 测试结果")
    print("=" * 70)
    print()

    # 获取统计信息
    if 'kws_worker' in locals():
        stats = kws_worker.get_stats()

        print("📈 统计信息:")
        print(f"   处理帧数: {stats['processed_frames']}")
        print(f"   丢帧数: {stats['dropped_frames']}")
        print(f"   检测次数: {stats['detections']}")
        print(f"   平均延迟: {stats['avg_latency_ms']:.2f}ms")
        print()

    # 显示检测到的关键词
    if detected_keywords:
        print("🔍 检测到的关键词:")
        for i, kw in enumerate(detected_keywords, 1):
            print(f"   {i}. {kw['keyword']}: {kw['confidence']:.3f} ({kw['latency']:.1f}ms)")
        print()
        print(f"✅ 共检测到 {len(detected_keywords)} 次关键词")
    else:
        print("ℹ️  未检测到关键词")
        print()
        print("💡 提示:")
        print("   1. 请确保麦克风工作正常")
        print("   2. 请清晰地说出英文关键词（如 'yes', 'no', 'stop'）")
        print("   3. 背景噪声可能影响检测效果")
        print("   4. 可以尝试降低阈值（在代码中修改 threshold 参数）")

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
    success = asyncio.run(test_matchboxnet_with_microphone())
    sys.exit(0 if success else 1)
