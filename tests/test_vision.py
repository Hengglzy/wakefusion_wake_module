"""
视觉组件测试脚本
用于验证Orbbec Gemini 330系列相机（335/336）和视觉门控是否正常工作
"""

import asyncio
import time
import numpy as np
from wakefusion.drivers import Gemini330Driver, CameraConfig
from wakefusion.routers import VisionRouter
from wakefusion.workers import FaceGateWorker, FaceGateConfig
from wakefusion.types import VisionFrame


async def test_camera_driver():
    """测试相机驱动"""
    print("=" * 60)
    print("相机驱动测试")
    print("=" * 60)

    frame_count = 0

    def on_vision_frame(frame: VisionFrame):
        nonlocal frame_count
        frame_count += 1

        if frame_count <= 5:
            print(f"\n✓ 接收到视觉帧 #{frame_count}:")
            print(f"  - 时间戳: {frame.ts:.3f}")
            if frame.rgb is not None:
                print(f"  - RGB: {frame.rgb.shape}")
            if frame.depth is not None:
                print(f"  - Depth: {frame.depth.shape}, range={frame.depth.min()}-{frame.depth.max()}")
                # 转换为米
                depth_m = frame.depth.astype(np.float32) / 1000.0
                valid_depth = depth_m[
                    (depth_m > 0.5) & (depth_m < 4.0)
                ]
                if len(valid_depth) > 0:
                    print(f"  - 有效深度范围: {valid_depth.min():.2f}m - {valid_depth.max():.2f}m")

    driver = Gemini330Driver(
        config=CameraConfig(
            rgb_width=1280,   # 匹配深度分辨率
            rgb_height=800,
            rgb_fps=15,       # 15FPS 对于展厅交互已经足够流畅
            depth_width=1280,
            depth_height=800,
            depth_fps=15,
            enable_rgb=False,  # 暂时只测试深度
            enable_depth=True
        ),
        callback=on_vision_frame
    )

    capture_task = None
    try:
        print("\n启动相机...")
        driver.start()
        
        # 启动后台采集循环（关键修复：确保帧数据被真正采集）
        capture_task = asyncio.create_task(driver.run_with_reconnect())
        
        print("采集中... (10秒)")
        await asyncio.sleep(10)

        print(f"\n✓ 采集完成!")
        print(f"  - 总帧数: {frame_count}")
        print(f"  - 实际FPS: {frame_count / 10:.1f}")

        if frame_count >= 100:  # 允许一些丢帧
            print("\n✓ 相机采集正常!")
        else:
            print("\n⚠  丢帧率较高，请检查")

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        print("\n可能的原因:")
        print("  1. 相机未连接")
        print("  2. pyorbbecsdk未安装")
        print("  3. 驱动问题")
    finally:
        # 安全停止采集任务
        if capture_task and not capture_task.done():
            capture_task.cancel()
            try:
                await capture_task
            except asyncio.CancelledError:
                pass
        
        # 停止驱动
        driver.stop()


async def test_vision_router():
    """测试视觉路由器"""
    print("\n" + "=" * 60)
    print("视觉路由器测试")
    print("=" * 60)

    router = VisionRouter(
        cache_ms=600,
        target_fps=15
    )

    # 生成测试帧
    print("\n生成测试帧...")
    current_time = time.time()

    for i in range(10):
        frame = VisionFrame(
            ts=current_time + i * 0.1,
            rgb=np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8),
            depth=np.random.randint(500, 4000, (480, 640), dtype=np.uint16),
            presence=(i % 3 == 0),  # 每3帧有1次presence
            faces=[],
            distance_m=2.0 + i * 0.1,
            confidence=0.8
        )
        router.process_frame(frame)

    # 测试缓存
    stats = router.get_cache_stats()
    print(f"\n✓ 缓存统计:")
    print(f"  - 缓存大小: {stats.size}/{stats.capacity}")
    print(f"  - 时长: {stats.duration_ms:.0f}ms")
    print(f"  - Presence帧数: {stats.presence_count}")

    # 测试获取最近帧
    recent_frames = router.get_recent_frames(300)
    print(f"\n✓ 最近300ms帧数: {len(recent_frames)}")

    # 测试presence摘要
    summary = router.get_presence_summary(500)
    print(f"\n✓ Presence摘要:")
    print(f"  - has_presence: {summary['has_presence']}")
    print(f"  - presence_count: {summary['presence_count']}")
    print(f"  - avg_confidence: {summary['avg_confidence']:.2f}")


async def test_face_gate():
    """测试人脸门控"""
    print("\n" + "=" * 60)
    print("人脸门控测试")
    print("=" * 60)

    gate = FaceGateWorker(
        config=FaceGateConfig(
            distance_m_max=4.0,
            distance_m_min=0.5,
            enable_face_detection=False,
            enable_depth_gate=True
        )
    )
    gate.start()

    # 生成测试帧
    print("\n生成测试帧...")
    current_time = time.time()

    # 帧1: 人在2米处
    frame1 = VisionFrame(
        ts=current_time,
        depth=np.random.randint(1900, 2100, (480, 640), dtype=np.uint16),  # ~2米
        rgb=None,
        presence=False,
        faces=[],
        distance_m=None,
        confidence=0.0
    )

    result1 = gate.process_frame(frame1)
    print(f"\n✓ 帧1 (2米处):")
    print(f"  - presence: {result1.presence if result1 else 'N/A'}")
    print(f"  - distance_m: {f'{result1.distance_m:.2f}m' if result1 and result1.distance_m is not None else 'N/A'}")
    print(f"  - valid: {result1.valid if result1 else 'N/A'}")
    print(f"  - confidence: {f'{result1.confidence:.2f}' if result1 else 'N/A'}")

    # 帧2: 人在5米外（超出范围）
    frame2 = VisionFrame(
        ts=current_time + 0.1,
        depth=np.random.randint(4900, 5100, (480, 640), dtype=np.uint16),  # ~5米
        rgb=None,
        presence=False,
        faces=[],
        distance_m=None,
        confidence=0.0
    )

    result2 = gate.process_frame(frame2)
    print(f"\n✓ 帧2 (5米外):")
    print(f"  - presence: {result2.presence if result2 else 'N/A'}")
    print(f"  - distance_m: {f'{result2.distance_m:.2f}m' if result2 and result2.distance_m is not None else 'N/A'}")
    print(f"  - valid: {result2.valid if result2 else 'N/A'}")

    # 统计
    stats = gate.get_stats()
    print(f"\n✓ 门控统计:")
    print(f"  - valid_user_count: {stats['valid_user_count']}")
    print(f"  - rejected_count: {stats['rejected_count']}")

    gate.stop()


async def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("WakeFusion 视觉组件测试")
    print("=" * 60)

    # 测试1: 相机驱动（需要硬件）
    print("\n提示: 相机驱动测试需要Orbbec Gemini 330系列硬件（335/336）")
    choice = input("是否测试相机驱动? (y/n): ")
    if choice.lower() == 'y':
        await test_camera_driver()

    # 测试2: 视觉路由器（无需硬件）
    await test_vision_router()

    # 测试3: 人脸门控（无需硬件）
    await test_face_gate()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    print("\n提示:")
    print("  - 运行完整系统: python -m wakefusion.runtime")
    print("  - 在config/config.yaml中启用vision.enabled")
    print()


if __name__ == "__main__":
    asyncio.run(main())
