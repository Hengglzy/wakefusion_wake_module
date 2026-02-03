"""
音频驱动测试脚本
用于验证XVF3800设备是否正常工作
"""

import asyncio
import numpy as np
from wakefusion.drivers import XVF3800Driver
from wakefusion.types import AudioFrameRaw


async def test_audio_devices():
    """测试音频设备枚举"""
    print("=" * 60)
    print("音频设备枚举测试")
    print("=" * 60)

    driver = XVF3800Driver()

    # 查找设备
    device = driver.find_device()

    if device:
        print(f"\n✓ 找到XVF3800设备:")
        print(f"  - 设备名称: {device.name}")
        print(f"  - 设备索引: {device.index}")
        print(f"  - 采样率: {device.sample_rate} Hz")
        print(f"  - 声道数: {device.channels}")
    else:
        print(f"\n✗ 未找到XVF3800设备")
        print("  请检查:")
        print("  1. 设备是否已连接")
        print("  2. Windows声音设置中是否可见")
        print("  3. config.yaml中device_match配置是否正确")


async def test_audio_capture():
    """测试音频采集"""
    print("\n" + "=" * 60)
    print("音频采集测试 (5秒)")
    print("=" * 60)

    frame_count = 0

    def on_audio_frame(frame: AudioFrameRaw):
        nonlocal frame_count
        frame_count += 1

        if frame_count <= 5:
            print(f"\n✓ 接收到音频帧 #{frame_count}:")
            print(f"  - 时间戳: {frame.ts:.3f}")
            print(f"  - 样本数: {len(frame.pcm16)}")
            print(f"  - 采样率: {frame.sample_rate} Hz")
            print(f"  - RMS能量: {np.sqrt(np.mean(frame.pcm16.astype(np.float32) ** 2)):.2f}")

    driver = XVF3800Driver(
        device_match="XVF3800",
        sample_rate=48000,
        frame_ms=20,
        callback=on_audio_frame
    )

    try:
        print("\n启动音频采集...")
        driver.start()
        print(f"\n采集中... (设备: {driver.device_info.name})")

        await asyncio.sleep(5)

        driver.stop()

        print(f"\n✓ 采集完成!")
        print(f"  - 总帧数: {frame_count}")
        print(f"  - 实际FPS: {frame_count / 5:.1f}")
        print(f"  - 预期FPS: 50.0")

        if frame_count >= 200:  # 允许一些丢帧
            print("\n✓ 音频采集正常!")
        else:
            print("\n⚠  丢帧率较高，请检查")

    except Exception as e:
        print(f"\n✗ 错误: {e}")
        print("\n可能的原因:")
        print("  1. 设备被其他应用占用")
        print("  2. 采样率不支持")
        print("  3. 驱动问题")


async def test_kws_model():
    """测试KWS模型加载"""
    print("\n" + "=" * 60)
    print("KWS模型测试")
    print("=" * 60)

    try:
        from openwakeword.model import Model

        print("\n加载openWakeWord模型...")
        model = Model()

        print("✓ 模型加载成功!")
        print(f"  - 可用模型: {list(model.models.keys())}")

        # 测试推理
        print("\n生成测试音频帧...")
        test_frame = np.random.randint(-1000, 1000, size=1600, dtype=np.int16)

        print("运行推理...")
        prediction = model.predict(test_frame)

        print("✓ 推理成功!")
        print(f"  - 预测结果: {prediction}")

    except ImportError:
        print("\n✗ openWakeWord未安装")
        print("  请运行: pip install openwakeword")
    except Exception as e:
        print(f"\n✗ 错误: {e}")


async def test_vad_model():
    """测试VAD模型加载"""
    print("\n" + "=" * 60)
    print("VAD模型测试")
    print("=" * 60)

    try:
        import webrtcvad

        print("\n加载webrtcvad...")
        vad = webrtcvad.Vad(2)

        print("✓ VAD加载成功!")

        # 测试推理
        print("\n生成测试音频帧...")
        test_frame = np.random.randint(-1000, 1000, size=320, dtype=np.int16)

        print("运行推理...")
        is_speech = vad.is_speech(test_frame.tobytes(), 16000)

        print("✓ 推理成功!")
        print(f"  - 检测结果: {'语音' if is_speech else '静音'}")

    except ImportError:
        print("\n✗ webrtcvad未安装")
        print("  请运行: pip install webrtcvad")
    except Exception as e:
        print(f"\n✗ 错误: {e}")


async def main():
    """主测试函数"""
    print("\n" + "=" * 60)
    print("WakeFusion 音频驱动测试")
    print("=" * 60)

    # 测试1: 设备枚举
    await test_audio_devices()

    # 测试2: 音频采集
    # await test_audio_capture()  # 取消注释以启用

    # 测试3: KWS模型
    await test_kws_model()

    # 测试4: VAD模型
    await test_vad_model()

    print("\n" + "=" * 60)
    print("测试完成")
    print("=" * 60)
    print("\n提示:")
    print("  - 如需测试音频采集，请取消注释 test_audio_capture()")
    print("  - 运行完整系统: python -m wakefusion.runtime")
    print()


if __name__ == "__main__":
    asyncio.run(main())
