"""
音频设备枚举工具
列出系统中所有可用的音频输入设备
"""

import pyaudio


def list_audio_devices():
    """列出所有音频设备"""
    print("=" * 70)
    print("可用的音频输入设备")
    print("=" * 70)

    p = pyaudio.PyAudio()

    print(f"\n总共找到 {p.get_device_count()} 个音频设备\n")

    input_devices = []

    for i in range(p.get_device_count()):
        info = p.get_device_info_by_index(i)

        # 只显示输入设备
        if info['maxInputChannels'] > 0:
            input_devices.append((i, info))

            print(f"设备 #{i}:")
            print(f"  名称: {info['name']}")
            print(f"  最大输入声道: {info['maxInputChannels']}")
            print(f"  默认采样率: {int(info['defaultSampleRate'])} Hz")
            print(f"  是否为默认输入: {info['isLoopbackDevice'] if 'isLoopbackDevice' in info else 'N/A'}")
            print()

    # 显示默认输入设备
    try:
        default_input = p.get_default_input_device_info()
        print("=" * 70)
        print(f"🎤 系统默认输入设备:")
        print(f"   设备 #{default_input['index']}: {default_input['name']}")
        print("=" * 70)
    except:
        print("无法获取默认输入设备")

    p.terminate()

    return input_devices


def test_device(device_index: int, duration_sec: int = 5):
    """
    测试特定设备是否能正常录音

    Args:
        device_index: 设备索引
        duration_sec: 测试时长（秒）
    """
    import numpy as np
    import time

    print(f"\n测试设备 #{device_index}...")

    p = pyaudio.PyAudio()

    try:
        device_info = p.get_device_info_by_index(device_index)
        sample_rate = int(device_info['defaultSampleRate'])
        channels = min(2, device_info['maxInputChannels'])

        print(f"  设备名称: {device_info['name']}")
        print(f"  采样率: {sample_rate} Hz")
        print(f"  声道数: {channels}")

        stream = p.open(
            format=pyaudio.paInt16,
            channels=channels,
            rate=sample_rate,
            input=True,
            input_device_index=device_index,
            frames_per_buffer=int(sample_rate * 0.02)  # 20ms
        )

        print(f"\n正在录音 {duration_sec} 秒...")
        print("请对着麦克风说话...")

        frames = []
        start_time = time.time()

        while time.time() - start_time < duration_sec:
            data = stream.read(int(sample_rate * 0.02), exception_on_overflow=False)
            frames.append(data)

            # 计算RMS能量
            audio_data = np.frombuffer(data, dtype=np.int16)
            rms = np.sqrt(np.mean(audio_data.astype(np.float32) ** 2))

            # 每秒显示一次
            elapsed = time.time() - start_time
            if int(elapsed) > int(elapsed - 0.02):
                energy_bar = min(50, int(rms / 50))
                bar = "█" * energy_bar + "░" * (50 - energy_bar)
                print(f"[{int(elapsed)}s] 音频能量: {bar} {rms:.1f}")

        stream.stop_stream()
        stream.close()

        print(f"\n✅ 测试成功!")
        print(f"   录音帧数: {len(frames)}")
        print(f"   实际时长: {len(frames) * 0.02:.1f} 秒")

    except Exception as e:
        print(f"\n❌ 测试失败: {e}")
    finally:
        p.terminate()


def main():
    """主函数"""
    import sys

    print("\n" + "=" * 70)
    print(" WakeFusion 音频设备工具")
    print("=" * 70)

    # 列出所有设备
    devices = list_audio_devices()

    if not devices:
        print("\n❌ 没有找到可用的音频输入设备!")
        print("\n请检查:")
        print("  1. 麦克风是否已连接")
        print("  2. Windows声音设置中是否启用")
        print("  3. 麦克风是否被其他应用占用")
        return

    # 询问用户是否要测试特定设备
    print("\n" + "=" * 70)
    choice = input("是否要测试特定设备? (输入设备编号，或按回车跳过): ").strip()

    if choice:
        try:
            device_index = int(choice)
            if device_index in [d[0] for d in devices]:
                test_device(device_index, duration_sec=5)
            else:
                print(f"\n❌ 无效的设备编号: {device_index}")
        except ValueError:
            print("\n❌ 无效的输入")

    # 显示配置建议
    print("\n" + "=" * 70)
    print("配置建议")
    print("=" * 70)

    if len(devices) == 1:
        print(f"\n只找到一个输入设备，建议使用:")
        print(f"  audio:")
        print(f"    device_match: \"default\"")
    else:
        print(f"\n找到 {len(devices)} 个输入设备，你可以:")
        print(f"\n  选项1: 使用系统默认设备")
        print(f"    audio:")
        print(f"      device_match: \"default\"")
        print(f"\n  选项2: 使用特定设备（输入设备编号）")
        print(f"    audio:")
        print(f"      device_match: \"{devices[0][1]['name']}\"")

    print("\n" + "=" * 70)


if __name__ == "__main__":
    main()
