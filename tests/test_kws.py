"""
使用音频文件测试KWS
当没有麦克风时，可以用预录制的音频文件测试KWS功能
"""

import asyncio
import numpy as np
from pathlib import Path

from wakefusion.workers import KWSWorker
from wakefusion.types import AudioFrame, EventType


async def test_kws_with_file(audio_file: str = None):
    """
    使用音频文件测试KWS

    Args:
        audio_file: 音频文件路径（WAV格式，16kHz, 16-bit, mono）
                  如果为None，则生成测试音频
    """
    print("=" * 70)
    print("KWS 测试 (音频文件模式)")
    print("=" * 70)

    # 初始化KWS工作线程
    kws = KWSWorker(
        keyword="hey_assistant",
        threshold=0.55,
        cooldown_ms=1200
    )
    kws.start()

    # 加载或生成音频
    if audio_file and Path(audio_file).exists():
        print(f"\n加载音频文件: {audio_file}")
        try:
            import wave
            with wave.open(audio_file, 'rb') as wf:
                frames = wf.getnframes()
                sample_rate = wf.getframerate()
                audio_data = wf.readframes(frames)

            print(f"  采样率: {sample_rate} Hz")
            print(f"  帧数: {frames}")
            print(f"  时长: {frames / sample_rate:.2f} 秒")

            # 转换为numpy数组
            pcm16 = np.frombuffer(audio_data, dtype=np.int16)

            # 如果不是16kHz，重采样
            if sample_rate != 16000:
                print(f"\n重采样到16kHz...")
                import librosa
                pcm16 = librosa.resample(pcm16.astype(np.float32), orig_sr=sample_rate, target_sr=16000)
                pcm16 = (pcm16 * 32767).astype(np.int16)

        except Exception as e:
            print(f"❌ 加载音频失败: {e}")
            print("   将使用生成的测试音频")
            audio_file = None
    else:
        # 生成测试音频（包含唤醒词模式）
        print("\n生成测试音频...")
        print("   (这只是模拟音频，不会真的触发KWS)")
        sample_rate = 16000
        duration = 5  # 秒
        pcm16 = np.random.randint(-1000, 1000, size=sample_rate * duration, dtype=np.int16)

    # 分帧处理
    frame_ms = 20
    frame_size = int(16000 * frame_ms / 1000)

    print(f"\n处理音频帧...")
    print(f"  帧长: {frame_ms}ms")
    print(f"  帧大小: {frame_size} samples")
    print(f"  总帧数: {len(pcm16) // frame_size}")

    detection_count = 0

    for i in range(0, len(pcm16), frame_size):
        frame_data = pcm16[i:i + frame_size]

        if len(frame_data) < frame_size:
            # 填充最后一帧
            frame_data = np.pad(frame_data, (0, frame_size - len(frame_data)))

        frame = AudioFrame(
            ts=i / 16000.0,
            pcm16=frame_data,
            sample_rate=16000
        )

        # 处理帧
        result = kws.process_frame(frame)

        if result:
            detection_count += 1
            print(f"\n✅ KWS 检测到唤醒词!")
            print(f"   关键词: {result.keyword}")
            print(f"   置信度: {result.confidence:.3f}")
            print(f"   时间戳: {result.ts:.3f}")

        # 每100帧显示一次进度
        if (i // frame_size) % 100 == 0:
            elapsed = (i // frame_size) * frame_ms / 1000.0
            print(f"   进度: {elapsed:.1f}秒 / {len(pcm16) / 16000:.1f}秒")

    print(f"\n" + "=" * 70)
    print(f"测试完成!")
    print(f"  总帧数: {len(pcm16) // frame_size}")
    print(f"  检测次数: {detection_count}")
    print(f"  检测率: {detection_count / (len(pcm16) // frame_size) * 100:.2f}%")

    if detection_count == 0:
        print("\n💡 没有检测到唤醒词，这是正常的，因为:")
        print("   1. 测试音频只是随机噪声")
        print("   2. 真实测试需要:")
        print("      - 预录制的唤醒词音频")
        print("      - 或者使用麦克风实时测试")

    kws.stop()


def create_test_audio_with_keyword(output_file: str = "test_hey_assistant.wav"):
    """
    创建包含唤醒词的测试音频

    注意: 这只是一个占位符函数。
    真实的测试需要预录制包含"hey assistant"的音频文件。

    Args:
        output_file: 输出文件路径
    """
    import wave

    print("\n创建测试音频文件...")
    print(f"  输出: {output_file}")
    print("  ⚠️  注意: 这只是随机噪声，不能真实测试KWS")
    print("  真实测试需要录制包含'hey assistant'的音频")

    sample_rate = 16000
    duration = 5  # 秒
    n_samples = sample_rate * duration

    # 生成随机噪声
    audio_data = np.random.randint(-5000, 5000, size=n_samples, dtype=np.int16)

    # 写入WAV文件
    with wave.open(output_file, 'w') as wf:
        wf.setnchannels(1)  # 单声道
        wf.setsampwidth(2)  # 16-bit
        wf.setframerate(sample_rate)
        wf.writeframes(audio_data.tobytes())

    print(f"  ✅ 测试音频已创建: {output_file}")
    print(f"  时长: {duration}秒")
    print(f"  采样率: {sample_rate}Hz")


async def main():
    """主函数"""
    import sys

    print("\n" + "=" * 70)
    print(" WakeFusion KWS 测试工具")
    print("=" * 70)

    print("\n选择测试模式:")
    print("  1. 生成测试音频文件（随机噪声）")
    print("  2. 使用音频文件测试KWS")
    print("  3. 直接测试（使用随机音频）")

    choice = input("\n请选择 (1/2/3): ").strip()

    if choice == "1":
        output_file = input("输出文件名 (默认: test_hey_assistant.wav): ").strip()
        if not output_file:
            output_file = "test_hey_assistant.wav"
        create_test_audio_with_keyword(output_file)
        print(f"\n💡 下一步:")
        print(f"   1. 使用音频编辑软件打开 {output_file}")
        print(f"   2. 录制或导入包含'hey assistant'的音频")
        print(f"   3. 运行: python tests/test_kws.py")
        print(f"   4. 选择模式2，使用该文件测试")

    elif choice == "2":
        audio_file = input("音频文件路径: ").strip()
        if audio_file:
            await test_kws_with_file(audio_file)
        else:
            print("❌ 请提供有效的文件路径")

    else:
        await test_kws_with_file()

    print("\n" + "=" * 70)
    print("💡 提示:")
    print("  - 要测试真实麦克风，请运行: python tests/list_audio_devices.py")
    print("  - 然后运行: python tests/test_audio_driver.py")
    print("=" * 70)


if __name__ == "__main__":
    asyncio.run(main())
