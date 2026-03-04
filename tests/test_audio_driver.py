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
        # 新增：根据设备标志判断是否为 XVF3800 专业阵列麦克风
        is_xvf3800 = getattr(device, "is_xvf3800", False)

        print(f"\n✓ 找到音频设备:")
        print(f"  - 设备名称: {getattr(device, 'name', 'Unknown')}")
        print(f"  - 设备索引: {getattr(device, 'index', 'N/A')}")
        print(f"  - 采样率: {getattr(device, 'sample_rate', 'N/A')} Hz")
        print(f"  - 声道数: {getattr(device, 'channels', 'N/A')}")

        if is_xvf3800:
            print("\n✓ 成功识别 XVF3800 专业阵列麦克风")
        else:
            print("\n⚠ 未找到 XVF3800，当前正在使用回退设备（普通麦克风）")
            print("  请检查 USB 连接和设备选择：")
            print("  1. 确认 XVF3800 已通过 USB 正确连接")
            print("  2. 在 Windows 声音设置中确认设备已启用且可见")
            print("  3. 检查 config.yaml 中的 device_match 是否正确匹配 XVF3800 名称")
    else:
        print(f"\n✗ 未找到任何可用的音频输入设备")
        print("  请检查:")
        print("  1. 设备是否已连接")
        print("  2. Windows声音设置中是否可见")
        print("  3. config.yaml中device_match配置是否正确")


async def test_audio_capture():
    """测试音频采集"""
    print("\n" + "=" * 60)
    print("音频采集测试 (5秒)")
    print("=" * 60)

    # 引入 VAD 依赖（实时语音检测）
    try:
        import webrtcvad
    except ImportError:
        print("\n✗ webrtcvad 未安装，无法进行实时 VAD 测试")
        print("  请运行: pip install webrtcvad")
        return

    # 初始化 VAD（攻击性级别 2）
    vad = webrtcvad.Vad(2)

    frame_count = 0
    last_print_frame = 0  # 控制打印频率（每 5 帧打印一次）

    def on_audio_frame(frame: AudioFrameRaw):
        nonlocal frame_count, last_print_frame
        frame_count += 1

        # 将 PCM16 数据转换为 numpy 数组
        pcm16_raw = frame.pcm16
        if pcm16_raw is None:
            return

        pcm16_raw = np.asarray(pcm16_raw, dtype=np.int16)
        actual_sample_rate = int(frame.sample_rate)
        target_sample_rate = 16000  # VAD 要求 16kHz

        # 计算原始音频的 RMS（用于调试和显示，使用原始数据）
        if len(pcm16_raw) > 0:
            rms_raw = float(np.sqrt(np.mean(pcm16_raw.astype(np.float32) ** 2)))
            pcm16_max = float(np.max(np.abs(pcm16_raw)))
            pcm16_min = float(np.min(pcm16_raw))
        else:
            rms_raw = 0.0
            pcm16_max = 0.0
            pcm16_min = 0.0

        # 调试信息：只在第一帧打印，检查音频数据是否正常
        if frame_count == 1:
            print(f"\n[调试] 第一帧音频数据:")
            print(f"  原始采样率: {actual_sample_rate} Hz")
            print(f"  原始帧大小: {len(pcm16_raw)} 采样点")
            print(f"  RMS: {rms_raw:.2f}")
            print(f"  数据范围: [{pcm16_min:.0f}, {pcm16_max:.0f}]")
            if rms_raw < 1.0:
                print(f"  ⚠️  警告: RMS 过低，可能麦克风没有输入或音量过低")

        # WebRTC VAD 只支持 8000, 16000, 32000, 48000 Hz
        # 计算期望的采样点数：16k 采样率下，20ms 帧长 => 320 个采样点
        expected_samples = int(target_sample_rate * 0.02)  # 20ms @ 16kHz = 320 采样点
        expected_bytes = expected_samples * 2  # 16-bit = 2 字节/采样点

        # 如果实际采样率不是 16kHz，需要下采样
        pcm16 = pcm16_raw.copy()  # 保存原始数据用于 RMS 计算
        if actual_sample_rate != target_sample_rate:
            # 优先使用 scipy.signal.resample 进行高质量下采样
            try:
                from scipy import signal
                # 使用 scipy 的 resample 进行下采样，直接下采样到期望的采样点数
                pcm16_resampled = signal.resample(pcm16, expected_samples).astype(np.int16)
                pcm16 = pcm16_resampled
                sample_rate = target_sample_rate
            except ImportError:
                # 如果 scipy 不可用，使用简单的线性插值下采样
            if actual_sample_rate > target_sample_rate:
                # 使用简单的索引下采样
                    indices = np.linspace(0, len(pcm16) - 1, expected_samples, dtype=np.int32)
                pcm16 = pcm16[indices]
                sample_rate = target_sample_rate
            else:
                # 上采样（不太可能，但为了完整性）
                ratio = target_sample_rate / actual_sample_rate
                    pcm16_upsampled = np.repeat(pcm16, int(ratio))
                    # 裁剪或填充到期望长度
                    if len(pcm16_upsampled) > expected_samples:
                        pcm16 = pcm16_upsampled[:expected_samples]
                    else:
                        padding = np.zeros(expected_samples - len(pcm16_upsampled), dtype=np.int16)
                        pcm16 = np.concatenate([pcm16_upsampled, padding])
                sample_rate = target_sample_rate
        else:
            sample_rate = target_sample_rate
            # 确保帧大小精确匹配（如果已经是 16kHz，但帧大小不匹配）
        if len(pcm16) != expected_samples:
            if len(pcm16) > expected_samples:
                pcm16 = pcm16[:expected_samples]
            else:
                padding = np.zeros(expected_samples - len(pcm16), dtype=np.int16)
                pcm16 = np.concatenate([pcm16, padding])

        # 转换为 bytes
        pcm_bytes = pcm16.tobytes()

        # 使用原始数据的 RMS（用于显示）
        rms = rms_raw

        try:
            is_speech = vad.is_speech(pcm_bytes, sample_rate)
        except Exception as vad_err:
            # 增加容错打印：输出 VAD 失败的具体错误原因，方便调试
            # 只在第一次失败时打印，避免刷屏
            if not hasattr(on_audio_frame, '_vad_error_printed'):
                print(f"\n[警告] VAD 推理失败: {vad_err}")
                print(f"  实际采样率: {actual_sample_rate} Hz")
                print(f"  目标采样率: {target_sample_rate} Hz")
                print(f"  帧大小: {len(pcm16)} 采样点 ({len(pcm_bytes)} 字节)")
                print(f"  期望大小: {expected_samples} 采样点 ({expected_bytes} 字节)")
                on_audio_frame._vad_error_printed = True
            return

        # 限制打印频率：每 5 帧打印一次最新状态
        if frame_count - last_print_frame < 5:
            return
        last_print_frame = frame_count

        # 使用单行刷新显示当前状态
        # 注意：在 Windows PowerShell 中，\r 可能被日志输出覆盖，所以使用 sys.stdout 直接写入
        import sys
        status_text = f"🎤 [SPEECH] | RMS: {rms:.2f}" if is_speech else f"☁️ [SILENCE] | RMS: {rms:.2f}"
        # 清除当前行并写入新内容
        sys.stdout.write(f"\r{' ' * 80}\r")  # 清除当前行（80字符宽度）
        sys.stdout.write(status_text)
        sys.stdout.flush()

    # 注意：XVF3800 设备可能不支持 16kHz，实际采样率可能是 44100 或 48000
    # 测试代码会自动下采样到 16kHz 以满足 VAD 要求
    # 使用设备实际采样率（通常为 44100 或 48000），然后在回调中下采样到 16kHz
    driver = XVF3800Driver(
        device_match="XVF3800",
        sample_rate=48000,  # 使用设备常见采样率，实际会使用设备支持的最高采样率
        frame_ms=20,
        callback=on_audio_frame
    )

    try:
        print("\n启动音频采集...")
        print("提示：现在您可以对着麦克风说话，观察 [SPEECH] 状态的切换")
        driver.start()
        print(f"\n采集中... (设备: {driver.device_info.name})")

        await asyncio.sleep(5)

        driver.stop()
        # 换行，避免最后一行状态覆盖后续输出
        import sys
        sys.stdout.write("\n")  # 确保换行
        sys.stdout.flush()

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
        import openwakeword
        from openwakeword.model import Model

        # 确保模型权重文件已下载到本地
        print("\n正在检查并同步模型权重文件...")
        try:
            openwakeword.utils.download_models()
            print("✓ 模型权重检查/同步完成")
        except Exception as e:
            print(f"⚠ 模型权重自动下载失败，将尝试使用本地已存在的模型文件。错误: {e}")

        print("\n加载openWakeWord模型...")
        print("强制使用 ONNX 推理后端尝试加载模型...")

        # 优先使用 ONNX 推理后端，避免 tflite-runtime 相关报错
        try:
            model = Model(inference_framework="onnx")
        except Exception as e:
            print(f"ONNX 推理后端加载失败，回退到默认后端。错误: {e}")
            # 回退到默认初始化（不带参数），保持兼容性
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
        # 打印 openwakeword 实际安装路径，便于排查环境/安装问题
        try:
            import openwakeword as _ow_debug
            print(f"openwakeword.__path__ = {_ow_debug.__path__}")
        except Exception as path_err:
            print(f"无法获取 openwakeword.__path__，错误: {path_err}")


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
    await test_audio_capture()  # 取消注释以启用

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
