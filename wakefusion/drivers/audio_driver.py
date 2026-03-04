"""
XVF3800 音频驱动
负责设备枚举、绑定、采集和自动重连
"""

import asyncio
import numpy as np
import pyaudio
from typing import Optional, Callable, Dict, Any
from dataclasses import dataclass
import time

from wakefusion.types import AudioFrameRaw
from wakefusion.logging import get_logger
from wakefusion.metrics import get_metrics, record_latency, set_gauge


logger = get_logger("audio_driver")
metrics = get_metrics()


@dataclass
class DeviceInfo:
    """设备信息"""
    index: int
    name: str
    sample_rate: int
    channels: int
    is_xvf3800: bool


class XVF3800Driver:
    """XVF3800音频驱动"""

    def __init__(
        self,
        device_match: str = "XVF3800",
        sample_rate: int = 48000,
        channels: int = 1,
        frame_ms: int = 20,
        callback: Optional[Callable[[AudioFrameRaw], None]] = None
    ):
        """
        初始化音频驱动

        Args:
            device_match: 设备匹配名称
            sample_rate: 采样率
            channels: 声道数
            frame_ms: 帧长（毫秒）
            callback: 音频帧回调函数
        """
        self.device_match = device_match
        self.sample_rate = sample_rate
        self.channels = channels
        self.frame_ms = frame_ms
        self.callback = callback

        # 计算帧大小
        self.frame_size = int(sample_rate * frame_ms / 1000)

        # PyAudio实例
        self.pyaudio: Optional[pyaudio.PyAudio] = None
        self.stream: Optional[pyaudio.Stream] = None

        # 设备信息
        self.device_info: Optional[DeviceInfo] = None

        # 状态
        self.is_running = False
        self.reconnect_interval = 1.0  # 重连间隔（秒）

        logger.info(
            "XVF3800Driver initialized",
            extra={
                "device_match": device_match,
                "sample_rate": sample_rate,
                "frame_ms": frame_ms
            }
        )

    def find_device(self) -> Optional[DeviceInfo]:
        """
        查找音频设备（支持XVF3800或默认设备）

        Returns:
            DeviceInfo: 设备信息，如果未找到则返回None
        """
        if not self.pyaudio:
            self.pyaudio = pyaudio.PyAudio()

        # 设备搜索日志改为 DEBUG，避免在终端反复刷屏
        logger.debug(f"Searching for audio device matching: {self.device_match}")

        # 如果配置为"default"，使用系统默认输入设备
        if self.device_match.lower() == "default":
            default_input = self.pyaudio.get_default_input_device_info()
            device_name = default_input['name']
            device_index = default_input['index']
            max_sample_rate = int(default_input['defaultSampleRate'])

            logger.info(
                f"Using system default audio device: {device_name}",
                extra={
                    "device_index": device_index,
                    "device_name": device_name,
                    "max_sample_rate": max_sample_rate
                }
            )

            return DeviceInfo(
                index=device_index,
                name=device_name,
                sample_rate=max_sample_rate,
                channels=min(self.channels, default_input['maxInputChannels']),
                is_xvf3800=False
            )

        # 否则查找匹配的设备
        for i in range(self.pyaudio.get_device_count()):
            device_info = self.pyaudio.get_device_info_by_index(i)

            # 检查是否是输入设备
            if device_info['maxInputChannels'] == 0:
                continue

            device_name = device_info['name']
            is_target_device = self.device_match.lower() in device_name.lower()

            logger.debug(
                f"Device {i}: {device_name}",
                extra={
                    "device_index": i,
                    "device_name": device_name,
                    "is_target_device": is_target_device,
                    "max_input_channels": device_info['maxInputChannels'],
                    "max_sample_rate": int(device_info['defaultSampleRate'])
                }
            )

            if is_target_device:
                # 使用设备支持的最大采样率
                max_sample_rate = int(device_info['defaultSampleRate'])

                logger.info(
                    f"Found target device: {device_name}",
                    extra={
                        "device_index": i,
                        "device_name": device_name,
                        "max_sample_rate": max_sample_rate
                    }
                )

                return DeviceInfo(
                    index=i,
                    name=device_name,
                    sample_rate=max_sample_rate,
                    channels=min(self.channels, device_info['maxInputChannels']),
                    is_xvf3800="xvf3800" in device_name.lower()
                )

        # 如果没找到，回退到默认设备
        logger.warning(f"No device found matching: {self.device_match}, falling back to default")
        return self.find_device_with_fallback()

    def find_device_with_fallback(self) -> Optional[DeviceInfo]:
        """回退到默认设备"""
        try:
            default_input = self.pyaudio.get_default_input_device_info()
            device_name = default_input['name']
            device_index = default_input['index']
            max_sample_rate = int(default_input['defaultSampleRate'])

            logger.warning(
                f"Using fallback device: {device_name}",
                extra={
                    "device_index": device_index,
                    "device_name": device_name
                }
            )

            return DeviceInfo(
                index=device_index,
                name=device_name,
                sample_rate=max_sample_rate,
                channels=min(self.channels, default_input['maxInputChannels']),
                is_xvf3800=False
            )
        except Exception as e:
            logger.error(f"Failed to get fallback device: {e}")
            return None

    def start(self):
        """启动音频采集"""
        if self.is_running:
            logger.warning("Audio driver already running")
            return

        # 查找设备
        self.device_info = self.find_device()
        if not self.device_info:
            raise RuntimeError(f"XVF3800 device not found: {self.device_match}")

        # 初始化PyAudio
        if not self.pyaudio:
            self.pyaudio = pyaudio.PyAudio()

        # 打开音频流
        try:
            self.stream = self.pyaudio.open(
                format=pyaudio.paInt16,
                channels=self.device_info.channels,
                rate=self.device_info.sample_rate,
                input=True,
                input_device_index=self.device_info.index,
                frames_per_buffer=self.frame_size,
                stream_callback=self._audio_callback
            )

            self.stream.start_stream()
            self.is_running = True

            logger.info(
                "Audio stream started",
                extra={
                    "device_name": self.device_info.name,
                    "sample_rate": self.device_info.sample_rate,
                    "channels": self.device_info.channels,
                    "frame_size": self.frame_size
                }
            )

        except Exception as e:
            logger.error(f"Failed to start audio stream: {e}")
            self.stop()
            raise

    def _audio_callback(
        self,
        in_data: bytes,
        frame_count: int,
        time_info: Dict[str, float],
        status: int
    ) -> tuple[Optional[bytes], int]:
        """
        音频回调函数（PyAudio线程）

        Args:
            in_data: 音频数据
            frame_count: 帧大小
            time_info: 时间信息
            status: 状态标志

        Returns:
            (out_data, flag): 输出数据和标志
        """
        start_time = time.perf_counter()

        try:
            # 转换为numpy数组
            pcm_data = np.frombuffer(in_data, dtype=np.int16)

            # 如果是多声道，转换为单声道
            if self.device_info.channels > 1:
                pcm_data = pcm_data.reshape(-1, self.device_info.channels)
                pcm_data = pcm_data.mean(axis=1).astype(np.int16)

            # 创建音频帧
            frame = AudioFrameRaw(
                ts=time_info.get('input_buffer_adctime', time.time()),
                pcm16=pcm_data,
                sample_rate=self.device_info.sample_rate,
                channels=1
            )

            # 回调处理
            if self.callback:
                self.callback(frame)

            # 记录指标
            latency_ms = (time.perf_counter() - start_time) * 1000
            record_latency("audio.callback_latency_ms", latency_ms)
            metrics.increment_counter("audio.frames_processed")

            return (None, pyaudio.paContinue)

        except Exception as e:
            logger.error(f"Error in audio callback: {e}")
            metrics.increment_counter("audio.callback_errors")
            return (None, pyaudio.paContinue)

    def stop(self):
        """停止音频采集"""
        if not self.is_running:
            return

        self.is_running = False

        if self.stream:
            try:
                self.stream.stop_stream()
                self.stream.close()
            except Exception as e:
                logger.error(f"Error stopping stream: {e}")
            self.stream = None

        if self.pyaudio:
            try:
                self.pyaudio.terminate()
            except Exception as e:
                logger.error(f"Error terminating PyAudio: {e}")
            self.pyaudio = None

        logger.info("Audio driver stopped")

    async def run_with_reconnect(self):
        """
        运行音频驱动并自动重连

        当检测到断连时，自动尝试重连
        """
        consecutive_errors = 0
        max_errors = 5

        while self.is_running:
            try:
                # 等待一小段时间
                await asyncio.sleep(self.reconnect_interval)

                # 检查流状态
                if self.stream and self.stream.is_active():
                    consecutive_errors = 0

                    # 更新设备状态指标
                    set_gauge("audio.device_connected", 1.0)
                    set_gauge("audio.fps", 1000 / self.frame_ms)

                elif consecutive_errors > max_errors:
                    # 检测到断连，尝试重连
                    logger.warning(f"Audio stream inactive, reconnecting... (attempt {consecutive_errors})")
                    metrics.increment_counter("audio.reconnect_count")

                    self.stop()
                    await asyncio.sleep(1.0)
                    self.start()

                    consecutive_errors = 0

            except Exception as e:
                logger.error(f"Error in reconnect loop: {e}")
                consecutive_errors += 1

                if consecutive_errors > max_errors:
                    logger.critical("Max reconnection errors reached, stopping")
                    self.stop()
                    break

    def get_device_status(self) -> Dict[str, Any]:
        """获取设备状态"""
        return {
            "device_name": self.device_info.name if self.device_info else "None",
            "sample_rate": self.device_info.sample_rate if self.device_info else 0,
            "channels": self.device_info.channels if self.device_info else 0,
            "is_running": self.is_running,
            "is_active": self.stream.is_active() if self.stream else False
        }


async def test_audio_driver():
    """测试音频驱动"""
    import sys

    received_frames = []

    def on_audio_frame(frame: AudioFrameRaw):
        received_frames.append(frame)
        if len(received_frames) <= 5:
            print(f"[{frame.ts:.3f}] Received frame: {len(frame.pcm16)} samples @ {frame.sample_rate}Hz")

    driver = XVF3800Driver(
        device_match="default",  # 使用默认麦克风
        sample_rate=48000,
        frame_ms=20,
        callback=on_audio_frame
    )

    try:
        driver.start()
        print("\n" + "="*60)
        print("🎤 正在录音...")
        print("   请说 'hey assistant' 来测试唤醒词检测")
        print("   按 Ctrl+C 停止")
        print("="*60 + "\n")

        await asyncio.sleep(10)  # 录音10秒
        driver.stop()

        print(f"\n✅ 录音完成!")
        print(f"   总帧数: {len(received_frames)}")
        print(f"   实际FPS: {len(received_frames) / 10:.1f}")

    except KeyboardInterrupt:
        print("\n⏹  停止录音")
        driver.stop()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n💡 可能的原因:")
        print("   1. 没有可用的麦克风")
        print("   2. 麦克风被其他应用占用")
        print("   3. pyaudio安装不完整")
        print("\n   请运行: python tests/list_audio_devices.py")
        print("          查看可用的音频设备")
        driver.stop()


if __name__ == "__main__":
    asyncio.run(test_audio_driver())
