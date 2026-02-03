"""
Porcupine 中文唤醒词集成示例
支持"小康小康"等中文唤醒词
"""

import asyncio
import pvporcupine
from pathlib import Path
from typing import Optional, Callable

from wakefusion.types import AudioFrame, BaseEvent, EventType
from wakefusion.logging import get_logger


logger = get_logger("porcupine_kws")
metrics = get_metrics()


class PorcupineKWSWorker:
    """Porcupine KWS工作线程（支持中文）"""

    def __init__(
        self,
        keyword_path: str,
        access_key: str,
        sensitivity: float = 0.5,
        library_path: Optional[str] = None,
        event_callback: Optional[Callable] = None
    ):
        """
        初始化Porcupine KWS工作线程

        Args:
            keyword_path: 唤醒词模型文件路径 (.ppn文件)
            access_key: Picovoice访问密钥
            sensitivity: 灵敏度 (0-1)
            library_path: Porcupine库路径（可选）
            event_callback: 事件回调函数
        """
        self.keyword_path = keyword_path
        self.access_key = access_key
        self.sensitivity = sensitivity
        self.library_path = library_path
        self.event_callback = event_callback

        # Porcupine实例
        self.porcupine: Optional[pvporcupine.Porcupine] = None

        # 状态
        self.is_running = False
        self.detections = 0

    def start(self):
        """启动Porcupine KWS"""
        try:
            logger.info("Initializing Porcupine...")

            # 初始化Porcupine
            self.porcupine = pvporcupine.create(
                keyword_paths=[self.keyword_path],
                sensitivities=[self.sensitivity],
                access_key=self.access_key,
                library_path=self.library_path
            )

            # 获取帧长度
            self.frame_length = self.porcupine.frame_length
            self.sample_rate = 16000

            logger.info(
                "Porcupine KWS started",
                extra={
                    "keyword_path": self.keyword_path,
                    "frame_length": self.frame_length,
                    "sample_rate": self.sample_rate
                }
            )

            self.is_running = True

        except Exception as e:
            logger.error(f"Failed to initialize Porcupine: {e}")
            raise

    def stop(self):
        """停止Porcupine KWS"""
        if self.porcupine:
            self.porcupine.delete()
            self.porcupine = None

        self.is_running = False
        logger.info("Porcupine KWS stopped")

    def process_frame(self, frame: AudioFrame) -> Optional[int]:
        """
        处理音频帧

        Args:
            frame: 音频帧

        Returns:
            int: 检测到的唤醒词索引，-1表示未检测到
        """
        if not self.is_running or not self.porcupine:
            return -1

        try:
            # 确保采样率正确
            if frame.sample_rate != 16000:
                logger.warning(f"Unexpected sample rate: {frame.sample_rate}, expected 16000")
                return -1

            # Porcupine期望int16 PCM
            pcm_data = frame.pcm16

            # 检查帧长度
            if len(pcm_data) != self.frame_length:
                # 填充或裁剪
                if len(pcm_data) < self.frame_length:
                    pcm_data = np.pad(pcm_data, (0, self.frame_length - len(pcm_data)))
                else:
                    pcm_data = pcm_data[:self.frame_length]

            # 运行检测
            keyword_index = self.porcupine.process(pcm_data)

            if keyword_index >= 0:
                self.detections += 1

                logger.info(
                    f"Porcupine detected keyword: {keyword_index}",
                    extra={
                        "keyword_index": keyword_index,
                        "timestamp": frame.ts
                    }
                )

                # 触发事件
                if self.event_callback:
                    event = BaseEvent(
                        type=EventType.KWS_HIT,
                        ts=frame.ts,
                        session_id=f"porcupine-{int(frame.ts)}",
                        priority=80,
                        **{
                            "payload": {
                                "keyword": "xiaokang_xiaokang",
                                "confidence": 0.8,  # Porcupine不提供置信度，使用固定值
                                "pre_roll_ms": 800,
                                "audio_start_ts": frame.ts - 0.8,
                                "audio_end_ts": frame.ts
                            }
                        }
                    )
                    self.event_callback(event)

            return keyword_index

        except Exception as e:
            logger.error(f"Error in Porcupine prediction: {e}")
            return -1

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "detections": self.detections,
            "is_running": self.is_running,
            "frame_length": self.frame_length if self.porcupine else 0,
            "sample_rate": self.sample_rate
        }


async def test_porcupine():
    """测试Porcupine中文唤醒词"""
    import numpy as np

    print("=" * 70)
    print("Porcupine 中文唤醒词测试")
    print("=" * 70)

    print("\n⚠️  注意:")
    print("   1. 需要先在 Picovoice Console 注册并下载唤醒词模型")
    print("   2. 访问: https://console.picovoice.ai/")
    print("   3. 创建新项目，训练中文唤醒词（如'小康小康'）")
    print("   4. 下载 .ppn 模型文件")
    print()

    keyword_path = input("请输入 .ppn 模型文件路径: ").strip()

    if not Path(keyword_path).exists():
        print(f"❌ 文件不存在: {keyword_path}")
        return

    access_key = input("请输入 Access Key: ").strip()

    print("\n正在初始化 Porcupine...")

    def on_kws_event(event):
        print(f"\n✅ 检测到唤醒词!")
        print(f"   时间戳: {event.ts}")

    kws = PorcupineKWSWorker(
        keyword_path=keyword_path,
        access_key=access_key,
        sensitivity=0.5,
        event_callback=on_kws_event
    )

    try:
        kws.start()

        print("\n" + "=" * 70)
        print("🎤 测试模式")
        print("   请说 '小康小康' 来测试")
        print("   按 Ctrl+C 停止")
        print("=" * 70 + "\n")

        # 这里需要真实的音频输入
        # 暂时使用模拟数据
        print("使用模拟音频数据（不会触发检测）...")
        print("真实测试需要麦克风输入...")

        await asyncio.sleep(5)

        kws.stop()

        print(f"\n测试完成!")
        print(f"检测次数: {kws.detections}")

    except KeyboardInterrupt:
        print("\n⏹  停止测试")
        kws.stop()
    except Exception as e:
        print(f"\n❌ 错误: {e}")
        print("\n可能的原因:")
        print("  1. Access Key无效")
        print("  2. 模型文件路径错误")
        print("  3. 网络问题（首次需要在线验证）")
        kws.stop()


if __name__ == "__main__":
    import sys
    asyncio.run(test_porcupine())
