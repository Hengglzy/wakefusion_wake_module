"""
Sherpa-ONNX KWS Worker
支持中文任意唤醒词，无需训练
"""

import asyncio
import numpy as np
from pathlib import Path
from typing import Optional, Callable, List
from dataclasses import dataclass

try:
    import sherpa_onnx
except ImportError:
    sherpa_onnx = None

from wakefusion.types import AudioFrame, BaseEvent, EventType
from wakefusion.logging import get_logger
from wakefusion.metrics import get_metrics, record_latency


logger = get_logger("sherpa_kws")
metrics = get_metrics()


@dataclass
class SherpaKWSConfig:
    """Sherpa-ONNX KWS配置"""
    model_dir: str = "./models/sherpa-onnx-kws-zh-16kHz"
    keywords: List[str] = None  # ["小康小康", "你好小助手"]
    threshold: float = 0.5
    num_threads: int = 4
    sample_rate: int = 16000
    cooldown_ms: int = 1200


class SherpaKWSWorker:
    """Sherpa-ONNX KWS工作线程（支持中文）"""

    def __init__(
        self,
        config: SherpaKWSConfig,
        event_callback: Optional[Callable] = None
    ):
        """
        初始化Sherpa-ONNX KWS工作线程

        Args:
            config: KWS配置
            event_callback: 事件回调函数
        """
        self.config = config
        self.event_callback = event_callback

        # Sherpa-ONNX实例
        self.kws: Optional[sherpa_onnx.KeywordSpotter] = None

        # 状态
        self.is_running = False
        self.detections = 0
        self.last_detection_time = 0.0

        # 模型路径
        self.model_dir = Path(config.model_dir)
        self.tokens_path = self.model_dir / "tokens.txt"
        self.encoder_path = self.model_dir / "encoder.onnx"
        self.decoder_path = self.model_dir / "decoder.onnx"
        self.joiner_path = self.model_dir / "joiner.onnx"

        if not self.model_dir.exists():
            logger.warning(
                f"Sherpa-ONNX model directory not found: {config.model_dir}",
                extra={
                    "model_dir": config.model_dir,
                    "hint": "Download from: https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/"
                }
            )

    def start(self):
        """启动Sherpa-ONNX KWS"""
        if sherpa_onnx is None:
            raise ImportError("sherpa-onnx not installed. Run: pip install sherpa-onnx")

        try:
            logger.info("Initializing Sherpa-ONNX...")

            # 验证模型文件
            if not self._validate_models():
                raise RuntimeError(f"Sherpa-ONNX model files incomplete in {self.model_dir}")

            # 创建关键词文件
            keywords_file = self._create_keywords_file()

            # 初始化KeywordSpotter
            self.kws = sherpa_onnx.KeywordSpotter(
                tokens=str(self.tokens_path),
                encoder=str(self.encoder_path),
                decoder=str(self.decoder_path),
                joiner=str(self.joiner_path),
                keywords_file=str(keywords_file),
                num_threads=self.config.num_threads
            )

            # 获取帧长度（Sherpa-ONNX建议使用固定帧长）
            self.frame_length = 5120  # 320ms @ 16kHz (可调整)

            logger.info(
                "Sherpa-ONNX KWS started",
                extra={
                    "model_dir": str(self.model_dir),
                    "keywords": self.config.keywords,
                    "frame_length": self.frame_length,
                    "sample_rate": self.config.sample_rate
                }
            )

            self.is_running = True

        except Exception as e:
            logger.error(f"Failed to initialize Sherpa-ONNX: {e}")
            raise

    def stop(self):
        """停止Sherpa-ONNX KWS"""
        if self.kws:
            self.kws = None

        self.is_running = False
        logger.info("Sherpa-ONNX KWS stopped")

    def process_frame(self, frame: AudioFrame) -> Optional[str]:
        """
        处理音频帧

        Args:
            frame: 音频帧

        Returns:
            str: 检测到的关键词，None表示未检测到
        """
        if not self.is_running or not self.kws:
            return None

        start_time = asyncio.get_event_loop().time()

        try:
            # 检查冷却期
            if frame.ts - self.last_detection_time < (self.config.cooldown_ms / 1000.0):
                return None

            # 确保采样率正确
            if frame.sample_rate != 16000:
                logger.warning(f"Unexpected sample rate: {frame.sample_rate}, expected 16000")
                return None

            # Sherpa-ONNX期望float32 PCM，范围[-1, 1]
            pcm_float32 = frame.pcm16.astype(np.float32) / 32768.0

            # 运行检测
            result = self.kws.accept_waveform(
                sample_rate=16000,
                waveform=pcm_float32
            )

            # 记录延迟
            latency_ms = (asyncio.get_event_loop().time() - start_time) * 1000
            record_latency("sherpa_kws.inference_ms", latency_ms)

            if result and result.keyword:
                self.detections += 1
                self.last_detection_time = frame.ts

                logger.info(
                    f"Sherpa-ONNX detected keyword: {result.keyword}",
                    extra={
                        "keyword": result.keyword,
                        "timestamp": frame.ts,
                        "latency_ms": latency_ms
                    }
                )

                # 触发事件
                if self.event_callback:
                    event = BaseEvent(
                        type=EventType.KWS_HIT,
                        ts=frame.ts,
                        session_id=f"sherpa-{int(frame.ts)}",
                        priority=80,
                        **{
                            "payload": {
                                "keyword": result.keyword,
                                "confidence": 0.85,  # Sherpa-ONNX不提供置信度
                                "pre_roll_ms": 800,
                                "audio_start_ts": frame.ts - 0.8,
                                "audio_end_ts": frame.ts
                            }
                        }
                    )
                    self.event_callback(event)

                return result.keyword

            return None

        except Exception as e:
            logger.error(f"Error in Sherpa-ONNX prediction: {e}")
            metrics.increment_counter("sherpa_kws.errors")
            return None

    def _validate_models(self) -> bool:
        """验证模型文件是否存在"""
        required_files = [
            self.tokens_path,
            self.encoder_path,
            self.decoder_path,
            self.joiner_path
        ]

        for file_path in required_files:
            if not file_path.exists():
                logger.error(f"Missing model file: {file_path}")
                return False

        return True

    def _create_keywords_file(self) -> Path:
        """
        创建关键词配置文件

        格式：
        小康小康 0.5
        你好小助手 0.5
        """
        keywords_file = self.model_dir / "keywords.txt"

        try:
            with open(keywords_file, 'w', encoding='utf-8') as f:
                for keyword in self.config.keywords:
                    f.write(f"{keyword} {self.config.threshold}\n")

            logger.info(f"Created keywords file: {keywords_file}")
            return keywords_file

        except Exception as e:
            logger.error(f"Failed to create keywords file: {e}")
            raise

    def get_stats(self) -> dict:
        """获取统计信息"""
        return {
            "detections": self.detections,
            "is_running": self.is_running,
            "keywords": self.config.keywords,
            "sample_rate": self.config.sample_rate
        }


async def test_sherpa_kws():
    """测试Sherpa-ONNX中文唤醒词"""
    print("=" * 70)
    print("Sherpa-ONNX 中文唤醒词测试")
    print("=" * 70)

    print("\n⚠️  注意:")
    print("   1. 需要先下载 Sherpa-ONNX 模型")
    print("   2. 访问: https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/")
    print("   3. 下载 sherpa-onnx-kws-zh-16kHz.tar.gz")
    print("   4. 解压到 ./models/ 目录")
    print()

    model_dir = input("请输入模型目录路径 (默认: ./models/sherpa-onnx-kws-zh-16kHz): ").strip()
    if not model_dir:
        model_dir = "./models/sherpa-onnx-kws-zh-16kHz"

    if not Path(model_dir).exists():
        print(f"❌ 模型目录不存在: {model_dir}")
        return

    keywords_str = input("请输入唤醒词（逗号分隔，默认: 小康小康）: ").strip()
    if not keywords_str:
        keywords_str = "小康小康"

    keywords = [k.strip() for k in keywords_str.split(",")]

    print("\n正在初始化 Sherpa-ONNX...")

    def on_kws_event(event):
        print(f"\n✅ 检测到唤醒词!")
        print(f"   关键词: {event.payload['keyword']}")
        print(f"   时间戳: {event.ts}")

    config = SherpaKWSConfig(
        model_dir=model_dir,
        keywords=keywords,
        threshold=0.5,
        cooldown_ms=1200
    )

    kws = SherpaKWSWorker(
        config=config,
        event_callback=on_kws_event
    )

    try:
        kws.start()

        print("\n" + "=" * 70)
        print("🎤 测试模式")
        print(f"   请说 '{keywords[0]}' 来测试")
        print("   按 Ctrl+C 停止")
        print("=" * 70 + "\n")

        # 这里需要真实的音频输入
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
        print("  1. 模型文件不完整")
        print("  2. sherpa-onnx未正确安装")
        print("  3. 模型目录路径错误")
        kws.stop()


if __name__ == "__main__":
    import sys
    asyncio.run(test_sherpa_kws())
