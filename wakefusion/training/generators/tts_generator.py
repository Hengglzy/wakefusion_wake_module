"""
TTS数据生成器
使用Edge-TTS生成唤醒词音频
"""

import asyncio
import edge_tts
from pathlib import Path
from typing import List, Optional
import librosa
import soundfile as sf
import numpy as np

from wakefusion.training.config import TTSConfig, TTSVoice, DataAugmentationConfig
from wakefusion.logging import get_logger


logger = get_logger("tts_generator")


class TTSDataGenerator:
    """TTS数据生成器"""

    def __init__(self, config: TTSConfig):
        """
        初始化TTS生成器

        Args:
            config: TTS配置
        """
        self.config = config
        self.semaphore = asyncio.Semaphore(5)  # 并发控制，最多5个并发TTS请求

        logger.info(
            "TTSDataGenerator initialized",
            extra={
                "voices": [v.value for v in config.voices],
                "samples_per_voice": config.samples_per_voice,
                "sample_rate": config.sample_rate
            }
        )

    async def generate(
        self,
        text: str,
        output_dir: Path,
        negative_samples: Optional[List[str]] = None
    ) -> List[Path]:
        """
        生成TTS音频数据

        Args:
            text: 唤醒词文本
            output_dir: 输出目录
            negative_samples: 负样本列表

        Returns:
            生成的音频文件路径列表
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        audio_files = []
        tasks = []

        # 生成正样本
        logger.info(f"开始生成正样本: {text}")
        for voice in self.config.voices:
            for i in range(self.config.samples_per_voice):
                task = self._generate_audio(
                    text=text,
                    voice=voice.value,
                    output_path=output_dir / f"positive_{voice.value}_{i:04d}.wav",
                    index=i
                )
                tasks.append(task)

        # 生成负样本
        if negative_samples:
            logger.info(f"开始生成负样本: {negative_samples}")
            for neg_text in negative_samples:
                for voice in self.config.voices:
                    for i in range(self.config.samples_per_voice // 5):  # 负样本少一些
                        task = self._generate_audio(
                            text=neg_text,
                            voice=voice.value,
                            output_path=output_dir / f"negative_{neg_text}_{voice.value}_{i:04d}.wav",
                            index=i
                        )
                        tasks.append(task)

        # 并发执行
        logger.info(f"总共 {len(tasks)} 个生成任务，并发执行...")
        results = await asyncio.gather(*tasks, return_exceptions=True)

        for result in results:
            if isinstance(result, Path):
                audio_files.append(result)
            elif isinstance(result, Exception):
                logger.error(f"TTS生成失败: {result}")

        logger.info(
            f"TTS生成完成，共 {len(audio_files)} 个文件",
            extra={"output_dir": str(output_dir)}
        )

        return audio_files

    async def _generate_audio(
        self,
        text: str,
        voice: str,
        output_path: Path,
        index: int
    ) -> Path:
        """
        生成单个音频文件

        Args:
            text: 文本内容
            voice: 音色
            output_path: 输出路径
            index: 索引

        Returns:
            生成的音频文件路径
        """
        async with self.semaphore:
            try:
                # 生成TTS音频（临时MP3文件）
                temp_mp3 = output_path.with_suffix('.mp3')
                communicate = edge_tts.Communicate(text, voice)
                await communicate.save(str(temp_mp3))

                # 转换为WAV 16kHz
                audio, sr = librosa.load(temp_mp3, sr=self.config.sample_rate)
                sf.write(output_path, audio, self.config.sample_rate)

                # 删除临时文件
                temp_mp3.unlink()

                if (index + 1) % 10 == 0:
                    logger.info(f"已生成 {index + 1} 个样本")

                return output_path

            except Exception as e:
                logger.error(f"生成音频失败 {text} ({voice}): {e}")
                raise

    async def augment(
        self,
        audio_files: List[Path],
        output_dir: Path,
        config: DataAugmentationConfig
    ) -> List[Path]:
        """
        数据增强

        Args:
            audio_files: 原始音频文件列表
            output_dir: 输出目录
            config: 增强配置

        Returns:
            增强后的音频文件路径列表
        """
        if not config.enabled:
            logger.info("数据增强已禁用")
            return audio_files

        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        augmented_files = list(audio_files)  # 包含原始文件
        total_augmented = 0

        for audio_file in audio_files:
            try:
                # 加载音频
                audio, sr = sf.read(audio_file)

                # 速度扰动
                if config.speed_perturbation:
                    for rate in config.speed_rates:
                        if rate == 1.0:
                            continue  # 跳过原始速度

                        augmented_audio = self._resample_audio(audio, sr, rate)
                        output_path = output_dir / f"{audio_file.stem}_speed{rate:.1f}.wav"
                        sf.write(output_path, augmented_audio, sr)
                        augmented_files.append(output_path)
                        total_augmented += 1

                # 增益扰动
                if config.gain_perturbation:
                    for gain_db in np.arange(
                        config.gain_range[0],
                        config.gain_range[1],
                        2.0
                    ):
                        augmented_audio = self._apply_gain(audio, gain_db)
                        output_path = output_dir / f"{audio_file.stem}_gain{gain_db:.0f}.wav"
                        sf.write(output_path, augmented_audio, sr)
                        augmented_files.append(output_path)
                        total_augmented += 1

                # 噪声注入
                if config.noise_injection:
                    for snr in np.arange(
                        config.noise_snr_range[0],
                        config.noise_snr_range[1],
                        5.0
                    ):
                        augmented_audio = self._add_noise(audio, snr)
                        output_path = output_dir / f"{audio_file.stem}_noise{snr:.0f}.wav"
                        sf.write(output_path, augmented_audio, sr)
                        augmented_files.append(output_path)
                        total_augmented += 1

            except Exception as e:
                logger.error(f"数据增强失败 {audio_file}: {e}")

        logger.info(
            f"数据增强完成，从 {len(audio_files)} 个样本增加到 {len(augmented_files)} 个"
        )

        return augmented_files

    def _resample_audio(self, audio: np.ndarray, sr: int, rate: float) -> np.ndarray:
        """重采样音频（速度扰动）"""
        # 使用时间拉伸改变速度而不改变音调
        return librosa.effects.time_stretch(audio, rate=rate)

    def _apply_gain(self, audio: np.ndarray, gain_db: float) -> np.ndarray:
        """应用增益"""
        gain_linear = 10 ** (gain_db / 20.0)
        return np.clip(audio * gain_linear, -1.0, 1.0)

    def _add_noise(self, audio: np.ndarray, snr_db: float) -> np.ndarray:
        """添加噪声"""
        signal_power = np.mean(audio ** 2)
        noise_power = signal_power / (10 ** (snr_db / 10.0))
        noise = np.random.normal(0, np.sqrt(noise_power), len(audio))
        return audio + noise
