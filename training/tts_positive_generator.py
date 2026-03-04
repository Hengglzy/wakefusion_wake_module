"""
WakeFusion TTS 正样本生成器 (Positive Sample Generator)
使用 edge-tts 的 10 种中文声音 × 6 种风格，批量生成"你好小康"正样本。
所有输出统一为 2 秒 / 16kHz / 单声道 WAV，与推理窗口完全一致。

运行环境：wakefusion 虚拟环境
运行命令：python training/tts_positive_generator.py
"""
import asyncio
import os
import edge_tts
import librosa
import soundfile as sf
import numpy as np
import uuid

# ================= 配置区 =================
OUTPUT_DIR = "custom_dataset/xiaokang"
SAMPLE_RATE = 16000
TARGET_DURATION = 2.0  # 🌟 核心：统一为 2 秒，与推理 BUFFER_DURATION 一致
MAX_CONCURRENT_TASKS = 4  # 并发控制，防止被微软限流

# 正样本短语（只有唤醒词本身，但用不同的断句方式增加多样性）
PHRASES = [
    "你好小康",
    "你好，小康",
    "你好小康！",
]

# 10 种中文声音（与负样本生成器保持一致，确保声音多样性）
VOICES = [
    "zh-CN-XiaoxiaoNeural",         # 标准女声
    "zh-CN-YunxiNeural",            # 标准男声
    "zh-CN-YunjianNeural",          # 体育解说男声
    "zh-CN-XiaoyiNeural",           # 活泼女声
    "zh-CN-YunxiaNeural",           # 活泼男童声
    "zh-CN-liaoning-XiaobeiNeural", # 东北话女声
    "zh-CN-shaanxi-XiaoniNeural",   # 陕西话女声
    "zh-TW-YunJheNeural",           # 台湾腔男声
    "zh-TW-HsiaoChenNeural",        # 台湾腔女声
    "zh-HK-HiuMaanNeural",          # 粤语女声
]

# 6 种语速/音调风格，增加发音多样性
STYLES = [
    {"kwargs": {}, "tag": "normal"},                                       # 正常
    {"kwargs": {"rate": "-20%", "pitch": "-10Hz"}, "tag": "slow"},         # 慢速低沉
    {"kwargs": {"rate": "+30%", "pitch": "+5Hz"}, "tag": "fast"},          # 快速明亮
    {"kwargs": {"rate": "-10%", "pitch": "-20Hz"}, "tag": "deep"},         # 深沉男性化
    {"kwargs": {"rate": "+15%", "pitch": "+15Hz"}, "tag": "bright"},       # 高亢清脆
    {"kwargs": {"rate": "-30%"}, "tag": "veryslow"},                       # 极慢（模拟老人/犹豫）
]
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 并发控制
semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
completed_count = 0
total_tasks = 0


def pad_or_trim_to_duration(audio, sr, target_sec):
    """
    将音频精确调整为目标时长。
    过短 → 尾部补零（静音填充）
    过长 → 截断
    """
    target_len = int(target_sec * sr)
    if len(audio) >= target_len:
        return audio[:target_len]
    else:
        return np.pad(audio, (0, target_len - len(audio)), mode='constant')


async def generate_single_positive(text, voice, style, output_wav_path, temp_mp3_path):
    global completed_count

    async with semaphore:
        for attempt in range(3):
            try:
                communicate = edge_tts.Communicate(text, voice, **style["kwargs"])
                await communicate.save(temp_mp3_path)

                # 加载并转换为 16kHz
                y, sr = librosa.load(temp_mp3_path, sr=SAMPLE_RATE)

                # 🌟 核心：精确裁剪/填充到 2 秒
                y = pad_or_trim_to_duration(y, SAMPLE_RATE, TARGET_DURATION)

                sf.write(output_wav_path, y, SAMPLE_RATE)

                completed_count += 1
                print(f"✅ 正样本生产进度: {completed_count} / {total_tasks}", end='\r')
                break

            except Exception as e:
                if attempt == 2:
                    print(f"\n❌ 最终失败 [{text}-{voice}]: {e}")
                else:
                    await asyncio.sleep(1.0)
            finally:
                if os.path.exists(temp_mp3_path):
                    try:
                        os.remove(temp_mp3_path)
                    except Exception:
                        pass

        await asyncio.sleep(0.2)


async def main():
    global total_tasks
    total_tasks = len(PHRASES) * len(VOICES) * len(STYLES)
    print("=" * 55)
    print("🎯 WakeFusion TTS 正样本生成工厂启动")
    print(f"预计生成: {len(PHRASES)}短语 × {len(VOICES)}声音 × {len(STYLES)}风格 = {total_tasks} 条正样本")
    print(f"所有音频统一为 {TARGET_DURATION} 秒 / {SAMPLE_RATE}Hz")
    print("=" * 55)
    print("⏳ 正在稳步生成，具备自动排队与防封禁机制...\n")

    tasks = []

    for phrase in PHRASES:
        for voice in VOICES:
            for style in STYLES:
                unique_id = uuid.uuid4().hex[:8]
                # 使用 tts_pos_ 前缀，方便 positive_data_factory 识别
                base_name = f"tts_pos_{voice.split('-')[-1]}_{style['tag']}_{unique_id}"

                filepath_wav = os.path.join(OUTPUT_DIR, f"{base_name}.wav")
                filepath_mp3 = os.path.join(OUTPUT_DIR, f"{base_name}.mp3")

                tasks.append(generate_single_positive(phrase, voice, style, filepath_wav, filepath_mp3))

    await asyncio.gather(*tasks)

    print(f"\n\n🎉 完美收工！成功生成 {completed_count} 条多声音正样本！")
    print(f"📁 输出目录: {os.path.abspath(OUTPUT_DIR)}")
    print("💡 下一步：运行 positive_data_factory.py 对真人录音做变调变速，对TTS做加噪")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 收到强制停止指令，已安全退出。部分数据已保存。")
