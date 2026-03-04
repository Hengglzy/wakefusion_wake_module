"""
WakeFusion 难负样本（Hard Negatives）生成器
============================================

目的：生成与唤醒词"你好小康"结构/节奏相似、但发音明显不同的语音样本。
      让模型学会：仅当完整听到"你好小康"四个字时才触发，其它相似结构一律拒绝。

原理（参考 Google KWS 论文 + Picovoice Porcupine 设计理念）：
    误唤醒的根源不是随机噪声，而是结构/节奏类似的语句。
    通过大量"结构相似但内容不同"的负样本训练，模型才能学会精确的判别边界。

⚠️ 重要设计原则：
    不收录同音字/近音字（如"小糠""小慷""晓康"），
    因为 MatchboxNet 基于 MFCC 频谱特征，无法区分同音词，
    将它们标为负样本会导致模型连真正的"你好小康"都认不出来。

运行环境：wakefusion 虚拟环境
运行命令：python training/tts_hard_negative_generator.py
"""
import asyncio
import os
import edge_tts
import librosa
import soundfile as sf
import uuid
import numpy as np

# ================= 配置区 =================
OUTPUT_DIR = "custom_dataset/others"
SAMPLE_RATE = 16000
TARGET_DURATION = 2.0         # 统一 2 秒长度
MAX_CONCURRENT_TASKS = 4

# 🎯 难负样本核心：结构/节奏与"你好小康"相似，但发音明显不同的短语
#
# ⚠️ 设计原则（极其重要）：
#   ✅ 收录：发音有明显差异（声母/韵母不同）的相似结构语句
#   ❌ 剔除：同音字/近音字（如"小糠""小慷""晓康""你号"），
#           因为 MatchboxNet 基于频谱特征，无法区分同音/近音词，
#           将它们标为负样本会导致模型连真正的"你好小康"都认不出来。
#
HARD_NEGATIVE_PHRASES = [
    # === 第一层：结构相同"你好小X"，但尾字发音明显不同 ===
    "你好小王",       # kang → wang（声母+韵母都不同）
    "你好小张",       # kang → zhang
    "你好小黄",       # kang → huang
    "你好小方",       # kang → fang
    "你好小杨",       # kang → yang
    "你好小红",       # kang → hong
    "你好小明",       # kang → ming
    "你好小白",       # kang → bai
    "你好小美",       # kang → mei
    "你好小李",       # kang → li
    "你好小陈",       # kang → chen
    "你好小林",       # kang → lin
    "你好小赵",       # kang → zhao
    "你好小周",       # kang → zhou
    "你好小吴",       # kang → wu

    # === 第二层：开头相同"你好"，但后半段明显不同 ===
    "你好老师",       # 小康 → 老师
    "你好大家",       # 小康 → 大家
    "你好同学",       # 小康 → 同学
    "你好朋友",       # 小康 → 朋友
    "你好世界",       # 小康 → 世界
    "你好小朋友",     # 比唤醒词多一个字

    # === 第三层：前两字"你好"的近似（但后续不同）===
    "你好",           # 只有前两个字，太短
    "你好啊",         # 三个字
    "你好吗",         # 三个字
    "你好呀",         # 三个字
    "你早",           # 结构相似但不同
    "你来了",         # "你"开头

    # === 第四层：四五字节奏相似但内容完全不同 ===
    "今天很好啊",
    "明天天气好",
    "快点过来吧",
    "我想吃东西",
    "帮我倒杯水",
    "现在几点了",
    "打开空调吧",
    "关灯睡觉了",
    "我要回家了",
    "吃饭了没有",
    "作业写完了",
    "出去走走吧",

    # === 第五层：日常口语短句（真实环境中最常见的声音）===
    "嗯",
    "哦",
    "好的",
    "是的",
    "行",
    "嗯嗯",
    "好好好",
    "对对对",
    "没问题",
    "知道了",
    "谢谢",
    "再见",
    "来了来了",
    "等一下",
    "马上就好",
]

VOICES = [
    "zh-CN-XiaoxiaoNeural",
    "zh-CN-YunxiNeural",
    "zh-CN-YunjianNeural",
    "zh-CN-XiaoyiNeural",
    "zh-CN-YunxiaNeural",
    "zh-CN-liaoning-XiaobeiNeural",
    "zh-CN-shaanxi-XiaoniNeural",
    "zh-TW-YunJheNeural",
    "zh-TW-HsiaoChenNeural",
    "zh-HK-HiuMaanNeural",
]

STYLES = [
    {"kwargs": {}, "tag": "normal"},
    {"kwargs": {"rate": "-15%"}, "tag": "slow"},
    {"kwargs": {"rate": "+20%"}, "tag": "fast"},
]
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
completed_count = 0
total_tasks = 0


def pad_or_trim(audio, target_len):
    """将音频统一为目标长度"""
    if len(audio) > target_len:
        return audio[:target_len]
    elif len(audio) < target_len:
        pad_len = target_len - len(audio)
        return np.pad(audio, (0, pad_len), mode='constant', constant_values=0)
    return audio


async def generate_single(text, voice, style, wav_path, mp3_path):
    global completed_count
    
    # 跳过已存在的文件
    if os.path.exists(wav_path):
        completed_count += 1
        return
    
    async with semaphore:
        for attempt in range(3):
            try:
                communicate = edge_tts.Communicate(text, voice, **style["kwargs"])
                await communicate.save(mp3_path)
                
                y, sr = librosa.load(mp3_path, sr=SAMPLE_RATE)
                
                # 统一为 2 秒长度
                target_len = int(TARGET_DURATION * SAMPLE_RATE)
                y = pad_or_trim(y, target_len)
                
                sf.write(wav_path, y, SAMPLE_RATE)
                
                completed_count += 1
                print(f"✅ [{completed_count}/{total_tasks}] {text} ({voice.split('-')[-1]}, {style['tag']})", end='\r')
                break
                
            except Exception as e:
                if attempt == 2:
                    print(f"\n❌ 最终失败 [{text}-{voice}]: {e}")
                else:
                    await asyncio.sleep(1.0)
            finally:
                if os.path.exists(mp3_path):
                    try:
                        os.remove(mp3_path)
                    except Exception:
                        pass
        
        await asyncio.sleep(0.2)


async def main():
    global total_tasks
    total_tasks = len(HARD_NEGATIVE_PHRASES) * len(VOICES) * len(STYLES)
    
    print("=" * 60)
    print("🎯 WakeFusion 难负样本（Hard Negatives）生成器")
    print(f"   词汇数: {len(HARD_NEGATIVE_PHRASES)}")
    print(f"   声线数: {len(VOICES)}")
    print(f"   风格数: {len(STYLES)}")
    print(f"   预计生成: {total_tasks} 条难负样本")
    print("=" * 60)
    print()
    print("🔑 这些样本的作用：")
    print("   让模型学会区分 '你好小康' 与 '你好小刚/小王/老康' 等相似语音")
    print("   这是降低误唤醒率的最有效手段（参考 Google KWS 论文）")
    print()
    print("⏳ 正在生成...\n")
    
    tasks = []
    for phrase in HARD_NEGATIVE_PHRASES:
        for voice in VOICES:
            for style in STYLES:
                uid = uuid.uuid4().hex[:8]
                base = f"tts_hardneg_{voice.split('-')[-1]}_{style['tag']}_{uid}"
                wav_path = os.path.join(OUTPUT_DIR, f"{base}.wav")
                mp3_path = os.path.join(OUTPUT_DIR, f"{base}.mp3")
                tasks.append(generate_single(phrase, voice, style, wav_path, mp3_path))

    await asyncio.gather(*tasks)
    
    print(f"\n\n🎉 完成！成功生成 {completed_count} 条难负样本。")
    print(f"\n📋 下一步操作：")
    print(f"   1. python training/split_dataset.py   (重新分割数据集)")
    print(f"   2. python fast_train.py               (重新训练模型)")


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 已安全退出。部分数据已保存。")
