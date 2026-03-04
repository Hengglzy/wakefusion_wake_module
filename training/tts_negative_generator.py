"""
WakeFusion 负样本批量生成器 V3 (反限流+自动重试版)
引入 Semaphore 智能排队机制和参数净化，完美绕过微软服务器限流。
"""
import asyncio
import os
import edge_tts
import librosa
import soundfile as sf
import uuid

# ================= 配置区 =================
OUTPUT_DIR = "custom_dataset/others"
SAMPLE_RATE = 16000
MAX_CONCURRENT_TASKS = 4  # 🌟 核心：最大并发数设为 4，极其稳定

PHRASES = [
    "小康小康", "你好老康", "你好小刚", "你好", "你好健康", "你好小框",
    "你好你好", "小康你好", "早上好", "今天天气怎么样", 
    "帮我打开电视", "是的", "Ok", "Yes", "没问题",
    "你好小红", "你好小明", "呼叫小康", "讲个笑话", "打开空调",
    "取消", "退出", "停止", "关闭", "现在几点了"
]

VOICES = [
    "zh-CN-XiaoxiaoNeural",         # 标准女声
    "zh-CN-YunxiNeural",            # 标准男声
    "zh-CN-YunjianNeural",          # 体育解说男声
    "zh-CN-XiaoyiNeural",           # 🌟 替换：极其稳定的新女声
    "zh-CN-YunxiaNeural",           # 活泼男童声
    "zh-CN-liaoning-XiaobeiNeural", # 东北话女声
    "zh-CN-shaanxi-XiaoniNeural",   # 陕西话女声
    "zh-TW-YunJheNeural",           # 台湾腔男声
    "zh-TW-HsiaoChenNeural",        # 台湾腔女声
    "zh-HK-HiuMaanNeural",          # 粤语女声
]

STYLES = [
    # 🌟 核心修复：正常风格不传 rate 和 pitch 参数
    {"kwargs": {}, "tag": "normal"},
    {"kwargs": {"rate": "-20%", "pitch": "-10Hz"}, "tag": "slow"}, 
    {"kwargs": {"rate": "+30%", "pitch": "+5Hz"}, "tag": "fast"}
]
# ==========================================

os.makedirs(OUTPUT_DIR, exist_ok=True)

# 限制并发数量的"发号器"
semaphore = asyncio.Semaphore(MAX_CONCURRENT_TASKS)
completed_count = 0
total_tasks = 0

async def generate_single_audio(text, voice, style, output_wav_path, temp_mp3_path):
    global completed_count
    
    # 拿到通行证才能执行，拿不到就在这里排队等候
    async with semaphore:
        for attempt in range(3):  # 🌟 核心：失败自动重试最多 3 次
            try:
                # 动态组装参数
                communicate = edge_tts.Communicate(text, voice, **style["kwargs"])
                await communicate.save(temp_mp3_path)
                
                # 转换并保存为 16kHz WAV
                y, sr = librosa.load(temp_mp3_path, sr=SAMPLE_RATE)
                sf.write(output_wav_path, y, SAMPLE_RATE)
                
                completed_count += 1
                print(f"✅ 生产进度: {completed_count} / {total_tasks}", end='\r')
                break  # 成功则跳出重试循环
                
            except Exception as e:
                if attempt == 2:
                    print(f"\n❌ 最终失败 (已放弃) [{text}-{voice}]: {e}")
                else:
                    await asyncio.sleep(1.0)  # 被限流了，休息 1 秒后重新尝试
            finally:
                # 清理临时文件
                if os.path.exists(temp_mp3_path):
                    try:
                        os.remove(temp_mp3_path)
                    except Exception:
                        pass
        
        # 为了极度安全，每个任务结束后稍微喘息 0.2 秒
        await asyncio.sleep(0.2)

async def main():
    global total_tasks
    total_tasks = len(PHRASES) * len(VOICES) * len(STYLES)
    print("=" * 50)
    print("😈 WakeFusion 负样本生成工厂 V3 启动")
    print(f"预计生成: 25词 × 10口音 × 3风格 = {total_tasks} 条数据")
    print("=" * 50)
    print("⏳ 正在稳步向服务器请求，具备自动排队与防封禁机制...\n")
    
    tasks = []
    
    for phrase in PHRASES:
        for voice in VOICES:
            for style in STYLES:
                unique_id = uuid.uuid4().hex[:8]
                base_name = f"tts_neg_{voice.split('-')[-1]}_{style['tag']}_{unique_id}"
                
                filepath_wav = os.path.join(OUTPUT_DIR, f"{base_name}.wav")
                filepath_mp3 = os.path.join(OUTPUT_DIR, f"{base_name}.mp3")
                
                tasks.append(generate_single_audio(phrase, voice, style, filepath_wav, filepath_mp3))

    # 一次性将所有任务放入事件循环（由 semaphore 自动控制并发数量）
    await asyncio.gather(*tasks)
        
    print(f"\n\n🎉 完美收工！成功生成 {completed_count} 条刁钻负样本！")

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n👋 收到强制停止指令，已安全退出。部分数据已保存。")
