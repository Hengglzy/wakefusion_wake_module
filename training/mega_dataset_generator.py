"""
WakeFusion 超级数据集生成器 (Mega Dataset Generator)
=====================================================
一键生成 20,000 条正样本 + 50,000 条负样本

引擎架构：
  阶段一：edge-tts 14 种中文声音 × 3 种语速 → 生成高多样性 TTS 基底
  阶段二：离线增强流水线 → 对每条基底施加随机增强组合，批量扩增至目标数量
  阶段三：纯噪声负样本 → 环境底噪、空调嗡声、静音等无语音片段

增强手段（6 维正交增强空间）：
  1. 声学特征变换 — 连续随机音高偏移（覆盖男/女/童全音域）
  2. 语速变化     — 连续随机（极速连读 ~ 树懒慢读）
  3. 时间偏移     — 唤醒词在 2 秒窗口中的起始位置随机
  4. 噪声注入     — 白噪声 / 粉红噪声 / 人声嘈杂 / 脉冲噪声
  5. 合成 RIR     — 模拟小房间 / 中型房间 / 大厅 / 展厅混响
  6. 远场模拟     — 低音量 + 轻微噪声，模拟说话者距离远

运行环境：wakefusion 虚拟环境
运行命令：python training/mega_dataset_generator.py
预计时间：TTS 基底 ~25 分钟 + 增强 ~30-60 分钟 = 约 1-1.5 小时
磁盘空间：约 5GB
断点续传：支持，已生成的文件会自动跳过

前置条件：pip install edge-tts librosa soundfile scipy numpy
"""
import asyncio
import os
import sys
import random
import time
import numpy as np
import librosa
import soundfile as sf
import scipy.signal
import edge_tts

# ====================== Windows 异步兼容 ======================
if sys.platform == 'win32':
    asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())

# ======================== 配置区 ========================
# 输出目录（与现有训练流水线一致）
POS_OUTPUT_DIR = "custom_dataset/xiaokang"
NEG_OUTPUT_DIR = "custom_dataset/others"
TEMP_DIR = "_temp_tts_base"   # TTS 基底临时目录

SAMPLE_RATE = 16000
TARGET_DURATION = 2.0
TARGET_SAMPLES = int(SAMPLE_RATE * TARGET_DURATION)

# 目标数量
TARGET_POS_COUNT = 20000       # 正样本总数
TARGET_NEG_TTS_COUNT = 45000   # TTS 负样本数
TARGET_NOISE_COUNT = 5000      # 纯噪声负样本数
TARGET_NEG_COUNT = TARGET_NEG_TTS_COUNT + TARGET_NOISE_COUNT  # 50000

# Edge-TTS 并发控制
MAX_CONCURRENT = 4

# ======================== 声音列表 ========================
# 14 种中文声音（自动发现验证过的完整列表）
# 覆盖：标准普通话男/女、方言、童声、播音腔、粤语、台湾腔
VOICES = [
    # --- 大陆普通话 (6 种) ---
    "zh-CN-XiaoxiaoNeural",          # 标准女声（温柔型）
    "zh-CN-XiaoyiNeural",            # 活泼女声
    "zh-CN-YunjianNeural",           # 体育解说男声（浑厚）
    "zh-CN-YunxiNeural",             # 标准男声
    "zh-CN-YunxiaNeural",            # 活泼男童声
    "zh-CN-YunyangNeural",           # 新闻播音男声（字正腔圆）
    # --- 方言 (2 种) ---
    "zh-CN-liaoning-XiaobeiNeural",  # 东北话女声
    "zh-CN-shaanxi-XiaoniNeural",    # 陕西话女声
    # --- 台湾腔 (3 种) ---
    "zh-TW-HsiaoChenNeural",         # 台湾腔女声
    "zh-TW-YunJheNeural",            # 台湾腔男声
    "zh-TW-HsiaoYuNeural",           # 台湾腔女声2
    # --- 粤语 (3 种) ---
    "zh-HK-HiuGaaiNeural",           # 粤语女声1
    "zh-HK-HiuMaanNeural",           # 粤语女声2
    "zh-HK-WanLungNeural",           # 粤语男声
]

# Edge-TTS 基底风格（离散档位，连续变化由增强流水线完成）
TTS_STYLES = [
    {"kwargs": {},              "tag": "normal"},
    {"kwargs": {"rate": "-20%"}, "tag": "slow"},
    {"kwargs": {"rate": "+25%"}, "tag": "fast"},
]

# ======================== 正样本短语 ========================
# 比例 2:1:1 → "你好小康"占 50%，其余各 25%
POS_PHRASES = [
    {"text": "你好小康",   "weight": 2},
    {"text": "你好小康！",  "weight": 1},
    {"text": "你好，小康",  "weight": 1},
]

# ======================== 负样本短语 ========================
# 高危词（每个占 TTS 负样本的 8%，合计 24%）
NEG_PHRASES_HIGH_RISK = ["你好你好", "小康小康", "你好"]
NEG_HIGH_RISK_WEIGHT = 0.08   # 每个高危短语占 TTS 负样本总数的比例

# 标准负样本短语（合并 tts_negative + tts_hard_negative，已去重）
NEG_PHRASES_STANDARD = [
    # "你好小X" 系列 — 结构相同尾字不同
    "你好小王", "你好小张", "你好小黄", "你好小方", "你好小杨",
    "你好小红", "你好小明", "你好小白", "你好小美", "你好小李",
    "你好小陈", "你好小林", "你好小赵", "你好小周", "你好小吴",
    # "你好+其他" 系列
    "你好老师", "你好大家", "你好同学", "你好朋友", "你好世界",
    "你好小朋友", "你好老康", "你好小刚", "你好健康", "你好小框",
    # "你好" 近似
    "你好啊", "你好吗", "你好呀", "你早", "你来了",
    # 四五字节奏相似
    "小康你好", "今天很好啊", "明天天气好", "快点过来吧", "我想吃东西",
    "帮我倒杯水", "打开空调吧", "关灯睡觉了", "我要回家了",
    "吃饭了没有", "作业写完了", "出去走走吧",
    # 日常口语
    "嗯", "哦", "好的", "是的", "行", "嗯嗯", "好好好",
    "对对对", "没问题", "知道了", "谢谢", "再见",
    "来了来了", "等一下", "马上就好",
    # 日常指令
    "早上好", "今天天气怎么样", "帮我打开电视",
    "呼叫小康", "讲个笑话", "打开空调",
    "取消", "退出", "停止", "关闭", "现在几点了",
    # 补充多样性
    "请问一下", "不好意思", "对不起", "没有关系",
    "怎么回事", "你说什么", "我听不到", "声音大一点",
    "一二三四五", "今天星期几", "Ok", "Yes",
]

# ======================== 增强参数 ========================
# 时间偏移范围（唤醒词在 2 秒窗口中的起始位置）
MAX_TIME_OFFSET = 0.8   # 秒

# 声学特征变换分布（通过 resampling 同时改变音高和语速）
# factor > 1 → 高音 + 快速 (女声/童声效果)
# factor < 1 → 低音 + 慢速 (男中音/低沉效果)
VOCAL_NORMAL_PROB = 0.70     # 70% 正常微调
VOCAL_NORMAL_RANGE = (0.87, 1.15)
VOCAL_DEEP_PROB = 0.15       # 15% 深沉（模拟大型男声）
VOCAL_DEEP_RANGE = (0.55, 0.87)
# 剩余 15% 高亢（模拟女高音/儿童）
VOCAL_HIGH_RANGE = (1.15, 1.70)

# 噪声/环境类型及分布
NOISE_TYPE_PROBS = {
    "clean":     0.10,   # 干净无噪声
    "rir_only":  0.10,   # 仅混响
    "white":     0.18,   # 白噪声
    "pink":      0.14,   # 粉红噪声（空调/风扇）
    "babble":    0.14,   # 合成人声嘈杂
    "impulse":   0.08,   # 脉冲噪声（碰撞/敲击）
    "far_field": 0.18,   # 远场模拟
    "hum":       0.08,   # 电力线嗡声
}
SNR_RANGE = (5, 30)                  # 信噪比范围 (dB)
FAR_FIELD_GAIN_RANGE = (0.12, 0.40)  # 远场音量缩放

# RIR 附加概率（对非 rir_only/clean 类型额外叠加 RIR 的概率）
RIR_EXTRA_PROB = 0.15


# ================================================================
#                        工具函数
# ================================================================

def pad_or_trim(audio, target_len):
    """将音频精确调整到目标长度（尾部补零或截断）"""
    if len(audio) >= target_len:
        return audio[:target_len]
    return np.pad(audio, (0, target_len - len(audio)), mode='constant').astype(np.float32)


def normalize_audio(audio, peak=0.90):
    """峰值归一化，防止削波"""
    max_val = np.max(np.abs(audio))
    if max_val > 1e-6:
        return (audio * (peak / max_val)).astype(np.float32)
    return audio.astype(np.float32)


def trim_silence(audio, threshold=0.005, margin=160):
    """去除头部静音，保留少量前置余量"""
    for i in range(len(audio)):
        if abs(audio[i]) > threshold:
            start = max(0, i - margin)
            return audio[start:]
    return audio


def voice_tag(voice_name):
    """提取声音短标签用于文件名：zh-CN-XiaoxiaoNeural → Xiaoxiao"""
    return voice_name.split("-")[-1].replace("Neural", "")


# ================================================================
#                     噪声与 RIR 生成器
# ================================================================

def generate_white_noise(length):
    """高斯白噪声"""
    return np.random.randn(length).astype(np.float32)


def generate_pink_noise(length):
    """粉红噪声 (1/f)，模拟空调/通风系统"""
    white = np.random.randn(length + 10)
    # Voss-McCartney 近似：用 IIR 滤波器将白噪声变为粉红噪声
    b = np.array([0.049922035, -0.095993537, 0.050612699, -0.004709510], dtype=np.float64)
    a = np.array([1.0, -2.494956002, 2.017265875, -0.522189400], dtype=np.float64)
    pink = scipy.signal.lfilter(b, a, white)
    return pink[:length].astype(np.float32)


def generate_babble_noise(length, sr):
    """合成人声嘈杂噪声（多人低语叠加）"""
    babble = np.zeros(length, dtype=np.float64)
    n_sources = random.randint(3, 7)
    for _ in range(n_sources):
        noise = np.random.randn(length)
        # 用随机频率的正弦调幅模拟语音包络
        mod_freq = random.uniform(2, 6)  # 人类语音节奏 2-6 Hz
        t = np.arange(length) / sr
        phase = random.uniform(0, 2 * np.pi)
        envelope = 0.5 + 0.5 * np.sin(2 * np.pi * mod_freq * t + phase)
        babble += noise * envelope
    return (babble / n_sources).astype(np.float32)


def generate_impulse_noise(length):
    """脉冲噪声（碰撞、敲击、按键声）"""
    noise = np.zeros(length, dtype=np.float32)
    n_impulses = random.randint(2, 8)
    for _ in range(n_impulses):
        pos = random.randint(0, length - 1)
        amp = random.uniform(0.3, 1.0)
        burst_len = random.randint(10, 160)
        end = min(pos + burst_len, length)
        noise[pos:end] = np.random.randn(end - pos).astype(np.float32) * amp
    return noise


def generate_hum_noise(length, sr):
    """电力线嗡声（50/60Hz 及谐波）"""
    t = np.arange(length, dtype=np.float64) / sr
    freq = random.choice([50, 60, 100, 120, 150])
    hum = np.sin(2 * np.pi * freq * t) * random.uniform(0.01, 0.08)
    # 叠加轻微谐波
    hum += np.sin(2 * np.pi * freq * 2 * t) * random.uniform(0.005, 0.03)
    hum += np.random.randn(length) * 0.003  # 极轻底噪
    return hum.astype(np.float32)


def generate_synthetic_rir(sr):
    """
    合成房间冲激响应 (RIR)
    模拟 4 种房间类型：小房间、中型房间、大房间、展厅
    """
    room_type = random.choice(['small', 'medium', 'large', 'hall'])
    configs = {
        'small':  {'rt60': (0.10, 0.30), 'length_sec': 0.3},
        'medium': {'rt60': (0.30, 0.60), 'length_sec': 0.6},
        'large':  {'rt60': (0.60, 1.00), 'length_sec': 1.0},
        'hall':   {'rt60': (1.00, 2.00), 'length_sec': 2.0},  # 展厅级混响
    }
    cfg = configs[room_type]
    rt60 = random.uniform(*cfg['rt60'])
    length = int(cfg['length_sec'] * sr)

    t = np.arange(length, dtype=np.float64) / sr
    decay = np.exp(-6.908 * t / max(rt60, 0.01))  # -60dB 衰减
    rir = np.random.randn(length) * decay
    rir[0] = 1.0  # 直达声
    max_val = np.max(np.abs(rir))
    if max_val > 0:
        rir = rir / max_val
    return rir.astype(np.float32)


def apply_rir(audio, rir):
    """将音频与 RIR 卷积，模拟房间混响"""
    augmented = scipy.signal.fftconvolve(audio.astype(np.float64),
                                         rir.astype(np.float64), mode='full')
    augmented = augmented[:len(audio)]
    max_val = np.max(np.abs(augmented))
    if max_val > 1e-6:
        augmented = augmented / max_val
    return augmented.astype(np.float32)


def add_noise(audio, noise, snr_db):
    """以指定信噪比 (SNR) 叠加噪声"""
    sig_power = np.mean(audio.astype(np.float64) ** 2)
    noise_power = np.mean(noise.astype(np.float64) ** 2)
    if noise_power < 1e-10 or sig_power < 1e-10:
        return audio
    scale = np.sqrt(sig_power / (noise_power * 10 ** (snr_db / 10)))
    mixed = audio.astype(np.float64) + noise.astype(np.float64) * scale
    return mixed.astype(np.float32)


# ================================================================
#                       增强流水线
# ================================================================

def sample_vocal_factor():
    """
    采样声学特征变换系数：
      factor > 1 → 音高上升 + 语速加快（高亢/童声）
      factor < 1 → 音高下降 + 语速减慢（低沉/男中音）
    """
    r = random.random()
    if r < VOCAL_NORMAL_PROB:
        return random.uniform(*VOCAL_NORMAL_RANGE)
    elif r < VOCAL_NORMAL_PROB + VOCAL_DEEP_PROB:
        return random.uniform(*VOCAL_DEEP_RANGE)
    else:
        return random.uniform(*VOCAL_HIGH_RANGE)


def sample_noise_type():
    """按权重采样噪声/环境类型"""
    types = list(NOISE_TYPE_PROBS.keys())
    weights = list(NOISE_TYPE_PROBS.values())
    return random.choices(types, weights=weights, k=1)[0]


def augment_audio(audio, sr, is_positive=True):
    """
    对一条基底音频施加完整增强流水线。
    
    流水线顺序（物理意义正确）：
      1. 声学变换（resampling → 改变音高+语速）
      2. 去除头部静音
      3. 随机时间偏移（唤醒词在窗口中的位置）
      4. 裁剪/填充到 2 秒
      5. RIR 混响（声音在房间中反射）
      6. 环境噪声（叠加在混响后的信号上）
      7. 峰值归一化
    """
    audio = audio.copy().astype(np.float32)

    # ---- 1. 声学特征变换 ----
    factor = sample_vocal_factor()
    if abs(factor - 1.0) > 0.03:
        new_len = max(800, int(len(audio) / factor))
        audio = scipy.signal.resample(audio, new_len).astype(np.float32)

    # ---- 2. 去除头部静音 ----
    audio = trim_silence(audio)

    # ---- 3 & 4. 时间偏移 + 裁剪/填充 ----
    speech_len = len(audio)
    if speech_len >= TARGET_SAMPLES:
        # 语音已超过 2 秒（极慢语速情况）
        if is_positive:
            # 正样本：居中截取，确保唤醒词完整
            excess = speech_len - TARGET_SAMPLES
            start = random.randint(0, max(0, excess))
            audio = audio[start:start + TARGET_SAMPLES]
        else:
            # 负样本：随机截取
            start = random.randint(0, speech_len - TARGET_SAMPLES)
            audio = audio[start:start + TARGET_SAMPLES]
    else:
        # 语音短于 2 秒：插入随机静音偏移
        available_space = TARGET_SAMPLES - speech_len
        max_offset = min(int(MAX_TIME_OFFSET * sr), available_space)
        offset_samples = random.randint(0, max(0, max_offset))
        audio = np.concatenate([
            np.zeros(offset_samples, dtype=np.float32),
            audio
        ])
        audio = pad_or_trim(audio, TARGET_SAMPLES)

    # ---- 5 & 6. 噪声/环境类型 ----
    noise_type = sample_noise_type()

    # 先处理 RIR
    if noise_type == "rir_only":
        rir = generate_synthetic_rir(sr)
        audio = apply_rir(audio, rir)
    elif noise_type not in ("clean",):
        # 对非 clean/rir_only 类型，有概率额外叠加 RIR
        if random.random() < RIR_EXTRA_PROB:
            rir = generate_synthetic_rir(sr)
            audio = apply_rir(audio, rir)

    # 叠加噪声
    if noise_type == "white":
        snr = random.uniform(*SNR_RANGE)
        noise = generate_white_noise(TARGET_SAMPLES)
        audio = add_noise(audio, noise, snr)

    elif noise_type == "pink":
        snr = random.uniform(*SNR_RANGE)
        noise = generate_pink_noise(TARGET_SAMPLES)
        audio = add_noise(audio, noise, snr)

    elif noise_type == "babble":
        snr = random.uniform(5, 20)  # 人声嘈杂 SNR 通常更低
        noise = generate_babble_noise(TARGET_SAMPLES, sr)
        audio = add_noise(audio, noise, snr)

    elif noise_type == "impulse":
        snr = random.uniform(10, 25)
        noise = generate_impulse_noise(TARGET_SAMPLES)
        audio = add_noise(audio, noise, snr)

    elif noise_type == "far_field":
        # 远场 = 降低音量 + 粉红噪声
        gain = random.uniform(*FAR_FIELD_GAIN_RANGE)
        audio = audio * gain
        snr = random.uniform(10, 20)
        noise = generate_pink_noise(TARGET_SAMPLES)
        audio = add_noise(audio, noise, snr)

    elif noise_type == "hum":
        snr = random.uniform(15, 30)
        noise = generate_hum_noise(TARGET_SAMPLES, sr)
        audio = add_noise(audio, noise, snr)

    # clean 和 rir_only 不叠加噪声

    # ---- 7. 峰值归一化 ----
    audio = normalize_audio(audio, peak=random.uniform(0.60, 0.95))

    return audio


# ================================================================
#                     Edge-TTS 基底生成
# ================================================================

# 全局并发控制
_semaphore = None
_tts_completed = 0
_tts_total = 0


async def generate_single_tts(text, voice, style, wav_path, mp3_path):
    """生成单条 TTS 基底音频"""
    global _tts_completed

    if os.path.exists(wav_path):
        _tts_completed += 1
        return True

    async with _semaphore:
        for attempt in range(3):
            try:
                communicate = edge_tts.Communicate(text, voice, **style["kwargs"])
                await communicate.save(mp3_path)

                y, _ = librosa.load(mp3_path, sr=SAMPLE_RATE)
                y = pad_or_trim(y, TARGET_SAMPLES)
                sf.write(wav_path, y, SAMPLE_RATE)

                _tts_completed += 1
                pct = _tts_completed / _tts_total * 100
                print(f"\r  TTS [{_tts_completed}/{_tts_total}] {pct:.0f}%", end="", flush=True)
                return True

            except Exception as e:
                if attempt == 2:
                    print(f"\n  [WARN] TTS fail: {voice} '{text}': {e}")
                    _tts_completed += 1
                    return False
                await asyncio.sleep(1.0)
            finally:
                if os.path.exists(mp3_path):
                    try:
                        os.remove(mp3_path)
                    except Exception:
                        pass

        await asyncio.sleep(0.15)
        return False


async def generate_tts_bases(phrases, output_subdir, prefix):
    """
    批量生成 TTS 基底音频。
    返回 dict: {phrase_idx: [wav_path, ...]}
    """
    global _semaphore, _tts_completed, _tts_total

    _semaphore = asyncio.Semaphore(MAX_CONCURRENT)

    base_dir = os.path.join(TEMP_DIR, output_subdir)
    os.makedirs(base_dir, exist_ok=True)

    # 构建任务列表
    tasks = []
    expected_files = {}

    phrase_list = phrases if isinstance(phrases, list) else phrases
    for p_idx, phrase_text in enumerate(phrase_list):
        expected_files[p_idx] = []
        for voice in VOICES:
            v_tag = voice_tag(voice)
            for style in TTS_STYLES:
                filename = f"{prefix}{p_idx:02d}_{v_tag}_{style['tag']}.wav"
                wav_path = os.path.join(base_dir, filename)
                mp3_path = wav_path.replace(".wav", ".mp3")

                expected_files[p_idx].append(wav_path)
                tasks.append(generate_single_tts(
                    phrase_text, voice, style, wav_path, mp3_path
                ))

    _tts_total += len(tasks)

    if tasks:
        await asyncio.gather(*tasks)

    # 过滤出实际存在的文件
    result = {}
    for p_idx, paths in expected_files.items():
        result[p_idx] = [p for p in paths if os.path.exists(p)]

    return result


# ================================================================
#                      批量增强与保存
# ================================================================

def augment_batch(base_files_dict, phrase_targets, output_dir, prefix, is_positive):
    """
    对基底音频进行批量增强并保存。
    
    Args:
        base_files_dict: {phrase_idx: [wav_path, ...]}
        phrase_targets:   {phrase_idx: target_count}
        output_dir:       输出目录
        prefix:           输出文件前缀
        is_positive:      是否为正样本
    """
    os.makedirs(output_dir, exist_ok=True)

    total_target = sum(phrase_targets.values())
    global_idx = 0
    generated = 0
    skipped = 0
    start_time = time.time()

    # 检查已有文件数量（断点续传）
    existing = set(f for f in os.listdir(output_dir) if f.startswith(prefix))

    print(f"  目标: {total_target} 条 | 已有: {len(existing)} 条")

    for p_idx in sorted(phrase_targets.keys()):
        target = phrase_targets[p_idx]
        bases = base_files_dict.get(p_idx, [])

        if not bases:
            print(f"  [WARN] 短语 {p_idx} 无可用基底，跳过")
            global_idx += target
            continue

        # 预加载此短语的所有基底音频
        base_audios = []
        for bp in bases:
            try:
                y, _ = librosa.load(bp, sr=SAMPLE_RATE)
                base_audios.append(y)
            except Exception:
                pass

        if not base_audios:
            global_idx += target
            continue

        for i in range(target):
            filename = f"{prefix}{global_idx:06d}.wav"
            output_path = os.path.join(output_dir, filename)
            global_idx += 1

            if filename in existing:
                skipped += 1
                continue

            # 随机选择一条基底
            base = random.choice(base_audios)

            try:
                augmented = augment_audio(base, SAMPLE_RATE, is_positive=is_positive)
                sf.write(output_path, augmented, SAMPLE_RATE)
                generated += 1
            except Exception as e:
                # 增强失败，重试一次
                try:
                    augmented = augment_audio(base, SAMPLE_RATE, is_positive=is_positive)
                    sf.write(output_path, augmented, SAMPLE_RATE)
                    generated += 1
                except Exception:
                    pass

            total_done = generated + skipped
            if total_done % 500 == 0 and total_done > 0:
                elapsed = time.time() - start_time
                rate = total_done / elapsed if elapsed > 0 else 0
                eta = (total_target - total_done) / rate if rate > 0 else 0
                print(f"\r  [{total_done}/{total_target}] "
                      f"{rate:.0f} files/s | ETA: {eta/60:.1f} min", end="", flush=True)

    elapsed = time.time() - start_time
    print(f"\n  Done: {generated} new + {skipped} exist = {generated + skipped} | "
          f"Time: {elapsed/60:.1f} min")
    return generated


def generate_pure_noise(output_dir, count, start_idx):
    """生成纯噪声负样本（无语音内容）"""
    os.makedirs(output_dir, exist_ok=True)
    generated = 0

    print(f"  目标: {count} 条纯噪声")

    for i in range(count):
        idx = start_idx + i
        filename = f"mega_noise_{idx:06d}.wav"
        output_path = os.path.join(output_dir, filename)

        if os.path.exists(output_path):
            generated += 1
            continue

        noise_type = random.choice(["white", "pink", "babble", "hum", "silence"])

        if noise_type == "white":
            audio = generate_white_noise(TARGET_SAMPLES) * random.uniform(0.02, 0.25)
        elif noise_type == "pink":
            audio = generate_pink_noise(TARGET_SAMPLES) * random.uniform(0.02, 0.25)
        elif noise_type == "babble":
            audio = generate_babble_noise(TARGET_SAMPLES, SAMPLE_RATE)
            audio = audio * random.uniform(0.02, 0.20)
        elif noise_type == "hum":
            audio = generate_hum_noise(TARGET_SAMPLES, SAMPLE_RATE)
        else:  # silence
            audio = np.random.randn(TARGET_SAMPLES).astype(np.float32) * 0.001

        # 部分叠加 RIR
        if random.random() < 0.2:
            rir = generate_synthetic_rir(SAMPLE_RATE)
            audio = apply_rir(audio, rir)

        audio = normalize_audio(audio, peak=random.uniform(0.05, 0.40))
        sf.write(output_path, audio, SAMPLE_RATE)
        generated += 1

        if (i + 1) % 500 == 0:
            print(f"\r  [{i + 1}/{count}]", end="", flush=True)

    print(f"\n  Done: {generated} noise samples")
    return generated


# ================================================================
#                          主流程
# ================================================================

async def main():
    global _tts_completed, _tts_total
    _tts_completed = 0
    _tts_total = 0

    print("=" * 70)
    print("  WakeFusion Mega Dataset Generator")
    print("  ==================================")
    print(f"  Target: {TARGET_POS_COUNT:,} positive + {TARGET_NEG_COUNT:,} negative = "
          f"{TARGET_POS_COUNT + TARGET_NEG_COUNT:,} total")
    print(f"  Voices: {len(VOICES)} | Styles: {len(TTS_STYLES)}")
    print(f"  Sample: {TARGET_DURATION}s / {SAMPLE_RATE}Hz / mono")
    print("=" * 70)

    os.makedirs(POS_OUTPUT_DIR, exist_ok=True)
    os.makedirs(NEG_OUTPUT_DIR, exist_ok=True)
    os.makedirs(TEMP_DIR, exist_ok=True)

    total_start = time.time()

    # ============================================================
    # Phase 1: TTS
    # ============================================================
    print("\n" + "=" * 50)
    print("Phase 1/4: Edge-TTS Base Generation")
    print("=" * 50)

    # --- 正样本 TTS 基底 ---
    pos_texts = [p["text"] for p in POS_PHRASES]
    n_pos_bases = len(pos_texts) * len(VOICES) * len(TTS_STYLES)
    print(f"\n  [POS] {len(pos_texts)} phrases x {len(VOICES)} voices x "
          f"{len(TTS_STYLES)} styles = {n_pos_bases} bases")
    pos_bases = await generate_tts_bases(pos_texts, "pos", "p")

    # --- 负样本 TTS 基底 ---
    all_neg_phrases = NEG_PHRASES_HIGH_RISK + NEG_PHRASES_STANDARD
    n_neg_bases = len(all_neg_phrases) * len(VOICES) * len(TTS_STYLES)
    print(f"\n\n  [NEG] {len(all_neg_phrases)} phrases x {len(VOICES)} voices x "
          f"{len(TTS_STYLES)} styles = {n_neg_bases} bases")
    neg_bases = await generate_tts_bases(all_neg_phrases, "neg", "n")

    tts_time = time.time() - total_start
    print(f"\n\n  Phase 1 done in {tts_time/60:.1f} min")

    # ============================================================
    # Phase 2: Positive Augmentation
    # ============================================================
    print("\n" + "=" * 50)
    print("Phase 2/4: Positive Sample Augmentation")
    print("=" * 50)

    total_pos_weight = sum(p["weight"] for p in POS_PHRASES)
    pos_targets = {}
    remaining = TARGET_POS_COUNT
    for i, p in enumerate(POS_PHRASES):
        if i == len(POS_PHRASES) - 1:
            pos_targets[i] = remaining
        else:
            count = int(TARGET_POS_COUNT * p["weight"] / total_pos_weight)
            pos_targets[i] = count
            remaining -= count

    for i, p in enumerate(POS_PHRASES):
        n_bases = len(pos_bases.get(i, []))
        aug_per = pos_targets[i] / max(1, n_bases)
        print(f"  '{p['text']}': {pos_targets[i]:,} target "
              f"({n_bases} bases x ~{aug_per:.0f} aug)")

    augment_batch(pos_bases, pos_targets, POS_OUTPUT_DIR, "mega_pos_", is_positive=True)

    # ============================================================
    # Phase 3: Negative TTS Augmentation
    # ============================================================
    print("\n" + "=" * 50)
    print("Phase 3/4: Negative Sample Augmentation")
    print("=" * 50)

    # 计算每个负样本短语的目标数量
    neg_targets = {}
    n_high = len(NEG_PHRASES_HIGH_RISK)
    n_std = len(NEG_PHRASES_STANDARD)

    # 高危词各占 8%
    high_risk_total = 0
    for i in range(n_high):
        count = int(TARGET_NEG_TTS_COUNT * NEG_HIGH_RISK_WEIGHT)
        neg_targets[i] = count
        high_risk_total += count

    # 剩余平均分配给标准词
    remaining_std = TARGET_NEG_TTS_COUNT - high_risk_total
    per_std = remaining_std // n_std
    extra = remaining_std % n_std

    for i in range(n_std):
        idx = n_high + i
        neg_targets[idx] = per_std + (1 if i < extra else 0)

    print(f"  High-risk ({n_high} phrases): "
          f"{high_risk_total:,} ({high_risk_total/TARGET_NEG_TTS_COUNT*100:.0f}%)")
    print(f"  Standard  ({n_std} phrases): "
          f"{remaining_std:,} ({remaining_std/TARGET_NEG_TTS_COUNT*100:.0f}%)")

    augment_batch(neg_bases, neg_targets, NEG_OUTPUT_DIR, "mega_neg_", is_positive=False)

    # ============================================================
    # Phase 4: Pure Noise
    # ============================================================
    print("\n" + "=" * 50)
    print("Phase 4/4: Pure Noise Negatives")
    print("=" * 50)

    generate_pure_noise(NEG_OUTPUT_DIR, TARGET_NOISE_COUNT, start_idx=0)

    # ============================================================
    # Statistics
    # ============================================================
    total_time = time.time() - total_start

    # 统计实际文件数
    pos_mega = len([f for f in os.listdir(POS_OUTPUT_DIR) if f.startswith("mega_pos_")])
    neg_mega = len([f for f in os.listdir(NEG_OUTPUT_DIR) if f.startswith("mega_neg_")])
    neg_noise = len([f for f in os.listdir(NEG_OUTPUT_DIR) if f.startswith("mega_noise_")])
    pos_other = len([f for f in os.listdir(POS_OUTPUT_DIR)
                     if f.endswith(".wav") and not f.startswith("mega_")])
    neg_other = len([f for f in os.listdir(NEG_OUTPUT_DIR)
                     if f.endswith(".wav") and not f.startswith("mega_")])

    print("\n" + "=" * 70)
    print("  Generation Complete!")
    print("=" * 70)
    print(f"\n  === Mega Dataset (new) ===")
    print(f"  Positive (mega_pos_*):  {pos_mega:>7,}")
    print(f"  Negative (mega_neg_*):  {neg_mega:>7,}")
    print(f"  Noise    (mega_noise_*): {neg_noise:>6,}")
    print(f"  {'':>25}--------")
    print(f"  {'Mega total:':>25} {pos_mega + neg_mega + neg_noise:>7,}")
    print(f"\n  === Existing Dataset ===")
    print(f"  Positive (other):       {pos_other:>7,}")
    print(f"  Negative (other):       {neg_other:>7,}")
    print(f"\n  === Grand Total ===")
    print(f"  All positive:           {pos_mega + pos_other:>7,}")
    print(f"  All negative:           {neg_mega + neg_noise + neg_other:>7,}")
    print(f"  Ratio (neg:pos):        1:{(neg_mega + neg_noise + neg_other) / max(1, pos_mega + pos_other):.1f}")
    print(f"\n  Total time: {total_time/60:.1f} min")
    print(f"  Temp dir: {os.path.abspath(TEMP_DIR)} (can be deleted after verification)")

    # 估算磁盘占用
    size_mb = (pos_mega + neg_mega + neg_noise) * TARGET_SAMPLES * 2 / 1024 / 1024
    print(f"  Est. disk: {size_mb/1024:.1f} GB")

    print(f"\n  Next steps:")
    print(f"    1. python training/oww_prepare_features.py  (extract features)")
    print(f"    2. python training/oww_train.py             (train OWW model)")
    print(f"    3. python training/split_dataset.py         (for NeMo)")
    print(f"    4. python fast_train.py                     (train NeMo model)")
    print("=" * 70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print("\n\n  Interrupted. Resume by running the script again.")
        print("  Already generated files will be skipped automatically.")
