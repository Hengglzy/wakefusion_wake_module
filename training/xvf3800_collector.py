"""
XVF3800 唤醒词专属采集与增强工具
功能：录制真人语音正/负样本，并自动进行基础数据增强（远场模拟 + 底噪注入）

文件命名规则：
  - 正样本原声: real_xiaokang_<timestamp>.wav
  - 负样本原声: real_others_<timestamp>.wav
  - 增强变体:   *_aug_quiet.wav (音量减半), *_aug_noisy.wav (加白噪声)

⚠️ 重要修复（防止文件数量爆炸）：
   - auto_augment_data() 现在只处理本次新录制的文件，不会重复处理已有文件
   - 如果录制 20 个新文件，只会生成 40 个增强文件（而不是处理所有已有文件）

与其他脚本的协作关系：
  - 本脚本生成: real_*_aug_*.wav
  - positive_data_factory.py 会处理 real_* 并生成 *_pitch_* 和 *_speed_*
  - 本脚本排除: tts_*, slice_*, mega_*, *_pitch_*, *_speed_*（避免冲突）

运行环境：wakefusion 虚拟环境
运行命令：python training/xvf3800_collector.py
"""
import os
import time
import numpy as np
import sounddevice as sd
from scipy.io import wavfile
import librosa

# ================= 配置区 =================
SAMPLE_RATE = 16000
DURATION = 2.0  # 每条录音 2 秒
# 请确保填入你之前查到的 XVF3800 的设备 ID（比如 14 或 31）
# 如果设为 None，将使用系统默认输入设备
DEVICE_ID = 14 

# 存储目录
OUTPUT_DIR = "custom_dataset"
POS_DIR = os.path.join(OUTPUT_DIR, "xiaokang")
NEG_DIR = os.path.join(OUTPUT_DIR, "others")
os.makedirs(POS_DIR, exist_ok=True)
os.makedirs(NEG_DIR, exist_ok=True)
# ==========================================


def list_audio_devices():
    """列出所有可用的音频输入设备"""
    print("\n📋 可用的音频输入设备：")
    print("=" * 60)
    devices = sd.query_devices()
    for i, device in enumerate(devices):
        if device['max_input_channels'] > 0:
            default_mark = " (默认)" if i == sd.default.device[0] else ""
            print(f"  [{i}] {device['name']} - {device['max_input_channels']} 通道{default_mark}")
    print("=" * 60)


def record_audio(duration, device=None):
    """底层录音函数（自动适配设备通道数）"""
    print(f"🔴 录音中... ({duration}秒)")

    # 尝试录音，如果失败则回退到默认设备
    attempts = [
        (device, "指定设备"),
        (None, "默认设备")
    ] if device is not None else [(None, "默认设备")]

    last_error = None
    for attempt_device, device_desc in attempts:
        try:
            # 查询设备信息
            if attempt_device is not None:
                device_info = sd.query_devices(attempt_device)
                max_channels = device_info['max_input_channels']
                if max_channels == 0:
                    raise ValueError(f"设备 {attempt_device} 不支持输入")

            # 执行录音（始终使用单声道）
            audio = sd.rec(
                int(duration * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                device=attempt_device
            )
            sd.wait()
            print("🟢 录音结束")

            # 如果是多通道，只取第一个通道
            if audio.ndim > 1 and audio.shape[1] > 1:
                audio = audio[:, 0]

            return audio.flatten()

        except Exception as e:
            last_error = e
            if attempt_device == device:
                print(f"⚠️ 使用{device_desc}失败: {e}")
                if len(attempts) > 1:
                    print("   尝试回退到默认设备...")
                continue
            else:
                break

    # 所有尝试都失败
    print(f"\n❌ 录音失败: {last_error}")
    print("\n💡 故障排除建议：")
    print("   1. 检查麦克风是否已连接并启用")
    print("   2. 运行选项 'd' 查看所有可用设备")
    print("   3. 尝试将脚本中的 DEVICE_ID 设为 None")
    print("   4. 检查其他程序是否占用了音频设备")
    raise RuntimeError(f"无法从任何设备录音: {last_error}")


def collect_samples(label_dir, prefix, count):
    """
    通用的连续采集与保存函数。
    返回本次新录制的文件列表（用于后续增强）。
    """
    new_files = []
    for i in range(count):
        input(f"👉 准备好后，按回车键开始录制第 {i+1}/{count} 条...")
        audio = record_audio(DURATION, DEVICE_ID)
        filename = f"{prefix}_{int(time.time())}.wav"
        filepath = os.path.join(label_dir, filename)
        wavfile.write(filepath, SAMPLE_RATE, (audio * 32767).astype(np.int16))
        new_files.append(filepath)
    return new_files


def auto_augment_data(new_files):
    """
    自动数据增强：仅对本次新录制的纯原声真人录音进行混音裂变。
    
    核心改进：只处理传入的 new_files 列表，而不是扫描整个目录。
    这样可以避免重复处理已有文件，防止文件数量爆炸。
    
    排除规则（不碰以下文件）：
      - tts_*      → TTS 生成的样本（已有多声音多风格）
      - slice_*    → negative_slicer 切片的环境音
      - mega_*     → mega_dataset_generator 生成的大规模数据
      - *_aug_*    → 本函数已生成的增强版本
      - *_pitch_*  → positive_data_factory 生成的变调文件
      - *_speed_*  → positive_data_factory 生成的变速文件
    """
    if not new_files:
        return 0
    
    print("\n" + "=" * 40)
    print("🪄 正在进行自动混音与数据裂变...")
    print(f"   处理范围: 仅本次新录制的 {len(new_files)} 个文件")
    
    augmented_count = 0
    for filepath in new_files:
        # 验证文件存在且符合命名规则
        if not os.path.exists(filepath):
            continue
        
        filename = os.path.basename(filepath)
        
        # 只处理 real_ 前缀的纯原声文件
        if not filename.startswith('real_'):
            continue
        
        # 排除已增强的文件
        if 'aug' in filename or 'pitch' in filename or 'speed' in filename:
            continue
        
        # 检查是否已经生成过增强数据，避免重复增强
        quiet_path = filepath.replace(".wav", "_aug_quiet.wav")
        noisy_path = filepath.replace(".wav", "_aug_noisy.wav")
        
        if os.path.exists(quiet_path) and os.path.exists(noisy_path):
            continue  # 已存在，跳过
        
        try:
            audio, _ = librosa.load(filepath, sr=SAMPLE_RATE)
            
            # 1. 音量减小版 (模拟远场)
            if not os.path.exists(quiet_path):
                audio_quiet = audio * 0.5
                wavfile.write(quiet_path, SAMPLE_RATE, (audio_quiet * 32767).astype(np.int16))
                augmented_count += 1
            
            # 2. 注入设备底噪 (白噪声模拟)
            if not os.path.exists(noisy_path):
                noise = np.random.normal(0, 0.005, len(audio))
                audio_noisy = np.clip(audio + noise, -1.0, 1.0)
                wavfile.write(noisy_path, SAMPLE_RATE, (audio_noisy * 32767).astype(np.int16))
                augmented_count += 1
        except Exception as e:
            print(f"  ⚠️ 处理 {filename} 时出错: {e}")
            continue
            
    print(f"✅ 数据裂变完成！本次新增了 {augmented_count} 条增强数据。")
    return augmented_count


def main():
    # 启动时显示当前设备信息
    print("\n" + "=" * 50)
    print("🎤 XVF3800 唤醒词专属采集与增强工具")
    print("=" * 50)
    if DEVICE_ID is not None:
        try:
            device_info = sd.query_devices(DEVICE_ID)
            print(f"📱 当前使用设备: [{DEVICE_ID}] {device_info['name']}")
            print(f"   最大输入通道: {device_info['max_input_channels']}")
        except Exception:
            print(f"⚠️ 警告: 无法访问设备 ID {DEVICE_ID}，将尝试使用默认设备")
    else:
        default_device = sd.default.device[0]
        device_info = sd.query_devices(default_device)
        print(f"📱 使用默认设备: [{default_device}] {device_info['name']}")

    print("💡 提示: 输入 'd' 可查看所有可用设备")
    print("=" * 50)

    while True:
        print("\n" + "=" * 50)
        print("🎤 XVF3800 唤醒词专属采集与增强工具")
        print("1. 仅录制正样本 (保存在 xiaokang)")
        print("2. 仅录制负样本 (保存在 others)")
        print("3. 完整录制模式 (先录正样本，再录负样本)")
        print("d. 查看所有音频设备")
        print("q. 退出程序")
        print("=" * 50)
        
        choice = input("请选择模式 (1/2/3/d/q): ").strip()
        
        if choice == '1':
            try:
                count = int(input("请输入要录制的正样本数量: "))
                print(f"\n🎯 准备录制 {count} 条唤醒词 '你好小康'...")
                new_files = collect_samples(POS_DIR, "real_xiaokang", count)
                if new_files:
                    auto_augment_data(new_files)
            except (ValueError, KeyboardInterrupt) as e:
                if isinstance(e, KeyboardInterrupt):
                    print("\n⚠️ 录制已中断")
                else:
                    print(f"❌ 输入错误: {e}")
            except Exception as e:
                print(f"❌ 录制失败: {e}")
                print("   请检查设备连接和配置")
            
        elif choice == '2':
            try:
                count = int(input("请输入要录制的负样本数量: "))
                print(f"\n🛡️ 准备录制 {count} 条负样本 (请专门录制导致误唤醒的废话，如 '小康小康', 'Yes' 等)...")
                new_files = collect_samples(NEG_DIR, "real_others", count)
                if new_files:
                    auto_augment_data(new_files)
            except (ValueError, KeyboardInterrupt) as e:
                if isinstance(e, KeyboardInterrupt):
                    print("\n⚠️ 录制已中断")
                else:
                    print(f"❌ 输入错误: {e}")
            except Exception as e:
                print(f"❌ 录制失败: {e}")
                print("   请检查设备连接和配置")
            
        elif choice == '3':
            try:
                pos_count = int(input("请输入正样本数量: "))
                neg_count = int(input("请输入负样本数量: "))
                print(f"\n🎯 阶段一：录制 {pos_count} 条唤醒词 '你好小康'...")
                pos_new_files = collect_samples(POS_DIR, "real_xiaokang", pos_count)
                print(f"\n🛡️ 阶段二：录制 {neg_count} 条负样本...")
                neg_new_files = collect_samples(NEG_DIR, "real_others", neg_count)
                # 合并所有新文件，一次性增强
                all_new_files = pos_new_files + neg_new_files
                if all_new_files:
                    auto_augment_data(all_new_files)
            except (ValueError, KeyboardInterrupt) as e:
                if isinstance(e, KeyboardInterrupt):
                    print("\n⚠️ 录制已中断")
                else:
                    print(f"❌ 输入错误: {e}")
            except Exception as e:
                print(f"❌ 录制失败: {e}")
                print("   请检查设备连接和配置")

        elif choice.lower() == 'd':
            list_audio_devices()
            print(f"\n💡 要更改设备，请编辑脚本中的 DEVICE_ID 变量（当前: {DEVICE_ID}）")
            
        elif choice.lower() == 'q':
            print("👋 退出程序。祝你炼丹顺利！")
            break
        else:
            print("❌ 无效的输入，请重新选择。")


if __name__ == "__main__":
    main()
