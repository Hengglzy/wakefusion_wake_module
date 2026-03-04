"""
NeMo vs OpenWakeWord 双模型对比测试台 V2
==========================================
核心改进（相比 V1）：

★ 修复训练-推理特征不一致（根本性修复）
  V1 的 predict_clip_streaming() 用 model.predict() 流式喂 80ms 块，
  训练时用 embed_clips() 批量处理完整 2 秒音频 —— 两者特征分布不同。
  V2 统一使用 embed_clips() + 直接调用 ONNX session，
  与 oww_prepare_features.py 特征提取流程完全一致。

★ 实时模式新增冷却期 + 连续确认
  原版无任何冷却，一次说话会连续触发十几次。
  V2 加入：连续确认 (OWW_CONSECUTIVE_REQUIRED=2) + 冷却期 (COOLDOWN_SECONDS=2.0)。

支持模式：
  1. 单次录音对比：录制 2 秒音频并对比两个模型的识别结果
  2. 实时麦克风持续对比：同时显示两个模型的得分
  3. 批量文件对比：扫描指定目录下的 WAV 文件进行批量对比

前置条件：
  - xiaokang_xvf3800_pro.nemo (NeMo 模型)
  - xiaokang_oww.onnx (OpenWakeWord 模型，由 training/oww_train.py 生成)

运行环境：wakefusion 虚拟环境
运行命令：python tests/test_compare_models.py
"""
import os
import sys
import numpy as np
import sounddevice as sd
import time
import torch

# ================= 配置区 =================
NEMO_MODEL_PATH = "xiaokang_xvf3800_pro.nemo"
OWW_MODEL_PATH = "xiaokang_oww.onnx"

SAMPLE_RATE = 16000
DURATION = 2.0                  # 录音时长（秒）
DEVICE_ID = 14                  # XVF3800 设备 ID
CHUNK_SIZE = 1280               # 每次读取 80ms（1280 samples）

NEMO_THRESHOLD = 0.75           # NeMo 唤醒阈值
OWW_THRESHOLD = 0.50            # OWW 唤醒阈值（CNN+FocalLoss 后模型更保守，0.5 合适）

COOLDOWN_SECONDS = 2.0          # 实时模式冷却期（秒）
OWW_CONSECUTIVE_REQUIRED = 2   # 实时模式：连续 N 次命中才触发唤醒
NEMO_CONSECUTIVE_REQUIRED = 2  # 实时模式：同上
# ==========================================


class NeMoModel:
    """NeMo MatchboxNet 模型封装"""
    def __init__(self, model_path):
        self.model = None
        self.labels = None
        self.model_path = model_path

    def load(self):
        if not os.path.exists(self.model_path):
            print(f"  ⚠️ NeMo 模型不存在: {self.model_path}")
            return False
        try:
            from nemo.collections.asr.models import EncDecClassificationModel
            torch.set_float32_matmul_precision('medium')
            self.model = EncDecClassificationModel.restore_from(self.model_path)
            self.model.eval()
            if torch.cuda.is_available():
                self.model = self.model.cuda()
            self.labels = self.model.cfg.labels
            return True
        except Exception as e:
            print(f"  ❌ NeMo 加载失败: {e}")
            return False

    def predict(self, audio_float):
        """从 float32 音频推理，返回 (label, confidence)"""
        audio_tensor = torch.FloatTensor(audio_float).unsqueeze(0)
        audio_len = torch.LongTensor([len(audio_float)])
        if torch.cuda.is_available():
            audio_tensor = audio_tensor.cuda()
            audio_len = audio_len.cuda()

        with torch.no_grad():
            logits = self.model.forward(input_signal=audio_tensor, input_signal_length=audio_len)
            probs = torch.softmax(logits, dim=-1)
            idx = torch.argmax(probs, dim=-1).item()
            conf = probs[0][idx].item()
            return self.labels[idx], conf


class OWWModel:
    """
    OpenWakeWord 模型封装 V2
    =========================
    核心设计：
    - predict_clip()：使用 embed_clips() 提取特征 + 直接调用 ONNX session，
      与训练流程完全一致，用于单次录音 / 批量文件测试。
    - predict_streaming()：用 openwakeword.Model.predict() 流式推理，
      用于实时麦克风测试（内部维护 2 秒滚动音频缓冲）。
    """
    def __init__(self, model_path):
        self.session = None         # ONNX 推理会话
        self.audio_features = None  # openWakeWord 特征提取器
        self.model_name = None
        self.model_path = model_path
        self._input_name = None

    def load(self):
        if not os.path.exists(self.model_path):
            print(f"  ⚠️ OWW 模型不存在: {self.model_path}")
            return False
        try:
            import onnxruntime as ort
            from openwakeword.utils import AudioFeatures

            # 直接 ONNX session（与训练特征对齐）
            self.session = ort.InferenceSession(self.model_path)
            self._input_name = self.session.get_inputs()[0].name

            # 特征提取器（与 oww_prepare_features.py 完全一致）
            self.audio_features = AudioFeatures(inference_framework="onnx")

            # 模型名称（取文件名不含扩展名）
            self.model_name = os.path.splitext(os.path.basename(self.model_path))[0]
            return True
        except Exception as e:
            print(f"  ❌ OWW 加载失败: {e}")
            return False

    def predict_clip(self, audio_int16):
        """
        完整音频片段预测（训练-推理特征完全一致）。
        用于：单次录音对比 + 批量文件对比。

        Args:
            audio_int16: int16 音频数组（任意长度，自动 pad/trim 至 2 秒）
        Returns:
            float: 唤醒概率 [0, 1]
        """
        # 统一为 2 秒长度（与训练一致）
        target = int(SAMPLE_RATE * DURATION)
        if len(audio_int16) > target:
            audio_int16 = audio_int16[:target]
        elif len(audio_int16) < target:
            audio_int16 = np.pad(audio_int16, (0, target - len(audio_int16)))

        # 使用 embed_clips 提取特征（与 oww_prepare_features.py 完全一致）
        audio_batch = audio_int16.reshape(1, -1)
        embeddings = self.audio_features.embed_clips(audio_batch)  # (1, 16, 96)

        # 直接调用 ONNX 模型
        result = self.session.run(
            None,
            {self._input_name: embeddings.astype(np.float32)}
        )[0]
        return float(result[0][0])


def test_recording_compare(nemo_model, oww_model):
    """模式 1：单次录音对比"""
    print(f"\n🎤 准备录音（{DURATION}秒）...")
    time.sleep(0.5)
    print("🔴 >>> 请说: '你好小康' <<<")

    recording = sd.rec(
        int(DURATION * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        device=DEVICE_ID,
        dtype='float32'
    )
    sd.wait()
    print("🟢 录音结束，正在双模型识别...\n")

    audio_float = recording.flatten()
    audio_int16 = (audio_float * 32767).astype(np.int16)

    # NeMo 推理
    results = {}
    if nemo_model and nemo_model.model:
        label, conf = nemo_model.predict(audio_float)
        nemo_hit = label == "xiaokang" and conf > NEMO_THRESHOLD
        results['nemo'] = {'label': label, 'conf': conf, 'hit': nemo_hit}
    else:
        results['nemo'] = None

    # OWW 推理（使用 predict_clip，与训练一致）
    if oww_model and oww_model.session:
        score = oww_model.predict_clip(audio_int16)
        oww_hit = score > OWW_THRESHOLD
        results['oww'] = {'score': score, 'hit': oww_hit}
    else:
        results['oww'] = None

    # 打印对比结果
    print("┌" + "─" * 48 + "┐")
    print("│" + " 🥊 双模型对比结果".center(44) + "│")
    print("├" + "─" * 48 + "┤")

    if results['nemo']:
        icon = "✅" if results['nemo']['hit'] else "❌"
        print(f"│ {icon} NeMo MatchboxNet                          │")
        print(f"│    标签: {results['nemo']['label']:>10}  置信度: {results['nemo']['conf']:.2%}       │")
        print(f"│    阈值: {NEMO_THRESHOLD:.2%}        判定: {'唤醒' if results['nemo']['hit'] else '未唤醒':>6}       │")
    else:
        print("│ ⚠️  NeMo 模型未加载                            │")

    print("├" + "─" * 48 + "┤")

    if results['oww']:
        icon = "✅" if results['oww']['hit'] else "❌"
        print(f"│ {icon} OpenWakeWord (CNN + Focal Loss)           │")
        print(f"│    得分: {results['oww']['score']:.2%}                              │")
        print(f"│    阈值: {OWW_THRESHOLD:.2%}        判定: {'唤醒' if results['oww']['hit'] else '未唤醒':>6}       │")
    else:
        print("│ ⚠️  OWW 模型未加载                              │")

    print("└" + "─" * 48 + "┘")


def test_realtime_compare(nemo_model, oww_model):
    """
    模式 2：实时麦克风双模型对比
    使用滚动 2 秒音频缓冲区 + embed_clips 推理（与训练一致）。
    加入连续确认 + 冷却期，防止重复触发。
    """
    print(f"\n🎙️ 实时对比模式 (按 Ctrl+C 退出)")
    print(f"   OWW 连续确认: {OWW_CONSECUTIVE_REQUIRED} 次  |  NeMo 连续确认: {NEMO_CONSECUTIVE_REQUIRED} 次  |  冷却: {COOLDOWN_SECONDS}s")
    print("=" * 65)

    # 滚动音频缓冲区（float32 给 NeMo，int16 给 OWW embed_clips）
    buffer_len = int(DURATION * SAMPLE_RATE)
    nemo_buffer = np.zeros(buffer_len, dtype=np.float32)
    oww_buffer = np.zeros(buffer_len, dtype=np.int16)

    # 连续命中计数器 + 冷却时间
    nemo_consecutive = 0
    nemo_cooldown_until = 0
    oww_consecutive = 0
    oww_cooldown_until = 0

    try:
        with sd.InputStream(
            samplerate=SAMPLE_RATE,
            channels=1,
            blocksize=CHUNK_SIZE,
            device=DEVICE_ID,
            dtype='float32'
        ) as stream:
            step_counter = 0
            while True:
                audio_chunk, _ = stream.read(CHUNK_SIZE)
                new_float = audio_chunk.flatten()
                new_int16 = (new_float * 32767).astype(np.int16)

                # 更新滚动缓冲区
                nemo_buffer = np.roll(nemo_buffer, -len(new_float))
                nemo_buffer[-len(new_float):] = new_float
                oww_buffer = np.roll(oww_buffer, -len(new_int16))
                oww_buffer[-len(new_int16):] = new_int16

                step_counter += 1
                if step_counter % 4 != 0:  # 每 ~320ms 推理一次
                    continue

                now = time.time()

                # ── NeMo 推理 ──────────────────────────────
                nemo_str = ""
                nemo_detected = False
                if nemo_model and nemo_model.model:
                    if now < nemo_cooldown_until:
                        nemo_consecutive = 0
                        label, conf = "cooldown", 0.0
                    else:
                        label, conf = nemo_model.predict(nemo_buffer)
                        if label == "xiaokang" and conf > NEMO_THRESHOLD:
                            nemo_consecutive += 1
                        else:
                            nemo_consecutive = 0

                        if nemo_consecutive >= NEMO_CONSECUTIVE_REQUIRED:
                            nemo_detected = True
                            nemo_cooldown_until = now + COOLDOWN_SECONDS
                            nemo_consecutive = 0

                    hit_icon = "✅" if nemo_detected else "  "
                    nemo_str = f"NeMo: {label:>9} {conf:.1%} {hit_icon}"

                # ── OWW 推理（embed_clips，与训练一致）──────
                oww_str = ""
                oww_detected = False
                oww_score = 0.0
                if oww_model and oww_model.session:
                    if now < oww_cooldown_until:
                        oww_consecutive = 0
                        oww_score = 0.0
                    else:
                        oww_score = oww_model.predict_clip(oww_buffer)
                        if oww_score > OWW_THRESHOLD:
                            oww_consecutive += 1
                        else:
                            oww_consecutive = 0

                        if oww_consecutive >= OWW_CONSECUTIVE_REQUIRED:
                            oww_detected = True
                            oww_cooldown_until = now + COOLDOWN_SECONDS
                            oww_consecutive = 0

                    hit_icon = "✅" if oww_detected else "  "
                    oww_str = f"OWW: {oww_score:.1%} {hit_icon}"

                # ── 打印 ──────────────────────────────────
                if nemo_detected or oww_detected:
                    now_str = time.strftime("%H:%M:%S")
                    winners = []
                    if nemo_detected:
                        winners.append(f"NeMo({conf:.1%})")
                    if oww_detected:
                        winners.append(f"OWW({oww_score:.1%})")
                    print(f"⚡ 唤醒！{' + '.join(winners)}  🕐 {now_str}")
                else:
                    print(f"  {nemo_str} | {oww_str}     ", end='\r')

    except KeyboardInterrupt:
        print(f"\n🛑 监听已停止。")


def test_batch_compare(nemo_model, oww_model):
    """模式 3：批量文件对比（使用 predict_clip，与训练一致）"""
    try:
        from tkinter import Tk, filedialog
        root = Tk()
        root.withdraw()
        root.attributes('-topmost', True)
        directory = filedialog.askdirectory(title="选择包含 WAV 文件的目录")
        root.destroy()
    except Exception:
        directory = input("请输入包含 WAV 文件的目录路径: ").strip().strip('"')

    if not directory or not os.path.isdir(directory):
        print("取消或目录不存在。")
        return

    import librosa
    wav_files = sorted([f for f in os.listdir(directory) if f.endswith('.wav')])

    if not wav_files:
        print(f"❌ 目录中没有 WAV 文件: {directory}")
        return

    print(f"\n📂 找到 {len(wav_files)} 个 WAV 文件")
    print("-" * 80)
    print(f"{'文件名':>35} | {'NeMo':>15} | {'OWW':>10} | {'一致?'}")
    print("-" * 80)

    nemo_hits = oww_hits = both_agree = 0

    for filename in wav_files:
        filepath = os.path.join(directory, filename)
        try:
            audio_float, _ = librosa.load(filepath, sr=SAMPLE_RATE)
            audio_int16 = (audio_float * 32767).astype(np.int16)
        except Exception:
            print(f"{filename[:33]:>35} | {'加载失败':>15} | {'':>10} | ")
            continue

        # NeMo
        nemo_result = ""
        nemo_is_hit = False
        if nemo_model and nemo_model.model:
            label, conf = nemo_model.predict(audio_float)
            nemo_is_hit = label == "xiaokang" and conf > NEMO_THRESHOLD
            nemo_result = f"{label} {conf:.1%}"
            if nemo_is_hit:
                nemo_hits += 1

        # OWW（predict_clip，与训练特征一致）
        oww_result = ""
        oww_is_hit = False
        if oww_model and oww_model.session:
            score = oww_model.predict_clip(audio_int16)
            oww_is_hit = score > OWW_THRESHOLD
            oww_result = f"{score:.1%}"
            if oww_is_hit:
                oww_hits += 1

        agree = nemo_is_hit == oww_is_hit
        if agree:
            both_agree += 1

        icon = "✅" if agree else "⚠️"
        short_name = filename[:33] if len(filename) > 33 else filename
        print(f"{short_name:>35} | {nemo_result:>15} | {oww_result:>10} | {icon}")

    total = len(wav_files)
    print("-" * 80)
    print(f"\n📊 统计:")
    print(f"   NeMo 唤醒次数: {nemo_hits}/{total}")
    print(f"   OWW  唤醒次数: {oww_hits}/{total}")
    print(f"   两者一致率:     {both_agree}/{total} ({both_agree / total:.1%})")


def main():
    print("=" * 65)
    print("🥊 NeMo vs OpenWakeWord 双模型对比测试台 V2")
    print("=" * 65)

    print("\n📦 加载模型...")
    nemo = NeMoModel(NEMO_MODEL_PATH)
    nemo_loaded = nemo.load()
    if nemo_loaded:
        print(f"  ✅ NeMo: {NEMO_MODEL_PATH}")
    else:
        print(f"  ⚠️ NeMo: 跳过")

    oww = OWWModel(OWW_MODEL_PATH)
    oww_loaded = oww.load()
    if oww_loaded:
        print(f"  ✅ OWW:  {OWW_MODEL_PATH}")
    else:
        print(f"  ⚠️ OWW:  跳过")

    if not nemo_loaded and not oww_loaded:
        print("\n❌ 两个模型都无法加载！")
        return

    while True:
        print("\n" + "=" * 50)
        print("🥊 双模型对比测试台")
        loaded = []
        if nemo_loaded:
            loaded.append("NeMo")
        if oww_loaded:
            loaded.append("OWW")
        print(f"   已加载: {' + '.join(loaded)}")
        print("=" * 50)
        print("1. 单次录音对比 (2秒)")
        print("2. 实时麦克风持续对比")
        print("3. 批量 WAV 文件对比")
        print("q. 退出程序")

        choice = input("请输入序号: ").strip()

        if choice == '1':
            test_recording_compare(
                nemo if nemo_loaded else None,
                oww if oww_loaded else None
            )
        elif choice == '2':
            test_realtime_compare(
                nemo if nemo_loaded else None,
                oww if oww_loaded else None
            )
        elif choice == '3':
            test_batch_compare(
                nemo if nemo_loaded else None,
                oww if oww_loaded else None
            )
        elif choice.lower() == 'q':
            print("👋 退出。")
            break
        else:
            print("❌ 无效输入。")


if __name__ == "__main__":
    main()
