"""
音频后台服务 (Audio Service) - 四层防御 + 无缝桥接版
核心防护：
1. 静音门限 (VAD)：RMS < 0.003 不推理，避免底噪误判。
2. 连续确认机制：连续 2 次推理命中才触发唤醒，过滤噪声尖峰。
3. 高阈值策略：阈值可按模型调整。
4. 线程安全缓冲区：原地写入环形缓冲区，消除竞态条件。
5. 零丢失桥接：唤醒时回捞缓冲区音频，确保指令开头不丢失。

支持两种唤醒模型（启动时交互选择）：
  1. NeMo MatchboxNet  (xiaokang_xvf3800_pro.nemo)
  2. OpenWakeWord CNN  (xiaokang_oww.onnx)
"""
import socket
import json
import threading
import base64
import numpy as np
import sounddevice as sd
import torch
import queue
import time
from datetime import datetime

# ================= 配置区 =================
# --- NeMo MatchboxNet ---
NEMO_MODEL_PATH = "xiaokang_xvf3800_pro.nemo"
NEMO_THRESHOLD = 0.75        # NeMo 唤醒阈值

# --- OpenWakeWord CNN ---
OWW_MODEL_PATH = "xiaokang_oww.onnx"
OWW_THRESHOLD = 0.75         # OWW 唤醒阈值（提高以减少误唤醒，治标方案）

# --- 公共参数 ---
SAMPLE_RATE = 16000
BUFFER_DURATION = 2.0
STEP_DURATION = 0.2          # 推理步长 0.2 秒，连续确认延迟约 0.4s
CONSECUTIVE_HITS_REQUIRED = 2  # 连续 2 次命中才唤醒
VAD_RMS_THRESHOLD = 0.003    # 静音门限：低于此值跳过推理
DEVICE_ID = 14
COOLDOWN_SECONDS = 2.0
AUDIO_GAIN = 1.2
RESCUE_SECONDS = 1.0         # 唤醒时回捞前 N 秒音频，防止指令开头丢失

UDP_DATA_TARGET = ("127.0.0.1", 10002)
UDP_CTRL_PORT = 10003
# ==========================================

is_streaming = False
cooldown_until = 0

# 环形缓冲区（线程安全：仅原地修改，不替换引用）
buffer_len = int(BUFFER_DURATION * SAMPLE_RATE)
audio_buffer = np.zeros(buffer_len, dtype=np.float32)
write_pos = 0  # 环形缓冲区写入位置

data_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ctrl_sock = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
ctrl_sock.bind(("127.0.0.1", UDP_CTRL_PORT))

stream_queue = queue.Queue()


def control_listener():
    global is_streaming, cooldown_until, write_pos
    print(f"🎧 监听控制指令端口: {UDP_CTRL_PORT}")
    while True:
        try:
            data, _ = ctrl_sock.recvfrom(1024)
            cmd = json.loads(data.decode('utf-8'))
            if cmd.get("command") == "stop":
                if is_streaming:
                    print("\n🛑 收到主控指令：立刻停止推流！进入冷却期...")
                    is_streaming = False
                    cooldown_until = time.time() + COOLDOWN_SECONDS
                    audio_buffer.fill(0.0)
                    write_pos = 0
        except Exception as e:
            pass


def network_sender():
    while True:
        try:
            chunk = stream_queue.get()
            chunk_int16 = (chunk * 32767).astype(np.int16)
            payload = {
                "type": "audio_stream",
                "data": base64.b64encode(chunk_int16.tobytes()).decode('utf-8')
            }
            data_sock.sendto(json.dumps(payload).encode('utf-8'), UDP_DATA_TARGET)
        except Exception as e:
            pass


def main():
    global is_streaming, cooldown_until, write_pos

    # ── 模型选择 ────────────────────────────────────────────────
    print("=" * 55)
    print("🎙️  WakeFusion Audio Service")
    print("=" * 55)
    print("请选择唤醒词模型：")
    print(f"  1. NeMo MatchboxNet  ({NEMO_MODEL_PATH})")
    print(f"  2. OpenWakeWord CNN  ({OWW_MODEL_PATH})")
    choice = input("请输入序号 (直接回车 = 1 NeMo): ").strip()
    use_oww = (choice == "2")

    # ── 加载模型，生成统一的 infer(audio_float32) 闭包 ─────────
    # infer() 接受 float32 音频数组，返回 (label_str, confidence_float)
    if use_oww:
        import onnxruntime as ort
        from openwakeword.utils import AudioFeatures

        print(f"\n📦 正在加载 OpenWakeWord 模型: {OWW_MODEL_PATH} ...")
        oww_session = ort.InferenceSession(OWW_MODEL_PATH)
        oww_input_name = oww_session.get_inputs()[0].name
        oww_features = AudioFeatures(inference_framework="onnx")
        active_threshold = OWW_THRESHOLD
        model_tag = f"OpenWakeWord CNN  (阈值 {OWW_THRESHOLD})"
        print("   ✅ 加载完成")

        def infer(audio_float32):
            """OWW 推理：embed_clips → ONNX session（与训练特征完全一致）"""
            audio_int16 = (audio_float32 * 32767).astype(np.int16)
            audio_batch = audio_int16.reshape(1, -1)
            embeddings = oww_features.embed_clips(audio_batch)          # (1, 16, 96)
            score = float(oww_session.run(
                None, {oww_input_name: embeddings.astype(np.float32)}
            )[0][0][0])
            # 统一为 (label, conf) 格式
            if score > 0.5:
                return "xiaokang", score
            else:
                return "others", 1.0 - score

    else:
        from nemo.collections.asr.models import EncDecClassificationModel

        print(f"\n📦 正在加载 NeMo 模型: {NEMO_MODEL_PATH} ...")
        torch.set_float32_matmul_precision('medium')
        nemo_model = EncDecClassificationModel.restore_from(NEMO_MODEL_PATH)
        nemo_model.eval()
        if torch.cuda.is_available():
            nemo_model = nemo_model.cuda()
        nemo_labels = nemo_model.cfg.labels
        active_threshold = NEMO_THRESHOLD
        model_tag = f"NeMo MatchboxNet  (阈值 {NEMO_THRESHOLD})"
        print("   ✅ 加载完成")

        def infer(audio_float32):
            """NeMo 推理：EncDecClassificationModel → softmax"""
            audio_tensor = torch.FloatTensor(audio_float32).unsqueeze(0)
            audio_len = torch.LongTensor([len(audio_float32)])
            if torch.cuda.is_available():
                audio_tensor = audio_tensor.cuda()
                audio_len = audio_len.cuda()
            with torch.no_grad():
                logits = nemo_model.forward(
                    input_signal=audio_tensor, input_signal_length=audio_len
                )
                probs = torch.softmax(logits, dim=-1)
                idx = torch.argmax(probs, dim=-1).item()
                conf = probs[0][idx].item()
                return nemo_labels[idx], conf

    # ── 线程启动 ────────────────────────────────────────────────
    threading.Thread(target=control_listener, daemon=True).start()
    threading.Thread(target=network_sender, daemon=True).start()

    # 重置缓冲区
    audio_buffer.fill(0.0)
    write_pos = 0

    # 连续命中计数器
    consecutive_hits = 0

    def audio_callback(indata, frames, time_info, status):
        global write_pos
        if status:
            pass  # 屏蔽底层警告

        new_data = indata[:, 0]

        # 增益补偿并防止爆音裁剪
        new_data = np.clip(new_data * AUDIO_GAIN, -1.0, 1.0)

        if is_streaming:
            try:
                stream_queue.put_nowait(new_data.copy())
            except queue.Full:
                pass
        else:
            # 🌟 第四层：环形缓冲区原地写入，避免 np.roll 创建新数组的竞态问题
            n = len(new_data)
            if write_pos + n <= buffer_len:
                audio_buffer[write_pos:write_pos + n] = new_data
                write_pos += n
            else:
                # 到达末尾，环形回绕
                first_part = buffer_len - write_pos
                audio_buffer[write_pos:] = new_data[:first_part]
                audio_buffer[:n - first_part] = new_data[first_part:]
                write_pos = n - first_part

    stream = sd.InputStream(
        samplerate=SAMPLE_RATE, channels=1, blocksize=int(SAMPLE_RATE * STEP_DURATION),
        device=DEVICE_ID, callback=audio_callback
    )

    print("\n" + "=" * 60)
    print("🎙️ Audio Service 已启动 (四层防御版)")
    print(f"   模型: {model_tag}")
    print(f"   推理步长: 每 {STEP_DURATION}s 一次")
    print(f"   连续确认: 需连续 {CONSECUTIVE_HITS_REQUIRED} 次命中")
    print(f"   静音门限: RMS < {VAD_RMS_THRESHOLD} 跳过推理")
    print(f"   音频桥接: 唤醒时回捞前 {RESCUE_SECONDS}s 音频")
    print("=" * 60)

    with stream:
        try:
            while True:
                sd.sleep(int(STEP_DURATION * 1000))

                if is_streaming:
                    continue

                if time.time() < cooldown_until:
                    audio_buffer.fill(0.0)
                    write_pos = 0
                    consecutive_hits = 0
                    print("❄️ 冷却中...          ", end='\r')
                    continue

                # 从环形缓冲区中按正确顺序读取完整音频
                pos = write_pos  # 快照当前写入位置
                current_audio = np.concatenate([
                    audio_buffer[pos:],
                    audio_buffer[:pos]
                ])

                # 🌟 第一层：静音门限 (VAD)
                rms = np.sqrt(np.mean(current_audio**2))
                if rms < VAD_RMS_THRESHOLD:
                    consecutive_hits = 0  # 静音时重置连续计数
                    continue

                # 🌟 推理（NeMo 或 OWW，由 infer() 闭包统一处理）
                label, conf = infer(current_audio)

                # 🌟 第二层 + 第三层：连续确认 + 阈值
                if label == "xiaokang" and conf > active_threshold:
                    consecutive_hits += 1

                    if consecutive_hits >= CONSECUTIVE_HITS_REQUIRED:
                        # ✅ 唤醒确认！
                        now_str = datetime.now().strftime("%H:%M")
                        print(f"⚡ 唤醒成功！(置信度={conf:.2%}) 🕐 {now_str}")
                        print(f"   切换为音频推流模式...")

                        # =======================================================
                        # 🚑 无缝二进制流桥接 (Zero-Loss Streaming)
                        # 严格时序：抢救音频 → 发唤醒事件 → 推入队列 → 切换流模式
                        # =======================================================

                        # Step 1: 从环形缓冲区按正确时序抢救最后 N 秒音频
                        rescue_samples = int(RESCUE_SECONDS * SAMPLE_RATE)
                        pos = write_pos  # 快照写入位置
                        ordered_audio = np.concatenate([
                            audio_buffer[pos:], audio_buffer[:pos]
                        ])
                        rescue_audio = ordered_audio[-rescue_samples:].copy()

                        # Step 2: 发送唤醒事件（让接收端知道数据流即将到来）
                        wake_payload = {
                            "type": "wake_word_hit",
                            "keyword": "xiaokang",
                            "confidence": conf
                        }
                        data_sock.sendto(json.dumps(wake_payload).encode('utf-8'), UDP_DATA_TARGET)

                        # Step 3: 将抢救的音频切片推入队列（必须在 is_streaming=True 之前！）
                        #   切成 0.1 秒小块，模拟麦克风连续吐数据，避免单个超大 UDP 包丢包
                        chunk_size = int(SAMPLE_RATE * 0.1)
                        for i in range(0, len(rescue_audio), chunk_size):
                            chunk = rescue_audio[i:i + chunk_size]
                            try:
                                stream_queue.put_nowait(chunk)
                            except queue.Full:
                                pass

                        # Step 4: 最后才切换流模式（回调线程开始向队列追加实时数据）
                        is_streaming = True

                        print(f"   🌊 已将前 {RESCUE_SECONDS}s 指令音频无缝桥接入推流队列！")

                        # Step 5: 清空缓存，进入冷却
                        audio_buffer.fill(0.0)
                        write_pos = 0
                        consecutive_hits = 0
                        cooldown_until = time.time() + COOLDOWN_SECONDS
                    else:
                        print(f"🔍 疑似唤醒... (连续{consecutive_hits}/{CONSECUTIVE_HITS_REQUIRED}, 置信度={conf:.2%})", end='\r')
                else:
                    consecutive_hits = 0  # 一旦中断，重置计数
                    print(f"听... ({label} {conf:.1%})        ", end='\r')

        except KeyboardInterrupt:
            print("\n🛑 服务已关闭。")

if __name__ == "__main__":
    main()
