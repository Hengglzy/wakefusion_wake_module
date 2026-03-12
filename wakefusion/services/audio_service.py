"""
音频后台服务 (Audio Service) - ZMQ版本
核心防护：
1. 静音门限 (VAD)：使用Silero VAD（深度学习）进行智能语音端点检测，替代传统RMS阈值。
2. 连续确认机制：连续 2 次推理命中才触发唤醒，过滤噪声尖峰。
3. 动态阈值策略：支持运行时动态调整阈值。
4. 线程安全缓冲区：原地写入环形缓冲区，消除竞态条件。
5. 零丢失桥接：唤醒时回捞缓冲区音频，确保指令开头不丢失。

支持两种唤醒模型（启动时交互选择）：
  1. NeMo MatchboxNet  (xiaokang_xvf3800_pro.nemo)
  2. OpenWakeWord CNN  (xiaokang_oww.onnx)

通信协议：
  - ZMQ PUB：发布音频数据流（Multipart Message：JSON元数据 + 二进制PCM）
  - ZMQ REP：接收控制指令（动态阈值调整）
"""
import json
import threading
import numpy as np
import sounddevice as sd
import torch
import queue
import time
import zmq
from datetime import datetime
from wakefusion.config import get_config
from wakefusion.services.vad_engine import SileroVADEngine

# ================= 配置区 =================
# --- NeMo MatchboxNet ---
NEMO_MODEL_PATH = "xiaokang_xvf3800_pro.nemo"

# --- OpenWakeWord CNN ---
OWW_MODEL_PATH = "xiaokang_oww.onnx"

# --- 公共参数 ---
SAMPLE_RATE = 16000
BUFFER_DURATION = 2.0
STEP_DURATION = 0.2          # 推理步长 0.2 秒，连续确认延迟约 0.4s
CONSECUTIVE_HITS_REQUIRED = 2  # 连续 2 次命中才唤醒
VAD_RMS_THRESHOLD = 0.003    # 静音门限：已废弃，由Silero VAD替代（保留用于兼容）
DEVICE_ID = 14
COOLDOWN_SECONDS = 2.0
AUDIO_GAIN = 1.2
RESCUE_SECONDS = 1.0         # 唤醒时回捞前 N 秒音频，防止指令开头丢失
# ==========================================

is_streaming = False
cooldown_until = 0

# 环形缓冲区（线程安全：仅原地修改，不替换引用）
buffer_len = int(BUFFER_DURATION * SAMPLE_RATE)
audio_buffer = np.zeros(buffer_len, dtype=np.float32)
write_pos = 0  # 环形缓冲区写入位置

# 动态阈值（初始值从配置读取）
active_threshold = 0.95  # 默认高阈值，将在main()中从配置读取

# VAD引擎（使用组合模式，完全解耦）
vad_engine = None  # 将在main()中根据配置初始化

# ZMQ Context和Sockets
zmq_context = None
zmq_pub_socket = None  # 数据流（PUB）
zmq_rep_socket = None  # 控制流（REP）

stream_queue = queue.Queue()


def control_listener_zmq():
    """ZMQ REP控制监听线程：接收动态阈值调整指令"""
    global active_threshold, cooldown_until, is_streaming, audio_buffer, write_pos
    while True:
        try:
            # 接收REQ请求（带超时）
            request = zmq_rep_socket.recv_json(zmq.NOBLOCK)
            command = request.get("command")
            if command == "set_threshold":
                new_threshold = float(request.get("value", active_threshold))
                active_threshold = new_threshold
                print(f"✅ 阈值已更新: {active_threshold:.2f}")
                # 快速响应
                zmq_rep_socket.send_json({"status": "ok", "threshold": active_threshold})
            elif command == "reset_cooldown":
                # 重置冷却期，允许立即再次唤醒
                cooldown_until = 0
                is_streaming = False  # 确保退出流模式
                if vad_engine is not None:
                    vad_engine.reset_states()  # 重置VAD，防止状态残留
                print("✅ 冷却期已重置，可以立即再次唤醒")
                zmq_rep_socket.send_json({"status": "ok", "cooldown_reset": True})
            elif command == "start_streaming":
                cooldown_until = 0
                is_streaming = True
                # 重置缓冲区，防止混入旧声音
                audio_buffer.fill(0.0)
                write_pos = 0
                print("✅ 收到中枢指令：进入免唤醒持续拾音模式 (开始推流)")
                zmq_rep_socket.send_json({"status": "ok"})
            elif command == "stop_streaming":
                # 🌟 修复：停止音频推流（进入PROCESSING状态时）
                is_streaming = False
                print("🛑 收到停止推流指令，退出流模式")
                if vad_engine is not None:
                    vad_engine.reset_states()  # 重置VAD，防止状态残留
                # 注意：只回复一次！删掉后面那些错乱的 print 和 send
                zmq_rep_socket.send_json({"status": "ok"})
            else:
                zmq_rep_socket.send_json({"status": "error", "message": "unknown command"})
        except zmq.Again:
            time.sleep(0.01)
            continue
        except Exception as e:
            try:
                zmq_rep_socket.send_json({"status": "error", "message": str(e)})
            except:
                pass


def network_sender():
    """ZMQ PUB数据发送线程：使用Silero VAD + Multipart Message发送音频数据"""
    global vad_engine, is_streaming
    # RMS 物理能量门限（Volume Gate）- 强杀底噪
    
    while True:
        try:
            chunk = stream_queue.get()
            chunk_int16 = (chunk * 32767).astype(np.int16)
            
            # --- 核心修复：RMS 物理能量门限 ---
            rms_energy = np.sqrt(np.mean(chunk_int16.astype(np.float32)**2))
            
            # 自动打印环境底噪（每 1 秒打印一次，不刷屏）
            log_counter = getattr(network_sender, "log_counter", 0) + 1
            if log_counter % 5 == 0 and not is_streaming:
                print(f"🎙️ [校准用] 当前环境底噪 RMS: {rms_energy:.1f}      ", end='\r')
            network_sender.log_counter = log_counter
            
            # 🌟 修复：提高默认阈值到 1500，强行压制 XVF3800 的 AGC 增益
            RMS_THRESHOLD = 1500.0  
            
            if rms_energy < RMS_THRESHOLD:
                vad_active = False
            else:
                if vad_engine is not None:
                    vad_active = vad_engine.is_speech(chunk_int16)
                else:
                    vad_active = True
            # ---------------------------------
            
            # 第一帧：JSON元数据
            metadata = {
                "vad": vad_active,
                "wake_word": {
                    "detected": False,  # 在唤醒时已发送，这里保持False
                    "confidence": 0.0
                },
                "timestamp": time.time()
            }
            
            # 第二帧：纯二进制PCM数据（int16）
            # 使用Multipart Message发送
            zmq_pub_socket.send_multipart([
                json.dumps(metadata).encode('utf-8'),
                chunk_int16.tobytes()
            ], zmq.NOBLOCK)
        except zmq.Again:
            # 发送缓冲区满，丢弃此帧
            pass
        except Exception as e:
            pass


def main():
    global is_streaming, cooldown_until, write_pos, active_threshold
    global zmq_context, zmq_pub_socket, zmq_rep_socket, vad_engine
    
    # 加载配置
    config = get_config()
    zmq_config = config.zmq
    audio_threshold_config = config.audio_threshold
    conversation_config = config.conversation
    vad_config = config.vad
    
    # 初始化动态阈值（从配置读取）
    active_threshold = audio_threshold_config.default
    
    # 初始化VAD RMS阈值（从配置读取，已废弃，保留用于向后兼容）
    global VAD_RMS_THRESHOLD
    VAD_RMS_THRESHOLD = conversation_config.vad_rms_threshold
    
    # 初始化Silero VAD引擎（使用组合模式）
    if vad_config.enabled and vad_config.engine == "silero":
        try:
            vad_engine = SileroVADEngine(
                threshold=vad_config.threshold,
                sample_rate=vad_config.sample_rate
            )
            print(f"✅ Silero VAD引擎已初始化（阈值={vad_config.threshold}，采样率={vad_config.sample_rate}Hz）")
        except Exception as e:
            print(f"⚠️ Silero VAD引擎初始化失败: {e}，将使用RMS阈值降级方案")
            vad_engine = None
    else:
        print(f"⚠️ VAD引擎未启用或不是silero，将使用RMS阈值降级方案")
        vad_engine = None
    
    # 初始化ZMQ Context和Sockets
    zmq_context = zmq.Context()
    
    # ZMQ PUB Socket（数据流）
    zmq_pub_socket = zmq_context.socket(zmq.PUB)
    audio_pub_port = zmq_config.audio_pub_port
    zmq_pub_socket.bind(f"tcp://127.0.0.1:{audio_pub_port}")
    print(f"✅ ZMQ PUB Socket bound to tcp://127.0.0.1:{audio_pub_port}")
    
    # ZMQ REP Socket（控制流）
    zmq_rep_socket = zmq_context.socket(zmq.REP)
    audio_ctrl_port = zmq_config.audio_ctrl_port
    zmq_rep_socket.bind(f"tcp://127.0.0.1:{audio_ctrl_port}")
    zmq_rep_socket.setsockopt(zmq.RCVTIMEO, zmq_config.req_rep_timeout_ms)
    print(f"✅ ZMQ REP Socket bound to tcp://127.0.0.1:{audio_ctrl_port}")
    
    # 启动控制监听线程
    ctrl_thread = threading.Thread(target=control_listener_zmq, daemon=True)
    ctrl_thread.start()
    print(f"✅ 控制监听线程已启动")

    # ── 模型选择 ────────────────────────────────────────────────
    print("=" * 55)
    print("🎙️  WakeFusion Audio Service")
    print("=" * 55)
    # 🌟 修复：直接默认使用 OpenWakeWord 模型，不再提供选择
    use_oww = True
    print(f"使用唤醒词模型: OpenWakeWord CNN ({OWW_MODEL_PATH})")

    # ── 加载模型，生成统一的 infer(audio_float32) 闭包 ─────────
    # infer() 接受 float32 音频数组，返回 (label_str, confidence_float)
    if use_oww:
        import onnxruntime as ort
        from openwakeword.utils import AudioFeatures

        print(f"\n📦 正在加载 OpenWakeWord 模型: {OWW_MODEL_PATH} ...")
        oww_session = ort.InferenceSession(OWW_MODEL_PATH)
        oww_input_name = oww_session.get_inputs()[0].name
        oww_features = AudioFeatures(inference_framework="onnx")
        # active_threshold 已在第137行从配置读取，这里不需要重新赋值
        model_tag = f"OpenWakeWord CNN  (阈值 {active_threshold:.2f})"
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
        # active_threshold 已在第137行从配置读取，这里不需要重新赋值
        model_tag = f"NeMo MatchboxNet  (阈值 {active_threshold:.2f})"
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
    # 注意：control_listener_zmq 已在第156行启动，这里不需要重复启动
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
    print("🎙️ Audio Service 已启动 (ZMQ版本)")
    print(f"   模型: {model_tag}")
    print(f"   初始阈值: {active_threshold:.2f}")
    print(f"   推理步长: 每 {STEP_DURATION}s 一次")
    print(f"   连续确认: 需连续 {CONSECUTIVE_HITS_REQUIRED} 次命中")
    if vad_engine is not None:
        print(f"   VAD引擎: Silero VAD (阈值={vad_engine.get_threshold()})")
    else:
        print(f"   VAD引擎: RMS阈值 (已废弃，< {VAD_RMS_THRESHOLD} 跳过推理)")
    print(f"   音频桥接: 唤醒时回捞前 {RESCUE_SECONDS}s 音频")
    print(f"   ZMQ PUB: tcp://127.0.0.1:{audio_pub_port}")
    print(f"   ZMQ REP: tcp://127.0.0.1:{audio_ctrl_port}")
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
                # 使用Silero VAD进行智能检测（如果已初始化）
                if vad_engine is not None:
                    # 【修复】只取最后 0.2 秒送给VAD，保证RNN时间线连续，避免喂入重复数据
                    # 如果传入整个 2.0 秒的缓冲区，会导致重叠数据破坏RNN的时间感知
                    latest_chunk_samples = int(STEP_DURATION * SAMPLE_RATE)
                    latest_audio = current_audio[-latest_chunk_samples:]
                    latest_audio_int16 = (latest_audio * 32767).astype(np.int16)
                    
                    if not vad_engine.is_speech(latest_audio_int16):
                        consecutive_hits = 0  # 静音时重置连续计数
                        continue
                else:
                    # 降级方案：使用RMS阈值（向后兼容）
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

                        # Step 2: 发送唤醒事件（通过ZMQ PUB，使用Multipart Message）
                        wake_metadata = {
                            "vad": True,
                            "wake_word": {
                                "detected": True,
                                "keyword": "xiaokang",
                                "confidence": float(conf)
                            },
                            "timestamp": time.time()
                        }
                        # 发送一个空的音频帧作为唤醒标记（或发送一个特殊标记）
                        wake_audio = np.zeros(int(SAMPLE_RATE * 0.1), dtype=np.int16)
                        zmq_pub_socket.send_multipart([
                            json.dumps(wake_metadata).encode('utf-8'),
                            wake_audio.tobytes()
                        ], zmq.NOBLOCK)

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
                        if vad_engine is not None:
                            vad_engine.reset_states()  # 唤醒成功，推流前洗脑，清空杂音记忆
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
        finally:
            # 关闭ZMQ sockets
            try:
                if zmq_pub_socket:
                    zmq_pub_socket.close()
                if zmq_rep_socket:
                    zmq_rep_socket.close()
                if zmq_context:
                    zmq_context.term()
            except Exception:
                pass

if __name__ == "__main__":
    main()
