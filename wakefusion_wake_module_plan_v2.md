# 展厅语音 + 图像唤醒模块技术方案（解耦版）
> 目标：在 **Windows 上位机**（展厅现场）实现“可交付、可长期运行”的 **唤醒/打断** 前端感知模块。  
> 本模块 **不做** ASR/LLM/TTS（由后续模块处理），只输出高层事件：`WAKE_CONFIRMED`、`BARGE_IN`、`SPEECH_START/END`、`PRESENCE` 等。  
> 适配硬件：**Femto Bolt + XVF3800（4麦）**。

---

## 0. 设计约束与结论
### 0.1 关键约束
- **音频链路必须“必过 XVF3800”**：采集设备绑定到 XVF3800；自动重连；不可默默切到系统默认麦。
- **打断（barge-in）必须实时**：数字人播报期间仍监听；触发后必须立即发出 `INTERRUPT` 事件给上层。
- **视觉门控（Vision-Gated）**：以音频为触发，视觉为校验（Audio-Triggered, Vision-Gated），避免视觉高频运行。
- **解耦**：本模块只负责感知与融合，不关心对话与TTS/ASR/模型调用实现。

### 0.2 推荐总体结构
- 主循环：`asyncio`（I/O与事件总线）
- 计算密集：KWS/VAD/人脸检测 → `ThreadPoolExecutor` 或独立进程（可选）
- 数据通道：音频/视觉各自维护 **ring buffer**（时间戳对齐）
- 输出：统一的 `EventBus`（本地）+ 一个对外 `Publisher`（WebSocket/gRPC/NamedPipe 任一）

---

## 1. 系统拓扑（System Topology）
```
┌─────────────────────────────── WakeFusion Runtime ───────────────────────────────┐
│                                                                                  │
│  ┌───────────────┐     frames      ┌─────────────────┐        events           │
│  │ AudioDriver    │ ─────────────▶ │ AudioRouter      │ ───────────────────┐    │
│  │ (XVF3800)       │               │ (ring buffer)    │                    │    │
│  └───────────────┘                 └─────────────────┘                    │    │
│             │                           │                                   │    │
│             │                           ├────────▶ VADWorker ───────────┐   │    │
│             │                           │                                │   │    │
│             │                           └────────▶ KWSWorker ───────────┼──▶│Event│
│             │                                                            │   │Bus  │
│  ┌───────────────┐     frames      ┌─────────────────┐                   │   │     │
│  │ CameraDriver   │ ─────────────▶ │ VisionRouter     │ ───────▶ FaceGate│   └──┬──┘
│  │ (Femto Bolt)   │               │ (frame cache)    │                   │      │
│  └───────────────┘                 └─────────────────┘                   │      │ publish
│                                                                            │      │
│                                                                            ▼      ▼
│                                                                         DecisionEngine
│                                                                              │
│                                                                              ▼
│                                                                         Publisher(out)
│                                                                  (WebSocket / gRPC / IPC)
└──────────────────────────────────────────────────────────────────────────────────────────┘
```
**说明**：
- `AudioRouter` 只做“搬运、对齐、缓存”，不做推理、不做日志（避免抖动）。
- `DecisionEngine` 实现融合策略：`KWS==True` 才触发视觉校验；支持降级策略。

---

## 2. 功能范围（Scope）
### 2.1 负责（IN）
- 音频采集（XVF3800绑定、重连、采样率统一、ring buffer）
- 语音端：VAD、KWS（唤醒词检测）、SPEECH_START/END 事件
- 视觉端：presence（是否有人）、face + depth gate（有效用户判定）
- 多模态融合：输出 `WAKE_CONFIRMED` / `WAKE_REJECTED`（可选）
- 打断：在“对外状态为 SPEAKING”时仍监听，KWS命中输出 `BARGE_IN`
- 观测指标：延迟、触发计数、帧率、丢帧、设备状态、噪声强度等

### 2.2 不负责（OUT）
- ASR（语音转文本）
- LLM（对话大模型）
- TTS（语音合成）
- 数字人动作/Unity控制（由上层接收事件后处理）

---

## 3. 核心机制设计

### 3.1 采样率与音频帧制式
- **推荐**：抓取设备原生采样率（常见 48k/16-bit/mono），在 `AudioRouter` 内部统一 **下采样到 16k** 供 KWS/VAD 使用。
- 音频帧建议使用固定帧长：**20ms**  
  - 16kHz：`320 samples/frame`
  - 48kHz：`960 samples/frame`（下采样后仍归一为 320 samples）
- ring buffer 长度：**2.0s**（100帧@20ms）

### 3.2 Ring Buffer（回捞）策略
- `AudioRouter` 维护：`deque[AudioFrame]`，每帧含：
  - `ts`（单调时钟时间戳）
  - `pcm16`（16k mono，bytes or np.int16）
  - `rms`、`peak`（可选）
- KWS 命中时：
  - 回捞窗口：`pre_roll = 800ms`（可配置）
  - 输出 `WakeContext`：包含回捞音频片段的起止 ts，用于后续模块拼接录音头部（避免丢首字）

### 3.3 Audio-Triggered, Vision-Gated（融合主导）
- KWS 命中 → 查询 `VisionRouter` 最近 `N=300ms` 内的视觉结果（人脸/深度/presence）
- 若视觉通过：输出 `WAKE_CONFIRMED`
- 若视觉缺失/低置信：进入 **Probation**（降级）策略：
  - 输出 `WAKE_PROBATION`（可选）或直接 `WAKE_CONFIRMED` with low_conf
  - 上层可选择开启短窗 ASR（不在本模块实现）做二次确认

### 3.4 打断（Barge-in）
- 上层（对话/渲染）会告诉本模块当前系统状态：`IDLE/LISTENING/SPEAKING`
- 当 `SPEAKING`：
  - VAD 仍运行（监听人声存在）
  - KWS 仍运行（更高阈值或更严格门控可选）
- KWS 命中并通过门控 → 输出 `BARGE_IN`（优先级最高）

---

## 4. 模块拆分与职责（Module Breakdown）
建议代码结构（供 Codex 生成）：
```
wakefusion/
  __init__.py
  config.py
  types.py
  logging.py
  metrics.py
  runtime.py

  drivers/
    audio_xvf3800.py
    camera_femto_bolt.py

  routers/
    audio_router.py
    vision_router.py

  workers/
    vad_worker.py
    kws_worker.py
    face_gate.py

  decision/
    decision_engine.py
    policy.py

  io/
    publisher_ws.py
    control_channel_ws.py   # 上层回写状态（SPEAKING等）
    health_server.py

  tests/
    test_ring_buffer.py
    test_policy.py
    test_event_schema.py
```

### 4.1 AudioDriver（XVF3800）
**职责**：
- 设备枚举、锁定 XVF3800（按设备名/VID:PID/序列号）
- 采集 PCM 流（优先 WASAPI Exclusive；不行则 Shared）
- 断连检测与自动重连
- 输出：`AudioFrameRaw`（原生采样率）到 `AudioRouter`

**注意**：不要在采集线程中做推理/日志。

### 4.2 CameraDriver（Femto Bolt）
**职责**：
- 获取 RGB + Depth（必要时对齐）
- 降频输出（例如 15FPS 足够）；或按需抓取（KWS触发时取最近帧）
- 输出：`VisionFrame`（含 ts、bbox、depth统计、presence）到 `VisionRouter`

### 4.3 Routers
- `AudioRouter`：下采样、ring buffer、广播给 KWS/VAD 消费
- `VisionRouter`：缓存最近 0.5~1.0s 视觉结果，用时间戳检索

### 4.4 Workers
- `VADWorker`：输出 `SPEECH_START/SPEECH_END`（可用于“有人讲话”提示、录音窗开启）
- `KWSWorker`：输出 `KWS_HIT`（携带 confidence、keyword、ts、回捞范围）
- `FaceGate`：将视觉信息转为门控信号：`VALID_USER`（distance阈值、ROI、朝向等）

### 4.5 DecisionEngine
输入：`KWS_HIT`、`VAD_*`、`VALID_USER`、`SYSTEM_STATE`  
输出：`WAKE_CONFIRMED`、`BARGE_IN`、`PRESENCE`、`NOISE_TOO_HIGH` 等。

---

## 5. 线程模型与执行模型（AsyncIO + Executors）
### 5.1 主循环（asyncio）
- `runtime.py` 负责启动各组件、管理生命周期、输出健康状态
- `EventBus` 建议用 `asyncio.Queue`（内部）

### 5.2 推理线程池
- KWS、VAD：`ThreadPoolExecutor(max_workers=2)`（或各1）
- 人脸/深度门控：`ThreadPoolExecutor(max_workers=1)` 或独立进程（可选）
- **原则**：推理慢了就丢帧，不允许阻塞音频搬运

---

## 6. 外部接口（完全解耦的契约）
### 6.1 上行事件（WakeFusion → 上层）
统一 JSON 事件：

#### Event 基础字段
```json
{
  "type": "WAKE_CONFIRMED",
  "ts": 1738450000.123,
  "session_id": "wf-20260201-0001",
  "priority": 100,
  "payload": {}
}
```

#### 关键事件类型
- `PRESENCE`：有人/无人、距离估计
- `SPEECH_START` / `SPEECH_END`：用于上层开启/关闭录音窗（但本模块不录音文件）
- `KWS_HIT`：可选（调试/观测）
- `WAKE_CONFIRMED`：唤醒确认（带 `pre_roll_audio_ms`、`confidence`、`vision_gate`）
- `BARGE_IN`：打断（优先级最高）
- `HEALTH`：设备状态、帧率、延迟、丢帧等

#### WAKE_CONFIRMED payload 建议
```json
{
  "keyword": "hey_assistant",
  "confidence": 0.87,
  "pre_roll_ms": 800,
  "audio": {
    "sample_rate": 16000,
    "channels": 1,
    "format": "pcm16",
    "pre_roll_ref": "rb:ts=...;dur=800" 
  },
  "vision": {
    "presence": true,
    "distance_m": 2.6,
    "face_conf": 0.76
  }
}
```
> `pre_roll_ref` 用“引用”而不是直接传大音频。需要传音频时可加一个 `AudioFetch` RPC/WS endpoint 拉取。

### 6.2 下行控制（上层 → WakeFusion）
- `SET_SYSTEM_STATE`：`IDLE/LISTENING/SPEAKING`
- `SET_POLICY`：动态调整阈值、门控策略（可选）
- `PING` / `GET_HEALTH`

示例：
```json
{ "type": "SET_SYSTEM_STATE", "payload": { "state": "SPEAKING" } }
```

### 6.3 传输方式建议
- 开发/联调：WebSocket（最简单）
- 生产/多进程：gRPC（更强的类型与流）
- 同机高可靠：NamedPipe（Windows）

本方案默认实现 WebSocket（`publisher_ws.py` + `control_channel_ws.py`）。

---

## 7. 策略（Policy）与默认参数
建议 `config.yaml`：
```yaml
audio:
  device_match: "XVF3800"
  capture_sample_rate: 48000
  work_sample_rate: 16000
  frame_ms: 20
  ring_buffer_sec: 2.0
  pre_roll_ms: 800

kws:
  model: "openwakeword"
  keyword: "hey_assistant"
  threshold: 0.55
  cooldown_ms: 1200

vad:
  enabled: true
  speech_start_ms: 120
  speech_end_ms: 500

vision:
  enabled: true
  gate_on_kws_only: true
  cache_ms: 600
  distance_m_max: 4.0
  face_conf_min: 0.55

fusion:
  probation_enabled: true
  probation_ms: 1000

runtime:
  health_interval_sec: 2
  log_level: "INFO"
```

---

## 8. 观测指标（Metrics & Logging）
必须提供的指标：
- `audio.capture_fps`、`audio.router_latency_ms_p95`、`audio.drop_frames`
- `kws.latency_ms_p95`、`kws.hit_count`
- `vad.speech_segments`
- `vision.fps`、`vision.latency_ms_p95`
- `fusion.wake_confirm_rate`、`fusion.false_reject_rate`（可通过标注回放统计）
- `device.reconnect_count`
- `noise.rms_dbfs`（用于噪声过高提示）

日志建议：结构化 JSON 日志，带 `session_id`、`event_id`、`ts`。

---

## 9. 可靠性与恢复策略（必须写进交付）
### 9.1 音频设备恢复
- 每 1s 检测采集心跳（帧时间戳增长）
- 异常：释放并重建音频流，尝试重新匹配 XVF3800
- 输出 `HEALTH` 中标记 `audio_status=DEGRADED/RECOVERING`

### 9.2 摄像头恢复
- 视觉模块可降级（vision off）：不因摄像头异常阻断音频唤醒
- 若 vision 不可用：`vision_gate=UNKNOWN`，走 probation 逻辑

### 9.3 过载保护
- 推理队列积压时：丢弃旧帧（只处理最新帧），并上报 `overload=true`
- KWS/VAD 的推理必须有超时（例如 100ms），超时丢弃本帧

---

## 10. 测试策略（可用性优先）
### 10.1 单元测试
- ring buffer 回捞正确性（pre-roll 拼接）
- policy 决策表（KWS hit + vision gate → WAKE CONFIRMED/REJECT）
- 事件 schema 校验（pydantic/jsonschema）

### 10.2 回放测试（强烈推荐）
- 录制真实展厅音频（含数字人播报、混响、噪声）与摄像头片段
- 离线回放驱动 `AudioRouter/VisionRouter`，评估：
  - 唤醒延迟
  - 误触发率（FAR）
  - 漏检率（FRR）

---

## 11. 实施建议（给 Codex 的任务拆分）
建议你在 Codex 里分 6 个子任务生成：
1) `types.py + config.py`：事件、帧、配置结构（pydantic）
2) `AudioDriver + AudioRouter`：设备匹配、帧制式、ring buffer
3) `KWSWorker + VADWorker`：接口先打通（可先 stub，再接 openWakeWord / webrtcvad）
4) `CameraDriver + VisionRouter + FaceGate`：先做 presence + distance gate（人脸可后置）
5) `DecisionEngine + Policy`：融合逻辑、barge-in、probation
6) `publisher_ws + control_channel_ws + runtime`：对外事件与状态回写、健康检查

---

## 12. 交付最小闭环（你现在就能跑通）
- 音频：XVF3800 → VAD + KWS → 输出 `WAKE_CONFIRMED/BARGE_IN`
- 视觉：Femto Bolt → presence + depth gate（先不做人脸）
- 对外：WebSocket 输出事件；从上层接收 `SET_SYSTEM_STATE`
- 上层（暂不实现）接到事件后可：
  - `WAKE_CONFIRMED` → 开启 ASR（后续模块）
  - `BARGE_IN` → 立刻打断 TTS/动作（后续模块）

---

## 13. 备注：关于“必过 XVF3800”的实现建议
- 优先使用 WASAPI 采集，并将设备匹配逻辑做成：
  1) 按设备友好名匹配（包含 `XVF3800`）
  2) 若有多个匹配，按设备 ID/序列号固定
  3) 启动时输出日志：最终绑定的设备 ID 与采样率
- 启动后每 2s 发一次 `HEALTH`，上报当前绑定设备与采样率，便于现场排障

---

## 14. 里程碑
- M1（1-2天）：音频链路 + KWS + VAD + WS输出（无视觉）
- M2（3-5天）：加入 Femto presence/depth gate + 融合策略
- M3（1周）：回放测试 + 指标看板 + 误触发/漏检调参

---

> 你可以把本 md 直接作为 Codex 的“实现规格文档”。  
> 如果你希望我再输出一份 `JSON Schema`（事件协议）或 `pydantic` 数据模型草案，我也可以基于此文档继续生成。



## 3.5 Ducking 与自声抑制（Ducking & Self-Voice Robustness）

### 3.5.1 Playback Ducking（播放链路 ducking）
当 VAD 检测到用户讲话时，本模块发出 DUCK_START/DUCK_END 事件，供上层降低 TTS 播放音量。

...

### 3.5.2 Self-Voice Robustness
在 SPEAKING 状态下提高 KWS/VAD 阈值，减少数字人自身声音触发误唤醒。

...

