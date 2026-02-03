# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 项目概述

**WakeFusion 唤醒模块** - 一个 Windows 上位机多模态唤醒/打断感知系统，用于展厅数字人交互场景。

**核心特点**:
- 多模态融合: 音频唤醒词检测(KWS) + 视觉人脸/深度验证(Vision-Gated)
- 实时打断: 在数字人播报期间持续监听并响应打断请求
- 硬件绑定: 必须使用 XVF3800(4麦阵列) + Femto Bolt(深度相机)
- 解耦设计: 只输出高层事件，不处理 ASR/LLM/TTS

**技术栈**:
- Python + asyncio (I/O与事件总线)
- WASAPI (Windows音频API, XVF3800设备绑定)
- openWakeWord (KWS) + webrtcVAD (VAD)
- WebSocket (事件发布与控制通道)
- pydantic (数据模型验证)

---

## 系统架构

### 数据流与拓扑

```
AudioDriver (XVF3800) → AudioRouter (ring buffer) → VADWorker/KWSWorker ─┐
                                                                      ├──▶ EventBus ─▶ DecisionEngine ─▶ Publisher
CameraDriver (Femto) → VisionRouter (frame cache) → FaceGate ───────────┘
```

**核心设计原则**:
1. **Audio-Triggered, Vision-Gated**: KWS命中才触发视觉验证,节省计算资源
2. **Ring Buffer回捞**: KWS命中时可回捞800ms预录音(避免丢失唤醒词首字)
3. **非阻塞推理**: 推理慢了就丢帧,不允许阻塞音频搬运线程
4. **模块化解耦**: 驱动/路由/工作线程/决策/IO 各司其职

### 关键组件职责

| 模块 | 职责 | 关键约束 |
|------|------|----------|
| `AudioDriver` | XVF3800设备绑定/采集/自动重连 | 必须锁定XVF3800,不可默默切到默认麦 |
| `CameraDriver` | Femto Bolt RGB+Depth采集 | 降频输出(15FPS)或按需抓取 |
| `AudioRouter` | 下采样到16k/维护2s ring buffer | 不做推理/不做日志(避免抖动) |
| `VisionRouter` | 缓存最近0.5-1s视觉结果 | 支持时间戳检索 |
| `VADWorker` | 语音活动检测 | 输出SPEECH_START/END |
| `KWSWorker` | 唤醒词检测 | 输出KWS_HIT(含置信度/回捞范围) |
| `FaceGate` | 视觉门控(距离/人脸/朝向) | 输出VALID_USER信号 |
| `DecisionEngine` | 多模态融合决策 | 支持probation降级策略 |
| `Publisher` | WebSocket事件发布 | 标准化JSON事件协议 |

---

## 开发命令

### 运行与测试
```bash
# 安装依赖(根据最终requirements.txt)
pip install pydantic numpy asyncio aiohttp websockets webrtcvad openwakeword

# 运行主程序
python -m wakefusion.runtime

# 运行单元测试
python -m pytest tests/

# 回放测试(基于录制数据)
python -m wakefusion.tests.replay_test --data path/to/recorded_data

# 健康检查
curl http://localhost:8080/health
```

### 代码生成建议顺序(根据方案文档第11节)
1. `types.py + config.py` - 事件/帧/配置结构
2. `AudioDriver + AudioRouter` - 设备匹配/帧制式/ring buffer
3. `KWSWorker + VADWorker` - 接口先打通(可stub,再接真实模型)
4. `CameraDriver + VisionRouter + FaceGate` - 先做presence+distance gate
5. `DecisionEngine + Policy` - 融合逻辑/barge-in/probation
6. `publisher_ws + control_channel_ws + runtime` - 对外事件与状态回写

---

## 关键配置参数

### 音频制式
- **采集采样率**: 48kHz (XVF3800原生)
- **工作采样率**: 16kHz (下采样后供KWS/VAD)
- **帧长**: 20ms (`320 samples @ 16kHz`)
- **Ring buffer**: 2.0s (100帧)
- **Pre-roll**: 800ms (KWS命中时回捞)

### 策略阈值(可配置)
```yaml
kws.threshold: 0.55          # KWS置信度阈值
kws.cooldown_ms: 1200        # KWS冷却期
vad.speech_start_ms: 120     # VAD语音起始阈值
vision.distance_m_max: 4.0   # 最大检测距离
vision.gate_on_kws_only: true # 仅KWS命中时启动视觉验证
```

---

## 事件协议

### 上行事件 (WakeFusion → 上层)

所有事件遵循统一格式:
```json
{
  "type": "WAKE_CONFIRMED",
  "ts": 1738450000.123,
  "session_id": "wf-20260201-0001",
  "priority": 100,
  "payload": {}
}
```

**关键事件类型**:
- `WAKE_CONFIRMED`: 唤醒确认(带pre_roll_ms/confidence/vision_gate)
- `BARGE_IN`: 打断事件(优先级最高,数字人播报时触发)
- `SPEECH_START/END`: 语音活动(用于上层开启/关闭录音窗)
- `PRESENCE`: 有人/无人/距离估计
- `HEALTH`: 设备状态/帧率/延迟/丢帧指标

### 下行控制 (上层 → WakeFusion)
- `SET_SYSTEM_STATE`: `IDLE/LISTENING/SPEAKING`
- `SET_POLICY`: 动态调整阈值/门控策略
- `PING/GET_HEALTH`: 健康检查

---

## 可靠性与恢复策略

### 设备恢复(必须实现)
- **音频**: 每1s检测心跳,异常时释放并重建流,重新匹配XVF3800
- **视觉**: 可降级(vision off),不因摄像头异常阻断音频唤醒
- **过载保护**: 推理队列积压时丢弃旧帧(只处理最新帧),上报overload=true

### Probation降级策略
当KWS命中但视觉验证缺失/低置信度时:
- 输出 `WAKE_PROBATION` 或 `WAKE_CONFIRMED` with low_conf
- 上层可选择开启短窗ASR做二次确认

---

## 里程碑与交付

- **M1** (1-2天): 音频链路 + KWS + VAD + WS输出(无视觉)
- **M2** (3-5天): Femto presence/depth gate + 融合策略
- **M3** (1周): 回放测试 + 指标看板 + 调参

**最小闭环**:
- XVF3800 → VAD + KWS → WAKE_CONFIRMED/BARGE_IN
- Femto Bolt → presence + depth gate
- WebSocket输出事件 + 接收SET_SYSTEM_STATE

---

## 重要注意事项

1. **音频链路必须"必过 XVF3800"**: 启动时输出日志:最终绑定的设备ID与采样率
2. **打断必须实时**: SPEAKING状态下VAD/KWS仍运行,KWS命中立即输出INTERRUPT事件
3. **视觉门控为主**: 避免视觉高频运行,以音频为触发
4. **观测性**: 每个关键组件必须输出指标(延迟/帧率/丢帧/设备状态)
5. **结构化日志**: 日志带session_id/event_id/ts,便于排障

---

## 参考资料

- 技术方案全文: `wakefusion_wake_module_plan_v2.md`
- 该文档为中文,涵盖详细的架构设计/接口协议/测试策略
