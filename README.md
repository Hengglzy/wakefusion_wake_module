# WakeFusion 唤醒模块

展厅多模态唤醒/打断前端感知模块

## 功能特性

### Phase 1 (已完成) - 音频链路
- ✅ XVF3800 麦克风阵列音频采集
- ✅ KWS (Keyword Spotting) 唤醒词检测 (openWakeWord)
- ✅ VAD (Voice Activity Detection) 语音活动检测 (webrtcvad)
- ✅ Ring Buffer 音频回捞
- ✅ WebSocket 事件发布
- ✅ 健康检查和指标监控

### Phase 2 (已完成) - 视觉融合
- ✅ Femto Bolt 深度相机集成
- ✅ Presence 检测（基于深度数据）
- ✅ Depth Gate（距离门控）
- ✅ 多模态融合决策（Audio-Triggered, Vision-Gated）
- ✅ 视觉帧缓存与时间戳对齐
- ⏳ Face Detection（可选，需要MediaPipe）

## 系统要求

- Windows 10/11
- Python 3.10+
- XVF3800 麦克风阵列
- Femto Bolt 深度相机

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置

编辑 `config/config.yaml`，根据需要调整参数：

```yaml
audio:
  device_match: "XVF3800"
  capture_sample_rate: 48000
  work_sample_rate: 16000

kws:
  keyword: "hey_assistant"
  threshold: 0.55
```

### 3. 运行

```bash
python -m wakefusion.runtime
```

### 4. 测试

**健康检查**:
```bash
curl http://localhost:8080/health
```

**WebSocket连接**:
```bash
wscat -c ws://localhost:8765
```

## 事件协议

### WAKE_CONFIRMED

```json
{
  "type": "WAKE_CONFIRMED",
  "ts": 1738450000.123,
  "session_id": "wf-20260201-0001",
  "priority": 90,
  "payload": {
    "keyword": "hey_assistant",
    "confidence": 0.87,
    "pre_roll_ms": 800,
    "vision_gate": false,
    "vision_confidence": 0.0
  }
}
```

### BARGE_IN

```json
{
  "type": "BARGE_IN",
  "ts": 1738450000.123,
  "session_id": "wf-20260201-0001",
  "priority": 100,
  "payload": {
    "keyword": "hey_assistant",
    "confidence": 0.92,
    "pre_roll_ms": 800
  }
}
```

## 性能指标

| 指标 | 目标值 |
|------|--------|
| 端到端延迟 | ≤100ms |
| 音频采集FPS | 稳定50fps (20ms) |
| KWS检测延迟 | ≤80ms |
| 丢帧率 | <1% |
| CPU使用率 | <50% (单核) |

## 项目结构

```
wakefusion/
├── __init__.py
├── config.py          # 配置管理
├── types.py           # 数据模型
├── logging.py         # 日志系统
├── metrics.py         # 指标收集
├── runtime.py         # 主运行时
├── drivers/           # 硬件驱动
│   ├── audio_driver.py    # XVF3800音频驱动
│   └── camera_driver.py   # Femto Bolt相机驱动
├── routers/           # 数据路由
│   ├── audio_router.py    # 音频Ring Buffer
│   └── vision_router.py   # 视觉帧缓存
├── workers/           # 工作线程
│   ├── kws_worker.py      # KWS检测
│   ├── vad_worker.py      # VAD检测
│   └── face_gate.py       # 视觉门控
├── decision/          # 决策引擎
│   └── decision_engine.py # 多模态融合决策
└── io/                # 外部接口
    ├── publisher_ws.py
    └── health_server.py
```

## 开发计划

- [x] Phase 1.1: 音频链路 (XVF3800 + KWS + VAD + WebSocket)
- [ ] Phase 1.2: 性能测试与优化
- [x] Phase 2.1: Femto Bolt 视觉集成
- [x] Phase 2.2: 多模态融合决策
- [ ] Phase 3: 回放测试与参数调优
- [ ] Phase 4: 生产环境部署

## 故障排查

### XVF3800未检测到

1. 检查设备连接
2. 检查Windows声音设置
3. 调整 `device_match` 配置

### KWS误触发率高

1. 调整 `kws.threshold` (提高阈值)
2. 启用 VAD 联动
3. 启用视觉门控

### Femto Bolt未检测到

1. 检查USB连接
2. 安装SDK: `pip install pyorbbecsdk`
3. 运行测试: `python tests/test_vision.py`

### 视觉门控过于严格

1. 调整 `vision.distance_m_max` (增加最大距离)
2. 调整 `vision.face_conf_min` (降低置信度阈值)
3. 检查深度数据质量

## License

MIT
