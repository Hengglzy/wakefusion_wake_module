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

**快速测试**:
```bash
# 运行快速测试工具
python scripts/quick_test.py
```

**详细测试手册**:
请参考 [测试手册](docs/TESTING_MANUAL.md) 了解完整的测试流程。

**模块独立测试**:
```bash
# 视觉模块测试
python wakefusion/tests/test_vision_sub.py

# 音频数据流测试
python wakefusion/tests/test_audio_sub.py

# 音频控制测试
python wakefusion/tests/test_audio_ctrl.py
```

**端到端集成测试**:
```bash
# 需要4个终端窗口，分别运行：
# 终端1: 视觉模块
conda activate wakefusion_vision
python -m wakefusion.services.vision_service --fps 15

# 终端2: 音频模块
conda activate wakefusion
python -m wakefusion.services.audio_service

# 终端3: Mock ASR
python wakefusion/tests/mock_asr_saver.py

# 终端4: Core Server
python -m wakefusion.services.core_server
```

**健康检查** (旧版):
```bash
curl http://localhost:8080/health
```

**WebSocket连接** (旧版):
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
├── core_server.py     # 核心决策模块 (ZMQ版本)
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
├── services/          # 服务模块
│   ├── audio_service.py   # 音频服务 (ZMQ版本)
│   └── vision_service.py  # 视觉服务 (ZMQ版本)
├── decision/          # 决策引擎
│   └── decision_engine.py # 多模态融合决策
├── tests/             # 测试脚本
│   ├── test_vision_sub.py  # 视觉模块测试
│   ├── test_audio_sub.py   # 音频数据流测试
│   ├── test_audio_ctrl.py  # 音频控制测试
│   └── mock_asr_saver.py   # Mock ASR服务
├── io/                # 外部接口
│   ├── publisher_ws.py
│   └── health_server.py
└── docs/              # 文档
    └── TESTING_MANUAL.md   # 测试手册
```

## 开发计划

- [x] Phase 1.1: 音频链路 (XVF3800 + KWS + VAD + WebSocket)
- [ ] Phase 1.2: 性能测试与优化
- [x] Phase 2.1: Femto Bolt 视觉集成
- [x] Phase 2.2: 多模态融合决策
- [x] Phase 2.3: ZMQ架构重构（前四步）
  - [x] 统一配置文件
  - [x] 视觉模块ZMQ改造
  - [x] 音频模块ZMQ改造
  - [x] 核心决策模块创建
- [ ] Phase 3: ASR和TTS模块集成
- [ ] Phase 4: 回放测试与参数调优
- [ ] Phase 5: 生产环境部署

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


为什么建议彻底卸载 MediaPipe？
是的，建议卸载。 * 解除封印：MediaPipe 强制要求的 protobuf<4 就像一个枷锁，导致你的 nemo-toolkit（语音模块）和 tensorflow（模型推理）处于随时可能崩溃的亚健康状态。

开发影响：卸载 MediaPipe 不会对后续开发产生负面影响。因为在你的 wakefusion 架构中，FaceGateWorker 是一个独立的组件，只要我们换个底层“引擎”但保持输出格式（返回人脸框和置信度）不变，系统的其他部分（如决策引擎）完全感知不到变化。

替代方案一：OpenCV YuNet（最推荐，零依赖）
这是目前避开依赖冲突的“银弹”。YuNet 是一个极其轻量且高性能的人脸检测模型。

原理：它是一个 .onnx 格式的模型文件。它不需要安装任何新的 Python 库，直接利用你已经安装好的 opencv-python 的 dnn 模块就能运行。

对比 MediaPipe：

性能：在 CPU 上的速度甚至快于 MediaPipe。

功能：同样支持人脸框（Bounding Box）和 5 个关键点（眼睛、鼻子、嘴角）。

兼容性：因为它是模型文件而不是库，所以完全没有 Protobuf 版本冲突。

实现思路：下载一个 2MB 的 face_detection_yunet.onnx 文件放在 models/ 目录下，在 face_gate.py 中用 cv2.FaceDetectorYN.create() 调用即可。

替代方案二：多进程/微服务隔离（最专业）
如果你非常喜欢 MediaPipe 的 API，或者未来需要它更复杂的功能（如手势识别），那么“隔离”是唯一出路。

原理：为 MediaPipe 单独创建一个 Conda 环境（比如叫 env_vision）。主程序运行在 wakefusion 环境中，两个环境通过 Socket 或 ZMQ 进行通信。

优点：

版本自由：视觉环境可以用 Protobuf 3.x，语音环境可以用 Protobuf 5.x，互不干扰。

系统稳定性：即使视觉模块因为相机掉线崩了，也不会拖累语音唤醒模块。