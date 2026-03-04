# 语音测试顺序与预期结果

## 一、推荐测试顺序（从基础到高级）

### 阶段1：硬件与驱动层测试（必须）

#### 测试1：音频设备枚举

**命令：**
```bash
python -m tests.list_audio_devices
```

**预期结果：**
- 显示所有可用音频输入设备
- 找到 XVF3800 或默认麦克风设备
- 显示设备索引、名称、采样率

**成功标志：**
- 至少显示1个输入设备
- 设备名称包含 "XVF3800" 或显示为默认设备

**失败处理：**
- 检查 USB 连接
- 检查 Windows 声音设置
- 检查设备是否被其他应用占用

---

#### 测试2：音频驱动基础测试

**命令：**
```bash
python -m tests.test_audio_driver
```

该脚本包含4个子测试，按顺序执行：

##### 2.1 设备枚举测试 (test_audio_devices)

**预期结果：**
- 显示 "✓ 找到音频设备"
- 显示设备名称、索引、采样率、声道数
- 如果找到 XVF3800，显示 "✓ 成功识别 XVF3800 专业阵列麦克风"

**成功标志：** 找到至少1个设备

##### 2.2 实时音频采集测试 (test_audio_capture)

**预期结果：**
- 显示 "🎤 [SPEECH]" 或 "☁️ [SILENCE]" 状态
- RMS 能量值随声音变化（说话时 > 100，静音时 < 50）
- 采集完成后显示：
  - 总帧数 ≥ 200（5秒 × 50帧/秒）
  - 实际FPS ≈ 50.0

**成功标志：**
- 帧数 ≥ 200
- 说话时显示 [SPEECH]
- RMS 值有明显波动

**失败处理：**
- 检查麦克风权限
- 检查采样率是否支持
- 检查 webrtcvad 是否安装

##### 2.3 KWS模型测试 (test_kws_model)

**预期结果：**
- 显示 "✓ 模型加载成功!"
- 显示可用模型列表（如 "hey_assistant", "alexa" 等）
- 显示 "✓ 推理成功!"
- 显示预测结果（字典格式）

**成功标志：**
- 模型加载无错误
- 推理返回有效结果

**失败处理：**
- 检查 openwakeword 是否安装：`pip install openwakeword`
- 检查模型文件是否下载
- 检查 ONNX/TFLite 后端是否可用

##### 2.4 VAD模型测试 (test_vad_model)

**预期结果：**
- 显示 "✓ VAD加载成功!"
- 显示 "✓ 推理成功!"
- 显示检测结果（"语音" 或 "静音"）

**成功标志：**
- VAD 初始化无错误
- 推理返回布尔值

**失败处理：**
- 检查 webrtcvad 是否安装：`pip install webrtcvad`

---

### 阶段2：KWS 模型测试（重要）

#### 测试3：openWakeWord 文件测试

**命令：**
```bash
python -m tests.test_kws
```

**选择模式：**
- 模式1：生成测试音频文件（随机噪声）
- 模式2：使用音频文件测试（需要准备包含唤醒词的 WAV 文件）
- 模式3：直接测试（使用随机音频）

**预期结果（模式2，使用真实音频）：**
- 显示 "✅ KWS 检测到唤醒词!"
- 显示关键词名称、置信度（0.0-1.0）、时间戳
- 检测率 > 0%（如果有真实唤醒词音频）

**成功标志：**
- 使用真实唤醒词音频时，至少检测到1次
- 置信度 > 0.5

**失败处理：**
- 检查音频文件格式（16kHz, 16-bit, mono）
- 检查唤醒词是否清晰
- 尝试降低阈值

---

#### 测试4：MatchboxNet KWS 模拟测试

**命令：**
```bash
python tests/test_matchboxnet_kws.py
```

**预期结果：**
- 显示 "✅ KWS Worker 已启动"
- 显示支持的关键词列表（30个 Google Speech Commands）
- 显示处理帧数、丢帧数、平均延迟
- 使用随机噪声时，检测次数为 0（符合预期）

**成功标志：**
- 模型加载成功
- 处理帧数 > 0
- 平均延迟 < 100ms

**失败处理：**
- 检查 NeMo 是否安装：`pip install nemo-toolkit[asr]>=1.14.0`
- 检查模型是否下载（首次运行会自动下载）
- 检查 torch 版本：`pip install torch>=2.0.0`

---

#### 测试5：MatchboxNet KWS 麦克风实时测试

**命令：**
```bash
python tests/test_matchboxnet_microphone.py
```

**预期结果：**
- 显示支持的关键词列表
- 显示 "✅ 音频采集已启动"
- 对着麦克风说英文关键词（如 "yes", "no", "stop", "go"）
- 检测到时显示：
  - "✅ 检测到关键词: [关键词]"
  - 置信度（0.0-1.0）
  - 延迟（毫秒）

**成功标志：**
- 清晰说出关键词时，至少检测到1次
- 置信度 > 0.5
- 延迟 < 100ms

**失败处理：**
- 检查麦克风是否工作
- 检查背景噪声是否过大
- 尝试降低阈值（在代码中修改 threshold=0.3）
- 确保说英文关键词

---

### 阶段3：高级功能测试（待补充）

#### 测试6：RNNoise 降噪测试（缺失，需要创建）

**建议文件名：** `tests/test_rnnoise.py`

**测试内容：**
- RNNoise 服务初始化
- 降噪效果测试（SNR 提升）
- 性能测试（延迟、CPU 使用率）
- 降噪对 KWS 准确率的影响

**预期结果：**
- 降噪后 SNR 提升 3-5 dB
- 处理延迟 < 5ms
- CPU 使用率增加 < 5%

---

#### 测试7：AudioRouter 测试（缺失，需要创建）

**建议文件名：** `tests/test_audio_router.py`

**测试内容：**
- 下采样功能（48kHz → 16kHz）
- Ring Buffer 管理（容量、溢出处理）
- 订阅者通知机制
- 时间戳对齐

**预期结果：**
- 下采样后采样率 = 16000 Hz
- Ring Buffer 容量 = 100 帧（2秒 @ 20ms）
- 无丢帧或丢帧率 < 1%

---

#### 测试8：端到端集成测试（缺失，需要创建）

**建议文件名：** `tests/test_audio_integration.py`

**测试内容：**
- AudioDriver → AudioRouter → KWS/VAD 完整链路
- 事件流测试（KWS_HIT, SPEECH_START, SPEECH_END）
- 延迟测试（端到端延迟 < 100ms）

**预期结果：**
- 完整链路正常工作
- 事件正确触发
- 端到端延迟 < 100ms

---

#### 测试9：xiaokang.nemo 模型测试（缺失，需要创建）

**建议文件名：** `tests/test_xiaokang_nemo.py`

**测试内容：**
- 模型加载（自定义 .nemo 文件）
- 中文唤醒词检测
- 与 MatchboxNet 预训练模型对比

**预期结果：**
- 模型成功加载
- 中文唤醒词检测准确率 > 80%
- 延迟 < 100ms

---

## 二、测试执行检查清单

### 环境准备

- [ ] 激活 wakefusion 环境：`conda activate wakefusion`
- [ ] 确认 XVF3800 已连接（或使用默认麦克风）
- [ ] 确认依赖已安装：
  ```bash
  pip install webrtcvad openwakeword
  pip install nemo-toolkit[asr]>=1.14.0 torch>=2.0.0
  pip install pyrnnoise  # 如果测试 RNNoise
  ```

### 测试顺序执行

- [ ] `python tests/list_audio_devices.py` - 设备枚举
- [ ] `python tests/test_audio_driver.py` - 驱动测试（4个子测试）
- [ ] `python tests/test_kws.py` - openWakeWord 测试（可选）
- [ ] `python tests/test_matchboxnet_kws.py` - MatchboxNet 模拟测试
- [ ] `python tests/test_matchboxnet_microphone.py` - MatchboxNet 实时测试

### 预期整体结果

- 所有基础测试通过
- 麦克风采集正常（FPS ≈ 50）
- KWS 模型加载成功
- VAD 工作正常
- MatchboxNet 能检测到英文关键词

---

## 三、需要补充的测试文件

### 优先级1：核心功能测试

#### 1. RNNoise 降噪测试 (tests/test_rnnoise.py)

**测试项：**
- RNNoise 服务初始化（启用/禁用）
- 降噪效果（SNR 对比）
- 性能影响（延迟、CPU）
- 降噪对 KWS 准确率的影响

#### 2. AudioRouter 测试 (tests/test_audio_router.py)

**测试项：**
- 下采样功能（48kHz → 16kHz）
- Ring Buffer 管理（容量、溢出）
- 订阅者机制
- 时间戳对齐

#### 3. 端到端集成测试 (tests/test_audio_integration.py)

**测试项：**
- AudioDriver → AudioRouter → KWS/VAD 完整链路
- 事件流测试
- 延迟测试
- 稳定性测试（长时间运行）

### 优先级2：自定义模型测试

#### 4. xiaokang.nemo 模型测试 (tests/test_xiaokang_nemo.py)

**测试项：**
- 模型加载（自定义 .nemo 文件）
- 中文唤醒词检测
- 准确率测试
- 与预训练模型对比

---

## 四、测试结果记录模板

建议创建测试结果记录文件：

```markdown
# 语音测试结果记录

## 测试日期：2024-XX-XX
## 测试环境：wakefusion (conda)

### 阶段1：硬件与驱动层
- [x] 设备枚举：通过
- [x] 音频采集：通过（FPS: 50.2）
- [x] KWS模型：通过
- [x] VAD模型：通过

### 阶段2：KWS模型测试
- [x] openWakeWord：通过（置信度: 0.87）
- [x] MatchboxNet模拟：通过（延迟: 45ms）
- [x] MatchboxNet实时：通过（检测到 "yes"）

### 阶段3：高级功能（待测试）
- [ ] RNNoise降噪：待测试
- [ ] AudioRouter：待测试
- [ ] 端到端集成：待测试
- [ ] xiaokang.nemo：待测试
```

---

## 五、常见问题排查

| 问题 | 可能原因 | 解决方案 |
|------|---------|---------|
| 找不到设备 | USB 未连接或驱动问题 | 检查设备管理器，重新插拔 USB |
| 采集帧数不足 | 采样率不匹配 | 检查设备支持的采样率 |
| KWS 模型加载失败 | 依赖缺失 | 安装 openwakeword 或 nemo-toolkit |
| VAD 检测不准确 | 采样率不是 16kHz | 确保下采样到 16kHz |
| MatchboxNet 检测不到 | 阈值过高或背景噪声 | 降低阈值或改善环境 |

---

## 总结

先完成阶段1和阶段2的基础测试，确认硬件和模型正常；再补充阶段3的高级测试，完善测试覆盖。建议按顺序执行，每个测试通过后再进行下一个。
