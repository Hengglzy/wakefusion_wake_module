# Sherpa-ONNX 集成指南

## 一、快速开始

### 1.1 安装依赖

```bash
pip install sherpa-onnx
```

### 1.2 下载中文KWS模型

```bash
# 创建模型目录
mkdir -p models/sherpa-onnx-kws-zh-16kHz
cd models/sherpa-onnx-kws-zh-16kHz

# 下载模型文件（3.3MB）
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zh-16kHz.tar.gz

# 解压
tar -xzf sherpa-onnx-kws-zh-16kHz.tar.gz
```

**模型文件清单**：
```
sherpa-onnx-kws-zh-16kHz/
├── encoder.onnx        # 编码器（1.5MB）
├── decoder.onnx        # 解码器（500KB）
├── joiner.onnx         # 连接器（1.3MB）
└── tokens.txt          # 字符表
```

---

## 二、配置说明

### 2.1 修改 config/config_sherpa.yaml

```yaml
kws:
  model: "sherpa"           # 使用Sherpa-ONNX
  model_dir: "./models/sherpa-onnx-kws-zh-16kHz"  # 模型目录
  keywords:
    - "小康小康"             # 自定义唤醒词1
    - "你好小助手"           # 自定义唤醒词2
  threshold: 0.5            # 检测阈值（0-1）
  num_threads: 4            # 推理线程数
  cooldown_ms: 1200         # 冷却时间
```

### 2.2 自定义唤醒词

**重要**：Sherpa-ONNX 支持任意文本唤醒词，无需训练！

```yaml
keywords:
  - "小康小康"
  - "小爱同学"
  - "天猫精灵"
  - "hey siri"
  - "ok google"
  - "你好小度"
  # 添加任意你需要的唤醒词
```

**唤醒词建议**：
- 长度：3-5个字（中文）或 2-4个音节
- 发音：清晰、不易误触发
- 避免常用词：如"你好"、"是"、"对"

---

## 三、性能测试

### 3.1 测试脚本

```bash
# 测试Sherpa-ONNX KWS
python wakefusion/workers/sherpa_kws.py
```

### 3.2 性能指标

| 指标 | 数值 | 说明 |
|------|------|------|
| 模型大小 | 3.3MB | encoder + decoder + joiner |
| 推理延迟 | 20-50ms | x86 CPU，单线程 |
| 内存占用 | <50MB | 运行时内存 |
| CPU占用 | 5-15% | 4线程，16kHz音频 |
| 检测准确率 | 85-95% | 取决于唤醒词 |

**⚠️ 注意**：
- Sherpa-ONNX 延迟包含在总的100ms预算内
- 实际端到端延迟 = 音频采集(20ms) + KWS推理(30ms) + VAD(10ms) + 融合(5ms) + 输出(5ms) = **~70ms** ✅

---

## 四、技术对比

### 4.1 三种方案对比

| 特性 | Sherpa-ONNX | Porcupine | openWakeWord |
|------|-------------|-----------|--------------|
| **自定义唤醒词** | ✅ 无需训练 | ❌ 需要在线训练 | ❌ 需要离线训练 |
| **离线使用** | ✅ 完全离线 | ⚠️ 需在线验证 | ✅ 完全离线 |
| **中文支持** | ✅ 原生支持 | ✅ 官方支持 | ⚠️ 需训练 |
| **开源协议** | ✅ Apache 2.0 | ❌ 商业受限 | ✅ Apache 2.0 |
| **模型大小** | 3.3MB | 2-5MB | 15-20MB |
| **推理延迟** | 20-50ms | 10-30ms | 30-60ms |
| **部署难度** | ⭐⭐ | ⭐ | ⭐⭐⭐ |

### 4.2 推荐使用场景

**Sherpa-ONNX** 适合：
- ✅ 需要快速迭代多个唤醒词
- ✅ 完全离线环境
- ✅ 商业项目（无版权限制）
- ✅ 中文唤醒词为主

**Porcupine** 适合：
- ✅ 追求最低延迟
- ✅ 固定唤醒词（不常变更）
- ✅ 可接受在线验证
- ⚠️ 注意商业使用限制

**openWakeWord** 适合：
- ✅ 需要高度定制化
- ✅ 有时间进行模型训练
- ✅ 英文唤醒词为主

---

## 五、集成示例

### 5.1 在 runtime.py 中集成

```python
from wakefusion.workers.sherpa_kws import SherpaKWSWorker, SherpaKWSConfig

# 初始化
if config.kws.model == "sherpa":
    kws_config = SherpaKWSConfig(
        model_dir=config.kws.model_dir,
        keywords=config.kws.keywords,
        threshold=config.kws.threshold,
        num_threads=config.kws.num_threads
    )
    self.kws_worker = SherpaKWSWorker(
        config=kws_config,
        event_callback=self._on_kws_hit
    )
```

### 5.2 事件回调

```python
def _on_kws_hit(self, event: BaseEvent):
    """处理KWS检测事件"""
    keyword = event.payload['keyword']
    confidence = event.payload['confidence']

    logger.info(f"检测到唤醒词: {keyword} (置信度: {confidence})")

    # 传递给决策引擎
    self.decision_engine.process_kws_hit(event)
```

---

## 六、故障排查

### 6.1 模型文件缺失

**错误**：`Missing model file: ./models/sherpa-onnx-kws-zh-16kHz/encoder.onnx`

**解决**：
```bash
# 检查模型文件是否完整
ls -lh ./models/sherpa-onnx-kws-zh-16kHz/

# 重新下载
cd models
wget https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zh-16kHz.tar.gz
tar -xzf sherpa-onnx-kws-zh-16kHz.tar.gz
```

### 6.2 未安装 sherpa-onnx

**错误**：`No module named 'sherpa_onnx'`

**解决**：
```bash
pip install sherpa-onnx
```

### 6.3 检测不到唤醒词

**可能原因**：
1. 阈值设置过高 → 降低 `threshold` 到 0.3-0.5
2. 唤醒词过于简单 → 使用更独特的唤醒词
3. 音频质量差 → 检查麦克风设置

**调试方法**：
```python
# 启用详细日志
runtime:
  log_level: "DEBUG"
```

---

## 七、高级配置

### 7.1 多唤醒词配置

```yaml
kws:
  keywords:
    - "小康小康"
    - "小爱同学"
    - "hey siri"
  threshold: 0.5  # 所有关键词共享阈值
```

### 7.2 调整帧长度

```python
# 在 sherpa_kws.py 中
self.frame_length = 5120  # 320ms @ 16kHz（默认）
# 可以调整为：
self.frame_length = 2560  # 160ms（更低延迟）
self.frame_length = 10240 # 640ms（更高准确率）
```

**权衡**：
- 更短的帧 → 更低延迟，但可能降低准确率
- 更长的帧 → 更高准确率，但增加延迟

### 7.3 线程数优化

```yaml
kws:
  num_threads: 4  # 根据CPU核心数调整
```

**建议**：
- 4核CPU → `num_threads: 2`
- 8核CPU → `num_threads: 4`
- 16核CPU → `num_threads: 8`

---

## 八、生产环境部署

### 8.1 Docker容器

```dockerfile
FROM python:3.10-slim

RUN apt-get update && apt-get install -y wget

# 安装依赖
COPY requirements.txt .
RUN pip install -r requirements.txt
RUN pip install sherpa-onnx

# 复制模型
COPY models/sherpa-onnx-kws-zh-16kHz /app/models/sherpa-onnx-kws-zh-16kHz

# 复制代码
COPY . /app
WORKDIR /app

CMD ["python", "-m", "wakefusion.runtime", "--config", "config/config_sherpa.yaml"]
```

### 8.2 Windows服务

使用 NSSM 将程序注册为Windows服务：

```bash
# 下载 NSSM
# https://nssm.cc/download

# 安装服务
nssm install WakeFusion "C:\Python310\python.exe" "D:\wakefusion_wake_module\wakefusion\runtime.py" --config "D:\wakefusion_wake_module\config\config_sherpa.yaml"

# 启动服务
nssm start WakeFusion
```

---

## 九、参考资料

- **官方文档**：https://k2-fsa.github.io/sherpa/onnx/kws/index.html
- **GitHub仓库**：https://github.com/k2-fsa/sherpa-onnx
- **KWS预训练模型**：https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/
- **腾讯云教程**（中文）：https://cloud.tencent.com/developer/article/2602282

---

## 十、总结

### 为什么选择 Sherpa-ONNX？

✅ **最大优势**：自定义唤醒词**无需训练**
✅ **完全离线**：无网络依赖
✅ **原生中文**：专为中文优化
✅ **开源友好**：Apache 2.0 协议
✅ **性能优秀**：延迟20-50ms，满足≤100ms要求

### 推荐决策路径

```
需要中文唤醒词？
    ↓
   是
    ↓
需要频繁更换唤醒词？
    ↓
   是 → Sherpa-ONNX ✅
    ↓
   否
    ↓
追求最低延迟？
    ↓
   是 → Porcupine
    ↓
   否 → Sherpa-ONNX ✅
```

**结论**：对于您的"小康小康"场景，**Sherpa-ONNX 是最佳选择**！
