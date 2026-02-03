# MatchboxNet 中文唤醒词训练指南

本文档介绍如何为 MatchboxNet 训练中文唤醒词（如"小康小康"）。

## 📋 目录

1. [概述](#概述)
2. [环境准备](#环境准备)
3. [数据准备](#数据准备)
4. [训练流程](#训练流程)
5. [模型导出](#模型导出)
6. [集成到 WakeFusion](#集成到-wakefusion)
7. [常见问题](#常见问题)

---

## 概述

### 为什么需要训练？

MatchboxNet 的预训练模型是针对 **Google Speech Commands 数据集**（英文）训练的，支持 30 个英文关键词。要检测中文唤醒词"小康小康"，需要使用中文数据集进行**微调（Fine-tuning）**。

### 训练流程概述

```
数据准备 → 格式转换 → 配置训练脚本 → 微调模型 → 导出 ONNX → 集成部署
```

**估计时间**: 1-2 周
- 数据准备: 3-5 天
- 训练: 2-4 小时（取决于数据量和硬件）
- 测试调优: 2-3 天

---

## 环境准备

### 1. 安装依赖

```bash
# 基础依赖
pip install nemo-toolkit[asr]>=1.14.0
pip install torch>=2.0.0 torchaudio>=2.0.0
pip install librosa>=0.10.0 soundfile>=0.12.0
pip install pyyaml tqdm

# 可选：GPU 加速
# 如果有 NVIDIA GPU，安装 CUDA 版本的 PyTorch
# 访问 https://pytorch.org/get-started/locally/ 获取正确版本
```

### 2. 验证安装

```bash
python -c "import nemo; import torch; print('NeMo:', nemo.__version__); print('PyTorch:', torch.__version__); print('CUDA available:', torch.cuda.is_available())"
```

---

## 数据准备

### 方案A: 使用 TTS 生成（推荐）

**优点**: 快速、成本可控、发音标准
**缺点**: 可能不够自然

#### 步骤:

1. **选择 TTS 引擎**

   **选项1: Edge-TTS (免费，推荐)**

   ```bash
   pip install edge-tts
   ```

   **选项2: 百度 AI (付费)**

   ```bash
   pip install baidu-aip
   ```

   **选项3: Azure TTS (付费)**

   ```bash
   pip install azure-cognitiveservices-speech
   ```

2. **生成音频脚本**

   创建 `scripts/generate_tts_data.py`:

   ```python
   import edge_tts
   import asyncio
   from pathlib import Path

   async def generate_wake_word_audio(
       text: str,
       output_dir: str,
       variations: int = 100
   ):
       """
       生成唤醒词音频

       Args:
           text: 唤醒词文本（如"小康小康"）
           output_dir: 输出目录
           variations: 生成变体数量
       """
       output_path = Path(output_dir)
       output_path.mkdir(parents=True, exist_ok=True)

       # 可用语音列表（中文）
       voices = [
           'zh-CN-XiaoxiaoNeural',  # 女声
           'zh-CN-YunxiNeural',     # 男声
           'zh-CN-YunyangNeural',   # 男声
       ]

       print(f"生成 {variations} 个音频样本...")

       for i in range(variations):
           # 轮换语音
           voice = voices[i % len(voices)]

           # 生成音频
           communicate = edge_tts.Communicate(text, voice)

           output_file = output_path / f"xiaokang_{i:04d}.mp3"
           await communicate.save(str(output_file))

           if (i + 1) % 10 == 0:
               print(f"已生成 {i + 1}/{variations} 个文件")

       print(f"✅ 完成！音频保存在: {output_path}")

   # 运行
   if __name__ == "__main__":
       asyncio.run(generate_wake_word_audio(
           text="小康小康",
       output_dir="data/xiaokang_tts",
       variations=100
   ))
   ```

   运行脚本:

   ```bash
   python scripts/generate_tts_data.py
   ```

3. **添加负样本（可选但推荐）**

   生成一些非唤醒词的音频，帮助模型学习区分：

   ```python
   negative_samples = [
       "小猫", "小刚", "消毒", "小康",  # 相似词
       "你好", "早上好", "晚安",        # 问候语
   ]

   for text in negative_samples:
       asyncio.run(generate_wake_word_audio(
           text=text,
           output_dir=f"data/negative/{text}",
           variations=20
       ))
   ```

### 方案B: 人工录制（最自然）

**优点**: 最自然、真实场景
**缺点**: 成本高、耗时长

#### 步骤:

1. **准备录制环境**
   - 安静房间
   - 质量好的麦克风
   - 距离麦克风 0.5-1 米

2. **录制样本**
   - 录制 100-200 个"小康小康"样本
   - 包含不同：
     - 音调（高/中/低）
     - 语速（快/慢）
     - 强度（大声/小声）
     - 口音（如有多人）

3. **录制工具**

   使用 `scripts/record_wake_word.py`:

   ```python
   import sounddevice as sd
   import soundfile as sf
   from pathlib import Path

   def record_audio(
       filename: str,
       duration: float = 2.0,
       sample_rate: int = 16000
   ):
       """录制音频"""
       print(f"开始录制: {filename}")
       print(f"时长: {duration}秒")
       print("3...")
       import time
       time.sleep(1)
       print("2...")
       time.sleep(1)
       print("1...")
       time.sleep(1)
       print("录音中...")

       # 录制
       recording = sd.rec(
           int(duration * sample_rate),
           samplerate=sample_rate,
           channels=1
       )
       sd.wait()

       # 保存
       sf.write(filename, recording, sample_rate)
       print(f"✅ 保存到: {filename}")

   # 批量录制
   output_dir = Path("data/xiaokang_recorded")
   output_dir.mkdir(parents=True, exist_ok=True)

   for i in range(100):
       record_audio(str(output_dir / f"xiaokang_{i:04d}.wav"))
       print(f"进度: {i+1}/100")
       print("休息 2 秒...")
       import time
       time.sleep(2)
   ```

### 方案C: 混合方案（平衡）

- TTS 生成 80 个样本
- 人工录制 20 个样本
- 总计 100 个样本

---

### 数据格式转换

NeMo 需要特定的清单文件格式：

1. **创建目录结构**

```
data/
├── train/
│   ├── xiaokang/
│   │   ├── 0001.wav
│   │   ├── 0002.wav
│   │   └── ...
│   └── _background_noise_/
│       ├── noise1.wav
│       └── ...
└── manifest.json
```

2. **生成清单文件**

创建 `scripts/create_manifest.py`:

```python
import json
from pathlib import Path
import random

def create_manifest(
    data_dir: str,
    output_file: str,
    split: str = "train"
):
    """
    创建 NeMo 清单文件

    Args:
        data_dir: 数据目录
        output_file: 输出清单文件
        split: train/val/test
    """
    data_path = Path(data_dir)

    manifest = []

    # 遍历所有音频文件
    audio_files = list(data_path.glob("**/*.wav"))
    print(f"找到 {len(audio_files)} 个音频文件")

    for audio_file in audio_files:
        # 获取相对路径
        rel_path = audio_file.relative_to(data_path)

        # 从目录名推断标签
        label = audio_file.parent.name

        # 获取时长
        import soundfile as sf
        with sf.open(audio_file) as f:
            duration = len(f) / f.samplerate

        entry = {
            "audio_filepath": str(audio_file),
            "label": label,
            "duration": duration
        }

        manifest.append(entry)

    # 打乱顺序
    random.shuffle(manifest)

    # 划分训练集/验证集
    if split == "train":
        manifest = manifest[:int(len(manifest) * 0.8)]
    elif split == "val":
        manifest = manifest[int(len(manifest) * 0.8):int(len(manifest) * 0.9)]
    else:  # test
        manifest = manifest[int(len(manifest) * 0.9):]

    # 保存清单
    with open(output_file, 'w', encoding='utf-8') as f:
        for entry in manifest:
            f.write(json.dumps(entry, ensure_ascii=False) + '\n')

    print(f"✅ 清单文件已创建: {output_file}")
    print(f"   样本数: {len(manifest)}")

# 运行
create_manifest(
    data_dir="data/xiaokang_tts",
    output_file="data/manifest_train.json",
    split="train"
)

create_manifest(
    data_dir="data/xiaokang_tts",
    output_file="data/manifest_val.json",
    split="val"
)
```

---

## 训练流程

### 1. 创建训练配置

创建 `configs/matchboxnet_xiaokang.yaml`:

```yaml
# Model args
model:
  sample_rate: 16000
  labels:
    - _background_noise_
    - xiaokang
    - 小康
    - 小猫
    # 添加更多负样本标签

  # MatchboxNet 架构
  # MatchboxNet 3x1x64: 3 blocks, 1 conv per block, 64 channels
  kernel_size: 31
  strides: [2, 1, 2, 1, 2, 1, 2, 1]
  num_blocks: 3
  num_layers_per_block: 1
  channels: 64

# Training args
training:
  batch_size: 32
  num_epochs: 50
  lr: 0.001
  weight_decay: 0.0001

# Optimizer
optimizer:
  name: adam
  lr: 0.001

# Scheduler
scheduler:
  name: CosineAnnealing
  params:
    T_max: 50
    eta_min: 1e-6

# Data
data:
  sample_rate: 16000
  train_manifest: "data/manifest_train.json"
  val_manifest: "data/manifest_val.json"
  num_workers: 4
  pin_memory: true

# Augmentation（可选）
augmentation:
  speed_perturbation: true
  speed_perturbation_rates: [0.9, 1.0, 1.1]
  gain_perturbation: true
  gain_perturbation_range: [-5, 5]
```

### 2. 训练脚本

创建 `scripts/train_matchboxnet.py`:

```python
import torch
from nemo.collections.asr.models import EncDecClassificationModel
from nemo.core.config import HydraConfig
from omegaconf import OmegaConf
import pytorch_lightning as pl
from pytorch_lightning.callbacks import ModelCheckpoint

# 加载配置
cfg = OmegaConf.load("configs/matchboxnet_xiaokang.yaml")

# 创建模型
model = EncDecClassificationModel(cfg=cfg.model)

# 设置训练参数
model.set_trainer(
    max_epochs=cfg.training.num_epochs,
    accelerator='gpu' if torch.cuda.is_available() else 'cpu',
    devices=1,
    callbacks=[
        ModelCheckpoint(
            monitor='val_accuracy',
            mode='max',
            save_top_k=3,
            dirpath='checkpoints/',
            filename='matchboxnet-xiaokang-{epoch:02d}-{val_accuracy:.3f}'
        )
    ]
)

# 加载预训练权重（可选，加快收敛）
pretrained_model = EncDecClassificationModel.from_pretrained(
    model_name='commandrecognition_en_matchboxnet3x1x64_v1'
)

# 迁移学习（保留底层特征提取器）
model.model.encoder = pretrained_model.model.encoder

# 修改最后一层以适应新的类别数
# 注意：需要重新初始化分类器

# 开始训练
print("🚀 开始训练 MatchboxNet...")
model.fit()

# 保存最终模型
model.save_to('checkpoints/matchboxnet_xiaokang.nemo')
print(f"✅ 模型已保存到: checkpoints/matchboxnet_xiaokang.nemo")
```

### 3. 运行训练

```bash
python scripts/train_matchboxnet.py
```

**预期输出**:

```
🚀 开始训练 MatchboxNet...
GPU available: True, using: NVIDIA GeForce RTX 3090
LOCAL_RANK: 0 - CUDA_VISIBLE_DEVICES: [0]

Epoch 0: 100%|██████████| 50/50 [00:15<00:00,  3.27it/s]
  train_loss: 2.123 - train_accuracy: 0.450
  val_loss: 1.876 - val_accuracy: 0.520

Epoch 1: 100%|██████████| 50/50 [00:14<00:00,  3.45it/s]
  train_loss: 1.654 - train_accuracy: 0.680
  val_loss: 1.234 - val_accuracy: 0.750

...

Epoch 49: 100%|██████████| 50/50 [00:14<00:00,  3.51it/s]
  train_loss: 0.123 - train_accuracy: 0.980
  val_loss: 0.234 - val_accuracy: 0.920

✅ 模型已保存到: checkpoints/matchboxnet_xiaokang.nemo
```

---

## 模型导出

### 导出为 ONNX（推荐用于生产）

创建 `scripts/export_onnx.py`:

```python
import torch
from nemo.collections.asr.models import EncDecClassificationModel

# 加载训练好的模型
model = EncDecClassificationModel.restore_from(
    "checkpoints/matchboxnet_xiaokang.nemo"
)
model.eval()

# 准备示例输入
dummy_input = torch.randn(1, 16000)  # 1 秒音频 @ 16kHz

# 导出 ONNX
torch.onnx.export(
    model,
    dummy_input,
    "checkpoints/matchboxnet_xiaokang.onnx",
    input_names=['audio_input'],
    output_names=['logits'],
    dynamic_axes={
        'audio_input': {0: 'batch_size', 1: 'time'},
        'logits': {0: 'batch_size'}
    },
    opset_version=14
)

print("✅ ONNX 模型已导出: checkpoints/matchboxnet_xiaokang.onnx")
```

运行导出:

```bash
python scripts/export_onnx.py
```

---

## 集成到 WakeFusion

### 1. 更新配置

编辑 `config/config.yaml`:

```yaml
kws:
  enabled: true
  engine: "matchboxnet"
  model: "matchboxnet"
  model_name: "local"  # 使用本地模型
  model_path: "checkpoints/matchboxnet_xiaokang.onnx"  # 本地模型路径
  device: "cpu"
  keyword: "xiaokang"  # 内部标签名
  threshold: 0.5  # 可能需要调整
  cooldown_ms: 1200
```

### 2. 测试

```bash
# 使用麦克风测试
python tests/test_matchboxnet_microphone.py

# 运行完整系统
python -m wakefusion.runtime
```

---

## 常见问题

### Q1: 训练需要多长时间？

**A**:
- **CPU**: ~6-8 小时（50 epochs）
- **GPU (RTX 3090)**: ~1-2 小时
- **GPU (GTX 1660)**: ~3-4 小时

### Q2: 需要多少训练样本？

**A**:
- **最少**: 50 个正样本 + 50 个负样本
- **推荐**: 100-200 个正样本 + 200-500 个负样本
- **最佳**: 500+ 个正样本 + 1000+ 个负样本

### Q3: 如何提高准确率？

**A**:
1. **增加数据量**: 更多样本 = 更好泛化
2. **数据增强**: 添加噪声、变速、变调
3. **调整阈值**: 降低/提高 `threshold`
4. **调整架构**: 增加 `num_blocks` 或 `channels`
5. **混合数据**: TTS + 人工录制

### Q4: 检测延迟是多少？

**A**:
- **CPU (i7)**: ~20-30ms
- **GPU (RTX 3090)**: ~10-15ms
- **边缘设备 (Jetson)**: ~30-50ms

### Q5: 如何支持多个唤醒词？

**A**: 在训练时添加多个类别：

```yaml
model:
  labels:
    - _background_noise_
    - xiaokang  # 小康小康
    - xiaoming  # 小明小明
    - xiaohong  # 小红小红
```

### Q6: 内存不足怎么办？

**A**:
1. 减小 `batch_size` (32 → 16 → 8)
2. 减小模型规模 (channels: 64 → 32)
3. 使用混合精度训练 `fp16=True`

---

## 📚 参考资料

- [NeMo 官方教程 - Speech Commands](https://github.com/NVIDIA/NeMo/blob/main/tutorials/asr/Speech_Commands.ipynb)
- [MatchboxNet 论文](https://arxiv.org/abs/2004.08531)
- [Google Speech Commands 数据集](https://www.kaggle.com/datasets/carlthome/google-speech-commands)
- [Edge-TTS 文档](https://github.com/rany2/edge-tts)

---

## ✅ 快速检查清单

训练前:
- [ ] 安装 NeMo 和 PyTorch
- [ ] 准备训练数据（TTS 或录制）
- [ ] 创建清单文件
- [ ] 创建训练配置

训练中:
- [ ] 监控 loss 和 accuracy
- [ ] 检查过拟合（train >> val）
- [ ] 保存最佳 checkpoint

训练后:
- [ ] 导出 ONNX 模型
- [ ] 测试模型性能
- [ ] 集成到 WakeFusion
- [ ] 调整阈值

---

**祝训练顺利！** 🚀

如有问题，请参考 NeMo 官方文档或提交 Issue。
