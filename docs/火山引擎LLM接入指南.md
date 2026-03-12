# 火山引擎LLM接入指南

## 概述

本项目已集成火山引擎豆包模型（Doubao Seed 2.0 Mini），可以通过统一的WebSocket协议与WakeFusion系统通信。

## 架构说明

```
WakeFusion Core Server (客户端)
    ↓ WebSocket (统一协议)
火山引擎LLM Agent (服务端)
    ↓ HTTP API
火山引擎豆包模型 API
```

## 配置说明

### 1. 配置文件 (`config/config.yaml`)

在 `llm_agent` 配置块中，已添加火山引擎API配置：

```yaml
llm_agent:
  host: "127.0.0.1:8080"  # LLM Agent服务地址
  device_id: "wakefusion-device-01"  # 设备标识
  token: "your-token-here"  # 认证令牌
  
  # 火山引擎API配置
  volcano_api_url: "https://ark.cn-beijing.volces.com/api/v3/responses"
  volcano_api_key: "13855083-26c0-423a-9720-f0d02f9e7668"  # 你的API密钥
  volcano_model: "doubao-seed-2-0-mini-260215"  # 模型名称
```

### 2. 修改API密钥

**重要**：请将 `volcano_api_key` 替换为你自己的API密钥！

从你提供的curl命令中提取的密钥：
```bash
-H "Authorization: Bearer 13855083-26c0-423a-9720-f0d02f9e7668"
```

如果你的密钥不同，请在配置文件中更新。

## 启动步骤

### 1. 安装依赖

确保已安装 `aiohttp` 库（用于HTTP请求）：

```bash
# 在 wakefusion 环境中
conda activate wakefusion
pip install aiohttp
```

### 2. 启动火山引擎LLM Agent

```bash
# 在 wakefusion 环境中
conda activate wakefusion
python -m wakefusion.services.llm_agent_volcano
```

**预期输出**：
```
2026-03-12 15:00:00 - llm_agent_volcano - INFO - 🌋 火山引擎LLM Agent初始化完成
2026-03-12 15:00:00 - llm_agent_volcano - INFO -   API地址: https://ark.cn-beijing.volces.com/api/v3/responses
2026-03-12 15:00:00 - llm_agent_volcano - INFO -   模型: doubao-seed-2-0-mini-260215
2026-03-12 15:00:00 - llm_agent_volcano - INFO - 🚀 启动火山引擎LLM Agent服务...
2026-03-12 15:00:00 - llm_agent_volcano - INFO -   WebSocket地址: ws://127.0.0.1:8080/api/voice/ws
2026-03-12 15:00:00 - llm_agent_volcano - INFO - ✅ 火山引擎LLM Agent服务已启动
2026-03-12 15:00:00 - llm_agent_volcano - INFO -   等待设备连接...
```

### 3. 启动WakeFusion系统

使用一键启动脚本 `launcher.ps1`，或手动启动各个服务：

1. Vision Service
2. Audio Service
3. ASR Service
4. TTS Service
5. **Core Server**（会自动连接到LLM Agent）

## 工作流程

1. **用户说话** → Audio Service 检测到语音
2. **ASR识别** → ASR Service 识别文本
3. **发送ASR结果** → Core Server 通过WebSocket发送 `asr` 消息（`stage=final`）
4. **LLM处理** → 火山引擎LLM Agent 接收ASR结果，调用火山引擎API
5. **生成回答** → 火山引擎API返回回答文本
6. **流式返回** → LLM Agent 通过WebSocket发送 `route` 消息（流式文本块）
7. **TTS合成** → Core Server 接收文本，发送给TTS Service合成语音
8. **播放语音** → 通过XVF3800播放回答

## 协议消息示例

### 上行消息（WakeFusion → LLM Agent）

**ASR最终结果**：
```json
{
  "type": "asr",
  "stage": "final",
  "text": "你好，今天天气怎么样？",
  "traceId": "7cc7d163-a970-4e97-a5f3-4fd43a4a4298",
  "deviceId": "wakefusion-device-01",
  "timestamp": 1730000000.123,
  "confidence": 0.95
}
```

**中断请求**：
```json
{
  "type": "interrupt",
  "traceId": "7cc7d163-a970-4e97-a5f3-4fd43a4a4298",
  "deviceId": "wakefusion-device-01",
  "reason": "barge-in"
}
```

### 下行消息（LLM Agent → WakeFusion）

**流式文本块**：
```json
{
  "type": "route",
  "text": "今天天气很好，",
  "isFinal": false,
  "traceId": "7cc7d163-a970-4e97-a5f3-4fd43a4a4298"
}
```

**最终文本块**：
```json
{
  "type": "route",
  "text": "适合出门。",
  "isFinal": true,
  "traceId": "7cc7d163-a970-4e97-a5f3-4fd43a4a4298"
}
```

**请求结束**：
```json
{
  "type": "final",
  "traceId": "7cc7d163-a970-4e97-a5f3-4fd43a4a4298",
  "status": "ok",
  "route": "fast"
}
```

## 故障排查

### 1. 连接失败

**问题**：Core Server 无法连接到LLM Agent

**检查**：
- LLM Agent 是否已启动？
- 端口是否正确（默认8080）？
- 防火墙是否阻止连接？

**日志位置**：
- Core Server: 查看 `core_server` 日志中的WebSocket连接错误
- LLM Agent: 查看 `llm_agent_volcano` 日志

### 2. API调用失败

**问题**：火山引擎API返回错误

**检查**：
- API密钥是否正确？
- API地址是否正确？
- 网络连接是否正常？
- API配额是否用完？

**日志位置**：
- LLM Agent日志中会显示API错误详情

### 3. 响应解析失败

**问题**：无法解析火山引擎API响应

**检查**：
- 查看LLM Agent日志中的完整API响应
- 根据实际响应格式调整 `_call_volcano_api` 方法中的解析逻辑

## 与Mock LLM Server的区别

| 特性 | Mock LLM Server | 火山引擎LLM Agent |
|------|----------------|-------------------|
| 功能 | 中继服务器，需要手动输入回答 | 自动调用API生成回答 |
| 适用场景 | 测试、调试 | 生产环境 |
| 依赖 | 无 | 需要火山引擎API密钥 |
| 响应速度 | 即时（手动） | 取决于API响应时间 |

## 下一步

1. **测试API响应格式**：首次运行时，检查日志中的API响应格式，如有需要可调整解析逻辑
2. **优化流式输出**：当前实现是按标点符号切分，可以优化为更智能的切分策略
3. **添加会话上下文**：可以在 `device_sessions` 中维护对话历史，实现多轮对话
4. **错误重试机制**：添加API调用失败时的重试逻辑

## 注意事项

1. **API密钥安全**：请勿将API密钥提交到Git仓库，建议使用环境变量或配置文件（不纳入版本控制）
2. **API配额**：注意火山引擎API的调用配额限制
3. **响应延迟**：API调用需要网络请求，会有一定延迟，建议优化用户体验
4. **错误处理**：当前实现已包含基本的错误处理，但可以根据实际需求进一步完善
