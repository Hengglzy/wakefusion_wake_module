# LLM Agent 接口文档

## 接口总览

| 接口 | 协议 | 端口 | 方向 | 用途 |
|------|------|------|------|------|
| ASR WebSocket | WebSocket | **8766** | ASR → LLM | 接收语音识别结果 |
| TTS WebSocket | WebSocket | **8767** | LLM → TTS | 发送文本进行语音合成 |
| Core Server 控制 | ZMQ REQ-REP | **5561** | LLM → Core Server | 发送控制指令 |

---

## 1. ASR WebSocket 接口（接收识别结果）

**连接地址**：`ws://127.0.0.1:8766`

**消息格式**（ASR → LLM）：
```json
{
  "type": "asr_result",
  "text": "你好，今天天气怎么样",
  "is_final": false,
  "confidence": 0.95,
  "timestamp": 1234567890.123
}
```

**字段说明**：
- `type`: 固定为 `"asr_result"`
- `text`: 识别出的文本内容
- `is_final`: `false`=中间结果（流式），`true`=最终结果（用于处理）
- `confidence`: 置信度（0.0-1.0）
- `timestamp`: Unix 时间戳（秒）

**Python 示例**：
```python
import asyncio
import websockets
import json

async def connect_asr():
    uri = "ws://127.0.0.1:8766"
    async with websockets.connect(uri) as websocket:
        async for message in websocket:
            data = json.loads(message)
            if data["type"] == "asr_result" and data["is_final"]:
                user_text = data["text"]
                # 调用LLM生成回复
                reply = await generate_reply(user_text)
                # 发送到TTS
                await send_tts(reply)

asyncio.run(connect_asr())
```

---

## 2. TTS WebSocket 接口（发送合成文本）

**连接地址**：`ws://127.0.0.1:8767`

**消息格式**（LLM → TTS）：

流式文本请求：
```json
{
  "type": "tts_request",
  "text": "今天天气很好",
  "is_final": false
}
```

停止合成信号：
```json
{
  "type": "stop_synthesis"
}
```

**字段说明**：
- `type`: `"tts_request"` 或 `"stop_synthesis"`
- `text`: 要合成的文本内容
- `is_final`: `false`=流式文本块，`true`=最后一块文本

**Python 示例**：
```python
async def send_tts(text: str):
    uri = "ws://127.0.0.1:8767"
    async with websockets.connect(uri) as websocket:
        # 流式发送
        sentences = text.split('。')
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                await websocket.send(json.dumps({
                    "type": "tts_request",
                    "text": sentence + ('。' if i < len(sentences)-1 else ''),
                    "is_final": (i == len(sentences) - 1)
                }))
```

---

## 3. Core Server 控制接口（可选）

**连接地址**：`tcp://127.0.0.1:5561`（ZMQ REQ-REP）

**消息格式**（LLM → Core Server）：
```json
{
  "command": "extend_window",
  "value": 15.0
}
```

**支持的命令**：
- `extend_window`: 延长免唤醒窗口（参数：`value` = 秒数，默认2.0秒，可延长到15.0秒）

**Python 示例**：
```python
import zmq
import json

context = zmq.Context()
req_socket = context.socket(zmq.REQ)
req_socket.connect("tcp://127.0.0.1:5561")

# 延长窗口到15秒
request = {"command": "extend_window", "value": 15.0}
req_socket.send_json(request)
response = req_socket.recv_json()
# {"status": "ok", "new_timeout": 15.0}
```

---

## 完整接入示例

```python
import asyncio
import websockets
import json

class LLMAgent:
    def __init__(self):
        self.asr_ws = None
        self.tts_ws = None
    
    async def connect(self):
        self.asr_ws = await websockets.connect("ws://127.0.0.1:8766")
        self.tts_ws = await websockets.connect("ws://127.0.0.1:8767")
        print("✅ 已连接ASR和TTS")
    
    async def listen_asr(self):
        async for message in self.asr_ws:
            data = json.loads(message)
            if data["type"] == "asr_result" and data["is_final"]:
                user_text = data["text"]
                reply = await self.generate_reply(user_text)  # 调用你的LLM
                await self.send_tts(reply)
    
    async def generate_reply(self, user_text: str) -> str:
        # TODO: 接入你的大模型API
        return f"我理解您说的是：{user_text}"
    
    async def send_tts(self, text: str):
        sentences = text.split('。')
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                await self.tts_ws.send(json.dumps({
                    "type": "tts_request",
                    "text": sentence + ('。' if i < len(sentences)-1 else ''),
                    "is_final": (i == len(sentences) - 1)
                }))
    
    async def run(self):
        await self.connect()
        await self.listen_asr()

if __name__ == "__main__":
    agent = LLMAgent()
    asyncio.run(agent.run())
```

---

## 配置说明

端口配置在 `config/config.yaml`：
```yaml
websocket:
  asr_port: 8766
  tts_port: 8767
zmq:
  core_control_rep_port: 5561
```
