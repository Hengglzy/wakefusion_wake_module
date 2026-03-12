# LLM Agent 接口文档（统一WebSocket协议）

## 接口总览

| 接口 | 协议 | 方向 | 用途 |
|------|------|------|------|
| **统一WebSocket** | WebSocket | WakeFusion ↔ LLM Agent | 双向通信：ASR结果、TTS文本、设备状态、打断控制 |
| Core Server 控制 | ZMQ REQ-REP | LLM → Core Server | 发送控制指令（可选） |

---

## 1. 统一WebSocket连接

### 1.1 连接地址

**设备侧（WakeFusion）作为客户端连接LLM Agent服务端**：

```
ws://<host>/api/voice/ws?deviceId=<deviceId>&token=<token>
```

生产环境建议使用：

```
wss://<host>/api/voice/ws?deviceId=<deviceId>&token=<token>
```

**连接参数**：
- `deviceId`: 设备唯一标识（必填）
- `token`: 认证令牌（必填）

**配置示例**（`config/config.yaml`）：
```yaml
llm_agent:
  host: "127.0.0.1:8080"  # LLM Agent服务地址
  device_id: "wakefusion-device-01"  # 设备标识
  token: "your-token-here"  # 认证令牌
  use_ssl: false  # 是否使用SSL（true for wss://, false for ws://）
  reconnect_interval_sec: 5.0  # 断线重连间隔（秒）
  ping_interval_sec: 30.0  # 保活ping间隔（秒）
```

### 1.2 设备隔离

- 每个设备通过唯一的 `deviceId` 标识
- 服务端必须按设备维度隔离会话上下文、流式输出、打断控制等
- 同一设备可重连，但同一时刻只能有一个活跃主连接

---

## 2. 上行消息（WakeFusion → LLM Agent）

### 2.1 ASR识别结果（asr）

设备上报语音识别结果（partial和final）。

**消息格式**：
```json
{
  "type": "asr",
  "stage": "partial",
  "text": "你好",
  "traceId": "7cc7d163-a970-4e97-a5f3-4fd43a4a4298",
  "deviceId": "wakefusion-device-01",
  "timestamp": 1730000000.123,
  "confidence": 0.95
}
```

**字段说明**：
- `type`: 固定为 `"asr"`
- `stage`: `"partial"`（中间结果）或 `"final"`（最终结果）
- `text`: 识别文本
- `traceId`: 请求唯一标识（UUID，每次进入LISTENING状态时生成）
- `deviceId`: 设备标识
- `timestamp`: 设备时间戳（秒）
- `confidence`: 置信度（0.0-1.0，可选）

**处理规则**：
- `stage = "partial"`: 服务端可记录状态，默认不触发LLM正式回答
- `stage = "final"`: 服务端必须发起一次正式回答流程

### 2.2 打断通知（interrupt）

设备通知服务端中断当前回答与播报。

**消息格式**：
```json
{
  "type": "interrupt",
  "traceId": "7cc7d163-a970-4e97-a5f3-4fd43a4a4298",
  "deviceId": "wakefusion-device-01",
  "reason": "barge-in",
  "timestamp": 1730000000.123
}
```

**字段说明**：
- `type`: 固定为 `"interrupt"`
- `traceId`: 被中断请求的标识
- `deviceId`: 设备标识
- `reason`: 打断原因（`"barge-in"`=硬打断，`"visual-cutoff"`=视觉斩断）
- `timestamp`: 设备时间戳（秒）

**触发场景**：
- **硬打断（barge-in）**：在SPEAKING状态下，用户说唤醒词触发硬打断
- **视觉斩断（visual-cutoff）**：在LISTENING或THINKING状态下，用户离开屏幕触发视觉强杀

### 2.3 设备状态上报（device_state）

设备自动上报当前状态。

**消息格式**：
```json
{
  "type": "device_state",
  "state": "listening",
  "deviceId": "wakefusion-device-01",
  "timestamp": 1730000000.123
}
```

**字段说明**：
- `type`: 固定为 `"device_state"`
- `state`: 设备状态（`"idle"`、`"listening"`、`"thinking"`、`"speaking"`）
- `deviceId`: 设备标识
- `timestamp`: 设备时间戳（秒）

**状态说明**：
- `idle`: 空闲状态，等待唤醒
- `listening`: 正在监听用户语音
- `thinking`: 已发送ASR final结果，等待LLM回复（数字人可做思考动作）
- `speaking`: 正在播放TTS音频

**状态转换时机**：
- `IDLE → LISTENING`: 检测到唤醒（视觉或音频）
- `LISTENING → THINKING`: 收到ASR final结果
- `THINKING → SPEAKING`: 收到LLM第一个token
- `SPEAKING → LISTENING`: TTS播放完成，进入持续对话

### 2.4 保活Ping（ping）

设备定期发送ping消息保持连接。

**消息格式**：
```json
{
  "type": "ping"
}
```

---

## 3. 下行消息（LLM Agent → WakeFusion）

### 3.1 TTS合成请求（route）

服务端发送文本进行语音合成。

**消息格式**：
```json
{
  "type": "route",
  "text": "今天天气很好",
  "isFinal": false,
  "traceId": "7cc7d163-a970-4e97-a5f3-4fd43a4a4298"
}
```

**字段说明**：
- `type`: 固定为 `"route"`
- `text`: 要合成的文本内容
- `isFinal`: `false`=流式文本块，`true`=最后一块文本
- `traceId`: 请求标识（可选，用于关联ASR请求）

**处理规则**：
- 设备接收文本后，通过ZMQ发送给TTS模块进行合成
- 流式文本会累积，直到收到 `isFinal=true` 才开始合成

### 3.2 停止TTS合成（stop_tts）

服务端请求停止当前TTS合成。

**消息格式**：
```json
{
  "type": "stop_tts"
}
```

### 3.3 错误消息（error）

服务端发送错误信息。

**消息格式**：
```json
{
  "type": "error",
  "message": "识别失败，请重试"
}
```

### 3.4 警告消息（warning）

服务端发送警告信息。

**消息格式**：
```json
{
  "type": "warning",
  "message": "识别置信度较低"
}
```

### 3.5 Ping响应（pong）

服务端响应ping消息。

**消息格式**：
```json
{
  "type": "pong"
}
```

---

## 4. traceId管理

### 4.1 生成规则

- **生成时机**：每次系统状态机进入 `LISTENING` 状态时生成新的 `traceId`
- **格式**：标准UUID（小写字符串）
- **生命周期**：从进入 `LISTENING` 到下一次进入 `LISTENING` 之前，所有partial和final结果共用同一个 `traceId`

### 4.2 使用规则

- 同一轮对话的所有ASR消息（partial和final）使用同一个 `traceId`
- interrupt消息必须携带被中断请求的 `traceId`
- 服务端必须基于 `traceId` 进行会话隔离和请求归属

---

## 5. 完整接入示例

### 5.1 LLM Agent服务端示例（Python）

```python
import asyncio
import websockets
import json
from typing import Dict, Set

class LLMAgentServer:
    def __init__(self, host: str = "127.0.0.1", port: int = 8080):
        self.host = host
        self.port = port
        self.devices: Dict[str, websockets.WebSocketServerProtocol] = {}
        self.device_sessions: Dict[str, str] = {}  # deviceId -> traceId
    
    async def handle_client(self, websocket, path):
        """处理客户端连接"""
        # 解析连接参数
        query_params = self._parse_query(path)
        device_id = query_params.get("deviceId")
        token = query_params.get("token")
        
        if not device_id or not token:
            await websocket.close(code=1008, reason="Missing deviceId or token")
            return
        
        # 验证token（示例）
        if not self._verify_token(device_id, token):
            await websocket.close(code=1008, reason="Invalid token")
            return
        
        self.devices[device_id] = websocket
        print(f"✅ 设备连接: {device_id}")
        
        try:
            async for message in websocket:
                data = json.loads(message)
                await self.handle_message(device_id, data)
        except websockets.exceptions.ConnectionClosed:
            pass
        finally:
            if device_id in self.devices:
                del self.devices[device_id]
            if device_id in self.device_sessions:
                del self.device_sessions[device_id]
            print(f"❌ 设备断开: {device_id}")
    
    async def handle_message(self, device_id: str, data: dict):
        """处理设备消息"""
        msg_type = data.get("type")
        
        if msg_type == "asr":
            # ASR识别结果
            stage = data.get("stage")
            text = data.get("text", "")
            trace_id = data.get("traceId")
            
            if trace_id:
                self.device_sessions[device_id] = trace_id
            
            if stage == "final":
                # 触发LLM生成回复
                reply = await self.generate_reply(text)
                # 发送TTS合成请求
                await self.send_tts(device_id, reply, trace_id)
        
        elif msg_type == "interrupt":
            # 打断通知
            trace_id = data.get("traceId")
            reason = data.get("reason", "unknown")
            print(f"🛑 设备 {device_id} 打断 (traceId={trace_id}, reason={reason})")
            # 停止当前TTS合成
            await self.send_stop_tts(device_id)
        
        elif msg_type == "device_state":
            # 设备状态上报
            state = data.get("state")
            print(f"📊 设备 {device_id} 状态: {state}")
        
        elif msg_type == "ping":
            # Ping响应
            await self.send_pong(device_id)
    
    async def generate_reply(self, user_text: str) -> str:
        """生成LLM回复（示例）"""
        # TODO: 接入你的大模型API
        return f"我理解您说的是：{user_text}"
    
    async def send_tts(self, device_id: str, text: str, trace_id: str = None):
        """发送TTS合成请求"""
        if device_id not in self.devices:
            return
        
        websocket = self.devices[device_id]
        
        # 流式发送文本（按句号切分）
        sentences = text.split('。')
        for i, sentence in enumerate(sentences):
            if sentence.strip():
                message = {
                    "type": "route",
                    "text": sentence + ('。' if i < len(sentences)-1 else ''),
                    "isFinal": (i == len(sentences) - 1)
                }
                if trace_id:
                    message["traceId"] = trace_id
                
                await websocket.send(json.dumps(message))
    
    async def send_stop_tts(self, device_id: str):
        """发送停止TTS信号"""
        if device_id not in self.devices:
            return
        
        websocket = self.devices[device_id]
        await websocket.send(json.dumps({"type": "stop_tts"}))
    
    async def send_pong(self, device_id: str):
        """发送Pong响应"""
        if device_id not in self.devices:
            return
        
        websocket = self.devices[device_id]
        await websocket.send(json.dumps({"type": "pong"}))
    
    def _parse_query(self, path: str) -> dict:
        """解析查询参数"""
        if "?" not in path:
            return {}
        query_string = path.split("?")[1]
        params = {}
        for param in query_string.split("&"):
            if "=" in param:
                key, value = param.split("=", 1)
                params[key] = value
        return params
    
    def _verify_token(self, device_id: str, token: str) -> bool:
        """验证token（示例）"""
        # TODO: 实现实际的token验证逻辑
        return True
    
    async def run(self):
        """启动服务器"""
        async with websockets.serve(
            self.handle_client,
            self.host,
            self.port
        ):
            print(f"🌐 LLM Agent服务器已启动: ws://{self.host}:{self.port}")
            await asyncio.Future()  # 永久运行

if __name__ == "__main__":
    server = LLMAgentServer()
    asyncio.run(server.run())
```

---

## 6. Core Server控制接口（可选）

**连接地址**：`tcp://127.0.0.1:5561`（ZMQ REQ-REP）

**消息格式**（LLM → Core Server）：
```json
{
  "command": "extend_window",
  "value": 15.0
}
```

**支持的命令**：
- `extend_window`: 延长免唤醒窗口（参数：`value` = 秒数，默认3.0秒，可延长到15.0秒）

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

## 7. 配置说明

### 7.1 WakeFusion配置（`config/config.yaml`）

```yaml
# LLM Agent配置
llm_agent:
  host: "127.0.0.1:8080"  # LLM Agent服务地址
  device_id: "wakefusion-device-01"  # 设备标识
  token: "your-token-here"  # 认证令牌
  use_ssl: false  # 是否使用SSL
  reconnect_interval_sec: 5.0  # 断线重连间隔（秒）
  ping_interval_sec: 30.0  # 保活ping间隔（秒）

# ZMQ配置
zmq:
  asr_result_push_port: 5562  # ASR推送识别结果给Core Server
  tts_text_pull_port: 5563  # TTS接收Core Server的合成文本
  core_control_rep_port: 5561  # Core Server控制端口
```

### 7.2 架构说明

**新的架构**：
- **WakeFusion（设备侧）**：作为WebSocket客户端，统一网关在Core Server
- **LLM Agent（服务端）**：作为WebSocket服务端，接收ASR结果，发送TTS文本
- **ASR模块**：通过ZMQ PUSH发送识别结果给Core Server
- **TTS模块**：通过ZMQ PULL接收Core Server的合成文本

**数据流**：
1. ASR识别结果：`ASR模块` → (ZMQ) → `Core Server` → (WebSocket) → `LLM Agent`
2. TTS合成文本：`LLM Agent` → (WebSocket) → `Core Server` → (ZMQ) → `TTS模块`
3. 设备状态：`Core Server` → (WebSocket) → `LLM Agent`
4. 打断通知：`Core Server` → (WebSocket) → `LLM Agent`

---

## 8. 注意事项

1. **连接管理**：
   - 设备侧会自动重连（间隔5秒）
   - 服务端应支持设备重连，保持会话上下文

2. **消息顺序**：
   - ASR partial消息可能乱序，服务端应基于timestamp排序
   - TTS文本按顺序发送，设备按顺序合成

3. **错误处理**：
   - 设备侧应处理WebSocket连接失败、消息发送失败等情况
   - 服务端应处理设备断开、消息解析失败等情况

4. **性能优化**：
   - 设备侧限制ZMQ队列积压（最多50个消息）
   - 服务端应支持多设备并发连接

5. **安全性**：
   - 生产环境必须使用SSL（wss://）
   - 实现token验证机制
   - 基于deviceId进行设备隔离

---

## 9. 协议参考

详细协议规范请参考：`unified-voice-ws-protocol.md`
