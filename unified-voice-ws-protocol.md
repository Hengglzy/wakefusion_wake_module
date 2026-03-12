# 统一语音 WebSocket 协议规范

## 1. 目标

本协议用于设备侧与 `industrial-assistant` 之间建立一条统一的双向 WebSocket 连接。

协议目标：

- 设备侧只维护一个统一客户端连接
- 服务端统一承接语音输入、流式文本输出、打断控制和设备状态
- 明确设备隔离模型，避免不同设备之间的会话串扰
- 为设备侧提供稳定的标准协议，便于封装 SDK 或统一客户端

本协议不考虑兼容旧版 `ASR/TTS/Core Server` 三通道模型。

---

## 2. 连接模型

### 2.1 WebSocket 地址

设备连接地址：

```text
ws://<host>/api/voice/ws?deviceId=<deviceId>&token=<token>
```

生产环境建议使用：

```text
wss://<host>/api/voice/ws?deviceId=<deviceId>&token=<token>
```

### 2.2 连接身份

每条连接必须唯一绑定一个设备身份。

必填连接参数：

- `deviceId`
- `token`

规则：

- 一个连接只代表一个设备
- 一个设备可重连，但同一时刻只能有一个活跃主连接
- 服务端必须基于 `deviceId` 做会话隔离和请求归属

### 2.3 设备隔离

服务端必须按设备维度隔离以下内容：

- 会话上下文
- 流式输出
- 打断控制
- 请求取消
- 审计日志
- 配额和限流

推荐映射：

- `userId = device:<deviceId>`
- `sessionKey = voice:<deviceId>`
- `context.channel = "voice-device"`
- `context.peerId = <deviceId>`

禁止：

- 不同设备复用同一个 `sessionKey`
- 不同设备共享同一个流式请求上下文

---

## 3. 消息总则

所有消息均为 JSON 文本帧。

每条消息必须包含：

- `type`: 消息类型

请求级消息建议包含：

- `traceId`: 一轮对话请求唯一标识
- `deviceId`: 设备标识

规范要求：

- `traceId` 由设备生成，推荐 UUID
- 同一轮请求的所有相关消息使用同一个 `traceId`
- 设备不得并发复用同一个 `traceId`

---

## 4. 上行消息

上行消息是“设备 -> 服务端”。

### 4.1 asr

设备上报语音识别结果。

```json
{
  "type": "asr",
  "stage": "partial",
  "text": "你好",
  "traceId": "7cc7d163-a970-4e97-a5f3-4fd43a4a4298",
  "deviceId": "dev-01",
  "timestamp": 1730000000.123,
  "confidence": 0.95
}
```

字段说明：

- `type`: 固定为 `"asr"`
- `stage`: `"partial"` 或 `"final"`
- `text`: 识别文本
- `traceId`: 请求标识
- `deviceId`: 设备标识
- `timestamp`: 设备时间戳，单位秒
- `confidence`: 可选，范围 `0.0-1.0`

处理规则：

- `stage = partial`
  - 服务端可记录状态
  - 默认不触发 LLM 正式回答
- `stage = final`
  - 服务端必须发起一次正式回答流程

### 4.2 interrupt

设备通知服务端中断当前回答与播报。

```json
{
  "type": "interrupt",
  "traceId": "7cc7d163-a970-4e97-a5f3-4fd43a4a4298",
  "deviceId": "dev-01",
  "reason": "barge-in"
}
```

字段说明：

- `type`: 固定为 `"interrupt"`
- `traceId`: 被中断请求的标识
- `deviceId`: 设备标识
- `reason`: 可选，例如 `"barge-in"`

处理规则：

- 服务端取消该 `traceId` 对应的流式回答
- 服务端向设备下发 `stop_tts`

### 4.3 device_state

设备上报自身状态。

```json
{
  "type": "device_state",
  "deviceId": "dev-01",
  "state": "listening",
  "timestamp": 1730000000.456
}
```

字段说明：

- `type`: 固定为 `"device_state"`
- `deviceId`: 设备标识
- `state`: 设备状态
- `timestamp`: 可选

允许的 `state`：

- `idle`
- `listening`
- `thinking`
- `speaking`
- `offline`

### 4.4 ping

设备主动保活。

```json
{
  "type": "ping",
  "deviceId": "dev-01",
  "timestamp": 1730000001.001
}
```

---

## 5. 下行消息

下行消息是“服务端 -> 设备”。

### 5.1 meta

服务端确认本轮请求已受理。

```json
{
  "type": "meta",
  "traceId": "7cc7d163-a970-4e97-a5f3-4fd43a4a4298",
  "deviceId": "dev-01",
  "sessionKey": "voice:dev-01",
  "agentId": "main",
  "tenantId": "default",
  "timestamp": "2026-03-11T18:00:00.000Z"
}
```

### 5.2 route

服务端告知本轮请求实际走的路径。

```json
{
  "type": "route",
  "traceId": "7cc7d163-a970-4e97-a5f3-4fd43a4a4298",
  "route": "fast",
  "reasons": ["preferred-fast-only"],
  "riskLevel": "low"
}
```

字段说明：

- `route`: `"fast"` 或 `"escalate"`
- `reasons`: 判定原因数组
- `riskLevel`: `"low" | "medium" | "high"`

### 5.3 token

服务端下发流式文本块。

```json
{
  "type": "token",
  "traceId": "7cc7d163-a970-4e97-a5f3-4fd43a4a4298",
  "text": "我在呢，请问有什么可以帮到你？"
}
```

字段说明：

- `type`: 固定为 `"token"`
- `traceId`: 请求标识
- `text`: 文本片段

设备侧要求：

- 必须按接收顺序拼接或直接送入本地 TTS
- 不得跨 `traceId` 混播

### 5.4 final

服务端告知本轮请求结束。

```json
{
  "type": "final",
  "traceId": "7cc7d163-a970-4e97-a5f3-4fd43a4a4298",
  "status": "ok",
  "route": "fast"
}
```

字段说明：

- `status`: `"ok" | "degraded" | "fallback"`
- `route`: `"fast"` 或 `"escalate"`

设备侧要求：

- 收到 `final` 后，当前请求结束
- 后续不得再将该 `traceId` 视为活跃播报请求

### 5.5 stop_tts

服务端通知设备立即停止当前播报。

```json
{
  "type": "stop_tts",
  "traceId": "7cc7d163-a970-4e97-a5f3-4fd43a4a4298",
  "reason": "interrupt"
}
```

触发场景：

- 用户打断
- 服务端取消当前回答
- 设备重发了新的最终语音

### 5.6 warning

服务端下发非致命告警。

```json
{
  "type": "warning",
  "traceId": "7cc7d163-a970-4e97-a5f3-4fd43a4a4298",
  "code": "fast_path_unavailable",
  "message": "Fast Path unavailable"
}
```

### 5.7 error

服务端下发致命错误。

```json
{
  "type": "error",
  "traceId": "7cc7d163-a970-4e97-a5f3-4fd43a4a4298",
  "code": "assistant_stream_failed",
  "message": "internal error"
}
```

### 5.8 pong

服务端响应保活。

```json
{
  "type": "pong",
  "deviceId": "dev-01",
  "timestamp": 1730000001.050
}
```

---

## 6. 服务端内部映射

当服务端收到：

```json
{
  "type": "asr",
  "stage": "final",
  "text": "...",
  "traceId": "...",
  "deviceId": "dev-01"
}
```

内部转换为：

```json
{
  "message": "...",
  "userId": "device:dev-01",
  "sessionKey": "voice:dev-01",
  "tenantId": "default",
  "agentId": "main",
  "traceId": "...",
  "context": {
    "channel": "voice-device",
    "peerId": "dev-01"
  },
  "options": {
    "preferredMode": "auto",
    "allowEscalation": true,
    "stream": true
  }
}
```

然后复用现有统一入口：

- `POST /api/assistant/stream`

---

## 7. 生命周期与状态机

建议设备侧状态机：

- `idle`
- `listening`
- `thinking`
- `speaking`

状态转换：

1. 设备已连接，待机为 `idle`
2. 开始监听用户语音时进入 `listening`
3. 发送 `asr.stage = final` 后进入 `thinking`
4. 收到首个 `token` 后进入 `speaking`
5. 收到 `final` 后回到 `idle`
6. 收到 `stop_tts` 后立即停止播放，并回到 `idle` 或 `listening`

---

## 8. 并发与幂等

### 8.1 单设备并发规则

建议每个设备同一时刻只保留一个活跃回答请求。

若设备发送了新的 `asr.stage = final`：

- 服务端应取消旧请求
- 服务端应下发旧请求对应的 `stop_tts`
- 新请求成为当前活跃请求

### 8.2 traceId 规则

- 一个 `traceId` 对应一轮完整回答
- 同设备不得并发复用同一 `traceId`
- `interrupt` 必须显式指向要取消的 `traceId`

---

## 9. 错误码建议

建议标准错误码：

- `unauthorized`
- `invalid_message`
- `invalid_device`
- `device_forbidden`
- `assistant_stream_failed`
- `request_cancelled`
- `rate_limited`
- `internal_error`

---

## 10. 设备侧客户端要求

设备侧开发应封装一个统一客户端，至少提供以下能力：

1. 建立并维持 WebSocket 连接
2. 发送 `asr partial/final`
3. 发送 `interrupt`
4. 接收 `token/final/stop_tts/error/warning`
5. 按 `traceId` 管理本地播报状态
6. 在断线后自动重连

推荐客户端接口：

```ts
connect(deviceId: string, token: string): Promise<void>
sendAsrPartial(traceId: string, text: string, confidence?: number): void
sendAsrFinal(traceId: string, text: string, confidence?: number): void
interrupt(traceId: string): void
onToken(handler: (traceId: string, text: string) => void): void
onFinal(handler: (traceId: string, status: string) => void): void
onStopTts(handler: (traceId: string) => void): void
onError(handler: (traceId: string, code: string, message: string) => void): void
close(): Promise<void>
```

---

## 11. 标准化边界

本协议中，以下字段为服务端硬占用字段，设备侧不得修改语义：

上行占用字段：

- `type`
- `stage`
- `text`
- `traceId`
- `deviceId`
- `timestamp`
- `confidence`
- `state`

下行占用字段：

- `type`
- `traceId`
- `deviceId`
- `sessionKey`
- `agentId`
- `tenantId`
- `route`
- `reasons`
- `riskLevel`
- `text`
- `status`
- `code`
- `message`
- `reason`
- `timestamp`

除上述字段外，后续如需增加设备自定义字段，应采用新增字段方式，不得修改既有字段语义。

---

## 12. 第一阶段实施范围

第一阶段只要求实现：

1. 设备建立统一 WS 连接
2. 上行 `asr.stage = final`
3. 下行 `meta / route / token / final / error`
4. 支持 `interrupt -> stop_tts`

第一阶段不要求：

1. `partial` 驱动增量回答
2. 多设备共享上下文
3. 复杂设备能力上报
4. 语音控制指令扩展

