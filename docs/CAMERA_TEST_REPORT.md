# Orbbec Gemini 330 系列相机测试报告

## 测试目标
将 WakeFusion 系统从 Femto Bolt 相机迁移到 Orbbec Gemini 330 系列（335/336）深度相机。

---

## 测试环境
- **硬件**: Orbbec Gemini 335
- **SDK**: pyorbbecsdk v2
- **系统**: Windows 10/11
- **Python**: 3.10+

---

## 测试过程与修改记录

### 1. 代码基础修复

#### 1.1 Metrics 模块统一
**问题**: 代码中混用 `metrics.increment()` 和 `metrics.increment_counter()`

**修改**: `wakefusion/metrics.py`
- 在 `MetricsCollector` 类中添加 `increment_counter()` 方法作为 `increment()` 的别名
- 保持向后兼容，无需修改其他文件

#### 1.2 测试脚本修复
**问题**: 
- 时间戳获取方式错误（使用 `asyncio.get_event_loop().time()`）
- `distance_m` 可能为 None 导致格式化错误

**修改**: `tests/test_vision.py`
- 导入 `time` 模块
- 将 `asyncio.get_event_loop().time()` 改为 `time.time()`
- 添加 None 判断：`f'{result.distance_m:.2f}m' if result.distance_m is not None else 'N/A'`

---

### 2. 硬件适配

#### 2.1 驱动类重命名
**修改**: `wakefusion/drivers/camera_driver.py`
- 类名：`FemtoBoltDriver` → `Gemini330Driver`
- 保留 `FemtoBoltDriver` 作为向后兼容别名
- 更新文档字符串

#### 2.2 默认分辨率调整
**修改**: `wakefusion/drivers/camera_driver.py` - `CameraConfig`
- RGB: `640x480` → `1280x800`
- Depth: `640x480` → `1280x800`
- FPS: 保持 `30fps`（测试时使用 `15fps`）

#### 2.3 DLL 路径配置
**修改**: `wakefusion/drivers/camera_driver.py`
- 添加 `os` 和 `sys` 导入
- 在文件顶部添加 DLL 搜索路径：
  ```python
  dll_path = r"D:\tools\cursor_project\Orbbec Gemini 335336\Orbbec Gemini 335336 SDK\bin"
  if os.path.exists(dll_path):
      os.add_dll_directory(dll_path)
  ```
  > 说明：以上为当前测试环境的真实路径，实际项目中请根据 pyorbbecsdk/Orbbec SDK 的安装位置进行调整。

---

### 3. SDK API 适配（pyorbbecsdk v2）

#### 3.1 设备名称获取
**问题**: `get_device_name()` 方法不存在

**修改**: `wakefusion/drivers/camera_driver.py`
- `self.device.get_device_name()` → `self.device.get_device_info().get_name()`
- 在 `start()` 和 `get_device_status()` 方法中统一修改

#### 3.2 流配置 API 重构
**问题**: 旧版 `pipeline.get_stream_config()` 在 v2 中不可用

**修改**: `wakefusion/drivers/camera_driver.py` - `start()` 方法

**旧版 API**:
```python
rgb_config = self.pipeline.get_stream_config(ob.VideoMode.RGB_VIDEO)
rgb_config.set_width(...)
rgb_config.set_height(...)
self.pipeline.start()
```

**新版 API**:
```python
config = ob.Config()
profile_list = self.pipeline.get_stream_profile_list(ob.OBSensorType.COLOR_SENSOR)
profile = self._find_best_profile(profile_list, width, height, format, fps)
config.enable_stream(profile)
self.pipeline.start(config)
```

#### 3.3 Profile 匹配逻辑
**新增**: `_find_best_profile()` 辅助方法
- 遍历所有可用 Profile
- 按格式、分辨率、帧率计算匹配分数
- 优先返回精确匹配，否则返回最佳近似匹配
- 如果找不到匹配，尝试使用默认 Profile

**关键特性**:
- 支持格式匹配：`ob.OBFormat.MJPG` (RGB，压缩流以节省 USB 带宽) / `ob.OBFormat.Y16` (Depth)
- 智能降级：精确匹配 → 最佳近似 → 默认配置
- 详细日志输出，便于调试

#### 3.4 深度对齐与彩色化增强
**修改**: `wakefusion/drivers/camera_driver.py` - `capture_frame()`
- 引入 `AlignFilter`，将深度流对齐到 RGB 流（`ob.OBStreamType.COLOR_STREAM`），修复坐标漂移问题，确保 Depth 像素与 RGB 像素一一对应。
- 对齐性能优化：
  - 使用 `time.perf_counter()` 监控两帧之间的时间间隔；若间隔过短（< 0.01s，疑似剧烈运动），本帧可跳过对齐并回退到上一帧成功对齐的备份。
  - 若单次 `AlignFilter.process()` 耗时超过 `0.05s`，同样回退使用上一帧备份，避免对齐导致的卡顿。
- 对齐结果验证：
  - 校验对齐后深度帧尺寸是否与 RGB 帧一致，不一致则视为失败并丢弃该对齐结果。
  - 检查深度数据是否全 0，全 0 视为失败；首次成功时打印有效像素比例日志，便于确认对齐质量。
- 深度彩色化：
  - 使用 OpenCV 对 `uint16` 深度图进行裁剪与归一化：固定范围 \[500mm, 4000mm]。
  - 通过 `cv2.applyColorMap(..., cv2.COLORMAP_JET)` 生成彩色深度图（与 Orbbec Viewer 视觉一致）：
    - 约 0.5m 为深蓝，1.0-1.5m 为青/绿，1.5-2.5m 为绿，2.5-3.5m 为黄，4.0m 为红。
  - 将最近一次有效的原始深度数组缓存在 `last_valid_depth_array` 中，用于在偶发空帧时做视觉兜底，减少深度画面闪烁。

#### 3.5 自动重连与采集循环
**新增**: `Gemini330Driver.run_with_reconnect()`
- 在后台异步循环中调用 `capture_frame()`，并根据帧获取情况自动维护 `camera.device_connected`、`camera.fps` 等指标。
- 当连续多次获取不到有效帧且超过 `max_reconnect_attempts` 时，进入 `RECONNECTING` 状态：
  - 调用 `stop()` 关闭当前 pipeline。
  - 等待 `reconnect_interval` 秒后重新调用 `start()`，自动完成重连。
- 在 `tests/test_vision.py` 和独立的 `test_camera_driver()` 中，统一通过 `asyncio.create_task(driver.run_with_reconnect())` 启动该循环，确保相机在测试过程中出现短暂抖动/断连时能够自动恢复。

---

### 4. 测试脚本更新

**修改**: `tests/test_vision.py`
- 导入：`FemtoBoltDriver` → `Gemini330Driver`，并显式使用 `CameraConfig`。
- 相机配置：
  - RGB/Depth 分辨率统一为 `1280x800`，帧率为 `15fps`（更适合展厅交互场景）。
  - 当前测试脚本中仅启用 Depth（`enable_rgb=False, enable_depth=True`），专注验证深度数据质量与距离范围。
- 采集方式：
  - 调用 `driver.start()` 启动硬件。
  - 通过 `asyncio.create_task(driver.run_with_reconnect())` 启动后台采集循环，确保在 10 秒测试期间持续产出帧，并在异常情况下自动重连。
- 输出内容：
  - 前 5 帧打印 `RGB/Depth` 的分辨率信息、Depth 原始取值范围，以及 0.5m–4.0m 内的“有效深度区间”。
  - 10 秒采集结束后，根据 `frame_count` 计算实际 FPS，并给出“采集正常/丢帧较高”的提示。
- 提示文字：文案统一更新为 "Orbbec Gemini 330系列" 并在开始时提醒需要 335/336 实际硬件支持。

---

## 测试结果

### ✅ 成功项
1. **设备识别**: 系统成功识别 "Orbbec Gemini 335"
2. **API 适配**: pyorbbecsdk v2 API 调用正常
3. **流配置**: Depth 流成功配置并启动
4. **代码兼容**: 向后兼容别名确保旧代码无需修改

### ⚠️ 注意事项
1. **分辨率匹配**: 需要确保目标分辨率在设备支持的 Profile 列表中
2. **格式要求**: 默认使用 `OBFormat.MJPG` 作为 RGB 压缩流（兼顾画质与 USB 带宽），Depth 使用 `OBFormat.Y16`；如需 `RGB888`，需确认设备 Profile 支持
3. **DLL 路径**: Windows 环境下需要正确配置 DLL 搜索路径

---

## 关键文件清单

| 文件 | 修改类型 | 说明 |
|------|---------|------|
| `wakefusion/metrics.py` | 新增方法 | 添加 `increment_counter()` 别名 |
| `wakefusion/drivers/camera_driver.py` | 重构 | 适配 pyorbbecsdk v2，重命名类，添加 Profile 匹配 |
| `wakefusion/drivers/__init__.py` | 更新导出 | 导出 `Gemini330Driver` |
| `tests/test_vision.py` | 修复+更新 | 修复时间戳和 None 判断，更新硬件配置 |

---

## 后续建议

1. **性能优化**: 根据实际使用场景调整帧率（建议 15fps 用于展厅交互）
2. **分辨率统一**: RGB 和 Depth 保持相同分辨率以获得更好的 D2C 对齐效果
3. **错误处理**: 增强 Profile 匹配失败时的降级策略
4. **文档更新**: 更新 README 和配置文档，说明 Gemini 330 系列支持

---

**测试日期**: 2024年
**测试人员**: AI Assistant
**硬件型号**: Orbbec Gemini 335
**SDK 版本**: pyorbbecsdk v2
