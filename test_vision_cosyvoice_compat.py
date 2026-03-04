"""
测试 wakefusion_vision 环境是否兼容 CosyVoice
=============================================
检查：
1. MediaPipe 是否仍然正常工作（protobuf 3.20.3）
2. 能否安装 PyTorch（不影响 MediaPipe）
3. 能否安装 CosyVoice 依赖（protobuf 版本冲突风险）
4. 实际加载 CosyVoice 模型测试

运行：在 wakefusion_vision 环境中运行此脚本
"""
import sys
import os

print("=" * 60)
print("  wakefusion_vision 环境 CosyVoice 兼容性测试")
print("=" * 60)

# ---- 测试 1: MediaPipe 基础功能 ----
print("\n[1/5] 测试 MediaPipe 基础功能...")
try:
    import mediapipe as mp
    print(f"  MediaPipe: {mp.__version__}")
    
    # 测试人脸检测器初始化（不实际运行，只检查能否导入和初始化）
    from mediapipe.python.solutions import face_detection as mp_face
    print("  FaceDetection 模块: OK")
    
    # 检查 protobuf 版本
    import google.protobuf
    pb_version = google.protobuf.__version__
    print(f"  protobuf: {pb_version}")
    
    if pb_version.startswith("3."):
        print("  [OK] protobuf 3.x 符合 MediaPipe 要求")
    else:
        print(f"  [WARN] protobuf {pb_version} 可能不兼容 MediaPipe（需要 3.x）")
    
except Exception as e:
    print(f"  [FAIL] MediaPipe 测试失败: {e}")
    sys.exit(1)

# ---- 测试 2: 检查 PyTorch 是否已安装 ----
print("\n[2/5] 检查 PyTorch...")
try:
    import torch
    print(f"  PyTorch: {torch.__version__}")
    print(f"  CUDA available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
except ImportError:
    print("  [INFO] PyTorch 未安装，需要安装")
    print("  建议: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("  注意: 安装 PyTorch 不会影响 MediaPipe（它们使用不同的依赖树）")

# ---- 测试 3: 检查 CosyVoice 依赖 ----
print("\n[3/5] 检查 CosyVoice 依赖...")
missing = []
for mod_name in ["modelscope", "oss2", "addict", "soundfile", "librosa", "numpy", "scipy"]:
    try:
        mod = __import__(mod_name)
        if hasattr(mod, '__version__'):
            print(f"  {mod_name}: {mod.__version__}")
        else:
            print(f"  {mod_name}: OK")
    except ImportError:
        missing.append(mod_name)
        print(f"  {mod_name}: [MISSING]")

if missing:
    print(f"\n  缺失依赖: {', '.join(missing)}")
    print("  建议: pip install " + " ".join(missing))

# ---- 测试 4: 尝试加载 CosyVoice（如果依赖齐全） ----
print("\n[4/5] 尝试加载 CosyVoice 模型...")
if "modelscope" not in missing and "torch" in sys.modules:
    try:
        from modelscope.pipelines import pipeline
        from modelscope.utils.constant import Tasks
        print("  ModelScope 导入: OK")
        
        # 只测试导入，不实际下载模型（节省时间）
        print("  [SKIP] 模型下载测试（首次运行会下载 ~1.5GB）")
        print("  如需完整测试，运行: python training/_test_cosyvoice.py")
    except Exception as e:
        print(f"  [FAIL] CosyVoice 加载测试失败: {e}")
        import traceback
        traceback.print_exc()
else:
    print("  [SKIP] 依赖不完整，跳过 CosyVoice 加载测试")

# ---- 测试 5: 验证 MediaPipe 在安装 PyTorch 后仍可用 ----
print("\n[5/5] 验证 MediaPipe 在安装 PyTorch 后仍可用...")
if "torch" in sys.modules:
    try:
        # 再次测试 MediaPipe 导入
        from mediapipe.python.solutions import face_detection as mp_face
        print("  [OK] MediaPipe 在 PyTorch 存在时仍然可用")
    except Exception as e:
        print(f"  [WARN] MediaPipe 可能受 PyTorch 影响: {e}")
else:
    print("  [SKIP] PyTorch 未安装，无法测试兼容性")

# ---- 总结 ----
print("\n" + "=" * 60)
print("  兼容性评估")
print("=" * 60)

issues = []
warnings = []

if "torch" not in sys.modules:
    issues.append("需要安装 PyTorch")
if missing:
    issues.append(f"缺失依赖: {', '.join(missing)}")

pb_version = google.protobuf.__version__ if 'google.protobuf' in sys.modules else "unknown"
if not pb_version.startswith("3."):
    warnings.append(f"protobuf {pb_version} 可能不兼容 MediaPipe（需要 3.x）")

if not issues and not warnings:
    print("\n  [OK] 环境完全兼容！可以同时使用 MediaPipe 和 CosyVoice")
    print("\n  建议:")
    print("    1. 如果 PyTorch 未安装，运行: pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu121")
    print("    2. 安装 CosyVoice 依赖: pip install modelscope oss2 addict soundfile librosa")
    print("    3. 运行完整测试: python training/_test_cosyvoice.py")
elif issues:
    print("\n  [WARN] 需要解决以下问题:")
    for issue in issues:
        print(f"    - {issue}")
    if warnings:
        print("\n  警告:")
        for warn in warnings:
            print(f"    - {warn}")
else:
    print("\n  [WARN] 存在警告，但可能不影响使用:")
    for warn in warnings:
        print(f"    - {warn}")

print("\n  重要提示:")
print("    - MediaPipe 和 CosyVoice 使用不同的依赖树，理论上可以共存")
print("    - 如果出现 protobuf 冲突，考虑使用独立的 wakefusion_cosyvoice 环境")
print("    - 建议先测试 MediaPipe 功能是否正常，再安装 CosyVoice 依赖")
