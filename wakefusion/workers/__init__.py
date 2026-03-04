"""
工作线程模块初始化
"""

from wakefusion.workers.kws_worker import KWSWorker, AsyncKWSWorker
from wakefusion.workers.vad_worker import VADWorker, AsyncVADWorker
from wakefusion.workers.face_gate import FaceGateWorker, AsyncFaceGateWorker, FaceGateConfig

# MatchboxNet KWS Worker 需要 torch，可选导入（如果环境没有 torch，则跳过）
try:
    # 修正：以下代码块现在已正确缩进
    from wakefusion.workers.matchboxnet_kws_worker import (
        MatchboxNetKWSWorker,
        MatchboxNetConfig,
        create_matchboxnet_worker
    )
    _matchboxnet_available = True
except ImportError:
    # torch 未安装，MatchboxNet 不可用
    MatchboxNetKWSWorker = None
    MatchboxNetConfig = None
    create_matchboxnet_worker = None
    _matchboxnet_available = False

__all__ = [
    'KWSWorker',
    'AsyncKWSWorker',
    'VADWorker',
    'AsyncVADWorker',
    'FaceGateWorker',
    'AsyncFaceGateWorker',
    'FaceGateConfig',
]

# 只有在 MatchboxNet 可用时才添加到 __all__
if _matchboxnet_available:
    __all__.extend([
        'MatchboxNetKWSWorker',
        'MatchboxNetConfig',
        'create_matchboxnet_worker'
    ])