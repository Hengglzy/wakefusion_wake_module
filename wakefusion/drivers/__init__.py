"""
驱动模块初始化
"""

from wakefusion.drivers.audio_driver import XVF3800Driver
from wakefusion.drivers.camera_driver import Gemini330Driver, FemtoBoltDriver, CameraConfig, CameraState

__all__ = [
    'XVF3800Driver',
    'Gemini330Driver',
    'FemtoBoltDriver',  # 向后兼容别名
    'CameraConfig',
    'CameraState'
]
