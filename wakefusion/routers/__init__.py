"""
路由器模块初始化
"""

from wakefusion.routers.audio_router import AudioRouter, AsyncAudioRouter
from wakefusion.routers.vision_router import VisionRouter, AsyncVisionRouter

__all__ = [
    'AudioRouter',
    'AsyncAudioRouter',
    'VisionRouter',
    'AsyncVisionRouter'
]
