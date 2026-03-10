"""
WakeFusion 项目安装配置
"""
from setuptools import setup, find_packages

# 读取版本号
from wakefusion import __version__

setup(
    name="wakefusion",
    version=__version__,
    description="WakeFusion - 展厅多模态唤醒模块",
    author="WakeFusion Team",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        # 核心依赖已在各环境的 requirements.txt 中安装
        # 这里只列出最基础的依赖，避免重复安装
        "pyyaml>=6.0",
        "pyzmq>=25.0.0",
        "pydantic>=2.0",
        "numpy>=1.24.0",
    ],
    # 不包含测试文件和数据文件
    exclude_package_data={
        "": ["*.pyc", "__pycache__", "*.log", "*.ckpt", "*.nemo", "*.onnx"],
    },
)
