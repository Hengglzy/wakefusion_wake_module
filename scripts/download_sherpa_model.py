#!/usr/bin/env python3
"""
Sherpa-ONNX 模型下载脚本
自动下载中文KWS模型
"""

import os
import sys
import tarfile
from pathlib import Path
from urllib.request import urlretrieve
from urllib.error import URLError


# 模型配置
MODEL_CONFIG = {
    "zh-16kHz": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-zh-16kHz.tar.gz",
        "filename": "sherpa-onnx-kws-zh-16kHz.tar.gz",
        "extract_dir": "sherpa-onnx-kws-zh-16kHz",
        "description": "中文KWS模型 (16kHz)"
    },
    "en-16kHz": {
        "url": "https://github.com/k2-fsa/sherpa-onnx/releases/download/kws-models/sherpa-onnx-kws-en-16kHz.tar.gz",
        "filename": "sherpa-onnx-kws-en-16kHz.tar.gz",
        "extract_dir": "sherpa-onnx-kws-en-16kHz",
        "description": "英文KWS模型 (16kHz)"
    }
}


def download_with_progress(url: str, dest_path: str):
    """
    下载文件并显示进度条

    Args:
        url: 下载URL
        dest_path: 目标路径
    """
    print(f"\n正在下载: {url}")
    print(f"保存到: {dest_path}")

    def progress(block_num, block_size, total_size):
        """显示下载进度"""
        downloaded = block_num * block_size
        percent = min(100, downloaded * 100.0 / total_size) if total_size > 0 else 0

        # 每10%显示一次
        if int(percent) % 10 == 0 and downloaded > 0:
            mb_downloaded = downloaded / (1024 * 1024)
            mb_total = total_size / (1024 * 1024)
            print(f"  进度: {percent:.1f}% ({mb_downloaded:.1f}MB / {mb_total:.1f}MB)", end='\r')

    try:
        urlretrieve(url, dest_path, reporthook=progress)
        print(f"\n✅ 下载完成!")
        return True

    except URLError as e:
        print(f"\n❌ 下载失败: {e}")
        print("\n可能的原因:")
        print("  1. 网络连接问题")
        print("  2. GitHub访问受限")
        print("\n请尝试手动下载:")
        print(f"  {url}")
        return False

    except Exception as e:
        print(f"\n❌ 下载失败: {e}")
        return False


def extract_tar_gz(tar_path: str, extract_to: str):
    """
    解压 tar.gz 文件

    Args:
        tar_path: 压缩文件路径
        extract_to: 解压目标目录
    """
    print(f"\n正在解压: {tar_path}")
    print(f"到: {extract_to}")

    try:
        with tarfile.open(tar_path, 'r:gz') as tar:
            tar.extractall(path=extract_to)

        print(f"✅ 解压完成!")
        return True

    except Exception as e:
        print(f"❌ 解压失败: {e}")
        return False


def verify_model_files(model_dir: Path) -> bool:
    """
    验证模型文件是否完整

    Args:
        model_dir: 模型目录

    Returns:
        bool: 文件是否完整
    """
    required_files = [
        "encoder.onnx",
        "decoder.onnx",
        "joiner.onnx",
        "tokens.txt"
    ]

    print("\n正在验证模型文件...")

    all_exist = True
    for filename in required_files:
        file_path = model_dir / filename
        if file_path.exists():
            size_mb = file_path.stat().st_size / (1024 * 1024)
            print(f"  ✅ {filename} ({size_mb:.2f}MB)")
        else:
            print(f"  ❌ {filename} 缺失")
            all_exist = False

    return all_exist


def download_model(model_key: str, models_dir: Path = None):
    """
    下载模型

    Args:
        model_key: 模型配置键（zh-16kHz 或 en-16kHz）
        models_dir: 模型根目录
    """
    if model_key not in MODEL_CONFIG:
        print(f"❌ 不支持的模型: {model_key}")
        print(f"可用的模型: {', '.join(MODEL_CONFIG.keys())}")
        return False

    config = MODEL_CONFIG[model_key]

    # 设置模型目录
    if models_dir is None:
        script_dir = Path(__file__).parent.parent
        models_dir = script_dir / "models"

    model_dir = models_dir / config["extract_dir"]
    tar_path = models_dir / config["filename"]

    # 创建模型目录
    models_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print(f"Sherpa-ONNX 模型下载工具")
    print("=" * 70)
    print(f"\n模型: {config['description']}")
    print(f"URL: {config['url']}")

    # 检查是否已下载
    if model_dir.exists():
        print(f"\n⚠️  模型目录已存在: {model_dir}")

        choice = input("是否重新下载？(y/N): ").strip().lower()
        if choice != 'y':
            print("跳过下载")

            # 验证现有模型
            if verify_model_files(model_dir):
                print(f"\n✅ 模型文件完整!")
                print(f"模型路径: {model_dir}")
                print(f"\n请在配置文件中设置:")
                print(f"  kws:")
                print(f"    model_dir: \"{model_dir}\"")
                return True
            else:
                print(f"\n❌ 模型文件不完整，请重新下载")
                return False

    # 下载
    if not download_with_progress(config["url"], tar_path):
        return False

    # 解压
    if not extract_tar_gz(tar_path, models_dir):
        return False

    # 验证
    if not verify_model_files(model_dir):
        print(f"\n❌ 模型文件不完整!")
        return False

    # 清理压缩文件
    print(f"\n清理压缩文件...")
    tar_path.unlink()

    print(f"\n" + "=" * 70)
    print(f"✅ 模型下载完成!")
    print(f"=" * 70)
    print(f"\n模型路径: {model_dir}")
    print(f"\n请在配置文件中设置:")
    print(f"  kws:")
    print(f"    model_dir: \"{model_dir}\"")
    print(f"    keywords:")
    print(f"      - \"小康小康\"")

    return True


def main():
    """主函数"""
    import argparse

    parser = argparse.ArgumentParser(description="下载Sherpa-ONNX KWS模型")
    parser.add_argument(
        "--model",
        choices=["zh-16kHz", "en-16kHz"],
        default="zh-16kHz",
        help="模型语言（默认: zh-16kHz 中文）"
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="模型输出目录（默认: ./models/）"
    )

    args = parser.parse_args()

    # 设置模型目录
    models_dir = Path(args.output_dir) if args.output_dir else None

    # 下载
    success = download_model(args.model, models_dir)

    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
