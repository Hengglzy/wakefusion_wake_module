"""
训练CLI入口
提供命令行接口启动自动化训练
"""

import asyncio
import sys
from pathlib import Path
from typing import Optional
import argparse

# 添加项目路径
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from wakefusion.training.orchestrator import TrainingOrchestrator
from wakefusion.training.config import TrainingConfig, TTSVoice
from wakefusion.logging import get_logger, set_log_level


logger = get_logger("train_cli")


def parse_arguments():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(
        description="WakeFusion 自动化中文唤醒词训练工具",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
示例:

  # 基础训练（使用默认参数）
  python -m cli.train "小康小康"

  # 自定义标签名
  python -m cli.train "小康小康" --label xiaokang

  # 指定输出目录
  python -m cli.train "小康小康" --output ./my_training

  # 添加负样本
  python -m cli.train "小康小康" --negative 小猫 小刚 消毒

  # GPU训练
  python -m cli.train "小康小康" --device cuda

  # 快速训练（减少样本和epochs）
  python -m cli.train "小康小康" --fast
        """
    )

    parser.add_argument(
        "wake_word",
        help="唤醒词文本（如：小康小康）"
    )

    parser.add_argument(
        "--label",
        default=None,
        help="模型标签名（英文，默认自动生成）"
    )

    parser.add_argument(
        "--output",
        default="training_output",
        help="输出目录（默认：training_output）"
    )

    parser.add_argument(
        "--samples",
        type=int,
        default=50,
        help="每个音色的样本数（默认：50）"
    )

    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="训练轮数（默认：50）"
    )

    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="批次大小（默认：32）"
    )

    parser.add_argument(
        "--device",
        choices=["cpu", "cuda"],
        default="cpu",
        help="训练设备（默认：cpu）"
    )

    parser.add_argument(
        "--negative",
        nargs="+",
        default=[],
        help="负样本列表（如：小猫 小刚 消毒）"
    )

    parser.add_argument(
        "--no-augmentation",
        action="store_true",
        help="禁用数据增强"
    )

    parser.add_argument(
        "--fast",
        action="store_true",
        help="快速训练模式（减少样本和epochs）"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="详细日志输出"
    )

    return parser.parse_args()


def generate_label_name(wake_word: str) -> str:
    """生成标签名（拼音化）"""
    import re
    # 简单实现：移除空格和特殊字符，保留英文和数字
    label = re.sub(r'[^a-zA-Z0-9]', '', wake_word)
    return label.lower() if label else "wake_word"


async def main():
    """主函数"""
    args = parse_arguments()

    # 设置日志级别
    set_log_level("DEBUG" if args.verbose else "INFO")

    # 生成标签名
    label_name = args.label or generate_label_name(args.wake_word)

    # 构建配置
    config = TrainingConfig(
        wake_word=args.wake_word,
        label_name=label_name,
        output_dir=args.output,
        negative_samples=args.negative
    )

    # 快速模式
    if args.fast:
        config.tts.samples_per_voice = 20
        config.training.num_epochs = 20
        config.training.batch_size = 16
        logger.info("启用快速训练模式")

    # 覆盖参数
    config.tts.samples_per_voice = args.samples
    config.training.num_epochs = args.epochs
    config.training.batch_size = args.batch_size
    config.device = args.device
    config.augmentation.enabled = not args.no_augmentation

    # 显示配置
    print("=" * 70)
    print("🚀 WakeFusion 自动化训练工具")
    print("=" * 70)
    print()
    print("训练配置:")
    print(f"  唤醒词: {config.wake_word}")
    print(f"  标签名: {config.label_name}")
    print(f"  样本数: {config.tts.samples_per_voice} × {len(config.tts.voices)} 音色 = {config.tts.samples_per_voice * len(config.tts.voices)} 个")
    print(f"  训练轮数: {config.training.num_epochs}")
    print(f"  批次大小: {config.training.batch_size}")
    print(f"  设备: {config.device}")
    print(f"  数据增强: {'启用' if config.augmentation.enabled else '禁用'}")
    print(f"  负样本: {', '.join(config.negative_samples) if config.negative_samples else '无'}")
    print()
    print(f"输出目录: {config.output_dir}")
    print()
    print("预计时间: " + ("30-60分钟（快速模式）" if args.fast else "1-2小时"))
    print()
    print("=" * 70)
    print()

    # 确认
    try:
        response = input("是否开始训练？ [Y/n]: ")
        if response.lower() == 'n':
            print("已取消")
            return
    except KeyboardInterrupt:
        print()
        print("已取消")
        return

    print()

    # 创建编排器
    orchestrator = TrainingOrchestrator(
        config=config,
        progress_callback=lambda stage, progress: print(f"\r[{stage}] {progress*100:.0f}%", end="", flush=True)
    )

    # 执行训练
    result = await orchestrator.train()

    print()
    print()
    print("=" * 70)

    if result.success:
        print("✅ 训练成功！")
        print()
        print("训练结果:")
        print(f"  训练准确率: {result.train_accuracy*100:.2f}%")
        print(f"  验证准确率: {result.val_accuracy*100:.2f}%")
        print(f"  测试准确率: {result.test_accuracy*100:.2f}%")
        print(f"  平均延迟: {result.avg_latency_ms:.2f}ms")
        print(f"  推荐阈值: {result.recommended_threshold:.2f}")
        print()
        print("输出文件:")
        print(f"  NeMo模型: {result.nemo_model_path}")
        print(f"  ONNX模型: {result.onnx_model_path}")
        print(f"  配置文件: {result.config_path}")
        print(f"  训练报告: {result.report_path}")
        print()
        print("下一步:")
        print(f"  1. 查看训练报告: cat {result.report_path}")
        print(f"  2. 测试模型: python tests/test_matchboxnet_microphone.py")
        print(f"  3. 更新配置: cp {result.config_path} config/config.yaml")
        print()
    else:
        print("❌ 训练失败")
        print()
        print(f"错误信息: {result.error_message}")
        print()
        print("💡 可能的解决方案:")
        print("  1. 检查网络连接（TTS需要网络）")
        print("  2. 安装依赖: pip install edge-tts nemo-toolkit[asr]")
        print("  3. 检查磁盘空间")
        print()

    print("=" * 70)


if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        print()
        print()
        print("⏹  训练中断")
        sys.exit(1)
