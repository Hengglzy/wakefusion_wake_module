"""
ONNX导出器
将NeMo模型转换为ONNX格式
"""

import torch
from pathlib import Path

from wakefusion.logging import get_logger


logger = get_logger("onnx_exporter")


class ONNXExporter:
    """ONNX导出器"""

    def __init__(self):
        """初始化ONNX导出器"""
        logger.info("ONNXExporter initialized")

    async def export(
        self,
        nemo_model_path: str,
        output_path: Path
    ) -> str:
        """
        导出ONNX模型

        Args:
            nemo_model_path: NeMo模型路径
            output_path: 输出ONNX模型路径

        Returns:
            导出的模型路径
        """
        try:
            logger.info(f"开始导出ONNX模型: {nemo_model_path}")

            from nemo.collections.asr.models import EncDecClassificationModel

            # 加载NeMo模型
            logger.info("加载NeMo模型")
            model = EncDecClassificationModel.restore_from(nemo_model_path)
            model.eval()

            # 准备示例输入
            dummy_input = torch.randn(1, 16000)  # 1秒音频 @ 16kHz

            # 确保输出目录存在
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)

            # 导出ONNX
            logger.info(f"导出ONNX到: {output_path}")
            torch.onnx.export(
                model,
                dummy_input,
                str(output_path),
                input_names=['audio_input'],
                output_names=['logits'],
                dynamic_axes={
                    'audio_input': {0: 'batch_size', 1: 'time'},
                    'logits': {0: 'batch_size'}
                },
                opset_version=14,
                do_constant_folding=True
            )

            logger.info(f"✅ ONNX模型已导出: {output_path}")

            # 验证ONNX模型
            self._verify_onnx_model(str(output_path))

            return str(output_path)

        except Exception as e:
            logger.error(f"ONNX导出失败: {e}", exc_info=True)
            raise

    def _verify_onnx_model(self, onnx_path: str):
        """
        验证ONNX模型

        Args:
            onnx_path: ONNX模型路径
        """
        try:
            import onnx
            import onnxruntime as ort

            # 加载并检查模型
            model = onnx.load(onnx_path)
            onnx.checker.check_model(model)

            logger.info("✅ ONNX模型验证通过")

            # 测试推理
            logger.info("测试ONNX推理...")
            session = ort.InferenceSession(onnx_path)

            # 准备测试输入
            dummy_input = {'audio_input': np.random.randn(1, 16000).astype(np.float32)}

            # 运行推理
            outputs = session.run(None, dummy_input)

            logger.info(f"✅ ONNX推理测试通过，输出shape: {outputs[0].shape}")

        except Exception as e:
            logger.error(f"ONNX模型验证失败: {e}")
            raise
