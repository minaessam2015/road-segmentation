"""Model optimization and export for production deployment.

Supports two target hardware profiles:
    - **CPU**: ONNX Runtime with INT8 quantization (fast, no GPU required)
    - **T4 GPU**: ONNX Runtime with FP16 (balances speed and accuracy)

Usage:
    python -m road_segmentation.api.optimize \\
        --checkpoint checkpoints/best.pth \\
        --output models/ \\
        --targets cpu gpu
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn as nn

from road_segmentation.models.factory import create_model

logger = logging.getLogger(__name__)


def load_model_from_checkpoint(
    checkpoint_path: str | Path,
    device: str = "cpu",
) -> tuple[nn.Module, dict]:
    """Load model and config from a training checkpoint."""
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    config = state.get("config", {})
    model_cfg = config.get("model", {})

    model = create_model(
        arch=model_cfg.get("arch", "Unet"),
        encoder_name=model_cfg.get("encoder_name", "resnet34"),
        encoder_weights=None,
        in_channels=model_cfg.get("in_channels", 3),
        classes=model_cfg.get("classes", 1),
    )
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model, config


def export_onnx(
    model: nn.Module,
    output_path: str | Path,
    image_size: int = 512,
    opset_version: int = 17,
) -> Path:
    """Export PyTorch model to ONNX format.

    Args:
        model: PyTorch model in eval mode.
        output_path: Path for the .onnx file.
        image_size: Input image dimensions (assumes square).
        opset_version: ONNX opset version.

    Returns:
        Path to the exported ONNX model.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    dummy_input = torch.randn(1, 3, image_size, image_size)

    # Use legacy TorchScript exporter — the new dynamo exporter fails
    # on UNet++ due to sympy constraint solving issues with dynamic shapes.
    # Also disable dynamic spatial dims — we use fixed input size anyway.
    torch.onnx.export(
        model,
        dummy_input,
        str(output_path),
        opset_version=max(opset_version, 18),
        input_names=["image"],
        output_names=["logits"],
        dynamo=False,
    )

    size_mb = output_path.stat().st_size / 1e6
    logger.info(f"ONNX model exported: {output_path} ({size_mb:.1f} MB)")
    return output_path


def quantize_onnx_dynamic(
    onnx_path: str | Path,
    output_path: Optional[str | Path] = None,
) -> Path:
    """Apply dynamic INT8 quantization to an ONNX model (CPU optimized).

    Dynamic quantization quantizes weights to INT8 and computes activations
    in INT8 at runtime. No calibration dataset needed.

    Args:
        onnx_path: Path to the input ONNX model.
        output_path: Path for the quantized model. Defaults to *_int8.onnx.

    Returns:
        Path to the quantized ONNX model.
    """
    from onnxruntime.quantization import QuantType, quantize_dynamic

    onnx_path = Path(onnx_path)
    if output_path is None:
        output_path = onnx_path.with_name(onnx_path.stem + "_int8.onnx")
    output_path = Path(output_path)

    quantize_dynamic(
        model_input=str(onnx_path),
        model_output=str(output_path),
        weight_type=QuantType.QUInt8,
    )

    original_size = onnx_path.stat().st_size / 1e6
    quantized_size = output_path.stat().st_size / 1e6
    ratio = quantized_size / original_size * 100

    logger.info(
        f"INT8 quantized: {output_path} ({quantized_size:.1f} MB, "
        f"{ratio:.0f}% of original {original_size:.1f} MB)"
    )
    return output_path


def convert_onnx_fp16(
    onnx_path: str | Path,
    output_path: Optional[str | Path] = None,
) -> Path:
    """Convert ONNX model weights to FP16 (GPU optimized).

    Reduces model size by ~50% with minimal accuracy loss.
    Inference runs in FP16 on GPU for ~2x speedup.

    Args:
        onnx_path: Path to the input ONNX model.
        output_path: Path for the FP16 model. Defaults to *_fp16.onnx.

    Returns:
        Path to the FP16 ONNX model.
    """
    import onnx
    from onnxruntime.transformers.float16 import convert_float_to_float16

    onnx_path = Path(onnx_path)
    if output_path is None:
        output_path = onnx_path.with_name(onnx_path.stem + "_fp16.onnx")
    output_path = Path(output_path)

    model = onnx.load(str(onnx_path))
    model_fp16 = convert_float_to_float16(model, keep_io_types=True)
    onnx.save(model_fp16, str(output_path))

    original_size = onnx_path.stat().st_size / 1e6
    fp16_size = output_path.stat().st_size / 1e6

    logger.info(
        f"FP16 converted: {output_path} ({fp16_size:.1f} MB, "
        f"from {original_size:.1f} MB)"
    )
    return output_path


class ONNXInferenceEngine:
    """ONNX Runtime inference engine for production deployment.

    Supports both CPU (with INT8 quantization) and GPU (with FP16).

    Usage:
        engine = ONNXInferenceEngine("models/model_int8.onnx", device="cpu")
        result = engine.predict(image_bytes)
    """

    def __init__(
        self,
        onnx_path: str | Path,
        device: str = "cpu",
        image_size: int = 512,
        threshold: float = 0.5,
        model_version: str = "v1.0",
    ) -> None:
        import onnxruntime as ort

        self.image_size = image_size
        self.threshold = threshold
        self.model_version = model_version
        self.device = device

        # Select execution provider
        if device == "gpu" or device == "cuda":
            providers = ["CUDAExecutionProvider", "CPUExecutionProvider"]
        else:
            providers = ["CPUExecutionProvider"]

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        if device == "cpu":
            sess_options.intra_op_num_threads = 4
            sess_options.inter_op_num_threads = 4

        self.session = ort.InferenceSession(
            str(onnx_path), sess_options=sess_options, providers=providers,
        )
        self.input_name = self.session.get_inputs()[0].name
        self.output_name = self.session.get_outputs()[0].name

        # Auto-detect image size from ONNX input shape if not specified
        input_shape = self.session.get_inputs()[0].shape  # e.g. [1, 3, 1024, 1024]
        if len(input_shape) == 4 and isinstance(input_shape[2], int) and input_shape[2] > 0:
            self.image_size = input_shape[2]

        active_provider = self.session.get_providers()[0]
        logger.info(f"ONNX Runtime: {onnx_path} | image_size={self.image_size} | provider={active_provider}")

    def predict(self, image_bytes: bytes) -> dict:
        """Run inference on raw image bytes.

        Returns dict with mask, road_coverage_pct, confidence_mean,
        inference_time_ms, and metadata.
        """
        import time

        import cv2

        t0 = time.time()

        # Decode image
        arr = np.frombuffer(image_bytes, dtype=np.uint8)
        image = cv2.imdecode(arr, cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError("Could not decode image")
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        original_h, original_w = image.shape[:2]

        # Preprocess
        resized = cv2.resize(image, (self.image_size, self.image_size))
        normalized = (resized.astype(np.float32) / 255.0 - [0.485, 0.456, 0.406]) / [0.229, 0.224, 0.225]
        tensor = normalized.transpose(2, 0, 1)[np.newaxis]  # (1, 3, H, W)

        # Inference
        logits = self.session.run([self.output_name], {self.input_name: tensor.astype(np.float32)})[0]

        # Post-process
        prob_map = 1 / (1 + np.exp(-logits[0, 0]))  # sigmoid
        prob_full = cv2.resize(prob_map, (original_w, original_h))
        mask = ((prob_full >= self.threshold) * 255).astype(np.uint8)

        road_pixels = (mask > 0).sum()
        total_pixels = mask.size
        road_coverage_pct = float(road_pixels / total_pixels * 100)
        confidence_mean = float(prob_full[prob_full >= self.threshold].mean()) if road_pixels > 0 else 0.0

        inference_time_ms = (time.time() - t0) * 1000

        return {
            "mask": mask,
            "probability_map": prob_full,
            "road_coverage_pct": round(road_coverage_pct, 2),
            "confidence_mean": round(confidence_mean, 4),
            "original_size": (original_w, original_h),
            "inference_time_ms": round(inference_time_ms, 1),
        }


def optimize_for_targets(
    checkpoint_path: str | Path,
    output_dir: str | Path,
    targets: list[str] = ("cpu", "gpu"),
) -> dict[str, Path]:
    """Full optimization pipeline: export ONNX and optimize for target hardware.

    Args:
        checkpoint_path: Path to training checkpoint (.pth).
        output_dir: Directory for optimized models.
        targets: List of targets — "cpu" (INT8) and/or "gpu" (FP16).

    Returns:
        Dict mapping target name to optimized model path.
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load model
    model, config = load_model_from_checkpoint(checkpoint_path)
    image_size = config.get("data", {}).get("image_size", 512)
    model_cfg = config.get("model", {})
    model_name = f"{model_cfg.get('arch', 'model')}_{model_cfg.get('encoder_name', 'encoder')}"

    # Export base ONNX
    base_onnx = output_dir / f"{model_name}.onnx"
    export_onnx(model, base_onnx, image_size=image_size)

    results = {"onnx_fp32": base_onnx}

    # CPU target: INT8 dynamic quantization
    if "cpu" in targets:
        int8_path = quantize_onnx_dynamic(base_onnx)
        results["cpu_int8"] = int8_path

    # GPU target: FP16 conversion
    if "gpu" in targets:
        try:
            fp16_path = convert_onnx_fp16(base_onnx)
            results["gpu_fp16"] = fp16_path
        except Exception as e:
            logger.warning(f"FP16 conversion failed: {e}. Use the FP32 ONNX for GPU.")
            results["gpu_fp16"] = base_onnx

    logger.info(f"Optimization complete. Models saved to: {output_dir}")
    for target, path in results.items():
        size = path.stat().st_size / 1e6
        logger.info(f"  {target}: {path.name} ({size:.1f} MB)")

    return results
