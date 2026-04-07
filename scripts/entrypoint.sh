#!/bin/bash
# Docker entrypoint: download model if not present, then start API server.
#
# If models/ already has an ONNX file (from volume mount), skip download.
# Otherwise, download from W&B artifacts (requires WANDB_API_KEY env var).

set -e

MODEL_DIR="/app/models"
ONNX_PATTERN="*.onnx"

# Check if any ONNX model already exists
if ls ${MODEL_DIR}/${ONNX_PATTERN} 1>/dev/null 2>&1; then
    echo "Model found in ${MODEL_DIR}"
else
    echo "No model found. Downloading from W&B..."
    if [ -z "$WANDB_API_KEY" ]; then
        echo "ERROR: No model files and no WANDB_API_KEY set."
        echo "Either mount models to /app/models or set WANDB_API_KEY."
        exit 1
    fi
    python scripts/download_model.py --onnx-only --output /app
    echo "Model downloaded."
fi

# Find the best ONNX model (prefer INT8 for CPU, FP16 for GPU)
if [ "${DEVICE:-cpu}" = "cpu" ]; then
    MODEL_FILE=$(ls ${MODEL_DIR}/*_int8.onnx 2>/dev/null | head -1)
fi
if [ -z "$MODEL_FILE" ]; then
    MODEL_FILE=$(ls ${MODEL_DIR}/*_fp16.onnx 2>/dev/null | head -1)
fi
if [ -z "$MODEL_FILE" ]; then
    MODEL_FILE=$(ls ${MODEL_DIR}/*.onnx 2>/dev/null | head -1)
fi

if [ -z "$MODEL_FILE" ]; then
    echo "ERROR: No ONNX model found in ${MODEL_DIR}"
    exit 1
fi

echo "Starting API with model: ${MODEL_FILE}"
exec python scripts/serve.py --onnx "${MODEL_FILE}" --device "${DEVICE:-cpu}" --host 0.0.0.0
