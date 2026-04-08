#!/bin/bash
# Docker entrypoint: download model if not present, then start API server.
#
# If models/ already has an ONNX file (from volume mount), skip download.
# Otherwise, download from Google Drive (no account needed).

set -e

MODEL_DIR="/app/models"
ONNX_PATTERN="*.onnx"

# Google Drive file IDs
INT8_ID="1vliNHcQ2fQ13Tk9VRDRPEllmwdoVzN6X"
FP16_ID="1nLfBITa-yjFZjdf_3XtEQjETn2VWG5lZ"

# Check if any ONNX model already exists
if ls ${MODEL_DIR}/${ONNX_PATTERN} 1>/dev/null 2>&1; then
    echo "Model found in ${MODEL_DIR}"
else
    echo "No model found. Downloading from Google Drive..."
    mkdir -p ${MODEL_DIR}

    if [ "${DEVICE:-cpu}" = "cpu" ]; then
        echo "Downloading INT8 model (21 MB, CPU optimized)..."
        curl -sL "https://drive.google.com/uc?export=download&id=${INT8_ID}" \
            -o ${MODEL_DIR}/UnetPlusPlus_efficientnet-b4_int8.onnx
    else
        echo "Downloading FP16 model (40 MB, GPU optimized)..."
        curl -sL "https://drive.google.com/uc?export=download&id=${FP16_ID}" \
            -o ${MODEL_DIR}/UnetPlusPlus_efficientnet-b4_fp16.onnx
    fi

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
