# ===========================================================
# Road Segmentation Inference Service
# ===========================================================
# Multi-stage build: builder installs deps, runtime is minimal.
#
# Build:
#   docker build -t road-segmentation .
#
# Run (CPU with PyTorch checkpoint):
#   docker run -p 8000:8000 -v ./checkpoints:/app/checkpoints \
#     road-segmentation python scripts/serve.py \
#     --checkpoint checkpoints/best.pth
#
# Run (CPU with ONNX model):
#   docker run -p 8000:8000 -v ./models:/app/models \
#     road-segmentation python scripts/serve.py \
#     --onnx models/model_int8.onnx --device cpu
# ===========================================================

FROM python:3.12-slim AS builder

WORKDIR /app

# Install build dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    gcc g++ libglib2.0-0 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies first (cached layer)
COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

# ===========================================================
FROM python:3.12-slim

WORKDIR /app

# Runtime dependencies only
RUN apt-get update && apt-get install -y --no-install-recommends \
    libglib2.0-0 libgl1-mesa-glx \
    && rm -rf /var/lib/apt/lists/*

# Copy installed packages from builder
COPY --from=builder /install /usr/local

# Copy project code
COPY pyproject.toml .
COPY src/ src/
COPY scripts/ scripts/
COPY configs/ configs/

# Install the package itself (editable not needed in production)
RUN pip install --no-cache-dir --no-deps .

# Create directories for results and models
RUN mkdir -p results models checkpoints

# Health check
HEALTHCHECK --interval=30s --timeout=5s --retries=3 \
    CMD python -c "import requests; r=requests.get('http://localhost:8000/health'); exit(0 if r.ok else 1)" || exit 1

EXPOSE 8000

# Default command — override with specific model path
CMD ["python", "scripts/serve.py", "--onnx", "models/model.onnx", "--device", "cpu", "--host", "0.0.0.0"]
