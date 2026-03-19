FROM python:3.12-slim

# FFmpeg + Tesseract + Intel VA-API (iHD driver) + Poppler (pdf2image)
RUN apt-get update && apt-get install -y --no-install-recommends \
        ffmpeg \
        tesseract-ocr tesseract-ocr-eng \
        poppler-utils \
        libgl1 libglib2.0-0 \
        ocl-icd-libopencl1 mesa-opencl-icd libdrm2 \
        vainfo libva-drm2 intel-media-va-driver \
        ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Run as non-root
RUN useradd -m -u 1000 appuser
WORKDIR /app
RUN mkdir -p /workspace && chown appuser:appuser /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Download CLIP ViT-B/32 ONNX visual encoder (Xenova export, ~350MB, cached in layer)
# Download YOLOv8n ONNX for object/human detection (~6MB)
RUN mkdir -p /app/models && \
    curl -fSL -o /app/models/clip_vit_b32_visual.onnx \
    "https://huggingface.co/Xenova/clip-vit-base-patch32/resolve/main/onnx/vision_model.onnx" && \
    curl -fSL -o /app/models/yolov8n.onnx \
    "https://huggingface.co/s1777/yolo-v8n-onnx/resolve/main/yolov8n.onnx" && \
    chown -R appuser:appuser /app/models

COPY app.py .
RUN chown -R appuser:appuser /app

USER appuser
EXPOSE 8099
HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8099/health', timeout=3)"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8099"]
