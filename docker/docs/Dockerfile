FROM python:3.12-slim

# System deps: poppler (pdf2image), libGL (OpenCV), Intel GPU (optional)
RUN apt-get update && apt-get install -y --no-install-recommends \
        poppler-utils \
        libgl1 libglib2.0-0 libmupdf-dev \
        ocl-icd-libopencl1 mesa-opencl-icd libdrm2 \
        ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Run as non-root
RUN useradd -m -u 1000 appuser
WORKDIR /app
RUN mkdir -p /workspace && chown appuser:appuser /workspace

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy both the main app and the PDF sub-service
COPY app.py pdf_service.py ./
RUN chown -R appuser:appuser /app

USER appuser
EXPOSE 8101
HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8101/health', timeout=3)"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8101"]
