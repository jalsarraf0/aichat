FROM python:3.12-slim

# Intel Arc / integrated GPU userspace for vision tools:
# - ocl-icd-libopencl1 + mesa-opencl-icd: OpenCL runtime plumbing for OpenCV.
# - libdrm2 + libva2 + intel-media-va-driver: DRM/VA-API access for Intel GPUs.
# - vainfo + clinfo: lightweight runtime diagnostics.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates \
        ocl-icd-libopencl1 mesa-opencl-icd \
        libdrm2 libva2 intel-media-va-driver \
        vainfo clinfo \
        nodejs npm \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r mcp && useradd -r -g mcp -d /app -s /sbin/nologin mcp

ENV SSL_CERT_FILE=/etc/ssl/certs/ca-certificates.crt

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt
COPY app.py .
RUN chown -R mcp:mcp /app

USER mcp

HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8096/health', timeout=3)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8096"]
