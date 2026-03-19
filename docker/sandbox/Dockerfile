FROM python:3.12-slim

# Sandbox for arbitrary user code — no network-accessible secrets
# Uses non-root, minimal packages, controlled filesystem mounts
RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl wget git jq bash \
        ocl-icd-libopencl1 mesa-opencl-icd libdrm2 && \
    rm -rf /var/lib/apt/lists/* && \
    update-ca-certificates && \
    git config --global --add safe.directory '*'

# Create non-root user
RUN useradd -m -u 1000 appuser
WORKDIR /app
RUN mkdir -p /data/tools /data/repos && chown -R appuser:appuser /data

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    python -c "import certifi; import ssl; open('/etc/ssl/certs/combined-ca.crt', 'w').write(open('/etc/ssl/certs/ca-certificates.crt').read() + '\n' + open(certifi.where()).read())" || true

COPY app.py .
RUN chown -R appuser:appuser /app

USER appuser
EXPOSE 8095
HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8095/health', timeout=3)"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8095"]
