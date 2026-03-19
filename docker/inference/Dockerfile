FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl libgomp1 && \
    rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
# Install CPU-only PyTorch first (smaller image), then the rest
RUN pip install --no-cache-dir torch>=2.2.0 --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir fastapi>=0.116.0 uvicorn==0.30.6 pydantic>=2.9.0 \
        numpy>=1.24.0 sentence-transformers>=3.0.0 einops "starlette>=0.49.1"

# Download model at build time (fast startup)
RUN python3 -c "\
from sentence_transformers import SentenceTransformer; \
m = SentenceTransformer('nomic-ai/nomic-embed-text-v1.5', trust_remote_code=True); \
m.save('/app/models/nomic-embed'); \
print('OK: nomic-embed-text-v1.5 downloaded')"

COPY app.py .

RUN groupadd -r inference && useradd -r -g inference -d /app -s /sbin/nologin inference && \
    chown -R inference:inference /app
USER inference

EXPOSE 8105

HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD curl -sf http://localhost:8105/health || exit 1

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8105", "--workers", "1"]
