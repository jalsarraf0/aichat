FROM python:3.12-slim
WORKDIR /app
RUN apt-get update && apt-get install -y --no-install-recommends curl openssl \
    && rm -rf /var/lib/apt/lists/*
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY server.py .
EXPOSE 8200 8247
HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
  CMD curl -sf --insecure https://localhost:8200/health || curl -sf http://localhost:8200/health || exit 1
CMD ["python", "-u", "server.py"]
