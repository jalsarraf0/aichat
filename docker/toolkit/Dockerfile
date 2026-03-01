FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ca-certificates curl wget git jq bash \
        ocl-icd-libopencl1 libdrm2 && \
    rm -rf /var/lib/apt/lists/* && \
    update-ca-certificates && \
    git config --global --add safe.directory '*'

WORKDIR /app
RUN mkdir -p /data/tools /data/repos
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt && \
    cat /etc/ssl/certs/ca-certificates.crt >> $(python -c "import certifi; print(certifi.where())")

COPY app.py .
EXPOSE 8095
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8095"]
