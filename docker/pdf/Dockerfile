FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        ocl-icd-libopencl1 mesa-opencl-icd \
        libdrm2 libva2 intel-media-va-driver \
        vainfo clinfo \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
EXPOSE 8103
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8103"]
