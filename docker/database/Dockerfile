FROM python:3.12-slim

# Intel Arc / integrated GPU: OpenCL ICD loader + DRM for GPU-accelerated
# image hashing (phash via PIL/Pillow) and any future vision operations.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ocl-icd-libopencl1 mesa-opencl-icd libdrm2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8091"]
