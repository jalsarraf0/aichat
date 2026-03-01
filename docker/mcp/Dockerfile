FROM python:3.12-slim

# Intel Arc / integrated GPU: OpenCL ICD loader + DRM for cv2/OpenCL acceleration.
# ocl-icd-libopencl1 provides the OpenCL dispatcher; libdrm2 exposes DRM device nodes.
RUN apt-get update && apt-get install -y --no-install-recommends \
        ocl-icd-libopencl1 libdrm2 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY app.py .
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8096"]
