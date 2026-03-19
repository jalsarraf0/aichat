FROM python:3.12-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
        gcc g++ libffi-dev \
    && rm -rf /var/lib/apt/lists/* \
    && groupadd -r jupyter && useradd -r -g jupyter -d /app -s /sbin/nologin jupyter

WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt \
    && python -m ipykernel install --sys-prefix

COPY app.py .

# Kernels store state in /kernels; mount a volume if persistence across restarts is needed
RUN mkdir -p /kernels && chown -R jupyter:jupyter /app /kernels

USER jupyter

HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8098/health', timeout=3)"

CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8098"]
