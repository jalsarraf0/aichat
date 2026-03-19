FROM python:3.12-slim

# Install build essentials for psycopg binary wheels
RUN apt-get update && apt-get install -y --no-install-recommends \
        libpq-dev gcc ca-certificates curl && \
    rm -rf /var/lib/apt/lists/*

# Run as non-root
RUN useradd -m -u 1000 appuser
WORKDIR /app
RUN mkdir -p /data && chown appuser:appuser /data

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

COPY app.py migrate.py sqlite_to_pg.py ./
COPY migrations/ ./migrations/
RUN chown -R appuser:appuser /app

USER appuser
EXPOSE 8091
HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8091/health', timeout=3)"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8091", "--workers", "2"]
