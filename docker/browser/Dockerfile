FROM python:3.12-slim

# System deps for Playwright Chromium
RUN apt-get update && apt-get install -y --no-install-recommends \
        libnss3 libnspr4 libatk1.0-0 libatk-bridge2.0-0 \
        libcups2 libdrm2 libxkbcommon0 libxcomposite1 \
        libxdamage1 libxfixes3 libxrandr2 libgbm1 libpango-1.0-0 \
        libcairo2 libasound2 libatspi2.0-0 libwayland-client0 \
        fonts-noto-cjk fonts-noto-color-emoji \
        ca-certificates curl dbus xvfb && \
    rm -rf /var/lib/apt/lists/*

# Non-root user
RUN useradd -m -u 1000 browser
WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install Playwright Chromium browser
RUN playwright install chromium && \
    chmod -R 755 /root/.cache/ms-playwright || true

COPY app.py .
RUN chown -R browser:browser /app

# Playwright needs writable home for browser profile
RUN mkdir -p /home/browser/.cache && \
    cp -r /root/.cache/ms-playwright /home/browser/.cache/ 2>/dev/null || true && \
    chown -R browser:browser /home/browser

USER browser
EXPOSE 8104
HEALTHCHECK --interval=15s --timeout=5s --retries=3 \
    CMD python3 -c "import urllib.request; urllib.request.urlopen('http://localhost:8104/health', timeout=3)"
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8104"]
