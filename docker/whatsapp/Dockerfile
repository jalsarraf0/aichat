FROM node:20-slim

WORKDIR /app

# Build tools for Baileys native optional modules (node-gyp) + git for npm deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 make g++ git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Rewrite SSH GitHub URLs → HTTPS so npm can fetch git deps without SSH keys
RUN git config --global url."https://github.com/".insteadOf "ssh://git@github.com/"

# Persistent auth state lives in a named volume at /data/auth
RUN mkdir -p /data/auth \
    && groupadd -r botuser && useradd -r -g botuser -d /app -s /sbin/nologin botuser \
    && chown -R botuser:botuser /data/auth

COPY package.json ./
RUN npm install --omit=dev

COPY bot.js .
RUN chown -R botuser:botuser /app

USER botuser

HEALTHCHECK --interval=20s --timeout=5s --retries=3 \
    CMD node -e "require('http').get('http://localhost:8097/health', r => process.exit(r.statusCode === 200 ? 0 : 1)).on('error', () => process.exit(1))"

# QR-code page + health endpoint
EXPOSE 8097

CMD ["node", "bot.js"]
