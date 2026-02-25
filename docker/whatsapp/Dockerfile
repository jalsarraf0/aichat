FROM node:20-slim

WORKDIR /app

# Build tools for Baileys native optional modules (node-gyp) + git for npm deps
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3 make g++ git ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Rewrite SSH GitHub URLs → HTTPS so npm can fetch git deps without SSH keys
RUN git config --global url."https://github.com/".insteadOf "ssh://git@github.com/"

# Persistent auth state lives in a named volume at /data/auth
RUN mkdir -p /data/auth

COPY package.json ./
RUN npm install --omit=dev

COPY bot.js .

# QR-code page + health endpoint
EXPOSE 8097

CMD ["node", "bot.js"]
