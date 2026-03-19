# AIChat

[![CI](https://github.com/Al-Sarraf-Tech/aichat/actions/workflows/ci.yml/badge.svg)](https://github.com/Al-Sarraf-Tech/aichat/actions/workflows/ci.yml)
[![Dart CI](https://github.com/Al-Sarraf-Tech/aichat/actions/workflows/dart-ci.yml/badge.svg)](https://github.com/Al-Sarraf-Tech/aichat/actions/workflows/dart-ci.yml)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![Dart](https://img.shields.io/badge/dart-stable-blue)
![Platform](https://img.shields.io/badge/platform-Linux-informational)

> CI runs on self-hosted runners managed by [haskell-ci-orchestrator](https://github.com/jalsarraf0/haskell-ci-orchestrator) with build attestation.

A **local-first AI chat platform** with two interfaces:

- **Web UI** — A polished, ChatGPT-like web interface (Dart/Shelf backend + vanilla JS frontend) with JWT auth, streaming, tool cards, image grids with lightbox, and per-model capability badges
- **Terminal UI** — A Textual-based TUI for terminal-native workflows with 14 keybinds and full MCP tool access

Both interfaces connect to [LM Studio](https://lmstudio.ai) for local LLM inference (no cloud dependency) and share a unified MCP tool server with 16 mega-tools backed by real services: web search, image search, browser automation, code execution, persistent memory, knowledge graphs, vector search, PDF processing, and more.

---

## Table of Contents

- [Architecture](#architecture)
- [Hardware Layout](#hardware-layout)
- [Services](#services)
- [Models](#models)
- [Tool Tier System](#tool-tier-system)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [Web UI](#web-ui)
- [TUI Usage](#tui-usage)
- [Arc A380 Preprocessing](#arc-a380-preprocessing)
- [Image Pipeline](#image-pipeline)
- [Personalities](#personalities)
- [CI/CD Pipeline](#cicd-pipeline)
- [Development](#development)
- [Replicating This Setup](#replicating-this-setup)
- [Known Limitations](#known-limitations)

---

## Architecture

```
                    ┌──────────────────────────────────────────────────────┐
                    │  User                                                │
                    │  Browser (:8200) ──── or ──── Terminal (aichat TUI)  │
                    └────────┬───────────────────────────────┬─────────────┘
                             │                               │
                             ▼                               ▼
                    ┌────────────────┐              ┌────────────────┐
                    │  aichat-auth   │              │  aichat TUI    │
                    │  Flask JWT     │              │  Python Textual│
                    │  :8200 (proxy) │              │  MCP stdio     │
                    │  :8247 (admin) │              └───────┬────────┘
                    └───────┬────────┘                      │
                            │                               │
                            ▼                               │
                    ┌────────────────┐                      │
                    │  aichat-web    │                      │
                    │  Dart/Shelf    │                      │
                    │  SSE streaming │                      │
                    │  :8200 (int)   │                      │
                    └───┬────────┬───┘                      │
                        │        │                          │
            ┌───────────┘        └──────────┐               │
            ▼                               ▼               ▼
   ┌─────────────────┐            ┌──────────────────────────────┐
   │  Intel Arc A380 │            │  aichat-mcp  :8096           │
   │  :1235 (router) │            │  MCP gateway — 16 mega-tools │
   │  Qwen2.5-3B     │            │  FastAPI + httpx              │
   │  Tool routing    │            └──────┬───────────────────────┘
   │  Prompt compress │                   │
   └─────────────────┘        ┌───────────┼───────────┬──────────┐
                              ▼           ▼           ▼          ▼
                    ┌──────────┐  ┌────────┐  ┌────────┐  ┌──────────┐
                    │aichat-   │  │aichat- │  │aichat- │  │aichat-   │
                    │data:8091 │  │vision  │  │browser │  │sandbox   │
                    │memory    │  │:8099   │  │:8104   │  │:8095     │
                    │graph     │  │OCR,face│  │Chromium│  │Python/JS │
                    │planner   │  │video   │  │scrape  │  │bash exec │
                    └────┬─────┘  └────────┘  └────────┘  └──────────┘
                         │
        ┌────────────────┼────────────────┐
        ▼                ▼                ▼
   ┌─────────┐    ┌───────────┐    ┌───────────┐
   │Postgres │    │Qdrant     │    │Redis      │
   │:5432    │    │:6333      │    │(Valkey)   │
   │auth+data│    │vectors    │    │ctx cache  │
   └─────────┘    └───────────┘    └───────────┘

                  ┌──────────────────────────────┐
                  │  LM Studio  (RTX 3090)       │
                  │  192.168.50.2:1234            │
                  │  Main inference — all models  │
                  │  JIST auto-load/swap          │
                  └──────────────────────────────┘
```

---

## Hardware Layout

This platform is designed for a **two-GPU split architecture** across a local network:

```
┌────────────────────────────────────────────┐     ┌────────────────────────────────────┐
│  SERVER  (Fedora 43, Intel Arc A380)       │     │  WORKSTATION  (Windows, RTX 3090)  │
│  hostname: amarillo                        │     │  hostname: dominus                 │
│  LAN: 192.168.50.5                         │     │  LAN: 192.168.50.2                 │
│                                            │     │                                    │
│  ┌──────────────────────────────────────┐  │     │  ┌──────────────────────────────┐  │
│  │  Docker Compose Stack               │  │     │  │  LM Studio                   │  │
│  │                                     │  │     │  │  :1234                        │  │
│  │  aichat-auth   :8200 (user-facing)  │  │     │  │                              │  │
│  │  aichat-web    :8200 (internal)     │──────────►│  7 chat models               │  │
│  │  aichat-mcp    :8096 (tools)        │  │     │  │  JIST auto-load/swap         │  │
│  │  aichat-redis  :6379 (cache)        │  │     │  │  24GB VRAM                   │  │
│  │  aichat-db     :5432 (postgres)     │  │     │  └──────────────────────────────┘  │
│  │  aichat-vector :6333 (qdrant)       │  │     │                                    │
│  │  aichat-data   :8091                │  │     └────────────────────────────────────┘
│  │  aichat-vision :8099                │  │
│  │  aichat-browser:8104                │  │
│  │  aichat-searxng:8080                │  │
│  │  aichat-sandbox:8095                │  │
│  │  aichat-docs   :8101                │  │
│  │  aichat-minio  :9001/9002           │  │
│  │  + 3 more services                  │  │
│  └──────────────────────────────────────┘  │
│                                            │
│  ┌──────────────────────────────────────┐  │
│  │  Intel Arc A380 (6GB VRAM)          │  │
│  │  LM Studio :1235 (via socat proxy)  │  │
│  │  Qwen2.5-3B-Instruct (always loaded)│  │
│  │                                     │  │
│  │  Roles:                             │  │
│  │  1. Tool routing    (~500ms-1.1s)   │  │
│  │  2. Prompt compress (~3-5s)         │  │
│  │  3. Context compact (cached Redis)  │  │
│  └──────────────────────────────────────┘  │
│                                            │
│  NVMe Persistence: /mnt/nvmeINT/aichat/   │
│  ├── postgres/  redis/  qdrant/            │
│  ├── web-db/  minio/  whatsapp/            │
│  └── ~930GB free                           │
└────────────────────────────────────────────┘
```

### Request Flow

```
User browser → :8200 (aichat-auth, JWT check)
  → aichat-web (Dart/Shelf)
    → Arc A380 :1235 (tool routing + prompt compression)
    → aichat-mcp :8096 (tool execution)
      → aichat-searxng (web/image search)
      → aichat-browser (page scraping)
      → aichat-vision (OCR, face detect)
      → aichat-sandbox (code execution)
    → RTX 3090 :1234 (main LLM generation, SSE streaming)
  ← SSE tokens stream back to browser
```

### Network Rules

- All ports/IPs are fixed and must not change
- Docker services communicate on internal Docker network
- Only `:8200` (web), `:8247` (admin), and `:8096` (MCP) are host-exposed
- LM Studio endpoints are on the LAN (`192.168.50.x`)
- Redis is internal-only (no host binding)

You can run everything on a single machine — just point `LM_STUDIO_URL` and `TOOL_ROUTER_URL` to `localhost`.

---

## Services

| Service | Port | Persistence | Purpose |
|---------|------|-------------|---------|
| `aichat-db` | 5432 | `/mnt/nvmeINT/aichat/postgres/` | PostgreSQL — auth users, articles, metadata |
| `aichat-vector` | 6333 | `/mnt/nvmeINT/aichat/qdrant/` | Qdrant — semantic search, CLIP embeddings |
| `aichat-redis` | 6379 | `/mnt/nvmeINT/aichat/redis/` | Valkey — context compaction cache, tool routing cache |
| `aichat-data` | 8091 | — | Consolidated data API: memory, articles, graph, planner, jobs, research |
| `aichat-vision` | 8099 | — | OCR (Tesseract), face detection (OpenCV), video analysis |
| `aichat-docs` | 8101 | — | Document ingestion, PDF extraction, full-text search |
| `aichat-sandbox` | 8095 | — | Isolated code execution: Python, bash, JavaScript |
| `aichat-searxng` | 8080 | — | Self-hosted meta-search (DuckDuckGo, Bing, Google) |
| `aichat-mcp` | **8096** | — | MCP HTTP/SSE gateway — 16 mega-tools |
| `aichat-browser` | 8104 | — | Headless Chromium for page screenshots and scraping |
| `aichat-jupyter` | 8098 | — | Stateful Jupyter kernel for code execution |
| `aichat-whatsapp` | 8097 | — | WhatsApp bot integration |
| `aichat-minio` | 9001/9002 | `/mnt/nvmeINT/aichat/minio/` | S3-compatible object storage |
| `aichat-inference` | 8105 | — | Intel Arc OpenVINO embeddings (offload from RTX 3090) |
| `aichat-web` | 8200 (int) | `/mnt/nvmeINT/aichat/web-db/` | Dart/Shelf web server + vanilla JS frontend |
| `aichat-auth` | **8200**, 8247 | — | Flask JWT auth proxy + admin panel |

All data volumes persist to NVMe at `/mnt/nvmeINT/aichat/`. Change this path in `docker-compose.yml` for your system.

---

## Models

Tested models (from LM Studio on RTX 3090):

| Model | Type | Quant | Context | Tools | Reasoning | Notes |
|-------|------|-------|---------|-------|-----------|-------|
| `openai/gpt-oss-20b` | LLM | MXFP4 | 131K | 9 | No | Strong general model |
| `dolphin-mistral-glm-4.7-flash-24b-*` | LLM | Q4_K_S | 32K | 9 | Yes | **UNRESTRICTED**, thinking, enforceTools |
| `qwen/qwen3.5-9b` | VLM | Q4_K_M | 262K | 7 | Yes | Vision + reasoning |
| `zai-org/glm-4.6v-flash` | VLM | Q8_0 | 131K | 7 | Yes | Vision + reasoning |
| `ibm/granite-4-h-tiny` | LLM | Q8_0 | 1M | 5 | No | Tiny, condensed prompts |
| `microsoft/phi-4-mini-reasoning` | LLM | Q8_0 | 131K | 2 | Yes | Very weak tools |

The `qwen2.5-3b-instruct` model runs permanently on the Intel Arc A380 as the tool router / prompt preprocessor.

---

## Tool Tier System

Instead of sending all 16 tools to every model (which causes timeouts and confusion), tools are organized into tiers:

### Default Tier (7 tools) — available to most models

| Tool | Actions | Description |
|------|---------|-------------|
| `web` | search, fetch, extract, news, wikipedia, arxiv, youtube | Web search and page fetching via SearXNG/DDG/Bing |
| `image` | search, generate, edit, fetch, caption, face_detect | Image search (SearXNG/DDG/Bing Images), generation, analysis |
| `browser` | navigate, screenshot, scrape, page_images | Headless Chromium browser automation |
| `research` | deep, rss_search | Multi-hop deep research with RSS feeds |
| `code` | python, javascript, bash | Sandboxed code execution |
| `document` | ingest, extract, ocr, pdf | Document processing and OCR |
| `memory` | store, recall, list, delete | Persistent key-value memory |

### Extended Tier (+2 for strong models)

| Tool | Description |
|------|-------------|
| `media` | Video analysis, TTS, object detection |
| `data` | Database operations, article storage, cache |

### Never-Routed (internal/infrastructure)

`knowledge`, `vector`, `jobs`, `custom_tools`, `planner`, `think`, `system` — used internally by the MCP server, never sent to chat models.

### Per-Model Tool Allocation

| Model | Tools | Count |
|-------|-------|-------|
| gpt-oss-20b, dolphin | Default + Extended | 9 |
| qwen3.5-9b | Default only | 7 |
| glm-4.6v-flash | web, image, browser, research, document, data, media | 7 |
| granite-tiny | web, image, browser, code, memory | 5 |
| phi-4-mini | web, browser | 2 |

### Image-Only Routing

When a user asks for images/pictures/photos, the tool router sends **only the `image` tool**. This prevents the model from using `web` search (which returns page links, not actual images) and hallucinating fake image URLs. The `image` tool does its own SearXNG/DDG/Bing image search and returns real CDN URLs + inline base64 images.

---

## Requirements

| Component | Minimum | Recommended |
|-----------|---------|-------------|
| OS | Linux (any distro with Docker) | Fedora 43 / Ubuntu 24.04 |
| Docker | 24.0+ with Compose v2 | Latest stable |
| Python | 3.12+ | 3.14 |
| Dart SDK | Stable channel | Latest stable |
| LM Studio | Any version | Latest with JIST support |
| GPU (inference) | 8GB VRAM | NVIDIA RTX 3090 (24GB) |
| GPU (preprocessing) | Optional | Intel Arc A380 (6GB) |
| RAM | 16 GB | 64 GB |
| Storage | 20 GB free | NVMe SSD for Docker volumes |

---

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/Al-Sarraf-Tech/aichat.git
cd aichat

# Create environment file
cat > .env << 'EOF'
POSTGRES_PASSWORD=your_secure_password_here
MINIO_ROOT_USER=minioadmin
MINIO_ROOT_PASSWORD=your_minio_password_here
JWT_SECRET=$(openssl rand -hex 32)
ADMIN_USER=your_username
ADMIN_INITIAL_PASSWORD=your_password_here
IMAGE_GEN_BASE_URL=http://your-lm-studio-host:1234
IMAGE_GEN_MODEL=openai/gpt-oss-20b
EOF
```

### 2. Start all services

```bash
docker compose up -d
```

First run builds all images (~10-15 min). Subsequent starts take under 30 seconds.

### 3. Verify the stack

```bash
make smoke    # hits /health on every service
# or manually:
curl http://localhost:8096/health   # MCP: {"ok":true}
curl http://localhost:8200/health   # Web: {"ok":true,"service":"dartboard"}
```

### 4. Access the Web UI

Open **http://localhost:8200** in your browser. Log in with the `ADMIN_USER` / `ADMIN_INITIAL_PASSWORD` from your `.env`.

### 5. Install the TUI (optional)

```bash
pip install -e ".[dev]"
aichat              # interactive TUI
aichat "question"   # one-shot chat
```

---

## Configuration

### Environment Variables

| Variable | Service | Default | Description |
|----------|---------|---------|-------------|
| `POSTGRES_PASSWORD` | db, auth, data | *required* | PostgreSQL password |
| `JWT_SECRET` | auth | *required* | JWT signing key (use `openssl rand -hex 32`) |
| `ADMIN_USER` | auth | `admin` | Admin username (auto-created on first start) |
| `ADMIN_INITIAL_PASSWORD` | auth | — | Initial admin password |
| `IMAGE_GEN_BASE_URL` | mcp | `http://192.168.50.2:1234` | LM Studio URL for MCP tools |
| `IMAGE_GEN_MODEL` | mcp | *(auto)* | Pin model for MCP; empty = auto-select |
| `MINIO_ROOT_USER` | minio | `minioadmin` | MinIO admin user |
| `MINIO_ROOT_PASSWORD` | minio | *required* | MinIO admin password |
| `LM_STUDIO_URL` | web | `http://192.168.50.2:1234` | LM Studio URL for web server |
| `LM_STUDIO_FALLBACK_URL` | web | `http://100.78.39.76:1234` | Failover LM Studio URL |
| `TOOL_ROUTER_URL` | web | — | Arc A380 URL (empty = rule-based routing) |
| `MCP_URL` | web | `http://aichat-mcp:8096` | MCP server URL |
| `MAX_TOOL_ITERATIONS` | web | `4` | Max tool call loops per request |

### Adapting to Your Hardware

**Single machine (no Arc GPU):**
```bash
# .env — leave TOOL_ROUTER_URL empty to use rule-based routing
LM_STUDIO_URL=http://localhost:1234
LM_STUDIO_FALLBACK_URL=http://localhost:1234
TOOL_ROUTER_URL=
```

**Two machines (like this setup):**
```bash
# .env
LM_STUDIO_URL=http://gpu-workstation:1234    # RTX 3090
TOOL_ROUTER_URL=http://gpu-workstation:1235  # Arc A380 (or same host)
```

---

## Web UI

The web interface is a vanilla JS single-page application with a ChatGPT-like design.

### Features

- **Streaming responses** with blinking cursor, token/s counter, and elapsed time
- **Dedicated tool cards** for active tool calls with live elapsed timers
- **Image grid** with responsive CSS Grid layout and lightbox (keyboard nav: Escape, arrows)
- **Image dedup** with URL normalization (strips CDN size suffixes, query params)
- **Model selector** with state dots (loaded/unloaded), quantization, context length, type (LLM/VLM), tool count
- **Thinking/reasoning** content in collapsible cards (hidden by default)
- **30 personalities** including category-filtered grid, custom prompts
- **File uploads** (images for vision models, text/code files inline)
- **Per-user chat isolation** via JWT auth
- **Admin panel** at `:8247` for user management, IP ban control

### SSE Event Types

The backend streams responses via Server-Sent Events:

| Event | Description |
|-------|-------------|
| `status` | Loading model indicator |
| `thinking` | Reasoning/thinking content (collapsible) |
| `token` | Content tokens (buffered, rendered once on `done`) |
| `tool_start` | Tool execution started (shows card if not suppressed) |
| `tool_result` | Tool execution complete (images collected for final render) |
| `error` | Error message |
| `done` | Stream complete, triggers single render pass |

### Buffer-Then-Render Architecture

Tokens are accumulated in memory during streaming. The DOM is updated with a lightweight plain-text preview only. On `done`, a single markdown render pass (marked.js + DOMPurify + highlight.js) produces the final output. This eliminates DOM thrashing and keeps the UI smooth during long generations.

---

## TUI Usage

```
aichat [OPTIONS] [MESSAGE]

Subcommands:
  aichat mcp          Run MCP server over stdio
  aichat repo create  Create and push GitHub repo
```

### Keyboard Shortcuts

| Key | Action | Key | Action |
|-----|--------|-----|--------|
| F1 | Help | F7 | Sessions |
| F2 | Model picker | F8 | Settings |
| F3 | Search | F9 | New Chat |
| F4 | Approval cycle | F10 | Clear |
| F5 | Theme picker | F11 | Cancel |
| F6 | Toggle streaming | F12 | Quit |
| Ctrl+S | Shell toggle | Ctrl+G | Personality |

---

## Arc A380 Preprocessing

The Intel Arc A380 runs a Qwen2.5-3B-Instruct model that serves three roles:

### 1. Tool Routing (3s timeout)
Classifies the user message and selects 1-3 tools from the available set. Falls back to keyword-based rule routing on timeout.

### 2. Prompt Compression (5s timeout)
For small-context models (`condensed` prompt size), compresses the personality system prompt from ~2KB to under 500 characters. Cached per personality+model combo.

### 3. Context Compaction Cache (via Redis)
Long conversations are summarized by the Arc model and cached in Redis (TTL 1h). On subsequent messages, the cached summary is used instead of re-summarizing.

```
User Request → Dart Server
  ├── Check Redis: cached context? (sub-ms)
  ├── If miss: Arc A380 compacts conversation (3-5s) → cache in Redis
  ├── Arc A380: select tools (3s)
  ├── Arc A380: compress prompt if condensed model (5s, cached)
  └── RTX 3090: main generation with compact context + selected tools
```

**Graceful degradation**: If Arc or Redis is unreachable, the system falls back to rule-based tool routing and uncompressed prompts. No single point of failure.

---

## Image Pipeline

Image requests go through a dedicated pipeline:

1. **Tool routing** detects image keywords → sends ONLY the `image` tool (not `web`)
2. **MCP image search** queries SearXNG (aggregates Google Images + Bing Images), DDG, and Bing directly
3. **Real CDN URLs** are returned (no hallucinated URLs) plus inline base64 thumbnails
4. **Dedup** via `normalizeImageUrl()` — strips query params, CDN size suffixes (`_300x200`, `_thumb`)
5. **Frontend render** — responsive CSS Grid, click for lightbox, broken images auto-hidden via `onerror`

Images from tool results are collected during streaming and rendered in a single pass on `done`. Markdown image tags in the model's text are only stripped when tool_result already provided the same images (dedup), otherwise preserved.

---

## Personalities

30 built-in personalities across 10 categories:

| Category | Examples |
|----------|---------|
| General | Master orchestrator (unrestricted) |
| Technology | Linux/DevOps, Full-stack programmer, AI/ML, Cybersecurity |
| News & Politics | Political analyst, Military/Defense, Legal |
| Science | Space, Economics, Symbolic logic |
| Entertainment | Anime, Gaming, Retro gaming, Film/TV |
| Creative | Writer, Music, Food/Chef |
| Health | Medical, Fitness, Psychology |
| Business | Finance, Automotive, Sports |
| Humanities | Philosophy, History |
| Special | Model-restricted personalities (dolphin-only) |

Model-restricted personalities are only visible when a compatible model is selected.

---

## CI/CD Pipeline

Two workflow files:

### `ci.yml` (Generated by haskell-ci-orchestrator)
- **Lint**: `ruff check .`
- **Test**: `pytest` — architecture tests
- **Security**: `gitleaks` secret scan
- Runs on self-hosted runners with ephemeral containers

### `dart-ci.yml` (Dart web server)
- **Lint**: `dart format --set-exit-if-changed`, `dart analyze`
- **Test**: `dart test test/dart/`
- **Frontend**: `node -c docker/web/web/app.js` syntax check
- Triggers only on Dart/web file changes

---

## Development

### Makefile Targets

```bash
# Docker stack
make build          # Build all service images
make up             # Start full stack
make down           # Stop and remove containers
make restart        # down + up
make logs           # Follow logs
make smoke          # Health check all services

# Python
make test           # Run pytest suite
make lint           # ruff + mypy

# Dart web server
make dart-get       # Install Dart dependencies
make dart-analyze   # Run dart analyze
make dart-test      # Run Dart tests
make dart-build     # Compile native binary
make dart-run       # Run web server locally
```

### Project Layout

```
aichat/
├── src/aichat/              # TUI source (Textual app)
│   ├── app.py               # Main TUI application
│   ├── ui/                  # Widgets, keybinds, modals
│   ├── themes.py            # Theme registry
│   └── model_labels.py      # Model capability badges
├── lib/                     # Dart web server source
│   ├── router.dart          # HTTP API, SSE streaming, tool execution loop
│   ├── llm_client.dart      # LM Studio client
│   ├── mcp_client.dart      # MCP JSON-RPC client
│   ├── tool_router.dart     # Rule-based + GPU tool routing
│   ├── model_profiles.dart  # Per-model tool tiers and parameters
│   ├── personalities.dart   # 30 AI personalities
│   ├── database.dart        # SQLite conversation storage
│   └── config.dart          # Environment-based configuration
├── bin/server.dart           # Dart server entry point
├── docker/
│   ├── web/                 # Web server Dockerfile + frontend
│   │   ├── Dockerfile
│   │   └── web/             # app.js, style.css, index.html
│   ├── auth/                # Auth proxy Dockerfile + server.py
│   ├── mcp/                 # MCP server (FastAPI, 16 mega-tools)
│   ├── data/                # Data service
│   ├── vision/              # Vision service (OCR, face, video)
│   ├── browser/             # Headless Chromium
│   ├── docs/                # Document processing
│   ├── sandbox/             # Code execution sandbox
│   ├── jupyter/             # Jupyter kernel
│   ├── searxng/             # Meta-search engine
│   ├── inference/           # Intel Arc OpenVINO embeddings
│   └── whatsapp/            # WhatsApp bot
├── test/                    # Tests (Python + Dart)
├── docker-compose.yml       # Full stack definition
├── pubspec.yaml             # Dart dependencies
├── pyproject.toml           # Python project config
├── Makefile                 # Build targets
└── .env                     # Local config (not committed)
```

---

## Replicating This Setup

### Minimal Setup (Single Machine, One GPU)

1. Install [LM Studio](https://lmstudio.ai) and load a chat model
2. Clone this repo, create `.env` with your PostgreSQL/JWT passwords
3. Set `LM_STUDIO_URL=http://localhost:1234` and leave `TOOL_ROUTER_URL` empty
4. `docker compose up -d` — wait for builds
5. Open `http://localhost:8200`, register, and start chatting

### Full Setup (Two GPUs, Network Split)

1. **Inference machine** (e.g., Windows desktop with RTX 3090):
   - Install LM Studio, load your models, expose on `:1234`
2. **Services machine** (e.g., Linux server with Intel Arc):
   - Clone this repo
   - Set up LM Studio with `qwen2.5-3b-instruct` on the Arc GPU, expose on `:1235`
   - Configure `.env` with the inference machine's IP
   - `docker compose up -d`
3. Access the web UI on `:8200` from any device on your network

### Key Design Decisions

- **Vanilla JS frontend**: No React/Vue/Svelte — the entire frontend is ~1200 lines of plain JavaScript. Easy to understand, modify, and deploy without a build step.
- **Buffer-then-render**: Tokens stream into a text buffer; DOM updates once on completion. No innerHTML thrashing.
- **Tool tier system**: Models get only the tools they can reliably use. Fewer tools = fewer hallucinated tool calls = faster responses.
- **Image-only routing**: Image requests bypass `web` search entirely. The `image` tool uses real image search engines (SearXNG/DDG/Bing Images) and returns actual CDN URLs.
- **Arc preprocessing**: A cheap 3B model handles classification/compression so the expensive 20B+ model spends its context window on actual generation.
- **Redis context cache**: Conversation summaries are cached for 1 hour. Repeat messages in long conversations skip re-summarization.

---

## Known Limitations

| Issue | Status |
|-------|--------|
| Arc A380 tool router requires LM Studio running locally on `:1235` | Falls back to rule-based routing when unavailable |
| Some CDNs block image HEAD requests | Backend trusts image tool URLs directly (no HEAD validation) |
| Dolphin model may call `browser` with invalid actions (`fetch_page_images`) | MCP returns error; model retries with valid actions |
| Max 4 images per response | Hardcoded in router.dart; increase `maxImagesPerResponse` if needed |
| WhatsApp QR must be scanned manually at `:8097` | By design — no credential storage |
| Reasoning models (phi-4) fill `max_tokens` with thinking tokens | Use dense instruction models for tool-heavy tasks |

---

## Disclaimer

The author is not responsible for how users use this program. Use at your own risk. All AI inference runs locally — no data is sent to external servers unless you configure external search engines.
