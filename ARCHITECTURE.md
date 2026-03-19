# aichat Platform Architecture

## Overview

aichat is a production-grade, local-first AI workstation platform exposing 100+
tools to LLM clients (LM Studio, Dartboard, and other MCP-compatible apps) via a single MCP
HTTP/SSE gateway.  All persistence is PostgreSQL-backed with a forward-only
migration system.

## Container Layout

```
┌──────────────────────────────────────────────────────────────┐
│  LM Studio / Dartboard (:8200) / MCP clients          │
│  → http://<host>:8096/sse   (MCP JSON-RPC over SSE)          │
└────────────────────────┬─────────────────────────────────────┘
                         │ MCP JSON-RPC (HTTP + SSE)
                         ▼
       ┌─────────────────────────────────┐
       │       aichat-mcp  :8096         │   MCP gateway (FastAPI)
       │  16 mega-tools, orchestrator,   │
       │  source strategy, streaming,    │
       │  async job dispatch             │
       └───┬──────┬──────┬──────┬───┬───┘
           │      │      │      │   │
   ┌───────┘  ┌───┘  ┌───┘  ┌──┘   └──────────────┐
   ▼          ▼      ▼      ▼                      ▼
┌──────────┐┌──────────┐┌──────────┐┌──────────┐┌──────────┐
│aichat-   ││aichat-   ││aichat-   ││aichat-   ││aichat-   │
│data :8091││vision    ││docs :8101││sandbox   ││browser   │
│          ││:8099     ││          ││:8095     ││:8104     │
│PostgreSQL││video+OCR ││docs+PDF  ││code exec ││Playwright│
│memory    ││CLIP      ││pdfminer  ││non-root  ││headless  │
│graph     ││YOLOv8n   ││openpyxl  ││          ││Chromium  │
│planner   ││FFmpeg    ││          ││          ││          │
│research  ││Tesseract ││          ││          ││          │
│jobs      │└──────────┘└──────────┘└──────────┘└──────────┘
│embeddings│
│batch ops │
└─────┬────┘
      │
      ▼
┌────────────────────────────────────────────┐
│ Infrastructure                              │
│  aichat-db     :5432   PostgreSQL 16        │
│  aichat-vector :6333   Qdrant               │
│  aichat-minio  :9001/9002   MinIO S3        │
│  aichat-searxng (internal 8080)  SearXNG    │
│  aichat-jupyter (internal 8098)  Kernels    │
└─────────────────────────────────────────────┘

Optional:
  aichat-whatsapp :8097  (WhatsApp bot)
```

## Services

### aichat-data `:8091`

Consolidated data service — all persistence is PostgreSQL-backed (no SQLite).

**Routes:**
- `/` — articles, images, cache, errors (PostgreSQL)
- `/memory/*` — key-value memory with TTL and compaction
- `/research/*` — RSS feed discovery and ingest
- `/graph/*` — knowledge graph (nodes, edges, paths, search)
- `/planner/*` — dependency-aware task queue
- `/jobs/*` — durable async job system
- `/embeddings/*` — embedding store with cosine similarity
- `/batch/*` — batch write operations

**Migration system:** Numbered SQL files in `docker/data/migrations/` tracked by
`_migrations` table.  Migrations run automatically at container startup.

**Compaction:** Background auto-purge of expired memory entries and terminal jobs
older than 7 days.  Manual compaction via `POST /memory/compact`.

### aichat-mcp `:8096`

MCP HTTP/SSE gateway — the primary interface for all LLM clients.

**Module structure:**
```
docker/mcp/
    app.py              — FastAPI app, CORS, health, lifespan
    helpers.py          — text_block(), json_or_err(), service URLs, constants
    orchestrator.py     — intent classification, bounded concurrency, resource governance
    source_strategy.py  — configurable news source preferences
    system_prompt.py    — production default system prompt
    search/
        searxng.py      — SearXNG multi-instance search with failover
        parsers.py      — DDG/Bing/Google HTML link extractors
        normalize.py    — query normalization, URL scoring, NSFW filter
    handlers/           — one file per mega-tool category (web, browser, image, etc.)
    imaging/            — image processing (rendering, hashing, GPU, face detection)
```

**16 mega-tools:** web, browser, image, document, media, data, memory, knowledge,
vector, code, custom_tools, planner, jobs, research, think, system

**Orchestrator:** First-class intent classification → routing → bounded concurrency
→ resource governance → progress streaming to Dartboard.

**Source strategy:** Configurable via `SOURCE_PROFILE` env var.  Default profile
prefers right-leaning sources for news discovery, neutral/primary for verification.

### aichat-vision `:8099`

Consolidated media analysis service.

**Routes:**
- `/` — video analysis (FFmpeg, Intel Arc VA-API HW accel)
- `/ocr/*` — Tesseract OCR (image, path, boxes, PDF)
- `/clip/*` — CLIP ViT-B/32 ONNX embeddings (512-dim)
- `/detect/*` — YOLOv8n object/human detection (80 COCO classes)
- `/transcode` — HW-accelerated video transcoding

### aichat-docs `:8101`

Document ingestion and PDF operations.

**Routes:**
- `/` — document ingestor (PDF, DOCX, XLSX, PPTX, HTML, TXT → Markdown)
- `/pdf/*` — PDF read, edit, form fill, merge, split (OCR delegated to vision)

### aichat-sandbox `:8095`

Dynamic custom tool host.  Python tools in `/data/tools` are discovered and
reloaded on each request (2-second TTL cache).

### aichat-searxng (internal `:8080`)

Self-hosted SearXNG meta-search engine.  Proxies Google, Bing, DDG, Brave —
no API keys, no bot detection.

**Configuration:** `docker/searxng/settings.yml`, `docker/searxng/limiter.toml`

**Integration:** The MCP gateway's `search/searxng.py` module queries SearXNG
with multi-instance failover (11 public instances + local).  The SearXNG
container itself is not modified by this codebase.

### aichat-browser `:8104`

Headless Chromium via Playwright for agentic browsing — navigate, click, type,
screenshot, scrape, tab management.

### aichat-jupyter (internal `:8098`)

Persistent Python kernels for stateful code execution.  Variables and DataFrames
survive between calls.

### aichat-minio `:9001`/`:9002`

S3-compatible object store for the image pipeline.

### aichat-whatsapp `:8097` (optional)

WhatsApp bot.  Visit `http://localhost:8097` to scan QR and pair.

## PostgreSQL Schema

All tables managed via numbered migrations in `docker/data/migrations/`:

| Migration | Tables |
|---|---|
| 001 | articles, images, cache, errors |
| 002 | memory (key-value with TTL) |
| 003 | graph_nodes, graph_edges |
| 004 | tasks (planner) |
| 005 | jobs (async execution) |
| 006 | embeddings |
| 007 | compaction_log, memory_summaries |
| 008 | tool_executions |
| 009 | fetch_cache (web fetch with TTL) |

## Source Layout

```
docker/
    data/           — data service (app.py, migrate.py, sqlite_to_pg.py, migrations/)
    mcp/            — MCP gateway (modular: app.py, helpers, search/, handlers/, imaging/)
    vision/         — vision service (video, OCR, CLIP, YOLO)
    docs/           — document service (ingestion, PDF)
    sandbox/        — code sandbox
    browser/        — Playwright browser automation
    jupyter/        — persistent Python kernels
    whatsapp/       — WhatsApp bot
    searxng/        — SearXNG config (DO NOT MODIFY)
src/
    aichat/         — Textual TUI application
tests/              — pytest test suite
```

## Operational Commands

```bash
docker compose up -d                    # start full stack
docker compose -f docker-compose.yml -f docker-compose.ports.yml up -d  # with host ports
make build                              # build all service images
make smoke                              # health endpoint check
make test                               # full pytest suite
make lint                               # ruff + mypy
make generate-lmstudio-json             # regenerate lmstudio-mcp.json from live /tools
docker compose logs -f <service>        # tail service logs
```

## Dartboard Integration

Dartboard (Dart/shelf web app at `:8200`) connects to aichat-mcp at `:8096` via
MCP JSON-RPC.  The orchestrator streams progress notifications via SSE using the
standard MCP `notifications/progress` method.

## GPU Acceleration

Intel Arc A380: `/dev/dri` device passthrough with GIDs 39 (video) and 105 (render).
Used by: aichat-vision (FFmpeg VA-API, OpenCL OCR preprocessing), aichat-mcp
(OpenCV face detection, image processing).
