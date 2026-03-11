# AIChat

[![CI](https://github.com/jalsarraf0/aichat/actions/workflows/ci.yml/badge.svg)](https://github.com/jalsarraf0/aichat/actions/workflows/ci.yml)
[![Security CI](https://github.com/jalsarraf0/aichat/actions/workflows/security.yml/badge.svg)](https://github.com/jalsarraf0/aichat/actions/workflows/security.yml)
[![Release](https://github.com/jalsarraf0/aichat/actions/workflows/release.yml/badge.svg)](https://github.com/jalsarraf0/aichat/actions/workflows/release.yml)
[![Latest Release](https://img.shields.io/github/v/release/jalsarraf0/aichat?display_name=tag)](https://github.com/jalsarraf0/aichat/releases)
![Python](https://img.shields.io/badge/python-3.12%2B-blue)
![Platform](https://img.shields.io/badge/platform-Linux-informational)

A **local-first AI chat platform** built on [Textual](https://github.com/Textualize/textual) (terminal UI) + Docker. It connects your LLM (via [LM Studio](https://lmstudio.ai)) to **96 real tools** through the [Model Context Protocol (MCP)](https://modelcontextprotocol.io): web search, image recognition, GPU-accelerated vision, code execution, persistent memory, knowledge graphs, vector search, PDF processing, and more — all running locally with no cloud dependency.

---

## Table of Contents

- [Architecture](#architecture)
- [Services](#services)
- [Requirements](#requirements)
- [Quick Start](#quick-start)
- [Configuration](#configuration)
- [MCP Tools Reference](#mcp-tools-reference)
- [TUI Usage](#tui-usage)
- [Model Selection](#model-selection)
- [CI/CD Pipeline](#cicd-pipeline)
- [Development](#development)
- [Known Limitations](#known-limitations)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────┐
│  Host Machine  (Fedora / Ubuntu / Debian)                               │
│                                                                         │
│  ┌─────────────────────┐     ┌──────────────────────────────────────┐  │
│  │   aichat TUI        │     │   LM Studio  :1234                   │  │
│  │   (Textual / CLI)   │◄───►│   LLM inference  •  embeddings       │  │
│  │                     │     │   (local GPU, no cloud)              │  │
│  └──────────┬──────────┘     └──────────────────────────────────────┘  │
│             │ MCP stdio                                                 │
│             ▼                                                           │
│  ┌──────────────────────────────────────────────────────────────────┐  │
│  │  aichat-mcp  :8096  (MCP HTTP/SSE gateway — 96 tools)           │  │
│  │  FastAPI + httpx  →  orchestrates all downstream services        │  │
│  └────┬──────────┬───────────┬──────────┬──────────┬───────────────┘  │
│       │          │           │          │          │                   │
│       ▼          ▼           ▼          ▼          ▼                   │
│  ┌─────────┐ ┌────────┐ ┌────────┐ ┌──────────┐ ┌────────────────┐   │
│  │aichat-  │ │aichat- │ │aichat- │ │aichat-   │ │aichat-vector   │   │
│  │data     │ │vision  │ │docs    │ │sandbox   │ │(Qdrant) :6333  │   │
│  │:8091    │ │:8099   │ │:8101   │ │:8095     │ │vector search   │   │
│  │memory   │ │OCR     │ │PDF     │ │code run  │ │embeddings      │   │
│  │graph    │ │face    │ │ingest  │ │JS/Python │ └────────────────┘   │
│  │planner  │ │video   │ │search  │ │bash exec │                       │
│  │articles │ │clothing│ │        │ │          │ ┌────────────────┐   │
│  └────┬────┘ └────────┘ └────────┘ └──────────┘ │aichat-searxng  │   │
│       │                                          │:8080 (meta-    │   │
│       ▼                                          │search engine)  │   │
│  ┌────────────┐    ┌──────────────────────┐      └────────────────┘   │
│  │aichat-db   │    │aichat-jupyter  :8098 │                           │
│  │(Postgres)  │    │Jupyter kernel exec   │                           │
│  │:5432       │    └──────────────────────┘                           │
│  └────────────┘                                                        │
└─────────────────────────────────────────────────────────────────────────┘
```

---

## Services

| Service | Internal Port | Purpose |
|---|---|---|
| `aichat-db` | 5432 | PostgreSQL — persistent storage for all services |
| `aichat-vector` | 6333 | Qdrant vector database — semantic search & embeddings |
| `aichat-data` | 8091 | Consolidated data API: memory, articles, graph, planner, jobs, research |
| `aichat-vision` | 8099 | Vision: OCR (Tesseract), face recognition (OpenCV), clothing detection (FashionCLIP), video analysis |
| `aichat-docs` | 8101 | Documents: PDF extraction, document ingestion, full-text search |
| `aichat-sandbox` | 8095 | Isolated code execution: Python, bash, JavaScript (Node.js) |
| `aichat-searxng` | 8080 | Self-hosted meta-search engine (DuckDuckGo, Bing, etc.) |
| `aichat-mcp` | **8096** | MCP HTTP/SSE gateway — the single entry point for all 96 tools |
| `aichat-jupyter` | 8098 | Jupyter kernel — stateful notebook-style code execution |
| `aichat-whatsapp` | 8097 | WhatsApp bot integration (QR scan at `:8097`) |

Only `aichat-mcp` (port **8096**) is exposed to the host by default. All other services communicate over the internal Docker network.

---

## Requirements

| Component | Version / Detail |
|---|---|
| Python | 3.12+ |
| Docker | 24.0+ with Compose v2 |
| LM Studio | Any version with at least one chat model loaded |
| GPU | Optional — Intel Arc/integrated (OpenCL) or NVIDIA; CPU fallback works |
| RAM | 8 GB minimum; 32 GB recommended for large models |

---

## Quick Start

### 1. Clone and configure

```bash
git clone https://github.com/jalsarraf0/aichat.git
cd aichat

# Create your environment file (see Configuration section below)
cat > .env << 'EOF'
IMAGE_GEN_BASE_URL=http://192.168.50.2:1234
IMAGE_GEN_MODEL=liquid/lfm2-24b-a2b
POSTGRES_PASSWORD=aichat
COMPREFACE_DB_PASSWORD=compreface_secret
EOF
```

### 2. Start all services

```bash
docker compose up -d
```

First run builds all images (~5–10 min). Subsequent starts take under 30 seconds.

### 3. Verify the stack is healthy

```bash
docker compose ps                   # all services should show "healthy"
curl http://localhost:8096/health   # {"ok":true,"tools":96}
```

### 4. Install the TUI

```bash
pip install -e ".[dev]"
```

### 5. Launch

```bash
aichat                     # interactive TUI
aichat "your question"     # one-shot chat
aichat --mcp "use tools"   # one-shot with MCP tools
```

---

## Configuration

### `.env` file

```bash
# LM Studio endpoint — where LM Studio is running
IMAGE_GEN_BASE_URL=http://192.168.50.2:1234

# Pin a specific model for ALL LLM operations.
# Empty = auto-select by VRAM (smallest model that fits the GPU).
IMAGE_GEN_MODEL=liquid/lfm2-24b-a2b

# Internal database password
POSTGRES_PASSWORD=aichat

# CompreFace face recognition password
COMPREFACE_DB_PASSWORD=compreface_secret
```

### Full Environment Variable Reference

| Variable | Service | Default | Description |
|---|---|---|---|
| `IMAGE_GEN_BASE_URL` | mcp | `http://192.168.50.2:1234` | LM Studio base URL |
| `IMAGE_GEN_MODEL` | mcp | *(auto)* | Pin a model; empty = auto-select by VRAM score |
| `DATABASE_URL` | all | `http://aichat-data:8091` | Internal data service URL |
| `MEMORY_URL` | mcp | `http://aichat-data:8091/memory` | Memory store endpoint |
| `VECTOR_URL` | mcp | `http://aichat-vector:6333` | Qdrant endpoint |
| `INTEL_GPU` | vision, mcp | `0` | Set `1` to force Intel GPU/OpenCL mode |
| `TOOL_TIMEOUT` | mcp | `30` | Max seconds per tool call |
| `LM_STUDIO_URL` | tui | *(from .env)* | Override LM Studio URL from shell |

---

## MCP Tools Reference

Connect any MCP client to `http://<host>:8096/sse`. The full tool list is available at `http://<host>:8096/mcp` (via `tools/list`).

### Vision

| Tool | Parameters | Description |
|---|---|---|
| `face_recognize` | `path`, `style` (`photo`/`anime`) | Detect and identify faces; use `style=anime` for illustrated images |
| `vision_query` | `path`, `question` | Ask a free-form question about an image |
| `detect_clothing` | `path` | Detect clothing items and attributes (FashionCLIP) |
| `ocr_image` | `path` | Extract text from an image (Tesseract OCR) |
| `image_describe` | `path` | Natural-language description of an image |
| `image_caption` | `path` | Short caption via vision-language model |

### Browser & Web

| Tool | Parameters | Description |
|---|---|---|
| `web_search` | `query`, `num_results` | Meta-search via SearXNG (DuckDuckGo, Bing, etc.) |
| `image_search` | `query`, `num_results` | Image search with vision verification |
| `web_fetch` | `url`, `format` (`text`/`markdown`) | Fetch and clean page content |
| `screenshot` | `url` | Full-page screenshot via headless Chromium |

### Code Execution

| Tool | Parameters | Description |
|---|---|---|
| `code_run` | `code`, `language` (`python`/`bash`) | Execute code in isolated sandbox |
| `run_javascript` | `code` | Execute JavaScript via Node.js |
| `jupyter_exec` | `code`, `kernel` | Run code in stateful Jupyter kernel |

### Memory & Knowledge

| Tool | Parameters | Description |
|---|---|---|
| `memory_store` | `key`, `value` | Persist a key-value fact |
| `memory_recall` | `key` | Retrieve a stored fact by key |
| `memory_list` | *(none)* | List all stored keys |
| `memory_delete` | `key` | Delete a stored fact |
| `vector_search` | `query`, `top_k`, `collection` | Semantic search over vector store |
| `vector_collections` | *(none)* | List all Qdrant collections |
| `graph_add_node` | `id`, `labels`, `properties` | Add a node to the knowledge graph |
| `graph_query` | `id` | Query graph neighbors by node ID |

### Documents & Media

| Tool | Parameters | Description |
|---|---|---|
| `pdf_extract` | `path` | Extract text and structure from a PDF |
| `docs_search` | `query` | Full-text search over ingested documents |
| `youtube_transcript` | `url` | Fetch transcript of a YouTube video |

### LLM Utilities

| Tool | Parameters | Description |
|---|---|---|
| `smart_summarize` | `content`, `style` (`brief`/`bullets`/`detailed`), `max_words` | Summarize text via local LLM |
| `structured_extract` | `content`, `schema_description` | Extract structured data from text |
| `inline_chat` | `prompt`, `model` | Direct LM Studio query |
| `smart_translate` | `text`, `target_language` | Translate text |
| `plan_task` | `task`, `max_steps`, `context` | Generate a multi-step execution plan |

### Database

| Tool | Parameters | Description |
|---|---|---|
| `db_search` | `query` | Full-text article search |
| `db_query` | `sql` | Read-only SQL query against the aichat database |

### Orchestration

| Tool | Parameters | Description |
|---|---|---|
| `orchestrate` | `steps` (array) | Run a multi-step workflow; parallel steps via `asyncio.gather` |
| `plan_create_task` | `title`, `description` | Create a planner task |
| `get_system_instructions` | *(none)* | Return orchestration guidelines for the active LLM |

---

## TUI Usage

```
aichat [OPTIONS] [MESSAGE]

Options:
  --mcp TEXT      One-shot query with MCP tools enabled
  --concise       Enable concise response mode
  --model TEXT    Override the model for this session
```

### Keyboard Shortcuts

| Key | Action |
|---|---|
| `Enter` | Send message |
| `Shift+Enter` | New line in input |
| `Ctrl+C` / `Ctrl+Q` | Quit |
| `Ctrl+L` | Clear conversation |
| `Ctrl+M` | Toggle MCP tools on/off |
| `Ctrl+T` | Toggle concise mode |
| `F1` | Help |

### How the LLM Uses Tools

When MCP is enabled, the LLM automatically:

1. Calls `get_system_instructions` to learn what tools are available
2. For complex multi-step tasks, calls `plan_task` to generate an ordered execution plan
3. Executes each step in the plan, passing prior results forward as `{{step_N.result}}`
4. For simple single-tool tasks, calls the tool directly

Example plan output for "Search for Klukai from Girls Frontline 2 and summarize":

```
PLAN: Klukai is a character from the Girls' Frontline series.
STEPS (2 total):
  1. web_search({"query":"Klukai Girls Frontline 2","num_results":5})
     ← Find relevant information about Klukai.
  2. smart_summarize({"content":"{{step_1_result}}","style":"bullets"})
     ← Summarize the search results.

EXECUTE: call each tool above in step order.
```

---

## Model Selection

The MCP server auto-selects the best available LM Studio model by **VRAM score** (smaller = faster for most tasks):

| Model Size | VRAM Score | Priority |
|---|---|---|
| ≤ 14B parameters | 0 | Highest (fastest) |
| ≤ 30B parameters | 1 | |
| ≤ 36B parameters | 2 | |
| > 36B parameters | 3 | Lowest |
| Tiny/nano (no size tag) | 4–5 | Last resort |

MoE models are scored by **total** parameter count. `lfm2-24b-a2b` = 24B total / 2B active — scores as 24B, not 2B.

To pin a specific model for all operations:

```bash
# .env
IMAGE_GEN_MODEL=liquid/lfm2-24b-a2b
```

### Benchmarks (RTX 3090 24 GB VRAM)

| Model | Speed | Tests Passed | Notes |
|---|---|---|---|
| `liquid/lfm2-24b-a2b` | 42–85 t/s | 6/6 | **Recommended** — fastest MoE |
| `ibm/granite-4-h-tiny` | 8–17 t/s | 6/6 | Good fallback |
| `dolphin-mistral-24b-venice-edition` | 27–34 t/s | 6/6 | Dense 24B alternative |
| `mistralai/devstral-small-2-2512` | ~8 t/s | 6/6 | Slow for daily use |
| `google/gemma-3-27b` | ~8 t/s | 6/6 | Slow for daily use |
| `bytedance/seed-oss-36b` | timeout | 0/6 | Too large for 24 GB |

---

## CI/CD Pipeline

The pipeline is defined in `.github/workflows/ci.yml` and runs on self-hosted Docker runners. Every job uses **ephemeral containers** — each container is started fresh, used for its single job, and torn down immediately after with `if: always()`.

```
push / PR
    │
    ├─► cancel-stale    ← cancels any queued/running duplicate runs for this ref
    │
    ├─► static          ← ruff lint + architecture tests (no Docker, ~30s)
    │
    ├─► regression      ← 1 ephemeral container per test file (25 parallel jobs)
    │   ├─ test_tui                    ┐
    │   ├─ test_tool_args              │  Each job:
    │   ├─ test_streaming              │  • pip install
    │   ├─ test_compaction             │  • pytest <test_file>.py
    │   ├─ test_database_tools         │  • teardown containers
    │   ├─ test_vision_tools           ┘
    │   └─ … (25 total, all parallel)
    │
    ├─► build-images    ← 1 build container per service (6 parallel jobs)
    │   ├─ aichat-data      → image tagged :ci-<sha> + :cache
    │   ├─ aichat-vision    → layer cache reused from previous run
    │   ├─ aichat-docs
    │   ├─ aichat-sandbox
    │   ├─ aichat-mcp
    │   └─ aichat-jupyter
    │
    ├─► smoke           ← full stack up → smoke tests → full stack down
    │
    ├─► package         ← build wheel + PyInstaller binary
    │
    ├─► sbom            ← 1 container per service (6 parallel)
    │   └─ SPDX + CycloneDX SBOM + grype vulnerability scan
    │       (ephemeral :ci-<sha> image removed after each SBOM job)
    │
    └─► cleanup         ← runs always, even on failure
        • removes all :ci-<sha> image tags
        • prunes dangling layers
        • retains :cache tags for next run's layer reuse
```

### Key CI Design Decisions

- **1 container per job**: Each regression test runs in complete isolation — no shared state, no port conflicts between parallel tests.
- **`cancel-in-progress: true`** on the concurrency group: a new push immediately cancels any in-progress run for the same branch, preventing queue buildup.
- **`styfle/cancel-workflow-action`**: cancels stale *queued* runs that the `cancel-in-progress` setting can't reach.
- **`:cache` tag persistence**: Docker layer cache is kept between runs on the same self-hosted runner, making image builds much faster after the first run.
- **`if: always()` teardown**: containers are removed even when a test fails — no orphaned containers accumulate on the runner.

---

## Development

### Install development dependencies

```bash
pip install -e ".[dev]"
```

### Run tests

```bash
# All unit tests (no Docker required)
pytest -q -m "not smoke" tests/

# Single test module
pytest tests/test_tool_args.py -v --tb=short

# Smoke tests (requires full Docker stack)
pytest -m smoke tests/test_smoke.py -v

# Lint
ruff check docker/ src/ tests/
```

### Rebuild a specific service

```bash
docker compose build aichat-mcp
docker compose up -d aichat-mcp
docker compose logs -f aichat-mcp
```

### Run the E2E MCP test suite (19 tools)

```bash
# Requires: docker compose up -d
python3 /tmp/mcp_e2e_test.py
# Expected: ✓ 19 passed   ✗ 0 failed
```

### Project Layout

```
aichat/
├── src/aichat/           # TUI source (Textual app + MCP client)
├── docker/
│   ├── mcp/              # MCP server (FastAPI, 96 tools)
│   │   ├── app.py        # ~9,300 lines — all tool implementations
│   │   └── Dockerfile
│   ├── data/             # Data service (PostgreSQL REST API)
│   ├── vision/           # Vision service (OCR, face, clothing)
│   ├── docs/             # Document service (PDF, search)
│   ├── sandbox/          # Code execution sandbox
│   └── jupyter/          # Jupyter kernel service
├── tests/                # 31 test modules, 940+ tests
├── .github/workflows/    # CI/CD pipeline definitions
├── docker-compose.yml    # Full stack definition
└── .env                  # Local configuration (not committed)
```

---

## Known Limitations

| Issue | Status |
|---|---|
| `screenshot` tool requires a separate `human_browser` container | Configure `BROWSER_URL` if you have it running; otherwise tool returns a connection error (non-fatal) |
| `ocr_image` returns 0 words on some test images | OCR engine limitation on low-resolution or non-text images; real documents work correctly |
| lfm2 generates 1–2 plan steps (not always 3) in some runs | MoE non-determinism; plans are valid and executable |
| WhatsApp QR must be scanned manually at `:8097` | By design — no credential storage |
| Reasoning models (phi-4, magistral) fill `max_tokens` with thinking tokens | These models are not suitable for tool use; use dense instruction models |

---

## Disclaimer

The author is not responsible for how users use this program. Use at your own risk. All AI inference runs locally — no data is sent to external servers.
