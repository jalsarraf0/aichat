# aichat Platform Architecture

## Overview

aichat is a production-grade MCP (Model Context Protocol) platform that exposes 83+ tools to LLM clients (LM Studio, Claude Desktop, etc.) via a single HTTP/SSE gateway.

## Container Layout

```
┌─────────────────────────────────────────────────────────────┐
│  LM Studio / Claude Desktop / any MCP client                │
│  → http://<host>:8096/sse                                   │
└──────────────────────────┬──────────────────────────────────┘
                           │ MCP JSON-RPC (HTTP + SSE)
                           ▼
        ┌──────────────────────────────┐
        │       aichat-mcp  :8096      │   MCP gateway (FastAPI)
        │  83 tools, orchestration,    │
        │  async job dispatch          │
        └───┬──────┬──────┬──────┬────┘
            │      │      │      │
    ┌───────┘  ┌───┘  ┌───┘  ┌──┘
    ▼          ▼      ▼      ▼
┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐
│aichat-   │ │aichat-   │ │aichat-   │ │aichat-   │
│data :8091│ │vision    │ │docs :8101│ │sandbox   │
│          │ │:8099     │ │          │ │:8095     │
│PostgreSQL│ │video+OCR │ │docs+PDF  │ │code exec │
│memory    │ │FFmpeg    │ │PyMuPDF   │ │non-root  │
│graph     │ │Tesseract │ │pdfminer  │ │          │
│planner   │ │          │ │          │ │          │
│research  │ └──────────┘ └──────────┘ └──────────┘
│jobs      │
└─────┬────┘
      │
 ┌────┘  ┌─────────────────────┐
 ▼       │ Infrastructure      │
┌──────┐ │  aichat-db  :5432   │  PostgreSQL 16
│SQLite│ │  aichat-vector:6333 │  Qdrant
│/data │ └─────────────────────┘
└──────┘

Optional:
  aichat-whatsapp :8097  (WhatsApp bot)
```

## Services

### aichat-data `:8091`
**Replaces:** aichat-database, aichat-memory, aichat-graph, aichat-planner, aichat-researchbox

Single FastAPI process with sub-routers mounted via `APIRouter`:

| Route prefix | Functionality | Storage |
|---|---|---|
| `/` | PostgreSQL REST (articles, images, cache, errors) | PostgreSQL |
| `/memory/*` | Episodic key-value memory store | SQLite WAL |
| `/graph/*` | Knowledge graph (node/edge, NetworkX paths) | SQLite WAL |
| `/planner/*` | DAG-aware task queue, dependency tracking | SQLite WAL |
| `/research/*` | RSS feed discovery and article indexing | SQLite WAL |
| `/jobs/*` | Durable async job system | SQLite WAL |

Volume: `datadb:/data` (consolidated from 3 separate volumes)

### aichat-vision `:8099`
**Replaces:** aichat-video (8099), aichat-ocr (8100)

| Route prefix | Functionality |
|---|---|
| `/` | Video analysis: `/info`, `/frames`, `/thumbnail` (FFmpeg) |
| `/ocr/*` | OCR: image, boxes, path, pdf (Tesseract + poppler) |

Intel Arc A380 GPU passthrough enabled (VA-API for FFmpeg, OpenCL for preprocessing).

### aichat-docs `:8101`
**Replaces:** aichat-docs (8101), aichat-pdf (8103)

| Route prefix | Functionality |
|---|---|
| `/` | Document ingestor: PDF/DOCX/XLSX/PPTX/HTML → Markdown |
| `/pdf/*` | Precise PDF operations: read/edit/fill-form/merge/split (PyMuPDF) |

PDF sub-service is mounted via `app.mount("/pdf", pdf_app)`. OCR is delegated to `aichat-vision:8099/ocr`.

### aichat-sandbox `:8095`
**Replaces:** aichat-toolkit (8095)

Isolated code execution in a non-root container with minimal attack surface. Provides `run_python`, `run_bash`, custom tool management.

### aichat-mcp `:8096`
The MCP gateway. Exposes all tools via:
- `GET  /sse`       — SSE transport for persistent connections
- `POST /mcp`       — JSON-RPC endpoint (tools/list, tools/call)
- `GET  /health`    — Health + tool count
- `GET  /tools`     — Tool manifest

### aichat-whatsapp `:8097` (optional)
WhatsApp bot using Baileys + LM Studio. Connects to MCP for full tool access. Requires QR code scan at `http://localhost:8097`.

---

## Async Job System

The `/jobs/*` API provides durable background job execution:

```
POST  /jobs           → create job (returns {id, status: "queued"})
GET   /jobs/{id}      → poll status (queued→running→succeeded/failed/cancelled)
PATCH /jobs/{id}      → update job (used internally by worker)
POST  /jobs/{id}/cancel → request cancellation
GET   /jobs           → list with ?status=&tool_name=&limit= filters
POST  /jobs/batch     → batch create multiple jobs
```

Job fields: `id`, `tool_name`, `args`, `status`, `priority`, `submitted_at`, `started_at`, `finished_at`, `progress`, `result`, `error`, `logs`, `retry_count`, `max_retries`, `timeout_s`.

MCP tools: `job_submit`, `job_status`, `job_result`, `job_cancel`, `job_list`, `batch_submit`.

---

## Tool Categories (83 tools)

| Category | Count | Tools |
|---|---|---|
| Browser | 9 | screenshot, browser, scroll_screenshot, bulk_screenshot, page_scrape, page_extract, page_images, browser_download_page_images, browser_save_images |
| Database | 6 | db_store_article, db_store_image, db_list_images, db_search, db_cache_store, db_cache_get |
| Memory | 2 | memory_store, memory_recall |
| Graph | 5 | graph_add_node, graph_add_edge, graph_query, graph_path, graph_search |
| Vector | 4 | vector_store, vector_search, vector_delete, vector_collections |
| Planner | 7 | plan_create_task, plan_get_task, plan_complete_task, plan_fail_task, plan_list_tasks, plan_delete_task, orchestrate |
| Jobs | 6 | job_submit, job_status, job_result, job_cancel, job_list, batch_submit |
| Vision/Image | 13 | image_generate, image_edit, image_annotate, image_caption, image_crop, image_diff, image_enhance, image_remix, image_scan, image_search, image_stitch, image_upscale, image_zoom |
| Video | 3 | video_info, video_frames, video_thumbnail |
| OCR | 2 | ocr_image, ocr_pdf |
| Documents | 2 | docs_ingest, docs_extract_tables |
| PDF | 5 | pdf_read, pdf_edit, pdf_fill_form, pdf_merge, pdf_split |
| Research | 2 | researchbox_search, researchbox_push |
| Sandbox | 5 | code_run, create_tool, list_custom_tools, call_custom_tool, delete_custom_tool |
| Web | 4 | web_search, web_fetch, extract_article, smart_summarize |
| Other | 7 | fetch_image, face_recognize, tts, screenshot_search, structured_extract, image_generate, embed_store, embed_search |

---

## Volume Map

| Volume | Service | Contents |
|---|---|---|
| `aichatdb` | aichat-db | PostgreSQL data files |
| `datadb` | aichat-data | SQLite: memory, graph, planner, research, jobs |
| `qdrantdb` | aichat-vector | Qdrant vector collections |
| `whatsappauth` | aichat-whatsapp | WhatsApp session credentials |

---

## Port Map

| Port | Service | Description |
|---|---|---|
| 5432 | aichat-db | PostgreSQL (localhost-only by default) |
| 6333 | aichat-vector | Qdrant HTTP |
| 6334 | aichat-vector | Qdrant gRPC |
| 8091 | aichat-data | Consolidated data service |
| 8095 | aichat-sandbox | Code execution |
| 8096 | aichat-mcp | MCP gateway (primary client endpoint) |
| 8097 | aichat-whatsapp | WhatsApp bot web UI |
| 8099 | aichat-vision | Video + OCR |
| 8101 | aichat-docs | Document ingestor + PDF |

---

## Quick Start

```bash
cp .env.example .env          # edit GPU GIDs if needed
make build                    # build all images
make up                       # start stack
make smoke                    # verify all /health endpoints
make test                     # run full test suite
```

Point LM Studio at: `http://<host-ip>:8096/sse`
