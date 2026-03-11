# CLAUDE.md — aichat

Local-first AI chat platform. Multi-container Docker Compose stack built on Python 3.12+,
Textual TUI, and the Model Context Protocol (MCP). LLM is served by LM Studio at
`host.docker.internal:1234`. No cloud dependency required.

---

## Stack Overview

| Service | Port | Role |
|---|---|---|
| aichat-db | 5432 | PostgreSQL |
| aichat-data | 8091 | FastAPI — articles, memory, graph, planner, RSS, jobs |
| aichat-vision | — | GPU-accelerated image analysis (Intel Arc) |
| aichat-docs | 8101 | Document ingestor |
| aichat-sandbox | — | Code execution sandbox |
| aichat-mcp | 8096 | MCP HTTP/SSE server (96 tools) |
| aichat-whatsapp | 8097 | WhatsApp bot |
| aichat-vector | 6333 | Qdrant vector DB |
| aichat-video | 8099 | FFmpeg video analysis |
| aichat-ocr | 8100 | Tesseract OCR |
| aichat-pdf | 8103 | PDF operations |

Intel Arc A380: pass `/dev/dri` via env vars `INTEL_DRI_DEVICE`, `INTEL_VIDEO_GID` (39),
`INTEL_RENDER_GID` (105). `IMAGE_GEN_BASE_URL` → LM Studio.

---

## Environment

Copy `.env.example` → `.env` before first run. Required: `POSTGRES_PASSWORD`.

---

## Common Commands

```bash
# Start full stack
docker compose up -d

# Build only changed services
make build

# Tail all logs
make logs
# or
docker compose logs -f <service>

# Smoke test (health endpoints)
make smoke

# Full test suite
make test

# Lint (ruff + mypy on docker/**/*.py)
make lint

# Security scan (shellcheck/bandit/safety/semgrep/trivy)
make security-checks

# Regenerate lmstudio-mcp.json from live /tools endpoint
make generate-lmstudio-json

# Stop stack
docker compose down
```

---

## Source Layout

```
docker/          # Per-service Dockerfiles and Python source
src/             # Shared Python library code
scripts/         # Operational helpers
entrypoint/      # Container entrypoint scripts
e2e_api_test.py  # End-to-end API test
```

---

## Development Rules

- All service Python lives under `docker/<service>/`.
- Do not change port assignments without updating `docker-compose.yml` and `ARCHITECTURE.md`.
- `aichat-data` consolidates five former services; do not recreate separate containers for
  memory, graph, planner, researchbox, or database.
- `make lint` and `make test` must pass before any commit.
- The `aichat-toolkit` container mounts `~/.config/aichat/tools` and `~/git` (read-only).
  Do not write to those paths from inside the container.
- Do not hardcode `POSTGRES_PASSWORD` anywhere. Use `${POSTGRES_PASSWORD:?...}`.

---

## Safe Edit Areas

- `docker/<service>/` Python source (lint + test after changes)
- `scripts/` operational helpers
- `.env` (never commit)
- `docker-compose.yml` service definitions (verify with `docker compose config`)

---

## Do Not Touch Without Review

- `entrypoint/` — container startup scripts; wrong changes cause silent boot failures
- `docker/data/` migrations — schema changes require a migration file, not in-place edits
- Port assignments — downstream MCP clients depend on fixed ports

---

## Validation

```bash
docker compose config          # validate compose syntax
make smoke                     # verify all health endpoints respond
make test                      # full pytest suite
make lint                      # zero warnings required
```

---

## Toolchain

| Tool | Path | Version |
|---|---|---|
| python3 | `/usr/bin/python3` | 3.14.3 (Fedora dnf) |
| pip3 | `/usr/bin/pip3` | system |
| Go | `/go/bin/go` | 1.26.1 — not used by this repo |
| Rust | `/usr/bin/rustc` | 1.93.1 — not used by this repo |

Use a project venv (`.venv/`) for all dependency management — never install to the system Python.
