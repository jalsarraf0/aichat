# AIChat

A local-first AI chat TUI (Terminal User Interface) built with [Textual](https://github.com/Textualize/textual). Connects to a local LLM via [LM Studio](https://lmstudio.ai) with a full suite of Docker-backed tools: real Chromium browsing (Playwright), image search and generation pipeline, GPU-accelerated image processing, shell execution, RSS/research, persistent memory, parallel thinking, conversation history, and a WhatsApp bot — all accessible through the Model Context Protocol (MCP).

---

## Architecture

```
┌────────────────────────────────────────────────────────────────────┐
│                          Host Machine                              │
│                                                                    │
│  aichat TUI (Textual)                                              │
│  ├── LM Studio  http://localhost:1234  (LLM / embeddings / TTS)   │
│  └── MCP stdio  (aichat mcp)                                       │
│                                                                    │
│  Docker services:                                                  │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐            │
│  │ aichat-db    │  │ aichat-      │  │ aichat-      │            │
│  │ postgres:16  │  │ database     │  │ researchbox  │            │
│  │ :5432        │  │ :8091        │  │ :8092        │            │
│  └──────┬───────┘  └──────┬───────┘  └──────┬───────┘            │
│         │                 │                  │                    │
│  ┌──────┴───────┐  ┌──────┴───────┐  ┌──────┴───────┐            │
│  │ aichat-      │  │ aichat-mcp   │  │ aichat-      │            │
│  │ memory       │  │ :8096 ◄──MCP │  │ toolkit      │            │
│  │ :8094        │  │  orchestrate │  │ :8095 + GPU  │            │
│  └──────────────┘  └──────────────┘  └──────────────┘            │
│                                                                    │
│  ┌──────────────┐  ┌──────────────┐                               │
│  │ aichat-      │  │ human_browser│                               │
│  │ whatsapp     │  │ :7081 + noVNC│                               │
│  │ :8097        │  │ :36411       │                               │
│  └──────────────┘  └──────────────┘                               │
└────────────────────────────────────────────────────────────────────┘
```

---

## Requirements

| Requirement | Details |
|---|---|
| Python | 3.12 or newer |
| LM Studio | Running locally or remotely (configurable via `base_url`) |
| Docker + Compose | Required for all tool services |
| `docker` group | Your user must be in the `docker` group |
| human_browser container | Required for all browser/screenshot/image tools (see below) |

---

## Installation

```bash
git clone https://github.com/jalsarraf0/aichat.git
cd aichat
bash install.sh
```

The installer:
1. Creates a Python virtualenv at `~/.local/share/aichat/venv`
2. Installs aichat into it
3. Creates a launcher at `~/.local/bin/aichat`
4. Creates `~/.config/aichat/tools/` and `~/git/`
5. Runs `docker compose up -d --build` to start all tool services

Add to your shell profile if not already present:

```bash
export PATH="$HOME/.local/bin:$PATH"
```

Then launch:

```bash
aichat
```

## Uninstall

```bash
bash uninstall.sh
```

Removes the virtualenv, launcher, and brings down Docker containers with `docker compose down --remove-orphans`. The PostgreSQL data volume (`aichatdb`) is preserved so stored articles and web cache survive reinstalls.

---

## Configuration

Config is stored at `~/.config/aichat/config.yml` and is created automatically on first run.

```yaml
base_url: http://localhost:1234        # LM Studio API endpoint
model: mistralai/magistral-small-2509  # Model ID (auto-selected if not found)
theme: cyberpunk                       # cyberpunk | dark | light | synth
approval: AUTO                         # AUTO | ASK | DENY
shell_enabled: true                    # Enable/disable shell tool
concise_mode: false                    # Shorter responses
active_personality: linux-shell-programming
context_length: 35063                  # Model context window (tokens); history trimmed to fit
max_response_tokens: 4096              # Tokens reserved for the assistant response
compact_model: ""                      # Model for compaction (defaults to main model)
tool_result_max_chars: 8000            # Max characters kept per tool result
rag_recency_days: 30                   # Days window for date-weighted RAG scoring
thinking_enabled: false                # Enable parallel thinking chains
thinking_paths: 3                      # Number of parallel reasoning chains
thinking_model: ""                     # Model for thinking (defaults to main model)
```

**Override the LM Studio URL** without editing the file:

```bash
LM_STUDIO_URL=http://localhost:1234 aichat
```

---

## Interface Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Header: model name | approval mode | streaming | THK:ON        │
├────────────────────────────────┬────────────────────────────────┤
│  Transcript (chat bubbles)     │  Tools panel (tool output log) │
│                                │  Sessions panel                │
├────────────────────────────────┴────────────────────────────────┤
│  Prompt input                                                   │
├─────────────────────────────────────────────────────────────────┤
│  Keybind bar: F1..F12  ^S  ^G | CTX%  ⟳N tools                 │
└─────────────────────────────────────────────────────────────────┘
```

- **Transcript**: Chat messages rendered as Markdown bubbles. Scroll with mouse wheel or PageUp/PageDown.
- **Tools panel**: Raw tool output streams here in real time.
- **Sessions panel**: Active shell sessions.
- **Prompt input**: Enter key sends, Shift+Enter inserts a newline.
- **CTX%**: Live context window utilisation indicator.
- **⟳N**: Number of tool calls in the current turn.

---

## Keybindings

| Key | Action |
|-----|--------|
| `Enter` | Send message |
| `Shift+Enter` | Insert newline |
| `PageUp` / `PageDown` | Scroll transcript |
| `Mouse wheel` | Scroll transcript |
| `F1` | Help (show keybinds) |
| `F2` | Model picker |
| `F3` | Search transcript |
| `F4` | Cycle approval mode (AUTO → ASK → DENY) |
| `F5` | Theme picker |
| `F6` | Toggle streaming on/off |
| `F7` | Sessions |
| `F8` | Settings |
| `F9` | New chat (archives context to `/tmp/context`) |
| `F10` | Clear transcript |
| `F11` | Cancel active streaming response |
| `F12` | Quit |
| `Ctrl+S` | Toggle shell access |
| `Ctrl+G` | Personality / persona menu |

---

## Slash Commands

### Chat

| Command | Description |
|---------|-------------|
| `/help` | Show keybindings |
| `/new` | Start a new chat (archives previous context to `/tmp/context`) |
| `/clear` | Clear the transcript display |
| `/copy` | Copy the last assistant response to clipboard |
| `/export` | Export the current chat to a Markdown file |
| `/concise on\|off` | Enable/disable concise (shorter) responses |
| `/verbose` | Alias for `/concise off` |
| `/shell on\|off` | Enable/disable shell tool |
| `!<cmd>` | Run a shell command directly (when shell is enabled) |

### Context & History

| Command | Description |
|---------|-------------|
| `/ctx` | Show context window usage summary |
| `/ctx graph` | ASCII bar chart of per-turn token usage |
| `/compact` | Summarise old turns to free context space |
| `/compact dry` | Preview what would be summarised (no change) |
| `/compact history` | Show all past compaction summaries |
| `/history` | Browse past conversation sessions |
| `/sessions` | List saved sessions with title + date |
| `/context` | Show RAG context used for the current session |
| `/fork` | Clone the current conversation into a new session |

### Thinking

| Command | Description |
|---------|-------------|
| `/think <question>` | Run parallel thinking chains on a question, synthesise best answer |
| `/thinking on\|off` | Toggle auto-thinking on every submission |
| `/thinking paths N` | Set number of parallel reasoning chains (1–8) |
| `/thinking model <id>` | Use a specific model for thinking chains |

### RSS & Research

| Command | Description |
|---------|-------------|
| `/rss <topic>` | Fetch latest stored RSS items for a topic |
| `/rss store <topic>` | Search for feeds and ingest items for a topic |
| `/rss ingest <topic> <url>` | Ingest a specific RSS feed URL |
| `/rss ingest` | Open a modal to enter topic + feed URL |

### Project / Coding

| Command | Description |
|---------|-------------|
| `/vibecode <project>` | Switch working directory to `~/git/<project>` for coding tasks |

### Personality

| Command | Description |
|---------|-------------|
| `/persona` | Open the personality selection menu |
| `/personality` | Alias for `/persona` |
| `/persona list` | List all available personalities |
| `/persona add` | Add a custom personality |

### Memory

| Command | Description |
|---------|-------------|
| `/memory store <key> <value>` | Store a value in persistent memory |
| `/memory recall <key>` | Recall a stored value |
| `/memory recall` | List all stored memory keys |

### Tools

| Command | Description |
|---------|-------------|
| `/tool list` | List all custom (dynamic) tools |
| `/fetch <url>` | Fetch a web page and show its content |

---

## MCP Tool Inventory (49 tools)

All tools are exposed via both the MCP HTTP server (`aichat-mcp` at port 8096) and the stdio MCP server (`aichat mcp`).

### Browser Tools — Real Chromium (Playwright)

All browser operations run inside the `human_browser` Docker container via a Playwright/FastAPI server (v18) that is auto-deployed and upgraded on first use.

**Setup**: Start the `human_browser` container once:
```bash
docker run -d --name human_browser \
  --shm-size=2g \
  -p 36411:6080 \
  -v /docker/human_browser/workspace:/workspace \
  -v /docker/human_browser/home:/home/ai \
  human-browser:latest
```

The browser server (v18) features:
- **15-section stealth JS**: removes `webdriver` flag, spoofs `navigator.plugins`, canvas fingerprint noise (per-session XOR seed), WebGL GPU string, AudioContext noise, screen geometry, `deviceMemory`, media devices stub — bypasses Cloudflare Turnstile, PerimeterX, DataDome
- **Human-like mouse movement**: real `mouse.move()` events after navigation
- **Route-based ad blocking**: 30+ ad/tracker domains aborted before load
- **Scroll jitter**: randomised inter-step timing (0.6–1.5×)
- **Screenshot block-retry**: Cloudflare/captcha detection with automatic context rotation
- **Browser crash recovery**: full Chromium restart on WebSocket close or OOM
- **GPU acceleration**: `BrowserGpuConfig` class dynamically enables VA-API hardware decode when `/dev/dri` is available

The noVNC web UI is accessible at `http://localhost:36411` to watch the browser live.

| Tool | Description |
|------|-------------|
| `screenshot` | Navigate to a URL and take a screenshot. `find_text` clips to a text region; `find_image` clips to a specific `<img>` element. Returns image inline. |
| `browser` | Navigate/read/click/fill/eval on the live page. Session persists between calls. |
| `scroll_screenshot` | Full-page screenshot assembled from multiple viewport-height scrolls. |
| `bulk_screenshot` | Screenshot multiple URLs in parallel. |
| `fetch_image` | Download an image by URL and return it inline; retries on 429. |
| `screenshot_search` | Web search + screenshot the top result pages. |
| `browser_save_images` | Download a list of image URLs to the workspace. |
| `browser_download_page_images` | Download all images found on a page to the workspace. |
| `page_scrape` | Full-page text extraction with lazy-scroll rendering. Supports `include_links`. |
| `page_images` | Extract all image URLs from a live page using 7 strategies (img src, srcset, og:image, background-image, picture, data-src, lazy-load attributes). |

### Web Tools

| Tool | Description |
|------|-------------|
| `web_fetch` | Fetch a URL and return clean readable text (HTML stripped). `max_chars` up to 16,000. |
| `web_search` | DuckDuckGo web search with three-tier fallback. Returns titles + snippets. |
| `extract_article` | Fetch a URL and extract the main article body (readability-style). |
| `page_extract` | Structured data extraction from a live page. |
| `image_search` | Multi-tier image search with diversity guarantee. Returns up to `count` images (default 4). Supports `offset` for pagination. |

#### `image_search` — How it works

- **Query expansion**: abbreviations auto-expand (`gfl2` → "Girls Frontline 2", `hsr` → "Honkai Star Rail")
- **Tier 1** — DDG web search → `page_images` on top article pages (1 scroll each)
- **Tier 2** — DuckDuckGo Images page → decodes proxy URLs
- **Tier 3** — Bing Images → different CDN corpus for variety
- **DB-first fast path**: returns cached confirmed images if enough exist in the database (~25× faster)
- **Perceptual hash dedup**: dHash (9×8) + Hamming distance < 8 removes near-duplicate images
- **Vision confirm**: optional LLM vision check on each candidate (disable with `IMAGE_VISION_CONFIRM=false`)
- **Domain cap**: max 2 images per domain per call
- **Cross-call dedup**: seen URLs stored in `aichat-memory` with 1-hour TTL

### Image Pipeline Tools

All image tools operate on files in the browser workspace (`/docker/human_browser/workspace`). Pass filenames between tools.

| Tool | Description |
|------|-------------|
| `image_crop` | Extract a rectangular region (x, y, width, height). |
| `image_zoom` | Enlarge a portion of an image with padding context. |
| `image_scan` | Pass the image to the vision model for OCR / detailed reading. |
| `image_enhance` | Adjust brightness, contrast, and saturation. |
| `image_stitch` | Combine multiple images side-by-side or in a grid. |
| `image_diff` | Pixel-level difference highlight between two images (ImageChops). |
| `image_annotate` | Draw bounding boxes and labels on an image (ImageDraw). |
| `image_upscale` | LANCZOS upscale (1.1–8.0×, default 2×) with optional sharpening. EXIF rotation applied; output capped at 8192 px per side. |
| `image_generate` | Text-to-image via LM Studio `/v1/images/generations`. Requires a loaded image-gen model. |
| `image_edit` | Image-to-image / inpainting via `/v1/images/edits`. |
| `image_remix` | Generate creative variants of an uploaded image using the vision+generate pipeline. |

**Pipeline examples:**
```
# Read fine text in a generated image
image_generate → image_upscale(scale=4) → image_scan

# Zoom in on a UI element
screenshot → image_crop → image_zoom → image_scan

# Compare two generated variants
image_generate × 2 → image_stitch → image_diff
```

### Database / Cache Tools

| Tool | Description |
|------|-------------|
| `db_store_article` | Store an article (title, url, body, topic) in PostgreSQL. |
| `db_search` | Full-text search across stored articles. Supports `offset` and `summary_only`. |
| `db_cache_store` | Store a key-value entry in the web cache (with optional TTL). |
| `db_cache_get` | Retrieve a cached entry by key. |
| `db_store_image` | Store an image URL + metadata (subject, description, phash, quality_score) in the database. |
| `db_list_images` | List stored images, optionally filtered by topic. |

### Memory Tools

| Tool | Description |
|------|-------------|
| `memory_store` | Store a key-value fact that persists across sessions. Supports `ttl_seconds`. |
| `memory_recall` | Recall a stored value. Supports SQL `LIKE` patterns for key matching. |

### Research Tools

| Tool | Description |
|------|-------------|
| `researchbox_search` | Find RSS feed sources for a given topic. |
| `researchbox_push` | Fetch a feed URL and store its items under a topic label. |

### Toolkit Tools

| Tool | Description |
|------|-------------|
| `create_tool` | Write and deploy a new custom Python tool at runtime. |
| `list_custom_tools` | List all custom tools you have created. |
| `delete_custom_tool` | Permanently remove a custom tool. |
| `call_custom_tool` | Call a custom tool by name with arguments. |

### LM Studio Tools

| Tool | Description |
|------|-------------|
| `tts` | Convert text to speech via LM Studio. Audio saved to workspace. |
| `embed_store` | Compute and store embeddings for a text chunk. |
| `embed_search` | Cosine-similarity search over stored embeddings. |
| `code_run` | Execute a Python code snippet in an isolated subprocess (120s cap). |
| `smart_summarize` | Summarise long text using the loaded model. |
| `image_caption` | Generate a text caption for an image using the vision model. |
| `structured_extract` | Extract structured data from text according to a JSON schema. |

### System Tools

| Tool | Description |
|------|-------------|
| `get_errors` | Retrieve recent tool errors from the server log. |
| `shell_exec` | Run a bash command on the host machine (stdio only). |

---

## `orchestrate` — Multi-Step Workflow Tool

The `orchestrate` tool lets an LLM compose a complete multi-step workflow in a single call, rather than issuing tool calls one-by-one across multiple turns.

### Key features

- **Automatic parallelism** — steps with no `depends_on` run concurrently via `asyncio.gather()`
- **Dependency management** — steps with `depends_on` wait for their prerequisites
- **Result interpolation** — inject an earlier step's output into a later step's args using `{step_id.result}`
- **Stop-on-error** — optionally abort remaining steps if any step fails
- **Structured report** — returns a formatted report with each step's label, status (`OK`/`FAILED`), timing, and a result preview

### Step schema

```json
{
  "id":         "unique_step_id",
  "tool":       "mcp_tool_name",
  "args":       { "param": "value or {other_step.result}" },
  "depends_on": ["step_id_1", "step_id_2"],
  "label":      "Human-readable label"
}
```

### Example 1 — Parallel web research + screenshot, then summarise

Two searches and a screenshot fire in parallel. The summary waits for both searches.

```json
{
  "steps": [
    {
      "id": "search_news",
      "tool": "web_search",
      "args": { "query": "Intel Arc A380 GPU review 2025" },
      "label": "Search for reviews"
    },
    {
      "id": "search_specs",
      "tool": "web_search",
      "args": { "query": "Intel Arc A380 specifications benchmarks" },
      "label": "Search for specs"
    },
    {
      "id": "summary",
      "tool": "smart_summarize",
      "args": { "text": "Reviews:\n{search_news.result}\n\nSpecs:\n{search_specs.result}" },
      "depends_on": ["search_news", "search_specs"],
      "label": "Combine and summarise"
    }
  ]
}
```

### Example 2 — Research pipeline: search → screenshot → store

```json
{
  "steps": [
    {
      "id": "find",
      "tool": "web_search",
      "args": { "query": "Fedora 43 release notes" },
      "label": "Find Fedora 43 release page"
    },
    {
      "id": "shot",
      "tool": "screenshot",
      "args": { "url": "https://fedoraproject.org/wiki/Releases/43/ChangeSet" },
      "label": "Screenshot change set"
    },
    {
      "id": "extract",
      "tool": "extract_article",
      "args": { "url": "https://fedoraproject.org/wiki/Releases/43/ChangeSet" },
      "label": "Extract article text"
    },
    {
      "id": "store",
      "tool": "db_store_article",
      "args": {
        "title": "Fedora 43 ChangeSet",
        "url": "https://fedoraproject.org/wiki/Releases/43/ChangeSet",
        "body": "{extract.result}",
        "topic": "linux"
      },
      "depends_on": ["extract"],
      "label": "Store in database"
    }
  ]
}
```

### Example 3 — Image pipeline: generate → upscale → scan

```json
{
  "steps": [
    {
      "id": "gen",
      "tool": "image_generate",
      "args": { "prompt": "a detailed circuit board diagram, technical, 4K" },
      "label": "Generate circuit board image"
    },
    {
      "id": "upscale",
      "tool": "image_upscale",
      "args": { "path": "{gen.result}", "scale": 3 },
      "depends_on": ["gen"],
      "label": "Upscale 3x"
    },
    {
      "id": "ocr",
      "tool": "image_scan",
      "args": { "path": "{upscale.result}", "prompt": "Read all visible text and labels" },
      "depends_on": ["upscale"],
      "label": "OCR scan upscaled image"
    }
  ]
}
```

---

## GPU Acceleration

AIChat includes hardware GPU acceleration for both the browser server and the toolkit sandbox, using Intel Arc / integrated GPUs via VA-API, OpenCL, and DRI device passthrough.

### Browser Server — `BrowserGpuConfig`

The `BrowserGpuConfig` class (in `src/aichat/tools/browser.py`) dynamically detects GPU availability at runtime and passes the appropriate Chromium launch arguments:

- **GPU available** (`/dev/dri/renderD*` present or `INTEL_GPU=1` env): enables `--use-gl=egl`, `--enable-features=VaapiVideoDecoder,VaapiVideoEncoder,CanvasOopRasterization`, `--enable-gpu-rasterization`, `--enable-zero-copy`
- **No GPU**: falls back to `--disable-gpu`

The `/health` endpoint on the browser server now reports GPU status:
```json
{"version": "18", "gpu": {"gpu_available": false, "dri_accessible": false, "intel_gpu_env": false}}
```

### Toolkit Sandbox — `ToolkitGpuRuntime`

The `ToolkitGpuRuntime` class (in `docker/toolkit/app.py`) reports which GPU-accelerated packages are importable in custom tools:

```json
{"available": true, "dri_accessible": true, "packages": ["numpy", "cv2"]}
```

Custom tools can use `numpy` and `cv2` with OpenCL acceleration when `/dev/dri` is mounted.

### Enabling GPU passthrough

The `docker-compose.yml` already includes GPU passthrough for `aichat-toolkit`:

```yaml
aichat-toolkit:
  devices:
    - /dev/dri:/dev/dri
  group_add:
    - video
  environment:
    INTEL_GPU: "1"
```

For NVIDIA GPUs, uncomment the `runtime: nvidia` block in `docker-compose.yml` and ensure `nvidia-container-toolkit` is installed on the host.

---

## Docker Services

All services start automatically with `docker compose up -d --build`.

| Service | Port | Description |
|---------|------|-------------|
| `aichat-db` | 5432 | PostgreSQL 16 — articles, images, web cache, conversations |
| `aichat-database` | 8091 | FastAPI REST wrapper for PostgreSQL |
| `aichat-researchbox` | 8092 | RSS/feed discovery service |
| `aichat-memory` | 8094 | Persistent key-value + embedding store (SQLite) |
| `aichat-toolkit` | 8095 | Dynamic custom tool execution sandbox (GPU-enabled) |
| `aichat-mcp` | 8096 | MCP HTTP/SSE server — 49 tools |
| `aichat-whatsapp` | 8097 | WhatsApp bot (Baileys + LM Studio + full MCP tool access) |
| `human_browser` | 7081 (API), 36411 (noVNC) | Chromium + Playwright — all web/screenshot ops |

```bash
docker compose up -d --build     # start all
docker compose down               # stop all (data volumes preserved)
docker compose logs -f aichat-mcp
docker logs human_browser
```

---

## MCP Server (LM Studio / Claude Desktop)

`aichat-mcp` exposes all 49 tools over the network via the Model Context Protocol. It supports both SSE (legacy 2024-11-05) and Streamable HTTP (2025-03-26) transports. Tool calls are **non-blocking** — the server returns immediately and delivers results asynchronously, preventing LM Studio timeouts on slow tools.

**LM Studio `mcp_servers.json` entry (Streamable HTTP, recommended):**
```json
{
  "mcpServers": {
    "aichat": {
      "url": "http://<YOUR_MACHINE_IP>:8096/mcp"
    }
  }
}
```

**Legacy SSE entry:**
```json
{
  "mcpServers": {
    "aichat": {
      "url": "http://<YOUR_MACHINE_IP>:8096/sse"
    }
  }
}
```

**Health check:**
```bash
curl http://localhost:8096/health
# {"ok": true, "sessions": 0, "tools": 49, "transports": [...]}
```

---

## Parallel Thinking

When `/thinking on` (or `thinking_enabled: true` in config), aichat fans out `N` parallel reasoning chains before generating the final answer. Each chain independently reasons about the question, a scoring heuristic picks the best chain, and the final response is synthesised from it.

```
User: What are the trade-offs between PostgreSQL JSONB and a normalised schema?

  Chain 1 ──► score: 0.82
  Chain 2 ──► score: 0.91  ◄── best
  Chain 3 ──► score: 0.75

Final answer synthesised from Chain 2.
```

Control with `/thinking paths N` (1–8 chains) and `/thinking model <id>` (use a faster/cheaper model for thinking).

---

## Contextual Compaction

When the conversation grows past 95% of the context window, aichat automatically summarises old turns instead of dropping them. This preserves important context across very long sessions.

- `/compact` — manually trigger compaction
- `/compact dry` — preview what would be summarised
- `/compact history` — see all past compaction summaries
- Compaction summaries are persisted to the conversation database and re-injected as system context in new sessions

---

## Conversation History

Every session is persisted to PostgreSQL (`conversation_sessions` + `conversation_turns` tables). The RAG system automatically searches past conversations for relevant context on each new query.

- `/history` — browse all sessions with date + title
- `/sessions` — list sessions (title, date, turn count)
- Session titles are auto-generated by the LLM after the first exchange
- RAG uses date-weighted scoring (recent sessions scored higher via exponential decay)

---

## WhatsApp Bot

`aichat-whatsapp` bridges WhatsApp messages to LM Studio via the full MCP tool suite.

**First-run setup** — scan the QR code at `http://localhost:8097` to pair your WhatsApp account. Session persisted in Docker volume `whatsappauth`.

**Features:**
- Full conversation history per contact (20-message window in `aichat-memory`)
- Tool calling: browse web, search images, run shell commands, query database
- Up to 5 tool iterations per user message
- Image/media extraction from incoming messages
- Group chat support (disabled by default; set `ALLOW_GROUPS=true`)

---

## Custom Tool Development

The `create_tool` MCP tool lets the LLM write and deploy tools at runtime:

```
Tool name: port_check
Description: Check if a TCP port is open on a host.
Parameters: {"host": "str", "port": "int"}
Code:
    import asyncio
    host = kwargs["host"]
    port = int(kwargs["port"])
    try:
        _, writer = await asyncio.wait_for(
            asyncio.open_connection(host, port), timeout=3
        )
        writer.close()
        return f"{host}:{port} is OPEN"
    except Exception:
        return f"{host}:{port} is CLOSED or unreachable"
```

Tools are saved to `~/.config/aichat/tools/<name>.py` and hot-reloaded by `aichat-toolkit` on next call (2-second TTL cache).

Available libraries in the sandbox: `asyncio`, `json`, `re`, `os`, `math`, `datetime`, `pathlib`, `shlex`, `subprocess`, `httpx`, `numpy` (GPU-accelerated), `cv2` (OpenCV, GPU-accelerated).

User git repos are available at `/data/repos/<reponame>` (read-only mount of `~/git`).

---

## Themes

Switch with **F5**:

| Theme | Description |
|-------|-------------|
| `cyberpunk` | Default — green/cyan neon on dark |
| `dark` | Neutral dark |
| `light` | Light background |
| `synth` | Purple/pink synthwave |

---

## Personalities

50 built-in expert personalities, switchable with **Ctrl+G** or `/persona`.

| Category | Personalities |
|----------|--------------|
| **Programming** | Linux/Shell, Python, Rust, Go, JavaScript/TypeScript, C++, Java, .NET |
| **Web** | Frontend/UX, Backend/APIs |
| **Data** | Database/SQL, Data Engineering, Data Science, ML Engineering, LLM Engineering |
| **Prompt** | Prompt Engineering |
| **DevOps** | DevOps, SRE, Observability, Incident Response, CI/CD |
| **Security** | AppSec, Red Team, Privacy/Compliance |
| **Infrastructure** | Network, AWS, GCP, Azure, Kubernetes, Docker |
| **Quality** | QA/Testing, Performance |
| **Systems** | Embedded, Systems Programming |
| **Platform** | Game Dev, Android, iOS, Desktop Apps |
| **Design** | UI Design, UX Research, Data Visualization |
| **Management** | Technical Writer, Product Manager, Project Manager |
| **Business** | Finance Analyst, Political Analyst, Legal Operations |
| **Other** | Educator/Coach |

Add custom personalities with `/persona add` — stored in `~/.config/aichat/config.yml`.

---

## Tool Approval Modes

| Mode | Behaviour |
|------|-----------|
| `AUTO` | All tools run immediately (default) |
| `ASK` | Each tool call shows a confirmation dialog |
| `DENY` | All tool calls are blocked |

Cycle with **F4** or set permanently in config (`approval: AUTO`).

---

## OOP Class Reference

Every major subsystem is encapsulated as an OOP class. Key classes:

| Class | File | Purpose |
|-------|------|---------|
| `ImageRenderer` | `docker/mcp/app.py` | Compress & inline-encode images to base64; enforces 3 MB LM Studio limit |
| `ImageRenderingPolicy` | `docker/mcp/app.py` | Guarantees every image tool call returns an inline image block — 3-step escalation (passthrough → fallback bytes → placeholder JPEG) |
| `VisionCache` | `docker/mcp/app.py` | LRU cache (deque, O(1) eviction) for perceptual hash → vision confirmation results; capacity 500 |
| `WorkflowStep` | `docker/mcp/app.py` | Dataclass: `id`, `tool`, `args`, `depends_on`, `label` for `orchestrate` |
| `WorkflowResult` | `docker/mcp/app.py` | Dataclass: `step_id`, `label`, `tool`, `result`, `ok`, `duration_ms` |
| `WorkflowExecutor` | `docker/mcp/app.py` | Topological wave execution; Kahn's algorithm; `asyncio.gather` parallelism; `{id.result}` interpolation |
| `ModelRegistry` | `docker/mcp/app.py` | Singleton; TTL-cached LM Studio model list; detects vision/image-gen/chat models |
| `GpuDetector` | `docker/mcp/app.py` | Cached GPU probe (NVIDIA/Intel/none) via nvidia-smi, vainfo, /dev/dri |
| `GpuUpscaler` | `docker/mcp/app.py` | LM Studio image upscaling with CPU LANCZOS fallback |
| `GpuImageProcessor` | `docker/mcp/app.py` | Static methods: resize, sharpen, enhance, diff, grayscale, annotate (OpenCV CUDA → OpenCV CPU → PIL) |
| `GpuCodeRuntime` | `docker/mcp/app.py` | Prepends DEVICE detection preamble to `code_run` payloads targeting torch/TF/CUDA |
| `BrowserGpuConfig` | `src/aichat/tools/browser.py` | Intel Arc GPU config for Chromium (DRI passthrough, VAAPI) |
| `LMStudioTool` | `src/aichat/tools/lm_studio.py` | TTS, embeddings, chat, caption, summarize, structured extract |
| `CodeInterpreterTool` | `src/aichat/tools/code_interpreter.py` | Async code execution with pip install, timeout, cleanup |
| `ThinkingTool` | `src/aichat/tools/thinking.py` | N parallel reasoning chains via `asyncio.gather`; heuristic scoring; synthesis |
| `ConversationStoreTool` | `src/aichat/tools/conversation_store.py` | Fail-open HTTP client for conversation sessions/turns/search |
| `ConversationSearcher` | `docker/database/app.py` | Cosine-similarity ranking with `ConvRow` namedtuple, `.exclude()` chaining, `.top(n)` |
| `ConvRow` | `docker/database/app.py` | `namedtuple("ConvRow", [id, session_id, role, content, embedding, timestamp])` — type-safe DB row access |
| `ErrorReporter` | `docker/memory/app.py` | Fire-and-forget OOP error reporter; swallows all exceptions; module-level shim for backwards compat |

---

## Running Tests

```bash
# Activate the venv first
source ~/.local/share/aichat/venv/bin/activate

# All tests (912+ pass, 3 skipped, 0 fail)
pytest -k "not smoke" -q

# Orchestration tool tests
pytest tests/test_orchestrate.py -v -k "not smoke"

# Image rendering guarantee + OOP tests
pytest tests/test_image_rendering.py -v -k "not smoke"

# Conversation DB + ConversationSearcher OOP tests
pytest tests/test_conversation_db.py -v -k "not smoke"

# GPU acceleration tests
pytest tests/test_browser_gpu.py -v -k "not smoke"

# Image pipeline
pytest tests/test_image_pipeline.py -q

# Smoke tests (require Docker services running)
pytest -m smoke -v
```

---

## Environment Variables

| Variable | Service | Description |
|----------|---------|-------------|
| `DATABASE_URL` | all | PostgreSQL FastAPI URL (default: `http://aichat-database:8091`) |
| `MEMORY_URL` | mcp | Memory service URL (default: `http://aichat-memory:8094`) |
| `BROWSER_URL` | mcp | Browser server URL (default: `http://human_browser:7081`) |
| `IMAGE_GEN_BASE_URL` | mcp | LM Studio base URL for image generation + embeddings |
| `IMAGE_GEN_MODEL` | mcp | Specific image model to use (empty = first loaded model) |
| `IMAGE_VISION_CONFIRM` | mcp | Set `false` to disable vision verification in `image_search` |
| `INTEL_GPU` | mcp, toolkit | Set `1` to force GPU mode even without `/dev/dri` detection |
| `TOOL_TIMEOUT` | mcp, toolkit | Max seconds per tool call (default: `30`) |
| `LM_STUDIO_URL` | tui | Override LM Studio URL from shell |

---

## Recent Changes

### Code review fixes (latest)
- **VisionCache LRU** — `_order` now uses `collections.deque` (O(1) `popleft`) and correctly re-inserts existing keys at the back of the queue so frequently-updated hashes are never spuriously evicted (was: re-insert did not update position)
- **`_handle_rpc` `isError` propagation** — `tools/call` responses that contain only text blocks starting with `"Error"` or `"Unknown tool"` now set `"isError": true` so MCP clients can distinguish tool failures from empty successes
- **`ConversationSearcher`** — replaced positional `_COL_*` magic integers with a `ConvRow` namedtuple for type-safe, reorder-proof column access; corrupt rows now emit `log.warning` with a count instead of silently disappearing
- **`conv_list_sessions` offset bounds** — `limit` and `offset` are now clamped (`limit ≤ 500`, `offset ≤ 100,000`) to prevent DoS via unbounded pagination queries
- **`ErrorReporter` OOP class** — extracted from inline `_report_error()` function in `docker/memory/app.py`; encapsulates fire-and-forget HTTP reporting with timeout; module-level shim preserved for backwards compatibility
- **sqlite3 timeout** — `sqlite3.connect()` in the memory service now uses `timeout=5.0` to prevent indefinite hangs on locked databases
- **`test_report_error_helper_does_not_raise`** — fixed `from docker.memory import app` (no `__init__.py`) with importlib load, consistent with all other test files
- **`test_search_excludes_current_session`** — uses orthogonal embedding `[0,…,0,1]` so new test entries always rank first regardless of accumulated historical data and service limit caps

### Image rendering guarantee
- **`ImageRenderingPolicy`** — all 20 image-returning MCP tools pass through a 3-step enforcement layer in `_handle_rpc`: (1) passthrough if image present, (2) compress fallback bytes via `_renderer`, (3) append dark-grey JPEG placeholder — LM Studio never shows "external image"
- **Screenshot fallback** — blocked-page image fetch now routes through `_renderer.encode_url_bytes()` for proper 3 MB compression instead of raw `base64.b64encode`

### Orchestration tool
- **`orchestrate`** (49th MCP tool) — declare a multi-step workflow; parallel steps use `asyncio.gather`; sequential steps respect `depends_on`; earlier results injectable via `{step_id.result}`; returns structured timing report

---

## Disclaimer

The author is not responsible for how users use this program. Use at your own risk.
