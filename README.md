# AIChat

A local-first AI chat TUI (Terminal User Interface) built with [Textual](https://github.com/Textualize/textual). Connects to a local LLM via [LM Studio](https://lmstudio.ai), with a full suite of Docker-backed tools: real Chromium browsing (Playwright), image search and processing pipeline, shell execution, RSS/research, persistent memory, and a WhatsApp bot.

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
```

**Override the LM Studio URL** without editing the file:

```bash
LM_STUDIO_URL=http://localhost:1234 aichat
```

---

## Interface Overview

```
┌─────────────────────────────────────────────────────────────────┐
│  Header (model name, approval mode, streaming indicator)        │
├────────────────────────────────┬────────────────────────────────┤
│  Transcript (chat bubbles)     │  Tools panel (tool output log) │
│                                │  Sessions panel                │
├────────────────────────────────┴────────────────────────────────┤
│  Prompt input                                                   │
├─────────────────────────────────────────────────────────────────┤
│  Keybind bar: F1..F12  ^S  ^G                                   │
└─────────────────────────────────────────────────────────────────┘
```

- **Transcript**: Chat messages rendered as Markdown bubbles. Scroll with mouse wheel or PageUp/PageDown.
- **Tools panel**: Raw tool output streams here in real time.
- **Sessions panel**: Active shell sessions.
- **Prompt input**: Enter key sends, Shift+Enter inserts a newline.

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

Type these directly in the prompt input.

### Chat

| Command | Description |
|---------|-------------|
| `/help` | Show keybindings |
| `/new` | Start a new chat (archives previous context to `/tmp/context`) |
| `/clear` | Clear the transcript display |
| `/copy` | Copy the last assistant response to clipboard |
| `/export` | Export the current chat to a Markdown file |
| `/concise on` | Enable concise (shorter) responses |
| `/concise off` | Disable concise mode |
| `/verbose` | Alias for `/concise off` |
| `/shell on` | Enable shell tool |
| `/shell off` | Disable shell tool |
| `!<cmd>` | Run a shell command directly (when shell is enabled) |

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

## Tool Approval Modes

| Mode | Behaviour |
|------|-----------|
| `AUTO` | All tools run immediately (default) |
| `ASK` | Each tool call shows a confirmation dialog |
| `DENY` | All tool calls are blocked |

Cycle with **F4** or set permanently in `~/.config/aichat/config.yml` (`approval: AUTO`).

---

## Built-in Tools

All tools are exposed via both the MCP HTTP server (`aichat-mcp`) and the stdio MCP server (`aichat mcp`). The complete tool inventory is grouped below.

### Browser Tools — Real Chromium (Playwright)

All browser/screenshot operations run inside the `human_browser` Docker container via a Playwright/FastAPI server that is auto-deployed and auto-upgraded on first use.

**Setup**: Start the `human_browser` container once:
```bash
docker run -d --name human_browser \
  --shm-size=2g \
  -p 36411:6080 \
  -v /docker/human_browser/workspace:/workspace \
  -v /docker/human_browser/home:/home/ai \
  human-browser:latest
```

The browser server auto-upgrades when a newer version is detected — simply use aichat and it redeployes transparently. The current v17 server includes:

- **15-section stealth JS**: removes `webdriver` flag, spoofs `navigator.plugins`, `navigator.languages`, canvas fingerprint noise (per-session XOR seed), WebGL GPU (NVIDIA RTX 3070), AudioContext noise, screen geometry, `deviceMemory = 8`, media devices stub, `navigator.vendor = "Google Inc."`, `maxTouchPoints`, `cookieEnabled`, `onLine`, `window.name` — eliminates every signal checked by Cloudflare Turnstile, PerimeterX, DataDome, and Kasada
- **Human-like mouse movement**: real Playwright `mouse.move()` events fire after navigation, satisfying bot detectors that require at least one `mousemove` event
- **Route-based ad blocking**: 30+ ad/tracker domains are aborted before they load (no extension required)
- **Scroll jitter**: randomised inter-step timing (`0.6–1.5×`) prevents robotic fixed-interval scrolling
- **Screenshot block-retry**: if a Cloudflare / captcha page is captured, the context is rotated and the screenshot retried automatically
- **Randomised viewport**: picks from common desktop sizes (1366×768, 1440×900, 1920×1080, etc.) on every new context
- **Context rotation on block**: `_ALWAYS_ROTATE_DOMAINS` (Twitter, Reddit, Pinterest, Instagram) get a fresh browser context on every visit
- **Browser crash recovery**: full Chromium restart on WebSocket close or OOM

The noVNC web UI is accessible at `http://localhost:36411` to watch the browser in real time.

| Tool | Description |
|------|-------------|
| `screenshot` | Navigate to a URL and take a screenshot. Supports `find_text` (scroll to and clip text region) and `find_image` (clip to a specific `<img>` element). Returns image inline. |
| `browser` | Navigate/read/click/fill/eval on the live page. Session persists between calls. |
| `scroll_screenshot` | Full-page screenshot assembled from multiple viewport-height scrolls. |
| `bulk_screenshot` | Screenshot multiple URLs in parallel. |
| `fetch_image` | Download an image by URL and return it inline; retries on 429. |
| `screenshot_search` | Web search + screenshot the top result pages. |
| `browser_save_images` | Download a list of image URLs to the workspace. |
| `browser_download_page_images` | Download all images found on a page to the workspace. |
| `page_scrape` | Full-page text extraction with lazy-scroll rendering. Supports `include_links`. |
| `page_images` | Extract all image URLs from a live page using 7 extraction strategies (img src, srcset, og:image, background-image, picture, data-src, lazy-load attributes). |

### Web Tools

| Tool | Description |
|------|-------------|
| `web_fetch` | Fetch a URL and return clean readable text (HTML stripped). `max_chars` up to 16 000. |
| `web_search` | DuckDuckGo web search with three-tier fallback. Returns titles + snippets. |
| `extract_article` | Fetch a URL and extract the main article body (readability-style). |
| `page_extract` | Structured data extraction from a live page. |
| `image_search` | Multi-tier image search with diversity guarantee. Returns up to `count` images (default 4). |

#### `image_search` — How it works

- **Query expansion**: game-specific abbreviations auto-expand (e.g. `gfl2` → "Girls Frontline 2", `hsr` → "Honkai Star Rail")
- **Tier 1** — DDG web search → `page_images` on top article pages (1 scroll each)
- **Tier 2** — DuckDuckGo Images page → decodes proxy URLs
- **Tier 3** — Bing Images → different CDN corpus from DDG for variety
- **Scoring**: `pbs.twimg.com` +10, known good image hosts +6, keyword in alt text +5, keyword in URL +2, natural width ≥ 500 px +4
- **Domain cap**: max 2 images per domain
- **Cross-call dedup**: seen URLs stored in `aichat-memory` with 1-hour TTL so repeated calls return new images
- **Pagination**: `offset` parameter skips the first N candidates

### Image Pipeline Tools

All image tools operate on files in the browser workspace (`/docker/human_browser/workspace`). Pass filenames (not full paths) between tools.

| Tool | Description |
|------|-------------|
| `image_crop` | Extract a rectangular region (x, y, width, height). |
| `image_zoom` | Enlarge a portion of an image with padding context. |
| `image_scan` | Pass the image to the vision model for OCR / detailed reading. |
| `image_enhance` | Adjust brightness, contrast, and saturation. |
| `image_stitch` | Combine multiple images side-by-side or in a grid. |
| `image_diff` | Pixel-level difference highlight between two images (ImageChops). |
| `image_annotate` | Draw bounding boxes and labels on an image (ImageDraw). |
| `image_upscale` | LANCZOS upscale (1.1–8.0×, default 2×) with optional UnsharpMask sharpening. EXIF rotation is applied automatically; output is capped at 8192 px per side to prevent OOM. |
| `image_generate` | Text-to-image via LM Studio OpenAI-compatible `/v1/images/generations`. Requires a loaded image-gen model. |
| `image_edit` | Image-to-image / inpainting via `/v1/images/edits`. Supports strength parameter. |

**Pipeline examples:**
```
image_generate → image_upscale(scale=4) → image_scan   # read fine text in a generated image
screenshot → image_crop → image_zoom → image_scan       # zoom in on a UI element
screenshot → image_edit("watercolor style") → image_diff(original, edited)
image_generate × 2 → image_stitch → image_diff         # compare two generated variants
```

### Database / Cache Tools

| Tool | Description |
|------|-------------|
| `db_store_article` | Store an article (title, url, body, topic) in PostgreSQL. |
| `db_search` | Full-text search across stored articles. Supports `offset` and `summary_only`. |
| `db_cache_store` | Store a key-value entry in the web cache (with optional TTL). |
| `db_cache_get` | Retrieve a cached entry by key. |
| `db_store_image` | Store an image URL + metadata in the database. |
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

### System Tools

| Tool | Description |
|------|-------------|
| `get_errors` | Retrieve recent tool errors from the server log. |
| `shell_exec` | Run a bash command on the host machine (stdio only). |

---

## Docker Services

All services start automatically with `docker compose up -d --build`.

| Service | Port | Description |
|---------|------|-------------|
| `aichat-db` | 5432 | PostgreSQL — stores articles, images, and web cache |
| `aichat-database` | 8091 | FastAPI REST wrapper for PostgreSQL |
| `aichat-researchbox` | 8092 | RSS/feed discovery service |
| `aichat-memory` | 8094 | Persistent key-value memory store (SQLite) |
| `aichat-toolkit` | 8095 | Dynamic tool execution sandbox |
| `aichat-mcp` | 8096 | MCP HTTP/SSE server (LM Studio / Claude Desktop integration) |
| `aichat-whatsapp` | 8097 | WhatsApp bot (Baileys + LM Studio + full MCP tool access) |
| `human_browser` | 36411 | Chromium browser + Playwright API + noVNC |

**Start all services:**
```bash
docker compose up -d --build
```

**Stop all services:**
```bash
docker compose down
```

**View logs:**
```bash
docker compose logs -f aichat-mcp
docker logs human_browser
```

---

## MCP Server (LM Studio / Claude Desktop)

`aichat-mcp` exposes all 40 tools over the network via the Model Context Protocol. It supports both SSE (legacy 2024-11-05) and Streamable HTTP (2025-03-26) transports. Tool calls are **non-blocking** — the server returns immediately and delivers results asynchronously, preventing LM Studio timeouts on slow tools.

**LM Studio `mcp_servers.json` entry (Streamable HTTP):**
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
```

Returns a JSON object listing active sessions, available tools, and supported transports.

---

## WhatsApp Bot

`aichat-whatsapp` bridges WhatsApp messages to LM Studio via the full MCP tool suite.

**First-run setup** — scan the QR code at `http://localhost:8097` to pair your WhatsApp account. The session is persisted in a Docker volume (`whatsappauth`) so you only need to pair once.

**Features:**
- Full conversation history per contact (stored in `aichat-memory` with 20-message window)
- Tool calling: the bot can browse the web, search for images, run shell commands, query the database, etc. — same capabilities as the TUI
- Up to 5 tool iterations per user message
- Image/media extraction from incoming messages
- Group chat support (disabled by default; set `ALLOW_GROUPS=true`)

**Environment variables** (in `docker-compose.yml`):

| Variable | Default | Description |
|----------|---------|-------------|
| `LM_STUDIO_URL` | `http://host.docker.internal:1234` | LM Studio endpoint |
| `BOT_NAME` | `AI Assistant` | Bot display name |
| `MAX_HISTORY` | `20` | Messages to keep in context per contact |
| `MAX_TOOL_ITER` | `5` | Max tool-call iterations per message |
| `MAX_TOKENS` | `1024` | Max response tokens |
| `ALLOW_GROUPS` | `false` | Enable group chat responses |

---

## Themes

Switch with **F5**. Four built-in themes:

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

Add custom personalities with `/persona add` — they're stored in `~/.config/aichat/config.yml`.

---

## Model Selection

- Press **F2** to open the model picker. Models are fetched live from LM Studio.
- If the configured model is not available, the first available model is selected automatically.
- The selected model is saved to `~/.config/aichat/config.yml`.

---

## Streaming

- **F6** toggles streaming on/off.
- When streaming is on, responses appear token by token.
- **F11** cancels an in-progress response.

---

## Transcript Behaviour

- All responses are sanitized: `<think>` blocks, XML tool-call artifacts, and special model tokens are stripped.
- Structured JSON/XML responses from tools are rendered as formatted Markdown code blocks.
- The full conversation history (including tool messages) is stored at `~/.local/share/aichat/transcript.jsonl`.
- `/new` starts a fresh context; the old context is archived to `/tmp/context`.

---

## Shell Tool Details

- Shell is enabled by default. Toggle with `Ctrl+S` or `/shell on|off`.
- Working directory persists across shell commands within a session.
- `sudo` commands are run non-interactively (`sudo -n`). If your sudo policy requires a password, they will fail.
- Output streams live to the Tools panel and also appears as a "Shell" bubble in the transcript.
- **Blocked commands** (regardless of approval mode): `rm -rf` on critical paths, fork bombs (`:(){:|:&};:`), `mkfs`, `wipefs`, `dd` to raw block devices, `chmod 777` on system paths, `kill -9 1`.

---

## Custom Tool Development

The `create_tool` command lets the LLM write and deploy tools on the fly:

```python
# Example: a tool that checks if a port is open
tool_name: port_check
description: Check if a TCP port is open on a host.
parameters: {"host": str, "port": int}

# Implementation (body of async def run(**kwargs) -> str:)
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

Tools are saved to `~/.config/aichat/tools/<name>.py` and loaded from `aichat-toolkit` on startup. They can access the user's git repos at `/data/repos/<reponame>` (read-only bind mount of `~/git`).

Available libraries: `asyncio`, `json`, `re`, `os`, `math`, `datetime`, `pathlib`, `shlex`, `subprocess`, `httpx`.

---

## GitHub Repo Management

```bash
aichat repo create --private
aichat repo create --public --owner <org>
```

Requirements:
- GitHub CLI (`gh`) installed and authenticated (`gh auth login`)
- A working SSH key in `~/.ssh` (`id_ed25519` or `id_rsa`)

---

## Running Tests

```bash
# All tests
pytest

# Only fast unit tests (no Docker required)
pytest tests/test_sanitizer.py tests/test_tool_args.py tests/test_keybinds.py

# Full image pipeline + MCP integration (requires Docker services running)
pytest tests/test_image_pipeline.py

# Full e2e (requires Docker services running)
pytest tests/test_tools_e2e.py
```

The test suite has 145 tests. Live tool tests skip gracefully when Docker services are not available.

---

## Disclaimer

The author is not responsible for how users use this program. Use at your own risk.
