# AIChat

A local-first AI chat TUI (Terminal User Interface) built with [Textual](https://github.com/Textualize/textual). Connects to a local LLM via [LM Studio](https://lmstudio.ai), with a full suite of Docker-backed tools: web browsing, shell execution, RSS feeds, research, persistent memory, and a live Chromium browser.

---

## Requirements

| Requirement | Details |
|---|---|
| Python | 3.12 or newer |
| LM Studio | Running locally or remotely (configurable via `base_url`) |
| Docker + Compose | Required for all tool services |
| `docker` group | Your user must be in the `docker` group |
| human_browser container | Required for the `browser` tool (see below) |

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

Removes the virtualenv, launcher, and brings down Docker containers with `docker compose down --volumes --remove-orphans`.

---

## Configuration

Config is stored at `~/.config/aichat/config.yml` and is created automatically on first run.

```yaml
base_url: http://localhost:1234   # LM Studio API endpoint
model: mistralai/magistral-small-2509 # Model ID (auto-selected if not found)
theme: cyberpunk                      # cyberpunk | dark | light | synth
approval: AUTO                        # AUTO | ASK | DENY
shell_enabled: true                   # Enable/disable shell tool
concise_mode: false                   # Shorter responses
active_personality: linux-shell-programming
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

Control whether tools run automatically or require confirmation:

| Mode | Behaviour |
|------|-----------|
| `AUTO` | All tools run immediately (default) |
| `ASK` | Each tool call shows a confirmation dialog |
| `DENY` | All tool calls are blocked |

Cycle with **F4** or set permanently in `~/.config/aichat/config.yml` (`approval: AUTO`).

---

## Built-in Tools

These tools are always available to the LLM without any setup beyond the Docker stack.

### `browser` — Real Chromium Browser

Controls a live Chromium browser in the `human_browser` Docker container via Playwright.

**Setup**: Start the `human_browser` container once:
```bash
docker run -d --name human_browser \
  --shm-size=2g \
  -p 36411:6080 \
  -v /docker/human_browser/workspace:/workspace \
  -v /docker/human_browser/home:/home/ai \
  human-browser:latest
```

On first use, aichat automatically deploys the Playwright API server into the container and starts it. No manual steps needed after the container is running.

The browser server **auto-upgrades** when a newer version is detected — simply use aichat and it will redeploy transparently. The v3 server includes:

- **Extended stealth**: spoofs `navigator.plugins`, `navigator.languages`, `window.chrome.runtime`, and the Permissions API — not just the `webdriver` flag
- **Randomized viewport**: picks from common desktop sizes (1366×768, 1440×900, 1920×1080, etc.) to avoid a fixed fingerprint
- **Human-like typing**: DuckDuckGo searches use `page.type()` with random per-keystroke delays (60–140 ms) and a natural pause before submitting
- **Pre-screenshot scroll**: simulates a user scrolling the page before capturing
- **Image fallback pipeline**: if a screenshot is blocked, the tool extracts `<img>` URLs from the page DOM and returns the first downloadable image inline

All httpx-based fetches (web_search, screenshot_search, web_fetch) use full Chrome browser headers (`Accept`, `Sec-Fetch-*`, `DNT`, etc.) and, for screenshot_search, a 2–5 second random pause between consecutive page loads to avoid triggering rate limits.

**Actions:**

| Action | Required params | Description |
|--------|----------------|-------------|
| `navigate` | `url` | Go to a URL; returns page title + readable text |
| `read` | — | Return current page title + text (no navigation) |
| `screenshot` | `url` (optional) | Take a PNG screenshot; returns file path at `/workspace/` |
| `click` | `selector` | Click a CSS selector; returns updated page content |
| `fill` | `selector`, `value` | Type text into an input field |
| `eval` | `code` | Run a JavaScript expression; returns its result |

The browser session persists between calls — navigate, click, fill, read all operate on the same live page.

If a screenshot fails (e.g. the page blocks headless browsers), the MCP server and aichat TUI both automatically fall back to fetching a real image from the page DOM and returning it inline.

The noVNC web UI is accessible at `http://localhost:36411` to watch the browser in real time.

### `web_fetch` — Fetch a Web Page

Fetches a URL and returns clean readable text (HTML stripped).

```
Fetch the content at https://docs.python.org/3/library/asyncio.html
```

- `url`: required
- `max_chars`: optional, default 4000, max 16000

### `shell_exec` — Run Shell Commands

Runs a bash command on the host machine. Working directory persists across calls within a session.

```
Run: ls -la ~/git/aichat/src
```

Dangerous commands (rm -rf on critical paths, fork bombs, dd to raw disks, etc.) are blocked automatically.
Shell must be enabled (`/shell on` or `Ctrl+S`).

### `rss_latest` — RSS Feed Reader

Fetches the latest stored RSS items for a topic from the RSS service.

```
What's the latest news on AI?
```

### `researchbox_search` — Find RSS Feeds

Searches for RSS feed sources for a given topic.

### `researchbox_push` — Ingest RSS Feed

Fetches a feed URL and stores its items under a topic label.

### `memory_store` / `memory_recall` — Persistent Memory

Store and retrieve facts or notes that persist across sessions.

```
Remember my preferred coding style is 4-space indentation
```

```
What was my preferred coding style?
```

### `create_tool` — Dynamic Tool Creation

Create a new custom tool at runtime. Tools are stored in `~/.config/aichat/tools/` and persist across restarts.

```
Create a tool that fetches the current Bitcoin price from CoinGecko
```

The tool code runs inside the `aichat-toolkit` Docker container (isolated). Available libraries: `asyncio`, `json`, `re`, `os`, `math`, `datetime`, `pathlib`, `shlex`, `subprocess`, `httpx`.

Tools can also:
- Make HTTP calls with `httpx`
- Run shell commands with `subprocess` or `asyncio.create_subprocess_exec`
- Read files from the user's git repos at `/data/repos/<reponame>`

### `list_custom_tools` — List Dynamic Tools

Lists all custom tools you have created.

### `delete_custom_tool` — Delete a Dynamic Tool

Permanently removes a custom tool.

---

## Docker Services

All services start automatically with `docker compose up -d --build`.

| Service | Port | Description |
|---------|------|-------------|
| `aichat-db` | 5432 | PostgreSQL — stores articles, images, and web cache |
| `aichat-database` | 8091 | FastAPI REST wrapper for PostgreSQL |
| `aichat-researchbox` | 8092 | RSS/feed discovery service |
| `aichat-memory` | 8094 | Persistent key-value memory store |
| `aichat-toolkit` | 8095 | Dynamic tool execution sandbox |
| `aichat-mcp` | 8096 | MCP HTTP/SSE server (LM Studio / Claude Desktop integration) |
| `human_browser` | 36411 | Chromium browser + noVNC (managed separately) |

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
docker compose logs -f aichat-toolkit
docker logs human_browser
```

---

## MCP Server (LM Studio / Claude Desktop)

`aichat-mcp` exposes all aichat tools over the network via the Model Context Protocol.
It supports both SSE (legacy) and Streamable HTTP (MCP 2025-03-26) transports.

**LM Studio `mcp_servers.json` entry:**
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

## Themes

Switch with **F5**. Four built-in themes:

| Theme | Description |
|-------|-------------|
| `cyberpunk` | Default — green/cyan neon on dark (default) |
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
- Structured JSON/XML responses from tools are rendered as formatted Markdown code blocks instead of being hidden.
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

Tools are saved to `~/.config/aichat/tools/<name>.py` and loaded from `aichat-toolkit` on startup.
They can access the user's git repos at `/data/repos/<reponame>` (read-only bind mount of `~/git`).

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

# With verbose output
pytest -v

# Only fast unit tests (no Docker required)
pytest tests/test_sanitizer.py tests/test_tool_args.py tests/test_keybinds.py

# TUI tests
pytest tests/test_tui.py

# Full e2e (requires Docker services running)
pytest tests/test_tools_e2e.py
```

The test suite has 130 tests. Live tool tests skip gracefully when Docker services are not available.

---

## Disclaimer

The author is not responsible for how users use this program. Use at your own risk.
