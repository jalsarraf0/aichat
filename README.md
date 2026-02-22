# AIChat

Codex-like Textual TUI with research/operator tooling, shell control, and theme switching.

## Quick Start

```bash
bash install.sh
export PATH="$HOME/.local/bin:$PATH"
aichat
```

## Requirements

- Python 3.12+.
- LM Studio running at `http://localhost:1234`.
- Docker (optional) for RSS and research tools.

## Install (user-local, no sudo)

```bash
bash install.sh
export PATH="$HOME/.local/bin:$PATH"
```

## Run

```bash
aichat
```

## Uninstall

```bash
bash uninstall.sh
```

## Docker Tools (optional)

If Docker is available, the installer attempts `docker compose up -d --build`.
The uninstaller attempts `docker compose down --volumes --remove-orphans`.
Make sure your user is in the `docker` group if you use these tools.

## Keybindings

- Enter: Send.
- Shift+Enter: Newline.
- PageUp/PageDown: Scroll transcript.
- Ctrl+S: Toggle shell access (shown after F12 in the keybind bar).
- Ctrl+; (shown as ^:): Personality menu.
- F1: Help.
- F2: Model picker.
- F3: Search transcript.
- F4: Cycle approval mode.
- F5: Theme picker.
- F6: Toggle streaming.
- F7: Sessions.
- F8: Settings.
- F9: New chat (archives previous context to `/tmp/context`).
- F10: Clear transcript.
- F11: Cancel streaming.
- F12: Quit.

Keybind bar order is always `F1..F12` then `^S` and `^:`.
Model picker labels include emojis that hint at capabilities (vision, code, tools, etc.).

## Commands

- `/help` shows keybinds.
- `/concise on` or `/concise off`.
- `/verbose` (alias for `/concise off`).
- `/new` starts a new chat and archives previous context to `/tmp/context`.
- `/clear` clears the transcript.
- `/copy` copies the last assistant response.
- `/export` exports the chat to markdown.
- `/vibecode <project>` creates/switches to `~/git/<project>` for shell and coding tasks.
- `/persona` or `/personality` opens the personality menu.
- `/persona list` lists available personalities.
- `/persona add` adds a custom personality.
- `/rss <topic>` shows latest stored RSS items.
- `/rss store <topic>` searches feeds and ingests items into the RSS service.
- `/rss ingest <topic> <feed_url>` ingests a specific feed URL.
- `/rss ingest` opens a modal to enter topic + feed URL.
- `/shell on` or `/shell off`.
- `/shell <cmd>` runs a command when shell is ON.
- `!<cmd>` shortcut when shell is ON.

## Model Selection

The model is loaded from LM Studio at startup. If the configured model is missing, the first available model is selected.
Selecting a model validates against the LM Studio API and saves the choice.

## Transcript Behavior

Responses are sanitized (no `<think>` or tool dumps), wrapped to the visible width, and more verbose by default.
The assistant may ask short follow-up questions when appropriate.

## Tools: batching + retries

Tool calls are queued and executed with a small concurrency limit (default 1). Rate-limited and transient errors
retry with exponential backoff and jitter, capped to avoid spam. Progress and raw tool output appear in the Tools panel.

## Shell Tool (controlled)

Shell access is ON by default. Approval mode still applies: `ASK` prompts, `AUTO` runs immediately.
Output streams to the Tools panel and also appears in the Transcript (truncated if very large).
Sudo commands are run non-interactively (`sudo -n`). If your sudo policy requires a password, the command fails.

## GitHub Repo Creation

`aichat repo create` creates and pushes a GitHub repo named `aichat` using SSH.
Alias: `aichat github init`.

Requirements:
- GitHub CLI (`gh`) installed and authenticated (`gh auth login`).
- A working SSH key in `~/.ssh` (prefers `id_ed25519`, then `id_rsa`).

Usage:

```bash
aichat repo create --private
aichat repo create --public --owner <org>
```

## Personalities

The app ships with ~50 expert personalities (Linux, shell, programming, security, cloud, political analysis, and more) and always keeps the built-in set.
You can switch between them with `Ctrl+;` or `/persona`, and add custom personalities with `/persona add`.
All personalities are stored in `~/.config/aichat/config.yml`.

## Disclaimer

The author is not responsible for how users use this program. Use at your own risk.
