# AIChat

Codex-like Textual TUI with research/operator tooling and theme switching.

## Install (user-local, no sudo)

```bash
bash install.sh
export PATH="$HOME/.local/bin:$PATH"
```

Then run:

```bash
aichat
```

Notes:
- Requires Python 3.12+.
- The LLM endpoint is fixed to `http://localhost:1234`.
- If Docker is available, the installer attempts `docker compose up -d --build` for plugin services.
- If Docker is installed, ensure your user is in the `docker` group.
- Concise mode is ON by default (final answers only). Toggle with `/concise on|off` or `/verbose`.

## Uninstall

```bash
bash uninstall.sh
```

Notes:
- The uninstaller attempts `docker compose down --volumes --remove-orphans` if Docker is available.

## Docker services (optional)

If you use docker-backed tools, ensure Docker is installed and your user is in the `docker` group.

```bash
docker compose up -d --build
docker compose down --volumes --remove-orphans
```

## Keybindings

- Enter: Send
- Shift+Enter: Newline
- Ctrl+P: Help (shown after F12 in the keybind bar)
- F1: Help
- F2: Model picker
- F3: Search transcript
- F4: Cycle approval mode
- F5: Theme picker
- F6: Toggle streaming
- F7: Sessions
- F8: Settings
- F9: Copy last assistant message
- F10: Export full chat to markdown
- F11: Cancel streaming
- F12: Quit

Keybind bar order is always `F1..F12` then `^P`.

## Concise Mode

By default, assistant responses are concise, final-only, and sanitized (no `<think>` blocks or tool payloads).

Commands:
- `/concise on` or `/concise off`
- `/verbose` (alias for `/concise off`)

## Tools: batching + retries

Tool calls are queued and executed with a small concurrency limit (default 1). Rate-limited and transient errors
retry with exponential backoff and jitter, capped to avoid spam. Progress and raw tool output appear in the Tools panel.

## Shell Tool (controlled)

Shell access is OFF by default.

Commands:
- `/shell on` or `/shell off`
- `/shell <cmd>` (runs when shell is ON)
- `!<cmd>` (shortcut when shell is ON)

Approval mode still applies: `ASK` prompts, `AUTO` runs immediately. Output streams to the Tools panel; the Transcript shows only a short summary.

## GitHub Repo Creation

`aichat repo create` creates and pushes a GitHub repo named `aichat` using SSH.
Alias: `aichat github init`.

Requirements:
- GitHub CLI (`gh`) installed and authenticated (`gh auth login`)
- A working SSH key in `~/.ssh` (prefers `id_ed25519`, then `id_rsa`)

Usage:

```bash
aichat repo create --private
aichat repo create --public --owner <org>
```
