# AIChat

Codex-like Textual TUI with research/operator tooling and theme switching.

Default OpenAI-compatible endpoint: `http://localhost:1234`.

## Install (user-local, no sudo)

```bash
bash install.sh
export PATH="$HOME/.local/bin:$PATH"
```

The installer also attempts to deploy plugin containers (`aichat-rssfeed-db`, `aichat-rssfeed`, `aichat-researchbox`) with `docker compose up -d --build` when Docker and permissions are available.

Then run:

```bash
aichat
```

## Uninstall

```bash
bash uninstall.sh
```

The uninstaller also tears down plugin containers with `docker compose down --volumes --remove-orphans` when available.

## Docker services

Artifacts from researchbox are mounted to `/tmp/research`.

Manual control:

```bash
docker compose up -d --build
docker compose down --volumes --remove-orphans
```

## Keybindings

- Enter: Send
- Shift+Enter: Newline
- F1: Help
- F2: Model picker
- F3: Search transcript
- F4: Cycle approval mode (DENY/ASK/AUTO)
- F5: Theme picker
- F6: Toggle streaming
- F7: Sessions
- F8: Settings
- F9: Copy last assistant message
- F10: Export full chat to markdown
- F11: Cancel active task
- F12: Quit
