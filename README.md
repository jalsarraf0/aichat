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
