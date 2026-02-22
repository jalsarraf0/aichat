# AIChat

Codex-like Textual TUI with research/operator tooling and theme switching.

Default OpenAI-compatible endpoint: `http://localhost:1234`.

## Install (user-local, no sudo)

```bash
bash install.sh
export PATH="$HOME/.local/bin:$PATH"
```

Then run:

```bash
aichat
```

## Uninstall

```bash
bash uninstall.sh
```

## Docker services (optional)

If you use docker-backed tools, ensure Docker is installed and your user is in the `docker` group.

```bash
docker compose up -d --build
docker compose down
```

## Keybindings

- Enter: Send
- Esc: Cancel
- Ctrl+Q: Quit
- Ctrl+I: Theme picker
- Ctrl+A: Cycle approval mode
- F2: Model picker
- F3: Search transcript
- F6: Toggle streaming
- F7: Sessions
- F8: Settings
- F10: Export full chat to markdown
