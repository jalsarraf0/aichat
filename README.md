# AIChat

Codex-like Textual TUI with research mode, operator mode, strict tool sequencing (one tool call at a time), tri-state approvals, theme picker, and Docker-backed RSS + researchbox services.

## Features Implemented

- Default OpenAI-compatible endpoint: `http://localhost:1234`
- Keybindings including required `Ctrl+O` command palette and `Ctrl+I` theme picker
- Approval tri-state (`DENY → ASK → AUTO`) on `Ctrl+A`
- Esc immediate cancellation for active chat and shell interruptions
- Research mode (`/research`), operator mode (`/operator` or `/code`), and `/continue`
- Full content persistence (JSONL transcript) and markdown export (`F10`)
- Tool pane separate from transcript (compact transcript entries)
- Theme system (Cyberpunk, Synth, Modern, Light, Pastel) persisted in config
- Docker stack:
  - Postgres
  - RSS service with scheduled purge every 6h and manual purge endpoint
  - Researchbox with feed discovery + feed push into RSS

## Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install -e .
```

## Run TUI

```bash
aichat
```

## Start/Stop Docker Stack

```bash
docker compose up -d --build
docker compose ps
docker compose down
```

Artifacts from researchbox are expected under `/tmp/research` (bind mount).

## Commands in chat

- `/research` – enable deep research mode
- `/operator` or `/code` – enable operator/vibe coding mode
- `/continue` – continue prior flow

## Keybindings

- Enter: Send
- Shift+Enter: Newline
- Esc: Cancel
- Ctrl+Q: Quit
- Ctrl+O: Command palette
- Ctrl+I: Theme picker
- Ctrl+A: Cycle approval mode
- F1: Help
- F2: Model picker
- F3: Search transcript
- F4: Toggle auto-approve (legacy)
- F5: Refresh models
- F6: Toggle streaming
- F7: Sessions
- F8: Settings
- F9: Copy last assistant full message
- F10: Export full chat to markdown
- F11: Focus mode toggle
- F12: Debug overlay

## Manual Test Checklist

- [ ] Ctrl+O opens command palette.
- [ ] Ctrl+I opens theme picker; switching themes persists and applies instantly.
- [ ] Ctrl+A cycles approval modes; status updates; tool gating respected.
- [ ] Base URL shows localhost:1234 UP/DOWN.
- [ ] F5 models refresh; F2 model switch.
- [ ] Enter send / Shift+Enter newline.
- [ ] Esc cancel mid-stream and mid-tool.
- [ ] Shell: sh_start → sh_send → sh_read; Esc cancels; sh_close.
- [ ] Operator mode creates program and runs it.
- [ ] Research mode uses rssfeed + feed discovery + refresh + summary.
- [ ] Purge older than 30 days works (scheduled + manual).
- [ ] Tool outputs never garble transcript; tool pane shows raw with paging/search.
- [ ] Export (F10) and copy (F9) include full content.
