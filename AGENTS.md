# aichat AGENTS

## What This Repo Does

`aichat` is a local-first AI chat platform built around Docker Compose, a Textual client, and an MCP HTTP/SSE gateway. The current stack centers on `aichat-data` as the consolidated data service and `aichat-mcp` on port `8096`.

## Main Entrypoints

- `docker-compose.yml`: live service graph and port layout.
- `Makefile`: common build, smoke, test, lint, and security targets.
- `docker/data/`: consolidated articles, memory, graph, planner, RSS, and jobs service.
- `docker/mcp/`: MCP gateway.
- `docker/vision/`, `docker/docs/`, `docker/sandbox/`, `docker/whatsapp/`: service implementations.
- `src/`: shared Python package code.
- `entrypoint/`: container startup scripts.

## Commands

- `docker compose config`
- `docker compose up -d`
- `make smoke`
- `make test`
- `make lint`
- `make security-checks`

## Repo-Specific Constraints

- Keep `aichat-data` consolidated; do not split old services back out without strong evidence from the codebase.
- Keep published ports stable unless the compose file, docs, and downstream clients are updated together.
- Treat `entrypoint/` as fragile startup logic; small mistakes can break containers silently.
- Handle schema changes through the repo's migration flow rather than ad hoc edits.
- `aichat-whatsapp` is Node-based; do not treat it like the Python services.
- Confirm `.env` values before assuming a live LM Studio endpoint or model name.

## Agent Notes

- Inspect the touched service directory before editing shared assumptions.
- Keep changes scoped to the affected service.
- Validate at least `docker compose config` plus the most relevant repo command for the changed area.
