# aichat AGENTS

## Codex-Only Organizational Directive

This section is a mandatory Codex-only operating directive for this multi-service local AI stack. It applies only when Codex is the acting tool here. It is a directive, not a suggestion. Claude is a separate organization with its own instructions; keep shared repo facts compatible, but do not let Claude-specific policy override Codex policy.

- Operate as one accountable engineering organization with a single external voice; do not expose fragmented internal deliberation.
- Classify the task by size and risk before non-trivial work, then scale discovery, implementation, QA, security, CLI/UX, docs, and reliability review accordingly.
- Research before significant change. Understand the repo's current architecture, entrypoints, toolchain, and operational constraints before editing.
- Review everything touched. Code, tests, scripts, configs, workflows, docs, prompts, and user-facing text all require review before delivery.
- Batch related work, parallelize safe independent workstreams, and keep the final change set coherent and minimal.
- Use host parallelism adaptively. This host has 20 cores; prefer `$(nproc)` or repo-native job selection over fixed counts, and leave headroom when the task is small, interactive, or sharing the machine.
- Keep repo-specific instructions authoritative. Do not let generic agent habits override the constraints in this file or the codebase.

**Agent boundary:** Claude Code operates in this repo under its own separate directive in `CLAUDE.md`. That file is Claude's territory. This file is Codex's territory. Neither directive governs the other agent.

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
