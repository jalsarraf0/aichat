SHELL := /usr/bin/env bash
COMPOSE := docker compose
SERVICES := aichat-data aichat-vision aichat-docs aichat-sandbox aichat-mcp

.PHONY: help build up down restart logs smoke test lint security-checks \
        generate-lmstudio-json

# ---------------------------------------------------------------------------
# Help
# ---------------------------------------------------------------------------
help:
	@echo "aichat platform — common targets"
	@echo ""
	@echo "  make build               Build all service images"
	@echo "  make up                  Start full stack (detached)"
	@echo "  make down                Stop and remove containers"
	@echo "  make restart             down + up"
	@echo "  make logs                Follow logs for all services"
	@echo "  make smoke               Quick health-endpoint check"
	@echo "  make test                Run full pytest test suite"
	@echo "  make lint                Run ruff + mypy on docker/**/*.py"
	@echo "  make security-checks     shellcheck/bandit/safety/semgrep/trivy"
	@echo "  make generate-lmstudio-json  Regenerate lmstudio-mcp.json from live /tools"

# ---------------------------------------------------------------------------
# Build / stack management
# ---------------------------------------------------------------------------
build:
	$(COMPOSE) build $(SERVICES)

up:
	$(COMPOSE) up -d

down:
	$(COMPOSE) down

restart: down up

logs:
	$(COMPOSE) logs -f

# ---------------------------------------------------------------------------
# Smoke test: hit /health on every application service
# ---------------------------------------------------------------------------
smoke:
	@set -e; \
	declare -A PORTS=( \
	  [aichat-data]=8091 \
	  [aichat-vision]=8099 \
	  [aichat-docs]=8101 \
	  [aichat-sandbox]=8095 \
	  [aichat-mcp]=8096 \
	  [aichat-searxng]=8098 \
	); \
	declare -A HEALTH_PATHS=( \
	  [aichat-searxng]="/" \
	); \
	for svc in "$${!PORTS[@]}"; do \
	  port=$${PORTS[$$svc]}; \
	  path=$${HEALTH_PATHS[$$svc]:-/health}; \
	  url="http://localhost:$${port}$${path}"; \
	  echo -n "  $$svc ($$url) ... "; \
	  status=$$(curl -s -o /dev/null -w "%{http_code}" --max-time 5 "$$url"); \
	  if [ "$$status" = "200" ]; then echo "OK"; \
	  else echo "FAIL (HTTP $$status)"; exit 1; fi; \
	done; \
	echo "All health checks passed."

# ---------------------------------------------------------------------------
# Tests (requires: pip install pytest httpx)
# ---------------------------------------------------------------------------
test:
	python3 -m pytest tests/ -v

# ---------------------------------------------------------------------------
# Lint (requires: pip install ruff mypy)
# ---------------------------------------------------------------------------
lint:
	@command -v ruff >/dev/null || { echo 'missing ruff — pip install ruff'; exit 1; }
	ruff check docker/
	@command -v mypy >/dev/null && mypy --ignore-missing-imports docker/ || true

# ---------------------------------------------------------------------------
# Security checks (existing targets preserved)
# ---------------------------------------------------------------------------
security-checks:
	@set -Eeuo pipefail; \
	command -v shellcheck >/dev/null || { echo 'missing shellcheck'; exit 1; }; \
	command -v shfmt      >/dev/null || { echo 'missing shfmt';      exit 1; }; \
	command -v bandit     >/dev/null || { echo 'missing bandit';     exit 1; }; \
	command -v safety     >/dev/null || { echo 'missing safety';     exit 1; }; \
	command -v semgrep    >/dev/null || { echo 'missing semgrep';    exit 1; }; \
	command -v trivy      >/dev/null || { echo 'missing trivy';      exit 1; }; \
	shellcheck install.sh uninstall.sh entrypoint \
	  scripts/bin/install scripts/bin/uninstall \
	  scripts/bootstrap/install_aichat.sh; \
	shfmt -d install.sh uninstall.sh entrypoint \
	  scripts/bin/install scripts/bin/uninstall \
	  scripts/bootstrap/install_aichat.sh; \
	bandit -q -r src docker -lll; \
	safety check --full-report; \
	semgrep --config p/security-audit src docker; \
	trivy fs --severity MEDIUM,HIGH,CRITICAL --exit-code 1 .

# ---------------------------------------------------------------------------
# Generate lmstudio-mcp.json from the live MCP gateway
# ---------------------------------------------------------------------------
generate-lmstudio-json:
	@echo "Fetching tool list from http://localhost:8096/tools ..."
	@python3 .github/scripts/gen_lmstudio_json.py \
	  --mcp-url http://localhost:8096 \
	  --out lmstudio-mcp.json
	@echo "Written: lmstudio-mcp.json"
