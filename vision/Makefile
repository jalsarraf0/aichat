.PHONY: help lint type-check test test-unit test-integration test-e2e \
        build-inference up-inference down-inference \
        build-mcp up-mcp down-mcp \
        smoke benchmark download-models sbom security clean

SHELL := /bin/bash
COMPOSE_INFERENCE := docker compose -f vision/compose/inference.yml
COMPOSE_MCP       := docker compose -f vision/compose/mcp.yml

help: ## Show this help message
	@grep -E '^[a-zA-Z_-]+:.*?## .*$$' $(MAKEFILE_LIST) | \
	  awk 'BEGIN {FS=":.*?## "}; {printf "  \033[36m%-22s\033[0m %s\n", $$1, $$2}'

# ---------------------------------------------------------------------------
# Code quality
# ---------------------------------------------------------------------------

lint: ## Run ruff linter on vision stack source code
	ruff check vision/mcp-server/app vision/services/vision-router/app

lint-fix: ## Run ruff linter and auto-fix safe issues
	ruff check --fix vision/mcp-server/app vision/services/vision-router/app

type-check: ## Run mypy type checker on vision stack source code
	mypy vision/mcp-server/app vision/services/vision-router/app --ignore-missing-imports

# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------

test-unit: ## Run unit tests (no services needed)
	cd vision && python -m pytest tests/unit -v --tb=short

test-integration: ## Run integration tests (requires running services)
	cd vision && python -m pytest tests/integration -v --tb=short

test-e2e: ## Run end-to-end tests (requires full stack running)
	cd vision && python -m pytest tests/e2e -v --tb=short

test: lint type-check test-unit ## Run all fast checks (lint + type-check + unit tests)

# ---------------------------------------------------------------------------
# Model management
# ---------------------------------------------------------------------------

download-models: ## Download and export ONNX models for Triton
	bash vision/triton/scripts/download_models.sh

# ---------------------------------------------------------------------------
# Inference stack (RTX 3090 host)
# ---------------------------------------------------------------------------

build-inference: ## Build inference stack images (RTX 3090)
	$(COMPOSE_INFERENCE) build

up-inference: ## Start inference stack on RTX 3090
	$(COMPOSE_INFERENCE) up -d

down-inference: ## Stop inference stack
	$(COMPOSE_INFERENCE) down

logs-inference: ## Tail inference stack logs
	$(COMPOSE_INFERENCE) logs -f

# ---------------------------------------------------------------------------
# MCP server (Fedora host)
# ---------------------------------------------------------------------------

build-mcp: ## Build vision MCP server image
	$(COMPOSE_MCP) build

up-mcp: ## Start vision MCP server
	$(COMPOSE_MCP) up -d

down-mcp: ## Stop vision MCP server
	$(COMPOSE_MCP) down

logs-mcp: ## Tail MCP server logs
	$(COMPOSE_MCP) logs -f

# ---------------------------------------------------------------------------
# Smoke test and benchmarks
# ---------------------------------------------------------------------------

smoke: ## Run smoke test against the running MCP server
	bash vision/scripts/smoke_test.sh

benchmark: ## Run inference latency benchmark against the running MCP server
	cd vision && python scripts/benchmark.py

# ---------------------------------------------------------------------------
# Security and SBOM
# ---------------------------------------------------------------------------

sbom: ## Generate SBOMs for mcp-server and vision-router
	syft vision/mcp-server -o cyclonedx-json > vision/sbom-mcp.json
	syft vision/services/vision-router -o cyclonedx-json > vision/sbom-router.json
	@echo "SBOMs written to vision/sbom-mcp.json and vision/sbom-router.json"

security: ## Run bandit security scan on vision stack
	bandit -r vision/mcp-server/app vision/services/vision-router/app \
	  -f json -o vision/bandit-report.json || true
	@python3 -c "\
import json, sys; \
r = json.load(open('vision/bandit-report.json')); \
results = r.get('results', []); \
high = len([x for x in results if x['issue_severity'] == 'HIGH']); \
med  = len([x for x in results if x['issue_severity'] == 'MEDIUM']); \
print(f'Security scan: High={high} Medium={med} Total={len(results)}'); \
sys.exit(1 if high > 0 else 0)"

# ---------------------------------------------------------------------------
# Housekeeping
# ---------------------------------------------------------------------------

clean: ## Remove build artifacts, caches, and generated reports
	find vision -type d -name __pycache__ | xargs rm -rf
	find vision -name "*.pyc" -delete
	find vision -name ".pytest_cache" -type d | xargs rm -rf
	find vision -name "*.egg-info" -type d | xargs rm -rf
	rm -f vision/sbom-*.json vision/bandit-report.json
	@echo "Clean complete"
