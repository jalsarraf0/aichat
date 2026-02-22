SHELL := /usr/bin/env bash

.PHONY: security-checks

security-checks:
	@set -Eeuo pipefail; \
	command -v shellcheck >/dev/null || { echo 'missing shellcheck'; exit 1; }; \
	command -v shfmt >/dev/null || { echo 'missing shfmt'; exit 1; }; \
	command -v bandit >/dev/null || { echo 'missing bandit'; exit 1; }; \
	command -v safety >/dev/null || { echo 'missing safety'; exit 1; }; \
	command -v semgrep >/dev/null || { echo 'missing semgrep'; exit 1; }; \
	command -v trivy >/dev/null || { echo 'missing trivy'; exit 1; }; \
	shellcheck install.sh uninstall.sh install uninstall entrypoint scripts/bootstrap/install_aichat.sh scripts/install/install.sh scripts/uninstall/uninstall.sh; \
	shfmt -d install.sh uninstall.sh scripts/bootstrap/install_aichat.sh scripts/install/install.sh scripts/uninstall/uninstall.sh; \
	bandit -q -r src docker -lll; \
	safety check --full-report; \
	semgrep --config p/security-audit src docker; \
	trivy fs --severity MEDIUM,HIGH,CRITICAL --exit-code 1 .
