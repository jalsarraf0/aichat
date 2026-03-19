#!/usr/bin/env bash
# aichat image cache wrapper.
# Delegates to the universal ~/scripts/cache-image.sh.
#
# Builds and caches all 7 locally-built aichat service images in parallel.
# One tar per service — stored in /tmp/ci-cache/aichat/.
#
# Usage:
#   bash scripts/cache-image.sh                         # warm all 7 images
#   bash scripts/cache-image.sh data mcp vision         # selected services only
#
# All images stored in /tmp/ci-cache/aichat/ — one source of truth per image.

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
UNIVERSAL="${HOME}/scripts/cache-image.sh"

if [[ ! -x "${UNIVERSAL}" ]]; then
    echo "ERROR: universal cache-image.sh not found at ${UNIVERSAL}" >&2
    exit 1
fi

# Service → Dockerfile directory (image tag = aichat-<service>)
declare -A SERVICES=(
    [data]="docker/data"
    [vision]="docker/vision"
    [docs]="docker/docs"
    [sandbox]="docker/sandbox"
    [mcp]="docker/mcp"
    [whatsapp]="docker/whatsapp"
    [jupyter]="docker/jupyter"
)

ALL_SERVICES=(data vision docs sandbox mcp whatsapp jupyter)

TARGETS=("${@:-${ALL_SERVICES[@]}}")

RED='\033[0;31m'; GREEN='\033[0;32m'; CYAN='\033[0;36m'; BOLD='\033[1m'; RESET='\033[0m'
info() { echo -e "${CYAN}▶${RESET} $*"; }

info "Caching ${#TARGETS[@]} aichat image(s) in parallel → /tmp/ci-cache/aichat/"

declare -A PIDS=()
declare -A LOGS=()

for svc in "${TARGETS[@]}"; do
    dir="${SERVICES[$svc]:-}"
    if [[ -z "${dir}" ]]; then
        echo "WARNING: unknown service '${svc}' — skipping" >&2
        continue
    fi
    log="$(mktemp /tmp/aichat-cache-${svc}-XXXXXX.log)"
    LOGS[$svc]="${log}"

    (
        "${UNIVERSAL}" \
            aichat \
            "aichat-${svc}" \
            "${REPO_ROOT}/${dir}/Dockerfile" \
            "${REPO_ROOT}/${dir}" \
            >"${log}" 2>&1
    ) &
    PIDS[$svc]=$!
done

FAILED=0
for svc in "${!PIDS[@]}"; do
    if wait "${PIDS[$svc]}"; then
        echo -e "${GREEN}${BOLD}  OK${RESET}  aichat-${svc}"
        rm -f "${LOGS[$svc]}"
    else
        echo -e "${RED}${BOLD}  FAIL${RESET}  aichat-${svc}"
        echo "  Log: ${LOGS[$svc]}"
        tail -10 "${LOGS[$svc]}" | sed 's/^/  | /'
        FAILED=$((FAILED + 1))
    fi
done

if [[ $FAILED -gt 0 ]]; then
    echo -e "\n${RED}${BOLD}${FAILED} image(s) failed to cache.${RESET}" >&2
    exit 1
fi

echo -e "\n${GREEN}${BOLD}All aichat images cached in /tmp/ci-cache/aichat/${RESET}"
