# Dartboard → Aichat Unified Refactor Log

## Session: 2026-03-18

### Phase 1: Backend Config Fixes (COMPLETE)
- **model_profiles.dart**: Fixed stale dolphin ID from `dolphin-mistral-24b-venice-edition` to `dolphin-mistral-glm-4.7-flash-24b-venice-edition-thinking-uncensored-i1`
- **model_profiles.dart**: Set `supportsReasoning: true` for dolphin (new model has "thinking" capability)
- **model_profiles.dart**: Implemented tool tier system — reduced from 16 tools to max 9 per model:
  - Strong models (gpt-oss-20b, dolphin): 9 tools (Default 7 + media, data)
  - Reasoning VLMs (qwen3.5-9b): 7 tools (Default tier)
  - GLM-4.6v-flash: 7 tools (probed subset)
  - Granite-tiny: 5 tools (minimal reliable set)
  - Phi-4-mini-reasoning: 2 tools (web, browser only)
- **model_profiles.dart**: Commented out stale models not in live inventory (ministral, deepseek)
- **personalities.dart**: Fixed dolphin allowedModels pattern from `['dolphin-mistral-24b-venice']` to `['dolphin']` for substring match compatibility
- **e2e_regression.py**: Updated model IDs, removed stale models from test matrix
- Validation: `dart analyze` clean (2 pre-existing warnings in router.dart), `dart test` 21/21 pass

### Phase 2: Frontend Refactor (COMPLETE)
- **style.css**: Replaced streaming cursor → blinking block cursor (`\2588`)
- **style.css**: Added `.stream-stats` for tok/s counter and elapsed time
- **style.css**: Replaced `.tool-images` flex → CSS Grid (responsive, auto-fill minmax(200px, 1fr))
- **style.css**: Added `.single-image` variant, lightbox overlay, model meta rows
- **app.js**: Added `getModelMeta()`, `formatCtx()`, `normalizeImageUrl()` utility functions
- **app.js**: Enhanced `renderModelMenu()` — state dots, quantization, ctx length, type, sorting (loaded first)
- **app.js**: Embedded models filter out from selector
- **app.js**: Added lightbox (`openLightbox`/`closeLightbox`) with keyboard nav (Escape, arrows)
- **app.js**: Refactored `send()` streaming loop:
  - Separated tool cards from thinking card (dedicated `makeToolCard()` per tool_start)
  - Live elapsed timer on tool cards (1s interval)
  - Token-per-second counter + line count + elapsed time stats bar
  - Image dedup uses `normalizeImageUrl()` (strips CDN size suffixes)
  - Mid-stream model/conversation switch guards
  - Double-done guard (`isDone` flag)
  - Orphan spinner cleanup on stream end
- **app.js**: Tightened `cleanResponse()` — removed aggressive number/JSON stripping, kept image markdown strip
- **app.js**: Updated tool count display from `/16` to dynamic count
- Validation: `node -c app.js` clean, `dart test` 21/21 pass

### Phase 3: Redis/Valkey (COMPLETE)
- **docker-compose.yml**: Added `aichat-redis` service (valkey:8-alpine)
  - 512MB max memory, LRU eviction, AOF persistence
  - Data persisted to `/mnt/nvmeINT/aichat/redis/`
  - Internal port 6379 only (no host binding)
  - Healthcheck: `valkey-cli ping`
- Created `/mnt/nvmeINT/aichat/redis/` directory
- Arc A380 preprocessing design documented (see below)

### Phase 4: TUI Polish (COMPLETE)
- **themes/dark.tcss**: Aligned colors with dartboard web palette (#212121 bg, #7c6bf5 accent)
- **model_labels.py**: Enhanced capability detection — added VLM patterns (vl, 4v, 4.6v, vlm), reasoning (qwen3.5, phi-4, think, glm), unrestricted (dolphin, uncensored), embedding detection
- **keybind_bar.py**: Added DEFAULT_CSS for consistent height/padding
- All keybindings unchanged: F1-F12, Ctrl+S, Ctrl+G

### Phase 5-6: Merge + Docker (COMPLETE)
- Copied dartboard source into aichat:
  - `lib/*.dart` → `~/git/aichat/lib/` (10 Dart files: router, llm_client, mcp_client, etc.)
  - `bin/server.dart` → `~/git/aichat/bin/`
  - `web/*` → `~/git/aichat/docker/web/web/` (app.js, style.css, index.html)
  - `docker/dart-board/Dockerfile` → `~/git/aichat/docker/web/`
  - `docker/dartboard-auth/*` → `~/git/aichat/docker/auth/`
  - `test/*.dart` → `~/git/aichat/test/dart/`
  - `pubspec.yaml`, `analysis_options.yaml` → `~/git/aichat/`
- **docker-compose.yml**: Added 3 new services:
  - `aichat-web` (Dart/Shelf, port 8200 internal, depends on MCP + Redis)
  - `aichat-auth` (Flask JWT proxy, ports 8200/8247 host-facing)
  - `aichat-redis` (Valkey, 512MB, AOF, /mnt/nvmeINT/aichat/redis/)
- **Makefile**: Added `aichat-web`, `aichat-auth`, `aichat-redis` to SERVICES
  - Added Dart targets: dart-get, dart-analyze, dart-test, dart-build, dart-run
  - Added aichat-web to smoke test
- Created NVMe persistence dirs: `/mnt/nvmeINT/aichat/redis/`, `/mnt/nvmeINT/aichat/web-db/`

### Phase 7-8: Repo Cleanup + CI (COMPLETE)
- **dart-ci.yml**: Created separate Dart CI workflow (lint, test, frontend syntax check)
  - Triggers only on Dart/web path changes (lib/, bin/, test/dart/, pubspec.*, docker/web/, docker/auth/)
  - Does not conflict with orchestrator-generated ci.yml
- **docs/ci-orchestrator.md**: Documented Haskell orchestrator relationship and notification flow
- GitHub repo deletion: PENDING user confirmation (will archive first, then delete)

## Arc A380 Preprocessing + Redis Architecture
```
User Request → Dart Server (lib/router.dart)
  ├── 1. Check Redis: compact:{conv_id}:{msg_count} → cached context (TTL 1h)
  ├── 2. If miss: Arc A380 (qwen2.5-3b on :1235) compacts conversation → cache in Redis
  ├── 3. Arc A380: route tools (existing, 3s timeout) → cache in Redis (TTL 60s)
  ├── 4. Arc A380: optimize prompt for condensed models (existing, 5s timeout) → cached in-memory
  └── 5. RTX 3090 (:1234): main generation with compacted context + selected tools
```
Fallback: If Redis unreachable, proceed without cache (graceful degradation).
Bypass: Short conversations (< compaction threshold) skip Arc compaction entirely.

## Ports (UNCHANGED)
- Auth proxy: :8200 (user-facing), :8247 (admin)
- Dart server: :8200 (internal)
- LM Studio RTX 3090: 192.168.50.2:1234
- Tool Router Arc A380: :1235
- MCP Server: :8096

## Models (Live Inventory)
1. qwen2.5-3b-instruct (Arc A380, router) - LOADED
2. dolphin-mistral-glm-4.7-flash-24b-venice-edition-thinking-uncensored-i1
3. openai/gpt-oss-20b
4. zai-org/glm-4.6v-flash (VLM)
5. qwen/qwen3.5-9b (VLM, reasoning)
6. ibm/granite-4-h-tiny
7. microsoft/phi-4-mini-reasoning
8. text-embedding-nomic-embed-text-v1.5
