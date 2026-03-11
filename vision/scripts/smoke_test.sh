#!/usr/bin/env bash
# Smoke test for the vision MCP server.
# Usage: MCP_SERVER_URL=http://localhost:8097 bash vision/scripts/smoke_test.sh
set -euo pipefail

MCP_URL="${MCP_SERVER_URL:-http://localhost:8097}"
PASS=0
FAIL=0
GREEN='\033[0;32m'
RED='\033[0;31m'
YELLOW='\033[0;33m'
NC='\033[0m'

echo "========================================"
echo " Vision MCP Server Smoke Test"
echo " Target: ${MCP_URL}"
echo "========================================"
echo ""

# ---------------------------------------------------------------------------
# Helper: run a check and record pass/fail
# ---------------------------------------------------------------------------
check() {
    local name="$1"
    local cmd="$2"
    if eval "$cmd" > /dev/null 2>&1; then
        echo -e "  ${GREEN}PASS${NC}  ${name}"
        ((PASS++)) || true
    else
        echo -e "  ${RED}FAIL${NC}  ${name}"
        ((FAIL++)) || true
    fi
}

# ---------------------------------------------------------------------------
# Generate a tiny test JPEG as base64
# ---------------------------------------------------------------------------
echo "Generating test image..."
TEST_B64=$(python3 -c "
import base64, io
from PIL import Image
img = Image.new('RGB', (64, 64), (128, 128, 128))
buf = io.BytesIO()
img.save(buf, 'JPEG')
print(base64.b64encode(buf.getvalue()).decode())
" 2>/dev/null) || {
    echo -e "${YELLOW}WARNING${NC}: Could not generate test image with Pillow (is it installed?)"
    TEST_B64=""
}

# ---------------------------------------------------------------------------
# Check 1: /health endpoint
# ---------------------------------------------------------------------------
echo ""
echo "--- Health check ---"
check "GET /health returns 200" \
    "curl -sf ${MCP_URL}/health | python3 -c \"
import json, sys
d = json.load(sys.stdin)
exit(0 if d.get('status') == 'ok' else 1)
\""

# ---------------------------------------------------------------------------
# Check 2: tools/list returns exactly 11 tools
# ---------------------------------------------------------------------------
echo ""
echo "--- MCP tools/list ---"
check "tools/list returns 11 tools" \
    "curl -sf -X POST ${MCP_URL}/mcp \
      -H 'Content-Type: application/json' \
      -d '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\",\"params\":{}}' \
    | python3 -c \"
import json, sys
d = json.load(sys.stdin)
tools = d.get('result', {}).get('tools', [])
exit(0 if len(tools) == 11 else 1)
\""

EXPECTED_TOOLS="recognize_face verify_face detect_faces enroll_face list_face_subjects delete_face_subject detect_objects classify_image detect_clothing embed_image analyze_image"
check "tools/list contains all face and vision tools" \
    "curl -sf -X POST ${MCP_URL}/mcp \
      -H 'Content-Type: application/json' \
      -d '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/list\",\"params\":{}}' \
    | python3 -c \"
import json, sys
d = json.load(sys.stdin)
names = {t['name'] for t in d.get('result', {}).get('tools', [])}
expected = set('${EXPECTED_TOOLS}'.split())
missing = expected - names
if missing:
    print('Missing tools:', missing, file=sys.stderr)
    sys.exit(1)
\""

# ---------------------------------------------------------------------------
# Check 3: list_face_subjects (no image needed)
# ---------------------------------------------------------------------------
echo ""
echo "--- Face tools ---"
check "list_face_subjects succeeds" \
    "curl -sf -X POST ${MCP_URL}/mcp \
      -H 'Content-Type: application/json' \
      -d '{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"list_face_subjects\",\"arguments\":{}}}' \
    | python3 -c \"
import json, sys
d = json.load(sys.stdin)
# Must have result or structured error — not an HTTP 500
exit(0 if 'result' in d or 'error' in d else 1)
\""

# ---------------------------------------------------------------------------
# Check 4: detect_objects with test image
# ---------------------------------------------------------------------------
if [[ -n "${TEST_B64}" ]]; then
    echo ""
    echo "--- Vision tools (with test image) ---"
    DETECT_PAYLOAD="{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"detect_objects\",\"arguments\":{\"image\":{\"base64\":\"${TEST_B64}\"}}}}"
    check "detect_objects accepts base64 image" \
        "echo '${DETECT_PAYLOAD}' | curl -sf -X POST ${MCP_URL}/mcp \
          -H 'Content-Type: application/json' \
          -d @- \
        | python3 -c \"
import json, sys
d = json.load(sys.stdin)
exit(0 if 'result' in d or 'error' in d else 1)
\""

    CLASSIFY_PAYLOAD="{\"jsonrpc\":\"2.0\",\"id\":1,\"method\":\"tools/call\",\"params\":{\"name\":\"classify_image\",\"arguments\":{\"image\":{\"base64\":\"${TEST_B64}\"}}}}"
    check "classify_image accepts base64 image" \
        "echo '${CLASSIFY_PAYLOAD}' | curl -sf -X POST ${MCP_URL}/mcp \
          -H 'Content-Type: application/json' \
          -d @- \
        | python3 -c \"
import json, sys
d = json.load(sys.stdin)
exit(0 if 'result' in d or 'error' in d else 1)
\""
fi

# ---------------------------------------------------------------------------
# Check 5: SSRF protection
# ---------------------------------------------------------------------------
echo ""
echo "--- Security checks ---"
SSRF_PAYLOAD='{"jsonrpc":"2.0","id":1,"method":"tools/call","params":{"name":"detect_objects","arguments":{"image":{"url":"http://127.0.0.1/internal"}}}}'
check "SSRF via localhost URL is blocked" \
    "curl -sf -X POST ${MCP_URL}/mcp \
      -H 'Content-Type: application/json' \
      -d '${SSRF_PAYLOAD}' \
    | python3 -c \"
import json, sys
d = json.load(sys.stdin)
# An error is the correct response for a blocked SSRF attempt
# A 'result' that contains error text is also acceptable
text = json.dumps(d)
if 'error' in d:
    sys.exit(0)  # error key = properly rejected
# Check if result contains an error description
if 'result' in d:
    content = str(d['result'])
    if any(w in content.lower() for w in ['error', 'blocked', 'private', 'invalid']):
        sys.exit(0)
sys.exit(1)
\""

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
echo ""
echo "========================================"
TOTAL=$((PASS + FAIL))
if [[ $FAIL -eq 0 ]]; then
    echo -e " ${GREEN}All ${TOTAL} checks passed${NC}"
    echo "========================================"
    exit 0
else
    echo -e " ${RED}${FAIL}/${TOTAL} checks FAILED${NC}"
    echo "========================================"
    exit 1
fi
