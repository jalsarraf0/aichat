# Vision Stack Testing Guide

## Test Structure

```
vision/tests/
├── conftest.py            # sys.path setup, shared fixtures
├── fixtures/
│   ├── __init__.py
│   └── images.py          # Test image factories (JPEG, PNG, base64)
├── unit/                  # No services required
│   ├── __init__.py
│   ├── test_config.py     # Settings, env var overrides, API key fan-out
│   ├── test_image_utils.py # Base64, MIME detection, SSRF, file path sandbox
│   ├── test_models.py     # Pydantic model validation and serialization
│   └── test_preprocessing.py # Image preprocessing for Triton models
├── integration/           # Requires individual services
│   ├── __init__.py
│   ├── test_compreface_client.py  # Requires COMPREFACE_URL
│   └── test_vision_router_client.py # Requires VISION_ROUTER_URL
└── e2e/                   # Requires full stack
    ├── __init__.py
    └── test_mcp_tools.py  # Requires MCP_SERVER_URL
```

## Running Unit Tests

Unit tests require no running services and complete in under 10 seconds.

```bash
# From the repo root
cd vision
python -m pytest tests/unit -v

# Or via Makefile
make test-unit
```

### Installing test dependencies

```bash
pip install pytest pytest-asyncio pydantic pydantic-settings fastapi httpx Pillow numpy scipy
```

Or with uv:
```bash
uv pip install pytest pytest-asyncio pydantic pydantic-settings fastapi httpx Pillow numpy scipy
```

### Expected output

```
tests/unit/test_config.py::TestDefaultConfig::test_compreface_default_url PASSED
tests/unit/test_config.py::TestDefaultConfig::test_server_default_port PASSED
tests/unit/test_config.py::TestApiKeyFanOut::test_master_key_fans_to_all_services PASSED
tests/unit/test_config.py::TestApiKeyFanOut::test_per_service_key_not_overridden_when_set PASSED
tests/unit/test_image_utils.py::TestDetectMime::test_jpeg_magic PASSED
tests/unit/test_image_utils.py::TestValidateUrl::test_blocks_localhost_hostname PASSED
tests/unit/test_image_utils.py::TestValidateUrl::test_blocks_private_ip_192_168 PASSED
tests/unit/test_models.py::TestImageSource::test_exactly_one_source_base64 PASSED
tests/unit/test_models.py::TestImageSource::test_no_source_raises PASSED
tests/unit/test_preprocessing.py::TestResizeForYolo::test_output_shape PASSED
tests/unit/test_preprocessing.py::TestResizeForYolo::test_output_range PASSED
tests/unit/test_preprocessing.py::TestResizeForEfficientnet::test_output_shape PASSED
tests/unit/test_preprocessing.py::TestResizeForClip::test_output_shape PASSED
...

=== 50+ passed in 2.14s ===
```

## Running Integration Tests

Integration tests require individual services to be reachable.

### CompreFace integration tests

```bash
COMPREFACE_URL=http://192.168.50.2:8080 python -m pytest tests/integration/test_compreface_client.py -v
```

### Vision router integration tests

```bash
VISION_ROUTER_URL=http://192.168.50.2:8090 python -m pytest tests/integration/test_vision_router_client.py -v
```

### All integration tests

```bash
COMPREFACE_URL=http://192.168.50.2:8080 \
VISION_ROUTER_URL=http://192.168.50.2:8090 \
python -m pytest tests/integration -v

# Or via Makefile (uses env vars)
make test-integration
```

## Running End-to-End Tests

E2E tests require the full vision stack (inference + MCP server).

```bash
# Start the stack first
make up-inference   # on RTX 3090 host
make up-mcp         # on Fedora host

# Run e2e tests
MCP_SERVER_URL=http://localhost:8097 python -m pytest tests/e2e -v

# Or via Makefile
make test-e2e
```

## Running All Tests

```bash
# Fast checks only (lint + type-check + unit)
make test

# All tests including integration (requires services)
COMPREFACE_URL=http://192.168.50.2:8080 \
VISION_ROUTER_URL=http://192.168.50.2:8090 \
MCP_SERVER_URL=http://localhost:8097 \
python -m pytest tests/ -v
```

## Running Benchmarks

The benchmark script measures inference latency for all tools.

```bash
# Requires the full stack to be running
MCP_SERVER_URL=http://localhost:8097 python vision/scripts/benchmark.py

# Custom iterations and URL
python vision/scripts/benchmark.py --url http://localhost:8097 --iterations 50

# Benchmark specific tool
python vision/scripts/benchmark.py --tool detect_objects --iterations 100
```

### Sample benchmark output

```
Vision MCP Inference Benchmark
  Target:     http://localhost:8097
  Iterations: 10
  Tools:      9
------------------------------------------------------------------------------------------
  list_face_subjects              p50=    12ms  p95=    18ms  p99=    22ms  min=    10ms  max=    35ms  err=0/10
  detect_faces                    p50=    85ms  p95=   120ms  p99=   145ms  min=    72ms  max=   180ms  err=0/10
  recognize_face                  p50=    92ms  p95=   138ms  p99=   155ms  min=    78ms  max=   210ms  err=0/10
  detect_objects                  p50=    18ms  p95=    27ms  p99=    32ms  min=    15ms  max=    45ms  err=0/10
  classify_image                  p50=    22ms  p95=    34ms  p99=    38ms  min=    18ms  max=    52ms  err=0/10
  detect_clothing                 p50=    35ms  p95=    52ms  p99=    58ms  min=    29ms  max=    78ms  err=0/10
  embed_image                     p50=    31ms  p95=    45ms  p99=    50ms  min=    25ms  max=    62ms  err=0/10
  analyze_image                   p50=    38ms  p95=    55ms  p99=    62ms  min=    31ms  max=    85ms  err=0/10
```

## Test Markers

Use pytest markers to select test subsets:

```bash
# Skip GPU-dependent tests
python -m pytest tests/ -m "not gpu"

# Only slow tests
python -m pytest tests/ -m slow

# Only integration tests
python -m pytest tests/ -m integration

# Only e2e tests
python -m pytest tests/ -m e2e
```

## Smoke Test

The shell-based smoke test is a quick sanity check for a running stack:

```bash
MCP_SERVER_URL=http://localhost:8097 bash vision/scripts/smoke_test.sh
```

The smoke test checks:
1. `/health` returns `{"status": "ok"}`
2. `tools/list` returns exactly 11 tools
3. All expected tool names are present
4. `list_face_subjects` succeeds (no image needed)
5. `detect_objects` accepts a base64 test image
6. `classify_image` accepts a base64 test image
7. SSRF via localhost URL is blocked

## CI Integration

Unit tests run automatically on every push to `vision/**` via
`.github/workflows/vision-ci.yml`. Integration and e2e tests are skipped in
CI because they require hardware (RTX 3090) and live services.

To check CI status, look for the "Vision Stack CI" workflow in GitHub Actions.

## Writing New Tests

### Unit test template

```python
from __future__ import annotations
import pytest

class TestMyFeature:
    def test_basic_behavior(self):
        from app.my_module import my_function  # type: ignore[import]
        result = my_function(input="test")
        assert result == "expected"

    def test_invalid_input_raises(self):
        from app.my_module import my_function  # type: ignore[import]
        with pytest.raises(ValueError, match="specific message"):
            my_function(input=None)
```

### Async test template

```python
from __future__ import annotations
import pytest

@pytest.mark.asyncio
async def test_async_operation():
    from app.clients.my_client import MyClient  # type: ignore[import]
    client = MyClient(url="http://test")
    result = await client.do_something()
    assert result["status"] == "ok"
```

### Using image fixtures

```python
def test_with_jpeg(test_jpeg_bytes, test_jpeg_b64):
    # test_jpeg_bytes: bytes — raw JPEG
    # test_jpeg_b64: str — base64-encoded JPEG
    assert test_jpeg_bytes[:3] == b"\xff\xd8\xff"
    assert len(test_jpeg_b64) > 0
```
