# Vision MCP Stack

GPU-first vision intelligence for the aichat MCP gateway. This stack provides
11 MCP tools covering face recognition (via CompreFace) and general computer
vision (object detection, image classification, clothing detection, and
semantic embeddings via NVIDIA Triton Inference Server on an RTX 3090).

## Architecture

```
                  ┌─────────────────────────────────┐
                  │         Fedora Host              │
                  │                                 │
                  │  ┌──────────────────────────┐   │
    MCP clients ──┼─►│  vision-mcp  :8097       │   │
    (aichat-mcp)  │  │  FastAPI + JSON-RPC 2.0  │   │
                  │  └───────────┬──────────────┘   │
                  └──────────────┼──────────────────┘
                                 │ HTTP
              ┌──────────────────┼───────────────────┐
              │  RTX 3090 Host   │                   │
              │                 │                   │
              │  ┌──────────────▼──────────────┐    │
              │  │  vision-router  :8090        │    │
              │  │  FastAPI preprocessing +     │    │
              │  │  Triton client               │    │
              │  └──────────────┬──────────────┘    │
              │                 │ gRPC              │
              │  ┌──────────────▼──────────────┐    │
              │  │  Triton Inference Server     │    │
              │  │  :8000 (HTTP) :8001 (gRPC)  │    │
              │  │  Models: yolov8n, efficientnet│   │
              │  │          clip_vit_b32,        │   │
              │  │          fashion_clip          │   │
              │  └─────────────────────────────┘    │
              │                                     │
              │  ┌──────────────────────────────┐   │
              │  │  CompreFace  :8080            │   │
              │  │  Face recognition service    │   │
              │  │  (PostgreSQL backend)        │   │
              │  └──────────────────────────────┘   │
              └─────────────────────────────────────┘
```

## Quick Start

### Step 1 — Start inference stack on RTX 3090 host

```bash
# Download ONNX models (first time only, ~2 GB)
make download-models

# Start Triton + CompreFace
make up-inference
```

### Step 2 — Start MCP server on Fedora host

```bash
# Copy and edit environment file
cp vision/mcp-server/.env.example vision/mcp-server/.env
# Edit COMPREFACE_URL, VISION_ROUTER_URL to point to RTX 3090 host

make up-mcp

# Verify
make smoke
```

## MCP Tools

All tools accept image input in one of three forms:
- `{"base64": "<base64-encoded-bytes>"}` — most common, no network call
- `{"url": "https://..."}` — public URLs only (SSRF-protected)
- `{"file_path": "/workspace/..."}` — server-local paths (sandboxed to /workspace, /tmp, /data)

### Face Tools (CompreFace backend)

#### recognize_face
Recognize faces in an image against enrolled subjects.

**Input:**
```json
{
  "image": {"base64": "<b64>"},
  "min_confidence": 0.7,
  "limit": 5
}
```

**Output:**
```json
{
  "faces": [
    {
      "box": {"x_min": 120, "y_min": 80, "x_max": 280, "y_max": 320, "confidence": 0.99},
      "matches": [{"subject": "alice", "similarity": 0.94}]
    }
  ],
  "count": 1,
  "timing": {"total_ms": 85.2, "backend_ms": 72.1},
  "backend": {"name": "compreface", "host": "192.168.50.2"}
}
```

#### verify_face
Compare two face images and determine if they are the same person.

**Input:**
```json
{
  "image_a": {"base64": "<b64_a>"},
  "image_b": {"base64": "<b64_b>"},
  "min_similarity": 0.85
}
```

**Output:**
```json
{
  "verified": true,
  "similarity": 0.92,
  "subject_a_face_count": 1,
  "subject_b_face_count": 1,
  "timing": {"total_ms": 110.0}
}
```

#### detect_faces
Detect all faces in an image without recognition.

**Input:**
```json
{
  "image": {"base64": "<b64>"},
  "min_confidence": 0.7,
  "return_landmarks": false
}
```

#### enroll_face
Add a face image to the recognition database under a subject name.

**Input:**
```json
{
  "image": {"base64": "<b64>"},
  "subject_name": "alice"
}
```

**Output:**
```json
{
  "subject": "alice",
  "image_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "timing": {"total_ms": 210.5}
}
```

#### list_face_subjects
List all enrolled subject names.

**Input:** `{}` (no arguments)

**Output:**
```json
{
  "subjects": ["alice", "bob", "charlie"],
  "count": 3,
  "backend": {"name": "compreface"}
}
```

#### delete_face_subject
Delete all face images for a subject.

**Input:**
```json
{"subject_name": "alice"}
```

### Vision Tools (Triton backend via vision-router)

#### detect_objects
Detect objects using YOLOv8n (80 COCO classes).

**Input:**
```json
{
  "image": {"base64": "<b64>"},
  "min_confidence": 0.4,
  "max_results": 20,
  "classes": ["person", "car"]
}
```

**Output:**
```json
{
  "objects": [
    {
      "label": "person",
      "confidence": 0.92,
      "box": {"x_min": 50, "y_min": 100, "x_max": 200, "y_max": 450, "confidence": 0.92}
    }
  ],
  "count": 1,
  "timing": {"total_ms": 18.3, "backend_ms": 12.1, "preprocess_ms": 3.2, "postprocess_ms": 3.0},
  "backend": {"name": "triton", "model": "yolov8n"}
}
```

#### classify_image
Classify image using EfficientNet-B0 (ImageNet-1K, 1000 classes).

**Input:**
```json
{
  "image": {"base64": "<b64>"},
  "top_k": 5,
  "min_confidence": 0.01
}
```

**Output:**
```json
{
  "labels": [
    {"name": "golden retriever", "confidence": 0.87},
    {"name": "Labrador retriever", "confidence": 0.08}
  ],
  "top_label": "golden retriever",
  "top_confidence": 0.87,
  "timing": {"total_ms": 22.1}
}
```

#### detect_clothing
Detect clothing items using FashionCLIP (zero-shot, 40+ categories).

**Input:**
```json
{
  "image": {"base64": "<b64>"},
  "min_confidence": 0.15,
  "top_k": 5
}
```

#### embed_image
Generate semantic embeddings using CLIP ViT-B/32 (512-dim).

**Input:**
```json
{
  "image": {"base64": "<b64>"},
  "model": "clip_vit_b32",
  "normalize": true
}
```

**Output:**
```json
{
  "embeddings": [0.023, -0.142, ...],
  "dim": 512,
  "model": "clip_vit_b32",
  "normalized": true,
  "timing": {"total_ms": 31.4}
}
```

#### analyze_image
Run multiple vision analyses in parallel (objects + classification + optional clothing/embeddings).

**Input:**
```json
{
  "image": {"base64": "<b64>"},
  "include_objects": true,
  "include_classification": true,
  "include_clothing": false,
  "include_embeddings": false
}
```

## Environment Variables

| Variable | Default | Description |
|---|---|---|
| `VISION_COMPREFACE__URL` | `http://192.168.50.2:8080` | CompreFace service URL |
| `VISION_COMPREFACE__API_KEY` | `""` | Master API key (fans out to per-service keys) |
| `VISION_COMPREFACE__RECOGNITION_API_KEY` | `""` | CompreFace Recognition API key |
| `VISION_COMPREFACE__DETECTION_API_KEY` | `""` | CompreFace Detection API key |
| `VISION_COMPREFACE__VERIFICATION_API_KEY` | `""` | CompreFace Verification API key |
| `VISION_COMPREFACE__SIMILARITY_THRESHOLD` | `0.85` | Minimum similarity for face matches |
| `VISION_COMPREFACE__TIMEOUT_S` | `30.0` | HTTP timeout for CompreFace calls |
| `VISION_ROUTER__URL` | `http://192.168.50.2:8090` | Vision router service URL |
| `VISION_ROUTER__TIMEOUT_S` | `60.0` | HTTP timeout for router calls |
| `VISION_SERVER__PORT` | `8097` | MCP server listen port |
| `VISION_SERVER__WORKERS` | `2` | Uvicorn worker count |
| `VISION_ENABLE_COMPREFACE` | `true` | Enable/disable CompreFace tools |
| `VISION_ENABLE_TRITON` | `true` | Enable/disable Triton vision tools |

## Models

| Model | Task | Input | Output | RTX 3090 FPS |
|---|---|---|---|---|
| YOLOv8n | Object detection | 640×640 | 80 COCO classes + boxes | ~250 |
| EfficientNet-B0 | Classification | 224×224 | 1000 ImageNet classes | ~400 |
| CLIP ViT-B/32 | Embeddings | 224×224 | 512-dim float32 | ~300 |
| FashionCLIP | Clothing detection | 224×224 | 40+ clothing categories | ~280 |

## Troubleshooting

**MCP server returns `backend_unavailable`:**
- Check `VISION_ROUTER__URL` and `VISION_COMPREFACE__URL` point to the correct hosts
- Run `make smoke` to pinpoint which backend is failing
- Check `docker compose -f vision/compose/inference.yml logs` on the RTX 3090 host

**Triton returns `model not found`:**
- Run `make download-models` to download and export ONNX models
- Verify `/opt/triton-models/` contains the model repository directories

**Face recognition returns `no faces detected`:**
- Ensure image is at least 100×100 pixels
- Lower `det_prob_threshold` (default 0.8) for difficult images
- CompreFace works best with frontal faces, good lighting, >50px face size

**GPU out of memory:**
- Reduce concurrent requests or restart the Triton container
- Check GPU usage: `nvidia-smi -l 1` on the RTX 3090 host

## Further Reading

- [Architecture](docs/architecture.md) — detailed service topology
- [API Reference](docs/api.md) — complete tool and REST endpoint docs
- [Models](docs/models.md) — model details and how to add new ones
- [Testing](docs/testing.md) — how to run tests and benchmarks
- [Security](docs/security.md) — SSRF protection, sandboxing, API keys
- [Operations](docs/operations.md) — deployment, monitoring, scaling
