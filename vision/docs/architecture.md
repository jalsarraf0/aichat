# Vision Stack Architecture

## Overview

The vision stack splits across two physical machines to separate GPU inference
from the general-purpose MCP gateway:

- **RTX 3090 host** — runs Triton Inference Server and CompreFace (both
  require GPU or benefit from it for face processing)
- **Fedora host** — runs the lightweight vision-mcp FastAPI server that
  translates MCP JSON-RPC calls into backend HTTP calls

The split keeps the MCP server simple and stateless. All model state lives
on the RTX 3090 host; the MCP server is horizontally scalable.

## Full Architecture Diagram

```
┌─────────────────────────────────────────────────────────────────────────────┐
│  CLIENT LAYER                                                               │
│                                                                             │
│   aichat-mcp (:8096)     LM Studio / Claude      Any JSON-RPC 2.0 client   │
│        │                       │                         │                  │
└────────┼───────────────────────┼─────────────────────────┼─────────────────┘
         │ HTTP POST /mcp        │                         │
         └───────────────────────┴─────────────────────────┘
                                 │
┌────────────────────────────────▼────────────────────────────────────────────┐
│  FEDORA HOST                                                                │
│                                                                             │
│  ┌─────────────────────────────────────────────────────────────────────┐   │
│  │  vision-mcp  (port 8097)                                            │   │
│  │                                                                     │   │
│  │  FastAPI application                                                │   │
│  │  ├── POST /mcp   — JSON-RPC 2.0 tool dispatch                      │   │
│  │  ├── GET  /health — service health + backend reachability           │   │
│  │  └── GET  /metrics — Prometheus metrics (optional)                  │   │
│  │                                                                     │   │
│  │  Tool routing:                                                      │   │
│  │  ├── recognize_face ────────────────────────────────────────┐      │   │
│  │  ├── verify_face ───────────────────────────────────────────┤      │   │
│  │  ├── detect_faces ──────────────────────────────────────────┤      │   │
│  │  ├── enroll_face ───────────────────────────────────────────┤      │   │
│  │  ├── list_face_subjects ────────────────────────────────────┤      │   │
│  │  └── delete_face_subject ───────────────────────────────────┘      │   │
│  │       (CompreFaceClient — HTTP to RTX 3090 host :8080)             │   │
│  │                                                                     │   │
│  │  ├── detect_objects ────────────────────────────────────────┐      │   │
│  │  ├── classify_image ────────────────────────────────────────┤      │   │
│  │  ├── detect_clothing ───────────────────────────────────────┤      │   │
│  │  ├── embed_image ───────────────────────────────────────────┤      │   │
│  │  └── analyze_image ─────────────────────────────────────────┘      │   │
│  │       (VisionRouterClient — HTTP to RTX 3090 host :8090)           │   │
│  │                                                                     │   │
│  │  Image loading pipeline:                                            │   │
│  │  base64 input → validate → decode → size check                     │   │
│  │  URL input    → SSRF check → fetch → size check                    │   │
│  │  file input   → path sandbox check → read                          │   │
│  └─────────────────────────────────────────────────────────────────────┘   │
└─────────────────────────────────────────────────────┼───────────────────────┘
                                                       │
                              ┌────────────────────────┘
                              │ HTTP (LAN / VPN)
┌─────────────────────────────▼───────────────────────────────────────────────┐
│  RTX 3090 HOST (192.168.50.2)                                               │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  vision-router  (port 8090)                                        │    │
│  │                                                                    │    │
│  │  FastAPI preprocessing service                                     │    │
│  │  ├── POST /v1/detect-objects   — YOLOv8n inference + NMS          │    │
│  │  ├── POST /v1/classify          — EfficientNet-B0 + softmax        │    │
│  │  ├── POST /v1/detect-clothing   — FashionCLIP zero-shot            │    │
│  │  ├── POST /v1/embed             — CLIP ViT-B/32 embedding          │    │
│  │  ├── POST /v1/analyze           — parallel multi-model inference   │    │
│  │  └── GET  /v1/health            — Triton reachability check        │    │
│  │                                                                    │    │
│  │  Preprocessing:                                                    │    │
│  │  ├── decode_b64_image → PIL.Image                                  │    │
│  │  ├── resize_for_yolo (letterbox 640×640, CHW float32 [0,1])       │    │
│  │  ├── resize_for_efficientnet (crop 224×224, ImageNet norm)         │    │
│  │  └── resize_for_clip (crop 224×224, CLIP norm)                    │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│           │ gRPC (:8001)                                                    │
│  ┌────────▼───────────────────────────────────────────────────────────┐    │
│  │  NVIDIA Triton Inference Server  (HTTP :8000, gRPC :8001)          │    │
│  │                                                                    │    │
│  │  Model repository (/opt/triton-models/):                           │    │
│  │  ├── yolov8n/       — YOLOv8 nano object detection (ONNX)         │    │
│  │  ├── efficientnet/  — EfficientNet-B0 classification (ONNX)       │    │
│  │  ├── clip_vit_b32/  — CLIP image encoder (ONNX)                   │    │
│  │  └── fashion_clip/  — FashionCLIP clothing (ONNX)                 │    │
│  │                                                                    │    │
│  │  GPU: NVIDIA RTX 3090 (24 GB VRAM)                                │    │
│  │  Runtime: TensorRT (FP16 optimization enabled)                    │    │
│  └────────────────────────────────────────────────────────────────────┘    │
│                                                                             │
│  ┌────────────────────────────────────────────────────────────────────┐    │
│  │  CompreFace  (port 8080)                                           │    │
│  │                                                                    │    │
│  │  Face recognition REST API                                        │    │
│  │  Services: recognition, detection, verification                   │    │
│  │  Backend: PostgreSQL (face embeddings store)                      │    │
│  │  GPU: RTX 3090 (optional, accelerates face processing)            │    │
│  └────────────────────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Service Descriptions

### vision-mcp (Fedora, :8097)

The MCP gateway. Exposes a single `POST /mcp` endpoint that speaks
[JSON-RPC 2.0](https://www.jsonrpc.org/specification) and implements the
[Model Context Protocol](https://modelcontextprotocol.io/). Responsibilities:

- Parse incoming `tools/call` requests
- Validate and load image bytes from base64, URL, or file path
- Route to the appropriate backend client (CompreFaceClient or VisionRouterClient)
- Translate backend responses into MCP-compatible tool results
- Enforce security policies (SSRF protection, file path sandboxing, size limits)

### vision-router (RTX 3090, :8090)

Preprocessing and Triton orchestration service. Responsibilities:

- Decode base64 images to PIL Images
- Apply model-specific preprocessing (letterbox, center crop, normalization)
- Build Triton InferInput tensors and submit gRPC inference requests
- Apply postprocessing (NMS for YOLO, softmax for classification, L2 norm for embeddings)
- Return structured JSON results

### Triton Inference Server (RTX 3090, :8000/:8001)

NVIDIA Triton serves ONNX models with TensorRT optimization. It handles
batching, concurrent model execution, and GPU memory management. Models are
loaded at startup from the repository at `/opt/triton-models/`.

### CompreFace (RTX 3090, :8080)

Open-source face recognition system. Maintains a PostgreSQL database of
face embeddings organized into named subjects. The REST API provides separate
API keys per service (recognition, detection, verification).

## Network Topology

| Service | Host | Protocol | Port | Accessible from |
|---|---|---|---|---|
| vision-mcp | Fedora | HTTP | 8097 | aichat-mcp, LM Studio, Claude |
| vision-router | RTX 3090 | HTTP | 8090 | vision-mcp |
| Triton HTTP | RTX 3090 | HTTP | 8000 | vision-router |
| Triton gRPC | RTX 3090 | gRPC | 8001 | vision-router |
| CompreFace | RTX 3090 | HTTP | 8080 | vision-mcp |
| CompreFace DB | RTX 3090 | PostgreSQL | 5433 | CompreFace only |

## Data Flow: Face Recognition

```
Client → vision-mcp (POST /mcp, tools/call: recognize_face)
  → load_image_bytes() → (bytes, mime)
  → CompreFaceClient.recognize_faces(bytes, mime)
    → POST http://RTX3090:8080/api/v1/recognition/recognize
    → Response: [{box, subjects: [{subject, similarity}]}]
  → Map to RecognizeFaceResult
← Return JSON-RPC result
```

## Data Flow: Object Detection

```
Client → vision-mcp (POST /mcp, tools/call: detect_objects)
  → load_image_bytes() → (bytes, mime)
  → base64-encode bytes for transport
  → VisionRouterClient.detect_objects(b64, min_confidence, max_results)
    → POST http://RTX3090:8090/v1/detect-objects
      → decode_b64_image(b64) → PIL.Image
      → resize_for_yolo(img, 640) → (CHW float32, scale, pad)
      → Triton InferInput("images", shape=[1,3,640,640])
      → tritonclient.grpc.InferenceServerClient.infer("yolov8n")
      → Postprocess: NMS → filter by confidence → scale boxes back
    → Response: {objects: [{label, confidence, box}]}
  → Map to DetectObjectsResult
← Return JSON-RPC result
```

## Performance Characteristics

All latencies measured on RTX 3090 with warm model cache:

| Operation | P50 | P95 | Notes |
|---|---|---|---|
| detect_faces | ~80ms | ~120ms | Depends on face count |
| recognize_face | ~90ms | ~140ms | Depends on enrolled subjects |
| detect_objects (640px) | ~18ms | ~28ms | YOLOv8n TensorRT FP16 |
| classify_image (224px) | ~22ms | ~35ms | EfficientNet TensorRT FP16 |
| embed_image (224px) | ~30ms | ~48ms | CLIP ViT-B/32 TensorRT FP16 |
| analyze_image (objects+classify) | ~35ms | ~55ms | Parallel Triton calls |
| vision-mcp overhead | ~5ms | ~10ms | Image decode + HTTP round-trip |

## Scaling Considerations

- **vision-mcp** is stateless and can run multiple replicas behind a load balancer
- **vision-router** is stateless (all state in Triton); scale by adding more instances
- **Triton** supports dynamic batching — set `max_batch_size` in model configs
- **CompreFace** is stateful (PostgreSQL); scale reads with read replicas
- GPU memory budget on RTX 3090 (24 GB): ~2 GB per model × 4 models = ~8 GB
  leaving ~16 GB for batching headroom
