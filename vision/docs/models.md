# Vision Stack Model Reference

All models run on the RTX 3090 via NVIDIA Triton Inference Server in ONNX
format, optionally converted to TensorRT FP16 for maximum throughput.

---

## YOLOv8n — Object Detection

| Attribute | Value |
|---|---|
| Source | Ultralytics YOLOv8 (`ultralytics` pip package) |
| Architecture | CSPDarknet + PANNet + Decoupled head |
| Parameters | 3.2M |
| Input shape | `(1, 3, 640, 640)` NCHW float32 `[0, 1]` |
| Output shape | `(1, 84, 8400)` — 80 class scores + 4 box coords |
| Preprocessing | Letterbox resize (gray pad 114/255), CHW float32 |
| Postprocessing | NMS (IoU threshold 0.45), filter by confidence |
| COCO classes | 80 (person, car, dog, cat, bicycle, ...) |
| mAP@0.5 (COCO val) | 37.3 |
| mAP@0.5:0.95 (COCO val) | 27.9 |
| FPS on RTX 3090 (TensorRT FP16) | ~250 at batch=1 |
| VRAM usage | ~800 MB |
| Triton model name | `yolov8n` |

### Export command
```bash
from ultralytics import YOLO
model = YOLO("yolov8n.pt")
model.export(format="onnx", dynamic=True, simplify=True, imgsz=640)
```

### Triton config snippet (`config.pbtxt`)
```protobuf
name: "yolov8n"
backend: "onnxruntime"
max_batch_size: 8

input [{ name: "images" data_type: TYPE_FP32 dims: [3, 640, 640] }]
output [{ name: "output0" data_type: TYPE_FP32 dims: [84, 8400] }]

dynamic_batching { preferred_batch_size: [1, 4, 8] }
```

---

## EfficientNet-B0 — Image Classification

| Attribute | Value |
|---|---|
| Source | `timm` library (`efficientnet_b0`, ImageNet pretrained) |
| Architecture | MBConv blocks with compound scaling |
| Parameters | 5.3M |
| Input shape | `(1, 3, 224, 224)` NCHW float32 |
| Output shape | `(1, 1000)` — 1000 class logits |
| Preprocessing | Resize shorter side → 224, center-crop 224×224, ImageNet normalization |
| Postprocessing | Softmax → top-k labels |
| Classes | 1000 ImageNet-1K classes |
| Top-1 accuracy (ImageNet val) | 77.7% |
| Top-5 accuracy (ImageNet val) | 93.6% |
| FPS on RTX 3090 (TensorRT FP16) | ~400 at batch=1 |
| VRAM usage | ~600 MB |
| Triton model name | `efficientnet_b0` |
| ImageNet mean | `[0.485, 0.456, 0.406]` |
| ImageNet std | `[0.229, 0.224, 0.225]` |

### Export command
```python
import timm, torch
model = timm.create_model("efficientnet_b0", pretrained=True, exportable=True)
model.eval()
dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(model, dummy, "efficientnet_b0.onnx",
    input_names=["input"], output_names=["output"],
    opset_version=17, dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}})
```

---

## CLIP ViT-B/32 — Semantic Embeddings

| Attribute | Value |
|---|---|
| Source | OpenAI CLIP (`openai/clip-vit-base-patch32` on Hugging Face) |
| Architecture | Vision Transformer ViT-B/32 image encoder |
| Parameters | 87M (image encoder only) |
| Input shape | `(1, 3, 224, 224)` NCHW float32 |
| Output shape | `(1, 512)` — 512-dim embedding vector |
| Preprocessing | Bicubic resize shorter side → 224, center-crop 224×224, CLIP normalization |
| Postprocessing | Optional L2 normalization |
| Embedding dim | 512 |
| CLIP mean | `[0.48145466, 0.4578275, 0.40821073]` |
| CLIP std | `[0.26862954, 0.26130258, 0.27577711]` |
| FPS on RTX 3090 (TensorRT FP16) | ~300 at batch=1 |
| VRAM usage | ~1.2 GB |
| Triton model name | `clip_vit_b32` |

### Use cases
- Semantic image search (embed images → vector store → nearest-neighbor search)
- Image-text similarity (compare image embeddings with text embeddings)
- Zero-shot classification (compare to text label embeddings)
- Visual deduplication

### Export command
```python
from transformers import CLIPModel
import torch

model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32")
image_encoder = model.vision_model
image_encoder.eval()

dummy = torch.randn(1, 3, 224, 224)
torch.onnx.export(image_encoder, dummy, "clip_vit_b32_image.onnx",
    input_names=["pixel_values"], output_names=["pooler_output"],
    opset_version=17, dynamic_axes={"pixel_values": {0: "batch"}, "pooler_output": {0: "batch"}})
```

---

## FashionCLIP — Clothing Detection

| Attribute | Value |
|---|---|
| Source | `patrickjohncyh/fashion-clip` on Hugging Face |
| Architecture | CLIP ViT-B/32 fine-tuned on fashion product dataset |
| Parameters | 87M (image encoder) |
| Input shape | `(1, 3, 224, 224)` NCHW float32 |
| Output shape | `(1, 512)` — embedding compared against clothing text embeddings |
| Preprocessing | Same as CLIP ViT-B/32 |
| Postprocessing | Cosine similarity with pre-computed text embeddings for clothing labels |
| Clothing categories | 40+ (t-shirt, jeans, dress, jacket, sneakers, boots, handbag, ...) |
| Approach | Zero-shot: embed image → compare against text embeddings of category names |
| FPS on RTX 3090 (TensorRT FP16) | ~280 at batch=1 |
| VRAM usage | ~1.2 GB (shared with CLIP if loaded jointly) |
| Triton model name | `fashion_clip` |

### How zero-shot clothing detection works

1. At startup, the router pre-computes text embeddings for each clothing
   category label (e.g. "a photo of a t-shirt", "a photo of jeans")
2. For each inference request, the image is encoded to a 512-dim vector
3. Cosine similarity is computed between the image embedding and each text
   label embedding
4. Categories above `min_confidence` are returned, sorted by similarity

---

## How to Add a New Model

### 1. Export the model to ONNX

```python
# See per-model export commands above as templates.
# Key requirements:
# - dynamic batch size (opset 17+)
# - known input/output names
# - float32 inputs/outputs (or ensure correct dtype in config)
```

### 2. Create a Triton model repository entry

```
vision/triton/models/
└── my_new_model/
    ├── config.pbtxt     # Triton model config
    └── 1/
        └── model.onnx   # or model.plan for TensorRT
```

Minimal `config.pbtxt`:
```protobuf
name: "my_new_model"
backend: "onnxruntime"
max_batch_size: 8

input [{
  name: "input"
  data_type: TYPE_FP32
  dims: [3, 224, 224]
}]

output [{
  name: "output"
  data_type: TYPE_FP32
  dims: [1000]
}]

dynamic_batching {
  preferred_batch_size: [1, 4, 8]
  max_queue_delay_microseconds: 100
}
```

### 3. Add preprocessing function in vision-router

In `vision/services/vision-router/app/preprocessing.py`, add a new function
following the pattern of `resize_for_yolo` or `resize_for_efficientnet`.

### 4. Add a new vision-router endpoint

In the vision-router app, add a new POST endpoint (e.g. `/v1/my-tool`) that:
- Accepts `image_b64` + any parameters
- Calls your preprocessing function
- Submits to Triton via `tritonclient.grpc`
- Postprocesses and returns structured JSON

### 5. Add a new MCP tool in vision-mcp

In `vision/mcp-server/app/tools/vision.py`, add:
- A new `@mcp.tool()` decorated function
- Input/output Pydantic models in `app/models.py`
- A VisionRouterClient method for the new endpoint

### 6. Add unit tests

Add test cases to:
- `vision/tests/unit/test_preprocessing.py` — for preprocessing
- `vision/tests/unit/test_models.py` — for new Pydantic models
- `vision/tests/integration/test_vision_router_client.py` — for the endpoint
