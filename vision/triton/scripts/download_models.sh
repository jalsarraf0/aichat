#!/usr/bin/env bash
# download_models.sh — Download and export vision models to the Triton model repository.
#
# Models acquired:
#   1. YOLOv8n      — ONNX from Ultralytics GitHub releases
#   2. EfficientNet-B0 — exported from timm via torch.onnx.export
#   3. CLIP ViT-B/32   — exported from openai/clip via torch.onnx.export
#   4. FashionCLIP     — exported from patrickjohncyh/fashion-clip via torch.onnx.export
#
# Usage:
#   bash download_models.sh [MODEL_REPO_DIR]
#
# The optional first argument overrides the default model repository path
# (../model_repository relative to this script).

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODEL_REPO="${1:-${SCRIPT_DIR}/../model_repository}"

log()  { echo "[INFO]  $*"; }
warn() { echo "[WARN]  $*" >&2; }
die()  { echo "[ERROR] $*" >&2; exit 1; }

require_cmd() {
    command -v "$1" &>/dev/null || die "Required command '$1' not found. Install it and retry."
}

require_cmd curl
require_cmd python3

# ---------------------------------------------------------------------------
# Helper: download a file if it doesn't already exist
# ---------------------------------------------------------------------------
download_if_missing() {
    local url="$1"
    local dest="$2"
    if [[ -f "$dest" ]]; then
        log "Already exists, skipping download: $dest"
        return 0
    fi
    log "Downloading: $url → $dest"
    curl --fail --location --progress-bar --output "$dest" "$url" \
        || die "Download failed: $url"
}

# ---------------------------------------------------------------------------
# 1. YOLOv8n ONNX
# ---------------------------------------------------------------------------
YOLO_DIR="${MODEL_REPO}/yolov8n/1"
YOLO_MODEL="${YOLO_DIR}/model.onnx"
mkdir -p "${YOLO_DIR}"

if [[ -f "${YOLO_MODEL}" ]]; then
    log "YOLOv8n already present: ${YOLO_MODEL}"
else
    log "=== Downloading and exporting YOLOv8n ONNX ==="
    YOLO_PT="${YOLO_DIR}/yolov8n.pt"
    download_if_missing "https://github.com/ultralytics/assets/releases/download/v8.4.0/yolov8n.pt" "${YOLO_PT}"
    log "Exporting YOLOv8n .pt -> ONNX..."
    export YOLO_PT="${YOLO_PT}" YOLO_MODEL="${YOLO_MODEL}"
    python3 - <<'PYEOF'
import sys, subprocess, shutil, os
try:
    from ultralytics import YOLO
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--break-system-packages", "ultralytics", "onnx", "onnxsim"])
    from ultralytics import YOLO
import os
pt = os.environ.get("YOLO_PT")
out = os.environ.get("YOLO_MODEL")
m = YOLO(pt)
m.export(format="onnx", opset=17, simplify=True, dynamic=True, imgsz=640)
exported = pt.replace(".pt", ".onnx")
shutil.move(exported, out)
os.remove(pt)
PYEOF
    log "YOLOv8n exported successfully."
fi

# ---------------------------------------------------------------------------
# 2. EfficientNet-B0 ONNX  (exported via timm)
# ---------------------------------------------------------------------------
EFFNET_DIR="${MODEL_REPO}/efficientnet_b0/1"
EFFNET_MODEL="${EFFNET_DIR}/model.onnx"
mkdir -p "${EFFNET_DIR}"

if [[ -f "${EFFNET_MODEL}" ]]; then
    log "EfficientNet-B0 already present: ${EFFNET_MODEL}"
else
    log "=== Exporting EfficientNet-B0 to ONNX via timm ==="
    python3 - <<PYEOF
import sys, subprocess

# Install timm if not present
try:
    import timm
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--break-system-packages", "timm", "torch", "torchvision"])
    import timm

import torch
import timm

model = timm.create_model("efficientnet_b0", pretrained=True)
model.eval()

dummy = torch.zeros(1, 3, 224, 224)
out_path = "${EFFNET_MODEL}"

torch.onnx.export(
    model,
    dummy,
    out_path,
    opset_version=17,
    input_names=["input_1"],
    output_names=["Predictions/Softmax"],
    dynamic_axes={
        "input_1": {0: "batch"},
        "Predictions/Softmax": {0: "batch"},
    },
    do_constant_folding=True,
)
print(f"Saved to {out_path}")
PYEOF
    log "EfficientNet-B0 exported successfully."
fi

# ---------------------------------------------------------------------------
# 3. CLIP ViT-B/32 ONNX  (image encoder only)
# ---------------------------------------------------------------------------
CLIP_DIR="${MODEL_REPO}/clip_vit_b32/1"
CLIP_MODEL="${CLIP_DIR}/model.onnx"
mkdir -p "${CLIP_DIR}"

if [[ -f "${CLIP_MODEL}" ]]; then
    log "CLIP ViT-B/32 already present: ${CLIP_MODEL}"
else
    log "=== Exporting CLIP ViT-B/32 image encoder to ONNX ==="
    python3 - <<PYEOF
import sys, subprocess, io

try:
    from transformers import CLIPVisionModel
    import torch
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--break-system-packages",
                           "transformers", "torch"])
    from transformers import CLIPVisionModel
    import torch

import torch

class CLIPVisionWrapper(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.model = CLIPVisionModel.from_pretrained("openai/clip-vit-base-patch32").float()
    def forward(self, x):
        return self.model(pixel_values=x).pooler_output

wrapper = CLIPVisionWrapper()
wrapper.eval()

dummy = torch.zeros(1, 3, 224, 224)
out_path = "${CLIP_MODEL}"

# Export to BytesIO to force inline weights (no external data split)
buf = io.BytesIO()
with torch.no_grad():
    torch.onnx.export(
        wrapper,
        dummy,
        buf,
        opset_version=17,
        input_names=["input"],
        output_names=["output"],
        dynamic_axes={"input": {0: "batch"}, "output": {0: "batch"}},
        do_constant_folding=True,
    )

with open(out_path, "wb") as f:
    f.write(buf.getvalue())
print(f"Saved {len(buf.getvalue()) // 1024 // 1024}MB to {out_path}")
PYEOF
    log "CLIP ViT-B/32 exported successfully."
fi

# ---------------------------------------------------------------------------
# 4. FashionCLIP ONNX  (image encoder, HuggingFace patrickjohncyh/fashion-clip)
# ---------------------------------------------------------------------------
FCLIP_DIR="${MODEL_REPO}/fashion_clip/1"
FCLIP_MODEL="${FCLIP_DIR}/model.onnx"
mkdir -p "${FCLIP_DIR}"

if [[ -f "${FCLIP_MODEL}" ]]; then
    log "FashionCLIP already present: ${FCLIP_MODEL}"
else
    log "=== Exporting FashionCLIP image encoder to ONNX ==="
    python3 - <<PYEOF
import sys, subprocess

try:
    from transformers import CLIPModel, CLIPProcessor
except ImportError:
    subprocess.check_call([sys.executable, "-m", "pip", "install", "--quiet", "--break-system-packages",
                           "transformers", "torch", "Pillow"])
    from transformers import CLIPModel, CLIPProcessor

import torch

MODEL_ID = "patrickjohncyh/fashion-clip"
print(f"Loading {MODEL_ID} from HuggingFace …")

model = CLIPModel.from_pretrained(MODEL_ID)
model.eval()
vision_model = model.vision_model
visual_proj = model.visual_projection

class FashionCLIPVisionWrapper(torch.nn.Module):
    """Vision encoder + projection head, returns float32 embeddings."""
    def __init__(self, vision_model, visual_proj):
        super().__init__()
        self.vision_model = vision_model
        self.visual_proj = visual_proj

    def forward(self, pixel_values: torch.Tensor) -> torch.Tensor:
        outputs = self.vision_model(pixel_values=pixel_values)
        pooled = outputs.pooler_output  # (B, hidden_size)
        return self.visual_proj(pooled).float()  # (B, 512)

wrapper = FashionCLIPVisionWrapper(vision_model, visual_proj)
wrapper.eval()

dummy = torch.zeros(1, 3, 224, 224)
out_path = "${FCLIP_MODEL}"

torch.onnx.export(
    wrapper,
    dummy,
    out_path,
    opset_version=17,
    input_names=["input"],
    output_names=["output"],
    dynamic_axes={
        "input": {0: "batch"},
        "output": {0: "batch"},
    },
    do_constant_folding=True,
)
print(f"Saved to {out_path}")
PYEOF
    log "FashionCLIP exported successfully."
fi

# ---------------------------------------------------------------------------
# 5. WD ViT Large Tagger v3 (anime/illustration character & content tagging)
#    Downloads model.onnx and selected_tags.csv from SmilingWolf on HuggingFace.
#    selected_tags.csv is a sidecar used at inference time; stored at the model
#    root (not inside the version directory) so the router can find it at
#    /models/wd_tagger/selected_tags.csv.
# ---------------------------------------------------------------------------
WD_DIR="${MODEL_REPO}/wd_tagger/1"
WD_MODEL="${WD_DIR}/model.onnx"
WD_TAGS="${MODEL_REPO}/wd_tagger/selected_tags.csv"
WD_HF_BASE="https://huggingface.co/SmilingWolf/wd-vit-large-tagger-v3/resolve/main"
mkdir -p "${WD_DIR}"

if [[ -f "${WD_MODEL}" ]] && [[ -f "${WD_TAGS}" ]]; then
    log "WD Tagger already present: ${WD_MODEL}"
else
    log "=== Downloading WD ViT Large Tagger v3 ==="
    download_if_missing "${WD_HF_BASE}/model.onnx" "${WD_MODEL}"
    download_if_missing "${WD_HF_BASE}/selected_tags.csv" "${WD_TAGS}"
    log "WD Tagger downloaded successfully."
fi

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------
log ""
log "=== Model repository status ==="
for model_name in yolov8n efficientnet_b0 clip_vit_b32 fashion_clip wd_tagger; do
    model_path="${MODEL_REPO}/${model_name}/1/model.onnx"
    if [[ -f "${model_path}" ]]; then
        size=$(du -sh "${model_path}" | cut -f1)
        log "  [OK]  ${model_name}: ${model_path} (${size})"
    else
        warn "  [MISSING] ${model_name}: ${model_path}"
    fi
done
log ""
log "All models are ready for Triton. Start Triton with:"
log "  tritonserver --model-repository=$(realpath "${MODEL_REPO}")"
