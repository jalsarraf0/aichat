"""aichat-inference — local embedding inference service.

Runs nomic-embed-text-v1.5 (768-dim) via sentence-transformers + ONNX Runtime,
offloading embedding work from LM Studio so it doesn't waste a model slot.

Endpoints:
  POST /v1/embeddings  — OpenAI-compatible embedding API
  GET  /health         — service status
  GET  /v1/models      — list available models
"""
from __future__ import annotations

import logging
import time
from typing import Any

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

log = logging.getLogger("inference")
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

app = FastAPI(title="aichat-inference", version="1.0.0")

MODEL_NAME = "nomic-embed-text-v1.5"
MODEL_DIR = "/app/models/nomic-embed"
EMBED_DIM = 768

_model = None
_load_time_ms = 0.0


def _ensure_loaded() -> None:
    global _model, _load_time_ms
    if _model is not None:
        return
    t0 = time.monotonic()
    from sentence_transformers import SentenceTransformer
    _model = SentenceTransformer(MODEL_DIR, trust_remote_code=True)
    _load_time_ms = (time.monotonic() - t0) * 1000
    log.info("Loaded %s in %.0f ms (torch CPU)", MODEL_NAME, _load_time_ms)


class EmbeddingRequest(BaseModel):
    input: str | list[str]
    model: str = MODEL_NAME
    encoding_format: str = "float"


class EmbeddingObj(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingObj]
    model: str
    usage: dict[str, int]


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok" if _model else "not_loaded",
        "service": "aichat-inference",
        "model": MODEL_NAME,
        "embed_dim": EMBED_DIM,
        "backend": "torch-cpu",
        "loaded": _model is not None,
        "load_time_ms": round(_load_time_ms, 1),
    }


@app.get("/v1/models")
def list_models() -> dict[str, Any]:
    return {
        "object": "list",
        "data": [{
            "id": MODEL_NAME,
            "object": "model",
            "owned_by": "aichat-inference",
            "type": "embeddings",
        }],
    }


@app.post("/v1/embeddings")
def create_embeddings(req: EmbeddingRequest) -> EmbeddingResponse:
    _ensure_loaded()

    texts = req.input if isinstance(req.input, list) else [req.input]
    if not texts:
        raise HTTPException(400, "input must be non-empty")
    if len(texts) > 128:
        raise HTTPException(400, "max 128 texts per batch")

    # nomic-embed requires task prefixes for best quality
    prefixed = []
    for t in texts:
        if t.startswith("search_document:") or t.startswith("search_query:"):
            prefixed.append(t)
        else:
            prefixed.append(f"search_document: {t}")

    embeddings = _model.encode(prefixed, normalize_embeddings=True)

    token_est = sum(len(t.split()) for t in texts)
    return EmbeddingResponse(
        data=[
            EmbeddingObj(embedding=emb.tolist(), index=i)
            for i, emb in enumerate(embeddings)
        ],
        model=MODEL_NAME,
        usage={"prompt_tokens": token_est, "total_tokens": token_est},
    )
