# Vision Stack API Reference

## MCP Protocol

All MCP tool calls use JSON-RPC 2.0 over HTTP POST to `/mcp`.

### Request format

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "method": "tools/call",
  "params": {
    "name": "<tool_name>",
    "arguments": { ... }
  }
}
```

### Success response format

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "result": {
    "content": [
      {
        "type": "text",
        "text": "<JSON-encoded tool result>"
      }
    ]
  }
}
```

### Error response format

```json
{
  "jsonrpc": "2.0",
  "id": 1,
  "error": {
    "code": -32000,
    "message": "Tool execution failed: <reason>",
    "data": { "tool": "<name>", "backend": "<name>" }
  }
}
```

### Common image input type

All tools that accept an image use the `ImageSource` type:

```json
{
  "url": "https://example.com/photo.jpg"
}
```
or
```json
{
  "base64": "<base64-encoded-image-bytes>"
}
```
or
```json
{
  "file_path": "/workspace/photo.jpg"
}
```

Exactly one field must be non-null.

---

## Face Tools (CompreFace backend)

### recognize_face

Recognize faces in an image against enrolled subjects.

**Input schema:**
```json
{
  "type": "object",
  "required": ["image"],
  "properties": {
    "image":            { "$ref": "#/ImageSource" },
    "subject_filter":  { "type": "array", "items": {"type": "string"}, "description": "Limit search to these subjects" },
    "min_confidence":  { "type": "number", "default": 0.7, "minimum": 0, "maximum": 1 },
    "limit":           { "type": "integer", "default": 5, "minimum": 1, "maximum": 50 }
  }
}
```

**Output:**
```json
{
  "faces": [
    {
      "box": {"x_min": 10, "y_min": 20, "x_max": 200, "y_max": 380, "confidence": 0.99},
      "matches": [
        {"subject": "alice", "similarity": 0.94},
        {"subject": "bob", "similarity": 0.71}
      ],
      "age": {"probability": 0.88, "high": 35, "low": 28},
      "gender": {"probability": 0.97, "value": "female"},
      "emotion": [{"probability": 0.85, "value": "neutral"}],
      "mask": {"probability": 0.02, "value": "without_mask"}
    }
  ],
  "count": 1,
  "timing": {"total_ms": 87.4, "backend_ms": 71.2},
  "backend": {"name": "compreface", "host": "192.168.50.2"}
}
```

---

### verify_face

Determine if two images contain the same person.

**Input schema:**
```json
{
  "type": "object",
  "required": ["image_a", "image_b"],
  "properties": {
    "image_a":        { "$ref": "#/ImageSource" },
    "image_b":        { "$ref": "#/ImageSource" },
    "min_similarity": { "type": "number", "default": 0.85 }
  }
}
```

**Output:**
```json
{
  "verified": true,
  "similarity": 0.93,
  "subject_a_face_count": 1,
  "subject_b_face_count": 1,
  "timing": {"total_ms": 112.3},
  "backend": {"name": "compreface"}
}
```

---

### detect_faces

Detect faces without recognition (no subject matching).

**Input schema:**
```json
{
  "type": "object",
  "required": ["image"],
  "properties": {
    "image":             { "$ref": "#/ImageSource" },
    "min_confidence":    { "type": "number", "default": 0.7 },
    "return_landmarks":  { "type": "boolean", "default": false }
  }
}
```

**Output:**
```json
{
  "faces": [
    {
      "box": {"x_min": 50, "y_min": 80, "x_max": 220, "y_max": 350, "confidence": 0.98},
      "landmarks": {
        "left_eye": [85.2, 155.0],
        "right_eye": [170.1, 152.3],
        "nose": [128.0, 200.5]
      }
    }
  ],
  "count": 1,
  "timing": {"total_ms": 68.1},
  "backend": {"name": "compreface"}
}
```

---

### enroll_face

Add a face image to the recognition database.

**Input schema:**
```json
{
  "type": "object",
  "required": ["image", "subject_name"],
  "properties": {
    "image":        { "$ref": "#/ImageSource" },
    "subject_name": { "type": "string", "minLength": 1, "maxLength": 200 }
  }
}
```

**Output:**
```json
{
  "subject": "alice",
  "image_id": "3fa85f64-5717-4562-b3fc-2c963f66afa6",
  "timing": {"total_ms": 215.0},
  "backend": {"name": "compreface"}
}
```

**Error:** Returns `FACE_NOT_FOUND` if no face is detected in the image.

---

### list_face_subjects

List all enrolled subject names.

**Input schema:**
```json
{ "type": "object", "properties": {} }
```

**Output:**
```json
{
  "subjects": ["alice", "bob", "charlie"],
  "count": 3,
  "backend": {"name": "compreface"}
}
```

---

### delete_face_subject

Delete all enrolled images for a subject.

**Input schema:**
```json
{
  "type": "object",
  "required": ["subject_name"],
  "properties": {
    "subject_name": { "type": "string", "minLength": 1 }
  }
}
```

**Output:**
```json
{
  "subject": "alice",
  "deleted": true,
  "backend": {"name": "compreface"}
}
```

---

## Vision Tools (Triton backend via vision-router)

### detect_objects

Detect objects using YOLOv8n (80 COCO classes).

**Input schema:**
```json
{
  "type": "object",
  "required": ["image"],
  "properties": {
    "image":          { "$ref": "#/ImageSource" },
    "min_confidence": { "type": "number", "default": 0.4, "minimum": 0, "maximum": 1 },
    "max_results":    { "type": "integer", "default": 20, "minimum": 1, "maximum": 100 },
    "classes":        { "type": "array", "items": {"type": "string"}, "description": "Filter to specific COCO class names" }
  }
}
```

**Output:**
```json
{
  "objects": [
    {
      "label": "person",
      "confidence": 0.92,
      "box": {"x_min": 45, "y_min": 110, "x_max": 198, "y_max": 445, "confidence": 0.92}
    },
    {
      "label": "car",
      "confidence": 0.88,
      "box": {"x_min": 300, "y_min": 200, "x_max": 580, "y_max": 400, "confidence": 0.88}
    }
  ],
  "count": 2,
  "timing": {"total_ms": 18.3, "backend_ms": 12.1, "preprocess_ms": 3.2, "postprocess_ms": 3.0},
  "backend": {"name": "triton", "model": "yolov8n"}
}
```

---

### classify_image

Classify image content using EfficientNet-B0 (1000 ImageNet classes).

**Input schema:**
```json
{
  "type": "object",
  "required": ["image"],
  "properties": {
    "image":          { "$ref": "#/ImageSource" },
    "top_k":          { "type": "integer", "default": 5, "minimum": 1, "maximum": 50 },
    "min_confidence": { "type": "number", "default": 0.01 }
  }
}
```

**Output:**
```json
{
  "labels": [
    {"name": "golden retriever", "confidence": 0.87},
    {"name": "Labrador retriever", "confidence": 0.08},
    {"name": "kuvasz", "confidence": 0.02}
  ],
  "top_label": "golden retriever",
  "top_confidence": 0.87,
  "timing": {"total_ms": 22.1},
  "backend": {"name": "triton", "model": "efficientnet_b0"}
}
```

---

### detect_clothing

Detect clothing items using FashionCLIP zero-shot classification.

**Input schema:**
```json
{
  "type": "object",
  "required": ["image"],
  "properties": {
    "image":          { "$ref": "#/ImageSource" },
    "min_confidence": { "type": "number", "default": 0.15 },
    "top_k":          { "type": "integer", "default": 5, "minimum": 1, "maximum": 20 }
  }
}
```

**Output:**
```json
{
  "items": [
    {"category": "t-shirt", "confidence": 0.82, "attributes": {"color": "white"}, "box": null},
    {"category": "jeans", "confidence": 0.74, "attributes": {}, "box": null}
  ],
  "count": 2,
  "dominant_category": "t-shirt",
  "timing": {"total_ms": 35.0},
  "backend": {"name": "triton", "model": "fashion_clip"}
}
```

---

### embed_image

Generate a semantic embedding vector for an image.

**Input schema:**
```json
{
  "type": "object",
  "required": ["image"],
  "properties": {
    "image":     { "$ref": "#/ImageSource" },
    "model":     { "type": "string", "default": "clip_vit_b32", "enum": ["clip_vit_b32"] },
    "normalize": { "type": "boolean", "default": true }
  }
}
```

**Output:**
```json
{
  "embeddings": [0.023, -0.142, 0.087, ...],
  "dim": 512,
  "model": "clip_vit_b32",
  "normalized": true,
  "timing": {"total_ms": 31.4},
  "backend": {"name": "triton", "model": "clip_vit_b32"}
}
```

---

### analyze_image

Run multiple vision analyses in parallel.

**Input schema:**
```json
{
  "type": "object",
  "required": ["image"],
  "properties": {
    "image":                  { "$ref": "#/ImageSource" },
    "include_objects":        { "type": "boolean", "default": true },
    "include_classification": { "type": "boolean", "default": true },
    "include_clothing":       { "type": "boolean", "default": false },
    "include_embeddings":     { "type": "boolean", "default": false },
    "object_confidence":      { "type": "number", "default": 0.4 },
    "classification_top_k":   { "type": "integer", "default": 3 }
  }
}
```

**Output:**
```json
{
  "objects": [
    {"label": "dog", "confidence": 0.95, "box": {"x_min": 50, "y_min": 80, "x_max": 350, "y_max": 460, "confidence": 0.95}}
  ],
  "classification": [
    {"name": "golden retriever", "confidence": 0.88}
  ],
  "clothing": null,
  "embeddings": null,
  "summary": "1 object detected (dog). Top class: golden retriever (0.88).",
  "timing": {"total_ms": 38.2},
  "backend": {"name": "triton"}
}
```

---

## Vision Router REST API

The vision-router service (on the RTX 3090 host) exposes these REST endpoints.
These are called internally by vision-mcp and are not exposed to MCP clients
directly.

### GET /v1/health

Check service and Triton connectivity.

**Response:**
```json
{
  "status": "ok",
  "triton_live": true,
  "triton_ready": true,
  "models": ["yolov8n", "efficientnet_b0", "clip_vit_b32", "fashion_clip"]
}
```

### POST /v1/detect-objects

**Request:**
```json
{
  "image_b64": "<base64>",
  "min_confidence": 0.4,
  "max_results": 20,
  "classes": null
}
```

**Response:** `DetectObjectsResult` (same schema as MCP tool output, minus timing/backend wrappers).

### POST /v1/classify

**Request:**
```json
{"image_b64": "<base64>", "top_k": 5, "min_confidence": 0.01}
```

### POST /v1/detect-clothing

**Request:**
```json
{"image_b64": "<base64>", "min_confidence": 0.15, "top_k": 5}
```

### POST /v1/embed

**Request:**
```json
{"image_b64": "<base64>", "model": "clip_vit_b32", "normalize": true}
```

### POST /v1/analyze

**Request:**
```json
{
  "image_b64": "<base64>",
  "include_objects": true,
  "include_classification": true,
  "include_clothing": false,
  "include_embeddings": false,
  "object_confidence": 0.4,
  "classification_top_k": 3
}
```

---

## Error Codes

| Code | Name | Description |
|---|---|---|
| -32700 | Parse error | Malformed JSON in request body |
| -32600 | Invalid request | JSON-RPC envelope is invalid |
| -32601 | Method not found | `method` is not `tools/call` or `tools/list` |
| -32602 | Invalid params | Tool arguments failed validation |
| -32000 | Tool error | Tool executed but returned an error |
| -32001 | Backend unavailable | CompreFace or vision-router is unreachable |
| -32002 | Image load failed | Could not decode/fetch the provided image |
| -32003 | SSRF blocked | URL points to a private/loopback address |
| -32004 | File path denied | File path is outside the allowed directories |
| -32005 | Image too large | Image exceeds the configured size limit |
| -32006 | Face not found | No face detected in the enrollment image |
