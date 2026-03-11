# Vision Stack Operations Guide

## Initial Setup

### Prerequisites

- RTX 3090 host with NVIDIA drivers installed
- Docker and Docker Compose v2 installed on both hosts
- NVIDIA Container Toolkit installed (see setup script below)

### Step 1 — Configure NVIDIA Container Toolkit (RTX 3090 host only)

```bash
sudo bash vision/scripts/setup_nvidia.sh
```

This script:
1. Validates `nvidia-smi` is available
2. Installs NVIDIA Container Toolkit if missing
3. Configures Docker runtime for GPU access
4. Restarts Docker
5. Validates GPU access with a test container

### Step 2 — Download and export models (RTX 3090 host)

```bash
make download-models
```

This runs `vision/triton/scripts/download_models.sh` which:
- Downloads YOLOv8n weights from Ultralytics
- Downloads EfficientNet-B0 from timm
- Downloads CLIP ViT-B/32 from Hugging Face
- Downloads FashionCLIP from Hugging Face
- Exports all models to ONNX format
- Places them in `/opt/triton-models/` (configurable via `TRITON_MODEL_REPO`)

Total download size: approximately 2 GB.

### Step 3 — Configure environment

```bash
cp vision/mcp-server/.env.example vision/mcp-server/.env
```

Edit the `.env` file:
```dotenv
# Point to your RTX 3090 host
VISION_COMPREFACE__URL=http://192.168.50.2:8080
VISION_COMPREFACE__API_KEY=your-compreface-api-key

VISION_ROUTER__URL=http://192.168.50.2:8090

# MCP server settings
VISION_SERVER__PORT=8097
VISION_SERVER__WORKERS=2
```

---

## Starting the Inference Stack (RTX 3090 host)

```bash
# Build images (first time or after changes)
make build-inference

# Start Triton + CompreFace + vision-router
make up-inference

# Check status
docker compose -f vision/compose/inference.yml ps

# Tail logs
make logs-inference
```

### Expected healthy state

```
NAME                  STATUS          PORTS
triton                running (1m)    0.0.0.0:8000-8002->8000-8002/tcp
compreface-core       running (30s)   0.0.0.0:8080->8080/tcp
compreface-db         running (35s)
vision-router         running (25s)   0.0.0.0:8090->8090/tcp
```

### Verify Triton is ready

```bash
curl http://192.168.50.2:8000/v2/health/ready
# Expected: {"live": true, "ready": true}

# Check models are loaded
curl http://192.168.50.2:8000/v2/models
```

### Verify CompreFace is ready

```bash
curl http://192.168.50.2:8080/actuator/health
# Expected: {"status":"UP"}
```

---

## Starting the MCP Server (Fedora host)

```bash
# Build MCP server image (first time or after changes)
make build-mcp

# Start MCP server
make up-mcp

# Verify
make smoke

# Tail logs
make logs-mcp
```

### Expected healthy state

```
NAME          STATUS          PORTS
vision-mcp    running (10s)   0.0.0.0:8097->8097/tcp
```

---

## Health Check Endpoints

| Endpoint | URL | Expected response |
|---|---|---|
| MCP server health | `GET :8097/health` | `{"status": "ok", "backends": {...}}` |
| Vision router health | `GET :8090/v1/health` | `{"status": "ok", "triton_ready": true}` |
| Triton liveness | `GET :8000/v2/health/live` | `{"live": true}` |
| Triton readiness | `GET :8000/v2/health/ready` | `{"ready": true}` |
| CompreFace health | `GET :8080/actuator/health` | `{"status": "UP"}` |

Automated health polling (useful for monitoring integration):

```bash
watch -n 5 'curl -s http://localhost:8097/health | python3 -m json.tool'
```

---

## Log Monitoring

### View recent logs

```bash
# MCP server (Fedora)
docker compose -f vision/compose/mcp.yml logs --tail=100 vision-mcp

# Inference stack (RTX 3090)
docker compose -f vision/compose/inference.yml logs --tail=100 vision-router
docker compose -f vision/compose/inference.yml logs --tail=100 triton
```

### Follow logs in real time

```bash
make logs-mcp
make logs-inference
```

### Log levels

Set the log level via environment variable:
```dotenv
VISION_SERVER__LOG_LEVEL=DEBUG   # very verbose, includes request/response bodies
VISION_SERVER__LOG_LEVEL=INFO    # default, one line per request
VISION_SERVER__LOG_LEVEL=WARNING # only warnings and errors
```

### What to look for

| Log pattern | Meaning | Action |
|---|---|---|
| `backend_unavailable` | vision-router or CompreFace unreachable | Check network, restart services |
| `SSRF blocked` | URL SSRF attempt | Review client, possibly block IP |
| `Image too large` | Client sending oversized images | Review `MAX_UPLOAD_MB` settings |
| `PermissionError: File path outside` | Path traversal attempt | Security alert, review client |
| `Triton inference failed` | GPU error or model issue | Check GPU memory, restart Triton |
| `CUDA out of memory` | GPU VRAM exhausted | Reduce concurrent requests, restart |

---

## GPU Memory Management

Monitor GPU memory:
```bash
# Live VRAM usage on RTX 3090 host
nvidia-smi -l 2

# Memory per model (Triton stats)
curl http://192.168.50.2:8000/v2/models/yolov8n/stats
```

### Memory budget (RTX 3090, 24 GB VRAM)

| Service | VRAM |
|---|---|
| Triton: yolov8n | ~800 MB |
| Triton: efficientnet_b0 | ~600 MB |
| Triton: clip_vit_b32 | ~1.2 GB |
| Triton: fashion_clip | ~1.2 GB |
| CompreFace (CUDA mode) | ~2.0 GB |
| System + CUDA overhead | ~1.5 GB |
| **Total in use** | **~7.3 GB** |
| **Available for batching** | **~16.7 GB** |

If you see OOM errors, try:
1. Unload unused models from Triton: `POST /v2/repository/models/fashion_clip/unload`
2. Reduce `preferred_batch_size` in model `config.pbtxt`
3. Restart the Triton container: `docker compose -f vision/compose/inference.yml restart triton`

---

## Updating Models

To update to a new model version:

```bash
# Download new model
make download-models

# Restart Triton to pick up new model files
docker compose -f vision/compose/inference.yml restart triton

# Verify new model is loaded
curl http://192.168.50.2:8000/v2/models
```

For zero-downtime updates with Triton's model control API:
```bash
# Load new version (assumes model repo has version 2)
curl -X POST http://192.168.50.2:8000/v2/repository/models/yolov8n/load

# Unload old version
curl -X POST http://192.168.50.2:8000/v2/repository/models/yolov8n/unload \
  -d '{"parameters": {"version": "1"}}'
```

---

## Scaling

### Scale vision-mcp horizontally

vision-mcp is stateless. Run multiple replicas behind a load balancer:

```yaml
# In vision/compose/mcp.yml
services:
  vision-mcp:
    deploy:
      replicas: 4
```

Or use Docker Swarm / Kubernetes for orchestration.

### Scale vision-router

vision-router is also stateless. Each instance connects independently to
Triton's gRPC endpoint which handles concurrent requests natively.

```yaml
services:
  vision-router:
    deploy:
      replicas: 2
```

### Triton dynamic batching

Triton automatically batches concurrent requests. Tune per model in `config.pbtxt`:

```protobuf
dynamic_batching {
  preferred_batch_size: [1, 4, 8]
  max_queue_delay_microseconds: 500
}
```

Higher `max_queue_delay_microseconds` → more batching → better GPU utilization
at the cost of slightly higher latency per request.

---

## CompreFace PostgreSQL Backup

CompreFace stores all face embeddings and metadata in PostgreSQL. Back it up regularly.

### Manual backup

```bash
# On RTX 3090 host
docker compose -f vision/compose/inference.yml exec compreface-db \
  pg_dump -U compreface compreface | gzip > "compreface-backup-$(date +%Y%m%d).sql.gz"
```

### Restore from backup

```bash
# Stop CompreFace core (not the DB)
docker compose -f vision/compose/inference.yml stop compreface-core

# Restore
gunzip < compreface-backup-20260101.sql.gz | \
  docker compose -f vision/compose/inference.yml exec -T compreface-db \
    psql -U compreface compreface

# Restart
docker compose -f vision/compose/inference.yml start compreface-core
```

### Automated daily backup

Add to crontab on RTX 3090 host:
```cron
0 2 * * * cd /path/to/repo && \
  docker compose -f vision/compose/inference.yml exec -T compreface-db \
    pg_dump -U compreface compreface | \
  gzip > /backup/compreface/compreface-$(date +\%Y\%m\%d).sql.gz && \
  find /backup/compreface -name "*.sql.gz" -mtime +30 -delete
```

---

## Stopping the Stack

```bash
# Stop MCP server (Fedora)
make down-mcp

# Stop inference stack (RTX 3090)
make down-inference

# Stop everything and remove volumes (DESTRUCTIVE — loses enrolled faces!)
docker compose -f vision/compose/inference.yml down -v
```

---

## Troubleshooting

### `backend_unavailable` in MCP responses

Check connectivity from Fedora to RTX 3090:
```bash
curl -v http://192.168.50.2:8090/v1/health
curl -v http://192.168.50.2:8080/actuator/health
```

If unreachable, check:
1. Firewall rules allow ports 8080, 8090 from Fedora host
2. Inference stack containers are running: `docker compose -f vision/compose/inference.yml ps`
3. Vision router is configured with correct Triton URL

### Triton model not found

```bash
# List loaded models
curl http://192.168.50.2:8000/v2/models

# Check model repository
ls -la /opt/triton-models/

# Re-run download if models are missing
make download-models
```

### CompreFace enrollment fails with "No face detected"

- Image must be at least 100×100 pixels
- Face must be clearly visible (frontal, good lighting)
- Lower `VISION_COMPREFACE__DET_PROB_THRESHOLD` (try 0.5)
- Check CompreFace logs: `docker compose -f vision/compose/inference.yml logs compreface-core`

### High latency / timeouts

1. Check GPU utilization: `nvidia-smi` on RTX 3090 host
2. Check Triton queue depth: `curl http://192.168.50.2:8000/v2/models/yolov8n/stats`
3. Increase timeouts: `VISION_ROUTER__TIMEOUT_S=120`
4. Consider enabling TensorRT optimization for faster inference
