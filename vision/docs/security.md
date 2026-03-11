# Vision Stack Security Reference

## Threat Model

The vision MCP server processes untrusted image input from LLM agents and
external clients. The primary threat surfaces are:

1. **SSRF** — attacker provides a URL that causes the server to make requests
   to internal services or cloud metadata endpoints
2. **Path traversal** — attacker provides a file path that escapes the allowed
   directories and reads sensitive files
3. **Denial of service** — attacker provides an extremely large image or a
   zip-bomb that exhausts memory
4. **API key exfiltration** — secrets leaking through logs or error messages
5. **Dependency vulnerabilities** — CVEs in transitive Python/system deps

---

## SSRF Protection

All URL image loading goes through `_validate_url()` in
`vision/mcp-server/app/utils/image.py` before any HTTP request is made.

### Blocked patterns

| Pattern | Example | Reason |
|---|---|---|
| Loopback IPs | `http://127.0.0.1/` | Accesses host's own services |
| Private IPv4 RFC 1918 | `http://10.0.0.1/`, `http://192.168.x.x/` | Internal network |
| Private IPv4 RFC 6598 | `http://100.64.x.x/` | Carrier-grade NAT |
| Link-local | `http://169.254.x.x/` | AWS/GCP metadata addresses |
| Blocked hostnames | `localhost`, `metadata.google.internal` | Known metadata hosts |
| Non-HTTP schemes | `file://`, `ftp://`, `gopher://` | Arbitrary file/protocol access |

### What is NOT blocked (by design)

- Legitimate public HTTPS URLs — the intended use case
- IPv6 public addresses — validated the same way as IPv4

### Defense-in-depth note

DNS rebinding attacks (public hostname that resolves to private IP after the
check) are not fully mitigated by the current implementation. For production
deployments with sensitive internal networks, consider running vision-mcp
behind an egress proxy that enforces its own SSRF controls, or disable URL
input entirely with `VISION_ENABLE_URL_INPUT=false`.

---

## File Path Sandboxing

Local file loading via `file_path` is restricted to three directories:

```
/workspace   — shared volume for processed assets
/tmp         — temporary upload directory
/data        — data volume (read-only assets)
```

Implementation in `_load_from_file()`:
1. `Path(path).resolve()` — resolves symlinks and `..` traversal
2. Checks that the resolved path starts with one of the allowed prefixes
3. Raises `PermissionError` with a descriptive message if the check fails

Example blocked paths:
- `/etc/passwd` → `PermissionError`
- `/workspace/../etc/shadow` → resolves to `/etc/shadow` → `PermissionError`
- `/tmp/../workspace/file.jpg` → resolves to `/workspace/file.jpg` → allowed

---

## Image Size Limits

| Limit | Value | Controlled by |
|---|---|---|
| Max raw image bytes | 25 MB | `MAX_BYTES` constant in `image.py` |
| Max upload via HTTP | 20 MB | `VISION_SERVER__MAX_UPLOAD_MB` env var |
| Max CompreFace upload | 10 MB | `VISION_COMPREFACE__MAX_FILE_SIZE_MB` |
| Max router upload | 20 MB | `VISION_ROUTER__MAX_FILE_SIZE_MB` |

Images exceeding the limit are rejected with a `ValueError` before any
decoding occurs. This prevents decompression bombs (e.g. a tiny PNG that
expands to gigabytes).

---

## API Key Authentication

### CompreFace per-service keys

CompreFace uses separate API keys for each service (recognition, detection,
verification). This limits the blast radius if a key is compromised:

```
VISION_COMPREFACE__RECOGNITION_API_KEY  — passed in X-Api-Key for recognition calls
VISION_COMPREFACE__DETECTION_API_KEY    — passed in X-Api-Key for detection calls
VISION_COMPREFACE__VERIFICATION_API_KEY — passed in X-Api-Key for verification calls
```

For convenience, set `VISION_COMPREFACE__API_KEY` as a master key and it
will be fanned out to all three service keys automatically (unless individual
keys are also set, which take precedence).

### Key storage

- Keys are read from environment variables or `.env` file via pydantic-settings
- Keys are **never** logged, even at DEBUG level
- Keys are **never** included in error responses or JSON-RPC error data
- The `.env` file is in `.gitignore` — never commit it

### Vision router authentication

The vision-router service is internal and should be network-isolated (not
exposed to the public internet). If you need authentication, set
`VISION_ROUTER__API_KEY` and the client will send it as a bearer token.

---

## No Face Data Persistence in Router

The vision-router service does **not** store or log any image data or face
embeddings. All inference is stateless — images are decoded, processed, and
the raw bytes are discarded immediately after inference.

Face data persistence is handled exclusively by CompreFace's PostgreSQL
database on the RTX 3090 host. Backup and access control for that database
is described in [operations.md](operations.md).

---

## Dependency Security

### SBOM generation

Software Bill of Materials is generated on every build:

```bash
make sbom
# Outputs: vision/sbom-mcp.json, vision/sbom-router.json (CycloneDX format)
```

### Vulnerability scanning

```bash
make security
# Runs bandit (Python SAST) + outputs severity counts
```

Grype scans the SBOM in CI (`.github/workflows/vision-ci.yml`). High-severity
CVEs cause CI to report a warning; critical CVEs should block deployment.

### Bandit configuration

Bandit is configured via `pyproject.toml`:
- `B101` (assert statements) is skipped — asserts are used intentionally in tests
- `B104` (binding to all interfaces) is expected — service listens on 0.0.0.0
- All other bandit rules run at level LOW and above

### CVE tracking

Check for new CVEs in dependencies:

```bash
# Using pip-audit
pip-audit -r vision/mcp-server/requirements.txt
pip-audit -r vision/services/vision-router/requirements.txt

# Using grype directly
grype vision/sbom-mcp.json --fail-on high
```

### Dependency update process

1. Run `pip-audit` monthly (automated via `.github/workflows/security.yml`)
2. For HIGH or CRITICAL CVEs: patch within 7 days
3. For MEDIUM CVEs: patch within 30 days
4. Update `requirements.txt` with pinned versions after testing

---

## Network Security Recommendations

For production deployments:

1. **Put vision-mcp behind a reverse proxy** (nginx/Caddy) with:
   - TLS termination
   - Request size limits (matching `MAX_UPLOAD_MB`)
   - Rate limiting (e.g. 100 req/min per IP)

2. **Isolate vision-router** — it should only be reachable from vision-mcp,
   not from the public internet. Use Docker network isolation or firewall rules.

3. **Isolate Triton** — Triton gRPC port 8001 should only be reachable from
   vision-router. Triton has no authentication by default.

4. **Isolate CompreFace** — port 8080 should only be reachable from vision-mcp.
   CompreFace's API key is the only auth mechanism.

5. **Use secrets management** — store API keys in Vault, AWS Secrets Manager,
   or Docker secrets rather than plain `.env` files in production.

---

## Incident Response

If a security issue is discovered:

1. Check logs for evidence of exploitation:
   ```bash
   docker compose -f vision/compose/mcp.yml logs vision-mcp | grep -E "SSRF|PermissionError|403"
   ```

2. Rotate API keys immediately via the CompreFace admin interface

3. If file path sandbox was bypassed: audit `/tmp`, `/workspace`, `/data` for
   unauthorized reads

4. Report vulnerabilities privately to the repo maintainer before public disclosure
