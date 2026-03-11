#!/usr/bin/env pwsh
<#
.SYNOPSIS
    Deploy the GPU vision inference stack on a Windows 11 + RTX 3090 workstation.
    Run this in an elevated PowerShell (Run as Administrator).

.DESCRIPTION
    Installs and starts: CompreFace (GPU), Triton Inference Server (GPU),
    and vision-router. Opens required Windows Firewall ports. Downloads ONNX models.

.NOTES
    Prerequisites: Docker Desktop with WSL2 backend + NVIDIA Container Toolkit
    Tested on: Windows 11 + RTX 3090 + Docker Desktop 4.x
#>

param(
    [string]$RepoRoot    = "$PSScriptRoot\..",      # vision/ directory
    [string]$ComposeFile = "$PSScriptRoot\..\compose\inference.yml",
    [string]$CompreFaceDbPassword = "compreface_secret",
    [string]$VisionRouterApiKey   = "",
    [switch]$SkipModelDownload,
    [switch]$SkipFirewall
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

# Colours
function Write-Step  { Write-Host "`n==> $args" -ForegroundColor Cyan   }
function Write-OK    { Write-Host "    OK: $args"  -ForegroundColor Green  }
function Write-Warn  { Write-Host "    WARN: $args" -ForegroundColor Yellow }
function Write-Fail  { Write-Host "    FAIL: $args" -ForegroundColor Red    }

# ─── 0. Authorize Fedora SSH key ─────────────────────────────────────────────
Write-Step "Authorising Fedora server SSH key"
$FedoraKey = "ssh-ed25519 AAAAC3NzaC1lZDI1NTE5AAAAIOf6AB1eztW+xEGApn892s69PD05WG003UeFHnVFHiA0 jalsarraf@fedora"
$SshDir     = "$env:USERPROFILE\.ssh"
$AuthKeys   = "$SshDir\authorized_keys"
if (!(Test-Path $SshDir)) { New-Item -ItemType Directory -Path $SshDir | Out-Null }
if (!(Select-String -Path $AuthKeys -Pattern $FedoraKey -Quiet -ErrorAction SilentlyContinue)) {
    Add-Content -Path $AuthKeys -Value $FedoraKey
    # Fix permissions — Windows SSH server requires strict ACLs
    icacls $AuthKeys /inheritance:r /grant "${env:USERNAME}:F" /grant "SYSTEM:F" | Out-Null
    Write-OK "Key added to $AuthKeys"
} else {
    Write-OK "Key already present"
}

# ─── 1. Verify Docker Desktop ─────────────────────────────────────────────────
Write-Step "Checking Docker Desktop"
try {
    $dockerVersion = docker version --format "{{.Server.Version}}" 2>&1
    if ($LASTEXITCODE -ne 0) { throw "Docker daemon not running" }
    Write-OK "Docker $dockerVersion"
} catch {
    Write-Fail "Docker Desktop is not running or not installed."
    Write-Host "  Download: https://www.docker.com/products/docker-desktop/" -ForegroundColor White
    exit 1
}

# ─── 2. Verify NVIDIA GPU + Container Toolkit ─────────────────────────────────
Write-Step "Checking NVIDIA GPU"
try {
    $gpuInfo = & "nvidia-smi" --query-gpu=name,memory.total,driver_version --format=csv,noheader 2>&1
    Write-OK $gpuInfo
} catch {
    Write-Fail "nvidia-smi not found. Install NVIDIA drivers first."
    exit 1
}

Write-Step "Checking NVIDIA Container Toolkit in Docker"
$testOutput = docker run --rm --gpus all nvcr.io/nvidia/cuda:12.0.0-base-ubuntu22.04 nvidia-smi 2>&1
if ($LASTEXITCODE -ne 0) {
    Write-Warn "GPU not accessible in Docker. Installing NVIDIA Container Toolkit via WSL2..."
    Write-Host @"

  Run the following inside WSL2 (wsl.exe):
    curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
    curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list
    sudo apt-get update && sudo apt-get install -y nvidia-container-toolkit
    sudo nvidia-ctk runtime configure --runtime=docker
  Then restart Docker Desktop and re-run this script.

"@ -ForegroundColor Yellow
    exit 1
} else {
    Write-OK "NVIDIA GPU accessible in Docker"
}

# ─── 3. Download ONNX models ───────────────────────────────────────────────────
$ModelRepo = "$RepoRoot\triton\model_repository"
if (!$SkipModelDownload) {
    Write-Step "Downloading ONNX models into $ModelRepo"

    # YOLOv8n — tiny COCO detector, ~6MB
    $yoloPath = "$ModelRepo\yolov8n\1\model.onnx"
    if (!(Test-Path $yoloPath)) {
        Write-Host "  Downloading YOLOv8n..."
        $yoloUrl = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8n.onnx"
        New-Item -ItemType Directory -Force -Path (Split-Path $yoloPath) | Out-Null
        Invoke-WebRequest -Uri $yoloUrl -OutFile $yoloPath -UseBasicParsing
        Write-OK "YOLOv8n downloaded"
    } else { Write-OK "YOLOv8n already present" }

    # EfficientNet-B0, CLIP, FashionCLIP — export via Python in WSL2
    $otherModels = @("efficientnet_b0", "clip_vit_b32", "fashion_clip")
    $missing = $otherModels | Where-Object { !(Test-Path "$ModelRepo\$_\1\model.onnx") }
    if ($missing) {
        Write-Host "  Exporting remaining models via WSL2 Python (this takes ~5 min)..."
        $wslScript = ($RepoRoot -replace "\\", "/") -replace "^([A-Za-z]):", "/mnt/`$1".ToLower()
        wsl.exe -- bash "$wslScript/triton/scripts/download_models.sh" "$wslScript/triton/model_repository"
        if ($LASTEXITCODE -ne 0) {
            Write-Warn "Model download had errors — Triton will start without unavailable models."
        } else {
            Write-OK "All models exported"
        }
    } else { Write-OK "All models already present" }
} else {
    Write-Warn "Skipping model download (--SkipModelDownload). Triton will start with empty model repo."
}

# ─── 4. Write .env file ───────────────────────────────────────────────────────
Write-Step "Writing .env for inference stack"
$envContent = @"
COMPREFACE_DB_PASSWORD=$CompreFaceDbPassword
VISION_ROUTER_API_KEY=$VisionRouterApiKey
"@
Set-Content -Path "$RepoRoot\compose\.env" -Value $envContent
Write-OK ".env written"

# ─── 5. Open Windows Firewall ─────────────────────────────────────────────────
if (!$SkipFirewall) {
    Write-Step "Opening Windows Firewall ports"
    $ports = @(
        @{ Port=8080; Name="CompreFace HTTP"          },
        @{ Port=8000; Name="Triton HTTP"              },
        @{ Port=8001; Name="Triton gRPC"              },
        @{ Port=8002; Name="Triton Metrics"           },
        @{ Port=8090; Name="Vision Router HTTP"       }
    )
    foreach ($p in $ports) {
        $ruleName = "VisionStack-$($p.Port)-$($p.Name -replace ' ','')"
        $existing = Get-NetFirewallRule -DisplayName $ruleName -ErrorAction SilentlyContinue
        if (!$existing) {
            New-NetFirewallRule -DisplayName $ruleName `
                -Direction Inbound -Action Allow -Protocol TCP `
                -LocalPort $p.Port | Out-Null
            Write-OK "Opened port $($p.Port) ($($p.Name))"
        } else {
            Write-OK "Port $($p.Port) already open"
        }
    }
}

# ─── 6. Deploy with Docker Compose ────────────────────────────────────────────
Write-Step "Deploying inference stack via Docker Compose"
Push-Location "$PSScriptRoot"
try {
    docker compose -f $ComposeFile pull --quiet 2>&1 | Out-Null
    docker compose -f $ComposeFile up -d --build
    if ($LASTEXITCODE -ne 0) { throw "docker compose up failed" }
    Write-OK "Services started"
} finally {
    Pop-Location
}

# ─── 7. Health checks ─────────────────────────────────────────────────────────
Write-Step "Waiting for services to become healthy..."
Start-Sleep -Seconds 20

$checks = @(
    @{ Name="CompreFace";    Url="http://localhost:8080/actuator/health" },
    @{ Name="Triton";        Url="http://localhost:8000/v2/health/ready" },
    @{ Name="Vision Router"; Url="http://localhost:8090/v1/health"       }
)

$allOk = $true
foreach ($chk in $checks) {
    try {
        $resp = Invoke-WebRequest -Uri $chk.Url -UseBasicParsing -TimeoutSec 10 -ErrorAction Stop
        if ($resp.StatusCode -eq 200) { Write-OK "$($chk.Name): healthy" }
        else { Write-Warn "$($chk.Name): HTTP $($resp.StatusCode)"; $allOk = $false }
    } catch {
        Write-Fail "$($chk.Name): unreachable — may still be starting"
        $allOk = $false
    }
}

Write-Host ""
if ($allOk) {
    Write-Host "✓ Vision inference stack is running on $(hostname)." -ForegroundColor Green
} else {
    Write-Host "⚠  Some services are not yet ready. Check with: docker compose -f inference.yml logs" -ForegroundColor Yellow
}

Write-Host @"

Next steps:
  1. On the Fedora server, vision-mcp is already running at http://192.168.50.5:8097
  2. In LM Studio → Settings → MCP Servers → Add Server:
       URL: http://192.168.50.5:8097/sse
  3. CompreFace admin UI: http://localhost:8080  (first run: create API keys, update .env)
  4. docker compose logs -f vision-router   (to watch inference logs)

"@ -ForegroundColor Cyan
