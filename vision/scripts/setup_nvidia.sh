#!/usr/bin/env bash
# Configure NVIDIA Container Toolkit and validate GPU access for Docker.
# Run this once on the RTX 3090 host before bringing up the inference stack.
#
# Usage: sudo bash vision/scripts/setup_nvidia.sh
set -euo pipefail

echo "=== NVIDIA Container Toolkit Setup for Vision Stack ==="
echo ""

# ---------------------------------------------------------------------------
# 1. Verify NVIDIA drivers are installed
# ---------------------------------------------------------------------------
if ! command -v nvidia-smi &> /dev/null; then
    echo "ERROR: nvidia-smi not found."
    echo "  Please install NVIDIA drivers for your OS before running this script."
    echo "  See: https://docs.nvidia.com/datacenter/tesla/tesla-installation-notes/"
    exit 1
fi

echo "GPU information:"
nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader | \
    awk -F',' '{printf "  GPU: %s  VRAM: %s  Driver: %s\n", $1, $2, $3}'
echo ""

# ---------------------------------------------------------------------------
# 2. Detect OS and install NVIDIA Container Toolkit
# ---------------------------------------------------------------------------
if ! command -v nvidia-ctk &> /dev/null; then
    echo "NVIDIA Container Toolkit not found — installing..."
    echo ""

    if command -v apt-get &> /dev/null; then
        # Debian / Ubuntu
        curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey | \
            sudo gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list | \
            sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#g' | \
            sudo tee /etc/apt/sources.list.d/nvidia-container-toolkit.list > /dev/null
        sudo apt-get update -qq
        sudo apt-get install -y nvidia-container-toolkit
    elif command -v dnf &> /dev/null; then
        # Fedora / RHEL
        curl -s -L https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo | \
            sudo tee /etc/yum.repos.d/nvidia-container-toolkit.repo > /dev/null
        sudo dnf install -y nvidia-container-toolkit
    elif command -v zypper &> /dev/null; then
        # OpenSUSE
        sudo zypper ar https://nvidia.github.io/libnvidia-container/stable/rpm/nvidia-container-toolkit.repo
        sudo zypper --gpg-auto-import-keys install -y nvidia-container-toolkit
    else
        echo "ERROR: Unsupported package manager."
        echo "  Please install nvidia-container-toolkit manually:"
        echo "  https://docs.nvidia.com/datacenter/cloud-native/container-toolkit/install-guide.html"
        exit 1
    fi
    echo ""
else
    echo "NVIDIA Container Toolkit already installed: $(nvidia-ctk --version)"
    echo ""
fi

# ---------------------------------------------------------------------------
# 3. Configure Docker runtime
# ---------------------------------------------------------------------------
echo "Configuring Docker NVIDIA runtime..."
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker
echo "Docker NVIDIA runtime configured and Docker restarted."
echo ""

# ---------------------------------------------------------------------------
# 4. Validate GPU access inside Docker
# ---------------------------------------------------------------------------
echo "Validating GPU access inside Docker..."
CUDA_IMAGE="nvcr.io/nvidia/cuda:12.0.0-base-ubuntu22.04"

if docker run --rm --gpus all "${CUDA_IMAGE}" nvidia-smi --query-gpu=name --format=csv,noheader 2>&1; then
    echo ""
    echo "GPU access in Docker: OK"
else
    echo ""
    echo "ERROR: GPU access in Docker failed."
    echo "  Try running: docker run --rm --gpus all ${CUDA_IMAGE} nvidia-smi"
    echo "  Check journalctl -u docker for errors."
    exit 1
fi

# ---------------------------------------------------------------------------
# 5. Verify Docker Compose GPU support
# ---------------------------------------------------------------------------
echo ""
echo "Checking Docker Compose version..."
docker compose version

echo ""
echo "=== Setup complete ==="
echo ""
echo "You can now start the inference stack with:"
echo "  make up-inference"
echo ""
echo "Or manually:"
echo "  docker compose -f vision/compose/inference.yml up -d"
