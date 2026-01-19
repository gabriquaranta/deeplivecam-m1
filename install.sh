#!/bin/bash
# =============================================================================
# Deep Live Cam - M1 Mac Installation Script
# =============================================================================
# This script sets up the Python environment with all dependencies correctly
# configured for Apple Silicon (M1/M2/M3) Macs.
# =============================================================================

set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=========================================="
echo "Deep Live Cam - M1 Mac Installation"
echo "=========================================="
echo ""

# Check for Python 3.10
if command -v python3.10 &> /dev/null; then
    PYTHON_CMD="python3.10"
elif command -v python3 &> /dev/null; then
    PYTHON_VERSION=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
    if [[ "$PYTHON_VERSION" == "3.10" || "$PYTHON_VERSION" == "3.11" ]]; then
        PYTHON_CMD="python3"
    else
        echo "❌ Python 3.10 or 3.11 required (found $PYTHON_VERSION)"
        echo "Install with: brew install python@3.10"
        exit 1
    fi
else
    echo "❌ Python 3 not found. Install with: brew install python@3.10"
    exit 1
fi

echo "✓ Using Python: $PYTHON_CMD"

# Create virtual environment if it doesn't exist
if [ ! -d "venv" ]; then
    echo ""
    echo "Creating virtual environment..."
    $PYTHON_CMD -m venv venv
    echo "✓ Virtual environment created"
fi

# Activate virtual environment
echo ""
echo "Activating virtual environment..."
source venv/bin/activate

# Upgrade pip
echo ""
echo "Upgrading pip..."
pip install --upgrade pip

# Install packages with constraints to keep numpy at 1.x
echo ""
echo "Installing dependencies (this may take a few minutes)..."

# Install numpy first to lock the version
pip install numpy==1.26.4

# Install all other packages with constraint
pip install -r requirements.txt --constraint constraints.txt

# Patch basicsr for torchvision compatibility
echo ""
echo "Patching basicsr for torchvision compatibility..."
BASICSR_FILE="venv/lib/python*/site-packages/basicsr/data/degradations.py"
if ls $BASICSR_FILE 1> /dev/null 2>&1; then
    for file in $BASICSR_FILE; do
        sed -i '' 's/from torchvision.transforms.functional_tensor import rgb_to_grayscale/from torchvision.transforms.functional import rgb_to_grayscale/' "$file"
    done
    echo "✓ basicsr patched successfully"
else
    echo "⚠ basicsr file not found, skipping patch"
fi

# Verify installation
echo ""
echo "Verifying installation..."
python -c "
import numpy as np
import cv2
import onnxruntime as ort
import torch
import insightface
import gfpgan

print(f'✓ NumPy: {np.__version__}')
print(f'✓ OpenCV: {cv2.__version__}')
print(f'✓ ONNX Runtime: {ort.__version__}')
print(f'✓ ONNX Providers: {ort.get_available_providers()}')
print(f'✓ PyTorch: {torch.__version__}')
print(f'✓ MPS Available: {torch.backends.mps.is_available()}')
print(f'✓ InsightFace: {insightface.__version__}')
print('✓ GFPGAN: OK')
"

echo ""
echo "=========================================="
echo "✅ Installation Complete!"
echo "=========================================="
echo ""
echo "To run Deep Live Cam:"
echo "  ./run-m1.sh"
echo ""
echo "Or manually:"
echo "  source venv/bin/activate"
echo "  python run.py"
echo ""
