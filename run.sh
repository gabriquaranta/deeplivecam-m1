#!/bin/bash

# Deep Live Cam - Apple Silicon (M1/M2/M3) Optimized Launch Script

# Check if running on Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "Warning: This script is optimized for Apple Silicon Macs"
fi

# Set environment variables for M1 optimization
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "No virtual environment found. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    pip install --upgrade pip
    pip install -r requirements.txt
fi

# Run with CoreML execution provider (optimal for M1)
python3 run.py --execution-provider coreml "$@"
