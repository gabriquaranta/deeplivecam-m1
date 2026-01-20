#!/bin/bash

# Deep Live Cam - Apple Silicon Advanced Launch Script
# This script auto-detects system resources and configures optimal settings

# Check if running on Apple Silicon
if [[ $(uname -m) != "arm64" ]]; then
    echo "Error: This script is only for Apple Silicon Macs (M1/M2/M3)"
    exit 1
fi

echo "=========================================="
echo "Deep Live Cam - M1 Optimized Launcher"
echo "=========================================="

# Set environment variables for M1 optimization
export PYTORCH_ENABLE_MPS_FALLBACK=1
export PYTORCH_MPS_HIGH_WATERMARK_RATIO=0.0

# Activate virtual environment
if [ -d "venv" ]; then
    source venv/bin/activate
elif [ -d ".venv" ]; then
    source .venv/bin/activate
else
    echo "No virtual environment found. Please run:"
    echo "  python3 -m venv venv"
    echo "  source venv/bin/activate"
    echo "  pip install -r requirements.txt"
    exit 1
fi

# Detect system resources
TOTAL_MEM_BYTES=$(sysctl -n hw.memsize)
TOTAL_MEM_GB=$((TOTAL_MEM_BYTES / 1024 / 1024 / 1024))
SUGGESTED_MEM=$((TOTAL_MEM_GB * 3 / 4))
CPU_COUNT=$(sysctl -n hw.ncpu)
# CoreML and MPS handle parallelism internally - using multiple execution threads
# causes thread contention and deadlocks. Use single thread for frame processing.
SUGGESTED_THREADS=1

echo ""
echo "System Configuration:"
echo "  Total Memory: ${TOTAL_MEM_GB}GB"
echo "  Using Memory: ${SUGGESTED_MEM}GB (75%)"
echo "  CPU Cores: ${CPU_COUNT}"
echo "  Execution Threads: ${SUGGESTED_THREADS} (CoreML/MPS handle parallelism)"
echo ""
echo "Execution Provider: CoreML (Apple Neural Engine)"
echo "Video Encoder: h264_videotoolbox (Hardware)"
echo "=========================================="
echo ""

# Run with optimal M1 settings
python3 run.py \
    --execution-provider coreml \
    --execution-threads $SUGGESTED_THREADS \
    --max-memory $SUGGESTED_MEM \
    --video-encoder h264_videotoolbox \
    "$@"
