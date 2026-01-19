#!/usr/bin/env python3

import platform

# Apply M1 optimizations before any other imports
# This must be done early to set environment variables before torch/onnxruntime load
if platform.system() == "Darwin" and platform.machine() == "arm64":
    try:
        from config_m1 import apply_m1_optimizations

        apply_m1_optimizations()
    except ImportError:
        pass  # config_m1.py not found, continue without M1 optimizations

from modules import core

if __name__ == "__main__":
    core.run()
