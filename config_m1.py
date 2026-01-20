"""
Apple Silicon (M1/M2/M3) Optimized Configuration for Deep Live Cam
This module applies performance optimizations specific to Apple Silicon Macs.
"""

import os
import platform
import multiprocessing


def apply_m1_optimizations():
    """
    Apply optimizations for Apple Silicon Macs.
    Call this before importing heavy libraries like torch, onnxruntime, etc.
    """
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        return False

    # PyTorch MPS optimizations
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"

    # Optimize memory usage for unified memory architecture
    # Setting to 0.0 allows PyTorch to use as much memory as needed
    os.environ["PYTORCH_MPS_HIGH_WATERMARK_RATIO"] = "0.0"

    # Thread optimizations for Apple Silicon
    # CoreML and MPS manage their own internal parallelism
    # Setting OMP_NUM_THREADS > 1 causes thread contention and deadlocks
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["MKL_NUM_THREADS"] = "1"

    # Reduce memory fragmentation
    os.environ["MALLOC_MMAP_THRESHOLD_"] = "1048576"

    cpu_count = multiprocessing.cpu_count()
    print(f"[M1 Optimization] Applied optimizations for {cpu_count}-core Apple Silicon")
    print(
        f"[M1 Optimization] Thread count: 1 (CoreML/MPS handle parallelism internally)"
    )

    return True


def get_system_memory_gb() -> int:
    """Get total system memory in GB (useful for M1's unified memory)."""
    import subprocess

    try:
        result = subprocess.run(
            ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
        )
        return int(result.stdout.strip()) // (1024**3)
    except:
        return 8  # Default assumption


def get_recommended_settings() -> dict:
    """Get recommended settings for M1 Macs."""
    total_memory = get_system_memory_gb()
    cpu_count = multiprocessing.cpu_count()

    return {
        "max_memory": max(4, int(total_memory * 0.75)),
        "execution_threads": min(cpu_count, 6),
        "execution_provider": "coreml",
        "video_encoder": "h264_videotoolbox",
    }


if __name__ == "__main__":
    # Test the configuration
    if apply_m1_optimizations():
        settings = get_recommended_settings()
        print(f"\nRecommended settings for your Mac:")
        for key, value in settings.items():
            print(f"  --{key.replace('_', '-')}: {value}")
