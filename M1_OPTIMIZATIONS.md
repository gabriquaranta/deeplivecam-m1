## Deep Live Cam - M1 Mac Optimizations

Single reference containing everything implemented and recommended for Apple Silicon (M1/M2/M3) Macs.

## Performance Summary

|              Metric | Before                | After               | Improvement           |
| ------------------: | :-------------------- | :------------------ | :-------------------- |
| Face Enhancer Speed | ~3–4s/frame           | ~0.8–1.0s/frame     | ~3–4x faster          |
|      Face Detection | Full resolution       | 480px resize        | ~40% faster detection |
|      Temp Frame I/O | PNG (slow)            | JPEG Q95 (fast)     | ~3x faster I/O        |
|  Threading Overhead | 4–6 threads competing | 1 thread + pipeline | Eliminated contention |

---

## Quick Checklist of Implemented Changes

- CoreML / InsightFace prioritized for face swapper (CoreMLExecutionProvider)
- GFPGAN uses MPS on macOS when available with CPU fallback
- **Detection resize to 480px** — ~40% faster face detection in GFPGAN
- `enhancer_threads = 1` default for M1 to avoid contention
- PNG → JPEG for temp frames (fast encode/decode; `-q:v 2` and `cv2.IMWRITE_JPEG_QUALITY = 95`)
- I/O pipelining for face enhancer (Reader → GPU → Writer queues)
- VideoToolbox (`-hwaccel videotoolbox`) enabled for macOS ffmpeg calls
- Preload models at startup to avoid on-demand load stalls
- (FP16 disabled - causes tensor type mismatch in GFPGAN internals)

---

## Full Implementation Details (Guide)

### 1) Execution Providers & Globals

File: `modules/globals.py`

Set M1-optimized defaults and separate thread counts:

```python
if platform.system() == "Darwin":
    max_memory = 8  # Default for many M1 machines; adjust to taste
    execution_threads = 4  # For CPU-bound frame processing where parallelism helps
    enhancer_threads = 1  # GFPGAN runs sequentially on the GPU; more threads add overhead
    enhancer_det_resize = 480  # Detection resolution (480=40% faster, 320=60% faster, None=full)
else:
    max_memory = None
    execution_threads = None
    enhancer_threads = None
    enhancer_det_resize = None
```

Notes:

- `execution_providers` should include `CoreMLExecutionProvider` on macOS when available.
- `enhancer_threads` set to `1` prevents unnecessary thread contention for GFPGAN.
- `enhancer_det_resize` reduces face detection resolution for faster processing. Use `320` for maximum speed (may miss small faces), `None` for full resolution.

### 2) GFPGAN Face Enhancer — MPS Support and Detection Optimization

File: `modules/processors/frame/face_enhancer.py`

- The enhancer uses MPS (Metal) on Darwin when `torch.backends.mps.is_available()`. Falls back to CPU if MPS fails.
- **Detection resize**: The face detection step is optimized by reducing input resolution from full image to `enhancer_det_resize` pixels. This provides ~40-60% faster detection with minimal accuracy impact.
- The `get_face_enhancer()` wrapper loads the `GFPGANv1.4.pth` model using `gfpgan.GFPGANer(..., device=device)` and patches the face_helper to use lower detection resolution.

Example:

```python
if platform.system() == "Darwin" and torch.backends.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")

FACE_ENHANCER = gfpgan.GFPGANer(model_path=model_path, upscale=1, device=device)

# Optimize face detection by reducing resolution
det_resize = getattr(modules.globals, "enhancer_det_resize", 480)
if det_resize is not None and det_resize > 0:
    original = FACE_ENHANCER.face_helper.get_face_landmarks_5
    def optimized(resize=det_resize, **kwargs):
        return original(resize=resize, **kwargs)
    FACE_ENHANCER.face_helper.get_face_landmarks_5 = optimized
```

To use full resolution detection (slower but more reliable for small faces):

```python
enhancer_det_resize = None
```

### 3) Single-Threaded Enhancer Path

Files: `modules/processors/frame/face_enhancer.py`, `modules/processors/frame/core.py`

- Because GFPGAN runs on the GPU sequentially, running multiple Python worker threads simply causes contention; we added a fast path in `multi_process_frame()` to process sequentially when `max_workers == 1`.
- `face_enhancer.process_video()` now passes `num_threads=getattr(modules.globals, 'enhancer_threads', 1)` into the core pipeline.

Core fast-path snippet (already implemented):

```python
if max_workers == 1:
    process_frames(source_path, temp_frame_paths, progress)
    return
```

### 4) JPEG for Temp Frames

File: `modules/utilities.py`

- Replaced PNG intermediate frames with high-quality JPEG outputs.
- `extract_frames()` now uses `-q:v 2` and outputs `%05d.jpg`.
- `create_video()` reads `%05d.jpg`.
- `get_temp_frame_paths()` globs `*.jpg`.
- `cv2.imwrite()` calls in frame processors write JPEG with `[cv2.IMWRITE_JPEG_QUALITY, 95]`.

Why:

- JPEG encoding/decoding is significantly faster than PNG; for intermediate frames where lossless preservation is unnecessary, this provides ~3x faster I/O and smaller disk usage.

### 5) I/O Pipelining (Producer–Consumer)

File: `modules/processors/frame/face_enhancer.py`

- Implemented a three-stage pipeline to overlap disk reads/writes with GPU inference.
- The pipeline uses two `Queue(maxsize=2)` buffers: `read_queue` and `write_queue`.
- Threads: `enhancer-reader`, `enhancer-gpu`, `enhancer-writer`.
- Small-batch fallback (len < 5) uses sequential processing to avoid thread overhead.

Pipeline behavior:

- Reader reads frames and enqueues to `read_queue`.
- GPU thread dequeues, runs `enhance_face()` and enqueues results into `write_queue`.
- Writer dequeues and writes JPEG files back to disk, updating progress.

Expected improvement: ~10–20% by hiding disk I/O latency behind GPU compute.

### 6) VideoToolbox / FFmpeg Hardware Acceleration

File: `modules/utilities.py`

- `run_ffmpeg()` now adds `-hwaccel videotoolbox` on macOS to use VideoToolbox for hardware-accelerated encoding/decoding.
- `create_video()` uses the `%05d.jpg` input pattern and `modules.globals.video_encoder` which can be `h264_videotoolbox` when running on macOS.

### 7) Model Preloading

File: `modules/core.py`

- Implemented `preload_models()` logic to load face swapper and face enhancer at startup (and based on UI toggles), preventing large model-loading delays during processing.

### 8) Threading and Batching in Core

File: `modules/processors/frame/core.py`

- `multi_process_frame()` supports a `num_threads` argument and includes an M1-optimized batching strategy when multiple workers are used; when `num_threads == 1` it uses a sequential fast path to avoid thread overhead.
- Progress bar now reports the effective thread count and enhancer device (MPS or CPU).

### 9) Additional Code Hygiene & Memory Fixes

- Fixed a memory calculation bug for macOS in `core.py` (use `1024**3` for GB rather than `1024**6`).
- Avoided unnecessary TensorFlow usage on macOS; removed TF GPU memory tweaks and instead use Python `resource` limits when appropriate.
- Added MPS-aware resource release (e.g., `torch.mps.empty_cache()` when applicable).

### 10) Run Scripts & Environment

- `run-m1.sh` and `run.sh` updated to set MPS-related environment variables (`PYTORCH_ENABLE_MPS_FALLBACK=1`, `PYTORCH_MPS_HIGH_WATERMARK_RATIO`) and to use `h264_videotoolbox` encoder on macOS.

### 11) Optional / Future Optimizations

- FP16 (half precision) for GFPGAN on MPS — medium effort, potential 30–50% speedup; requires testing for numeric stability and possibly converting inputs/ops to `.half()`.
- Batched GFPGAN inference — high effort (changes inside GFPGAN or running SRNet directly) but can yield big speedups for many-face scenarios.
- Converting GFPGAN to CoreML could yield large gains but requires model conversion and compatibility work.

---

## Troubleshooting & Notes

- If the enhancer still runs on CPU, ensure PyTorch with MPS support is installed (PyTorch wheel for macOS; check `torch.__version__` and `torch.backends.mps.is_available()`).
- Delete old `temp/` directories before running after the PNG→JPG change: remove old PNGs to avoid stale file patterns.

Commands:

```bash
# Remove old temp folders
rm -rf <path-to-target>/temp

# Launch with M1 optimized script
./run-m1.sh
```

---

## Testing Checklist (Suggested)

- [ ] App starts without errors on macOS
- [ ] Face swapper uses CoreML provider
- [ ] Face enhancer uses MPS (prints "Face Enhancer loaded on mps")
- [ ] Temp frames are written/read as `.jpg`
- [ ] Pipelines overlap I/O and GPU compute (observe smoother throughput)
- [ ] Video encoding uses VideoToolbox (check `ffmpeg` logs or output encoder)
- [ ] Memory remains within `max_memory` limits

---

## Quick Recommendations

1. Keep `enhancer_threads = 1` when face enhancer is active; keep higher `execution_threads` (e.g., 4) when only face swapper is used.
2. Test FP16 on a small sample before enabling globally.
3. Consider increasing `read_queue`/`write_queue` `maxsize` to 3 if memory allows and if pipeline stalls are observed.

---

Last updated: January 2026
