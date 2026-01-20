import os
import sys
import platform

# M1 Mac optimizations - set before other imports
if platform.system() == "Darwin":
    # CoreML and MPS handle their own parallelism - setting OMP_NUM_THREADS > 1
    # causes thread contention and deadlocks. Use single thread for OpenMP.
    os.environ["OMP_NUM_THREADS"] = "1"
    os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
else:
    # single thread doubles cuda performance - needs to be set before torch import
    if any(arg.startswith("--execution-provider") for arg in sys.argv):
        os.environ["OMP_NUM_THREADS"] = "1"

import warnings
from typing import List
import signal
import shutil
import argparse
import torch
import onnxruntime

import modules.globals
import modules.metadata
import modules.ui as ui
from modules.processors.frame.core import get_frame_processors_modules
from modules.utilities import (
    has_image_extension,
    is_image,
    is_video,
    detect_fps,
    create_video,
    extract_frames,
    get_temp_frame_paths,
    restore_audio,
    create_temp,
    move_temp,
    clean_temp,
    normalize_output_path,
)

warnings.filterwarnings("ignore", category=FutureWarning, module="insightface")
warnings.filterwarnings("ignore", category=UserWarning, module="torchvision")


def parse_args() -> None:
    signal.signal(signal.SIGINT, lambda signal_number, frame: destroy())
    program = argparse.ArgumentParser()
    program.add_argument(
        "-s", "--source", help="select an source image", dest="source_path"
    )
    program.add_argument(
        "-t", "--target", help="select an target image or video", dest="target_path"
    )
    program.add_argument(
        "-o", "--output", help="select output file or directory", dest="output_path"
    )
    program.add_argument(
        "--frame-processor",
        help="pipeline of frame processors",
        dest="frame_processor",
        default=["face_swapper"],
        choices=["face_swapper", "face_enhancer"],
        nargs="+",
    )
    program.add_argument(
        "--keep-fps",
        help="keep original fps",
        dest="keep_fps",
        action="store_true",
        default=False,
    )
    program.add_argument(
        "--keep-audio",
        help="keep original audio",
        dest="keep_audio",
        action="store_true",
        default=True,
    )
    program.add_argument(
        "--keep-frames",
        help="keep temporary frames",
        dest="keep_frames",
        action="store_true",
        default=False,
    )
    program.add_argument(
        "--many-faces",
        help="process every face",
        dest="many_faces",
        action="store_true",
        default=False,
    )
    program.add_argument(
        "--nsfw-filter",
        help="filter the NSFW image or video",
        dest="nsfw_filter",
        action="store_true",
        default=False,
    )
    program.add_argument(
        "--map-faces",
        help="map source target faces",
        dest="map_faces",
        action="store_true",
        default=False,
    )
    program.add_argument(
        "--mouth-mask",
        help="mask the mouth region",
        dest="mouth_mask",
        action="store_true",
        default=False,
    )
    program.add_argument(
        "--video-encoder",
        help="adjust output video encoder",
        dest="video_encoder",
        default="h264_videotoolbox" if platform.system() == "Darwin" else "libx264",
        choices=[
            "libx264",
            "libx265",
            "libvpx-vp9",
            "h264_videotoolbox",
            "hevc_videotoolbox",
        ],
    )
    program.add_argument(
        "--video-quality",
        help="adjust output video quality",
        dest="video_quality",
        type=int,
        default=18,
        choices=range(52),
        metavar="[0-51]",
    )
    program.add_argument("-l", "--lang", help="Ui language", default="en")
    program.add_argument(
        "--live-mirror",
        help="The live camera display as you see it in the front-facing camera frame",
        dest="live_mirror",
        action="store_true",
        default=False,
    )
    program.add_argument(
        "--live-resizable",
        help="The live camera frame is resizable",
        dest="live_resizable",
        action="store_true",
        default=False,
    )
    program.add_argument(
        "--max-memory",
        help="maximum amount of RAM in GB",
        dest="max_memory",
        type=int,
        default=suggest_max_memory(),
    )
    program.add_argument(
        "--execution-provider",
        help="execution provider",
        dest="execution_provider",
        default=get_default_execution_provider(),
        choices=suggest_execution_providers(),
        nargs="+",
    )
    program.add_argument(
        "--execution-threads",
        help="number of execution threads",
        dest="execution_threads",
        type=int,
        default=suggest_execution_threads(),
    )
    program.add_argument(
        "-v",
        "--version",
        action="version",
        version=f"{modules.metadata.name} {modules.metadata.version}",
    )

    # register deprecated args
    program.add_argument(
        "-f", "--face", help=argparse.SUPPRESS, dest="source_path_deprecated"
    )
    program.add_argument(
        "--cpu-cores", help=argparse.SUPPRESS, dest="cpu_cores_deprecated", type=int
    )
    program.add_argument(
        "--gpu-vendor", help=argparse.SUPPRESS, dest="gpu_vendor_deprecated"
    )
    program.add_argument(
        "--gpu-threads", help=argparse.SUPPRESS, dest="gpu_threads_deprecated", type=int
    )

    args = program.parse_args()

    modules.globals.source_path = args.source_path
    modules.globals.target_path = args.target_path
    modules.globals.output_path = normalize_output_path(
        modules.globals.source_path, modules.globals.target_path, args.output_path
    )
    modules.globals.frame_processors = args.frame_processor
    modules.globals.headless = args.source_path or args.target_path or args.output_path
    modules.globals.keep_fps = args.keep_fps
    modules.globals.keep_audio = args.keep_audio
    modules.globals.keep_frames = args.keep_frames
    modules.globals.many_faces = args.many_faces
    modules.globals.mouth_mask = args.mouth_mask
    modules.globals.nsfw_filter = args.nsfw_filter
    modules.globals.map_faces = args.map_faces
    modules.globals.video_encoder = args.video_encoder
    modules.globals.video_quality = args.video_quality
    modules.globals.live_mirror = args.live_mirror
    modules.globals.live_resizable = args.live_resizable
    modules.globals.max_memory = args.max_memory
    modules.globals.execution_providers = decode_execution_providers(
        args.execution_provider
    )
    modules.globals.execution_threads = args.execution_threads
    modules.globals.lang = args.lang

    # for ENHANCER tumbler:
    if "face_enhancer" in args.frame_processor:
        modules.globals.fp_ui["face_enhancer"] = True
    else:
        modules.globals.fp_ui["face_enhancer"] = False

    # translate deprecated args
    if args.source_path_deprecated:
        print(
            "\033[33mArgument -f and --face are deprecated. Use -s and --source instead.\033[0m"
        )
        modules.globals.source_path = args.source_path_deprecated
        modules.globals.output_path = normalize_output_path(
            args.source_path_deprecated, modules.globals.target_path, args.output_path
        )
    if args.cpu_cores_deprecated:
        print(
            "\033[33mArgument --cpu-cores is deprecated. Use --execution-threads instead.\033[0m"
        )
        modules.globals.execution_threads = args.cpu_cores_deprecated
    if args.gpu_vendor_deprecated == "apple":
        print(
            "\033[33mArgument --gpu-vendor apple is deprecated. Use --execution-provider coreml instead.\033[0m"
        )
        modules.globals.execution_providers = decode_execution_providers(["coreml"])
    if args.gpu_vendor_deprecated == "nvidia":
        print(
            "\033[33mArgument --gpu-vendor nvidia is deprecated. Use --execution-provider cuda instead.\033[0m"
        )
        modules.globals.execution_providers = decode_execution_providers(["cuda"])
    if args.gpu_vendor_deprecated == "amd":
        print(
            "\033[33mArgument --gpu-vendor amd is deprecated. Use --execution-provider cuda instead.\033[0m"
        )
        modules.globals.execution_providers = decode_execution_providers(["rocm"])
    if args.gpu_threads_deprecated:
        print(
            "\033[33mArgument --gpu-threads is deprecated. Use --execution-threads instead.\033[0m"
        )
        modules.globals.execution_threads = args.gpu_threads_deprecated


def encode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [
        execution_provider.replace("ExecutionProvider", "").lower()
        for execution_provider in execution_providers
    ]


def decode_execution_providers(execution_providers: List[str]) -> List[str]:
    return [
        provider
        for provider, encoded_execution_provider in zip(
            onnxruntime.get_available_providers(),
            encode_execution_providers(onnxruntime.get_available_providers()),
        )
        if any(
            execution_provider in encoded_execution_provider
            for execution_provider in execution_providers
        )
    ]


def get_default_execution_provider() -> List[str]:
    """Get the optimal default execution provider for the current platform."""
    if platform.system().lower() == "darwin":
        available = onnxruntime.get_available_providers()
        if "CoreMLExecutionProvider" in available:
            return ["coreml"]
    return ["cpu"]


def suggest_max_memory() -> int:
    if platform.system().lower() == "darwin":
        import subprocess

        try:
            # Get actual system memory on macOS
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"], capture_output=True, text=True
            )
            total_memory_gb = int(result.stdout.strip()) // (1024**3)
            # Use 75% of available memory
            return max(4, int(total_memory_gb * 0.75))
        except:
            return 8  # Default for M1 Macs (most have 8-16GB)
    return 16


def suggest_execution_providers() -> List[str]:
    return encode_execution_providers(onnxruntime.get_available_providers())


def suggest_execution_threads() -> int:
    if platform.system().lower() == "darwin":
        # M1 Macs benefit from moderate parallelism due to unified memory
        # Use performance cores effectively
        return min(os.cpu_count() or 4, 6)
    if "DmlExecutionProvider" in modules.globals.execution_providers:
        return 1
    if "ROCMExecutionProvider" in modules.globals.execution_providers:
        return 1
    return 8


def limit_resources() -> None:
    # limit memory usage (TensorFlow removed - not needed for M1)
    if modules.globals.max_memory:
        memory = modules.globals.max_memory * 1024**3  # Convert GB to bytes
        if platform.system().lower() == "darwin":
            # macOS: resource limits don't work the same way, skip
            pass
        elif platform.system().lower() != "windows":
            import resource

            try:
                soft, hard = resource.getrlimit(resource.RLIMIT_DATA)
                # Only set if requested memory is less than hard limit
                if memory <= hard or hard == resource.RLIM_INFINITY:
                    resource.setrlimit(resource.RLIMIT_DATA, (memory, hard))
            except (ValueError, OSError):
                pass  # Ignore if we can't set the limit


def release_resources() -> None:
    if "CUDAExecutionProvider" in modules.globals.execution_providers:
        torch.cuda.empty_cache()
    elif "CoreMLExecutionProvider" in modules.globals.execution_providers:
        # Clear MPS cache if using Apple Silicon
        if torch.backends.mps.is_available():
            torch.mps.empty_cache()


def _release_resources_legacy() -> None:
    # Legacy function kept for reference
    if "CUDAExecutionProvider" in modules.globals.execution_providers:
        torch.cuda.empty_cache()


def pre_check() -> bool:
    if sys.version_info < (3, 9):
        update_status(
            "Python version is not supported - please upgrade to 3.9 or higher."
        )
        return False
    if not shutil.which("ffmpeg"):
        update_status("ffmpeg is not installed.")
        return False
    return True


def preload_models() -> None:
    """Preload face swapper and enhancer models at startup to avoid loading delays during processing."""
    try:
        update_status("Preloading models...")
    except Exception:
        print("[DLC.CORE] Preloading models...")

    # Preload face_swapper if it's in the frame processors or enabled in UI
    if "face_swapper" in modules.globals.frame_processors or modules.globals.fp_ui.get(
        "face_swapper", True
    ):
        try:
            from modules.processors.frame import face_swapper

            try:
                update_status("Loading Face Swapper model...")
            except Exception:
                print("[DLC.CORE] Loading Face Swapper model...")
            face_swapper.get_face_swapper()
            try:
                update_status("Face Swapper ready")
            except Exception:
                print("[DLC.CORE] Face Swapper ready")
        except Exception as e:
            try:
                update_status(f"Warning: Could not preload Face Swapper: {e}")
            except Exception:
                print(f"[DLC.CORE] Warning: Could not preload Face Swapper: {e}")

    # Preload face_enhancer if it's in the frame processors or enabled in UI
    if (
        "face_enhancer" in modules.globals.frame_processors
        or modules.globals.fp_ui.get("face_enhancer", False)
    ):
        try:
            from modules.processors.frame import face_enhancer

            try:
                update_status("Loading Face Enhancer model...")
            except Exception:
                print("[DLC.CORE] Loading Face Enhancer model...")
            face_enhancer.get_face_enhancer()
            try:
                update_status("Face Enhancer ready")
            except Exception:
                print("[DLC.CORE] Face Enhancer ready")
        except Exception as e:
            try:
                update_status(f"Warning: Could not preload Face Enhancer: {e}")
            except Exception:
                print(f"[DLC.CORE] Warning: Could not preload Face Enhancer: {e}")

    try:
        update_status("All models loaded successfully")
    except Exception:
        print("[DLC.CORE] All models loaded successfully")


def update_status(message: str, scope: str = "DLC.CORE") -> None:
    print(f"[{scope}] {message}")
    if not modules.globals.headless:
        ui.update_status(message)


def update_progress(current: int, total: int, stage: str = None) -> None:
    """Update UI progress bar."""
    if not modules.globals.headless:
        try:
            ui.update_progress(current, total, stage)
        except Exception:
            pass


def reset_progress() -> None:
    """Reset UI progress bar."""
    if not modules.globals.headless:
        try:
            ui.reset_progress()
        except Exception:
            pass


def start() -> None:
    # Reset progress bar at start
    reset_progress()

    # Ensure models are preloaded before processing starts
    # This handles cases where processors are added via UI after initial preload
    preload_models()

    for frame_processor in get_frame_processors_modules(
        modules.globals.frame_processors
    ):
        if not frame_processor.pre_start():
            return
    update_status("Processing...")
    # process image to image
    if has_image_extension(modules.globals.target_path):
        if modules.globals.nsfw_filter and ui.check_and_ignore_nsfw(
            modules.globals.target_path, destroy
        ):
            return
        try:
            shutil.copy2(modules.globals.target_path, modules.globals.output_path)
        except Exception as e:
            print("Error copying file:", str(e))
        for frame_processor in get_frame_processors_modules(
            modules.globals.frame_processors
        ):
            update_status("Progressing...", frame_processor.NAME)
            frame_processor.process_image(
                modules.globals.source_path,
                modules.globals.output_path,
                modules.globals.output_path,
            )
            release_resources()
        if is_image(modules.globals.target_path):
            update_status("Processing to image succeed!")
        else:
            update_status("Processing to image failed!")
        return
    # process image to videos
    if modules.globals.nsfw_filter and ui.check_and_ignore_nsfw(
        modules.globals.target_path, destroy
    ):
        return

    if not modules.globals.map_faces:
        update_status("Creating temp resources...")
        create_temp(modules.globals.target_path)
        update_status("Extracting frames...")
        update_progress(0, 1, "Extracting")  # Show extracting stage
        extract_frames(modules.globals.target_path)
        update_status("Frames extracted!")

    temp_frame_paths = get_temp_frame_paths(modules.globals.target_path)
    total_frames = len(temp_frame_paths)

    for frame_processor in get_frame_processors_modules(
        modules.globals.frame_processors
    ):
        update_status("Progressing...", frame_processor.NAME)
        frame_processor.process_video(modules.globals.source_path, temp_frame_paths)
        update_status("Done!", frame_processor.NAME)
        release_resources()
    # handles fps
    update_progress(0, 1, "Encoding")  # Show encoding stage
    if modules.globals.keep_fps:
        update_status("Detecting fps...")
        fps = detect_fps(modules.globals.target_path)
        update_status(f"Creating video with {fps} fps...")
        create_video(modules.globals.target_path, fps)
    else:
        update_status("Creating video with 30.0 fps...")
        create_video(modules.globals.target_path)
    # handle audio
    if modules.globals.keep_audio:
        if modules.globals.keep_fps:
            update_status("Restoring audio...")
        else:
            update_status("Restoring audio might cause issues as fps are not kept...")
        restore_audio(modules.globals.target_path, modules.globals.output_path)
    else:
        move_temp(modules.globals.target_path, modules.globals.output_path)
    # clean and validate
    clean_temp(modules.globals.target_path)
    update_progress(1, 1, "Encoding")  # Complete encoding stage
    reset_progress()  # Reset progress bar when done
    if is_video(modules.globals.target_path):
        update_status("Processing to video succeed!")
    else:
        update_status("Processing to video failed!")


def destroy(to_quit=True) -> None:
    if modules.globals.target_path:
        clean_temp(modules.globals.target_path)
    if to_quit:
        quit()


def run() -> None:
    parse_args()
    if not pre_check():
        return
    for frame_processor in get_frame_processors_modules(
        modules.globals.frame_processors
    ):
        if not frame_processor.pre_check():
            return
    limit_resources()

    # Preload models after all checks pass
    preload_models()

    if modules.globals.headless:
        start()
    else:
        window = ui.init(start, destroy, modules.globals.lang)
        window.mainloop()
