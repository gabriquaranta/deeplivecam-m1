import sys
import importlib
import platform
from concurrent.futures import ThreadPoolExecutor
from types import ModuleType
from typing import Any, List, Callable
from tqdm import tqdm

import modules
import modules.globals

FRAME_PROCESSORS_MODULES: List[ModuleType] = []
FRAME_PROCESSORS_INTERFACE = [
    "pre_check",
    "pre_start",
    "process_frame",
    "process_image",
    "process_video",
]


def load_frame_processor_module(frame_processor: str) -> Any:
    try:
        frame_processor_module = importlib.import_module(
            f"modules.processors.frame.{frame_processor}"
        )
        for method_name in FRAME_PROCESSORS_INTERFACE:
            if not hasattr(frame_processor_module, method_name):
                sys.exit()
    except ImportError:
        print(f"Frame processor {frame_processor} not found")
        sys.exit()
    return frame_processor_module


def get_frame_processors_modules(frame_processors: List[str]) -> List[ModuleType]:
    global FRAME_PROCESSORS_MODULES

    if not FRAME_PROCESSORS_MODULES:
        for frame_processor in frame_processors:
            frame_processor_module = load_frame_processor_module(frame_processor)
            FRAME_PROCESSORS_MODULES.append(frame_processor_module)
    set_frame_processors_modules_from_ui(frame_processors)
    return FRAME_PROCESSORS_MODULES


def set_frame_processors_modules_from_ui(frame_processors: List[str]) -> None:
    global FRAME_PROCESSORS_MODULES
    for frame_processor, state in modules.globals.fp_ui.items():
        if state == True and frame_processor not in frame_processors:
            frame_processor_module = load_frame_processor_module(frame_processor)
            FRAME_PROCESSORS_MODULES.append(frame_processor_module)
            modules.globals.frame_processors.append(frame_processor)
        if state == False:
            try:
                frame_processor_module = load_frame_processor_module(frame_processor)
                FRAME_PROCESSORS_MODULES.remove(frame_processor_module)
                modules.globals.frame_processors.remove(frame_processor)
            except:
                pass


def multi_process_frame(
    source_path: str,
    temp_frame_paths: List[str],
    process_frames: Callable[[str, List[str], Any], None],
    progress: Any = None,
    num_threads: int = None,
) -> None:
    max_workers = num_threads if num_threads else modules.globals.execution_threads

    # Single-threaded: process sequentially without thread pool overhead
    if max_workers == 1:
        process_frames(source_path, temp_frame_paths, progress)
        return

    # On M1 Mac, process in batches to optimize for unified memory architecture
    if platform.system() == "Darwin":
        batch_size = max(1, len(temp_frame_paths) // max_workers)
        batches = [
            temp_frame_paths[i : i + batch_size]
            for i in range(0, len(temp_frame_paths), batch_size)
        ]

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for batch in batches:
                future = executor.submit(process_frames, source_path, batch, progress)
                futures.append(future)
            for future in futures:
                future.result()
    else:
        # Original per-frame threading for other platforms
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = []
            for path in temp_frame_paths:
                future = executor.submit(process_frames, source_path, [path], progress)
                futures.append(future)
            for future in futures:
                future.result()


def process_video(
    source_path: str,
    frame_paths: list[str],
    process_frames: Callable[[str, List[str], Any], None],
    num_threads: int = None,
    stage_name: str = "Processing",
) -> None:
    import platform
    import torch

    # Use provided thread count or global default
    effective_threads = (
        num_threads if num_threads else modules.globals.execution_threads
    )

    # Determine enhancer device for display
    enhancer_device = "CPU"
    if (
        platform.system() == "Darwin"
        and hasattr(torch.backends, "mps")
        and torch.backends.mps.is_available()
    ):
        use_fp16 = getattr(modules.globals, "use_fp16_enhancer", False)
        enhancer_device = "MPS-FP16" if use_fp16 else "MPS"

    progress_bar_format = (
        "{l_bar}{bar}| {n_fmt}/{total_fmt} [{elapsed}<{remaining}, {rate_fmt}{postfix}]"
    )
    total = len(frame_paths)

    # Try to get UI progress callback
    try:
        from modules import ui

        ui_update = lambda n, t: ui.update_progress(n, t, stage_name)
    except Exception:
        ui_update = lambda n, t: None

    with tqdm(
        total=total,
        desc=stage_name,
        unit="frame",
        dynamic_ncols=True,
        bar_format=progress_bar_format,
    ) as progress:
        progress.set_postfix(
            {
                "exec_providers": f"CoreML(swapper),{enhancer_device}(enhancer)",
                "threads": effective_threads,
                "max_mem": f"{modules.globals.max_memory}GB",
            }
        )

        # Wrap progress to also update UI
        original_update = progress.update

        def wrapped_update(n=1):
            original_update(n)
            ui_update(progress.n, total)

        progress.update = wrapped_update

        # Initial UI update
        ui_update(0, total)

        multi_process_frame(
            source_path, frame_paths, process_frames, progress, num_threads
        )
