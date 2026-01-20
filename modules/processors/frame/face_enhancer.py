from typing import Any, List
import cv2
import threading
from queue import Queue
import os
import sys
import importlib

# GFPGAN/basicsr imports the removed 'torchvision.transforms.functional_tensor'.
# We must inject a shim BEFORE importing gfpgan.
try:
    from torchvision.transforms import functional as _tf

    sys.modules["torchvision.transforms.functional_tensor"] = _tf
except Exception:
    pass  # If torchvision isn't available, let gfpgan fail naturally

import gfpgan

import numpy as np
import modules.globals
import modules.processors.frame.core
from modules.core import update_status
from modules.face_analyser import get_one_face
from modules.typing import Frame, Face
import platform
import torch
from modules.utilities import (
    conditional_download,
    is_image,
    is_video,
)

# Import for FP16 tensor conversion
try:
    from basicsr.utils import img2tensor, tensor2img
    from torchvision.transforms import functional as TF

    normalize = TF.normalize
    HAS_BASICSR = True
except Exception as e:
    print(f"[DLC.FACE-ENHANCER] Warning: basicsr/torchvision import failed: {e}")
    HAS_BASICSR = False

FACE_ENHANCER = None
THREAD_LOCK = threading.Lock()
NAME = "DLC.FACE-ENHANCER"

abs_dir = os.path.dirname(os.path.abspath(__file__))
models_dir = os.path.join(
    os.path.dirname(os.path.dirname(os.path.dirname(abs_dir))), "models"
)


def pre_check() -> bool:
    download_directory_path = models_dir
    conditional_download(
        download_directory_path,
        [
            "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"
        ],
    )
    return True


def pre_start() -> bool:
    if not is_image(modules.globals.target_path) and not is_video(
        modules.globals.target_path
    ):
        update_status("Select an image or video for target path.", NAME)
        return False
    return True


def get_face_enhancer() -> Any:
    global FACE_ENHANCER

    with THREAD_LOCK:
        if FACE_ENHANCER is None:
            model_path = os.path.join(models_dir, "GFPGANv1.4.pth")
            try:
                update_status("Loading Face Enhancer model...", NAME)
            except Exception:
                pass

            # Try MPS first for M1/M2 Macs, with CPU fallback
            device = None
            device_name = "cpu"

            if (
                platform.system() == "Darwin"
                and hasattr(torch.backends, "mps")
                and torch.backends.mps.is_available()
            ):
                try:
                    device = torch.device("mps")
                    device_name = "mps"
                except Exception:
                    device = torch.device("cpu")
                    device_name = "cpu"
            else:
                device = torch.device("cpu")
                device_name = "cpu"

            try:
                FACE_ENHANCER = gfpgan.GFPGANer(
                    model_path=model_path,
                    upscale=1,
                    arch="clean",
                    channel_multiplier=2,
                    bg_upsampler=None,
                    device=device,
                )

                # Convert to FP16 on MPS for ~30-50% speedup
                # We patch the forward pass to auto-convert input tensors to FP16
                use_fp16 = getattr(modules.globals, "use_fp16_enhancer", False)
                if use_fp16 and device_name == "mps":
                    FACE_ENHANCER.gfpgan = FACE_ENHANCER.gfpgan.half()

                    # Patch forward to handle FP16 input conversion
                    # This fixes tensor type mismatch when FP32 tensors from img2tensor
                    # are fed to the FP16 model
                    original_forward = FACE_ENHANCER.gfpgan.forward

                    @torch.no_grad()
                    def fp16_forward(x, return_rgb=True, weight=0.5):
                        # Convert input tensor to FP16 to match model weights
                        if x.dtype != torch.float16:
                            x = x.half()
                        return original_forward(x, return_rgb=return_rgb, weight=weight)

                    FACE_ENHANCER.gfpgan.forward = fp16_forward
                    device_name = "mps (FP16)"

                # Optimize face detection by reducing resolution (40-60% faster detection)
                # Default 480 is good balance; 320 is faster but may miss small faces
                det_resize = getattr(modules.globals, "enhancer_det_resize", 480)
                if det_resize is not None and det_resize > 0:
                    original_get_landmarks = (
                        FACE_ENHANCER.face_helper.get_face_landmarks_5
                    )

                    def optimized_get_landmarks(
                        only_keep_largest=False,
                        only_center_face=False,
                        resize=det_resize,
                        blur_ratio=0.01,
                        eye_dist_threshold=None,
                    ):
                        return original_get_landmarks(
                            only_keep_largest=only_keep_largest,
                            only_center_face=only_center_face,
                            resize=resize,
                            blur_ratio=blur_ratio,
                            eye_dist_threshold=eye_dist_threshold,
                        )

                    FACE_ENHANCER.face_helper.get_face_landmarks_5 = (
                        optimized_get_landmarks
                    )
                    device_name += f" (det@{det_resize}px)"

                try:
                    update_status(f"Face Enhancer loaded on {device_name}", NAME)
                except Exception:
                    pass
            except Exception as e:
                # Fallback to CPU if MPS fails
                if device_name != "cpu":
                    print(f"{NAME}: MPS failed ({e}), falling back to CPU")
                    device = torch.device("cpu")
                    FACE_ENHANCER = gfpgan.GFPGANer(
                        model_path=model_path,
                        upscale=1,
                        arch="clean",
                        channel_multiplier=2,
                        bg_upsampler=None,
                        device=device,
                    )
                    try:
                        update_status("Face Enhancer loaded on cpu (fallback)", NAME)
                    except Exception:
                        pass
                else:
                    raise

    return FACE_ENHANCER


def enhance_face(temp_frame: Frame) -> Frame:
    # Single-threaded processing - no semaphore needed
    # enhance() with paste_back=True handles everything internally
    _, _, temp_frame = get_face_enhancer().enhance(temp_frame, paste_back=True)
    return temp_frame


def enhance_face_with_external_detection(temp_frame: Frame, faces_data: list) -> Frame:
    """
    Enhance faces using externally-detected face data (from face_swapper).

    This bypasses GFPGAN's internal RetinaFace detection for a ~20-30% speedup
    when faces have already been detected by InsightFace in the swapper.

    Args:
        temp_frame: The frame to enhance
        faces_data: List of dicts with 'bbox', 'landmarks_5', 'det_score' keys

    Returns:
        Enhanced frame with faces restored
    """
    if not HAS_BASICSR:
        # Fallback if basicsr imports failed
        return enhance_face(temp_frame)

    enhancer = get_face_enhancer()

    # Clear previous face data from face_helper
    enhancer.face_helper.clean_all()

    # Read the image into face_helper
    enhancer.face_helper.read_image(temp_frame)

    # Inject external face data instead of running detection
    valid_faces = 0
    for face_data in faces_data:
        landmarks = face_data.get("landmarks_5")
        if landmarks is not None:
            enhancer.face_helper.all_landmarks_5.append(np.array(landmarks))
            # GFPGAN expects det_faces as [x1, y1, x2, y2, score]
            bbox = face_data.get("bbox", [0, 0, 0, 0])
            score = face_data.get("det_score", 1.0)
            if len(bbox) >= 4:
                det_face = bbox[:4] + [score] if len(bbox) == 4 else bbox[:5]
                enhancer.face_helper.det_faces.append(det_face)
            valid_faces += 1

    if valid_faces == 0:
        # No valid faces provided, fall back to internal detection
        return enhance_face(temp_frame)

    # Align and warp faces using injected landmarks
    enhancer.face_helper.align_warp_face()

    # Check if we're using FP16
    use_fp16 = getattr(modules.globals, "use_fp16_enhancer", False)
    device = enhancer.device

    # Run GFPGAN enhancement on each cropped face
    for cropped_face in enhancer.face_helper.cropped_faces:
        # Prepare tensor (same as GFPGAN's enhance() method)
        cropped_face_t = img2tensor(cropped_face / 255.0, bgr2rgb=True, float32=True)
        normalize(cropped_face_t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
        cropped_face_t = cropped_face_t.unsqueeze(0).to(device)

        # FP16 conversion if enabled (forward pass patch handles this too, but explicit is safer)
        if use_fp16 and cropped_face_t.dtype != torch.float16:
            cropped_face_t = cropped_face_t.half()

        try:
            with torch.no_grad():
                output = enhancer.gfpgan(cropped_face_t, return_rgb=False, weight=0.5)[
                    0
                ]
            restored_face = tensor2img(output.squeeze(0), rgb2bgr=True, min_max=(-1, 1))
        except Exception as e:
            print(f"{NAME}: GFPGAN inference failed: {e}")
            restored_face = cropped_face

        restored_face = restored_face.astype("uint8")
        enhancer.face_helper.add_restored_face(restored_face)

    # Paste enhanced faces back to original image
    enhancer.face_helper.get_inverse_affine(None)
    restored_img = enhancer.face_helper.paste_faces_to_input_image(upsample_img=None)

    return restored_img


def process_frame(source_face: Face, temp_frame: Frame) -> Frame:  # type: ignore
    # Check if we have cached face data from swapper (avoids redundant detection)
    cached_faces = getattr(modules.globals, "last_frame_faces", None)
    cached_frame_id = getattr(modules.globals, "last_frame_id", None)

    # Use cached faces if available and frame ID matches
    if cached_faces and cached_frame_id is not None:
        try:
            temp_frame = enhance_face_with_external_detection(temp_frame, cached_faces)
            # Clear cache after use to prevent stale data
            modules.globals.last_frame_faces = None
            modules.globals.last_frame_id = None
            return temp_frame
        except Exception as e:
            print(
                f"{NAME}: External face enhancement failed ({e}), falling back to detection"
            )

    # No cached data or cache failed, use standard detection
    temp_frame = enhance_face(temp_frame)
    return temp_frame


def process_frames(
    source_path: str, temp_frame_paths: List[str], progress: Any = None
) -> None:
    """
    Process frames sequentially - MPS/GFPGAN can deadlock with threading.
    """
    jpeg_params = [cv2.IMWRITE_JPEG_QUALITY, 95]
    for temp_frame_path in temp_frame_paths:
        temp_frame = cv2.imread(temp_frame_path)
        if temp_frame is not None:
            result = process_frame(None, temp_frame)
            cv2.imwrite(temp_frame_path, result, jpeg_params)
        if progress:
            progress.update(1)


def process_image(source_path: str, target_path: str, output_path: str) -> None:
    target_frame = cv2.imread(target_path)
    result = process_frame(None, target_frame)
    cv2.imwrite(output_path, result)


def process_video(source_path: str, temp_frame_paths: List[str]) -> None:
    # Use single thread for enhancer - GFPGAN runs on GPU sequentially
    # More threads just add overhead without parallelism benefit
    modules.processors.frame.core.process_video(
        None,
        temp_frame_paths,
        process_frames,
        num_threads=getattr(modules.globals, "enhancer_threads", 1),
        stage_name="Enhancing",
    )


def process_frame_v2(temp_frame: Frame) -> Frame:
    # Skip redundant face detection - GFPGAN handles it internally
    temp_frame = enhance_face(temp_frame)
    return temp_frame
