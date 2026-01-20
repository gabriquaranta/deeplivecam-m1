import os
import platform
from typing import List, Dict, Any

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(ROOT_DIR, "workflow")

file_types = [
    ("Image", ("*.png", "*.jpg", "*.jpeg", "*.gif", "*.bmp")),
    ("Video", ("*.mp4", "*.mkv")),
]

source_target_map = []
simple_map = {}

source_path = None
target_path = None
output_path = None
frame_processors: List[str] = []
keep_fps = True
keep_audio = True
keep_frames = False
many_faces = False
map_faces = False
color_correction = False  # New global variable for color correction toggle
nsfw_filter = False
video_encoder = None
video_quality = None
live_mirror = False
live_resizable = True

# M1-optimized defaults
if platform.system() == "Darwin":
    max_memory = 8  # Better default for M1 unified memory
    # CoreML/MPS handle parallelism internally - using multiple execution threads
    # causes thread contention and deadlocks with these backends
    execution_threads = 1
    # GFPGAN runs sequentially on GPU - more threads just add overhead
    enhancer_threads = 1
    # FP16 for GFPGAN - disabled due to input tensor type mismatch in GFPGAN internals
    # Enabling requires modifying GFPGAN's face_helper to also use FP16 tensors
    use_fp16_enhancer = False
    # Detection resolution for GFPGAN face detection (480=40% faster, 320=60% faster, None=full)
    enhancer_det_resize = 480
else:
    max_memory = None
    execution_threads = None
    enhancer_threads = None
    use_fp16_enhancer = False
    enhancer_det_resize = None

execution_providers: List[str] = []
headless = None
log_level = "error"
fp_ui: Dict[str, bool] = {"face_enhancer": False}
camera_input_combobox = None
webcam_preview_running = False
show_fps = False
mouth_mask = False
show_mouth_mask_box = False
mask_feather_ratio = 8
mask_down_size = 0.50
mask_size = 1
