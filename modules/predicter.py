import numpy
import cv2
import modules.globals

from modules.typing import Frame

# NSFW detection is optional - TensorFlow/opennsfw2 can be heavy
# This makes it gracefully degrade if not installed
try:
    import opennsfw2
    from PIL import Image

    NSFW_AVAILABLE = True
except ImportError:
    NSFW_AVAILABLE = False
    print("[NSFW Filter] opennsfw2 not available - NSFW detection disabled")

MAX_PROBABILITY = 0.85

# Preload the model once for efficiency
model = None


def predict_frame(target_frame: Frame) -> bool:
    if not NSFW_AVAILABLE:
        return False  # Skip NSFW check if not available

    # Convert the frame to RGB before processing if color correction is enabled
    if modules.globals.color_correction:
        target_frame = cv2.cvtColor(target_frame, cv2.COLOR_BGR2RGB)

    image = Image.fromarray(target_frame)
    image = opennsfw2.preprocess_image(image, opennsfw2.Preprocessing.YAHOO)
    global model
    if model is None:
        model = opennsfw2.make_open_nsfw_model()

    views = numpy.expand_dims(image, axis=0)
    _, probability = model.predict(views)[0]
    return probability > MAX_PROBABILITY


def predict_image(target_path: str) -> bool:
    if not NSFW_AVAILABLE:
        return False
    return opennsfw2.predict_image(target_path) > MAX_PROBABILITY


def predict_video(target_path: str) -> bool:
    if not NSFW_AVAILABLE:
        return False
    _, probabilities = opennsfw2.predict_video_frames(
        video_path=target_path, frame_interval=100
    )
    return any(probability > MAX_PROBABILITY for probability in probabilities)
