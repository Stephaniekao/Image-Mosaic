# mosaic_generator/utils.py
import os
import cv2
import numpy as np
from .config import VALID_EXT, CACHE_META_NAME

def apply_tone(image_rgb: np.ndarray, tone_value: float) -> np.ndarray:
    """
    One-knob tone control from Black and White -> Normal -> Negative.
    tone_value in [0, 2]:
      [0,1]: blend gray -> original
      (1,2]: blend original -> negative
    
    Arg:
        img_rgb: The input image in RGB format as a numpy array.
        t: The tone adjustment value (0 to 2).

    Return:
        Image changed after tone adjustment.
    """
    # Convert to float32 for processing
    tone_image = image_rgb.astype(np.float32)
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY).astype(np.float32)
    gray3 = np.stack([gray, gray, gray], axis=2)
    negative_image = 255.0 - tone_image

    if tone_value <= 1.0:
        black_white = np.clip(tone_value, 0.0, 1.0)
        mixed = (1.0 - black_white) * gray3 + black_white * tone_image
    else:
        negative = np.clip(tone_value - 1.0, 0.0, 1.0)
        mixed = (1.0 - negative) * tone_image + negative * negative_image

    mixed = np.clip(mixed, 0.0, 255.0).astype(np.uint8)

    return mixed

def lab_mean_for_matching(original_image: np.ndarray, tone_value: float = 1.0) -> np.ndarray:
    """
    Compute mean LAB value after tone adjustment for matching.

    Arg:
        original_image: The input image in BGR format as a numpy array.
        tone_value: The tone adjustment value (0 to 2).
    
    Return:
        Mean LAB value.
    """
    # BGR -> RGB（因為你的 apply_tone 接 RGB）
    image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    if tone_value != 1.0:
        toned_rgb = apply_tone(image_rgb, tone_value)
        toned_bgr = cv2.cvtColor(toned_rgb, cv2.COLOR_RGB2BGR)
    else:
        # No tone adjustment needed, reuse original BGR image
        toned_bgr = original_image

    lab = cv2.cvtColor(toned_bgr, cv2.COLOR_BGR2LAB)

    if tone_value <= 1.0:
        L = float(lab[:, :, 0].mean())
        return np.array([L, 0.0, 0.0], dtype=np.float32)
    else:
        return lab.mean(axis=(0, 1)).astype(np.float32)
    
def latest_mosaic_path(folder: str, prefix: str = "mosaic_result_", ext: str = ".png"):
    """
    Find the newest saved mosaic result your code wrote.

    Arg:
        folder: The directory where mosaic result images are stored.
        prefix: The prefix of the mosaic result filenames.
        ext: The file extension of the mosaic result images.
    """
    # Find most recent file matching pattern
    try:
        files = [f for f in os.listdir(folder) if f.startswith(prefix) and f.endswith(ext)]
        if not files:
            return None
        files.sort(key=lambda f: os.path.getmtime(os.path.join(folder, f)), reverse=True)
        return os.path.join(folder, files[0])
    except Exception:
        return None