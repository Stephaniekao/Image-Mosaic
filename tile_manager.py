# mosaic_generator/tile_manager.py
import os
import cv2
import numpy as np
from .config import VALID_EXT, CACHE_META_NAME
from .utils import lab_mean_for_matching

class TileManager:
    """
    A class to manage tile-related operations such as preprocessing tilt images
    and computing LAB means for matching.
    """

    def __init__(self, mosaic_size: int, tilt_folder: str, preprocess_folder: str, tone_value: float = 1.0):
        self.mosaic_size = int(mosaic_size)
        self.tilt_folder = tilt_folder
        self.preprocess_folder = preprocess_folder
        self.tone_value = float(tone_value)

        os.makedirs(self.preprocess_folder, exist_ok=True)

    # ---------- Tilt preprocessing ----------
    
    def tilts_image_preprocessing(self):
        """
        Preprocess the tilt images by resizing. 

        Args:
            mosaic_size: The size to which each tilt image should be resized.
            tilt_folder_path: Path to the folder containing tilt images.
            preprocessing_folder_path: Path to the folder where preprocessed images will be saved.
        """
        # Ensure the preprocessing folder exists
        mosaic_size = self.mosaic_size
        tilt_folder_path = self.tilt_folder
        preprocessing_folder_path = self.preprocess_folder

        valid_ext = ('.png', '.jpg', '.jpeg', '.bmp', '.webp')  

        # Metadata file to cache the mosaic_size used for the preprocessed tiles
        cache_meta_path = os.path.join(preprocessing_folder_path, "_cache_info.txt")
        cached_size = None  

        # Try to read the cached mosaic_size from metadata file
        if os.path.exists(cache_meta_path):
            try:
                with open(cache_meta_path, "r", encoding="utf-8") as f:
                    line = f.read().strip()
                cached_size = int(line)
            except Exception:
                cached_size = None  

        # Collect existing preprocessed images (exclude metadata file)
        existing_images = [
            f for f in os.listdir(preprocessing_folder_path)
            if f.lower().endswith(valid_ext)
        ]   

        # If we already have preprocessed tiles for this mosaic_size, reuse them
        if cached_size == mosaic_size and len(existing_images) > 0:
            print(f"[tilts_image_preprocessing] Using cached preprocessed tiles, mosaic_size = {mosaic_size}")
            return  

        # Otherwise, remove old preprocessed tiles and regenerate them
        print(f"[tilts_image_preprocessing] Regenerating preprocessed tiles, mosaic_size = {mosaic_size}")
        for filename in existing_images:
            file_path = os.path.join(preprocessing_folder_path, filename)
            if os.path.isfile(file_path) or os.path.islink(file_path):
                os.remove(file_path)    

        # Read original tilt images, resize, and save to preprocessing folder
        for filename in os.listdir(tilt_folder_path):
            if not filename.lower().endswith(valid_ext):
                continue    

            src_path = os.path.join(tilt_folder_path, filename)
            tilt_image = cv2.imread(src_path, cv2.IMREAD_COLOR) 

            # Skip unreadable or empty image
            if tilt_image is None or tilt_image.size == 0:
                continue    

            # Normalize to 3 channels in case of grayscale images
            if len(tilt_image.shape) == 2:
                tilt_image = cv2.cvtColor(tilt_image, cv2.COLOR_GRAY2BGR)   

            # Directly resize to mosaic_size x mosaic_size
            # (avoid pre-cropping that could cause 0x0 errors)
            try:
                resized_tilt_image = cv2.resize(
                    tilt_image,
                    (mosaic_size, mosaic_size),
                    interpolation=cv2.INTER_AREA
                )
            except cv2.error:
                resized_tilt_image = cv2.resize(
                    tilt_image,
                    (mosaic_size, mosaic_size),
                    interpolation=cv2.INTER_LINEAR
                )   

            dst_path = os.path.join(preprocessing_folder_path, filename)
            cv2.imwrite(dst_path, resized_tilt_image)   

        # Update cache metadata with the current mosaic_size
        try:
            with open(cache_meta_path, "w", encoding="utf-8") as f:
                f.write(str(mosaic_size))
        except Exception:
            pass    

        return  
    