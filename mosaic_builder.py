# mosaic_generator/mosaic_builder.py
import os
from datetime import datetime
import cv2
import numpy    
import numpy as np

from .utils import apply_tone, lab_mean_for_matching
from .tile_manager import TileManager
from .image_processor import empty_split_folder, split_image


class MosaicBuilder:
    """
    Main mosaic construction logic:
    - match split tiles to tilt tiles
    - assemble final mosaic image
    """

    def __init__(self, tile_manager: TileManager, mosaic_size: int):
        self.tile_manager = tile_manager
        self.mosaic_size = int(mosaic_size)

    # ---------- Matching algorithms ----------

    def loop_function_call(self, split_folder_path, best_match, preprocessing_folder_path, mosaic_size, tone_value=1.0):
        """
        Call the functions in sequence to process the images.
    
        Arg:
            split_folder_path: Path to the folder containing split images.
            best_match: Dictionary to store the best matching tilt image for each split image.
            preprocessing_folder_path: Path to the folder containing preprocessed tilt images.
            mosaic_size: The size of each tile (e.g., 128 for 128x128 tiles).
    
        return:
            best_match: Updated dictionary with best matches.
        """
        #   Read and process the split images, using loop method and cvtColor and save mean LAB values
        split_files = sorted([f for f in os.listdir(split_folder_path) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))])
        split_lab_dict = {}
        for filename in split_files:
            split_image = cv2.imread(os.path.join(split_folder_path, filename))
            if split_image is not None:
                split_mean_lab = lab_mean_for_matching(split_image, tone_value)  
                split_lab_dict[filename] = split_mean_lab
    
        #   Read and process the tilt images, using loop method and cvtColor and save mean LAB values
        tilt_files  = sorted([f for f in os.listdir(preprocessing_folder_path) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))])
        tilt_lab_dict = {}
        for tilt_filename in tilt_files:
            tilt_image = cv2.imread(os.path.join(preprocessing_folder_path, tilt_filename))
            if tilt_image is not None:
                tilt_mean_lab = lab_mean_for_matching(tilt_image, tone_value)
                tilt_lab_dict[tilt_filename] = tilt_mean_lab

        #   Match split images with tilt images based on mean LAB values got above
        #   For each split image, find the best matching tilt image
        #   max reuse each tilt image for 8192 / mosaic_size times, in case of using too many same tilt images
        max_reuse = int(8192 / mosaic_size)
        tilt_usage = {tilt_name: 0 for tilt_name in tilt_lab_dict}
        for split_name, split_mean_lab in split_lab_dict.items():
            min_diff = float('inf')
            best_tilt = None
            candidate_tilts = [tilt_name for tilt_name in tilt_lab_dict if tilt_usage[tilt_name] < max_reuse]
            if candidate_tilts:
                for tilt_name in candidate_tilts:
                    tilt_mean_lab = tilt_lab_dict[tilt_name]
                    lab_diff = [a - b for a, b in zip(split_mean_lab, tilt_mean_lab)]
                    l2_norm = sum((d ** 2 for d in lab_diff)) ** 0.5
                    if l2_norm < min_diff:
                        min_diff = l2_norm
                        best_tilt = tilt_name
            else:
                min_usage = min(tilt_usage.values())
                for tilt_name, tilt_mean_lab in tilt_lab_dict.items():
                    if tilt_usage[tilt_name] == min_usage:
                        lab_diff = [a - b for a, b in zip(split_mean_lab, tilt_mean_lab)]
                        l2_norm = sum((d ** 2 for d in lab_diff)) ** 0.5
                        if l2_norm < min_diff:
                            min_diff = l2_norm
                            best_tilt = tilt_name
            best_match[split_name] = best_tilt
            if best_tilt is not None:
                tilt_usage[best_tilt] += 1
        return best_match
    
    def vectorized_function_call(self, split_folder_path, best_match, preprocessing_folder_path, mosaic_size, tone_value=1.0):
        """
        Call the functions in sequence to process the images using vectorized operations.
    
        Arg:
            split_folder_path: Path to the folder containing split images.
            best_match: Dictionary to store the best matching tilt image for each split image.
            preprocessing_folder_path: Path to the folder containing preprocessed tilt images.
            mosaic_size: The size of each tile (e.g., 128 for 128x128 tiles).

        return:
            best_match: Updated dictionary with best matches.
        """
        #   Read and process the split images, using vectorized method and cvtColor and save mean LAB values
        split_files = sorted([f for f in os.listdir(split_folder_path) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))])
        split_means = []
        for filename in split_files:
            split_image = cv2.imread(os.path.join(split_folder_path, filename))
            if split_image is not None:
                split_means.append(lab_mean_for_matching(split_image, tone_value))
        split_means = np.asarray(split_means, dtype=np.float32)
    
        #   Read and process the tilt images, using vectorized method and cvtColor and save mean LAB values
        tilt_files  = sorted([f for f in os.listdir(preprocessing_folder_path) if f.lower().endswith(('.png','.jpg','.jpeg','.bmp'))])
        tilt_means = []
        for tilt_filename in tilt_files:
            tilt_image = cv2.imread(os.path.join(preprocessing_folder_path, tilt_filename))
            if tilt_image is not None:
                tilt_means.append(lab_mean_for_matching(tilt_image, tone_value))
        tilt_means = np.asarray(tilt_means, dtype=np.float32)

        #   Match split images with tilt images based on mean LAB values got above
        #   For each split image, find the best matching tilt image
        #   max reuse each tilt image for 8192 / mosaic_size times, in case of using too many same tilt images
        max_reuse = int(8192 / mosaic_size)
        tilt_usage = np.zeros(len(tilt_files), dtype=int)
        diff = split_means[:, None, :] - tilt_means[None, :, :]  
        l2_norm = np.linalg.norm(diff, axis=2) 
        for i, split_name in enumerate(split_files):
            valid_idx = np.where(tilt_usage < max_reuse)[0]
            if len(valid_idx) > 0:
                idx = valid_idx[np.argmin(l2_norm[i, valid_idx])]
            else:
                min_usage = tilt_usage.min()
                min_idx = np.where(tilt_usage == min_usage)[0]
                idx = min_idx[np.argmin(l2_norm[i, min_idx])]
            best_match[split_name] = tilt_files[idx]
            tilt_usage[idx] += 1
    
        return best_match
    
    def mosaic_image(self, original_image, mosaic_size, best_match, split_folder_path, preprocessing_folder_path, out_path, tone_value=1.0):
        """
        Create a mosaic image based on the best matches.
    
        Arg:
            best_match: Dictionary mapping split image filenames to their best matching tilt image filenames.
            split_folder_path: Path to the folder containing split images.
            preprocessing_folder_path: Path to the folder containing preprocessed tilt images.
            out_path: Path to the folder where the final mosaic image will be saved.

        Return:
            mosaic_result: The final mosaic image as a numpy array.
        """
        #   Create an empty mosaic image
        mosaic_result = numpy.zeros_like(original_image)
        #   Fill the mosaic image with the best matching tilt images, using set to avoid duplicate loading
        need = set(best_match.values())
        #   Cache for loaded tile images
        tile_cache = {}

        #   Load the necessary tile images into the cache
        for name in need:
            if name is None: 
                continue
            path = os.path.join(preprocessing_folder_path, name)
            mosaic_image = cv2.imread(path)
            if mosaic_image is not None:
                tile_rgb = cv2.cvtColor(mosaic_image, cv2.COLOR_BGR2RGB)
                toned_rgb = apply_tone(tile_rgb, float(tone_value))
                toned_bgr = cv2.cvtColor(toned_rgb, cv2.COLOR_RGB2BGR)
                tile_cache[name] = toned_bgr

        #   Fill the mosaic image
        for filename in sorted([f for f in os.listdir(split_folder_path) if f.lower().endswith('.png')]):
            tilt_filename = best_match.get(filename)
            if not tilt_filename or tilt_filename not in tile_cache:
                continue
            parts = filename.replace('.png', '').split('_')
            i, j = int(parts[1]), int(parts[2])
            mosaic_result[i:i+mosaic_size, j:j+mosaic_size] = tile_cache[tilt_filename]

        #   Save the final mosaic image with timestamp
        os.makedirs(out_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")  
        out_path = os.path.join(out_path, f"mosaic_result_{timestamp}.png")
        cv2.imwrite(out_path, mosaic_result)
    
        return mosaic_result

    def mosaic_process(self, gradio_direction, original_image, mosaic_size, tone_value=1.0, function_choosing="2"):
        """
        The main process to create a mosaic image.

        Arg:
            gradio_direction: Base folder that contains 'original_split', 'tilts', 'tilts_preprocessing'.
            original_image: The original image as a numpy array (BGR).
            mosaic_size: The size of each tile (e.g., 128 for 128x128 tiles).
            tone_value: Tone adjustment value (0 to 2).
            function_choosing: '1' for loop-based, '2' for vectorized.
        """
        # --- Use the SAME folder structure as your original Lab 1 ---
        split_folder_path = os.path.join(gradio_direction, "original_split")
        tilt_folder_path = os.path.join(gradio_direction, "tilts")
        preprocessing_folder_path = os.path.join(gradio_direction, "tilts_preprocessing")
        out_path = gradio_direction

        # Make sure folders exist
        os.makedirs(split_folder_path, exist_ok=True)
        os.makedirs(tilt_folder_path, exist_ok=True)
        os.makedirs(preprocessing_folder_path, exist_ok=True)

        # 1. Clean old split images (keep preprocessed tiles cache logic inside tilts_image_preprocessing)
        empty_split_folder(split_folder_path, preprocessing_folder_path)

        # 2. Split original image into tiles
        split_image(original_image, mosaic_size, split_folder_path)

        # 3. Preprocess tilt images (resize & cache)
        #    NOTE: function signature = (mosaic_size, tilt_folder_path, preprocessing_folder_path)
        self.tile_manager.tilts_image_preprocessing()

        # 4. Build best_match dictionary
        best_match = {}

        if function_choosing == "1":
            best_match = self.loop_function_call(
                split_folder_path, best_match, preprocessing_folder_path, mosaic_size, tone_value
            )
        else:
            best_match = self.vectorized_function_call(
                split_folder_path, best_match, preprocessing_folder_path, mosaic_size, tone_value
            )

        # 5. Create final mosaic image and save it
        self.mosaic_image(
            original_image,
            mosaic_size,
            best_match,
            split_folder_path,
            preprocessing_folder_path,
            out_path,
            tone_value=float(tone_value),
        )

        return