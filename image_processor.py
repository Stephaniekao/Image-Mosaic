# mosaic_generator/image_processor.py
import os
import cv2
import numpy as np
from .config import VALID_EXT

def empty_split_folder(split_folder_path, preprocessing_folder_path):
    """
    Empty the folder before saving new split images.
 
    Arg:
        split_folder_path: Path to the folder containing split images.
        preprocessing_folder_path: Path to the folder containing preprocessed tilt images.
    """

    if not os.path.exists(split_folder_path):
        os.makedirs(split_folder_path, exist_ok=True)

    # Remove all files in the split folder (split folder is original image after splitting)
    for filename in os.listdir(split_folder_path):
        split_file_path = os.path.join(split_folder_path, filename)
        if os.path.isfile(split_file_path) or os.path.islink(split_file_path):
            os.remove(split_file_path)

    return
 
def split_image(original_image, mosaic_size, split_folder_path):
    """
    Split the original image into smaller tiles of size mosaic_size.
    
    Arg:
        original_image: The original image as a numpy array.
        mosaic_size: The size of each tile (e.g., 128 for 128x128 tiles).
        split_folder_path: Path to the folder where split images will be saved.
    """
    #   Crop the original image to be divisible by mosaic_size
    Height, Width = original_image.shape[:2]
    shape = mosaic_size
    Height_center, Width_center = (Height // shape) * shape, (Width // shape) * shape
    original_image = original_image[:Height_center, :Width_center]
    
    grid_height = Height_center // shape
    grid_width = Width_center // shape
    
    cells = original_image.reshape(grid_height, shape, grid_width, shape, 3)
    cells = cells.transpose(0, 2, 1, 3, 4)
    cells = cells.reshape(-1, shape, shape, 3)

    if not os.path.exists(split_folder_path):
        os.makedirs(split_folder_path, exist_ok=True)

    #   Save the split images
    for idx, tile in enumerate(cells):
        i = idx // grid_width
        j = idx % grid_width
        filename = f"split_{i * shape}_{j * shape}.png"
        cv2.imwrite(os.path.join(split_folder_path, filename), tile)

    return grid_height, grid_width, (Height_center, Width_center)