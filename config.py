# mosaic_generator/config.py
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

MOSAC_PACKAGE_ROOT = os.path.dirname(BASE_DIR)   #
MOSAIC_DIR = os.path.join(MOSAC_PACKAGE_ROOT, "Mosaic")
os.makedirs(MOSAIC_DIR, exist_ok=True)

VALID_EXT = (".png", ".jpg", ".jpeg", ".bmp", ".webp")

CACHE_META_NAME = "_cache_info.txt"

DEFAULT_GRID_SIZE = 16
MAX_REUSE_BASE = 8192  
