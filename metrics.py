import numpy as np
from skimage.metrics import structural_similarity as ssim

def perform_metric(original_image, mosaic_result):
    """
    Perform metric calculation between the original image and the mosaic result.

    Arg:
        original_image: The original image as a numpy array.
        mosaic_result: The mosaic image as a numpy array.
    
    Return:
        A tuple containing the MSE and SSIM values.

    Raises:
        ValueError: If the dimensions of the original image and mosaic result do not match.
    """
    #   Ensure both images have the same dimensions
    if original_image.shape != mosaic_result.shape:
        raise ValueError(f"Size mismatch: {original_image.shape} vs {mosaic_result.shape}")
    
    #   Normalize images to [0, 1] float32 for metric calculation
    original = original_image.astype(np.float32) / 255.0
    mosaic = mosaic_result.astype(np.float32) / 255.0
    #   Compute MSE and SSIM
    mse_value  = float(np.mean((original - mosaic) ** 2))
    ssim_value = float(ssim(original, mosaic, channel_axis=2, data_range=1.0))

    return mse_value, ssim_value


def perform_metric_rgb(original_rgb_state, adjusted):
    """
    Compute MSE/SSIM on two RGB images of the same size after tone adjustment.

    Arg:
        image_a_rgb: The first image in RGB format as a numpy array.
        image_b_rgb: The second image in RGB format as a numpy array.
    """
    if original_rgb_state.shape != adjusted.shape:
        raise ValueError(f"Size mismatch: {original_rgb_state.shape} vs {adjusted.shape}")
    
    original = original_rgb_state.astype(np.float32) / 255.0
    adjusted = adjusted.astype(np.float32) / 255.0
    
    mse_value  = float(np.mean((original - adjusted) ** 2))
    ssim_value = float(ssim(original, adjusted, channel_axis=2, data_range=1.0))

    return mse_value, ssim_value