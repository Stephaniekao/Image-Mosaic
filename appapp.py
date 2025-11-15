import cv2
import time
import gradio as gr
import numpy as np
import os

from mosaic_generator.config import MOSAIC_DIR, DEFAULT_GRID_SIZE
from mosaic_generator.tile_manager import TileManager
from mosaic_generator.mosaic_builder import MosaicBuilder   
from mosaic_generator.utils import apply_tone, latest_mosaic_path
from mosaic_generator.metrics import perform_metric, perform_metric_rgb

def gradio_runner(input_image_rgb, grid_size, algo_choice, tone_value):
    """
    Gradio runner

    Arg:
        input_image_rgb: The input image in RGB format as a numpy array.
        grid_size: The size of each tile (e.g., 128 for 128x128 tiles).
        algo_choice: The choice of algorithm ("Vectorized" or "Loop-based").
    
    Return:
        A tuple containing the mosaic image in RGB format and an information string.
    """
    if input_image_rgb is None:
        return None, "Please upload an image."

    # Convert RGB (Gradio) to BGR (OpenCV)
    original_image = cv2.cvtColor(input_image_rgb, cv2.COLOR_RGB2BGR)

    # Using global variable behavior to select algorithm
    function_choosing = '1' if algo_choice == "Loop-based" else '2'
    grid_size = int(grid_size)
    tone_value = float(tone_value)

    split_folder_path = os.path.join(MOSAIC_DIR, "original_split")
    tilt_folder_path = os.path.join(MOSAIC_DIR, "tilts")
    preprocessing_folder_path = os.path.join(MOSAIC_DIR, "tilts_preprocessing")

    os.makedirs(split_folder_path, exist_ok=True)
    os.makedirs(tilt_folder_path, exist_ok=True)
    os.makedirs(preprocessing_folder_path, exist_ok=True)

    tile_manager = TileManager(
        mosaic_size=grid_size,
        tilt_folder=tilt_folder_path,
        preprocess_folder=preprocessing_folder_path,
        tone_value=tone_value,
    )

    builder = MosaicBuilder(tile_manager=tile_manager, mosaic_size=grid_size)

    # Call mosaic_process exactly and capture runtime
    time_start = time.perf_counter()
    builder.mosaic_process(
        gradio_direction=MOSAIC_DIR,          #
        original_image=original_image,
        mosaic_size=int(grid_size),
        tone_value=float(tone_value),
        function_choosing=function_choosing,
    )
    time_end = time.perf_counter()

    # Load the newest saved mosaic file your code produced
    latest_path = latest_mosaic_path(MOSAIC_DIR)
    if latest_path is None:
        return None, "No mosaic file was generated. Please check your tilts folder and paths."

    mosaic_image = cv2.imread(latest_path)
    if mosaic_image is None:
        return None, f"Generated file not readable: {latest_path}"

    # Compute Performance Metric
    mse_value, ssim_value = perform_metric(original_image, mosaic_image)

    # Convert to RGB for display
    mosaic_rgb = cv2.cvtColor(mosaic_image, cv2.COLOR_BGR2RGB)

    # Prepare information string
    information = (
        f"Saved: {latest_path}\n"
        f"MSE: {mse_value:.4f}\n"
        f"SSIM: {ssim_value:.4f}\n"
        f"Runtime: {time_end - time_start:.3f} s\n"
        f"Algorithm: {'loop-based' if function_choosing=='1' else 'vectorized'} | "
        f"Grid: {int(grid_size)} | Tone used for matching: {float(tone_value):.2f}"
    )
    return mosaic_rgb, information, mosaic_rgb, input_image_rgb

def tone_update(mosaic_rgb_state, original_rgb_state, tone_value):
    """
    Apply tone change to the last mosaic and recompute metrics vs original.

    Arg:
        mosaic_rgb_state: The current mosaic image in RGB format as a numpy array.
        original_rgb_state: The original image in RGB format as a numpy array.
        tone_value: The tone adjustment value (0 to 2).
    
    Return:
        A tuple containing the adjusted mosaic image in RGB format and an information string.
    """

    # Ensure we have both images
    if mosaic_rgb_state is None or original_rgb_state is None:
        return None, "Please build a mosaic first."

    # Apply tone adjustment
    adjusted = apply_tone(mosaic_rgb_state, float(tone_value))
    mse_value, ssim_value = perform_metric_rgb(original_rgb_state, adjusted)
    
    info = (
        f"Tone: {tone_value:.2f} (0=Black/White, 1=Original, 2=Negative)\n"
        f"MSE (with tone): {mse_value:.4f}\n"
        f"SSIM (with tone): {ssim_value:.4f}"
    )
    return adjusted, info

### Main Gradio App ###
# Gradio UI
with gr.Blocks(title="Image Mosaic (Your Original Pipeline)") as demo:
    gr.Markdown("## 🧩 Image Mosaic (Keeps your function names & folders) 🧩\nUpload an image → choose grid size & algorithm → build.")
    
    # State to hold last mosaic and original images
    state_mosaic = gr.State(value=None)   # RGB mosaic
    state_original = gr.State(value=None) # RGB original

    with gr.Row():
        with gr.Column():
            in_img = gr.Image(type="numpy", label="Upload image")
            grid = gr.Slider(8, 128, value=16, step=8, label="Grid size (tile size)")
            algo = gr.Radio(["Vectorized", "Loop-based"], value="Vectorized", label="Algorithm")
            btn = gr.Button("Build Mosaic")
        with gr.Column():
            out_img = gr.Image(type="numpy", label="Mosaic result")
            out_info = gr.Textbox(label="Info", lines=6)

    # Tone slider row
    with gr.Row():
        tone = gr.Slider(0.0, 2.0, value=1.0, step=0.01, label="Tone (0=B/W, 1=Normal, 2=Negative)")

    # Wire up actions
    btn.click(
        fn=gradio_runner,
        inputs=[in_img, grid, algo, tone],
        outputs=[out_img, out_info, state_mosaic, state_original]
    )

    # Tone adjustment
    tone.change(
        fn=tone_update,
        inputs=[state_mosaic, state_original, tone],
        outputs=[out_img, out_info]
    )
    

# Run the Gradio app
if __name__ == "__main__":
    demo.launch()

