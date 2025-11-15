🧩 Image Mosaic Generator — Modular Version

A fully modularized and optimized image mosaic generator based on my Lab 1, and repaired Lav1 problem.

Lab 1 problem:
When the grid size increases, the conversion fails. For example, with grid_size = 32 the output is incorrect.

Operating Logic:
Upload an image → choose grid size & algorithm → choose tone → build.

Features include:

- Image splitting
- Tile preprocessing with caching
- Loop-based & Vectorized matching algorithms
- Tone-adjustable final output
- Comprehensive Gradio interface
- Production-ready package structure

📦 Installation
1. Clone or download this project
git clone https://github.com/Stephaniekao/Image-Mosaic.git
cd mosaic-generator

2. Create and activate a virtual environment

(Recommended so dependencies won’t conflict)

python -m venv venv
source venv/bin/activate      # macOS / Linux
venv\Scripts\activate         # Windows

3. Install required packages
pip install -r requirements.txt

numpy==2.2.6
opencv-python==4.12.0.88
pillow==11.3.0
scikit-image==0.25.2
gradio==5.46.0

🚀 Usage
1. Prepare folders

Inside your project, make sure these exist:

Mosaic/
├── tilts/               # your tile images
├── tilts_preprocessing/ # auto-generated
└── original_split/      # auto-generated


Place your tile images in:

Mosaic/tilts/

2. Launch the Gradio App
python app.py

3. In the UI:

Upload an image

Select grid size (tile size)

Choose algorithm

Vectorized → fastest

Loop-based → original behavior

Adjust tone

View metrics (MSE / SSIM) and time

The newest mosaic is saved automatically inside the Mosaic/ folder

🧱 Project Structure
mosaic_generator/
├── __init__.py
├── config.py
├── image_processor.py
├── tile_manager.py
├── mosaic_builder.py
├── metrics.py
└── utils.py

app.py
requirements.txt
README.md


This structure follows your Lab 1 functions but reorganized into clear modules:

- image_processor.py
- empty_split_folder
- split_image
- tile_manager.py
- tilts_image_preprocessing (your original)
- compute_split_features
- compute_tilt_features
- mosaic_builder.py
- loop_function_call
- vectorized_function_call
- mosaic_image
- mosaic_process (main pipeline)
- metrics.py
- mse_ssim_bgr
- mse_ssim_rgb
- utils.py
- apply_tone
- lab_mean_for_matching
- latest_mosaic_path

⚡ Performance Benchmarks

Mosaic Pipeline Performance (Lab 1 vs Modular Version)

Benchmark based on second-run timing (cached tile preprocessing).

Grid Size: 8
- Lab 1 Runtime     : 20.086 s
- Modular Runtime   : 14.771 s
- Speedup           : 1.36x

Grid Size: 16
- Lab 1 Runtime     : 10.259 s
- Modular Runtime   : 4.498 s
- Speedup           : 2.28x

Grid Size: 32
- Lab 1 Runtime     : 7.598 s
- Modular Runtime   : 1.924 s
- Speedup           : 3.95x

The refactored modular mosaic pipeline achieved significant performance improvements compared to the original Lab 1 single-file implementation.
- At small grid sizes (8×8), the pipeline is already 1.36× faster, showing that the new processing flow reduces unnecessary repeated operations.
- At medium grid sizes (16×16), speedup increases to 2.28×, demonstrating that the reorganized execution order and reduced tile I/O overhead scale better as computation grows.
- At larger grid sizes (32×32), the benefit becomes more obvious, reaching nearly 4× faster performance. This highlights how the modular design eliminates redundant tone conversions and per-tile processing in the original version.

Overall, the new architecture provides:
- More efficient execution
- Cleaner module boundaries
- A stable caching system for tile preprocessing
- Dramatic performance improvements especially at larger grid sizes

This benchmark verifies that the modular refactor not only improves maintainability but also significantly boosts efficiency.

🌐 Deploying the Gradio App

You can deploy using:

Option A — Gradio Hosted Spaces 
gradio deploy

Option B — HuggingFace Spaces

Upload:
app.py
mosaic_generator/
requirements.txt

Option C — Share temporary link

In app.py:
demo.launch(share=True)
Gradio will generate a public link instantly:
https://xxxx.gradio.live

📝 Example Code Snippet
from mosaic_generator.mosaic_builder import mosaic_process
from mosaic_generator.config import MOSAIC_DIR

mosaic_process(
    gradio_direction=MOSAIC_DIR,
    original_image=cv2.imread("input.jpg"),
    mosaic_size=32,
    tone_value=1.0,
    function_choosing="2",  # vectorized
)

🤝 Credits

This project is based on the original coursework implementation from
CS5130 — NU Image Mosaic Project
by Stephanie Kao (Wan Chi Kao)
