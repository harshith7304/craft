# CRAFT Text Detection MVP

This project implements a text detection service using **CRAFT (Character Region Awareness for Text)**. It detects text regions in AI-generated ad images, extracts tight bounding boxes, and provides cropped text images as Base64 strings.

## 🚀 Features

*   **Robust Text Detection**: Uses the official `clovaai/CRAFT-pytorch` implementation.
*   **Tight Bounding Boxes**: Extracts precise polygons and axis-aligned bounding boxes for text regions.
*   **Base64 Crops**: Automatically crops text regions and encodes them as Base64 for easy API integration.
*   **JSON Output**: structured output containing image dimensions, detection count, and detailed region info.
*   **Batch Processing**: Support for processing single images or entire folders.
*   **Visualization**: Generates debug images showing detected bounding boxes and reading order.

## 🛠️ Setup

1.  **Clone the repository** (if you haven't already).

2.  **Create and activate a virtual environment**:
    ```bash
    python -m venv venv
    .\venv\Scripts\Activate.ps1  # Windows
    # source venv/bin/activate   # Linux/Mac
    ```

3.  **Install dependencies**:
    ```bash
    pip install -r requirements.txt
    ```
    *Note: This project requires `torch`, `torchvision`, `opencv-python`, and `Pillow`.*

4.  **Download Pretrained Model**:
    The system uses the `craft_mlt_25k.pth` model. If not already present in `CRAFT-pytorch/`, download it:
    ```bash
    pip install gdown
    gdown --id 1Jk4eGD7crsqCCg9C9VjCLkMN3ze8kutZ -O CRAFT-pytorch/craft_mlt_25k.pth
    ```

## 💻 Usage

The main interface is `text_detector_craft.py`, but you can use `test_craft.py` for easy testing.

### Quick Start (Testing)

Open `test_craft.py` and configure the bottom section to choose your mode:

```python
# Option 1: Process a single image
process_single_image(
    "image/your_image.png",
    detector,
    results_base="results"
)

# Option 2: Process an entire folder
# process_batch(
#     "image",
#     detector,
#     results_base="results"
# )
```

Run the test script:
```bash
python test_craft.py
```

### CLI Usage

You can also run the detector directly from the command line:

```bash
# Single Image
python text_detector_craft.py --image image/test.png --visualize

# Entire Folder
python text_detector_craft.py --folder image/ --output results.json --visualize
```

## 📂 Output Structure

Results are saved in the `outputs/` directory. Each execution creates a new numbered run folder (`run_1`, `run_2`, etc.) to keep results organized.

```text
outputs/
└── run_N/
    └── image_filename/
        ├── image_filename.json          # Main result file
        ├── image_filename_detected.png  # Visualization image
        └── crops/                       # Folder containing individual text crops
            ├── region_1.png
            ├── region_2.png
            └── ...
```

### JSON Format

The output JSON follows this structure:

```json
{
  "source_image": "...",
  "image_dimensions": { "width": 1024, "height": 1024 },
  "total_regions": 2,
  "text_regions": [
    {
      "id": 1,
      "bbox": { "x": 100, "y": 200, "width": 50, "height": 20 },
      "polygon": [[100, 200], [150, 200], [150, 220], [100, 220]],
      "area": 1000,
      "cropped_base64": "data:image/png;base64,..."
    }
  ]
}
```

## 🔧 Troubleshooting

*   **Torchvision Error**: If you encounter errors related to `model_urls`, ensure you are using the patched version of `vgg16_bn.py` provided in this repo, or stick to the specific versions in `requirements.txt`.
*   **Model Loading Error**: The wrapper script handles removing the `module.` prefix from weights saved with `DataParallel`. Standard CRAFT weights should work out of the box.

## 📜 Credits

Based on the [CRAFT-pytorch](https://github.com/clovaai/CRAFT-pytorch) repository by Clova AI.
