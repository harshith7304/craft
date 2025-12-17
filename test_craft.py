"""
CRAFT Text Detection Test Script
==================================
Easy to use script for testing CRAFT text detection on single or multiple images.

Results Structure:
results/
  image_name/
    image_name.json          # Detection results with base64
    crops/                   # All cropped text regions
      region_1.png
      region_2.png
      ...
    image_name_detected.png  # Visualization with bounding boxes
"""

import json
from pathlib import Path
from text_detector_craft import CraftTextDetector
import cv2
import base64
from io import BytesIO
from PIL import Image


def save_cropped_regions(result, output_folder):
    """
    Save all cropped regions as separate image files.
    
    Args:
        result: Detection result dict with text_regions
        output_folder: Folder to save crops
    """
    crops_folder = Path(output_folder) / "crops"
    crops_folder.mkdir(parents=True, exist_ok=True)
    
    for region in result['text_regions']:
        # Get base64 data
        base64_data = region['cropped_base64']
        
        # Remove data URI prefix
        if 'base64,' in base64_data:
            base64_data = base64_data.split('base64,')[1]
        
        # Decode base64
        img_data = base64.b64decode(base64_data)
        
        # Save as PNG
        region_file = crops_folder / f"region_{region['id']}.png"
        with open(region_file, 'wb') as f:
            f.write(img_data)
        
        print(f"  Saved crop: {region_file.name}")


def process_single_image(image_path, detector, results_base="results"):
    """
    Process a single image and save results in organized structure.
    
    Args:
        image_path: Path to image file
        detector: CraftTextDetector instance
        results_base: Base results directory
    """
    image_path = Path(image_path)
    
    print(f"\n{'='*60}")
    print(f"Processing: {image_path.name}")
    print('='*60)
    
    # Create output folder
    output_folder = Path(results_base) / image_path.stem
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Detect text
    result = detector.detect(str(image_path))
    
    # Print summary
    print(f"\nImage size: {result['image_dimensions']['width']}x{result['image_dimensions']['height']}")
    print(f"Text regions found: {result['total_regions']}")
    
    # Print each region
    for region in result['text_regions']:
        bbox = region['bbox']
        print(f"\n  Region #{region['id']}:")
        print(f"    Position: ({bbox['x']}, {bbox['y']})")
        print(f"    Size: {bbox['width']}x{bbox['height']}")
        print(f"    Area: {region['area']} pixels")
    
    # Save cropped regions
    print(f"\nSaving cropped regions...")
    save_cropped_regions(result, output_folder)
    
    # Create visualization
    viz_path = output_folder / f"{image_path.stem}_detected.png"
    detector.visualize(str(image_path), str(viz_path))
    print(f"\nVisualization saved: {viz_path}")
    
    # Save JSON result
    json_path = output_folder / f"{image_path.stem}.json"
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(result, f, indent=2)
    print(f"JSON saved: {json_path}")
    
    print(f"\n✅ Results saved to: {output_folder}")
    
    return result


def process_batch(folder_path, detector, results_base="results1"):
    """
    Process all images in a folder.
    
    Args:
        folder_path: Path to folder with images
        detector: CraftTextDetector instance
        results_base: Base results directory
    """
    folder = Path(folder_path)
    
    # Find all images
    extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp']
    image_files = []
    for ext in extensions:
        image_files.extend(folder.glob(f"*{ext}"))
        image_files.extend(folder.glob(f"*{ext.upper()}"))
    
    print(f"Found {len(image_files)} images in {folder_path}")
    print(f"{'='*60}\n")
    
    # Process each image
    results = []
    for image_path in image_files:
        try:
            result = process_single_image(image_path, detector, results_base)
            results.append(result)
        except Exception as e:
            print(f"❌ Error processing {image_path.name}: {e}")
    
    print(f"\n{'='*60}")
    print(f"✅ Batch processing complete!")
    print(f"Total images processed: {len(results)}/{len(image_files)}")
    print(f"Results saved to: {Path(results_base).absolute()}")
    
    return results


def get_next_run_dir(base_dir="runs"):
    """
    Get the next available run directory (e.g., runs/run_1, runs/run_2).
    
    Args:
        base_dir: Base directory for all runs
        
    Returns:
        Path object for the new run directory
    """
    base_path = Path(base_dir)
    base_path.mkdir(parents=True, exist_ok=True)
    
    # improved run folder logic
    i = 1
    while True:
        run_addr = base_path / f"run_{i}"
        if not run_addr.exists():
            run_addr.mkdir(parents=True, exist_ok=True)
            print(f"Created run directory: {run_addr}")
            return run_addr
        i += 1

if __name__ == "__main__":
    print("CRAFT Text Detection Test")
    print("="*60)
    
    # Initialize run directory
    run_dir = get_next_run_dir("outputs_all_line")
    
    # Initialize detector
    detector = CraftTextDetector(
        text_threshold=0.7,
        link_threshold=0.4,
        cuda=False,  # Set to True if you have GPU
        merge_lines=True
    )
    
    # =====================================================
    # CONFIGURATION: Choose what to run
    # =====================================================
    
    # Option 1: Single image test
    # Uncomment the line below to test a single image
    # process_single_image(
    #     "image/18f5c3c6-e7cf-4ea1-b674-35467e779422.png",
    #     detector,
    #     results_base=run_dir
    # )
    
    # Option 2: Batch process all images
    # Uncomment the line below to process all images in folder
    process_batch(
        "image",
        detector,
        results_base=run_dir
    )
    
    # Option 3: Process specific images
    # Uncomment the lines below to process multiple specific images
    # test_images = [
    #     "image/18f5c3c6-e7cf-4ea1-b674-35467e779422.png",
    #     "image/1ce357b5-2293-4b66-8ea5-caed661ba3d6.png",
    # ]
    # for img_path in test_images:
    #     process_single_image(img_path, detector, results_base=run_dir)
    
    print(f"\n✅ All results saved to: {run_dir}")
    print("\n✅ Done!")
