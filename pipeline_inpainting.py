import os
import argparse
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
# from simple_lama_inpainting import SimpleLama # Imported dynamically
from mask_generator import generate_inpaint_mask

def run_pipeline(run_dir: str, exclude_ids: list = None):
    """
    Run inpainting pipeline on a CRAFT output run directory.
    
    Structure assumptions:
    run_dir/
      image_name/
        image_name.json
    """
    run_path = Path(run_dir)
    if not run_path.exists():
        print(f"Error: Directory {run_dir} does not exist.")
        return

    # PATCH: Numpy 2.0 compatibility for older libraries
    if hasattr(np, 'float'):
        pass
    else:
        # Map removed types to standard python types
        np.float = float
        np.int = int
        np.bool = bool
    
    from simple_lama_inpainting import SimpleLama
    print("Initializing LaMa model...")
    lama = SimpleLama()

    # Iterate through all subdirectories
    subdirs = [d for d in run_path.iterdir() if d.is_dir()]
    
    print(f"Found {len(subdirs)} images to process in {run_dir}")

    for subdir in subdirs:
        print(f"\nProcessing: {subdir.name}")
        
        # Find JSON file
        json_files = list(subdir.glob("*.json"))
        if not json_files:
            print("  No JSON found, skipping.")
            continue
        
        json_path = json_files[0]
        
        # Determine source image path
        # 1. Try to find the image in the ../../image/ folder based on folder name
        #    (Assuming standard test_craft.py structure)
        # 2. Or read 'source_image' key from JSON if valid
        
        # Method 2 is safer if JSON has absolute path
        image_path = None
        import json
        with open(json_path, 'r') as f:
            data = json.load(f)
            if 'source_image' in data:
                possible_path = data['source_image']
                if os.path.exists(possible_path):
                    image_path = possible_path
        
        # Fallback logic if JSON path is invalid/local
        if not image_path:
             # Try standard relative location
             # .../craft/image/image_name.png
             # We are in .../craft/outputs/run_X/image_name
             base_craft_dir = run_path.parent.parent # adjust based on where script runs
             # Actually, simpler: construct path from known image folder
             possible_extensions = ['.png', '.jpg', '.jpeg', 'webp']
             for ext in possible_extensions:
                 try_path = Path("image") / f"{subdir.name}{ext}"
                 if try_path.exists():
                     image_path = str(try_path)
                     break
        
        if not image_path:
            print("  Could not locate source image file.")
            continue
            
        print(f"  Source Image: {image_path}")
        
        # Paths
        mask_path = subdir / "inpainting_Lama_mask.png"
        output_path = subdir / "inpainted_Lama_result.png"
        
        # 1. Generate Mask
        print("  Generating mask...")
        try:
            generate_inpaint_mask(
                image_path=image_path,
                craft_json_path=str(json_path),
                output_mask_path=str(mask_path),
                exclude_ids=exclude_ids
            )
        except Exception as e:
            print(f"  Error generating mask: {e}")
            continue

        # 2. Run LaMa Inpainting
        print("  Running LaMa inpainting...")
        try:
            image = Image.open(image_path).convert("RGB")
            mask = Image.open(mask_path).convert("L")
            
            result = lama(image, mask)
            result.save(output_path)
            print(f"  Inpainted result saved to: {output_path}")
            
        except Exception as e:
            print(f"  Error running LaMa: {e}")

if __name__ == "__main__":
    # =====================================================
    # CONFIGURATION: Set your run directory here
    # =====================================================
    
    # Default directory to process
    RUN_DIR = "outputs_all_line/run_2"
    
    # Optional: IDs to exclude from inpainting (e.g. [5, 6] for specific logos)
    EXCLUDE_IDS = [] 

    # =====================================================
    
    parser = argparse.ArgumentParser(description="Run inpainting pipeline on CRAFT results")
    parser.add_argument("--run-dir", default=RUN_DIR, help="Path to the run directory")
    parser.add_argument("--exclude-ids", nargs="+", type=int, default=EXCLUDE_IDS, help="List of region IDs to exclude")
    
    args = parser.parse_args()
    
    print(f"Starting inpainting pipeline on: {args.run_dir}")
    if args.exclude_ids:
        print(f"Excluding regions: {args.exclude_ids}")
        
    run_pipeline(args.run_dir, args.exclude_ids)
