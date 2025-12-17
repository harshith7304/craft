import json
import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional

def generate_inpaint_mask(
    image_path: str,
    craft_json_path: str,
    output_mask_path: str,
    exclude_ids: Optional[List[int]] = None
):
    """
    Generates a binary inpainting mask from CRAFT detection results.
    
    Args:
        image_path: Path to the original image (used for dimensions)
        craft_json_path: Path to the CRAFT result JSON
        output_mask_path: Path to save the generated mask
        exclude_ids: List of region IDs to exclude from the mask (e.g. design elements)
    """
    # Load image to get dimensions
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError(f"Image not found: {image_path}")

    h, w = image.shape[:2]

    # Load CRAFT JSON
    with open(craft_json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    # Create empty mask (black)
    mask = np.zeros((h, w), dtype=np.uint8)
    
    exclude_ids = exclude_ids or []

    # Create empty mask (black)
    mask = np.zeros((h, w), dtype=np.uint8)
    
    exclude_ids = exclude_ids or []

    # Draw text polygons with per-region dilation
    for region in data.get("text_regions", []):
        if region.get("id") in exclude_ids:
            continue
            
        # Draw single polygon on temporary mask
        temp_mask = np.zeros((h, w), dtype=np.uint8)
        polygon = np.array(region["polygon"], dtype=np.int32)
        cv2.fillPoly(temp_mask, [polygon], 255)
        
        # Per-region dilation based on text height
        bbox = region["bbox"]
        text_h = bbox["height"]
        
        # Adaptive dilation: 15% of text height, clamped between 4 and 12 pixels
        dilation_val = max(4, min(12, int(0.15 * text_h)))
        kernel = np.ones((dilation_val, dilation_val), np.uint8)
        
        # Dilate this specific region
        temp_mask = cv2.dilate(temp_mask, kernel, iterations=1)
        
        # Add to main mask
        mask = cv2.bitwise_or(mask, temp_mask)

    # Save mask
    cv2.imwrite(output_mask_path, mask)
    print(f"Mask saved to: {output_mask_path}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True, help="Input image path")
    parser.add_argument("--json", required=True, help="CRAFT output JSON path")
    parser.add_argument("--out", required=True, help="Output mask path")
    parser.add_argument("--exclude", nargs="+", type=int, help="Region IDs to exclude")
    args = parser.parse_args()
    
    generate_inpaint_mask(args.image, args.json, args.out, args.exclude)
