"""
CRAFT Text Detection MVP
========================
Detects text regions in images using CRAFT (Character Region Awareness for Text)
Outputs tight bounding boxes with cropped images as base64.

Usage:
    python text_detector_craft.py --image path/to/image.png
    python text_detector_craft.py --folder path/to/images/
"""

import os
import json
import base64
import argparse
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any, Tuple

import cv2
import numpy as np
from PIL import Image

# CRAFT detector
from craft_text_detector import Craft


class CraftTextDetector:
    """
    Text detector using CRAFT model.
    Provides tight bounding boxes and base64 cropped images.
    """
    
    def __init__(
        self,
        text_threshold: float = 0.7,
        link_threshold: float = 0.4,
        low_text: float = 0.4,
        cuda: bool = False,
        long_size: int = 1280
    ):
        """
        Initialize CRAFT detector.
        
        Args:
            text_threshold: Text confidence threshold
            link_threshold: Link confidence threshold  
            low_text: Low text score threshold
            cuda: Use GPU if available
            long_size: Maximum image dimension for processing
        """
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        self.cuda = cuda
        self.long_size = long_size
        self.craft = None
        
    def _init_model(self):
        """Initialize CRAFT model (lazy loading)."""
        if self.craft is None:
            print("Loading CRAFT model...")
            self.craft = Craft(
                output_dir=None,  # Don't save intermediate files
                crop_type="poly",
                cuda=self.cuda,
                text_threshold=self.text_threshold,
                link_threshold=self.link_threshold,
                low_text=self.low_text,
                long_size=self.long_size
            )
            print("CRAFT model loaded successfully!")
    
    def _crop_polygon(self, image: np.ndarray, polygon: np.ndarray) -> np.ndarray:
        """
        Crop image using polygon coordinates with tight bounding box.
        
        Args:
            image: Original image (BGR)
            polygon: 4-point polygon [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            
        Returns:
            Cropped image as numpy array
        """
        # Get tight bounding box from polygon
        x_coords = polygon[:, 0]
        y_coords = polygon[:, 1]
        
        x_min = max(0, int(np.floor(x_coords.min())))
        x_max = min(image.shape[1], int(np.ceil(x_coords.max())))
        y_min = max(0, int(np.floor(y_coords.min())))
        y_max = min(image.shape[0], int(np.ceil(y_coords.max())))
        
        # Crop the region
        cropped = image[y_min:y_max, x_min:x_max]
        
        return cropped
    
    def _image_to_base64(self, image: np.ndarray, format: str = "PNG") -> str:
        """
        Convert numpy image to base64 data URI.
        
        Args:
            image: Image as numpy array (BGR)
            format: Image format (PNG, JPEG)
            
        Returns:
            Base64 data URI string
        """
        # Convert BGR to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            image_rgb = image
            
        # Convert to PIL Image
        pil_image = Image.fromarray(image_rgb)
        
        # Save to bytes buffer
        buffer = BytesIO()
        pil_image.save(buffer, format=format)
        buffer.seek(0)
        
        # Encode to base64
        img_base64 = base64.b64encode(buffer.getvalue()).decode('utf-8')
        
        # Create data URI
        mime_type = f"image/{format.lower()}"
        data_uri = f"data:{mime_type};base64,{img_base64}"
        
        return data_uri
    
    def _polygon_to_bbox(self, polygon: np.ndarray) -> Dict[str, int]:
        """
        Convert polygon to axis-aligned bounding box.
        
        Args:
            polygon: 4-point polygon
            
        Returns:
            Dict with x, y, width, height
        """
        x_coords = polygon[:, 0]
        y_coords = polygon[:, 1]
        
        x = int(np.floor(x_coords.min()))
        y = int(np.floor(y_coords.min()))
        width = int(np.ceil(x_coords.max() - x_coords.min()))
        height = int(np.ceil(y_coords.max() - y_coords.min()))
        
        return {
            "x": x,
            "y": y,
            "width": width,
            "height": height
        }
    
    def detect(self, image_path: str) -> Dict[str, Any]:
        """
        Detect text regions in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict with image info and detected text regions
        """
        # Initialize model if not done
        self._init_model()
        
        # Load image
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        height, width = image.shape[:2]
        
        # Run CRAFT detection
        print(f"Processing: {image_path.name}")
        prediction_result = self.craft.detect_text(str(image_path))
        
        # Extract polygons (boxes)
        boxes = prediction_result["boxes"]
        
        # Process each detected region
        text_regions = []
        for idx, polygon in enumerate(boxes):
            polygon = np.array(polygon, dtype=np.float32)
            
            # Get bounding box
            bbox = self._polygon_to_bbox(polygon)
            
            # Skip very small regions (likely noise)
            if bbox["width"] < 10 or bbox["height"] < 10:
                continue
            
            # Crop the text region
            cropped = self._crop_polygon(image, polygon)
            
            # Skip if crop is empty
            if cropped.size == 0:
                continue
            
            # Convert to base64
            cropped_base64 = self._image_to_base64(cropped)
            
            # Create region info
            region = {
                "id": idx + 1,
                "bbox": bbox,
                "polygon": polygon.astype(int).tolist(),
                "cropped_base64": cropped_base64,
                "area": bbox["width"] * bbox["height"]
            }
            
            text_regions.append(region)
        
        # Sort by position (top-to-bottom, left-to-right)
        text_regions.sort(key=lambda r: (r["bbox"]["y"], r["bbox"]["x"]))
        
        # Re-assign IDs after sorting
        for idx, region in enumerate(text_regions):
            region["id"] = idx + 1
        
        # Build result
        result = {
            "source_image": str(image_path.absolute()),
            "image_name": image_path.name,
            "image_dimensions": {
                "width": width,
                "height": height
            },
            "total_regions": len(text_regions),
            "text_regions": text_regions
        }
        
        return result
    
    def detect_batch(self, folder_path: str, extensions: List[str] = None) -> List[Dict[str, Any]]:
        """
        Detect text in all images in a folder.
        
        Args:
            folder_path: Path to folder containing images
            extensions: List of file extensions to process
            
        Returns:
            List of detection results
        """
        if extensions is None:
            extensions = ['.png', '.jpg', '.jpeg', '.webp', '.bmp']
        
        folder = Path(folder_path)
        if not folder.exists():
            raise FileNotFoundError(f"Folder not found: {folder_path}")
        
        # Find all images
        image_files = []
        for ext in extensions:
            image_files.extend(folder.glob(f"*{ext}"))
            image_files.extend(folder.glob(f"*{ext.upper()}"))
        
        print(f"Found {len(image_files)} images in {folder_path}")
        
        # Process each image
        results = []
        for image_path in image_files:
            try:
                result = self.detect(str(image_path))
                results.append(result)
            except Exception as e:
                print(f"Error processing {image_path.name}: {e}")
                results.append({
                    "source_image": str(image_path.absolute()),
                    "image_name": image_path.name,
                    "error": str(e)
                })
        
        return results
    
    def visualize(self, image_path: str, output_path: str = None) -> str:
        """
        Visualize detected text regions on the image.
        
        Args:
            image_path: Path to input image
            output_path: Path to save visualization (optional)
            
        Returns:
            Path to visualization image
        """
        # Run detection
        result = self.detect(image_path)
        
        # Load original image
        image = cv2.imread(image_path)
        
        # Draw bounding boxes
        for region in result["text_regions"]:
            bbox = region["bbox"]
            x, y, w, h = bbox["x"], bbox["y"], bbox["width"], bbox["height"]
            
            # Draw rectangle
            cv2.rectangle(image, (x, y), (x + w, y + h), (0, 255, 0), 2)
            
            # Draw region ID
            cv2.putText(
                image, 
                f"#{region['id']}", 
                (x, y - 5),
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 255, 0), 
                2
            )
        
        # Determine output path
        if output_path is None:
            input_path = Path(image_path)
            output_path = input_path.parent / f"{input_path.stem}_detected{input_path.suffix}"
        
        # Save visualization
        cv2.imwrite(str(output_path), image)
        print(f"Visualization saved: {output_path}")
        
        return str(output_path)


def main():
    """Main entry point for CLI."""
    parser = argparse.ArgumentParser(
        description="CRAFT Text Detection - Detect text regions in images"
    )
    parser.add_argument(
        "--image", 
        type=str, 
        help="Path to single image"
    )
    parser.add_argument(
        "--folder", 
        type=str, 
        help="Path to folder containing images"
    )
    parser.add_argument(
        "--output", 
        type=str, 
        default="detection_result.json",
        help="Output JSON file path"
    )
    parser.add_argument(
        "--visualize", 
        action="store_true",
        help="Save visualization images"
    )
    parser.add_argument(
        "--cuda", 
        action="store_true",
        help="Use GPU acceleration"
    )
    parser.add_argument(
        "--text-threshold", 
        type=float, 
        default=0.7,
        help="Text confidence threshold (0-1)"
    )
    parser.add_argument(
        "--link-threshold", 
        type=float, 
        default=0.4,
        help="Link confidence threshold (0-1)"
    )
    
    args = parser.parse_args()
    
    if not args.image and not args.folder:
        parser.error("Please provide --image or --folder argument")
    
    # Initialize detector
    detector = CraftTextDetector(
        text_threshold=args.text_threshold,
        link_threshold=args.link_threshold,
        cuda=args.cuda
    )
    
    # Process single image or batch
    if args.image:
        result = detector.detect(args.image)
        
        if args.visualize:
            detector.visualize(args.image)
        
        # Save result
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f"Results saved to: {args.output}")
        
    elif args.folder:
        results = detector.detect_batch(args.folder)
        
        if args.visualize:
            for r in results:
                if "error" not in r:
                    detector.visualize(r["source_image"])
        
        # Save results
        with open(args.output, 'w', encoding='utf-8') as f:
            json.dump({"results": results}, f, indent=2)
        print(f"Results saved to: {args.output}")
    
    print("Done!")


if __name__ == "__main__":
    main()
