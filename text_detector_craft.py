"""
CRAFT Text Detection MVP - Using Official CRAFT-pytorch
=========================================================
Detects text regions in images using CRAFT (Character Region Awareness for Text)
Outputs tight bounding boxes with cropped images as base64.

Usage:
    python text_detector_craft.py --image path/to/image.png
    python text_detector_craft.py --folder path/to/images/
"""

import os
import sys
import json
import base64
import argparse
from io import BytesIO
from pathlib import Path
from typing import List, Dict, Any

import cv2
import numpy as np
from PIL import Image

# Add CRAFT-pytorch to path
CRAFT_DIR = Path(__file__).parent / "CRAFT-pytorch"
sys.path.insert(0, str(CRAFT_DIR))

import torch
import torch.backends.cudnn as cudnn
from torch.autograd import Variable

# Import CRAFT modules
from craft import CRAFT
from craft_utils import getDetBoxes, adjustResultCoordinates
from imgproc import resize_aspect_ratio, normalizeMeanVariance


class CraftTextDetector:
    """
    Text detector using official CRAFT model.
    Provides tight bounding boxes and base64 cropped images.
    """
    
    def __init__(
        self,
        model_path: str = None,
        text_threshold: float = 0.7,
        link_threshold: float = 0.4,
        low_text: float = 0.4,
        cuda: bool = False,
        canvas_size: int = 1280,
        mag_ratio: float = 1.5,
        poly: bool = False,
        padding_percentage: float = 0.035,
        merge_lines: bool = False
    ):
        """
        Initialize CRAFT detector.
        
        Args:
            model_path: Path to pretrained model
            text_threshold: Text confidence threshold
            link_threshold: Link confidence threshold  
            low_text: Low text score threshold
            cuda: Use GPU if available
            canvas_size: Maximum canvas size for processing
            mag_ratio: Image magnification ratio
            poly: Use polygon output
            padding_percentage: Percentage of width/height to add as padding (default 0.035 = 3.5%)
            merge_lines: Whether to merge close words into single lines
        """
        self.text_threshold = text_threshold
        self.link_threshold = link_threshold
        self.low_text = low_text
        self.cuda = cuda and torch.cuda.is_available()
        self.canvas_size = canvas_size
        self.mag_ratio = mag_ratio
        self.poly = poly
        self.padding_percentage = padding_percentage
        self.merge_lines = merge_lines
        
        # Default model path
        if model_path is None:
            model_path = str(CRAFT_DIR / "craft_mlt_25k.pth")
        self.model_path = model_path
        
        self.net = None
        
    def _load_model(self):
        """Load CRAFT model."""
        if self.net is None:
            print("Loading CRAFT model...")
            self.net = CRAFT()
            
            print(f"Loading weights from: {self.model_path}")
            
            # Load state dict
            if self.cuda:
                state_dict = torch.load(self.model_path)
            else:
                state_dict = torch.load(self.model_path, map_location='cpu')
            
            # Handle DataParallel saved weights (remove 'module.' prefix)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in state_dict.items():
                if k.startswith('module.'):
                    name = k[7:]  # remove 'module.' prefix
                else:
                    name = k
                new_state_dict[name] = v
            
            self.net.load_state_dict(new_state_dict)
            
            if self.cuda:
                self.net = self.net.cuda()
                self.net = torch.nn.DataParallel(self.net)
                cudnn.benchmark = False
            
            self.net.eval()
            print("CRAFT model loaded successfully!")
    
    def _crop_polygon(self, image: np.ndarray, polygon: np.ndarray) -> np.ndarray:
        """
        Crop image using polygon coordinates with tight bounding box + padding.
        
        Args:
            image: Original image (BGR)
            polygon: 4-point polygon [[x1,y1], [x2,y2], [x3,y3], [x4,y4]]
            
        Returns:
            Cropped image as numpy array
        """
        # Get raw coordinates
        x_coords = polygon[:, 0]
        y_coords = polygon[:, 1]
        
        # Calculate dynamic padding
        w = x_coords.max() - x_coords.min()
        h = y_coords.max() - y_coords.min()
        
        pad_x = w * self.padding_percentage
        pad_y = h * self.padding_percentage
        
        x_min = int(np.floor(x_coords.min() - pad_x))
        x_max = int(np.ceil(x_coords.max() + pad_x))
        y_min = int(np.floor(y_coords.min() - pad_y))
        y_max = int(np.ceil(y_coords.max() + pad_y))
        
        # Clamp to image boundaries
        x_min = max(0, x_min)
        x_max = min(image.shape[1], x_max)
        y_min = max(0, y_min)
        y_max = min(image.shape[0], y_max)
        
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
        Convert polygon to axis-aligned bounding box with padding.
        
        Args:
            polygon: 4-point polygon
            
        Returns:
            Dict with x, y, width, height
        """
        x_coords = polygon[:, 0]
        y_coords = polygon[:, 1]
        
        # Calculate dynamic padding
        w = x_coords.max() - x_coords.min()
        h = y_coords.max() - y_coords.min()
        
        pad_x = w * self.padding_percentage
        pad_y = h * self.padding_percentage
        
        x_min = int(np.floor(x_coords.min() - pad_x))
        x_max = int(np.ceil(x_coords.max() + pad_x))
        y_min = int(np.floor(y_coords.min() - pad_y))
        y_max = int(np.ceil(y_coords.max() + pad_y))

        
        # Don't clamp here, or do we? Usually bbox can handle being slightly out if not used for cropping directly
        # But for consistency, let's keep it clean
        x = x_min
        y = y_min
        width = x_max - x_min
        height = y_max - y_min
        
        return {
            "x": x,
            "y": y,
            "width": width,
            "height": height
        }

    def _merge_close_regions(self, polys: List[np.ndarray]) -> List[np.ndarray]:
        """
        Merge close polygons that are likely on the same line.
        """
        if not polys or len(polys) == 0:
            return []
            
        # 1. Convert to detailed objects
        regions = []
        for p in polys:
            if p is None: continue
            xs, ys = p[:, 0], p[:, 1]
            regions.append({
                "poly": p,
                "x1": xs.min(), "x2": xs.max(),
                "y1": ys.min(), "y2": ys.max(),
                "cy": (ys.min() + ys.max()) / 2,
                "height": ys.max() - ys.min()
            })
        
        if not regions:
            return []
            
        # 2. Group by lines (simple Y clustering)
        regions.sort(key=lambda r: r["cy"])
        lines = []
        current_line = [regions[0]]
        
        for r in regions[1:]:
            # Compare with average cy of current line
            avg_cy = sum(item["cy"] for item in current_line) / len(current_line)
            
            # Use max height of current line as reference
            avg_h = sum(item["height"] for item in current_line) / len(current_line)
            
            # If cy diff is small relative to height (0.5 * h)
            if abs(r["cy"] - avg_cy) < (0.5 * avg_h):
                current_line.append(r)
            else:
                lines.append(current_line)
                current_line = [r]
        lines.append(current_line)
        
        # 3. Merge within lines
        final_polys = []
        
        for line in lines:
            # Sort by x1 (left to right)
            line.sort(key=lambda r: r["x1"])
            
            merged_line = [line[0]]
            for r in line[1:]:
                prev = merged_line[-1]
                
                # Horizontal gap check
                # Limit gap to 1.5 * max height of the pair
                max_h = max(prev["height"], r["height"])
                gap = r["x1"] - prev["x2"]
                
                if gap < (1.5 * max_h): 
                    # Merge them
                    new_x1 = min(prev["x1"], r["x1"])
                    new_x2 = max(prev["x2"], r["x2"])
                    new_y1 = min(prev["y1"], r["y1"])
                    new_y2 = max(prev["y2"], r["y2"])
                    
                    # Create simple rect poly for merged region
                    new_poly = np.array([
                        [new_x1, new_y1], [new_x2, new_y1],
                        [new_x2, new_y2], [new_x1, new_y2]
                    ], dtype=np.float32)
                    
                    # Update prev (the last item in merged_line)
                    merged_line[-1] = {
                        "poly": new_poly,
                        "x1": new_x1, "x2": new_x2,
                        "y1": new_y1, "y2": new_y2,
                        "cy": (new_y1 + new_y2) / 2,
                        "height": new_y2 - new_y1
                    }
                else:
                    merged_line.append(r)
            
            final_polys.extend([m["poly"] for m in merged_line])
            
        return final_polys
    
    def _run_craft(self, image: np.ndarray):
        """
        Run CRAFT detection on image.
        
        Args:
            image: Image as numpy array (RGB)
            
        Returns:
            boxes, polys, score_text
        """
        # Resize
        img_resized, target_ratio, size_heatmap = resize_aspect_ratio(
            image, self.canvas_size, interpolation=cv2.INTER_LINEAR, mag_ratio=self.mag_ratio
        )
        ratio_h = ratio_w = 1 / target_ratio
        
        # Preprocessing
        x = normalizeMeanVariance(img_resized)
        x = torch.from_numpy(x).permute(2, 0, 1)  # [h, w, c] to [c, h, w]
        x = Variable(x.unsqueeze(0))  # [c, h, w] to [b, c, h, w]
        
        if self.cuda:
            x = x.cuda()
        
        # Forward pass
        with torch.no_grad():
            y, feature = self.net(x)
        
        # Make score and link map
        score_text = y[0, :, :, 0].cpu().data.numpy()
        score_link = y[0, :, :, 1].cpu().data.numpy()
        
        # Post-processing
        boxes, polys = getDetBoxes(
            score_text, score_link, 
            self.text_threshold, self.link_threshold, self.low_text, self.poly
        )
        
        # Coordinate adjustment
        boxes = adjustResultCoordinates(boxes, ratio_w, ratio_h)
        # Handle polys adjustment manually to avoid numpy ragged array errors
        for k in range(len(polys)):
            if polys[k] is not None:
                polys[k] = np.array(polys[k])
                polys[k] *= (ratio_w * 2, ratio_h * 2)  # ratio_net is 2 by default
        
        for k in range(len(polys)):
            if polys[k] is None:
                polys[k] = boxes[k]
        
        return boxes, polys, score_text
    
    def detect(self, image_path: str) -> Dict[str, Any]:
        """
        Detect text regions in an image.
        
        Args:
            image_path: Path to the image file
            
        Returns:
            Dict with image info and detected text regions
        """
        # Load model if not loaded
        self._load_model()
        
        # Load image
        image_path = Path(image_path)
        if not image_path.exists():
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        image = cv2.imread(str(image_path))
        if image is None:
            raise ValueError(f"Failed to load image: {image_path}")
        
        height, width = image.shape[:2]
        
        # Convert to RGB for CRAFT
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Run CRAFT detection
        print(f"Processing: {image_path.name}")
        boxes, polys, _ = self._run_craft(image_rgb)
        
        # Merge if requested
        if self.merge_lines:
            print("Merging close regions (line mode)...")
            polys = self._merge_close_regions(polys)
        
        # Process each detected region
        text_regions = []
        for idx, poly in enumerate(polys):
            polygon = np.array(poly, dtype=np.float32)
            
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
    parser.add_argument(
        "--padding-percentage", 
        type=float, 
        default=0.035,
        help="Padding percentage (default 0.035 = 3.5%)"
    )
    parser.add_argument(
        "--merge-lines", 
        action="store_true",
        help="Merge close words into single lines"
    )
    
    args = parser.parse_args()
    
    if not args.image and not args.folder:
        parser.error("Please provide --image or --folder argument")
    
    # Initialize detector
    detector = CraftTextDetector(
        text_threshold=args.text_threshold,
        link_threshold=args.link_threshold,
        cuda=args.cuda,
        padding_percentage=args.padding_percentage,
        merge_lines=args.merge_lines
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
