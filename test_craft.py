"""
Quick Test Script for CRAFT Text Detection
============================================
Run this to test the detector on sample images.
"""

import json
from pathlib import Path
from text_detector_craft import CraftTextDetector


def test_single_image():
    """Test on a single image and show results."""
    
    # Initialize detector
    detector = CraftTextDetector(
        text_threshold=0.7,
        link_threshold=0.4,
        cuda=False  # Set to True if you have GPU
    )
    
    # Test image path - using one of your ad images
    image_folder = Path("./image")
    test_images = list(image_folder.glob("*.png"))[:3]  # Test first 3 images
    
    if not test_images:
        print("No images found in ./image folder")
        return
    
    for image_path in test_images:
        print(f"\n{'='*60}")
        print(f"Processing: {image_path.name}")
        print('='*60)
        
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
            print(f"    Base64 length: {len(region['cropped_base64'])} chars")
        
        # Create visualization
        viz_path = detector.visualize(str(image_path))
        print(f"\nVisualization saved: {viz_path}")
        
        # Save JSON result for this image
        output_file = f"result_{image_path.stem}.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, indent=2)
        print(f"JSON saved: {output_file}")


def test_batch():
    """Test on all images in folder."""
    
    detector = CraftTextDetector(cuda=False)
    
    results = detector.detect_batch("./image")
    
    # Save all results
    with open("all_results.json", 'w', encoding='utf-8') as f:
        json.dump({"results": results}, f, indent=2)
    
    print(f"\n{'='*60}")
    print(f"Batch processing complete!")
    print(f"Total images: {len(results)}")
    print(f"Results saved to: all_results.json")


if __name__ == "__main__":
    print("CRAFT Text Detection Test")
    print("="*60)
    
    # Run single image test first
    test_single_image()
    
    # Uncomment below to test all images:
    # test_batch()
