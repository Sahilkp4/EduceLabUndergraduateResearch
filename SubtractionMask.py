#!/usr/bin/env python3
"""
Automatic Background Subtraction Masking for Photogrammetry

This script performs background subtraction on a set of images to create
mask images suitable for photogrammetry processing.

Compares each image against a background reference to isolate the foreground object.
Uses Triangle thresholding on the difference image to automatically determine the
optimal separation point. Detects and corrects inverted thresholds by checking that
background-similar pixels are correctly classified. Filters to keep only the largest
connected component, removing noise and disconnected regions.

Triangle thresholding is used to automatically determine the optimal threshold
for separating foreground from background based on the difference image histogram.

Masks are saved as [original_name]_mask.png in the same directory as the original images.

python3 SubtractionMask.py <folder> <background_image>

python3 SubtractionMask.py /Users/photolab/Downloads/PhotogrammetryMask/test_images/jpg /Users/photolab/Downloads/PhotogrammetryMask/test_images/jpg/DSC_0028.jpg
"""

import cv2
import os
import sys
import argparse
from pathlib import Path


# Supported image extensions
SUPPORTED_EXTENSIONS = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')


def get_image_files(folder_path):
    """
    Get all image files in the folder with supported extensions.
    
    Args:
        folder_path: Path object to the folder
        
    Returns:
        Sorted list of Path objects for image files
    """
    image_files = []
    for item in folder_path.iterdir():
        if item.is_file() and item.suffix in SUPPORTED_EXTENSIONS:
            image_files.append(item)
    return sorted(image_files)


def load_image_grayscale(image_path):
    """
    Load an image in grayscale format, converting to 8-bit if necessary.
    
    Args:
        image_path: Path to the image file
        
    Returns:
        Grayscale image as numpy array (8-bit single channel)
        
    Raises:
        ValueError: If image cannot be loaded
    """
    # Load with unchanged flag to detect bit depth
    img = cv2.imread(str(image_path), cv2.IMREAD_UNCHANGED)
    
    if img is None:
        raise ValueError(f"Could not load image: {image_path}")
    
    # Handle different bit depths
    if img.dtype == 'uint16':
        print(f"  Note: Converting 16-bit image to 8-bit: {image_path.name}")
        img = (img / 256).astype('uint8')
    
    # Convert to grayscale if needed
    if len(img.shape) == 3:
        if img.shape[2] == 4:
            img = cv2.cvtColor(img, cv2.COLOR_BGRA2GRAY)
        else:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    return img


def create_foreground_mask(image_gray, background_gray):
    """
    Create a binary mask based on absolute difference from background.
    
    Uses Triangle thresholding to automatically determine the optimal threshold
    for separating foreground from background. Automatically detects and corrects
    if the threshold direction is inverted.
    
    Args:
        image_gray: Grayscale image to process
        background_gray: Grayscale background reference image
        
    Returns:
        Binary mask (uint8) with 0=background, 255=foreground
    """
    # Compute absolute difference
    diff = cv2.absdiff(image_gray, background_gray)
    
    # Create mask using Triangle thresholding to automatically determine threshold
    # cv2.threshold returns (threshold_value, thresholded_image)
    _, mask = cv2.threshold(diff, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_TRIANGLE)
    
    # Check if threshold went the wrong direction
    # Pixels with low difference (similar to background) should be 0 in the mask
    low_diff_pixels = diff < 10
    if low_diff_pixels.any():
        avg_mask_for_low_diff = mask[low_diff_pixels].mean()
        if avg_mask_for_low_diff > 127:
            mask = cv2.bitwise_not(mask)
    
    return mask


def filter_largest_component(mask):
    """
    Filter the mask to keep only the largest connected component.
    
    Uses 8-connectivity (pixels touching including diagonals are considered connected).
    Only the largest contiguous foreground region is preserved; all other
    disconnected regions are removed (set to background).
    
    Args:
        mask: Binary mask (0=background, 255=foreground)
        
    Returns:
        Filtered binary mask with only the largest component
    """
    num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
        mask, connectivity=8, ltype=cv2.CV_32S
    )
    
    # If no foreground components (only background label 0), return original
    if num_labels <= 1:
        return mask
    
    # Find label with largest area (label 0 is background, skip it)
    # stats[:, cv2.CC_STAT_AREA] gives areas for all labels
    areas = stats[1:, cv2.CC_STAT_AREA]  # exclude background
    largest_label = 1 + areas.argmax()    # offset by 1 since we sliced from index 1
    
    # Create mask with only the largest component
    filtered_mask = (labels == largest_label).astype('uint8') * 255
    
    return filtered_mask


def main():
    """Main function to run background subtraction masking."""
    
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description='Automatic background subtraction masking for photogrammetry.',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python SubtractionMask.py ./images background.jpg
  python SubtractionMask.py /path/to/photos bg.png
  
Masks are saved as [original_name]_mask.png in the same directory as the originals.
        '''
    )
    parser.add_argument(
        'folder',
        type=str,
        help='Path to the folder containing all images'
    )
    parser.add_argument(
        'background',
        type=str,
        help='Filename of the background image (must be in the folder)'
    )
    
    args = parser.parse_args()
    
    folder_path = Path(args.folder).resolve()
    background_filename = args.background
    
    # Validate folder exists
    if not folder_path.is_dir():
        print(f"Error: Folder not found: {folder_path}")
        sys.exit(1)
    
    # Validate background image exists
    background_path = folder_path / background_filename
    if not background_path.is_file():
        print(f"Error: Background image not found: {background_path}")
        sys.exit(1)
    
    print("=" * 60)
    print("Background Subtraction Masking for Photogrammetry")
    print("=" * 60)
    print(f"Folder: {folder_path}")
    print(f"Background: {background_filename}")
    print(f"Threshold: Triangle thresholding (automatic)")
    print("=" * 60)
    
    # Load background image
    print(f"\nLoading background image: {background_filename}")
    try:
        background_gray = load_image_grayscale(background_path)
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    
    background_shape = background_gray.shape
    print(f"Background dimensions: {background_shape[1]}x{background_shape[0]} pixels")
    
    # Get all image files in folder
    all_images = get_image_files(folder_path)
    
    # Filter out the background image and any existing mask files
    images_to_process = [
        f for f in all_images 
        if f.name != background_filename and not f.stem.endswith('_mask')
    ]
    
    if not images_to_process:
        print("\nNo images found to process (excluding background image).")
        print("Make sure your images have .png, .jpg, or .jpeg extensions.")
        sys.exit(0)
    
    print(f"\nFound {len(images_to_process)} images to process.\n")
    print("-" * 60)
    
    # Track statistics
    processed_count = 0
    error_count = 0
    
    # Process each image
    for i, img_path in enumerate(images_to_process, 1):
        print(f"[{i}/{len(images_to_process)}] Processing: {img_path.name}")
        
        try:
            # Load grayscale version
            img_gray = load_image_grayscale(img_path)
            
            # Create foreground mask
            mask = create_foreground_mask(img_gray, background_gray)
            
            # Filter to keep only the largest connected component
            mask = filter_largest_component(mask)
            
            # Save mask as [original_name]_mask.png in the same directory
            mask_filename = f"{img_path.stem}_mask.png"
            output_path = img_path.parent / mask_filename
            
            # Save mask image
            success = cv2.imwrite(str(output_path), mask)
            
            if success:
                print(f"  -> Saved: {mask_filename}")
                processed_count += 1
            else:
                print(f"  -> Error: Failed to save {mask_filename}")
                error_count += 1
                
        except ValueError as e:
            print(f"  -> Error: {e}")
            error_count += 1
            continue
        except Exception as e:
            print(f"  -> Unexpected error: {e}")
            error_count += 1
            continue
    
    # Print summary
    print("-" * 60)
    print("\nProcessing complete!")
    print(f"  Successfully processed: {processed_count} images")
    if error_count > 0:
        print(f"  Errors: {error_count} images")
    print(f"  Masks saved alongside original images")
    print("=" * 60)


if __name__ == '__main__':
    main()