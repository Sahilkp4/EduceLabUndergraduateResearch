#!/usr/bin/env python3
"""
Photogrammetry Turntable Object Masking Script (Variance-Based Version)
========================================================================

Automates masking of objects on a turntable for photogrammetry.
Supports white and black backdrops with controlled lighting.

MODIFIED: Uses variance-based detection to identify the object by analyzing
which pixels CHANGE THE MOST across all images in the dataset.

Logic:
1. For each pixel position, calculate how much it varies across ALL images
2. Use standard deviation (or other variance metrics) as the change magnitude
3. Keep only pixels in the top X percentile of change magnitude
4. Apply backdrop filtering as a secondary filter for static elements
5. Apply intensity filtering to remove shadows and lighting artifacts

Key principle: Pixels that vary significantly are part of the rotating object.
Pixels with low variance (wobbling lines, subtle shadows, static elements) are removed.

Usage:
    python Mask.py <image_folder> <backdrop_image> <white|black> [options]

Example:
    python Mask.py ./captures ./backdrop.jpg white --variance-percentile 40
    python Mask.py ./captures ./backdrop.jpg black --variance-method range --variance-percentile 60
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List


# Supported image extensions
SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.tif', '.tiff', '.bmp'}


def load_image(path: str) -> np.ndarray:
    """Load an image from disk as BGR."""
    img = cv2.imread(path, cv2.IMREAD_COLOR)
    if img is None:
        raise FileNotFoundError(f"Could not load image: {path}")
    return img


def bgr_to_gray(image: np.ndarray) -> np.ndarray:
    """Convert BGR image to grayscale."""
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def get_image_files(folder: Path) -> list:
    """Get sorted list of supported image files in a folder."""
    return sorted([
        f for f in folder.iterdir()
        if f.is_file() and f.suffix.lower() in SUPPORTED_EXTENSIONS
    ])


# =============================================================================
# VARIANCE-BASED DETECTION LOGIC
# =============================================================================

def compute_pixel_variance(grayscale_images: List[np.ndarray], 
                          method: str = 'std') -> np.ndarray:
    """
    Compute variance map showing how much each pixel changes across all images.
    
    This is the core of the variance-based detection approach. For each pixel
    position (x,y), we collect all its grayscale values across the dataset and
    compute a measure of how much it varies.
    
    Args:
        grayscale_images: List of grayscale images (all same dimensions)
        method: Variance calculation method
            - 'std': Standard deviation (default, most common)
            - 'range': Max - Min value (simple, interpretable)
            - 'mad': Median Absolute Deviation (robust to outliers)
    
    Returns:
        2D array (h x w) where each value represents the change magnitude
        for that pixel position. Higher values = more change = likely object.
    """
    if not grayscale_images:
        raise ValueError("No images provided")
    
    # Stack all images into 3D array: (num_images, height, width)
    image_stack = np.stack(grayscale_images, axis=0).astype(np.float32)
    
    if method == 'std':
        # Standard deviation across images for each pixel
        variance_map = np.std(image_stack, axis=0)
        
    elif method == 'range':
        # Range: max - min across images for each pixel
        max_vals = np.max(image_stack, axis=0)
        min_vals = np.min(image_stack, axis=0)
        variance_map = max_vals - min_vals
        
    elif method == 'mad':
        # Median Absolute Deviation (more robust to outliers)
        median_vals = np.median(image_stack, axis=0)
        # Absolute deviations from median
        abs_deviations = np.abs(image_stack - median_vals[np.newaxis, :, :])
        # MAD is the median of these absolute deviations
        variance_map = np.median(abs_deviations, axis=0)
        
    else:
        raise ValueError(f"Unknown variance method: {method}")
    
    return variance_map


def detect_high_variance_pixels(grayscale_images: List[np.ndarray],
                               percentile: float = 50.0,
                               method: str = 'std') -> np.ndarray:
    """
    Detect pixels with high variance across images using percentile threshold.
    
    This replaces the old static/non-static binary logic with a more nuanced
    approach that filters out low-variance artifacts like wobbling lines,
    shifting shadows, and subtle lighting changes.
    
    Args:
        grayscale_images: List of grayscale images from the dataset
        percentile: Percentile threshold (0-100). Keep pixels above this percentile.
                   50 = keep top 50% of variance, 75 = keep top 25%, etc.
        method: Variance calculation method ('std', 'range', or 'mad')
    
    Returns:
        Binary mask: 255 = high variance (likely object), 0 = low variance (background)
    """
    # Compute variance map
    variance_map = compute_pixel_variance(grayscale_images, method=method)
    
    # Calculate threshold based on percentile
    # Higher percentile = higher threshold = fewer pixels kept
    threshold_value = np.percentile(variance_map, percentile)
    
    # Create binary mask: pixels above threshold are considered object
    high_variance_mask = (variance_map > threshold_value).astype(np.uint8) * 255
    
    return high_variance_mask


def apply_backdrop_filter(mask: np.ndarray, image: np.ndarray, backdrop: np.ndarray,
                         diff_threshold: int = 20) -> np.ndarray:
    """
    Apply backdrop filtering as a secondary filter to remove static elements.
    
    This catches any pixels that made it through variance filtering but are
    actually just static background elements very similar to the backdrop.
    
    Args:
        mask: Current object mask (255 = object, 0 = background)
        image: The current image (BGR)
        backdrop: The backdrop reference image (BGR)
        diff_threshold: Minimum difference from backdrop to keep pixel
    
    Returns:
        Filtered mask with backdrop-similar pixels removed
    """
    # Convert to grayscale
    image_gray = bgr_to_gray(image)
    backdrop_gray = bgr_to_gray(backdrop)
    
    # Compute difference from backdrop
    diff = cv2.absdiff(image_gray, backdrop_gray)
    
    # Pixels too similar to backdrop should be removed
    # diff <= threshold means similar to backdrop
    similar_to_backdrop = diff <= diff_threshold
    
    # Remove these pixels from the mask
    filtered_mask = mask.copy()
    filtered_mask[similar_to_backdrop] = 0
    
    return filtered_mask


# =============================================================================
# MASK REFINEMENT AND CLEANUP
# =============================================================================

def clean_mask(mask: np.ndarray, min_area_ratio: float = 0.001) -> np.ndarray:
    """
    Clean up the mask using morphological operations and contour filtering.
    
    1. Remove small noise with opening
    2. Fill holes with closing
    3. Find external contours and fill them (ensures solid object)
    4. Filter out tiny contours
    
    Args:
        mask: Binary mask to clean
        min_area_ratio: Minimum contour area as fraction of image area
    
    Returns:
        Cleaned binary mask
    """
    h, w = mask.shape[:2]
    image_area = h * w
    
    # Morphological kernels
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    
    # Opening: remove small noise specks
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel_small, iterations=2)
    
    # Closing: fill small holes inside the object
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_CLOSE, kernel_medium, iterations=2)
    
    # Find external contours only
    contours, _ = cv2.findContours(cleaned, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by minimum area
    min_area = image_area * min_area_ratio
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    # If no valid contours but we had some, use the largest
    if not valid_contours and contours:
        valid_contours = [max(contours, key=cv2.contourArea)]
    
    # Create filled mask from contours
    result = np.zeros((h, w), dtype=np.uint8)
    if valid_contours:
        cv2.drawContours(result, valid_contours, -1, 255, thickness=cv2.FILLED)
    
    # Final closing to ensure solid interior
    result = cv2.morphologyEx(result, cv2.MORPH_CLOSE, kernel_medium, iterations=1)
    
    return result


def find_and_fill_object(mask: np.ndarray, min_area_ratio: float = 0.001) -> np.ndarray:
    """
    Find the object boundary from mask and fill it completely.
    
    This creates a solid mask where ALL interior pixels are included,
    protecting white/black spots inside the object from being filtered out.
    
    Args:
        mask: Binary mask
        min_area_ratio: Minimum contour area as fraction of image area
    
    Returns:
        Filled binary mask (solid object interior)
    """
    h, w = mask.shape[:2]
    image_area = h * w
    
    # Morphological kernels
    kernel_small = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    kernel_medium = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
    kernel_large = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (11, 11))
    
    # Light cleanup to connect nearby regions
    cleaned = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel_small, iterations=2)
    cleaned = cv2.morphologyEx(cleaned, cv2.MORPH_OPEN, kernel_small, iterations=1)
    
    # Dilate slightly to ensure we capture the full object boundary
    dilated = cv2.dilate(cleaned, kernel_small, iterations=2)
    
    # Find external contours
    contours, _ = cv2.findContours(dilated, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filter by minimum area
    min_area = image_area * min_area_ratio
    valid_contours = [c for c in contours if cv2.contourArea(c) > min_area]
    
    if not valid_contours and contours:
        valid_contours = [max(contours, key=cv2.contourArea)]
    
    # Create FILLED mask - this is the key: completely fill the interior
    filled = np.zeros((h, w), dtype=np.uint8)
    if valid_contours:
        cv2.drawContours(filled, valid_contours, -1, 255, thickness=cv2.FILLED)
    
    # Close any remaining holes
    filled = cv2.morphologyEx(filled, cv2.MORPH_CLOSE, kernel_large, iterations=2)
    
    return filled


def smooth_edges(mask: np.ndarray, blur_size: int = 5) -> np.ndarray:
    """
    Smooth the edges of the mask using Gaussian blur.
    
    This creates softer, more natural-looking edges for the mask.
    
    Args:
        mask: Binary mask
        blur_size: Size of Gaussian kernel (must be odd)
    
    Returns:
        Smoothed mask with softer edges
    """
    if blur_size % 2 == 0:
        blur_size += 1  # Ensure odd number
    
    # Apply Gaussian blur
    blurred = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    
    # Re-threshold to maintain binary nature but with smoother edges
    _, smoothed = cv2.threshold(blurred, 127, 255, cv2.THRESH_BINARY)
    
    return smoothed.astype(np.uint8)


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply mask to image to create RGBA output with transparency.
    
    Args:
        image: Original BGR image
        mask: Binary mask (255 = keep, 0 = transparent)
    
    Returns:
        BGRA image with alpha channel from mask
    """
    # Convert BGR to BGRA
    bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    
    # Set alpha channel to mask
    bgra[:, :, 3] = mask
    
    return bgra


# =============================================================================
# MAIN PROCESSING PIPELINE
# =============================================================================

def process_image(image_path: str, backdrop: np.ndarray, backdrop_color: str,
                 dataset_grayscale_images: List[np.ndarray],
                 variance_percentile: float = 50.0,
                 variance_method: str = 'std',
                 diff_threshold: int = 20) -> np.ndarray:
    """
    Process a single image using variance-based detection.
    
    Pipeline:
    1. Detect high-variance pixels across all dataset images
    2. Apply backdrop filtering to remove static elements
    3. Apply intensity filtering to remove shadows/lighting artifacts
    4. Refine mask with morphological operations
    5. Smooth edges and apply to original image
    
    Args:
        image_path: Path to image to process
        backdrop: Empty backdrop reference image (BGR)
        backdrop_color: 'white' or 'black'
        dataset_grayscale_images: List of ALL dataset images in grayscale
        variance_percentile: Percentile threshold for variance filtering
        variance_method: Variance calculation method ('std', 'range', 'mad')
        diff_threshold: Threshold for backdrop filtering
    
    Returns:
        BGRA image with alpha mask applied
    """
    # Load original image
    image = load_image(image_path)
    h, w = image.shape[:2]
    
    # Resize backdrop if dimensions don't match
    if image.shape[:2] != backdrop.shape[:2]:
        backdrop_resized = cv2.resize(backdrop, (w, h))
    else:
        backdrop_resized = backdrop
    
    # =========================================================================
    # STEP 1: Detect high-variance pixels across all images
    # =========================================================================
    if dataset_grayscale_images and len(dataset_grayscale_images) > 1:
        print(f"(variance-based, {variance_method}, {variance_percentile}%)", end=" ")
        object_mask = detect_high_variance_pixels(
            dataset_grayscale_images,
            percentile=variance_percentile,
            method=variance_method
        )
    else:
        # Fallback: simple backdrop difference if no dataset provided
        print("(backdrop-only)", end=" ")
        image_gray = bgr_to_gray(image)
        backdrop_gray = bgr_to_gray(backdrop_resized)
        diff = cv2.absdiff(image_gray, backdrop_gray)
        _, object_mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
        object_mask = object_mask.astype(np.uint8)
    
    # =========================================================================
    # STEP 2: Apply backdrop filtering (secondary filter)
    # =========================================================================
    # This catches static elements that somehow got through variance filtering
    object_mask = apply_backdrop_filter(
        object_mask, image, backdrop_resized, diff_threshold
    )
    
    # =========================================================================
    # STEP 3: Apply intensity filtering to remove shadows/lighting artifacts
    # =========================================================================
    # This removes shadow regions and lighting artifacts that might have
    # high variance but are actually just lighting effects
    
    image_gray = bgr_to_gray(image)
    
    if backdrop_color == "white":
        # Remove near-white pixels (likely shadows or backdrop remnants)
        near_white = image_gray > 235
        object_mask[near_white] = 0
        
    else:  # black backdrop
        # Remove near-black pixels
        near_black = image_gray < 20
        object_mask[near_black] = 0
    
    # =========================================================================
    # STEP 4: Refine mask with morphological operations
    # =========================================================================
    # Find and fill the main object boundary to ensure solid interior
    object_boundary = find_and_fill_object(object_mask, min_area_ratio=0.001)
    
    # Clean up the mask
    refined_mask = clean_mask(object_boundary, min_area_ratio=0.001)
    
    # =========================================================================
    # STEP 5: Smooth edges
    # =========================================================================
    final_mask = smooth_edges(refined_mask, blur_size=5)
    
    # =========================================================================
    # STEP 6: Apply mask to original image
    # =========================================================================
    masked_image = apply_mask(image, final_mask)
    
    return masked_image


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Mask objects on turntable using variance-based detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Mask.py ./captures ./backdrop.jpg white
  python Mask.py ./captures ./backdrop.jpg white --variance-percentile 40
  python Mask.py ./photos ./bg.png black --variance-method range --variance-percentile 60

The script uses variance-based detection to identify the object:
  1. Calculates how much each pixel varies across ALL images in the dataset
  2. Keeps only pixels with high variance (top X percentile)
  3. Applies backdrop filtering to remove static elements
  4. Applies intensity filtering to remove shadows and lighting artifacts
  
Key principle: Pixels that change significantly are part of the rotating object.
Pixels with low variance (wobbling lines, shadows, static artifacts) are removed.

Variance Methods:
  - 'std': Standard deviation (default, most common)
  - 'range': Max - Min value (simple, interpretable)
  - 'mad': Median Absolute Deviation (robust to outliers)

Percentile Guide:
  - 50 (default): Keep top 50% of variance (balanced)
  - 40: Keep top 60% (more inclusive, good for complex objects)
  - 60: Keep top 40% (more selective, filters more noise)
  - 70: Keep top 30% (very selective, use for clean captures)
        """
    )
    parser.add_argument("image_folder", help="Path to folder containing object images")
    parser.add_argument("backdrop_image", help="Path to empty backdrop reference image")
    parser.add_argument("backdrop_color", choices=["white", "black"],
                        help="Backdrop color: 'white' or 'black'")
    parser.add_argument("-o", "--output", help="Output folder (default: <image_folder>_masked)")
    parser.add_argument("-t", "--threshold", type=int, default=20,
                        help="Backdrop difference threshold (default: 20). Lower=more sensitive")
    parser.add_argument("--variance-percentile", type=float, default=50.0,
                        help="Variance percentile threshold (0-100, default: 50). "
                             "Higher value = more selective = fewer pixels kept. "
                             "E.g., 70 keeps top 30%% of variance.")
    parser.add_argument("--variance-method", choices=['std', 'range', 'mad'], default='std',
                        help="Variance calculation method (default: std). "
                             "std=standard deviation, range=max-min, mad=median absolute deviation")
    
    args = parser.parse_args()
    
    # Print header
    print("\n" + "="*70)
    print("PHOTOGRAMMETRY VARIANCE-BASED MASKING SCRIPT")
    print("="*70)
    print("This script identifies objects by analyzing pixel variance across")
    print("all images in your dataset, filtering out static artifacts.")
    print("="*70)
    
    # Validate inputs
    if not 0 <= args.variance_percentile <= 100:
        print("\nERROR: --variance-percentile must be between 0 and 100")
        print(f"You provided: {args.variance_percentile}")
        sys.exit(1)
    
    # Validate paths
    image_folder = Path(args.image_folder)
    backdrop_path = Path(args.backdrop_image)
    
    if not image_folder.exists():
        print(f"\nERROR: Image folder does not exist")
        print(f"Path: {image_folder}")
        print(f"Absolute path: {image_folder.resolve()}")
        sys.exit(1)
    
    if not image_folder.is_dir():
        print(f"\nERROR: Image folder path is not a directory")
        print(f"Path: {image_folder}")
        sys.exit(1)
    
    if not backdrop_path.exists():
        print(f"\nERROR: Backdrop image does not exist")
        print(f"Path: {backdrop_path}")
        print(f"Absolute path: {backdrop_path.resolve()}")
        sys.exit(1)
    
    if not backdrop_path.is_file():
        print(f"\nERROR: Backdrop path is not a file")
        print(f"Path: {backdrop_path}")
        sys.exit(1)
    
    # Setup output folder
    if args.output:
        output_folder = Path(args.output)
    else:
        output_folder = image_folder.parent / f"{image_folder.name}_masked"
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Load backdrop image
    print(f"\n{'='*70}")
    print(f"SETUP")
    print(f"{'='*70}")
    print(f"Loading backdrop: {backdrop_path.name}")
    backdrop = load_image(str(backdrop_path))
    print(f"  Path: {backdrop_path}")
    print(f"  Size: {backdrop.shape[1]}x{backdrop.shape[0]}")
    
    # Get list of ALL images in folder
    all_images = get_image_files(image_folder)
    print(f"\nFound {len(all_images)} total image files in folder")
    
    # Filter out the backdrop image from the dataset
    # Use absolute paths for comparison to handle backdrop in same folder
    backdrop_abs = str(backdrop_path.resolve())
    image_files = []
    
    for img_file in all_images:
        img_abs = str(img_file.resolve())
        if img_abs == backdrop_abs:
            print(f"  - Skipping backdrop: {img_file.name}")
        else:
            image_files.append(img_file)
    
    print(f"\nDataset contains {len(image_files)} images (excluding backdrop)")
    
    if not image_files:
        print("\nERROR: No object images to process!")
        print("Make sure:")
        print("  1. Your image folder contains multiple images")
        print("  2. The backdrop image path is correct")
        print("  3. You have images other than the backdrop")
        sys.exit(1)
    
    if len(image_files) < 2:
        print(f"\nWARNING: Only {len(image_files)} image(s) found.")
        print("Variance-based detection works best with 10+ images.")
        print("With fewer images, results may not be optimal.\n")
    
    # Preload ALL dataset images as grayscale for variance calculation
    print(f"\n{'='*70}")
    print(f"LOADING IMAGES")
    print(f"{'='*70}")
    print(f"Loading {len(image_files)} images as grayscale for variance analysis...")
    
    dataset_grayscale_images = []
    image_files_valid = []
    failed_loads = []
    
    for i, f in enumerate(image_files, 1):
        try:
            print(f"  [{i}/{len(image_files)}] Loading {f.name}...", end=" ")
            img = load_image(str(f))
            gray = bgr_to_gray(img)
            dataset_grayscale_images.append(gray)
            image_files_valid.append(f)
            print("✓")
        except Exception as e:
            print(f"✗ FAILED: {e}")
            failed_loads.append((f.name, str(e)))
    
    if failed_loads:
        print(f"\nWarning: Failed to load {len(failed_loads)} image(s):")
        for name, error in failed_loads:
            print(f"  - {name}: {error}")
    
    if len(dataset_grayscale_images) == 0:
        print("\nERROR: Could not load any images!")
        print("Check that:")
        print("  1. Image files are valid (not corrupted)")
        print("  2. File extensions are supported (.jpg, .png, .tif, .bmp)")
        print("  3. You have read permissions on the files")
        sys.exit(1)
    
    print(f"\nSuccessfully loaded {len(dataset_grayscale_images)} images")
    
    # Verify all images have same dimensions
    first_shape = dataset_grayscale_images[0].shape
    print(f"  Image dimensions: {first_shape[1]}x{first_shape[0]} pixels")
    
    mismatched = []
    for i, (img, f) in enumerate(zip(dataset_grayscale_images, image_files_valid)):
        if img.shape != first_shape:
            mismatched.append((f.name, img.shape))
    
    if mismatched:
        print("\nERROR: Not all images have the same dimensions!")
        print(f"Expected: {first_shape[1]}x{first_shape[0]}")
        print("Mismatched images:")
        for name, shape in mismatched:
            print(f"  - {name}: {shape[1]}x{shape[0]}")
        print("\nAll images must be the same size for variance analysis.")
        print("Please resize your images to match.")
        sys.exit(1)
    
    print(f"\n{'='*70}")
    print(f"PROCESSING")
    print(f"{'='*70}")
    print(f"Processing {len(image_files_valid)} images...")
    print(f"  Mode: {args.backdrop_color} backdrop")
    print(f"  Variance method: {args.variance_method}")
    print(f"  Variance percentile: {args.variance_percentile} (keep top {100-args.variance_percentile:.1f}%)")
    print(f"  Backdrop threshold: {args.threshold}")
    print(f"{'='*70}")
    
    # Process each image
    successful = 0
    failed = 0
    
    for i, image_path in enumerate(image_files_valid, 1):
        print(f"[{i}/{len(image_files_valid)}] {image_path.name}", end=" ")
        
        try:
            masked_image = process_image(
                str(image_path),
                backdrop,
                args.backdrop_color,
                dataset_grayscale_images=dataset_grayscale_images,
                variance_percentile=args.variance_percentile,
                variance_method=args.variance_method,
                diff_threshold=args.threshold
            )
            
            # Save as PNG to preserve transparency
            output_filename = image_path.stem + ".png"
            output_path = output_folder / output_filename
            cv2.imwrite(str(output_path), masked_image)
            
            print("✓")
            successful += 1
            
        except Exception as e:
            print(f"✗ Error: {e}")
            failed += 1
    
    # Print summary
    print(f"{'='*70}")
    print(f"COMPLETE")
    print(f"{'='*70}")
    print(f"Successfully processed: {successful}/{len(image_files_valid)} images")
    if failed > 0:
        print(f"Failed: {failed}")
    print(f"\nOutput directory: {output_folder}")
    print(f"Output format: PNG with transparency (alpha channel)")
    print(f"\nNext steps:")
    print(f"  1. Review masks in: {output_folder}")
    print(f"  2. If quality issues, adjust --variance-percentile")
    print(f"  3. Import masked images to photogrammetry software")
    print(f"{'='*70}")


if __name__ == "__main__":
    main()