#!/usr/bin/env python3
"""
Photogrammetry Turntable Object Masking Script (Grayscale Version)
===================================================================

Automates masking of objects on a turntable for photogrammetry.
Supports white and black backdrops with controlled lighting.

MODIFIED: Uses grayscale comparisons instead of color to better detect
black lines and other intensity-based features on the turntable.

Logic:
1. For each image, compare it to all other images in the dataset (using grayscale)
2. Identify pixels that are STATIC (consistent across ALL images)
3. Remove static pixels - these are background elements, turntable marks, etc.
4. Keep pixels that CHANGE between images (the rotating object)
5. Apply intensity filtering to remove shadows and lighting artifacts

Key principle: Cross-image consistency is the primary test.
A pixel that appears in the same location in all images is background,
regardless of whether it matches the backdrop image.

Usage:
    python Mask.py <image_folder> <backdrop_image> <white|black>

Example:
    python Mask.py ./captures ./backdrop.jpg white
"""

import os
import sys
import argparse
import cv2
import numpy as np
from pathlib import Path


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
# CORE MASKING LOGIC
# =============================================================================

def create_difference_mask(image: np.ndarray, backdrop: np.ndarray, 
                           diff_threshold: int = 20) -> np.ndarray:
    """
    Create mask based on difference from backdrop using grayscale comparison.
    
    Pixels that differ significantly from the backdrop are considered
    part of the object. Pixels that are similar to the backdrop are
    considered background.
    
    Args:
        image: The object image (BGR)
        backdrop: The empty backdrop image (BGR)
        diff_threshold: Minimum difference to consider a pixel as "different"
                       Lower = more sensitive, Higher = less sensitive
    
    Returns:
        Binary mask where 255 = object (different from backdrop), 0 = background
    """
    # Convert both images to grayscale for comparison
    image_gray = bgr_to_gray(image)
    backdrop_gray = bgr_to_gray(backdrop)
    
    # Compute absolute difference in grayscale
    diff = cv2.absdiff(image_gray, backdrop_gray)
    
    # Threshold: pixels with difference > threshold are the object
    _, mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
    
    return mask.astype(np.uint8)


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
    Find the object boundary from difference mask and fill it completely.
    
    This creates a solid mask where ALL interior pixels are included,
    protecting white/black spots inside the object from being filtered out.
    
    Args:
        mask: Binary difference mask
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


def detect_static_pixels(image: np.ndarray, backdrop: np.ndarray, 
                        dataset_images: dict, current_image_path: str,
                        diff_threshold: int = 20) -> np.ndarray:
    """
    Detect pixels that are STATIC (unchanging) across all images using grayscale comparison.
    
    NEW LOGIC: A pixel is static if it's consistent across ALL dataset images,
    regardless of whether it matches the backdrop. The backdrop is used as a
    helpful reference but NOT as a hard requirement.
    
    This identifies background elements, turntable marks, and any artifacts that
    don't change as the object rotates.
    
    MODIFIED: Uses grayscale comparisons to better detect intensity-based features
    like black lines on the turntable.
    
    Args:
        image: Current image being processed (BGR)
        backdrop: Empty backdrop reference (BGR)
        dataset_images: Dictionary mapping image paths to BGR arrays
        current_image_path: Path to current image (to skip self-comparison)
        diff_threshold: Maximum difference to consider pixels "the same"
    
    Returns:
        Binary mask where 255 = static (remove), 0 = changing (keep)
    """
    h, w = image.shape[:2]
    
    # Convert current image to grayscale once
    image_gray = bgr_to_gray(image)
    
    # Start by assuming all pixels are static
    # We'll mark pixels as non-static if they differ between images
    static_mask = np.ones((h, w), dtype=bool)
    
    # PRIMARY TEST: Compare to all other images in the dataset (using grayscale)
    # If a pixel differs in ANY other image, it's not static (it's the rotating object)
    current_key = str(Path(current_image_path).resolve())
    
    for img_path, other_image in dataset_images.items():
        # Skip self-comparison
        if img_path == current_key:
            continue
        
        # Resize if necessary
        other_resized = other_image if other_image.shape[:2] == image.shape[:2] else \
                       cv2.resize(other_image, (w, h))
        
        # Convert to grayscale for comparison
        other_gray = bgr_to_gray(other_resized)
        
        # Compute grayscale difference
        other_diff = cv2.absdiff(image_gray, other_gray)
        differs_from_other = other_diff >= diff_threshold
        
        # If pixel differs in this image, it's NOT static
        static_mask &= ~differs_from_other
    
    # OPTIONAL REFINEMENT: Use backdrop as a helpful reference
    # If a pixel is static across all images AND matches the backdrop,
    # we have extra confidence it's background. But if it's static across
    # all images and doesn't match backdrop (like black stage lines on white
    # backdrop), we still remove it because cross-image consistency is primary.
    
    # Note: We're NOT using backdrop as a filter anymore - static_mask already
    # contains pixels that are consistent across all dataset images, which is
    # what we want to remove regardless of backdrop match.
    
    # Convert to uint8 mask (255 = static/remove, 0 = changing/keep)
    return (static_mask * 255).astype(np.uint8)


def smooth_edges(mask: np.ndarray, blur_size: int = 5) -> np.ndarray:
    """
    Smooth mask edges for cleaner results.
    """
    if blur_size % 2 == 0:
        blur_size += 1
    
    smoothed = cv2.GaussianBlur(mask, (blur_size, blur_size), 0)
    _, smoothed = cv2.threshold(smoothed, 127, 255, cv2.THRESH_BINARY)
    
    return smoothed


def apply_mask(image: np.ndarray, mask: np.ndarray) -> np.ndarray:
    """
    Apply mask to image, creating RGBA output with transparency.
    
    White (255) in mask = keep pixel (opaque)
    Black (0) in mask = remove pixel (transparent)
    """
    # Ensure mask is binary
    _, binary_mask = cv2.threshold(mask, 127, 255, cv2.THRESH_BINARY)
    
    # Create BGRA image with alpha channel
    bgra = cv2.cvtColor(image, cv2.COLOR_BGR2BGRA)
    bgra[:, :, 3] = binary_mask
    
    return bgra


# =============================================================================
# MAIN PROCESSING
# =============================================================================

def process_image(image_path: str, backdrop: np.ndarray, backdrop_color: str,
                  dataset_images: dict | None = None, diff_threshold: int = 20) -> np.ndarray:
    """
    Process a single image using cross-image consistency detection with grayscale.
    
    Pipeline:
    1. Detect static pixels by comparing current image to ALL other images (grayscale)
    2. Create initial mask keeping only changing (non-static) pixels
    3. Apply intensity filtering to remove shadows/lighting artifacts (grayscale)
    4. Refine mask with morphological operations
    5. Smooth edges
    6. Apply mask to original image
    
    Note: Cross-image consistency is the PRIMARY test. The backdrop is used
    for intensity filtering but not as a hard constraint for pixel removal.
    
    Args:
        image_path: Path to image to process
        backdrop: Empty backdrop reference image
        backdrop_color: "white" or "black"
        dataset_images: Dict of all dataset images {path: BGR_array}
        diff_threshold: Threshold for detecting differences
    
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
    # STEP 1: Detect static pixels across backdrop and all images (grayscale)
    # =========================================================================
    if dataset_images and len(dataset_images) > 1:
        print("(multi-image)", end=" ")
        static_mask = detect_static_pixels(
            image, backdrop_resized, dataset_images, 
            image_path, diff_threshold
        )
        # Invert: we want to KEEP non-static pixels (the object)
        # static_mask has 255 for static, 0 for changing
        # We want 255 for object (changing), 0 for background (static)
        object_mask = cv2.bitwise_not(static_mask)
    else:
        # Fallback: simple backdrop difference if no dataset provided
        print("(backdrop-only)", end=" ")
        image_gray = bgr_to_gray(image)
        backdrop_gray = bgr_to_gray(backdrop_resized)
        diff = cv2.absdiff(image_gray, backdrop_gray)
        _, object_mask = cv2.threshold(diff, diff_threshold, 255, cv2.THRESH_BINARY)
        object_mask = object_mask.astype(np.uint8)
    
    # =========================================================================
    # STEP 2: Apply intensity filtering to remove shadows/lighting artifacts
    # =========================================================================
    # This removes shadow regions and lighting artifacts that might be
    # classified as "changing" but are actually just lighting effects
    # NOW USING GRAYSCALE VALUES
    
    # Convert image to grayscale for intensity filtering
    image_gray = bgr_to_gray(image)
    
    if backdrop_color == "white":
        # Remove near-white pixels (likely shadows or backdrop remnants)
        near_white = image_gray > 235
        
        # Create intensity filter mask (255 = remove, 0 = keep)
        intensity_filter = (near_white * 255).astype(np.uint8)
        
    else:  # black backdrop
        # Remove near-black pixels
        near_black = image_gray < 20
        
        intensity_filter = (near_black * 255).astype(np.uint8)
    
    # Apply intensity filter: remove pixels that match the filter
    # object_mask currently has 255 for object, 0 for background
    # intensity_filter has 255 for pixels to remove
    # We want to set object_mask to 0 where intensity_filter is 255
    object_mask[intensity_filter == 255] = 0
    
    # =========================================================================
    # STEP 3: Refine mask with morphological operations
    # =========================================================================
    # Find and fill the main object boundary to ensure solid interior
    object_boundary = find_and_fill_object(object_mask, min_area_ratio=0.001)
    
    # Clean up the mask
    refined_mask = clean_mask(object_boundary, min_area_ratio=0.001)
    
    # =========================================================================
    # STEP 4: Smooth edges
    # =========================================================================
    final_mask = smooth_edges(refined_mask, blur_size=5)
    
    # =========================================================================
    # STEP 5: Apply mask to original image
    # =========================================================================
    masked_image = apply_mask(image, final_mask)
    
    return masked_image


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Mask objects on turntable for photogrammetry (Grayscale version)",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python Mask.py ./captures ./backdrop.jpg white
  python Mask.py ./photos ./empty_bg.png black

The script uses cross-image consistency detection with grayscale comparisons:
  1. Compares each image to ALL other images in the dataset (in grayscale)
  2. Removes pixels that are STATIC (identical across all images)
  3. Keeps pixels that CHANGE (the rotating object)
  4. Applies intensity filtering to remove shadows and lighting artifacts
  
Key principle: If a pixel is in the same location in all images,
it's background/turntable/artifacts - removed regardless of backdrop match.
This ensures stage lines and marks are properly filtered out.

MODIFIED: Uses grayscale comparisons instead of color to better detect
intensity-based features like black lines on white turntables.
        """
    )
    parser.add_argument("image_folder", help="Path to folder containing object images")
    parser.add_argument("backdrop_image", help="Path to empty backdrop reference image")
    parser.add_argument("backdrop_color", choices=["white", "black"],
                        help="Backdrop color: 'white' or 'black'")
    parser.add_argument("-o", "--output", help="Output folder (default: <image_folder>_masked)")
    parser.add_argument("-t", "--threshold", type=int, default=20,
                        help="Difference threshold (default: 20). Lower=more sensitive")
    
    args = parser.parse_args()
    
    # Validate paths
    image_folder = Path(args.image_folder)
    backdrop_path = Path(args.backdrop_image)
    
    if not image_folder.is_dir():
        print(f"Error: Image folder not found: {image_folder}")
        sys.exit(1)
    
    if not backdrop_path.is_file():
        print(f"Error: Backdrop image not found: {backdrop_path}")
        sys.exit(1)
    
    # Setup output folder
    if args.output:
        output_folder = Path(args.output)
    else:
        output_folder = image_folder.parent / f"{image_folder.name}_masked"
    
    output_folder.mkdir(parents=True, exist_ok=True)
    
    # Load backdrop image
    print(f"Loading backdrop: {backdrop_path.name}")
    backdrop = load_image(str(backdrop_path))
    print(f"  Size: {backdrop.shape[1]}x{backdrop.shape[0]}")
    
    # Get list of images (excluding backdrop)
    all_images = get_image_files(image_folder)
    image_files = [f for f in all_images if f.resolve() != backdrop_path.resolve()]

    # Preload dataset images into memory (mapping resolved_path -> numpy array)
    # This is used so we can compare each image to every other image in the set
    print("Loading dataset images into memory...")
    dataset_images = {}
    for f in image_files:
        try:
            dataset_images[str(f.resolve())] = load_image(str(f))
        except Exception as e:
            print(f"Warning: failed to load {f.name}: {e}")
    
    if not image_files:
        print("Error: No object images to process")
        sys.exit(1)
    
    print(f"\nProcessing {len(image_files)} images...")
    print(f"  Mode: {args.backdrop_color} backdrop")
    print(f"  Difference threshold: {args.threshold}")
    print(f"  Comparison method: GRAYSCALE (intensity-based)")
    print("-" * 50)
    
    # Process each image
    successful = 0
    failed = 0
    
    for i, image_path in enumerate(image_files, 1):
        print(f"[{i}/{len(image_files)}] {image_path.name}", end=" ")
        
        try:
            masked_image = process_image(
                str(image_path),
                backdrop,
                args.backdrop_color,
                dataset_images=dataset_images,
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
    print("-" * 50)
    print(f"Complete: {successful}/{len(image_files)} successful")
    if failed > 0:
        print(f"Failed: {failed}")
    print(f"Output: {output_folder}")


if __name__ == "__main__":
    main()