#!/usr/bin/env python3
"""
GrabCut Interactive Masking for Photogrammetry

This script performs interactive object segmentation using OpenCV's GrabCut algorithm
to create mask images suitable for photogrammetry processing.

The user paints foreground regions on a few sample images, corrects the GrabCut
results with green (foreground) and red (background) strokes, then the learned
segmentation model is propagated to all images in the sequence.

Workflow:
1. Select w evenly-spaced sample images from the sequence
2. For each sample:
   a. User paints GREEN strokes on guaranteed foreground areas
   b. GrabCut generates an initial mask from the painted regions
   c. User corrects errors: GREEN on missed foreground, RED on false foreground
   d. GrabCut re-trains with corrections; user accepts or repeats corrections
3. Models are propagated to all images (parallel processing by default)
4. Masks are expanded to ensure object coverage
5. User reviews results and can accept or retry with more samples

Training uses downscaled images for speed; propagation runs at full resolution.

Configuration:
All parameters can be set via config.yaml (placed next to the script) or CLI args.
Precedence: built-in defaults < config.yaml < CLI arguments.
Use --config PATH to specify an alternate config file.
Requires PyYAML for config file support (pip install pyyaml).
The script works without PyYAML, falling back to built-in defaults.

Masks are saved as [original_name]_mask.png in the same directory as the original images.

Usage:
    python3 GrabcutExtraction.py <folder>
    python3 GrabcutExtraction.py <folder> --samples 5 --train-iterations 7
    python3 GrabcutExtraction.py <folder> --workers 4
    python3 GrabcutExtraction.py <folder> --no-parallel
    python3 GrabcutExtraction.py <folder> --training-max-dim 800
    python3 GrabcutExtraction.py <folder> --config my_config.yaml
"""

import cv2
import numpy as np
import sys
import argparse
import os
import time
import tkinter as tk
from tkinter import scrolledtext
from pathlib import Path
from typing import List, Tuple, Optional
from concurrent.futures import ProcessPoolExecutor, as_completed

try:
    from PIL import Image as PILImage, ImageTk
except ImportError:
    print("Error: Pillow is required for the UI.")
    print("       Install with: pip install pillow")
    sys.exit(1)


def get_default_config() -> dict:
    """Return the complete default configuration dictionary (single source of truth)."""
    return {
        "initial_samples": 3,
        "include_first_image": True,
        "train_iterations": 5,
        "propagate_iterations": 3,
        "max_retries": 5,
        "expansion_pixels": 3,
        "mask_cleanup": True,
        "mask_cleanup_kernel_size": 5,
        "mask_fill_holes": True,
        "template_margin": 0.1,
        "min_template_size": 50,
        "use_parallel": True,
        "worker_count": None,
        "large_template_threshold": 2_000_000,
        "very_large_template_threshold": 4_000_000,
        "large_template_max_workers": 5,
        "very_large_template_max_workers": 3,
        "selector_window_width": 1200,
        "selector_window_height": 800,
        "review_window_width": 1800,
        "review_window_height": 600,
        "review_max_display_height": 800,
        "retry_sample_increment": 2,
        "retry_train_increment": 1,
        "retry_propagate_increment": 1,
        "large_image_pixel_threshold": 12_000_000,
        "template_coverage_warning_pct": 75,
        "time_estimate_pixel_rate": 850_000,
        "refine_brush_radius": 15,
        "refine_iterations": 5,
        "training_max_dimension": 1200,
        "show_instructions_popup": True,
        "use_fullscreen": True,
        "zoom_max": 10.0,
        "zoom_step_factor": 1.15,
        "brush_min_radius": 2,
        "brush_max_radius": 200,
        "supported_extensions": [".png", ".jpg", ".jpeg", ".PNG", ".JPG", ".JPEG"],
    }


def load_yaml_config(config_path: Optional[str]) -> dict:
    """
    Load configuration from a YAML file.

    Returns an empty dict if the file doesn't exist (when using default path),
    PyYAML is not installed, or the file has syntax errors.
    Exits with an error if --config was explicitly specified and the file is
    missing, unparseable, or PyYAML is not installed.
    """
    explicit = config_path is not None

    if config_path is None:
        script_dir = Path(__file__).resolve().parent
        default_path = script_dir / "GrabcutExtractionConfig.yaml"
        if not default_path.exists():
            return {}
        config_path = default_path
    else:
        config_path = Path(config_path)
        if not config_path.exists():
            print(f"Error: Config file not found: {config_path}")
            sys.exit(1)

    try:
        import yaml
    except ImportError:
        msg = "PyYAML is not installed. Install with: pip install pyyaml"
        if explicit:
            print(f"Error: {msg}")
            sys.exit(1)
        else:
            print(f"Warning: {msg}")
            print("         Ignoring config.yaml, using built-in defaults.")
            return {}

    try:
        with open(config_path, "r") as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        print(f"Error: Failed to parse {config_path}:")
        print(f"  {e}")
        if explicit:
            sys.exit(1)
        else:
            print("  Using built-in defaults.")
            return {}

    if data is None:
        return {}
    if not isinstance(data, dict):
        print(f"Warning: {config_path} does not contain a YAML mapping. Ignoring.")
        return {}

    print(f"Loaded config from: {config_path}")
    return data


def merge_config(defaults: dict, yaml_cfg: dict, cli_args: argparse.Namespace) -> dict:
    """
    Merge configuration sources: defaults <- yaml <- CLI args.

    CLI args override YAML values only when explicitly provided (not None).
    """
    config = dict(defaults)

    # Layer 2: YAML overrides defaults
    for key, value in yaml_cfg.items():
        if key in config:
            config[key] = value
        else:
            print(f"Warning: Unknown config key '{key}' in YAML file. Ignoring.")

    # Layer 3: CLI args override YAML (only when explicitly provided)
    cli_mapping = {
        "samples": "initial_samples",
        "train_iterations": "train_iterations",
        "propagate_iterations": "propagate_iterations",
        "max_retries": "max_retries",
        "workers": "worker_count",
        "training_max_dim": "training_max_dimension",
    }

    for cli_name, config_key in cli_mapping.items():
        cli_value = getattr(cli_args, cli_name, None)
        if cli_value is not None:
            config[config_key] = cli_value

    # --no-parallel flag: if set, override use_parallel
    if cli_args.no_parallel:
        config["use_parallel"] = False

    return config


def process_single_image_parallel(
    img_path: str,
    bgdModel: Optional[np.ndarray],
    fgdModel: Optional[np.ndarray],
    temp_mask_dir: str,
    iterations: int,
    expansion_pixels: int = 3,
    mask_fill_holes: bool = True,
    seed_mask: Optional[np.ndarray] = None,
    mask_cleanup: bool = True,
    mask_cleanup_kernel_size: int = 5,
) -> Tuple[str, bool, Optional[str]]:
    """
    Propagate pre-blended GMM models to a single image and save the resulting mask.

    Designed to be pickled and executed in a separate process — all parameters
    are serializable (strings instead of Path objects).

    For training images (seed_mask is not None), the accepted mask is used directly
    and GMM classification is skipped entirely.

    For non-training images, pixels are classified using classify_with_frozen_gmm(),
    which applies GMM likelihood scoring followed by optional graph-cut spatial
    smoothing with GC_EVAL_FREEZE_MODEL (training GMMs are never re-estimated).

    Args:
        img_path:               Absolute path to image file (string for pickling)
        bgdModel:               Pre-blended BGD GMM model for this image (1×65 float64).
                                None for training images (seed_mask used instead).
        fgdModel:               Pre-blended FGD GMM model for this image (1×65 float64).
                                None for training images (seed_mask used instead).
        temp_mask_dir:          Absolute path to temporary mask directory (string)
        iterations:             Graph-cut refinement iterations (0 = GMM-only, no graph-cut)
        expansion_pixels:       Pixels to dilate mask boundary after segmentation
        mask_fill_holes:        Fill interior background holes enclosed by foreground
        seed_mask:              Pre-accepted binary mask for training images
        mask_cleanup:           Keep only largest foreground component (removes noise blobs)
        mask_cleanup_kernel_size: Morphological kernel size for cleanup opening

    Returns:
        Tuple of (image_name, success, error_message)
    """
    try:
        img_path_obj = Path(img_path)
        mask_dir_obj = Path(temp_mask_dir)

        if seed_mask is not None:
            # Training image — accepted mask is ground truth; skip classification
            expanded_mask = seed_mask
        else:
            image = load_image_color(img_path_obj)
            binary_mask = classify_with_frozen_gmm(image, bgdModel, fgdModel, iterations)
            binary_mask = keep_largest_component(binary_mask)

            if mask_cleanup:
                binary_mask = clean_mask(binary_mask, kernel_size=mask_cleanup_kernel_size)
            if mask_fill_holes:
                binary_mask = fill_mask_holes(binary_mask)
            expanded_mask = expand_mask(binary_mask, expansion_pixels=expansion_pixels)

        mask_filename = f"{img_path_obj.stem}_mask.png"
        mask_path = mask_dir_obj / mask_filename
        success = cv2.imwrite(str(mask_path), expanded_mask)

        if not success:
            return (img_path_obj.name, False, "Failed to save mask")

        return (img_path_obj.name, True, None)

    except Exception as e:
        return (Path(img_path).name, False, str(e))


def get_image_files(folder_path: Path, supported_extensions: tuple = ('.png', '.jpg', '.jpeg', '.PNG', '.JPG', '.JPEG')) -> List[Path]:
    """
    Get all image files in the folder with supported extensions.

    Args:
        folder_path: Path object to the folder
        supported_extensions: Tuple of file extensions to include

    Returns:
        Sorted list of Path objects for image files
    """
    image_files = []
    for item in folder_path.iterdir():
        if item.is_file() and item.suffix in supported_extensions and not item.stem.endswith('_mask'):
            image_files.append(item)
    return sorted(image_files)


def load_image_color(image_path: Path) -> np.ndarray:
    """
    Load an image in BGR color format, converting to 8-bit if necessary.

    Args:
        image_path: Path to the image file

    Returns:
        BGR color image as numpy array (8-bit 3-channel)

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
        # Use bitshift instead of division for 5-10x faster conversion
        img = (img >> 8).astype('uint8')

    # Convert to BGR if grayscale or BGRA
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
    elif img.shape[2] == 4:
        img = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)

    return img


def compute_training_scale(image_shape: Tuple[int, int],
                           max_training_dimension: int = 1200) -> float:
    """
    Compute scale factor for downscaling images during interactive training.

    Args:
        image_shape: (height, width) of the original image
        max_training_dimension: Maximum dimension in pixels. Only downscales.

    Returns:
        Scale factor in (0.0, 1.0]. Returns 1.0 if image already fits.
    """
    if max_training_dimension <= 0:
        return 1.0  # 0 (or negative) means "disabled" — no downscaling
    h, w = image_shape
    max_dim = max(h, w)
    if max_dim <= max_training_dimension:
        return 1.0
    return max_training_dimension / max_dim


def downscale_image(image: np.ndarray, scale: float) -> np.ndarray:
    """Downscale image by the given factor using INTER_AREA interpolation."""
    if scale >= 1.0:
        return image
    h, w = image.shape[:2]
    new_w = int(w * scale)
    new_h = int(h * scale)
    return cv2.resize(image, (new_w, new_h), interpolation=cv2.INTER_AREA)


def upscale_mask(mask: np.ndarray, target_shape: Tuple[int, int]) -> np.ndarray:
    """Upscale a mask to target (height, width) using nearest-neighbor interpolation."""
    h, w = target_shape
    if mask.shape[0] == h and mask.shape[1] == w:
        return mask
    return cv2.resize(mask, (w, h), interpolation=cv2.INTER_NEAREST)


def select_sample_indices(n_images: int, w: int, include_first: bool = False) -> List[int]:
    """
    Select w evenly-spaced sample indices from n images.

    When include_first is False (original behavior):
        Positions: n/w, 2n/w, 3n/w, ... w*n/w

    When include_first is True:
        First sample is always index 0 (first image).
        Remaining w-1 samples use the formula i * n / (w-1) for i=1..w-1,
        ensuring the last sample is always the final image.

    Args:
        n_images: Total number of images
        w: Number of samples to select
        include_first: If True, always include the first image as sample 0

    Returns:
        List of w integer indices (0-based)

    Raises:
        ValueError: If w > n_images or w < 1

    Examples:
        n=10, w=3, include_first=False -> [2, 5, 8]
        n=10, w=3, include_first=True  -> [0, 4, 9]
        n=100, w=5, include_first=True -> [0, 24, 49, 74, 99]
    """
    if w < 1:
        raise ValueError(f"w must be >= 1, got {w}")
    if w > n_images:
        raise ValueError(f"w ({w}) cannot exceed number of images ({n_images})")

    if include_first:
        indices = [0]
        if w > 1:
            remaining = w - 1
            for i in range(1, remaining + 1):
                pos_1based = int((i * n_images) / remaining)
                pos_0based = pos_1based - 1
                pos_0based = max(0, min(pos_0based, n_images - 1))
                indices.append(pos_0based)
        return indices

    indices = []
    for i in range(1, w + 1):
        # Calculate 1-based position: i * n / w
        pos_1based = int((i * n_images) / w)
        # Convert to 0-based index
        pos_0based = pos_1based - 1
        # Clamp to valid range
        pos_0based = max(0, min(pos_0based, n_images - 1))
        indices.append(pos_0based)

    return indices


def _get_screen_size() -> Tuple[int, int]:
    """Get screen dimensions using tkinter (stdlib). Returns (width, height)."""
    import tkinter as tk
    root = tk.Tk()
    root.withdraw()
    w = root.winfo_screenwidth()
    h = root.winfo_screenheight()
    root.destroy()
    return w, h


def show_instructions_dialog():
    """Show a dark-themed startup dialog explaining the workflow and controls."""
    BG = '#1c1c1e'
    TOOLBAR = '#2c2c2e'
    TEXT = '#f2f2f7'
    DIM = '#aeaeb2'
    GREEN = '#32d74b'
    BORDER = '#48484a'

    root = tk.Tk()
    root.title("GrabCut Masking — Instructions")
    root.geometry("760x580")
    root.resizable(False, False)
    root.configure(bg=BG)

    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - 380
    y = (root.winfo_screenheight() // 2) - 290
    root.geometry(f"+{x}+{y}")

    # Header bar
    header = tk.Frame(root, bg=TOOLBAR, pady=14)
    header.pack(fill=tk.X)
    tk.Label(header, text="GrabCut Interactive Masking", bg=TOOLBAR, fg=TEXT,
             font=('SF Pro Display', 16, 'bold')).pack()
    tk.Label(header, text="Read the workflow below, then click Start to begin.",
             bg=TOOLBAR, fg=DIM, font=('SF Pro Display', 11)).pack(pady=(2, 0))

    tk.Frame(root, bg=BORDER, height=1).pack(fill=tk.X)

    instructions = (
        "WORKFLOW\n"
        "────────────────────────────────────────────────────\n"
        "1.  Select training images that show unique angles of the object.\n"
        "2.  For each selected image, paint GREEN strokes on the object (foreground).\n"
        "3.  GrabCut generates an initial mask; correct it with GREEN (include)\n"
        "    and RED (exclude) strokes, then accept.\n"
        "4.  Accepted samples accumulate a shared model.\n"
        "5.  Model propagates to all images automatically (parallel).\n"
        "6.  Review results → Accept to save, or Reject to re-select images and retry.\n"
        "\n"
        "TIPS FOR BEST RESULTS\n"
        "────────────────────────────────────────────────────\n"
        "  ·  Select images that show unique angles, lighting conditions,\n"
        "     and object orientations — diversity improves propagation.\n"
        "  ·  Paint generously on the object interior, not just edges.\n"
        "  ·  Cover different colours and textures of the object.\n"
        "  ·  Use RED liberally on background regions — this teaches GrabCut\n"
        "     what the background looks like and greatly improves propagation.\n"
        "  ·  Zoom in to check edges and correct fine details.\n"
        "\n"
        "KEYBOARD SHORTCUTS\n"
        "────────────────────────────────────────────────────\n"
        "  Painting mode\n"
        "    Left-drag      Paint green foreground stroke\n"
        "    E / P          Switch to Erase / Paint mode\n"
        "    Right-drag     Pan view (when zoomed)\n"
        "    Scroll         Zoom in / out\n"
        "    [ / ]          Decrease / Increase brush size\n"
        "    ENTER          Confirm painting\n"
        "    C              Clear all strokes\n"
        "    ESC            Cancel\n"
        "\n"
        "  Correction mode\n"
        "    Left-drag      Draw correction stroke (current colour)\n"
        "    G / R          Switch to Green (FG) / Red (BG) mode\n"
        "    Right-drag     Pan view\n"
        "    Scroll         Zoom in / out\n"
        "    [ / ]          Decrease / Increase brush size\n"
        "    ENTER          Re-run GrabCut with corrections\n"
        "    A              Accept mask\n"
        "    C              Clear corrections & reset\n"
        "    ESC            Restart this sample\n"
        "\n"
        "  Review mode\n"
        "    ← / →          Navigate images\n"
        "    Scroll         Zoom in / out\n"
        "    Right-drag     Pan view\n"
        "    Y              Accept all masks\n"
        "    N              Reject and re-select images\n"
        "    Q / ESC        Quit\n"
        "\n"
        "  All buttons in the toolbar above each image perform the same actions.\n"
    )

    text_widget = scrolledtext.ScrolledText(
        root, wrap=tk.WORD,
        font=('Menlo', 11),
        bg='#2c2c2e', fg=TEXT,
        insertbackground=TEXT,
        selectbackground='#3a3a3c',
        relief=tk.FLAT, borderwidth=0,
        padx=16, pady=12
    )
    text_widget.insert(tk.END, instructions)
    text_widget.config(state=tk.DISABLED)
    text_widget.pack(padx=12, pady=(12, 6), fill=tk.BOTH, expand=True)

    tk.Frame(root, bg=BORDER, height=1).pack(fill=tk.X)

    btn_frame = tk.Frame(root, bg=BG)
    btn_frame.pack(fill=tk.X, pady=12)

    ok_button = tk.Button(
        btn_frame, text="✓   Continue to Image Selection",
        font=('SF Pro Display', 13, 'bold'),
        bg=GREEN, fg='black',
        relief=tk.FLAT, borderwidth=0,
        padx=24, pady=10,
        cursor='hand2',
        activebackground='#28b840', activeforeground='black',
        command=root.destroy
    )
    ok_button.pack()

    root.mainloop()


def select_sample_images(image_files: List[Path], config: dict,
                         locked_indices: Optional[List[int]] = None) -> Optional[List[int]]:
    """
    Show a dark-themed dialog with a scrollable thumbnail grid so the user can
    manually choose which images to use as GrabCut training samples.

    Args:
        image_files:    All images in the dataset.
        config:         Merged configuration dictionary.
        locked_indices: Indices of images whose GMMs were already trained in a
                        previous run. They are pre-selected and shown in blue;
                        the user cannot deselect them.

    Returns a sorted list of 0-based indices into *image_files*, or None if the
    user cancels.
    """
    BG      = '#1c1c1e'
    TOOLBAR = '#2c2c2e'
    TEXT    = '#f2f2f7'
    DIM     = '#aeaeb2'
    GREEN   = '#32d74b'
    BORDER  = '#48484a'
    SEL_BG  = '#0a84ff'   # blue tint for selected cell background

    THUMB_W    = 190
    THUMB_H    = 130
    CELL_PAD   = 8
    COLS       = 5
    CELL_W     = THUMB_W + CELL_PAD * 2
    CELL_H     = THUMB_H + CELL_PAD * 2 + 20   # 20 px for filename label

    locked: set = set(locked_indices or [])
    selected: set = set(locked)   # locked images are always selected
    result_holder = [None]   # mutable container so inner functions can write

    # ------------------------------------------------------------------
    # Build root window
    # ------------------------------------------------------------------
    root = tk.Tk()
    root.title("Select Training Images")
    root.configure(bg=BG)

    sw = root.winfo_screenwidth()
    sh = root.winfo_screenheight()
    win_w = min(sw - 80, COLS * CELL_W + 32)
    win_h = min(sh - 80, 760)
    x = (sw - win_w) // 2
    y = (sh - win_h) // 2
    root.geometry(f"{win_w}x{win_h}+{x}+{y}")
    root.resizable(True, True)

    # ------------------------------------------------------------------
    # Header
    # ------------------------------------------------------------------
    header = tk.Frame(root, bg=TOOLBAR, pady=10)
    header.pack(fill=tk.X)
    tk.Label(header, text="Select Training Images",
             bg=TOOLBAR, fg=TEXT,
             font=('SF Pro Display', 15, 'bold')).pack()
    tk.Label(header,
             text="Choose images that show unique angles, lighting, and orientations of the object.",
             bg=TOOLBAR, fg=DIM,
             font=('SF Pro Display', 11)).pack(pady=(2, 0))
    if locked:
        tk.Label(header,
                 text=f"■ Blue ({len(locked)}) = kept from previous run — cannot be deselected   "
                      f"■ Green = newly selected",
                 bg=TOOLBAR, fg=SEL_BG,
                 font=('SF Pro Display', 10)).pack(pady=(4, 0))
    tk.Frame(root, bg=BORDER, height=1).pack(fill=tk.X)

    # ------------------------------------------------------------------
    # Scrollable thumbnail grid
    # ------------------------------------------------------------------
    grid_frame = tk.Frame(root, bg=BG)
    grid_frame.pack(fill=tk.BOTH, expand=True)

    canvas = tk.Canvas(grid_frame, bg=BG, highlightthickness=0)
    scrollbar = tk.Scrollbar(grid_frame, orient=tk.VERTICAL, command=canvas.yview)
    canvas.configure(yscrollcommand=scrollbar.set)

    scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
    canvas.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    inner = tk.Frame(canvas, bg=BG)
    inner_window = canvas.create_window((0, 0), window=inner, anchor='nw')

    def _on_inner_configure(event):
        canvas.configure(scrollregion=canvas.bbox('all'))

    def _on_canvas_configure(event):
        canvas.itemconfig(inner_window, width=event.width)

    inner.bind('<Configure>', _on_inner_configure)
    canvas.bind('<Configure>', _on_canvas_configure)

    # Mousewheel scroll
    def _on_mousewheel(event):
        if event.num == 4:
            canvas.yview_scroll(-1, 'units')
        elif event.num == 5:
            canvas.yview_scroll(1, 'units')
        else:
            canvas.yview_scroll(int(-1 * (event.delta / 120)), 'units')

    canvas.bind('<MouseWheel>', _on_mousewheel)
    canvas.bind('<Button-4>', _on_mousewheel)
    canvas.bind('<Button-5>', _on_mousewheel)

    # ------------------------------------------------------------------
    # Footer
    # ------------------------------------------------------------------
    tk.Frame(root, bg=BORDER, height=1).pack(fill=tk.X)
    footer = tk.Frame(root, bg=TOOLBAR, pady=10)
    footer.pack(fill=tk.X)

    count_var = tk.StringVar(value="0 images selected")
    tk.Label(footer, textvariable=count_var,
             bg=TOOLBAR, fg=DIM,
             font=('SF Pro Display', 12)).pack(side=tk.LEFT, padx=16)

    confirm_var = tk.StringVar(value="Confirm Selection (0)")
    confirm_btn = tk.Button(
        footer, textvariable=confirm_var,
        font=('SF Pro Display', 13, 'bold'),
        bg='#3a3a3c', fg='#686868',
        relief=tk.FLAT, borderwidth=0,
        padx=20, pady=8, cursor='arrow',
        state=tk.DISABLED,
        activebackground='#28b840', activeforeground='black'
    )
    confirm_btn.pack(side=tk.RIGHT, padx=8)

    cancel_btn = tk.Button(
        footer, text="Cancel",
        font=('SF Pro Display', 12),
        bg='#3a3a3c', fg=DIM,
        relief=tk.FLAT, borderwidth=0,
        padx=16, pady=8, cursor='hand2',
        command=root.destroy
    )
    cancel_btn.pack(side=tk.RIGHT, padx=4)

    # ------------------------------------------------------------------
    # Selection state helpers
    # ------------------------------------------------------------------
    cell_frames: list = []   # list of (outer_frame, name_label) per image

    def _refresh_cell(idx: int):
        """Update visual state of cell at *idx*."""
        outer, name_lbl = cell_frames[idx]
        if idx in locked:
            outer.configure(bg=SEL_BG, highlightbackground=SEL_BG, highlightthickness=2)
            name_lbl.configure(bg=SEL_BG, fg='white')
        elif idx in selected:
            outer.configure(bg=GREEN, highlightbackground=GREEN, highlightthickness=2)
            name_lbl.configure(bg=GREEN, fg='black')
        else:
            outer.configure(bg='#2c2c2e', highlightbackground='#3a3a3c', highlightthickness=1)
            name_lbl.configure(bg='#2c2c2e', fg=DIM)

    def _update_footer():
        n = len(selected)
        count_var.set(f"{n} image{'s' if n != 1 else ''} selected")
        confirm_var.set(f"Confirm Selection ({n})")
        if n >= 1:
            confirm_btn.configure(
                bg=GREEN, fg='black', cursor='hand2', state=tk.NORMAL)
        else:
            confirm_btn.configure(
                bg='#3a3a3c', fg='#686868', cursor='arrow', state=tk.DISABLED)

    def _toggle(idx: int):
        if idx in locked:
            return  # locked images cannot be deselected
        if idx in selected:
            selected.discard(idx)
        else:
            selected.add(idx)
        _refresh_cell(idx)
        _update_footer()

    def _do_confirm():
        if selected:
            result_holder[0] = sorted(selected)
            root.destroy()

    confirm_btn.configure(command=_do_confirm)
    root.bind('<Return>',   lambda e: _do_confirm())
    root.bind('<KP_Enter>', lambda e: _do_confirm())
    root.bind('<Escape>',   lambda e: root.destroy())

    # ------------------------------------------------------------------
    # Load thumbnails and build grid
    # ------------------------------------------------------------------
    thumb_photos: list = []   # keep PhotoImage refs alive

    for idx, img_path in enumerate(image_files):
        # Load thumbnail with PIL
        try:
            pil_img = PILImage.open(str(img_path))
            pil_img.thumbnail((THUMB_W, THUMB_H), PILImage.LANCZOS)
            # Centre on a fixed-size background
            bg_img = PILImage.new('RGB', (THUMB_W, THUMB_H), (44, 44, 46))
            paste_x = (THUMB_W - pil_img.width) // 2
            paste_y = (THUMB_H - pil_img.height) // 2
            bg_img.paste(pil_img, (paste_x, paste_y))
            photo = ImageTk.PhotoImage(bg_img)
        except Exception:
            # Fallback: plain dark rectangle
            bg_img = PILImage.new('RGB', (THUMB_W, THUMB_H), (44, 44, 46))
            photo = ImageTk.PhotoImage(bg_img)

        thumb_photos.append(photo)

        row = idx // COLS
        col = idx % COLS

        # Outer cell frame (border changes on selection)
        outer = tk.Frame(inner, bg='#2c2c2e',
                         highlightbackground='#3a3a3c', highlightthickness=1)
        outer.grid(row=row, column=col, padx=4, pady=4, sticky='nw')

        # Thumbnail label
        img_lbl = tk.Label(outer, image=photo, bg='#2c2c2e',
                           cursor='hand2', relief=tk.FLAT)
        img_lbl.pack(padx=CELL_PAD, pady=(CELL_PAD, 2))

        # Filename label (truncated)
        fname = img_path.name
        if len(fname) > 22:
            fname = fname[:10] + '…' + fname[-10:]
        name_lbl = tk.Label(outer, text=fname, bg='#2c2c2e', fg=DIM,
                             font=('SF Pro Display', 9),
                             cursor='hand2')
        name_lbl.pack(pady=(0, CELL_PAD))

        # Bind click on all sub-widgets
        for widget in (outer, img_lbl, name_lbl):
            widget.bind('<Button-1>', lambda e, i=idx: _toggle(i))

        cell_frames.append((outer, name_lbl))

    # Render initial blue highlight for locked (pre-selected) cells
    for _idx in locked:
        _refresh_cell(_idx)
    _update_footer()

    root.mainloop()
    return result_holder[0]


# ---------------------------------------------------------------------------
# Shared dark-theme palette used by all Tk windows
# ---------------------------------------------------------------------------
_C = {
    'bg':          '#1c1c1e',
    'toolbar':     '#2c2c2e',
    'btn':         '#3a3a3c',
    'btn_hover':   '#4a4a4c',
    'green':       '#32d74b',
    'red':         '#ff453a',
    'orange':      '#ff9f0a',
    'blue':        '#0a84ff',
    'text':        '#f2f2f7',
    'dim':         '#aeaeb2',
    'sep':         '#48484a',
    'btn_dis':     '#303032',
    'text_dis':    '#686868',
    'hint':        '#ffe066',
}


class TkViewport:
    """
    Base class for all interactive tkinter windows.

    Each subclass creates a fullscreen window with:
      • A fixed 58 px dark toolbar at the top (built by _build_toolbar())
      • A canvas filling the remaining area showing the image
      • Scroll-wheel zoom centred on the cursor
      • Right-click drag to pan
      • Keyboard shortcuts bound by _bind_keys()

    The blocking entry point is run() which calls root.mainloop() and
    returns self._result once the window is closed.
    """

    def __init__(self, image: np.ndarray,
                 brush_radius: int = 15, min_brush: int = 2, max_brush: int = 100,
                 use_fullscreen: bool = True, fallback_width: int = 1200,
                 fallback_height: int = 800, zoom_max: float = 10.0,
                 zoom_step_factor: float = 1.15):
        self.image = image                     # full-res BGR numpy array
        self.img_h, self.img_w = image.shape[:2]
        self.brush_radius    = brush_radius
        self.min_brush       = min_brush
        self.max_brush       = max_brush
        self.use_fullscreen  = use_fullscreen
        self.fallback_width  = fallback_width
        self.fallback_height = fallback_height
        self.zoom_max        = zoom_max
        self.zoom_step_factor = zoom_step_factor

        # Viewport state
        self.zoom_level   = 1.0
        self.base_scale   = 1.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        self.canvas_w     = 800   # updated by Configure event
        self.canvas_h     = 600

        # Drawing state
        self.drawing       = False
        self.last_draw_pos = None

        # Panning state
        self.panning            = False
        self.pan_start_x        = 0
        self.pan_start_y        = 0
        self.pan_start_offset_x = 0.0
        self.pan_start_offset_y = 0.0

        # Result returned from run()
        self._result = None

        # Tkinter widgets (created in _create_window)
        self.root    = None
        self.toolbar = None
        self.canvas  = None
        self._photo           = None    # kept alive to prevent GC
        self._canvas_img_id   = None
        self._overlay_cache   = None

        # Optional toolbar widget vars (set by helpers)
        self._zoom_label_var  = None
        self._brush_scale_var = None
        self._brush_label_var = None

        # Hint label (populated by _add_hint_label)
        self._hint_var = None

        # Temporary canvas items drawn during drag (cleared on mouse-up)
        self._stroke_items: list = []

        # Cursor ring canvas item IDs (created in _create_window)
        self._cursor_outer = None
        self._cursor_inner = None
        self._cursor_ch    = None   # centre crosshair horizontal
        self._cursor_cv    = None   # centre crosshair vertical
        self._last_cursor_pos = (0, 0)

        # Set False in subclasses that don't paint (e.g. MaskReviewer)
        self._use_cursor_ring = True

    # ------------------------------------------------------------------
    # Window creation
    # ------------------------------------------------------------------
    def _create_window(self, title: str):
        self.root = tk.Tk()
        self.root.title(title)
        self.root.configure(bg=_C['bg'])

        if self.use_fullscreen:
            self.root.attributes('-fullscreen', True)
        else:
            self.root.geometry(f"{self.fallback_width}x{self.fallback_height}")

        # Toolbar strip (fixed height)
        self.toolbar = tk.Frame(self.root, bg=_C['toolbar'], height=58)
        self.toolbar.pack(side=tk.TOP, fill=tk.X)
        self.toolbar.pack_propagate(False)

        # 1-px separator
        tk.Frame(self.root, bg=_C['sep'], height=1).pack(side=tk.TOP, fill=tk.X)

        # Image canvas
        self.canvas = tk.Canvas(self.root, bg=_C['bg'],
                                cursor='crosshair', highlightthickness=0)
        self.canvas.pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        # Placeholder image item
        self._canvas_img_id = self.canvas.create_image(0, 0, anchor=tk.NW)

        # Dual-ring brush cursor (hidden native cursor, drawn as canvas items)
        if self._use_cursor_ring:
            self._cursor_outer = self.canvas.create_oval(
                -100, -100, -100, -100, outline='black', width=3, tags='cursor_ring')
            self._cursor_inner = self.canvas.create_oval(
                -100, -100, -100, -100, outline='white', width=1.5, tags='cursor_ring')
            self._cursor_ch = self.canvas.create_line(
                -100, -100, -100, -100, fill='white', width=1, tags='cursor_ring')
            self._cursor_cv = self.canvas.create_line(
                -100, -100, -100, -100, fill='white', width=1, tags='cursor_ring')
            self.canvas.configure(cursor='none')

        # Mouse bindings
        self.canvas.bind('<ButtonPress-1>',   self._on_lb_down)
        self.canvas.bind('<B1-Motion>',       self._on_lb_move)
        self.canvas.bind('<ButtonRelease-1>', self._on_lb_up)
        self.canvas.bind('<ButtonPress-3>',   self._on_rb_down)
        self.canvas.bind('<B3-Motion>',       self._on_rb_move)
        self.canvas.bind('<ButtonRelease-3>', self._on_rb_up)
        self.canvas.bind('<ButtonPress-2>',   self._on_rb_down)  # middle=pan too
        self.canvas.bind('<B2-Motion>',       self._on_rb_move)
        self.canvas.bind('<ButtonRelease-2>', self._on_rb_up)
        # Scroll zoom (macOS/Windows delta; Linux button 4/5)
        self.canvas.bind('<MouseWheel>', self._on_scroll)
        self.canvas.bind('<Button-4>',   self._on_scroll)
        self.canvas.bind('<Button-5>',   self._on_scroll)
        # Canvas resize
        self.canvas.bind('<Configure>', self._on_canvas_resize)
        # Cursor ring tracking
        self.canvas.bind('<Motion>', self._on_canvas_motion)
        self.canvas.bind('<B1-Motion>', self._on_canvas_motion, add='+')
        self.canvas.bind('<Leave>',  self._on_canvas_leave)
        self.canvas.bind('<Enter>',  self._on_canvas_enter)

        # Let subclasses populate the toolbar and bind keys
        self._build_toolbar()
        self._bind_keys()
        self._bind_common_keys()   # arrow-key pan (after subclass binds)

        # First render after layout settles
        self.root.after(80, self._first_render)

    def _first_render(self):
        """Called once after the window has its real geometry."""
        self.canvas_w = max(1, self.canvas.winfo_width())
        self.canvas_h = max(1, self.canvas.winfo_height())
        self._compute_base_scale()
        self._update_zoom_label()
        self._refresh_display()

    # ------------------------------------------------------------------
    # Scale / pan helpers
    # ------------------------------------------------------------------
    def _compute_base_scale(self):
        scale_w = self.canvas_w / self.img_w
        scale_h = self.canvas_h / self.img_h
        self.base_scale = min(scale_w, scale_h)
        self.min_zoom   = 1.0
        self.max_zoom   = max(self.zoom_max, 1.0 / self.base_scale)

    def _on_canvas_resize(self, event):
        self.canvas_w = max(1, event.width)
        self.canvas_h = max(1, event.height)
        self._compute_base_scale()
        self._refresh_display()

    def _screen_to_image(self, cx: int, cy: int) -> Tuple[float, float]:
        eff = self.base_scale * self.zoom_level
        return self.pan_offset_x + cx / eff, self.pan_offset_y + cy / eff

    def _clamp_pan(self):
        eff    = self.base_scale * self.zoom_level
        view_w = self.canvas_w / eff
        view_h = self.canvas_h / eff
        self.pan_offset_x = max(0.0, min(self.pan_offset_x, self.img_w - view_w))
        self.pan_offset_y = max(0.0, min(self.pan_offset_y, self.img_h - view_h))

    # ------------------------------------------------------------------
    # Rendering
    # ------------------------------------------------------------------
    def _render_to_canvas(self, source: np.ndarray):
        """Crop visible region of *source*, scale to canvas, update PhotoImage."""
        if self.canvas_w < 2 or self.canvas_h < 2:
            return
        eff = self.base_scale * self.zoom_level
        self._clamp_pan()

        src_h, src_w = source.shape[:2]
        x0 = max(0, min(int(self.pan_offset_x), src_w - 1))
        y0 = max(0, min(int(self.pan_offset_y), src_h - 1))
        x1 = min(src_w, int(self.pan_offset_x + self.canvas_w / eff) + 1)
        y1 = min(src_h, int(self.pan_offset_y + self.canvas_h / eff) + 1)
        crop = source[y0:y1, x0:x1]
        if crop.size == 0:
            return

        tw = max(1, int((x1 - x0) * eff))
        th = max(1, int((y1 - y0) * eff))
        # cv2 resize is significantly faster than PIL LANCZOS for both up- and
        # downscaling.  INTER_AREA gives high quality for shrinking (fit-to-screen);
        # INTER_NEAREST is instant for zoomed-in pixel viewing.
        cv2_interp = cv2.INTER_NEAREST if eff > 1.0 else cv2.INTER_AREA

        rgb     = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        resized = cv2.resize(rgb, (tw, th), interpolation=cv2_interp)
        pil_img = PILImage.fromarray(resized)

        # Pad to canvas size with dark background
        if tw < self.canvas_w or th < self.canvas_h:
            padded = PILImage.new('RGB', (self.canvas_w, self.canvas_h), (28, 28, 30))
            padded.paste(pil_img, (0, 0))
            pil_img = padded

        self._photo = ImageTk.PhotoImage(pil_img)
        self.canvas.itemconfig(self._canvas_img_id, image=self._photo)

    def _refresh_display(self):
        self._overlay_cache = self._compose_overlay()
        self._render_to_canvas(self._overlay_cache)

    def _quick_display_update(self):
        if self._overlay_cache is not None:
            self._render_to_canvas(self._overlay_cache)
        else:
            self._refresh_display()

    def _compose_overlay(self) -> np.ndarray:
        """Override in subclasses: return full-res BGR image with overlays."""
        return self.image.copy()

    # ------------------------------------------------------------------
    # Mouse event handlers
    # ------------------------------------------------------------------
    def _on_scroll(self, event):
        cx, cy = event.x, event.y
        img_x, img_y = self._screen_to_image(cx, cy)
        if getattr(event, 'delta', 0) > 0 or event.num == 4:
            self.zoom_level = min(self.max_zoom, self.zoom_level * self.zoom_step_factor)
        else:
            self.zoom_level = max(self.min_zoom, self.zoom_level / self.zoom_step_factor)
        eff = self.base_scale * self.zoom_level
        self.pan_offset_x = img_x - cx / eff
        self.pan_offset_y = img_y - cy / eff
        self._clamp_pan()
        self._update_zoom_label()
        # Overlay content unchanged — reuse cache, just re-render at new zoom/pan
        self._quick_display_update()

    def _on_rb_down(self, event):
        self.panning            = True
        self.pan_start_x        = event.x
        self.pan_start_y        = event.y
        self.pan_start_offset_x = self.pan_offset_x
        self.pan_start_offset_y = self.pan_offset_y

    def _on_rb_move(self, event):
        if not self.panning:
            return
        eff = self.base_scale * self.zoom_level
        self.pan_offset_x = self.pan_start_offset_x - (event.x - self.pan_start_x) / eff
        self.pan_offset_y = self.pan_start_offset_y - (event.y - self.pan_start_y) / eff
        self._clamp_pan()
        self._quick_display_update()

    def _on_rb_up(self, event):
        self.panning = False

    # ------------------------------------------------------------------
    # Cursor ring handlers
    # ------------------------------------------------------------------
    def _on_canvas_motion(self, event):
        self._last_cursor_pos = (event.x, event.y)
        self._update_cursor_ring(event.x, event.y)

    def _on_canvas_leave(self, _event):
        if self._cursor_outer is None:
            return
        for item in (self._cursor_outer, self._cursor_inner,
                     self._cursor_ch, self._cursor_cv):
            if item is not None:
                self.canvas.coords(item, -200, -200, -200, -200)

    def _on_canvas_enter(self, _event):
        pass  # ring appears on next Motion event

    def _update_cursor_ring(self, x: int, y: int):
        """Reposition the dual-ring brush cursor to (x, y) in canvas pixels."""
        if self._cursor_outer is None:
            return
        r = max(2.0, self.brush_radius * self.base_scale * self.zoom_level)
        self.canvas.coords(self._cursor_outer, x-r-1, y-r-1, x+r+1, y+r+1)
        self.canvas.coords(self._cursor_inner, x-r,   y-r,   x+r,   y+r)
        # 5-px crosshair at centre
        clen = 5
        self.canvas.coords(self._cursor_ch, x-clen, y, x+clen, y)
        self.canvas.coords(self._cursor_cv, x, y-clen, x, y+clen)
        # tag_raise omitted: _draw_canvas_stroke raises cursor_ring after each
        # new stroke item; when not drawing, nothing covers the ring anyway.

    # Override in subclasses
    def _on_lb_down(self, event): pass
    def _on_lb_move(self, event): pass
    def _on_lb_up(self, event):   pass

    # ------------------------------------------------------------------
    # Toolbar / key extension points
    # ------------------------------------------------------------------
    def _build_toolbar(self): pass
    def _bind_keys(self):     pass

    def _bind_common_keys(self):
        """Arrow-key panning, bound for all windows after subclass _bind_keys()."""
        self.root.bind('<Left>',  lambda e: self._pan_arrow(-40,  0))
        self.root.bind('<Right>', lambda e: self._pan_arrow( 40,  0))
        self.root.bind('<Up>',    lambda e: self._pan_arrow(  0, -40))
        self.root.bind('<Down>',  lambda e: self._pan_arrow(  0,  40))

    def _pan_arrow(self, dx_screen: int, dy_screen: int):
        """Pan the viewport by *dx_screen* / *dy_screen* canvas pixels."""
        eff = self.base_scale * self.zoom_level
        self.pan_offset_x += dx_screen / eff
        self.pan_offset_y += dy_screen / eff
        self._clamp_pan()
        self._quick_display_update()

    # ------------------------------------------------------------------
    # Zoom toolbar actions
    # ------------------------------------------------------------------
    def _zoom_step(self, direction: int):
        cx, cy = self.canvas_w / 2, self.canvas_h / 2
        img_x, img_y = self._screen_to_image(cx, cy)
        if direction > 0:
            self.zoom_level = min(self.max_zoom, self.zoom_level * self.zoom_step_factor)
        else:
            self.zoom_level = max(self.min_zoom, self.zoom_level / self.zoom_step_factor)
        eff = self.base_scale * self.zoom_level
        self.pan_offset_x = img_x - cx / eff
        self.pan_offset_y = img_y - cy / eff
        self._clamp_pan()
        self._update_zoom_label()
        # Overlay content unchanged — skip recompose, re-render cache at new zoom
        self._quick_display_update()

    def _zoom_fit(self):
        self.zoom_level   = 1.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        self._update_zoom_label()
        self._refresh_display()

    def _update_zoom_label(self):
        if self._zoom_label_var is not None:
            pct = int(round(self.zoom_level * self.base_scale * 100))
            self._zoom_label_var.set(f"{pct}%")

    # ------------------------------------------------------------------
    # Brush helpers
    # ------------------------------------------------------------------
    def _decrease_brush(self):
        step = max(1, self.brush_radius // 5)
        self.brush_radius = max(self.min_brush, self.brush_radius - step)
        self._sync_brush_widgets()

    def _increase_brush(self):
        step = max(1, self.brush_radius // 5)
        self.brush_radius = min(self.max_brush, self.brush_radius + step)
        self._sync_brush_widgets()

    def _on_brush_slider(self, val):
        self.brush_radius = int(float(val))
        if self._brush_label_var is not None:
            self._brush_label_var.set(str(self.brush_radius))

    def _sync_brush_widgets(self):
        if self._brush_scale_var is not None:
            self._brush_scale_var.set(self.brush_radius)
        if self._brush_label_var is not None:
            self._brush_label_var.set(str(self.brush_radius))
        # Refresh cursor ring immediately so it resizes with the brush
        x, y = self._last_cursor_pos
        self._update_cursor_ring(x, y)

    # ------------------------------------------------------------------
    # Drawing helpers (same algorithm as old InteractiveViewport)
    # ------------------------------------------------------------------
    def _draw_on_mask_interpolated(self, mask, orig_x, orig_y, value, radius):
        if self.last_draw_pos is not None:
            lx, ly = self.last_draw_pos
            cv2.line(mask, (lx, ly), (orig_x, orig_y), value, thickness=radius * 2)
        cv2.circle(mask, (orig_x, orig_y), radius, value, -1)

    def _draw_on_overlay_interpolated(self, orig_x, orig_y, color):
        if self._overlay_cache is None:
            return
        if self.last_draw_pos is not None:
            lx, ly = self.last_draw_pos
            cv2.line(self._overlay_cache, (lx, ly), (orig_x, orig_y),
                     color, thickness=self.brush_radius * 2)
        cv2.circle(self._overlay_cache, (orig_x, orig_y), self.brush_radius, color, -1)

    def _draw_canvas_stroke(self, cx: int, cy: int,
                            prev_cx: Optional[int], prev_cy: Optional[int],
                            color_hex: str):
        """Draw a temporary canvas stroke item for instant visual feedback.

        Bypasses PIL entirely — Tk renders canvas items at native speed
        with no numpy/PIL round-trip. Items are cleared and replaced by a
        full numpy render when _commit_stroke() is called on mouse-up.
        """
        r = max(1.5, self.brush_radius * self.base_scale * self.zoom_level)
        if prev_cx is not None and prev_cy is not None:
            item = self.canvas.create_line(
                prev_cx, prev_cy, cx, cy,
                fill=color_hex, width=max(1, int(r * 2)), capstyle=tk.ROUND)
            self._stroke_items.append(item)
        item = self.canvas.create_oval(
            cx - r, cy - r, cx + r, cy + r, fill=color_hex, outline='')
        self._stroke_items.append(item)
        self.canvas.tag_raise('cursor_ring')

    def _commit_stroke(self):
        """Remove temporary canvas stroke items and do a full numpy→PIL render."""
        for item in self._stroke_items:
            self.canvas.delete(item)
        self._stroke_items = []
        self._refresh_display()

    # ------------------------------------------------------------------
    # Toolbar widget factories
    # ------------------------------------------------------------------
    def _sep(self, parent):
        f = tk.Frame(parent, bg=_C['sep'], width=1)
        f.pack(side=tk.LEFT, fill=tk.Y, padx=6, pady=10)

    def _btn(self, parent, text, cmd, bg=None, fg=None,
             state=tk.NORMAL, padx=8):
        bg = bg or _C['btn']
        fg = fg or _C['text']
        b  = tk.Label(parent, text=text,
                      bg=bg, fg=fg, relief=tk.FLAT, borderwidth=0,
                      padx=padx, pady=6, cursor='hand2',
                      font=('SF Pro Display', 12))
        b.pack(side=tk.LEFT, padx=4, pady=8)
        if state == tk.NORMAL:
            b.bind('<Button-1>', lambda e: cmd())
            b.bind('<Enter>', lambda e: b.configure(bg=_C['btn_hover']))
            b.bind('<Leave>', lambda e: b.configure(bg=bg))
        return b

    def _lbl(self, parent, text, fg=None, font=None):
        fg   = fg or _C['text']
        font = font or ('SF Pro Display', 12)
        l    = tk.Label(parent, text=text, bg=_C['toolbar'], fg=fg, font=font)
        l.pack(side=tk.LEFT, padx=4)
        return l

    def _add_zoom_controls(self, parent):
        self._sep(parent)
        self._btn(parent, '−', lambda: self._zoom_step(-1), padx=8)
        self._zoom_label_var = tk.StringVar(value='100%')
        tk.Label(parent, textvariable=self._zoom_label_var,
                 bg=_C['toolbar'], fg=_C['dim'],
                 font=('SF Pro Display', 11), width=5).pack(side=tk.LEFT)
        self._btn(parent, '+', lambda: self._zoom_step(1), padx=8)
        self._btn(parent, '⊡ Fit', self._zoom_fit, padx=8)

    def _add_brush_controls(self, parent):
        self._lbl(parent, 'Brush:', fg=_C['dim'], font=('SF Pro Display', 11))
        self._brush_scale_var = tk.DoubleVar(value=self.brush_radius)
        tk.Scale(parent,
                 from_=self.min_brush, to=self.max_brush,
                 orient=tk.HORIZONTAL,
                 variable=self._brush_scale_var,
                 command=self._on_brush_slider,
                 bg=_C['toolbar'], fg=_C['text'],
                 troughcolor=_C['btn'], highlightthickness=0,
                 bd=0, width=10, length=110, showvalue=False,
                 sliderlength=16).pack(side=tk.LEFT, padx=4, pady=12)
        self._brush_label_var = tk.StringVar(value=str(self.brush_radius))
        tk.Label(parent, textvariable=self._brush_label_var,
                 bg=_C['toolbar'], fg=_C['dim'],
                 font=('SF Pro Display', 11), width=3,
                 anchor='e').pack(side=tk.LEFT)
        self._lbl(parent, 'px', fg=_C['dim'], font=('SF Pro Display', 11))

    # ------------------------------------------------------------------
    # Hint / status label
    # ------------------------------------------------------------------
    def _add_hint_label(self, parent, initial_text: str):
        """Add a context-sensitive italic hint label to a toolbar frame."""
        self._hint_var = tk.StringVar(value=initial_text)
        tk.Label(parent, textvariable=self._hint_var,
                 bg=_C['toolbar'], fg=_C['hint'],
                 font=('SF Pro Display', 10, 'italic'),
                 wraplength=400, anchor='w').pack(side=tk.LEFT, padx=14)

    def _set_hint(self, text: str):
        """Update the hint label text."""
        if self._hint_var is not None:
            self._hint_var.set(text)

    # ------------------------------------------------------------------
    # Event loop
    # ------------------------------------------------------------------
    def run(self):
        """Start mainloop; return self._result when the window closes."""
        self.root.mainloop()
        return self._result


# ---------------------------------------------------------------------------
# ForegroundPainter  (replaces the old class of the same name)
# ---------------------------------------------------------------------------
class ForegroundPainter(TkViewport):
    """
    Fullscreen window for painting GREEN foreground strokes.

    Toolbar controls (all keyboard shortcuts still work):
      ✓ Confirm  — ENTER   (enabled only after ≥100 px painted)
        Clear    — C
      ✕ Cancel   — ESC
      Brush slider / [ ]
      Zoom − % + Fit / scroll
    """

    def __init__(self, window_name: str, image: np.ndarray,
                 brush_radius: int = 15, use_fullscreen: bool = True,
                 fallback_width: int = 1200, fallback_height: int = 800,
                 min_brush: int = 2, max_brush: int = 100,
                 zoom_max: float = 10.0, zoom_step_factor: float = 1.15,
                 sample_num: int = 1, total_samples: int = 1):
        super().__init__(image, brush_radius, min_brush, max_brush,
                         use_fullscreen, fallback_width, fallback_height,
                         zoom_max, zoom_step_factor)
        self._window_name   = window_name
        self._sample_num    = sample_num
        self._total_samples = total_samples
        self.paint_mask     = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
        self._painted_px    = 0
        self._confirm_btn   = None
        self._paint_btn     = None
        self._erase_btn     = None
        self._erase_mode    = False
        self._undo_stack: list = []   # list of paint_mask copies (max 20)

    # -- toolbar -----------------------------------------------------------
    def _build_toolbar(self):
        tb = self.toolbar

        # Right-side actions FIRST — reserves space before left items are placed
        right = tk.Frame(tb, bg=_C['toolbar'])
        right.pack(side=tk.RIGHT, padx=8)

        self._add_zoom_controls(right)
        self._sep(right)

        self._btn(right, '↩ Undo',  self._undo)
        self._btn(right, 'Clear',   self._do_clear)
        self._btn(right, '✕ Cancel', self._do_cancel, fg=_C['dim'])
        self._confirm_btn = tk.Label(
            right, text='✓  Confirm',
            bg=_C['btn_dis'], fg=_C['text_dis'],
            relief=tk.FLAT, borderwidth=0,
            padx=8, pady=6,
            font=('SF Pro Display', 12),
            cursor='arrow')
        self._confirm_btn.pack(side=tk.LEFT, padx=4, pady=8)

        # Sample badge
        tk.Label(tb, text=f"  Sample {self._sample_num} / {self._total_samples}  ",
                 bg=_C['blue'], fg='white',
                 font=('SF Pro Display', 11, 'bold')).pack(
            side=tk.LEFT, padx=(12, 4), pady=12)

        # Filename (truncated)
        fname = self._window_name.split(': ')[-1]
        if len(fname) > 30:
            fname = '…' + fname[-27:]
        self._lbl(tb, fname, fg=_C['dim'], font=('SF Pro Display', 11))

        self._sep(tb)

        # Paint / Erase mode toggle
        self._paint_btn = self._btn(tb, '● Paint', self._set_paint_mode,
                                    bg=_C['green'], fg='black')
        self._erase_btn = self._btn(tb, '◻ Erase', self._set_erase_mode)

        self._sep(tb)
        self._add_brush_controls(tb)
        self._sep(tb)

        # Context hint (last — gets clipped first on narrow toolbars)
        self._add_hint_label(
            tb, 'Paint GREEN on the object')

    def _bind_keys(self):
        self.root.bind('<Return>',    lambda e: self._do_confirm())
        self.root.bind('<KP_Enter>',  lambda e: self._do_confirm())
        self.root.bind('<Escape>',    lambda e: self._do_cancel())
        self.root.bind('c',           lambda e: self._do_clear())
        self.root.bind('[',           lambda e: self._decrease_brush())
        self.root.bind(']',           lambda e: self._increase_brush())
        self.root.bind('e',           lambda e: self._set_erase_mode())
        self.root.bind('p',           lambda e: self._set_paint_mode())
        self.root.bind('<Control-z>', lambda e: self._undo())
        self.root.bind('<Command-z>', lambda e: self._undo())

    # -- overlay -----------------------------------------------------------
    def _compose_overlay(self) -> np.ndarray:
        overlay = self.image.copy()
        painted = self.paint_mask > 0
        if np.any(painted):
            overlay[painted] = (
                overlay[painted] * 0.4 + np.array([0, 200, 0]) * 0.6
            ).astype('uint8')
        return overlay

    # -- drawing -----------------------------------------------------------
    def _on_lb_down(self, event):
        self._push_undo()          # snapshot before stroke begins
        self.drawing       = True
        self.last_draw_pos = None
        self._draw_stroke(event.x, event.y)

    def _on_lb_move(self, event):
        if self.drawing:
            self._draw_stroke(event.x, event.y)

    def _on_lb_up(self, event):
        self.drawing       = False
        self.last_draw_pos = None
        # Count painted pixels once per stroke (not per motion event)
        self._painted_px = int(np.count_nonzero(self.paint_mask))
        self._update_confirm_state()
        self._commit_stroke()      # clear canvas items + full re-render

    def _draw_stroke(self, cx: int, cy: int):
        img_x, img_y = self._screen_to_image(cx, cy)
        ox = max(0, min(int(img_x), self.img_w - 1))
        oy = max(0, min(int(img_y), self.img_h - 1))

        # Convert previous image-space pos back to screen coords for canvas line
        prev_cx = prev_cy = None
        if self.last_draw_pos is not None:
            lx, ly = self.last_draw_pos
            eff = self.base_scale * self.zoom_level
            prev_cx = int((lx - self.pan_offset_x) * eff)
            prev_cy = int((ly - self.pan_offset_y) * eff)

        if self._erase_mode:
            self._draw_on_mask_interpolated(self.paint_mask, ox, oy, 0, self.brush_radius)
            color_hex = _C['orange']
        else:
            self._draw_on_mask_interpolated(self.paint_mask, ox, oy, 255, self.brush_radius)
            color_hex = _C['green']

        # Instant canvas-item feedback — no PIL round-trip during drag
        self._draw_canvas_stroke(cx, cy, prev_cx, prev_cy, color_hex)
        self.last_draw_pos = (ox, oy)
        # Pixel count updated on mouse-up only (see _on_lb_up) to avoid the
        # cost of np.count_nonzero on every motion event for large images.

    def _update_confirm_state(self):
        if self._confirm_btn is None:
            return
        if self._painted_px >= 100:
            self._confirm_btn.configure(
                cursor='hand2', bg=_C['green'], fg='black')
            self._confirm_btn.bind('<Button-1>', lambda e: self._do_confirm())
            self._confirm_btn.bind('<Enter>',
                lambda e: self._confirm_btn.configure(bg='#28b840'))
            self._confirm_btn.bind('<Leave>',
                lambda e: self._confirm_btn.configure(bg=_C['green']))
        else:
            self._confirm_btn.configure(
                cursor='arrow', bg=_C['btn_dis'], fg=_C['text_dis'])
            self._confirm_btn.unbind('<Button-1>')
            self._confirm_btn.unbind('<Enter>')
            self._confirm_btn.unbind('<Leave>')

    # -- mode switches -----------------------------------------------------
    def _set_paint_mode(self):
        self._erase_mode = False
        if self._paint_btn:
            self._paint_btn.configure(bg=_C['green'], fg='black')
            self._paint_btn.bind('<Enter>',
                lambda e: self._paint_btn.configure(bg='#28b840'))
            self._paint_btn.bind('<Leave>',
                lambda e: self._paint_btn.configure(bg=_C['green']))
        if self._erase_btn:
            self._erase_btn.configure(bg=_C['btn'], fg=_C['text'])
            self._erase_btn.bind('<Enter>',
                lambda e: self._erase_btn.configure(bg=_C['btn_hover']))
            self._erase_btn.bind('<Leave>',
                lambda e: self._erase_btn.configure(bg=_C['btn']))
        self._set_hint('Paint GREEN on the object' )

    def _set_erase_mode(self):
        self._erase_mode = True
        if self._erase_btn:
            self._erase_btn.configure(bg=_C['orange'], fg='black')
            self._erase_btn.bind('<Enter>',
                lambda e: self._erase_btn.configure(bg='#e08800'))
            self._erase_btn.bind('<Leave>',
                lambda e: self._erase_btn.configure(bg=_C['orange']))
        if self._paint_btn:
            self._paint_btn.configure(bg=_C['btn'], fg=_C['text'])
            self._paint_btn.bind('<Enter>',
                lambda e: self._paint_btn.configure(bg=_C['btn_hover']))
            self._paint_btn.bind('<Leave>',
                lambda e: self._paint_btn.configure(bg=_C['btn']))
        self._set_hint('ERASE mode — drag to remove painted areas.  [ ] resize brush.')

    # -- undo --------------------------------------------------------------
    def _push_undo(self):
        if len(self._undo_stack) >= 20:
            self._undo_stack.pop(0)
        self._undo_stack.append(self.paint_mask.copy())

    def _undo(self):
        if not self._undo_stack:
            return
        self.paint_mask    = self._undo_stack.pop()
        self._painted_px   = int(np.count_nonzero(self.paint_mask))
        self._overlay_cache = None
        self._update_confirm_state()
        self._refresh_display()

    # -- actions -----------------------------------------------------------
    def _do_confirm(self):
        if self._painted_px >= 100:
            self._result = self.paint_mask
            self.root.quit()
            self.root.destroy()
        else:
            print("  Paint more of the foreground before confirming")

    def _do_clear(self):
        self.paint_mask[:] = 0
        self._painted_px   = 0
        self._undo_stack   = []
        self._overlay_cache = None
        self._update_confirm_state()
        self._refresh_display()
        self._set_paint_mode()     # reset to paint mode on full clear

    def _do_cancel(self):
        self._result = None
        self.root.quit()
        self.root.destroy()

    # -- public API (same as old class) ------------------------------------
    def paint(self) -> np.ndarray:
        """Show window; return paint_mask or raise ValueError on cancel."""
        self._create_window(self._window_name)
        result = self.run()
        if result is None:
            raise ValueError("User cancelled foreground painting")
        return result


# ---------------------------------------------------------------------------
# MaskCorrector  (replaces the old class of the same name)
# ---------------------------------------------------------------------------
class MaskCorrector(TkViewport):
    """
    Fullscreen window for correcting a GrabCut mask with FG/BG scribbles.

    Toolbar controls:
      ● Foreground / ● Background  toggle  — G / R
      Brush slider / [ ]
      ↺ Re-run GrabCut  — ENTER  (enabled once a scribble is drawn)
      ✓ Accept          — A
        Clear           — C
      ← Re-paint        — ESC
      Zoom − % + Fit / scroll
    """

    FOREGROUND_MODE = 'foreground'
    BACKGROUND_MODE = 'background'

    def __init__(self, window_name: str, image: np.ndarray,
                 brush_radius: int = 15, use_fullscreen: bool = True,
                 fallback_width: int = 1200, fallback_height: int = 800,
                 min_brush: int = 2, max_brush: int = 100,
                 zoom_max: float = 10.0, zoom_step_factor: float = 1.15,
                 sample_num: int = 1, total_samples: int = 1):
        super().__init__(image, brush_radius, min_brush, max_brush,
                         use_fullscreen, fallback_width, fallback_height,
                         zoom_max, zoom_step_factor)
        self._window_name   = window_name
        self._sample_num    = sample_num
        self._total_samples = total_samples
        self.current_mode   = self.BACKGROUND_MODE

        self.fg_scribble_mask = np.zeros((self.img_h, self.img_w), dtype=np.uint8)
        self.bg_scribble_mask = np.zeros((self.img_h, self.img_w), dtype=np.uint8)

        # Set during correct()
        self._gc_mask      = None
        self._bgdModel     = None
        self._fgdModel     = None
        self._gc_iters     = 5
        self._initial_mask = None
        self._initial_bgd  = None
        self._initial_fgd  = None

        self._fg_btn       = None
        self._bg_btn       = None
        self._rerun_btn    = None
        self._rerun_enabled = False

        # Stroke-level undo: list of (fg_scribble_copy, bg_scribble_copy) tuples (max 20)
        self._stroke_undo_stack: list = []
        # Last committed GrabCut state (after most recent Re-run); None = not yet run
        self._last_gc_mask:   Optional[np.ndarray] = None
        self._last_bgdModel:  Optional[np.ndarray] = None
        self._last_fgdModel:  Optional[np.ndarray] = None
        # Overlay blend alpha (mask preview opacity)
        self._overlay_alpha: float = 0.5

    # -- toolbar -----------------------------------------------------------
    def _build_toolbar(self):
        tb = self.toolbar

        # Right-side actions FIRST — reserves space before left items are placed
        right = tk.Frame(tb, bg=_C['toolbar'])
        right.pack(side=tk.RIGHT, padx=8)

        self._add_zoom_controls(right)
        self._sep(right)

        self._btn(right, '↩ Undo',      self._undo)
        self._btn(right, 'Clear',        self._do_clear)
        self._btn(right, '↺ Restart',    self._do_back, fg=_C['dim'])

        self._rerun_btn = tk.Label(
            right, text='↺  Run GrabCut',
            bg=_C['btn'], fg=_C['text_dis'],
            relief=tk.FLAT, borderwidth=0,
            padx=8, pady=6, cursor='arrow',
            font=('SF Pro Display', 12))
        self._rerun_btn.pack(side=tk.LEFT, padx=4, pady=8)

        self._btn(right, '✓  Accept', self._do_accept,
                  bg=_C['green'], fg='black')

        # Sample badge
        tk.Label(tb, text=f"  Sample {self._sample_num} / {self._total_samples}  ",
                 bg=_C['blue'], fg='white',
                 font=('SF Pro Display', 11, 'bold')).pack(
            side=tk.LEFT, padx=(12, 4), pady=12)

        # Filename
        fname = self._window_name.split(': ')[-1]
        if len(fname) > 26:
            fname = '…' + fname[-23:]
        self._lbl(tb, fname, fg=_C['dim'], font=('SF Pro Display', 11))

        self._sep(tb)

        # Mode toggle
        self._lbl(tb, 'Draw:', fg=_C['dim'], font=('SF Pro Display', 11))

        self._fg_btn = tk.Label(
            tb, text='  ● Foreground  ',
            bg=_C['btn'], fg=_C['dim'],
            font=('SF Pro Display', 11, 'bold'),
            relief=tk.FLAT, borderwidth=0,
            padx=8, pady=6, cursor='hand2')
        self._fg_btn.bind('<Button-1>', lambda e: self._set_fg_mode())
        self._fg_btn.pack(side=tk.LEFT, padx=(4, 0), pady=8)

        self._bg_btn = tk.Label(
            tb, text='  ● Background  ',
            bg=_C['btn'], fg=_C['dim'],
            font=('SF Pro Display', 11, 'bold'),
            relief=tk.FLAT, borderwidth=0,
            padx=8, pady=6, cursor='hand2')
        self._bg_btn.bind('<Button-1>', lambda e: self._set_bg_mode())
        self._bg_btn.pack(side=tk.LEFT, padx=(0, 4), pady=8)

        self._update_mode_buttons()
        self._sep(tb)
        self._add_brush_controls(tb)
        self._sep(tb)

        # Context hint (last — gets clipped first on narrow toolbars)
        self._add_hint_label(
            tb, 'Draw corrections: GREEN = include,  RED = exclude.  '
                'ENTER to re-run GrabCut.')

    def _update_mode_buttons(self):
        if self.current_mode == self.FOREGROUND_MODE:
            self._fg_btn.configure(bg=_C['green'], fg='black')
            self._bg_btn.configure(bg=_C['btn'],   fg=_C['dim'])
        else:
            self._fg_btn.configure(bg=_C['btn'],  fg=_C['dim'])
            self._bg_btn.configure(bg=_C['red'],  fg='white')

    def _bind_keys(self):
        self.root.bind('<Return>',    lambda e: self._do_rerun())
        self.root.bind('<KP_Enter>',  lambda e: self._do_rerun())
        self.root.bind('<Escape>',    lambda e: self._do_back())
        self.root.bind('a',           lambda e: self._do_accept())
        self.root.bind('c',           lambda e: self._do_clear())
        self.root.bind('g',           lambda e: self._set_fg_mode())
        self.root.bind('r',           lambda e: self._set_bg_mode())
        self.root.bind('[',           lambda e: self._decrease_brush())
        self.root.bind(']',           lambda e: self._increase_brush())
        self.root.bind('<Control-z>', lambda e: self._undo())
        self.root.bind('<Command-z>', lambda e: self._undo())

    # -- overlay -----------------------------------------------------------
    def _compose_overlay(self) -> np.ndarray:
        overlay = self.image.copy()
        alpha = self._overlay_alpha
        if self._gc_mask is not None:
            fg = (self._gc_mask == cv2.GC_FGD) | (self._gc_mask == cv2.GC_PR_FGD)
            overlay[fg] = (
                overlay[fg] * (1.0 - alpha) + np.array([0, 200, 0]) * alpha
            ).astype('uint8')
        fgs = self.fg_scribble_mask > 0
        if np.any(fgs):
            overlay[fgs] = (overlay[fgs] * 0.3 + np.array([0, 200, 0]) * 0.7).astype('uint8')
        bgs = self.bg_scribble_mask > 0
        if np.any(bgs):
            overlay[bgs] = (overlay[bgs] * 0.3 + np.array([0, 0, 220]) * 0.7).astype('uint8')
        return overlay

    # -- drawing -----------------------------------------------------------
    def _on_lb_down(self, event):
        # Snapshot scribble state before stroke begins (for stroke-level undo)
        if len(self._stroke_undo_stack) >= 20:
            self._stroke_undo_stack.pop(0)
        self._stroke_undo_stack.append((
            self.fg_scribble_mask.copy(),
            self.bg_scribble_mask.copy()
        ))
        self.drawing       = True
        self.last_draw_pos = None
        self._draw_scribble(event.x, event.y)

    def _on_lb_move(self, event):
        if self.drawing:
            self._draw_scribble(event.x, event.y)

    def _on_lb_up(self, event):
        self.drawing       = False
        self.last_draw_pos = None
        self._commit_stroke()      # clear canvas items + full re-render

    def _draw_scribble(self, cx: int, cy: int):
        img_x, img_y = self._screen_to_image(cx, cy)
        ox = max(0, min(int(img_x), self.img_w - 1))
        oy = max(0, min(int(img_y), self.img_h - 1))

        # Convert previous image-space pos to screen coords for canvas line
        prev_cx = prev_cy = None
        if self.last_draw_pos is not None:
            lx, ly = self.last_draw_pos
            eff = self.base_scale * self.zoom_level
            prev_cx = int((lx - self.pan_offset_x) * eff)
            prev_cy = int((ly - self.pan_offset_y) * eff)

        if self.current_mode == self.FOREGROUND_MODE:
            self._draw_on_mask_interpolated(
                self.fg_scribble_mask, ox, oy, 1, self.brush_radius)
            color_hex = _C['green']
        else:
            self._draw_on_mask_interpolated(
                self.bg_scribble_mask, ox, oy, 1, self.brush_radius)
            color_hex = _C['red']

        # Instant canvas-item feedback — no PIL round-trip during drag
        self._draw_canvas_stroke(cx, cy, prev_cx, prev_cy, color_hex)
        self.last_draw_pos = (ox, oy)

        # Enable Re-run button once anything is drawn
        if self._rerun_btn and not self._rerun_enabled:
            self._rerun_enabled = True
            self._rerun_btn.configure(
                cursor='hand2', bg=_C['blue'], fg='white')
            self._rerun_btn.bind('<Button-1>', lambda e: self._do_rerun())
            self._rerun_btn.bind(
                '<Enter>', lambda e: self._rerun_btn.configure(bg='#0070d0'))
            self._rerun_btn.bind(
                '<Leave>', lambda e: self._rerun_btn.configure(bg=_C['blue']))

    # -- mode switches -----------------------------------------------------
    def _set_fg_mode(self):
        self.current_mode  = self.FOREGROUND_MODE
        self.last_draw_pos = None
        self._update_mode_buttons()
        self._set_hint('GREEN strokes — paint missed foreground areas.  '
                       'ENTER to re-run GrabCut.')

    def _set_bg_mode(self):
        self.current_mode  = self.BACKGROUND_MODE
        self.last_draw_pos = None
        self._update_mode_buttons()
        self._set_hint('RED strokes — paint false-positive areas.  '
                       'ENTER to re-run GrabCut.')

    # -- actions -----------------------------------------------------------
    def _do_rerun(self):
        has_fg = np.any(self.fg_scribble_mask)
        has_bg = np.any(self.bg_scribble_mask)
        if not (has_fg or has_bg):
            return
        print("    Re-running GrabCut with corrections...", end=" ", flush=True)
        if has_fg:
            self._gc_mask[self.fg_scribble_mask > 0] = cv2.GC_FGD
        if has_bg:
            self._gc_mask[self.bg_scribble_mask > 0] = cv2.GC_BGD
        cv2.grabCut(self.image, self._gc_mask, None,
                    self._bgdModel, self._fgdModel,
                    self._gc_iters, cv2.GC_INIT_WITH_MASK)
        print("Done", flush=True)
        # Save the committed GrabCut state so Clear can revert to it
        self._last_gc_mask  = self._gc_mask.copy()
        self._last_bgdModel = self._bgdModel.copy()
        self._last_fgdModel = self._fgdModel.copy()
        self.fg_scribble_mask[:] = 0
        self.bg_scribble_mask[:] = 0
        # Scribbles cleared — stroke undo history no longer meaningful
        self._stroke_undo_stack = []
        if self._rerun_btn:
            self._rerun_enabled = False
            self._rerun_btn.configure(
                cursor='arrow', bg=_C['btn'], fg=_C['text_dis'])
            self._rerun_btn.unbind('<Button-1>')
            self._rerun_btn.unbind('<Enter>')
            self._rerun_btn.unbind('<Leave>')
        self._overlay_cache = None
        self._refresh_display()

    def _undo(self):
        """Revert the most recent scribble stroke."""
        if not self._stroke_undo_stack:
            return
        fg_prev, bg_prev = self._stroke_undo_stack.pop()
        self.fg_scribble_mask[:] = fg_prev
        self.bg_scribble_mask[:] = bg_prev
        # Disable Re-run button if all strokes have been undone
        if not np.any(self.fg_scribble_mask) and not np.any(self.bg_scribble_mask):
            if self._rerun_btn:
                self._rerun_enabled = False
                self._rerun_btn.configure(
                    cursor='arrow', bg=_C['btn'], fg=_C['text_dis'])
                self._rerun_btn.unbind('<Button-1>')
                self._rerun_btn.unbind('<Enter>')
                self._rerun_btn.unbind('<Leave>')
        self._overlay_cache = None
        self._refresh_display()

    def _do_accept(self):
        self._result = (self._gc_mask, self._bgdModel, self._fgdModel)
        self.root.quit()
        self.root.destroy()

    def _do_clear(self):
        """Clear pending scribbles and revert GC state to the last Re-run result.

        Restores to the initial state if Re-run GrabCut has never been clicked.
        Use ↺ Restart to go all the way back to before any corrections.
        """
        self.fg_scribble_mask[:] = 0
        self.bg_scribble_mask[:] = 0
        self._stroke_undo_stack  = []
        # Restore to the last committed GrabCut state (not the very beginning)
        last_mask = self._last_gc_mask  if self._last_gc_mask  is not None else self._initial_mask
        last_bgd  = self._last_bgdModel if self._last_bgdModel is not None else self._initial_bgd
        last_fgd  = self._last_fgdModel if self._last_fgdModel is not None else self._initial_fgd
        self._gc_mask[:]  = last_mask
        self._bgdModel[:] = last_bgd
        self._fgdModel[:] = last_fgd
        if self._rerun_btn:
            self._rerun_enabled = False
            self._rerun_btn.configure(
                cursor='arrow', bg=_C['btn'], fg=_C['text_dis'])
            self._rerun_btn.unbind('<Button-1>')
            self._rerun_btn.unbind('<Enter>')
            self._rerun_btn.unbind('<Leave>')
        self._overlay_cache = None
        self._refresh_display()

    def _do_back(self):
        self._result = None
        self.root.quit()
        self.root.destroy()

    # -- public API (same as old class) ------------------------------------
    def correct(self, mask: np.ndarray,
                bgdModel: np.ndarray, fgdModel: np.ndarray,
                iterations: int = 5) -> Optional[Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """Show corrector; return (mask, bgd, fgd) or None on back/cancel."""
        self._gc_mask      = mask
        self._bgdModel     = bgdModel
        self._fgdModel     = fgdModel
        self._gc_iters     = iterations
        self._initial_mask = mask.copy()
        self._initial_bgd  = bgdModel.copy()
        self._initial_fgd  = fgdModel.copy()
        self._create_window(self._window_name)
        return self.run()


def apply_grabcut_with_paint_mask(
    image: np.ndarray,
    paint_mask: np.ndarray,
    iterations: int = 5,
    bgdModel: Optional[np.ndarray] = None,
    fgdModel: Optional[np.ndarray] = None
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Apply GrabCut using user-painted foreground pixels as initialization.

    Initializes a GrabCut mask where:
      - Painted pixels (paint_mask > 0) = GC_FGD (definite foreground)
      - Image border pixels = GC_BGD (definite background anchor)
      - All other pixels = GC_PR_BGD (probable background)

    Then runs GrabCut with GC_INIT_WITH_MASK.

    Args:
        image: BGR color image
        paint_mask: Binary mask from ForegroundPainter (255=foreground, 0=unmarked)
        iterations: Number of GrabCut iterations
        bgdModel: Optional pre-existing background model (for subsequent samples).
                  If None, initializes fresh zeros.
        fgdModel: Optional pre-existing foreground model (for subsequent samples).
                  If None, initializes fresh zeros.

    Returns:
        Tuple of (gc_mask, bgdModel, fgdModel)
    """
    h, w = image.shape[:2]

    # Initialize mask: everything is probable background
    gc_mask = np.full((h, w), cv2.GC_PR_BGD, dtype=np.uint8)

    # Painted pixels are definite foreground
    gc_mask[paint_mask > 0] = cv2.GC_FGD

    # Add definite background border to give GrabCut clear background anchors
    border = max(5, int(min(h, w) * 0.02))
    gc_mask[0:border, :] = cv2.GC_BGD
    gc_mask[h-border:h, :] = cv2.GC_BGD
    gc_mask[:, 0:border] = cv2.GC_BGD
    gc_mask[:, w-border:w] = cv2.GC_BGD

    if bgdModel is None:
        bgdModel = np.zeros((1, 65), dtype=np.float64)
    else:
        bgdModel = bgdModel.copy()
    if fgdModel is None:
        fgdModel = np.zeros((1, 65), dtype=np.float64)
    else:
        fgdModel = fgdModel.copy()

    for i in range(1, iterations + 1):
        mode = cv2.GC_INIT_WITH_MASK if i == 1 else cv2.GC_EVAL
        cv2.grabCut(image, gc_mask, None, bgdModel, fgdModel, 1, mode)
        print(f"    Iteration {i}/{iterations} complete", flush=True)

    return gc_mask, bgdModel, fgdModel


def _classify_pixels_by_gmm(
    image: np.ndarray,
    bgdModel: np.ndarray,
    fgdModel: np.ndarray,
) -> np.ndarray:
    """Produce a GrabCut mask initialised purely from GMM colour likelihoods.

    For each pixel, computes log-likelihood under the 5-component foreground
    and background Gaussian Mixture Models stored in the OpenCV GrabCut model
    arrays.  Pixels with higher FG likelihood are marked GC_PR_FGD; others
    GC_PR_BGD.  A narrow definite-background border anchors the subsequent
    graph-cut optimisation.

    OpenCV stores each GMM component as 13 consecutive float64 values:
        [weight, mean_B, mean_G, mean_R, cov_00..cov_22]

    This function uses only colour information — no spatial/positional data.
    Processes the image in 128-row chunks to limit peak memory usage.

    Args:
        image:    Full-resolution BGR image.
        bgdModel: Background GMM model (1×65 float64) from training.
        fgdModel: Foreground GMM model (1×65 float64) from training.

    Returns:
        uint8 GrabCut mask with values in {GC_PR_FGD, GC_PR_BGD, GC_BGD}.
    """
    h, w = image.shape[:2]
    _LOG2PI3 = 3.0 * np.log(2.0 * np.pi)

    def _precompute(model: np.ndarray) -> list:
        """Pre-invert covariance matrices once per model component."""
        params = model.ravel()
        comps = []
        for k in range(5):
            base = k * 13
            wt = params[base]
            if wt <= 0.0:
                continue
            mu  = params[base + 1 : base + 4]
            cov = params[base + 4 : base + 13].reshape(3, 3).copy()
            cov += np.eye(3, dtype=np.float64) * 1e-6
            try:
                sign, ldet = np.linalg.slogdet(cov)
                if sign <= 0:
                    continue
                comps.append((np.log(wt), mu, np.linalg.inv(cov), ldet))
            except np.linalg.LinAlgError:
                continue
        return comps

    def _log_ll_chunk(px: np.ndarray, comps: list) -> np.ndarray:
        """Vectorised GMM log-likelihood for a chunk of BGR pixels."""
        log_p = np.full(len(px), -np.inf)
        for log_wt, mu, cov_inv, ldet in comps:
            diff  = px - mu
            mah   = np.einsum('ni,ij,nj->n', diff, cov_inv, diff)
            log_c = log_wt - 0.5 * (_LOG2PI3 + ldet + mah)
            log_p = np.logaddexp(log_p, log_c)
        return log_p

    fg_comps = _precompute(fgdModel)
    bg_comps = _precompute(bgdModel)

    ROWS = 128
    mask = np.empty((h, w), dtype=np.uint8)
    for r0 in range(0, h, ROWS):
        r1 = min(r0 + ROWS, h)
        chunk = image[r0:r1].reshape(-1, 3).astype(np.float64)
        if fg_comps and bg_comps:
            fg_ll  = _log_ll_chunk(chunk, fg_comps)
            bg_ll  = _log_ll_chunk(chunk, bg_comps)
            labels = np.where(fg_ll > bg_ll, cv2.GC_PR_FGD, cv2.GC_PR_BGD)
        else:
            labels = np.full(len(chunk), cv2.GC_PR_BGD)
        mask[r0:r1] = labels.astype(np.uint8).reshape(r1 - r0, w)

    # Definite-background border anchors the GrabCut graph-cut
    border = max(5, int(min(h, w) * 0.02))
    mask[:border, :]  = cv2.GC_BGD
    mask[-border:, :] = cv2.GC_BGD
    mask[:, :border]  = cv2.GC_BGD
    mask[:, -border:] = cv2.GC_BGD
    return mask


def classify_with_frozen_gmm(
    image: np.ndarray,
    bgdModel: np.ndarray,
    fgdModel: np.ndarray,
    iterations: int = 1,
) -> np.ndarray:
    """
    Classify image pixels using pre-trained GMM models and return a binary mask.

    Step 1: Each pixel is assigned to foreground or background based on which
    GMM (FG or BG) gives a higher log-likelihood.

    Step 2: If iterations > 0, cv2.grabCut is run with GC_EVAL_FREEZE_MODEL,
    which applies graph-cut spatial smoothing WITHOUT re-estimating the GMMs.
    This preserves the training-learned models exactly — no drift occurs.

    Args:
        image:      BGR colour image (full resolution)
        bgdModel:   Background GMM model (1×65 float64) — not mutated
        fgdModel:   Foreground GMM model (1×65 float64) — not mutated
        iterations: Graph-cut refinement iterations (0 = skip, pure GMM only)

    Returns:
        Binary mask (uint8): 255 = foreground, 0 = background
    """
    gc_mask = _classify_pixels_by_gmm(image, bgdModel, fgdModel)

    if iterations > 0:
        bgd_work = bgdModel.copy()
        fgd_work = fgdModel.copy()
        cv2.grabCut(image, gc_mask, None, bgd_work, fgd_work,
                    iterations, cv2.GC_EVAL_FREEZE_MODEL)

    return extract_binary_mask(gc_mask)


def compute_template_rectangle(sample_masks: List[np.ndarray],
                                image_shape: Tuple[int, int],
                                margin: float = 0.1,
                                min_size: int = 50) -> Tuple[int, int, int, int]:
    """
    Compute average bounding box from sample masks for propagation.

    This creates a template rectangle that constrains GrabCut propagation to the
    object region, dramatically improving performance (80-100x speedup).

    Args:
        sample_masks: List of binary masks from sample images (255=foreground, 0=background)
        image_shape: (height, width) of images to clamp rectangle bounds
        margin: Safety margin as fraction of bounding box size (default 0.1 = 10%)
        min_size: Minimum rectangle dimension in pixels (default 50)

    Returns:
        Tuple (x, y, w, h) suitable for cv2.GC_INIT_WITH_RECT

    Raises:
        ValueError: If no foreground pixels found in any sample mask
    """
    bounding_boxes = []
    h_img, w_img = image_shape

    for mask in sample_masks:
        # Find foreground (non-zero) pixels
        coords = np.argwhere(mask > 0)
        if len(coords) > 0:
            y_min, x_min = coords.min(axis=0)
            y_max, x_max = coords.max(axis=0)
            bounding_boxes.append((x_min, y_min, x_max - x_min + 1, y_max - y_min + 1))

    if not bounding_boxes:
        raise ValueError("No foreground pixels found in sample masks")

    # Average the bounding boxes
    avg_x = int(np.mean([bb[0] for bb in bounding_boxes]))
    avg_y = int(np.mean([bb[1] for bb in bounding_boxes]))
    avg_w = int(np.mean([bb[2] for bb in bounding_boxes]))
    avg_h = int(np.mean([bb[3] for bb in bounding_boxes]))

    # Add margin for safety
    margin_w = int(avg_w * margin)
    margin_h = int(avg_h * margin)

    final_x = max(0, avg_x - margin_w)
    final_y = max(0, avg_y - margin_h)
    final_w = min(w_img - final_x, avg_w + 2 * margin_w)
    final_h = min(h_img - final_y, avg_h + 2 * margin_h)

    # Ensure minimum size
    final_w = max(min_size, final_w)
    final_h = max(min_size, final_h)

    return (final_x, final_y, final_w, final_h)




def extract_binary_mask(grabcut_mask: np.ndarray) -> np.ndarray:
    """
    Extract binary mask from GrabCut output.

    GrabCut mask values:
        0 = GC_BGD (definite background)
        1 = GC_FGD (definite foreground)
        2 = GC_PR_BGD (probable background)
        3 = GC_PR_FGD (probable foreground)

    Binary mask: Keep pixels classified as foreground (1 or 3)

    Args:
        grabcut_mask: GrabCut output mask

    Returns:
        Binary mask with 0=background, 255=foreground
    """
    # Keep GC_FGD (1) and GC_PR_FGD (3)
    binary_mask = np.where((grabcut_mask == 1) | (grabcut_mask == 3), 255, 0).astype('uint8')
    return binary_mask


def keep_largest_component(mask: np.ndarray) -> np.ndarray:
    """
    Retain only the largest foreground connected component.

    Applied unconditionally after GMM classification to ensure stray blobs
    (e.g. background regions mislabelled as foreground) are discarded before
    any subsequent hole-filling or expansion.  Does NOT apply morphological
    opening — the full object silhouette is preserved at pixel level.

    Args:
        mask: Binary mask (0=background, 255=foreground)

    Returns:
        Mask containing only the largest foreground component.
        Falls back to the original if no foreground is found.
    """
    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask, connectivity=8, ltype=cv2.CV_32S
    )
    if num_labels <= 1:
        return mask  # no foreground

    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + int(areas.argmax())
    result = (labels == largest_label).astype('uint8') * 255

    if cv2.countNonZero(result) == 0:
        return mask
    return result


def clean_mask(mask: np.ndarray, kernel_size: int = 5) -> np.ndarray:
    """
    Remove small noise blobs from binary mask using morphological opening
    followed by largest connected component extraction.

    Uses connectedComponentsWithStats for pixel-perfect component selection
    with exact area counts and 8-connectivity.

    Args:
        mask: Binary mask (0=background, 255=foreground)
        kernel_size: Diameter of elliptical kernel for morphological opening (default 5)

    Returns:
        Cleaned binary mask. Falls back to original if cleanup removes all foreground.
    """
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    opened = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(
        opened, connectivity=8, ltype=cv2.CV_32S
    )

    # Only background label exists — fall back to original
    if num_labels <= 1:
        return mask

    # Find largest foreground component (label 0 is background, skip it)
    areas = stats[1:, cv2.CC_STAT_AREA]
    largest_label = 1 + areas.argmax()
    result = (labels == largest_label).astype('uint8') * 255

    if cv2.countNonZero(result) == 0:
        return mask

    return result


def fill_mask_holes(mask: np.ndarray) -> np.ndarray:
    """
    Fill interior background holes completely enclosed by foreground.

    Uses connected-component analysis on the inverted mask to identify
    background regions unreachable from the image border, then fills them.
    Handles colour-ambiguous regions (e.g. white text against a white
    background) that GrabCut cannot distinguish by colour alone.

    Args:
        mask: Binary mask (0=background, 255=foreground)

    Returns:
        Mask with interior holes filled. Returns original if no holes found.
    """
    inv = cv2.bitwise_not(mask)   # FG=0, BG=255
    n_labels, labels = cv2.connectedComponents(inv, connectivity=8)

    # Collect all component labels that touch the image border (external BG)
    border_labels = set()
    border_labels.update(labels[0, :].tolist())
    border_labels.update(labels[-1, :].tolist())
    border_labels.update(labels[:, 0].tolist())
    border_labels.update(labels[:, -1].tolist())

    # Any BG component not touching the border is an interior hole
    holes = np.zeros_like(mask)
    for label in range(1, n_labels):
        if label not in border_labels:
            holes[labels == label] = 255

    if cv2.countNonZero(holes) == 0:
        return mask
    return cv2.bitwise_or(mask, holes)


def expand_mask(mask: np.ndarray, expansion_pixels: int = 3) -> np.ndarray:
    """
    Expand mask region by specified number of pixels using dilation.

    Args:
        mask: Binary mask (0=background, 255=foreground)
        expansion_pixels: Number of pixels to expand (default 3)

    Returns:
        Expanded binary mask
    """
    # Create structuring element (kernel)
    # For 3-pixel expansion, use 7x7 kernel (3 pixels in each direction)
    kernel_size = 2 * expansion_pixels + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))

    # Dilate mask
    expanded = cv2.dilate(mask, kernel, iterations=1)

    return expanded


def compute_weighted_gmm(
    img_idx: int,
    sample_indices: List[int],
    per_sample_bgd: dict,
    per_sample_fgd: dict,
) -> Tuple[np.ndarray, np.ndarray]:
    """Compute inverse-distance-weighted average GMM models for a non-training image.

    All training images contribute to the blend. Each training image k receives
    weight = 1 / (1 + |img_idx - k|), so images closest in sequence index
    (closest rotation angle) get the most influence, while all images still
    contribute a meaningful global average.

    Weights are normalised to sum to 1.0 before computing the weighted average.

    Args:
        img_idx: 0-based index of the image being processed
        sample_indices: List of training image indices (0-based)
        per_sample_bgd: Dict mapping training image index to BGD GMM model (1×65)
        per_sample_fgd: Dict mapping training image index to FGD GMM model (1×65)

    Returns:
        Tuple (blended_bgdModel, blended_fgdModel) as new (1×65 float64) arrays
    """
    weights = np.array([1.0 / (1.0 + abs(img_idx - s)) for s in sample_indices],
                       dtype=np.float64)
    weights /= weights.sum()

    bgd = sum(w * per_sample_bgd[s] for w, s in zip(weights, sample_indices))
    fgd = sum(w * per_sample_fgd[s] for w, s in zip(weights, sample_indices))
    return bgd, fgd


class MaskReviewer(TkViewport):
    """
    Fullscreen window for reviewing masks with navigation, zoom/pan,
    and view-mode switching.

    Toolbar controls:
      ◀ Prev / Next ▶   — ← / → or A / D
      View selector      — All 3 panels | Overlay only | Mask only | Original
      ✕ Reject & Retry   — N
      ✓ Accept All       — Y
        Quit             — Q / ESC
      Zoom − % + Fit     — scroll wheel
    """

    VIEW_ALL3     = 'all3'
    VIEW_OVERLAY  = 'overlay'
    VIEW_MASK     = 'mask'
    VIEW_ORIGINAL = 'original'

    def __init__(self, image_files: List[Path], mask_dir: Path,
                 max_display_height: int = 800, use_fullscreen: bool = True,
                 fallback_width: int = 1800, fallback_height: int = 600,
                 zoom_max: float = 10.0, zoom_step_factor: float = 1.15):
        self.image_files        = image_files
        self.mask_dir           = mask_dir
        self.current_index      = 0
        self.max_display_height = max_display_height
        self._view_mode         = self.VIEW_ALL3

        # Toolbar widget refs (set in _build_toolbar)
        self._counter_var  = None
        self._filename_var = None
        self._view_btns    = {}

        # Build first composite to give TkViewport a valid image
        first_display = self._build_view(0)
        if first_display is None:
            raise ValueError("No valid masks found to review")

        super().__init__(first_display,
                         brush_radius=0, min_brush=0, max_brush=0,
                         use_fullscreen=use_fullscreen,
                         fallback_width=fallback_width,
                         fallback_height=fallback_height,
                         zoom_max=zoom_max,
                         zoom_step_factor=zoom_step_factor)
        # No painting in review mode — keep native cursor, skip cursor ring
        self._use_cursor_ring = False

    # -- image loading ----------------------------------------------------
    def _load_pair(self, index: int):
        """Return (image_bgr, mask_gray) or (None, None) if unavailable."""
        img_path  = self.image_files[index]
        mask_path = self.mask_dir / f"{img_path.stem}_mask.png"
        if not mask_path.exists():
            return None, None
        image = load_image_color(img_path)
        mask  = cv2.imread(str(mask_path), cv2.IMREAD_GRAYSCALE)
        if mask is None:
            return None, None
        h, w = image.shape[:2]
        if h > self.max_display_height:
            scale = self.max_display_height / h
            nw    = int(w * scale)
            image = cv2.resize(image, (nw, self.max_display_height))
            mask  = cv2.resize(mask,  (nw, self.max_display_height))
        return image, mask

    def _build_view(self, index: int) -> Optional[np.ndarray]:
        """Build display image for current view mode at *index*."""
        image, mask = self._load_pair(index)
        if image is None:
            return None
        overlay = image.copy()
        overlay[mask == 255] = (
            overlay[mask == 255] * 0.5 + np.array([0, 200, 0]) * 0.5
        ).astype('uint8')
        if self._view_mode == self.VIEW_OVERLAY:
            return overlay
        elif self._view_mode == self.VIEW_MASK:
            return cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        elif self._view_mode == self.VIEW_ORIGINAL:
            return image
        else:  # VIEW_ALL3
            return np.hstack([image, cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR), overlay])

    def _compose_overlay(self) -> np.ndarray:
        return self.image.copy()

    # -- toolbar ----------------------------------------------------------
    def _build_toolbar(self):
        tb = self.toolbar

        # Right-side actions FIRST — reserves space before left items are placed
        right = tk.Frame(tb, bg=_C['toolbar'])
        right.pack(side=tk.RIGHT, padx=8)
        self._add_zoom_controls(right)
        self._sep(right)
        self._btn(right, 'Quit',              self._do_quit,   fg=_C['dim'])
        self._btn(right, '✕  Reject & Retry', self._do_reject,
                  bg=_C['red'], fg='white')
        self._btn(right, '✓  Accept All',     self._do_accept,
                  bg=_C['green'], fg='black')

        self._btn(tb, '◀  Prev', lambda: self._navigate(-1))

        self._counter_var = tk.StringVar(
            value=f"  {self.current_index + 1} / {len(self.image_files)}  ")
        tk.Label(tb, textvariable=self._counter_var,
                 bg=_C['toolbar'], fg=_C['text'],
                 font=('SF Pro Display', 12, 'bold')).pack(side=tk.LEFT)

        self._filename_var = tk.StringVar(value=self.image_files[0].name)
        tk.Label(tb, textvariable=self._filename_var,
                 bg=_C['toolbar'], fg=_C['dim'],
                 font=('SF Pro Display', 11)).pack(side=tk.LEFT, padx=(4, 8))

        self._btn(tb, 'Next  ▶', lambda: self._navigate(1))
        self._sep(tb)

        # View-mode segmented control
        self._lbl(tb, 'View:', fg=_C['dim'], font=('SF Pro Display', 11))
        for label, mode in [('All 3',    self.VIEW_ALL3),
                             ('Overlay',  self.VIEW_OVERLAY),
                             ('Mask',     self.VIEW_MASK),
                             ('Original', self.VIEW_ORIGINAL)]:
            b = tk.Label(tb, text=label,
                         font=('SF Pro Display', 11),
                         relief=tk.FLAT, borderwidth=0,
                         padx=8, pady=5, cursor='hand2',
                         bg=_C['btn'], fg=_C['text'])
            b.bind('<Button-1>', lambda e, m=mode: self._set_view(m))
            b.pack(side=tk.LEFT, padx=2, pady=8)
            self._view_btns[mode] = b
        self._refresh_view_buttons()
        self._sep(tb)

        # Context hint (last — gets clipped first on narrow toolbars)
        self._add_hint_label(
            tb, '← → navigate  ·  Y accept all  ·  N retry  ·  scroll zoom  ·  ↑↓ pan')

    def _refresh_view_buttons(self):
        for mode, btn in self._view_btns.items():
            if mode == self._view_mode:
                btn.configure(bg=_C['blue'], fg='white')
            else:
                btn.configure(bg=_C['btn'], fg=_C['text'])

    def _set_view(self, mode: str):
        self._view_mode = mode
        self._refresh_view_buttons()
        view = self._build_view(self.current_index)
        if view is not None:
            self.image = view
            self.img_h, self.img_w = view.shape[:2]
            self._compute_base_scale()
            self._overlay_cache = None
            self._refresh_display()

    def _bind_keys(self):
        self.root.bind('<Left>',     lambda e: self._navigate(-1))
        self.root.bind('<Right>',    lambda e: self._navigate(1))
        self.root.bind('a',          lambda e: self._navigate(-1))
        self.root.bind('d',          lambda e: self._navigate(1))
        self.root.bind('y',          lambda e: self._do_accept())
        self.root.bind('n',          lambda e: self._do_reject())
        self.root.bind('q',          lambda e: self._do_quit())
        self.root.bind('<Escape>',   lambda e: self._do_quit())

    def _bind_common_keys(self):
        # ← → navigate images in review mode — only bind ↑↓ for vertical pan
        self.root.bind('<Up>',   lambda e: self._pan_arrow(0, -40))
        self.root.bind('<Down>', lambda e: self._pan_arrow(0,  40))

    def _navigate(self, delta: int):
        """Navigate to a different image, skipping missing masks."""
        start = self.current_index
        while True:
            self.current_index = (self.current_index + delta) % len(self.image_files)
            composite = self._build_view(self.current_index)
            if composite is not None:
                break
            if self.current_index == start:
                print("\nError: No valid masks found. Cannot review.")
                raise ValueError("No valid masks to review")

        # Update image and viewport
        self.image = composite
        self.img_h, self.img_w = composite.shape[:2]
        self.zoom_level   = 1.0
        self.pan_offset_x = 0.0
        self.pan_offset_y = 0.0
        self._compute_base_scale()
        self._update_zoom_label()
        self._overlay_cache = None
        self._refresh_display()

        # Sync toolbar labels
        if self._counter_var is not None:
            self._counter_var.set(
                f"  {self.current_index + 1} / {len(self.image_files)}  ")
        if self._filename_var is not None:
            self._filename_var.set(self.image_files[self.current_index].name)

    # -- actions ----------------------------------------------------------
    def _do_accept(self):
        self._result = 'accept'
        self.root.quit()
        self.root.destroy()

    def _do_reject(self):
        self._result = 'reject'
        self.root.quit()
        self.root.destroy()

    def _do_quit(self):
        self._result = 'quit'
        self.root.quit()
        self.root.destroy()

    # -- public API -------------------------------------------------------
    def review(self) -> str:
        """Open review window; return 'accept', 'reject', or 'quit'."""
        self._create_window("GrabCut Mask Review")
        return self.run()


def process_grabcut_extraction(folder_path: Path, config: dict) -> bool:
    """
    Main processing function with retry logic.

    Args:
        folder_path: Path to folder containing images
        config: Merged configuration dictionary (from get_default_config + YAML + CLI)

    Returns:
        True if successful, False if user quit
    """
    # Show instructions popup if enabled
    if config.get("show_instructions_popup", True):
        show_instructions_dialog()

    # Load all image files
    extensions = tuple(config["supported_extensions"])
    image_files = get_image_files(folder_path, extensions)
    if not image_files:
        raise ValueError(f"No images found in {folder_path}")

    print(f"Found {len(image_files)} images")

    train_iters = config["train_iterations"]
    propagate_iters = config["propagate_iterations"]
    retry_count = 0

    # State accumulated across retries — persists through reject+retry cycles
    locked_indices: List[int] = []   # indices whose GMMs are already trained
    accumulated_bgd: dict = {}       # {image_idx: bgdModel} from all prior runs
    accumulated_fgd: dict = {}       # {image_idx: fgdModel} from all prior runs
    accumulated_masks: dict = {}     # {image_idx: binary_mask} from all prior runs

    while retry_count <= config["max_retries"]:
        # Phase 1: User selects training images
        sample_indices = select_sample_images(image_files, config, locked_indices=locked_indices)
        if sample_indices is None:
            print("\nImage selection cancelled. Exiting.")
            return False
        w = len(sample_indices)

        print(f"\n{'='*60}")
        print(f"ATTEMPT {retry_count + 1}: {w} images selected, train={train_iters} iters, propagate={propagate_iters} iters")
        print(f"{'='*60}")

        print(f"\nSelected {w} training images:")
        for idx in sample_indices:
            print(f"  - Image {idx + 1}: {image_files[idx].name}")

        try:
            # Phase 2: Train on samples with interactive painting
            new_count = sum(1 for idx in sample_indices if idx not in accumulated_bgd)
            print(f"\nPhase 2: Training on {w} samples "
                  f"({len(accumulated_bgd)} reused, {new_count} new)")

            # Pre-load GMMs and masks from prior runs; new images will be added below
            per_sample_bgd: dict = accumulated_bgd.copy()
            per_sample_fgd: dict = accumulated_fgd.copy()
            mask_by_idx: dict = dict(accumulated_masks)   # {image_idx: binary_mask}

            for i, idx in enumerate(sample_indices):
                img_path = image_files[idx]

                if idx in accumulated_bgd:
                    # GMM already trained in a previous run — skip interactive training
                    print(f"\n[Sample {i+1}/{w}] {img_path.name} — reusing saved GMM")
                    continue

                print(f"\n[Sample {i+1}/{w}] {img_path.name}")

                # Load full-resolution image
                image_full = load_image_color(img_path)
                full_shape = image_full.shape[:2]

                # Compute downscale factor for training
                training_scale = compute_training_scale(
                    full_shape, config["training_max_dimension"]
                )
                image_small = downscale_image(image_full, training_scale)

                if training_scale < 1.0:
                    print(f"  Downscaled {full_shape[1]}x{full_shape[0]} -> "
                          f"{image_small.shape[1]}x{image_small.shape[0]} "
                          f"(scale={training_scale:.3f})")

                # Per-sample loop: draw FG/BG scribbles -> Re-run GrabCut -> accept
                while True:
                    # Initialise a fresh GrabCut mask: probable-background everywhere,
                    # with a definite-background border as a GrabCut anchor.
                    h_s, w_s = image_small.shape[:2]
                    gc_mask = np.full((h_s, w_s), cv2.GC_PR_BGD, dtype=np.uint8)
                    border = max(5, int(min(h_s, w_s) * 0.02))
                    gc_mask[0:border, :]       = cv2.GC_BGD
                    gc_mask[h_s-border:h_s, :] = cv2.GC_BGD
                    gc_mask[:, 0:border]        = cv2.GC_BGD
                    gc_mask[:, w_s-border:w_s]  = cv2.GC_BGD
                    # Fresh models for each sample — no cross-contamination between images
                    bgd_init = np.zeros((1, 65), dtype=np.float64)
                    fgd_init = np.zeros((1, 65), dtype=np.float64)

                    corrector_brush = max(1, int(config["refine_brush_radius"] * training_scale))
                    corrector = MaskCorrector(
                        f"Sample {i+1}/{w}: {img_path.name}",
                        image_small,
                        brush_radius=corrector_brush,
                        use_fullscreen=config.get("use_fullscreen", True),
                        fallback_width=config["selector_window_width"],
                        fallback_height=config["selector_window_height"],
                        min_brush=config.get("brush_min_radius", 2),
                        max_brush=config.get("brush_max_radius", 100),
                        zoom_max=config.get("zoom_max", 10.0),
                        zoom_step_factor=config.get("zoom_step_factor", 1.15),
                        sample_num=i + 1,
                        total_samples=w,
                    )
                    result = corrector.correct(
                        gc_mask, bgd_init, fgd_init,
                        iterations=config["refine_iterations"]
                    )

                    if result is None:
                        # Restart — reopen corrector for this sample from scratch
                        print("  Restarting sample...")
                        continue

                    gc_mask, trained_bgd, trained_fgd = result
                    per_sample_bgd[idx] = trained_bgd.copy()
                    per_sample_fgd[idx] = trained_fgd.copy()
                    break  # Sample accepted

                # Step 4: Upscale mask to full resolution and extract binary
                if training_scale < 1.0:
                    gc_mask_full = upscale_mask(gc_mask, full_shape)
                else:
                    gc_mask_full = gc_mask

                binary_mask = extract_binary_mask(gc_mask_full)
                if config["mask_cleanup"]:
                    binary_mask = clean_mask(binary_mask, kernel_size=config["mask_cleanup_kernel_size"])
                if config.get("mask_fill_holes", True):
                    binary_mask = fill_mask_holes(binary_mask)
                expanded_mask = expand_mask(binary_mask, expansion_pixels=config["expansion_pixels"])
                mask_by_idx[idx] = expanded_mask

                print(f"  Sample {i+1} accepted")

            # Build lookup: image_index → accepted mask for Phase 4 ground-truth bypass
            sample_mask_lookup = {idx: mask_by_idx[idx] for idx in sample_indices}

            print(f"\nPhase 3: Computing per-image GMMs from {len(per_sample_bgd)} training samples")
            print("  Weighting: closer training images get higher weight (inverse-distance)")

            first_image = load_image_color(image_files[0])
            img_h, img_w = first_image.shape[:2]
            total_pixels = img_h * img_w
            print(f"  Image dimensions: {img_w}×{img_h} ({total_pixels:,} pixels)")

            # Pre-compute inverse-distance-weighted GMMs for every non-training image.
            # Training images receive None placeholders — their accepted masks are used
            # directly in Phase 4 and GMM classification is skipped entirely.
            per_image_bgd: List[Optional[np.ndarray]] = []
            per_image_fgd: List[Optional[np.ndarray]] = []
            for _i in range(len(image_files)):
                if _i in sample_mask_lookup:
                    per_image_bgd.append(None)
                    per_image_fgd.append(None)
                else:
                    _b, _f = compute_weighted_gmm(
                        _i, sample_indices, per_sample_bgd, per_sample_fgd)
                    per_image_bgd.append(_b)
                    per_image_fgd.append(_f)

            # Warn and estimate time for large images
            if total_pixels > config["large_template_threshold"]:
                est_time_per_image = total_pixels / config["time_estimate_pixel_rate"]
                est_total_minutes = len(image_files) * est_time_per_image
                print(f"\n  WARNING: Large images detected ({total_pixels:,} pixels each)")
                print(f"  Estimated Phase 4 time: {est_total_minutes:.0f} minutes ({est_total_minutes/60:.1f} hours)")
                print(f"  Tip: Use --propagate-iterations 0 for faster (GMM-only) processing")
                print()

            # Phase 4: Propagate to all images
            print(f"\nPhase 4: Propagating to all {len(image_files)} images")
            temp_mask_dir = folder_path / ".temp_masks"
            temp_mask_dir.mkdir(exist_ok=True)

            failed_images = []

            # Determine parallelization strategy
            if config["use_parallel"]:
                cpu_count = os.cpu_count() or 1
                if config["worker_count"] is not None:
                    max_workers = config["worker_count"]
                else:
                    base_workers = max(1, cpu_count - 1)
                    if total_pixels > config["very_large_template_threshold"]:
                        max_workers = min(base_workers, config["very_large_template_max_workers"])
                        print(f"  Note: Using {max_workers} workers (images are very large)")
                    elif total_pixels > config["large_template_threshold"]:
                        max_workers = min(base_workers, config["large_template_max_workers"])
                        print(f"  Note: Using {max_workers} workers (images are large)")
                    else:
                        max_workers = min(len(image_files), base_workers)

                print(f"Using parallel processing with {max_workers} workers")

                img_paths_str = [str(p.resolve()) for p in image_files]
                temp_mask_dir_str = str(temp_mask_dir.resolve())

                with ProcessPoolExecutor(max_workers=max_workers) as executor:
                    future_to_img = {
                        executor.submit(
                            process_single_image_parallel,
                            img_path,
                            per_image_bgd[i],
                            per_image_fgd[i],
                            temp_mask_dir_str,
                            propagate_iters,
                            config["expansion_pixels"],
                            config.get("mask_fill_holes", True),
                            sample_mask_lookup.get(i),
                            config.get("mask_cleanup", True),
                            config.get("mask_cleanup_kernel_size", 5),
                        ): img_path
                        for i, img_path in enumerate(img_paths_str)
                    }

                    completed = 0
                    total = len(image_files)
                    start_time = time.time()

                    for future in as_completed(future_to_img):
                        completed += 1
                        img_name, success, error_msg = future.result()

                        elapsed = time.time() - start_time
                        rate = completed / elapsed if elapsed > 0 else 0
                        eta_seconds = (total - completed) / rate if rate > 0 else 0

                        if success:
                            print(f"[{completed}/{total}] {img_name}... Done ({rate:.1f} img/s, ETA: {eta_seconds:.0f}s)", flush=True)
                        else:
                            print(f"[{completed}/{total}] {img_name}... Error: {error_msg}", flush=True)
                            failed_images.append(img_name)

            else:
                print("Using sequential processing")

                for i, img_path in enumerate(image_files):
                    print(f"[{i+1}/{len(image_files)}] Processing {img_path.name}...", end=" ")

                    try:
                        seed_mask = sample_mask_lookup.get(i)
                        if seed_mask is not None:
                            expanded_mask = seed_mask
                        else:
                            image = load_image_color(img_path)
                            binary_mask = classify_with_frozen_gmm(
                                image, per_image_bgd[i], per_image_fgd[i], propagate_iters)
                            binary_mask = keep_largest_component(binary_mask)

                            if config.get("mask_cleanup", True):
                                binary_mask = clean_mask(
                                    binary_mask,
                                    kernel_size=config.get("mask_cleanup_kernel_size", 5))
                            if config.get("mask_fill_holes", True):
                                binary_mask = fill_mask_holes(binary_mask)
                            expanded_mask = expand_mask(binary_mask, expansion_pixels=config["expansion_pixels"])

                        mask_filename = f"{img_path.stem}_mask.png"
                        mask_path = temp_mask_dir / mask_filename
                        success = cv2.imwrite(str(mask_path), expanded_mask)

                        if not success:
                            raise ValueError(f"Failed to save mask to {mask_path}")

                        print("Done")

                    except Exception as e:
                        print(f"Error: {e}")
                        failed_images.append(img_path.name)
                        continue

            # Check if too many images failed
            if failed_images:
                print(f"\nWarning: {len(failed_images)} image(s) failed to process:")
                for name in failed_images[:5]:  # Show first 5
                    print(f"  - {name}")
                if len(failed_images) > 5:
                    print(f"  ... and {len(failed_images) - 5} more")

                if len(failed_images) == len(image_files):
                    print("\nError: All images failed to process. Cannot continue.")
                    # Clean up
                    for temp_file in temp_mask_dir.glob("*.png"):
                        temp_file.unlink()
                    try:
                        temp_mask_dir.rmdir()
                    except:
                        pass
                    return False

            # Phase 5: Review
            print(f"\nPhase 5: Review results")
            reviewer = MaskReviewer(
                image_files, temp_mask_dir,
                max_display_height=config["review_max_display_height"],
                use_fullscreen=config.get("use_fullscreen", True),
                fallback_width=config["review_window_width"],
                fallback_height=config["review_window_height"],
                zoom_max=config.get("zoom_max", 10.0),
                zoom_step_factor=config.get("zoom_step_factor", 1.15)
            )
            result = reviewer.review()

            if result == 'accept':
                # Move masks to final location
                print(f"\nAccepted! Saving masks...")
                for img_path in image_files:
                    temp_path = temp_mask_dir / f"{img_path.stem}_mask.png"
                    final_path = img_path.parent / f"{img_path.stem}_mask.png"
                    if temp_path.exists():
                        temp_path.rename(final_path)

                # Clean up temp directory
                try:
                    temp_mask_dir.rmdir()
                except:
                    pass

                print(f"Successfully saved {len(image_files)} masks")
                return True

            elif result == 'reject':
                # Persist trained GMMs and masks so the next retry reuses them
                locked_indices = list(sample_indices)
                accumulated_bgd.update(per_sample_bgd)
                accumulated_fgd.update(per_sample_fgd)
                accumulated_masks.update(mask_by_idx)

                # Clean up temp masks
                for temp_file in temp_mask_dir.glob("*.png"):
                    temp_file.unlink()
                try:
                    temp_mask_dir.rmdir()
                except Exception:
                    pass

                retry_count += 1
                print(f"\nRejected. Returning to image selection (attempt {retry_count + 1})...")

                continue

            else:  # quit
                # Clean up temp masks
                for temp_file in temp_mask_dir.glob("*.png"):
                    temp_file.unlink()
                try:
                    temp_mask_dir.rmdir()
                except:
                    pass

                print("\nQuitting without saving")
                return False

        except ValueError as e:
            print(f"\nError: {e}")
            return False
        except Exception as e:
            print(f"\nUnexpected error: {e}")
            import traceback
            traceback.print_exc()
            return False

    print(f"\nMaximum retries ({config['max_retries']}) reached")
    return False


def main():
    """Main entry point with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Interactive GrabCut-based masking for photogrammetry',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog='''
Examples:
  python GrabcutExtraction.py ./images
  python GrabcutExtraction.py ./photos --samples 5 --train-iterations 7
  python GrabcutExtraction.py ./dataset --workers 4
  python GrabcutExtraction.py ./photos --no-parallel
  python GrabcutExtraction.py ./photos --config my_config.yaml

Configuration:
  Settings are loaded in order: built-in defaults < config.yaml < CLI args.
  Place a config.yaml next to this script for automatic loading, or use
  --config to specify a custom path. CLI arguments always take priority.

Interactive Workflow:
  1. Script selects w sample images evenly across dataset
  2. User paints GREEN foreground strokes on each sample
  3. GrabCut generates mask; user corrects with GREEN/RED strokes
  4. User accepts each sample or re-paints; models accumulate
  5. Models are propagated to all images (parallel, full resolution)
  6. Masks are expanded and reviewed
  7. If rejected, samples and iterations increase, then retry

Masks are saved as [original_name]_mask.png in the same directory.
        '''
    )
    parser.add_argument(
        'folder',
        type=str,
        help='Path to folder containing image sequence'
    )
    parser.add_argument(
        '--config',
        type=str,
        default=None,
        metavar='PATH',
        help='Path to YAML config file (default: config.yaml next to script)'
    )
    parser.add_argument(
        '-w', '--samples',
        type=int,
        default=None,
        help='Initial number of sample images (default: from config or 3)'
    )
    parser.add_argument(
        '--train-iterations',
        type=int,
        default=None,
        help='GrabCut iterations for training samples (default: from config or 3)'
    )
    parser.add_argument(
        '--propagate-iterations',
        type=int,
        default=None,
        help='GrabCut iterations for mask propagation (default: from config or 1)'
    )
    parser.add_argument(
        '--max-retries',
        type=int,
        default=None,
        help='Maximum retry attempts (default: from config or 5)'
    )
    parser.add_argument(
        '--no-parallel',
        action='store_true',
        help='Disable parallel processing (use sequential mode)'
    )
    parser.add_argument(
        '--workers',
        type=int,
        default=None,
        metavar='N',
        help='Number of parallel workers (default: from config or CPU count - 1)'
    )
    parser.add_argument(
        '--training-max-dim',
        type=int,
        default=None,
        metavar='PX',
        help='Max image dimension during training (default: from config or 1200)'
    )

    args = parser.parse_args()

    # Build merged configuration: defaults <- YAML <- CLI args
    defaults = get_default_config()
    yaml_cfg = load_yaml_config(args.config)
    config = merge_config(defaults, yaml_cfg, args)

    # Validate workers argument
    if config["worker_count"] is not None:
        if config["worker_count"] < 1:
            print("Error: worker_count must be >= 1")
            sys.exit(1)
        if not config["use_parallel"]:
            print("Warning: worker_count specified but parallel processing is disabled. Ignoring worker_count.")

    # Validate folder
    folder_path = Path(args.folder).resolve()
    if not folder_path.is_dir():
        print(f"Error: Folder not found: {folder_path}")
        sys.exit(1)

    # Print header
    print("="*60)
    print("GrabCut Interactive Masking for Photogrammetry")
    print("="*60)
    print(f"Folder: {folder_path}")
    print(f"Initial samples (w): {config['initial_samples']}")
    print(f"Training iterations: {config['train_iterations']}")
    print(f"Propagation iterations: {config['propagate_iterations']}")
    print(f"Expansion pixels: {config['expansion_pixels']}")
    print(f"Parallel processing: {config['use_parallel']}")
    print(f"Training max dimension: {config['training_max_dimension']}px")
    print("="*60)

    # Run processing
    try:
        success = process_grabcut_extraction(folder_path, config)

        if success:
            print("\n" + "="*60)
            print("COMPLETE")
            print("="*60)
            print(f"Masks saved in: {folder_path}")
            sys.exit(0)
        else:
            print("\nProcessing was not completed")
            sys.exit(1)

    except KeyboardInterrupt:
        print("\n\nInterrupted by user")
        sys.exit(1)
    except Exception as e:
        print(f"\nFatal error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == '__main__':
    main()
