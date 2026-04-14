# EduceLab Undergraduate Research

**Python tools developed during undergraduate research at [EduceLab](https://educelab.engr.uky.edu/)**, focused on computer vision and hardware control for photogrammetry and cultural heritage imaging pipelines.

---

## Repository Overview

This repository contains a suite of Python scripts built to support multi-spectral imaging and 3D photogrammetry workflows used in real research settings. The tools span hardware control, interactive computer vision, image processing, and batch automation.

| Script | Description |
|---|---|
| `MISHA.py` | Full-featured GUI desktop application for controlling a multi-wavelength LED illumination board over USB serial |
| `GrabcutExtraction.py` | Interactive, semi-automatic GrabCut segmentation tool for generating photogrammetry masks across large image sequences |
| `Mask.py` | Automated object masking for turntable photogrammetry using cross-image consistency detection |
| `MaskChanges.py` | Utility script for reviewing and modifying existing mask outputs |
| `SubtractionMask.py` | Background subtraction-based masking using reference images |
| `GrabcutExtractionConfig.yaml` | YAML configuration file for `GrabcutExtraction.py` with full parameter control |

---

## Featured Scripts

### MISHA.py — Multi-Wavelength LED Controller GUI

**653 lines · Python · tkinter + pyserial**

MISHA is a polished desktop GUI application built from scratch for controlling the MISHA multi-wavelength LED illumination board used in multi-spectral imaging experiments. The board supports 16 distinct LED wavelengths ranging from UV (365 nm) through visible light to near-infrared (940 nm).

#### What it does

The application sends real-time serial commands to an Arduino-based LED controller, allowing researchers to select a specific wavelength and set its output intensity — without touching any code.

#### Key technical features

**Custom UI component architecture.** Because macOS's Aqua theme overrides `tk.Button` background colors, a custom `_FlatButton` widget class was implemented from scratch using `tk.Frame` + `tk.Label` composition, with manual hover-state tracking via mouse bindings. This ensures the UI renders correctly and consistently across platforms.

**Physics-accurate wavelength visualization.** A `_nm_to_hex()` function maps all 16 supported wavelengths to approximate sRGB hex values, using perceptual luminance calculation (`0.299R + 0.587G + 0.114B`) to automatically choose readable foreground text color for each colored tile. UV wavelengths render as violet and IR as deep red.

**Thread-safe serial communication.** Serial reads run in a background daemon thread (`_rx_thread`) that streams Arduino responses to a live serial log window without blocking the UI. All log updates are marshaled to the main thread via `self.after()` to prevent race conditions. The connection teardown sequence signals the RX thread, sends a "turn off all LEDs" command, then closes the port — in that order — to prevent hardware state leakage.

**Intelligent port detection.** On connect, the app automatically scans available serial ports and prefers USB serial adapters (`usbserial`/`usbmodem`) — the typical profile of an Arduino — over other devices, saving the researcher from manually hunting for the correct port.

**Live intensity control with deferred serial writes.** The intensity slider updates the on-screen percentage label in real time via a `trace` callback, but only writes to serial on mouse release (`ButtonRelease`). This prevents flooding the Arduino with commands during a drag operation while still giving instant visual feedback.

**Per-tile active state and hover highlighting.** Each of the 16 wavelength tiles responds to hover and click independently, updating background color, text contrast, and border highlight while preserving the "active" state of any currently-selected wavelength.

#### Technologies

`tkinter`, `pyserial`, `threading`, `ttk.Style`, custom widget composition, USB serial protocol

---

### GrabcutExtraction.py — Interactive Segmentation Pipeline for Photogrammetry

**3,118 lines · Python · OpenCV + tkinter + Pillow + concurrent.futures**

GrabcutExtraction is the most complex script in the repository. It implements a full end-to-end interactive segmentation pipeline that uses OpenCV's GrabCut algorithm to generate binary masks for every image in a photogrammetry capture sequence — potentially hundreds or thousands of images — with minimal human effort.

#### The problem it solves

Photogrammetry requires clean masks isolating the subject from the background in every input image. Manual masking is impractical at scale. Fully automatic methods often fail on complex backgrounds or low-contrast objects. This tool finds a middle ground: the user annotates a small number of "training" images interactively, and the learned segmentation model is automatically propagated to the entire sequence.

#### Workflow

1. The user selects a small set of evenly-spaced sample images from the sequence (configurable; defaults to 3).
2. For each sample, a fullscreen interactive canvas opens. The user paints green strokes over the object (foreground).
3. GrabCut runs from those strokes, producing an initial binary mask.
4. The user corrects the mask using green (include) and red (exclude) brush strokes; GrabCut refines iteratively.
5. The blended GMM (Gaussian Mixture Model) is accumulated across all training samples.
6. The final GMM is propagated in parallel to every image in the sequence at full resolution.
7. Results are reviewed in a scrollable side-by-side comparison UI; the user can accept or retry with more samples.

#### Key technical features

**Frozen GMM propagation with spatial graph-cut.** Training images teach two Gaussian Mixture Models — one for foreground, one for background. For non-training images, `classify_with_frozen_gmm()` applies GMM likelihood scoring followed by optional graph-cut spatial smoothing using `GC_EVAL_FREEZE_MODEL` — the training GMMs are never re-estimated during propagation, preventing model drift across the sequence.

**Training/propagation resolution split.** Training runs on downscaled images (configurable max dimension, default 1200 px) for interactive speed. Propagation runs at full resolution. `compute_training_scale()` and `upscale_mask()` manage the coordinate transform transparently.

**Parallel batch processing with adaptive worker scaling.** Mask propagation uses `concurrent.futures.ProcessPoolExecutor` with worker count automatically scaled based on image pixel count: large images (>2 MP) get fewer workers to avoid memory pressure; very large images (>4 MP) fewer still. All propagation work is serializable via string paths rather than Path objects to ensure picklability across processes.

**Three-layer configuration system.** A `merge_config()` function implements a clean precedence chain: built-in defaults → YAML config file → CLI arguments. Unknown YAML keys are warned about and ignored. The optional PyYAML dependency is handled gracefully: if not installed, the script falls back to built-in defaults and prints a notice instead of crashing.

**Full-featured zoomable/pannable viewport base class.** `TkViewport` is an abstract base class that all interactive windows inherit from. It provides scroll-wheel zoom centered on the cursor, right-click drag panning, canvas resize handling, a dual-ring brush cursor (rendered as canvas items rather than a system cursor for precise visual feedback), and coordinate transforms between screen space and image space. Subclasses (`PaintingUI`, `CorrectionUI`, `MaskReviewUI`) layer their specific behavior on top.

**Multi-image sample selector with thumbnail grid.** A scrollable dark-themed thumbnail grid UI lets the user visually browse and select training images. Previously-locked (already-trained) images are highlighted in blue and cannot be deselected — ensuring previously invested training work is preserved across retry iterations.

**Mask post-processing pipeline.** After GrabCut, each mask goes through: largest-connected-component retention, morphological cleanup (opening to remove noise blobs), hole filling, and configurable edge dilation (`expansion_pixels`) to ensure object coverage.

#### Technologies

`cv2` (GrabCut, GMM, morphological ops), `tkinter`, `Pillow/ImageTk`, `concurrent.futures`, `argparse`, `PyYAML`, `numpy`, object-oriented viewport architecture

---

### Mask.py — Automated Turntable Masking via Cross-Image Consistency

**534 lines · Python · OpenCV + NumPy**

A batch processing script for masking objects on a turntable using a principled computer vision approach: rather than relying solely on backdrop subtraction, it detects pixels that are *static across the entire image sequence*. Any pixel that appears in the same location in every image is classified as background — regardless of whether it matches the reference backdrop. This elegantly handles turntable markings, stage lines, and other scene elements that simple backdrop differencing would miss.

The pipeline per image: grayscale cross-image consistency detection → intensity filtering (removes shadows) → contour-fill object boundary → morphological cleanup → edge smoothing → RGBA output with alpha channel for photogrammetry software compatibility.

#### Technologies

`cv2`, `numpy`, `argparse`, BGRA alpha-channel mask output

---

## Skills Demonstrated

- **Computer vision:** GrabCut segmentation, GMM-based pixel classification, morphological image processing, multi-image consistency detection, background subtraction
- **GUI application development:** Full tkinter desktop apps with custom widgets, threading, real-time serial communication, zoomable canvas viewports
- **Hardware interfacing:** USB serial protocol, Arduino communication, real-time LED control
- **Software architecture:** Abstract base classes, three-layer config systems, parallel processing pipelines, picklable cross-process data
- **Research engineering:** Designing tools that non-programmer researchers can use effectively, with robust error handling and configurable parameters

---

## Dependencies

```
opencv-python
numpy
Pillow
pyserial
PyYAML          # optional — used for GrabcutExtraction.py config files
```

Install with:
```bash
pip install -r requirements.txt
```

---

## Usage

```bash
# MISHA LED Controller GUI
python MISHA.py

# GrabCut interactive masking (basic)
python GrabcutExtraction.py <image_folder>

# GrabCut with options
python GrabcutExtraction.py <image_folder> --samples 5 --workers 4 --training-max-dim 800

# Turntable backdrop masking
python Mask.py <image_folder> <backdrop_image> <white|black>
```

---

*Developed as part of undergraduate research at [EduceLab, University of Kentucky](https://educelab.engr.uky.edu/) *
