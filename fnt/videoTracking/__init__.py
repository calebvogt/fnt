#!/usr/bin/env python3
"""
Video Tracking Module for FieldNeuroToolbox

Simple tracking tools for behavioral tests using SAM (Segment Anything Model)
and classical computer vision techniques.

Available trackers:
- Open Field Test (OFT) - Single or multi-animal tracking in open arena
- Light-Dark Box (LDB) - Tracking with occlusion handling for covered compartment

Each tracker is optimized for its specific behavioral paradigm.
"""

__version__ = "0.1.0"
__author__ = "FieldNeuroToolbox Contributors"

from pathlib import Path

# Module root directory
MODULE_DIR = Path(__file__).parent

# Check for dependencies
try:
    import cv2
    OPENCV_AVAILABLE = True
except ImportError:
    OPENCV_AVAILABLE = False
    print("Warning: OpenCV not available. Install with: pip install opencv-python")

try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    print("Warning: PyTorch not available. Install with: pip install torch")

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    print("Warning: Segment Anything not available. Install with: pip install segment-anything")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: Pandas not available. Install with: pip install pandas")

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False
    print("Warning: NumPy not available. Install with: pip install numpy")

# Check if all dependencies are available
ALL_DEPENDENCIES_AVAILABLE = (
    OPENCV_AVAILABLE and 
    TORCH_AVAILABLE and 
    SAM_AVAILABLE and 
    PANDAS_AVAILABLE and 
    NUMPY_AVAILABLE
)

if not ALL_DEPENDENCIES_AVAILABLE:
    print("\nVideo Tracking requires additional dependencies.")
    print("Install with: pip install opencv-python torch segment-anything pandas numpy")

__all__ = [
    'MODULE_DIR',
    'OPENCV_AVAILABLE',
    'TORCH_AVAILABLE', 
    'SAM_AVAILABLE',
    'PANDAS_AVAILABLE',
    'NUMPY_AVAILABLE',
    'ALL_DEPENDENCIES_AVAILABLE'
]
