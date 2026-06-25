#!/usr/bin/env python3
"""
Video Tracking Module for FieldNeuroethologyToolbox

Tracking tools for behavioral tests using SAM2 (Segment Anything Model 2),
YOLOv11, and classical computer vision techniques.

Available trackers:
- Mask Tracker Tool (MTT) - SAM2-based annotation + YOLO/MaskRCNN training
- Simple Tracker - Classical blob tracking for open arena tests

Heavy dependencies (torch, sam2, ultralytics) are imported lazily inside
class methods, so this module is safe to import without them installed.
"""

__version__ = "0.1.0"
__author__ = "FieldNeuroethologyToolbox Contributors"

from pathlib import Path

# Module root directory
MODULE_DIR = Path(__file__).parent

__all__ = [
    'MODULE_DIR',
]
