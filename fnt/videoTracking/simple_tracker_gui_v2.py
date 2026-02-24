#!/usr/bin/env python3
"""
Simple Tracker GUI V3 - Movement-Based Blob Detection with ROI Analysis

Fast, CPU-only tracking using background subtraction for static camera setups.
Features blob refinement, ROI analysis, and comprehensive export pipeline.

Key features:
- Background subtraction (MOG2) with configurable blob refinement
- Multi-object centroid tracking with Hungarian matching
- Single-animal mode (largest blob only)
- ROI drawing (OFT, LDB, Custom) with scale bar calibration
- Toggle between background subtraction view and standard view
- Batch processing with per-video configuration
- Comprehensive exports: CSVs, plots, tracked/data view video

Author: FieldNeuroToolbox Contributors
"""

import sys
import os
import json
import cv2
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QMessageBox,
    QGroupBox, QSpinBox, QDoubleSpinBox, QListWidget, QListWidgetItem,
    QSlider, QScrollArea, QCheckBox, QTableWidget, QTableWidgetItem,
    QInputDialog, QFrame
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QRect
from PyQt5.QtGui import QImage, QPixmap, QFont, QPainter, QColor, QPen, QBrush, QPainterPath

# Check for required dependencies
try:
    import cv2
    CV2_AVAILABLE = True
except ImportError:
    CV2_AVAILABLE = False
    print("Warning: OpenCV not available")

try:
    import pandas as pd
    PANDAS_AVAILABLE = True
except ImportError:
    PANDAS_AVAILABLE = False
    print("Warning: pandas not available")

try:
    from scipy.optimize import linear_sum_assignment
    SCIPY_AVAILABLE = True
except ImportError:
    SCIPY_AVAILABLE = False
    print("Warning: scipy not available - using simple nearest neighbor matching")


# ═══════════════════════════════════════════════════════════════════════════════
# VideoTrackingConfig - Per-video configuration
# ═══════════════════════════════════════════════════════════════════════════════

class VideoTrackingConfig:
    """Configuration for a single video's tracking and ROI analysis."""

    def __init__(self, video_path: str):
        self.video_path = video_path

        # Video properties
        self.width = None
        self.height = None
        self.total_frames = 0
        self.fps = None
        self.first_frame = None
        self.current_frame_idx = 0

        # Tracking area
        self.tracking_area = []  # List of (x, y) polygon points
        self.tracking_area_set = False

        # Animal count
        self.num_animals = 1  # Default: single animal (largest blob only)

        # Detection settings
        self.min_object_area = 100
        self.max_object_area = 10000
        self.history = 500
        self.var_threshold = 16
        self.contrast = 1.0
        self.brightness = 0

        # Blob refinement
        self.morph_open_size = 3
        self.morph_close_size = 5
        self.morph_open_iterations = 1
        self.morph_close_iterations = 2
        self.gaussian_blur_size = 0  # 0 = off, odd numbers 3,5,7...
        self.fill_holes = True
        self.convex_hull = False

        # ROIs
        self.rois = []  # List of (roi_name, polygon_points) tuples
        self.scale_bar_set = False
        self.scale_bar_pixels = None
        self.scale_bar_cm = None
        self.pixels_per_cm = None

        # Interpolation
        self.interpolate_tracks = True

        # Export options
        self.save_position_coords = True
        self.save_roi_occupancy = True
        self.save_roi_summary = True
        self.save_config = True
        self.save_tracking_plots = True
        self.create_tracked_video = True
        self.show_tracking_area = True
        self.show_rois = True
        self.show_blob_ids = True
        self.show_trail = True
        self.trail_length = 30
        self.save_data_view = True
        self.overwrite_files = True

        # State
        self.configured = False

    def add_roi(self, name: str, polygon: List[Tuple[int, int]]):
        """Add or update an ROI (maintains order if updating)."""
        for i, (roi_name, _) in enumerate(self.rois):
            if roi_name == name:
                self.rois[i] = (name, polygon)
                return
        self.rois.append((name, polygon))

    def get_roi_names(self) -> List[str]:
        """Get list of ROI names in priority order."""
        return [name for name, _ in self.rois]

    def get_roi_polygon(self, name: str) -> Optional[List[Tuple[int, int]]]:
        """Get polygon for a specific ROI."""
        for roi_name, polygon in self.rois:
            if roi_name == name:
                return polygon
        return None

    def rename_roi(self, old_name: str, new_name: str):
        """Rename an ROI."""
        for i, (roi_name, polygon) in enumerate(self.rois):
            if roi_name == old_name:
                self.rois[i] = (new_name, polygon)
                return

    def reorder_rois(self, new_order: List[str]):
        """Reorder ROIs based on list of names."""
        roi_dict = {name: polygon for name, polygon in self.rois}
        self.rois = [(name, roi_dict[name]) for name in new_order if name in roi_dict]

    def to_config_dict(self) -> dict:
        """Export configuration to dictionary for saving."""
        return {
            'video_path': self.video_path,
            'tracking_area': self.tracking_area,
            'tracking_area_set': self.tracking_area_set,
            'num_animals': self.num_animals,
            'min_object_area': self.min_object_area,
            'max_object_area': self.max_object_area,
            'history': self.history,
            'var_threshold': self.var_threshold,
            'contrast': self.contrast,
            'brightness': self.brightness,
            'morph_open_size': self.morph_open_size,
            'morph_close_size': self.morph_close_size,
            'morph_open_iterations': self.morph_open_iterations,
            'morph_close_iterations': self.morph_close_iterations,
            'gaussian_blur_size': self.gaussian_blur_size,
            'fill_holes': self.fill_holes,
            'convex_hull': self.convex_hull,
            'rois': [(name, polygon) for name, polygon in self.rois],
            'scale_bar_set': self.scale_bar_set,
            'scale_bar_pixels': self.scale_bar_pixels,
            'scale_bar_cm': self.scale_bar_cm,
            'pixels_per_cm': self.pixels_per_cm,
            'interpolate_tracks': self.interpolate_tracks,
            'save_position_coords': self.save_position_coords,
            'save_roi_occupancy': self.save_roi_occupancy,
            'save_roi_summary': self.save_roi_summary,
            'save_config': self.save_config,
            'save_tracking_plots': self.save_tracking_plots,
            'create_tracked_video': self.create_tracked_video,
            'show_tracking_area': self.show_tracking_area,
            'show_rois': self.show_rois,
            'show_blob_ids': self.show_blob_ids,
            'show_trail': self.show_trail,
            'trail_length': self.trail_length,
            'save_data_view': self.save_data_view,
            'overwrite_files': self.overwrite_files,
        }

    def from_config_dict(self, config_dict: dict):
        """Load configuration from dictionary."""
        self.tracking_area = config_dict.get('tracking_area', [])
        self.tracking_area_set = config_dict.get('tracking_area_set', False)
        self.num_animals = config_dict.get('num_animals', 1)
        self.min_object_area = config_dict.get('min_object_area', 100)
        self.max_object_area = config_dict.get('max_object_area', 10000)
        self.history = config_dict.get('history', 500)
        self.var_threshold = config_dict.get('var_threshold', 16)
        self.contrast = config_dict.get('contrast', 1.0)
        self.brightness = config_dict.get('brightness', 0)
        self.morph_open_size = config_dict.get('morph_open_size', 3)
        self.morph_close_size = config_dict.get('morph_close_size', 5)
        self.morph_open_iterations = config_dict.get('morph_open_iterations', 1)
        self.morph_close_iterations = config_dict.get('morph_close_iterations', 2)
        self.gaussian_blur_size = config_dict.get('gaussian_blur_size', 0)
        self.fill_holes = config_dict.get('fill_holes', True)
        self.convex_hull = config_dict.get('convex_hull', False)
        self.rois = [(name, polygon) for name, polygon in config_dict.get('rois', [])]
        self.scale_bar_set = config_dict.get('scale_bar_set', False)
        self.scale_bar_pixels = config_dict.get('scale_bar_pixels', None)
        self.scale_bar_cm = config_dict.get('scale_bar_cm', None)
        self.pixels_per_cm = config_dict.get('pixels_per_cm', None)
        self.interpolate_tracks = config_dict.get('interpolate_tracks', True)
        self.save_position_coords = config_dict.get('save_position_coords', True)
        self.save_roi_occupancy = config_dict.get('save_roi_occupancy', True)
        self.save_roi_summary = config_dict.get('save_roi_summary', True)
        self.save_config = config_dict.get('save_config', True)
        self.save_tracking_plots = config_dict.get('save_tracking_plots', True)
        self.create_tracked_video = config_dict.get('create_tracked_video', True)
        self.show_tracking_area = config_dict.get('show_tracking_area', True)
        self.show_rois = config_dict.get('show_rois', True)
        self.show_blob_ids = config_dict.get('show_blob_ids', True)
        self.show_trail = config_dict.get('show_trail', True)
        self.trail_length = config_dict.get('trail_length', 30)
        self.save_data_view = config_dict.get('save_data_view', True)
        self.overwrite_files = config_dict.get('overwrite_files', True)


# ═══════════════════════════════════════════════════════════════════════════════
# BackgroundSubtractionTracker - Core detection engine
# ═══════════════════════════════════════════════════════════════════════════════

class BackgroundSubtractionTracker:
    """
    Multi-object tracker using MOG2 background subtraction with configurable
    blob refinement. Supports single-animal mode (largest blob only).
    """

    def __init__(
        self,
        video_path: str,
        min_object_area: int = 100,
        max_object_area: int = 10000,
        history: int = 500,
        var_threshold: int = 16,
        num_animals: int = 1,
        morph_open_size: int = 3,
        morph_close_size: int = 5,
        morph_open_iterations: int = 1,
        morph_close_iterations: int = 2,
        gaussian_blur_size: int = 0,
        fill_holes: bool = True,
        convex_hull: bool = False,
    ):
        self.video_path = video_path
        self.min_object_area = min_object_area
        self.max_object_area = max_object_area
        self.history = history
        self.var_threshold = var_threshold
        self.num_animals = num_animals

        # Blob refinement params
        self.morph_open_size = morph_open_size
        self.morph_close_size = morph_close_size
        self.morph_open_iterations = morph_open_iterations
        self.morph_close_iterations = morph_close_iterations
        self.gaussian_blur_size = gaussian_blur_size
        self.fill_holes = fill_holes
        self.convex_hull = convex_hull

        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False
        )

        # Video properties
        self.cap = None
        self.fps = None
        self.total_frames = None
        self.frame_width = None
        self.frame_height = None

        # Multi-object tracking state
        self.trajectories = defaultdict(list)
        self.next_object_id = 0
        self.previous_centroids = {}
        self.max_distance_threshold = 50
        self.max_frames_disappeared = 30
        self.disappeared = {}

    def initialize_video(self):
        """Open video and read properties."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")

        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

    def apply_roi_mask(self, frame: np.ndarray, tracking_area: List[Tuple[int, int]]) -> np.ndarray:
        """Zero out pixels outside the tracking area polygon."""
        if not tracking_area or len(tracking_area) < 3:
            return frame
        mask = np.zeros(frame.shape[:2], dtype=np.uint8)
        pts = np.array(tracking_area, dtype=np.int32)
        cv2.fillPoly(mask, [pts], 255)
        if len(frame.shape) == 3:
            return cv2.bitwise_and(frame, frame, mask=mask)
        return cv2.bitwise_and(frame, frame, mask=mask)

    def _refine_mask(self, fg_mask: np.ndarray) -> np.ndarray:
        """Apply morphological refinement operations to foreground mask."""
        # Opening (remove noise)
        if self.morph_open_size > 0 and self.morph_open_iterations > 0:
            k_size = max(1, self.morph_open_size)
            if k_size % 2 == 0:
                k_size += 1
            kernel_open = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel_open,
                                       iterations=self.morph_open_iterations)

        # Closing (fill gaps)
        if self.morph_close_size > 0 and self.morph_close_iterations > 0:
            k_size = max(1, self.morph_close_size)
            if k_size % 2 == 0:
                k_size += 1
            kernel_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_size, k_size))
            fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel_close,
                                       iterations=self.morph_close_iterations)

        # Gaussian blur
        if self.gaussian_blur_size > 0:
            blur_size = self.gaussian_blur_size
            if blur_size % 2 == 0:
                blur_size += 1
            fg_mask = cv2.GaussianBlur(fg_mask, (blur_size, blur_size), 0)
            _, fg_mask = cv2.threshold(fg_mask, 127, 255, cv2.THRESH_BINARY)

        # Fill holes using floodfill
        if self.fill_holes:
            h, w = fg_mask.shape
            flood_mask = np.zeros((h + 2, w + 2), np.uint8)
            fg_inv = cv2.bitwise_not(fg_mask)
            cv2.floodFill(fg_inv, flood_mask, (0, 0), 255)
            fg_inv = cv2.bitwise_not(fg_inv)
            fg_mask = fg_mask | fg_inv

        return fg_mask

    def process_frame(self, frame: np.ndarray, frame_idx: int,
                      tracking_area: Optional[List[Tuple[int, int]]] = None
                      ) -> Tuple[Dict[int, Tuple[float, float, float]], np.ndarray]:
        """
        Process single frame with background subtraction.

        Returns:
            - Dictionary: {object_id: (x, y, area)}
            - Refined binary mask for visualization
        """
        # Convert to grayscale
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame.copy()

        # Apply tracking area mask before BG subtraction
        if tracking_area:
            gray = self.apply_roi_mask(gray, tracking_area)

        # Background subtraction
        fg_mask = self.bg_subtractor.apply(gray)

        # Refine mask with morphological operations
        fg_mask = self._refine_mask(fg_mask)

        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(
            fg_mask, connectivity=8
        )

        # Extract valid objects (filter by area)
        current_objects = []
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if self.min_object_area <= area <= self.max_object_area:
                if self.convex_hull:
                    # Compute convex hull centroid
                    blob_mask = (labels == i).astype(np.uint8) * 255
                    contours, _ = cv2.findContours(blob_mask, cv2.RETR_EXTERNAL,
                                                   cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        hull = cv2.convexHull(contours[0])
                        M = cv2.moments(hull)
                        if M["m00"] > 0:
                            cx = M["m10"] / M["m00"]
                            cy = M["m01"] / M["m00"]
                            current_objects.append((cx, cy, area))
                            continue
                # Standard centroid
                cx, cy = centroids[i]
                current_objects.append((cx, cy, area))

        # Single-animal mode: keep only the largest blob
        if self.num_animals == 1 and len(current_objects) > 1:
            current_objects = [max(current_objects, key=lambda o: o[2])]

        # Multi-animal mode: keep top N blobs by area
        elif self.num_animals > 1 and len(current_objects) > self.num_animals:
            current_objects = sorted(current_objects, key=lambda o: o[2], reverse=True)
            current_objects = current_objects[:self.num_animals]

        # Single-animal mode: always assign to ID 0 (no matching needed)
        if self.num_animals == 1:
            matched_objects = {}
            if current_objects:
                cx, cy, area = current_objects[0]  # Already filtered to largest
                self.trajectories[0].append((frame_idx, cx, cy, area))
                matched_objects[0] = (cx, cy, area)
            return matched_objects, fg_mask

        # Multi-animal mode: match to previous frame and assign IDs
        matched_objects = self._match_and_update(current_objects, frame_idx)

        return matched_objects, fg_mask

    def _match_and_update(self, current_objects, frame_idx):
        """Match current objects to previous frame using Hungarian algorithm."""
        matched = {}

        if not self.previous_centroids:
            for cx, cy, area in current_objects:
                obj_id = self.next_object_id
                self.next_object_id += 1
                self.previous_centroids[obj_id] = (cx, cy)
                self.trajectories[obj_id].append((frame_idx, cx, cy, area))
                matched[obj_id] = (cx, cy, area)
            return matched

        if not current_objects:
            for obj_id in list(self.previous_centroids.keys()):
                self.disappeared[obj_id] = self.disappeared.get(obj_id, 0) + 1
                if self.disappeared[obj_id] > self.max_frames_disappeared:
                    del self.previous_centroids[obj_id]
                    del self.disappeared[obj_id]
            return matched

        prev_ids = list(self.previous_centroids.keys())
        prev_centroids = np.array([self.previous_centroids[oid] for oid in prev_ids])
        curr_centroids = np.array([(cx, cy) for cx, cy, _ in current_objects])

        distances = np.linalg.norm(
            prev_centroids[:, np.newaxis] - curr_centroids[np.newaxis, :], axis=2
        )

        if SCIPY_AVAILABLE and len(prev_ids) > 0 and len(current_objects) > 0:
            row_indices, col_indices = linear_sum_assignment(distances)

            used_cols = set()
            for row, col in zip(row_indices, col_indices):
                if distances[row, col] < self.max_distance_threshold:
                    obj_id = prev_ids[row]
                    cx, cy, area = current_objects[col]
                    self.previous_centroids[obj_id] = (cx, cy)
                    self.trajectories[obj_id].append((frame_idx, cx, cy, area))
                    self.disappeared[obj_id] = 0
                    matched[obj_id] = (cx, cy, area)
                    used_cols.add(col)

            for col, (cx, cy, area) in enumerate(current_objects):
                if col not in used_cols:
                    obj_id = self.next_object_id
                    self.next_object_id += 1
                    self.previous_centroids[obj_id] = (cx, cy)
                    self.trajectories[obj_id].append((frame_idx, cx, cy, area))
                    matched[obj_id] = (cx, cy, area)

            matched_prev_ids = set(
                prev_ids[i] for i in row_indices
                if distances[i, col_indices[list(row_indices).index(i)]] < self.max_distance_threshold
            )
            for obj_id in prev_ids:
                if obj_id not in matched_prev_ids:
                    self.disappeared[obj_id] = self.disappeared.get(obj_id, 0) + 1
                    if self.disappeared[obj_id] > self.max_frames_disappeared:
                        del self.previous_centroids[obj_id]
                        if obj_id in self.disappeared:
                            del self.disappeared[obj_id]

        else:
            # Simple nearest neighbor matching (fallback)
            used_curr = set()
            for obj_id in prev_ids:
                prev_x, prev_y = self.previous_centroids[obj_id]
                min_dist = float('inf')
                best_idx = None
                for idx, (cx, cy, area) in enumerate(current_objects):
                    if idx in used_curr:
                        continue
                    dist = np.sqrt((cx - prev_x)**2 + (cy - prev_y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = idx

                if best_idx is not None and min_dist < self.max_distance_threshold:
                    cx, cy, area = current_objects[best_idx]
                    self.previous_centroids[obj_id] = (cx, cy)
                    self.trajectories[obj_id].append((frame_idx, cx, cy, area))
                    self.disappeared[obj_id] = 0
                    matched[obj_id] = (cx, cy, area)
                    used_curr.add(best_idx)
                else:
                    self.disappeared[obj_id] = self.disappeared.get(obj_id, 0) + 1
                    if self.disappeared[obj_id] > self.max_frames_disappeared:
                        del self.previous_centroids[obj_id]
                        if obj_id in self.disappeared:
                            del self.disappeared[obj_id]

            for idx, (cx, cy, area) in enumerate(current_objects):
                if idx not in used_curr:
                    obj_id = self.next_object_id
                    self.next_object_id += 1
                    self.previous_centroids[obj_id] = (cx, cy)
                    self.trajectories[obj_id].append((frame_idx, cx, cy, area))
                    matched[obj_id] = (cx, cy, area)

        return matched

    def get_standard_view_frame(self, frame: np.ndarray, mask: np.ndarray,
                                objects: Dict, tracking_area=None, rois=None) -> np.ndarray:
        """Render camera view with green semi-transparent mask overlay + IDs."""
        display = frame.copy()

        # Green semi-transparent overlay for detected blobs
        if mask is not None:
            mask_colored = np.zeros_like(display)
            mask_colored[:, :, 1] = mask  # Green channel
            display = cv2.addWeighted(display, 0.7, mask_colored, 0.3, 0)

        # Draw tracking area boundary
        if tracking_area and len(tracking_area) >= 3:
            pts = np.array(tracking_area, dtype=np.int32)
            cv2.polylines(display, [pts], True, (0, 255, 0), 2)

        # Draw ROIs
        if rois:
            for roi_name, roi_polygon in rois:
                if len(roi_polygon) < 3:
                    continue
                pts = np.array(roi_polygon, dtype=np.int32)
                if 'center' in roi_name.lower():
                    color = (0, 165, 255)
                else:
                    color = (0, 255, 255)  # Cyan for ROIs
                cv2.polylines(display, [pts], True, color, 2)
                cv2.putText(display, roi_name, tuple(roi_polygon[0]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw object center dots + IDs
        colors = [(255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0)]
        for obj_id, (cx, cy, area) in objects.items():
            color = colors[obj_id % len(colors)]
            cv2.circle(display, (int(cx), int(cy)), 8, color, -1)
            cv2.circle(display, (int(cx), int(cy)), 11, color, 2)
            cv2.putText(display, f"ID{obj_id}", (int(cx) + 14, int(cy)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

        return display

    def get_mask_view_frame(self, frame: np.ndarray, mask: np.ndarray,
                            objects: Dict, tracking_area=None, rois=None) -> np.ndarray:
        """Render white background with black blobs (inside tracking area only)."""
        h, w = frame.shape[:2]

        # White background with black blobs
        display = np.ones((h, w, 3), dtype=np.uint8) * 255
        if mask is not None:
            # Black where the mask is active (blobs)
            blob_pixels = mask > 127
            display[blob_pixels] = [0, 0, 0]

        # Grey out area outside tracking area
        if tracking_area and len(tracking_area) >= 3:
            ta_mask = np.zeros((h, w), dtype=np.uint8)
            pts = np.array(tracking_area, dtype=np.int32)
            cv2.fillPoly(ta_mask, [pts], 255)
            outside = ta_mask == 0
            display[outside] = [40, 40, 40]  # Dark grey outside tracking area
            cv2.polylines(display, [pts], True, (0, 180, 0), 2)

        # Draw ROIs
        if rois:
            for roi_name, roi_polygon in rois:
                if len(roi_polygon) < 3:
                    continue
                pts = np.array(roi_polygon, dtype=np.int32)
                if 'center' in roi_name.lower():
                    color = (0, 130, 200)
                else:
                    color = (0, 180, 180)
                cv2.polylines(display, [pts], True, color, 2)
                cv2.putText(display, roi_name, tuple(roi_polygon[0]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw object centers with red dots for visibility on white bg
        for obj_id, (cx, cy, area) in objects.items():
            cv2.circle(display, (int(cx), int(cy)), 8, (220, 40, 40), -1)
            cv2.circle(display, (int(cx), int(cy)), 11, (220, 40, 40), 2)
            cv2.putText(display, f"ID{obj_id}", (int(cx) + 14, int(cy)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, (220, 40, 40), 2)

        return display

    def cleanup(self):
        """Release video capture."""
        if self.cap:
            self.cap.release()


# ═══════════════════════════════════════════════════════════════════════════════
# Trajectory Interpolation
# ═══════════════════════════════════════════════════════════════════════════════

def interpolate_trajectories(trajectories: Dict[int, List[Tuple]],
                             total_frames: int,
                             max_gap: int = 30) -> Dict[int, List[Tuple]]:
    """
    Fill gaps in trajectories using linear interpolation.

    Parameters
    ----------
    trajectories : dict
        {object_id: [(frame, x, y, area), ...]}
    total_frames : int
        Total number of frames in the video.
    max_gap : int
        Maximum gap size (in frames) to interpolate across.

    Returns
    -------
    dict : Interpolated trajectories in the same format.
    """
    interpolated = {}

    for obj_id, traj in trajectories.items():
        if len(traj) < 2:
            interpolated[obj_id] = list(traj)
            continue

        # Sort by frame
        sorted_traj = sorted(traj, key=lambda t: t[0])
        new_traj = [sorted_traj[0]]

        for i in range(1, len(sorted_traj)):
            prev_frame, prev_x, prev_y, prev_area = sorted_traj[i - 1]
            curr_frame, curr_x, curr_y, curr_area = sorted_traj[i]

            gap = curr_frame - prev_frame

            if 1 < gap <= max_gap:
                # Linear interpolation for the gap
                for f in range(prev_frame + 1, curr_frame):
                    t = (f - prev_frame) / gap
                    ix = prev_x + t * (curr_x - prev_x)
                    iy = prev_y + t * (curr_y - prev_y)
                    ia = prev_area + t * (curr_area - prev_area)
                    new_traj.append((f, ix, iy, ia))

            new_traj.append(sorted_traj[i])

        interpolated[obj_id] = new_traj

    return interpolated


# ═══════════════════════════════════════════════════════════════════════════════
# PLACEHOLDER: BatchTrackingWorker and SimpleTrackerGUI follow in subsequent chunks
# ═══════════════════════════════════════════════════════════════════════════════

class BatchTrackingWorker(QThread):
    """Worker thread for batch processing multiple videos."""

    progress = pyqtSignal(int, int, int)  # video_idx, frame_idx, total_frames
    status = pyqtSignal(str)
    video_finished = pyqtSignal(int, bool, str)  # video_idx, success, message
    all_finished = pyqtSignal(bool, str)

    def __init__(self, video_configs: List[VideoTrackingConfig]):
        super().__init__()
        self.video_configs = video_configs
        self.cancelled = False

    def get_output_folder(self, video_path: str) -> str:
        """Get/create analysis output folder for a video."""
        video_base = os.path.splitext(os.path.basename(video_path))[0]
        video_dir = os.path.dirname(video_path)
        folder = os.path.join(video_dir, f"{video_base}_FNT_SimpleTracker_analysis")
        os.makedirs(folder, exist_ok=True)
        return folder

    def get_output_path(self, video_path: str, suffix: str) -> str:
        """Generate output path inside analysis folder."""
        folder = self.get_output_folder(video_path)
        video_base = os.path.splitext(os.path.basename(video_path))[0]
        return os.path.join(folder, f"{video_base}{suffix}")

    def clear_existing_outputs(self, config: VideoTrackingConfig):
        """Clear all existing output files for this video."""
        try:
            folder = self.get_output_folder(config.video_path)
            if not os.path.exists(folder):
                return
            files_deleted = 0
            for fname in os.listdir(folder):
                fpath = os.path.join(folder, fname)
                if os.path.isfile(fpath):
                    try:
                        os.remove(fpath)
                        files_deleted += 1
                    except Exception as e:
                        self.status.emit(f"Warning: Could not delete {fname}: {e}")
            if files_deleted > 0:
                self.status.emit(f"\u2713 Cleared {files_deleted} existing output file(s)")
        except Exception as e:
            self.status.emit(f"Warning: Error clearing outputs: {e}")

    def point_in_polygon(self, point, polygon):
        """Check if point is inside polygon using cv2."""
        if not polygon or len(polygon) < 3:
            return False
        poly_arr = np.array(polygon, dtype=np.int32)
        return cv2.pointPolygonTest(poly_arr, point, False) >= 0

    def run(self):
        """Process all videos in batch."""
        total_videos = len(self.video_configs)
        successful = 0
        failed = 0

        for video_idx, config in enumerate(self.video_configs):
            if self.cancelled:
                self.all_finished.emit(False, "Batch processing cancelled")
                return

            self.status.emit(
                f"Processing video {video_idx + 1}/{total_videos}: "
                f"{os.path.basename(config.video_path)}"
            )

            try:
                # Clear existing outputs if overwrite enabled
                if config.overwrite_files:
                    self.clear_existing_outputs(config)

                # ── Phase 1: Detection ────────────────────────────
                tracker = BackgroundSubtractionTracker(
                    video_path=config.video_path,
                    min_object_area=config.min_object_area,
                    max_object_area=config.max_object_area,
                    history=config.history,
                    var_threshold=config.var_threshold,
                    num_animals=config.num_animals,
                    morph_open_size=config.morph_open_size,
                    morph_close_size=config.morph_close_size,
                    morph_open_iterations=config.morph_open_iterations,
                    morph_close_iterations=config.morph_close_iterations,
                    gaussian_blur_size=config.gaussian_blur_size,
                    fill_holes=config.fill_holes,
                    convex_hull=config.convex_hull,
                )
                tracker.initialize_video()

                tracking_area = config.tracking_area if config.tracking_area else None
                fps = tracker.fps or 30
                total_frames = tracker.total_frames

                frame_idx = 0
                while not self.cancelled:
                    ret, frame = tracker.cap.read()
                    if not ret:
                        break

                    if len(frame.shape) == 3:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

                    # Apply contrast/brightness
                    if config.contrast != 1.0 or config.brightness != 0:
                        adj = frame_rgb.astype(np.float32)
                        adj = config.contrast * (adj - 128) + 128 + config.brightness
                        frame_rgb = np.clip(adj, 0, 255).astype(np.uint8)

                    objects, mask = tracker.process_frame(
                        frame_rgb, frame_idx, tracking_area=tracking_area
                    )

                    if frame_idx % 50 == 0:
                        self.progress.emit(video_idx, frame_idx, total_frames)

                    frame_idx += 1

                if self.cancelled:
                    tracker.cleanup()
                    self.all_finished.emit(False, "Batch processing cancelled")
                    return

                self.status.emit(f"Detection complete: {frame_idx} frames processed")

                # ── Phase 2: Post-processing ──────────────────────
                trajectories = dict(tracker.trajectories)
                tracker.cleanup()

                if config.interpolate_tracks:
                    self.status.emit("Interpolating trajectories...")
                    # For single-animal mode, use a much larger max_gap since
                    # all detections belong to the same animal and we want
                    # continuous smooth interpolation across detection gaps
                    gap = 300 if config.num_animals == 1 else 30
                    trajectories = interpolate_trajectories(
                        trajectories, total_frames, max_gap=gap
                    )

                # Build position DataFrame
                rows = []
                for obj_id, traj in trajectories.items():
                    for f, x, y, area in traj:
                        rows.append({
                            'frame': f, 'timestamp': f / fps,
                            'object_id': obj_id, 'x': x, 'y': y, 'area': area
                        })

                if not rows:
                    self.video_finished.emit(video_idx, False, "No objects detected")
                    failed += 1
                    continue

                traj_df = pd.DataFrame(rows).sort_values(['frame', 'object_id'])

                # Pivot to wide format for position coordinates
                object_ids = sorted(traj_df['object_id'].unique())
                pos_frames = list(range(total_frames))
                pos_data = {'frame': pos_frames,
                            'timestamp': [f / fps for f in pos_frames]}

                for oid in object_ids:
                    obj_traj = traj_df[traj_df['object_id'] == oid].set_index('frame')
                    pos_data[f'object_{oid}_x'] = [
                        obj_traj.loc[f, 'x'] if f in obj_traj.index else np.nan
                        for f in pos_frames
                    ]
                    pos_data[f'object_{oid}_y'] = [
                        obj_traj.loc[f, 'y'] if f in obj_traj.index else np.nan
                        for f in pos_frames
                    ]
                    pos_data[f'object_{oid}_area'] = [
                        obj_traj.loc[f, 'area'] if f in obj_traj.index else np.nan
                        for f in pos_frames
                    ]

                pos_df = pd.DataFrame(pos_data)

                # ── Phase 3: ROI Analysis ─────────────────────────
                occupancy_dfs = {}
                if config.rois:
                    self.status.emit("Calculating ROI occupancy...")
                    for oid in object_ids:
                        occ_data = []
                        x_col = f'object_{oid}_x'
                        y_col = f'object_{oid}_y'
                        for _, row in pos_df.iterrows():
                            x, y = row[x_col], row[y_col]
                            if pd.isna(x) or pd.isna(y):
                                occ_data.append({'frame': int(row['frame']), 'region': 'none'})
                                continue
                            found = False
                            for roi_name, roi_poly in config.rois:
                                if self.point_in_polygon((float(x), float(y)), roi_poly):
                                    occ_data.append({'frame': int(row['frame']), 'region': roi_name})
                                    found = True
                                    break
                            if not found:
                                occ_data.append({'frame': int(row['frame']), 'region': 'none'})
                        occupancy_dfs[oid] = pd.DataFrame(occ_data)

                # ── Phase 4: Exports ──────────────────────────────
                # 4a: Position Coordinates CSV
                if config.save_position_coords:
                    path = self.get_output_path(config.video_path, '_positionCoordinates.csv')
                    pos_df.to_csv(path, index=False)
                    self.status.emit(f"\u2713 Saved position coordinates")

                # 4b: ROI Occupancy CSV (per object)
                if config.save_roi_occupancy and config.rois:
                    for oid in object_ids:
                        path = self.get_output_path(
                            config.video_path, f'_roiOccupancy_object{oid}.csv'
                        )
                        occupancy_dfs[oid].to_csv(path, index=False)
                    self.status.emit(f"\u2713 Saved ROI occupancy for {len(object_ids)} object(s)")

                # 4c: ROI Summary CSV
                if config.save_roi_summary and config.rois:
                    summary_rows = []
                    for oid in object_ids:
                        row_data = {
                            'video': os.path.basename(config.video_path),
                            'object_id': oid,
                            'video_duration_s': total_frames / fps,
                        }

                        # Total distance
                        obj_traj = traj_df[traj_df['object_id'] == oid].sort_values('frame')
                        if len(obj_traj) > 1:
                            dists = np.sqrt(
                                np.diff(obj_traj['x'].values)**2 +
                                np.diff(obj_traj['y'].values)**2
                            )
                            total_dist_pix = float(np.nansum(dists))
                        else:
                            total_dist_pix = 0.0

                        if config.scale_bar_set and config.pixels_per_cm:
                            row_data['total_distance_cm'] = total_dist_pix / config.pixels_per_cm
                        else:
                            row_data['total_distance_pix'] = total_dist_pix

                        # Per-ROI metrics
                        occ = occupancy_dfs[oid]
                        for roi_name, _ in config.rois:
                            safe = roi_name.replace(' ', '_').replace('-', '_')
                            in_roi = occ['region'] == roi_name
                            frames_in = int(in_roi.sum())
                            time_in = frames_in / fps

                            row_data[f'time_s_{safe}'] = time_in

                            # Latency
                            first_entry = None
                            for idx_r, r in occ.iterrows():
                                if r['region'] == roi_name:
                                    first_entry = idx_r
                                    break
                            row_data[f'latency_enter_s_{safe}'] = (
                                first_entry / fps if first_entry is not None else np.nan
                            )

                            # Entry count
                            entries = 0
                            if len(in_roi) > 0:
                                if in_roi.iloc[0]:
                                    entries = 1
                                entries += int(np.sum(np.diff(in_roi.astype(int)) > 0))
                            row_data[f'roi_entry_count_{safe}'] = entries
                            row_data[f'frames_{safe}'] = frames_in

                        summary_rows.append(row_data)

                    summary_df = pd.DataFrame(summary_rows)
                    path = self.get_output_path(config.video_path, '_roiSummary.csv')
                    summary_df.to_csv(path, index=False)
                    self.status.emit(f"\u2713 Saved ROI summary")

                # 4d: Config JSON
                if config.save_config:
                    path = self.get_output_path(config.video_path, '_trackerConfig.json')
                    with open(path, 'w') as f:
                        json.dump(config.to_config_dict(), f, indent=2)
                    self.status.emit(f"\u2713 Saved configuration")

                # 4e: Tracking plots
                if config.save_tracking_plots:
                    self._create_tracking_plots(config, traj_df, pos_df, object_ids, fps)

                # 4f: Tracked video / data view video
                if config.create_tracked_video:
                    if config.save_data_view:
                        self._create_data_view_video(
                            config, traj_df, pos_df, occupancy_dfs,
                            object_ids, fps, total_frames, video_idx
                        )
                    else:
                        self._create_tracked_video(
                            config, traj_df, pos_df, occupancy_dfs,
                            object_ids, fps, total_frames, video_idx
                        )

                self.video_finished.emit(
                    video_idx, True,
                    f"Completed: {os.path.basename(config.video_path)}"
                )
                successful += 1

            except Exception as e:
                import traceback
                self.video_finished.emit(video_idx, False, f"Failed: {str(e)}")
                self.status.emit(f"Error: {traceback.format_exc()}")
                failed += 1

        summary = f"Batch complete: {successful} successful, {failed} failed"
        self.all_finished.emit(failed == 0, summary)

    def cancel(self):
        self.cancelled = True

    # ── Plot and Video Export Methods ─────────────────────────────────────

    def _create_tracking_plots(self, config, traj_df, pos_df, object_ids, fps):
        """Create trajectory plots per object and a combined heatmap."""
        self.status.emit("Creating tracking plots...")

        for oid in object_ids:
            try:
                obj_data = traj_df[traj_df['object_id'] == oid].sort_values('frame')
                if len(obj_data) < 2:
                    continue

                fig, ax = plt.subplots(figsize=(12, 10))
                ax.plot(obj_data['x'].values, obj_data['y'].values,
                       color='blue', linewidth=2, alpha=0.8, label='Trajectory')

                # Start/end markers
                ax.plot(obj_data['x'].iloc[0], obj_data['y'].iloc[0],
                       'ko', markersize=10, label='Start', zorder=5)
                ax.plot(obj_data['x'].iloc[-1], obj_data['y'].iloc[-1],
                       'ks', markersize=10, label='End', zorder=5)

                # Draw ROIs
                for roi_name, roi_poly in config.rois:
                    if len(roi_poly) < 3:
                        continue
                    rp = np.array(roi_poly + [roi_poly[0]], dtype=np.int32)
                    ax.plot(rp[:, 0], rp[:, 1], color='black', linewidth=2)

                ax.set_xlabel('X (pixels)', fontsize=12)
                ax.set_ylabel('Y (pixels)', fontsize=12)
                ax.set_title(
                    f'Trajectory: Object {oid}\n{os.path.basename(config.video_path)}',
                    fontsize=14, fontweight='bold'
                )
                ax.set_aspect('equal')
                ax.grid(True, alpha=0.3)
                ax.legend(loc='upper right')
                ax.invert_yaxis()
                plt.tight_layout()

                path = self.get_output_path(
                    config.video_path, f'_trajectoryPlot_object{oid}.png'
                )
                fig.savefig(path, dpi=150, bbox_inches='tight')
                plt.close(fig)
                self.status.emit(f"\u2713 Saved trajectory plot for object {oid}")

            except Exception as e:
                self.status.emit(f"Warning: Plot error for object {oid}: {e}")

        # Heatmap
        try:
            fig, ax = plt.subplots(figsize=(12, 10))

            # Use first frame as background if available
            cap = cv2.VideoCapture(config.video_path)
            ret, frame = cap.read()
            cap.release()
            if ret:
                if len(frame.shape) == 3:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame_rgb = frame
                ax.imshow(frame_rgb, alpha=0.5)

            all_x = traj_df['x'].dropna().values
            all_y = traj_df['y'].dropna().values
            if len(all_x) > 10:
                width = config.width or int(np.max(all_x)) + 1
                height = config.height or int(np.max(all_y)) + 1
                heatmap, _, _ = np.histogram2d(
                    all_y, all_x, bins=[height // 10, width // 10],
                    range=[[0, height], [0, width]]
                )
                ax.imshow(heatmap, cmap='hot', alpha=0.6,
                         extent=[0, width, height, 0])

            ax.set_title(
                f'Position Heatmap\n{os.path.basename(config.video_path)}',
                fontsize=14, fontweight='bold'
            )
            ax.set_xlabel('X (pixels)')
            ax.set_ylabel('Y (pixels)')
            plt.tight_layout()

            path = self.get_output_path(config.video_path, '_heatmap.png')
            fig.savefig(path, dpi=150, bbox_inches='tight')
            plt.close(fig)
            self.status.emit(f"\u2713 Saved heatmap")

        except Exception as e:
            self.status.emit(f"Warning: Heatmap error: {e}")

    def _create_tracked_video(self, config, traj_df, pos_df, occupancy_dfs,
                              object_ids, fps, total_frames, video_idx):
        """Create tracked video with overlays."""
        self.status.emit("Rendering tracked video...")

        cap = cv2.VideoCapture(config.video_path)
        if not cap.isOpened():
            return

        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        output_path = self.get_output_path(config.video_path, '_tracked.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))

        # Build frame lookup per object
        obj_positions = {}
        for oid in object_ids:
            obj_data = traj_df[traj_df['object_id'] == oid]
            obj_positions[oid] = {
                int(row['frame']): (row['x'], row['y'])
                for _, row in obj_data.iterrows()
            }

        trail_history = {oid: [] for oid in object_ids}
        colors = [(255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0)]

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % 50 == 0:
                self.progress.emit(video_idx, frame_num, total_frames)

            # Draw tracking area
            if config.show_tracking_area and config.tracking_area:
                pts = np.array(config.tracking_area, dtype=np.int32)
                cv2.polylines(frame, [pts], True, (255, 0, 255), 2)

            # Draw ROIs
            if config.show_rois and config.rois:
                for roi_name, roi_poly in config.rois:
                    if len(roi_poly) < 3:
                        continue
                    pts = np.array(roi_poly, dtype=np.int32)
                    if 'center' in roi_name.lower():
                        color = (0, 165, 255)
                    else:
                        color = (0, 255, 0)
                    cv2.polylines(frame, [pts], True, color, 2)
                    cv2.putText(frame, roi_name, tuple(roi_poly[0]),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Draw trails first (behind objects)
            if config.show_trail:
                for oid in object_ids:
                    c = colors[oid % len(colors)]
                    trail = trail_history[oid]
                    if len(trail) > 1:
                        for i in range(len(trail) - 1):
                            alpha = (i + 1) / len(trail)
                            thickness = max(1, int(3 * alpha))
                            cv2.line(frame, trail[i], trail[i + 1], c, thickness)

            # Draw objects
            for oid in object_ids:
                if frame_num in obj_positions[oid]:
                    x, y = obj_positions[oid][frame_num]
                    ix, iy = int(x), int(y)
                    c = colors[oid % len(colors)]

                    cv2.circle(frame, (ix, iy), 8, c, -1)
                    cv2.circle(frame, (ix, iy), 12, c, 2)

                    if config.show_blob_ids:
                        cv2.putText(frame, f"ID{oid}", (ix + 15, iy),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)

                    # Update trail
                    if config.show_trail:
                        trail_history[oid].append((ix, iy))
                        if len(trail_history[oid]) > config.trail_length:
                            trail_history[oid].pop(0)

            out.write(frame)
            frame_num += 1

        cap.release()
        out.release()
        self.status.emit(f"\u2713 Saved tracked video: {os.path.basename(output_path)}")

    def _create_data_view_video(self, config, traj_df, pos_df, occupancy_dfs,
                                object_ids, fps, total_frames, video_idx):
        """Create data view video: tracked video on left, live stats on right."""
        self.status.emit("Rendering data view video...")

        cap = cv2.VideoCapture(config.video_path)
        if not cap.isOpened():
            return

        vid_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        vid_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        # Stats panel width = 40% of video width
        stats_w = max(300, int(vid_w * 0.4))
        output_w = vid_w + stats_w
        output_h = vid_h

        output_path = self.get_output_path(config.video_path, '_tracked.mp4')
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (output_w, output_h))

        # Build lookups
        obj_positions = {}
        for oid in object_ids:
            obj_data = traj_df[traj_df['object_id'] == oid]
            obj_positions[oid] = {
                int(row['frame']): (row['x'], row['y'])
                for _, row in obj_data.iterrows()
            }

        # Pre-compute cumulative distances per object
        cum_distances = {oid: {} for oid in object_ids}
        for oid in object_ids:
            obj_sorted = traj_df[traj_df['object_id'] == oid].sort_values('frame')
            cum_dist = 0.0
            prev_x, prev_y = None, None
            for _, row in obj_sorted.iterrows():
                f = int(row['frame'])
                if prev_x is not None:
                    cum_dist += np.sqrt((row['x'] - prev_x)**2 + (row['y'] - prev_y)**2)
                cum_distances[oid][f] = cum_dist
                prev_x, prev_y = row['x'], row['y']

        trail_history = {oid: [] for oid in object_ids}
        colors_bgr = [(255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0)]

        # ROI time accumulators
        roi_time = {oid: {rn: 0 for rn, _ in config.rois} for oid in object_ids}
        roi_entries = {oid: {rn: 0 for rn, _ in config.rois} for oid in object_ids}
        prev_region = {oid: 'none' for oid in object_ids}

        frame_num = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_num % 50 == 0:
                self.progress.emit(video_idx, frame_num, total_frames)

            # ── Left side: tracked video ──
            # Draw tracking area
            if config.show_tracking_area and config.tracking_area:
                pts = np.array(config.tracking_area, dtype=np.int32)
                cv2.polylines(frame, [pts], True, (255, 0, 255), 2)

            # Draw ROIs
            if config.show_rois and config.rois:
                for rn, rp in config.rois:
                    if len(rp) < 3:
                        continue
                    pts = np.array(rp, dtype=np.int32)
                    c = (0, 165, 255) if 'center' in rn.lower() else (0, 255, 0)
                    cv2.polylines(frame, [pts], True, c, 2)
                    cv2.putText(frame, rn, tuple(rp[0]),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 1)

            # Draw trails
            if config.show_trail:
                for oid in object_ids:
                    c = colors_bgr[oid % len(colors_bgr)]
                    trail = trail_history[oid]
                    for i in range(len(trail) - 1):
                        alpha = (i + 1) / len(trail)
                        cv2.line(frame, trail[i], trail[i + 1], c, max(1, int(3 * alpha)))

            # Draw objects and update ROI accumulators
            for oid in object_ids:
                if frame_num in obj_positions[oid]:
                    x, y = obj_positions[oid][frame_num]
                    ix, iy = int(x), int(y)
                    c = colors_bgr[oid % len(colors_bgr)]
                    cv2.circle(frame, (ix, iy), 8, c, -1)
                    cv2.circle(frame, (ix, iy), 12, c, 2)
                    if config.show_blob_ids:
                        cv2.putText(frame, f"ID{oid}", (ix + 15, iy),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, c, 2)
                    if config.show_trail:
                        trail_history[oid].append((ix, iy))
                        if len(trail_history[oid]) > config.trail_length:
                            trail_history[oid].pop(0)

                    # Update ROI time
                    if config.rois and oid in occupancy_dfs:
                        occ_df = occupancy_dfs[oid]
                        if frame_num < len(occ_df):
                            region = occ_df.iloc[frame_num]['region']
                            for rn, _ in config.rois:
                                if region == rn:
                                    roi_time[oid][rn] += 1.0 / fps
                            if region != prev_region[oid] and region != 'none':
                                roi_entries[oid][region] = roi_entries[oid].get(region, 0) + 1
                            prev_region[oid] = region

            # ── Right side: stats panel ──
            stats_panel = np.zeros((output_h, stats_w, 3), dtype=np.uint8)
            stats_panel[:] = (30, 30, 30)  # Dark background

            y_pos = 30
            font = cv2.FONT_HERSHEY_SIMPLEX

            # Title
            cv2.putText(stats_panel, "Live Statistics", (10, y_pos),
                       font, 0.7, (255, 255, 255), 2)
            y_pos += 30

            # Frame/time
            cv2.putText(stats_panel, f"Frame: {frame_num}/{total_frames}", (10, y_pos),
                       font, 0.5, (180, 180, 180), 1)
            y_pos += 22
            cv2.putText(stats_panel, f"Time: {frame_num / fps:.1f}s", (10, y_pos),
                       font, 0.5, (180, 180, 180), 1)
            y_pos += 30

            # Per-object stats
            for oid in object_ids:
                c = colors_bgr[oid % len(colors_bgr)]
                cv2.putText(stats_panel, f"Object {oid}", (10, y_pos),
                           font, 0.6, c, 2)
                y_pos += 22

                # Distance
                dist = cum_distances[oid].get(frame_num, 0.0)
                if config.scale_bar_set and config.pixels_per_cm:
                    cv2.putText(stats_panel, f"  Distance: {dist / config.pixels_per_cm:.1f} cm",
                               (10, y_pos), font, 0.45, (180, 180, 180), 1)
                else:
                    cv2.putText(stats_panel, f"  Distance: {dist:.0f} px",
                               (10, y_pos), font, 0.45, (180, 180, 180), 1)
                y_pos += 18

                # ROI info
                if config.rois:
                    for rn, _ in config.rois:
                        t = roi_time[oid].get(rn, 0)
                        e = roi_entries[oid].get(rn, 0)
                        cv2.putText(stats_panel,
                                   f"  {rn}: {t:.1f}s, {e} entries",
                                   (10, y_pos), font, 0.4, (150, 150, 150), 1)
                        y_pos += 16

                y_pos += 10

                if y_pos > output_h - 20:
                    break

            # Combine frames
            combined = np.zeros((output_h, output_w, 3), dtype=np.uint8)
            combined[:vid_h, :vid_w] = frame
            combined[:, vid_w:] = stats_panel

            out.write(combined)
            frame_num += 1

        cap.release()
        out.release()
        self.status.emit(f"\u2713 Saved data view video: {os.path.basename(output_path)}")


class DarkCheckBox(QCheckBox):
    """Checkbox with a visible checkmark drawn on dark theme backgrounds."""

    def __init__(self, text="", parent=None):
        super().__init__(text, parent)
        self.setStyleSheet("")  # Clear any parent style for indicator

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        # Draw indicator box
        indicator_size = 16
        y_center = self.height() // 2
        box_rect = QRect(0, y_center - indicator_size // 2,
                         indicator_size, indicator_size)

        # Box background
        if self.isChecked():
            p.setBrush(QBrush(QColor(0, 120, 212)))
            p.setPen(QPen(QColor(0, 120, 212), 1))
        else:
            p.setBrush(QBrush(QColor(63, 63, 63)))
            p.setPen(QPen(QColor(85, 85, 85), 2))
        p.drawRoundedRect(box_rect, 3, 3)

        # Draw checkmark
        if self.isChecked():
            p.setPen(QPen(QColor(255, 255, 255), 2.2, Qt.SolidLine, Qt.RoundCap, Qt.RoundJoin))
            x0, y0 = box_rect.x(), box_rect.y()
            p.drawLine(x0 + 3, y0 + 8, x0 + 6, y0 + 12)
            p.drawLine(x0 + 6, y0 + 12, x0 + 12, y0 + 4)

        # Draw label text
        text_x = indicator_size + 8
        p.setPen(QColor(204, 204, 204))
        p.setFont(self.font())
        text_rect = QRect(text_x, 0, self.width() - text_x, self.height())
        p.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, self.text())

        p.end()


class RangeSlider(QWidget):
    """Custom dual-handle range slider for min/max object size."""

    rangeChanged = pyqtSignal(int, int)
    sliderReleased = pyqtSignal()

    def __init__(self, minimum=0, maximum=50000, parent=None):
        super().__init__(parent)
        self._min = minimum
        self._max = maximum
        self._low = minimum
        self._high = maximum
        self._pressed = None  # 'low', 'high', or None
        self._handle_w = 14
        self._bar_h = 6
        self.setMinimumHeight(28)
        self.setMinimumWidth(100)
        self.setCursor(Qt.PointingHandCursor)

    def setRange(self, minimum, maximum):
        self._min = minimum
        self._max = maximum
        self._low = max(self._low, minimum)
        self._high = min(self._high, maximum)
        self.update()

    def setLow(self, value):
        self._low = max(self._min, min(value, self._high))
        self.update()

    def setHigh(self, value):
        self._high = min(self._max, max(value, self._low))
        self.update()

    def low(self):
        return self._low

    def high(self):
        return self._high

    def _val_to_x(self, val):
        w = self.width() - self._handle_w
        if self._max == self._min:
            return self._handle_w // 2
        return int(self._handle_w // 2 + (val - self._min) / (self._max - self._min) * w)

    def _x_to_val(self, x):
        w = self.width() - self._handle_w
        if w <= 0:
            return self._min
        ratio = max(0.0, min(1.0, (x - self._handle_w // 2) / w))
        return int(self._min + ratio * (self._max - self._min))

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)

        h = self.height()
        bar_y = h // 2 - self._bar_h // 2

        # Track background
        p.setPen(Qt.NoPen)
        p.setBrush(QBrush(QColor(63, 63, 63)))
        p.drawRoundedRect(self._handle_w // 2, bar_y,
                          self.width() - self._handle_w, self._bar_h, 3, 3)

        # Active range
        x_low = self._val_to_x(self._low)
        x_high = self._val_to_x(self._high)
        p.setBrush(QBrush(QColor(0, 120, 212)))
        p.drawRoundedRect(x_low, bar_y, max(1, x_high - x_low), self._bar_h, 3, 3)

        # Handles
        for x, is_pressed in [(x_low, self._pressed == 'low'),
                               (x_high, self._pressed == 'high')]:
            r = self._handle_w // 2
            color = QColor(30, 110, 190) if is_pressed else QColor(0, 120, 212)
            p.setBrush(QBrush(color))
            p.setPen(QPen(QColor(200, 200, 200), 1))
            p.drawEllipse(x - r, h // 2 - r, self._handle_w, self._handle_w)

        p.end()

    def mousePressEvent(self, event):
        x = event.pos().x()
        x_low = self._val_to_x(self._low)
        x_high = self._val_to_x(self._high)

        dist_low = abs(x - x_low)
        dist_high = abs(x - x_high)

        if dist_low <= dist_high:
            self._pressed = 'low'
        else:
            self._pressed = 'high'
        self._move_handle(x)
        self.update()

    def mouseMoveEvent(self, event):
        if self._pressed:
            self._move_handle(event.pos().x())

    def mouseReleaseEvent(self, event):
        if self._pressed:
            self._pressed = None
            self.update()
            self.sliderReleased.emit()

    def _move_handle(self, x):
        val = self._x_to_val(x)
        if self._pressed == 'low':
            val = min(val, self._high)
            if val != self._low:
                self._low = val
                self.rangeChanged.emit(self._low, self._high)
                self.update()
        elif self._pressed == 'high':
            val = max(val, self._low)
            if val != self._high:
                self._high = val
                self.rangeChanged.emit(self._low, self._high)
                self.update()


class SimpleTrackerGUI(QMainWindow):
    """
    Simple Tracker V3 - Movement-based blob detection with ROI analysis.
    Complete rewrite with blob refinement, ROI drawing, and export pipeline.
    """

    # Dark theme + blue scrollbar stylesheet
    DARK_THEME = """
        QMainWindow { background-color: #2b2b2b; color: #cccccc; }
        QWidget { background-color: #2b2b2b; color: #cccccc; }
        QGroupBox {
            font-weight: bold; border: 1px solid #3f3f3f;
            border-radius: 4px; margin-top: 6px; padding-top: 6px;
            padding-bottom: 2px; color: #cccccc;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 10px; padding: 0 5px; }
        QPushButton {
            background-color: #0078d4; color: white; border: none;
            padding: 4px 10px; border-radius: 3px; font-weight: bold; font-size: 12px;
        }
        QPushButton:hover { background-color: #106ebe; }
        QPushButton:pressed { background-color: #005a9e; }
        QPushButton:disabled { background-color: #3f3f3f; color: #888888; }
        QLabel { color: #cccccc; background-color: transparent; font-size: 11px; }
        QSpinBox, QDoubleSpinBox {
            background-color: #3f3f3f; color: #cccccc;
            border: 1px solid #555555; border-radius: 3px; padding: 2px;
        }
        QListWidget {
            background-color: #1e1e1e; color: #cccccc;
            border: 1px solid #3f3f3f; border-radius: 3px;
        }
        QListWidget::item { padding: 3px; border-bottom: 1px solid #3f3f3f; }
        QListWidget::item:selected { background-color: #0078d4; }
        QProgressBar {
            border: 1px solid #555555; border-radius: 3px;
            background-color: #3f3f3f; text-align: center; color: #cccccc;
        }
        QProgressBar::chunk { background-color: #0078d4; }
        QStatusBar { background-color: #1e1e1e; color: #cccccc; border-top: 1px solid #3f3f3f; }
        QCheckBox { color: #cccccc; spacing: 8px; }
        QCheckBox::indicator {
            width: 16px; height: 16px; border: 2px solid #555; border-radius: 3px;
            background-color: #3f3f3f;
        }
        QCheckBox::indicator:checked {
            background-color: #0078d4; border-color: #0078d4;
            image: none;
        }
        QTableWidget {
            background-color: #1e1e1e; color: #cccccc;
            border: 1px solid #3f3f3f; gridline-color: #3f3f3f;
        }
        QTableWidget::item { padding: 3px; }
        QTableWidget::item:selected { background-color: #0078d4; }
        QHeaderView::section {
            background-color: #2b2b2b; color: #cccccc;
            border: 1px solid #3f3f3f; padding: 3px;
        }
        QScrollBar:vertical {
            background-color: #2b2b2b; width: 12px; border-radius: 6px;
        }
        QScrollBar::handle:vertical {
            background-color: #0078d4; border-radius: 4px; min-height: 20px;
        }
        QScrollBar::handle:vertical:hover { background-color: #106ebe; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical { background: none; }
        QScrollBar:horizontal {
            background-color: #2b2b2b; height: 12px; border-radius: 6px;
        }
        QScrollBar::handle:horizontal {
            background-color: #0078d4; border-radius: 4px; min-width: 20px;
        }
        QScrollBar::handle:horizontal:hover { background-color: #106ebe; }
        QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0px; }
        QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal { background: none; }
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple Tracker - FieldNeuroToolbox")
        self.setGeometry(100, 100, 1400, 900)

        # State
        self.video_configs = []
        self.current_config_idx = 0
        self.preview_frame = None
        self.preview_cap = None
        self.preview_tracker = None
        self.auto_preview_enabled = False

        # Drawing state
        self.drawing_mode = None  # None, 'tracking_area', 'oft', 'ldb_light', 'custom', 'scale_bar'
        self.polygon_points = []
        self.scale_bar_points = []
        self.oft_corners = []

        # View mode
        self.view_mode = 'mask'  # 'mask' or 'standard'

        # Worker thread
        self.tracking_worker = None

        # Debounce timer for preview
        self.preview_update_timer = QTimer()
        self.preview_update_timer.setSingleShot(True)
        self.preview_update_timer.timeout.connect(self.update_preview_with_current_settings)

        # Apply theme
        self.setStyleSheet(self.DARK_THEME)
        self.init_ui()

    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() in (Qt.Key_Return, Qt.Key_Enter):
            if self.drawing_mode and len(self.polygon_points) >= 3:
                self.finish_polygon()
        elif event.key() == Qt.Key_Escape:
            if self.drawing_mode:
                self.cancel_drawing()

    # ── UI Initialization ─────────────────────────────────────────────────

    def init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)

        # Left panel - scrollable controls
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, stretch=2)

        # Right panel - preview
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, stretch=3)

        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready - Select videos to begin")

    def create_left_panel(self):
        """Create left control panel with scroll area."""
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        panel = QWidget()
        self.left_layout = QVBoxLayout(panel)
        self.left_layout.setSpacing(4)
        self.left_layout.setContentsMargins(4, 4, 4, 4)

        # Section 1: Video Selection
        self._create_section_video_selection()

        # Section 2: Set Tracking Area
        self._create_section_tracking_area()

        # Section 3: Number of Animals
        self._create_section_num_animals()

        # Section 4: Object Detection Settings
        self._create_section_detection_settings()

        # Section 5: Draw ROIs
        self._create_section_draw_rois()

        # Section 6: Export Options
        self._create_section_export_options()

        # Section 7: Batch Processing
        self._create_section_batch_processing()

        self.left_layout.addStretch()
        scroll_area.setWidget(panel)
        return scroll_area

    def create_right_panel(self):
        """Create right preview panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # Title
        self.preview_title = QLabel("<h2>Preview</h2>")
        self.preview_title.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.preview_title)

        # Drawing instruction banner (hidden by default, shown during drawing)
        self.drawing_instruction_label = QLabel("")
        self.drawing_instruction_label.setAlignment(Qt.AlignCenter)
        self.drawing_instruction_label.setWordWrap(True)
        self.drawing_instruction_label.setStyleSheet(
            "background-color: #1a3a5c; color: #66bbff; "
            "padding: 6px 12px; border-radius: 4px; font-size: 12px; font-weight: bold;"
        )
        self.drawing_instruction_label.hide()
        layout.addWidget(self.drawing_instruction_label)

        # Preview label
        self.preview_label = QLabel("Select videos to begin")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(800, 600)
        self.preview_label.setStyleSheet(
            "background-color: #1e1e1e; border: 1px solid #3f3f3f; "
            "border-radius: 4px; color: #888888; font-size: 14px;"
        )
        self.preview_label.setMouseTracking(True)
        self.preview_label.mousePressEvent = self.on_preview_click
        layout.addWidget(self.preview_label)

        # Frame slider
        frame_slider_layout = QHBoxLayout()
        frame_slider_layout.addWidget(QLabel("Frame:"))

        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setRange(0, 0)
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.on_frame_slider_changed)
        frame_slider_layout.addWidget(self.frame_slider, stretch=1)

        self.frame_label = QLabel("0 / 0")
        self.frame_label.setMinimumWidth(80)
        frame_slider_layout.addWidget(self.frame_label)
        layout.addLayout(frame_slider_layout)

        # Video info
        self.video_info_label = QLabel("No videos selected")
        self.video_info_label.setAlignment(Qt.AlignCenter)
        self.video_info_label.setWordWrap(True)
        self.video_info_label.setStyleSheet(
            "background-color: #2d2d30; color: #cccccc; "
            "padding: 10px; border-radius: 4px; font-size: 12px;"
        )
        layout.addWidget(self.video_info_label)

        # Detection info
        self.detection_info_label = QLabel("")
        self.detection_info_label.setAlignment(Qt.AlignCenter)
        self.detection_info_label.setWordWrap(True)
        layout.addWidget(self.detection_info_label)

        return panel

    # ── Section 1: Video Selection ────────────────────────────────────────

    def _create_section_video_selection(self):
        video_group = QGroupBox("1. Video Selection")
        video_layout = QVBoxLayout()

        btn_layout = QHBoxLayout()
        self.select_video_btn = QPushButton("Select Videos")
        self.select_video_btn.clicked.connect(self.select_videos)
        btn_layout.addWidget(self.select_video_btn)

        self.clear_videos_btn = QPushButton("Clear")
        self.clear_videos_btn.clicked.connect(self.clear_videos)
        self.clear_videos_btn.setStyleSheet(
            "QPushButton { background-color: #d47800; }"
            "QPushButton:hover { background-color: #e68a00; }"
        )
        btn_layout.addWidget(self.clear_videos_btn)
        video_layout.addLayout(btn_layout)

        self.video_list = QListWidget()
        self.video_list.currentRowChanged.connect(self.on_video_selected)
        video_layout.addWidget(self.video_list)

        # Navigation arrows
        nav_layout = QHBoxLayout()
        self.prev_video_btn = QPushButton("\u25C0 Previous")
        self.prev_video_btn.clicked.connect(self.previous_video)
        self.prev_video_btn.setEnabled(False)
        nav_layout.addWidget(self.prev_video_btn)

        self.next_video_btn = QPushButton("Next \u25B6")
        self.next_video_btn.clicked.connect(self.next_video)
        self.next_video_btn.setEnabled(False)
        nav_layout.addWidget(self.next_video_btn)
        video_layout.addLayout(nav_layout)

        video_group.setLayout(video_layout)
        self.left_layout.addWidget(video_group)

    # ── Section 2: Set Tracking Area ──────────────────────────────────────

    def _create_section_tracking_area(self):
        roi_group = QGroupBox("2. Set Tracking Area")
        roi_layout = QVBoxLayout()

        roi_info = QLabel("Define the arena boundary (optional)")
        roi_info.setWordWrap(True)
        roi_info.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        roi_layout.addWidget(roi_info)

        roi_btn_layout = QHBoxLayout()
        self.set_roi_btn = QPushButton("Draw Area")
        self.set_roi_btn.clicked.connect(self.start_tracking_area_drawing)
        self.set_roi_btn.setEnabled(False)
        self.set_roi_btn.setStyleSheet(
            "QPushButton { background-color: #00aa00; }"
            "QPushButton:hover { background-color: #00cc00; }"
            "QPushButton:disabled { background-color: #3f3f3f; color: #888888; }"
        )
        roi_btn_layout.addWidget(self.set_roi_btn)

        self.skip_roi_btn = QPushButton("Skip")
        self.skip_roi_btn.clicked.connect(self.skip_tracking_area)
        self.skip_roi_btn.setEnabled(False)
        self.skip_roi_btn.setStyleSheet(
            "QPushButton { background-color: #666666; }"
            "QPushButton:hover { background-color: #777777; }"
            "QPushButton:disabled { background-color: #3f3f3f; color: #888888; }"
        )
        roi_btn_layout.addWidget(self.skip_roi_btn)

        self.clear_roi_btn = QPushButton("Clear")
        self.clear_roi_btn.clicked.connect(self.clear_tracking_area)
        self.clear_roi_btn.setEnabled(False)
        self.clear_roi_btn.setStyleSheet(
            "QPushButton { background-color: #d47800; }"
            "QPushButton:hover { background-color: #e68a00; }"
            "QPushButton:disabled { background-color: #3f3f3f; color: #888888; }"
        )
        roi_btn_layout.addWidget(self.clear_roi_btn)
        roi_layout.addLayout(roi_btn_layout)

        self.roi_status_label = QLabel("Click Draw Area or Skip to continue")
        self.roi_status_label.setStyleSheet("color: #FFA500; font-size: 10px;")
        roi_layout.addWidget(self.roi_status_label)

        roi_group.setLayout(roi_layout)
        self.left_layout.addWidget(roi_group)

    # ── Section 3: Number of Animals ─────────────────────────────────────

    def _create_section_num_animals(self):
        group = QGroupBox("3. Number of Animals")
        layout = QVBoxLayout()

        h = QHBoxLayout()
        h.addWidget(QLabel("Animals to track:"))
        self.num_animals_spin = QSpinBox()
        self.num_animals_spin.setRange(1, 20)
        self.num_animals_spin.setValue(1)
        self.num_animals_spin.valueChanged.connect(self._on_num_animals_changed)
        h.addWidget(self.num_animals_spin)
        h.addStretch()
        layout.addLayout(h)

        self.num_animals_info = QLabel(
            "Set to 1 for single-animal tracking (only largest blob used)"
        )
        self.num_animals_info.setWordWrap(True)
        self.num_animals_info.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        layout.addWidget(self.num_animals_info)

        group.setLayout(layout)
        self.left_layout.addWidget(group)

    def _on_num_animals_changed(self, value):
        if self.video_configs:
            self.video_configs[self.current_config_idx].num_animals = value
        if value == 1:
            self.num_animals_info.setText(
                "Set to 1 for single-animal tracking (only largest blob used)"
            )
        else:
            self.num_animals_info.setText(
                f"Tracking top {value} blobs by area with ID assignment"
            )
        self._schedule_preview_update()

    # ── Section 4: Object Detection Settings ──────────────────────────────

    def _create_section_detection_settings(self):
        group = QGroupBox("4. Object Detection Settings")
        settings_layout = QVBoxLayout()

        # View toggle
        view_layout = QHBoxLayout()
        self.btn_mask_view = QPushButton("Background Subtraction View")
        self.btn_mask_view.setCheckable(True)
        self.btn_mask_view.setChecked(True)
        self.btn_mask_view.clicked.connect(lambda: self._set_view_mode('mask'))
        view_layout.addWidget(self.btn_mask_view)

        self.btn_standard_view = QPushButton("Standard View")
        self.btn_standard_view.setCheckable(True)
        self.btn_standard_view.setChecked(False)
        self.btn_standard_view.clicked.connect(lambda: self._set_view_mode('standard'))
        view_layout.addWidget(self.btn_standard_view)
        settings_layout.addLayout(view_layout)
        self._update_view_toggle_styles()

        # Object size range (dual-handle slider)
        self.area_range_label = QLabel("Object Size: 100 – 10000 px²")
        settings_layout.addWidget(self.area_range_label)

        area_spin_h = QHBoxLayout()
        area_spin_h.addWidget(QLabel("Min:"))
        self.min_area_spinbox = QSpinBox()
        self.min_area_spinbox.setRange(10, 50000)
        self.min_area_spinbox.setValue(100)
        self.min_area_spinbox.setSuffix(" px²")
        self.min_area_spinbox.valueChanged.connect(self._on_min_spinbox_changed)
        self.min_area_spinbox.setEnabled(False)
        area_spin_h.addWidget(self.min_area_spinbox)
        area_spin_h.addWidget(QLabel("Max:"))
        self.max_area_spinbox = QSpinBox()
        self.max_area_spinbox.setRange(10, 50000)
        self.max_area_spinbox.setValue(10000)
        self.max_area_spinbox.setSuffix(" px²")
        self.max_area_spinbox.valueChanged.connect(self._on_max_spinbox_changed)
        self.max_area_spinbox.setEnabled(False)
        area_spin_h.addWidget(self.max_area_spinbox)
        settings_layout.addLayout(area_spin_h)

        self.area_range_slider = RangeSlider(10, 50000)
        self.area_range_slider.setLow(100)
        self.area_range_slider.setHigh(10000)
        self.area_range_slider.rangeChanged.connect(self._on_area_range_changed)
        self.area_range_slider.sliderReleased.connect(self._on_slider_released)
        self.area_range_slider.setEnabled(False)
        settings_layout.addWidget(self.area_range_slider)

        # Sensitivity
        self.sensitivity_label = QLabel("Sensitivity: 16")
        settings_layout.addWidget(self.sensitivity_label)
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(1, 100)
        self.sensitivity_slider.setValue(16)
        self.sensitivity_slider.setToolTip("Lower = more sensitive. Higher = less sensitive.")
        self.sensitivity_slider.valueChanged.connect(self._on_sensitivity_changed)
        self.sensitivity_slider.sliderReleased.connect(self._on_slider_released)
        self.sensitivity_slider.setEnabled(False)
        settings_layout.addWidget(self.sensitivity_slider)

        # Contrast
        self.contrast_label = QLabel("Contrast: 1.0x")
        settings_layout.addWidget(self.contrast_label)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(50, 300)
        self.contrast_slider.setValue(100)
        self.contrast_slider.valueChanged.connect(self._on_contrast_changed)
        self.contrast_slider.sliderReleased.connect(self._on_slider_released)
        self.contrast_slider.setEnabled(False)
        settings_layout.addWidget(self.contrast_slider)

        # Brightness
        self.brightness_label = QLabel("Brightness: 0")
        settings_layout.addWidget(self.brightness_label)
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.valueChanged.connect(self._on_brightness_changed)
        self.brightness_slider.sliderReleased.connect(self._on_slider_released)
        self.brightness_slider.setEnabled(False)
        settings_layout.addWidget(self.brightness_slider)

        # Blob Refinement sub-group
        refine_group = QGroupBox("Blob Refinement")
        refine_layout = QVBoxLayout()

        # Opening kernel
        h = QHBoxLayout()
        h.addWidget(QLabel("Opening Kernel:"))
        self.open_kernel_spin = QSpinBox()
        self.open_kernel_spin.setRange(1, 15)
        self.open_kernel_spin.setValue(3)
        self.open_kernel_spin.setSingleStep(2)
        self.open_kernel_spin.valueChanged.connect(self._on_refinement_changed)
        h.addWidget(self.open_kernel_spin)
        h.addWidget(QLabel("Iter:"))
        self.open_iter_spin = QSpinBox()
        self.open_iter_spin.setRange(0, 10)
        self.open_iter_spin.setValue(1)
        self.open_iter_spin.valueChanged.connect(self._on_refinement_changed)
        h.addWidget(self.open_iter_spin)
        refine_layout.addLayout(h)

        # Closing kernel
        h2 = QHBoxLayout()
        h2.addWidget(QLabel("Closing Kernel:"))
        self.close_kernel_spin = QSpinBox()
        self.close_kernel_spin.setRange(1, 15)
        self.close_kernel_spin.setValue(5)
        self.close_kernel_spin.setSingleStep(2)
        self.close_kernel_spin.valueChanged.connect(self._on_refinement_changed)
        h2.addWidget(self.close_kernel_spin)
        h2.addWidget(QLabel("Iter:"))
        self.close_iter_spin = QSpinBox()
        self.close_iter_spin.setRange(0, 10)
        self.close_iter_spin.setValue(2)
        self.close_iter_spin.valueChanged.connect(self._on_refinement_changed)
        h2.addWidget(self.close_iter_spin)
        refine_layout.addLayout(h2)

        # Gaussian blur
        h3 = QHBoxLayout()
        h3.addWidget(QLabel("Gaussian Blur:"))
        self.blur_spin = QSpinBox()
        self.blur_spin.setRange(0, 31)
        self.blur_spin.setValue(0)
        self.blur_spin.setSingleStep(2)
        self.blur_spin.setToolTip("0 = off, odd numbers (3, 5, 7...)")
        self.blur_spin.valueChanged.connect(self._on_refinement_changed)
        h3.addWidget(self.blur_spin)
        refine_layout.addLayout(h3)

        # Checkboxes
        self.chk_fill_holes = DarkCheckBox("Fill Holes")
        self.chk_fill_holes.setChecked(True)
        self.chk_fill_holes.stateChanged.connect(self._on_refinement_changed)
        refine_layout.addWidget(self.chk_fill_holes)

        self.chk_convex_hull = DarkCheckBox("Convex Hull")
        self.chk_convex_hull.setChecked(False)
        self.chk_convex_hull.stateChanged.connect(self._on_refinement_changed)
        refine_layout.addWidget(self.chk_convex_hull)

        refine_group.setLayout(refine_layout)
        settings_layout.addWidget(refine_group)

        # Info
        info_label = QLabel(
            "<b>Settings:</b><br>"
            "\u2022 <b>Min/Max Size:</b> Filter objects by area<br>"
            "\u2022 <b>Sensitivity:</b> Background subtraction threshold<br>"
            "\u2022 Changes apply automatically on slider release"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #aaaaaa; font-size: 10px; padding: 2px;")
        settings_layout.addWidget(info_label)

        group.setLayout(settings_layout)
        self.left_layout.addWidget(group)

    # ── Section 4 Event Handlers ──────────────────────────────────────────

    def _set_view_mode(self, mode):
        self.view_mode = mode
        self._update_view_toggle_styles()
        self._schedule_preview_update()

    def _update_view_toggle_styles(self):
        active = "QPushButton { background-color: #0078d4; color: white; font-weight: bold; }"
        inactive = "QPushButton { background-color: #555555; color: #aaaaaa; }"
        if self.view_mode == 'mask':
            self.btn_mask_view.setStyleSheet(active)
            self.btn_standard_view.setStyleSheet(inactive)
            self.btn_mask_view.setChecked(True)
            self.btn_standard_view.setChecked(False)
        else:
            self.btn_mask_view.setStyleSheet(inactive)
            self.btn_standard_view.setStyleSheet(active)
            self.btn_mask_view.setChecked(False)
            self.btn_standard_view.setChecked(True)

    def _on_area_range_changed(self, low, high):
        """Handle range slider drag updates."""
        self.min_area_spinbox.blockSignals(True)
        self.max_area_spinbox.blockSignals(True)
        self.min_area_spinbox.setValue(low)
        self.max_area_spinbox.setValue(high)
        self.min_area_spinbox.blockSignals(False)
        self.max_area_spinbox.blockSignals(False)
        self.area_range_label.setText(f"Object Size: {low} – {high} px²")

    def _on_min_spinbox_changed(self, value):
        if value > self.max_area_spinbox.value():
            self.min_area_spinbox.blockSignals(True)
            self.min_area_spinbox.setValue(self.max_area_spinbox.value())
            self.min_area_spinbox.blockSignals(False)
            value = self.max_area_spinbox.value()
        self.area_range_slider.setLow(value)
        self.area_range_label.setText(
            f"Object Size: {value} – {self.max_area_spinbox.value()} px²"
        )
        self._update_current_config()
        self._schedule_preview_update()

    def _on_max_spinbox_changed(self, value):
        if value < self.min_area_spinbox.value():
            self.max_area_spinbox.blockSignals(True)
            self.max_area_spinbox.setValue(self.min_area_spinbox.value())
            self.max_area_spinbox.blockSignals(False)
            value = self.min_area_spinbox.value()
        self.area_range_slider.setHigh(value)
        self.area_range_label.setText(
            f"Object Size: {self.min_area_spinbox.value()} – {value} px²"
        )
        self._update_current_config()
        self._schedule_preview_update()

    def _on_sensitivity_changed(self, value):
        self.sensitivity_label.setText(f"Sensitivity: {value}")

    def _on_contrast_changed(self, value):
        self.contrast_label.setText(f"Contrast: {value / 100.0:.1f}x")

    def _on_brightness_changed(self, value):
        self.brightness_label.setText(f"Brightness: {value:+d}")

    def _on_slider_released(self):
        self._update_current_config()
        self._schedule_preview_update()

    def _on_refinement_changed(self, _=None):
        self._update_current_config()
        self._schedule_preview_update()

    def _schedule_preview_update(self):
        """Debounced preview update."""
        self.preview_update_timer.start(300)

    def _update_current_config(self):
        """Sync all UI values to current video config."""
        if not self.video_configs:
            return
        config = self.video_configs[self.current_config_idx]
        config.num_animals = self.num_animals_spin.value()
        config.min_object_area = self.area_range_slider.low()
        config.max_object_area = self.area_range_slider.high()
        config.var_threshold = self.sensitivity_slider.value()
        config.contrast = self.contrast_slider.value() / 100.0
        config.brightness = self.brightness_slider.value()
        config.morph_open_size = self.open_kernel_spin.value()
        config.morph_close_size = self.close_kernel_spin.value()
        config.morph_open_iterations = self.open_iter_spin.value()
        config.morph_close_iterations = self.close_iter_spin.value()
        config.gaussian_blur_size = self.blur_spin.value()
        config.fill_holes = self.chk_fill_holes.isChecked()
        config.convex_hull = self.chk_convex_hull.isChecked()
        config.configured = True
        self.update_config_status()

    def _enable_detection_settings(self):
        """Enable detection controls after tracking area is set."""
        self.area_range_slider.setEnabled(True)
        self.min_area_spinbox.setEnabled(True)
        self.max_area_spinbox.setEnabled(True)
        self.sensitivity_slider.setEnabled(True)
        self.contrast_slider.setEnabled(True)
        self.brightness_slider.setEnabled(True)

    def _disable_detection_settings(self):
        """Disable detection controls when tracking area not set."""
        self.area_range_slider.setEnabled(False)
        self.min_area_spinbox.setEnabled(False)
        self.max_area_spinbox.setEnabled(False)
        self.sensitivity_slider.setEnabled(False)
        self.contrast_slider.setEnabled(False)
        self.brightness_slider.setEnabled(False)

    # ── Section 5: Draw ROIs ─────────────────────────────────────────────

    def _create_section_draw_rois(self):
        group = QGroupBox("5. Draw ROIs")
        layout = QVBoxLayout()

        self.btn_oft = QPushButton("Open Field Test Layout")
        self.btn_oft.clicked.connect(self._start_oft_drawing)
        layout.addWidget(self.btn_oft)

        self.btn_ldb = QPushButton("Light-Dark Box Layout")
        self.btn_ldb.clicked.connect(self._start_ldb_drawing)
        layout.addWidget(self.btn_ldb)

        self.btn_custom_roi = QPushButton("Custom ROI")
        self.btn_custom_roi.clicked.connect(self._start_custom_roi)
        layout.addWidget(self.btn_custom_roi)

        self.btn_scale_bar = QPushButton("Set Scale Bar (optional)")
        self.btn_scale_bar.clicked.connect(self._start_scale_bar)
        self.btn_scale_bar.setStyleSheet(
            "QPushButton { background-color: #666666; }"
            "QPushButton:hover { background-color: #777777; }"
        )
        layout.addWidget(self.btn_scale_bar)

        self.scale_status_label = QLabel("")
        self.scale_status_label.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        layout.addWidget(self.scale_status_label)

        # ROI priority table
        self.roi_table = QTableWidget()
        self.roi_table.setColumnCount(1)
        self.roi_table.setHorizontalHeaderLabels(["ROI Name (priority order)"])
        self.roi_table.horizontalHeader().setStretchLastSection(True)
        self.roi_table.setMaximumHeight(150)
        self.roi_table.itemChanged.connect(self._on_roi_renamed)
        layout.addWidget(self.roi_table)

        roi_ctrl = QHBoxLayout()
        self.roi_up_btn = QPushButton("\u25B2")
        self.roi_up_btn.setMaximumWidth(40)
        self.roi_up_btn.clicked.connect(self._roi_move_up)
        roi_ctrl.addWidget(self.roi_up_btn)
        self.roi_down_btn = QPushButton("\u25BC")
        self.roi_down_btn.setMaximumWidth(40)
        self.roi_down_btn.clicked.connect(self._roi_move_down)
        roi_ctrl.addWidget(self.roi_down_btn)
        roi_ctrl.addStretch()
        self.roi_clear_btn = QPushButton("Clear ROIs")
        self.roi_clear_btn.clicked.connect(self._clear_all_rois)
        self.roi_clear_btn.setStyleSheet(
            "QPushButton { background-color: #d47800; }"
            "QPushButton:hover { background-color: #e68a00; }"
        )
        roi_ctrl.addWidget(self.roi_clear_btn)
        layout.addLayout(roi_ctrl)

        group.setLayout(layout)
        self.left_layout.addWidget(group)

    # ── ROI Drawing Methods ───────────────────────────────────────────────

    def _show_drawing_instructions(self, text):
        """Show drawing instructions above the preview pane."""
        self.drawing_instruction_label.setText(text)
        self.drawing_instruction_label.show()

    def _hide_drawing_instructions(self):
        """Hide drawing instructions."""
        self.drawing_instruction_label.hide()

    def _start_oft_drawing(self):
        """Start OFT layout: 4 corner clicks."""
        if not self.video_configs:
            return
        self.drawing_mode = 'oft'
        self.oft_corners = []
        self._show_drawing_instructions("OFT: Click 4 corners of the outer boundary")
        self.status_bar.showMessage("OFT: Click 4 corners of the outer boundary")

    def _start_ldb_drawing(self):
        """Start LDB layout: draw light box polygon."""
        if not self.video_configs:
            return
        self.drawing_mode = 'ldb_light'
        self.polygon_points = []
        self._show_drawing_instructions(
            "LDB: Click to draw the LIGHT box boundary  •  Enter = finish  •  Esc = cancel"
        )
        self.status_bar.showMessage(
            "LDB: Click to draw the LIGHT box boundary. Enter to finish."
        )

    def _start_custom_roi(self):
        """Start custom ROI polygon drawing."""
        if not self.video_configs:
            return
        self.drawing_mode = 'custom'
        self.polygon_points = []
        self._show_drawing_instructions(
            "Custom ROI: Click to draw polygon  •  Enter = finish  •  Esc = cancel"
        )
        self.status_bar.showMessage(
            "Custom ROI: Click to draw polygon. Enter to finish."
        )

    def _start_scale_bar(self):
        """Start scale bar: 2-point click."""
        if not self.video_configs:
            return
        self.drawing_mode = 'scale_bar'
        self.scale_bar_points = []
        self._show_drawing_instructions(
            "Scale bar: Click two endpoints of a known distance  •  Esc = cancel"
        )
        self.status_bar.showMessage(
            "Scale bar: Click two endpoints of a known distance"
        )

    def _finish_oft(self):
        """Finish OFT drawing from 4 corners: create outer + center ROIs."""
        if len(self.oft_corners) != 4:
            return
        config = self.video_configs[self.current_config_idx]
        self._hide_drawing_instructions()

        # Outer boundary = the 4 corners
        outer_polygon = self.oft_corners.copy()
        config.add_roi('oft_outer', outer_polygon)

        # Calculate center zone (50% area scaled from centroid)
        center = self._calculate_oft_center(outer_polygon)
        config.add_roi('oft_center', center)

        self.drawing_mode = None
        self.oft_corners = []
        self._update_roi_table()
        self._redraw_preview()
        self.status_bar.showMessage("OFT layout created (outer + center)")

        # Auto-transition to scale bar
        self._prompt_scale_bar()

    def _calculate_oft_center(self, outer_polygon):
        """Calculate OFT center zone as 50% area scaled polygon from centroid."""
        pts = np.array(outer_polygon, dtype=np.float64)
        cx = np.mean(pts[:, 0])
        cy = np.mean(pts[:, 1])
        scale = np.sqrt(0.5)  # 50% area = sqrt(0.5) linear scale
        center_pts = []
        for x, y in outer_polygon:
            nx = cx + (x - cx) * scale
            ny = cy + (y - cy) * scale
            center_pts.append((int(nx), int(ny)))
        return center_pts

    def _finish_scale_bar(self):
        """Finish scale bar measurement."""
        if len(self.scale_bar_points) != 2:
            return
        config = self.video_configs[self.current_config_idx]
        self._hide_drawing_instructions()

        p1, p2 = self.scale_bar_points
        pixel_dist = np.sqrt((p1[0] - p2[0])**2 + (p1[1] - p2[1])**2)

        cm_val, ok = QInputDialog.getDouble(
            self, "Scale Bar", "Enter the real-world distance (cm):",
            value=10.0, min=0.01, max=10000.0, decimals=2
        )
        if ok:
            config.scale_bar_set = True
            config.scale_bar_pixels = pixel_dist
            config.scale_bar_cm = cm_val
            config.pixels_per_cm = pixel_dist / cm_val
            self.scale_status_label.setText(
                f"Scale: {config.pixels_per_cm:.1f} px/cm "
                f"({pixel_dist:.0f} px = {cm_val} cm)"
            )
            self.scale_status_label.setStyleSheet("color: #90EE90; font-size: 10px;")

        self.drawing_mode = None
        self.scale_bar_points = []
        self._redraw_preview()

    def _prompt_scale_bar(self):
        """Ask user if they want to set a scale bar after ROI drawing."""
        reply = QMessageBox.question(
            self, "Scale Bar",
            "Would you like to set a scale bar for distance measurements?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply == QMessageBox.Yes:
            self._start_scale_bar()

    # ── ROI Table Methods ─────────────────────────────────────────────────

    def _update_roi_table(self):
        """Sync ROI table with current config."""
        if not self.video_configs:
            return
        config = self.video_configs[self.current_config_idx]

        self.roi_table.blockSignals(True)
        self.roi_table.setRowCount(len(config.rois))
        for i, (name, _) in enumerate(config.rois):
            item = QTableWidgetItem(name)
            self.roi_table.setItem(i, 0, item)
        self.roi_table.blockSignals(False)

    def _on_roi_renamed(self, item):
        """Handle ROI rename in table."""
        if not self.video_configs:
            return
        config = self.video_configs[self.current_config_idx]
        row = item.row()
        if 0 <= row < len(config.rois):
            old_name = config.rois[row][0]
            new_name = item.text().strip()
            if new_name and new_name != old_name:
                config.rename_roi(old_name, new_name)
                self._redraw_preview()

    def _roi_move_up(self):
        """Move selected ROI up in priority."""
        row = self.roi_table.currentRow()
        if row <= 0 or not self.video_configs:
            return
        config = self.video_configs[self.current_config_idx]
        config.rois[row], config.rois[row - 1] = config.rois[row - 1], config.rois[row]
        self._update_roi_table()
        self.roi_table.setCurrentCell(row - 1, 0)

    def _roi_move_down(self):
        """Move selected ROI down in priority."""
        if not self.video_configs:
            return
        config = self.video_configs[self.current_config_idx]
        row = self.roi_table.currentRow()
        if row < 0 or row >= len(config.rois) - 1:
            return
        config.rois[row], config.rois[row + 1] = config.rois[row + 1], config.rois[row]
        self._update_roi_table()
        self.roi_table.setCurrentCell(row + 1, 0)

    def _clear_all_rois(self):
        """Clear all ROIs and scale bar."""
        if not self.video_configs:
            return
        config = self.video_configs[self.current_config_idx]
        config.rois = []
        config.scale_bar_set = False
        config.scale_bar_pixels = None
        config.scale_bar_cm = None
        config.pixels_per_cm = None
        self.scale_status_label.setText("")
        self._update_roi_table()
        self._redraw_preview()
        self.status_bar.showMessage("ROIs and scale bar cleared")

    # ── Section 6: Export Options ─────────────────────────────────────────

    def _create_section_export_options(self):
        group = QGroupBox("6. Export Options")
        layout = QVBoxLayout()

        self.chk_save_coords = DarkCheckBox("Save Position Coordinates CSV")
        self.chk_save_coords.setChecked(True)
        layout.addWidget(self.chk_save_coords)

        self.chk_save_occupancy = DarkCheckBox("Save ROI Occupancy CSV")
        self.chk_save_occupancy.setChecked(True)
        layout.addWidget(self.chk_save_occupancy)

        self.chk_save_summary = DarkCheckBox("Save ROI Summary CSV")
        self.chk_save_summary.setChecked(True)
        layout.addWidget(self.chk_save_summary)

        self.chk_save_config = DarkCheckBox("Save Configuration JSON")
        self.chk_save_config.setChecked(True)
        layout.addWidget(self.chk_save_config)

        self.chk_save_plots = DarkCheckBox("Save Tracking Plots")
        self.chk_save_plots.setChecked(True)
        layout.addWidget(self.chk_save_plots)

        self.chk_tracked_video = DarkCheckBox("Create Tracked Video")
        self.chk_tracked_video.setChecked(True)
        self.chk_tracked_video.stateChanged.connect(self._on_tracked_video_toggled)
        layout.addWidget(self.chk_tracked_video)

        # Indented video sub-options
        self.video_opts_widget = QWidget()
        vopts = QVBoxLayout(self.video_opts_widget)
        vopts.setContentsMargins(20, 0, 0, 0)
        vopts.setSpacing(2)

        self.chk_show_area = DarkCheckBox("Show tracking area boundary")
        self.chk_show_area.setChecked(True)
        vopts.addWidget(self.chk_show_area)

        self.chk_show_rois = DarkCheckBox("Show ROI boundaries")
        self.chk_show_rois.setChecked(True)
        vopts.addWidget(self.chk_show_rois)

        self.chk_show_ids = DarkCheckBox("Show blob ID labels")
        self.chk_show_ids.setChecked(True)
        vopts.addWidget(self.chk_show_ids)

        trail_h = QHBoxLayout()
        self.chk_show_trail = DarkCheckBox("Show trail:")
        self.chk_show_trail.setChecked(True)
        trail_h.addWidget(self.chk_show_trail)
        self.trail_length_spin = QSpinBox()
        self.trail_length_spin.setRange(5, 300)
        self.trail_length_spin.setValue(30)
        self.trail_length_spin.setSuffix(" frames")
        trail_h.addWidget(self.trail_length_spin)
        trail_h.addStretch()
        vopts.addLayout(trail_h)

        self.chk_data_view = DarkCheckBox("Data view (video + live stats)")
        self.chk_data_view.setChecked(True)
        vopts.addWidget(self.chk_data_view)

        layout.addWidget(self.video_opts_widget)

        self.chk_interpolate = DarkCheckBox("Interpolate tracks (fill gaps)")
        self.chk_interpolate.setChecked(True)
        layout.addWidget(self.chk_interpolate)

        self.chk_overwrite = DarkCheckBox("Overwrite existing output files")
        self.chk_overwrite.setChecked(True)
        layout.addWidget(self.chk_overwrite)

        group.setLayout(layout)
        self.left_layout.addWidget(group)

    def _on_tracked_video_toggled(self, state):
        self.video_opts_widget.setEnabled(state == Qt.Checked)

    # ── Section 7: Batch Processing ───────────────────────────────────────

    def _create_section_batch_processing(self):
        group = QGroupBox("7. Batch Processing")
        batch_layout = QVBoxLayout()

        self.add_queue_btn = QPushButton("Add Video to Queue")
        self.add_queue_btn.clicked.connect(self._add_to_queue)
        batch_layout.addWidget(self.add_queue_btn)

        self.queue_list = QListWidget()
        self.queue_list.setMaximumHeight(150)
        batch_layout.addWidget(self.queue_list)

        self.start_batch_btn = QPushButton("Start Batch Processing")
        self.start_batch_btn.clicked.connect(self.start_batch_tracking)
        self.start_batch_btn.setEnabled(False)
        self.start_batch_btn.setStyleSheet(
            "QPushButton { background-color: #0078d4; font-weight: bold; font-size: 14px; }"
            "QPushButton:hover { background-color: #0086f0; }"
            "QPushButton:disabled { background-color: #3f3f3f; color: #888888; }"
        )
        batch_layout.addWidget(self.start_batch_btn)

        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_tracking)
        self.cancel_btn.setEnabled(False)
        batch_layout.addWidget(self.cancel_btn)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        batch_layout.addWidget(self.progress_bar)

        self.progress_label = QLabel("Ready")
        self.progress_label.setAlignment(Qt.AlignCenter)
        batch_layout.addWidget(self.progress_label)

        group.setLayout(batch_layout)
        self.left_layout.addWidget(group)

    # ── Batch Processing Methods ──────────────────────────────────────────

    def _add_to_queue(self):
        """Add current video to processing queue."""
        if not self.video_configs:
            return
        config = self.video_configs[self.current_config_idx]

        # Sync export options to config
        self._sync_export_options(config)

        config.configured = True
        self.update_config_status()

        # Update video list icon
        item = self.video_list.item(self.current_config_idx)
        if item:
            name = Path(config.video_path).name
            item.setText(f"\u2713 {self.current_config_idx + 1}. {name}")

        # Add to queue list if not already there
        already = any(
            self.queue_list.item(i).data(Qt.UserRole) == self.current_config_idx
            for i in range(self.queue_list.count())
        )
        if not already:
            q_item = QListWidgetItem(Path(config.video_path).name)
            q_item.setData(Qt.UserRole, self.current_config_idx)
            self.queue_list.addItem(q_item)

        self.start_batch_btn.setEnabled(self.queue_list.count() > 0)

        # Auto-advance to next video
        if self.current_config_idx < len(self.video_configs) - 1:
            self.next_video()

        self.status_bar.showMessage(
            f"Added to queue ({self.queue_list.count()} videos)"
        )

    def _sync_export_options(self, config):
        """Sync export option checkboxes to config."""
        config.save_position_coords = self.chk_save_coords.isChecked()
        config.save_roi_occupancy = self.chk_save_occupancy.isChecked()
        config.save_roi_summary = self.chk_save_summary.isChecked()
        config.save_config = self.chk_save_config.isChecked()
        config.save_tracking_plots = self.chk_save_plots.isChecked()
        config.create_tracked_video = self.chk_tracked_video.isChecked()
        config.show_tracking_area = self.chk_show_area.isChecked()
        config.show_rois = self.chk_show_rois.isChecked()
        config.show_blob_ids = self.chk_show_ids.isChecked()
        config.show_trail = self.chk_show_trail.isChecked()
        config.trail_length = self.trail_length_spin.value()
        config.save_data_view = self.chk_data_view.isChecked()
        config.interpolate_tracks = self.chk_interpolate.isChecked()
        config.overwrite_files = self.chk_overwrite.isChecked()

    def start_batch_tracking(self):
        """Start batch processing queued videos."""
        if self.queue_list.count() == 0:
            return

        # Collect configs for queued videos
        queued_configs = []
        for i in range(self.queue_list.count()):
            idx = self.queue_list.item(i).data(Qt.UserRole)
            queued_configs.append(self.video_configs[idx])

        self.tracking_worker = BatchTrackingWorker(queued_configs)
        self.tracking_worker.progress.connect(self._update_progress)
        self.tracking_worker.status.connect(self._update_status)
        self.tracking_worker.video_finished.connect(self._video_finished)
        self.tracking_worker.all_finished.connect(self._all_finished)
        self.tracking_worker.start()

        self.start_batch_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.select_video_btn.setEnabled(False)
        self.status_bar.showMessage("Batch processing started...")

    def cancel_tracking(self):
        if self.tracking_worker:
            self.tracking_worker.cancel()
            self.cancel_btn.setEnabled(False)
            self.status_bar.showMessage("Cancelling...")

    def _update_progress(self, video_idx, frame_idx, total_frames):
        total_videos = self.queue_list.count()
        video_pct = (video_idx / total_videos) * 100
        frame_pct = (frame_idx / max(1, total_frames)) * (100 / total_videos)
        self.progress_bar.setValue(int(video_pct + frame_pct))
        self.progress_label.setText(
            f"Video {video_idx + 1}/{total_videos} - Frame {frame_idx}/{total_frames}"
        )

    def _update_status(self, message):
        self.status_bar.showMessage(message)

    def _video_finished(self, video_idx, success, message):
        item = self.queue_list.item(video_idx)
        if item:
            if success:
                item.setText(f"\u2713 {item.text()}")
            else:
                item.setText(f"\u2717 {item.text()}")

    def _all_finished(self, success, message):
        self.start_batch_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.select_video_btn.setEnabled(True)
        self.progress_bar.setValue(100 if success else 0)

        if success:
            QMessageBox.information(self, "Batch Complete", message)
        else:
            QMessageBox.warning(self, "Batch Incomplete", message)
        self.status_bar.showMessage(message)

    # ── Video Selection Methods ───────────────────────────────────────────

    def select_videos(self):
        """Select video files for batch processing."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self, "Select Video Files", "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        )
        if not file_paths:
            return

        self.video_configs = [VideoTrackingConfig(path) for path in file_paths]
        self.current_config_idx = 0

        self.video_list.clear()
        for idx, config in enumerate(self.video_configs):
            item_text = f"\u25CB {idx + 1}. {Path(config.video_path).name}"
            self.video_list.addItem(QListWidgetItem(item_text))

        self.video_list.setCurrentRow(0)
        self.auto_preview_enabled = True
        self.load_first_frame_for_current_video()
        self.update_config_status()
        self._update_nav_buttons()
        self.status_bar.showMessage(f"Loaded {len(file_paths)} video(s)")

    def clear_videos(self):
        """Clear all selected videos."""
        self.video_configs = []
        self.video_list.clear()
        self.current_config_idx = 0
        self.preview_label.setText("Select videos to begin")
        self.detection_info_label.setText("")
        self.video_info_label.setText("No videos selected")
        self.frame_slider.setEnabled(False)
        self.frame_slider.setRange(0, 0)
        self.frame_label.setText("0 / 0")
        self._update_nav_buttons()
        self.status_bar.showMessage("Ready - Select videos to begin")

    def on_video_selected(self, row):
        """Handle video selection from list."""
        if 0 <= row < len(self.video_configs):
            self.current_config_idx = row
            self.load_current_config()
            self.update_config_status()
            self._update_nav_buttons()
            if self.auto_preview_enabled:
                self.load_first_frame_for_current_video()

    def previous_video(self):
        if self.current_config_idx > 0:
            self.current_config_idx -= 1
            self.video_list.setCurrentRow(self.current_config_idx)

    def next_video(self):
        if self.current_config_idx < len(self.video_configs) - 1:
            self.current_config_idx += 1
            self.video_list.setCurrentRow(self.current_config_idx)

    def _update_nav_buttons(self):
        n = len(self.video_configs)
        self.prev_video_btn.setEnabled(self.current_config_idx > 0)
        self.next_video_btn.setEnabled(self.current_config_idx < n - 1)

    def load_current_config(self):
        """Load current video configuration into UI controls."""
        if not self.video_configs:
            return
        config = self.video_configs[self.current_config_idx]

        # Update slider ranges if video has been loaded
        if config.width and config.height:
            max_area = config.width * config.height
            self.area_range_slider.setRange(10, max_area)
            self.min_area_spinbox.setRange(10, max_area)
            self.max_area_spinbox.setRange(10, max_area)

        # Block signals while syncing
        for w in (self.sensitivity_slider,
                  self.contrast_slider, self.brightness_slider, self.min_area_spinbox,
                  self.max_area_spinbox, self.num_animals_spin, self.open_kernel_spin,
                  self.close_kernel_spin, self.open_iter_spin, self.close_iter_spin,
                  self.blur_spin, self.chk_fill_holes, self.chk_convex_hull):
            w.blockSignals(True)

        self.area_range_slider.setLow(config.min_object_area)
        self.area_range_slider.setHigh(config.max_object_area)
        self.min_area_spinbox.setValue(config.min_object_area)
        self.max_area_spinbox.setValue(config.max_object_area)
        self.sensitivity_slider.setValue(config.var_threshold)
        self.contrast_slider.setValue(int(config.contrast * 100))
        self.brightness_slider.setValue(config.brightness)
        self.num_animals_spin.setValue(config.num_animals)
        self.open_kernel_spin.setValue(config.morph_open_size)
        self.close_kernel_spin.setValue(config.morph_close_size)
        self.open_iter_spin.setValue(config.morph_open_iterations)
        self.close_iter_spin.setValue(config.morph_close_iterations)
        self.blur_spin.setValue(config.gaussian_blur_size)
        self.chk_fill_holes.setChecked(config.fill_holes)
        self.chk_convex_hull.setChecked(config.convex_hull)

        for w in (self.sensitivity_slider,
                  self.contrast_slider, self.brightness_slider, self.min_area_spinbox,
                  self.max_area_spinbox, self.num_animals_spin, self.open_kernel_spin,
                  self.close_kernel_spin, self.open_iter_spin, self.close_iter_spin,
                  self.blur_spin, self.chk_fill_holes, self.chk_convex_hull):
            w.blockSignals(False)

        # Update labels
        self.area_range_label.setText(
            f"Object Size: {config.min_object_area} – {config.max_object_area} px²"
        )
        self.sensitivity_label.setText(f"Sensitivity: {config.var_threshold}")
        self.contrast_label.setText(f"Contrast: {config.contrast:.1f}x")
        self.brightness_label.setText(f"Brightness: {config.brightness:+d}")

    def update_config_status(self):
        """Update configuration status in preview panel."""
        if not self.video_configs:
            self.video_info_label.setText("No videos selected")
            return

        configured_count = sum(1 for c in self.video_configs if c.configured)
        total = len(self.video_configs)
        config = self.video_configs[self.current_config_idx]

        self.video_info_label.setText(
            f"<b>Video {self.current_config_idx + 1} of {total}</b> - "
            f"{Path(config.video_path).name}<br>"
            f"<b>Configured:</b> {configured_count}/{total} | "
            f"<b>Status:</b> {'<span style=\"color:#90EE90\">Configured</span>' if config.configured else '<span style=\"color:#FFA500\">Not configured</span>'}"
        )
        self.preview_title.setText(
            f"<h2>Preview - Video {self.current_config_idx + 1}/{total}</h2>"
        )

    def load_first_frame_for_current_video(self):
        """Load first frame and video properties for current video."""
        if not self.video_configs:
            return

        config = self.video_configs[self.current_config_idx]

        if config.first_frame is not None:
            self.display_frame(config.first_frame)
            self.frame_slider.setRange(0, max(0, config.total_frames - 1))
            self.frame_slider.setValue(config.current_frame_idx)
            self.frame_label.setText(f"{config.current_frame_idx} / {config.total_frames}")
            self._update_tracking_area_ui()
            return

        try:
            cap = cv2.VideoCapture(config.video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video")

            config.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            config.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            config.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            config.fps = cap.get(cv2.CAP_PROP_FPS)
            config.current_frame_idx = min(10, config.total_frames - 1)

            self.frame_slider.setRange(0, max(0, config.total_frames - 1))
            self.frame_slider.setValue(config.current_frame_idx)
            self.frame_slider.setEnabled(True)
            self.frame_label.setText(f"{config.current_frame_idx} / {config.total_frames}")

            # Smart defaults based on resolution
            if not config.configured:
                max_area = config.width * config.height
                config.min_object_area = max(50, int(max_area * 0.0001))
                config.max_object_area = int(max_area * 0.1)

            cap.set(cv2.CAP_PROP_POS_FRAMES, config.current_frame_idx)
            ret, frame = cap.read()
            cap.release()
            if not ret:
                raise ValueError("Could not read frame")

            if len(frame.shape) == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

            config.first_frame = frame_rgb
            self.display_frame(frame_rgb)
            self._update_tracking_area_ui()
            self.status_bar.showMessage(
                f"Loaded: {Path(config.video_path).name} ({config.width}x{config.height})"
            )

        except Exception as e:
            self.status_bar.showMessage(f"Error loading video: {str(e)}")
            QMessageBox.warning(self, "Load Error", f"Could not load video:\n{str(e)}")

    def _update_tracking_area_ui(self):
        """Update tracking area UI based on current config state."""
        if not self.video_configs:
            return
        config = self.video_configs[self.current_config_idx]

        if config.tracking_area_set:
            self.set_roi_btn.setEnabled(False)
            self.skip_roi_btn.setEnabled(False)
            self.clear_roi_btn.setEnabled(True)
            if config.tracking_area:
                self.roi_status_label.setText(
                    f"Tracking area defined ({len(config.tracking_area)} points)"
                )
            else:
                self.roi_status_label.setText("Tracking area skipped (full frame)")
            self.roi_status_label.setStyleSheet("color: #90EE90; font-size: 10px;")
            self._enable_detection_settings()
        else:
            self.set_roi_btn.setEnabled(True)
            self.skip_roi_btn.setEnabled(True)
            self.clear_roi_btn.setEnabled(False)
            self.roi_status_label.setText("Click Draw Area or Skip to continue")
            self.roi_status_label.setStyleSheet("color: #FFA500; font-size: 10px;")
            self._disable_detection_settings()

    # ── Tracking Area Drawing ─────────────────────────────────────────────

    def start_tracking_area_drawing(self):
        """Start tracking area polygon drawing."""
        if not self.video_configs:
            return
        config = self.video_configs[self.current_config_idx]
        if config.first_frame is None:
            return

        self.drawing_mode = 'tracking_area'
        self.polygon_points = []
        self.set_roi_btn.setEnabled(False)
        self.skip_roi_btn.setEnabled(False)
        self.clear_roi_btn.setEnabled(False)
        self.roi_status_label.setText("Drawing tracking area...")
        self.roi_status_label.setStyleSheet("color: #FFA500; font-size: 10px;")
        self._show_drawing_instructions(
            "Click to add polygon points  •  Enter = finish  •  Esc = cancel"
        )
        self.status_bar.showMessage("Click on preview to draw polygon. Press Enter when done.")

    def skip_tracking_area(self):
        """Skip tracking area - use full frame."""
        if not self.video_configs:
            return
        config = self.video_configs[self.current_config_idx]
        config.tracking_area = []
        config.tracking_area_set = True

        self.roi_status_label.setText("Tracking area skipped (full frame)")
        self.roi_status_label.setStyleSheet("color: #90EE90; font-size: 10px;")
        self.set_roi_btn.setEnabled(False)
        self.skip_roi_btn.setEnabled(False)
        self.clear_roi_btn.setEnabled(True)
        self.status_bar.showMessage("Tracking area skipped - adjust detection settings")

    def clear_tracking_area(self):
        """Clear tracking area and reset workflow."""
        if not self.video_configs:
            return
        config = self.video_configs[self.current_config_idx]
        config.tracking_area = []
        config.tracking_area_set = False

        self._update_tracking_area_ui()
        if config.first_frame is not None:
            self.display_frame(config.first_frame)
        self.status_bar.showMessage("Tracking area cleared")

    # ── Preview Click & Polygon Drawing ───────────────────────────────────

    def _label_to_frame_coords(self, label_x, label_y):
        """Convert label click coordinates to frame pixel coordinates."""
        if not self.video_configs:
            return 0, 0
        config = self.video_configs[self.current_config_idx]
        if config.first_frame is None:
            return 0, 0

        label_w = self.preview_label.width()
        label_h = self.preview_label.height()
        frame_h, frame_w = config.first_frame.shape[:2]

        scale_x = frame_w / label_w
        scale_y = frame_h / label_h
        scale = max(scale_x, scale_y)

        display_w = int(frame_w / scale)
        display_h = int(frame_h / scale)

        offset_x = (label_w - display_w) // 2
        offset_y = (label_h - display_h) // 2

        frame_x = int((label_x - offset_x) * scale)
        frame_y = int((label_y - offset_y) * scale)

        frame_x = max(0, min(frame_x, frame_w - 1))
        frame_y = max(0, min(frame_y, frame_h - 1))
        return frame_x, frame_y

    def on_preview_click(self, event):
        """Handle mouse clicks on the preview label."""
        if not self.drawing_mode:
            return
        if event.button() != Qt.LeftButton:
            return

        frame_x, frame_y = self._label_to_frame_coords(event.pos().x(), event.pos().y())

        if self.drawing_mode == 'scale_bar':
            self.scale_bar_points.append((frame_x, frame_y))
            self._redraw_preview()
            if len(self.scale_bar_points) == 2:
                self._finish_scale_bar()
            return

        if self.drawing_mode == 'oft':
            self.oft_corners.append((frame_x, frame_y))
            self._redraw_preview()
            if len(self.oft_corners) == 4:
                self._finish_oft()
            else:
                msg = f"OFT corner {len(self.oft_corners)}/4 placed. Click next corner.  •  Esc = cancel"
                self._show_drawing_instructions(msg)
                self.status_bar.showMessage(msg)
            return

        # General polygon drawing (tracking_area, ldb_light, custom)
        self.polygon_points.append((frame_x, frame_y))
        self._redraw_preview()
        self.status_bar.showMessage(
            f"Point {len(self.polygon_points)} at ({frame_x}, {frame_y}). Enter to finish."
        )

    def finish_polygon(self):
        """Finish current polygon drawing."""
        if not self.drawing_mode or len(self.polygon_points) < 3:
            if self.drawing_mode and len(self.polygon_points) < 3:
                QMessageBox.warning(self, "Invalid Polygon",
                                    "Need at least 3 points.")
            return

        config = self.video_configs[self.current_config_idx]
        self._hide_drawing_instructions()

        if self.drawing_mode == 'tracking_area':
            config.tracking_area = self.polygon_points.copy()
            config.tracking_area_set = True
            self.drawing_mode = None
            self.polygon_points = []
            self._update_tracking_area_ui()
            self._redraw_preview()
            self.status_bar.showMessage("Tracking area defined")

        elif self.drawing_mode == 'ldb_light':
            config.add_roi('ldb_light', self.polygon_points.copy())
            self.drawing_mode = None
            self.polygon_points = []
            self._update_roi_table()
            self._redraw_preview()
            # Auto-transition to scale bar
            self._prompt_scale_bar()

        elif self.drawing_mode == 'custom':
            # Ask for name
            name, ok = QInputDialog.getText(self, "ROI Name", "Enter ROI name:")
            if ok and name.strip():
                config.add_roi(name.strip(), self.polygon_points.copy())
            self.drawing_mode = None
            self.polygon_points = []
            self._update_roi_table()
            self._redraw_preview()

    def cancel_drawing(self):
        """Cancel current drawing mode."""
        self.drawing_mode = None
        self.polygon_points = []
        self.scale_bar_points = []
        self.oft_corners = []
        self._hide_drawing_instructions()
        self._update_tracking_area_ui()
        self._redraw_preview()
        self.status_bar.showMessage("Drawing cancelled")

    def _redraw_preview(self):
        """Redraw preview with all overlays (tracking area, ROIs, in-progress drawing)."""
        if not self.video_configs:
            return
        config = self.video_configs[self.current_config_idx]
        if config.first_frame is None:
            return

        display = config.first_frame.copy()

        # Draw tracking area
        if config.tracking_area and len(config.tracking_area) >= 3:
            pts = np.array(config.tracking_area, np.int32)
            cv2.polylines(display, [pts.reshape((-1, 1, 2))], True, (0, 255, 0), 2)

        # Draw ROIs
        for roi_name, roi_polygon in config.rois:
            if len(roi_polygon) < 3:
                continue
            pts = np.array(roi_polygon, np.int32)
            if 'center' in roi_name.lower():
                color = (0, 165, 255)
            elif 'light' in roi_name.lower():
                color = (0, 255, 0)
            else:
                color = (0, 255, 0)
            cv2.polylines(display, [pts.reshape((-1, 1, 2))], True, color, 2)
            cv2.putText(display, roi_name, tuple(roi_polygon[0]),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

        # Draw scale bar
        if config.scale_bar_set and config.scale_bar_pixels:
            # Just a visual indicator - no persistent overlay needed
            pass

        # Draw in-progress polygon
        if self.drawing_mode in ('tracking_area', 'ldb_light', 'custom'):
            if len(self.polygon_points) > 1:
                pts = np.array(self.polygon_points, np.int32)
                cv2.polylines(display, [pts.reshape((-1, 1, 2))], False, (0, 255, 0), 2)
            for pt in self.polygon_points:
                cv2.circle(display, pt, 8, (0, 255, 0), -1)
                cv2.circle(display, pt, 10, (255, 255, 255), 2)

        # Draw in-progress OFT corners
        if self.drawing_mode == 'oft':
            for i, pt in enumerate(self.oft_corners):
                cv2.circle(display, pt, 8, (0, 165, 255), -1)
                cv2.circle(display, pt, 10, (255, 255, 255), 2)
                cv2.putText(display, str(i + 1), (pt[0] + 12, pt[1]),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 165, 255), 2)
            if len(self.oft_corners) > 1:
                pts = np.array(self.oft_corners, np.int32)
                cv2.polylines(display, [pts.reshape((-1, 1, 2))], False, (0, 165, 255), 2)

        # Draw in-progress scale bar
        if self.drawing_mode == 'scale_bar' and self.scale_bar_points:
            for pt in self.scale_bar_points:
                cv2.circle(display, pt, 6, (255, 255, 0), -1)
            if len(self.scale_bar_points) == 2:
                cv2.line(display, self.scale_bar_points[0],
                        self.scale_bar_points[1], (255, 255, 0), 2)

        self.display_frame(display)

    # ── Frame Display & Navigation ────────────────────────────────────────

    def display_frame(self, frame: np.ndarray):
        """Display frame in preview label."""
        height, width = frame.shape[:2]
        channels = frame.shape[2] if len(frame.shape) == 3 else 1
        if channels == 1:
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            channels = 3
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation
        )
        self.preview_label.setPixmap(scaled_pixmap)

    def on_frame_slider_changed(self, frame_idx):
        """Handle frame slider movement."""
        if not self.video_configs:
            return
        config = self.video_configs[self.current_config_idx]
        config.current_frame_idx = frame_idx
        self.frame_label.setText(f"{frame_idx} / {config.total_frames}")
        self._load_and_show_frame(frame_idx)

    def _load_and_show_frame(self, frame_idx):
        """Load a specific frame and display it (with or without detection)."""
        if not self.video_configs:
            return
        config = self.video_configs[self.current_config_idx]

        try:
            cap = cv2.VideoCapture(config.video_path)
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            cap.release()

            if ret:
                if len(frame.shape) == 3:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

                frame_rgb = self.apply_contrast_brightness(
                    frame_rgb, config.contrast, config.brightness
                )
                config.first_frame = frame_rgb
                self._redraw_preview()
        except Exception as e:
            print(f"Error loading frame {frame_idx}: {e}")

    def apply_contrast_brightness(self, frame, contrast, brightness):
        """Apply contrast and brightness adjustments."""
        adjusted = frame.astype(np.float32)
        adjusted = contrast * (adjusted - 128) + 128
        adjusted = adjusted + brightness
        return np.clip(adjusted, 0, 255).astype(np.uint8)

    def update_preview_with_current_settings(self):
        """Update preview with detection overlay - debounced."""
        if not self.video_configs:
            return
        config = self.video_configs[self.current_config_idx]
        if config.first_frame is None:
            return

        if self.drawing_mode:
            self._redraw_preview()
            return

        if not config.tracking_area_set:
            self._redraw_preview()
            return

        try:
            self.status_bar.showMessage("Updating detection preview...")

            temp_tracker = BackgroundSubtractionTracker(
                video_path=config.video_path,
                min_object_area=config.min_object_area,
                max_object_area=config.max_object_area,
                history=config.history,
                var_threshold=config.var_threshold,
                num_animals=config.num_animals,
                morph_open_size=config.morph_open_size,
                morph_close_size=config.morph_close_size,
                morph_open_iterations=config.morph_open_iterations,
                morph_close_iterations=config.morph_close_iterations,
                gaussian_blur_size=config.gaussian_blur_size,
                fill_holes=config.fill_holes,
                convex_hull=config.convex_hull,
            )
            temp_tracker.initialize_video()

            # Build background model
            target = config.current_frame_idx
            start = max(0, target - 60)
            temp_tracker.cap.set(cv2.CAP_PROP_POS_FRAMES, start)
            for _ in range(min(60, target - start + 1)):
                ret, frame = temp_tracker.cap.read()
                if not ret:
                    break
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                temp_tracker.bg_subtractor.apply(gray)

            # Get target frame
            temp_tracker.cap.set(cv2.CAP_PROP_POS_FRAMES, target)
            ret, frame = temp_tracker.cap.read()

            if ret:
                if len(frame.shape) == 3:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

                frame_adjusted = self.apply_contrast_brightness(
                    frame_rgb, config.contrast, config.brightness
                )
                config.first_frame = frame_rgb

                tracking_area = config.tracking_area if config.tracking_area else None
                objects, mask = temp_tracker.process_frame(
                    frame_adjusted, target, tracking_area=tracking_area
                )

                # Render based on view mode
                if self.view_mode == 'mask':
                    display = temp_tracker.get_mask_view_frame(
                        frame_adjusted, mask, objects,
                        tracking_area=config.tracking_area,
                        rois=config.rois
                    )
                else:
                    display = temp_tracker.get_standard_view_frame(
                        frame_adjusted, mask, objects,
                        tracking_area=config.tracking_area,
                        rois=config.rois
                    )

                self.display_frame(display)

                # Update detection info
                info_text = f"<b>Objects Detected: {len(objects)}</b><br>"
                for obj_id, (cx, cy, area) in objects.items():
                    info_text += f"\u2022 ID {obj_id}: {int(area)}px\u00B2 at ({int(cx)}, {int(cy)})<br>"
                if not objects:
                    info_text += "<span style='color: #ff9900;'>No objects detected</span>"
                self.detection_info_label.setText(info_text)

            temp_tracker.cleanup()
            self.status_bar.showMessage("Preview updated")

        except Exception as e:
            self.status_bar.showMessage(f"Preview error: {str(e)}")
            print(f"Preview error: {e}")

def main():
    """Run the Simple Tracker GUI."""
    if not CV2_AVAILABLE or not PANDAS_AVAILABLE:
        print("Missing dependencies!")
        print("Install with: pip install opencv-python pandas numpy scipy")
        return

    app = QApplication(sys.argv)
    window = SimpleTrackerGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
