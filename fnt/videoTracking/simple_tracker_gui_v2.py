#!/usr/bin/env python3
"""
Simple Tracker GUI - Optimized for Static Camera Background Subtraction

Fast, CPU-only tracking using background subtraction for static camera setups.
Perfect for black/white videos with moving objects.

Optimized for:
- Black and white video
- Static camera position
- One or more moving objects
- Potential occlusions

Key features:
- Background subtraction (MOG2) - 200-500 fps on CPU
- Multi-object centroid tracking with Hungarian matching
- Batch processing with per-video configuration
- ROI definition ready (for future implementation)
- No manual clicking required - auto-detects all moving objects

Workflow:
1. Select multiple videos
2. For each video: Preview and configure settings (min/max size, ROI - future)
3. Start batch tracking - processes all videos sequentially
4. Export CSV with trajectories for each video

Author: FieldNeuroToolbox Contributors
"""

import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple, Dict
from collections import defaultdict
from scipy.optimize import linear_sum_assignment

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QMessageBox,
    QGroupBox, QSpinBox, QDoubleSpinBox, QListWidget, QListWidgetItem,
    QSlider, QScrollArea
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap

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


class BackgroundSubtractionTracker:
    """
    Ultra-fast multi-object tracker using background subtraction.
    Perfect for static camera with moving objects (200-500 fps on CPU).
    """
    
    def __init__(
        self,
        video_path: str,
        min_object_area: int = 100,
        max_object_area: int = 10000,
        history: int = 500,
        var_threshold: int = 16
    ):
        self.video_path = video_path
        self.min_object_area = min_object_area
        self.max_object_area = max_object_area
        self.history = history
        self.var_threshold = var_threshold
        
        # Background subtractor
        self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
            history=history,
            varThreshold=var_threshold,
            detectShadows=False  # Faster without shadow detection
        )
        
        # Video properties
        self.cap = None
        self.fps = None
        self.total_frames = None
        self.frame_width = None
        self.frame_height = None
        
        # Multi-object tracking state
        self.trajectories = defaultdict(list)  # object_id -> [(frame, x, y, area), ...]
        self.next_object_id = 0
        self.previous_centroids = {}  # object_id -> (x, y)
        self.max_distance_threshold = 50  # Max pixels between frames to consider same object
        self.max_frames_disappeared = 30  # Max frames before removing lost object
        self.disappeared = {}  # object_id -> frames_disappeared
    
    def initialize_video(self):
        """Open video and read properties."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Tuple[Dict[int, Tuple[float, float, float]], np.ndarray]:
        """
        Process single frame with background subtraction.
        
        Returns:
            - Dictionary: {object_id: (x, y, area)}
            - Binary mask for visualization
        """
        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(gray)
        
        # Morphological operations to clean up noise
        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel, iterations=1)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel, iterations=2)
        
        # Find connected components
        num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(fg_mask, connectivity=8)
        
        # Extract valid objects (filter by area)
        current_objects = []
        for i in range(1, num_labels):  # Skip background (label 0)
            area = stats[i, cv2.CC_STAT_AREA]
            if self.min_object_area <= area <= self.max_object_area:
                cx, cy = centroids[i]
                current_objects.append((cx, cy, area))
        
        # Match to previous frame and assign IDs
        matched_objects = self._match_and_update(current_objects, frame_idx)
        
        return matched_objects, fg_mask
    
    def _match_and_update(self, current_objects: List[Tuple[float, float, float]], frame_idx: int) -> Dict[int, Tuple[float, float, float]]:
        """
        Match current objects to previous frame using Hungarian algorithm or nearest neighbor.
        Assigns consistent IDs across frames.
        """
        matched = {}
        
        # If no previous objects, create new IDs for all
        if not self.previous_centroids:
            for cx, cy, area in current_objects:
                obj_id = self.next_object_id
                self.next_object_id += 1
                self.previous_centroids[obj_id] = (cx, cy)
                self.trajectories[obj_id].append((frame_idx, cx, cy, area))
                matched[obj_id] = (cx, cy, area)
            return matched
        
        # If no current objects, mark all as disappeared
        if not current_objects:
            for obj_id in list(self.previous_centroids.keys()):
                self.disappeared[obj_id] = self.disappeared.get(obj_id, 0) + 1
                if self.disappeared[obj_id] > self.max_frames_disappeared:
                    del self.previous_centroids[obj_id]
                    del self.disappeared[obj_id]
            return matched
        
        # Calculate distance matrix
        prev_ids = list(self.previous_centroids.keys())
        prev_centroids = np.array([self.previous_centroids[oid] for oid in prev_ids])
        curr_centroids = np.array([(cx, cy) for cx, cy, _ in current_objects])
        
        # Compute pairwise distances
        distances = np.linalg.norm(prev_centroids[:, np.newaxis] - curr_centroids[np.newaxis, :], axis=2)
        
        # Use Hungarian algorithm if scipy available, else nearest neighbor
        if SCIPY_AVAILABLE and len(prev_ids) > 0 and len(current_objects) > 0:
            row_indices, col_indices = linear_sum_assignment(distances)
            
            # Process matched pairs
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
            
            # Create new IDs for unmatched current objects
            for col, (cx, cy, area) in enumerate(current_objects):
                if col not in used_cols:
                    obj_id = self.next_object_id
                    self.next_object_id += 1
                    self.previous_centroids[obj_id] = (cx, cy)
                    self.trajectories[obj_id].append((frame_idx, cx, cy, area))
                    matched[obj_id] = (cx, cy, area)
            
            # Mark unmatched previous objects as disappeared
            matched_prev_ids = set(prev_ids[i] for i in row_indices if distances[i, col_indices[list(row_indices).index(i)]] < self.max_distance_threshold)
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
                
                # Find nearest current object
                min_dist = float('inf')
                best_idx = None
                for idx, (cx, cy, area) in enumerate(current_objects):
                    if idx in used_curr:
                        continue
                    dist = np.sqrt((cx - prev_x)**2 + (cy - prev_y)**2)
                    if dist < min_dist:
                        min_dist = dist
                        best_idx = idx
                
                # Match if close enough
                if best_idx is not None and min_dist < self.max_distance_threshold:
                    cx, cy, area = current_objects[best_idx]
                    self.previous_centroids[obj_id] = (cx, cy)
                    self.trajectories[obj_id].append((frame_idx, cx, cy, area))
                    self.disappeared[obj_id] = 0
                    matched[obj_id] = (cx, cy, area)
                    used_curr.add(best_idx)
                else:
                    # Mark as disappeared
                    self.disappeared[obj_id] = self.disappeared.get(obj_id, 0) + 1
                    if self.disappeared[obj_id] > self.max_frames_disappeared:
                        del self.previous_centroids[obj_id]
                        if obj_id in self.disappeared:
                            del self.disappeared[obj_id]
            
            # Create new IDs for unmatched current objects
            for idx, (cx, cy, area) in enumerate(current_objects):
                if idx not in used_curr:
                    obj_id = self.next_object_id
                    self.next_object_id += 1
                    self.previous_centroids[obj_id] = (cx, cy)
                    self.trajectories[obj_id].append((frame_idx, cx, cy, area))
                    matched[obj_id] = (cx, cy, area)
        
        return matched
    
    def export_trajectories(self, output_path: str) -> Dict[str, any]:
        """
        Export all trajectories to CSV.
        
        Returns summary statistics.
        """
        # Convert trajectories to DataFrame
        rows = []
        for obj_id, trajectory in self.trajectories.items():
            for frame, x, y, area in trajectory:
                rows.append({
                    'object_id': obj_id,
                    'frame': frame,
                    'timestamp': frame / self.fps,
                    'x': x,
                    'y': y,
                    'area': area
                })
        
        if not rows:
            raise ValueError("No trajectories to export")
        
        df = pd.DataFrame(rows)
        
        # Calculate distance traveled for each object
        distance_by_object = {}
        for obj_id in df['object_id'].unique():
            obj_data = df[df['object_id'] == obj_id].sort_values('frame')
            if len(obj_data) > 1:
                distances = np.sqrt(
                    np.diff(obj_data['x'].values)**2 + 
                    np.diff(obj_data['y'].values)**2
                )
                distance_by_object[obj_id] = np.sum(distances)
            else:
                distance_by_object[obj_id] = 0
        
        # Add distance column
        df['distance_traveled'] = df['object_id'].map(distance_by_object)
        
        # Save to CSV
        df.to_csv(output_path, index=False)
        
        # Calculate summary statistics
        stats = {
            'num_objects': len(self.trajectories),
            'total_frames': self.total_frames,
            'objects_detected': list(self.trajectories.keys()),
            'total_distance': sum(distance_by_object.values()),
            'avg_distance': np.mean(list(distance_by_object.values())) if distance_by_object else 0
        }
        
        return stats
    
    def cleanup(self):
        """Release video capture."""
        if self.cap:
            self.cap.release()


class VideoConfig:
    """Configuration for a single video in batch processing."""
    
    def __init__(self, video_path: str):
        self.video_path = video_path
        self.min_object_area = 100
        self.max_object_area = 10000
        self.history = 500
        self.var_threshold = 16
        self.contrast = 1.0  # Contrast multiplier (0.5-3.0, 1.0 = normal)
        self.brightness = 0  # Brightness offset (-100 to 100, 0 = normal)
        self.roi_rectangles = []  # List of (x, y) polygon points
        self.tracking_area_set = False  # True if user drew area or clicked skip
        self.configured = False
        self.width = None
        self.height = None
        self.first_frame = None
        self.current_frame_idx = 0  # Current frame being previewed
        self.total_frames = 0  # Total frames in video


class BatchTrackingWorker(QThread):
    """Worker thread for batch processing multiple videos."""
    
    progress = pyqtSignal(int, int, int)  # video_idx, frame_idx, total_frames
    status = pyqtSignal(str)
    video_finished = pyqtSignal(int, bool, str, dict)  # video_idx, success, message, stats
    all_finished = pyqtSignal(bool, str)  # success, message
    frame_ready = pyqtSignal(np.ndarray, dict, np.ndarray)  # frame, objects, mask
    
    def __init__(self, video_configs: List[VideoConfig]):
        super().__init__()
        self.video_configs = video_configs
        self.cancelled = False
    
    def run(self):
        """Process all videos in batch."""
        total_videos = len(self.video_configs)
        successful = 0
        failed = 0
        
        for video_idx, config in enumerate(self.video_configs):
            if self.cancelled:
                self.all_finished.emit(False, "Batch tracking cancelled")
                return
            
            try:
                self.status.emit(f"Processing video {video_idx + 1}/{total_videos}: {Path(config.video_path).name}")
                
                # Create tracker
                tracker = BackgroundSubtractionTracker(
                    video_path=config.video_path,
                    min_object_area=config.min_object_area,
                    max_object_area=config.max_object_area,
                    history=config.history,
                    var_threshold=config.var_threshold
                )
                
                # Initialize video
                tracker.initialize_video()
                
                # Process all frames
                frame_idx = 0
                while not self.cancelled:
                    ret, frame = tracker.cap.read()
                    if not ret:
                        break
                    
                    # Convert to RGB for display
                    if len(frame.shape) == 3:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                    
                    # Process frame
                    objects, mask = tracker.process_frame(frame_rgb, frame_idx)
                    
                    # Emit preview every 30 frames
                    if frame_idx % 30 == 0:
                        self.frame_ready.emit(frame_rgb, objects, mask)
                    
                    # Update progress
                    self.progress.emit(video_idx, frame_idx, tracker.total_frames)
                    
                    frame_idx += 1
                
                if self.cancelled:
                    tracker.cleanup()
                    self.all_finished.emit(False, "Batch tracking cancelled")
                    return
                
                # Export trajectories
                output_path = str(Path(config.video_path).with_suffix('.csv'))
                stats = tracker.export_trajectories(output_path)
                
                tracker.cleanup()
                
                self.video_finished.emit(video_idx, True, f"Completed: {Path(config.video_path).name}", stats)
                successful += 1
                
            except Exception as e:
                self.video_finished.emit(video_idx, False, f"Failed: {str(e)}", {})
                failed += 1
        
        # All videos processed
        summary = f"Batch complete: {successful} successful, {failed} failed"
        self.all_finished.emit(failed == 0, summary)
    
    def cancel(self):
        """Cancel batch processing."""
        self.cancelled = True


class SimpleTrackerGUI(QMainWindow):
    """
    Simplified tracker GUI optimized for batch processing with background subtraction.
    """
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple Tracker - FieldNeuroToolbox")
        self.setGeometry(100, 100, 1400, 900)
        
        # State
        self.video_configs = []  # List of VideoConfig objects
        self.current_config_idx = 0
        self.preview_frame = None
        self.preview_cap = None
        self.preview_tracker = None  # Temporary tracker for live preview
        self.auto_preview_enabled = False  # Flag to enable auto-preview after first load
        
        # Polygon drawing state
        self.drawing_polygon = False
        self.polygon_points = []
        
        # Worker thread
        self.tracking_worker = None
        
        # Timer for debounced preview updates
        self.preview_update_timer = QTimer()
        self.preview_update_timer.setSingleShot(True)
        self.preview_update_timer.timeout.connect(self.update_preview_with_current_settings)
        
        # Apply dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #cccccc;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #cccccc;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3f3f3f;
                border-radius: 4px;
                margin-top: 10px;
                padding-top: 8px;
                color: #cccccc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #3f3f3f;
                color: #888888;
            }
            QLabel {
                color: #cccccc;
                background-color: transparent;
                font-size: 12px;
            }
            QSpinBox, QDoubleSpinBox {
                background-color: #3f3f3f;
                color: #cccccc;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 4px;
            }
            QListWidget {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #3f3f3f;
                border-radius: 3px;
            }
            QListWidget::item {
                padding: 5px;
                border-bottom: 1px solid #3f3f3f;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
            }
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 3px;
                background-color: #3f3f3f;
                text-align: center;
                color: #cccccc;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
            }
            QStatusBar {
                background-color: #1e1e1e;
                color: #cccccc;
                border-top: 1px solid #3f3f3f;
            }
        """)
        
        self.init_ui()
    
    def keyPressEvent(self, event):
        """Handle key press events."""
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            if self.drawing_polygon and len(self.polygon_points) >= 3:
                self.finish_polygon()
        elif event.key() == Qt.Key_Escape:
            if self.drawing_polygon:
                self.cancel_polygon_drawing()
    
    def init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - controls
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, stretch=1)
        
        # Right panel - preview
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, stretch=2)
        
        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready - Select videos to begin")
    
    def create_left_panel(self):
        """Create left control panel with scroll area."""
        # Create scroll area
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        
        # Create the actual panel content
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 1. Video Selection
        video_group = QGroupBox("1. Video Selection")
        video_layout = QVBoxLayout()
        
        btn_layout = QHBoxLayout()
        self.select_video_btn = QPushButton("Select Videos")
        self.select_video_btn.clicked.connect(self.select_videos)
        btn_layout.addWidget(self.select_video_btn)
        
        self.clear_videos_btn = QPushButton("Clear")
        self.clear_videos_btn.clicked.connect(self.clear_videos)
        self.clear_videos_btn.setStyleSheet("""
            QPushButton {
                background-color: #d47800;
            }
            QPushButton:hover {
                background-color: #e68a00;
            }
        """)
        btn_layout.addWidget(self.clear_videos_btn)
        video_layout.addLayout(btn_layout)
        
        self.video_list = QListWidget()
        self.video_list.currentRowChanged.connect(self.on_video_selected)
        video_layout.addWidget(self.video_list)
        
        video_group.setLayout(video_layout)
        layout.addWidget(video_group)
        
        # 2. Set Tracking Area
        roi_group = QGroupBox("2. Set Tracking Area")
        roi_layout = QVBoxLayout()
        
        roi_info = QLabel("Define the area to track objects (optional)")
        roi_info.setWordWrap(True)
        roi_info.setStyleSheet("color: #aaaaaa; font-size: 10px;")
        roi_layout.addWidget(roi_info)
        
        roi_btn_layout = QHBoxLayout()
        self.set_roi_btn = QPushButton("Draw Area")
        self.set_roi_btn.clicked.connect(self.start_polygon_drawing)
        self.set_roi_btn.setEnabled(False)
        self.set_roi_btn.setStyleSheet("""
            QPushButton {
                background-color: #00aa00;
            }
            QPushButton:hover {
                background-color: #00cc00;
            }
            QPushButton:disabled {
                background-color: #3f3f3f;
                color: #888888;
            }
        """)
        roi_btn_layout.addWidget(self.set_roi_btn)
        
        self.skip_roi_btn = QPushButton("Skip")
        self.skip_roi_btn.clicked.connect(self.skip_tracking_area)
        self.skip_roi_btn.setEnabled(False)
        self.skip_roi_btn.setStyleSheet("""
            QPushButton {
                background-color: #666666;
            }
            QPushButton:hover {
                background-color: #777777;
            }
            QPushButton:disabled {
                background-color: #3f3f3f;
                color: #888888;
            }
        """)
        roi_btn_layout.addWidget(self.skip_roi_btn)
        
        self.clear_roi_btn = QPushButton("Clear")
        self.clear_roi_btn.clicked.connect(self.clear_polygon)
        self.clear_roi_btn.setEnabled(False)
        self.clear_roi_btn.setStyleSheet("""
            QPushButton {
                background-color: #d47800;
            }
            QPushButton:hover {
                background-color: #e68a00;
            }
            QPushButton:disabled {
                background-color: #3f3f3f;
                color: #888888;
            }
        """)
        roi_btn_layout.addWidget(self.clear_roi_btn)
        roi_layout.addLayout(roi_btn_layout)
        
        self.roi_status_label = QLabel("Click Draw Area or Skip to continue")
        self.roi_status_label.setStyleSheet("color: #FFA500; font-size: 10px;")
        roi_layout.addWidget(self.roi_status_label)
        
        roi_group.setLayout(roi_layout)
        layout.addWidget(roi_group)
        
        # 3. Object Detection Settings
        settings_group = QGroupBox("3. Object Detection Settings")
        settings_layout = QVBoxLayout()
        
        # Min object area
        self.min_area_label = QLabel("Min Object Size: 100 px²")
        settings_layout.addWidget(self.min_area_label)
        
        # Min spinbox and slider
        min_area_widget_layout = QHBoxLayout()
        self.min_area_spinbox = QSpinBox()
        self.min_area_spinbox.setRange(10, 50000)
        self.min_area_spinbox.setValue(100)
        self.min_area_spinbox.setSuffix(" px²")
        self.min_area_spinbox.valueChanged.connect(self.on_min_spinbox_changed)
        self.min_area_spinbox.setEnabled(False)
        min_area_widget_layout.addWidget(self.min_area_spinbox)
        settings_layout.addLayout(min_area_widget_layout)
        
        self.min_area_slider = QSlider(Qt.Horizontal)
        self.min_area_slider.setRange(10, 50000)
        self.min_area_slider.setValue(100)
        self.min_area_slider.setTickPosition(QSlider.TicksBelow)
        self.min_area_slider.setTickInterval(5000)
        self.min_area_slider.valueChanged.connect(self.on_min_slider_changed)
        self.min_area_slider.sliderReleased.connect(self.on_slider_released)
        self.min_area_slider.setEnabled(False)
        settings_layout.addWidget(self.min_area_slider)
        
        # Max object area
        self.max_area_label = QLabel("Max Object Size: 10000 px²")
        settings_layout.addWidget(self.max_area_label)
        
        # Max spinbox and slider
        max_area_widget_layout = QHBoxLayout()
        self.max_area_spinbox = QSpinBox()
        self.max_area_spinbox.setRange(10, 50000)
        self.max_area_spinbox.setValue(10000)
        self.max_area_spinbox.setSuffix(" px²")
        self.max_area_spinbox.valueChanged.connect(self.on_max_spinbox_changed)
        self.max_area_spinbox.setEnabled(False)
        max_area_widget_layout.addWidget(self.max_area_spinbox)
        settings_layout.addLayout(max_area_widget_layout)
        
        self.max_area_slider = QSlider(Qt.Horizontal)
        self.max_area_slider.setRange(10, 50000)
        self.max_area_slider.setValue(10000)
        self.max_area_slider.setTickPosition(QSlider.TicksBelow)
        self.max_area_slider.setTickInterval(5000)
        self.max_area_slider.valueChanged.connect(self.on_max_slider_changed)
        self.max_area_slider.sliderReleased.connect(self.on_slider_released)
        self.max_area_slider.setEnabled(False)
        settings_layout.addWidget(self.max_area_slider)
        
        # Sensitivity
        self.sensitivity_label = QLabel("Sensitivity: 16")
        settings_layout.addWidget(self.sensitivity_label)
        
        self.sensitivity_slider = QSlider(Qt.Horizontal)
        self.sensitivity_slider.setRange(1, 100)
        self.sensitivity_slider.setValue(16)
        self.sensitivity_slider.setTickPosition(QSlider.TicksBelow)
        self.sensitivity_slider.setTickInterval(10)
        self.sensitivity_slider.setToolTip("Lower = more sensitive (detects more). Higher = less sensitive.")
        self.sensitivity_slider.valueChanged.connect(self.on_sensitivity_slider_changed)
        self.sensitivity_slider.sliderReleased.connect(self.on_slider_released)
        self.sensitivity_slider.setEnabled(False)
        settings_layout.addWidget(self.sensitivity_slider)
        
        # Contrast
        self.contrast_label = QLabel("Contrast: 1.0x")
        settings_layout.addWidget(self.contrast_label)
        
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(50, 300)  # 0.5 to 3.0 (scaled by 100)
        self.contrast_slider.setValue(100)  # 1.0x default
        self.contrast_slider.setTickPosition(QSlider.TicksBelow)
        self.contrast_slider.setTickInterval(50)
        self.contrast_slider.setToolTip("Adjust contrast to make objects more distinct from background")
        self.contrast_slider.valueChanged.connect(self.on_contrast_slider_changed)
        self.contrast_slider.sliderReleased.connect(self.on_slider_released)
        self.contrast_slider.setEnabled(False)
        settings_layout.addWidget(self.contrast_slider)
        
        # Brightness
        self.brightness_label = QLabel("Brightness: 0")
        settings_layout.addWidget(self.brightness_label)
        
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(-100, 100)
        self.brightness_slider.setValue(0)
        self.brightness_slider.setTickPosition(QSlider.TicksBelow)
        self.brightness_slider.setTickInterval(25)
        self.brightness_slider.setToolTip("Adjust brightness to enhance visibility")
        self.brightness_slider.valueChanged.connect(self.on_brightness_slider_changed)
        self.brightness_slider.sliderReleased.connect(self.on_slider_released)
        self.brightness_slider.setEnabled(False)
        settings_layout.addWidget(self.brightness_slider)
        
        # Detect Objects button
        self.detect_objects_btn = QPushButton("Detect Objects")
        self.detect_objects_btn.clicked.connect(self.detect_objects_preview)
        self.detect_objects_btn.setEnabled(False)
        self.detect_objects_btn.setStyleSheet("""
            QPushButton {
                background-color: #0078d4;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #0086f0;
            }
            QPushButton:disabled {
                background-color: #3f3f3f;
                color: #888888;
            }
        """)
        settings_layout.addWidget(self.detect_objects_btn)
        
        info_label = QLabel(
            "<b>Settings:</b><br>"
            "• <b>Min/Max Size:</b> Filter objects by area<br>"
            "• <b>Sensitivity:</b> Background subtraction threshold<br>"
            "• Changes apply immediately to preview"
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #aaaaaa; font-size: 10px; padding: 5px;")
        settings_layout.addWidget(info_label)
        
        settings_group.setLayout(settings_layout)
        layout.addWidget(settings_group)
        
        # 4. Batch Processing
        batch_group = QGroupBox("4. Batch Processing")
        batch_layout = QVBoxLayout()
        
        self.start_batch_btn = QPushButton("Start Batch Tracking")
        self.start_batch_btn.clicked.connect(self.start_batch_tracking)
        self.start_batch_btn.setEnabled(False)
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
        
        batch_group.setLayout(batch_layout)
        layout.addWidget(batch_group)
        
        layout.addStretch()
        
        # Set the panel as the scroll area's widget
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
        
        # Video info label (replaces old preview section)
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
    
    def select_videos(self):
        """Select video files for batch processing."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video Files",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        )
        
        if file_paths:
            self.video_configs = [VideoConfig(path) for path in file_paths]
            self.current_config_idx = 0
            
            # Update video list
            self.video_list.clear()
            for idx, config in enumerate(self.video_configs):
                item_text = f"{idx + 1}. {Path(config.video_path).name}"
                item = QListWidgetItem(item_text)
                self.video_list.addItem(item)
            
            self.video_list.setCurrentRow(0)
            
            # Enable controls
            self.start_batch_btn.setEnabled(True)
            
            # Auto-load first frame of first video
            self.auto_preview_enabled = True
            self.load_first_frame_for_current_video()
            
            # Update status
            self.update_config_status()
            self.status_bar.showMessage(f"Loaded {len(file_paths)} video(s) - Adjust settings using sliders")
    
    def clear_videos(self):
        """Clear all selected videos."""
        self.video_configs = []
        self.video_list.clear()
        self.current_config_idx = 0
        self.preview_label.setText("Select videos to begin")
        self.detection_info_label.setText("")
        self.video_info_label.setText("No videos selected")
        self.start_batch_btn.setEnabled(False)
        self.frame_slider.setEnabled(False)
        self.frame_slider.setRange(0, 0)
        self.frame_label.setText("0 / 0")
        self.status_bar.showMessage("Ready - Select videos to begin")
    
    def on_video_selected(self, row):
        """Handle video selection from list."""
        if row >= 0 and row < len(self.video_configs):
            self.current_config_idx = row
            self.load_current_config()
            self.update_config_status()
            
            # Auto-load first frame if not already loaded
            if self.auto_preview_enabled:
                self.load_first_frame_for_current_video()
    
    def load_current_config(self):
        """Load current video configuration into UI."""
        if not self.video_configs:
            return
        
        config = self.video_configs[self.current_config_idx]
        
        # Update slider ranges based on video resolution
        if config.width and config.height:
            max_possible_area = config.width * config.height
            self.min_area_slider.setRange(10, max_possible_area)
            self.max_area_slider.setRange(10, max_possible_area)
        
        # Block signals while updating to prevent triggering events
        self.min_area_slider.blockSignals(True)
        self.max_area_slider.blockSignals(True)
        self.sensitivity_slider.blockSignals(True)
        self.contrast_slider.blockSignals(True)
        self.brightness_slider.blockSignals(True)
        self.min_area_spinbox.blockSignals(True)
        self.max_area_spinbox.blockSignals(True)
        
        self.min_area_slider.setValue(config.min_object_area)
        self.max_area_slider.setValue(config.max_object_area)
        self.sensitivity_slider.setValue(config.var_threshold)
        self.contrast_slider.setValue(int(config.contrast * 100))
        self.brightness_slider.setValue(config.brightness)
        self.min_area_spinbox.setValue(config.min_object_area)
        self.max_area_spinbox.setValue(config.max_object_area)
        
        self.min_area_slider.blockSignals(False)
        self.max_area_slider.blockSignals(False)
        self.sensitivity_slider.blockSignals(False)
        self.contrast_slider.blockSignals(False)
        self.brightness_slider.blockSignals(False)
        self.min_area_spinbox.blockSignals(False)
        self.max_area_spinbox.blockSignals(False)
        
        # Update labels
        self.min_area_label.setText(f"Min Object Size: {config.min_object_area} px²")
        self.max_area_label.setText(f"Max Object Size: {config.max_object_area} px²")
        self.sensitivity_label.setText(f"Sensitivity: {config.var_threshold}")
        self.contrast_label.setText(f"Contrast: {config.contrast:.1f}x")
        self.brightness_label.setText(f"Brightness: {config.brightness:+d}")
    
    def on_min_slider_changed(self, value):
        """Handle min slider value change - update label, spinbox, and enforce constraints."""
        # Ensure min doesn't exceed max
        if value > self.max_area_slider.value():
            self.min_area_slider.blockSignals(True)
            self.min_area_slider.setValue(self.max_area_slider.value())
            self.min_area_slider.blockSignals(False)
            value = self.max_area_slider.value()
        
        self.min_area_label.setText(f"Min Object Size: {value} px²")
        
        # Update spinbox without triggering its signal
        self.min_area_spinbox.blockSignals(True)
        self.min_area_spinbox.setValue(value)
        self.min_area_spinbox.blockSignals(False)
    
    def on_max_slider_changed(self, value):
        """Handle max slider value change - update label, spinbox, and enforce constraints."""
        # Ensure max doesn't go below min
        if value < self.min_area_slider.value():
            self.max_area_slider.blockSignals(True)
            self.max_area_slider.setValue(self.min_area_slider.value())
            self.max_area_slider.blockSignals(False)
            value = self.min_area_slider.value()
        
        self.max_area_label.setText(f"Max Object Size: {value} px²")
        
        # Update spinbox without triggering its signal
        self.max_area_spinbox.blockSignals(True)
        self.max_area_spinbox.setValue(value)
        self.max_area_spinbox.blockSignals(False)
    
    def on_sensitivity_slider_changed(self, value):
        """Handle sensitivity slider value change - update label."""
        self.sensitivity_label.setText(f"Sensitivity: {value}")
    
    def on_contrast_slider_changed(self, value):
        """Handle contrast slider value change - update label."""
        contrast_value = value / 100.0
        self.contrast_label.setText(f"Contrast: {contrast_value:.1f}x")
    
    def on_brightness_slider_changed(self, value):
        """Handle brightness slider value change - update label."""
        self.brightness_label.setText(f"Brightness: {value:+d}")
    
    def on_min_spinbox_changed(self, value):
        """Handle min spinbox value change - update slider."""
        # Block slider signals to avoid recursion
        self.min_area_slider.blockSignals(True)
        self.min_area_slider.setValue(value)
        self.min_area_slider.blockSignals(False)
        
        # Update label and enforce constraints
        if value > self.max_area_spinbox.value():
            self.min_area_spinbox.blockSignals(True)
            self.min_area_spinbox.setValue(self.max_area_spinbox.value())
            self.min_area_spinbox.blockSignals(False)
            value = self.max_area_spinbox.value()
            self.min_area_slider.setValue(value)
        
        self.min_area_label.setText(f"Min Object Size: {value} px²")
        self.update_current_config()
        
        # Immediately update preview
        if self.video_configs:
            config = self.video_configs[self.current_config_idx]
            if config.first_frame is not None and config.tracking_area_set:
                self.load_and_preview_frame(config.current_frame_idx)
    
    def on_max_spinbox_changed(self, value):
        """Handle max spinbox value change - update slider."""
        # Block slider signals to avoid recursion
        self.max_area_slider.blockSignals(True)
        self.max_area_slider.setValue(value)
        self.max_area_slider.blockSignals(False)
        
        # Update label and enforce constraints
        if value < self.min_area_spinbox.value():
            self.max_area_spinbox.blockSignals(True)
            self.max_area_spinbox.setValue(self.min_area_spinbox.value())
            self.max_area_spinbox.blockSignals(False)
            value = self.min_area_spinbox.value()
            self.max_area_slider.setValue(value)
        
        self.max_area_label.setText(f"Max Object Size: {value} px²")
        self.update_current_config()
        
        # Immediately update preview
        if self.video_configs:
            config = self.video_configs[self.current_config_idx]
            if config.first_frame is not None and config.tracking_area_set:
                self.load_and_preview_frame(config.current_frame_idx)
    
    def on_slider_released(self):
        """When slider is released, update config and immediately refresh preview."""
        self.update_current_config()
        
        # Immediately update preview
        if self.video_configs:
            config = self.video_configs[self.current_config_idx]
            if config.first_frame is not None and config.tracking_area_set:
                self.load_and_preview_frame(config.current_frame_idx)
    
    def update_current_config(self):
        """Update current video configuration from UI."""
        if not self.video_configs:
            return
        
        config = self.video_configs[self.current_config_idx]
        config.min_object_area = self.min_area_slider.value()
        config.max_object_area = self.max_area_slider.value()
        config.var_threshold = self.sensitivity_slider.value()
        config.contrast = self.contrast_slider.value() / 100.0
        config.brightness = self.brightness_slider.value()
        config.configured = True
        
        self.update_config_status()
    
    def update_config_status(self):
        """Update configuration status label in preview panel."""
        if not self.video_configs:
            self.video_info_label.setText("No videos selected")
            return
        
        configured_count = sum(1 for c in self.video_configs if c.configured)
        total = len(self.video_configs)
        
        config = self.video_configs[self.current_config_idx]
        status_text = (
            f"<b>Video {self.current_config_idx + 1} of {total}</b> - {Path(config.video_path).name}<br>"
            f"<b>Configured:</b> {configured_count}/{total} videos | "
            f"<b>Status:</b> {'✓ Configured' if config.configured else '⚠ Not configured'}"
        )
        self.video_info_label.setText(status_text)
        
        self.preview_title.setText(
            f"<h2>Preview - Video {self.current_config_idx + 1}/{total}</h2>"
        )
    
    def previous_video(self):
        """Navigate to previous video."""
        if self.current_config_idx > 0:
            self.current_config_idx -= 1
            self.video_list.setCurrentRow(self.current_config_idx)
    
    def next_video(self):
        """Navigate to next video."""
        if self.current_config_idx < len(self.video_configs) - 1:
            self.current_config_idx += 1
            self.video_list.setCurrentRow(self.current_config_idx)
    
    def load_first_frame_for_current_video(self):
        """Load first frame and video properties for current video."""
        if not self.video_configs:
            return
        
        config = self.video_configs[self.current_config_idx]
        
        # If already loaded, just display it
        if config.first_frame is not None:
            self.display_frame(config.first_frame)
            # Update frame slider
            self.frame_slider.setRange(0, max(0, config.total_frames - 1))
            self.frame_slider.setValue(config.current_frame_idx)
            self.frame_label.setText(f"{config.current_frame_idx} / {config.total_frames}")
            return
        
        try:
            # Open video temporarily
            cap = cv2.VideoCapture(config.video_path)
            if not cap.isOpened():
                raise ValueError("Could not open video")
            
            # Get video properties
            config.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            config.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            config.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            config.current_frame_idx = 10  # Start at frame 10 to skip black frames
            
            # Setup frame slider
            self.frame_slider.setRange(0, max(0, config.total_frames - 1))
            self.frame_slider.setValue(10)
            self.frame_slider.setEnabled(True)
            self.frame_label.setText(f"10 / {config.total_frames}")
            
            # Update slider and spinbox ranges based on resolution
            max_possible_area = config.width * config.height
            self.min_area_slider.setRange(10, max_possible_area)
            self.max_area_slider.setRange(10, max_possible_area)
            self.min_area_spinbox.setRange(10, max_possible_area)
            self.max_area_spinbox.setRange(10, max_possible_area)
            
            # Set smart defaults based on resolution
            default_min = max(50, int(max_possible_area * 0.0001))  # 0.01% of frame
            default_max = int(max_possible_area * 0.1)  # 10% of frame
            
            if not config.configured:
                config.min_object_area = default_min
                config.max_object_area = default_max
                self.min_area_slider.setValue(default_min)
                self.max_area_slider.setValue(default_max)
                self.min_area_spinbox.setValue(default_min)
                self.max_area_spinbox.setValue(default_max)
                self.min_area_label.setText(f"Min Object Size: {default_min} px²")
                self.max_area_label.setText(f"Max Object Size: {default_max} px²")
            
            # Read frame at current_frame_idx
            cap.set(cv2.CAP_PROP_POS_FRAMES, config.current_frame_idx)
            ret, frame = cap.read()
            if not ret:
                cap.release()
                raise ValueError("Could not read frame")
            
            # Convert to RGB
            if len(frame.shape) == 3:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            else:
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
            
            config.first_frame = frame_rgb
            cap.release()
            
            # Display frame (no detection yet - user must click Detect Objects)
            self.display_frame(frame_rgb)
            
            # Set button states based on tracking_area_set flag
            if config.tracking_area_set:
                # Tracking area already set/skipped for this video
                self.set_roi_btn.setEnabled(False)
                self.skip_roi_btn.setEnabled(False)
                self.clear_roi_btn.setEnabled(True)
                self.enable_detection_settings()
                
                if config.roi_rectangles:
                    self.roi_status_label.setText(f"Tracking area defined ({len(config.roi_rectangles)} points)")
                else:
                    self.roi_status_label.setText("Tracking area skipped (full frame)")
                self.roi_status_label.setStyleSheet("color: #90EE90;")
            else:
                # Need to set tracking area
                self.set_roi_btn.setEnabled(True)
                self.skip_roi_btn.setEnabled(True)
                self.clear_roi_btn.setEnabled(False)
                
                # Lock detection settings until tracking area is set
                self.min_area_slider.setEnabled(False)
                self.max_area_slider.setEnabled(False)
                self.min_area_spinbox.setEnabled(False)
                self.max_area_spinbox.setEnabled(False)
                self.sensitivity_slider.setEnabled(False)
                self.contrast_slider.setEnabled(False)
                self.brightness_slider.setEnabled(False)
                self.detect_objects_btn.setEnabled(False)
                
                self.roi_status_label.setText("Click Draw Area or Skip to continue")
                self.roi_status_label.setStyleSheet("color: #FFA500;")
            
            self.status_bar.showMessage(f"Loaded: {Path(config.video_path).name} ({config.width}x{config.height})")
            
        except Exception as e:
            self.status_bar.showMessage(f"Error loading video: {str(e)}")
            QMessageBox.warning(self, "Load Error", f"Could not load video:\n{str(e)}")
    
    def start_polygon_drawing(self):
        """Start polygon drawing mode."""
        if not self.video_configs or self.current_config_idx < 0:
            QMessageBox.warning(self, "No Video", "Please select a video first.")
            return
        
        config = self.video_configs[self.current_config_idx]
        if config.first_frame is None:
            QMessageBox.warning(self, "No Frame", "Please wait for the frame to load.")
            return
        
        self.drawing_polygon = True
        self.polygon_points = []
        
        # Update UI state
        self.set_roi_btn.setEnabled(False)
        self.skip_roi_btn.setEnabled(False)
        self.clear_roi_btn.setEnabled(False)
        self.roi_status_label.setText("Drawing... (Click to add points, Enter to finish, Esc to cancel)")
        self.roi_status_label.setStyleSheet("color: #FFA500;")  # Orange while drawing
        
        self.status_bar.showMessage("Click on the preview to draw polygon points. Press Enter when done.")
    
    def on_preview_click(self, event):
        """Handle mouse clicks on the preview label."""
        if not self.drawing_polygon:
            return
        
        if event.button() != Qt.LeftButton:
            return
        
        config = self.video_configs[self.current_config_idx]
        if config.first_frame is None:
            return
        
        # Get click position on the label
        label_x = event.pos().x()
        label_y = event.pos().y()
        
        # Get label and frame dimensions
        label_width = self.preview_label.width()
        label_height = self.preview_label.height()
        frame_height, frame_width = config.first_frame.shape[:2]
        
        # Calculate scaling factor (the frame is scaled to fit the label while maintaining aspect ratio)
        scale_x = frame_width / label_width
        scale_y = frame_height / label_height
        scale = max(scale_x, scale_y)
        
        # Calculate the actual displayed size
        display_width = int(frame_width / scale)
        display_height = int(frame_height / scale)
        
        # Calculate offset (centering)
        offset_x = (label_width - display_width) // 2
        offset_y = (label_height - display_height) // 2
        
        # Convert label coordinates to frame coordinates
        frame_x = int((label_x - offset_x) * scale)
        frame_y = int((label_y - offset_y) * scale)
        
        # Clamp to frame boundaries
        frame_x = max(0, min(frame_x, frame_width - 1))
        frame_y = max(0, min(frame_y, frame_height - 1))
        
        # Add point
        self.polygon_points.append((frame_x, frame_y))
        
        # Update preview to show the polygon being drawn
        self.update_preview_with_current_settings()
        
        self.status_bar.showMessage(f"Point {len(self.polygon_points)} added at ({frame_x}, {frame_y}). Press Enter to finish.")
    
    def finish_polygon(self):
        """Finish drawing the polygon and save it."""
        if not self.drawing_polygon:
            return
        
        if len(self.polygon_points) < 3:
            QMessageBox.warning(
                self,
                "Invalid Polygon",
                "A polygon must have at least 3 points. Please add more points or press Esc to cancel."
            )
            return
        
        # Save polygon to current video config
        config = self.video_configs[self.current_config_idx]
        config.roi_rectangles = self.polygon_points.copy()
        config.tracking_area_set = True
        
        # Reset drawing state
        self.drawing_polygon = False
        self.polygon_points = []
        
        # Update UI state
        self.set_roi_btn.setEnabled(False)
        self.skip_roi_btn.setEnabled(False)
        self.clear_roi_btn.setEnabled(True)
        self.roi_status_label.setText(f"Tracking area defined ({len(config.roi_rectangles)} points)")
        self.roi_status_label.setStyleSheet("color: #90EE90;")  # Light green when set
        
        # Enable object detection settings
        self.enable_detection_settings()
        
        # Update preview with final polygon (no detection overlay yet)
        if config.first_frame is not None:
            frame_display = config.first_frame.copy()
            
            # Draw the completed polygon
            if len(config.roi_rectangles) > 0:
                pts = np.array(config.roi_rectangles, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame_display, [pts], True, (0, 255, 0), 2)
            
            self.display_frame(frame_display)
        
        self.status_bar.showMessage("Tracking area defined - Click 'Detect Objects' to preview detection")
    
    def clear_polygon(self):
        """Clear the tracking area polygon and reset workflow."""
        if not self.video_configs or self.current_config_idx < 0:
            return
        
        config = self.video_configs[self.current_config_idx]
        config.roi_rectangles = []
        config.tracking_area_set = False
        
        # Reset workflow - lock detection settings
        self.min_area_slider.setEnabled(False)
        self.max_area_slider.setEnabled(False)
        self.min_area_spinbox.setEnabled(False)
        self.max_area_spinbox.setEnabled(False)
        self.sensitivity_slider.setEnabled(False)
        self.contrast_slider.setEnabled(False)
        self.brightness_slider.setEnabled(False)
        self.detect_objects_btn.setEnabled(False)
        
        # Enable draw/skip buttons
        self.set_roi_btn.setEnabled(True)
        self.skip_roi_btn.setEnabled(True)
        
        self.roi_status_label.setText("Click Draw Area or Skip to continue")
        self.roi_status_label.setStyleSheet("color: #FFA500;")  # Orange for action needed
        
        # Clear preview detection overlay
        if config.first_frame is not None:
            self.display_frame(config.first_frame)
        
        self.status_bar.showMessage("Tracking area cleared - Set area or skip to continue")
    
    def skip_tracking_area(self):
        """Skip tracking area definition and proceed to object detection settings."""
        if not self.video_configs or self.current_config_idx < 0:
            return
        
        config = self.video_configs[self.current_config_idx]
        config.roi_rectangles = []  # Clear any existing polygon
        config.tracking_area_set = True  # Mark as set (skipped)
        
        # Update UI
        self.roi_status_label.setText("Tracking area skipped (full frame)")
        self.roi_status_label.setStyleSheet("color: #90EE90;")  # Green for completed
        
        # Disable draw/skip buttons, enable clear
        self.set_roi_btn.setEnabled(False)
        self.skip_roi_btn.setEnabled(False)
        self.clear_roi_btn.setEnabled(True)
        
        # Enable object detection settings
        self.enable_detection_settings()
        
        self.status_bar.showMessage("Tracking area skipped - Click 'Detect Objects' to preview")
    
    def enable_detection_settings(self):
        """Enable object detection settings after tracking area is set/skipped."""
        self.min_area_slider.setEnabled(True)
        self.max_area_slider.setEnabled(True)
        self.min_area_spinbox.setEnabled(True)
        self.max_area_spinbox.setEnabled(True)
        self.sensitivity_slider.setEnabled(True)
        self.contrast_slider.setEnabled(True)
        self.brightness_slider.setEnabled(True)
        self.detect_objects_btn.setEnabled(True)
    
    def apply_contrast_brightness(self, frame, contrast, brightness):
        """Apply contrast and brightness adjustments to frame."""
        # Convert to float for processing
        adjusted = frame.astype(np.float32)
        
        # Apply contrast: new_value = contrast * (old_value - 128) + 128
        adjusted = contrast * (adjusted - 128) + 128
        
        # Apply brightness
        adjusted = adjusted + brightness
        
        # Clip to valid range and convert back
        adjusted = np.clip(adjusted, 0, 255).astype(np.uint8)
        
        return adjusted
    
    def detect_objects_preview(self):
        """Run object detection preview (manual trigger)."""
        self.update_current_config()
        if self.video_configs:
            config = self.video_configs[self.current_config_idx]
            if config.first_frame is not None:
                self.load_and_preview_frame(config.current_frame_idx)
    
    def on_frame_slider_changed(self, frame_idx):
        """Handle frame slider movement."""
        if not self.video_configs:
            return
        
        config = self.video_configs[self.current_config_idx]
        config.current_frame_idx = frame_idx
        self.frame_label.setText(f"{frame_idx} / {config.total_frames}")
        
        # Load and preview the selected frame
        if config.tracking_area_set:
            self.load_and_preview_frame(frame_idx)
        else:
            # Just show frame without detection if tracking area not set
            self.load_frame_only(frame_idx)
    
    def load_frame_only(self, frame_idx):
        """Load and display a specific frame without detection."""
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
                
                # Apply contrast and brightness adjustments
                frame_rgb = self.apply_contrast_brightness(frame_rgb, config.contrast, config.brightness)
                
                config.first_frame = frame_rgb
                self.display_frame(frame_rgb)
        except Exception as e:
            print(f"Error loading frame {frame_idx}: {e}")
    
    def load_and_preview_frame(self, frame_idx):
        """Load a specific frame and run detection with ROI filtering."""
        if not self.video_configs:
            return
        
        config = self.video_configs[self.current_config_idx]
        
        try:
            self.status_bar.showMessage(f"Processing frame {frame_idx}...")
            
            # Create temporary tracker with current settings
            temp_tracker = BackgroundSubtractionTracker(
                video_path=config.video_path,
                min_object_area=config.min_object_area,
                max_object_area=config.max_object_area,
                history=config.history,
                var_threshold=config.var_threshold
            )
            
            # Initialize video
            temp_tracker.initialize_video()
            
            # Build background model from frames before target frame
            start_frame = max(0, frame_idx - 60)
            temp_tracker.cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
            
            for i in range(min(60, frame_idx - start_frame + 1)):
                ret, frame = temp_tracker.cap.read()
                if not ret:
                    break
                
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame
                
                _ = temp_tracker.bg_subtractor.apply(gray)
            
            # Get target frame
            temp_tracker.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = temp_tracker.cap.read()
            
            if ret:
                if len(frame.shape) == 3:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                
                # Apply contrast and brightness adjustments before detection
                frame_adjusted = self.apply_contrast_brightness(frame_rgb, config.contrast, config.brightness)
                
                config.first_frame = frame_rgb  # Store original for reference
                
                # Process adjusted frame with detection
                objects, mask = temp_tracker.process_frame(frame_adjusted, frame_idx)
                
                # Filter objects by ROI polygon if defined
                if config.roi_rectangles:
                    roi_polygon = np.array(config.roi_rectangles, dtype=np.int32)
                    filtered_objects = {}
                    
                    for obj_id, (cx, cy, area) in objects.items():
                        # Check if centroid is inside polygon
                        result = cv2.pointPolygonTest(roi_polygon, (float(cx), float(cy)), False)
                        if result >= 0:  # Inside or on edge
                            filtered_objects[obj_id] = (cx, cy, area)
                    
                    objects = filtered_objects
                
                # Visualize using adjusted frame
                preview = frame_adjusted.copy()
                
                # Create filtered mask showing only objects within size range
                filtered_mask = np.zeros_like(mask)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if config.min_object_area <= area <= config.max_object_area:
                        # Check ROI if defined
                        if config.roi_rectangles:
                            cx, cy = centroids[i]
                            roi_polygon = np.array(config.roi_rectangles, dtype=np.int32)
                            result = cv2.pointPolygonTest(roi_polygon, (float(cx), float(cy)), False)
                            if result >= 0:
                                filtered_mask[labels == i] = 255
                        else:
                            filtered_mask[labels == i] = 255
                
                # Draw filtered mask overlay
                mask_colored = np.zeros_like(preview)
                mask_colored[:, :, 1] = filtered_mask
                preview = cv2.addWeighted(preview, 0.6, mask_colored, 0.4, 0)
                
                # Draw tracking polygon if defined
                if config.roi_rectangles:
                    points = np.array(config.roi_rectangles, dtype=np.int32)
                    cv2.polylines(preview, [points], isClosed=True, color=(0, 255, 0), thickness=2)
                
                # Draw objects
                colors = [(255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0)]
                for obj_id, (cx, cy, area) in objects.items():
                    color = colors[obj_id % len(colors)]
                    cv2.circle(preview, (int(cx), int(cy)), 10, color, -1)
                    cv2.circle(preview, (int(cx), int(cy)), 14, color, 2)
                    label = f"ID{obj_id}"
                    cv2.putText(preview, label, (int(cx) + 18, int(cy)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                self.display_frame(preview)
                
                # Update info
                info_text = f"<b>Objects Detected: {len(objects)}</b><br>"
                if objects:
                    for obj_id, (cx, cy, area) in objects.items():
                        info_text += f"• ID {obj_id}: {int(area)}px² at ({int(cx)}, {int(cy)})<br>"
                else:
                    info_text += "<span style='color: #ff9900;'>No objects detected - adjust sliders</span>"
                
                self.detection_info_label.setText(info_text)
            
            temp_tracker.cleanup()
            self.status_bar.showMessage(f"Frame {frame_idx} processed")
            
        except Exception as e:
            self.status_bar.showMessage(f"Error processing frame: {str(e)}")
            print(f"Frame processing error: {e}")
    
    def cancel_polygon_drawing(self):
        """Cancel polygon drawing mode without saving."""
        if not self.drawing_polygon:
            return
        
        self.drawing_polygon = False
        self.polygon_points = []
        
        config = self.video_configs[self.current_config_idx]
        
        # Restore UI state based on whether tracking area was already set
        if config.tracking_area_set:
            # Already had tracking area set - restore to that state
            self.set_roi_btn.setEnabled(False)
            self.skip_roi_btn.setEnabled(False)
            self.clear_roi_btn.setEnabled(True)
            
            if config.roi_rectangles:
                self.roi_status_label.setText(f"Tracking area defined ({len(config.roi_rectangles)} points)")
            else:
                self.roi_status_label.setText("Tracking area skipped (full frame)")
            self.roi_status_label.setStyleSheet("color: #90EE90;")
            
            # Keep detection settings enabled
            self.min_area_slider.setEnabled(True)
            self.max_area_slider.setEnabled(True)
            self.min_area_spinbox.setEnabled(True)
            self.max_area_spinbox.setEnabled(True)
            self.sensitivity_slider.setEnabled(True)
            self.detect_objects_btn.setEnabled(True)
        else:
            # Tracking area not set yet - return to initial state
            self.set_roi_btn.setEnabled(True)
            self.skip_roi_btn.setEnabled(True)
            self.clear_roi_btn.setEnabled(False)
            
            self.roi_status_label.setText("Click Draw Area or Skip to continue")
            self.roi_status_label.setStyleSheet("color: #FFA500;")
            
            # Keep detection settings locked
            self.min_area_slider.setEnabled(False)
            self.max_area_slider.setEnabled(False)
            self.min_area_spinbox.setEnabled(False)
            self.max_area_spinbox.setEnabled(False)
            self.sensitivity_slider.setEnabled(False)
            self.contrast_slider.setEnabled(False)
            self.brightness_slider.setEnabled(False)
            self.detect_objects_btn.setEnabled(False)
        
        # Just show the frame without detection
        if config.first_frame is not None:
            frame_display = config.first_frame.copy()
            
            # Draw existing polygon if any
            if config.roi_rectangles:
                pts = np.array(config.roi_rectangles, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame_display, [pts], True, (0, 255, 0), 2)
            
            self.display_frame(frame_display)
        
        self.status_bar.showMessage("Polygon drawing cancelled")
    
    def update_preview_with_current_settings(self):
        """Update preview with detection using current slider settings."""
        if not self.video_configs:
            return
        
        config = self.video_configs[self.current_config_idx]
        if config.first_frame is None:
            return
        
        # If drawing polygon, just show the frame with partial polygon and dots
        if self.drawing_polygon:
            frame_display = config.first_frame.copy()
            
            # Draw polygon lines in progress
            if len(self.polygon_points) > 1:
                pts = np.array(self.polygon_points, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(frame_display, [pts], False, (0, 255, 0), 2)
            
            # Draw green dots at each point
            for point in self.polygon_points:
                cv2.circle(frame_display, point, 8, (0, 255, 0), -1)
                cv2.circle(frame_display, point, 10, (255, 255, 255), 2)
            
            self.display_frame(frame_display)
            return
        
        try:
            self.status_bar.showMessage("Updating detection preview...")
            
            # Create temporary tracker with current settings
            temp_tracker = BackgroundSubtractionTracker(
                video_path=config.video_path,
                min_object_area=config.min_object_area,
                max_object_area=config.max_object_area,
                history=config.history,
                var_threshold=config.var_threshold
            )
            
            # Initialize video to get properties
            temp_tracker.initialize_video()
            
            # Process first 60 frames to build background model
            for i in range(60):
                ret, frame = temp_tracker.cap.read()
                if not ret:
                    break
                
                if len(frame.shape) == 3:
                    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                else:
                    gray = frame
                
                _ = temp_tracker.bg_subtractor.apply(gray)
            
            # Get next frame for detection
            ret, frame = temp_tracker.cap.read()
            if ret:
                if len(frame.shape) == 3:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                else:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)
                
                # Process frame with detection
                objects, mask = temp_tracker.process_frame(frame_rgb, 60)
                
                # Visualize
                preview = frame_rgb.copy()
                
                # Create filtered mask showing only objects within size range
                filtered_mask = np.zeros_like(mask)
                num_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(mask, connectivity=8)
                for i in range(1, num_labels):
                    area = stats[i, cv2.CC_STAT_AREA]
                    if config.min_object_area <= area <= config.max_object_area:
                        filtered_mask[labels == i] = 255
                
                # Draw filtered mask overlay (only shows objects within size range)
                mask_colored = np.zeros_like(preview)
                mask_colored[:, :, 1] = filtered_mask
                preview = cv2.addWeighted(preview, 0.6, mask_colored, 0.4, 0)
                
                # Draw tracking polygon if defined
                if config.roi_rectangles:
                    points = np.array(config.roi_rectangles, dtype=np.int32)
                    cv2.polylines(preview, [points], isClosed=True, color=(0, 255, 0), thickness=2)
                
                # Draw objects
                colors = [(255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0), (0, 255, 0)]
                for obj_id, (cx, cy, area) in objects.items():
                    color = colors[obj_id % len(colors)]
                    cv2.circle(preview, (int(cx), int(cy)), 10, color, -1)
                    cv2.circle(preview, (int(cx), int(cy)), 14, color, 2)
                    label = f"ID{obj_id}"
                    cv2.putText(preview, label, (int(cx) + 18, int(cy)), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                self.display_frame(preview)
                
                # Update info
                info_text = f"<b>Objects Detected: {len(objects)}</b><br>"
                if objects:
                    for obj_id, (cx, cy, area) in objects.items():
                        info_text += f"• ID {obj_id}: {int(area)}px² at ({int(cx)}, {int(cy)})<br>"
                else:
                    info_text += "<span style='color: #ff9900;'>No objects detected - adjust sliders</span>"
                
                self.detection_info_label.setText(info_text)
            
            temp_tracker.cleanup()
            self.status_bar.showMessage("Preview updated")
            
        except Exception as e:
            self.status_bar.showMessage(f"Preview error: {str(e)}")
            print(f"Preview error: {e}")
    
    def preview_detection(self):
        """Preview object detection on first frame - manual trigger."""
        # Just call the auto-update method
        self.update_preview_with_current_settings()
    
    def display_frame(self, frame: np.ndarray):
        """Display frame in preview label."""
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.preview_label.setPixmap(scaled_pixmap)
    
    def start_batch_tracking(self):
        """Start batch processing all videos."""
        if not self.video_configs:
            return
        
        # Check if all videos configured
        unconfigured = [i for i, c in enumerate(self.video_configs) if not c.configured]
        if unconfigured:
            reply = QMessageBox.question(
                self,
                "Unconfigured Videos",
                f"{len(unconfigured)} video(s) not configured. Use default settings for these?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.No:
                return
            
            # Mark as configured with defaults
            for idx in unconfigured:
                self.video_configs[idx].configured = True
        
        # Create worker
        self.tracking_worker = BatchTrackingWorker(self.video_configs)
        self.tracking_worker.progress.connect(self.update_progress)
        self.tracking_worker.status.connect(self.update_status)
        self.tracking_worker.video_finished.connect(self.video_finished)
        self.tracking_worker.all_finished.connect(self.all_finished)
        self.tracking_worker.frame_ready.connect(self.update_preview)
        
        # Start tracking
        self.tracking_worker.start()
        
        # Update UI
        self.start_batch_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.select_video_btn.setEnabled(False)
        self.status_bar.showMessage("Batch tracking started...")
    
    def cancel_tracking(self):
        """Cancel batch tracking."""
        if self.tracking_worker:
            self.tracking_worker.cancel()
            self.cancel_btn.setEnabled(False)
            self.status_bar.showMessage("Cancelling...")
    
    def update_progress(self, video_idx: int, frame_idx: int, total_frames: int):
        """Update progress bar."""
        # Overall progress across all videos
        videos_complete = video_idx
        total_videos = len(self.video_configs)
        
        # Calculate weighted progress
        video_progress = (videos_complete / total_videos) * 100
        frame_progress = (frame_idx / total_frames) * (100 / total_videos)
        overall_progress = int(video_progress + frame_progress)
        
        self.progress_bar.setValue(overall_progress)
        self.progress_label.setText(
            f"Video {video_idx + 1}/{total_videos} - Frame {frame_idx}/{total_frames}"
        )
    
    def update_status(self, message: str):
        """Update status message."""
        self.status_bar.showMessage(message)
    
    def update_preview(self, frame: np.ndarray, objects: Dict, mask: np.ndarray):
        """Update preview during tracking."""
        # Draw visualization
        preview = frame.copy()
        
        # Draw mask overlay
        mask_colored = np.zeros_like(preview)
        mask_colored[:, :, 1] = mask
        preview = cv2.addWeighted(preview, 0.7, mask_colored, 0.3, 0)
        
        # Draw objects
        colors = [(255, 0, 0), (0, 255, 255), (255, 0, 255), (255, 255, 0)]
        for obj_id, (cx, cy, area) in objects.items():
            color = colors[obj_id % len(colors)]
            cv2.circle(preview, (int(cx), int(cy)), 8, color, -1)
            cv2.putText(preview, f"ID:{obj_id}", (int(cx) + 10, int(cy)),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
        
        self.display_frame(preview)
    
    def video_finished(self, video_idx: int, success: bool, message: str, stats: Dict):
        """Handle single video completion."""
        # Update video list item to show completion
        item = self.video_list.item(video_idx)
        if success:
            item.setText(f"✓ {item.text()}")
            item.setForeground(Qt.green)
        else:
            item.setText(f"✗ {item.text()}")
            item.setForeground(Qt.red)
    
    def all_finished(self, success: bool, message: str):
        """Handle batch completion."""
        # Reset UI
        self.start_batch_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.select_video_btn.setEnabled(True)
        self.progress_bar.setValue(100 if success else 0)
        
        # Show completion message
        if success:
            QMessageBox.information(self, "Batch Complete", message)
        else:
            QMessageBox.warning(self, "Batch Incomplete", message)
        
        self.status_bar.showMessage(message)


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
