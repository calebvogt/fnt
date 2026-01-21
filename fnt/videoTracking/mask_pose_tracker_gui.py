#!/usr/bin/env python3
"""
Mask Pose Tracker GUI

SAM-based pose tracking using mask shape analysis for behavioral tests.
Segments the animal on every frame and extracts rich pose features from mask geometry.

User workflow:
1. Select video file(s)
2. Click on animal in first frame -> SAM segments automatically
3. Draw rectangular ROI for arena boundary
4. Track and export trajectory with pose features for behavioral clustering

Features:
- Frame-by-frame SAM segmentation with ROI cropping for speed
- Rich mask-based pose feature extraction (shape, orientation, skeleton)
- Real-time tracking preview
- Distance traveled calculation
- CSV export with behavioral metrics and pose features
- Designed for UMAP clustering and behavioral motif discovery

Author: FieldNeuroToolbox Contributors
"""

import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple

# Import torch for CUDA availability check
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QMessageBox,
    QGroupBox, QLineEdit, QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox, QSlider
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor

# Import SAM 2 components
try:
    from .sam2_tracker import SAM2MultiObjectTracker
    from .sam2_checkpoint_manager import get_sam2_checkpoint, SAM2CheckpointDialog
    SAM2_AVAILABLE = True
except ImportError:
    SAM2_AVAILABLE = False
    print("Warning: SAM 2 not available")


class InteractiveVideoWidget(QLabel):
    """Widget for displaying video and capturing user clicks/drawings."""
    
    # Signals
    click_signal = pyqtSignal(int, int)  # (x, y) click coordinate
    rectangle_drawn = pyqtSignal(int, int, int, int, object, float)  # (x, y, width, height, corners, rotation_angle)
    first_corner_clicked = pyqtSignal()  # Signal when first corner is clicked (for status update)
    
    def __init__(self):
        super().__init__()
        self.setMinimumSize(800, 600)
        self.setAlignment(Qt.AlignCenter)
        self.setStyleSheet("background-color: #1e1e1e; border: 1px solid #3f3f3f;")
        
        # Display state
        self.current_frame = None
        self.original_frame = None
        self.scale_factor = 1.0
        self.display_offset = (0, 0)
        self.is_rgb = False  # Flag to indicate if frame is RGB (True) or BGR (False)
        
        # Drawing state
        self.drawing_mode = None  # None, 'click', 'rectangle', 'manipulate'
        self.rectangle_start = None
        self.rectangle_end = None
        self.temp_rectangle_point = None
        self.rectangle_corners = None  # For rotated rectangle: [(x1,y1), (x2,y2), (x3,y3), (x4,y4)]
        self.rotation_angle = 0.0  # Rotation angle in degrees (0-360, continuous)
        self.rectangle_center = None  # Center point of rectangle
        self.rectangle_width = None  # Original width
        self.rectangle_height = None  # Original height
        
        # Rectangle manipulation state
        self.manipulation_mode = None  # None, 'move', 'resize_tl', 'resize_tr', 'resize_bl', 'resize_br', 'rotate'
        self.drag_start_pos = None
        self.drag_start_angle = None  # Starting angle for rotation
        self.original_rectangle = None
        
        # Tracking visualization
        self.trajectory_points = []  # List of (x, y) tuples
        self.current_position = None
        
    def set_frame(self, frame: np.ndarray, is_rgb: bool = False):
        """Display frame (RGB or BGR format)."""
        self.original_frame = frame.copy()
        self.is_rgb = is_rgb
        self._update_display()
        
    def _update_display(self):
        """Update display with current frame and overlays."""
        if self.original_frame is None:
            return
            
        # Convert BGR to RGB if needed
        if self.is_rgb:
            display_frame = self.original_frame.copy()
        else:
            display_frame = cv2.cvtColor(self.original_frame, cv2.COLOR_BGR2RGB)
        
        # Draw overlays - rectangle arena
        if self.rectangle_start is not None and self.rectangle_end is not None:
            # Draw simple green rectangle (no handles, no inner zones)
            x1, y1 = self.rectangle_start
            x2, y2 = self.rectangle_end
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
        # Draw trajectory
        if len(self.trajectory_points) > 1:
            points = np.array(self.trajectory_points, dtype=np.int32)
            cv2.polylines(display_frame, [points], False, (255, 0, 255), 2)
            
        # Draw current position
        if self.current_position is not None:
            cv2.circle(display_frame, self.current_position, 8, (0, 255, 255), -1)
            cv2.circle(display_frame, self.current_position, 10, (0, 0, 255), 2)
            
        # Convert to QPixmap
        height, width, channel = display_frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(display_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit widget
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        
        # Calculate scale factor and offset for coordinate mapping
        self.scale_factor = scaled_pixmap.width() / pixmap.width()
        self.display_offset = (
            (self.width() - scaled_pixmap.width()) // 2,
            (self.height() - scaled_pixmap.height()) // 2
        )
        
        self.setPixmap(scaled_pixmap)
        self.current_frame = display_frame
        
    def widget_to_image_coords(self, widget_x: int, widget_y: int) -> Tuple[int, int]:
        """Convert widget coordinates to original image coordinates."""
        if self.original_frame is None:
            return (0, 0)
            
        # Get the current pixmap
        pixmap = self.pixmap()
        if pixmap is None:
            return (0, 0)
        
        # Calculate where the pixmap is actually displayed (it's centered in the QLabel)
        pixmap_width = pixmap.width()
        pixmap_height = pixmap.height()
        widget_width = self.width()
        widget_height = self.height()
        
        # Calculate offset (pixmap is centered)
        offset_x = (widget_width - pixmap_width) // 2
        offset_y = (widget_height - pixmap_height) // 2
        
        # Convert to pixmap coordinates
        pixmap_x = widget_x - offset_x
        pixmap_y = widget_y - offset_y
        
        # Check if click is outside pixmap bounds
        if pixmap_x < 0 or pixmap_y < 0 or pixmap_x >= pixmap_width or pixmap_y >= pixmap_height:
            return (0, 0)
        
        # Calculate scale factor (pixmap size / original image size)
        original_height, original_width = self.original_frame.shape[:2]
        scale_x = pixmap_width / original_width
        scale_y = pixmap_height / original_height
        
        # Convert to original image coordinates
        img_x = int(pixmap_x / scale_x)
        img_y = int(pixmap_y / scale_y)
        
        # Clamp to image bounds
        img_x = max(0, min(img_x, original_width - 1))
        img_y = max(0, min(img_y, original_height - 1))
        
        return (img_x, img_y)
    
    def get_rectangle_corners(self) -> Optional[Tuple]:
        """Get rectangle corners as (tl, tr, br, bl) tuples."""
        # If we have rotated corners, return those
        if self.rectangle_corners and abs(self.rotation_angle) > 0.1:
            return tuple(self.rectangle_corners)
        
        # Otherwise use bounding box
        if self.rectangle_start is None or self.rectangle_end is None:
            return None
        x1, y1 = self.rectangle_start
        x2, y2 = self.rectangle_end
        return (
            (min(x1, x2), min(y1, y2)),  # Top-left
            (max(x1, x2), min(y1, y2)),  # Top-right
            (max(x1, x2), max(y1, y2)),  # Bottom-right
            (min(x1, x2), max(y1, y2))   # Bottom-left
        )
    
    def get_rotation_handle_position(self) -> Optional[Tuple[int, int]]:
        """Get position of rotation handle (above center of rectangle)."""
        corners = self.get_rectangle_corners()
        if corners is None:
            return None
        tl, tr, br, bl = corners
        center_x = (tl[0] + tr[0]) // 2
        top_y = tl[1]
        # Place rotation handle 40 pixels above top edge (or at y=20 if near top)
        handle_y = max(20, top_y - 40)
        return (center_x, handle_y)
    
    def rotate_rectangle_90_degrees(self):
        """Rotate the rectangle 90 degrees clockwise around its center."""
        # This is now replaced by continuous rotation in mouseMoveEvent
        # Kept for backwards compatibility
        self.rotation_angle = (self.rotation_angle + 90) % 360
        self.update_rectangle_corners()
    
    def update_rectangle_corners(self):
        """Update rectangle corners based on center, dimensions, and rotation angle."""
        if self.rectangle_center is None or self.rectangle_width is None or self.rectangle_height is None:
            return
        
        cx, cy = self.rectangle_center
        w = self.rectangle_width
        h = self.rectangle_height
        
        # Calculate 4 corners of unrotated rectangle centered at origin
        half_w = w / 2
        half_h = h / 2
        corners_unrotated = [
            (-half_w, -half_h),  # Top-left
            (half_w, -half_h),   # Top-right
            (half_w, half_h),    # Bottom-right
            (-half_w, half_h)    # Bottom-left
        ]
        
        # Rotate corners
        angle_rad = np.radians(self.rotation_angle)
        cos_a = np.cos(angle_rad)
        sin_a = np.sin(angle_rad)
        
        rotated_corners = []
        for x, y in corners_unrotated:
            # Rotate: x' = x*cos - y*sin, y' = x*sin + y*cos
            rx = x * cos_a - y * sin_a
            ry = x * sin_a + y * cos_a
            # Translate to center
            rotated_corners.append((int(cx + rx), int(cy + ry)))
        
        self.rectangle_corners = rotated_corners
        
        # Update bounding box for compatibility
        xs = [c[0] for c in rotated_corners]
        ys = [c[1] for c in rotated_corners]
        self.rectangle_start = (min(xs), min(ys))
        self.rectangle_end = (max(xs), max(ys))
    
    def point_near_position(self, px: int, py: int, tx: int, ty: int, threshold: int = 15) -> bool:
        """Check if point (px, py) is near target (tx, ty) within threshold."""
        return abs(px - tx) <= threshold and abs(py - ty) <= threshold
    
    def detect_rectangle_manipulation_zone(self, x: int, y: int) -> Optional[str]:
        """
        Detect which manipulation zone was clicked.
        Returns: 'move', 'resize_tl', 'resize_tr', 'resize_br', 'resize_bl', 'rotate', or None
        """
        corners = self.get_rectangle_corners()
        if corners is None:
            return None
        
        tl, tr, br, bl = corners
        
        # Check rotation handle first (has priority)
        rotation_handle = self.get_rotation_handle_position()
        if rotation_handle and self.point_near_position(x, y, *rotation_handle, threshold=20):
            return 'rotate'
        
        # Check corner handles (resize)
        if self.point_near_position(x, y, *tl):
            return 'resize_tl'
        if self.point_near_position(x, y, *tr):
            return 'resize_tr'
        if self.point_near_position(x, y, *br):
            return 'resize_br'
        if self.point_near_position(x, y, *bl):
            return 'resize_bl'
        
        # Check if inside rectangle (move) - works for both rotated and non-rotated
        # Use point-in-polygon test for rotated rectangles
        if self.rectangle_corners and abs(self.rotation_angle) > 0.1:
            # Point in polygon test using cv2
            contour = np.array(self.rectangle_corners, dtype=np.int32)
            result = cv2.pointPolygonTest(contour, (float(x), float(y)), False)
            if result >= 0:  # Inside or on edge
                return 'move'
        else:
            # Simple bounding box check for axis-aligned rectangles
            xs = [c[0] for c in corners]
            ys = [c[1] for c in corners]
            if min(xs) <= x <= max(xs) and min(ys) <= y <= max(ys):
                return 'move'
        
        return None
        
    def mousePressEvent(self, event):
        """Handle mouse press for rectangle drawing (two clicks)."""
        if self.original_frame is None:
            return
        
        img_x, img_y = self.widget_to_image_coords(event.x(), event.y())
        
        if self.drawing_mode is None:
            return
        
        if self.drawing_mode == 'click':
            # Single click selection
            self.click_signal.emit(img_x, img_y)
        
        elif self.drawing_mode == 'rectangle':
            # Two-click rectangle drawing
            if self.rectangle_start is None:
                # First click - set start corner
                self.rectangle_start = (img_x, img_y)
                self.temp_rectangle_point = (img_x, img_y)
                self._update_display()
                # Signal to parent to update status
                self.first_corner_clicked.emit()
            else:
                # Second click - complete rectangle
                self.rectangle_end = (img_x, img_y)
                x1, y1 = self.rectangle_start
                x2, y2 = self.rectangle_end
                # Emit normalized coordinates (x, y, width, height)
                x = min(x1, x2)
                y = min(y1, y2)
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                # Create non-rotated corners for initial rectangle
                corners = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
                self.rectangle_drawn.emit(x, y, w, h, corners, 0.0)  # 0 rotation for initial draw
                self.drawing_mode = None  # Disable drawing mode after completion
                self._update_display()
                
    def mouseMoveEvent(self, event):
        """Handle mouse move for rectangle preview during two-click drawing."""
        if self.original_frame is None:
            return
        
        img_x, img_y = self.widget_to_image_coords(event.x(), event.y())
        
        # Handle rectangle drawing preview (between first and second click)
        if self.drawing_mode == 'rectangle' and self.rectangle_start is not None and self.rectangle_end is None:
            self.temp_rectangle_point = (img_x, img_y)
            
            # Draw preview
            preview_frame = self.original_frame.copy()
            if not self.is_rgb:
                preview_frame = cv2.cvtColor(preview_frame, cv2.COLOR_BGR2RGB)
            x1, y1 = self.rectangle_start
            cv2.rectangle(preview_frame, (x1, y1), (img_x, img_y), (0, 255, 0), 2)
            
            height, width, channel = preview_frame.shape
            bytes_per_line = 3 * width
            q_image = QImage(preview_frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
            pixmap = QPixmap.fromImage(q_image)
            scaled_pixmap = pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
            self.setPixmap(scaled_pixmap)

            
    def clear_overlays(self):
        """Clear all drawing overlays."""
        self.rectangle_start = None
        self.rectangle_end = None
        self.temp_rectangle_point = None
        self.trajectory_points = []
        self.current_position = None
        if self.original_frame is not None:
            self._update_display()
            
    def add_trajectory_point(self, x: int, y: int):
        """Add point to trajectory visualization."""
        self.trajectory_points.append((int(x), int(y)))
        if len(self.trajectory_points) > 1000:  # Limit for performance
            self.trajectory_points = self.trajectory_points[-1000:]
        self._update_display()
        
    def update_position(self, x: int, y: int):
        """Update current position marker."""
        self.current_position = (int(x), int(y))
        self._update_display()


class TrackingWorker(QThread):
    """Worker thread for SAM 2 video tracking."""
    
    # Signals
    progress_signal = pyqtSignal(int, int)  # (current_frame, total_frames)
    position_signal = pyqtSignal(float, float, float)  # (x, y, confidence)
    frame_signal = pyqtSignal(np.ndarray)  # current frame with tracking overlay
    finished_signal = pyqtSignal(str)  # output_path
    error_signal = pyqtSignal(str)  # error_message
    
    def __init__(
        self,
        video_path: str,
        sam_checkpoint: str,
        sam_config: str,
        bounding_box: Tuple[int, int, int, int],  # (x1, y1, x2, y2)
        device: str
    ):
        super().__init__()
        self.video_path = video_path
        self.sam_checkpoint = sam_checkpoint
        self.sam_config = sam_config
        self.bounding_box = bounding_box
        self.device = device
        self.is_cancelled = False
        
    def run(self):
        """Run SAM 2 tracking process."""
        try:
            # Initialize SAM 2 tracker
            tracker = SAM2MultiObjectTracker(
                checkpoint_path=self.sam_checkpoint,
                config_name=self.sam_config,
                device=self.device
            )
            
            # Initialize video
            tracker.init_video(self.video_path)
            
            # Add bounding box prompt
            x1, y1, x2, y2 = self.bounding_box
            box = np.array([x1, y1, x2, y2], dtype=np.float32)
            tracker.add_box_prompt(box=box, frame_idx=0, obj_id=1, label="Animal")
            
            # Open video for visualization
            cap = cv2.VideoCapture(self.video_path)
            
            # Track through video
            for frame_idx, object_ids, video_masks in tracker.track_objects():
                if self.is_cancelled:
                    break
                
                # Read frame for visualization
                cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
                ret, frame = cap.read()
                if not ret:
                    continue
                
                # Get mask and centroid (use position index 0, not object ID)
                mask = video_masks[0][0].cpu().numpy() if TORCH_AVAILABLE and torch else video_masks[0][0]
                
                # Calculate centroid
                moments = cv2.moments(mask.astype(np.uint8))
                if moments["m00"] > 0:
                    cx = moments["m10"] / moments["m00"]
                    cy = moments["m01"] / moments["m00"]
                    
                    # Emit position update
                    self.position_signal.emit(cx, cy, 1.0)
                    
                    # Create visualization
                    display_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Draw mask overlay
                    mask_overlay = np.zeros_like(display_frame)
                    mask_overlay[mask > 0] = [0, 255, 0]
                    display_frame = cv2.addWeighted(display_frame, 1.0, mask_overlay, 0.3, 0)
                    
                    # Draw centroid
                    cv2.circle(display_frame, (int(cx), int(cy)), 8, (0, 255, 0), -1)
                    
                    # Emit frame update
                    self.frame_signal.emit(display_frame)
                
                # Emit progress
                self.progress_signal.emit(frame_idx, tracker.total_frames)
            
            cap.release()
            
            # Export trajectory
            trajectories = tracker.export_trajectories()
            output_path = str(Path(self.video_path).parent / f"{Path(self.video_path).stem}_trajectory.csv")
            
            self.finished_signal.emit(output_path)
            
        except Exception as e:
            self.error_signal.emit(f"Tracking error: {str(e)}")
            import traceback
            traceback.print_exc()
            
    def cancel(self):
        """Cancel tracking."""
        self.is_cancelled = True


class MaskPoseTrackerGUI(QMainWindow):
    """Main GUI for Mask Pose Tracker - SAM-based behavioral tracking."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Mask Tracker - FieldNeuroToolbox")
        self.setGeometry(100, 100, 1200, 800)
        
        # State
        self.video_paths = []
        self.current_video_idx = 0
        self.sam_checkpoint_path = None
        self.sam_config_name = None
        self.bounding_box = None  # (x1, y1, x2, y2) for tracking
        self.tracking_worker = None
        
        # Apply dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #cccccc;
            }
            QGroupBox {
                border: 1px solid #3f3f3f;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 8px;
                font-weight: bold;
                color: #cccccc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px 0 5px;
            }
            QLabel {
                color: #cccccc;
                background-color: transparent;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #3f3f3f;
                color: #666666;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #3f3f3f;
                padding: 4px;
                border-radius: 2px;
            }
            QProgressBar {
                border: 1px solid #3f3f3f;
                border-radius: 4px;
                text-align: center;
                color: #cccccc;
                background-color: #1e1e1e;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
            }
            QCheckBox {
                color: #cccccc;
            }
        """)
        
        self.init_ui()
        
        # Check for SAM 2 checkpoint after UI is initialized
        self.check_sam2_checkpoint()
        
    def init_ui(self):
        """Initialize user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - Controls
        left_panel = QWidget()
        left_layout = QVBoxLayout(left_panel)
        left_panel.setMaximumWidth(400)
        
        # File selection group
        file_group = QGroupBox("1. Video Selection")
        file_layout = QVBoxLayout()
        
        self.select_videos_btn = QPushButton("Select Video Files")
        self.select_videos_btn.clicked.connect(self.select_videos)
        file_layout.addWidget(self.select_videos_btn)
        
        self.video_list_label = QLabel("No videos selected")
        self.video_list_label.setWordWrap(True)
        file_layout.addWidget(self.video_list_label)
        
        # Frame navigation slider
        file_layout.addWidget(QLabel("Frame Navigation:"))
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setMinimum(0)
        self.frame_slider.setMaximum(0)
        self.frame_slider.setValue(0)
        self.frame_slider.setEnabled(False)
        self.frame_slider.valueChanged.connect(self.on_frame_slider_changed)
        file_layout.addWidget(self.frame_slider)
        
        self.frame_label = QLabel("Frame: 0 / 0")
        self.frame_label.setStyleSheet("color: #888888; font-size: 9px;")
        file_layout.addWidget(self.frame_label)
        
        file_group.setLayout(file_layout)
        left_layout.addWidget(file_group)
        
        # SAM model group
        sam_group = QGroupBox("2. SAM Model Setup")
        sam_layout = QVBoxLayout()
        
        self.select_sam_btn = QPushButton("Select SAM Checkpoint")
        self.select_sam_btn.clicked.connect(self.select_sam_checkpoint)
        sam_layout.addWidget(self.select_sam_btn)
        
        self.sam_path_label = QLabel("No checkpoint selected")
        self.sam_path_label.setWordWrap(True)
        sam_layout.addWidget(self.sam_path_label)
        
        # Model type (auto-detected from filename)
        self.model_type_label = QLabel("Model Type: (will auto-detect from filename)")
        self.model_type_label.setStyleSheet("color: #888888; font-style: italic; font-size: 10px;")
        sam_layout.addWidget(self.model_type_label)
        
        # Device selection with auto-detect
        device_layout = QHBoxLayout()
        device_layout.addWidget(QLabel("Device:"))
        self.device_combo = QLineEdit()
        # Auto-detect CUDA availability
        if TORCH_AVAILABLE and torch is not None and torch.cuda.is_available():
            gpu_name = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else "GPU"
            self.device_combo.setText("cuda")
            self.device_combo.setPlaceholderText(f"cuda ({gpu_name})")
            self.device_combo.setStyleSheet("QLineEdit { color: #00ff00; font-weight: bold; }")
            device_help = QLabel("✓ GPU detected - SAM will be ~50x faster")
            device_help.setStyleSheet("color: #00ff00; font-size: 10px;")
        else:
            self.device_combo.setText("cpu")
            self.device_combo.setPlaceholderText("cpu (no GPU detected)")
            self.device_combo.setStyleSheet("QLineEdit { color: #ff9900; }")
            device_help = QLabel("⚠ CPU mode - SAM will be VERY slow (~10s per frame). Consider GPU.")
            device_help.setStyleSheet("color: #ff9900; font-size: 10px; font-weight: bold;")
        self.device_combo.setMaximumWidth(150)
        device_layout.addWidget(self.device_combo)
        device_layout.addStretch()
        sam_layout.addLayout(device_layout)
        sam_layout.addWidget(device_help)
        
        sam_group.setLayout(sam_layout)
        left_layout.addWidget(sam_group)
        
        # Tracking setup group
        setup_group = QGroupBox("3. Tracking Setup")
        setup_layout = QVBoxLayout()
        
        self.load_first_frame_btn = QPushButton("Load First Frame")
        self.load_first_frame_btn.clicked.connect(self.load_first_frame)
        self.load_first_frame_btn.setEnabled(False)
        setup_layout.addWidget(self.load_first_frame_btn)
        
        self.draw_box_btn = QPushButton("Draw Bounding Box on Animal")
        self.draw_box_btn.clicked.connect(self.start_draw_bounding_box)
        self.draw_box_btn.setEnabled(False)
        setup_layout.addWidget(self.draw_box_btn)
        
        self.redraw_box_btn = QPushButton("Redraw Bounding Box")
        self.redraw_box_btn.clicked.connect(self.redraw_bounding_box)
        self.redraw_box_btn.setEnabled(False)
        setup_layout.addWidget(self.redraw_box_btn)
        
        self.box_status_label = QLabel("Status: Not initialized")
        setup_layout.addWidget(self.box_status_label)
        
        flow_help = QLabel("Window = pixel neighborhood around centroid tracked between SAM updates")
        flow_help.setWordWrap(True)
        flow_help.setStyleSheet("color: #888888; font-size: 9px; font-style: italic; padding: 2px 10px;")
        setup_layout.addWidget(flow_help)
        
        setup_group.setLayout(setup_layout)
        left_layout.addWidget(setup_group)
        
        # Tracking control group
        control_group = QGroupBox("4. Run Tracking")
        control_layout = QVBoxLayout()
        
        self.start_tracking_btn = QPushButton("Start Tracking")
        self.start_tracking_btn.clicked.connect(self.start_tracking)
        self.start_tracking_btn.setEnabled(False)
        control_layout.addWidget(self.start_tracking_btn)
        
        self.cancel_tracking_btn = QPushButton("Cancel")
        self.cancel_tracking_btn.clicked.connect(self.cancel_tracking)
        self.cancel_tracking_btn.setEnabled(False)
        control_layout.addWidget(self.cancel_tracking_btn)
        
        self.progress_bar = QProgressBar()
        control_layout.addWidget(self.progress_bar)
        
        self.status_label = QLabel("Ready")
        self.status_label.setWordWrap(True)
        control_layout.addWidget(self.status_label)
        
        control_group.setLayout(control_layout)
        left_layout.addWidget(control_group)
        
        left_layout.addStretch()
        
        # Right panel - Video display
        self.video_widget = InteractiveVideoWidget()
        self.video_widget.rectangle_drawn.connect(self.on_bounding_box_drawn)
        self.video_widget.first_corner_clicked.connect(self.on_first_corner_clicked)
        
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.video_widget, stretch=1)
        
    def select_videos(self):
        """Open file dialog to select video files."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video Files (Ctrl+Click for multiple)",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        
        if file_paths:
            self.video_paths = file_paths
            self.current_video_idx = 0
            video_names = "\n".join([Path(p).name for p in file_paths[:3]])
            more_text = f"\n... and {len(file_paths) - 3} more" if len(file_paths) > 3 else ""
            self.video_list_label.setText(
                f"<b>Selected {len(file_paths)} video(s)</b> (Video 1/{len(file_paths)}):<br>"
                f"{video_names}{more_text}"
            )
            # Auto-load first frame with slider
            self.load_first_frame_with_slider()
            
    def select_sam_checkpoint(self):
        """Select or download SAM 2 checkpoint."""
        if not SAM2_AVAILABLE:
            QMessageBox.warning(
                self,
                "SAM 2 Not Available",
                "SAM 2 is not installed. Please install with:\n\n"
                "pip install git+https://github.com/facebookresearch/sam2.git"
            )
            return
        
        # Create custom dialog with proper button labels
        msg = QMessageBox(self)
        msg.setIcon(QMessageBox.Question)
        msg.setWindowTitle("Select SAM 2 Checkpoint")
        msg.setText("Would you like to:")
        msg.setInformativeText("• Download a new checkpoint (recommended)\n• Browse for an existing checkpoint file")
        
        download_btn = msg.addButton("Download New", QMessageBox.AcceptRole)
        browse_btn = msg.addButton("Browse", QMessageBox.ActionRole)
        cancel_btn = msg.addButton("Cancel", QMessageBox.RejectRole)
        
        msg.exec_()
        clicked_button = msg.clickedButton()
        
        if clicked_button == download_btn:
            # Download new checkpoint
            checkpoint_path, config_name = get_sam2_checkpoint(parent=self)
            if checkpoint_path:
                self.sam_checkpoint_path = str(checkpoint_path)
                self.sam_config_name = config_name
                self.sam_path_label.setText(f"✓ {checkpoint_path.name}")
                self.sam_path_label.setStyleSheet("color: #4CAF50;")
                self.model_type_label.setText(f"Config: {config_name}")
                
        elif clicked_button == browse_btn:
            # Browse for existing file
            file_path, _ = QFileDialog.getOpenFileName(
                self,
                "Select SAM 2 Checkpoint",
                "",
                "PyTorch Model (*.pt *.pth);;All Files (*)"
            )
            
            if file_path:
                self.sam_checkpoint_path = file_path
                filename = Path(file_path).name
                self.sam_path_label.setText(f"Checkpoint: {filename}")
                
                # Auto-detect config from filename
                if "tiny" in filename.lower():
                    self.sam_config_name = "sam2.1_hiera_t.yaml"
                elif "small" in filename.lower():
                    self.sam_config_name = "sam2.1_hiera_s.yaml"
                elif "base" in filename.lower():
                    self.sam_config_name = "sam2.1_hiera_b+.yaml"
                elif "large" in filename.lower():
                    self.sam_config_name = "sam2.1_hiera_l.yaml"
                else:
                    self.sam_config_name = "sam2.1_hiera_l.yaml"
                
                self.model_type_label.setText(f"Config: {self.sam_config_name}")
                self.model_type_label.setStyleSheet("color: #0078d4; font-weight: bold;")
    
    def check_sam2_checkpoint(self):
        """Check if SAM 2 checkpoint exists, prompt user to download if needed."""
        if not SAM2_AVAILABLE:
            QMessageBox.warning(
                self,
                "SAM 2 Not Available",
                "SAM 2 is not installed. Please install with:\n\n"
                "pip install git+https://github.com/facebookresearch/sam2.git"
            )
            return
        
        # Check default SAM_models directory
        default_dir = Path(__file__).parent.parent.parent / "SAM_models"
        
        # Check if any checkpoints exist
        existing_checkpoints = []
        if default_dir.exists():
            existing_checkpoints = list(default_dir.glob("sam2*.pt"))
        
        if not existing_checkpoints:
            # Create custom dialog with proper button labels
            msg = QMessageBox(self)
            msg.setIcon(QMessageBox.Question)
            msg.setWindowTitle("SAM 2 Checkpoint Required")
            msg.setText("The Mask Tracker requires a SAM 2 model checkpoint.")
            msg.setInformativeText("Would you like to download one now?\n\n(Models will be saved to SAM_models folder for future use)")
            
            download_btn = msg.addButton("Download", QMessageBox.AcceptRole)
            browse_btn = msg.addButton("Browse", QMessageBox.ActionRole)
            cancel_btn = msg.addButton("Cancel", QMessageBox.RejectRole)
            
            msg.exec_()
            clicked_button = msg.clickedButton()
            
            if clicked_button == download_btn:
                checkpoint_path, config_name = get_sam2_checkpoint(parent=self)
                if checkpoint_path:
                    self.sam_checkpoint_path = str(checkpoint_path)
                    self.sam_config_name = config_name
                    self.sam_path_label.setText(f"✓ {checkpoint_path.name}")
                    self.sam_path_label.setStyleSheet("color: #4CAF50;")
            elif clicked_button == browse_btn:
                # Browse for existing file
                file_path, _ = QFileDialog.getOpenFileName(
                    self,
                    "Select SAM 2 Checkpoint",
                    "",
                    "PyTorch Model (*.pt *.pth);;All Files (*)"
                )
                
                if file_path:
                    self.sam_checkpoint_path = file_path
                    filename = Path(file_path).name
                    self.sam_path_label.setText(f"✓ {filename}")
                    self.sam_path_label.setStyleSheet("color: #4CAF50;")
                    
                    # Auto-detect config from filename
                    if "tiny" in filename.lower():
                        self.sam_config_name = "sam2.1_hiera_t.yaml"
                    elif "small" in filename.lower():
                        self.sam_config_name = "sam2.1_hiera_s.yaml"
                    elif "base" in filename.lower():
                        self.sam_config_name = "sam2.1_hiera_b+.yaml"
                    elif "large" in filename.lower():
                        self.sam_config_name = "sam2.1_hiera_l.yaml"
                    else:
                        self.sam_config_name = "sam2.1_hiera_l.yaml"
        else:
            # Use first available checkpoint
            self.sam_checkpoint_path = str(existing_checkpoints[0])
            # Determine config from checkpoint name
            ckpt_name = existing_checkpoints[0].name
            if "tiny" in ckpt_name:
                self.sam_config_name = "sam2.1_hiera_t.yaml"
            elif "small" in ckpt_name:
                self.sam_config_name = "sam2.1_hiera_s.yaml"
            elif "base" in ckpt_name:
                self.sam_config_name = "sam2.1_hiera_b+.yaml"
            elif "large" in ckpt_name:
                self.sam_config_name = "sam2.1_hiera_l.yaml"
            else:
                self.sam_config_name = "sam2.1_hiera_l.yaml"
            
            self.sam_path_label.setText(f"✓ {existing_checkpoints[0].name}")
            self.sam_path_label.setStyleSheet("color: #4CAF50;")
            
    def get_model_type_from_checkpoint(self) -> str:
        """Extract model type from checkpoint filename."""
        if not self.sam_checkpoint_path:
            return "vit_h"  # Default
        
        filename = Path(self.sam_checkpoint_path).name.lower()
        if "vit_h" in filename:
            return "vit_h"
        elif "vit_l" in filename:
            return "vit_l"
        elif "vit_b" in filename:
            return "vit_b"
        else:
            return "vit_h"  # Default to best quality
            
    def load_first_frame(self):
        """Load first frame of current video."""
        if not self.video_paths:
            return
            
        video_path = self.video_paths[self.current_video_idx]
        
        # Update video list to show current video
        video_names = "\n".join([Path(p).name for p in self.video_paths[:3]])
        more_text = f"\n... and {len(self.video_paths) - 3} more" if len(self.video_paths) > 3 else ""
        self.video_list_label.setText(
            f"<b>Selected {len(self.video_paths)} video(s)</b> (Video {self.current_video_idx + 1}/{len(self.video_paths)}):<br>"
            f"<span style='color: #0078d4;'>► {Path(video_path).name}</span><br>"
            f"{video_names}{more_text}"
        )
        
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            self.video_widget.set_frame(frame)
            self.video_widget.clear_overlays()
            self.bounding_box = None
            self.box_status_label.setText("Status: Frame loaded - draw bounding box around animal")
            self.draw_box_btn.setEnabled(True)
            self.start_tracking_btn.setEnabled(False)
        else:
            QMessageBox.warning(self, "Error", "Failed to load first frame")
    
    def load_first_frame_with_slider(self):
        """Load video and enable frame slider for navigation."""
        if not self.video_paths:
            return
        
        video_path = self.video_paths[self.current_video_idx]
        
        # Open video to get frame count
        cap = cv2.VideoCapture(video_path)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            # Store video capture for slider navigation
            self.video_cap = cv2.VideoCapture(video_path)
            self.total_frames = total_frames
            
            # Setup slider
            self.frame_slider.setMaximum(total_frames - 1)
            self.frame_slider.setValue(0)
            self.frame_slider.setEnabled(True)
            self.frame_label.setText(f"Frame: 0 / {total_frames}")
            
            # Display first frame
            self.video_widget.set_frame(frame)
            self.video_widget.clear_overlays()
            self.bounding_box = None
            self.box_status_label.setText("Status: Use slider to find good frame, then draw bounding box")
            self.draw_box_btn.setEnabled(True)
            self.start_tracking_btn.setEnabled(False)
            self.load_first_frame_btn.setEnabled(False)
        else:
            QMessageBox.warning(self, "Error", "Failed to load video")
    
    def on_frame_slider_changed(self, value):
        """Update display when slider moves."""
        if hasattr(self, 'video_cap') and self.video_cap is not None:
            self.video_cap.set(cv2.CAP_PROP_POS_FRAMES, value)
            ret, frame = self.video_cap.read()
            if ret:
                self.video_widget.set_frame(frame)
                # Preserve existing bounding box overlay
                if self.bounding_box:
                    self.video_widget.rectangle_start = (self.bounding_box[0], self.bounding_box[1])
                    self.video_widget.rectangle_end = (self.bounding_box[2], self.bounding_box[3])
                    self.video_widget._update_display()
                self.frame_label.setText(f"Frame: {value} / {self.total_frames}")
            
    def start_draw_bounding_box(self):
        """Enable bounding box drawing mode for animal selection."""
        self.video_widget.drawing_mode = 'rectangle'
        self.video_widget.rectangle_start = None
        self.video_widget.rectangle_end = None
        self.video_widget.temp_rectangle_point = None
        self.draw_box_btn.setEnabled(False)
        self.redraw_box_btn.setEnabled(False)
        self.box_status_label.setText("Status: Click first corner of box")
        
    def redraw_bounding_box(self):
        """Clear current box and redraw."""
        self.bounding_box = None
        self.video_widget.rectangle_start = None
        self.video_widget.rectangle_end = None
        self.video_widget.temp_rectangle_point = None
        self.video_widget._update_display()
        self.start_draw_bounding_box()
    
    def on_first_corner_clicked(self):
        """Update status after first corner is clicked."""
        self.box_status_label.setText("Status: Click opposite corner to complete box")
    
    def on_bounding_box_drawn(self, x: int, y: int, width: int, height: int, corners, rotation_angle: float):
        """Handle bounding box drawn around animal."""
        # Convert to x1, y1, x2, y2 format
        x1 = x
        y1 = y
        x2 = x + width
        y2 = y + height
        
        self.bounding_box = (x1, y1, x2, y2)
        
        self.box_status_label.setText(f"Bounding box: ({x1}, {y1}) to ({x2}, {y2})")
        self.box_status_label.setStyleSheet("QLabel { color: #00ff00; font-weight: bold; }")
        
        self.draw_box_btn.setEnabled(True)
        self.redraw_box_btn.setEnabled(True)
        self.start_tracking_btn.setEnabled(True)
        
    def start_tracking(self):
        """Start SAM 2 video tracking."""
        if not self.bounding_box or not self.sam_checkpoint_path:
            QMessageBox.warning(self, "Setup Incomplete", "Please complete all setup steps:\n1. Draw bounding box on animal\n2. Select SAM checkpoint")
            return
            
        video_path = self.video_paths[self.current_video_idx]
        device = self.device_combo.text().strip() or "cpu"
        
        # Create worker thread
        self.tracking_worker = TrackingWorker(
            video_path=video_path,
            sam_checkpoint=self.sam_checkpoint_path,
            sam_config=self.sam_config_name,
            bounding_box=self.bounding_box,
            device=device
        )
        
        # Connect signals
        self.tracking_worker.progress_signal.connect(self.on_tracking_progress)
        self.tracking_worker.position_signal.connect(self.on_position_update)
        self.tracking_worker.frame_signal.connect(self.on_frame_update)
        self.tracking_worker.finished_signal.connect(self.on_tracking_finished)
        self.tracking_worker.error_signal.connect(self.on_tracking_error)
        
        # Update UI
        self.start_tracking_btn.setEnabled(False)
        self.cancel_tracking_btn.setEnabled(True)
        self.status_label.setText("Tracking in progress...")
        self.progress_bar.setValue(0)
        
        # Clear previous trajectory
        self.video_widget.trajectory_points = []
        
        # Start tracking
        self.tracking_worker.start()
        
    def cancel_tracking(self):
        """Cancel ongoing tracking."""
        if self.tracking_worker:
            self.tracking_worker.cancel()
            self.status_label.setText("Cancelling...")
            
    def on_tracking_progress(self, current_frame: int, total_frames: int):
        """Update progress bar."""
        progress = int((current_frame / total_frames) * 100)
        self.progress_bar.setValue(progress)
        
    def on_position_update(self, x: float, y: float, confidence: float):
        """Update position visualization."""
        self.video_widget.add_trajectory_point(int(x), int(y))
        self.video_widget.update_position(int(x), int(y))
        
    def on_frame_update(self, frame: np.ndarray):
        """Update video widget with current tracking frame."""
        self.video_widget.set_frame(frame, is_rgb=True)
        
    def on_tracking_method_changed(self, index: int):
        """Handle tracking method change."""
        # Show/hide optical flow window control based on method
        is_optical_flow = (index == 0)
        # Optical flow window control is always visible but tooltip explains it's only used for optical flow
        # No need to hide it, just update the tooltip if needed
        pass
    
    def on_tracking_finished(self, output_path: str):
        """Handle tracking completion."""
        self.status_label.setText(f"Tracking complete! Saved to:\n{output_path}")
        self.progress_bar.setValue(100)
        self.start_tracking_btn.setEnabled(True)
        self.cancel_tracking_btn.setEnabled(False)
        
        QMessageBox.information(
            self,
            "Tracking Complete",
            f"Trajectory exported to:\n{output_path}\n\nMetrics saved to corresponding _oft_metrics.txt file"
        )
        
        # Move to next video if in batch mode
        if self.current_video_idx < len(self.video_paths) - 1:
            reply = QMessageBox.question(
                self,
                "Next Video",
                f"Process next video ({self.current_video_idx + 2}/{len(self.video_paths)})?",
                QMessageBox.Yes | QMessageBox.No
            )
            if reply == QMessageBox.Yes:
                self.current_video_idx += 1
                self.load_first_frame()
                
    def on_tracking_error(self, error_message: str):
        """Handle tracking error."""
        self.status_label.setText(f"Error: {error_message}")
        self.start_tracking_btn.setEnabled(True)
        self.cancel_tracking_btn.setEnabled(False)
        QMessageBox.critical(self, "Tracking Error", error_message)


def main():
    """Run the Mask Tracker GUI."""
    app = QApplication(sys.argv)
    
    if not SAM2_AVAILABLE:
        QMessageBox.critical(
            None,
            "SAM 2 Not Available",
            "SAM 2 is required for mask tracking.\n\nInstall with:\npip install git+https://github.com/facebookresearch/sam2.git"
        )
        return
        
    window = MaskPoseTrackerGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
