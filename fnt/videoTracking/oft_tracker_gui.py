#!/usr/bin/env python3
"""
Open Field Test (OFT) Tracker GUI

Interactive SAM-based tracking for open field behavioral tests.
User workflow:
1. Select video file(s)
2. Click on animal in first frame -> SAM segments automatically
3. Draw rectangular ROI for arena boundary
4. Track and export trajectory with center zone metrics

Features:
- Batch processing support
- Real-time tracking preview
- Distance traveled calculation
- Time in center zone analysis (inner 60% of arena)
- CSV export with behavioral metrics

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
    QGroupBox, QLineEdit, QCheckBox, QSpinBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QPainter, QPen, QColor

try:
    from .sam_tracker_base import SAMTrackerBase, calculate_distance_traveled, calculate_time_in_zone
    SAM_TRACKER_AVAILABLE = True
except ImportError:
    SAM_TRACKER_AVAILABLE = False
    print("Warning: SAM tracker base not available")


class InteractiveVideoWidget(QLabel):
    """Widget for displaying video and capturing user clicks/drawings."""
    
    # Signals
    click_signal = pyqtSignal(int, int)  # (x, y) click coordinate
    rectangle_drawn = pyqtSignal(int, int, int, int)  # (x, y, width, height)
    
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
        
        # Drawing state
        self.drawing_mode = None  # None, 'click', 'rectangle'
        self.rectangle_start = None
        self.rectangle_end = None
        self.temp_rectangle_point = None
        
        # Tracking visualization
        self.trajectory_points = []  # List of (x, y) tuples
        self.current_position = None
        
    def set_frame(self, frame: np.ndarray):
        """Display frame (BGR format from OpenCV)."""
        self.original_frame = frame.copy()
        self._update_display()
        
    def _update_display(self):
        """Update display with current frame and overlays."""
        if self.original_frame is None:
            return
            
        # Convert BGR to RGB
        display_frame = cv2.cvtColor(self.original_frame, cv2.COLOR_BGR2RGB)
        
        # Draw overlays - rectangle arena
        if self.rectangle_start is not None and self.rectangle_end is not None:
            x1, y1 = self.rectangle_start
            x2, y2 = self.rectangle_end
            # Draw outer arena boundary
            cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
            
            # Draw center zone (inner 60% of arena)
            width = abs(x2 - x1)
            height = abs(y2 - y1)
            center_x = (x1 + x2) // 2
            center_y = (y1 + y2) // 2
            inner_w = int(width * 0.6 / 2)
            inner_h = int(height * 0.6 / 2)
            cv2.rectangle(
                display_frame,
                (center_x - inner_w, center_y - inner_h),
                (center_x + inner_w, center_y + inner_h),
                (255, 255, 0), 2
            )
            
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
        if self.scale_factor == 0:
            return (0, 0)
        img_x = int((widget_x - self.display_offset[0]) / self.scale_factor)
        img_y = int((widget_y - self.display_offset[1]) / self.scale_factor)
        return (img_x, img_y)
        
    def mousePressEvent(self, event):
        """Handle mouse press for click selection or rectangle drawing."""
        if self.drawing_mode is None or self.original_frame is None:
            return
            
        img_x, img_y = self.widget_to_image_coords(event.x(), event.y())
        
        if self.drawing_mode == 'click':
            # Single click selection
            self.click_signal.emit(img_x, img_y)
            
        elif self.drawing_mode == 'rectangle':
            # Rectangle drawing - first click sets corner, second click sets opposite corner
            if self.rectangle_start is None:
                self.rectangle_start = (img_x, img_y)
                self.temp_rectangle_point = (img_x, img_y)
            else:
                # Second click completes rectangle
                self.rectangle_end = (img_x, img_y)
                x1, y1 = self.rectangle_start
                x2, y2 = self.rectangle_end
                # Emit normalized coordinates (x, y, width, height)
                x = min(x1, x2)
                y = min(y1, y2)
                w = abs(x2 - x1)
                h = abs(y2 - y1)
                self.rectangle_drawn.emit(x, y, w, h)
                self.drawing_mode = None  # Finish drawing
                self._update_display()
                
    def mouseMoveEvent(self, event):
        """Handle mouse move for rectangle preview."""
        if self.drawing_mode == 'rectangle' and self.rectangle_start is not None and self.rectangle_end is None:
            img_x, img_y = self.widget_to_image_coords(event.x(), event.y())
            self.temp_rectangle_point = (img_x, img_y)
            
            # Draw preview
            preview_frame = self.original_frame.copy()
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
    """Worker thread for video tracking."""
    
    # Signals
    progress_signal = pyqtSignal(int, int)  # (current_frame, total_frames)
    position_signal = pyqtSignal(float, float, float)  # (x, y, confidence)
    finished_signal = pyqtSignal(str)  # output_path
    error_signal = pyqtSignal(str)  # error_message
    
    def __init__(
        self,
        video_path: str,
        sam_checkpoint: str,
        click_point: Tuple[int, int],
        arena_rectangle: Tuple[int, int, int, int],  # Changed from arena_circle
        sam_update_interval: int,
        model_type: str,
        device: str
    ):
        super().__init__()
        self.video_path = video_path
        self.sam_checkpoint = sam_checkpoint
        self.click_point = click_point
        self.arena_rectangle = arena_rectangle  # (x, y, width, height)
        self.sam_update_interval = sam_update_interval
        self.model_type = model_type
        self.device = device
        self.is_cancelled = False
        
    def run(self):
        """Run tracking process."""
        try:
            # Initialize tracker
            tracker = SAMTrackerBase(
                video_path=self.video_path,
                sam_checkpoint=self.sam_checkpoint,
                model_type=self.model_type,
                device=self.device,
                sam_update_interval=self.sam_update_interval
            )
            
            # Open video
            if not tracker.initialize_video():
                self.error_signal.emit("Failed to open video")
                return
                
            # Load SAM model
            if not tracker.initialize_sam():
                self.error_signal.emit("Failed to load SAM model")
                return
                
            # Read first frame
            tracker.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            ret, frame = tracker.cap.read()
            if not ret:
                self.error_signal.emit("Failed to read first frame")
                return
                
            # Convert BGR to RGB for SAM
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            
            # Initialize tracking
            if not tracker.initialize_tracking(frame_rgb, self.click_point):
                self.error_signal.emit("Failed to initialize tracking")
                return
                
            # Process all frames
            frame_idx = 0
            while not self.is_cancelled:
                ret, frame = tracker.cap.read()
                if not ret:
                    break
                    
                frame_idx += 1
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Track
                result = tracker.process_frame(frame_rgb, frame_idx)
                if result is not None:
                    x, y, confidence = result
                    tracker.trajectory.append((frame_idx, x, y, confidence))
                    self.position_signal.emit(x, y, confidence)
                    
                # Emit progress
                if frame_idx % 10 == 0:
                    self.progress_signal.emit(frame_idx, tracker.total_frames)
                    
            # Export trajectory
            df = tracker.export_trajectory()
            
            # Calculate OFT-specific metrics
            if len(df) > 0:
                # Distance traveled
                distance = calculate_distance_traveled(df)
                
                # Time in center zone (inner 60% of arena rectangle)
                x, y, width, height = self.arena_rectangle
                # Create center zone as inner 60% of rectangle
                center_margin_w = int(width * 0.2)  # 20% margin on each side = 60% center
                center_margin_h = int(height * 0.2)
                center_poly = np.array([
                    [x + center_margin_w, y + center_margin_h],
                    [x + width - center_margin_w, y + center_margin_h],
                    [x + width - center_margin_w, y + height - center_margin_h],
                    [x + center_margin_w, y + height - center_margin_h]
                ], dtype=np.int32)
                time_in_center = calculate_time_in_zone(df, center_poly, tracker.fps)
                
                # Add metadata
                metadata_path = Path(self.video_path).parent / f"{Path(self.video_path).stem}_oft_metrics.txt"
                with open(metadata_path, 'w') as f:
                    f.write(f"OFT Tracking Metrics\n")
                    f.write(f"====================\n\n")
                    f.write(f"Video: {Path(self.video_path).name}\n")
                    f.write(f"Total frames: {tracker.total_frames}\n")
                    f.write(f"Duration: {tracker.total_frames / tracker.fps:.2f} s\n")
                    f.write(f"FPS: {tracker.fps:.2f}\n\n")
                    f.write(f"Distance traveled: {distance:.1f} pixels\n")
                    f.write(f"Time in center zone: {time_in_center:.2f} s ({time_in_center / (tracker.total_frames / tracker.fps) * 100:.1f}%)\n")
                    f.write(f"Average speed: {df['speed'].mean():.2f} pixels/s\n")
                    
            tracker.cleanup()
            
            output_path = str(Path(self.video_path).parent / f"{Path(self.video_path).stem}_trajectory.csv")
            self.finished_signal.emit(output_path)
            
        except Exception as e:
            self.error_signal.emit(f"Tracking error: {str(e)}")
            
    def cancel(self):
        """Cancel tracking."""
        self.is_cancelled = True


class OFTTrackerGUI(QMainWindow):
    """Main GUI for Open Field Test tracking."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Open Field Test Tracker - FieldNeuroToolbox")
        self.setGeometry(100, 100, 1200, 800)
        
        # State
        self.video_paths = []
        self.current_video_idx = 0
        self.sam_checkpoint_path = None
        self.click_point = None
        self.arena_rectangle = None  # Changed from arena_circle
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
            self.device_combo.setText("cuda")
            self.device_combo.setPlaceholderText("cuda (GPU detected)")
        else:
            self.device_combo.setText("cpu")
            self.device_combo.setPlaceholderText("cpu (no GPU detected)")
        self.device_combo.setMaximumWidth(150)
        device_layout.addWidget(self.device_combo)
        device_layout.addStretch()
        sam_layout.addLayout(device_layout)
        
        sam_group.setLayout(sam_layout)
        left_layout.addWidget(sam_group)
        
        # Tracking setup group
        setup_group = QGroupBox("3. Tracking Setup")
        setup_layout = QVBoxLayout()
        
        self.load_first_frame_btn = QPushButton("Load First Frame")
        self.load_first_frame_btn.clicked.connect(self.load_first_frame)
        self.load_first_frame_btn.setEnabled(False)
        setup_layout.addWidget(self.load_first_frame_btn)
        
        self.click_animal_btn = QPushButton("Click on Animal")
        self.click_animal_btn.clicked.connect(self.start_click_mode)
        self.click_animal_btn.setEnabled(False)
        setup_layout.addWidget(self.click_animal_btn)
        
        self.click_status_label = QLabel("Status: Not initialized")
        setup_layout.addWidget(self.click_status_label)
        
        self.draw_arena_btn = QPushButton("Draw Arena Rectangle")
        self.draw_arena_btn.clicked.connect(self.start_draw_rectangle)
        self.draw_arena_btn.setEnabled(False)
        setup_layout.addWidget(self.draw_arena_btn)
        
        self.arena_status_label = QLabel("Arena: Not defined")
        setup_layout.addWidget(self.arena_status_label)
        
        # SAM Update Interval with explanation
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("SAM Update Interval:"))
        self.sam_interval_spin = QSpinBox()
        self.sam_interval_spin.setRange(10, 300)
        self.sam_interval_spin.setValue(30)
        self.sam_interval_spin.setSuffix(" frames")
        self.sam_interval_spin.setToolTip(
            "How often to refine tracking with SAM (in frames).\n\n"
            "Lower values (10-20): More accurate, slower processing\n"
            "Higher values (50-100): Faster processing, may drift\n\n"
            "Default (30): Good balance for most videos"
        )
        interval_layout.addWidget(self.sam_interval_spin)
        setup_layout.addLayout(interval_layout)
        
        # Add explanation label
        interval_help = QLabel(
            "ℹ️ SAM re-segments every N frames. Optical flow tracks between updates. "
            "Lower = more accurate but slower."
        )
        interval_help.setWordWrap(True)
        interval_help.setStyleSheet("color: #888888; font-size: 9px; font-style: italic; padding: 2px 10px;")
        setup_layout.addWidget(interval_help)
        
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
        self.video_widget.click_signal.connect(self.on_click_selected)
        self.video_widget.rectangle_drawn.connect(self.on_rectangle_drawn)
        
        main_layout.addWidget(left_panel)
        main_layout.addWidget(self.video_widget, stretch=1)
        
    def select_videos(self):
        """Open file dialog to select video files."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video Files",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*)"
        )
        
        if file_paths:
            self.video_paths = file_paths
            self.current_video_idx = 0
            self.video_list_label.setText(f"Selected {len(file_paths)} video(s):\n" + "\n".join([Path(p).name for p in file_paths[:3]]) + ("..." if len(file_paths) > 3 else ""))
            self.load_first_frame_btn.setEnabled(True)
            
    def select_sam_checkpoint(self):
        """Select SAM model checkpoint file and auto-detect model type."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select SAM Checkpoint",
            "",
            "PyTorch Model (*.pth);;All Files (*)"
        )
        
        if file_path:
            self.sam_checkpoint_path = file_path
            filename = Path(file_path).name
            self.sam_path_label.setText(f"Checkpoint: {filename}")
            
            # Auto-detect model type from filename
            if "vit_h" in filename.lower():
                model_type = "vit_h (Huge - Best accuracy)"
            elif "vit_l" in filename.lower():
                model_type = "vit_l (Large - Balanced)"
            elif "vit_b" in filename.lower():
                model_type = "vit_b (Base - Fastest)"
            else:
                model_type = "Unknown (will try vit_h)"
            
            self.model_type_label.setText(f"Model Type: {model_type}")
            self.model_type_label.setStyleSheet("color: #0078d4; font-weight: bold;")
            
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
        cap = cv2.VideoCapture(video_path)
        ret, frame = cap.read()
        cap.release()
        
        if ret:
            self.video_widget.set_frame(frame)
            self.video_widget.clear_overlays()
            self.click_animal_btn.setEnabled(True)
            self.click_point = None
            self.arena_rectangle = None  # Changed from arena_circle
            self.click_status_label.setText("Status: Frame loaded - click on animal")
            self.arena_status_label.setText("Arena: Not defined")
            self.start_tracking_btn.setEnabled(False)
        else:
            QMessageBox.warning(self, "Error", "Failed to load first frame")
            
    def start_click_mode(self):
        """Enable click mode for animal selection."""
        self.video_widget.drawing_mode = 'click'
        self.click_animal_btn.setEnabled(False)
        self.click_status_label.setText("Status: Click on the animal in the video")
        
    def on_click_selected(self, x: int, y: int):
        """Handle animal selection click."""
        self.click_point = (x, y)
        self.video_widget.drawing_mode = None
        self.click_status_label.setText(f"Status: Animal selected at ({x}, {y})")
        self.draw_arena_btn.setEnabled(True)
        self.click_animal_btn.setEnabled(True)
        
    def start_draw_rectangle(self):
        """Enable rectangle drawing mode for arena definition."""
        self.video_widget.drawing_mode = 'rectangle'
        self.draw_arena_btn.setEnabled(False)
        self.arena_status_label.setText("Arena: Click first corner, then opposite corner")
        
    def on_rectangle_drawn(self, x: int, y: int, width: int, height: int):
        """Handle arena rectangle drawn."""
        self.arena_rectangle = (x, y, width, height)
        self.arena_status_label.setText(f"Arena: {width}x{height} px at ({x}, {y})")
        self.draw_arena_btn.setEnabled(True)
        
        # Enable tracking if all prerequisites met
        if self.click_point and self.sam_checkpoint_path:
            self.start_tracking_btn.setEnabled(True)
            
    def start_tracking(self):
        """Start tracking process."""
        if not self.click_point or not self.arena_rectangle or not self.sam_checkpoint_path:
            QMessageBox.warning(self, "Setup Incomplete", "Please complete all setup steps:\n1. Select animal\n2. Draw arena\n3. Select SAM checkpoint")
            return
            
        video_path = self.video_paths[self.current_video_idx]
        
        # Create worker thread
        self.tracking_worker = TrackingWorker(
            video_path=video_path,
            sam_checkpoint=self.sam_checkpoint_path,
            click_point=self.click_point,
            arena_rectangle=self.arena_rectangle,  # Changed from arena_circle
            sam_update_interval=self.sam_interval_spin.value(),
            model_type=self.get_model_type_from_checkpoint(),  # Auto-detect from filename
            device=self.device_combo.text()
        )
        
        # Connect signals
        self.tracking_worker.progress_signal.connect(self.on_tracking_progress)
        self.tracking_worker.position_signal.connect(self.on_position_update)
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
    """Run the OFT Tracker GUI."""
    app = QApplication(sys.argv)
    
    if not SAM_TRACKER_AVAILABLE:
        QMessageBox.critical(
            None,
            "Dependencies Missing",
            "SAM tracker dependencies not available.\n\nInstall with:\npip install opencv-python torch segment-anything pandas numpy"
        )
        return
        
    window = OFTTrackerGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
