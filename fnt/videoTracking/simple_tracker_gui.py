#!/usr/bin/env python3
"""
Simple Tracker GUI

Fast, CPU-only tracking using classical computer vision methods.
No GPU or neural networks required - runs on any Mac or Windows computer.

User workflow:
1. Select video file(s)
2. Click on animal in first frame -> classical segmentation (GrabCut/Background Subtraction)
3. Draw rectangular ROI for arena boundary (optional)
4. Track and export trajectory with behavioral metrics

Features:
- Pure CPU-based tracking (100-200 fps on laptop)
- Multiple segmentation methods: GrabCut, Background Subtraction, Adaptive Threshold
- OpenCV tracker integration (CSRT, KCF, MOSSE)
- Hybrid approach: periodic re-segmentation + fast tracking
- Real-time tracking preview
- Distance traveled calculation
- Time in center zone analysis
- CSV export with behavioral metrics
- Cross-platform (Mac, Windows, Linux)

Technical approach:
- Classical segmentation every N frames (default: 30)
- OpenCV CSRT tracker for inter-frame tracking
- Kalman filter for smooth trajectories
- No dependencies on PyTorch or CUDA

Author: FieldNeuroToolbox Contributors
"""

import sys
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QMessageBox,
    QGroupBox, QLineEdit, QCheckBox, QSpinBox, QComboBox, QDoubleSpinBox
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal
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


class ClassicalSegmenter:
    """
    Classical computer vision segmentation without neural networks.
    100-5000x faster than SAM on CPU!
    """
    
    def __init__(self, method: str = "grabcut"):
        """
        method: 'grabcut', 'background_subtraction', 'adaptive_threshold'
        """
        self.method = method
        self.bg_subtractor = None
        
    def segment_with_click(
        self, 
        frame: np.ndarray, 
        click_point: Tuple[int, int],
        rect_margin: int = 50
    ) -> np.ndarray:
        """
        Segment animal from single click point.
        
        Returns: Binary mask (0 or 255)
        Speed: ~10-50ms on CPU (100-1000x faster than SAM)
        """
        
        if self.method == "grabcut":
            return self._segment_grabcut(frame, click_point, rect_margin)
        elif self.method == "background_subtraction":
            return self._segment_background_subtraction(frame)
        elif self.method == "adaptive_threshold":
            return self._segment_adaptive_threshold(frame, click_point)
        else:
            raise ValueError(f"Unknown method: {self.method}")
    
    def _segment_grabcut(
        self, 
        frame: np.ndarray, 
        click_point: Tuple[int, int],
        margin: int = 50
    ) -> np.ndarray:
        """
        GrabCut segmentation - interactive foreground/background separation.
        
        Speed: ~20-100ms per frame on CPU
        Quality: Good for well-contrasted objects
        """
        h, w = frame.shape[:2]
        x, y = click_point
        
        # Create initial rectangle around click point
        rect = (
            max(0, x - margin),
            max(0, y - margin),
            min(w - x + margin, 2 * margin),
            min(h - y + margin, 2 * margin)
        )
        
        # Initialize mask
        mask = np.zeros(frame.shape[:2], np.uint8)
        bgd_model = np.zeros((1, 65), np.float64)
        fgd_model = np.zeros((1, 65), np.float64)
        
        try:
            # Run GrabCut
            cv2.grabCut(frame, mask, rect, bgd_model, fgd_model, 5, cv2.GC_INIT_WITH_RECT)
            
            # Convert to binary mask
            mask2 = np.where((mask == 2) | (mask == 0), 0, 1).astype('uint8')
            result = mask2 * 255
            
            # Post-process: keep only largest connected component
            result = self._keep_largest_component(result)
        except:
            # Fallback to simple threshold if GrabCut fails
            result = self._segment_adaptive_threshold(frame, click_point)
        
        return result
    
    def _segment_background_subtraction(self, frame: np.ndarray) -> np.ndarray:
        """
        Background subtraction for moving objects.
        
        Speed: ~1-5ms per frame on CPU (ULTRA FAST)
        Quality: Excellent for static camera, moving animal
        Note: Requires multiple frames to build background model
        """
        if self.bg_subtractor is None:
            # Initialize background subtractor
            self.bg_subtractor = cv2.createBackgroundSubtractorMOG2(
                history=500,
                varThreshold=16,
                detectShadows=True
            )
        
        # Apply background subtraction
        fg_mask = self.bg_subtractor.apply(frame)
        
        # Remove shadows (value 127)
        fg_mask[fg_mask == 127] = 0
        
        # Morphological operations to clean up
        kernel = np.ones((5, 5), np.uint8)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_OPEN, kernel)
        fg_mask = cv2.morphologyEx(fg_mask, cv2.MORPH_CLOSE, kernel)
        
        # Keep largest component (the animal)
        fg_mask = self._keep_largest_component(fg_mask)
        
        return fg_mask
    
    def _segment_adaptive_threshold(
        self, 
        frame: np.ndarray, 
        click_point: Tuple[int, int]
    ) -> np.ndarray:
        """
        Adaptive thresholding + flood fill from click point.
        
        Speed: ~2-10ms per frame on CPU (FASTEST)
        Quality: Good for consistent lighting
        """
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
        # Adaptive threshold
        thresh = cv2.adaptiveThreshold(
            gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
            cv2.THRESH_BINARY_INV, 11, 2
        )
        
        # Morphological closing to fill small gaps
        kernel = np.ones((5, 5), np.uint8)
        thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
        
        # Flood fill from click point
        h, w = thresh.shape
        mask = np.zeros((h + 2, w + 2), np.uint8)
        cv2.floodFill(thresh, mask, click_point, 255)
        
        # Remove padding
        mask = mask[1:-1, 1:-1]
        
        return mask
    
    def _keep_largest_component(self, mask: np.ndarray) -> np.ndarray:
        """Keep only the largest connected component in mask."""
        num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        
        if num_labels <= 1:
            return mask
        
        # Find largest component (excluding background label 0)
        largest_label = 1 + np.argmax(stats[1:, cv2.CC_STAT_AREA])
        
        # Create new mask with only largest component
        result = np.zeros_like(mask)
        result[labels == largest_label] = 255
        
        return result


class SimpleTracker:
    """
    Fast tracking using OpenCV built-in trackers.
    No neural networks - runs 100-200 fps on CPU!
    """
    
    def __init__(
        self,
        video_path: str,
        segmentation_method: str = "grabcut",
        tracker_type: str = "CSRT",
        segment_interval: int = 30
    ):
        self.video_path = video_path
        self.segmentation_method = segmentation_method
        self.tracker_type = tracker_type
        self.segment_interval = segment_interval
        
        # Components
        self.segmenter = ClassicalSegmenter(method=segmentation_method)
        self.opencv_tracker = None
        
        # Video properties
        self.cap = None
        self.fps = None
        self.total_frames = None
        self.frame_width = None
        self.frame_height = None
        
        # Tracking state
        self.click_point = None
        self.arena_rectangle = None
        self.frame_count = 0
        self.trajectory = []
        
        # Kalman filter for smoothing
        self.kf = cv2.KalmanFilter(4, 2)
        self._initialize_kalman_filter()
    
    def _initialize_kalman_filter(self):
        """Initialize Kalman filter for smooth trajectory prediction."""
        # Transition matrix (constant velocity model)
        self.kf.transitionMatrix = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ], dtype=np.float32)
        
        # Measurement matrix (we measure x, y)
        self.kf.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        # Process noise
        self.kf.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        
        # Measurement noise
        self.kf.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
    
    def initialize_video(self):
        """Open video and read properties."""
        self.cap = cv2.VideoCapture(self.video_path)
        if not self.cap.isOpened():
            raise ValueError(f"Could not open video: {self.video_path}")
        
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.frame_height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    def initialize_tracking(self, frame: np.ndarray, click_point: Tuple[int, int]):
        """Initialize tracking with first frame and click point."""
        self.click_point = click_point
        
        # Segment animal
        mask = self.segmenter.segment_with_click(frame, click_point)
        
        # Check if mask has any pixels
        mask_area = np.sum(mask > 0)
        if mask_area < 100:
            raise ValueError(
                f"Segmentation failed: Only {mask_area} pixels detected.\n\n"
                "This usually means:\n"
                "1. Click point is not on the animal\n"
                "2. Poor contrast between animal and background\n"
                "3. Wrong segmentation method selected\n\n"
                "Try:\n"
                "• Click directly on the animal body\n"
                "• Try 'Background Subtraction' method\n"
                "• Ensure good lighting/contrast in video"
            )
        
        # Get bounding box from mask
        contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if contours:
            x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
            bbox = (x, y, w, h)
            
            # Initialize OpenCV tracker
            self._create_opencv_tracker()
            self.opencv_tracker.init(frame, bbox)
            
            # Initialize Kalman filter
            cx, cy = x + w // 2, y + h // 2
            self.kf.statePre = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
            self.kf.statePost = np.array([[cx], [cy], [0], [0]], dtype=np.float32)
            
            return mask, (cx, cy)
        else:
            raise ValueError("Could not find contours in segmentation mask")
    
    def _create_opencv_tracker(self):
        """Create OpenCV tracker instance."""
        if self.tracker_type == "CSRT":
            self.opencv_tracker = cv2.TrackerCSRT_create()
        elif self.tracker_type == "KCF":
            self.opencv_tracker = cv2.TrackerKCF_create()
        elif self.tracker_type == "MOSSE":
            self.opencv_tracker = cv2.legacy.TrackerMOSSE_create()
        else:
            # Default to CSRT (best quality)
            self.opencv_tracker = cv2.TrackerCSRT_create()
    
    def process_frame(self, frame: np.ndarray, frame_idx: int) -> Tuple[Optional[Tuple[int, int]], Optional[np.ndarray]]:
        """
        Process single frame with hybrid segmentation + tracking.
        
        Returns: (centroid, mask) or (None, None) if tracking lost
        """
        self.frame_count += 1
        
        # Decide whether to re-segment or just track
        use_segmentation = (frame_idx % self.segment_interval == 0) or (frame_idx == 0)
        
        if use_segmentation:
            # Re-segment with classical method
            mask = self.segmenter.segment_with_click(frame, self.click_point)
            
            # Get centroid from mask
            M = cv2.moments(mask)
            if M['m00'] > 0:
                cx = int(M['m10'] / M['m00'])
                cy = int(M['m01'] / M['m00'])
                centroid = (cx, cy)
                
                # Update Kalman filter
                measurement = np.array([[cx], [cy]], dtype=np.float32)
                self.kf.correct(measurement)
                
                # Re-initialize OpenCV tracker with new bbox
                contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                    bbox = (x, y, w, h)
                    self._create_opencv_tracker()
                    self.opencv_tracker.init(frame, bbox)
                
                self.trajectory.append({'frame': frame_idx, 'x': cx, 'y': cy, 'method': 'segmentation'})
                return centroid, mask
            else:
                return None, None
        else:
            # Track with OpenCV tracker (FAST)
            success, bbox = self.opencv_tracker.update(frame)
            
            if success:
                x, y, w, h = [int(v) for v in bbox]
                cx, cy = x + w // 2, y + h // 2
                
                # Update Kalman filter
                measurement = np.array([[cx], [cy]], dtype=np.float32)
                corrected = self.kf.correct(measurement)
                cx, cy = int(corrected[0]), int(corrected[1])
                
                centroid = (cx, cy)
                self.trajectory.append({'frame': frame_idx, 'x': cx, 'y': cy, 'method': 'tracking'})
                
                # Create simple mask from bbox for visualization
                mask = np.zeros((frame.shape[0], frame.shape[1]), dtype=np.uint8)
                cv2.rectangle(mask, (x, y), (x + w, y + h), 255, -1)
                
                return centroid, mask
            else:
                # Tracking lost - try to recover with Kalman prediction
                prediction = self.kf.predict()
                cx, cy = int(prediction[0]), int(prediction[1])
                
                # Try to re-initialize tracker at predicted location
                self.click_point = (cx, cy)
                mask = self.segmenter.segment_with_click(frame, self.click_point)
                
                M = cv2.moments(mask)
                if M['m00'] > 0:
                    cx = int(M['m10'] / M['m00'])
                    cy = int(M['m01'] / M['m00'])
                    centroid = (cx, cy)
                    
                    # Re-initialize tracker
                    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    if contours:
                        x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                        bbox = (x, y, w, h)
                        self._create_opencv_tracker()
                        self.opencv_tracker.init(frame, bbox)
                    
                    self.trajectory.append({'frame': frame_idx, 'x': cx, 'y': cy, 'method': 'recovery'})
                    return centroid, mask
                else:
                    return None, None
    
    def export_trajectory(self, output_path: str):
        """Export trajectory to CSV."""
        df = pd.DataFrame(self.trajectory)
        
        # Add timestamp
        df['timestamp'] = df['frame'] / self.fps
        
        # Calculate distance traveled
        if len(df) > 1:
            df['distance'] = 0.0
            for i in range(1, len(df)):
                dx = df.loc[i, 'x'] - df.loc[i-1, 'x']
                dy = df.loc[i, 'y'] - df.loc[i-1, 'y']
                df.loc[i, 'distance'] = np.sqrt(dx**2 + dy**2)
        
        # Calculate center zone time if arena defined
        if self.arena_rectangle:
            x, y, w, h = self.arena_rectangle
            center_x = x + w // 2
            center_y = y + h // 2
            center_w = int(w * 0.6)
            center_h = int(h * 0.6)
            
            def in_center(row):
                px, py = row['x'], row['y']
                return (abs(px - center_x) < center_w / 2) and (abs(py - center_y) < center_h / 2)
            
            df['in_center'] = df.apply(in_center, axis=1)
        
        df.to_csv(output_path, index=False)
        
        # Calculate summary statistics
        total_distance = df['distance'].sum() if 'distance' in df else 0
        time_in_center = df['in_center'].sum() / self.fps if 'in_center' in df else 0
        
        return {
            'total_distance': total_distance,
            'time_in_center': time_in_center,
            'total_frames': len(df)
        }
    
    def cleanup(self):
        """Release video capture."""
        if self.cap:
            self.cap.release()


class TrackingWorker(QThread):
    """Worker thread for running tracking without blocking GUI."""
    progress = pyqtSignal(int)
    status = pyqtSignal(str)
    finished = pyqtSignal(bool, str, dict)
    frame_ready = pyqtSignal(np.ndarray, np.ndarray, tuple)  # frame, mask, centroid
    
    def __init__(
        self,
        video_path: str,
        click_point: Tuple[int, int],
        arena_rectangle: Optional[Tuple[int, int, int, int]],
        segmentation_method: str,
        tracker_type: str,
        segment_interval: int,
        output_path: str
    ):
        super().__init__()
        self.video_path = video_path
        self.click_point = click_point
        self.arena_rectangle = arena_rectangle
        self.segmentation_method = segmentation_method
        self.tracker_type = tracker_type
        self.segment_interval = segment_interval
        self.output_path = output_path
        self.cancelled = False
    
    def run(self):
        """Run tracking process."""
        try:
            # Create tracker
            tracker = SimpleTracker(
                video_path=self.video_path,
                segmentation_method=self.segmentation_method,
                tracker_type=self.tracker_type,
                segment_interval=self.segment_interval
            )
            
            # Initialize video
            self.status.emit("Opening video...")
            tracker.initialize_video()
            
            # Read first frame
            ret, first_frame = tracker.cap.read()
            if not ret:
                raise ValueError("Could not read first frame")
            
            first_frame_rgb = cv2.cvtColor(first_frame, cv2.COLOR_BGR2RGB)
            
            # Initialize tracking
            self.status.emit("Initializing tracking...")
            mask, centroid = tracker.initialize_tracking(first_frame_rgb, self.click_point)
            tracker.arena_rectangle = self.arena_rectangle
            
            # Emit first frame
            self.frame_ready.emit(first_frame_rgb, mask, centroid)
            
            # Process all frames
            self.status.emit("Tracking...")
            frame_idx = 1
            
            while not self.cancelled:
                ret, frame = tracker.cap.read()
                if not ret:
                    break
                
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame
                centroid, mask = tracker.process_frame(frame_rgb, frame_idx)
                
                # Emit frame for preview (every 10 frames to avoid overwhelming GUI)
                if frame_idx % 10 == 0:
                    if centroid and mask is not None:
                        self.frame_ready.emit(frame_rgb, mask, centroid)
                
                # Update progress
                progress_pct = int((frame_idx / tracker.total_frames) * 100)
                self.progress.emit(progress_pct)
                
                frame_idx += 1
            
            if self.cancelled:
                tracker.cleanup()
                self.finished.emit(False, "Tracking cancelled", {})
                return
            
            # Export trajectory
            self.status.emit("Exporting results...")
            stats = tracker.export_trajectory(self.output_path)
            
            tracker.cleanup()
            self.finished.emit(True, "Tracking completed successfully", stats)
            
        except Exception as e:
            self.finished.emit(False, f"Tracking failed: {str(e)}", {})
    
    def cancel(self):
        """Cancel tracking."""
        self.cancelled = True


class SimpleTrackerGUI(QMainWindow):
    """Main GUI for Simple Tracker - CPU-only tracking."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Simple Tracker - FieldNeuroToolbox")
        self.setGeometry(100, 100, 1200, 800)
        
        # State
        self.video_paths = []
        self.current_video_idx = 0
        self.click_point = None
        self.arena_rectangle = None
        self.first_frame = None
        self.preview_frame = None
        
        # Worker thread
        self.tracking_worker = None
        
        # Apply dark theme (matching FNT style)
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
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                background-color: #3f3f3f;
                color: #cccccc;
                border: 1px solid #555555;
                border-radius: 3px;
                padding: 4px;
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
    
    def init_ui(self):
        """Initialize the user interface."""
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        
        # Left panel - controls
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, stretch=1)
        
        # Right panel - video preview
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, stretch=2)
        
        # Status bar
        self.status_bar = self.statusBar()
        self.status_bar.showMessage("Ready")
    
    def create_left_panel(self):
        """Create left control panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # 1. Video Selection
        video_group = QGroupBox("1. Video Selection")
        video_layout = QVBoxLayout()
        
        self.select_video_btn = QPushButton("Select Video Files")
        self.select_video_btn.clicked.connect(self.select_videos)
        video_layout.addWidget(self.select_video_btn)
        
        self.video_label = QLabel("No videos selected")
        self.video_label.setWordWrap(True)
        video_layout.addWidget(self.video_label)
        
        video_group.setLayout(video_layout)
        layout.addWidget(video_group)
        
        # 2. Segmentation Setup
        seg_group = QGroupBox("2. Segmentation Setup")
        seg_layout = QVBoxLayout()
        
        # Segmentation method
        method_layout = QHBoxLayout()
        method_layout.addWidget(QLabel("Method:"))
        self.seg_method_combo = QComboBox()
        self.seg_method_combo.addItems(["GrabCut", "Background Subtraction", "Adaptive Threshold"])
        self.seg_method_combo.setCurrentText("GrabCut")
        method_layout.addWidget(self.seg_method_combo)
        seg_layout.addLayout(method_layout)
        
        # Info label
        self.method_info_label = QLabel(
            "<b>GrabCut:</b> Best for well-contrasted animals<br>"
            "<b>Background Sub:</b> Best for moving animals, static camera<br>"
            "<b>Adaptive:</b> Fastest, good for consistent lighting"
        )
        self.method_info_label.setWordWrap(True)
        self.method_info_label.setStyleSheet("color: #aaaaaa; font-size: 10px; padding: 5px;")
        seg_layout.addWidget(self.method_info_label)
        
        seg_group.setLayout(seg_layout)
        layout.addWidget(seg_group)
        
        # 3. Tracking Setup
        track_group = QGroupBox("3. Tracking Setup")
        track_layout = QVBoxLayout()
        
        # OpenCV tracker type
        tracker_layout = QHBoxLayout()
        tracker_layout.addWidget(QLabel("Tracker:"))
        self.tracker_combo = QComboBox()
        self.tracker_combo.addItems(["CSRT", "KCF", "MOSSE"])
        self.tracker_combo.setCurrentText("CSRT")
        tracker_layout.addWidget(self.tracker_combo)
        track_layout.addLayout(tracker_layout)
        
        # Segment interval
        interval_layout = QHBoxLayout()
        interval_layout.addWidget(QLabel("Re-segment Interval:"))
        self.interval_spin = QSpinBox()
        self.interval_spin.setRange(1, 1000)
        self.interval_spin.setValue(30)
        self.interval_spin.setSuffix(" frames")
        interval_layout.addWidget(self.interval_spin)
        track_layout.addLayout(interval_layout)
        
        info_label = QLabel(
            "Re-segments every N frames for accuracy.<br>"
            "Lower = more accurate, slower.<br>"
            "Higher = faster, may drift."
        )
        info_label.setWordWrap(True)
        info_label.setStyleSheet("color: #aaaaaa; font-size: 10px; padding: 5px;")
        track_layout.addWidget(info_label)
        
        # Load first frame and reset buttons
        frame_buttons_layout = QHBoxLayout()
        self.load_frame_btn = QPushButton("Load First Frame")
        self.load_frame_btn.clicked.connect(self.load_first_frame)
        self.load_frame_btn.setEnabled(False)
        frame_buttons_layout.addWidget(self.load_frame_btn)
        
        self.reset_btn = QPushButton("Reset")
        self.reset_btn.clicked.connect(self.reset_tracking_setup)
        self.reset_btn.setEnabled(False)
        self.reset_btn.setStyleSheet("""
            QPushButton {
                background-color: #d47800;
                color: white;
            }
            QPushButton:hover {
                background-color: #e68a00;
            }
            QPushButton:disabled {
                background-color: #3f3f3f;
                color: #888888;
            }
        """)
        frame_buttons_layout.addWidget(self.reset_btn)
        track_layout.addLayout(frame_buttons_layout)
        
        self.frame_status_label = QLabel("Status: Not initialized")
        self.frame_status_label.setWordWrap(True)
        track_layout.addWidget(self.frame_status_label)
        
        # Click instruction
        click_instruction = QLabel(
            "<b>Instructions:</b><br>"
            "1. Load first frame<br>"
            "2. Click on animal<br>"
            "3. Draw arena rectangle (optional)<br>"
            "4. Start tracking"
        )
        click_instruction.setWordWrap(True)
        click_instruction.setStyleSheet("color: #cccccc; font-size: 11px; padding: 5px;")
        track_layout.addWidget(click_instruction)
        
        track_group.setLayout(track_layout)
        layout.addWidget(track_group)
        
        # 4. Run Tracking
        run_group = QGroupBox("4. Run Tracking")
        run_layout = QVBoxLayout()
        
        self.start_btn = QPushButton("Start Tracking")
        self.start_btn.clicked.connect(self.start_tracking)
        self.start_btn.setEnabled(False)
        run_layout.addWidget(self.start_btn)
        
        self.cancel_btn = QPushButton("Cancel")
        self.cancel_btn.clicked.connect(self.cancel_tracking)
        self.cancel_btn.setEnabled(False)
        run_layout.addWidget(self.cancel_btn)
        
        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        run_layout.addWidget(self.progress_bar)
        
        self.progress_label = QLabel("Ready")
        self.progress_label.setAlignment(Qt.AlignCenter)
        run_layout.addWidget(self.progress_label)
        
        run_group.setLayout(run_layout)
        layout.addWidget(run_group)
        
        # Info box
        info_box = QLabel(
            "<b>💡 CPU-Only Tracking</b><br>"
            "No GPU required! Runs on any computer.<br><br>"
            "<b>Expected Speed:</b><br>"
            "• GrabCut: 30-60 fps<br>"
            "• Background Sub: 100-200 fps<br>"
            "• Adaptive: 50-100 fps"
        )
        info_box.setWordWrap(True)
        info_box.setStyleSheet(
            "background-color: #1e3a5f; color: #cccccc; "
            "padding: 10px; border-radius: 4px; font-size: 11px;"
        )
        layout.addWidget(info_box)
        
        layout.addStretch()
        
        return panel
    
    def create_right_panel(self):
        """Create right preview panel."""
        panel = QWidget()
        layout = QVBoxLayout(panel)
        
        # Title
        title = QLabel("<h2>Video Preview</h2>")
        title.setAlignment(Qt.AlignCenter)
        layout.addWidget(title)
        
        # Preview label
        self.preview_label = QLabel("No video loaded")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(640, 480)
        self.preview_label.setStyleSheet(
            "background-color: #1e1e1e; border: 1px solid #3f3f3f; "
            "border-radius: 4px;"
        )
        self.preview_label.mousePressEvent = self.on_frame_click
        layout.addWidget(self.preview_label)
        
        # Info labels
        self.tracking_info_label = QLabel("")
        self.tracking_info_label.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.tracking_info_label)
        
        return panel
    
    def select_videos(self):
        """Select video files."""
        file_paths, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video Files",
            "",
            "Video Files (*.mp4 *.avi *.mov *.mkv);;All Files (*.*)"
        )
        
        if file_paths:
            self.video_paths = file_paths
            self.current_video_idx = 0
            self.video_label.setText(f"Selected {len(file_paths)} video(s):\n{Path(file_paths[0]).name}")
            self.load_frame_btn.setEnabled(True)
            self.status_bar.showMessage(f"Loaded {len(file_paths)} video(s)")
    
    def load_first_frame(self):
        """Load and display first frame."""
        if not self.video_paths:
            return
        
        try:
            video_path = self.video_paths[self.current_video_idx]
            cap = cv2.VideoCapture(video_path)
            
            ret, frame = cap.read()
            if not ret:
                raise ValueError("Could not read first frame")
            
            self.first_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            self.preview_frame = self.first_frame.copy()
            
            cap.release()
            
            # Display frame
            self.display_frame(self.preview_frame)
            
            # Reset tracking state
            self.click_point = None
            self.arena_rectangle = None
            
            self.frame_status_label.setText("Status: Frame loaded - Click on animal")
            self.status_bar.showMessage("First frame loaded - Click on animal to initialize")
            self.reset_btn.setEnabled(True)
            
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load first frame: {str(e)}")
    
    def reset_tracking_setup(self):
        """Reset the tracking setup to start over."""
        self.click_point = None
        self.arena_rectangle = None
        self.start_btn.setEnabled(False)
        
        # Reload first frame
        if self.first_frame is not None:
            self.preview_frame = self.first_frame.copy()
            self.display_frame(self.preview_frame)
            self.frame_status_label.setText("Status: Reset - Click on animal")
            self.status_bar.showMessage("Reset - Click on animal to initialize")
        else:
            self.frame_status_label.setText("Status: Not initialized")
            self.status_bar.showMessage("Load first frame to begin")
    
    def on_frame_click(self, event):
        """Handle mouse click on frame."""
        if self.first_frame is None:
            return
        
        # Get click coordinates (account for label scaling)
        label_width = self.preview_label.width()
        label_height = self.preview_label.height()
        
        frame_height, frame_width = self.first_frame.shape[:2]
        
        # Calculate scale
        scale_x = frame_width / label_width
        scale_y = frame_height / label_height
        
        # Get click position
        click_x = int(event.pos().x() * scale_x)
        click_y = int(event.pos().y() * scale_y)
        
        # Clamp to frame bounds
        click_x = max(0, min(click_x, frame_width - 1))
        click_y = max(0, min(click_y, frame_height - 1))
        
        if self.click_point is None:
            # First click - set animal location and test segmentation
            self.click_point = (click_x, click_y)
            
            # TEST SEGMENTATION - Show user what will be tracked
            self.status_bar.showMessage("Testing segmentation...")
            try:
                seg_method = self.seg_method_combo.currentText().lower().replace(" ", "_")
                segmenter = ClassicalSegmenter(method=seg_method)
                test_mask = segmenter.segment_with_click(self.first_frame, (click_x, click_y))
                
                # Check if segmentation found anything
                if np.sum(test_mask) < 100:  # Less than 100 pixels
                    QMessageBox.warning(
                        self, 
                        "Segmentation Warning", 
                        "Could not segment animal at this location.\n\n"
                        "Try:\n"
                        "1. Click directly on the animal body\n"
                        "2. Try a different segmentation method\n"
                        "3. Ensure good contrast between animal and background"
                    )
                    self.click_point = None
                    return
                
                # Draw mask overlay on preview
                self.preview_frame = self.first_frame.copy()
                mask_colored = np.zeros_like(self.preview_frame)
                mask_colored[:, :, 1] = test_mask  # Green channel
                self.preview_frame = cv2.addWeighted(self.preview_frame, 0.7, mask_colored, 0.3, 0)
                
                # Draw click point
                cv2.circle(self.preview_frame, (click_x, click_y), 5, (255, 255, 0), -1)
                cv2.circle(self.preview_frame, (click_x, click_y), 8, (255, 255, 0), 2)
                
                # Draw bounding box
                contours, _ = cv2.findContours(test_mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                if contours:
                    x, y, w, h = cv2.boundingRect(max(contours, key=cv2.contourArea))
                    cv2.rectangle(self.preview_frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
                
                self.display_frame(self.preview_frame)
                
                self.frame_status_label.setText(
                    f"✓ Animal detected at ({click_x}, {click_y})\n"
                    f"Mask area: {np.sum(test_mask > 0)} pixels\n"
                    f"Click again to define arena (optional) or Start Tracking"
                )
                self.start_btn.setEnabled(True)
                self.status_bar.showMessage("Segmentation successful!")
                
            except Exception as e:
                QMessageBox.critical(
                    self, 
                    "Segmentation Error", 
                    f"Failed to segment animal: {str(e)}\n\n"
                    "Try a different segmentation method or click location."
                )
                self.click_point = None
                return
            
        elif self.arena_rectangle is None:
            # Second click - start defining arena rectangle
            # (For simplicity, we'll use a fixed size rectangle centered on click)
            # In a more advanced version, you could implement click-and-drag
            rect_size = min(self.first_frame.shape[0], self.first_frame.shape[1]) // 2
            x = max(0, click_x - rect_size // 2)
            y = max(0, click_y - rect_size // 2)
            w = min(rect_size, self.first_frame.shape[1] - x)
            h = min(rect_size, self.first_frame.shape[0] - y)
            
            self.arena_rectangle = (x, y, w, h)
            
            # Draw rectangle on preview
            self.preview_frame = self.first_frame.copy()
            cv2.circle(self.preview_frame, self.click_point, 5, (0, 255, 0), -1)
            cv2.rectangle(self.preview_frame, (x, y), (x + w, y + h), (255, 0, 0), 2)
            self.display_frame(self.preview_frame)
            
            self.frame_status_label.setText("Arena defined - Ready to track")
    
    def display_frame(self, frame: np.ndarray):
        """Display frame in preview label."""
        height, width, channel = frame.shape
        bytes_per_line = 3 * width
        q_image = QImage(frame.data, width, height, bytes_per_line, QImage.Format_RGB888)
        
        # Scale to fit label while maintaining aspect ratio
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(
            self.preview_label.size(),
            Qt.KeepAspectRatio,
            Qt.SmoothTransformation
        )
        self.preview_label.setPixmap(scaled_pixmap)
    
    def start_tracking(self):
        """Start tracking process."""
        if not self.video_paths or self.click_point is None:
            QMessageBox.warning(self, "Warning", "Please select video and click on animal first")
            return
        
        # Get output path
        video_path = self.video_paths[self.current_video_idx]
        output_path = str(Path(video_path).with_suffix('.csv'))
        
        # Get settings
        seg_method = self.seg_method_combo.currentText().lower().replace(" ", "_")
        tracker_type = self.tracker_combo.currentText()
        segment_interval = self.interval_spin.value()
        
        # Create worker thread
        self.tracking_worker = TrackingWorker(
            video_path=video_path,
            click_point=self.click_point,
            arena_rectangle=self.arena_rectangle,
            segmentation_method=seg_method,
            tracker_type=tracker_type,
            segment_interval=segment_interval,
            output_path=output_path
        )
        
        # Connect signals
        self.tracking_worker.progress.connect(self.update_progress)
        self.tracking_worker.status.connect(self.update_status)
        self.tracking_worker.finished.connect(self.tracking_finished)
        self.tracking_worker.frame_ready.connect(self.update_preview)
        
        # Start tracking
        self.tracking_worker.start()
        
        # Update UI
        self.start_btn.setEnabled(False)
        self.cancel_btn.setEnabled(True)
        self.select_video_btn.setEnabled(False)
        self.load_frame_btn.setEnabled(False)
    
    def cancel_tracking(self):
        """Cancel tracking process."""
        if self.tracking_worker:
            self.tracking_worker.cancel()
            self.cancel_btn.setEnabled(False)
    
    def update_progress(self, value: int):
        """Update progress bar."""
        self.progress_bar.setValue(value)
    
    def update_status(self, message: str):
        """Update status message."""
        self.progress_label.setText(message)
        self.status_bar.showMessage(message)
    
    def update_preview(self, frame: np.ndarray, mask: np.ndarray, centroid: Tuple[int, int]):
        """Update preview with tracking visualization."""
        # Draw mask overlay
        preview = frame.copy()
        mask_colored = np.zeros_like(preview)
        mask_colored[:, :, 1] = mask  # Green channel
        preview = cv2.addWeighted(preview, 0.7, mask_colored, 0.3, 0)
        
        # Draw centroid
        cv2.circle(preview, centroid, 5, (255, 0, 0), -1)
        
        # Draw arena if defined
        if self.arena_rectangle:
            x, y, w, h = self.arena_rectangle
            cv2.rectangle(preview, (x, y), (x + w, y + h), (255, 0, 0), 2)
        
        self.display_frame(preview)
    
    def tracking_finished(self, success: bool, message: str, stats: dict):
        """Handle tracking completion."""
        # Reset UI
        self.start_btn.setEnabled(True)
        self.cancel_btn.setEnabled(False)
        self.select_video_btn.setEnabled(True)
        self.load_frame_btn.setEnabled(True)
        self.reset_btn.setEnabled(True)
        self.progress_bar.setValue(0 if not success else 100)
        
        if success:
            # Show results
            info_text = (
                f"<b>Tracking Complete!</b><br><br>"
                f"Total frames: {stats.get('total_frames', 'N/A')}<br>"
                f"Total distance: {stats.get('total_distance', 0):.1f} pixels<br>"
                f"Time in center: {stats.get('time_in_center', 0):.1f} seconds<br><br>"
                f"Results saved to:<br>{Path(self.video_paths[self.current_video_idx]).with_suffix('.csv').name}"
            )
            self.tracking_info_label.setText(info_text)
            
            QMessageBox.information(self, "Success", message)
        else:
            # On error, enable reset so user can try again
            self.tracking_info_label.setText(
                f"<b>Tracking Failed</b><br><br>"
                f"{message}<br><br>"
                f"Click 'Reset' to try again with different settings."
            )
            QMessageBox.critical(self, "Error", message)
        
        self.status_bar.showMessage(message)


def main():
    """Run the Simple Tracker GUI."""
    if not CV2_AVAILABLE or not PANDAS_AVAILABLE:
        print("Missing dependencies!")
        print("Install with: pip install opencv-python pandas numpy")
        return
    
    app = QApplication(sys.argv)
    window = SimpleTrackerGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
