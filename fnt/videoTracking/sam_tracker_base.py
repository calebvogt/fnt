#!/usr/bin/env python3
"""
SAM Tracker Base Class

Provides interactive tracking using Meta's Segment Anything Model (SAM) combined
with optical flow for fast frame-to-frame tracking. User clicks on animal once
to initialize, then tracking runs automatically.

Architecture:
1. User clicks on animal -> SAM segments the blob
2. Optical flow tracks blob between frames (fast)
3. Periodic SAM updates refine segmentation (every N frames)
4. Kalman filter predicts position during occlusions
5. Export trajectories to CSV with behavioral metrics

Author: FieldNeuroToolbox Contributors
"""

import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, Tuple, List, Dict
import warnings

# Check for SAM and torch availability
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    torch = None
    warnings.warn("PyTorch not available. Install with: pip install torch")

try:
    from segment_anything import sam_model_registry, SamPredictor
    SAM_AVAILABLE = True
except ImportError:
    SAM_AVAILABLE = False
    warnings.warn("Segment Anything not available. Install with: pip install segment-anything")


class SAMTrackerBase:
    """
    Base class for SAM-based interactive tracking.
    
    Workflow:
    1. Load video and initialize SAM model
    2. User clicks on animal in first frame
    3. SAM generates segmentation mask
    4. Track using optical flow + periodic SAM updates
    5. Handle occlusions with Kalman filter
    6. Export trajectory to CSV
    """
    
    def __init__(
        self,
        video_path: str,
        sam_checkpoint: Optional[str] = None,
        model_type: str = "vit_h",
        device: str = "cuda",
        sam_update_interval: int = 30,
        confidence_threshold: float = 0.5,
        flow_window_size: int = 101,
        tracking_method: str = "optical_flow"
    ):
        """
        Initialize SAM tracker.
        
        Args:
            video_path: Path to input video file
            sam_checkpoint: Path to SAM model checkpoint (.pth file)
                          If None, will attempt to download automatically
            model_type: SAM model type - "vit_h" (huge), "vit_l" (large), or "vit_b" (base)
            device: "cuda" for GPU or "cpu" for CPU
            sam_update_interval: Update SAM segmentation every N frames
            confidence_threshold: Minimum confidence for tracking (0-1)
            flow_window_size: Optical flow window size (must be odd, default 101)
            tracking_method: "optical_flow" (precise but slow) or "centroid" (fast but less precise)
        """
        self.video_path = Path(video_path)
        self.sam_checkpoint = sam_checkpoint
        self.model_type = model_type
        # Check if CUDA is available, fallback to CPU
        if TORCH_AVAILABLE and torch is not None:
            self.device = device if torch.cuda.is_available() else "cpu"
        else:
            self.device = "cpu"
        self.sam_update_interval = sam_update_interval
        self.confidence_threshold = confidence_threshold
        self.flow_window_size = flow_window_size
        self.tracking_method = tracking_method
        
        # Video properties
        self.cap = None
        self.total_frames = 0
        self.fps = 0
        self.width = 0
        self.height = 0
        
        # Tracking state
        self.sam_predictor = None
        self.current_mask = None
        self.current_centroid = None
        self.tracking_initialized = False
        
        # Trajectory storage
        self.trajectory = []  # List of (frame, x, y, confidence) tuples
        
        # Kalman filter for occlusion handling
        self.kalman = None
        self.kalman_initialized = False
        
        # Optical flow parameters
        self.lk_params = dict(
            winSize=(self.flow_window_size, self.flow_window_size),
            maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01)
        )
        self.prev_gray = None
        self.prev_points = None
        
    def initialize_video(self) -> bool:
        """Open video and get properties."""
        self.cap = cv2.VideoCapture(str(self.video_path))
        if not self.cap.isOpened():
            print(f"Error: Could not open video {self.video_path}")
            return False
            
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"Video loaded: {self.width}x{self.height} @ {self.fps:.2f} fps, {self.total_frames} frames")
        return True
        
    def initialize_sam(self) -> bool:
        """Initialize SAM model."""
        if not SAM_AVAILABLE:
            print("Error: Segment Anything not installed")
            return False
            
        try:
            if self.sam_checkpoint is None:
                print("Error: SAM checkpoint path required")
                print("Download from: https://github.com/facebookresearch/segment-anything#model-checkpoints")
                return False
                
            print(f"Loading SAM model ({self.model_type}) on {self.device}...")
            sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
            sam.to(device=self.device)
            self.sam_predictor = SamPredictor(sam)
            print("SAM model loaded successfully")
            return True
            
        except Exception as e:
            print(f"Error loading SAM model: {e}")
            return False
            
    def initialize_kalman_filter(self, initial_x: float, initial_y: float):
        """
        Initialize Kalman filter for position prediction.
        
        State vector: [x, y, vx, vy]
        Measurement: [x, y]
        """
        self.kalman = cv2.KalmanFilter(4, 2)  # 4 state variables, 2 measurements
        self.kalman.measurementMatrix = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ], dtype=np.float32)
        
        self.kalman.transitionMatrix = np.array([
            [1, 0, 1, 0],  # x = x + vx
            [0, 1, 0, 1],  # y = y + vy
            [0, 0, 1, 0],  # vx = vx
            [0, 0, 0, 1]   # vy = vy
        ], dtype=np.float32)
        
        self.kalman.processNoiseCov = np.eye(4, dtype=np.float32) * 0.03
        self.kalman.measurementNoiseCov = np.eye(2, dtype=np.float32) * 0.1
        
        # Set initial state
        self.kalman.statePost = np.array([[initial_x], [initial_y], [0], [0]], dtype=np.float32)
        self.kalman_initialized = True
        
    def segment_with_sam(self, frame: np.ndarray, point: Tuple[int, int]) -> Optional[np.ndarray]:
        """
        Use SAM to segment object at given point.
        
        Args:
            frame: RGB frame
            point: (x, y) click coordinate
            
        Returns:
            Binary mask or None if segmentation failed
        """
        if self.sam_predictor is None:
            return None
            
        try:
            # Set image for SAM
            self.sam_predictor.set_image(frame)
            
            # Predict mask with positive point prompt
            input_point = np.array([[point[0], point[1]]])
            input_label = np.array([1])  # 1 = foreground point
            
            masks, scores, logits = self.sam_predictor.predict(
                point_coords=input_point,
                point_labels=input_label,
                multimask_output=True  # Get multiple mask options
            )
            
            # Choose mask with highest score
            best_idx = np.argmax(scores)
            mask = masks[best_idx]
            
            return mask.astype(np.uint8)
            
        except Exception as e:
            print(f"SAM segmentation error: {e}")
            return None
            
    def get_mask_centroid(self, mask: np.ndarray) -> Optional[Tuple[float, float]]:
        """Calculate centroid of binary mask."""
        moments = cv2.moments(mask)
        if moments["m00"] == 0:
            return None
        cx = moments["m10"] / moments["m00"]
        cy = moments["m01"] / moments["m00"]
        return (cx, cy)
        
    def track_optical_flow(self, frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Track using Lucas-Kanade optical flow.
        
        Args:
            frame: Current grayscale frame
            
        Returns:
            (x, y, confidence) or None if tracking failed
        """
        if self.prev_gray is None or self.prev_points is None:
            return None
            
        # Calculate optical flow
        next_points, status, err = cv2.calcOpticalFlowPyrLK(
            self.prev_gray,
            frame,
            self.prev_points,
            None,
            **self.lk_params
        )
        
        # Keep only good points
        if next_points is not None and status is not None:
            good_new = next_points[status == 1]
            good_old = self.prev_points[status == 1]
            
            if len(good_new) > 0:
                # Use median position of tracked points
                median_x = np.median(good_new[:, 0])
                median_y = np.median(good_new[:, 1])
                confidence = np.mean(status)
                
                # Update points for next frame
                self.prev_points = good_new.reshape(-1, 1, 2)
                
                return (median_x, median_y, confidence)
                
        return None
        
    def track_centroid(self, frame: np.ndarray) -> Optional[Tuple[float, float, float]]:
        """
        Track using simple centroid-based blob detection (faster than optical flow).
        Searches for dark blob near previous position.
        
        Args:
            frame: Current grayscale frame
            
        Returns:
            (x, y, confidence) or None if tracking failed
        """
        if self.current_centroid is None:
            return None
            
        # Define search region around previous position (e.g., 100px radius)
        search_radius = 100
        cx, cy = int(self.current_centroid[0]), int(self.current_centroid[1])
        x1 = max(0, cx - search_radius)
        y1 = max(0, cy - search_radius)
        x2 = min(frame.shape[1], cx + search_radius)
        y2 = min(frame.shape[0], cy + search_radius)
        
        # Extract search region
        search_region = frame[y1:y2, x1:x2]
        
        # Threshold to find dark blob (assumes animal is darker than background)
        # Use Otsu's method for adaptive thresholding
        _, binary = cv2.threshold(search_region, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)
        
        # Find contours
        contours, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        
        if len(contours) == 0:
            return None
            
        # Find largest contour (assumed to be the animal)
        largest_contour = max(contours, key=cv2.contourArea)
        area = cv2.contourArea(largest_contour)
        
        # Calculate centroid of largest contour
        M = cv2.moments(largest_contour)
        if M["m00"] == 0:
            return None
            
        local_cx = M["m10"] / M["m00"]
        local_cy = M["m01"] / M["m00"]
        
        # Convert back to full frame coordinates
        global_x = x1 + local_cx
        global_y = y1 + local_cy
        
        # Confidence based on blob size consistency
        expected_area = self.current_mask.sum() if self.current_mask is not None else 1000
        confidence = min(1.0, area / max(expected_area, 1))
        
        return (global_x, global_y, confidence)
    
    def predict_with_kalman(self) -> Tuple[float, float]:
        """Predict next position using Kalman filter."""
        if not self.kalman_initialized:
            return self.current_centroid if self.current_centroid else (0, 0)
            
        prediction = self.kalman.predict()
        return (prediction[0, 0], prediction[1, 0])
        
    def update_kalman(self, x: float, y: float):
        """Update Kalman filter with new measurement."""
        if not self.kalman_initialized:
            return
            
        measurement = np.array([[x], [y]], dtype=np.float32)
        self.kalman.correct(measurement)
        
    def initialize_tracking(self, frame: np.ndarray, click_point: Tuple[int, int]) -> bool:
        """
        Initialize tracking with user click.
        
        Args:
            frame: First frame (RGB)
            click_point: (x, y) where user clicked on animal
            
        Returns:
            True if initialization successful
        """
        # Segment with SAM
        print(f"Segmenting animal at point {click_point}...")
        mask = self.segment_with_sam(frame, click_point)
        
        if mask is None:
            print("Error: SAM segmentation failed")
            return False
            
        # Get centroid
        centroid = self.get_mask_centroid(mask)
        if centroid is None:
            print("Error: Could not compute centroid")
            return False
            
        self.current_mask = mask
        self.current_centroid = centroid
        
        # Initialize Kalman filter
        self.initialize_kalman_filter(centroid[0], centroid[1])
        
        # Initialize optical flow
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        self.prev_gray = gray
        
        # Extract feature points from mask
        points = cv2.goodFeaturesToTrack(
            gray,
            mask=mask,
            maxCorners=100,
            qualityLevel=0.01,
            minDistance=10
        )
        
        if points is None or len(points) < 5:
            print("Warning: Few feature points detected, tracking may be unstable")
            
        self.prev_points = points
        
        # Record first trajectory point
        self.trajectory.append((0, centroid[0], centroid[1], 1.0))
        
        self.tracking_initialized = True
        print(f"Tracking initialized at ({centroid[0]:.1f}, {centroid[1]:.1f})")
        return True
        
    def process_frame(
        self,
        frame: np.ndarray,
        frame_idx: int,
        force_sam_update: bool = False
    ) -> Optional[Tuple[float, float, float]]:
        """
        Process single frame and return tracked position.
        
        Args:
            frame: Current frame (RGB)
            frame_idx: Frame index
            force_sam_update: Force SAM segmentation update
            
        Returns:
            (x, y, confidence) or None if tracking lost
        """
        if not self.tracking_initialized:
            return None
            
        gray = cv2.cvtColor(frame, cv2.COLOR_RGB2GRAY)
        
        # Decide whether to use SAM or optical flow
        use_sam = (frame_idx % self.sam_update_interval == 0) or force_sam_update
        
        if use_sam and self.current_centroid is not None:
            # Update with SAM
            mask = self.segment_with_sam(frame, (int(self.current_centroid[0]), int(self.current_centroid[1])))
            if mask is not None:
                centroid = self.get_mask_centroid(mask)
                if centroid is not None:
                    self.current_mask = mask
                    self.current_centroid = centroid
                    
                    # Update Kalman filter
                    self.update_kalman(centroid[0], centroid[1])
                    
                    # Update optical flow points (only if using optical flow)
                    if self.tracking_method == "optical_flow":
                        points = cv2.goodFeaturesToTrack(gray, mask=mask, maxCorners=100, qualityLevel=0.01, minDistance=10)
                        if points is not None:
                            self.prev_points = points
                        
                    self.prev_gray = gray
                    return (centroid[0], centroid[1], 1.0)
        else:
            # Track with selected method
            if self.tracking_method == "optical_flow":
                result = self.track_optical_flow(gray)
            elif self.tracking_method == "centroid":
                result = self.track_centroid(gray)
            else:
                print(f"Unknown tracking method: {self.tracking_method}, falling back to optical flow")
                result = self.track_optical_flow(gray)
                
            if result is not None:
                x, y, confidence = result
                self.prev_gray = gray
                
                if confidence >= self.confidence_threshold:
                    self.current_centroid = (x, y)
                    self.update_kalman(x, y)
                    return (x, y, confidence)
        
        # Tracking lost - use Kalman prediction
        pred_x, pred_y = self.predict_with_kalman()
        return (pred_x, pred_y, 0.3)  # Low confidence
        
    def export_trajectory(self, output_path: Optional[str] = None) -> pd.DataFrame:
        """
        Export trajectory to CSV.
        
        Args:
            output_path: Path for CSV output. If None, uses video name.
            
        Returns:
            DataFrame with trajectory data
        """
        if len(self.trajectory) == 0:
            print("Warning: No trajectory data to export")
            return pd.DataFrame()
            
        # Create DataFrame
        df = pd.DataFrame(self.trajectory, columns=['frame', 'x', 'y', 'confidence'])
        
        # Add time column (seconds)
        df['time_s'] = df['frame'] / self.fps if self.fps > 0 else df['frame']
        
        # Add velocity (pixels/second)
        df['vx'] = df['x'].diff() * self.fps
        df['vy'] = df['y'].diff() * self.fps
        df['speed'] = np.sqrt(df['vx']**2 + df['vy']**2)
        
        # Determine output path
        if output_path is None:
            output_path = self.video_path.parent / f"{self.video_path.stem}_trajectory.csv"
        else:
            output_path = Path(output_path)
            
        # Save to CSV
        df.to_csv(output_path, index=False)
        print(f"Trajectory exported to: {output_path}")
        
        return df
        
    def cleanup(self):
        """Release resources."""
        if self.cap is not None:
            self.cap.release()
        cv2.destroyAllWindows()


# Utility functions for trajectory analysis

def calculate_distance_traveled(trajectory: pd.DataFrame) -> float:
    """Calculate total distance traveled in pixels."""
    if len(trajectory) < 2:
        return 0.0
    dx = trajectory['x'].diff()
    dy = trajectory['y'].diff()
    distances = np.sqrt(dx**2 + dy**2)
    return distances.sum()


def calculate_time_in_zone(
    trajectory: pd.DataFrame,
    zone_poly: np.ndarray,
    fps: float
) -> float:
    """
    Calculate time spent in polygonal zone.
    
    Args:
        trajectory: DataFrame with 'x', 'y' columns
        zone_poly: Polygon vertices as Nx2 array
        fps: Frame rate
        
    Returns:
        Time in seconds
    """
    points = trajectory[['x', 'y']].values
    inside = np.array([cv2.pointPolygonTest(zone_poly, tuple(pt), False) >= 0 for pt in points])
    frames_inside = inside.sum()
    return frames_inside / fps


def detect_zone_transitions(
    trajectory: pd.DataFrame,
    zone_poly: np.ndarray
) -> List[int]:
    """
    Detect frames where animal enters/exits zone.
    
    Args:
        trajectory: DataFrame with 'x', 'y' columns
        zone_poly: Polygon vertices
        
    Returns:
        List of frame indices where transitions occur
    """
    points = trajectory[['x', 'y']].values
    inside = np.array([cv2.pointPolygonTest(zone_poly, tuple(pt), False) >= 0 for pt in points])
    
    # Find transitions (changes in inside/outside state)
    transitions = np.where(np.diff(inside.astype(int)) != 0)[0] + 1
    return transitions.tolist()
