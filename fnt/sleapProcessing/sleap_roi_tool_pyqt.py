#!/usr/bin/env python3
"""
SLEAP ROI Analysis Tool

Post-processing tool for SLEAP tracking data with ROI (Region of Interest) analysis.
Analyzes animal position within defined regions and generates occupancy statistics.

Workflow:
1. Select videos + corresponding CSV files (predictions.analysis.csv)
2. Set tracking area (optional) - exclude areas outside arena
3. Draw ROIs (Open Field Test, Light-Dark Box, Custom)
4. Select keypoints for analysis
5. Process batch - generate occupancy data, summary stats, and tracked videos

Features:
- Flexible ROI definition (polygon-based)
- Multiple ROI types (OFT outer/center, LDB light/dark, custom)
- Per-keypoint analysis
- Occupancy timeseries and summary statistics
- Video rendering with ROI overlays

Author: FieldNeuroToolbox Contributors
"""

import sys
import os
import cv2
import numpy as np
import pandas as pd
from pathlib import Path
from typing import Optional, List, Tuple, Dict
import re

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QProgressBar, QMessageBox,
    QGroupBox, QListWidget, QListWidgetItem, QTableWidget, QTableWidgetItem,
    QCheckBox, QScrollArea, QFrame, QComboBox, QInputDialog
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer
from PyQt5.QtGui import QImage, QPixmap, QFont

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


class VideoROIConfig:
    """Configuration for a single video's ROI analysis."""
    
    def __init__(self, video_path: str, csv_path: str):
        self.video_path = video_path
        self.csv_path = csv_path
        self.tracking_area = []  # List of (x, y) polygon points
        self.tracking_area_set = False
        self.rois = []  # List of (roi_name, polygon_points) tuples - ORDER MATTERS for priority
        self.selected_keypoints = []  # List of keypoint names to analyze
        self.configured = False
        self.width = None
        self.height = None
        self.first_frame = None
        self.total_frames = 0
        self.available_keypoints = []  # Extracted from CSV
        
        # Scale bar calibration
        self.scale_bar_set = False
        self.scale_bar_pixels = None  # Distance in pixels
        self.scale_bar_cm = None  # Real-world distance in cm
        self.pixels_per_cm = None  # Conversion factor
    
    def add_roi(self, name: str, polygon: List[Tuple[int, int]]):
        """Add or update an ROI (maintains order if updating)."""
        # Check if ROI already exists
        for i, (roi_name, _) in enumerate(self.rois):
            if roi_name == name:
                self.rois[i] = (name, polygon)
                return
        # Add new ROI
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


class ROIProcessor(QThread):
    """Worker thread for processing videos with ROI analysis."""
    
    progress = pyqtSignal(int, int, int)  # video_idx, frame_idx, total_frames
    status = pyqtSignal(str)
    video_finished = pyqtSignal(int, bool, str)  # video_idx, success, message
    all_finished = pyqtSignal(bool, str)
    
    def __init__(self, video_configs: List[VideoROIConfig], create_video: bool, 
                 show_tracking_area: bool = True, show_rois: bool = True, 
                 show_all_keypoints: bool = True, show_edges: bool = False,
                 show_keypoint_labels: bool = False, interpolate_keypoints: bool = False):
        super().__init__()
        self.video_configs = video_configs
        self.create_video = create_video
        self.show_tracking_area = show_tracking_area
        self.show_rois = show_rois
        self.show_all_keypoints = show_all_keypoints
        self.show_edges = show_edges
        self.show_keypoint_labels = show_keypoint_labels
        self.interpolate_keypoints = interpolate_keypoints
        self.cancelled = False
        self.show_all_keypoints = show_all_keypoints
        self.show_edges = show_edges
        self.cancelled = False
    
    def get_output_path(self, video_path: str, csv_path: str, suffix: str) -> str:
        """
        Generate output path that includes the tracking session identifier.
        
        Example:
            video_path: 'F9039_PreOFT.mp4'
            csv_path: 'F9039_PreOFT.mp4.251106_125759.predictions.analysis.csv'
            suffix: '_roiOccupancy.csv'
            output: 'F9039_PreOFT.mp4.251106_125759_roiOccupancy.csv'
        
        Args:
            video_path: Path to the video file
            csv_path: Path to the tracking CSV file
            suffix: Suffix to append (e.g., '_roiOccupancy.csv')
            
        Returns:
            Output path with tracking session identifier
        """
        # Extract the base video filename (e.g., 'F9039_PreOFT.mp4')
        video_base = os.path.basename(video_path)
        csv_base = os.path.basename(csv_path)
        
        # Check if CSV has the pattern: video_name.DATETIME.predictions.analysis.csv
        if csv_base.startswith(video_base):
            # Extract the datetime part between video name and '.predictions'
            # e.g., from 'F9039_PreOFT.mp4.251106_125759.predictions.analysis.csv'
            # extract '.251106_125759'
            after_video = csv_base[len(video_base):]  # '.251106_125759.predictions.analysis.csv'
            
            if '.predictions' in after_video:
                datetime_part = after_video.split('.predictions')[0]  # '.251106_125759'
                
                # Build output path with datetime included
                video_dir = os.path.dirname(video_path)
                output_name = f"{video_base}{datetime_part}{suffix}"
                return os.path.join(video_dir, output_name)
        
        # Fallback: if pattern doesn't match, use simple replacement
        return video_path.replace('.mp4', suffix)
    
    def run(self):
        """Process all videos."""
        total_videos = len(self.video_configs)
        successful = 0
        failed = 0
        
        for video_idx, config in enumerate(self.video_configs):
            if self.cancelled:
                break
            
            self.status.emit(f"Processing video {video_idx + 1}/{total_videos}: {os.path.basename(config.video_path)}")
            
            try:
                # Load tracking data
                df = pd.read_csv(config.csv_path)
                
                # STEP 1: Filter by tracking area (remove points outside - DELETE them)
                df = self.filter_by_tracking_area(df, config)
                
                # STEP 1.5: Fill in missing frames (frames where SLEAP detected nothing)
                df = self.fill_missing_frames(df)
                
                # STEP 2: Interpolate ALL keypoints (if requested)
                if self.interpolate_keypoints:
                    df = self.interpolate_all_keypoints(df)
                
                # STEP 3: Save tracked coordinates (all keypoints, filtered and interpolated)
                coordinates_df = self.save_tracked_coordinates(df)
                coordinates_path = self.get_output_path(config.video_path, config.csv_path, '_roiPositionCoordinates.csv')
                coordinates_df.to_csv(coordinates_path, index=False)
                
                # Process occupancy
                occupancy_df = self.calculate_occupancy(config, df, video_idx)
                
                # Save occupancy data
                occupancy_path = self.get_output_path(config.video_path, config.csv_path, '_roiOccupancy.csv')
                occupancy_df.to_csv(occupancy_path, index=False)
                
                # Calculate summary statistics
                summary_df = self.calculate_summary(config, occupancy_df, df)
                
                # Save summary
                summary_path = self.get_output_path(config.video_path, config.csv_path, '_roiSummary.csv')
                summary_df.to_csv(summary_path, index=False)
                
                # Create tracked video if requested
                if self.create_video:
                    self.create_tracked_video(config, occupancy_df, df, video_idx)
                
                self.video_finished.emit(video_idx, True, f"Completed: {os.path.basename(config.video_path)}")
                successful += 1
                
            except Exception as e:
                self.video_finished.emit(video_idx, False, f"Failed: {str(e)}")
                failed += 1
        
        summary = f"Batch complete: {successful} successful, {failed} failed"
        self.all_finished.emit(failed == 0, summary)
    
    def calculate_occupancy(self, config: VideoROIConfig, df: pd.DataFrame, video_idx: int) -> pd.DataFrame:
        """
        Calculate which ROI each keypoint is in for each frame.
        Note: Tracking area filtering has already been applied - points outside are NaN.
        """
        occupancy_data = []
        
        total_frames = len(df)
        
        for frame_idx, row in df.iterrows():
            if self.cancelled:
                break
            
            # Emit progress update (every 10 frames to reduce overhead)
            if frame_idx % 10 == 0:
                self.progress.emit(video_idx, frame_idx, total_frames)
            
            frame_data = {'frame': row.get('frame_idx', frame_idx)}
            
            # For each selected keypoint
            for keypoint in config.selected_keypoints:
                x_col = f"{keypoint}.x"
                y_col = f"{keypoint}.y"
                
                if x_col not in df.columns or y_col not in df.columns:
                    continue
                
                x = row[x_col]
                y = row[y_col]
                
                # Check if point is valid (NaN means missing or outside tracking area)
                if pd.isna(x) or pd.isna(y):
                    frame_data[f"{keypoint}_region"] = "none"
                    continue
                
                # Check which ROI the point is in (priority order - first match wins)
                roi_found = False
                for roi_name, roi_polygon in config.rois:
                    if self.point_in_polygon((x, y), roi_polygon):
                        frame_data[f"{keypoint}_region"] = roi_name
                        roi_found = True
                        break
                
                if not roi_found:
                    frame_data[f"{keypoint}_region"] = "none"
            
            occupancy_data.append(frame_data)
        
        return pd.DataFrame(occupancy_data)
    
    def point_in_polygon(self, point: Tuple[float, float], polygon: List[Tuple[int, int]]) -> bool:
        """Check if point is inside polygon using cv2.pointPolygonTest."""
        if not polygon or len(polygon) < 3:
            return False
        
        polygon_array = np.array(polygon, dtype=np.int32)
        result = cv2.pointPolygonTest(polygon_array, point, False)
        return result >= 0
    
    def filter_by_tracking_area(self, df: pd.DataFrame, config: VideoROIConfig) -> pd.DataFrame:
        """
        Filter tracking data to remove keypoints outside the tracking area.
        This is the FIRST step - points outside are deleted (set to NaN).
        
        Args:
            df: DataFrame with tracking data
            config: Video configuration with tracking area
            
        Returns:
            DataFrame with points outside tracking area removed (set to NaN)
        """
        if not config.tracking_area_set or not config.tracking_area:
            # No tracking area defined, return as-is
            return df
        
        df = df.copy()
        
        # Get all keypoints from CSV
        all_keypoints = []
        for col in df.columns:
            if col.endswith('.x') and not col.startswith('track'):
                keypoint_name = col[:-2]  # Remove '.x'
                all_keypoints.append(keypoint_name)
        
        points_filtered = 0
        
        # For each keypoint, check if it's inside tracking area
        for keypoint in all_keypoints:
            x_col = f"{keypoint}.x"
            y_col = f"{keypoint}.y"
            
            if x_col not in df.columns or y_col not in df.columns:
                continue
            
            for frame_idx, row in df.iterrows():
                x = row[x_col]
                y = row[y_col]
                
                # If point is valid but outside tracking area, remove it
                if not pd.isna(x) and not pd.isna(y):
                    if not self.point_in_polygon((x, y), config.tracking_area):
                        df.at[frame_idx, x_col] = np.nan
                        df.at[frame_idx, y_col] = np.nan
                        points_filtered += 1
        
        if points_filtered > 0:
            self.status.emit(f"Filtered {points_filtered} points outside tracking area")
        
        return df
    
    def fill_missing_frames(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Fill in missing frames that SLEAP skipped (when no keypoints were detected).
        Creates rows with NaN values for missing frame_idx values.
        This MUST happen before interpolation so interpolation can fill the gaps.
        
        Args:
            df: DataFrame with tracking data (potentially with missing frames)
            
        Returns:
            DataFrame with all frames present (missing frames filled with NaN)
        """
        # Check if we have frame_idx column
        if 'frame_idx' not in df.columns:
            # No frame_idx, assume sequential - nothing to do
            return df
        
        # Get the range of frames
        min_frame = int(df['frame_idx'].min())
        max_frame = int(df['frame_idx'].max())
        expected_frames = max_frame - min_frame + 1
        actual_frames = len(df)
        
        missing_count = expected_frames - actual_frames
        
        if missing_count == 0:
            # No missing frames
            return df
        
        self.status.emit(f"Found {missing_count} missing frames - adding them for interpolation")
        
        # Create a complete frame sequence
        all_frames = pd.DataFrame({'frame_idx': range(min_frame, max_frame + 1)})
        
        # Merge with existing data (missing frames will have NaN for all keypoint columns)
        df_complete = all_frames.merge(df, on='frame_idx', how='left')
        
        # Sort by frame_idx to ensure proper order
        df_complete = df_complete.sort_values('frame_idx').reset_index(drop=True)
        
        return df_complete
    
    def interpolate_all_keypoints(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Interpolate missing coordinates for ALL keypoints (forward-only after first detection).
        This is the SECOND step - applied after tracking area filtering.
        
        Args:
            df: DataFrame with tracking data (already filtered by tracking area)
            
        Returns:
            DataFrame with interpolated values for all keypoints
        """
        df = df.copy()
        
        # Get all keypoints from CSV
        all_keypoints = []
        for col in df.columns:
            if col.endswith('.x') and not col.startswith('track'):
                keypoint_name = col[:-2]  # Remove '.x'
                all_keypoints.append(keypoint_name)
        
        # Track interpolation stats for debugging
        total_interpolated = 0
        
        # For each keypoint, interpolate x and y coordinates
        for kp in all_keypoints:
            x_col = f"{kp}.x"
            y_col = f"{kp}.y"
            
            if x_col in df.columns and y_col in df.columns:
                # Count NaNs before interpolation
                x_nans_before = df[x_col].isna().sum()
                y_nans_before = df[y_col].isna().sum()
                
                # Forward-only interpolation: only starts after first valid value
                df[x_col] = df[x_col].interpolate(method='linear', limit_direction='forward')
                df[y_col] = df[y_col].interpolate(method='linear', limit_direction='forward')
                
                # Count NaNs after interpolation
                x_nans_after = df[x_col].isna().sum()
                y_nans_after = df[y_col].isna().sum()
                
                x_filled = x_nans_before - x_nans_after
                y_filled = y_nans_before - y_nans_after
                total_interpolated += x_filled + y_filled
        
        if total_interpolated > 0:
            self.status.emit(f"Interpolated {total_interpolated} coordinate values across all keypoints")
        else:
            self.status.emit("No missing values to interpolate")
        
        return df
    
    def save_tracked_coordinates(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Save all keypoint coordinates (after filtering and interpolation).
        
        Args:
            df: Tracking data (filtered and interpolated)
            
        Returns:
            DataFrame with frame and all keypoint coordinates
        """
        # Get all keypoints from CSV
        all_keypoints = []
        for col in df.columns:
            if col.endswith('.x') and not col.startswith('track'):
                keypoint_name = col[:-2]  # Remove '.x'
                all_keypoints.append(keypoint_name)
        
        # Build output dataframe with frame and all keypoint coordinates
        output_data = []
        
        for frame_idx, row in df.iterrows():
            frame_data = {'frame': row.get('frame_idx', frame_idx)}
            
            # Add all keypoint coordinates
            for keypoint in all_keypoints:
                x_col = f"{keypoint}.x"
                y_col = f"{keypoint}.y"
                
                if x_col in df.columns and y_col in df.columns:
                    frame_data[x_col] = row[x_col]
                    frame_data[y_col] = row[y_col]
            
            output_data.append(frame_data)
        
        return pd.DataFrame(output_data)
    
    def calculate_summary(self, config: VideoROIConfig, occupancy_df: pd.DataFrame, tracking_df: pd.DataFrame) -> pd.DataFrame:
        """Calculate summary statistics for each keypoint and ROI."""
        summary_data = []
        
        # Assume FPS (or extract from video)
        cap = cv2.VideoCapture(config.video_path)
        fps = cap.get(cv2.CAP_PROP_FPS) if cap.isOpened() else 30
        cap.release()
        
        # Total video frames and time
        total_frames = len(occupancy_df)
        total_time_seconds = total_frames / fps
        
        for keypoint in config.selected_keypoints:
            region_col = f"{keypoint}_region"
            if region_col not in occupancy_df.columns:
                continue
            
            # Time in each ROI
            for roi_name, _ in config.rois:
                frames_in_roi = (occupancy_df[region_col] == roi_name).sum()
                time_in_roi = frames_in_roi / fps
                
                summary_data.append({
                    'keypoint': keypoint,
                    'roi': roi_name,
                    'frames': frames_in_roi,
                    'time_seconds': time_in_roi,
                    'percent_time': (time_in_roi / total_time_seconds * 100) if total_time_seconds > 0 else 0
                })
            
            # Calculate distance traveled
            x_col = f"{keypoint}.x"
            y_col = f"{keypoint}.y"
            
            if x_col in tracking_df.columns and y_col in tracking_df.columns:
                coords = tracking_df[[x_col, y_col]].dropna()
                if len(coords) > 1:
                    distances = np.sqrt(np.sum(np.diff(coords.values, axis=0)**2, axis=1))
                    total_distance_pixels = np.sum(distances)
                    
                    # Convert to cm if scale bar is set
                    if config.scale_bar_set and config.pixels_per_cm:
                        total_distance_cm = total_distance_pixels / config.pixels_per_cm
                        avg_velocity_cm_s = total_distance_cm / (len(tracking_df) / fps) if len(tracking_df) > 0 else 0
                        
                        summary_data.append({
                            'keypoint': keypoint,
                            'roi': 'ALL',
                            'total_distance_pixels': total_distance_pixels,
                            'total_distance_cm': total_distance_cm,
                            'avg_velocity_pixels_s': total_distance_pixels / (len(tracking_df) / fps) if len(tracking_df) > 0 else 0,
                            'avg_velocity_cm_s': avg_velocity_cm_s
                        })
                    else:
                        # No scale bar - only pixels
                        avg_velocity_pixels = total_distance_pixels / (len(tracking_df) / fps) if len(tracking_df) > 0 else 0
                        
                        summary_data.append({
                            'keypoint': keypoint,
                            'roi': 'ALL',
                            'total_distance_pixels': total_distance_pixels,
                            'avg_velocity_pixels_s': avg_velocity_pixels
                        })
        
        return pd.DataFrame(summary_data)
    
    def detect_skeleton_edges(self, df: pd.DataFrame, keypoints: List[str], 
                              max_distance_pixels: float = 150.0, 
                              sample_frames: int = 10) -> List[Tuple[str, str]]:
        """
        Automatically detect skeleton edges based on proximity between keypoints.
        
        Args:
            df: DataFrame with tracking data
            keypoints: List of keypoint names
            max_distance_pixels: Maximum distance to consider keypoints connected
            sample_frames: Number of frames to sample for building skeleton
            
        Returns:
            List of (keypoint1, keypoint2) tuples representing edges
        """
        # Sample multiple frames to build robust skeleton
        sample_indices = np.linspace(0, len(df) - 1, min(sample_frames, len(df)), dtype=int)
        
        # Calculate average distances between all keypoint pairs
        keypoint_distances = {}
        
        for i, kp1 in enumerate(keypoints):
            for kp2 in keypoints[i+1:]:  # Only upper triangle to avoid duplicates
                distances = []
                
                for idx in sample_indices:
                    row = df.iloc[idx]
                    
                    x1, y1 = row.get(f"{kp1}.x"), row.get(f"{kp1}.y")
                    x2, y2 = row.get(f"{kp2}.x"), row.get(f"{kp2}.y")
                    
                    # Skip if any coordinate is NaN
                    if pd.isna(x1) or pd.isna(y1) or pd.isna(x2) or pd.isna(y2):
                        continue
                    
                    # Calculate Euclidean distance
                    dist = np.sqrt((x2 - x1)**2 + (y2 - y1)**2)
                    distances.append(dist)
                
                # Store median distance (robust to outliers)
                if distances:
                    keypoint_distances[(kp1, kp2)] = np.median(distances)
        
        # Sort pairs by distance and select closest neighbors
        sorted_pairs = sorted(keypoint_distances.items(), key=lambda x: x[1])
        
        # Build skeleton using minimum spanning tree approach
        # Each keypoint should connect to its nearest neighbors
        edges = []
        connected_keypoints = set()
        
        # First pass: add edges under threshold, ensuring connectivity
        for (kp1, kp2), dist in sorted_pairs:
            if dist <= max_distance_pixels:
                edges.append((kp1, kp2))
                connected_keypoints.add(kp1)
                connected_keypoints.add(kp2)
        
        # Second pass: ensure all keypoints are connected (add closest edge for isolated points)
        for kp in keypoints:
            if kp not in connected_keypoints:
                # Find closest keypoint
                closest_kp = None
                min_dist = float('inf')
                
                for (kp1, kp2), dist in keypoint_distances.items():
                    if kp1 == kp and kp2 in connected_keypoints:
                        if dist < min_dist:
                            min_dist = dist
                            closest_kp = kp2
                    elif kp2 == kp and kp1 in connected_keypoints:
                        if dist < min_dist:
                            min_dist = dist
                            closest_kp = kp1
                
                if closest_kp:
                    edges.append((kp, closest_kp))
                    connected_keypoints.add(kp)
        
        return edges
    
    def create_tracked_video(self, config: VideoROIConfig, occupancy_df: pd.DataFrame, df: pd.DataFrame, video_idx: int):
        """Create video with ROI overlays and tracked keypoints.
        
        Args:
            config: Video configuration
            occupancy_df: ROI occupancy results
            df: Tracking data (potentially interpolated)
            video_idx: Index of current video being processed
        """
        self.status.emit(f"Rendering tracked video...")
        
        # Open video
        cap = cv2.VideoCapture(config.video_path)
        if not cap.isOpened():
            return
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_video_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Output path with tracking session identifier
        output_path = self.get_output_path(config.video_path, config.csv_path, '_roiTracked.mp4')
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        out = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
        
        # Use the passed tracking data (which may be interpolated)
        # df is already loaded and potentially interpolated in run()
        
        # Get all available keypoints from CSV
        all_keypoints = []
        for col in df.columns:
            if col.endswith('.x') and not col.startswith('track'):
                keypoint_name = col[:-2]  # Remove '.x'
                all_keypoints.append(keypoint_name)
        
        # Automatically detect skeleton edges based on proximity in first valid frame
        skeleton_edges = self.detect_skeleton_edges(df, all_keypoints)
        
        # Check if CSV has frame_idx column, otherwise use index
        if 'frame_idx' in df.columns:
            # Create a lookup dictionary: frame_number -> row_index
            # Use enumerate to get actual position, not pandas index
            frame_to_row = {int(row['frame_idx']): idx for idx, row in enumerate(df.to_dict('records'))}
        elif 'frame' in df.columns:
            frame_to_row = {int(row['frame']): idx for idx, row in enumerate(df.to_dict('records'))}
        else:
            # Assume CSV rows match video frames sequentially
            frame_to_row = {idx: idx for idx in range(len(df))}
        
        video_frame_number = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Emit progress update
            if video_frame_number % 10 == 0:  # Update every 10 frames to avoid flooding
                self.progress.emit(video_idx, video_frame_number, total_video_frames)
            
            # Draw ROIs (if enabled)
            if self.show_rois:
                for roi_name, roi_polygon in config.rois:
                    if 'center' in roi_name.lower() or 'inner' in roi_name.lower():
                        color = (0, 165, 255)  # Orange
                    elif 'light' in roi_name.lower():
                        color = (0, 255, 0)  # Green
                    elif 'dark' in roi_name.lower():
                        color = (0, 255, 255)  # Yellow
                    else:
                        color = (0, 255, 0)  # Green default
                    
                    pts = np.array(roi_polygon, dtype=np.int32)
                    cv2.polylines(frame, [pts], True, color, 2)
                    
                    # Add label
                    if len(roi_polygon) > 0:
                        cv2.putText(frame, roi_name, tuple(roi_polygon[0]), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw tracking area (if enabled)
            if self.show_tracking_area and config.tracking_area_set and config.tracking_area:
                pts = np.array(config.tracking_area, dtype=np.int32)
                cv2.polylines(frame, [pts], True, (255, 0, 255), 2)  # Magenta
            
            # Draw keypoints - look up the correct row based on frame number
            if video_frame_number in frame_to_row:
                row_idx = frame_to_row[video_frame_number]
                row = df.iloc[row_idx]
                
                # Determine which keypoints to display
                if self.show_all_keypoints:
                    keypoints_to_show = all_keypoints
                else:
                    keypoints_to_show = config.selected_keypoints
                
                # Store keypoint positions for edge drawing
                keypoint_positions = {}
                
                # First pass: collect keypoint positions
                # Note: Points outside tracking area are already NaN from filtering step
                for keypoint in keypoints_to_show:
                    x_col = f"{keypoint}.x"
                    y_col = f"{keypoint}.y"
                    
                    if x_col in df.columns and y_col in df.columns:
                        x = row[x_col]
                        y = row[y_col]
                        
                        # If valid (not NaN), include it
                        if not pd.isna(x) and not pd.isna(y):
                            keypoint_positions[keypoint] = (int(x), int(y))
                
                # Draw skeleton edges FIRST (so they appear below keypoints)
                if self.show_edges:
                    for kp1, kp2 in skeleton_edges:
                        if kp1 in keypoint_positions and kp2 in keypoint_positions:
                            pt1 = keypoint_positions[kp1]
                            pt2 = keypoint_positions[kp2]
                            cv2.line(frame, pt1, pt2, (0, 255, 255), 2)  # Yellow lines
                
                # Draw keypoints SECOND (so they appear above edges)
                for keypoint in keypoints_to_show:
                    if keypoint in keypoint_positions:
                        x, y = keypoint_positions[keypoint]
                        
                        # Color: blue for selected keypoints, red for others
                        if keypoint in config.selected_keypoints:
                            color = (255, 0, 0)  # Blue for selected
                        else:
                            color = (0, 0, 255)  # Red for unselected
                        
                        cv2.circle(frame, (x, y), 5, color, -1)
                        
                        # Draw label if enabled
                        if self.show_keypoint_labels:
                            cv2.putText(frame, keypoint, (x + 10, y), 
                                       cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
            
            out.write(frame)
            video_frame_number += 1
        
        cap.release()
        out.release()
    
    def cancel(self):
        """Cancel processing."""
        self.cancelled = True


class ROIToolGUI(QMainWindow):
    """Main GUI for SLEAP ROI Analysis Tool."""
    
    def __init__(self):
        super().__init__()
        self.setWindowTitle("SLEAP ROI Analysis Tool - FieldNeuroToolbox")
        self.setGeometry(100, 100, 1400, 900)
        
        # State
        self.video_configs = []
        self.current_config_idx = 0
        self.preview_frame = None
        self.preview_cap = None
        
        # Drawing state
        self.drawing_mode = None  # 'tracking_area', 'oft', 'ldb_light', 'ldb_dark', 'custom', 'scale_bar'
        self.current_polygon = []
        self.oft_outer_polygon = []
        
        # Scale bar state
        self.scale_bar_points = []  # Two points for scale bar
        
        # Worker thread
        self.processor = None
        
        self.init_ui()
    
    def init_ui(self):
        """Initialize the user interface."""
        
        # Apply dark theme
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #cccccc;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 3px;
                font-weight: bold;
                font-size: 11px;
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
            QGroupBox {
                font-weight: bold;
                border: 2px solid #3f3f3f;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 15px;
                font-size: 13px;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 10px;
                padding: 0 5px;
            }
            QLabel {
                color: #cccccc;
                font-size: 11px;
            }
            QListWidget {
                background-color: #1e1e1e;
                border: 1px solid #3f3f3f;
                border-radius: 3px;
                padding: 5px;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
            }
            QTableWidget {
                background-color: #1e1e1e;
                border: 1px solid #3f3f3f;
                border-radius: 3px;
            }
            QTableWidget::item:selected {
                background-color: #0078d4;
            }
            QCheckBox {
                color: #cccccc;
                spacing: 8px;
            }
            QProgressBar {
                border: 1px solid #3f3f3f;
                border-radius: 3px;
                text-align: center;
                background-color: #1e1e1e;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
            }
        """)
        
        # Main layout
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout()
        central_widget.setLayout(main_layout)
        
        # Left panel (controls)
        left_panel = self.create_left_panel()
        main_layout.addWidget(left_panel, stretch=2)
        
        # Right panel (preview)
        right_panel = self.create_right_panel()
        main_layout.addWidget(right_panel, stretch=3)
    
    def create_left_panel(self):
        """Create left control panel."""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Title
        title = QLabel("SLEAP ROI Analysis Tool")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setStyleSheet("color: #0078d4; margin: 10px;")
        layout.addWidget(title)
        
        # Scroll area for all sections
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll_widget = QWidget()
        scroll_layout = QVBoxLayout()
        scroll_widget.setLayout(scroll_layout)
        
        # Section 1: Video + CSV Selection
        scroll_layout.addWidget(self.create_video_selection_section())
        
        # Section 2: Set Tracking Area
        scroll_layout.addWidget(self.create_tracking_area_section())
        
        # Section 3: Draw ROIs
        scroll_layout.addWidget(self.create_roi_section())
        
        # Section 4: Select Keypoints
        scroll_layout.addWidget(self.create_keypoint_section())
        
        # Section 5: Analysis Options
        scroll_layout.addWidget(self.create_analysis_options_section())
        
        # Section 6: Processing
        scroll_layout.addWidget(self.create_processing_section())
        
        scroll_layout.addStretch()
        scroll.setWidget(scroll_widget)
        layout.addWidget(scroll)
        
        return panel
    
    def create_video_selection_section(self):
        """Create video selection section."""
        group = QGroupBox("1. Video + CSV Selection")
        layout = QVBoxLayout()
        
        # Buttons
        btn_layout = QHBoxLayout()
        self.btn_add_folder = QPushButton("📁 Add Folder")
        self.btn_add_folder.clicked.connect(self.add_folder)
        btn_layout.addWidget(self.btn_add_folder)
        
        self.btn_add_videos = QPushButton("🎬 Add Video(s)")
        self.btn_add_videos.clicked.connect(self.add_videos)
        btn_layout.addWidget(self.btn_add_videos)
        
        self.btn_clear_videos = QPushButton("🗑️ Clear All")
        self.btn_clear_videos.clicked.connect(self.clear_videos)
        btn_layout.addWidget(self.btn_clear_videos)
        layout.addLayout(btn_layout)
        
        # Video list with arrow buttons
        list_layout = QHBoxLayout()
        
        self.video_list = QListWidget()
        self.video_list.currentRowChanged.connect(self.on_video_selected)
        list_layout.addWidget(self.video_list)
        
        # Arrow buttons for navigation
        arrow_layout = QVBoxLayout()
        arrow_layout.addStretch()
        
        self.btn_prev = QPushButton("▲")
        self.btn_prev.setMaximumWidth(40)
        self.btn_prev.clicked.connect(self.previous_video)
        self.btn_prev.setEnabled(False)
        self.btn_prev.setToolTip("Previous video")
        arrow_layout.addWidget(self.btn_prev)
        
        self.btn_next = QPushButton("▼")
        self.btn_next.setMaximumWidth(40)
        self.btn_next.clicked.connect(self.next_video)
        self.btn_next.setEnabled(False)
        self.btn_next.setToolTip("Next video")
        arrow_layout.addWidget(self.btn_next)
        
        arrow_layout.addStretch()
        list_layout.addLayout(arrow_layout)
        
        layout.addLayout(list_layout)
        
        group.setLayout(layout)
        return group
    
    def create_tracking_area_section(self):
        """Create tracking area section."""
        group = QGroupBox("2. Set Tracking Area")
        layout = QVBoxLayout()
        
        label = QLabel("Define arena boundary (optional)")
        label.setStyleSheet("color: #999999; font-style: italic;")
        layout.addWidget(label)
        
        self.btn_set_tracking_area = QPushButton("✏️ Draw Tracking Area")
        self.btn_set_tracking_area.clicked.connect(self.start_tracking_area_drawing)
        self.btn_set_tracking_area.setEnabled(False)
        layout.addWidget(self.btn_set_tracking_area)
        
        btn_row = QHBoxLayout()
        self.btn_clear_tracking_area = QPushButton("Clear")
        self.btn_clear_tracking_area.clicked.connect(self.clear_tracking_area)
        self.btn_clear_tracking_area.setEnabled(False)
        btn_row.addWidget(self.btn_clear_tracking_area)
        
        self.btn_skip_tracking_area = QPushButton("Skip")
        self.btn_skip_tracking_area.clicked.connect(self.skip_tracking_area)
        self.btn_skip_tracking_area.setEnabled(False)
        btn_row.addWidget(self.btn_skip_tracking_area)
        layout.addLayout(btn_row)
        
        self.lbl_tracking_status = QLabel("")
        layout.addWidget(self.lbl_tracking_status)
        
        group.setLayout(layout)
        return group
    
    def create_roi_section(self):
        """Create ROI drawing section."""
        group = QGroupBox("3. Draw ROIs")
        layout = QVBoxLayout()
        
        self.btn_oft = QPushButton("🔲 Open Field Test Layout")
        self.btn_oft.clicked.connect(self.start_oft_drawing)
        self.btn_oft.setEnabled(False)
        layout.addWidget(self.btn_oft)
        
        self.btn_ldb = QPushButton("◐ Light-Dark Box Layout")
        self.btn_ldb.clicked.connect(self.start_ldb_drawing)
        self.btn_ldb.setEnabled(False)
        layout.addWidget(self.btn_ldb)
        
        self.btn_custom = QPushButton("✏️ Custom ROI")
        self.btn_custom.clicked.connect(self.start_custom_roi_drawing)
        self.btn_custom.setEnabled(False)
        layout.addWidget(self.btn_custom)
        
        # Scale bar button
        self.btn_scale_bar = QPushButton("📏 Set Scale Bar (optional)")
        self.btn_scale_bar.clicked.connect(self.start_scale_bar_setting)
        self.btn_scale_bar.setEnabled(False)
        layout.addWidget(self.btn_scale_bar)
        
        self.lbl_scale_status = QLabel("")
        self.lbl_scale_status.setStyleSheet("color: #4CAF50; font-style: italic; font-size: 10px;")
        layout.addWidget(self.lbl_scale_status)
        
        # ROI table with priority ordering
        priority_label = QLabel("ROI Priority (select row and use arrows, top = highest priority):")
        priority_label.setStyleSheet("color: #999999; font-style: italic; font-size: 10px; margin-top: 5px;")
        layout.addWidget(priority_label)
        
        # Table and arrow buttons in horizontal layout
        table_layout = QHBoxLayout()
        
        self.roi_table = QTableWidget()
        self.roi_table.setColumnCount(1)
        self.roi_table.setHorizontalHeaderLabels(['ROI Name'])
        self.roi_table.horizontalHeader().setStretchLastSection(True)
        self.roi_table.setMinimumHeight(200)
        self.roi_table.setMaximumHeight(300)
        self.roi_table.setSelectionBehavior(QTableWidget.SelectRows)
        self.roi_table.itemChanged.connect(self.on_roi_renamed)
        self.roi_table.itemSelectionChanged.connect(self.on_roi_selection_changed)
        table_layout.addWidget(self.roi_table)
        
        # Arrow buttons for reordering
        arrow_layout = QVBoxLayout()
        arrow_layout.addStretch()
        
        self.btn_move_up = QPushButton("▲")
        self.btn_move_up.setMaximumWidth(40)
        self.btn_move_up.setEnabled(False)
        self.btn_move_up.clicked.connect(self.move_roi_up)
        self.btn_move_up.setToolTip("Move ROI up (higher priority)")
        arrow_layout.addWidget(self.btn_move_up)
        
        self.btn_move_down = QPushButton("▼")
        self.btn_move_down.setMaximumWidth(40)
        self.btn_move_down.setEnabled(False)
        self.btn_move_down.clicked.connect(self.move_roi_down)
        self.btn_move_down.setToolTip("Move ROI down (lower priority)")
        arrow_layout.addWidget(self.btn_move_down)
        
        arrow_layout.addStretch()
        table_layout.addLayout(arrow_layout)
        
        layout.addLayout(table_layout)
        
        self.btn_skip_roi = QPushButton("Skip ROIs")
        self.btn_skip_roi.clicked.connect(self.skip_rois)
        self.btn_skip_roi.setEnabled(False)
        layout.addWidget(self.btn_skip_roi)
        
        group.setLayout(layout)
        return group
    
    def create_keypoint_section(self):
        """Create keypoint selection section."""
        group = QGroupBox("4. Select Keypoints for Analysis")
        layout = QVBoxLayout()
        
        label = QLabel("Select which keypoints to track:")
        layout.addWidget(label)
        
        # Keypoint checkboxes (dynamically populated)
        self.keypoint_checkboxes = []
        self.keypoint_layout = QVBoxLayout()
        layout.addLayout(self.keypoint_layout)
        
        self.lbl_config_status = QLabel("")
        self.lbl_config_status.setStyleSheet("color: #4caf50; font-weight: bold;")
        layout.addWidget(self.lbl_config_status)
        
        group.setLayout(layout)
        return group
    
    def create_analysis_options_section(self):
        """Create analysis options section."""
        group = QGroupBox("5. Analysis Options")
        layout = QVBoxLayout()
        
        # Interpolation checkbox
        self.chk_interpolate = QCheckBox("Interpolate missing keypoints")
        self.chk_interpolate.setChecked(True)
        layout.addWidget(self.chk_interpolate)
        
        # Video creation checkbox
        self.chk_create_video = QCheckBox("Create tracked video")
        self.chk_create_video.setChecked(True)
        self.chk_create_video.stateChanged.connect(self.on_create_video_toggled)
        layout.addWidget(self.chk_create_video)
        
        # Video display options (indented, only enabled if create_video is checked)
        self.video_options_widget = QWidget()
        video_options_layout = QVBoxLayout()
        video_options_layout.setContentsMargins(20, 0, 0, 0)  # Indent
        
        video_label = QLabel("Video Display Options:")
        video_label.setStyleSheet("color: #999999; font-style: italic; font-size: 10px;")
        video_options_layout.addWidget(video_label)
        
        self.chk_show_tracking_area = QCheckBox("Show tracking area boundary")
        self.chk_show_tracking_area.setChecked(True)
        video_options_layout.addWidget(self.chk_show_tracking_area)
        
        self.chk_show_rois = QCheckBox("Show ROI boundaries and labels")
        self.chk_show_rois.setChecked(True)
        video_options_layout.addWidget(self.chk_show_rois)
        
        self.chk_show_all_keypoints = QCheckBox("Show all keypoints (selected keypoints in blue)")
        self.chk_show_all_keypoints.setChecked(True)
        video_options_layout.addWidget(self.chk_show_all_keypoints)
        
        self.chk_show_keypoint_labels = QCheckBox("Show keypoint labels")
        self.chk_show_keypoint_labels.setChecked(False)
        video_options_layout.addWidget(self.chk_show_keypoint_labels)
        
        self.chk_show_edges = QCheckBox("Draw skeleton edges between keypoints")
        self.chk_show_edges.setChecked(False)
        video_options_layout.addWidget(self.chk_show_edges)
        
        self.chk_data_view = QCheckBox("Data view (video + live statistics panel)")
        self.chk_data_view.setChecked(False)
        video_options_layout.addWidget(self.chk_data_view)
        
        self.video_options_widget.setLayout(video_options_layout)
        layout.addWidget(self.video_options_widget)
        
        group.setLayout(layout)
        return group
    
    def create_processing_section(self):
        """Create processing section."""
        group = QGroupBox("6. Start Processing")
        layout = QVBoxLayout()
        
        self.chk_overwrite = QCheckBox("Overwrite existing output files")
        self.chk_overwrite.setChecked(True)
        layout.addWidget(self.chk_overwrite)
        
        btn_layout = QHBoxLayout()
        self.btn_process = QPushButton("▶️ Start Batch Processing")
        self.btn_process.clicked.connect(self.start_processing)
        self.btn_process.setEnabled(False)
        btn_layout.addWidget(self.btn_process)
        
        self.btn_cancel = QPushButton("⏹️ Cancel")
        self.btn_cancel.clicked.connect(self.cancel_processing)
        self.btn_cancel.setEnabled(False)
        btn_layout.addWidget(self.btn_cancel)
        layout.addLayout(btn_layout)
        
        self.progress_bar = QProgressBar()
        layout.addWidget(self.progress_bar)
        
        self.lbl_status = QLabel("Ready")
        layout.addWidget(self.lbl_status)
        
        group.setLayout(layout)
        return group
    
    def create_right_panel(self):
        """Create right preview panel."""
        panel = QWidget()
        layout = QVBoxLayout()
        panel.setLayout(layout)
        
        # Preview label
        preview_label = QLabel("Video Preview")
        preview_label.setFont(QFont("Arial", 12, QFont.Bold))
        preview_label.setStyleSheet("color: #0078d4;")
        layout.addWidget(preview_label)
        
        # Preview frame
        self.preview_label = QLabel()
        self.preview_label.setStyleSheet("border: 2px solid #3f3f3f; background-color: #1e1e1e;")
        self.preview_label.setAlignment(Qt.AlignCenter)
        self.preview_label.setMinimumSize(800, 600)
        self.preview_label.mousePressEvent = self.on_preview_click
        layout.addWidget(self.preview_label)
        
        # Instructions
        self.lbl_instructions = QLabel("Select a video to begin")
        self.lbl_instructions.setAlignment(Qt.AlignCenter)
        self.lbl_instructions.setStyleSheet("color: #999999; font-style: italic; margin: 10px;")
        layout.addWidget(self.lbl_instructions)
        
        return panel
    
    # === Video Selection Methods ===
    
    def add_folder(self):
        """Add folder with videos and CSV files."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder with Videos and CSVs")
        if not folder:
            return
        
        # Find video files
        video_files = []
        for ext in ['.mp4', '.avi', '.mov']:
            video_files.extend(list(Path(folder).glob(f'*{ext}')))
        
        if not video_files:
            QMessageBox.warning(self, "No Videos", "No video files found in selected folder")
            return
        
        # Match videos with CSV files
        added = 0
        for video_path in video_files:
            # Look for corresponding predictions.analysis.csv
            csv_pattern = str(video_path).replace('.mp4', '.*.predictions.analysis.csv')
            csv_files = list(Path(folder).glob(os.path.basename(csv_pattern)))
            
            if csv_files:
                csv_path = str(csv_files[0])
                config = VideoROIConfig(str(video_path), csv_path)
                self.video_configs.append(config)
                
                item = QListWidgetItem(f"○ {os.path.basename(str(video_path))}")
                self.video_list.addItem(item)
                added += 1
        
        if added > 0:
            self.video_list.setCurrentRow(0)
            self.btn_prev.setEnabled(True)
            self.btn_next.setEnabled(True)
            QMessageBox.information(self, "Videos Added", f"Added {added} video(s) with matching CSV files")
        else:
            QMessageBox.warning(self, "No Matches", "No videos with matching .predictions.analysis.csv files found")
    
    def add_videos(self):
        """Add individual video files with their CSV files."""
        video_files, _ = QFileDialog.getOpenFileNames(
            self,
            "Select Video File(s)",
            "",
            "Video Files (*.mp4 *.avi *.mov);;All Files (*)"
        )
        
        if not video_files:
            return
        
        added = 0
        skipped = 0
        
        for video_path in video_files:
            video_path = Path(video_path)
            folder = video_path.parent
            
            # Look for corresponding predictions.analysis.csv in the same folder
            csv_pattern = str(video_path.name).replace(video_path.suffix, '.*.predictions.analysis.csv')
            csv_files = list(folder.glob(csv_pattern))
            
            if csv_files:
                csv_path = str(csv_files[0])
                config = VideoROIConfig(str(video_path), csv_path)
                self.video_configs.append(config)
                
                item = QListWidgetItem(f"○ {video_path.name}")
                self.video_list.addItem(item)
                added += 1
            else:
                skipped += 1
        
        if added > 0:
            self.video_list.setCurrentRow(0)
            self.btn_prev.setEnabled(True)
            self.btn_next.setEnabled(True)
            
            message = f"Added {added} video(s) with matching CSV files"
            if skipped > 0:
                message += f"\nSkipped {skipped} video(s) without matching .predictions.analysis.csv files"
            QMessageBox.information(self, "Videos Added", message)
        else:
            QMessageBox.warning(self, "No Matches", 
                              f"None of the {len(video_files)} selected video(s) have matching .predictions.analysis.csv files")
    
    def clear_videos(self):
        """Clear all videos."""
        self.video_configs.clear()
        self.video_list.clear()
        self.current_config_idx = 0
        self.preview_label.clear()
        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)
    
    def on_video_selected(self, row):
        """Handle video selection."""
        if row >= 0 and row < len(self.video_configs):
            self.current_config_idx = row
            self.load_current_video()
    
    def previous_video(self):
        """Go to previous video."""
        if self.current_config_idx > 0:
            self.video_list.setCurrentRow(self.current_config_idx - 1)
    
    def next_video(self):
        """Go to next video."""
        if self.current_config_idx < len(self.video_configs) - 1:
            self.video_list.setCurrentRow(self.current_config_idx + 1)
    
    def load_current_video(self):
        """Load and display current video."""
        if not self.video_configs:
            return
        
        config = self.video_configs[self.current_config_idx]
        
        # Load first frame if not already loaded
        if config.first_frame is None:
            cap = cv2.VideoCapture(config.video_path)
            if not cap.isOpened():
                QMessageBox.critical(self, "Error", "Failed to open video")
                return
            
            ret, frame = cap.read()
            if ret:
                config.first_frame = frame.copy()
                config.height, config.width = frame.shape[:2]
                config.total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            
            cap.release()
        
        # Display the frame with overlays
        self.redraw_preview()
        
        # Load available keypoints from CSV if not already loaded
        if not config.available_keypoints:
            self.load_keypoints_from_csv(config)
        
        # Restore UI state for this video
        self.restore_video_ui_state(config)
        
        # Enable controls
        self.btn_set_tracking_area.setEnabled(True)
        self.btn_skip_tracking_area.setEnabled(True)
        
        self.update_config_status()
    
    def restore_video_ui_state(self, config: VideoROIConfig):
        """Restore UI state for the selected video."""
        # Restore tracking area status
        if config.tracking_area_set:
            self.lbl_tracking_status.setText("✓ Tracking area set")
            self.lbl_tracking_status.setStyleSheet("color: #4caf50;")
            self.enable_roi_buttons()
            self.btn_clear_tracking_area.setEnabled(True)
        else:
            self.lbl_tracking_status.setText("")
            self.btn_oft.setEnabled(False)
            self.btn_ldb.setEnabled(False)
            self.btn_custom.setEnabled(False)
            self.btn_scale_bar.setEnabled(False)
            self.btn_skip_roi.setEnabled(False)
            self.btn_clear_tracking_area.setEnabled(False)
        
        # Restore scale bar status
        if config.scale_bar_set:
            self.lbl_scale_status.setText(f"✓ Scale set: {config.scale_bar_cm} cm = {config.scale_bar_pixels:.1f} pixels ({config.pixels_per_cm:.2f} pixels/cm)")
        else:
            self.lbl_scale_status.setText("")
        
        # Restore ROI table
        self.roi_table.setRowCount(0)
        for roi_name, _ in config.rois:
            row = self.roi_table.rowCount()
            self.roi_table.insertRow(row)
            self.roi_table.setItem(row, 0, QTableWidgetItem(roi_name))
        
        # Restore keypoint selection
        self.update_keypoint_checkboxes(config)
    
    def update_keypoint_checkboxes(self, config: VideoROIConfig):
        """Update keypoint checkboxes to reflect current selection."""
        # Find all checkboxes in the keypoint section
        keypoint_group = None
        for widget in self.findChildren(QGroupBox):
            if widget.windowTitle() == "4. Select Keypoints for Analysis":
                keypoint_group = widget
                break
        
        if keypoint_group:
            # Update checkbox states
            for checkbox in keypoint_group.findChildren(QCheckBox):
                keypoint_name = checkbox.text()
                if keypoint_name in config.available_keypoints:
                    checkbox.setChecked(keypoint_name in config.selected_keypoints)
    
    def load_keypoints_from_csv(self, config: VideoROIConfig):
        """Extract available keypoints from CSV columns."""
        try:
            df = pd.read_csv(config.csv_path, nrows=0)  # Just read headers
            
            # Extract unique keypoint names from columns like "hatFront.x", "hatFront.y"
            keypoints = set()
            for col in df.columns:
                if '.' in col:
                    keypoint = col.rsplit('.', 1)[0]
                    keypoints.add(keypoint)
            
            config.available_keypoints = sorted(list(keypoints))
            
            # Update keypoint checkboxes
            self.update_keypoint_checkboxes(config)
            
        except Exception as e:
            QMessageBox.warning(self, "CSV Error", f"Could not read CSV: {str(e)}")
    
    def update_keypoint_checkboxes(self, config: VideoROIConfig):
        """Update keypoint selection checkboxes."""
        # Clear existing
        for i in reversed(range(self.keypoint_layout.count())):
            self.keypoint_layout.itemAt(i).widget().setParent(None)
        
        self.keypoint_checkboxes.clear()
        
        # Create checkboxes
        for keypoint in config.available_keypoints:
            chk = QCheckBox(keypoint)
            chk.stateChanged.connect(self.on_keypoint_selection_changed)
            self.keypoint_layout.addWidget(chk)
            self.keypoint_checkboxes.append(chk)
    
    # === Tracking Area Methods ===
    
    def start_tracking_area_drawing(self):
        """Start drawing tracking area."""
        self.drawing_mode = 'tracking_area'
        self.current_polygon = []
        self.lbl_instructions.setText("Click to draw tracking area boundary. Press ENTER when done, ESC to cancel.")
        self.lbl_instructions.setStyleSheet("color: #4caf50; font-weight: bold; margin: 10px;")
    
    def clear_tracking_area(self):
        """Clear tracking area."""
        if self.video_configs:
            config = self.video_configs[self.current_config_idx]
            config.tracking_area = []
            config.tracking_area_set = False
            self.lbl_tracking_status.setText("")
            self.redraw_preview()
    
    def skip_tracking_area(self):
        """Skip tracking area definition."""
        if self.video_configs:
            config = self.video_configs[self.current_config_idx]
            config.tracking_area_set = True
            self.lbl_tracking_status.setText("✓ Tracking area skipped")
            self.lbl_tracking_status.setStyleSheet("color: #4caf50;")
            self.enable_roi_buttons()
    
    # === ROI Drawing Methods ===
    
    def start_oft_drawing(self):
        """Start OFT layout drawing."""
        self.drawing_mode = 'oft'
        self.current_polygon = []
        self.lbl_instructions.setText("Click 4 corners of the Open Field Test floor. Press ENTER when done.")
        self.lbl_instructions.setStyleSheet("color: #4caf50; font-weight: bold; margin: 10px;")
    
    def start_ldb_drawing(self):
        """Start Light-Dark Box drawing."""
        self.drawing_mode = 'ldb_light'
        self.current_polygon = []
        self.lbl_instructions.setText("Draw the LIGHT box area (polygon). Press ENTER when done.")
        self.lbl_instructions.setStyleSheet("color: #00ff00; font-weight: bold; margin: 10px;")
    
    def start_custom_roi_drawing(self):
        """Start custom ROI drawing."""
        self.drawing_mode = 'custom'
        self.current_polygon = []
        
        # Get custom ROI count
        config = self.video_configs[self.current_config_idx]
        custom_count = sum(1 for name in config.get_roi_names() if name.startswith('custom_'))
        self.current_custom_idx = custom_count + 1
        
        self.lbl_instructions.setText(f"Draw custom ROI #{self.current_custom_idx}. Press ENTER when done.")
        self.lbl_instructions.setStyleSheet("color: #4caf50; font-weight: bold; margin: 10px;")
    
    def start_scale_bar_setting(self):
        """Start scale bar calibration."""
        self.drawing_mode = 'scale_bar'
        self.scale_bar_points = []
        
        self.lbl_instructions.setText("Click two points to set scale bar, then enter the known distance in cm.")
        self.lbl_instructions.setStyleSheet("color: #4caf50; font-weight: bold; margin: 10px;")
    
    def finish_scale_bar(self):
        """Finish scale bar setting and get real-world distance."""
        if len(self.scale_bar_points) != 2:
            return
        
        # Calculate pixel distance
        p1, p2 = self.scale_bar_points
        pixel_distance = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
        
        # Ask user for real-world distance
        distance_cm, ok = QInputDialog.getDouble(
            self,
            "Set Scale Bar",
            f"Distance between points: {pixel_distance:.1f} pixels\n\nEnter the real-world distance in cm:",
            value=10.0,
            min=0.1,
            max=1000.0,
            decimals=2
        )
        
        if ok and distance_cm > 0:
            config = self.video_configs[self.current_config_idx]
            config.scale_bar_set = True
            config.scale_bar_pixels = pixel_distance
            config.scale_bar_cm = distance_cm
            config.pixels_per_cm = pixel_distance / distance_cm
            
            self.lbl_scale_status.setText(f"✓ Scale set: {distance_cm} cm = {pixel_distance:.1f} pixels ({config.pixels_per_cm:.2f} pixels/cm)")
            self.lbl_instructions.setText("")
            
            # Update list item
            item = self.video_list.item(self.current_config_idx)
            if config.configured:
                item.setText(f"✓ {os.path.basename(config.video_path)}")
            else:
                item.setText(f"◐ {os.path.basename(config.video_path)}")
        
        self.drawing_mode = None
        self.scale_bar_points = []
        self.redraw_preview()
    
    def skip_rois(self):
        """Skip ROI definition."""
        if self.video_configs:
            self.enable_keypoint_selection()
    
    # === Mouse and Keyboard Handlers ===
    
    def on_preview_click(self, event):
        """Handle mouse click on preview."""
        if not self.drawing_mode or not self.video_configs:
            return
        
        # Get click position relative to displayed image
        label_width = self.preview_label.width()
        label_height = self.preview_label.height()
        
        config = self.video_configs[self.current_config_idx]
        if config.first_frame is None:
            return
        
        # Calculate scaling
        frame_height, frame_width = config.first_frame.shape[:2]
        scale = min(label_width / frame_width, label_height / frame_height)
        
        # Get click coordinates in original frame space
        x = int((event.x() - (label_width - frame_width * scale) / 2) / scale)
        y = int((event.y() - (label_height - frame_height * scale) / 2) / scale)
        
        # Clamp to frame bounds
        x = max(0, min(x, frame_width - 1))
        y = max(0, min(y, frame_height - 1))
        
        # Handle scale bar mode (only needs 2 points)
        if self.drawing_mode == 'scale_bar':
            self.scale_bar_points.append((x, y))
            self.redraw_preview()
            
            if len(self.scale_bar_points) == 2:
                self.finish_scale_bar()
            return
        
        # Handle polygon drawing modes
        self.current_polygon.append((x, y))
        self.redraw_preview()
    
    def keyPressEvent(self, event):
        """Handle keyboard events."""
        if event.key() == Qt.Key_Return or event.key() == Qt.Key_Enter:
            self.finish_current_drawing()
        elif event.key() == Qt.Key_Escape:
            self.cancel_current_drawing()
    
    def finish_current_drawing(self):
        """Finish current polygon drawing."""
        if not self.drawing_mode or not self.current_polygon:
            return
        
        config = self.video_configs[self.current_config_idx]
        
        if self.drawing_mode == 'tracking_area':
            config.tracking_area = self.current_polygon.copy()
            config.tracking_area_set = True
            self.lbl_tracking_status.setText("✓ Tracking area set")
            self.lbl_tracking_status.setStyleSheet("color: #4caf50;")
            self.enable_roi_buttons()
            
        elif self.drawing_mode == 'oft':
            if len(self.current_polygon) < 4:
                QMessageBox.warning(self, "Invalid", "OFT requires 4 corners")
                return
            
            # Create outer and center ROIs (center first for priority)
            center_roi = self.calculate_oft_center(self.current_polygon)
            config.add_roi('oft_center', center_roi)
            config.add_roi('oft_outer', self.current_polygon.copy())
            
            self.update_roi_table()
            
        elif self.drawing_mode == 'ldb_light':
            config.add_roi('ldb_light', self.current_polygon.copy())
            self.update_roi_table()
            
            # Now prompt for dark box
            self.drawing_mode = 'ldb_dark'
            self.current_polygon = []
            self.lbl_instructions.setText("Draw the DARK box area (polygon). Press ENTER when done.")
            self.lbl_instructions.setStyleSheet("color: #ffff00; font-weight: bold; margin: 10px;")
            self.redraw_preview()
            return
            
        elif self.drawing_mode == 'ldb_dark':
            config.add_roi('ldb_dark', self.current_polygon.copy())
            self.update_roi_table()
            
        elif self.drawing_mode == 'custom':
            config.add_roi(f'custom_{self.current_custom_idx}', self.current_polygon.copy())
            self.update_roi_table()
        
        self.drawing_mode = None
        self.current_polygon = []
        self.lbl_instructions.setText("ROI saved. Draw more or continue to keypoint selection.")
        self.lbl_instructions.setStyleSheet("color: #999999; font-style: italic; margin: 10px;")
        self.enable_keypoint_selection()
        self.redraw_preview()
    
    def cancel_current_drawing(self):
        """Cancel current drawing."""
        self.drawing_mode = None
        self.current_polygon = []
        self.lbl_instructions.setText("Drawing cancelled")
        self.lbl_instructions.setStyleSheet("color: #999999; font-style: italic; margin: 10px;")
        self.redraw_preview()
    
    def calculate_oft_center(self, outer_polygon: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        """Calculate center box (50% of area) by scaling the polygon shape from its centroid."""
        # Calculate centroid of the polygon
        xs = [p[0] for p in outer_polygon]
        ys = [p[1] for p in outer_polygon]
        
        centroid_x = sum(xs) / len(xs)
        centroid_y = sum(ys) / len(ys)
        
        # Scale factor: 50% area means sqrt(0.5) ≈ 0.707 of each dimension
        scale_factor = np.sqrt(0.5)
        
        # Scale each point toward the centroid
        center_polygon = []
        for x, y in outer_polygon:
            # Vector from centroid to point
            dx = x - centroid_x
            dy = y - centroid_y
            
            # Scale the vector
            new_x = centroid_x + dx * scale_factor
            new_y = centroid_y + dy * scale_factor
            
            center_polygon.append((int(new_x), int(new_y)))
        
        return center_polygon
    
    # === UI Update Methods ===
    
    def update_roi_table(self):
        """Update ROI table display."""
        if not self.video_configs:
            return
        
        config = self.video_configs[self.current_config_idx]
        
        # Temporarily disconnect itemChanged signal
        try:
            self.roi_table.itemChanged.disconnect(self.on_roi_renamed)
        except:
            pass
        
        self.roi_table.setRowCount(len(config.rois))
        for i, (roi_name, _) in enumerate(config.rois):
            item = QTableWidgetItem(roi_name)
            item.setFlags(item.flags() | Qt.ItemIsEditable)
            self.roi_table.setItem(i, 0, item)
        
        # Reconnect signal
        self.roi_table.itemChanged.connect(self.on_roi_renamed)
    
    def on_roi_renamed(self, item):
        """Handle ROI rename."""
        if not self.video_configs:
            return
        
        config = self.video_configs[self.current_config_idx]
        row = item.row()
        
        roi_names = config.get_roi_names()
        if row < len(roi_names):
            old_name = roi_names[row]
            new_name = item.text()
            
            if new_name and new_name != old_name:
                config.rename_roi(old_name, new_name)
    
    def on_roi_selection_changed(self):
        """Handle ROI table selection change - enable/disable arrow buttons."""
        selected_rows = self.roi_table.selectedIndexes()
        
        if not selected_rows:
            self.btn_move_up.setEnabled(False)
            self.btn_move_down.setEnabled(False)
            return
        
        row = selected_rows[0].row()
        row_count = self.roi_table.rowCount()
        
        # Enable/disable buttons based on position
        self.btn_move_up.setEnabled(row > 0)
        self.btn_move_down.setEnabled(row < row_count - 1)
    
    def move_roi_up(self):
        """Move selected ROI up in priority (higher priority)."""
        if not self.video_configs:
            return
        
        current_row = self.roi_table.currentRow()
        if current_row <= 0:
            return
        
        config = self.video_configs[self.current_config_idx]
        
        # Swap ROIs in config
        config.rois[current_row], config.rois[current_row - 1] = \
            config.rois[current_row - 1], config.rois[current_row]
        
        # Update table
        self.update_roi_table()
        
        # Reselect the moved row
        self.roi_table.selectRow(current_row - 1)
        
        # Update preview
        self.redraw_preview()
    
    def move_roi_down(self):
        """Move selected ROI down in priority (lower priority)."""
        if not self.video_configs:
            return
        
        current_row = self.roi_table.currentRow()
        if current_row < 0 or current_row >= self.roi_table.rowCount() - 1:
            return
        
        config = self.video_configs[self.current_config_idx]
        
        # Swap ROIs in config
        config.rois[current_row], config.rois[current_row + 1] = \
            config.rois[current_row + 1], config.rois[current_row]
        
        # Update table
        self.update_roi_table()
        
        # Reselect the moved row
        self.roi_table.selectRow(current_row + 1)
        
        # Update preview
        self.redraw_preview()
    
    def on_keypoint_selection_changed(self):
        """Handle keypoint selection change."""
        if not self.video_configs:
            return
        
        config = self.video_configs[self.current_config_idx]
        config.selected_keypoints = [
            chk.text() for chk in self.keypoint_checkboxes if chk.isChecked()
        ]
        
        self.update_config_status()
    
    def on_create_video_toggled(self):
        """Enable/disable video options based on create_video checkbox."""
        enabled = self.chk_create_video.isChecked()
        self.video_options_widget.setEnabled(enabled)
    
    def update_config_status(self):
        """Update configuration status."""
        if not self.video_configs:
            return
        
        config = self.video_configs[self.current_config_idx]
        
        # Check if configured
        has_tracking_area = config.tracking_area_set
        has_rois = len(config.rois) > 0
        has_keypoints = len(config.selected_keypoints) > 0
        
        config.configured = has_tracking_area and has_keypoints
        
        if config.configured:
            video_name = os.path.basename(config.video_path)
            self.lbl_config_status.setText(f"✓ Video {video_name} is ready for processing")
            self.lbl_config_status.setStyleSheet("color: #4caf50; font-weight: bold;")
            
            # Update list item
            item = self.video_list.item(self.current_config_idx)
            if item:
                item.setText(f"✓ {video_name}")
            
            # Enable processing if all configured
            all_configured = all(c.configured for c in self.video_configs)
            self.btn_process.setEnabled(all_configured)
        else:
            self.lbl_config_status.setText("")
            self.btn_process.setEnabled(False)
    
    def enable_roi_buttons(self):
        """Enable ROI drawing buttons."""
        self.btn_oft.setEnabled(True)
        self.btn_ldb.setEnabled(True)
        self.btn_custom.setEnabled(True)
        self.btn_scale_bar.setEnabled(True)
        self.btn_skip_roi.setEnabled(True)
        self.btn_clear_tracking_area.setEnabled(True)
    
    def enable_keypoint_selection(self):
        """Enable keypoint selection."""
        # Keypoints are always selectable after CSV is loaded
        pass
    
    def redraw_preview(self):
        """Redraw preview with current overlays."""
        if not self.video_configs:
            return
        
        config = self.video_configs[self.current_config_idx]
        if config.first_frame is None:
            return
        
        frame = config.first_frame.copy()
        
        # Draw tracking area
        if config.tracking_area:
            pts = np.array(config.tracking_area, dtype=np.int32)
            cv2.polylines(frame, [pts], True, (255, 0, 255), 2)  # Magenta
        
        # Draw scale bar points
        if self.scale_bar_points:
            for pt in self.scale_bar_points:
                cv2.circle(frame, pt, 8, (0, 255, 255), -1)  # Cyan circles
            
            if len(self.scale_bar_points) == 2:
                cv2.line(frame, self.scale_bar_points[0], self.scale_bar_points[1], (0, 255, 255), 3)
        
        # Draw existing scale bar if set
        if config.scale_bar_set and hasattr(config, 'scale_bar_pixels'):
            # Draw a small indicator in corner
            text = f"Scale: {config.scale_bar_cm}cm"
            cv2.putText(frame, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        
        # Draw current polygon being drawn
        if self.current_polygon:
            for i, pt in enumerate(self.current_polygon):
                cv2.circle(frame, pt, 5, (0, 255, 0), -1)
                if i > 0:
                    cv2.line(frame, self.current_polygon[i-1], pt, (0, 255, 0), 2)
        
        # Draw ROIs
        for roi_name, roi_polygon in config.rois:
            if 'center' in roi_name.lower() or 'inner' in roi_name.lower():
                color = (0, 165, 255)  # Orange
            elif 'light' in roi_name.lower():
                color = (0, 255, 0)  # Green
            elif 'dark' in roi_name.lower():
                color = (0, 255, 255)  # Yellow
            else:
                color = (0, 255, 0)  # Green
            
            pts = np.array(roi_polygon, dtype=np.int32)
            cv2.polylines(frame, [pts], True, color, 2)
            
            # Add label
            if len(roi_polygon) > 0:
                cv2.putText(frame, roi_name, tuple(roi_polygon[0]), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
        
        self.display_frame(frame)
    
    def display_frame(self, frame: np.ndarray):
        """Display frame in preview label."""
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, ch = frame_rgb.shape
        bytes_per_line = ch * w
        qt_image = QImage(frame_rgb.data, w, h, bytes_per_line, QImage.Format_RGB888)
        
        pixmap = QPixmap.fromImage(qt_image)
        scaled_pixmap = pixmap.scaled(self.preview_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.preview_label.setPixmap(scaled_pixmap)
    
    # === Processing Methods ===
    
    def start_processing(self):
        """Start batch processing."""
        if not self.video_configs:
            return
        
        # Verify all configured
        if not all(c.configured for c in self.video_configs):
            QMessageBox.warning(self, "Not Ready", "Not all videos are configured")
            return
        
        self.btn_process.setEnabled(False)
        self.btn_cancel.setEnabled(True)
        
        # Get display options
        show_tracking_area = self.chk_show_tracking_area.isChecked()
        show_rois = self.chk_show_rois.isChecked()
        show_all_keypoints = self.chk_show_all_keypoints.isChecked()
        show_edges = self.chk_show_edges.isChecked()
        show_keypoint_labels = self.chk_show_keypoint_labels.isChecked()
        interpolate_keypoints = self.chk_interpolate.isChecked()
        
        self.processor = ROIProcessor(
            self.video_configs, 
            self.chk_create_video.isChecked(),
            show_tracking_area,
            show_rois,
            show_all_keypoints,
            show_edges,
            show_keypoint_labels,
            interpolate_keypoints
        )
        self.processor.progress.connect(self.update_progress)
        self.processor.status.connect(self.update_status)
        self.processor.video_finished.connect(self.on_video_finished)
        self.processor.all_finished.connect(self.on_all_finished)
        self.processor.start()
    
    def cancel_processing(self):
        """Cancel processing."""
        if self.processor:
            self.processor.cancel()
    
    def update_progress(self, video_idx, frame_idx, total_frames):
        """Update progress bar based on overall batch progress."""
        total_videos = len(self.video_configs)
        
        if total_videos > 0 and total_frames > 0:
            # Calculate progress for current video (0-1)
            video_progress = frame_idx / total_frames
            
            # Calculate overall progress across all videos
            # Each video contributes (1/total_videos) to the total progress
            overall_progress = (video_idx / total_videos) + (video_progress / total_videos)
            
            # Convert to percentage
            progress_percent = int(overall_progress * 100)
            self.progress_bar.setValue(progress_percent)
    
    def update_status(self, message: str):
        """Update status label."""
        self.lbl_status.setText(message)
    
    def on_video_finished(self, video_idx, success, message):
        """Handle video processing completion."""
        self.lbl_status.setText(message)
    
    def on_all_finished(self, success, message):
        """Handle all processing completion."""
        self.btn_process.setEnabled(True)
        self.btn_cancel.setEnabled(False)
        self.progress_bar.setValue(100 if success else 0)
        
        QMessageBox.information(self, "Complete", message)


def main():
    """Run the ROI Tool GUI."""
    if not CV2_AVAILABLE or not PANDAS_AVAILABLE:
        print("Error: Required dependencies not available")
        print("Install with: pip install opencv-python pandas")
        sys.exit(1)
    
    app = QApplication(sys.argv)
    window = ROIToolGUI()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
