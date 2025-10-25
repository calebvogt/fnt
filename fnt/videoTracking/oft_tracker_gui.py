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
    QGroupBox, QLineEdit, QCheckBox, QSpinBox, QDoubleSpinBox, QComboBox
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
    rectangle_drawn = pyqtSignal(int, int, int, int, object, float)  # (x, y, width, height, corners, rotation_angle)
    
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
            # If we have rotated corners, use them; otherwise use bounding box
            if self.rectangle_corners and abs(self.rotation_angle) > 0.1:
                # Draw rotated rectangle using corners
                pts = np.array(self.rectangle_corners, np.int32)
                pts = pts.reshape((-1, 1, 2))
                cv2.polylines(display_frame, [pts], True, (0, 255, 0), 2)
                
                # Calculate center from stored value
                center_x, center_y = self.rectangle_center if self.rectangle_center else (
                    int(sum(c[0] for c in self.rectangle_corners) / 4),
                    int(sum(c[1] for c in self.rectangle_corners) / 4)
                )
                
                # Draw manipulation handles if in rectangle mode
                if self.drawing_mode == 'rectangle':
                    # Draw corner handles at actual corners
                    for corner in self.rectangle_corners:
                        cv2.circle(display_frame, corner, 8, (0, 255, 0), -1)
                        cv2.circle(display_frame, corner, 10, (255, 255, 255), 2)
                    
                    # Draw rotation handle above rectangle
                    # Use the topmost point
                    ys = [c[1] for c in self.rectangle_corners]
                    top_y = min(ys)
                    rotation_handle = (int(center_x), max(20, top_y - 40))
                    
                    cv2.line(display_frame, (int(center_x), top_y), rotation_handle, (255, 255, 0), 2)
                    cv2.circle(display_frame, rotation_handle, 12, (255, 255, 0), -1)
                    cv2.circle(display_frame, rotation_handle, 14, (255, 255, 255), 2)
                    cv2.putText(display_frame, "R", (rotation_handle[0] - 7, rotation_handle[1] + 7),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
                # Draw center zone (60% of original dimensions, also rotated)
                # Scale corners towards center by 60%
                inner_corners = []
                for corner in self.rectangle_corners:
                    # Vector from center to corner
                    dx = corner[0] - center_x
                    dy = corner[1] - center_y
                    # Scale by 0.6
                    inner_x = int(center_x + dx * 0.6)
                    inner_y = int(center_y + dy * 0.6)
                    inner_corners.append((inner_x, inner_y))
                
                inner_pts = np.array(inner_corners, np.int32).reshape((-1, 1, 2))
                cv2.polylines(display_frame, [inner_pts], True, (255, 255, 0), 2)
                
            else:
                # Draw normal axis-aligned rectangle
                x1, y1 = self.rectangle_start
                x2, y2 = self.rectangle_end
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
                
                # Draw manipulation handles if in rectangle mode
                if self.drawing_mode == 'rectangle':
                    corners = self.get_rectangle_corners()
                    if corners:
                        # Draw corner handles (resize)
                        for corner in corners:
                            cv2.circle(display_frame, corner, 8, (0, 255, 0), -1)
                            cv2.circle(display_frame, corner, 10, (255, 255, 255), 2)
                        
                        # Draw rotation handle (above top center)
                        rotation_handle = self.get_rotation_handle_position()
                        if rotation_handle:
                            # Draw line from top center to rotation handle
                            center_top = ((x1 + x2) // 2, min(y1, y2))
                            cv2.line(display_frame, center_top, rotation_handle, (255, 255, 0), 2)
                            # Draw circular rotation handle
                            cv2.circle(display_frame, rotation_handle, 12, (255, 255, 0), -1)
                            cv2.circle(display_frame, rotation_handle, 14, (255, 255, 255), 2)
                            # Draw "R" for rotate
                            cv2.putText(display_frame, "R", (rotation_handle[0] - 7, rotation_handle[1] + 7),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 2)
                
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
        """Handle mouse press for click selection, rectangle drawing, or manipulation."""
        if self.original_frame is None:
            return
            
        img_x, img_y = self.widget_to_image_coords(event.x(), event.y())
        
        # Check if clicking on existing rectangle for manipulation
        if self.drawing_mode == 'rectangle' and self.rectangle_start is not None and self.rectangle_end is not None:
            manipulation_zone = self.detect_rectangle_manipulation_zone(img_x, img_y)
            if manipulation_zone:
                # Special case for rotation - start drag mode
                if manipulation_zone == 'rotate':
                    self.manipulation_mode = 'rotate'
                    self.drag_start_pos = (img_x, img_y)
                    
                    # Initialize rectangle center and dimensions if not set
                    if self.rectangle_center is None:
                        x1, y1 = self.rectangle_start
                        x2, y2 = self.rectangle_end
                        self.rectangle_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                        self.rectangle_width = abs(x2 - x1)
                        self.rectangle_height = abs(y2 - y1)
                    
                    # Calculate starting angle from center to mouse
                    cx, cy = self.rectangle_center
                    dx = img_x - cx
                    dy = img_y - cy
                    self.drag_start_angle = np.degrees(np.arctan2(dy, dx))
                    return
                else:
                    # For other manipulations, start drag mode
                    self.manipulation_mode = manipulation_zone
                    self.drag_start_pos = (img_x, img_y)
                    self.original_rectangle = (self.rectangle_start, self.rectangle_end)
                    
                    # Store original rotation state for resize/move operations
                    if self.rectangle_center is None:
                        x1, y1 = self.rectangle_start
                        x2, y2 = self.rectangle_end
                        self.rectangle_center = ((x1 + x2) / 2, (y1 + y2) / 2)
                        self.rectangle_width = abs(x2 - x1)
                        self.rectangle_height = abs(y2 - y1)
                    
                    self.original_center = self.rectangle_center
                    self.original_width = self.rectangle_width
                    self.original_height = self.rectangle_height
                    return
        
        if self.drawing_mode is None:
            return
        
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
                # Create non-rotated corners for initial rectangle
                corners = [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
                self.rectangle_drawn.emit(x, y, w, h, corners, 0.0)  # 0 rotation for initial draw
                # Don't disable drawing mode anymore - allow manipulation
                self._update_display()
                
    def mouseMoveEvent(self, event):
        """Handle mouse move for rectangle preview or manipulation."""
        if self.original_frame is None:
            return
        
        img_x, img_y = self.widget_to_image_coords(event.x(), event.y())
        
        # Handle rotation manipulation
        if self.manipulation_mode == 'rotate' and self.rectangle_center and self.drag_start_angle is not None:
            # Calculate current angle from center to mouse
            cx, cy = self.rectangle_center
            dx = img_x - cx
            dy = img_y - cy
            current_angle = np.degrees(np.arctan2(dy, dx))
            
            # Calculate angle difference
            angle_delta = current_angle - self.drag_start_angle
            
            # Update rotation angle
            self.rotation_angle = angle_delta % 360
            
            # Update rectangle corners
            self.update_rectangle_corners()
            self._update_display()
            return
        
        # Handle other rectangle manipulations
        if self.manipulation_mode and self.drag_start_pos:
            dx = img_x - self.drag_start_pos[0]
            dy = img_y - self.drag_start_pos[1]
            
            if self.manipulation_mode == 'move':
                # Move entire rectangle - translate center
                if hasattr(self, 'original_center'):
                    orig_cx, orig_cy = self.original_center
                    self.rectangle_center = (orig_cx + dx, orig_cy + dy)
                    self.update_rectangle_corners()
                else:
                    # Fallback for non-rotated
                    if self.original_rectangle:
                        orig_start, orig_end = self.original_rectangle
                        x1, y1 = orig_start
                        x2, y2 = orig_end
                        self.rectangle_start = (x1 + dx, y1 + dy)
                        self.rectangle_end = (x2 + dx, y2 + dy)
                
            elif self.manipulation_mode.startswith('resize_'):
                # Resize operations - need to handle rotated rectangles
                if abs(self.rotation_angle) > 0.1 and hasattr(self, 'original_width'):
                    # For rotated rectangles, calculate resize based on drag distance
                    # Project the drag vector onto the width/height axes of the rotated rectangle
                    angle_rad = np.radians(self.rotation_angle)
                    cos_a = np.cos(angle_rad)
                    sin_a = np.sin(angle_rad)
                    
                    # Project drag onto rotated axes
                    dx_local = dx * cos_a + dy * sin_a
                    dy_local = -dx * sin_a + dy * cos_a
                    
                    # Adjust width/height based on which corner is being dragged
                    if self.manipulation_mode == 'resize_tl':
                        self.rectangle_width = max(20, self.original_width - 2 * dx_local)
                        self.rectangle_height = max(20, self.original_height - 2 * dy_local)
                    elif self.manipulation_mode == 'resize_tr':
                        self.rectangle_width = max(20, self.original_width + 2 * dx_local)
                        self.rectangle_height = max(20, self.original_height - 2 * dy_local)
                    elif self.manipulation_mode == 'resize_br':
                        self.rectangle_width = max(20, self.original_width + 2 * dx_local)
                        self.rectangle_height = max(20, self.original_height + 2 * dy_local)
                    elif self.manipulation_mode == 'resize_bl':
                        self.rectangle_width = max(20, self.original_width - 2 * dx_local)
                        self.rectangle_height = max(20, self.original_height + 2 * dy_local)
                    
                    self.update_rectangle_corners()
                else:
                    # Non-rotated resize - original logic
                    if self.original_rectangle:
                        orig_start, orig_end = self.original_rectangle
                        x1, y1 = orig_start
                        x2, y2 = orig_end
                        
                        if self.manipulation_mode == 'resize_tl':
                            self.rectangle_start = (x1 + dx, y1 + dy)
                            self.rectangle_end = orig_end
                        elif self.manipulation_mode == 'resize_tr':
                            self.rectangle_start = (x1, y1 + dy)
                            self.rectangle_end = (x2 + dx, y2)
                        elif self.manipulation_mode == 'resize_br':
                            self.rectangle_start = orig_start
                            self.rectangle_end = (x2 + dx, y2 + dy)
                        elif self.manipulation_mode == 'resize_bl':
                            self.rectangle_start = (x1 + dx, y1)
                            self.rectangle_end = (x2, y2 + dy)
            
            self._update_display()
            return
        
        # Handle rectangle drawing preview
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
    
    def mouseReleaseEvent(self, event):
        """Handle mouse release to finish manipulation."""
        if self.manipulation_mode:
            # Emit updated rectangle coordinates
            if self.rectangle_corners and abs(self.rotation_angle) > 0.1:
                # For rotated rectangles, use bounding box
                xs = [c[0] for c in self.rectangle_corners]
                ys = [c[1] for c in self.rectangle_corners]
                x = min(xs)
                y = min(ys)
                w = max(xs) - min(xs)
                h = max(ys) - min(ys)
            else:
                # For non-rotated, use normal bounding box
                x1, y1 = self.rectangle_start
                x2, y2 = self.rectangle_end
                x = min(x1, x2)
                y = min(y1, y2)
                w = abs(x2 - x1)
                h = abs(y2 - y1)
            
            # Emit with corners and rotation angle
            corners = self.rectangle_corners if self.rectangle_corners else [(x, y), (x+w, y), (x+w, y+h), (x, y+h)]
            self.rectangle_drawn.emit(x, y, w, h, corners, self.rotation_angle)
            
            # Reset manipulation state
            self.manipulation_mode = None
            self.drag_start_pos = None
            self.drag_start_angle = None
            self.original_rectangle = None
            
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
    frame_signal = pyqtSignal(np.ndarray)  # current frame with tracking overlay
    finished_signal = pyqtSignal(str)  # output_path
    error_signal = pyqtSignal(str)  # error_message
    
    def __init__(
        self,
        video_path: str,
        sam_checkpoint: str,
        click_point: Tuple[int, int],
        arena_rectangle: Tuple[int, int, int, int],  # Changed from arena_circle
        arena_corners,  # List of 4 (x,y) tuples for rotated rectangle
        arena_rotation: float,  # Rotation angle in degrees
        sam_update_interval: int,
        model_type: str,
        device: str,
        flow_window_size: int = 21,
        tracking_method: str = "optical_flow"
    ):
        super().__init__()
        self.video_path = video_path
        self.sam_checkpoint = sam_checkpoint
        self.click_point = click_point
        self.arena_rectangle = arena_rectangle  # (x, y, width, height)
        self.arena_corners = arena_corners  # Rotated corners
        self.arena_rotation = arena_rotation  # Rotation angle
        self.sam_update_interval = sam_update_interval
        self.model_type = model_type
        self.device = device
        self.flow_window_size = flow_window_size
        self.tracking_method = tracking_method
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
                sam_update_interval=self.sam_update_interval,
                flow_window_size=self.flow_window_size,
                tracking_method=self.tracking_method
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
                    
                    # Create frame overlay for live preview
                    display_frame = frame_rgb.copy()
                    
                    # Draw current position
                    cv2.circle(display_frame, (int(x), int(y)), 8, (0, 255, 0), -1)
                    
                    # Draw arena rectangle (rotated if applicable)
                    if self.arena_corners:
                        # Draw rotated rectangle using corners
                        corners_array = np.array(self.arena_corners, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(display_frame, [corners_array], True, (255, 165, 0), 2)
                    else:
                        # Draw axis-aligned rectangle (fallback)
                        arena_x, arena_y, arena_w, arena_h = self.arena_rectangle
                        cv2.rectangle(
                            display_frame,
                            (arena_x, arena_y),
                            (arena_x + arena_w, arena_y + arena_h),
                            (255, 165, 0),
                            2
                        )
                    
                    # Draw center zone (inner 60%)
                    arena_x, arena_y, arena_w, arena_h = self.arena_rectangle
                    if self.arena_corners and abs(self.arena_rotation) > 0.1:
                        # For rotated rectangles, calculate center zone corners
                        # Scale down by 0.6 around center
                        cx = arena_x + arena_w / 2
                        cy = arena_y + arena_h / 2
                        inner_w = arena_w * 0.6
                        inner_h = arena_h * 0.6
                        
                        # Calculate inner rectangle corners (before rotation)
                        inner_corners = [
                            (-inner_w/2, -inner_h/2),
                            (inner_w/2, -inner_h/2),
                            (inner_w/2, inner_h/2),
                            (-inner_w/2, inner_h/2)
                        ]
                        
                        # Rotate and translate inner corners
                        angle_rad = np.radians(self.arena_rotation)
                        cos_a = np.cos(angle_rad)
                        sin_a = np.sin(angle_rad)
                        
                        rotated_inner = []
                        for x, y in inner_corners:
                            x_rot = x * cos_a - y * sin_a + cx
                            y_rot = x * sin_a + y * cos_a + cy
                            rotated_inner.append((int(x_rot), int(y_rot)))
                        
                        inner_array = np.array(rotated_inner, dtype=np.int32).reshape((-1, 1, 2))
                        cv2.polylines(display_frame, [inner_array], True, (255, 255, 0), 1)
                    else:
                        # For non-rotated rectangles, use simple rectangle
                        center_margin_w = int(arena_w * 0.2)
                        center_margin_h = int(arena_h * 0.2)
                        cv2.rectangle(
                            display_frame,
                            (arena_x + center_margin_w, arena_y + center_margin_h),
                            (arena_x + arena_w - center_margin_w, arena_y + arena_h - center_margin_h),
                            (255, 255, 0),
                            1
                        )
                    
                    # Draw trajectory (last 50 points)
                    if len(tracker.trajectory) > 1:
                        recent_points = tracker.trajectory[-50:]
                        for i in range(1, len(recent_points)):
                            pt1 = (int(recent_points[i-1][1]), int(recent_points[i-1][2]))
                            pt2 = (int(recent_points[i][1]), int(recent_points[i][2]))
                            cv2.line(display_frame, pt1, pt2, (0, 200, 255), 1)
                    
                    # Emit frame for live preview
                    self.frame_signal.emit(display_frame)
                    
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
        self.arena_corners = None  # Store rotated corners
        self.arena_rotation = 0.0  # Store rotation angle
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
        self.sam_interval_spin.setRange(10, 1000)  # Increased from 300 to 1000 for CPU users
        self.sam_interval_spin.setValue(30)
        self.sam_interval_spin.setSuffix(" frames")
        self.sam_interval_spin.setToolTip(
            "How often SAM re-segments the animal:\n\n"
            "ARCHITECTURE:\n"
            "• Frame 1, 30, 60, etc.: SAM segments (SLOW but accurate)\n"
            "• Frames 2-29, 31-59, etc.: Optical flow tracks (FAST)\n\n"
            "GPU users: 20-50 frames recommended\n"
            "CPU users: 100-500 frames (SAM is very slow on CPU)\n\n"
            "If animal moves quickly or arena is cluttered, use lower values."
        )
        interval_layout.addWidget(self.sam_interval_spin)
        setup_layout.addLayout(interval_layout)
        
        # Add explanation label
        interval_help = QLabel(
            "<b>How it works:</b> SAM segments every N frames (accurate but slow). "
            "Between SAM updates, fast optical flow tracks the blob. "
            "Hybrid approach = SAM accuracy + optical flow speed."
        )
        interval_help.setWordWrap(True)
        interval_help.setStyleSheet("color: #0078d4; font-size: 10px; padding: 2px 10px; background-color: #1e1e1e; border-left: 3px solid #0078d4;")
        setup_layout.addWidget(interval_help)
        
        # Tracking method selector
        tracking_method_layout = QHBoxLayout()
        tracking_method_layout.addWidget(QLabel("Tracking Method:"))
        self.tracking_method_combo = QComboBox()
        self.tracking_method_combo.addItems(["Optical Flow (Precise)", "Centroid Matching (Fast)"])
        self.tracking_method_combo.setCurrentIndex(0)  # Default to optical flow
        self.tracking_method_combo.setToolTip(
            "Choose tracking algorithm:\n\n"
            "Optical Flow: Very precise, tracks feature points, slower (~5-10 fps)\n"
            "  - Best for: High-quality tracking, detailed analysis\n"
            "  - Use when: Accuracy is critical\n\n"
            "Centroid Matching: Fast blob detection, ~10-30x faster\n"
            "  - Best for: Quick previews, real-time tracking\n"
            "  - Use when: Speed matters more than precision\n\n"
            "Both use SAM for periodic refinement."
        )
        self.tracking_method_combo.currentIndexChanged.connect(self.on_tracking_method_changed)
        tracking_method_layout.addWidget(self.tracking_method_combo)
        tracking_method_layout.addStretch()
        setup_layout.addLayout(tracking_method_layout)
        
        tracking_help = QLabel("Fast centroid is good for most cases; optical flow for maximum accuracy")
        tracking_help.setWordWrap(True)
        tracking_help.setStyleSheet("color: #888888; font-size: 9px; font-style: italic; padding: 2px 10px;")
        setup_layout.addWidget(tracking_help)
        
        # Optical Flow Window Size
        flow_window_layout = QHBoxLayout()
        flow_window_layout.addWidget(QLabel("Optical Flow Window:"))
        self.flow_window_spin = QSpinBox()
        self.flow_window_spin.setRange(5, 201)  # Increased max from 51 to 201
        self.flow_window_spin.setSingleStep(2)  # Step by 2 to keep odd
        self.flow_window_spin.setValue(101)  # Default 101x101 (better for erratic movement)
        self.flow_window_spin.setSuffix(" px")
        self.flow_window_spin.setToolTip(
            "Size of the pixel neighborhood tracked by optical flow.\n\n"
            "Smaller (7-21): Faster, tracks fine details, may lose track easier\n"
            "Medium (31-71): Balanced - good for most cases\n"
            "Larger (81-201): Very robust, less likely to lose track, slower\n\n"
            "If tracking is unstable, try increasing this value.\n"
            "Values above 100 are recommended for challenging videos."
        )
        flow_window_layout.addWidget(self.flow_window_spin)
        flow_window_layout.addStretch()
        setup_layout.addLayout(flow_window_layout)
        
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
        self.video_widget.click_signal.connect(self.on_click_selected)
        self.video_widget.rectangle_drawn.connect(self.on_rectangle_drawn)
        
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
            self.click_point = None
            self.arena_rectangle = None
            self.click_status_label.setText("Status: Frame loaded - click on animal to see SAM segmentation")
            self.arena_status_label.setText("Arena: Not defined")
            self.click_animal_btn.setEnabled(True)
            self.start_tracking_btn.setEnabled(False)
        else:
            QMessageBox.warning(self, "Error", "Failed to load first frame")
            
    def start_click_mode(self):
        """Enable click mode for animal selection."""
        self.video_widget.drawing_mode = 'click'
        self.click_animal_btn.setEnabled(False)
        self.click_status_label.setText("Status: Click on the animal in the video")
        
    def on_click_selected(self, x: int, y: int):
        """Handle animal selection click and show SAM mask for clicked point."""
        self.click_point = (x, y)
        self.video_widget.drawing_mode = None
        self.click_status_label.setText(f"Status: Generating SAM mask at ({x}, {y})...")
        
        # Run SAM segmentation to show mask overlay for the clicked point
        if self.sam_checkpoint_path and self.video_widget.original_frame is not None:
            try:
                from fnt.videoTracking.sam_tracker_base import SAMTrackerBase
                import torch
                
                # Create temporary tracker just for mask preview
                temp_tracker = SAMTrackerBase(
                    video_path=self.video_paths[self.current_video_idx],
                    sam_checkpoint=self.sam_checkpoint_path,
                    model_type=self.get_model_type_from_checkpoint(),
                    device=self.device_combo.text()
                )
                
                # Initialize SAM
                if temp_tracker.initialize_sam():
                    # Get current frame
                    frame = self.video_widget.original_frame.copy()
                    if not self.video_widget.is_rgb:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    else:
                        frame_rgb = frame.copy()
                    
                    # Initialize tracking to get mask
                    if temp_tracker.initialize_tracking(frame_rgb, (x, y)):
                        # Get the mask
                        mask = temp_tracker.current_mask
                        
                        # Create overlay
                        overlay = frame_rgb.copy()
                        # Color the mask area (semi-transparent green)
                        overlay[mask > 0] = overlay[mask > 0] * 0.5 + np.array([0, 255, 0]) * 0.5
                        
                        # Draw click point
                        cv2.circle(overlay, (x, y), 5, (255, 0, 0), -1)
                        cv2.circle(overlay, (x, y), 7, (255, 255, 255), 2)
                        
                        # Update display
                        self.video_widget.set_frame(overlay, is_rgb=True)
                        self.click_status_label.setText(f"Status: Animal selected at ({x}, {y}) - SAM mask shown in green")
                    else:
                        self.click_status_label.setText(f"Status: Animal selected at ({x}, {y}) - SAM mask failed")
                else:
                    self.click_status_label.setText(f"Status: Animal selected at ({x}, {y}) - SAM initialization failed")
                    
                # Cleanup
                temp_tracker.cleanup()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                
            except Exception as e:
                self.click_status_label.setText(f"Status: Animal selected at ({x}, {y}) - Mask preview error: {str(e)}")
        else:
            self.click_status_label.setText(f"Status: Animal selected at ({x}, {y})")
        
        self.draw_arena_btn.setEnabled(True)
        self.click_animal_btn.setEnabled(True)
        
    def start_draw_rectangle(self):
        """Enable rectangle drawing mode for arena definition."""
        self.video_widget.drawing_mode = 'rectangle'
        self.draw_arena_btn.setEnabled(False)
        self.arena_status_label.setText("Arena: Click first corner, then opposite corner")
        
    def on_rectangle_drawn(self, x: int, y: int, width: int, height: int, corners, rotation_angle: float):
        """Handle arena rectangle drawn."""
        self.arena_rectangle = (x, y, width, height)
        self.arena_corners = corners  # Store rotated corners
        self.arena_rotation = rotation_angle  # Store rotation angle
        self.arena_status_label.setText(f"Arena: {width}x{height} px at ({x}, {y}), rotation: {rotation_angle:.1f}°")
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
        
        # Determine tracking method
        tracking_method = "optical_flow" if self.tracking_method_combo.currentIndex() == 0 else "centroid"
        
        # Create worker thread
        self.tracking_worker = TrackingWorker(
            video_path=video_path,
            sam_checkpoint=self.sam_checkpoint_path,
            click_point=self.click_point,
            arena_rectangle=self.arena_rectangle,  # Changed from arena_circle
            arena_corners=self.arena_corners,  # Rotated corners
            arena_rotation=self.arena_rotation,  # Rotation angle
            sam_update_interval=self.sam_interval_spin.value(),
            model_type=self.get_model_type_from_checkpoint(),  # Auto-detect from filename
            device=self.device_combo.text(),
            flow_window_size=self.flow_window_spin.value(),
            tracking_method=tracking_method
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
