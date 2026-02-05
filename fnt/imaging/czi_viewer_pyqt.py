"""
CZI Viewer - PyQt5 GUI for viewing and processing Zeiss CZI microscopy images.

Follows the FNT pattern: left panel (controls) + right panel (image preview).
"""

import os
import sys
import math
import json
from pathlib import Path
from datetime import datetime
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPointF, QRectF, QPoint, QTimer
from PyQt5.QtGui import (
    QFont, QImage, QPixmap, QPainter, QColor, QPen, QWheelEvent,
    QBrush, QFontMetrics, QTransform, QCursor, QPolygonF, QPainterPath
)
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QGroupBox, QScrollArea,
    QSpinBox, QDoubleSpinBox, QComboBox, QListWidget, QListWidgetItem,
    QCheckBox, QSlider, QStatusBar, QMessageBox, QSizePolicy, QLineEdit,
    QInputDialog, QButtonGroup, QToolButton, QStyle, QStyleOptionButton
)

from .czi_reader import CZIFileReader, CZIImageData, HAS_AICSPYLIBCZI, HAS_AICSIMAGEIO
from .image_processor import (
    CZIImageProcessor, ChannelDisplaySettings, ShapeAnnotation, HAS_SKIMAGE
)


class CZILoadWorker(QThread):
    """Worker thread for loading CZI files."""
    progress = pyqtSignal(str)
    loaded = pyqtSignal(object)
    error = pyqtSignal(str)

    def __init__(self, filepath: str):
        super().__init__()
        self.filepath = filepath

    def run(self):
        try:
            self.progress.emit(f"Loading {Path(self.filepath).name}...")
            reader = CZIFileReader(self.filepath)
            data = reader.load_all_channels()
            self.loaded.emit(data)
        except Exception as e:
            self.error.emit(str(e))


class ImagePreviewWidget(QWidget):
    """
    Custom widget for displaying microscopy images with zoom/pan
    and interactive annotation editing.
    """

    annotation_clicked = pyqtSignal(int, int)  # x, y in image coordinates
    annotation_selected = pyqtSignal(int)  # annotation index
    annotation_modified = pyqtSignal()  # annotation was moved/resized/rotated
    annotation_edit_requested = pyqtSignal(int)  # double-click to edit

    # Shape drawing signals
    shape_drawn = pyqtSignal(object)  # Emits ShapeAnnotation when drawing finishes
    shape_selected = pyqtSignal(int)  # shape index
    shape_modified = pyqtSignal()  # shape was moved/resized

    # ROI drawing signal
    roi_drawn = pyqtSignal(int, int, int, int)  # x, y, w, h in image coords

    # Drawing mode exited signal (Escape pressed)
    drawing_mode_exited = pyqtSignal()

    # Handle types for resize/rotate
    HANDLE_NONE = 0
    HANDLE_TL = 1  # Top-left
    HANDLE_TR = 2  # Top-right
    HANDLE_BL = 3  # Bottom-left
    HANDLE_BR = 4  # Bottom-right
    HANDLE_ROTATE = 5  # Rotation handle (above top center)
    HANDLE_MOVE = 6  # Move the whole annotation
    # Shape-specific handles
    HANDLE_START = 7  # Start point (line/arrow)
    HANDLE_END = 8  # End point (line/arrow)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 400)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)
        self.setFocusPolicy(Qt.StrongFocus)

        # Image state
        self.display_image: Optional[QImage] = None
        self.image_width = 0
        self.image_height = 0

        # View state
        self.zoom_level: float = 1.0
        self.pan_offset = QPointF(0, 0)

        # Interaction state
        self.is_panning: bool = False
        self.last_mouse_pos: Optional[QPoint] = None
        self.annotation_mode: bool = False

        # Annotation interaction state
        self.annotations: List = []  # List of TextAnnotation objects
        self.selected_annotation_idx: int = -1
        self.active_handle: int = self.HANDLE_NONE
        self.drag_start_pos: Optional[QPointF] = None
        self.drag_start_annotation_pos: Optional[Tuple[int, int]] = None
        self.drag_start_rotation: float = 0.0
        self.drag_start_size: Optional[Tuple[int, int]] = None

        # Shape drawing state
        self.shape_annotations: List = []  # List of ShapeAnnotation objects
        self.drawing_mode: str = ""  # "", "arrow", "line", "circle", "ellipse", "rectangle", "freehand"
        self.current_draw_shape: Optional[object] = None  # Shape being drawn
        self.is_drawing: bool = False
        self.selected_shape_idx: int = -1
        self.active_shape_handle: int = self.HANDLE_NONE
        self.drag_start_shape_data: Optional[dict] = None

        # ROI drawing state
        self.roi_mode: bool = False
        self.roi_rect: Optional[Tuple[int, int, int, int]] = None  # x, y, w, h in image coords
        self.roi_start_pos: Optional[Tuple[float, float]] = None

        # Scale bar state
        self.show_scale_bar: bool = True
        self.pixel_size_um: float = 0.0

        # Zoom reference: the zoom level that corresponds to "fit to window" = 100%
        self._fit_zoom_level: float = 1.0

        # Scale bar position as fraction of widget (0.0-1.0), None = default bottom-right
        self.scale_bar_pos: Optional[Tuple[float, float]] = None
        self._scale_bar_widget_rect: Optional[QRectF] = None
        self._dragging_scale_bar: bool = False
        self._scale_bar_drag_offset: Optional[QPointF] = None

        # Drawing tool defaults (set by the parent window)
        self._draw_color: str = "white"
        self._draw_line_width: float = 2.0
        self._draw_line_style: str = "solid"

        # Handle size
        self.handle_size = 8
        self.rotate_handle_distance = 25

        # Double-click detection
        self.last_click_time = 0
        self.last_click_pos = None
        self.double_click_threshold = 300  # ms

        # Styling
        self.setStyleSheet("background-color: #1e1e1e;")

    def set_annotations(self, annotations: List):
        """Set the list of annotations to display."""
        self.annotations = annotations
        self.update()

    def set_shapes(self, shapes: List):
        """Set the list of shape annotations to display."""
        self.shape_annotations = shapes
        self.update()

    def set_image(self, image: np.ndarray):
        """Set image data (RGB numpy array) for display."""
        if image is None:
            self.display_image = None
            self.update()
            return

        h, w = image.shape[:2]
        self.image_width = w
        self.image_height = h

        if len(image.shape) == 3 and image.shape[2] == 3:
            bytes_per_line = 3 * w
            self.display_image = QImage(
                image.data, w, h, bytes_per_line, QImage.Format_RGB888
            ).copy()
        else:
            gray = image if len(image.shape) == 2 else image[:, :, 0]
            rgb = np.stack([gray, gray, gray], axis=2)
            bytes_per_line = 3 * w
            self.display_image = QImage(
                rgb.data, w, h, bytes_per_line, QImage.Format_RGB888
            ).copy()

        self.update()

    def fit_to_window(self):
        """Reset zoom to fit image in widget."""
        if self.display_image is None:
            return

        widget_w = self.width()
        widget_h = self.height()

        scale_x = widget_w / self.image_width
        scale_y = widget_h / self.image_height

        self.zoom_level = min(scale_x, scale_y) * 0.95
        self._fit_zoom_level = self.zoom_level
        self.pan_offset = QPointF(0, 0)
        self.update()

    def _recalc_fit_zoom(self):
        """Recalculate what zoom level corresponds to 'fit to window'."""
        if self.display_image is None or self.image_width == 0 or self.image_height == 0:
            return
        widget_w = self.width()
        widget_h = self.height()
        scale_x = widget_w / self.image_width
        scale_y = widget_h / self.image_height
        self._fit_zoom_level = min(scale_x, scale_y) * 0.95

    def zoom_to_100(self):
        """Set zoom to 100% (fit-to-window level)."""
        self._recalc_fit_zoom()
        self.zoom_level = self._fit_zoom_level
        self.pan_offset = QPointF(0, 0)
        self.update()

    def set_zoom(self, zoom: float):
        """Set zoom level."""
        self.zoom_level = max(0.01, min(50.0, zoom))
        self.update()

    def get_zoom_percent(self) -> int:
        """Get current zoom as percentage relative to fit-to-window (fit = 100%)."""
        if self._fit_zoom_level > 0:
            return int((self.zoom_level / self._fit_zoom_level) * 100)
        return int(self.zoom_level * 100)

    def resizeEvent(self, event):
        """Recalculate fit zoom reference on resize."""
        super().resizeEvent(event)
        self._recalc_fit_zoom()

    def _image_to_widget_coords(self, img_x: float, img_y: float) -> QPointF:
        """Convert image coordinates to widget coordinates."""
        scaled_w = self.image_width * self.zoom_level
        scaled_h = self.image_height * self.zoom_level
        offset_x = (self.width() - scaled_w) / 2 + self.pan_offset.x()
        offset_y = (self.height() - scaled_h) / 2 + self.pan_offset.y()
        return QPointF(offset_x + img_x * self.zoom_level,
                       offset_y + img_y * self.zoom_level)

    def _widget_to_image_coords(self, widget_pos: QPoint) -> Optional[Tuple[float, float]]:
        """Convert widget coordinates to image coordinates."""
        if self.display_image is None:
            return None

        scaled_w = self.image_width * self.zoom_level
        scaled_h = self.image_height * self.zoom_level

        x = (self.width() - scaled_w) / 2 + self.pan_offset.x()
        y = (self.height() - scaled_h) / 2 + self.pan_offset.y()

        img_x = (widget_pos.x() - x) / self.zoom_level
        img_y = (widget_pos.y() - y) / self.zoom_level

        return (img_x, img_y)

    def _get_annotation_bounds(self, ann) -> QRectF:
        """Get the bounding rectangle for an annotation in image coordinates."""
        # Estimate text size if not set
        if ann.width == 0 or ann.height == 0:
            # Approximate based on font size and text length
            ann.width = max(50, len(ann.text) * ann.font_size * 0.6)
            ann.height = ann.font_size * 1.5
        return QRectF(ann.x, ann.y, ann.width, ann.height)

    def _get_annotation_handles(self, ann) -> Dict[int, QRectF]:
        """Get handle rectangles for an annotation in widget coordinates."""
        bounds = self._get_annotation_bounds(ann)
        handles = {}
        hs = self.handle_size

        # Get corners in image coords
        corners = [
            (bounds.left(), bounds.top()),      # TL
            (bounds.right(), bounds.top()),     # TR
            (bounds.left(), bounds.bottom()),   # BL
            (bounds.right(), bounds.bottom()),  # BR
        ]

        # Rotation handle position (above center top)
        center_top = (bounds.center().x(), bounds.top() - self.rotate_handle_distance / self.zoom_level)

        # Apply rotation around annotation center
        cx, cy = bounds.center().x(), bounds.center().y()
        angle_rad = math.radians(ann.rotation)

        def rotate_point(px, py):
            dx, dy = px - cx, py - cy
            rx = dx * math.cos(angle_rad) - dy * math.sin(angle_rad) + cx
            ry = dx * math.sin(angle_rad) + dy * math.cos(angle_rad) + cy
            return rx, ry

        rotated_corners = [rotate_point(c[0], c[1]) for c in corners]
        rotated_rotate = rotate_point(center_top[0], center_top[1])

        # Convert to widget coords and create handle rects
        handle_types = [self.HANDLE_TL, self.HANDLE_TR, self.HANDLE_BL, self.HANDLE_BR]
        for i, (ix, iy) in enumerate(rotated_corners):
            wp = self._image_to_widget_coords(ix, iy)
            handles[handle_types[i]] = QRectF(wp.x() - hs/2, wp.y() - hs/2, hs, hs)

        # Rotation handle
        rwp = self._image_to_widget_coords(rotated_rotate[0], rotated_rotate[1])
        handles[self.HANDLE_ROTATE] = QRectF(rwp.x() - hs/2, rwp.y() - hs/2, hs, hs)

        return handles

    def _hit_test_annotation(self, widget_pos: QPoint) -> Tuple[int, int]:
        """
        Test if a point hits an annotation or its handles.
        Returns (annotation_index, handle_type) or (-1, HANDLE_NONE).
        """
        # Check in reverse order (topmost first)
        for i in range(len(self.annotations) - 1, -1, -1):
            ann = self.annotations[i]
            handles = self._get_annotation_handles(ann)

            # Check handles first (if this annotation is selected)
            if i == self.selected_annotation_idx:
                for handle_type, handle_rect in handles.items():
                    if handle_rect.contains(QPointF(widget_pos)):
                        return (i, handle_type)

            # Check if inside the annotation box
            bounds = self._get_annotation_bounds(ann)
            img_coords = self._widget_to_image_coords(widget_pos)
            if img_coords:
                # Apply inverse rotation to test point
                cx, cy = bounds.center().x(), bounds.center().y()
                angle_rad = -math.radians(ann.rotation)
                dx, dy = img_coords[0] - cx, img_coords[1] - cy
                rx = dx * math.cos(angle_rad) - dy * math.sin(angle_rad) + cx
                ry = dx * math.sin(angle_rad) + dy * math.cos(angle_rad) + cy

                if bounds.contains(QPointF(rx, ry)):
                    return (i, self.HANDLE_MOVE)

        return (-1, self.HANDLE_NONE)

    def paintEvent(self, event):
        """Paint the image with current zoom/pan state and annotations."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.SmoothPixmapTransform)
        painter.setRenderHint(QPainter.Antialiasing)

        # Fill background
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        if self.display_image is None:
            painter.setPen(QColor(100, 100, 100))
            painter.setFont(QFont("Arial", 14))
            painter.drawText(self.rect(), Qt.AlignCenter, "No image loaded")
            return

        # Calculate scaled size
        scaled_w = int(self.image_width * self.zoom_level)
        scaled_h = int(self.image_height * self.zoom_level)

        # Center in widget plus pan offset
        x = (self.width() - scaled_w) / 2 + self.pan_offset.x()
        y = (self.height() - scaled_h) / 2 + self.pan_offset.y()

        # Draw scaled image
        target_rect = QRectF(x, y, scaled_w, scaled_h)
        source_rect = QRectF(0, 0, self.image_width, self.image_height)
        painter.drawImage(target_rect, self.display_image, source_rect)

        # Draw annotations with selection boxes
        self._draw_annotations(painter)

        # Draw shapes
        self._draw_shapes(painter)

        # Draw current shape being drawn
        if self.current_draw_shape is not None:
            self._draw_single_shape(painter, self.current_draw_shape, is_preview=True)

        # Draw ROI rectangle overlay
        if self.roi_rect is not None:
            self._draw_roi_overlay(painter)

        # Draw scale bar
        if self.show_scale_bar and self.pixel_size_um > 0:
            self._draw_scale_bar(painter)

        # Draw mode indicators
        if self.annotation_mode:
            painter.setPen(QPen(QColor(255, 200, 0), 2))
            painter.drawRect(2, 2, self.width() - 4, self.height() - 4)
            painter.setPen(QColor(255, 200, 0))
            painter.setFont(QFont("Arial", 10))
            painter.drawText(10, 20, "Placing text — click to place")
            painter.setPen(QColor(255, 255, 0))
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            painter.drawText(10, 38, "Press Escape to stop drawing")
        elif self.drawing_mode:
            painter.setPen(QPen(QColor(0, 200, 255), 2))
            painter.drawRect(2, 2, self.width() - 4, self.height() - 4)
            painter.setPen(QColor(0, 200, 255))
            painter.setFont(QFont("Arial", 10))
            painter.drawText(10, 20, f"Drawing: {self.drawing_mode}")
            painter.setPen(QColor(255, 255, 0))
            painter.setFont(QFont("Arial", 10, QFont.Bold))
            painter.drawText(10, 38, "Press Escape to stop drawing")
        elif self.roi_mode:
            painter.setPen(QPen(QColor(255, 100, 0), 2))
            painter.drawRect(2, 2, self.width() - 4, self.height() - 4)
            painter.setPen(QColor(255, 100, 0))
            painter.setFont(QFont("Arial", 10))
            painter.drawText(10, 20, "Draw ROI: click and drag a rectangle")

    def _draw_annotations(self, painter: QPainter):
        """Draw all annotations with their text and selection handles."""
        for i, ann in enumerate(self.annotations):
            bounds = self._get_annotation_bounds(ann)
            center = self._image_to_widget_coords(bounds.center().x(), bounds.center().y())

            # Save painter state
            painter.save()

            # Translate to annotation center and rotate
            painter.translate(center)
            painter.rotate(ann.rotation)

            # Draw the text
            font = QFont("Arial", int(ann.font_size * self.zoom_level))
            painter.setFont(font)

            # Get color
            color_map = {
                'white': QColor(255, 255, 255),
                'black': QColor(0, 0, 0),
                'red': QColor(255, 0, 0),
                'green': QColor(0, 255, 0),
                'blue': QColor(0, 0, 255),
                'yellow': QColor(255, 255, 0),
                'cyan': QColor(0, 255, 255),
                'magenta': QColor(255, 0, 255),
                'gray': QColor(128, 128, 128),
            }
            text_color = color_map.get(ann.color, QColor(255, 255, 255))
            painter.setPen(text_color)

            # Calculate text rect centered at origin
            fm = QFontMetrics(font)
            text_rect = fm.boundingRect(ann.text)

            # Update annotation dimensions
            ann.width = text_rect.width() / self.zoom_level + 10
            ann.height = text_rect.height() / self.zoom_level + 6

            # Draw text centered
            scaled_w = bounds.width() * self.zoom_level
            scaled_h = bounds.height() * self.zoom_level
            draw_rect = QRectF(-scaled_w/2, -scaled_h/2, scaled_w, scaled_h)
            painter.drawText(draw_rect, Qt.AlignCenter, ann.text)

            # Draw selection box if selected
            if i == self.selected_annotation_idx:
                # Draw dashed selection rectangle
                pen = QPen(QColor(0, 120, 215), 2, Qt.DashLine)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(draw_rect.adjusted(-3, -3, 3, 3))

            painter.restore()

            # Draw handles if selected (in widget coords, no rotation transform)
            if i == self.selected_annotation_idx:
                self._draw_handles(painter, ann)

    def _draw_handles(self, painter: QPainter, ann):
        """Draw resize and rotate handles for an annotation."""
        handles = self._get_annotation_handles(ann)

        # Draw corner handles
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.setPen(QPen(QColor(0, 120, 215), 1))

        for handle_type, rect in handles.items():
            if handle_type == self.HANDLE_ROTATE:
                # Draw rotation handle as a circle
                painter.setBrush(QBrush(QColor(0, 200, 100)))
                painter.drawEllipse(rect)

                # Draw line from top center to rotation handle
                bounds = self._get_annotation_bounds(ann)
                cx, cy = bounds.center().x(), bounds.center().y()
                top_center = (bounds.center().x(), bounds.top())

                # Rotate the top center point
                angle_rad = math.radians(ann.rotation)
                dx, dy = top_center[0] - cx, top_center[1] - cy
                rx = dx * math.cos(angle_rad) - dy * math.sin(angle_rad) + cx
                ry = dx * math.sin(angle_rad) + dy * math.cos(angle_rad) + cy

                tc_widget = self._image_to_widget_coords(rx, ry)
                painter.setPen(QPen(QColor(0, 200, 100), 1, Qt.DashLine))
                painter.drawLine(tc_widget, rect.center())
            else:
                # Draw corner handles as squares
                painter.setBrush(QBrush(QColor(255, 255, 255)))
                painter.setPen(QPen(QColor(0, 120, 215), 1))
                painter.drawRect(rect)

    def _draw_shapes(self, painter: QPainter):
        """Draw all shape annotations."""
        for i, shape in enumerate(self.shape_annotations):
            is_selected = (i == self.selected_shape_idx)
            self._draw_single_shape(painter, shape, is_selected=is_selected)

    def _draw_single_shape(self, painter: QPainter, shape, is_selected=False, is_preview=False):
        """Draw a single shape annotation."""
        color_map = {
            'white': QColor(255, 255, 255), 'black': QColor(0, 0, 0),
            'red': QColor(255, 0, 0), 'green': QColor(0, 255, 0),
            'blue': QColor(0, 0, 255), 'yellow': QColor(255, 255, 0),
            'cyan': QColor(0, 255, 255), 'magenta': QColor(255, 0, 255),
            'gray': QColor(128, 128, 128),
        }
        color = color_map.get(shape.color, QColor(255, 255, 255))
        if is_preview:
            color.setAlpha(180)

        lw = max(1, int(shape.line_width * self.zoom_level))
        pen = QPen(color, lw)
        if shape.line_style == "dashed":
            pen.setDashPattern([8, 4])
        elif shape.line_style == "dotted":
            pen.setDashPattern([2, 4])
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)

        if shape.shape_type in ("line", "arrow"):
            p1 = self._image_to_widget_coords(shape.x, shape.y)
            p2 = self._image_to_widget_coords(shape.x2, shape.y2)
            painter.drawLine(p1, p2)

            if shape.shape_type == "arrow":
                # Draw arrowhead
                dx = p2.x() - p1.x()
                dy = p2.y() - p1.y()
                length = math.sqrt(dx * dx + dy * dy)
                if length > 0:
                    ux, uy = dx / length, dy / length
                    px, py = -uy, ux
                    hs = shape.arrow_head_size * self.zoom_level
                    tip = p2
                    left = QPointF(tip.x() - ux * hs + px * hs * 0.4,
                                   tip.y() - uy * hs + py * hs * 0.4)
                    right = QPointF(tip.x() - ux * hs - px * hs * 0.4,
                                    tip.y() - uy * hs - py * hs * 0.4)
                    painter.setBrush(QBrush(color))
                    painter.drawPolygon(QPolygonF([tip, left, right]))
                    painter.setBrush(Qt.NoBrush)

            # Draw endpoint handles if selected
            if is_selected:
                self._draw_point_handle(painter, p1)
                self._draw_point_handle(painter, p2)

        elif shape.shape_type == "rectangle":
            tl = self._image_to_widget_coords(shape.x, shape.y)
            w = shape.width * self.zoom_level
            h = shape.height * self.zoom_level
            painter.drawRect(QRectF(tl.x(), tl.y(), w, h))

            if is_selected:
                hs = self.handle_size
                corners = [
                    QPointF(tl.x(), tl.y()),
                    QPointF(tl.x() + w, tl.y()),
                    QPointF(tl.x(), tl.y() + h),
                    QPointF(tl.x() + w, tl.y() + h),
                ]
                for c in corners:
                    self._draw_point_handle(painter, c)

        elif shape.shape_type in ("circle", "ellipse"):
            center = self._image_to_widget_coords(shape.x, shape.y)
            w = shape.width * self.zoom_level
            h = shape.height * self.zoom_level
            painter.drawEllipse(QRectF(center.x() - w/2, center.y() - h/2, w, h))

            if is_selected:
                # Handles on axes
                self._draw_point_handle(painter, QPointF(center.x() + w/2, center.y()))
                self._draw_point_handle(painter, QPointF(center.x() - w/2, center.y()))
                self._draw_point_handle(painter, QPointF(center.x(), center.y() + h/2))
                self._draw_point_handle(painter, QPointF(center.x(), center.y() - h/2))

        elif shape.shape_type == "freehand" and shape.points:
            if len(shape.points) >= 2:
                points = [self._image_to_widget_coords(px, py) for px, py in shape.points]
                painter.drawPolyline(QPolygonF(points))

                if is_selected:
                    self._draw_point_handle(painter, points[0])
                    self._draw_point_handle(painter, points[-1])

    def _draw_point_handle(self, painter: QPainter, pos: QPointF):
        """Draw a small handle square at a position."""
        hs = self.handle_size
        painter.setBrush(QBrush(QColor(255, 255, 255)))
        painter.setPen(QPen(QColor(0, 120, 215), 1))
        painter.drawRect(QRectF(pos.x() - hs/2, pos.y() - hs/2, hs, hs))

    def _draw_roi_overlay(self, painter: QPainter):
        """Draw the ROI rectangle overlay."""
        if self.roi_rect is None:
            return
        x, y, w, h = self.roi_rect
        tl = self._image_to_widget_coords(x, y)
        br = self._image_to_widget_coords(x + w, y + h)
        pen = QPen(QColor(255, 100, 0), 2, Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(255, 100, 0, 40)))
        painter.drawRect(QRectF(tl, br))
        painter.setPen(QColor(255, 100, 0))
        painter.setFont(QFont("Arial", 9))
        painter.drawText(int(tl.x()) + 4, int(tl.y()) - 4, "ROI")

    def _draw_scale_bar(self, painter: QPainter):
        """Draw a scale bar (draggable). Default position: bottom-right corner."""
        if self.pixel_size_um <= 0 or self.display_image is None:
            return

        # Choose a nice scale bar length
        visible_image_width = self.width() / self.zoom_level
        target_um = visible_image_width * self.pixel_size_um * 0.2

        nice_values = [1, 2, 5, 10, 20, 25, 50, 100, 200, 250, 500, 1000, 2000, 5000]
        bar_um = nice_values[0]
        for v in nice_values:
            if v <= target_um:
                bar_um = v
            else:
                break

        bar_pixels = bar_um / self.pixel_size_um
        bar_widget_px = bar_pixels * self.zoom_level

        bar_height = 6
        margin = 20

        if self.scale_bar_pos is not None:
            # Custom position: scale_bar_pos is (x_frac, y_frac) of widget dimensions
            x = self.scale_bar_pos[0] * self.width()
            y = self.scale_bar_pos[1] * self.height()
        else:
            # Default: bottom-right corner
            x = self.width() - margin - bar_widget_px
            y = self.height() - margin - bar_height - 15

        # Cache the widget-space bounding rect for hit-testing
        self._scale_bar_widget_rect = QRectF(x - 4, y - 4, bar_widget_px + 8, 28)

        # Draw bar background (dark outline)
        painter.setPen(QPen(QColor(0, 0, 0), bar_height + 2))
        painter.drawLine(QPointF(x - 1, y), QPointF(x + bar_widget_px + 1, y))

        # Draw bar (white)
        painter.setPen(QPen(QColor(255, 255, 255), bar_height))
        painter.drawLine(QPointF(x, y), QPointF(x + bar_widget_px, y))

        # Draw label
        if bar_um >= 1000:
            label = f"{bar_um / 1000:.0f} mm"
        else:
            label = f"{bar_um:.0f} µm"

        painter.setPen(QColor(255, 255, 255))
        painter.setFont(QFont("Arial", 10, QFont.Bold))
        text_rect = QRectF(x, y + 4, bar_widget_px, 16)
        painter.drawText(text_rect, Qt.AlignCenter, label)

        # Draw subtle drag indicator when hovered
        if self._dragging_scale_bar:
            painter.setPen(QPen(QColor(0, 120, 215, 128), 1, Qt.DashLine))
            painter.setBrush(Qt.NoBrush)
            painter.drawRect(self._scale_bar_widget_rect)

    def _hit_test_shape(self, widget_pos: QPoint) -> Tuple[int, int]:
        """
        Test if a point hits a shape or its handles.
        Returns (shape_index, handle_type) or (-1, HANDLE_NONE).
        """
        threshold = 8  # pixel distance threshold for hit

        for i in range(len(self.shape_annotations) - 1, -1, -1):
            shape = self.shape_annotations[i]
            wp = QPointF(widget_pos)

            if shape.shape_type in ("line", "arrow"):
                p1 = self._image_to_widget_coords(shape.x, shape.y)
                p2 = self._image_to_widget_coords(shape.x2, shape.y2)

                # Check endpoints (handles) if selected
                if i == self.selected_shape_idx:
                    if self._point_distance(wp, p1) < threshold:
                        return (i, self.HANDLE_START)
                    if self._point_distance(wp, p2) < threshold:
                        return (i, self.HANDLE_END)

                # Check line proximity
                dist = self._point_to_line_distance(wp, p1, p2)
                if dist < threshold:
                    return (i, self.HANDLE_MOVE)

            elif shape.shape_type == "rectangle":
                tl = self._image_to_widget_coords(shape.x, shape.y)
                w = shape.width * self.zoom_level
                h = shape.height * self.zoom_level
                rect = QRectF(tl.x(), tl.y(), w, h)

                # Check corners if selected
                if i == self.selected_shape_idx:
                    corners = [
                        (QPointF(tl.x(), tl.y()), self.HANDLE_TL),
                        (QPointF(tl.x() + w, tl.y()), self.HANDLE_TR),
                        (QPointF(tl.x(), tl.y() + h), self.HANDLE_BL),
                        (QPointF(tl.x() + w, tl.y() + h), self.HANDLE_BR),
                    ]
                    for corner, handle in corners:
                        if self._point_distance(wp, corner) < threshold:
                            return (i, handle)

                # Check edge proximity or interior
                expanded = rect.adjusted(-threshold, -threshold, threshold, threshold)
                if expanded.contains(wp):
                    return (i, self.HANDLE_MOVE)

            elif shape.shape_type in ("circle", "ellipse"):
                center = self._image_to_widget_coords(shape.x, shape.y)
                w = shape.width * self.zoom_level
                h = shape.height * self.zoom_level

                # Check axis handles if selected
                if i == self.selected_shape_idx:
                    axis_pts = [
                        (QPointF(center.x() + w/2, center.y()), self.HANDLE_TR),
                        (QPointF(center.x() - w/2, center.y()), self.HANDLE_TL),
                        (QPointF(center.x(), center.y() + h/2), self.HANDLE_BR),
                        (QPointF(center.x(), center.y() - h/2), self.HANDLE_BL),
                    ]
                    for pt, handle in axis_pts:
                        if self._point_distance(wp, pt) < threshold:
                            return (i, handle)

                # Check if near ellipse edge
                if w > 0 and h > 0:
                    dx = (wp.x() - center.x()) / (w / 2)
                    dy = (wp.y() - center.y()) / (h / 2)
                    dist_norm = math.sqrt(dx * dx + dy * dy)
                    if abs(dist_norm - 1.0) < threshold / min(w/2, h/2) or dist_norm <= 1.0:
                        return (i, self.HANDLE_MOVE)

            elif shape.shape_type == "freehand" and shape.points:
                points = [self._image_to_widget_coords(px, py) for px, py in shape.points]
                for j in range(len(points) - 1):
                    dist = self._point_to_line_distance(wp, points[j], points[j + 1])
                    if dist < threshold:
                        return (i, self.HANDLE_MOVE)

        return (-1, self.HANDLE_NONE)

    def _point_distance(self, p1: QPointF, p2: QPointF) -> float:
        """Calculate distance between two points."""
        dx = p1.x() - p2.x()
        dy = p1.y() - p2.y()
        return math.sqrt(dx * dx + dy * dy)

    def _point_to_line_distance(self, point: QPointF, line_start: QPointF, line_end: QPointF) -> float:
        """Calculate minimum distance from point to line segment."""
        dx = line_end.x() - line_start.x()
        dy = line_end.y() - line_start.y()
        length_sq = dx * dx + dy * dy
        if length_sq < 1e-10:
            return self._point_distance(point, line_start)

        t = max(0, min(1, ((point.x() - line_start.x()) * dx + (point.y() - line_start.y()) * dy) / length_sq))
        proj = QPointF(line_start.x() + t * dx, line_start.y() + t * dy)
        return self._point_distance(point, proj)

    def wheelEvent(self, event: QWheelEvent):
        """Handle mouse wheel for zooming."""
        if self.display_image is None:
            return

        mouse_pos = event.pos()
        delta = event.angleDelta().y()
        factor = 1.15 if delta > 0 else 1 / 1.15

        old_zoom = self.zoom_level
        new_zoom = max(0.01, min(50.0, old_zoom * factor))

        widget_center = QPointF(self.width() / 2, self.height() / 2)
        mouse_offset = QPointF(mouse_pos) - widget_center - self.pan_offset

        scale_change = new_zoom / old_zoom
        new_offset = mouse_offset * scale_change

        self.pan_offset = self.pan_offset + mouse_offset - new_offset
        self.zoom_level = new_zoom

        self.update()

    def mousePressEvent(self, event):
        """Handle mouse press for panning, annotation placement, shape drawing, or selection."""
        if event.button() == Qt.LeftButton:
            # Check for double-click
            import time
            current_time = int(time.time() * 1000)
            if (self.last_click_pos is not None and
                self.last_click_pos == event.pos() and
                current_time - self.last_click_time < self.double_click_threshold):
                ann_idx, _ = self._hit_test_annotation(event.pos())
                if ann_idx >= 0:
                    self.annotation_edit_requested.emit(ann_idx)
                    return

            self.last_click_time = current_time
            self.last_click_pos = event.pos()

            img_coords = self._widget_to_image_coords(event.pos())

            # Scale bar dragging (takes priority when visible)
            if (self.show_scale_bar and self._scale_bar_widget_rect is not None and
                    self._scale_bar_widget_rect.contains(QPointF(event.pos()))):
                self._dragging_scale_bar = True
                self._scale_bar_drag_offset = QPointF(event.pos()) - self._scale_bar_widget_rect.topLeft()
                self.update()
                return

            # ROI drawing mode
            if self.roi_mode and img_coords:
                self.roi_start_pos = img_coords
                self.roi_rect = (int(img_coords[0]), int(img_coords[1]), 0, 0)
                self.update()
                return

            # Shape drawing mode
            if self.drawing_mode and img_coords:
                self.is_drawing = True
                ix, iy = img_coords
                if self.drawing_mode in ("line", "arrow"):
                    self.current_draw_shape = ShapeAnnotation(
                        shape_type=self.drawing_mode, x=ix, y=iy, x2=ix, y2=iy,
                        color=self._draw_color, line_width=self._draw_line_width,
                        line_style=self._draw_line_style)
                elif self.drawing_mode == "rectangle":
                    self.current_draw_shape = ShapeAnnotation(
                        shape_type="rectangle", x=ix, y=iy, width=0, height=0,
                        color=self._draw_color, line_width=self._draw_line_width,
                        line_style=self._draw_line_style)
                elif self.drawing_mode in ("circle", "ellipse"):
                    self.current_draw_shape = ShapeAnnotation(
                        shape_type=self.drawing_mode, x=ix, y=iy, width=0, height=0,
                        color=self._draw_color, line_width=self._draw_line_width,
                        line_style=self._draw_line_style)
                elif self.drawing_mode == "freehand":
                    self.current_draw_shape = ShapeAnnotation(
                        shape_type="freehand", x=ix, y=iy, points=[(ix, iy)],
                        color=self._draw_color, line_width=self._draw_line_width,
                        line_style=self._draw_line_style)
                self.update()
                return

            # Annotation placement mode
            if self.annotation_mode and self.display_image is not None:
                if img_coords and 0 <= img_coords[0] < self.image_width and 0 <= img_coords[1] < self.image_height:
                    self.annotation_clicked.emit(int(img_coords[0]), int(img_coords[1]))
                return

            # Check if clicking on a shape
            shape_idx, shape_handle = self._hit_test_shape(event.pos())
            if shape_idx >= 0:
                self.selected_shape_idx = shape_idx
                self.selected_annotation_idx = -1
                self.active_shape_handle = shape_handle
                self.drag_start_pos = QPointF(event.pos())
                s = self.shape_annotations[shape_idx]
                self.drag_start_shape_data = {
                    'x': s.x, 'y': s.y, 'x2': s.x2, 'y2': s.y2,
                    'width': s.width, 'height': s.height,
                    'points': list(s.points) if s.points else []
                }
                self.shape_selected.emit(shape_idx)
                self.update()
                return

            # Check if clicking on an annotation or handle
            ann_idx, handle = self._hit_test_annotation(event.pos())
            if ann_idx >= 0:
                self.selected_annotation_idx = ann_idx
                self.selected_shape_idx = -1
                self.active_handle = handle
                self.drag_start_pos = QPointF(event.pos())

                ann = self.annotations[ann_idx]
                self.drag_start_annotation_pos = (ann.x, ann.y)
                self.drag_start_rotation = ann.rotation
                self.drag_start_size = (ann.width, ann.height)

                self.annotation_selected.emit(ann_idx)
                self.update()
                return

            # Deselect all and start panning
            if self.selected_annotation_idx >= 0 or self.selected_shape_idx >= 0:
                self.selected_annotation_idx = -1
                self.selected_shape_idx = -1
                self.update()

            self.is_panning = True
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        """Handle mouse drag for panning, annotation/shape manipulation, or drawing."""
        # Scale bar dragging — store as widget-relative fractions (0.0-1.0)
        if self._dragging_scale_bar:
            new_top_left = QPointF(event.pos()) - self._scale_bar_drag_offset
            x_frac = (new_top_left.x() + 4) / self.width()
            y_frac = (new_top_left.y() + 4) / self.height()
            self.scale_bar_pos = (max(0.0, min(1.0, x_frac)), max(0.0, min(1.0, y_frac)))
            self.update()
            return

        # ROI dragging
        if self.roi_mode and self.roi_start_pos is not None:
            img_coords = self._widget_to_image_coords(event.pos())
            if img_coords:
                x0, y0 = self.roi_start_pos
                x1, y1 = img_coords
                rx = int(min(x0, x1))
                ry = int(min(y0, y1))
                rw = int(abs(x1 - x0))
                rh = int(abs(y1 - y0))
                self.roi_rect = (rx, ry, rw, rh)
                self.update()
            return

        # Shape drawing in progress
        if self.is_drawing and self.current_draw_shape is not None:
            img_coords = self._widget_to_image_coords(event.pos())
            if img_coords:
                ix, iy = img_coords
                shape = self.current_draw_shape
                if shape.shape_type in ("line", "arrow"):
                    shape.x2 = ix
                    shape.y2 = iy
                elif shape.shape_type == "rectangle":
                    shape.width = ix - shape.x
                    shape.height = iy - shape.y
                elif shape.shape_type in ("circle", "ellipse"):
                    dx = ix - shape.x
                    dy = iy - shape.y
                    if shape.shape_type == "circle":
                        r = math.sqrt(dx * dx + dy * dy)
                        shape.width = r * 2
                        shape.height = r * 2
                    else:
                        shape.width = abs(dx) * 2
                        shape.height = abs(dy) * 2
                elif shape.shape_type == "freehand":
                    shape.points.append((ix, iy))
                self.update()
            return

        if self.is_panning and self.last_mouse_pos is not None:
            delta = event.pos() - self.last_mouse_pos
            self.pan_offset += QPointF(delta)
            self.last_mouse_pos = event.pos()
            self.update()

        elif self.active_shape_handle != self.HANDLE_NONE and self.selected_shape_idx >= 0:
            # Manipulating a shape
            shape = self.shape_annotations[self.selected_shape_idx]
            current_pos = QPointF(event.pos())
            delta = current_pos - self.drag_start_pos
            delta_img_x = delta.x() / self.zoom_level
            delta_img_y = delta.y() / self.zoom_level
            d = self.drag_start_shape_data

            if self.active_shape_handle == self.HANDLE_MOVE:
                if shape.shape_type in ("line", "arrow"):
                    shape.x = d['x'] + delta_img_x
                    shape.y = d['y'] + delta_img_y
                    shape.x2 = d['x2'] + delta_img_x
                    shape.y2 = d['y2'] + delta_img_y
                elif shape.shape_type == "rectangle":
                    shape.x = d['x'] + delta_img_x
                    shape.y = d['y'] + delta_img_y
                elif shape.shape_type in ("circle", "ellipse"):
                    shape.x = d['x'] + delta_img_x
                    shape.y = d['y'] + delta_img_y
                elif shape.shape_type == "freehand" and d['points']:
                    shape.points = [(px + delta_img_x, py + delta_img_y) for px, py in d['points']]
            elif self.active_shape_handle == self.HANDLE_START:
                img_coords = self._widget_to_image_coords(event.pos())
                if img_coords:
                    shape.x, shape.y = img_coords
            elif self.active_shape_handle == self.HANDLE_END:
                img_coords = self._widget_to_image_coords(event.pos())
                if img_coords:
                    shape.x2, shape.y2 = img_coords
            elif self.active_shape_handle in (self.HANDLE_TL, self.HANDLE_TR, self.HANDLE_BL, self.HANDLE_BR):
                img_coords = self._widget_to_image_coords(event.pos())
                if img_coords:
                    ix, iy = img_coords
                    if shape.shape_type == "rectangle":
                        if self.active_shape_handle == self.HANDLE_BR:
                            shape.width = ix - shape.x
                            shape.height = iy - shape.y
                        elif self.active_shape_handle == self.HANDLE_TL:
                            shape.width = (d['x'] + d['width']) - ix
                            shape.height = (d['y'] + d['height']) - iy
                            shape.x = ix
                            shape.y = iy
                        elif self.active_shape_handle == self.HANDLE_TR:
                            shape.width = ix - shape.x
                            shape.height = (d['y'] + d['height']) - iy
                            shape.y = iy
                        elif self.active_shape_handle == self.HANDLE_BL:
                            shape.width = (d['x'] + d['width']) - ix
                            shape.x = ix
                            shape.height = iy - shape.y
                    elif shape.shape_type in ("circle", "ellipse"):
                        dx = abs(ix - shape.x)
                        dy = abs(iy - shape.y)
                        shape.width = dx * 2
                        shape.height = dy * 2

            self.shape_modified.emit()
            self.update()

        elif self.active_handle != self.HANDLE_NONE and self.selected_annotation_idx >= 0:
            ann = self.annotations[self.selected_annotation_idx]
            current_pos = QPointF(event.pos())

            if self.active_handle == self.HANDLE_MOVE:
                delta = current_pos - self.drag_start_pos
                delta_img_x = delta.x() / self.zoom_level
                delta_img_y = delta.y() / self.zoom_level
                ann.x = int(self.drag_start_annotation_pos[0] + delta_img_x)
                ann.y = int(self.drag_start_annotation_pos[1] + delta_img_y)

            elif self.active_handle == self.HANDLE_ROTATE:
                bounds = self._get_annotation_bounds(ann)
                center = self._image_to_widget_coords(bounds.center().x(), bounds.center().y())
                dx = current_pos.x() - center.x()
                dy = current_pos.y() - center.y()
                angle = math.degrees(math.atan2(dx, -dy))
                ann.rotation = angle

            elif self.active_handle in [self.HANDLE_TL, self.HANDLE_TR, self.HANDLE_BL, self.HANDLE_BR]:
                delta = current_pos - self.drag_start_pos
                scale_factor = 1.0 + (delta.x() + delta.y()) / 200.0
                scale_factor = max(0.1, scale_factor)
                original_font_size = self.drag_start_size[1] if self.drag_start_size else 24
                ann.font_size = max(4, int(original_font_size * scale_factor))

            self.annotation_modified.emit()
            self.update()

        else:
            # Update cursor based on what's under the mouse
            shape_idx, shape_handle = self._hit_test_shape(event.pos())
            if shape_handle == self.HANDLE_MOVE:
                self.setCursor(Qt.SizeAllCursor)
                return
            elif shape_handle in (self.HANDLE_START, self.HANDLE_END, self.HANDLE_TL, self.HANDLE_TR, self.HANDLE_BL, self.HANDLE_BR):
                self.setCursor(Qt.CrossCursor)
                return

            ann_idx, handle = self._hit_test_annotation(event.pos())
            if handle == self.HANDLE_ROTATE:
                self.setCursor(Qt.CrossCursor)
            elif handle == self.HANDLE_MOVE:
                self.setCursor(Qt.SizeAllCursor)
            elif handle in [self.HANDLE_TL, self.HANDLE_BR]:
                self.setCursor(Qt.SizeFDiagCursor)
            elif handle in [self.HANDLE_TR, self.HANDLE_BL]:
                self.setCursor(Qt.SizeBDiagCursor)
            else:
                self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() == Qt.LeftButton:
            # Finish scale bar dragging
            if self._dragging_scale_bar:
                self._dragging_scale_bar = False
                self.update()
                return

            # Finish ROI drawing
            if self.roi_mode and self.roi_start_pos is not None:
                self.roi_start_pos = None
                if self.roi_rect and self.roi_rect[2] > 2 and self.roi_rect[3] > 2:
                    x, y, w, h = self.roi_rect
                    self.roi_drawn.emit(x, y, w, h)
                self.roi_mode = False
                self.update()
                return

            # Finish shape drawing
            if self.is_drawing and self.current_draw_shape is not None:
                shape = self.current_draw_shape
                # Validate shape has some size
                valid = False
                if shape.shape_type in ("line", "arrow"):
                    dx = shape.x2 - shape.x
                    dy = shape.y2 - shape.y
                    valid = math.sqrt(dx*dx + dy*dy) > 3
                elif shape.shape_type == "rectangle":
                    valid = abs(shape.width) > 3 and abs(shape.height) > 3
                    # Normalize negative dimensions
                    if shape.width < 0:
                        shape.x += shape.width
                        shape.width = abs(shape.width)
                    if shape.height < 0:
                        shape.y += shape.height
                        shape.height = abs(shape.height)
                elif shape.shape_type in ("circle", "ellipse"):
                    valid = shape.width > 3 and shape.height > 3
                elif shape.shape_type == "freehand":
                    valid = len(shape.points) > 2

                if valid:
                    self.shape_drawn.emit(shape)

                self.current_draw_shape = None
                self.is_drawing = False
                self.update()
                return

            self.is_panning = False
            self.last_mouse_pos = None
            self.active_handle = self.HANDLE_NONE
            self.active_shape_handle = self.HANDLE_NONE
            self.drag_start_pos = None
            self.drag_start_shape_data = None
            self.setCursor(Qt.ArrowCursor)

    def keyPressEvent(self, event):
        """Handle key presses for deleting annotations/shapes."""
        if event.key() in [Qt.Key_Delete, Qt.Key_Backspace]:
            if self.selected_shape_idx >= 0:
                self.shape_selected.emit(-1)  # Special signal for delete
            elif self.selected_annotation_idx >= 0:
                self.annotation_selected.emit(-1)  # Special signal for delete
        elif event.key() == Qt.Key_Escape:
            # Cancel current drawing or mode
            exited = False
            if self.is_drawing:
                self.current_draw_shape = None
                self.is_drawing = False
                self.update()
                exited = True
            if self.drawing_mode:
                self.drawing_mode = ""
                self.update()
                exited = True
            if self.roi_mode:
                self.roi_mode = False
                self.roi_start_pos = None
                self.update()
                exited = True
            if self.annotation_mode:
                self.annotation_mode = False
                self.update()
                exited = True
            if exited:
                self.drawing_mode_exited.emit()
        event.accept()

    def select_annotation(self, idx: int):
        """Programmatically select an annotation."""
        self.selected_annotation_idx = idx
        self.update()


class CheckmarkCheckBox(QCheckBox):
    """Custom checkbox that draws an actual ✓ checkmark inside the indicator."""

    def paintEvent(self, event):
        """Override paint to draw a visible checkmark."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Get the indicator rect using style
        opt = QStyleOptionButton()
        self.initStyleOption(opt)
        indicator_rect = self.style().subElementRect(
            QStyle.SE_CheckBoxIndicator, opt, self
        )

        # Draw indicator background
        r = indicator_rect
        painter.setPen(QPen(QColor("#3f3f3f") if not self.isChecked() else QColor("#0078d4"), 1))
        painter.setBrush(QBrush(QColor("#0078d4") if self.isChecked() else QColor("#1e1e1e")))
        painter.drawRoundedRect(r.adjusted(0, 0, -1, -1), 2, 2)

        # Draw checkmark if checked
        if self.isChecked():
            pen = QPen(QColor(255, 255, 255), 2)
            pen.setCapStyle(Qt.RoundCap)
            pen.setJoinStyle(Qt.RoundJoin)
            painter.setPen(pen)
            path = QPainterPath()
            # Checkmark path within the indicator rect
            cx = r.center().x()
            cy = r.center().y()
            hw = r.width() * 0.3
            hh = r.height() * 0.3
            path.moveTo(cx - hw, cy)
            path.lineTo(cx - hw * 0.2, cy + hh)
            path.lineTo(cx + hw, cy - hh)
            painter.drawPath(path)

        # Draw the label text
        text_rect = self.style().subElementRect(
            QStyle.SE_CheckBoxContents, opt, self
        )
        if self.text():
            painter.setPen(QColor("#cccccc"))
            painter.setFont(self.font())
            painter.drawText(text_rect, Qt.AlignVCenter | Qt.AlignLeft, self.text())

        painter.end()


class DualHandleSlider(QWidget):
    """Custom dual-handle range slider for threshold control."""

    range_changed = pyqtSignal(float, float)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(22)
        self.setMinimumWidth(80)

        self._min_val = 0
        self._max_val = 1000
        self._low = 0
        self._high = 1000
        self._handle_radius = 6
        self._dragging = None  # "low", "high", or None
        self._enabled = False
        self.setMouseTracking(True)

    @property
    def low_value(self) -> float:
        return self._low / self._max_val

    @property
    def high_value(self) -> float:
        return self._high / self._max_val

    def set_range(self, low: float, high: float):
        self._low = int(low * self._max_val)
        self._high = int(high * self._max_val)
        self.update()

    def set_slider_enabled(self, enabled: bool):
        self._enabled = enabled
        self.update()

    def _val_to_x(self, val):
        margin = self._handle_radius + 2
        usable = self.width() - 2 * margin
        return margin + (val / self._max_val) * usable

    def _x_to_val(self, x):
        margin = self._handle_radius + 2
        usable = self.width() - 2 * margin
        val = ((x - margin) / usable) * self._max_val
        return max(self._min_val, min(self._max_val, int(val)))

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        h = self.height()
        track_y = h // 2
        track_h = 4

        # Track background
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor(30, 30, 30))
        painter.drawRoundedRect(
            self._handle_radius, track_y - track_h // 2,
            self.width() - 2 * self._handle_radius, track_h, 2, 2
        )

        if self._enabled:
            # Highlighted range between handles
            lx = self._val_to_x(self._low)
            hx = self._val_to_x(self._high)
            painter.setBrush(QColor(0, 120, 212))
            painter.drawRect(int(lx), track_y - track_h // 2, int(hx - lx), track_h)

            # Low handle
            painter.setBrush(QColor(0, 120, 212))
            painter.setPen(QPen(QColor(200, 200, 200), 1))
            painter.drawEllipse(QPointF(lx, track_y), self._handle_radius, self._handle_radius)

            # High handle
            painter.drawEllipse(QPointF(hx, track_y), self._handle_radius, self._handle_radius)
        else:
            # Grayed out
            painter.setBrush(QColor(60, 60, 60))
            lx = self._val_to_x(self._low)
            hx = self._val_to_x(self._high)
            painter.setPen(Qt.NoPen)
            painter.drawEllipse(QPointF(lx, track_y), self._handle_radius, self._handle_radius)
            painter.drawEllipse(QPointF(hx, track_y), self._handle_radius, self._handle_radius)

    def mousePressEvent(self, event):
        if not self._enabled:
            return
        x = event.pos().x()
        lx = self._val_to_x(self._low)
        hx = self._val_to_x(self._high)
        # Check which handle is closer
        if abs(x - lx) < abs(x - hx) and abs(x - lx) < self._handle_radius * 2:
            self._dragging = "low"
        elif abs(x - hx) < self._handle_radius * 2:
            self._dragging = "high"
        elif abs(x - lx) < self._handle_radius * 2:
            self._dragging = "low"

    def mouseMoveEvent(self, event):
        if not self._enabled or self._dragging is None:
            return
        val = self._x_to_val(event.pos().x())
        if self._dragging == "low":
            self._low = min(val, self._high)
        elif self._dragging == "high":
            self._high = max(val, self._low)
        self.update()
        self.range_changed.emit(self.low_value, self.high_value)

    def mouseReleaseEvent(self, event):
        self._dragging = None


class ChannelControlWidget(QWidget):
    """Widget for controlling a single channel's display settings."""

    settings_changed = pyqtSignal()

    def __init__(self, channel_idx: int, channel_name: str,
                 suggested_color: str = "gray", parent=None):
        super().__init__(parent)
        self.channel_idx = channel_idx

        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 4, 0, 4)
        main_layout.setSpacing(2)

        label_width = 70
        val_width = 30
        label_style = "font-size: 10px; color: #cccccc;"

        # Row 0: Channel name (bold, own row above checkbox)
        name_label = QLabel(channel_name)
        name_label.setStyleSheet("font-size: 11px; font-weight: bold; color: #ffffff;")
        main_layout.addWidget(name_label)

        # Row 1: Enable checkbox + color combo
        row1 = QHBoxLayout()
        row1.setSpacing(4)
        self.enabled_cb = CheckmarkCheckBox()
        self.enabled_cb.setChecked(True)
        self.enabled_cb.stateChanged.connect(self._on_change)
        row1.addWidget(self.enabled_cb)
        row1.addStretch()

        self.color_combo = QComboBox()
        self.color_combo.addItems(['green', 'magenta', 'cyan', 'red', 'blue', 'yellow', 'gray'])
        self.color_combo.setCurrentText(suggested_color)
        self.color_combo.currentTextChanged.connect(self._on_change)
        self.color_combo.setFixedWidth(80)
        self.color_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        row1.addWidget(self.color_combo)
        main_layout.addLayout(row1)

        # Row 2: Brightness slider
        row2 = QHBoxLayout()
        row2.setSpacing(4)
        lbl = QLabel("Brightness:")
        lbl.setFixedWidth(label_width)
        lbl.setStyleSheet(label_style)
        row2.addWidget(lbl)
        self.brightness_slider = QSlider(Qt.Horizontal)
        self.brightness_slider.setRange(0, 300)
        self.brightness_slider.setValue(100)
        self.brightness_slider.setFixedHeight(16)
        self.brightness_slider.valueChanged.connect(self._on_change)
        row2.addWidget(self.brightness_slider)
        self.brightness_val = QLabel("1.0")
        self.brightness_val.setFixedWidth(val_width)
        self.brightness_val.setStyleSheet(label_style)
        row2.addWidget(self.brightness_val)
        main_layout.addLayout(row2)

        # Row 3: Contrast slider
        row3 = QHBoxLayout()
        row3.setSpacing(4)
        lbl = QLabel("Contrast:")
        lbl.setFixedWidth(label_width)
        lbl.setStyleSheet(label_style)
        row3.addWidget(lbl)
        self.contrast_slider = QSlider(Qt.Horizontal)
        self.contrast_slider.setRange(0, 300)
        self.contrast_slider.setValue(100)
        self.contrast_slider.setFixedHeight(16)
        self.contrast_slider.valueChanged.connect(self._on_change)
        row3.addWidget(self.contrast_slider)
        self.contrast_val = QLabel("1.0")
        self.contrast_val.setFixedWidth(val_width)
        self.contrast_val.setStyleSheet(label_style)
        row3.addWidget(self.contrast_val)
        main_layout.addLayout(row3)

        # Row 4: Gamma slider (above Sharpness per user preference)
        row4 = QHBoxLayout()
        row4.setSpacing(4)
        lbl = QLabel("Gamma:")
        lbl.setFixedWidth(label_width)
        lbl.setStyleSheet(label_style)
        row4.addWidget(lbl)
        self.gamma_slider = QSlider(Qt.Horizontal)
        self.gamma_slider.setRange(10, 300)  # 0.1-3.0
        self.gamma_slider.setValue(100)
        self.gamma_slider.setFixedHeight(16)
        self.gamma_slider.valueChanged.connect(self._on_change)
        row4.addWidget(self.gamma_slider)
        self.gamma_val = QLabel("1.0")
        self.gamma_val.setFixedWidth(val_width)
        self.gamma_val.setStyleSheet(label_style)
        row4.addWidget(self.gamma_val)
        main_layout.addLayout(row4)

        # Row 5: Sharpness slider
        row5 = QHBoxLayout()
        row5.setSpacing(4)
        lbl = QLabel("Sharpness:")
        lbl.setFixedWidth(label_width)
        lbl.setStyleSheet(label_style)
        row5.addWidget(lbl)
        self.sharpness_slider = QSlider(Qt.Horizontal)
        self.sharpness_slider.setRange(0, 500)  # 0-5.0
        self.sharpness_slider.setValue(0)
        self.sharpness_slider.setFixedHeight(16)
        self.sharpness_slider.valueChanged.connect(self._on_change)
        row5.addWidget(self.sharpness_slider)
        self.sharpness_val = QLabel("0.0")
        self.sharpness_val.setFixedWidth(val_width)
        self.sharpness_val.setStyleSheet(label_style)
        row5.addWidget(self.sharpness_val)
        main_layout.addLayout(row5)

        # Row 6: Threshold — dual-handle range slider with enable checkbox
        row6 = QHBoxLayout()
        row6.setSpacing(4)
        self.thresh_cb = CheckmarkCheckBox("Threshold:")
        self.thresh_cb.setChecked(False)
        self.thresh_cb.setStyleSheet("font-size: 10px;")
        self.thresh_cb.stateChanged.connect(self._on_thresh_toggle)
        row6.addWidget(self.thresh_cb)
        self.thresh_slider = DualHandleSlider()
        self.thresh_slider.range_changed.connect(self._on_thresh_changed)
        row6.addWidget(self.thresh_slider)
        self.thresh_range_label = QLabel("0.00–1.00")
        self.thresh_range_label.setFixedWidth(60)
        self.thresh_range_label.setStyleSheet("font-size: 9px; color: #aaaaaa;")
        row6.addWidget(self.thresh_range_label)
        main_layout.addLayout(row6)

        # Row 7: Background subtraction — method combo + radius slider
        row7 = QHBoxLayout()
        row7.setSpacing(4)
        lbl = QLabel("BG Sub:")
        lbl.setFixedWidth(50)
        lbl.setStyleSheet(label_style)
        row7.addWidget(lbl)
        self.bg_method_combo = QComboBox()
        self.bg_method_combo.addItems(["None", "Rolling Ball", "Gaussian"])
        self.bg_method_combo.setFixedWidth(90)
        self.bg_method_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        self.bg_method_combo.currentTextChanged.connect(self._on_bg_method_changed)
        row7.addWidget(self.bg_method_combo)
        row7.addStretch()
        main_layout.addLayout(row7)

        # Row 7b: Radius slider (hidden by default, shown for Rolling Ball / Gaussian)
        self.bg_radius_row = QWidget()
        bg_r_layout = QHBoxLayout(self.bg_radius_row)
        bg_r_layout.setContentsMargins(0, 0, 0, 0)
        bg_r_layout.setSpacing(4)
        lbl = QLabel("Radius:")
        lbl.setFixedWidth(50)
        lbl.setStyleSheet(label_style)
        bg_r_layout.addWidget(lbl)
        self.bg_radius_slider = QSlider(Qt.Horizontal)
        self.bg_radius_slider.setRange(1, 500)
        self.bg_radius_slider.setValue(50)
        self.bg_radius_slider.setFixedHeight(16)
        self.bg_radius_slider.setToolTip(
            "Radius should be larger than the largest foreground object.\n"
            "Typical values: 50-200 for cells, 200-500 for large structures."
        )
        # Debounce: update label immediately but delay recompute by 400ms
        self._bg_radius_debounce_timer = QTimer()
        self._bg_radius_debounce_timer.setSingleShot(True)
        self._bg_radius_debounce_timer.setInterval(400)
        self._bg_radius_debounce_timer.timeout.connect(self._on_bg_radius_debounce_fire)
        self.bg_radius_slider.valueChanged.connect(self._on_bg_radius_slider_moved)
        bg_r_layout.addWidget(self.bg_radius_slider)
        self.bg_radius_val = QLabel("50")
        self.bg_radius_val.setFixedWidth(val_width)
        self.bg_radius_val.setStyleSheet(label_style)
        bg_r_layout.addWidget(self.bg_radius_val)
        main_layout.addWidget(self.bg_radius_row)
        self.bg_radius_row.hide()

        # Separator line
        sep = QWidget()
        sep.setFixedHeight(1)
        sep.setStyleSheet("background-color: #3f3f3f;")
        main_layout.addWidget(sep)

    def _on_change(self):
        """Update display labels and emit settings_changed."""
        self.brightness_val.setText(f"{self.brightness_slider.value() / 100:.1f}")
        self.contrast_val.setText(f"{self.contrast_slider.value() / 100:.1f}")
        self.sharpness_val.setText(f"{self.sharpness_slider.value() / 100:.1f}")
        self.gamma_val.setText(f"{self.gamma_slider.value() / 100:.1f}")
        self.bg_radius_val.setText(str(self.bg_radius_slider.value()))
        self.settings_changed.emit()

    def _on_bg_radius_slider_moved(self, value):
        """Update label immediately but debounce the expensive recompute."""
        self.bg_radius_val.setText(str(value))
        self._bg_radius_debounce_timer.start()

    def _on_bg_radius_debounce_fire(self):
        """Called after debounce timer — trigger actual recompute."""
        self.settings_changed.emit()

    def _on_thresh_toggle(self, state):
        enabled = bool(state)
        self.thresh_slider.set_slider_enabled(enabled)
        self._on_change()

    def _on_thresh_changed(self, low: float, high: float):
        self.thresh_range_label.setText(f"{low:.2f}–{high:.2f}")
        self.settings_changed.emit()

    def _on_bg_method_changed(self, method_text: str):
        is_radius = method_text in ("Rolling Ball", "Gaussian")
        self.bg_radius_row.setVisible(is_radius)
        self._on_change()

    def get_settings(self) -> ChannelDisplaySettings:
        """Get current settings for this channel."""
        method_map = {
            "None": "none",
            "Rolling Ball": "rolling_ball",
            "Gaussian": "gaussian",
        }
        return ChannelDisplaySettings(
            enabled=self.enabled_cb.isChecked(),
            color=self.color_combo.currentText(),
            brightness=self.brightness_slider.value() / 100.0,
            contrast=self.contrast_slider.value() / 100.0,
            sharpness=self.sharpness_slider.value() / 100.0,
            gamma=self.gamma_slider.value() / 100.0,
            threshold_enabled=self.thresh_cb.isChecked(),
            threshold_low=self.thresh_slider.low_value,
            threshold_high=self.thresh_slider.high_value,
            bg_subtract_method=method_map.get(self.bg_method_combo.currentText(), "none"),
            bg_subtract_radius=float(self.bg_radius_slider.value()),
        )

    def reset(self):
        """Reset all per-channel controls to defaults."""
        self.brightness_slider.setValue(100)
        self.contrast_slider.setValue(100)
        self.sharpness_slider.setValue(0)
        self.gamma_slider.setValue(100)
        self.thresh_cb.setChecked(False)
        self.thresh_slider.set_range(0.0, 1.0)
        self.thresh_range_label.setText("0.00–1.00")
        self.bg_method_combo.setCurrentText("None")
        self.bg_radius_slider.setValue(50)

    def set_enabled(self, enabled: bool):
        self.enabled_cb.setChecked(enabled)

    def set_color(self, color: str):
        self.color_combo.setCurrentText(color)

    def apply_settings(self, settings: ChannelDisplaySettings):
        """Apply saved settings to this channel's controls (used for settings preservation)."""
        # Block signals during restoration to avoid redundant recomputes
        self.blockSignals(True)
        self.enabled_cb.setChecked(settings.enabled)
        self.color_combo.setCurrentText(settings.color)
        self.brightness_slider.setValue(int(settings.brightness * 100))
        self.contrast_slider.setValue(int(settings.contrast * 100))
        self.gamma_slider.setValue(int(settings.gamma * 100))
        self.sharpness_slider.setValue(int(settings.sharpness * 100))
        self.thresh_cb.setChecked(settings.threshold_enabled)
        self.thresh_slider.set_range(settings.threshold_low, settings.threshold_high)
        self.thresh_range_label.setText(f"{settings.threshold_low:.2f}–{settings.threshold_high:.2f}")

        # BG subtraction settings
        method_reverse = {
            "none": "None", "rolling_ball": "Rolling Ball",
            "gaussian": "Gaussian",
        }
        self.bg_method_combo.setCurrentText(method_reverse.get(settings.bg_subtract_method, "None"))
        self.bg_radius_slider.setValue(int(settings.bg_subtract_radius))
        self._bg_roi_value = settings.bg_subtract_roi_value
        self.bg_roi_value_label.setText(
            f"ROI: {settings.bg_subtract_roi_value:.4f}" if settings.bg_subtract_roi_value else "ROI: —"
        )

        self.blockSignals(False)
        # Update display labels
        self._on_change()


class CZIViewerWindow(QMainWindow):
    """
    Main window for CZI microscopy image viewer.
    """

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FNT CZI Viewer")
        self.setMinimumSize(1000, 700)
        self.resize(1400, 900)

        # State
        self.czi_files: List[str] = []
        self.current_file_idx: int = 0
        self.image_data: Optional[CZIImageData] = None
        self.processor = CZIImageProcessor()
        self.channel_controls: Dict[int, ChannelControlWidget] = {}
        self.load_worker: Optional[CZILoadWorker] = None

        # Per-file settings cache: filepath -> {channel_idx: ChannelDisplaySettings}
        self._file_settings_cache: Dict[str, Dict[int, ChannelDisplaySettings]] = {}

        self._setup_ui()
        self._apply_styles()

    def _setup_ui(self):
        """Setup main UI layout."""
        central = QWidget()
        main_layout = QHBoxLayout(central)
        main_layout.setSpacing(10)
        main_layout.setContentsMargins(10, 10, 10, 10)

        # Left scrollable panel
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setFixedWidth(340)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setSpacing(10)

        # Build sections
        self._create_input_section(left_layout)
        self._create_channel_section(left_layout)
        self._create_drawing_tools_section(left_layout)
        self._create_export_section(left_layout)

        left_layout.addStretch()
        left_scroll.setWidget(left_widget)

        # Right preview panel
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(5)

        self.preview = ImagePreviewWidget()
        self.preview.annotation_clicked.connect(self._on_annotation_click)
        self.preview.annotation_selected.connect(self._on_annotation_selected)
        self.preview.annotation_modified.connect(self._on_annotation_modified)
        self.preview.annotation_edit_requested.connect(self._on_annotation_edit)
        self.preview.shape_drawn.connect(self._on_shape_drawn)
        self.preview.shape_selected.connect(self._on_shape_selected)
        self.preview.shape_modified.connect(self._on_shape_modified)
        self.preview.roi_drawn.connect(self._on_roi_drawn)
        self.preview.drawing_mode_exited.connect(self._on_drawing_mode_exited)
        right_layout.addWidget(self.preview, 1)

        # Zoom controls below preview (right-aligned)
        zoom_bar = QHBoxLayout()
        zoom_bar.addStretch()

        self.btn_zoom_out = QPushButton("-")
        self.btn_zoom_out.setFixedWidth(30)
        self.btn_zoom_out.clicked.connect(self._zoom_out)
        zoom_bar.addWidget(self.btn_zoom_out)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setMinimumWidth(50)
        self.zoom_label.setAlignment(Qt.AlignCenter)
        zoom_bar.addWidget(self.zoom_label)

        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setFixedWidth(30)
        self.btn_zoom_in.clicked.connect(self._zoom_in)
        zoom_bar.addWidget(self.btn_zoom_in)

        self.btn_fit = QPushButton("Fit")
        self.btn_fit.setFixedWidth(50)
        self.btn_fit.clicked.connect(self._fit_to_window)
        zoom_bar.addWidget(self.btn_fit)

        right_layout.addLayout(zoom_bar)

        # Add to main
        main_layout.addWidget(left_scroll)
        main_layout.addWidget(right_panel, 1)

        self.setCentralWidget(central)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Ready - Load a CZI file to begin")

    def _create_input_section(self, layout):
        """Create file input section."""
        group = QGroupBox("Input")
        group_layout = QVBoxLayout()

        # Buttons row
        btn_layout = QHBoxLayout()

        self.btn_add_folder = QPushButton("Add Folder")
        self.btn_add_folder.clicked.connect(self._add_folder)
        btn_layout.addWidget(self.btn_add_folder)

        self.btn_add_files = QPushButton("Add Files")
        self.btn_add_files.clicked.connect(self._add_files)
        btn_layout.addWidget(self.btn_add_files)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self._clear_files)
        self.btn_clear.setStyleSheet("background-color: #5c5c5c;")
        btn_layout.addWidget(self.btn_clear)

        group_layout.addLayout(btn_layout)

        # File list
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(100)
        self.file_list.itemClicked.connect(self._on_file_selected)
        group_layout.addWidget(self.file_list)

        # Navigation row
        nav_layout = QHBoxLayout()

        self.btn_prev = QPushButton("<")
        self.btn_prev.setFixedWidth(40)
        self.btn_prev.clicked.connect(self._prev_file)
        nav_layout.addWidget(self.btn_prev)

        self.file_label = QLabel("No files loaded")
        self.file_label.setAlignment(Qt.AlignCenter)
        nav_layout.addWidget(self.file_label, 1)

        self.btn_next = QPushButton(">")
        self.btn_next.setFixedWidth(40)
        self.btn_next.clicked.connect(self._next_file)
        nav_layout.addWidget(self.btn_next)

        group_layout.addLayout(nav_layout)

        # Image info labels (shown below navigation)
        info_style = "font-size: 10px; color: #aaaaaa;"
        self.info_size_label = QLabel("Size: —")
        self.info_size_label.setStyleSheet(info_style)
        group_layout.addWidget(self.info_size_label)

        self.info_pixel_label = QLabel("Pixel Size: —")
        self.info_pixel_label.setStyleSheet(info_style)
        group_layout.addWidget(self.info_pixel_label)

        self.info_objective_label = QLabel("Objective: —")
        self.info_objective_label.setStyleSheet(info_style)
        group_layout.addWidget(self.info_objective_label)

        self.info_date_label = QLabel("Date: —")
        self.info_date_label.setStyleSheet(info_style)
        group_layout.addWidget(self.info_date_label)

        self.cb_scale_bar = CheckmarkCheckBox("Show Scale Bar")
        self.cb_scale_bar.setChecked(True)
        self.cb_scale_bar.stateChanged.connect(self._toggle_scale_bar)
        group_layout.addWidget(self.cb_scale_bar)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _create_channel_section(self, layout):
        """Create channel display section."""
        group = QGroupBox("Channels")
        self.channel_layout = QVBoxLayout()

        self.channel_placeholder = QLabel("Load a file to see channels")
        self.channel_placeholder.setStyleSheet("color: #888888; font-style: italic;")
        self.channel_layout.addWidget(self.channel_placeholder)

        self.btn_reset = QPushButton("Reset All Channels")
        self.btn_reset.clicked.connect(self._reset_adjustments)
        self.btn_reset.setStyleSheet("background-color: #5c5c5c;")
        self.channel_layout.addWidget(self.btn_reset)

        group.setLayout(self.channel_layout)
        layout.addWidget(group)

    def _create_drawing_tools_section(self, layout):
        """Create unified drawing tools section (text + shapes in one toolbar)."""
        group = QGroupBox("Drawing Tools")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(4)

        # Unified toolbar: [T] [→] [╱] [○] [▢] [～]
        tool_layout = QHBoxLayout()
        tool_layout.setSpacing(2)
        self.shape_tool_group = QButtonGroup(self)
        self.shape_tool_group.setExclusive(True)

        tool_icons = [
            ("T", "text"), ("→", "arrow"), ("╱", "line"),
            ("○", "circle"), ("▢", "rectangle"), ("～", "freehand"),
        ]
        self._tool_buttons = {}
        for label_text, tool_name in tool_icons:
            btn = QToolButton()
            btn.setText(label_text)
            btn.setFixedSize(36, 28)
            btn.setCheckable(True)
            btn.setProperty("tool_name", tool_name)
            btn.setStyleSheet("font-size: 14px; font-weight: bold;")
            self.shape_tool_group.addButton(btn)
            tool_layout.addWidget(btn)
            self._tool_buttons[tool_name] = btn
        tool_layout.addStretch()
        group_layout.addLayout(tool_layout)

        # Connect tool selection to switch options
        self.shape_tool_group.buttonClicked.connect(self._on_tool_selected)

        # ---- Text options (shown when T is selected) ----
        self.text_options_widget = QWidget()
        text_opts = QVBoxLayout(self.text_options_widget)
        text_opts.setContentsMargins(0, 0, 0, 0)
        text_opts.setSpacing(4)

        t_input = QHBoxLayout()
        t_input.addWidget(QLabel("Text:"))
        self.annotation_text = QLineEdit()
        self.annotation_text.setPlaceholderText("Enter annotation text")
        t_input.addWidget(self.annotation_text)
        text_opts.addLayout(t_input)

        t_options = QHBoxLayout()
        t_options.addWidget(QLabel("Size:"))
        self.font_size_spin = QSpinBox()
        self.font_size_spin.setRange(4, 500)
        self.font_size_spin.setValue(24)
        t_options.addWidget(self.font_size_spin)
        t_options.addWidget(QLabel("Color:"))
        self.annotation_color = QComboBox()
        self.annotation_color.addItems(['white', 'green', 'magenta', 'cyan', 'red', 'blue', 'yellow', 'gray'])
        self.annotation_color.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        t_options.addWidget(self.annotation_color)
        text_opts.addLayout(t_options)

        group_layout.addWidget(self.text_options_widget)

        # ---- Shape options (shown when a shape tool is selected) ----
        self.shape_options_widget = QWidget()
        shape_opts = QVBoxLayout(self.shape_options_widget)
        shape_opts.setContentsMargins(0, 0, 0, 0)
        shape_opts.setSpacing(4)

        s_opts1 = QHBoxLayout()
        s_opts1.addWidget(QLabel("Color:"))
        self.shape_color_combo = QComboBox()
        self.shape_color_combo.addItems(['white', 'green', 'magenta', 'cyan', 'red', 'blue', 'yellow', 'gray'])
        self.shape_color_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        s_opts1.addWidget(self.shape_color_combo)
        s_opts1.addWidget(QLabel("Width:"))
        self.shape_width_spin = QDoubleSpinBox()
        self.shape_width_spin.setRange(0.5, 20)
        self.shape_width_spin.setValue(2.0)
        self.shape_width_spin.setSingleStep(0.5)
        s_opts1.addWidget(self.shape_width_spin)
        shape_opts.addLayout(s_opts1)

        s_opts2 = QHBoxLayout()
        s_opts2.addWidget(QLabel("Style:"))
        self.shape_style_combo = QComboBox()
        self.shape_style_combo.addItems(["solid", "dashed", "dotted"])
        self.shape_style_combo.setSizeAdjustPolicy(QComboBox.AdjustToContents)
        s_opts2.addWidget(self.shape_style_combo)
        s_opts2.addStretch()
        shape_opts.addLayout(s_opts2)

        group_layout.addWidget(self.shape_options_widget)

        # No tool selected by default — both option panels hidden
        self.text_options_widget.hide()
        self.shape_options_widget.hide()

        # Allow no selection in the button group
        self.shape_tool_group.setExclusive(False)

        # Connect property changes for live editing of selected items
        self.font_size_spin.valueChanged.connect(self._on_text_property_changed)
        self.annotation_color.currentTextChanged.connect(self._on_text_property_changed)
        self.annotation_text.textChanged.connect(self._on_text_property_changed)
        self.shape_color_combo.currentTextChanged.connect(self._on_shape_property_changed)
        self.shape_width_spin.valueChanged.connect(self._on_shape_property_changed)
        self.shape_style_combo.currentTextChanged.connect(self._on_shape_property_changed)

        # Shared list for all annotations/shapes
        list_label = QLabel("Annotations:")
        list_label.setStyleSheet("font-size: 10px; color: #aaaaaa; margin-top: 4px;")
        group_layout.addWidget(list_label)

        self.annotation_list = QListWidget()
        self.annotation_list.setMaximumHeight(80)
        self.annotation_list.itemClicked.connect(self._on_annotation_list_click)
        group_layout.addWidget(self.annotation_list)

        # Buttons row
        btn_row = QHBoxLayout()

        self.btn_add_drawing = QPushButton("Add")
        self.btn_add_drawing.clicked.connect(self._add_drawing)
        btn_row.addWidget(self.btn_add_drawing)

        self.btn_delete_annotation = QPushButton("Delete Selected")
        self.btn_delete_annotation.clicked.connect(self._delete_selected_item)
        self.btn_delete_annotation.setStyleSheet("background-color: #5c5c5c;")
        btn_row.addWidget(self.btn_delete_annotation)

        self.btn_clear_all_drawings = QPushButton("Clear All")
        self.btn_clear_all_drawings.clicked.connect(self._clear_all_drawings)
        self.btn_clear_all_drawings.setStyleSheet("background-color: #5c5c5c;")
        btn_row.addWidget(self.btn_clear_all_drawings)
        group_layout.addLayout(btn_row)

        # Help text
        help_label = QLabel("Select a tool, then click and drag on the image. Press Escape to deselect. Double-click text to edit.")
        help_label.setWordWrap(True)
        help_label.setStyleSheet("color: #888888; font-size: 9px; font-style: italic;")
        group_layout.addWidget(help_label)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _create_export_section(self, layout):
        """Create export section."""
        group = QGroupBox("Export")
        group_layout = QVBoxLayout()

        format_layout = QHBoxLayout()
        format_layout.addWidget(QLabel("Format:"))
        self.export_format = QComboBox()
        self.export_format.addItems(['PNG', 'TIFF'])
        self.export_format.setMinimumWidth(80)
        format_layout.addWidget(self.export_format, 1)
        group_layout.addLayout(format_layout)

        btn_layout = QHBoxLayout()
        self.btn_export = QPushButton("Export Current")
        self.btn_export.clicked.connect(self._export_current)
        btn_layout.addWidget(self.btn_export)

        self.btn_export_all = QPushButton("Export All")
        self.btn_export_all.clicked.connect(self._export_all)
        btn_layout.addWidget(self.btn_export_all)
        group_layout.addLayout(btn_layout)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _apply_styles(self):
        """Apply FNT dark theme stylesheet."""
        self.setStyleSheet("""
            QMainWindow {
                background-color: #2b2b2b;
                color: #cccccc;
            }
            QWidget {
                background-color: #2b2b2b;
                color: #cccccc;
            }
            QLabel {
                color: #cccccc;
                background-color: transparent;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 4px 10px;
                border-radius: 4px;
                font-weight: bold;
                font-size: 11px;
                min-height: 16px;
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
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                padding: 5px;
                border: 1px solid #3f3f3f;
                border-radius: 3px;
                background-color: #1e1e1e;
                color: #cccccc;
            }
            QComboBox {
                min-width: 70px;
            }
            QComboBox QAbstractItemView {
                min-width: 140px;
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #3f3f3f;
            }
            QListWidget {
                background-color: #1e1e1e;
                border: 1px solid #3f3f3f;
                border-radius: 3px;
                color: #cccccc;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
            }
            QSlider::groove:horizontal {
                border: 1px solid #3f3f3f;
                height: 6px;
                background: #1e1e1e;
                border-radius: 3px;
            }
            QSlider::handle:horizontal {
                background: #0078d4;
                border: none;
                width: 16px;
                margin: -5px 0;
                border-radius: 8px;
            }
            QScrollArea {
                border: none;
            }
            QScrollBar:vertical {
                background: #1e1e1e;
                width: 10px;
                border-radius: 5px;
            }
            QScrollBar::handle:vertical {
                background: #0078d4;
                border-radius: 5px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background: #106ebe;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: #1e1e1e;
            }
            QStatusBar {
                background-color: #1e1e1e;
                color: #cccccc;
                border-top: 1px solid #3f3f3f;
            }
            QCheckBox {
                color: #cccccc;
            }
            QCheckBox::indicator {
                width: 16px;
                height: 16px;
            }
            QCheckBox::indicator:unchecked {
                border: 1px solid #3f3f3f;
                background-color: #1e1e1e;
            }
            QCheckBox::indicator:checked {
                border: 1px solid #0078d4;
                background-color: #0078d4;
            }
        """)

    # =========================================================================
    # File handling
    # =========================================================================

    def _add_folder(self):
        """Browse for folder and add all CZI files."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if folder:
            files = list(Path(folder).glob("*.czi"))
            files += list(Path(folder).glob("*.CZI"))
            self.czi_files.extend([str(f) for f in files])
            self._update_file_list()
            if self.czi_files and self.image_data is None:
                self.current_file_idx = 0
                self._load_current_file()

    def _add_files(self):
        """Browse for individual CZI files."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select CZI Files", "", "CZI Files (*.czi *.CZI);;All Files (*.*)"
        )
        if files:
            self.czi_files.extend(files)
            self._update_file_list()
            if self.image_data is None:
                self.current_file_idx = len(self.czi_files) - len(files)
                self._load_current_file()

    def _clear_files(self):
        """Clear all loaded files."""
        self.czi_files.clear()
        self.current_file_idx = 0
        self.image_data = None
        self._update_file_list()
        self._clear_channel_controls()
        self.preview.set_image(None)
        self.status_bar.showMessage("Cleared all files")

    def _update_file_list(self):
        """Update the file list widget."""
        self.file_list.clear()
        for filepath in self.czi_files:
            self.file_list.addItem(Path(filepath).name)

        if self.czi_files:
            self.file_list.setCurrentRow(self.current_file_idx)
            self.file_label.setText(f"File {self.current_file_idx + 1}/{len(self.czi_files)}")
        else:
            self.file_label.setText("No files loaded")

    def _on_file_selected(self, item):
        """Handle file selection from list."""
        idx = self.file_list.row(item)
        if idx != self.current_file_idx:
            self._save_current_file_settings()
            self.current_file_idx = idx
            self._load_current_file()

    def _prev_file(self):
        """Go to previous file."""
        if self.czi_files and self.current_file_idx > 0:
            self._save_current_file_settings()
            self.current_file_idx -= 1
            self._load_current_file()
            self._update_file_list()

    def _next_file(self):
        """Go to next file."""
        if self.czi_files and self.current_file_idx < len(self.czi_files) - 1:
            self._save_current_file_settings()
            self.current_file_idx += 1
            self._load_current_file()
            self._update_file_list()

    def _save_current_file_settings(self):
        """Save current channel settings for the current file to the cache."""
        if self.czi_files and self.channel_controls:
            filepath = self.czi_files[self.current_file_idx]
            self._file_settings_cache[filepath] = self._get_current_settings()

    def _restore_file_settings(self, filepath: str):
        """Restore cached channel settings for a file, if available."""
        cached = self._file_settings_cache.get(filepath)
        if cached:
            for ch_idx, settings in cached.items():
                ctrl = self.channel_controls.get(ch_idx)
                if ctrl:
                    ctrl.apply_settings(settings)

    def _load_current_file(self):
        """Load the current CZI file."""
        if not self.czi_files:
            return

        if not HAS_AICSPYLIBCZI and not HAS_AICSIMAGEIO:
            QMessageBox.warning(
                self, "Missing Dependencies",
                "CZI file support requires additional packages.\n\n"
                "Install with:\n"
                "pip install aicspylibczi fsspec Pillow"
            )
            return

        filepath = self.czi_files[self.current_file_idx]

        # Use worker thread
        self.load_worker = CZILoadWorker(filepath)
        self.load_worker.progress.connect(self._on_load_progress)
        self.load_worker.loaded.connect(self._on_load_complete)
        self.load_worker.error.connect(self._on_load_error)
        self.load_worker.start()

    def _on_load_progress(self, message: str):
        self.status_bar.showMessage(message)

    def _on_load_complete(self, data: CZIImageData):
        self.image_data = data
        self.processor.clear_bg_cache()
        self._setup_channel_controls()

        # Restore cached settings if available
        filepath = data.metadata.filepath
        self._restore_file_settings(filepath)

        self._update_image_info()
        self._update_display()
        self.preview.fit_to_window()
        self._update_zoom_label()

        n_channels = len(data.channel_data)
        self.status_bar.showMessage(
            f"Loaded: {Path(data.metadata.filepath).name} "
            f"({data.metadata.width}x{data.metadata.height}, {n_channels} channels)"
        )

    def _on_load_error(self, error: str):
        QMessageBox.critical(self, "Load Error", f"Failed to load file:\n{error}")
        self.status_bar.showMessage(f"Error: {error}")

    # =========================================================================
    # Channel controls
    # =========================================================================

    def _clear_channel_controls(self):
        """Clear all channel control widgets."""
        for ctrl in self.channel_controls.values():
            ctrl.setParent(None)
            ctrl.deleteLater()
        self.channel_controls.clear()

        # Show placeholder
        self.channel_placeholder.show()

    def _setup_channel_controls(self):
        """Setup channel control widgets based on loaded image."""
        self._clear_channel_controls()

        if self.image_data is None:
            return

        self.channel_placeholder.hide()

        for channel_info in self.image_data.metadata.channels:
            ctrl = ChannelControlWidget(
                channel_info.index,
                channel_info.name,
                channel_info.suggested_color
            )
            ctrl.settings_changed.connect(self._on_channel_changed)
            self.channel_layout.insertWidget(self.channel_layout.count() - 1, ctrl)  # Before Reset button
            self.channel_controls[channel_info.index] = ctrl

    def _on_channel_changed(self):
        """Handle channel settings change."""
        self._update_display()

    # =========================================================================
    # Display
    # =========================================================================

    def _get_current_settings(self) -> Dict[int, ChannelDisplaySettings]:
        """Get current display settings from all channel controls."""
        settings = {}
        for idx, ctrl in self.channel_controls.items():
            settings[idx] = ctrl.get_settings()
        return settings

    def _update_display(self):
        """Update the image preview with current settings."""
        if self.image_data is None:
            return

        settings = self._get_current_settings()

        # Merge channels
        merged = self.processor.merge_channels(
            self.image_data.channel_data,
            settings
        )

        # Convert to uint8 for display
        display_image = self.processor.to_uint8(merged)

        self.preview.set_image(display_image)

        # Update preview with annotations and shapes for interactive editing
        self.preview.set_annotations(self.processor.annotations)
        self.preview.set_shapes(self.processor.shape_annotations)

    def _reset_adjustments(self):
        """Reset all per-channel controls to defaults."""
        for ctrl in self.channel_controls.values():
            ctrl.reset()
        self.processor.clear_bg_cache()
        self._update_display()

    # =========================================================================
    # Drawing tools (unified text + shapes)
    # =========================================================================

    def _on_tool_selected(self, button):
        """Handle tool selection from the unified toolbar. Selects tool and shows options, but does NOT enter drawing mode."""
        tool_name = button.property("tool_name")

        # Toggle behavior: if button was already checked, uncheck it (deselect tool)
        if not button.isChecked():
            # Button was unchecked — deselect tool, hide options, exit any active drawing mode
            self.text_options_widget.hide()
            self.shape_options_widget.hide()
            self.preview.drawing_mode = ""
            self.preview.annotation_mode = False
            self.status_bar.showMessage("Drawing tool deselected")
            return

        # Uncheck all other buttons (manual exclusive since we use non-exclusive group)
        for name, btn in self._tool_buttons.items():
            if btn is not button:
                btn.setChecked(False)

        # Exit any active drawing mode when switching tools
        self.preview.drawing_mode = ""
        self.preview.annotation_mode = False

        if tool_name == "text":
            self.text_options_widget.show()
            self.shape_options_widget.hide()
        else:
            self.text_options_widget.hide()
            self.shape_options_widget.show()

        self.status_bar.showMessage(f"Selected {tool_name} tool. Click 'Add' to draw.")

    def _on_drawing_mode_exited(self):
        """Handle Escape pressed in preview — uncheck all tool buttons and hide options."""
        for btn in self._tool_buttons.values():
            btn.setChecked(False)
        self.text_options_widget.hide()
        self.shape_options_widget.hide()
        self.status_bar.showMessage("Drawing tool deselected")

    def _add_drawing(self):
        """Enter drawing mode for the currently selected tool."""
        # Find which tool is selected
        selected_tool = None
        for name, btn in self._tool_buttons.items():
            if btn.isChecked():
                selected_tool = name
                break

        if selected_tool is None:
            self.status_bar.showMessage("Select a drawing tool first")
            return

        if selected_tool == "text":
            self.preview.annotation_mode = True
            self.preview.drawing_mode = ""
            self.text_options_widget.show()
            self.shape_options_widget.hide()
            self.status_bar.showMessage("Click on the image to place text.")
        else:
            self.preview.annotation_mode = False
            self.preview.drawing_mode = selected_tool
            self.preview._draw_color = self.shape_color_combo.currentText()
            self.preview._draw_line_width = self.shape_width_spin.value()
            self.preview._draw_line_style = self.shape_style_combo.currentText()
            self.text_options_widget.hide()
            self.shape_options_widget.show()
            self.status_bar.showMessage(f"Click and drag on the image to draw {selected_tool}. Press Escape to stop.")

    def _on_annotation_click(self, x: int, y: int):
        """Handle annotation placement click (text mode). Auto-exits after placing."""
        text = self.annotation_text.text().strip() or "Text"
        self.processor.add_annotation(
            text, x, y,
            self.font_size_spin.value(),
            self.annotation_color.currentText()
        )
        self._refresh_drawing_list()
        self._update_display()

        # Auto-exit text drawing mode after placing
        self.preview.annotation_mode = False
        for btn in self._tool_buttons.values():
            btn.setChecked(False)
        self.text_options_widget.hide()
        self.status_bar.showMessage("Text placed.")

    def _on_annotation_selected(self, idx: int):
        """Handle annotation selection in preview."""
        if idx == -1 and self.preview.selected_annotation_idx >= 0:
            self._delete_selected_annotation()
        elif idx >= 0:
            # Select in list (text annotations start at index 0)
            self.annotation_list.setCurrentRow(idx)
            # Populate text options from selected annotation
            self._populate_text_options_from_annotation(idx)
            self.status_bar.showMessage(f"Selected text annotation {idx + 1}")

    def _on_annotation_list_click(self, item):
        """Handle click on item in drawing list widget."""
        idx = self.annotation_list.row(item)
        n_text = len(self.processor.annotations)
        if idx < n_text:
            self.preview.select_annotation(idx)
            self.preview.selected_shape_idx = -1
            self._populate_text_options_from_annotation(idx)
        else:
            shape_idx = idx - n_text
            self.preview.selected_shape_idx = shape_idx
            self.preview.selected_annotation_idx = -1
            self._populate_shape_options_from_shape(shape_idx)
            self.preview.update()
        self.status_bar.showMessage(f"Selected item {idx + 1}")

    def _on_annotation_modified(self):
        """Handle annotation modification (move/resize/rotate)."""
        self._refresh_drawing_list()
        self.preview.update()

    def _on_annotation_edit(self, idx: int):
        """Handle double-click to edit annotation text."""
        if idx < 0 or idx >= len(self.processor.annotations):
            return

        ann = self.processor.annotations[idx]
        new_text, ok = QInputDialog.getText(
            self, "Edit Annotation", "Enter new text:", QLineEdit.Normal, ann.text
        )
        if ok and new_text.strip():
            ann.text = new_text.strip()
            self._refresh_drawing_list()
            self._update_display()
            self.status_bar.showMessage("Annotation updated")

    def _delete_selected_annotation(self):
        """Delete the currently selected text annotation."""
        idx = self.preview.selected_annotation_idx
        if 0 <= idx < len(self.processor.annotations):
            self.processor.remove_annotation(idx)
            self.preview.selected_annotation_idx = -1
            self._refresh_drawing_list()
            self._update_display()
            self.status_bar.showMessage("Text annotation deleted")

    # =========================================================================
    # Shape drawing
    # =========================================================================

    def _on_shape_drawn(self, shape):
        """Handle shape drawn on the preview."""
        self.processor.add_shape(shape)
        self._refresh_drawing_list()
        self._update_display()
        # Keep drawing mode active for continuous drawing
        self.status_bar.showMessage(f"Shape added. Continue drawing or press Esc to stop.")

    def _on_shape_selected(self, idx: int):
        """Handle shape selection in preview."""
        if idx == -1 and self.preview.selected_shape_idx >= 0:
            self._delete_selected_shape()
        elif idx >= 0:
            n_text = len(self.processor.annotations)
            self.annotation_list.setCurrentRow(n_text + idx)
            # Populate shape options from selected shape
            self._populate_shape_options_from_shape(idx)
            self.status_bar.showMessage(f"Selected shape {idx + 1}")

    def _on_shape_modified(self):
        """Handle shape modification (move/resize)."""
        self._refresh_drawing_list()
        self.preview.update()

    def _populate_text_options_from_annotation(self, idx: int):
        """Populate text options panel from a selected text annotation."""
        if idx < 0 or idx >= len(self.processor.annotations):
            return
        ann = self.processor.annotations[idx]
        # Block signals to prevent feedback loops
        self.annotation_text.blockSignals(True)
        self.font_size_spin.blockSignals(True)
        self.annotation_color.blockSignals(True)
        self.annotation_text.setText(ann.text)
        self.font_size_spin.setValue(ann.font_size)
        self.annotation_color.setCurrentText(ann.color)
        self.annotation_text.blockSignals(False)
        self.font_size_spin.blockSignals(False)
        self.annotation_color.blockSignals(False)
        # Show text options panel
        self.text_options_widget.show()
        self.shape_options_widget.hide()

    def _populate_shape_options_from_shape(self, idx: int):
        """Populate shape options panel from a selected shape."""
        if idx < 0 or idx >= len(self.processor.shape_annotations):
            return
        shape = self.processor.shape_annotations[idx]
        # Block signals to prevent feedback loops
        self.shape_color_combo.blockSignals(True)
        self.shape_width_spin.blockSignals(True)
        self.shape_style_combo.blockSignals(True)
        self.shape_color_combo.setCurrentText(shape.color)
        self.shape_width_spin.setValue(shape.line_width)
        self.shape_style_combo.setCurrentText(shape.line_style)
        self.shape_color_combo.blockSignals(False)
        self.shape_width_spin.blockSignals(False)
        self.shape_style_combo.blockSignals(False)
        # Show shape options panel
        self.shape_options_widget.show()
        self.text_options_widget.hide()

    def _on_text_property_changed(self):
        """Update selected text annotation when text options change."""
        idx = self.preview.selected_annotation_idx
        if idx < 0 or idx >= len(self.processor.annotations):
            return
        ann = self.processor.annotations[idx]
        ann.font_size = self.font_size_spin.value()
        ann.color = self.annotation_color.currentText()
        # Update text if changed
        new_text = self.annotation_text.text().strip()
        if new_text:
            ann.text = new_text
        self._refresh_drawing_list()
        self._update_display()

    def _on_shape_property_changed(self):
        """Update selected shape when shape options change."""
        idx = self.preview.selected_shape_idx
        if idx < 0 or idx >= len(self.processor.shape_annotations):
            return
        shape = self.processor.shape_annotations[idx]
        shape.color = self.shape_color_combo.currentText()
        shape.line_width = self.shape_width_spin.value()
        shape.line_style = self.shape_style_combo.currentText()
        self._refresh_drawing_list()
        self._update_display()

    def _delete_selected_shape(self):
        """Delete the currently selected shape."""
        idx = self.preview.selected_shape_idx
        if 0 <= idx < len(self.processor.shape_annotations):
            self.processor.remove_shape(idx)
            self.preview.selected_shape_idx = -1
            self._refresh_drawing_list()
            self._update_display()
            self.status_bar.showMessage("Shape deleted")

    def _delete_selected_item(self):
        """Delete whichever item is selected (text or shape)."""
        if self.preview.selected_annotation_idx >= 0:
            self._delete_selected_annotation()
        elif self.preview.selected_shape_idx >= 0:
            self._delete_selected_shape()

    def _clear_all_drawings(self):
        """Clear all text annotations and shapes."""
        self.processor.clear_annotations()
        self.processor.clear_shapes()
        self.preview.selected_annotation_idx = -1
        self.preview.selected_shape_idx = -1
        self._refresh_drawing_list()
        self._update_display()

    def _refresh_drawing_list(self):
        """Refresh the combined drawing list (text annotations + shapes)."""
        self.annotation_list.clear()
        for ann in self.processor.annotations:
            rotation_str = f", rot={ann.rotation:.0f}°" if ann.rotation != 0 else ""
            self.annotation_list.addItem(f'📝 "{ann.text}" ({ann.x}, {ann.y}){rotation_str}')
        for shape in self.processor.shape_annotations:
            self.annotation_list.addItem(f'🔷 {shape.shape_type} ({shape.color}, {shape.line_style})')

    # =========================================================================
    # Scale bar / Image info
    # =========================================================================

    def _toggle_scale_bar(self):
        """Toggle scale bar overlay."""
        self.preview.show_scale_bar = self.cb_scale_bar.isChecked()
        self.preview.update()

    def _update_image_info(self):
        """Update image info labels from metadata."""
        if self.image_data is None:
            return

        meta = self.image_data.metadata
        self.info_size_label.setText(f"Size: {meta.width} × {meta.height}")

        if meta.pixel_size_um and meta.pixel_size_um > 0:
            self.info_pixel_label.setText(f"Pixel Size: {meta.pixel_size_um:.4f} µm/px")
            self.preview.pixel_size_um = meta.pixel_size_um
        else:
            self.info_pixel_label.setText("Pixel Size: —")
            self.preview.pixel_size_um = 0.0

        if meta.objective:
            obj_str = meta.objective
            if hasattr(meta, 'magnification') and meta.magnification:
                obj_str += f" {meta.magnification}x"
            if hasattr(meta, 'numerical_aperture') and meta.numerical_aperture:
                obj_str += f"/{meta.numerical_aperture}"
            self.info_objective_label.setText(f"Objective: {obj_str}")
        else:
            self.info_objective_label.setText("Objective: —")

        if meta.acquisition_date:
            self.info_date_label.setText(f"Date: {meta.acquisition_date[:19]}")
        else:
            self.info_date_label.setText("Date: —")

    # =========================================================================
    # Export
    # =========================================================================

    def _build_export_settings_dict(self) -> dict:
        """Build a dictionary of all current settings for export traceability."""
        settings = self._get_current_settings()
        export_dict = {
            'source_file': self.image_data.metadata.filepath,
            'export_date': datetime.now().isoformat(),
            'image_dimensions': {
                'width': self.image_data.metadata.width,
                'height': self.image_data.metadata.height,
            },
            'pixel_size_um': self.image_data.metadata.pixel_size_um,
            'channels': {},
            'scale_bar': {
                'visible': self.preview.show_scale_bar,
                'position': list(self.preview.scale_bar_pos) if self.preview.scale_bar_pos else None,
            },
            'annotations': {
                'text_count': len(self.processor.annotations),
                'shape_count': len(self.processor.shape_annotations),
            }
        }
        # Add per-channel settings
        for idx, s in settings.items():
            ch_name = f"channel_{idx}"
            for ch_info in self.image_data.metadata.channels:
                if ch_info.index == idx:
                    ch_name = ch_info.name
                    break
            export_dict['channels'][ch_name] = {
                'enabled': s.enabled,
                'color': s.color,
                'brightness': s.brightness,
                'contrast': s.contrast,
                'gamma': s.gamma,
                'sharpness': s.sharpness,
                'threshold_enabled': s.threshold_enabled,
                'threshold_low': s.threshold_low,
                'threshold_high': s.threshold_high,
                'bg_subtract_method': s.bg_subtract_method,
                'bg_subtract_radius': s.bg_subtract_radius,
            }
        return export_dict

    def _export_current(self):
        """Export current image with companion JSON settings file."""
        if self.image_data is None:
            QMessageBox.warning(self, "No Image", "Load an image first")
            return

        ext = self.export_format.currentText().lower()
        source_dir = str(Path(self.image_data.metadata.filepath).parent)
        date_suffix = datetime.now().strftime('%Y%m%d')
        default_name = Path(self.image_data.metadata.filepath).stem + f"_{date_suffix}.{ext}"
        default_path = os.path.join(source_dir, default_name)

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Image", default_path,
            f"{ext.upper()} Files (*.{ext});;All Files (*.*)"
        )

        if filepath:
            try:
                settings = self._get_current_settings()
                merged = self.processor.merge_channels(
                    self.image_data.channel_data,
                    settings
                )
                # Build scale bar info if visible
                scale_bar_info = None
                if self.preview.show_scale_bar and self.preview.pixel_size_um > 0:
                    scale_bar_info = {
                        'pixel_size_um': self.preview.pixel_size_um,
                        'position': self.preview.scale_bar_pos,
                    }
                self.processor.export_image(merged, filepath,
                                            include_annotations=True,
                                            scale_bar_info=scale_bar_info)

                # Write companion JSON with settings for traceability
                json_path = str(Path(filepath).with_suffix('.json'))
                export_settings = self._build_export_settings_dict()
                with open(json_path, 'w') as f:
                    json.dump(export_settings, f, indent=2)

                self.status_bar.showMessage(f"Exported: {filepath} + settings.json")
            except Exception as e:
                QMessageBox.critical(self, "Export Error", str(e))

    def _export_all(self):
        """Export all loaded images with companion JSON settings files."""
        if not self.czi_files:
            QMessageBox.warning(self, "No Files", "No files loaded")
            return

        source_dir = str(Path(self.czi_files[0]).parent) if self.czi_files else ""
        folder = QFileDialog.getExistingDirectory(self, "Select Output Folder", source_dir)
        if not folder:
            return

        ext = self.export_format.currentText().lower()
        date_suffix = datetime.now().strftime('%Y%m%d')
        exported = 0

        for filepath in self.czi_files:
            try:
                reader = CZIFileReader(filepath)
                data = reader.load_all_channels()

                settings = self._get_current_settings()
                merged = self.processor.merge_channels(data.channel_data, settings)

                output_name = Path(filepath).stem + f"_{date_suffix}.{ext}"
                output_path = os.path.join(folder, output_name)
                # Build scale bar info if visible
                scale_bar_info = None
                if self.preview.show_scale_bar and self.preview.pixel_size_um > 0:
                    scale_bar_info = {
                        'pixel_size_um': self.preview.pixel_size_um,
                        'position': self.preview.scale_bar_pos,
                    }
                self.processor.export_image(merged, output_path,
                                            include_annotations=False,
                                            scale_bar_info=scale_bar_info)

                # Write companion JSON with settings for traceability
                json_path = str(Path(output_path).with_suffix('.json'))
                export_settings = {
                    'source_file': filepath,
                    'export_date': datetime.now().isoformat(),
                    'image_dimensions': {
                        'width': data.metadata.width,
                        'height': data.metadata.height,
                    },
                    'pixel_size_um': data.metadata.pixel_size_um,
                    'channels': {},
                    'scale_bar': {
                        'visible': self.preview.show_scale_bar,
                        'position': list(self.preview.scale_bar_pos) if self.preview.scale_bar_pos else None,
                    },
                    'annotations': {
                        'text_count': 0,
                        'shape_count': 0,
                    }
                }
                # Add per-channel settings
                for idx, s in settings.items():
                    ch_name = f"channel_{idx}"
                    for ch_info in data.metadata.channels:
                        if ch_info.index == idx:
                            ch_name = ch_info.name
                            break
                    export_settings['channels'][ch_name] = {
                        'enabled': s.enabled,
                        'color': s.color,
                        'brightness': s.brightness,
                        'contrast': s.contrast,
                        'gamma': s.gamma,
                        'sharpness': s.sharpness,
                        'threshold_enabled': s.threshold_enabled,
                        'threshold_low': s.threshold_low,
                        'threshold_high': s.threshold_high,
                        'bg_subtract_method': s.bg_subtract_method,
                        'bg_subtract_radius': s.bg_subtract_radius,
                    }
                with open(json_path, 'w') as f:
                    json.dump(export_settings, f, indent=2)

                exported += 1

            except Exception as e:
                self.status_bar.showMessage(f"Error exporting {filepath}: {e}")

        QMessageBox.information(
            self, "Export Complete",
            f"Exported {exported} of {len(self.czi_files)} files (+ settings.json) to:\n{folder}"
        )

    # =========================================================================
    # View controls
    # =========================================================================

    def _zoom_in(self):
        """Zoom in by 25%."""
        current = self.preview.zoom_level
        self.preview.set_zoom(current * 1.25)
        self._update_zoom_label()

    def _zoom_out(self):
        """Zoom out by 25%."""
        current = self.preview.zoom_level
        self.preview.set_zoom(current / 1.25)
        self._update_zoom_label()

    def _fit_to_window(self):
        """Fit image to window."""
        self.preview.fit_to_window()
        self._update_zoom_label()

    def _update_zoom_label(self):
        """Update zoom percentage label."""
        self.zoom_label.setText(f"{self.preview.get_zoom_percent()}%")


def main():
    """Run the CZI Viewer standalone."""
    app = QApplication(sys.argv)
    window = CZIViewerWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
