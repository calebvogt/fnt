"""
Image Quantification Tool - Standalone PyQt5 GUI for cell counting and analysis.

Provides multi-channel cell/particle detection, colocalization analysis,
and ROI-based density measurement for Zeiss CZI microscopy images.
"""

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QPointF, QRectF, QPoint
from PyQt5.QtGui import (
    QFont, QImage, QPainter, QColor, QPen, QWheelEvent,
    QBrush, QCursor, QKeySequence
)
from PyQt5.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QGroupBox, QScrollArea,
    QListWidget, QSplitter, QSizePolicy, QMessageBox, QShortcut
)

from .czi_reader import CZIFileReader, CZIImageData, HAS_AICSPYLIBCZI, HAS_AICSIMAGEIO
from .image_processor import CZIImageProcessor, ChannelDisplaySettings


# =========================================================================
# Background file loader (duplicated from czi_viewer for independence)
# =========================================================================

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


# =========================================================================
# Batch analysis worker
# =========================================================================

class BatchAnalysisWorker(QThread):
    """Worker thread for batch analysis across multiple CZI files."""
    progress = pyqtSignal(str, int, int)  # filename, current, total
    finished = pyqtSignal(str, int)        # filepath, n_files
    error = pyqtSignal(str)

    def __init__(self, filepaths: List[str], quant_panel, processor, output_path: str):
        super().__init__()
        self.filepaths = filepaths
        self.output_path = output_path
        self.processor = processor
        # Capture current settings from the panel
        self._bg_method = quant_panel.get_bg_method()
        self._bg_radius = quant_panel.get_bg_radius()
        self._min_area = quant_panel.min_area_spin.value()
        self._max_area = quant_panel.max_area_spin.value()
        self._use_watershed = quant_panel.cb_watershed.isChecked()
        # Per-channel settings (method + manual threshold)
        self._channel_settings = {}
        for ch_idx, row in quant_panel._channel_rows.items():
            if row.is_enabled():
                self._channel_settings[ch_idx] = {
                    'method': row.get_method(),
                    'manual_threshold': row.get_manual_threshold(),
                }

    def run(self):
        import csv
        from .quantification import (
            ImageQuantifier, QuantificationConfig, MultiChannelConfig,
        )

        try:
            quantifier = ImageQuantifier()
            total = len(self.filepaths)
            all_rows = []

            for file_i, filepath in enumerate(self.filepaths):
                filename = Path(filepath).name
                self.progress.emit(filename, file_i + 1, total)

                try:
                    reader = CZIFileReader(filepath)
                    data = reader.load_all_channels()
                except Exception as e:
                    all_rows.append([filename, "LOAD_ERROR", str(e)])
                    continue

                pixel_size = data.metadata.pixel_size_um
                channel_names = {
                    info.index: info.name
                    for info in data.metadata.channels
                }

                # Build config from saved settings
                channel_configs = {}
                images = {}
                for ch_idx in data.channel_data:
                    if ch_idx not in self._channel_settings:
                        # Use default settings for channels not in panel
                        settings = {'method': 'otsu', 'manual_threshold': 0.5}
                    else:
                        settings = self._channel_settings[ch_idx]

                    channel_configs[ch_idx] = QuantificationConfig(
                        channel_index=ch_idx,
                        threshold_method=settings['method'],
                        manual_threshold=settings['manual_threshold'],
                        min_area_um2=self._min_area,
                        max_area_um2=self._max_area,
                        use_watershed=self._use_watershed,
                    )

                    # Normalize and apply BG sub
                    raw = data.channel_data[ch_idx]
                    normalized = self.processor.normalize_image(raw)
                    if self._bg_method != "none":
                        from .image_processor import ChannelDisplaySettings
                        bg_settings = ChannelDisplaySettings(
                            bg_subtract_method=self._bg_method,
                            bg_subtract_radius=self._bg_radius,
                        )
                        normalized = self.processor.apply_background_subtraction(
                            normalized, bg_settings, ch_idx + 10000 * file_i
                        )
                    images[ch_idx] = normalized

                config = MultiChannelConfig(channel_configs=channel_configs)
                result = quantifier.detect_and_measure_multi(
                    images, config, pixel_size, channel_names
                )

                # Collect summary rows per channel
                for ch_idx in sorted(result.channel_results.keys()):
                    ch_res = result.channel_results[ch_idx]
                    ch_name = ch_res.channel_name or f"Ch{ch_idx}"
                    all_rows.append([
                        filename, ch_name,
                        ch_res.particle_count,
                        f"{ch_res.total_area_px:.2f}",
                        f"{ch_res.total_area_um2:.2f}" if ch_res.total_area_um2 is not None else "N/A",
                        f"{ch_res.mean_area_px:.2f}",
                        f"{ch_res.mean_area_um2:.2f}" if ch_res.mean_area_um2 is not None else "N/A",
                        f"{ch_res.mean_intensity:.6f}",
                        f"{ch_res.area_fraction:.6f}",
                        f"{ch_res.integrated_density:.4f}",
                        ch_res.threshold_method,
                        f"{ch_res.threshold_value:.6f}",
                    ])

            # Write CSV
            with open(self.output_path, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([
                    "Filename", "Channel", "Particle Count",
                    "Total Area (px)", "Total Area (um2)",
                    "Mean Area (px)", "Mean Area (um2)",
                    "Mean Intensity", "Area Fraction",
                    "Integrated Density",
                    "Threshold Method", "Threshold Value",
                ])
                writer.writerows(all_rows)

            self.finished.emit(self.output_path, total)
        except Exception as e:
            self.error.emit(str(e))


# =========================================================================
# QuantPreviewWidget — simplified image preview for quantification
# =========================================================================

class QuantPreviewWidget(QWidget):
    """
    Simplified image preview with zoom/pan, quantification overlay, and ROI drawing.

    Does NOT include annotation editing, shape drawing, or scale bar features
    (those belong to the CZI Viewer's ImagePreviewWidget).
    """

    # ROI drawing signal
    roi_drawn = pyqtSignal(int, int, int, int)  # x, y, w, h in image coords

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
        self._fit_zoom_level: float = 1.0

        # Interaction state
        self.is_panning: bool = False
        self.last_mouse_pos: Optional[QPoint] = None

        # ROI drawing state
        self.roi_mode: bool = False
        self.roi_rect: Optional[Tuple[int, int, int, int]] = None
        self.roi_start_pos: Optional[Tuple[float, float]] = None

        # Quantification overlay state
        self.quant_overlay: Optional[dict] = None
        self._quant_overlay_qimage: Optional[QImage] = None
        self._quant_overlay_mask_id = None

        # Styling
        self.setStyleSheet("background-color: #1e1e1e;")

    def set_image(self, image: Optional[np.ndarray]):
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

    def zoom_to_point(self, img_x: float, img_y: float):
        """Zoom in and center on a specific image coordinate (e.g. particle centroid)."""
        if self.display_image is None:
            return
        # Zoom to 300% of fit level
        target_zoom = self._fit_zoom_level * 3.0
        self.zoom_level = target_zoom
        # Pan so the target point is at widget center
        widget_cx = self.width() / 2
        widget_cy = self.height() / 2
        scaled_w = self.image_width * self.zoom_level
        scaled_h = self.image_height * self.zoom_level
        # The image top-left in widget coords is:
        # img_origin_x = (widget_w - scaled_w)/2 + pan_offset.x
        # We want: img_origin_x + img_x * zoom = widget_cx
        # => pan_offset.x = widget_cx - img_x * zoom - (widget_w - scaled_w)/2
        pan_x = widget_cx - img_x * self.zoom_level - (self.width() - scaled_w) / 2
        pan_y = widget_cy - img_y * self.zoom_level - (self.height() - scaled_h) / 2
        self.pan_offset = QPointF(pan_x, pan_y)
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

    # =====================================================================
    # Paint
    # =====================================================================

    def paintEvent(self, event):
        """Paint the image with quant overlay and ROI rectangle."""
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

        # Draw ROI rectangle overlay (in-progress drawing)
        if self.roi_rect is not None:
            self._draw_roi_overlay(painter)

        # Draw quantification overlay (mask + centroids + labels + ROI rects)
        if self.quant_overlay is not None:
            self._draw_quant_overlay(painter)

        # Draw mode indicator
        if self.roi_mode:
            painter.setPen(QPen(QColor(255, 100, 0), 2))
            painter.drawRect(2, 2, self.width() - 4, self.height() - 4)
            painter.setPen(QColor(255, 100, 0))
            painter.setFont(QFont("Arial", 10))
            painter.drawText(10, 20, "Draw ROI: click and drag a rectangle")

    def _draw_roi_overlay(self, painter: QPainter):
        """Draw the in-progress ROI rectangle overlay."""
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

    def _draw_quant_overlay(self, painter: QPainter):
        """Draw multi-channel quantification overlay with masks, centroids, ROIs."""
        overlay = self.quant_overlay
        if overlay is None:
            return

        channel_overlays = overlay.get('channel_overlays', {})
        show_mask = overlay.get('show_mask', False)
        contour_mode = overlay.get('contour_mode', True)
        highlight_channel = overlay.get('highlight_channel')
        highlight_idx = overlay.get('highlight_idx', -1)

        # Compute image-to-widget transform for mask drawing
        scaled_w = int(self.image_width * self.zoom_level)
        scaled_h = int(self.image_height * self.zoom_level)
        img_x = (self.width() - scaled_w) / 2 + self.pan_offset.x()
        img_y = (self.height() - scaled_h) / 2 + self.pan_offset.y()
        target = QRectF(img_x, img_y, scaled_w, scaled_h)

        # Draw per-channel binary masks
        if show_mask and channel_overlays:
            if contour_mode:
                # Contour outline mode — draw mask boundaries as colored lines
                self._draw_contour_outlines(painter, channel_overlays, overlay)
            else:
                # Filled mask mode — semi-transparent colored overlay
                self._draw_filled_mask(painter, channel_overlays, overlay, target)

        # Draw centroids per channel
        for ch_idx, co in channel_overlays.items():
            centroids = co.get('centroids', [])
            labels = co.get('labels', [])
            r_color, g_color, b_color = co.get('color', (255, 255, 0))

            is_highlight_ch = (highlight_channel == ch_idx)

            for i, (cx, cy) in enumerate(centroids):
                wp = self._image_to_widget_coords(cx, cy)
                r = max(3, 5 * self.zoom_level)

                if is_highlight_ch and i == highlight_idx:
                    painter.setPen(QPen(QColor(255, 80, 80), 3))
                    painter.setBrush(QBrush(QColor(255, 80, 80, 80)))
                    painter.drawEllipse(wp, r * 2, r * 2)
                else:
                    painter.setPen(QPen(QColor(r_color, g_color, b_color), 2))
                    painter.setBrush(Qt.NoBrush)
                    painter.drawEllipse(wp, r, r)

            # ID labels (only when zoomed in enough)
            if (self.zoom_level >= self._fit_zoom_level * 0.5
                    and labels and centroids):
                font_size = max(
                    7, int(8 * self.zoom_level / self._fit_zoom_level)
                )
                painter.setFont(QFont("Arial", font_size))
                for i, ((cx, cy), label_id) in enumerate(
                    zip(centroids, labels)
                ):
                    wp = self._image_to_widget_coords(cx, cy)
                    if is_highlight_ch and i == highlight_idx:
                        painter.setPen(QColor(255, 80, 80))
                    else:
                        painter.setPen(QColor(r_color, g_color, b_color))
                    painter.drawText(
                        int(wp.x()) + 6, int(wp.y()) - 4, str(label_id)
                    )

        # Draw ROI rectangles
        roi_defs = overlay.get('roi_definitions', [])
        for roi in roi_defs:
            tl = self._image_to_widget_coords(roi.x, roi.y)
            br = self._image_to_widget_coords(roi.x + roi.w, roi.y + roi.h)
            pen = QPen(QColor(255, 200, 0), 2, Qt.DashLine)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(255, 200, 0, 25)))
            painter.drawRect(QRectF(tl, br))
            # Label
            painter.setPen(QColor(255, 200, 0))
            font_size = max(
                8, int(10 * self.zoom_level / max(self._fit_zoom_level, 0.01))
            )
            painter.setFont(QFont("Arial", font_size, QFont.Bold))
            painter.drawText(
                int(tl.x()) + 4, int(tl.y()) - 4, roi.label
            )

    def _draw_contour_outlines(self, painter: QPainter,
                               channel_overlays: dict, overlay: dict):
        """Draw mask boundaries as colored contour outlines (no fill)."""
        try:
            from skimage.measure import find_contours
        except ImportError:
            # Fallback to filled mask if skimage unavailable
            scaled_w = int(self.image_width * self.zoom_level)
            scaled_h = int(self.image_height * self.zoom_level)
            img_x = (self.width() - scaled_w) / 2 + self.pan_offset.x()
            img_y = (self.height() - scaled_h) / 2 + self.pan_offset.y()
            target = QRectF(img_x, img_y, scaled_w, scaled_h)
            self._draw_filled_mask(painter, channel_overlays, overlay, target)
            return

        line_width = max(1, min(3, int(1.5 * self.zoom_level / max(self._fit_zoom_level, 0.01))))

        for ch_idx, co in channel_overlays.items():
            mask = co.get('binary_mask')
            if mask is None:
                continue
            r, g, b = co.get('color', (0, 255, 255))
            pen = QPen(QColor(r, g, b, 200), line_width)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)

            contours = find_contours(mask.astype(np.float64), 0.5)
            for contour in contours:
                if len(contour) < 3:
                    continue
                # Convert contour points (row, col) to widget coords
                points = []
                for row, col in contour:
                    wp = self._image_to_widget_coords(col, row)
                    points.append(wp)
                # Draw as connected line segments
                for i in range(len(points) - 1):
                    painter.drawLine(points[i], points[i + 1])
                # Close the contour
                if len(points) > 2:
                    painter.drawLine(points[-1], points[0])

    def _draw_filled_mask(self, painter: QPainter,
                          channel_overlays: dict, overlay: dict,
                          target: QRectF):
        """Draw filled semi-transparent mask overlay (original behavior)."""
        cache_key = tuple(
            (ch_idx, id(co.get('binary_mask')))
            for ch_idx, co in sorted(channel_overlays.items())
            if co.get('binary_mask') is not None
        )

        if cache_key != self._quant_overlay_mask_id:
            first_mask = next(
                (co['binary_mask'] for co in channel_overlays.values()
                 if co.get('binary_mask') is not None), None
            )
            if first_mask is not None:
                h, w = first_mask.shape
                composite = np.zeros((h, w, 4), dtype=np.uint8)

                for ch_idx, co in channel_overlays.items():
                    mask = co.get('binary_mask')
                    if mask is None:
                        continue
                    r, g, b = co.get('color', (0, 255, 255))
                    # Additive blend per-channel mask at alpha 40
                    composite[mask, 0] = np.minimum(
                        255, composite[mask, 0].astype(np.uint16) + r // 4
                    ).astype(np.uint8)
                    composite[mask, 1] = np.minimum(
                        255, composite[mask, 1].astype(np.uint16) + g // 4
                    ).astype(np.uint8)
                    composite[mask, 2] = np.minimum(
                        255, composite[mask, 2].astype(np.uint16) + b // 4
                    ).astype(np.uint8)
                    composite[mask, 3] = np.minimum(
                        255, composite[mask, 3].astype(np.uint16) + 40
                    ).astype(np.uint8)

                # Overlap mask in white at higher alpha
                overlap_mask = overlay.get('overlap_mask')
                if overlap_mask is not None:
                    composite[overlap_mask, 0] = 255
                    composite[overlap_mask, 1] = 255
                    composite[overlap_mask, 2] = 255
                    composite[overlap_mask, 3] = 80

                self._quant_overlay_qimage = QImage(
                    composite.data, w, h, 4 * w, QImage.Format_RGBA8888
                ).copy()
                self._quant_overlay_mask_id = cache_key

        if self._quant_overlay_qimage is not None:
            source = QRectF(
                0, 0,
                self._quant_overlay_qimage.width(),
                self._quant_overlay_qimage.height()
            )
            painter.drawImage(target, self._quant_overlay_qimage, source)

    # =====================================================================
    # Mouse / keyboard events
    # =====================================================================

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
        """Handle mouse press — ROI drawing or panning."""
        if event.button() == Qt.LeftButton:
            img_coords = self._widget_to_image_coords(event.pos())

            # ROI drawing mode
            if self.roi_mode and img_coords:
                self.roi_start_pos = img_coords
                self.roi_rect = (int(img_coords[0]), int(img_coords[1]), 0, 0)
                self.update()
                return

            # Default: start panning
            self.is_panning = True
            self.last_mouse_pos = event.pos()
            self.setCursor(Qt.ClosedHandCursor)

    def mouseMoveEvent(self, event):
        """Handle mouse drag — ROI drawing or panning."""
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

        # Panning
        if self.is_panning and self.last_mouse_pos is not None:
            delta = event.pos() - self.last_mouse_pos
            self.pan_offset += QPointF(delta)
            self.last_mouse_pos = event.pos()
            self.update()

    def mouseReleaseEvent(self, event):
        """Handle mouse release — finish ROI or panning."""
        if event.button() == Qt.LeftButton:
            # Finish ROI drawing
            if self.roi_mode and self.roi_start_pos is not None:
                self.roi_start_pos = None
                if self.roi_rect and self.roi_rect[2] > 2 and self.roi_rect[3] > 2:
                    x, y, w, h = self.roi_rect
                    self.roi_drawn.emit(x, y, w, h)
                self.roi_mode = False
                self.update()
                return

            # Finish panning
            self.is_panning = False
            self.last_mouse_pos = None
            self.setCursor(Qt.ArrowCursor)

    def keyPressEvent(self, event):
        """Handle key presses — Escape cancels ROI mode."""
        if event.key() == Qt.Key_Escape:
            if self.roi_mode:
                self.roi_mode = False
                self.roi_start_pos = None
                self.roi_rect = None
                self.update()


# =========================================================================
# QuantificationToolWindow — main standalone window
# =========================================================================

class QuantificationToolWindow(QWidget):
    """
    Standalone Image Quantification tool window.

    Provides CZI file loading, multi-channel cell counting with colocalization
    analysis, ROI-based density measurement, and CSV export.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Image Quantification — FNT")
        self.resize(1200, 800)

        # State
        self.czi_files: List[str] = []
        self.current_file_idx: int = 0
        self.image_data: Optional[CZIImageData] = None
        self.processor = CZIImageProcessor()
        self.load_worker: Optional[CZILoadWorker] = None
        self.batch_worker: Optional[BatchAnalysisWorker] = None

        self._setup_ui()
        self._apply_styles()
        self._setup_shortcuts()

    def _setup_ui(self):
        """Build the main UI layout."""
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(8, 8, 8, 8)

        # Title
        title = QLabel("Image Quantification")
        title.setFont(QFont("Arial", 16, QFont.Bold))
        title.setStyleSheet("color: #0078d4; background-color: transparent;")
        main_layout.addWidget(title)

        # Description
        desc = QLabel("Cell counting and analysis for CZI microscopy images")
        desc.setFont(QFont("Arial", 10))
        desc.setStyleSheet("color: #999999; font-style: italic; margin-bottom: 6px;")
        main_layout.addWidget(desc)

        # Splitter: left panel + right preview
        splitter = QSplitter(Qt.Horizontal)

        # --- Left panel (scroll area) ---
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setFixedWidth(340)

        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(4, 4, 4, 4)

        # Input section
        self._create_input_section(left_layout)

        # Quantification panel (from quantification_widgets.py)
        from .quantification_widgets import QuantificationPanel
        self.quant_panel = QuantificationPanel(title="Quantification")

        # Connect quant panel signals
        self.quant_panel.request_analysis.connect(self._on_quant_requested)
        self.quant_panel.request_threshold_preview.connect(
            self._on_quant_threshold_preview
        )
        self.quant_panel.overlay_updated.connect(self._on_quant_overlay_updated)
        self.quant_panel.overlay_cleared.connect(self._on_quant_overlay_cleared)
        self.quant_panel.request_roi_mode.connect(self._on_quant_roi_mode)
        self.quant_panel.request_overlay_export.connect(self._export_overlay_image)
        self.quant_panel.request_batch_analysis.connect(self._on_batch_requested)
        self.quant_panel.zoom_to_particle.connect(self._on_zoom_to_particle)
        self.quant_panel.quant_mode_entered.connect(self._update_preview)
        self.quant_panel.quant_mode_exited.connect(self._update_preview)

        left_layout.addWidget(self.quant_panel)
        left_layout.addStretch()

        left_scroll.setWidget(left_container)
        splitter.addWidget(left_scroll)

        # --- Right panel (preview + zoom bar) ---
        right_container = QWidget()
        right_layout = QVBoxLayout(right_container)
        right_layout.setContentsMargins(0, 0, 0, 0)

        # Preview widget
        self.preview = QuantPreviewWidget()
        self.preview.roi_drawn.connect(self._on_roi_drawn)
        right_layout.addWidget(self.preview, 1)

        # Zoom controls bar
        zoom_bar = QHBoxLayout()
        zoom_bar.setContentsMargins(4, 2, 4, 2)

        self.btn_zoom_out = QPushButton("−")
        self.btn_zoom_out.setFixedWidth(30)
        self.btn_zoom_out.clicked.connect(self._zoom_out)
        zoom_bar.addWidget(self.btn_zoom_out)

        self.zoom_label = QLabel("100%")
        self.zoom_label.setAlignment(Qt.AlignCenter)
        self.zoom_label.setFixedWidth(60)
        zoom_bar.addWidget(self.zoom_label)

        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setFixedWidth(30)
        self.btn_zoom_in.clicked.connect(self._zoom_in)
        zoom_bar.addWidget(self.btn_zoom_in)

        self.btn_zoom_fit = QPushButton("Fit")
        self.btn_zoom_fit.setFixedWidth(50)
        self.btn_zoom_fit.clicked.connect(self._zoom_fit)
        zoom_bar.addWidget(self.btn_zoom_fit)

        zoom_bar.addStretch()
        right_layout.addLayout(zoom_bar)

        splitter.addWidget(right_container)

        # Splitter proportions
        splitter.setStretchFactor(0, 0)  # Left: fixed
        splitter.setStretchFactor(1, 1)  # Right: expanding

        main_layout.addWidget(splitter, 1)

        # Status bar
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(
            "color: #aaaaaa; font-size: 10px; padding: 2px 4px; "
            "border-top: 1px solid #3f3f3f;"
        )
        main_layout.addWidget(self.status_label)

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

        # Image info labels
        info_style = "font-size: 10px; color: #aaaaaa;"
        self.info_size_label = QLabel("Size: \u2014")
        self.info_size_label.setStyleSheet(info_style)
        group_layout.addWidget(self.info_size_label)

        self.info_pixel_label = QLabel("Pixel Size: \u2014")
        self.info_pixel_label.setStyleSheet(info_style)
        group_layout.addWidget(self.info_pixel_label)

        self.info_objective_label = QLabel("Objective: \u2014")
        self.info_objective_label.setStyleSheet(info_style)
        group_layout.addWidget(self.info_objective_label)

        group.setLayout(group_layout)
        layout.addWidget(group)

    # =====================================================================
    # File management
    # =====================================================================

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
        """Clear all loaded files and reset state."""
        self.czi_files.clear()
        self.current_file_idx = 0
        self.image_data = None
        self._update_file_list()
        self.quant_panel.clear_results()
        self.preview.set_image(None)
        self.preview.quant_overlay = None
        self.preview._quant_overlay_qimage = None
        self.preview._quant_overlay_mask_id = None
        self.status_label.setText("Cleared all files")

    def _update_file_list(self):
        """Update the file list widget."""
        self.file_list.clear()
        for filepath in self.czi_files:
            self.file_list.addItem(Path(filepath).name)

        if self.czi_files:
            self.file_list.setCurrentRow(self.current_file_idx)
            self.file_label.setText(
                f"File {self.current_file_idx + 1}/{len(self.czi_files)}"
            )
        else:
            self.file_label.setText("No files loaded")

    def _on_file_selected(self, item):
        """Handle file selection from list."""
        idx = self.file_list.row(item)
        if idx != self.current_file_idx:
            self.current_file_idx = idx
            self._load_current_file()

    def _prev_file(self):
        """Go to previous file."""
        if self.czi_files and self.current_file_idx > 0:
            self.current_file_idx -= 1
            self._load_current_file()
            self._update_file_list()

    def _next_file(self):
        """Go to next file."""
        if self.czi_files and self.current_file_idx < len(self.czi_files) - 1:
            self.current_file_idx += 1
            self._load_current_file()
            self._update_file_list()

    # =====================================================================
    # File loading
    # =====================================================================

    def _load_current_file(self):
        """Load the current CZI file in a background thread."""
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

        self.load_worker = CZILoadWorker(filepath)
        self.load_worker.progress.connect(
            lambda msg: self.status_label.setText(msg)
        )
        self.load_worker.loaded.connect(self._on_load_complete)
        self.load_worker.error.connect(self._on_load_error)
        self.load_worker.start()

    def _on_load_complete(self, data: CZIImageData):
        """Handle successful file load."""
        self.image_data = data
        self.processor.clear_bg_cache()

        # Update info labels
        self._update_image_info()

        # Update quantification panel channels
        self.quant_panel.set_channels(data.metadata.channels)
        self.quant_panel.clear_results()

        # Build initial preview
        self._update_preview()
        self.preview.fit_to_window()
        self._update_zoom_label()

        n_channels = len(data.channel_data)
        self.status_label.setText(
            f"Loaded: {Path(data.metadata.filepath).name} "
            f"({data.metadata.width}\u00d7{data.metadata.height}, "
            f"{n_channels} channels)"
        )

    def _on_load_error(self, error: str):
        """Handle file load error."""
        QMessageBox.critical(self, "Load Error", f"Failed to load file:\n{error}")
        self.status_label.setText(f"Error: {error}")

    def _update_image_info(self):
        """Update image info labels from metadata."""
        if self.image_data is None:
            return

        meta = self.image_data.metadata
        self.info_size_label.setText(f"Size: {meta.width} \u00d7 {meta.height}")

        if meta.pixel_size_um and meta.pixel_size_um > 0:
            self.info_pixel_label.setText(
                f"Pixel Size: {meta.pixel_size_um:.4f} \u00b5m/px"
            )
        else:
            self.info_pixel_label.setText("Pixel Size: \u2014")

        if meta.objective:
            obj_str = meta.objective
            if hasattr(meta, 'magnification') and meta.magnification:
                obj_str += f" {meta.magnification}x"
            if hasattr(meta, 'numerical_aperture') and meta.numerical_aperture:
                obj_str += f"/{meta.numerical_aperture}"
            self.info_objective_label.setText(f"Objective: {obj_str}")
        else:
            self.info_objective_label.setText("Objective: \u2014")

    # =====================================================================
    # Preview display
    # =====================================================================

    # Channel color name -> RGB tuple for false-color composite
    CHANNEL_COLORS = {
        'white': (1.0, 1.0, 1.0),
        'green': (0.0, 1.0, 0.0),
        'magenta': (1.0, 0.0, 1.0),
        'cyan': (0.0, 1.0, 1.0),
        'red': (1.0, 0.0, 0.0),
        'blue': (0.0, 0.4, 1.0),
        'yellow': (1.0, 1.0, 0.0),
        'gray': (0.7, 0.7, 0.7),
    }

    def _update_preview(self):
        """
        Update the preview image.

        Shows a false-colored composite of all enabled quantification channels
        with additive blending, using each channel's assigned color.
        The quant panel's own BG subtraction is applied.
        """
        if self.image_data is None:
            return

        enabled = self.quant_panel.get_enabled_channels()
        if not enabled:
            # Show first available channel if none enabled
            if self.image_data.channel_data:
                first_key = next(iter(self.image_data.channel_data))
                raw = self.image_data.channel_data[first_key]
                normalized = self.processor.normalize_image(raw)
                display = np.clip(normalized * 255, 0, 255).astype(np.uint8)
                display_rgb = np.stack([display, display, display], axis=2)
                self.preview.set_image(display_rgb)
            return

        quant_bg_method = self.quant_panel.get_bg_method()
        quant_bg_radius = self.quant_panel.get_bg_radius()

        # Build false-color composite from enabled channels
        first_ch = self.image_data.channel_data.get(enabled[0])
        if first_ch is None:
            return

        h, w = first_ch.shape
        composite = np.zeros((h, w, 3), dtype=np.float32)

        for ch_idx in enabled:
            raw = self.image_data.channel_data.get(ch_idx)
            if raw is None:
                continue
            normalized = self.processor.normalize_image(raw)

            if quant_bg_method != "none":
                quant_settings = ChannelDisplaySettings(
                    bg_subtract_method=quant_bg_method,
                    bg_subtract_radius=quant_bg_radius,
                )
                normalized = self.processor.apply_background_subtraction(
                    normalized, quant_settings, ch_idx
                )

            # Get channel color for false-coloring
            color_name = self.quant_panel._channel_colors.get(ch_idx, 'gray')
            color_rgb = self.CHANNEL_COLORS.get(color_name, (0.7, 0.7, 0.7))

            # Additive blend: channel intensity * color
            composite[:, :, 0] += normalized * color_rgb[0]
            composite[:, :, 1] += normalized * color_rgb[1]
            composite[:, :, 2] += normalized * color_rgb[2]

        # Clip and convert to uint8 RGB
        display_rgb = np.clip(composite * 255, 0, 255).astype(np.uint8)
        self.preview.set_image(display_rgb)

    def _get_quant_images(self):
        """
        Prepare normalized images for all enabled channels.

        Uses the quantification panel's own BG subtraction settings.
        Returns (images_dict, pixel_size_um, channel_names_dict) or
        (None, None, None) if no data.
        """
        if self.image_data is None:
            return None, None, None

        enabled_channels = self.quant_panel.get_enabled_channels()
        if not enabled_channels:
            return None, None, None

        quant_bg_method = self.quant_panel.get_bg_method()
        quant_bg_radius = self.quant_panel.get_bg_radius()

        images = {}
        for ch_idx in enabled_channels:
            if ch_idx not in self.image_data.channel_data:
                continue
            raw_channel = self.image_data.channel_data[ch_idx]
            normalized = self.processor.normalize_image(raw_channel)

            if quant_bg_method != "none":
                quant_settings = ChannelDisplaySettings(
                    bg_subtract_method=quant_bg_method,
                    bg_subtract_radius=quant_bg_radius,
                )
                normalized = self.processor.apply_background_subtraction(
                    normalized, quant_settings, ch_idx
                )
            images[ch_idx] = normalized

        if not images:
            return None, None, None

        pixel_size = self.image_data.metadata.pixel_size_um

        channel_names = {}
        for info in self.image_data.metadata.channels:
            if info.index in images:
                channel_names[info.index] = info.name

        return images, pixel_size, channel_names

    # =====================================================================
    # Quantification signal handlers
    # =====================================================================

    def _on_quant_requested(self):
        """Prepare data and run full multi-channel quantification."""
        images, pixel_size, channel_names = self._get_quant_images()
        if images is None:
            self.status_label.setText("No channels enabled for analysis")
            return
        self.quant_panel.run_multi_with_data(images, pixel_size, channel_names)

    def _on_quant_threshold_preview(self):
        """Prepare data and run live threshold preview (mask only)."""
        images, pixel_size, channel_names = self._get_quant_images()
        if images is None:
            return
        self.quant_panel.preview_threshold_with_data(
            images, pixel_size, channel_names
        )

    def _on_quant_overlay_updated(self, overlay: dict):
        """Set quantification overlay on the preview widget."""
        self.preview.quant_overlay = overlay
        self.preview.update()

    def _on_quant_overlay_cleared(self):
        """Clear quantification overlay from the preview widget."""
        self.preview.quant_overlay = None
        self.preview._quant_overlay_qimage = None
        self.preview._quant_overlay_mask_id = None
        self.preview.update()

    def _on_quant_roi_mode(self):
        """Enter ROI drawing mode for quantification."""
        self.preview.roi_mode = True
        self.preview.roi_rect = None
        self.preview.update()
        self.status_label.setText(
            "Draw ROI: click and drag on the image to select a region"
        )

    def _on_roi_drawn(self, x: int, y: int, w: int, h: int):
        """Handle ROI rectangle drawn on the preview widget."""
        self.quant_panel.set_roi(x, y, w, h)
        self.status_label.setText(f"ROI set: ({x}, {y}) {w}\u00d7{h} px")

    def _on_zoom_to_particle(self, cx: float, cy: float):
        """Zoom and pan to center on the selected particle centroid."""
        self.preview.zoom_to_point(cx, cy)
        self._update_zoom_label()

    # =====================================================================
    # Overlay image export (Improvement 1)
    # =====================================================================

    def _export_overlay_image(self):
        """Export the current preview with quant overlay as PNG/TIFF."""
        if self.image_data is None or self.preview.display_image is None:
            self.status_label.setText("No image to export")
            return

        filepath, _ = QFileDialog.getSaveFileName(
            self, "Export Overlay Image", "",
            "PNG Files (*.png);;TIFF Files (*.tiff *.tif);;All Files (*.*)"
        )
        if not filepath:
            return

        try:
            # Render to an offscreen QImage at full resolution
            img_w = self.image_data.metadata.width
            img_h = self.image_data.metadata.height

            # Build the base composite at full resolution
            export_image = QImage(img_w, img_h, QImage.Format_RGB888)
            export_image.fill(QColor(0, 0, 0))

            painter = QPainter(export_image)
            painter.setRenderHint(QPainter.Antialiasing)

            # Draw the base image
            source = QRectF(0, 0,
                            self.preview.display_image.width(),
                            self.preview.display_image.height())
            target = QRectF(0, 0, img_w, img_h)
            painter.drawImage(target, self.preview.display_image, source)

            # Draw quant overlay at full resolution
            overlay = self.preview.quant_overlay
            if overlay is not None:
                channel_overlays = overlay.get('channel_overlays', {})
                show_mask = overlay.get('show_mask', False)

                # Draw filled mask overlay at full res
                if show_mask and channel_overlays:
                    first_mask = next(
                        (co['binary_mask'] for co in channel_overlays.values()
                         if co.get('binary_mask') is not None), None
                    )
                    if first_mask is not None:
                        h, w = first_mask.shape
                        composite = np.zeros((h, w, 4), dtype=np.uint8)
                        for ch_idx, co in channel_overlays.items():
                            mask = co.get('binary_mask')
                            if mask is None:
                                continue
                            r, g, b = co.get('color', (0, 255, 255))
                            composite[mask, 0] = np.minimum(
                                255, composite[mask, 0].astype(np.uint16) + r // 3
                            ).astype(np.uint8)
                            composite[mask, 1] = np.minimum(
                                255, composite[mask, 1].astype(np.uint16) + g // 3
                            ).astype(np.uint8)
                            composite[mask, 2] = np.minimum(
                                255, composite[mask, 2].astype(np.uint16) + b // 3
                            ).astype(np.uint8)
                            composite[mask, 3] = np.minimum(
                                255, composite[mask, 3].astype(np.uint16) + 50
                            ).astype(np.uint8)

                        mask_qimg = QImage(
                            composite.data, w, h, 4 * w, QImage.Format_RGBA8888
                        ).copy()
                        mask_source = QRectF(0, 0, w, h)
                        painter.drawImage(target, mask_qimg, mask_source)

                # Draw centroids
                for ch_idx, co in channel_overlays.items():
                    centroids = co.get('centroids', [])
                    labels = co.get('labels', [])
                    r_c, g_c, b_c = co.get('color', (255, 255, 0))

                    for cx, cy in centroids:
                        painter.setPen(QPen(QColor(r_c, g_c, b_c), 2))
                        painter.setBrush(Qt.NoBrush)
                        radius = max(4, min(img_w, img_h) * 0.003)
                        painter.drawEllipse(QPointF(cx, cy), radius, radius)

                    # Labels
                    if labels and centroids:
                        font_size = max(8, min(img_w, img_h) // 150)
                        painter.setFont(QFont("Arial", font_size))
                        painter.setPen(QColor(r_c, g_c, b_c))
                        for (cx, cy), label_id in zip(centroids, labels):
                            painter.drawText(
                                int(cx) + int(radius) + 2,
                                int(cy) - 3,
                                str(label_id)
                            )

                # Draw ROI rectangles
                roi_defs = overlay.get('roi_definitions', [])
                for roi in roi_defs:
                    pen = QPen(QColor(255, 200, 0), 2, Qt.DashLine)
                    painter.setPen(pen)
                    painter.setBrush(QBrush(QColor(255, 200, 0, 25)))
                    painter.drawRect(QRectF(roi.x, roi.y, roi.w, roi.h))
                    font_size = max(10, min(img_w, img_h) // 100)
                    painter.setFont(QFont("Arial", font_size, QFont.Bold))
                    painter.setPen(QColor(255, 200, 0))
                    painter.drawText(roi.x + 4, roi.y - 4, roi.label)

            painter.end()
            export_image.save(filepath)
            self.status_label.setText(f"Overlay exported: {Path(filepath).name}")

        except Exception as e:
            QMessageBox.critical(
                self, "Export Error", f"Failed to export overlay:\n{str(e)}"
            )

    # =====================================================================
    # Batch processing (Improvement 2)
    # =====================================================================

    def _on_batch_requested(self):
        """Run batch analysis on all loaded files with current settings."""
        if not self.czi_files:
            self.status_label.setText("No files loaded for batch analysis")
            return

        if len(self.czi_files) < 2:
            self.status_label.setText("Batch analysis requires 2+ files")
            return

        # Ask for output file
        filepath, _ = QFileDialog.getSaveFileName(
            self, "Save Batch Results", "",
            "CSV Files (*.csv);;All Files (*.*)"
        )
        if not filepath:
            return

        # Disable UI during batch
        self.quant_panel.btn_run.setEnabled(False)
        self.status_label.setText("Starting batch analysis...")

        self.batch_worker = BatchAnalysisWorker(
            self.czi_files,
            self.quant_panel,
            self.processor,
            filepath,
        )
        self.batch_worker.progress.connect(self._on_batch_progress)
        self.batch_worker.finished.connect(self._on_batch_complete)
        self.batch_worker.error.connect(self._on_batch_error)
        self.batch_worker.start()

    def _on_batch_progress(self, filename: str, current: int, total: int):
        """Update status during batch processing."""
        self.status_label.setText(
            f"Batch: {filename} ({current}/{total})"
        )

    def _on_batch_complete(self, filepath: str, n_files: int):
        """Handle completed batch analysis."""
        self.quant_panel.btn_run.setEnabled(True)
        self.status_label.setText(
            f"Batch complete: {n_files} files \u2192 {Path(filepath).name}"
        )

    def _on_batch_error(self, error_msg: str):
        """Handle batch analysis error."""
        self.quant_panel.btn_run.setEnabled(True)
        self.status_label.setText("Batch analysis failed")
        QMessageBox.warning(
            self, "Batch Error", f"Batch analysis failed:\n{error_msg}"
        )

    # =====================================================================
    # Zoom controls
    # =====================================================================

    def _zoom_in(self):
        self.preview.zoom_level = min(50.0, self.preview.zoom_level * 1.25)
        self.preview.update()
        self._update_zoom_label()

    def _zoom_out(self):
        self.preview.zoom_level = max(0.01, self.preview.zoom_level / 1.25)
        self.preview.update()
        self._update_zoom_label()

    def _zoom_fit(self):
        self.preview.fit_to_window()
        self._update_zoom_label()

    def _update_zoom_label(self):
        self.zoom_label.setText(f"{self.preview.get_zoom_percent()}%")

    # =====================================================================
    # Styling
    # =====================================================================

    def _apply_styles(self):
        """Apply FNT dark theme stylesheet."""
        self.setStyleSheet("""
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

    def _setup_shortcuts(self):
        """Set up keyboard shortcuts for common actions."""
        # Ctrl+R: Run Analysis
        sc_run = QShortcut(QKeySequence("Ctrl+R"), self)
        sc_run.activated.connect(self._on_quant_requested)
        self.quant_panel.btn_run.setToolTip("Run Analysis (Ctrl+R)")

        # Ctrl+E: Export CSV
        sc_export = QShortcut(QKeySequence("Ctrl+E"), self)
        sc_export.activated.connect(self.quant_panel._export_csv)
        self.quant_panel.btn_export_csv.setToolTip("Export CSV (Ctrl+E)")

        # Ctrl+O: Add files
        sc_open = QShortcut(QKeySequence("Ctrl+O"), self)
        sc_open.activated.connect(self._add_files)
        self.btn_add_files.setToolTip("Add CZI files (Ctrl+O)")

        # F: Fit to window
        sc_fit = QShortcut(QKeySequence("F"), self)
        sc_fit.activated.connect(self._zoom_fit)
        self.btn_zoom_fit.setToolTip("Fit to window (F)")

        # M: Toggle mask
        sc_mask = QShortcut(QKeySequence("M"), self)
        sc_mask.activated.connect(
            lambda: self.quant_panel.cb_show_mask.setChecked(
                not self.quant_panel.cb_show_mask.isChecked()
            )
        )
        self.quant_panel.cb_show_mask.setToolTip("Toggle mask overlay (M)")

        # Left/Right: Navigate files
        sc_prev = QShortcut(QKeySequence("Left"), self)
        sc_prev.activated.connect(self._prev_file)
        sc_next = QShortcut(QKeySequence("Right"), self)
        sc_next.activated.connect(self._next_file)
        self.btn_prev.setToolTip("Previous file (\u2190)")
        self.btn_next.setToolTip("Next file (\u2192)")

        # Ctrl++/Ctrl+-: Zoom
        sc_zoom_in = QShortcut(QKeySequence("Ctrl+="), self)
        sc_zoom_in.activated.connect(self._zoom_in)
        sc_zoom_out = QShortcut(QKeySequence("Ctrl+-"), self)
        sc_zoom_out.activated.connect(self._zoom_out)
        self.btn_zoom_in.setToolTip("Zoom in (Ctrl++)")
        self.btn_zoom_out.setToolTip("Zoom out (Ctrl+-)")


# =========================================================================
# Standalone entry point
# =========================================================================

def main():
    """Run the Image Quantification tool standalone."""
    app = QApplication(sys.argv)
    window = QuantificationToolWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
