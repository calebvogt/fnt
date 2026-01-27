"""
USV Studio - Unified tool for USV detection, inspection, and classification.

A comprehensive PyQt5 application for:
1. Loading and browsing audio files
2. Running DSP-based USV detection
3. Manual labeling and ground-truthing
4. Training Random Forest classifiers
5. Applying ML models for automated detection

Author: FNT Project
"""

import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRectF, QTimer, QEvent
from PyQt5.QtGui import QFont, QColor, QPainter, QImage, QPen, QBrush
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QProgressBar, QGroupBox,
    QSpinBox, QDoubleSpinBox, QComboBox, QListWidget, QListWidgetItem,
    QScrollArea, QSplitter, QStatusBar, QMessageBox, QScrollBar,
    QSizePolicy, QFrame, QCheckBox
)
from scipy import signal

# Optional imports
try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except ImportError:
    HAS_SOUNDDEVICE = False

try:
    import soundfile as sf
    HAS_SOUNDFILE = True
except ImportError:
    HAS_SOUNDFILE = False


# =============================================================================
# Spectrogram Widget (from Inspector, with improvements)
# =============================================================================

class SpectrogramWidget(QWidget):
    """Custom widget for displaying spectrogram with detection boxes."""

    detection_selected = pyqtSignal(int)
    box_adjusted = pyqtSignal(int, float, float, float, float)
    drag_complete = pyqtSignal()  # Emitted when box drag is finished

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        # Audio data
        self.audio_data = None
        self.sample_rate = None
        self.total_duration = 0.0

        # Cached spectrogram
        self.spec_image = None
        self.cached_view_start = None
        self.cached_view_end = None

        # View state
        self.view_start = 0.0
        self.view_end = 10.0
        self.min_freq = 0
        self.max_freq = 125000

        # Detections
        self.detections = []
        self.current_detection_idx = -1

        # Interaction state
        self.drag_mode = None
        self.drag_start = None
        self.drag_start_box = None
        self.drag_detection_idx = None

        # Drawing new box
        self.is_drawing = False
        self.draw_start = None
        self.draw_current = None

        # Colormap
        self.colormap_lut = self._create_colormap_lut()

    def _create_colormap_lut(self):
        """Create viridis-like colormap lookup table."""
        colors = np.array([
            [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142],
            [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89],
            [180, 222, 44], [253, 231, 37],
        ], dtype=np.float32)

        n_colors = len(colors)
        indices = np.linspace(0, n_colors - 1, 256)
        lut = np.zeros((256, 3), dtype=np.uint8)

        for i, t in enumerate(indices):
            idx = int(t)
            frac = t - idx
            if idx >= n_colors - 1:
                lut[i] = colors[-1]
            else:
                lut[i] = (colors[idx] * (1 - frac) + colors[idx + 1] * frac).astype(np.uint8)

        return lut

    def set_audio_data(self, audio_data, sample_rate):
        """Set audio data for spectrogram computation."""
        self.audio_data = audio_data
        self.sample_rate = sample_rate

        if audio_data is not None and sample_rate is not None and len(audio_data) > 0:
            self.total_duration = len(audio_data) / sample_rate
            self.view_start = 0
            self.view_end = min(2.0, self.total_duration)
            self.max_freq = min(125000, sample_rate / 2)
        else:
            self.total_duration = 0
            self.view_start = 0
            self.view_end = 10.0

        self.cached_view_start = None
        self.cached_view_end = None
        self.spec_image = None

        if self.total_duration > 0:
            self._compute_view_spectrogram()
        self.update()

    def set_detections(self, detections, current_idx=-1):
        """Set detection boxes."""
        self.detections = detections
        self.current_detection_idx = current_idx
        self.update()

    def update_detection(self, idx, start_s, stop_s, min_freq, max_freq):
        """Update a single detection in place (used during drag)."""
        if 0 <= idx < len(self.detections):
            self.detections[idx]['start_seconds'] = start_s
            self.detections[idx]['stop_seconds'] = stop_s
            self.detections[idx]['min_freq_hz'] = min_freq
            self.detections[idx]['max_freq_hz'] = max_freq
            self.update()

    def set_view_range(self, start_s, end_s):
        """Set visible time range."""
        if self.total_duration > 0:
            self.view_start = max(0, start_s)
            self.view_end = min(self.total_duration, end_s)
            self._compute_view_spectrogram()
            self.update()

    def get_view_range(self):
        """Get current view range."""
        return self.view_start, self.view_end

    def get_total_duration(self):
        """Get total audio duration."""
        return self.total_duration

    def _compute_view_spectrogram(self):
        """Compute spectrogram for current view window."""
        if self.audio_data is None or self.sample_rate is None:
            self.spec_image = None
            return

        # Check cache - only use if position is nearly identical
        current_window = self.view_end - self.view_start
        if (self.cached_view_start is not None and
            self.cached_view_end is not None and
            self.spec_image is not None):
            cached_window = self.cached_view_end - self.cached_view_start
            zoom_similar = abs(current_window - cached_window) / (cached_window + 1e-10) < 0.05
            position_tolerance = current_window * 0.01
            same_position = (abs(self.view_start - self.cached_view_start) < position_tolerance and
                            abs(self.view_end - self.cached_view_end) < position_tolerance)
            if zoom_similar and same_position:
                return

        # Extract segment with padding
        pad_time = 0.1
        start_time = max(0, self.view_start - pad_time)
        end_time = min(self.total_duration, self.view_end + pad_time)

        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)
        segment = self.audio_data[start_sample:end_sample]

        if len(segment) < 512:
            self.spec_image = None
            return

        # Downsample if needed
        effective_sr = self.sample_rate
        if len(segment) > 2_000_000:
            downsample_factor = int(np.ceil(len(segment) / 2_000_000))
            segment = segment[::downsample_factor]
            effective_sr = self.sample_rate / downsample_factor

        # Compute spectrogram
        nperseg = min(512, len(segment) // 10)
        nperseg = max(128, nperseg)
        noverlap = int(nperseg * 0.75)
        nfft = max(nperseg, 512)

        frequencies, times, Sxx = signal.spectrogram(
            segment, fs=effective_sr,
            nperseg=nperseg, noverlap=noverlap, nfft=nfft, window='hann'
        )

        times = times + start_time
        spec_db = 10 * np.log10(Sxx + 1e-10)

        # Filter to view range
        time_mask = (times >= self.view_start) & (times <= self.view_end)
        if not np.any(time_mask):
            self.spec_image = None
            return

        view_spec = spec_db[:, time_mask]

        # Normalize and apply colormap
        vmin = np.percentile(view_spec, 5)
        vmax = np.percentile(view_spec, 99)
        normalized = np.clip((view_spec - vmin) / (vmax - vmin + 1e-10), 0, 1)
        indices = (normalized * 255).astype(np.uint8)
        indices = np.flipud(indices)

        rgb_data = self.colormap_lut[indices]

        height, width = indices.shape
        self.spec_image = QImage(
            rgb_data.data, width, height, width * 3,
            QImage.Format_RGB888
        ).copy()

        self.cached_view_start = self.view_start
        self.cached_view_end = self.view_end

    def paintEvent(self, event):
        """Paint spectrogram and detection boxes."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        painter.fillRect(self.rect(), QColor(30, 30, 30))

        if self.spec_image is None:
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignCenter, "No spectrogram data\nLoad audio files to begin")
            return

        spec_rect = self._get_spec_rect()
        scaled_image = self.spec_image.scaled(
            int(spec_rect.width()), int(spec_rect.height()),
            Qt.IgnoreAspectRatio, Qt.SmoothTransformation
        )
        painter.drawImage(spec_rect.topLeft(), scaled_image)

        # Draw detection boxes
        for i, det in enumerate(self.detections):
            self._draw_detection_box(painter, det, i, spec_rect)

        # Draw temp box if drawing
        if self.is_drawing and self.draw_start and self.draw_current:
            self._draw_temp_box(painter, spec_rect)

        self._draw_axes(painter, spec_rect)

    def _get_spec_rect(self):
        """Get spectrogram drawing area."""
        margin_left = 50
        margin_right = 10
        margin_top = 10
        margin_bottom = 40
        return QRectF(
            margin_left, margin_top,
            self.width() - margin_left - margin_right,
            self.height() - margin_top - margin_bottom
        )

    def _time_to_x(self, time_s, spec_rect):
        """Convert time to x coordinate."""
        if self.view_end <= self.view_start:
            return spec_rect.left()
        t = (time_s - self.view_start) / (self.view_end - self.view_start)
        return spec_rect.left() + t * spec_rect.width()

    def _x_to_time(self, x, spec_rect):
        """Convert x coordinate to time."""
        if spec_rect.width() <= 0:
            return self.view_start
        t = (x - spec_rect.left()) / spec_rect.width()
        return self.view_start + t * (self.view_end - self.view_start)

    def _freq_to_y(self, freq_hz, spec_rect):
        """Convert frequency to y coordinate."""
        if self.max_freq <= self.min_freq:
            return spec_rect.top()
        f = (freq_hz - self.min_freq) / (self.max_freq - self.min_freq)
        return spec_rect.bottom() - f * spec_rect.height()

    def _y_to_freq(self, y, spec_rect):
        """Convert y coordinate to frequency."""
        if spec_rect.height() <= 0:
            return self.min_freq
        f = (spec_rect.bottom() - y) / spec_rect.height()
        return self.min_freq + f * (self.max_freq - self.min_freq)

    def _draw_detection_box(self, painter, det, idx, spec_rect):
        """Draw a detection bounding box."""
        start_s = det.get('start_seconds', 0)
        stop_s = det.get('stop_seconds', 0)
        min_freq = det.get('min_freq_hz', self.min_freq)
        max_freq = det.get('max_freq_hz', self.max_freq)
        status = det.get('status', 'pending')

        # Handle NaN values
        import math
        if math.isnan(start_s) or math.isnan(stop_s):
            return
        if math.isnan(min_freq):
            min_freq = self.min_freq
        if math.isnan(max_freq):
            max_freq = self.max_freq

        x1 = self._time_to_x(start_s, spec_rect)
        x2 = self._time_to_x(stop_s, spec_rect)
        y1 = self._freq_to_y(max_freq, spec_rect)
        y2 = self._freq_to_y(min_freq, spec_rect)

        is_selected = idx == self.current_detection_idx

        # Color based on status - yellow for pending, green/red for labeled
        if status == 'accepted':
            color = QColor(0, 200, 0, 220)  # Bright green
            label = "A"
        elif status == 'rejected':
            color = QColor(255, 80, 80, 220)  # Bright red
            label = "R"
        else:
            color = QColor(255, 255, 0, 200)  # Yellow for pending
            label = None

        # Selected = white border; non-selected = yellow outline
        if is_selected:
            pen = QPen(QColor(255, 255, 255))
            pen.setWidth(4)
        else:
            pen = QPen(color)
            pen.setWidth(2)

        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), 30)))
        painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))

        # Draw status label above the box (for accepted/rejected)
        if label:
            center_x = (x1 + x2) / 2
            label_y = y1 - 4
            font = QFont("Arial", 14, QFont.Bold)
            painter.setFont(font)
            # White label for selected box, yellow for others
            if is_selected:
                painter.setPen(QColor(255, 255, 255))
            else:
                painter.setPen(QColor(255, 255, 0))  # Yellow
            painter.drawText(int(center_x - 10), int(label_y - 16), 20, 18, Qt.AlignCenter, label)

        # Draw frequency contour dots+lines for accepted detections
        if status == 'accepted':
            self._draw_freq_contour(painter, det, x1, x2, spec_rect)

    def _draw_freq_contour(self, painter, det, x1, x2, spec_rect):
        """Draw peak frequency sample dots and connecting lines."""
        import math
        # Collect peak_freq_N values
        points = []
        i = 1
        while True:
            key = f'peak_freq_{i}'
            val = det.get(key)
            if val is None:
                break
            if not (isinstance(val, (int, float)) and not math.isnan(val)):
                i += 1
                continue
            points.append(val)
            i += 1

        if len(points) < 2:
            return

        n = len(points)
        # Evenly space dots horizontally across the box
        dot_coords = []
        for j, freq in enumerate(points):
            t = j / (n - 1) if n > 1 else 0.5
            px = x1 + t * (x2 - x1)
            py = self._freq_to_y(freq, spec_rect)
            dot_coords.append((px, py))

        # Draw connecting lines (cyan)
        pen = QPen(QColor(0, 220, 255, 200))
        pen.setWidth(2)
        painter.setPen(pen)
        for j in range(len(dot_coords) - 1):
            painter.drawLine(
                int(dot_coords[j][0]), int(dot_coords[j][1]),
                int(dot_coords[j+1][0]), int(dot_coords[j+1][1])
            )

        # Draw dots
        painter.setBrush(QBrush(QColor(0, 220, 255, 230)))
        painter.setPen(QPen(QColor(255, 255, 255), 1))
        radius = 4
        for px, py in dot_coords:
            painter.drawEllipse(int(px - radius), int(py - radius), radius * 2, radius * 2)

    def _draw_temp_box(self, painter, spec_rect):
        """Draw temporary box while drawing."""
        x1 = self._time_to_x(self.draw_start[0], spec_rect)
        x2 = self._time_to_x(self.draw_current[0], spec_rect)
        y1 = self._freq_to_y(self.draw_start[1], spec_rect)
        y2 = self._freq_to_y(self.draw_current[1], spec_rect)

        pen = QPen(QColor(0, 120, 212))
        pen.setWidth(2)
        pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(QBrush(QColor(0, 120, 212, 40)))
        painter.drawRect(QRectF(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)))

    def _draw_axes(self, painter, spec_rect):
        """Draw axis labels with adaptive tick density."""
        painter.setPen(QColor(200, 200, 200))
        font = QFont("Arial", 8)
        painter.setFont(font)

        # Time axis - adaptive tick spacing based on view window
        view_duration = self.view_end - self.view_start
        if view_duration <= 0:
            return

        # Choose nice tick intervals based on view duration
        if view_duration < 0.05:
            tick_interval = 0.005
        elif view_duration < 0.1:
            tick_interval = 0.01
        elif view_duration < 0.5:
            tick_interval = 0.05
        elif view_duration < 1.0:
            tick_interval = 0.1
        elif view_duration < 5.0:
            tick_interval = 0.5
        elif view_duration < 20.0:
            tick_interval = 1.0
        elif view_duration < 60.0:
            tick_interval = 5.0
        else:
            tick_interval = 10.0

        # Determine decimal places for label
        if tick_interval < 0.01:
            fmt = "{:.3f}"
        elif tick_interval < 0.1:
            fmt = "{:.2f}"
        elif tick_interval < 1.0:
            fmt = "{:.1f}"
        else:
            fmt = "{:.1f}"

        # Draw minor ticks (subdivisions between major ticks)
        import math as _math
        minor_interval = tick_interval / 10.0
        first_minor = _math.ceil(self.view_start / minor_interval) * minor_interval
        t = first_minor
        painter.setPen(QColor(120, 120, 120))
        while t <= self.view_end:
            x = self._time_to_x(t, spec_rect)
            if spec_rect.left() <= x <= spec_rect.right():
                # Short tick mark at bottom of spectrogram
                painter.drawLine(int(x), int(spec_rect.bottom()), int(x), int(spec_rect.bottom() + 4))
            t += minor_interval

        # Draw major ticks with labels
        painter.setPen(QColor(200, 200, 200))
        first_tick = _math.ceil(self.view_start / tick_interval) * tick_interval
        t = first_tick
        while t <= self.view_end:
            x = self._time_to_x(t, spec_rect)
            if spec_rect.left() <= x <= spec_rect.right():
                # Taller tick mark for major ticks
                painter.drawLine(int(x), int(spec_rect.bottom()), int(x), int(spec_rect.bottom() + 8))
                painter.drawText(
                    int(x - 25), int(spec_rect.bottom() + 10),
                    50, 15, Qt.AlignCenter, fmt.format(t)
                )
            t += tick_interval

        painter.drawText(
            int(spec_rect.center().x() - 30), int(spec_rect.bottom() + 25),
            60, 15, Qt.AlignCenter, "Time (s)"
        )

        # Frequency axis
        n_freq_ticks = 5
        for i in range(n_freq_ticks + 1):
            freq = self.min_freq + (self.max_freq - self.min_freq) * i / n_freq_ticks
            y = self._freq_to_y(freq, spec_rect)
            painter.drawText(0, int(y - 7), 45, 14, Qt.AlignRight, f"{freq/1000:.0f}")

        painter.drawText(5, int(spec_rect.center().y() - 20), "kHz")

    def mousePressEvent(self, event):
        """Handle mouse press."""
        if event.button() != Qt.LeftButton:
            return

        spec_rect = self._get_spec_rect()
        pos = event.pos()

        if not spec_rect.contains(pos):
            return

        time_s = self._x_to_time(pos.x(), spec_rect)
        freq_hz = self._y_to_freq(pos.y(), spec_rect)

        edge, det_idx = self._find_edge_at_pos(pos, spec_rect)

        if edge:
            self.drag_mode = edge
            self.drag_detection_idx = det_idx
            self.drag_start = (time_s, freq_hz)

            if edge == 'move' and det_idx is not None and det_idx < len(self.detections):
                det = self.detections[det_idx]
                self.drag_start_box = (
                    det.get('start_seconds', 0),
                    det.get('stop_seconds', 0),
                    det.get('min_freq_hz', self.min_freq),
                    det.get('max_freq_hz', self.max_freq)
                )
        else:
            det_idx = self._find_detection_at_pos(pos, spec_rect)
            if det_idx >= 0:
                self.detection_selected.emit(det_idx)
            else:
                self.is_drawing = True
                self.draw_start = (time_s, freq_hz)
                self.draw_current = (time_s, freq_hz)

    def mouseMoveEvent(self, event):
        """Handle mouse move."""
        spec_rect = self._get_spec_rect()
        pos = event.pos()

        if self.is_drawing and self.draw_start:
            time_s = self._x_to_time(pos.x(), spec_rect)
            freq_hz = self._y_to_freq(pos.y(), spec_rect)
            self.draw_current = (time_s, freq_hz)
            self.update()
        elif self.drag_mode:
            self._handle_drag(pos, spec_rect)
        else:
            edge, _ = self._find_edge_at_pos(pos, spec_rect)
            if edge in ('resize_left', 'resize_right'):
                self.setCursor(Qt.SizeHorCursor)
            elif edge in ('resize_top', 'resize_bottom'):
                self.setCursor(Qt.SizeVerCursor)
            elif edge == 'move':
                self.setCursor(Qt.SizeAllCursor)
            else:
                self.setCursor(Qt.ArrowCursor)

    def mouseReleaseEvent(self, event):
        """Handle mouse release."""
        if event.button() != Qt.LeftButton:
            return

        was_dragging = self.drag_mode is not None

        if self.is_drawing and self.draw_start and self.draw_current:
            t1, f1 = self.draw_start
            t2, f2 = self.draw_current
            if abs(t2 - t1) > 0.001 and abs(f2 - f1) > 100:
                # Emit signal for new box (parent will handle creation)
                pass

        self.is_drawing = False
        self.draw_start = None
        self.draw_current = None
        self.drag_mode = None
        self.drag_detection_idx = None
        self.drag_start_box = None
        self.update()

        # Notify parent that drag is complete (for saving)
        if was_dragging:
            self.drag_complete.emit()

    def _find_edge_at_pos(self, pos, spec_rect, threshold=8):
        """Find if position is near a detection edge."""
        for i, det in enumerate(self.detections):
            start_s = det.get('start_seconds', 0)
            stop_s = det.get('stop_seconds', 0)
            min_freq = det.get('min_freq_hz', self.min_freq)
            max_freq = det.get('max_freq_hz', self.max_freq)

            x1 = self._time_to_x(start_s, spec_rect)
            x2 = self._time_to_x(stop_s, spec_rect)
            y1 = self._freq_to_y(max_freq, spec_rect)
            y2 = self._freq_to_y(min_freq, spec_rect)

            if abs(pos.x() - x1) < threshold and y1 <= pos.y() <= y2:
                return 'resize_left', i
            if abs(pos.x() - x2) < threshold and y1 <= pos.y() <= y2:
                return 'resize_right', i
            if abs(pos.y() - y1) < threshold and x1 <= pos.x() <= x2:
                return 'resize_top', i
            if abs(pos.y() - y2) < threshold and x1 <= pos.x() <= x2:
                return 'resize_bottom', i

            if i == self.current_detection_idx:
                if x1 < pos.x() < x2 and y1 < pos.y() < y2:
                    return 'move', i

        return None, None

    def _find_detection_at_pos(self, pos, spec_rect):
        """Find detection at position."""
        for i, det in enumerate(self.detections):
            start_s = det.get('start_seconds', 0)
            stop_s = det.get('stop_seconds', 0)

            x1 = self._time_to_x(start_s, spec_rect)
            x2 = self._time_to_x(stop_s, spec_rect)

            if x1 <= pos.x() <= x2:
                return i
        return -1

    def _handle_drag(self, pos, spec_rect):
        """Handle drag for resize or move."""
        if self.drag_detection_idx is None or self.drag_detection_idx >= len(self.detections):
            return

        det = self.detections[self.drag_detection_idx]
        time_s = self._x_to_time(pos.x(), spec_rect)
        freq_hz = self._y_to_freq(pos.y(), spec_rect)

        start_s = det.get('start_seconds', 0)
        stop_s = det.get('stop_seconds', 0)
        min_freq = det.get('min_freq_hz', self.min_freq)
        max_freq = det.get('max_freq_hz', self.max_freq)

        if self.drag_mode == 'resize_left':
            start_s = min(time_s, stop_s - 0.001)
        elif self.drag_mode == 'resize_right':
            stop_s = max(time_s, start_s + 0.001)
        elif self.drag_mode == 'resize_top':
            max_freq = max(freq_hz, min_freq + 100)
        elif self.drag_mode == 'resize_bottom':
            min_freq = min(freq_hz, max_freq - 100)
        elif self.drag_mode == 'move' and self.drag_start and self.drag_start_box:
            delta_t = time_s - self.drag_start[0]
            delta_f = freq_hz - self.drag_start[1]

            orig_start, orig_stop, orig_min_f, orig_max_f = self.drag_start_box
            box_duration = orig_stop - orig_start
            box_freq_range = orig_max_f - orig_min_f

            start_s = max(0, orig_start + delta_t)
            stop_s = start_s + box_duration
            if stop_s > self.total_duration:
                stop_s = self.total_duration
                start_s = stop_s - box_duration

            min_freq = max(0, orig_min_f + delta_f)
            max_freq = min_freq + box_freq_range
            if max_freq > self.max_freq:
                max_freq = self.max_freq
                min_freq = max_freq - box_freq_range

        self.box_adjusted.emit(self.drag_detection_idx, start_s, stop_s, min_freq, max_freq)


# =============================================================================
# Worker Threads
# =============================================================================

class DSPDetectionWorker(QThread):
    """Worker thread for DSP detection."""
    progress = pyqtSignal(str, int, int)  # filename, current, total
    file_complete = pyqtSignal(str, int)  # filename, n_detections
    all_complete = pyqtSignal(dict)  # results dict
    error = pyqtSignal(str, str)  # filename, error message

    def __init__(self, files: List[str], config: dict):
        super().__init__()
        self.files = files
        self.config = config
        self.results = {}
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        from fnt.usv.usv_detector.dsp_detector import DSPDetector
        from fnt.usv.usv_detector.config import USVDetectorConfig

        config = USVDetectorConfig(
            min_freq_hz=self.config.get('min_freq_hz', 25000),
            max_freq_hz=self.config.get('max_freq_hz', 65000),
            energy_threshold_db=self.config.get('energy_threshold_db', 10.0),
            min_duration_ms=self.config.get('min_duration_ms', 5.0),
            max_duration_ms=self.config.get('max_duration_ms', 300.0),
            min_gap_ms=self.config.get('min_gap_ms', 5.0),
            noise_percentile=self.config.get('noise_percentile', 25.0),
            nperseg=self.config.get('nperseg', 512),
            noverlap=self.config.get('noverlap', 384),
        )

        detector = DSPDetector(config)
        total = len(self.files)

        for i, filepath in enumerate(self.files):
            if self._stop_requested:
                break

            filename = os.path.basename(filepath)
            self.progress.emit(filename, i, total)

            try:
                detections = detector.detect_file(filepath)
                self.results[filepath] = detections
                self.file_complete.emit(filename, len(detections))
            except Exception as e:
                self.error.emit(filename, str(e))
                self.results[filepath] = []

        self.all_complete.emit(self.results)


class RFDetectionWorker(QThread):
    """Worker thread for Random Forest detection."""
    progress = pyqtSignal(str, int, int)
    file_complete = pyqtSignal(str, int)
    all_complete = pyqtSignal(dict)
    error = pyqtSignal(str, str)

    def __init__(self, files: List[str], model_dir: str, dsp_config: dict):
        super().__init__()
        self.files = files
        self.model_dir = model_dir
        self.dsp_config = dsp_config
        self.results = {}
        self._stop_requested = False

    def stop(self):
        self._stop_requested = True

    def run(self):
        from fnt.usv.usv_classifier import USVClassifier, extract_features_from_detection
        from fnt.usv.usv_detector.dsp_detector import DSPDetector
        from fnt.usv.usv_detector.config import USVDetectorConfig

        # Load classifier
        try:
            classifier = USVClassifier.load(self.model_dir)
        except Exception as e:
            self.error.emit("Model", f"Failed to load model: {e}")
            return

        # Create DSP detector for candidate generation
        config = USVDetectorConfig(
            min_freq_hz=self.dsp_config.get('min_freq_hz', 25000),
            max_freq_hz=self.dsp_config.get('max_freq_hz', 65000),
            energy_threshold_db=self.dsp_config.get('energy_threshold_db', 8.0),  # Lower threshold
            min_duration_ms=self.dsp_config.get('min_duration_ms', 3.0),
            max_duration_ms=self.dsp_config.get('max_duration_ms', 500.0),
        )
        detector = DSPDetector(config)

        total = len(self.files)

        for i, filepath in enumerate(self.files):
            if self._stop_requested:
                break

            filename = os.path.basename(filepath)
            self.progress.emit(filename, i, total)

            try:
                # Load audio
                audio_data, sample_rate = self._load_audio(filepath)

                # Get DSP candidates
                candidates = detector.detect_file(filepath)

                if len(candidates) == 0:
                    self.results[filepath] = []
                    self.file_complete.emit(filename, 0)
                    continue

                # Extract features for each candidate
                features_list = []
                for _, row in candidates.iterrows():
                    features = extract_features_from_detection(
                        audio_data, sample_rate,
                        row['start_seconds'], row['stop_seconds'],
                        row.get('min_freq_hz', config.min_freq_hz),
                        row.get('max_freq_hz', config.max_freq_hz)
                    )
                    features_list.append(features)

                features_df = pd.DataFrame(features_list)

                # Predict
                predictions = classifier.predict(features_df)
                probabilities = classifier.predict_proba(features_df)

                # Filter to USV predictions
                usv_mask = predictions == 'usv'
                usv_detections = candidates[usv_mask].copy()
                usv_detections['rf_confidence'] = probabilities[usv_mask, 1]
                usv_detections['status'] = 'pending'
                usv_detections['source'] = 'rf'

                self.results[filepath] = usv_detections
                self.file_complete.emit(filename, len(usv_detections))

            except Exception as e:
                self.error.emit(filename, str(e))
                self.results[filepath] = pd.DataFrame()

        self.all_complete.emit(self.results)

    def _load_audio(self, filepath):
        """Load audio file."""
        if HAS_SOUNDFILE:
            try:
                audio, sr = sf.read(filepath, dtype='float32')
                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
                return audio, sr
            except:
                pass

        # Fallback to ffmpeg
        import subprocess
        import tempfile

        probe_cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'a:0',
            '-show_entries', 'stream=sample_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1', filepath
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        sr = int(result.stdout.strip())

        with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as tmp:
            temp_path = tmp.name

        try:
            convert_cmd = [
                'ffmpeg', '-i', filepath, '-f', 'f32le', '-acodec', 'pcm_f32le',
                '-ac', '1', '-y', temp_path
            ]
            subprocess.run(convert_cmd, capture_output=True, check=True)
            audio = np.fromfile(temp_path, dtype=np.float32)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        return audio, sr


# =============================================================================
# Main USV Studio Window
# =============================================================================

class USVStudioWindow(QMainWindow):
    """Main window for USV Studio."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("FNT USV Studio")
        self.setMinimumSize(1000, 700)
        self.resize(1400, 900)  # Initial size, but resizable

        # State
        self.audio_files = []  # List of audio file paths
        self.current_file_idx = 0
        self.audio_data = None
        self.sample_rate = None
        self.detections_df = None  # Current file's detections
        self.current_detection_idx = 0
        self.all_detections = {}  # filepath -> DataFrame
        self.dsp_queue = []  # Files queued for DSP detection
        self.current_model_dir = None

        # Playback state
        self.is_playing = False
        self.playback_speed = 0.1
        self.use_heterodyne = False

        # Workers
        self.dsp_worker = None
        self.rf_worker = None

        self._setup_ui()
        self._apply_styles()

        # Install event filter to capture arrow keys before child widgets consume them
        QApplication.instance().installEventFilter(self)

    def eventFilter(self, obj, event):
        """Intercept key presses globally to handle arrow key shortcuts."""
        if event.type() == QEvent.KeyPress:
            key = event.key()
            if key == Qt.Key_Right and self.btn_next_det.isEnabled():
                self._flash_button(self.btn_next_det)
                self.next_detection()
                return True
            elif key == Qt.Key_Left and self.btn_prev_det.isEnabled():
                self._flash_button(self.btn_prev_det)
                self.prev_detection()
                return True
        return super().eventFilter(obj, event)

    def _setup_ui(self):
        """Setup the main UI."""
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Left panel (scrollable - vertical only)
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setFixedWidth(340)  # Fixed width to prevent horizontal scroll

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(8)

        # Build sections
        self._create_input_section(left_layout)
        self._create_dsp_section(left_layout)
        self._create_detection_section(left_layout)
        self._create_labeling_section(left_layout)
        self._create_playback_section(left_layout)
        self._create_ml_section(left_layout)

        left_layout.addStretch()
        left_scroll.setWidget(left_widget)

        # Right panel (spectrogram)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(2)

        # Spectrogram
        self.spectrogram = SpectrogramWidget()
        self.spectrogram.detection_selected.connect(self.on_detection_selected)
        self.spectrogram.box_adjusted.connect(self.on_box_adjusted)
        self.spectrogram.drag_complete.connect(self.on_drag_complete)
        right_layout.addWidget(self.spectrogram, 1)

        # Navigation bar
        nav_bar = QWidget()
        nav_layout = QHBoxLayout(nav_bar)
        nav_layout.setContentsMargins(5, 2, 5, 2)

        self.btn_pan_left = QPushButton("<")
        self.btn_pan_left.setObjectName("small_btn")
        self.btn_pan_left.setFixedWidth(24)
        self.btn_pan_left.clicked.connect(self.pan_left)
        nav_layout.addWidget(self.btn_pan_left)

        self.time_scrollbar = QScrollBar(Qt.Horizontal)
        self.time_scrollbar.setMinimum(0)
        self.time_scrollbar.setMaximum(1000)
        self.time_scrollbar.valueChanged.connect(self.on_scrollbar_changed)
        nav_layout.addWidget(self.time_scrollbar, 1)

        self.btn_pan_right = QPushButton(">")
        self.btn_pan_right.setObjectName("small_btn")
        self.btn_pan_right.setFixedWidth(24)
        self.btn_pan_right.clicked.connect(self.pan_right)
        nav_layout.addWidget(self.btn_pan_right)

        nav_layout.addWidget(QLabel("Window:"))

        self.btn_zoom_out = QPushButton("-")
        self.btn_zoom_out.setObjectName("small_btn")
        self.btn_zoom_out.setFixedWidth(24)
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        nav_layout.addWidget(self.btn_zoom_out)

        self.spin_view_window = QDoubleSpinBox()
        self.spin_view_window.setRange(0.1, 600.0)
        self.spin_view_window.setValue(2.0)
        self.spin_view_window.setSuffix(" s")
        self.spin_view_window.setFixedWidth(80)
        self.spin_view_window.valueChanged.connect(self.on_view_window_changed)
        nav_layout.addWidget(self.spin_view_window)

        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setObjectName("small_btn")
        self.btn_zoom_in.setFixedWidth(24)
        self.btn_zoom_in.clicked.connect(self.zoom_in)
        nav_layout.addWidget(self.btn_zoom_in)

        right_layout.addWidget(nav_bar)

        # Add to main layout
        main_layout.addWidget(left_scroll)
        main_layout.addWidget(right_panel, 1)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Welcome to USV Studio - Load audio files to begin")

    def _create_input_section(self, layout):
        """Create input/file selection section."""
        group = QGroupBox("1. Input")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(4)

        # Buttons
        btn_row = QHBoxLayout()
        btn_row.setSpacing(2)

        self.btn_add_folder = QPushButton("Add Folder")
        self.btn_add_folder.clicked.connect(self.add_folder)
        btn_row.addWidget(self.btn_add_folder)

        self.btn_add_files = QPushButton("Add Files")
        self.btn_add_files.clicked.connect(self.add_files)
        btn_row.addWidget(self.btn_add_files)

        self.btn_clear_files = QPushButton("Clear")
        self.btn_clear_files.setStyleSheet("background-color: #5c5c5c;")
        self.btn_clear_files.clicked.connect(self.clear_files)
        btn_row.addWidget(self.btn_clear_files)

        group_layout.addLayout(btn_row)

        # File list
        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(100)
        self.file_list.currentRowChanged.connect(self.on_file_selected)
        group_layout.addWidget(self.file_list)

        # File navigation
        nav_row = QHBoxLayout()
        nav_row.setSpacing(2)

        self.btn_prev_file = QPushButton("< Prev")
        self.btn_prev_file.setObjectName("small_btn")
        self.btn_prev_file.clicked.connect(self.prev_file)
        self.btn_prev_file.setEnabled(False)
        nav_row.addWidget(self.btn_prev_file)

        self.lbl_file_num = QLabel("File 0/0")
        self.lbl_file_num.setAlignment(Qt.AlignCenter)
        nav_row.addWidget(self.lbl_file_num, 1)

        self.btn_next_file = QPushButton("Next >")
        self.btn_next_file.setObjectName("small_btn")
        self.btn_next_file.clicked.connect(self.next_file)
        self.btn_next_file.setEnabled(False)
        nav_row.addWidget(self.btn_next_file)

        group_layout.addLayout(nav_row)

        # Open Folder button
        self.btn_open_folder = QPushButton("Open Folder")
        self.btn_open_folder.setStyleSheet("background-color: #5c5c5c;")
        self.btn_open_folder.clicked.connect(self.open_output_folder)
        self.btn_open_folder.setEnabled(False)
        group_layout.addWidget(self.btn_open_folder)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _create_dsp_section(self, layout):
        """Create DSP detection section."""
        group = QGroupBox("2. DSP Detection")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(4)

        # Frequency range
        freq_row = QHBoxLayout()
        freq_row.setSpacing(2)
        freq_row.addWidget(QLabel("Freq:"))

        self.spin_min_freq = QSpinBox()
        self.spin_min_freq.setRange(1000, 150000)
        self.spin_min_freq.setSingleStep(1000)
        self.spin_min_freq.setValue(25000)
        self.spin_min_freq.setSuffix(" Hz")
        freq_row.addWidget(self.spin_min_freq)

        freq_row.addWidget(QLabel("-"))

        self.spin_max_freq = QSpinBox()
        self.spin_max_freq.setRange(1000, 150000)
        self.spin_max_freq.setSingleStep(1000)
        self.spin_max_freq.setValue(65000)
        self.spin_max_freq.setSuffix(" Hz")
        freq_row.addWidget(self.spin_max_freq)

        group_layout.addLayout(freq_row)

        # Threshold
        thresh_row = QHBoxLayout()
        thresh_row.setSpacing(2)
        thresh_row.addWidget(QLabel("Threshold:"))

        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(1.0, 30.0)
        self.spin_threshold.setSingleStep(0.5)
        self.spin_threshold.setValue(10.0)
        self.spin_threshold.setSuffix(" dB")
        thresh_row.addWidget(self.spin_threshold)
        thresh_row.addStretch()

        group_layout.addLayout(thresh_row)

        # Duration
        dur_row = QHBoxLayout()
        dur_row.setSpacing(2)
        dur_row.addWidget(QLabel("Duration:"))

        self.spin_min_dur = QDoubleSpinBox()
        self.spin_min_dur.setRange(1.0, 100.0)
        self.spin_min_dur.setValue(5.0)
        self.spin_min_dur.setSuffix(" ms")
        dur_row.addWidget(self.spin_min_dur)

        dur_row.addWidget(QLabel("-"))

        self.spin_max_dur = QDoubleSpinBox()
        self.spin_max_dur.setRange(10.0, 5000.0)
        self.spin_max_dur.setValue(300.0)
        self.spin_max_dur.setSuffix(" ms")
        dur_row.addWidget(self.spin_max_dur)

        group_layout.addLayout(dur_row)

        # Advanced options (collapsible)
        self.chk_advanced = QCheckBox("Advanced Options")
        self.chk_advanced.toggled.connect(self._toggle_advanced_options)
        group_layout.addWidget(self.chk_advanced)

        self.advanced_widget = QWidget()
        adv_layout = QVBoxLayout(self.advanced_widget)
        adv_layout.setContentsMargins(10, 0, 0, 0)
        adv_layout.setSpacing(2)

        # Min gap
        gap_row = QHBoxLayout()
        gap_row.addWidget(QLabel("Min Gap:"))
        self.spin_min_gap = QDoubleSpinBox()
        self.spin_min_gap.setRange(0.0, 100.0)
        self.spin_min_gap.setValue(5.0)
        self.spin_min_gap.setSuffix(" ms")
        gap_row.addWidget(self.spin_min_gap)
        gap_row.addStretch()
        adv_layout.addLayout(gap_row)

        # Noise percentile
        noise_row = QHBoxLayout()
        noise_row.addWidget(QLabel("Noise %ile:"))
        self.spin_noise_pct = QDoubleSpinBox()
        self.spin_noise_pct.setRange(1.0, 50.0)
        self.spin_noise_pct.setValue(25.0)
        noise_row.addWidget(self.spin_noise_pct)
        noise_row.addStretch()
        adv_layout.addLayout(noise_row)

        # FFT params
        fft_row = QHBoxLayout()
        fft_row.addWidget(QLabel("FFT:"))
        self.spin_nperseg = QSpinBox()
        self.spin_nperseg.setRange(64, 2048)
        self.spin_nperseg.setSingleStep(64)
        self.spin_nperseg.setValue(512)
        fft_row.addWidget(self.spin_nperseg)
        fft_row.addWidget(QLabel("Overlap:"))
        self.spin_noverlap = QSpinBox()
        self.spin_noverlap.setRange(0, 1024)
        self.spin_noverlap.setSingleStep(64)
        self.spin_noverlap.setValue(384)
        fft_row.addWidget(self.spin_noverlap)
        adv_layout.addLayout(fft_row)

        # Frequency samples
        freq_samp_row = QHBoxLayout()
        freq_samp_row.addWidget(QLabel("Freq Samples:"))
        self.spin_freq_samples = QSpinBox()
        self.spin_freq_samples.setRange(3, 10)
        self.spin_freq_samples.setValue(5)
        self.spin_freq_samples.setToolTip("Number of evenly-spaced peak frequency samples per call")
        freq_samp_row.addWidget(self.spin_freq_samples)
        freq_samp_row.addStretch()
        adv_layout.addLayout(freq_samp_row)

        self.advanced_widget.setVisible(False)
        group_layout.addWidget(self.advanced_widget)

        # Queue display
        self.lbl_queue = QLabel("Queue: 0 files")
        self.lbl_queue.setStyleSheet("color: #999999;")
        group_layout.addWidget(self.lbl_queue)

        # Queue buttons
        queue_row = QHBoxLayout()
        queue_row.setSpacing(2)

        self.btn_add_to_queue = QPushButton("Add to Queue")
        self.btn_add_to_queue.clicked.connect(self.add_to_queue)
        self.btn_add_to_queue.setEnabled(False)
        queue_row.addWidget(self.btn_add_to_queue)

        self.btn_clear_queue = QPushButton("Clear")
        self.btn_clear_queue.setStyleSheet("background-color: #5c5c5c;")
        self.btn_clear_queue.clicked.connect(self.clear_queue)
        queue_row.addWidget(self.btn_clear_queue)

        group_layout.addLayout(queue_row)

        # Run button
        self.btn_run_dsp = QPushButton("Run DSP Detection")
        self.btn_run_dsp.setStyleSheet("background-color: #0078d4;")
        self.btn_run_dsp.clicked.connect(self.run_dsp_detection)
        self.btn_run_dsp.setEnabled(False)
        group_layout.addWidget(self.btn_run_dsp)

        # Progress
        self.dsp_progress = QProgressBar()
        self.dsp_progress.setValue(0)
        self.dsp_progress.setVisible(False)
        group_layout.addWidget(self.dsp_progress)

        self.lbl_dsp_status = QLabel("")
        self.lbl_dsp_status.setStyleSheet("color: #999999; font-size: 9px;")
        group_layout.addWidget(self.lbl_dsp_status)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _create_detection_section(self, layout):
        """Create current detection section."""
        group = QGroupBox("3. Current Detection")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(4)

        # Navigation
        nav_row = QHBoxLayout()
        nav_row.setSpacing(2)

        self.btn_prev_det = QPushButton("<")
        self.btn_prev_det.setObjectName("small_btn")
        self.btn_prev_det.clicked.connect(self.prev_detection)
        self.btn_prev_det.setEnabled(False)
        nav_row.addWidget(self.btn_prev_det)

        self.lbl_det_num = QLabel("Det 0/0")
        self.lbl_det_num.setAlignment(Qt.AlignCenter)
        nav_row.addWidget(self.lbl_det_num, 1)

        self.btn_next_det = QPushButton(">")
        self.btn_next_det.setObjectName("small_btn")
        self.btn_next_det.clicked.connect(self.next_detection)
        self.btn_next_det.setEnabled(False)
        nav_row.addWidget(self.btn_next_det)

        group_layout.addLayout(nav_row)

        # Time
        time_row = QHBoxLayout()
        time_row.setSpacing(2)
        time_row.addWidget(QLabel("Start:"))
        self.spin_start = QDoubleSpinBox()
        self.spin_start.setDecimals(4)
        self.spin_start.setRange(0, 9999)
        self.spin_start.setSuffix(" s")
        self.spin_start.valueChanged.connect(self.on_time_changed)
        time_row.addWidget(self.spin_start, 1)

        time_row.addWidget(QLabel("Stop:"))
        self.spin_stop = QDoubleSpinBox()
        self.spin_stop.setDecimals(4)
        self.spin_stop.setRange(0, 9999)
        self.spin_stop.setSuffix(" s")
        self.spin_stop.valueChanged.connect(self.on_time_changed)
        time_row.addWidget(self.spin_stop, 1)

        group_layout.addLayout(time_row)

        # Frequency
        freq_row = QHBoxLayout()
        freq_row.setSpacing(2)
        freq_row.addWidget(QLabel("Min F:"))
        self.spin_det_min_freq = QSpinBox()
        self.spin_det_min_freq.setRange(0, 200000)
        self.spin_det_min_freq.setSingleStep(1000)
        self.spin_det_min_freq.setSuffix(" Hz")
        self.spin_det_min_freq.valueChanged.connect(self.on_freq_changed)
        freq_row.addWidget(self.spin_det_min_freq, 1)

        freq_row.addWidget(QLabel("Max:"))
        self.spin_det_max_freq = QSpinBox()
        self.spin_det_max_freq.setRange(0, 200000)
        self.spin_det_max_freq.setSingleStep(1000)
        self.spin_det_max_freq.setSuffix(" Hz")
        self.spin_det_max_freq.valueChanged.connect(self.on_freq_changed)
        freq_row.addWidget(self.spin_det_max_freq, 1)

        group_layout.addLayout(freq_row)

        # Info
        self.lbl_det_info = QLabel("Peak: -- Hz | Dur: -- ms")
        self.lbl_det_info.setStyleSheet("color: #999999;")
        group_layout.addWidget(self.lbl_det_info)

        # Status
        self.lbl_det_status = QLabel("Status: --")
        self.lbl_det_status.setStyleSheet("font-weight: bold;")
        group_layout.addWidget(self.lbl_det_status)

        # Progress (moved from section 6)
        self.lbl_progress = QLabel("0/0 reviewed")
        self.lbl_progress.setStyleSheet("color: #999999;")
        group_layout.addWidget(self.lbl_progress)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        group_layout.addWidget(self.progress_bar)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _create_labeling_section(self, layout):
        """Create labeling/action section."""
        group = QGroupBox("4. Labeling")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(4)

        # Accept/Reject
        btn_row1 = QHBoxLayout()
        btn_row1.setSpacing(2)

        self.btn_accept = QPushButton("Accept")
        self.btn_accept.setObjectName("accept_btn")
        self.btn_accept.clicked.connect(self.accept_detection)
        self.btn_accept.setEnabled(False)
        btn_row1.addWidget(self.btn_accept)

        self.btn_reject = QPushButton("Reject")
        self.btn_reject.setObjectName("reject_btn")
        self.btn_reject.clicked.connect(self.reject_detection)
        self.btn_reject.setEnabled(False)
        btn_row1.addWidget(self.btn_reject)

        group_layout.addLayout(btn_row1)

        # Add/Delete
        btn_row2 = QHBoxLayout()
        btn_row2.setSpacing(2)

        self.btn_add_usv = QPushButton("+ Add USV")
        self.btn_add_usv.setStyleSheet("background-color: #6b4c9a;")
        self.btn_add_usv.clicked.connect(self.add_new_usv)
        self.btn_add_usv.setEnabled(False)
        btn_row2.addWidget(self.btn_add_usv)

        self.btn_delete = QPushButton("Delete")
        self.btn_delete.setStyleSheet("background-color: #5c5c5c;")
        self.btn_delete.clicked.connect(self.delete_current)
        self.btn_delete.setEnabled(False)
        btn_row2.addWidget(self.btn_delete)

        group_layout.addLayout(btn_row2)

        # Delete pending
        self.btn_delete_pending = QPushButton("Delete All Pending")
        self.btn_delete_pending.setStyleSheet("background-color: #5c5c5c;")
        self.btn_delete_pending.clicked.connect(self.delete_all_pending)
        self.btn_delete_pending.setEnabled(False)
        group_layout.addWidget(self.btn_delete_pending)

        # Instructions
        lbl_hint = QLabel("Drag edges to resize, center to move")
        lbl_hint.setStyleSheet("color: #666666; font-size: 9px; font-style: italic;")
        group_layout.addWidget(lbl_hint)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _create_playback_section(self, layout):
        """Create playback section."""
        group = QGroupBox("5. Playback")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(4)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(2)

        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.toggle_playback)
        self.btn_play.setEnabled(False)
        btn_row.addWidget(self.btn_play)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet("background-color: #5c5c5c;")
        self.btn_stop.clicked.connect(self.stop_playback)
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_stop)

        group_layout.addLayout(btn_row)

        # Speed
        speed_row = QHBoxLayout()
        speed_row.setSpacing(2)
        speed_row.addWidget(QLabel("Speed:"))

        self.combo_speed = QComboBox()
        self.combo_speed.addItem("0.01x", 0.01)
        self.combo_speed.addItem("0.05x", 0.05)
        self.combo_speed.addItem("0.1x", 0.1)
        self.combo_speed.addItem("0.2x", 0.2)
        self.combo_speed.addItem("0.5x", 0.5)
        self.combo_speed.addItem("1x", 1.0)
        self.combo_speed.addItem("Heterodyne", "heterodyne")
        self.combo_speed.setCurrentIndex(2)
        self.combo_speed.currentIndexChanged.connect(self.on_speed_changed)
        speed_row.addWidget(self.combo_speed, 1)

        group_layout.addLayout(speed_row)

        if not HAS_SOUNDDEVICE:
            lbl_warn = QLabel("sounddevice not installed")
            lbl_warn.setStyleSheet("color: #d13438; font-size: 9px;")
            group_layout.addWidget(lbl_warn)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _create_ml_section(self, layout):
        """Create machine learning train and predict sections."""
        # --- Train section ---
        train_group = QGroupBox("7. Train ML Model")
        train_layout = QVBoxLayout()
        train_layout.setSpacing(4)

        self.lbl_training_data = QLabel("Training: 0 USV, 0 rejected")
        self.lbl_training_data.setStyleSheet("color: #999999;")
        train_layout.addWidget(self.lbl_training_data)

        self.btn_train = QPushButton("Train Model")
        self.btn_train.setStyleSheet("background-color: #2d7d46;")
        self.btn_train.clicked.connect(self.train_model)
        self.btn_train.setEnabled(False)
        train_layout.addWidget(self.btn_train)

        train_group.setLayout(train_layout)
        layout.addWidget(train_group)

        # --- Predict section ---
        predict_group = QGroupBox("8. Predict with ML Model")
        predict_layout = QVBoxLayout()
        predict_layout.setSpacing(4)

        model_row = QHBoxLayout()
        model_row.setSpacing(2)
        self.btn_browse_model = QPushButton("Browse Model...")
        self.btn_browse_model.clicked.connect(self.browse_model)
        model_row.addWidget(self.btn_browse_model)
        predict_layout.addLayout(model_row)

        self.lbl_model_name = QLabel("No model selected")
        self.lbl_model_name.setStyleSheet("color: #999999; font-size: 9px;")
        self.lbl_model_name.setWordWrap(True)
        predict_layout.addWidget(self.lbl_model_name)

        apply_row = QHBoxLayout()
        apply_row.setSpacing(2)

        self.btn_apply_current = QPushButton("Apply Current")
        self.btn_apply_current.clicked.connect(self.apply_model_current)
        self.btn_apply_current.setEnabled(False)
        apply_row.addWidget(self.btn_apply_current)

        self.btn_apply_all = QPushButton("Apply All")
        self.btn_apply_all.clicked.connect(self.apply_model_all)
        self.btn_apply_all.setEnabled(False)
        apply_row.addWidget(self.btn_apply_all)

        predict_layout.addLayout(apply_row)

        self.rf_progress = QProgressBar()
        self.rf_progress.setValue(0)
        self.rf_progress.setVisible(False)
        predict_layout.addWidget(self.rf_progress)

        self.lbl_rf_status = QLabel("")
        self.lbl_rf_status.setStyleSheet("color: #999999; font-size: 9px;")
        predict_layout.addWidget(self.lbl_rf_status)

        predict_group.setLayout(predict_layout)
        layout.addWidget(predict_group)

    def _toggle_advanced_options(self, checked):
        """Toggle advanced options visibility."""
        self.advanced_widget.setVisible(checked)

    def _apply_styles(self):
        """Apply dark theme styles."""
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #cccccc;
                font-family: Arial;
            }
            QLabel {
                color: #cccccc;
                background-color: transparent;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 6px 12px;
                border-radius: 3px;
                font-weight: bold;
                min-height: 18px;
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
            QPushButton#small_btn {
                padding: 4px 8px;
                min-height: 16px;
            }
            QPushButton#accept_btn {
                background-color: #107c10;
            }
            QPushButton#accept_btn:hover {
                background-color: #0e6b0e;
            }
            QPushButton#reject_btn {
                background-color: #d13438;
            }
            QPushButton#reject_btn:hover {
                background-color: #b52e31;
            }
            QGroupBox {
                font-weight: bold;
                border: 1px solid #3f3f3f;
                border-radius: 4px;
                margin-top: 8px;
                padding-top: 6px;
                color: #cccccc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                left: 8px;
                padding: 0 4px 0 4px;
            }
            QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {
                padding: 4px;
                border: 1px solid #3f3f3f;
                border-radius: 3px;
                background-color: #1e1e1e;
                color: #cccccc;
            }
            QListWidget {
                background-color: #1e1e1e;
                border: 1px solid #3f3f3f;
                border-radius: 3px;
            }
            QListWidget::item {
                padding: 2px;
            }
            QListWidget::item:selected {
                background-color: #0078d4;
            }
            QProgressBar {
                border: 1px solid #3f3f3f;
                border-radius: 3px;
                text-align: center;
                background-color: #1e1e1e;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
                border-radius: 2px;
            }
            QScrollArea {
                border: none;
            }
            QScrollBar:vertical {
                background-color: #2b2b2b;
                width: 12px;
            }
            QScrollBar::handle:vertical {
                background-color: #5c5c5c;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar:horizontal {
                background-color: #1e1e1e;
                height: 14px;
                border: 1px solid #3f3f3f;
                border-radius: 3px;
            }
            QScrollBar::handle:horizontal {
                background-color: #0078d4;
                border-radius: 3px;
                min-width: 30px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #106ebe;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: none;
            }
            QCheckBox {
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
            }
        """)

    # =========================================================================
    # File Management
    # =========================================================================

    def add_folder(self):
        """Add all WAV files from a folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return

        wav_files = list(Path(folder).glob("*.wav")) + list(Path(folder).glob("*.WAV"))
        if not wav_files:
            QMessageBox.warning(self, "No Files", "No WAV files found in folder.")
            return

        for f in wav_files:
            if str(f) not in self.audio_files:
                self.audio_files.append(str(f))

        # Pre-load detection counts from existing CSVs
        self._scan_existing_csvs()
        self._update_file_list()

    def add_files(self):
        """Add individual WAV files."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio Files", "",
            "WAV Files (*.wav *.WAV);;All Files (*.*)"
        )
        if not files:
            return

        for f in files:
            if f not in self.audio_files:
                self.audio_files.append(f)

        self._update_file_list()

    def _scan_existing_csvs(self):
        """Pre-scan for existing CSV detection files to show counts in file list."""
        for filepath in self.audio_files:
            if filepath in self.all_detections:
                continue
            base = Path(filepath).stem
            parent = Path(filepath).parent
            for suffix in ['_usv_dsp', '_usv_rf', '_usv_detections']:
                csv_path = parent / f"{base}{suffix}.csv"
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        if 'status' not in df.columns:
                            df['status'] = 'pending'
                        self.all_detections[filepath] = df
                        break
                    except Exception:
                        pass

    def clear_files(self):
        """Clear all files."""
        self.audio_files = []
        self.current_file_idx = 0
        self.audio_data = None
        self.detections_df = None
        self.all_detections = {}
        self.dsp_queue = []
        self._update_file_list()
        self.spectrogram.set_audio_data(None, None)
        self.spectrogram.set_detections([], -1)
        self._update_ui_state()

    def _update_file_list(self):
        """Update file list display."""
        self.file_list.clear()

        for filepath in self.audio_files:
            filename = os.path.basename(filepath)
            # Build display text with queue indicator and detection count
            parts = []
            if filepath in self.dsp_queue:
                parts.append("\u2713")  # Checkmark for queued files
            parts.append(filename)
            if filepath in self.all_detections:
                n_det = len(self.all_detections[filepath])
                parts.append(f"({n_det})")
            display_text = " ".join(parts)
            item = QListWidgetItem(display_text)
            item.setData(Qt.UserRole, filepath)
            self.file_list.addItem(item)

        n = len(self.audio_files)
        self.lbl_file_num.setText(f"File {self.current_file_idx + 1 if n > 0 else 0}/{n}")
        self.btn_prev_file.setEnabled(self.current_file_idx > 0)
        self.btn_next_file.setEnabled(self.current_file_idx < n - 1)
        self.btn_add_to_queue.setEnabled(n > 0)

        if n > 0 and self.current_file_idx < n:
            self.file_list.setCurrentRow(self.current_file_idx)
            self._load_current_file()

        self._update_ui_state()

    def on_file_selected(self, row):
        """Handle file selection from list."""
        if row >= 0 and row < len(self.audio_files):
            # Save current file's detections before switching
            self._store_current_detections()
            self.current_file_idx = row
            self._load_current_file()
            self._update_file_navigation()

    def prev_file(self):
        """Go to previous file."""
        if self.current_file_idx > 0:
            self.current_file_idx -= 1
            self.file_list.setCurrentRow(self.current_file_idx)

    def next_file(self):
        """Go to next file."""
        if self.current_file_idx < len(self.audio_files) - 1:
            self.current_file_idx += 1
            self.file_list.setCurrentRow(self.current_file_idx)

    def _update_file_navigation(self):
        """Update file navigation buttons."""
        n = len(self.audio_files)
        self.lbl_file_num.setText(f"File {self.current_file_idx + 1}/{n}")
        self.btn_prev_file.setEnabled(self.current_file_idx > 0)
        self.btn_next_file.setEnabled(self.current_file_idx < n - 1)

    def _refresh_file_list_items(self):
        """Refresh file list display text without changing selection."""
        current_row = self.file_list.currentRow()
        self.file_list.blockSignals(True)

        for i, filepath in enumerate(self.audio_files):
            item = self.file_list.item(i)
            if item is None:
                continue
            filename = os.path.basename(filepath)
            # Build display text with queue indicator and detection count
            parts = []
            if filepath in self.dsp_queue:
                parts.append("\u2713")  # Checkmark for queued files
            parts.append(filename)
            if filepath in self.all_detections:
                n_det = len(self.all_detections[filepath])
                parts.append(f"({n_det})")
            item.setText(" ".join(parts))

        self.file_list.blockSignals(False)
        if current_row >= 0:
            self.file_list.setCurrentRow(current_row)

    def _load_current_file(self):
        """Load the current audio file."""
        if not self.audio_files or self.current_file_idx >= len(self.audio_files):
            return

        filepath = self.audio_files[self.current_file_idx]
        self.status_bar.showMessage(f"Loading {os.path.basename(filepath)}...")
        QApplication.processEvents()

        try:
            # Load audio
            if HAS_SOUNDFILE:
                try:
                    self.audio_data, self.sample_rate = sf.read(filepath, dtype='float32')
                except:
                    self.audio_data, self.sample_rate = self._load_with_ffmpeg(filepath)
            else:
                self.audio_data, self.sample_rate = self._load_with_ffmpeg(filepath)

            if self.audio_data.ndim > 1:
                self.audio_data = np.mean(self.audio_data, axis=1)

            self.spectrogram.set_audio_data(self.audio_data, self.sample_rate)

            # Load detections if available
            if filepath in self.all_detections:
                self.detections_df = self.all_detections[filepath].copy()
                self.current_detection_idx = 0
            else:
                # Try to load from CSV
                self._try_load_detections(filepath)

            self._update_display()
            self.status_bar.showMessage(f"Loaded: {os.path.basename(filepath)}")

        except Exception as e:
            self.status_bar.showMessage(f"Error loading file: {e}")
            self.audio_data = None

        self._update_ui_state()

    def _load_with_ffmpeg(self, filepath):
        """Load audio using ffmpeg."""
        import subprocess
        import tempfile

        probe_cmd = [
            'ffprobe', '-v', 'error', '-select_streams', 'a:0',
            '-show_entries', 'stream=sample_rate',
            '-of', 'default=noprint_wrappers=1:nokey=1', filepath
        ]
        result = subprocess.run(probe_cmd, capture_output=True, text=True, check=True)
        sr = int(result.stdout.strip())

        with tempfile.NamedTemporaryFile(suffix='.raw', delete=False) as tmp:
            temp_path = tmp.name

        try:
            convert_cmd = [
                'ffmpeg', '-i', filepath, '-f', 'f32le', '-acodec', 'pcm_f32le',
                '-ac', '1', '-y', temp_path
            ]
            subprocess.run(convert_cmd, capture_output=True, check=True)
            audio = np.fromfile(temp_path, dtype=np.float32)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        return audio, sr

    def _try_load_detections(self, filepath):
        """Try to load existing detections for a file."""
        base = Path(filepath).stem
        parent = Path(filepath).parent

        # Try different suffixes
        for suffix in ['_usv_dsp', '_usv_rf', '_usv_detections']:
            csv_path = parent / f"{base}{suffix}.csv"
            if csv_path.exists():
                try:
                    self.detections_df = pd.read_csv(csv_path)
                    if 'status' not in self.detections_df.columns:
                        self.detections_df['status'] = 'pending'
                    self.all_detections[filepath] = self.detections_df.copy()
                    self.current_detection_idx = 0
                    return
                except:
                    pass

        self.detections_df = None
        self.current_detection_idx = 0

    # =========================================================================
    # DSP Detection
    # =========================================================================

    def add_to_queue(self):
        """Add current file to DSP queue."""
        if not self.audio_files or self.current_file_idx >= len(self.audio_files):
            return

        filepath = self.audio_files[self.current_file_idx]
        if filepath not in self.dsp_queue:
            # Check if file already has detections
            if filepath in self.all_detections and len(self.all_detections[filepath]) > 0:
                reply = QMessageBox.question(
                    self, "Existing Detections",
                    f"This file already has {len(self.all_detections[filepath])} detections.\n"
                    "Running DSP detection will overwrite them.\n\nAdd to queue anyway?",
                    QMessageBox.Yes | QMessageBox.No,
                    QMessageBox.No
                )
                if reply != QMessageBox.Yes:
                    return

            self.dsp_queue.append(filepath)
            self._update_queue_display()

    def clear_queue(self):
        """Clear DSP queue."""
        self.dsp_queue = []
        self._update_queue_display()

    def _update_queue_display(self):
        """Update queue display and file list checkmarks."""
        n = len(self.dsp_queue)
        self.lbl_queue.setText(f"Queue: {n} file{'s' if n != 1 else ''}")
        self.btn_run_dsp.setEnabled(n > 0)
        # Update file list to show/hide checkmarks
        self._refresh_file_list_items()

    def run_dsp_detection(self):
        """Run DSP detection on queued files."""
        if not self.dsp_queue:
            return

        # Gather config
        config = {
            'min_freq_hz': self.spin_min_freq.value(),
            'max_freq_hz': self.spin_max_freq.value(),
            'energy_threshold_db': self.spin_threshold.value(),
            'min_duration_ms': self.spin_min_dur.value(),
            'max_duration_ms': self.spin_max_dur.value(),
            'min_gap_ms': self.spin_min_gap.value(),
            'noise_percentile': self.spin_noise_pct.value(),
            'nperseg': self.spin_nperseg.value(),
            'noverlap': self.spin_noverlap.value(),
            'freq_samples': self.spin_freq_samples.value(),
        }

        # Start worker
        self.dsp_worker = DSPDetectionWorker(self.dsp_queue.copy(), config)
        self.dsp_worker.progress.connect(self._on_dsp_progress)
        self.dsp_worker.file_complete.connect(self._on_dsp_file_complete)
        self.dsp_worker.all_complete.connect(self._on_dsp_complete)
        self.dsp_worker.error.connect(self._on_dsp_error)

        self.dsp_progress.setVisible(True)
        self.dsp_progress.setValue(0)
        self.btn_run_dsp.setEnabled(False)
        self.btn_run_dsp.setText("Running...")

        self.dsp_worker.start()

    def _on_dsp_progress(self, filename, current, total):
        """Handle DSP progress update."""
        self.dsp_progress.setValue(int(current / total * 100))
        self.lbl_dsp_status.setText(f"Processing: {filename}")

    def _on_dsp_file_complete(self, filename, n_detections):
        """Handle DSP file completion."""
        self.lbl_dsp_status.setText(f"{filename}: {n_detections} detections")

    def _on_dsp_complete(self, results):
        """Handle DSP detection completion."""
        # Store results
        for filepath, detections in results.items():
            if isinstance(detections, pd.DataFrame):
                df = detections.copy()
            else:
                df = pd.DataFrame(detections)

            if len(df) > 0 and 'status' not in df.columns:
                df['status'] = 'pending'
                df['source'] = 'dsp'

            self.all_detections[filepath] = df

            # Save CSV
            base = Path(filepath).stem
            parent = Path(filepath).parent
            csv_path = parent / f"{base}_usv_dsp.csv"
            df.to_csv(csv_path, index=False)

        # Update UI
        self.dsp_progress.setValue(100)
        self.dsp_progress.setVisible(False)
        self.btn_run_dsp.setEnabled(True)
        self.btn_run_dsp.setText("Run DSP Detection")

        total_det = sum(len(d) for d in results.values())
        self.lbl_dsp_status.setText(f"Complete: {total_det} total detections")
        self.status_bar.showMessage(f"DSP detection complete: {total_det} detections in {len(results)} files")

        # Clear queue and reload current file
        self.dsp_queue = []
        self._update_queue_display()
        self._update_file_list()
        self._load_current_file()

    def _on_dsp_error(self, filename, error):
        """Handle DSP error."""
        self.lbl_dsp_status.setText(f"Error: {filename} - {error}")

    # =========================================================================
    # Detection Navigation and Display
    # =========================================================================

    def _update_display(self):
        """Update all display elements."""
        if self.detections_df is None or len(self.detections_df) == 0:
            self.lbl_det_num.setText("Det 0/0")
            self.lbl_det_info.setText("Peak: -- Hz | Dur: -- ms")
            self.lbl_det_status.setText("Status: --")
            self.spectrogram.set_detections([], -1)
            self._update_progress()
            return

        n_det = len(self.detections_df)
        self.current_detection_idx = min(self.current_detection_idx, n_det - 1)
        self.current_detection_idx = max(0, self.current_detection_idx)

        det = self.detections_df.iloc[self.current_detection_idx]

        # Navigation
        self.lbl_det_num.setText(f"Det {self.current_detection_idx + 1}/{n_det}")
        self.btn_prev_det.setEnabled(self.current_detection_idx > 0)
        self.btn_next_det.setEnabled(self.current_detection_idx < n_det - 1)

        # Time spinboxes
        self.spin_start.blockSignals(True)
        self.spin_stop.blockSignals(True)
        self.spin_start.setValue(det['start_seconds'])
        self.spin_stop.setValue(det['stop_seconds'])
        self.spin_start.blockSignals(False)
        self.spin_stop.blockSignals(False)

        # Frequency spinboxes (handle NaN values safely)
        self.spin_det_min_freq.blockSignals(True)
        self.spin_det_max_freq.blockSignals(True)
        min_f = det.get('min_freq_hz', 20000)
        max_f = det.get('max_freq_hz', 80000)
        # Handle NaN values
        min_f = 20000 if pd.isna(min_f) else int(min_f)
        max_f = 80000 if pd.isna(max_f) else int(max_f)
        self.spin_det_min_freq.setValue(min_f)
        self.spin_det_max_freq.setValue(max_f)
        self.spin_det_min_freq.blockSignals(False)
        self.spin_det_max_freq.blockSignals(False)

        # Info (handle NaN values)
        peak = det.get('peak_freq_hz', 0)
        dur = det.get('duration_ms', 0)
        peak = 0 if pd.isna(peak) else peak
        dur = 0 if pd.isna(dur) else dur
        self.lbl_det_info.setText(f"Peak: {peak:.0f} Hz | Dur: {dur:.1f} ms")

        # Status
        status = det.get('status', 'pending')
        self.lbl_det_status.setText(f"Status: {status.capitalize()}")
        if status == 'accepted':
            self.lbl_det_status.setStyleSheet("font-weight: bold; color: #107c10;")
        elif status == 'rejected':
            self.lbl_det_status.setStyleSheet("font-weight: bold; color: #d13438;")
        elif status == 'noise':
            self.lbl_det_status.setStyleSheet("font-weight: bold; color: #8b4513;")
        else:
            self.lbl_det_status.setStyleSheet("font-weight: bold; color: #999999;")

        # Update spectrogram view
        self._update_spectrogram_view()
        self._update_progress()
        self._update_training_data_label()
        self._update_button_counts()

    def _update_spectrogram_view(self):
        """Update spectrogram to show current detection."""
        if self.detections_df is None or len(self.detections_df) == 0:
            return

        det = self.detections_df.iloc[self.current_detection_idx]
        det_center = (det['start_seconds'] + det['stop_seconds']) / 2

        window = self.spin_view_window.value()
        view_start = det_center - window / 2
        view_end = det_center + window / 2

        total_dur = self.spectrogram.get_total_duration()
        if total_dur > 0:
            view_start = max(0, view_start)
            view_end = min(total_dur, view_end)

        self.spectrogram.set_view_range(view_start, view_end)
        self._update_scrollbar()

        # Update detection boxes
        detections = []
        for _, row in self.detections_df.iterrows():
            detections.append({
                'start_seconds': row['start_seconds'],
                'stop_seconds': row['stop_seconds'],
                'min_freq_hz': row.get('min_freq_hz', 20000),
                'max_freq_hz': row.get('max_freq_hz', 80000),
                'status': row.get('status', 'pending')
            })

        self.spectrogram.set_detections(detections, self.current_detection_idx)

    def _update_scrollbar(self):
        """Update scrollbar position."""
        total_dur = self.spectrogram.get_total_duration()
        if total_dur <= 0:
            return

        view_start, view_end = self.spectrogram.get_view_range()
        view_center = (view_start + view_end) / 2

        pos = int(view_center / total_dur * 1000)
        self.time_scrollbar.blockSignals(True)
        self.time_scrollbar.setValue(pos)
        self.time_scrollbar.blockSignals(False)

    def _update_progress(self):
        """Update progress display."""
        if self.detections_df is None or len(self.detections_df) == 0:
            self.lbl_progress.setText("0/0 reviewed")
            self.progress_bar.setValue(0)
            return

        total = len(self.detections_df)
        reviewed = len(self.detections_df[self.detections_df['status'] != 'pending'])
        self.lbl_progress.setText(f"{reviewed}/{total} reviewed")
        self.progress_bar.setValue(int(reviewed / total * 100) if total > 0 else 0)

    def _update_training_data_label(self):
        """Update training data summary."""
        n_usv = 0
        n_noise = 0

        for filepath, df in self.all_detections.items():
            if len(df) > 0 and 'status' in df.columns:
                n_usv += (df['status'] == 'accepted').sum()
                n_noise += (df['status'].isin(['rejected', 'noise'])).sum()

        self.lbl_training_data.setText(f"Training: {n_usv} USV, {n_noise} rejected")
        self.btn_train.setEnabled(n_usv >= 3 and n_noise >= 3)

    def prev_detection(self):
        """Go to previous detection."""
        if self.current_detection_idx > 0:
            self.current_detection_idx -= 1
            self._update_display()

    def next_detection(self):
        """Go to next detection."""
        if self.detections_df is not None and self.current_detection_idx < len(self.detections_df) - 1:
            self.current_detection_idx += 1
            self._update_display()

    def on_detection_selected(self, idx):
        """Handle detection selection from spectrogram."""
        if self.detections_df is not None and 0 <= idx < len(self.detections_df):
            self.current_detection_idx = idx
            self._update_display()

    def on_box_adjusted(self, idx, start_s, stop_s, min_freq, max_freq):
        """Handle box adjustment from spectrogram."""
        if self.detections_df is None or idx < 0 or idx >= len(self.detections_df):
            return

        self.detections_df.at[idx, 'start_seconds'] = start_s
        self.detections_df.at[idx, 'stop_seconds'] = stop_s
        self.detections_df.at[idx, 'min_freq_hz'] = min_freq
        self.detections_df.at[idx, 'max_freq_hz'] = max_freq
        self.detections_df.at[idx, 'duration_ms'] = (stop_s - start_s) * 1000

        # Update current file's stored detections (don't save yet - wait for drag complete)
        if self.audio_files and self.current_file_idx < len(self.audio_files):
            filepath = self.audio_files[self.current_file_idx]
            self.all_detections[filepath] = self.detections_df.copy()

        # Update the spectrogram widget's detection in place (don't replace entire list during drag)
        self.spectrogram.update_detection(idx, start_s, stop_s, min_freq, max_freq)

        if idx == self.current_detection_idx:
            self._update_detection_info_only()

    def on_drag_complete(self):
        """Handle completion of box drag - save to CSV."""
        self._store_current_detections()
        self._update_button_counts()
        self._refresh_file_list_items()
        self._update_detection_boxes()

    def _update_detection_info_only(self):
        """Update detection info without changing view."""
        if self.detections_df is None or len(self.detections_df) == 0:
            return

        det = self.detections_df.iloc[self.current_detection_idx]

        self.spin_start.blockSignals(True)
        self.spin_stop.blockSignals(True)
        self.spin_det_min_freq.blockSignals(True)
        self.spin_det_max_freq.blockSignals(True)

        self.spin_start.setValue(det['start_seconds'])
        self.spin_stop.setValue(det['stop_seconds'])

        # Handle NaN values safely
        min_f = det.get('min_freq_hz', 20000)
        max_f = det.get('max_freq_hz', 80000)
        min_f = 20000 if pd.isna(min_f) else int(min_f)
        max_f = 80000 if pd.isna(max_f) else int(max_f)
        self.spin_det_min_freq.setValue(min_f)
        self.spin_det_max_freq.setValue(max_f)

        self.spin_start.blockSignals(False)
        self.spin_stop.blockSignals(False)
        self.spin_det_min_freq.blockSignals(False)
        self.spin_det_max_freq.blockSignals(False)

        peak = det.get('peak_freq_hz', 0)
        dur = det.get('duration_ms', 0)
        self.lbl_det_info.setText(f"Peak: {peak:.0f} Hz | Dur: {dur:.1f} ms")

    def _update_detection_boxes(self):
        """Update detection boxes on spectrogram."""
        if self.detections_df is None or len(self.detections_df) == 0:
            self.spectrogram.set_detections([], -1)
            return

        detections = []
        for _, row in self.detections_df.iterrows():
            # Use direct column access with safe defaults
            min_freq = row['min_freq_hz'] if 'min_freq_hz' in row and not pd.isna(row['min_freq_hz']) else 20000
            max_freq = row['max_freq_hz'] if 'max_freq_hz' in row and not pd.isna(row['max_freq_hz']) else 80000
            status = row['status'] if 'status' in row and not pd.isna(row['status']) else 'pending'

            detections.append({
                'start_seconds': row['start_seconds'],
                'stop_seconds': row['stop_seconds'],
                'min_freq_hz': min_freq,
                'max_freq_hz': max_freq,
                'status': status
            })

        self.spectrogram.set_detections(detections, self.current_detection_idx)

    # =========================================================================
    # View Controls
    # =========================================================================

    def on_scrollbar_changed(self, value):
        """Handle scrollbar change."""
        total_dur = self.spectrogram.get_total_duration()
        if total_dur <= 0:
            return

        center = value / 1000.0 * total_dur
        window = self.spin_view_window.value()

        view_start = max(0, center - window / 2)
        view_end = min(total_dur, center + window / 2)

        self.spectrogram.set_view_range(view_start, view_end)
        self._update_detection_boxes()

    def pan_left(self):
        """Pan view left."""
        view_start, view_end = self.spectrogram.get_view_range()
        window = view_end - view_start
        shift = window * 0.25

        new_start = max(0, view_start - shift)
        new_end = new_start + window

        self.spectrogram.set_view_range(new_start, new_end)
        self._update_scrollbar()
        self._update_detection_boxes()

    def pan_right(self):
        """Pan view right."""
        view_start, view_end = self.spectrogram.get_view_range()
        window = view_end - view_start
        shift = window * 0.25
        total = self.spectrogram.get_total_duration()

        new_end = min(total, view_end + shift)
        new_start = new_end - window

        self.spectrogram.set_view_range(new_start, new_end)
        self._update_scrollbar()
        self._update_detection_boxes()

    def zoom_in(self):
        """Zoom in maintaining center."""
        current_start, current_end = self.spectrogram.get_view_range()
        center = (current_start + current_end) / 2
        new_window = max(0.1, (current_end - current_start) / 1.5)

        self.spin_view_window.blockSignals(True)
        self.spin_view_window.setValue(new_window)
        self.spin_view_window.blockSignals(False)

        total_dur = self.spectrogram.get_total_duration()
        new_start = max(0, center - new_window / 2)
        new_end = min(total_dur, center + new_window / 2)
        self.spectrogram.set_view_range(new_start, new_end)
        self._update_scrollbar()
        self._update_detection_boxes()

    def zoom_out(self):
        """Zoom out maintaining center."""
        current_start, current_end = self.spectrogram.get_view_range()
        center = (current_start + current_end) / 2
        total_dur = self.spectrogram.get_total_duration()
        new_window = min(total_dur if total_dur > 0 else 600, (current_end - current_start) * 1.5)

        self.spin_view_window.blockSignals(True)
        self.spin_view_window.setValue(new_window)
        self.spin_view_window.blockSignals(False)

        new_start = max(0, center - new_window / 2)
        new_end = min(total_dur, center + new_window / 2)
        self.spectrogram.set_view_range(new_start, new_end)
        self._update_scrollbar()
        self._update_detection_boxes()

    def on_view_window_changed(self):
        """Handle view window spinbox change."""
        self._update_spectrogram_view()

    def on_time_changed(self):
        """Handle time spinbox change."""
        if self.detections_df is None:
            return

        start = self.spin_start.value()
        stop = self.spin_stop.value()

        self.detections_df.at[self.current_detection_idx, 'start_seconds'] = start
        self.detections_df.at[self.current_detection_idx, 'stop_seconds'] = stop
        self.detections_df.at[self.current_detection_idx, 'duration_ms'] = (stop - start) * 1000

        self._update_detection_boxes()

    def on_freq_changed(self):
        """Handle frequency spinbox change."""
        if self.detections_df is None:
            return

        min_f = self.spin_det_min_freq.value()
        max_f = self.spin_det_max_freq.value()

        self.detections_df.at[self.current_detection_idx, 'min_freq_hz'] = min_f
        self.detections_df.at[self.current_detection_idx, 'max_freq_hz'] = max_f

        self._update_detection_boxes()

    # =========================================================================
    # Labeling Actions
    # =========================================================================

    def accept_detection(self):
        """Mark current detection as accepted (USV)."""
        if self.detections_df is None:
            return
        self.detections_df.at[self.current_detection_idx, 'status'] = 'accepted'
        self._store_current_detections()
        self._update_display()
        self._update_button_counts()
        self._auto_advance()

    def reject_detection(self):
        """Mark current detection as rejected."""
        if self.detections_df is None:
            return
        self.detections_df.at[self.current_detection_idx, 'status'] = 'rejected'
        self._store_current_detections()
        self._update_display()
        self._update_button_counts()
        self._auto_advance()

    def _update_button_counts(self):
        """Update Accept/Reject/Noise button labels with counts."""
        if self.detections_df is None or len(self.detections_df) == 0:
            self.btn_accept.setText("Accept")
            self.btn_reject.setText("Reject")
            return

        counts = self.detections_df['status'].value_counts()
        n_accepted = counts.get('accepted', 0)
        n_rejected = counts.get('rejected', 0)

        self.btn_accept.setText(f"Accept ({n_accepted})")
        self.btn_reject.setText(f"Reject ({n_rejected})")

    def add_new_usv(self):
        """Add a new USV detection at view center."""
        if self.audio_data is None:
            return

        view_start, view_end = self.spectrogram.get_view_range()
        view_center = (view_start + view_end) / 2
        view_duration = view_end - view_start

        new_duration = min(0.05, view_duration * 0.1)
        new_start = view_center - new_duration / 2
        new_stop = view_center + new_duration / 2

        new_row = {
            'start_seconds': new_start,
            'stop_seconds': new_stop,
            'duration_ms': new_duration * 1000,
            'min_freq_hz': 30000,
            'max_freq_hz': 60000,
            'peak_freq_hz': 45000,
            'status': 'pending',
            'source': 'manual'
        }

        if self.detections_df is None:
            self.detections_df = pd.DataFrame([new_row])
        else:
            self.detections_df = pd.concat([self.detections_df, pd.DataFrame([new_row])],
                                           ignore_index=True)

        self.current_detection_idx = len(self.detections_df) - 1
        self._store_current_detections()
        self._update_display()
        self._update_ui_state()
        self.status_bar.showMessage("Added new USV detection")

    def delete_current(self):
        """Delete current detection."""
        if self.detections_df is None or len(self.detections_df) == 0:
            return

        self.detections_df = self.detections_df.drop(self.current_detection_idx).reset_index(drop=True)

        if len(self.detections_df) == 0:
            self.detections_df = None
            self.current_detection_idx = 0
        elif self.current_detection_idx >= len(self.detections_df):
            self.current_detection_idx = len(self.detections_df) - 1

        self._store_current_detections()
        self._update_display()
        self._update_ui_state()
        self.status_bar.showMessage("Deleted detection")

    def delete_all_pending(self):
        """Delete all pending detections."""
        if self.detections_df is None or len(self.detections_df) == 0:
            return

        pending_mask = self.detections_df['status'] == 'pending'
        n_pending = pending_mask.sum()

        if n_pending == 0:
            self.status_bar.showMessage("No pending detections to delete")
            return

        reply = QMessageBox.question(
            self, "Delete Pending",
            f"Delete {n_pending} pending detections?\n\n"
            "This keeps only labeled detections.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        self.detections_df = self.detections_df[~pending_mask].reset_index(drop=True)

        if len(self.detections_df) == 0:
            self.detections_df = None
            self.current_detection_idx = 0
        else:
            self.current_detection_idx = min(self.current_detection_idx, len(self.detections_df) - 1)

        self._store_current_detections()
        self._update_display()
        self._update_ui_state()

        n_remaining = len(self.detections_df) if self.detections_df is not None else 0
        self.status_bar.showMessage(f"Deleted {n_pending} pending | {n_remaining} labeled remain")

    def _auto_advance(self):
        """Auto-advance to next detection."""
        if self.detections_df is not None and self.current_detection_idx < len(self.detections_df) - 1:
            self.current_detection_idx += 1
            self._update_display()

    def _store_current_detections(self):
        """Store current detections in all_detections dict and save to CSV."""
        if self.audio_files and self.current_file_idx < len(self.audio_files):
            filepath = self.audio_files[self.current_file_idx]
            if self.detections_df is not None:
                self.all_detections[filepath] = self.detections_df.copy()
                # Auto-save to CSV
                self._save_detections_csv(filepath)
            elif filepath in self.all_detections:
                del self.all_detections[filepath]

    def _save_detections_csv(self, filepath):
        """Save current detections to CSV file (updates _usv_dsp.csv in place)."""
        if self.detections_df is None or len(self.detections_df) == 0:
            return

        base = Path(filepath).stem
        parent = Path(filepath).parent
        csv_path = parent / f"{base}_usv_dsp.csv"

        try:
            self.detections_df.to_csv(csv_path, index=False)
        except Exception as e:
            self.status_bar.showMessage(f"Error saving CSV: {e}")

    # =========================================================================
    # Playback
    # =========================================================================

    def on_speed_changed(self):
        """Handle playback speed change."""
        mode = self.combo_speed.currentData()
        if mode == "heterodyne":
            self.use_heterodyne = True
            self.playback_speed = 1.0
        else:
            self.use_heterodyne = False
            self.playback_speed = mode

    def toggle_playback(self):
        """Toggle playback."""
        if self.is_playing:
            self.stop_playback()
        else:
            self.play_visible()

    def play_visible(self):
        """Play visible spectrogram region."""
        if not HAS_SOUNDDEVICE or self.audio_data is None:
            return

        start_s, stop_s = self.spectrogram.get_view_range()
        total_duration = len(self.audio_data) / self.sample_rate
        start_s = max(0, start_s)
        stop_s = min(total_duration, stop_s)

        start_sample = int(start_s * self.sample_rate)
        stop_sample = int(stop_s * self.sample_rate)
        segment = self.audio_data[start_sample:stop_sample].copy()

        try:
            if self.use_heterodyne:
                segment = self._heterodyne(segment)
                play_sr = 44100
            else:
                # Resample for slow playback
                target_sr = int(self.sample_rate * self.playback_speed)
                target_sr = max(8000, min(target_sr, 192000))
                play_sr = target_sr

            sd.play(segment, play_sr)
            self.is_playing = True
            self.btn_play.setText("Playing...")

        except Exception as e:
            self.status_bar.showMessage(f"Playback error: {e}")

    def stop_playback(self):
        """Stop playback."""
        if HAS_SOUNDDEVICE:
            sd.stop()
        self.is_playing = False
        self.btn_play.setText("Play")

    def _heterodyne(self, segment, carrier_freq=40000):
        """Apply heterodyne transformation."""
        t = np.arange(len(segment)) / self.sample_rate
        mixed = segment * np.cos(2 * np.pi * carrier_freq * t)

        # Low-pass filter
        from scipy.signal import butter, filtfilt
        nyq = self.sample_rate / 2
        cutoff = min(10000, nyq * 0.9)
        b, a = butter(4, cutoff / nyq, btype='low')
        filtered = filtfilt(b, a, mixed)

        # Resample to 44.1kHz
        target_len = int(len(filtered) * 44100 / self.sample_rate)
        resampled = signal.resample(filtered, target_len)

        return resampled.astype(np.float32)

    # =========================================================================
    # Save/Load
    # =========================================================================

    def open_output_folder(self):
        """Open folder containing current file."""
        if self.audio_files and self.current_file_idx < len(self.audio_files):
            folder = str(Path(self.audio_files[self.current_file_idx]).parent)
            import subprocess
            subprocess.run(['open', folder])

    # =========================================================================
    # Machine Learning
    # =========================================================================

    def train_model(self):
        """Train Random Forest classifier using labeled data from loaded files."""
        try:
            from fnt.usv.usv_classifier import USVClassifier
        except ImportError:
            QMessageBox.warning(
                self, "Missing Dependencies",
                "scikit-learn and joblib are required.\n"
                "Install with: pip install scikit-learn joblib"
            )
            return

        # Gather all labeled data from loaded files
        all_features = []
        files_processed = 0
        files_with_labels = 0

        for filepath, df in self.all_detections.items():
            files_processed += 1
            if df is None or len(df) == 0:
                continue
            if 'status' not in df.columns:
                continue

            labeled_mask = df['status'].isin(['accepted', 'rejected', 'noise'])
            labeled_df = df[labeled_mask]

            if len(labeled_df) == 0:
                continue

            files_with_labels += 1

            # Load audio for feature extraction (same fallback logic as main loader)
            try:
                if HAS_SOUNDFILE:
                    try:
                        audio, sr = sf.read(filepath, dtype='float32')
                    except Exception:
                        audio, sr = self._load_with_ffmpeg(filepath)
                else:
                    audio, sr = self._load_with_ffmpeg(filepath)

                if audio.ndim > 1:
                    audio = np.mean(audio, axis=1)
            except Exception as e:
                self.status_bar.showMessage(f"Error loading {filepath}: {e}")
                continue

            # Extract features
            for _, row in labeled_df.iterrows():
                try:
                    features = self._extract_features(audio, sr, row)
                    features['label'] = 'usv' if row['status'] == 'accepted' else 'noise'
                    all_features.append(features)
                except Exception as e:
                    self.status_bar.showMessage(f"Feature extraction error: {e}")
                    continue

        if len(all_features) < 10:
            QMessageBox.warning(self, "Insufficient Data",
                               f"Need at least 10 labeled samples. Have {len(all_features)}.")
            return

        features_df = pd.DataFrame(all_features)
        n_usv = (features_df['label'] == 'usv').sum()
        n_noise = (features_df['label'] == 'noise').sum()

        if n_usv < 3 or n_noise < 3:
            QMessageBox.warning(self, "Imbalanced Data",
                               f"Need at least 3 of each class.\n"
                               f"USV: {n_usv}, Noise: {n_noise}")
            return

        # Ask for save location
        folder = QFileDialog.getExistingDirectory(self, "Select Model Save Location")
        if not folder:
            return

        # Create model folder with timestamp
        timestamp = datetime.now().strftime("%Y.%m.%d_%H%M%S")
        model_name = f"fntAudioModel_{timestamp}"
        model_dir = Path(folder) / model_name

        self.status_bar.showMessage("Training classifier...")
        QApplication.processEvents()

        try:
            classifier = USVClassifier()
            metrics = classifier.train(features_df, features_df['label'])
            classifier.save(str(model_dir))

            report = (
                f"Training Complete!\n\n"
                f"Samples: {len(features_df)} ({n_usv} USV, {n_noise} noise)\n\n"
                f"Performance:\n"
                f"  Accuracy:  {metrics['accuracy']:.1%}\n"
                f"  Precision: {metrics['precision']:.1%}\n"
                f"  Recall:    {metrics['recall']:.1%}\n"
                f"  F1 Score:  {metrics['f1']:.1%}\n\n"
                f"Model saved to:\n{model_dir}"
            )

            self.status_bar.showMessage(f"Model saved: {model_dir}")
            QMessageBox.information(self, "Training Complete", report)

            # Auto-select the new model
            self.current_model_dir = str(model_dir)
            self.lbl_model_name.setText(model_name)
            self._update_ui_state()

        except Exception as e:
            self.status_bar.showMessage(f"Training failed: {e}")
            QMessageBox.critical(self, "Training Error", f"Training failed:\n{e}")

    def _extract_features(self, audio, sr, row):
        """Extract features from a detection."""
        from fnt.usv.usv_classifier import extract_features_from_detection
        return extract_features_from_detection(
            audio, sr,
            row['start_seconds'], row['stop_seconds'],
            row.get('min_freq_hz', 25000), row.get('max_freq_hz', 65000)
        )

    def browse_model(self):
        """Browse for model folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Model Folder")
        if not folder:
            return

        # Check for valid model
        model_path = Path(folder)
        if not (model_path / "model.joblib").exists():
            QMessageBox.warning(self, "Invalid Model",
                               "Selected folder does not contain a valid model.")
            return

        self.current_model_dir = folder
        self.lbl_model_name.setText(model_path.name)
        self._update_ui_state()

    def apply_model_current(self):
        """Apply model to current file."""
        if not self.current_model_dir or not self.audio_files:
            return

        filepath = self.audio_files[self.current_file_idx]
        self._apply_model_to_files([filepath])

    def apply_model_all(self):
        """Apply model to all files."""
        if not self.current_model_dir or not self.audio_files:
            return

        self._apply_model_to_files(self.audio_files)

    def _apply_model_to_files(self, files):
        """Apply RF model to files."""
        config = {
            'min_freq_hz': self.spin_min_freq.value(),
            'max_freq_hz': self.spin_max_freq.value(),
            'energy_threshold_db': max(5.0, self.spin_threshold.value() - 3),  # Lower for candidates
            'min_duration_ms': max(3.0, self.spin_min_dur.value() - 2),
            'max_duration_ms': self.spin_max_dur.value() + 100,
        }

        self.rf_worker = RFDetectionWorker(files, self.current_model_dir, config)
        self.rf_worker.progress.connect(self._on_rf_progress)
        self.rf_worker.file_complete.connect(self._on_rf_file_complete)
        self.rf_worker.all_complete.connect(self._on_rf_complete)
        self.rf_worker.error.connect(self._on_rf_error)

        self.rf_progress.setVisible(True)
        self.rf_progress.setValue(0)
        self.btn_apply_current.setEnabled(False)
        self.btn_apply_all.setEnabled(False)

        self.rf_worker.start()

    def _on_rf_progress(self, filename, current, total):
        """Handle RF progress."""
        self.rf_progress.setValue(int(current / total * 100))
        self.lbl_rf_status.setText(f"Processing: {filename}")

    def _on_rf_file_complete(self, filename, n_det):
        """Handle RF file completion."""
        self.lbl_rf_status.setText(f"{filename}: {n_det} USVs detected")

    def _on_rf_complete(self, results):
        """Handle RF completion."""
        for filepath, detections in results.items():
            if isinstance(detections, pd.DataFrame) and len(detections) > 0:
                self.all_detections[filepath] = detections

                # Save CSV
                base = Path(filepath).stem
                parent = Path(filepath).parent
                csv_path = parent / f"{base}_usv_rf.csv"
                detections.to_csv(csv_path, index=False)

        self.rf_progress.setValue(100)
        self.rf_progress.setVisible(False)
        self.btn_apply_current.setEnabled(True)
        self.btn_apply_all.setEnabled(True)

        total_det = sum(len(d) for d in results.values() if isinstance(d, pd.DataFrame))
        self.lbl_rf_status.setText(f"Complete: {total_det} USVs detected")
        self.status_bar.showMessage(f"RF detection complete: {total_det} USVs")

        self._update_file_list()
        self._load_current_file()

    def _on_rf_error(self, filename, error):
        """Handle RF error."""
        self.lbl_rf_status.setText(f"Error: {filename}")

    # =========================================================================
    # UI State Management
    # =========================================================================

    def _update_ui_state(self):
        """Update UI enabled states based on current state."""
        has_files = len(self.audio_files) > 0
        has_audio = self.audio_data is not None
        has_det = self.detections_df is not None and len(self.detections_df) > 0
        has_model = self.current_model_dir is not None

        # Input
        self.btn_add_to_queue.setEnabled(has_files)

        # Detection navigation
        self.btn_prev_det.setEnabled(has_det and self.current_detection_idx > 0)
        self.btn_next_det.setEnabled(has_det and self.current_detection_idx < len(self.detections_df) - 1 if has_det else False)

        # Labeling
        self.btn_accept.setEnabled(has_det)
        self.btn_reject.setEnabled(has_det)
        self.btn_add_usv.setEnabled(has_audio)
        self.btn_delete.setEnabled(has_det)

        has_pending = has_det and (self.detections_df['status'] == 'pending').any()
        self.btn_delete_pending.setEnabled(has_pending)

        # Playback
        self.btn_play.setEnabled(has_audio and HAS_SOUNDDEVICE)
        self.btn_stop.setEnabled(has_audio and HAS_SOUNDDEVICE)

        # Open folder
        self.btn_open_folder.setEnabled(has_files)

        # ML
        self.btn_apply_current.setEnabled(has_model and has_audio)
        self.btn_apply_all.setEnabled(has_model and has_files)

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts for accept/reject."""
        key = event.key()
        if key == Qt.Key_A and self.btn_accept.isEnabled():
            self._flash_button(self.btn_accept)
            self.accept_detection()
        elif key == Qt.Key_R and self.btn_reject.isEnabled():
            self._flash_button(self.btn_reject)
            self.reject_detection()
        elif key == Qt.Key_Space and self.btn_play.isEnabled():
            self._flash_button(self.btn_play)
            self.toggle_playback()
        else:
            super().keyPressEvent(event)

    def _flash_button(self, button):
        """Briefly highlight a button to give visual feedback."""
        original_style = button.styleSheet()
        button.setStyleSheet("background-color: #ffffff; color: #000000;")
        QTimer.singleShot(120, lambda: button.setStyleSheet(original_style))


# =============================================================================
# Main Entry Point
# =============================================================================

def main():
    app = QApplication(sys.argv)
    window = USVStudioWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
