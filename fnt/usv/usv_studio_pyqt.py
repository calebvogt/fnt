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

import math
import os
import sys
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QRectF, QPointF, QTimer, QSettings, QUrl
from PyQt5.QtGui import QFont, QColor, QPainter, QImage, QPen, QBrush, QPolygonF, QKeySequence, QDesktopServices
from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QLabel, QPushButton, QFileDialog, QProgressBar, QGroupBox,
    QSpinBox, QDoubleSpinBox, QComboBox, QListWidget, QListWidgetItem,
    QScrollArea, QSplitter, QStatusBar, QMessageBox, QScrollBar,
    QSizePolicy, QFrame, QCheckBox, QShortcut, QSlider,
    QDialog, QDialogButtonBox
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
    zoom_requested = pyqtSignal(float, float)  # factor, center_time

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
        self.cached_min_freq = None
        self.cached_max_freq = None

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

        # Playback position indicator (None = not playing)
        self.playback_position = None

        # Colormap
        self.colormap_name = 'viridis'
        self.colormap_lut = self._create_colormap_lut(self.colormap_name)

    def set_colormap(self, name):
        """Set colormap by name and recompute spectrogram."""
        self.colormap_name = name
        self.colormap_lut = self._create_colormap_lut(name)
        # Force recompute
        self.cached_view_start = None
        self.cached_view_end = None
        self.spec_image = None
        if self.total_duration > 0:
            self._compute_view_spectrogram()
        self.update()

    @staticmethod
    def _create_colormap_lut(name='viridis'):
        """Create colormap lookup table by name."""
        colormaps = {
            'viridis': [
                [68, 1, 84], [72, 40, 120], [62, 74, 137], [49, 104, 142],
                [38, 130, 142], [31, 158, 137], [53, 183, 121], [109, 205, 89],
                [180, 222, 44], [253, 231, 37],
            ],
            'magma': [
                [0, 0, 4], [28, 16, 68], [79, 18, 123], [129, 37, 129],
                [181, 54, 122], [229, 89, 100], [251, 135, 97], [254, 194, 140],
                [254, 237, 176], [252, 253, 191],
            ],
            'inferno': [
                [0, 0, 4], [22, 11, 57], [66, 10, 104], [106, 23, 110],
                [147, 38, 103], [188, 55, 84], [221, 81, 58], [243, 120, 25],
                [249, 173, 10], [252, 255, 164],
            ],
            'grayscale': [
                [0, 0, 0], [28, 28, 28], [57, 57, 57], [85, 85, 85],
                [113, 113, 113], [142, 142, 142], [170, 170, 170], [198, 198, 198],
                [227, 227, 227], [255, 255, 255],
            ],
        }
        colors = np.array(colormaps.get(name, colormaps['viridis']), dtype=np.float32)

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

    def set_audio_data(self, audio_data, sample_rate, preserve_view=False):
        """Set audio data for spectrogram computation.

        Args:
            preserve_view: If True, keep current view_start/view_end/freq settings.
        """
        self.audio_data = audio_data
        self.sample_rate = sample_rate

        if audio_data is not None and sample_rate is not None and len(audio_data) > 0:
            self.total_duration = len(audio_data) / sample_rate
            nyquist = min(125000, sample_rate / 2)
            if not preserve_view:
                self.view_start = 0
                self.view_end = min(2.0, self.total_duration)
                self.max_freq = nyquist
            else:
                # Clamp view to new file duration
                if self.view_start >= self.total_duration:
                    self.view_start = 0
                self.view_end = min(self.view_end, self.total_duration)
                if self.view_end <= self.view_start:
                    self.view_end = min(self.view_start + 2.0, self.total_duration)
                # Clamp max_freq to Nyquist
                self.max_freq = min(self.max_freq, nyquist)
        else:
            self.total_duration = 0
            self.view_start = 0
            self.view_end = 10.0

        self.cached_view_start = None
        self.cached_view_end = None
        self.cached_min_freq = None
        self.cached_max_freq = None
        self.spec_image = None

        if self.total_duration > 0:
            self._compute_view_spectrogram()
        self.update()

    def set_detections(self, detections, current_idx=-1):
        """Set detection boxes. Clears any active drag to prevent stale references."""
        # Cancel any active drag — the detection list is being replaced
        if self.drag_mode is not None:
            self.drag_mode = None
            self.drag_detection_idx = None
            self.drag_start = None
            self.drag_start_box = None
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

        # Check cache - only use if position and freq range are nearly identical
        current_window = self.view_end - self.view_start
        if (self.cached_view_start is not None and
            self.cached_view_end is not None and
            self.spec_image is not None):
            cached_window = self.cached_view_end - self.cached_view_start
            zoom_similar = abs(current_window - cached_window) / (cached_window + 1e-10) < 0.05
            position_tolerance = current_window * 0.01
            same_position = (abs(self.view_start - self.cached_view_start) < position_tolerance and
                            abs(self.view_end - self.cached_view_end) < position_tolerance)
            same_freq = (self.cached_min_freq == self.min_freq and
                        self.cached_max_freq == self.max_freq)
            if zoom_similar and same_position and same_freq:
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

        # Downsample if needed, but ensure Nyquist stays above max display freq
        effective_sr = self.sample_rate
        if len(segment) > 2_000_000:
            downsample_factor = int(np.ceil(len(segment) / 2_000_000))
            # Limit downsample so Nyquist (effective_sr/2) >= max_freq
            if self.max_freq > 0:
                max_allowed = max(1, int(self.sample_rate / (2 * self.max_freq)))
                downsample_factor = min(downsample_factor, max_allowed)
            if downsample_factor > 1:
                segment = segment[::downsample_factor]
                effective_sr = self.sample_rate / downsample_factor

        # For very large segments, reduce time resolution instead of frequency
        # Target ~2000 time columns max for the spectrogram image
        target_time_cols = 2000
        nperseg = min(512, len(segment) // 10)
        nperseg = max(128, nperseg)
        # Adjust overlap to limit output columns for wide windows
        min_hop = max(1, len(segment) // target_time_cols)
        noverlap = max(0, nperseg - min_hop)
        noverlap = min(noverlap, int(nperseg * 0.75))  # Cap at 75% overlap
        nfft = max(nperseg, 512)

        frequencies, times, Sxx = signal.spectrogram(
            segment, fs=effective_sr,
            nperseg=nperseg, noverlap=noverlap, nfft=nfft, window='hann'
        )

        times = times + start_time
        spec_db = 10 * np.log10(Sxx + 1e-10)

        # Filter to view range (time)
        time_mask = (times >= self.view_start) & (times <= self.view_end)
        if not np.any(time_mask):
            self.spec_image = None
            return

        view_spec = spec_db[:, time_mask]

        # Filter to frequency display range
        freq_mask = (frequencies >= self.min_freq) & (frequencies <= self.max_freq)
        if not np.any(freq_mask):
            self.spec_image = None
            return
        view_spec = view_spec[freq_mask, :]

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
        self.cached_min_freq = self.min_freq
        self.cached_max_freq = self.max_freq

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

        # Draw playback position line
        if self.playback_position is not None:
            x = self._time_to_x(self.playback_position, spec_rect)
            if spec_rect.left() <= x <= spec_rect.right():
                painter.setPen(QPen(QColor(255, 255, 255, 220), 2))
                painter.drawLine(int(x), int(spec_rect.top()),
                                 int(x), int(spec_rect.bottom()))

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
        try:
            if math.isnan(start_s) or math.isnan(stop_s):
                return
        except TypeError:
            return
        if isinstance(min_freq, float) and math.isnan(min_freq):
            min_freq = self.min_freq
        if isinstance(max_freq, float) and math.isnan(max_freq):
            max_freq = self.max_freq

        x1 = self._time_to_x(start_s, spec_rect)
        x2 = self._time_to_x(stop_s, spec_rect)
        y1 = self._freq_to_y(max_freq, spec_rect)
        y2 = self._freq_to_y(min_freq, spec_rect)

        is_selected = idx == self.current_detection_idx

        # Color based on status - yellow for pending, green for accepted, red for rejected
        if status == 'accepted':
            color = QColor(0, 200, 0, 220)  # Bright green
            label = "A"
        elif status == 'rejected':
            color = QColor(255, 80, 80, 220)  # Bright red
            label = "R"
        elif status == 'negative':
            color = QColor(255, 50, 50, 180)  # Dark red for negative/background
            label = "N"
        else:
            color = QColor(255, 255, 0, 200)  # Yellow for pending
            label = "P"

        # Selected = white border; non-selected = yellow outline
        if is_selected:
            pen = QPen(QColor(255, 255, 255))
            pen.setWidth(4)
        else:
            pen = QPen(color)
            pen.setWidth(2)

        painter.setPen(pen)
        fill_alpha = 50 if status == 'negative' else 30
        painter.setBrush(QBrush(QColor(color.red(), color.green(), color.blue(), fill_alpha)))
        painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))

        # Draw status label above the box
        if label:
            center_x = (x1 + x2) / 2
            label_y = y1 - 4
            font = QFont("Arial", 14, QFont.Bold)
            painter.setFont(font)
            if is_selected:
                painter.setPen(QColor(255, 255, 255))
            else:
                # Label color matches status
                painter.setPen(color)
            painter.drawText(int(center_x - 10), int(label_y - 16), 20, 18, Qt.AlignCenter, label)

        # Draw frequency contour dots+lines for accepted detections
        if status == 'accepted':
            self._draw_freq_contour(painter, det, x1, x2, spec_rect)

    def _draw_freq_contour(self, painter, det, x1, x2, spec_rect):
        """Draw peak frequency sample dots and connecting lines."""
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
        # Aim for roughly 8-12 major ticks across the view
        nice_intervals = [0.001, 0.002, 0.005, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5,
                          1, 2, 5, 10, 15, 30, 60, 120, 300, 600]
        target_ticks = 10
        ideal_interval = view_duration / target_ticks
        tick_interval = nice_intervals[-1]
        for ni in nice_intervals:
            if ni >= ideal_interval:
                tick_interval = ni
                break

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
        minor_interval = tick_interval / 10.0
        first_minor = math.ceil(self.view_start / minor_interval) * minor_interval
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
        first_tick = math.ceil(self.view_start / tick_interval) * tick_interval
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

        # Frequency axis — adaptive tick intervals like time axis
        freq_range_khz = (self.max_freq - self.min_freq) / 1000.0
        nice_freq_intervals = [0.5, 1, 2, 5, 10, 25, 50]  # kHz
        target_freq_ticks = 8
        ideal_freq_interval = freq_range_khz / target_freq_ticks
        freq_tick_khz = nice_freq_intervals[-1]
        for ni in nice_freq_intervals:
            if ni >= ideal_freq_interval:
                freq_tick_khz = ni
                break

        # Draw minor frequency ticks (subdivisions)
        minor_freq_khz = freq_tick_khz / 5.0
        painter.setPen(QColor(120, 120, 120))
        first_minor_f = math.ceil(self.min_freq / 1000.0 / minor_freq_khz) * minor_freq_khz
        f_khz = first_minor_f
        while f_khz * 1000.0 <= self.max_freq:
            freq = f_khz * 1000.0
            y = self._freq_to_y(freq, spec_rect)
            if spec_rect.top() <= y <= spec_rect.bottom():
                painter.drawLine(int(spec_rect.left() - 4), int(y),
                                 int(spec_rect.left()), int(y))
            f_khz += minor_freq_khz

        # Draw major frequency ticks with labels (centered vertically on tick)
        painter.setPen(QColor(200, 200, 200))
        first_major_f = math.ceil(self.min_freq / 1000.0 / freq_tick_khz) * freq_tick_khz
        f_khz = first_major_f
        label_h = 16
        while f_khz * 1000.0 <= self.max_freq:
            freq = f_khz * 1000.0
            y = self._freq_to_y(freq, spec_rect)
            if spec_rect.top() <= y <= spec_rect.bottom():
                painter.drawLine(int(spec_rect.left() - 8), int(y),
                                 int(spec_rect.left()), int(y))
                painter.drawText(0, int(y - label_h // 2), 42, label_h,
                                 Qt.AlignRight | Qt.AlignVCenter, f"{f_khz:.0f}")
            f_khz += freq_tick_khz

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

        # First: check if clicking inside ANY detection box (for selection)
        clicked_det = self._find_detection_at_pos(pos, spec_rect)

        if clicked_det >= 0 and clicked_det != self.current_detection_idx:
            # Clicked a non-selected detection — select it (don't drag)
            self.detection_selected.emit(clicked_det)
            return

        # Now check edges/move on the SELECTED detection only
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
            # Not on any detection — start drawing a new box
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

    def wheelEvent(self, event):
        """Handle mouse wheel for zoom. Zoom centered on mouse position."""
        if self.total_duration <= 0:
            return
        spec_rect = self._get_spec_rect()
        if not spec_rect.contains(event.pos()):
            return
        delta = event.angleDelta().y()
        if delta == 0:
            return
        # Zoom factor: scroll up = zoom in, scroll down = zoom out
        factor = 1.3 if delta > 0 else 1 / 1.3
        center_time = self._x_to_time(event.pos().x(), spec_rect)
        self.zoom_requested.emit(factor, center_time)
        event.accept()

    def _find_edge_at_pos(self, pos, spec_rect, threshold=8):
        """Find if position is near a detection edge. Prioritizes selected detection."""
        def _check_det(i, det):
            start_s = det.get('start_seconds', 0)
            stop_s = det.get('stop_seconds', 0)
            min_freq = det.get('min_freq_hz', self.min_freq)
            max_freq = det.get('max_freq_hz', self.max_freq)

            x1 = self._time_to_x(start_s, spec_rect)
            x2 = self._time_to_x(stop_s, spec_rect)
            y1 = self._freq_to_y(max_freq, spec_rect)
            y2 = self._freq_to_y(min_freq, spec_rect)

            # Expand y-range slightly for edge detection
            y_pad = threshold
            if abs(pos.x() - x1) < threshold and y1 - y_pad <= pos.y() <= y2 + y_pad:
                return 'resize_left', i
            if abs(pos.x() - x2) < threshold and y1 - y_pad <= pos.y() <= y2 + y_pad:
                return 'resize_right', i
            if abs(pos.y() - y1) < threshold and x1 <= pos.x() <= x2:
                return 'resize_top', i
            if abs(pos.y() - y2) < threshold and x1 <= pos.x() <= x2:
                return 'resize_bottom', i

            if i == self.current_detection_idx:
                if x1 < pos.x() < x2 and y1 < pos.y() < y2:
                    return 'move', i

            return None, None

        # Check currently selected detection first (gives it priority)
        if 0 <= self.current_detection_idx < len(self.detections):
            result = _check_det(self.current_detection_idx, self.detections[self.current_detection_idx])
            if result[0] is not None:
                return result

        # Then check all others
        for i, det in enumerate(self.detections):
            if i == self.current_detection_idx:
                continue
            result = _check_det(i, det)
            if result[0] is not None:
                return result

        return None, None

    def _find_detection_at_pos(self, pos, spec_rect):
        """Find detection at position (checks both x and y axes).

        Uses a generous click target — the box itself plus a 6px padding zone,
        so small boxes are easier to click on.
        """
        pad = 6  # pixels of padding around each box
        for i, det in enumerate(self.detections):
            start_s = det.get('start_seconds', 0)
            stop_s = det.get('stop_seconds', 0)
            min_freq = det.get('min_freq_hz', self.min_freq)
            max_freq = det.get('max_freq_hz', self.max_freq)

            x1 = self._time_to_x(start_s, spec_rect)
            x2 = self._time_to_x(stop_s, spec_rect)
            y1 = self._freq_to_y(max_freq, spec_rect)  # top
            y2 = self._freq_to_y(min_freq, spec_rect)  # bottom

            if (x1 - pad) <= pos.x() <= (x2 + pad) and (y1 - pad) <= pos.y() <= (y2 + pad):
                return i
        return -1

    def _handle_drag(self, pos, spec_rect):
        """Handle drag for resize or move with clamped coordinates."""
        if self.drag_detection_idx is None or self.drag_detection_idx >= len(self.detections):
            self.drag_mode = None
            return

        det = self.detections[self.drag_detection_idx]
        time_s = self._x_to_time(pos.x(), spec_rect)
        freq_hz = self._y_to_freq(pos.y(), spec_rect)

        # Clamp to valid ranges
        time_s = max(0, min(time_s, self.total_duration))
        freq_hz = max(0, min(freq_hz, self.max_freq))

        start_s = det.get('start_seconds', 0)
        stop_s = det.get('stop_seconds', 0)
        min_freq = det.get('min_freq_hz', self.min_freq)
        max_freq = det.get('max_freq_hz', self.max_freq)

        if self.drag_mode == 'resize_left':
            start_s = max(0, min(time_s, stop_s - 0.001))
        elif self.drag_mode == 'resize_right':
            stop_s = min(self.total_duration, max(time_s, start_s + 0.001))
        elif self.drag_mode == 'resize_top':
            max_freq = min(self.max_freq, max(freq_hz, min_freq + 100))
        elif self.drag_mode == 'resize_bottom':
            min_freq = max(0, min(freq_hz, max_freq - 100))
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
# Waveform Overview Widget
# =============================================================================

class WaveformOverviewWidget(QWidget):
    """Thin waveform strip showing full file with viewport highlight."""

    view_changed = pyqtSignal(float)  # center time clicked

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedHeight(40)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)

        self.envelope = None  # Downsampled envelope
        self.total_duration = 0.0
        self.view_start = 0.0
        self.view_end = 1.0
        self._dragging = False
        self.detection_times = []  # List of (start_s, stop_s) for tick marks

    def set_audio_data(self, audio_data, sample_rate):
        """Compute downsampled envelope for display."""
        if audio_data is None or sample_rate is None or len(audio_data) == 0:
            self.envelope = None
            self.total_duration = 0.0
            self.update()
            return

        self.total_duration = len(audio_data) / sample_rate
        # Downsample to ~1000 points for the overview
        n_bins = min(1000, len(audio_data))
        bin_size = max(1, len(audio_data) // n_bins)
        trimmed = audio_data[:bin_size * n_bins]
        reshaped = trimmed.reshape(n_bins, bin_size)
        self.envelope = np.max(np.abs(reshaped), axis=1)
        # Normalize
        peak = np.max(self.envelope)
        if peak > 0:
            self.envelope = self.envelope / peak
        self.update()

    def set_detections(self, detection_times):
        """Set detection positions for tick marks.

        Args:
            detection_times: List of (start_seconds, stop_seconds) tuples
        """
        self.detection_times = detection_times or []
        self.update()

    def set_view_range(self, start, end):
        """Update viewport highlight."""
        self.view_start = start
        self.view_end = end
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(20, 20, 20))

        if self.envelope is None or self.total_duration <= 0:
            return

        w = self.width()
        h = self.height()

        # Draw waveform envelope
        painter.setPen(QPen(QColor(0, 120, 212, 180), 1))
        n = len(self.envelope)
        mid_y = h / 2
        for i in range(n):
            x = int(i / n * w)
            amp = self.envelope[i] * mid_y * 0.9
            painter.drawLine(x, int(mid_y - amp), x, int(mid_y + amp))

        # Draw detection tick marks
        if self.detection_times and self.total_duration > 0:
            painter.setPen(QPen(QColor(255, 200, 50, 180), 1))
            for start_s, stop_s in self.detection_times:
                cx = int((start_s + stop_s) / 2.0 / self.total_duration * w)
                painter.drawLine(cx, 0, cx, h)

        # Draw viewport highlight
        x1 = int(self.view_start / self.total_duration * w)
        x2 = int(self.view_end / self.total_duration * w)
        painter.setBrush(QBrush(QColor(0, 120, 212, 50)))
        painter.setPen(QPen(QColor(0, 120, 212, 200), 1))
        painter.drawRect(x1, 0, max(2, x2 - x1), h)

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton and self.total_duration > 0:
            self._dragging = True
            center = event.pos().x() / self.width() * self.total_duration
            self.view_changed.emit(center)

    def mouseMoveEvent(self, event):
        if self._dragging and self.total_duration > 0:
            center = event.pos().x() / self.width() * self.total_duration
            center = max(0, min(center, self.total_duration))
            self.view_changed.emit(center)

    def mouseReleaseEvent(self, event):
        self._dragging = False


# =============================================================================
# Worker Threads
# =============================================================================

class DSPDetectionWorker(QThread):
    """Worker thread for DSP detection."""
    progress = pyqtSignal(str, int, int)  # filename, current, total
    file_progress = pyqtSignal(float)  # fraction 0.0-1.0 within current file
    file_complete = pyqtSignal(str, str, list, int)  # filename, filepath, detections, n_detections
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
            min_freq_hz=self.config.get('min_freq_hz', 20000),
            max_freq_hz=self.config.get('max_freq_hz', 65000),
            energy_threshold_db=self.config.get('energy_threshold_db', 10.0),
            min_duration_ms=self.config.get('min_duration_ms', 10.0),
            max_duration_ms=self.config.get('max_duration_ms', 1000.0),
            max_bandwidth_hz=self.config.get('max_bandwidth_hz', 20000.0),
            min_tonality=self.config.get('min_tonality', 0.3),
            min_call_freq_hz=self.config.get('min_call_freq_hz', 0.0),
            harmonic_filter=self.config.get('harmonic_filter', True),
            min_freq_gap_hz=self.config.get('min_freq_gap_hz', 5000.0),
            min_gap_ms=self.config.get('min_gap_ms', 5.0),
            noise_percentile=self.config.get('noise_percentile', 25.0),
            nperseg=self.config.get('nperseg', 512),
            noverlap=self.config.get('noverlap', 384),
            gpu_enabled=self.config.get('gpu_enabled', False),
            gpu_device=self.config.get('gpu_device', 'auto'),
        )

        detector = DSPDetector(config)
        total = len(self.files)

        for i, filepath in enumerate(self.files):
            if self._stop_requested:
                break

            filename = os.path.basename(filepath)
            self.progress.emit(filename, i, total)
            self.file_progress.emit(0.0)

            try:
                def on_chunk_progress(fraction):
                    self.file_progress.emit(fraction)

                detections = detector.detect_file(filepath, progress_callback=on_chunk_progress)
                # If stop was requested during processing, discard this file's results
                if self._stop_requested:
                    break
                self.results[filepath] = detections
                self.file_complete.emit(filename, filepath, detections, len(detections))
            except Exception as e:
                if self._stop_requested:
                    break
                self.error.emit(filename, str(e))
                self.results[filepath] = []

        self.all_complete.emit(self.results)


class YOLOTrainingWorker(QThread):
    """Worker thread for YOLO model training."""
    progress = pyqtSignal(str)       # status message
    complete = pyqtSignal(str)       # model_path
    error = pyqtSignal(str)          # error message

    def __init__(self, dataset_yaml, output_dir, model_name,
                 pretrained_weights=None):
        super().__init__()
        self.dataset_yaml = dataset_yaml
        self.output_dir = output_dir
        self.model_name = model_name
        self.pretrained_weights = pretrained_weights

    def run(self):
        try:
            from fnt.usv.usv_detector.yolo_detector import train_yolo_model

            self.progress.emit(f"Training {self.model_name} (auto early stopping)...")
            model_path = train_yolo_model(
                self.dataset_yaml, self.output_dir, self.model_name,
                pretrained_weights=self.pretrained_weights,
            )
            self.complete.emit(model_path)
        except Exception as e:
            self.error.emit(str(e))


class YOLOInferenceWorker(QThread):
    """Worker thread for YOLO inference on audio files."""
    progress = pyqtSignal(str, int, int)       # filename, current, total
    file_complete = pyqtSignal(str, str, list, int)  # filename, filepath, detections, count
    all_complete = pyqtSignal(dict)             # filepath -> detections
    error = pyqtSignal(str, str)               # filename, error

    def __init__(self, files, model_path, config, confidence_threshold=0.25):
        super().__init__()
        self.files = files
        self.model_path = model_path
        self.config = config
        self.confidence_threshold = confidence_threshold
        self.results = {}

    def run(self):
        from fnt.usv.usv_detector.yolo_detector import run_yolo_inference

        total = len(self.files)
        for i, filepath in enumerate(self.files):
            filename = os.path.basename(filepath)
            self.progress.emit(filename, i, total)

            try:
                detections = run_yolo_inference(
                    self.model_path, filepath, self.config,
                    confidence_threshold=self.confidence_threshold,
                )
                self.results[filepath] = detections
                self.file_complete.emit(filename, filepath, detections, len(detections))
            except Exception as e:
                self.error.emit(filename, str(e))
                self.results[filepath] = []

        self.all_complete.emit(self.results)


# =============================================================================
# Main USV Studio Window
# =============================================================================

class USVStudioWindow(QMainWindow):
    """Main window for USV Studio."""

    # Species-specific DSP detection presets.
    # 'Manual' means user controls all parameters directly.
    # Add new species by adding a key with a dict of DSP config values.
    SPECIES_PROFILES = {
        'Manual': None,
        'Prairie Vole USVs': {
            'min_freq_hz': 20000, 'max_freq_hz': 65000,
            'energy_threshold_db': 10.0,
            'min_duration_ms': 20.0, 'max_duration_ms': 1000.0,
            'max_bandwidth_hz': 25000, 'min_tonality': 0.50,
            'min_call_freq_hz': 15000, 'harmonic_filter': True,
            'min_freq_gap_hz': 5000,
            'min_gap_ms': 5.0, 'noise_percentile': 25.0,
            'nperseg': 512, 'noverlap': 384,
        },
    }

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
        self.detection_sources = {}  # filepath -> 'dsp' | 'ml' | 'detections' (tracks CSV origin)
        self.dsp_queue = []  # Files queued for DSP detection

        # Playback state
        self.is_playing = False
        self.playback_speed = 1.0
        self.use_heterodyne = False
        self._playback_start_time = None  # Wall-clock time when playback started
        self._playback_start_s = None     # Audio time (seconds) where playback begins
        self._playback_end_s = None       # Audio time (seconds) where playback ends
        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(30)  # ~33 fps update
        self._playback_timer.timeout.connect(self._update_playback_position)

        # Undo stack (stores (action, data) tuples)
        self.undo_stack = deque(maxlen=50)

        # Filter state
        self.filter_status = 'all'  # 'all', 'pending', 'accepted', 'rejected'

        # Workers
        self.dsp_worker = None

        self._setup_ui()
        self._apply_styles()
        self._setup_shortcuts()
        self._setup_pan_timers()
        self._restore_settings()

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
        left_scroll.setFixedWidth(380)  # Fixed width to prevent horizontal scroll

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(8)

        # Build sections
        self._create_input_section(left_layout)
        self._create_dsp_section(left_layout)
        self._create_detection_section(left_layout)
        self._create_labeling_section(left_layout)
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
        self.spectrogram.zoom_requested.connect(self.on_wheel_zoom)
        right_layout.addWidget(self.spectrogram, 1)

        # Waveform overview strip
        self.waveform_overview = WaveformOverviewWidget()
        self.waveform_overview.view_changed.connect(self.on_overview_clicked)
        right_layout.addWidget(self.waveform_overview)

        # Scrollbar row (full width, matching spectrogram)
        scroll_bar_row = QWidget()
        scroll_layout = QHBoxLayout(scroll_bar_row)
        scroll_layout.setContentsMargins(5, 2, 5, 2)
        scroll_layout.setSpacing(2)

        self.btn_pan_left = QPushButton("<")
        self.btn_pan_left.setObjectName("small_btn")
        self.btn_pan_left.setFixedWidth(24)
        self.btn_pan_left.setToolTip("Pan the spectrogram view left in time")
        self.btn_pan_left.clicked.connect(self.pan_left)
        scroll_layout.addWidget(self.btn_pan_left)

        self.time_scrollbar = QScrollBar(Qt.Horizontal)
        self.time_scrollbar.setMinimum(0)
        self.time_scrollbar.setMaximum(1000)
        self.time_scrollbar.setToolTip("Scroll through the recording timeline")
        self.time_scrollbar.valueChanged.connect(self.on_scrollbar_changed)
        scroll_layout.addWidget(self.time_scrollbar, 1)

        self.btn_pan_right = QPushButton(">")
        self.btn_pan_right.setObjectName("small_btn")
        self.btn_pan_right.setFixedWidth(24)
        self.btn_pan_right.setToolTip("Pan the spectrogram view right in time")
        self.btn_pan_right.clicked.connect(self.pan_right)
        scroll_layout.addWidget(self.btn_pan_right)

        right_layout.addWidget(scroll_bar_row)

        # Controls row (window/zoom, freq, colormap, playback)
        controls_bar = QWidget()
        controls_layout = QHBoxLayout(controls_bar)
        controls_layout.setContentsMargins(5, 2, 5, 2)
        controls_layout.setSpacing(4)

        # Window / Zoom
        controls_layout.addWidget(QLabel("Window:"))

        self.btn_zoom_out = QPushButton("-")
        self.btn_zoom_out.setObjectName("small_btn")
        self.btn_zoom_out.setFixedWidth(24)
        self.btn_zoom_out.setToolTip("Zoom out (show more time)")
        self.btn_zoom_out.clicked.connect(self.zoom_out)
        controls_layout.addWidget(self.btn_zoom_out)

        self.spin_view_window = QDoubleSpinBox()
        self.spin_view_window.setRange(0.1, 600.0)
        self.spin_view_window.setValue(2.0)
        self.spin_view_window.setSuffix(" s")
        self.spin_view_window.setFixedWidth(80)
        self.spin_view_window.setToolTip("Time window duration (seconds).\nControls how much of the recording is visible.\nSmaller = zoomed in, larger = zoomed out.\nAlso adjustable with mouse wheel on spectrogram.")
        self.spin_view_window.valueChanged.connect(self.on_view_window_changed)
        controls_layout.addWidget(self.spin_view_window)

        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setObjectName("small_btn")
        self.btn_zoom_in.setFixedWidth(24)
        self.btn_zoom_in.setToolTip("Zoom in (show less time)")
        self.btn_zoom_in.clicked.connect(self.zoom_in)
        controls_layout.addWidget(self.btn_zoom_in)

        # Separator
        sep1 = QFrame()
        sep1.setFrameShape(QFrame.VLine)
        sep1.setStyleSheet("color: #3f3f3f;")
        controls_layout.addWidget(sep1)

        # Frequency display range
        controls_layout.addWidget(QLabel("Freq:"))
        self.spin_display_min_freq = QSpinBox()
        self.spin_display_min_freq.setRange(0, 200000)
        self.spin_display_min_freq.setSingleStep(5000)
        self.spin_display_min_freq.setValue(0)
        self.spin_display_min_freq.setSuffix(" Hz")
        self.spin_display_min_freq.setFixedWidth(90)
        self.spin_display_min_freq.setToolTip("Minimum frequency displayed on the spectrogram (Hz).\nAdjust to zoom into the frequency range of interest.")
        self.spin_display_min_freq.valueChanged.connect(self.on_display_freq_changed)
        controls_layout.addWidget(self.spin_display_min_freq)

        controls_layout.addWidget(QLabel("-"))
        self.spin_display_max_freq = QSpinBox()
        self.spin_display_max_freq.setRange(1000, 250000)
        self.spin_display_max_freq.setSingleStep(5000)
        self.spin_display_max_freq.setValue(125000)
        self.spin_display_max_freq.setSuffix(" Hz")
        self.spin_display_max_freq.setFixedWidth(90)
        self.spin_display_max_freq.setToolTip("Maximum frequency displayed on the spectrogram (Hz).\nPrairie vole USVs are typically 30-110 kHz.")
        self.spin_display_max_freq.valueChanged.connect(self.on_display_freq_changed)
        controls_layout.addWidget(self.spin_display_max_freq)

        # Separator
        sep2 = QFrame()
        sep2.setFrameShape(QFrame.VLine)
        sep2.setStyleSheet("color: #3f3f3f;")
        controls_layout.addWidget(sep2)

        # Colormap selector
        controls_layout.addWidget(QLabel("Color Map:"))
        self.combo_colormap = QComboBox()
        self.combo_colormap.addItems(['viridis', 'magma', 'inferno', 'grayscale'])
        self.combo_colormap.setFixedWidth(85)
        self.combo_colormap.setToolTip("Spectrogram color scheme.\nViridis (default) and magma work well for USV visualization.\nGrayscale can be helpful for publications.")
        self.combo_colormap.currentTextChanged.connect(self.on_colormap_changed)
        controls_layout.addWidget(self.combo_colormap)

        # Separator
        sep3 = QFrame()
        sep3.setFrameShape(QFrame.VLine)
        sep3.setStyleSheet("color: #3f3f3f;")
        controls_layout.addWidget(sep3)

        # Playback controls (moved from left panel)
        self.btn_play = QPushButton("Play")
        self.btn_play.setToolTip("Play the current detection audio (Space key).\nAudio is slowed down according to the speed setting\nso ultrasonic calls become audible.")
        self.btn_play.clicked.connect(self.toggle_playback)
        self.btn_play.setEnabled(False)
        controls_layout.addWidget(self.btn_play)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet("background-color: #5c5c5c;")
        self.btn_stop.setToolTip("Stop audio playback immediately.")
        self.btn_stop.clicked.connect(self.stop_playback)
        self.btn_stop.setEnabled(False)
        controls_layout.addWidget(self.btn_stop)

        controls_layout.addWidget(QLabel("Speed:"))
        # Speed slider: positions map to [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.5, 1.0]
        self._speed_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.5, 1.0]
        self.slider_speed = QSlider(Qt.Horizontal)
        self.slider_speed.setRange(0, len(self._speed_values) - 1)
        self.slider_speed.setValue(len(self._speed_values) - 1)  # Default 1.0x
        self.slider_speed.setFixedWidth(100)
        self.slider_speed.setToolTip("Playback speed multiplier.\nLower values slow audio down, shifting\nultrasonic frequencies into the audible range.\n0.1x is a good default for ~50 kHz USV calls.")
        self.slider_speed.valueChanged.connect(self.on_speed_changed)
        controls_layout.addWidget(self.slider_speed)
        self.lbl_speed = QLabel("1.0x")
        self.lbl_speed.setFixedWidth(35)
        controls_layout.addWidget(self.lbl_speed)

        if not HAS_SOUNDDEVICE:
            lbl_warn = QLabel("No audio")
            lbl_warn.setStyleSheet("color: #d13438; font-size: 9px;")
            controls_layout.addWidget(lbl_warn)

        controls_layout.addStretch()

        right_layout.addWidget(controls_bar)

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
        self.btn_add_folder.setToolTip("Add all WAV files from a folder")
        self.btn_add_folder.clicked.connect(self.add_folder)
        btn_row.addWidget(self.btn_add_folder)

        self.btn_add_files = QPushButton("Add Files")
        self.btn_add_files.setToolTip("Select individual WAV files to add")
        self.btn_add_files.clicked.connect(self.add_files)
        btn_row.addWidget(self.btn_add_files)

        self.btn_clear_files = QPushButton("Clear")
        self.btn_clear_files.setStyleSheet("background-color: #5c5c5c;")
        self.btn_clear_files.setToolTip("Remove all loaded files")
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
        self.btn_prev_file.setToolTip("Load the previous file in the list")
        self.btn_prev_file.clicked.connect(self.prev_file)
        self.btn_prev_file.setEnabled(False)
        nav_row.addWidget(self.btn_prev_file)

        self.lbl_file_num = QLabel("File 0/0")
        self.lbl_file_num.setAlignment(Qt.AlignCenter)
        nav_row.addWidget(self.lbl_file_num, 1)

        self.btn_next_file = QPushButton("Next >")
        self.btn_next_file.setObjectName("small_btn")
        self.btn_next_file.setToolTip("Load the next file in the list")
        self.btn_next_file.clicked.connect(self.next_file)
        self.btn_next_file.setEnabled(False)
        nav_row.addWidget(self.btn_next_file)

        group_layout.addLayout(nav_row)

        # Open Folder button
        self.btn_open_folder = QPushButton("Open Folder")
        self.btn_open_folder.setStyleSheet("background-color: #5c5c5c;")
        self.btn_open_folder.setToolTip("Open the folder containing the current file\nin the system file browser.")
        self.btn_open_folder.clicked.connect(self.open_output_folder)
        self.btn_open_folder.setEnabled(False)
        group_layout.addWidget(self.btn_open_folder)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _make_label(self, text, tooltip=None, min_width=0):
        """Helper to create a label with optional tooltip and minimum width."""
        lbl = QLabel(text)
        if tooltip:
            lbl.setToolTip(tooltip)
        if min_width > 0:
            lbl.setMinimumWidth(min_width)
        return lbl

    def _create_dsp_section(self, layout):
        """Create DSP detection section."""
        group = QGroupBox("2. DSP Detection")
        group.setToolTip("Configure and run the DSP-based ultrasonic vocalization detector")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(4)

        # Species profile dropdown
        profile_row = QHBoxLayout()
        profile_row.setSpacing(4)
        profile_row.addWidget(self._make_label("Profile:", "Select a species profile to auto-fill\nDSP parameters, or Manual to set them yourself.", min_width=90))
        self.combo_species_profile = QComboBox()
        for name in self.SPECIES_PROFILES.keys():
            self.combo_species_profile.addItem(name)
        self.combo_species_profile.setCurrentText("Manual")
        self.combo_species_profile.setToolTip("Species profile presets.\nSelecting a profile auto-fills all DSP parameters\nand locks them. Choose Manual to customize.")
        self.combo_species_profile.currentTextChanged.connect(self._on_species_profile_changed)
        profile_row.addWidget(self.combo_species_profile, 1)
        group_layout.addLayout(profile_row)

        # Frequency range
        freq_tip = ("Bandpass filter range for detection.\n"
                     "Only energy within this band is analyzed.\n"
                     "Prairie voles: typically 25-65 kHz.\n"
                     "Mice: typically 30-110 kHz.")
        freq_row = QHBoxLayout()
        freq_row.setSpacing(4)
        freq_row.addWidget(self._make_label("Freq Range:", freq_tip, min_width=90))

        self.spin_min_freq = QSpinBox()
        self.spin_min_freq.setRange(1000, 150000)
        self.spin_min_freq.setSingleStep(1000)
        self.spin_min_freq.setValue(20000)
        self.spin_min_freq.setSuffix(" Hz")
        self.spin_min_freq.setToolTip("Minimum frequency of the detection bandpass filter (Hz)")
        freq_row.addWidget(self.spin_min_freq, 1)

        freq_row.addWidget(QLabel("-"))

        self.spin_max_freq = QSpinBox()
        self.spin_max_freq.setRange(1000, 150000)
        self.spin_max_freq.setSingleStep(1000)
        self.spin_max_freq.setValue(65000)
        self.spin_max_freq.setSuffix(" Hz")
        self.spin_max_freq.setToolTip("Maximum frequency of the detection bandpass filter (Hz)")
        freq_row.addWidget(self.spin_max_freq, 1)

        group_layout.addLayout(freq_row)

        # Threshold
        thresh_tip = ("Energy threshold above the noise floor (dB).\n"
                      "Lower = more sensitive (more detections, more noise).\n"
                      "Higher = more conservative (fewer false positives).\n"
                      "Typical range: 6-15 dB.")
        thresh_row = QHBoxLayout()
        thresh_row.setSpacing(4)
        thresh_row.addWidget(self._make_label("Threshold:", thresh_tip, min_width=90))

        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(1.0, 30.0)
        self.spin_threshold.setSingleStep(0.5)
        self.spin_threshold.setValue(10.0)
        self.spin_threshold.setSuffix(" dB")
        self.spin_threshold.setToolTip(thresh_tip)
        thresh_row.addWidget(self.spin_threshold)
        thresh_row.addStretch()

        group_layout.addLayout(thresh_row)

        # Duration
        dur_tip = ("Min/max call duration filter.\n"
                   "Detections outside this range are discarded.\n"
                   "Most USV calls are 5-200 ms.")
        dur_row = QHBoxLayout()
        dur_row.setSpacing(4)
        dur_row.addWidget(self._make_label("Duration:", dur_tip, min_width=90))

        self.spin_min_dur = QDoubleSpinBox()
        self.spin_min_dur.setRange(1.0, 100.0)
        self.spin_min_dur.setValue(10.0)
        self.spin_min_dur.setSuffix(" ms")
        self.spin_min_dur.setToolTip("Minimum call duration — shorter events are discarded (ms)")
        dur_row.addWidget(self.spin_min_dur, 1)

        dur_row.addWidget(QLabel("-"))

        self.spin_max_dur = QDoubleSpinBox()
        self.spin_max_dur.setRange(10.0, 5000.0)
        self.spin_max_dur.setValue(1000.0)
        self.spin_max_dur.setSuffix(" ms")
        self.spin_max_dur.setToolTip("Maximum call duration — longer events are discarded (ms)")
        dur_row.addWidget(self.spin_max_dur, 1)

        group_layout.addLayout(dur_row)

        # --- Noise Rejection Filters ---
        # Max bandwidth
        bw_tip = ("Maximum frequency bandwidth for a detection (Hz).\n"
                  "Detections spanning a wider frequency range are\n"
                  "discarded as broadband noise (cage bumps, movement).\n"
                  "Real USV calls are narrowband (typically <15 kHz).\n"
                  "Set to 0 to disable this filter.")
        bw_row = QHBoxLayout()
        bw_row.setSpacing(4)
        bw_row.addWidget(self._make_label("Max Bandwidth:", bw_tip, min_width=90))
        self.spin_max_bw = QSpinBox()
        self.spin_max_bw.setRange(0, 100000)
        self.spin_max_bw.setSingleStep(1000)
        self.spin_max_bw.setValue(20000)
        self.spin_max_bw.setSuffix(" Hz")
        self.spin_max_bw.setToolTip(bw_tip)
        bw_row.addWidget(self.spin_max_bw, 1)
        group_layout.addLayout(bw_row)

        # --- Advanced Options (collapsible) ---
        self.btn_advanced_toggle = QPushButton("▶ Advanced Options")
        self.btn_advanced_toggle.setFlat(True)
        self.btn_advanced_toggle.setStyleSheet("text-align: left; color: #aaaaaa; font-size: 11px; padding: 2px 0px;")
        self.btn_advanced_toggle.setCursor(Qt.PointingHandCursor)
        self.btn_advanced_toggle.clicked.connect(self._toggle_advanced_options)
        group_layout.addWidget(self.btn_advanced_toggle)

        self.advanced_options_widget = QWidget()
        advanced_layout = QVBoxLayout()
        advanced_layout.setContentsMargins(0, 0, 0, 0)
        advanced_layout.setSpacing(4)

        # Tonality (spectral purity)
        ton_tip = ("Minimum spectral purity score (0.0 - 1.0).\n"
                   "Measures how tonal vs. noisy a detection is.\n"
                   "1.0 = pure tone (energy at one frequency).\n"
                   "0.0 = broadband noise (energy spread everywhere).\n"
                   "Real USV calls are tonal (typically >0.3).\n"
                   "Higher = stricter, fewer false positives.\n"
                   "Set to 0 to disable this filter.")
        ton_row = QHBoxLayout()
        ton_row.setSpacing(4)
        ton_row.addWidget(self._make_label("Tonality:", ton_tip, min_width=90))
        self.spin_tonality = QDoubleSpinBox()
        self.spin_tonality.setRange(0.0, 1.0)
        self.spin_tonality.setSingleStep(0.05)
        self.spin_tonality.setDecimals(2)
        self.spin_tonality.setValue(0.30)
        self.spin_tonality.setToolTip(ton_tip)
        ton_row.addWidget(self.spin_tonality, 1)
        advanced_layout.addLayout(ton_row)

        # Min call frequency
        mcf_tip = ("Minimum actual frequency of a detection (Hz).\n"
                   "Detections whose lowest frequency falls below\n"
                   "this are discarded as likely broadband noise.\n"
                   "Broadband artifacts often extend down to the\n"
                   "detection floor, while real USVs start higher.\n"
                   "Set to 0 to disable this filter.")
        mcf_row = QHBoxLayout()
        mcf_row.setSpacing(4)
        mcf_row.addWidget(self._make_label("Min Call Freq:", mcf_tip, min_width=90))
        self.spin_min_call_freq = QSpinBox()
        self.spin_min_call_freq.setRange(0, 100000)
        self.spin_min_call_freq.setSingleStep(1000)
        self.spin_min_call_freq.setValue(0)
        self.spin_min_call_freq.setSuffix(" Hz")
        self.spin_min_call_freq.setToolTip(mcf_tip)
        mcf_row.addWidget(self.spin_min_call_freq, 1)
        advanced_layout.addLayout(mcf_row)

        # Harmonic filter checkbox
        harmonic_tip = ("Merge temporally overlapping detections that are\n"
                        "likely harmonics of the same call.\n"
                        "When enabled, if two detections overlap by >=80%\n"
                        "in time, only the lowest-frequency one (the\n"
                        "fundamental) is kept and the harmonic is discarded.\n"
                        "Recommended for species that produce strong harmonics.")
        self.chk_harmonic_filter = QCheckBox("Harmonic Filter")
        self.chk_harmonic_filter.setChecked(True)
        self.chk_harmonic_filter.setToolTip(harmonic_tip)
        advanced_layout.addWidget(self.chk_harmonic_filter)

        # Frequency gap splitting
        freq_gap_tip = ("Minimum vertical frequency gap (Hz) to split\n"
                        "a single connected detection into two.\n"
                        "When a fundamental and harmonic are connected\n"
                        "by noise, this splits them so the harmonic\n"
                        "filter can remove the upper one.\n"
                        "Set to 0 to disable. 5000 Hz works well for\n"
                        "prairie vole USVs.")
        freq_gap_row = QHBoxLayout()
        freq_gap_row.setSpacing(4)
        freq_gap_row.addWidget(self._make_label("Freq Gap:", freq_gap_tip, min_width=90))
        self.spin_freq_gap = QSpinBox()
        self.spin_freq_gap.setRange(0, 30000)
        self.spin_freq_gap.setSingleStep(1000)
        self.spin_freq_gap.setValue(5000)
        self.spin_freq_gap.setSuffix(" Hz")
        self.spin_freq_gap.setToolTip(freq_gap_tip)
        freq_gap_row.addWidget(self.spin_freq_gap, 1)
        advanced_layout.addLayout(freq_gap_row)

        # Min gap
        gap_tip = ("Minimum silent gap between calls (ms).\n"
                   "Calls closer together than this are merged\n"
                   "into a single detection.")
        gap_row = QHBoxLayout()
        gap_row.setSpacing(4)
        gap_row.addWidget(self._make_label("Min Gap:", gap_tip, min_width=90))
        self.spin_min_gap = QDoubleSpinBox()
        self.spin_min_gap.setRange(0.0, 100.0)
        self.spin_min_gap.setValue(5.0)
        self.spin_min_gap.setSuffix(" ms")
        self.spin_min_gap.setToolTip(gap_tip)
        gap_row.addWidget(self.spin_min_gap, 1)
        advanced_layout.addLayout(gap_row)

        # Noise percentile
        noise_tip = ("Percentile of spectrogram power used to\n"
                     "estimate the background noise floor.\n"
                     "Lower = assumes quieter background.\n"
                     "Typical: 20-30.")
        noise_row = QHBoxLayout()
        noise_row.setSpacing(4)
        noise_row.addWidget(self._make_label("Noise %tile:", noise_tip, min_width=90))
        self.spin_noise_pct = QDoubleSpinBox()
        self.spin_noise_pct.setRange(1.0, 50.0)
        self.spin_noise_pct.setValue(25.0)
        self.spin_noise_pct.setToolTip(noise_tip)
        noise_row.addWidget(self.spin_noise_pct, 1)
        advanced_layout.addLayout(noise_row)

        # FFT params
        fft_tip = ("FFT window size (samples). Larger = better\n"
                   "frequency resolution but worse time resolution.")
        overlap_tip = ("FFT overlap (samples). Higher overlap gives\n"
                       "smoother spectrograms but costs more compute.\n"
                       "Typical: 75% of FFT size.")
        fft_row = QHBoxLayout()
        fft_row.setSpacing(4)
        fft_row.addWidget(self._make_label("FFT:", fft_tip, min_width=34))
        self.spin_nperseg = QSpinBox()
        self.spin_nperseg.setRange(64, 2048)
        self.spin_nperseg.setSingleStep(64)
        self.spin_nperseg.setValue(512)
        self.spin_nperseg.setToolTip(fft_tip)
        fft_row.addWidget(self.spin_nperseg, 1)
        fft_row.addWidget(self._make_label("Overlap:", overlap_tip, min_width=50))
        self.spin_noverlap = QSpinBox()
        self.spin_noverlap.setRange(0, 1024)
        self.spin_noverlap.setSingleStep(64)
        self.spin_noverlap.setValue(384)
        self.spin_noverlap.setToolTip(overlap_tip)
        fft_row.addWidget(self.spin_noverlap, 1)
        advanced_layout.addLayout(fft_row)

        # Frequency samples (optional)
        freq_samp_tip = ("Sample peak frequency at N evenly-spaced\n"
                         "time points across each call. Enables\n"
                         "frequency contour visualization on accepted\n"
                         "detections (cyan line overlay).")
        freq_samp_row = QHBoxLayout()
        freq_samp_row.setSpacing(4)
        self.chk_freq_samples = QCheckBox("Freq Samples:")
        self.chk_freq_samples.setChecked(False)
        self.chk_freq_samples.setToolTip(freq_samp_tip)
        self.chk_freq_samples.toggled.connect(lambda checked: self.spin_freq_samples.setEnabled(checked))
        freq_samp_row.addWidget(self.chk_freq_samples)
        self.spin_freq_samples = QSpinBox()
        self.spin_freq_samples.setRange(3, 10)
        self.spin_freq_samples.setValue(5)
        self.spin_freq_samples.setEnabled(False)
        self.spin_freq_samples.setToolTip(freq_samp_tip)
        self.spin_freq_samples.setFixedWidth(60)
        freq_samp_row.addWidget(self.spin_freq_samples)
        freq_samp_row.addStretch()
        advanced_layout.addLayout(freq_samp_row)

        self.advanced_options_widget.setLayout(advanced_layout)
        self.advanced_options_widget.setVisible(False)
        group_layout.addWidget(self.advanced_options_widget)

        # GPU acceleration checkbox
        gpu_row = QHBoxLayout()
        gpu_row.setSpacing(4)
        self.chk_gpu_accel = QCheckBox("Enable GPU Acceleration")
        self.chk_gpu_accel.setChecked(False)
        self.chk_gpu_accel.setToolTip(
            "Use GPU for spectrogram computation (FFT).\n"
            "Supports NVIDIA CUDA and Apple Silicon MPS.\n"
            "Falls back to CPU if no compatible GPU found."
        )
        self.chk_gpu_accel.toggled.connect(self._on_gpu_toggle)
        gpu_row.addWidget(self.chk_gpu_accel)
        self.lbl_gpu_status = QLabel("")
        self.lbl_gpu_status.setStyleSheet("color: #999999; font-size: 9px;")
        gpu_row.addWidget(self.lbl_gpu_status)
        gpu_row.addStretch()
        group_layout.addLayout(gpu_row)
        self._selected_gpu_device = "auto"

        # Collect all DSP parameter widgets for profile enable/disable
        self._dsp_param_widgets = [
            self.spin_min_freq, self.spin_max_freq,
            self.spin_threshold,
            self.spin_min_dur, self.spin_max_dur,
            self.spin_max_bw, self.spin_tonality, self.spin_min_call_freq,
            self.chk_harmonic_filter, self.spin_freq_gap,
            self.spin_min_gap, self.spin_noise_pct,
            self.spin_nperseg, self.spin_noverlap,
            self.chk_freq_samples, self.spin_freq_samples,
        ]

        # Queue display
        self.lbl_queue = QLabel("Queue: 0 files")
        self.lbl_queue.setStyleSheet("color: #999999;")
        group_layout.addWidget(self.lbl_queue)

        # Queue buttons - row 1
        queue_row = QHBoxLayout()
        queue_row.setSpacing(2)

        self.btn_add_to_queue = QPushButton("Add Current")
        self.btn_add_to_queue.setToolTip("Add the currently selected file to the detection queue")
        self.btn_add_to_queue.clicked.connect(self.add_to_queue)
        self.btn_add_to_queue.setEnabled(False)
        queue_row.addWidget(self.btn_add_to_queue)

        self.btn_add_all_to_queue = QPushButton("Add All")
        self.btn_add_all_to_queue.setToolTip("Add all imported files to the detection queue")
        self.btn_add_all_to_queue.clicked.connect(self.add_all_to_queue)
        self.btn_add_all_to_queue.setEnabled(False)
        queue_row.addWidget(self.btn_add_all_to_queue)

        self.btn_clear_queue = QPushButton("Clear")
        self.btn_clear_queue.setStyleSheet("background-color: #5c5c5c;")
        self.btn_clear_queue.setToolTip("Remove all files from the detection queue")
        self.btn_clear_queue.clicked.connect(self.clear_queue)
        queue_row.addWidget(self.btn_clear_queue)

        group_layout.addLayout(queue_row)

        # Run button
        self.btn_run_dsp = QPushButton("Run DSP Detection")
        self.btn_run_dsp.setStyleSheet("background-color: #0078d4;")
        self.btn_run_dsp.setToolTip("Run DSP-based detection on all queued files")
        self.btn_run_dsp.clicked.connect(self.run_dsp_detection)
        self.btn_run_dsp.setEnabled(False)
        group_layout.addWidget(self.btn_run_dsp)

        # Batch progress (tracks files completed)
        self.dsp_progress = QProgressBar()
        self.dsp_progress.setValue(0)
        self.dsp_progress.setVisible(False)
        self.dsp_progress.setFormat("Batch: %p%")
        group_layout.addWidget(self.dsp_progress)

        # Per-file progress (tracks chunks within current file)
        self.dsp_file_progress = QProgressBar()
        self.dsp_file_progress.setValue(0)
        self.dsp_file_progress.setVisible(False)
        self.dsp_file_progress.setFormat("File: %p%")
        self.dsp_file_progress.setMaximumHeight(12)
        group_layout.addWidget(self.dsp_file_progress)

        self.btn_stop_dsp = QPushButton("Stop Detection")
        self.btn_stop_dsp.setStyleSheet("background-color: #d13438;")
        self.btn_stop_dsp.setToolTip("Stop DSP detection after the current file finishes.\nResults for the file being processed will be discarded.")
        self.btn_stop_dsp.clicked.connect(self.stop_dsp_detection)
        self.btn_stop_dsp.setVisible(False)
        group_layout.addWidget(self.btn_stop_dsp)

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

        # Filter row
        filter_row = QHBoxLayout()
        filter_row.setSpacing(2)
        filter_row.addWidget(QLabel("Show:"))
        self.combo_filter = QComboBox()
        self.combo_filter.addItem("All", "all")
        self.combo_filter.addItem("Pending", "pending")
        self.combo_filter.addItem("Accepted", "accepted")
        self.combo_filter.addItem("Rejected", "rejected")
        self.combo_filter.addItem("Negative", "negative")
        self.combo_filter.setToolTip("Filter which detections to navigate through")
        self.combo_filter.currentIndexChanged.connect(self.on_filter_changed)
        filter_row.addWidget(self.combo_filter, 1)

        # Jump-to spinner
        filter_row.addWidget(QLabel("Go:"))
        self.spin_jump = QSpinBox()
        self.spin_jump.setRange(1, 1)
        self.spin_jump.setFixedWidth(60)
        self.spin_jump.setToolTip("Jump directly to a detection by number.\nType a number and press Enter.")
        self.spin_jump.editingFinished.connect(self.on_jump_to_detection)
        filter_row.addWidget(self.spin_jump)
        group_layout.addLayout(filter_row)

        # Navigation
        nav_row = QHBoxLayout()
        nav_row.setSpacing(2)

        self.btn_prev_det = QPushButton("<")
        self.btn_prev_det.setObjectName("small_btn")
        self.btn_prev_det.setToolTip("Go to previous detection (Left arrow key)")
        self.btn_prev_det.clicked.connect(self.prev_detection)
        self.btn_prev_det.setEnabled(False)
        nav_row.addWidget(self.btn_prev_det)

        self.lbl_det_num = QLabel("Det 0/0")
        self.lbl_det_num.setAlignment(Qt.AlignCenter)
        nav_row.addWidget(self.lbl_det_num, 1)

        self.btn_next_det = QPushButton(">")
        self.btn_next_det.setObjectName("small_btn")
        self.btn_next_det.setToolTip("Go to next detection (Right arrow key)")
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
        self.spin_start.setToolTip("Start time of the selected detection (seconds).\nEdit to adjust the left boundary of the detection box.")
        self.spin_start.valueChanged.connect(self.on_time_changed)
        time_row.addWidget(self.spin_start, 1)

        time_row.addWidget(QLabel("Stop:"))
        self.spin_stop = QDoubleSpinBox()
        self.spin_stop.setDecimals(4)
        self.spin_stop.setRange(0, 9999)
        self.spin_stop.setSuffix(" s")
        self.spin_stop.setToolTip("End time of the selected detection (seconds).\nEdit to adjust the right boundary of the detection box.")
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
        self.spin_det_min_freq.setToolTip("Minimum frequency of the selected detection (Hz).\nEdit to adjust the bottom boundary of the detection box.")
        self.spin_det_min_freq.valueChanged.connect(self.on_freq_changed)
        freq_row.addWidget(self.spin_det_min_freq, 1)

        freq_row.addWidget(QLabel("Max:"))
        self.spin_det_max_freq = QSpinBox()
        self.spin_det_max_freq.setRange(0, 200000)
        self.spin_det_max_freq.setSingleStep(1000)
        self.spin_det_max_freq.setSuffix(" Hz")
        self.spin_det_max_freq.setToolTip("Maximum frequency of the selected detection (Hz).\nEdit to adjust the top boundary of the detection box.")
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
        self.btn_accept.setToolTip("Mark current detection as a valid USV call (A key).\nAccepted calls are saved and used for ML training.")
        self.btn_accept.clicked.connect(self.accept_detection)
        self.btn_accept.setEnabled(False)
        btn_row1.addWidget(self.btn_accept)

        self.btn_reject = QPushButton("Reject")
        self.btn_reject.setObjectName("reject_btn")
        self.btn_reject.setToolTip("Mark current detection as a false positive (R key).\nRejected calls are excluded from analysis but kept for ML training.")
        self.btn_reject.clicked.connect(self.reject_detection)
        self.btn_reject.setEnabled(False)
        btn_row1.addWidget(self.btn_reject)

        self.btn_skip = QPushButton("Skip")
        self.btn_skip.setStyleSheet("background-color: #5c5c5c;")
        self.btn_skip.clicked.connect(self.skip_detection)
        self.btn_skip.setEnabled(False)
        self.btn_skip.setToolTip("Skip to the next detection without changing\nits current status (S key).\nUseful for reviewing without modifying labels.")
        btn_row1.addWidget(self.btn_skip)

        self.btn_undo = QPushButton("Undo")
        self.btn_undo.setStyleSheet("background-color: #5c5c5c;")
        self.btn_undo.clicked.connect(self.undo_action)
        self.btn_undo.setEnabled(False)
        self.btn_undo.setToolTip("Undo last labeling action (Ctrl+Z).\nSupports undoing single and batch operations.")
        btn_row1.addWidget(self.btn_undo)

        group_layout.addLayout(btn_row1)

        # Batch labeling
        batch_row = QHBoxLayout()
        batch_row.setSpacing(2)

        self.btn_accept_all_pending = QPushButton("Accept All Pending")
        self.btn_accept_all_pending.setObjectName("accept_btn")
        self.btn_accept_all_pending.setToolTip("Accept all unreviewed detections at once.\nUseful after reviewing and only rejecting false positives.")
        self.btn_accept_all_pending.clicked.connect(self.accept_all_pending)
        self.btn_accept_all_pending.setEnabled(False)
        batch_row.addWidget(self.btn_accept_all_pending)

        self.btn_reject_all_pending = QPushButton("Reject All Pending")
        self.btn_reject_all_pending.setObjectName("reject_btn")
        self.btn_reject_all_pending.setToolTip("Reject all unreviewed detections at once.\nUseful to clear remaining detections after accepting valid calls.")
        self.btn_reject_all_pending.clicked.connect(self.reject_all_pending)
        self.btn_reject_all_pending.setEnabled(False)
        batch_row.addWidget(self.btn_reject_all_pending)

        group_layout.addLayout(batch_row)

        # Add/Delete
        btn_row2 = QHBoxLayout()
        btn_row2.setSpacing(2)

        self.btn_add_usv = QPushButton("+ Add Label")
        self.btn_add_usv.setStyleSheet("background-color: #6b4c9a;")
        self.btn_add_usv.setToolTip("Manually draw a new USV detection box on the spectrogram.\nClick and drag on the spectrogram to define the region.")
        self.btn_add_usv.clicked.connect(self.add_new_usv)
        self.btn_add_usv.setEnabled(False)
        btn_row2.addWidget(self.btn_add_usv)

        self.btn_delete = QPushButton("Delete")
        self.btn_delete.setStyleSheet("background-color: #5c5c5c;")
        self.btn_delete.setToolTip("Permanently delete the currently selected detection (D key).\nThis cannot be undone.")
        self.btn_delete.clicked.connect(self.delete_current)
        self.btn_delete.setEnabled(False)
        btn_row2.addWidget(self.btn_delete)

        group_layout.addLayout(btn_row2)

        # Delete pending / Delete all labels
        delete_row = QHBoxLayout()
        delete_row.setSpacing(2)

        self.btn_delete_pending = QPushButton("Delete Pending")
        self.btn_delete_pending.setStyleSheet("background-color: #5c5c5c;")
        self.btn_delete_pending.setToolTip("Permanently delete all unreviewed detections.\nUseful for clearing noise after accepting valid calls.")
        self.btn_delete_pending.clicked.connect(self.delete_all_pending)
        self.btn_delete_pending.setEnabled(False)
        delete_row.addWidget(self.btn_delete_pending)

        self.btn_delete_all_labels = QPushButton("Delete All Labels")
        self.btn_delete_all_labels.setStyleSheet("background-color: #8b0000;")
        self.btn_delete_all_labels.setToolTip("Delete ALL detections (pending + accepted + rejected)\nfor the current file. This cannot be undone.")
        self.btn_delete_all_labels.clicked.connect(self.delete_all_labels)
        self.btn_delete_all_labels.setEnabled(False)
        delete_row.addWidget(self.btn_delete_all_labels)

        group_layout.addLayout(delete_row)

        # Statistics label
        self.lbl_stats = QLabel("")
        self.lbl_stats.setStyleSheet("color: #888888; font-size: 9px;")
        self.lbl_stats.setWordWrap(True)
        group_layout.addWidget(self.lbl_stats)

        # Instructions
        lbl_hint = QLabel("Keys: A=Accept R=Reject X=Negative D=Delete Space=Play Ctrl+Z=Undo")
        lbl_hint.setStyleSheet("color: #666666; font-size: 9px; font-style: italic;")
        lbl_hint.setWordWrap(True)
        group_layout.addWidget(lbl_hint)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _create_ml_section(self, layout):
        """Create YOLO-based ML detection section."""
        group = QGroupBox("5. ML Detection (YOLO)")
        group.setToolTip(
            "Train and run a YOLO neural network for USV detection.\n"
            "Label examples with Accept (A) and Negative (X),\n"
            "then train a model to detect calls automatically."
        )
        group_layout = QVBoxLayout()
        group_layout.setSpacing(4)

        # Project row
        project_row = QHBoxLayout()
        project_row.setSpacing(2)
        self.btn_ml_project = QPushButton("Open/Create Project")
        self.btn_ml_project.setToolTip(
            "Open an existing YOLO project or create a new one.\n"
            "Projects store training data, models, and config."
        )
        self.btn_ml_project.clicked.connect(self._ml_open_project)
        project_row.addWidget(self.btn_ml_project)
        group_layout.addLayout(project_row)

        self.lbl_ml_project = QLabel("No project")
        self.lbl_ml_project.setStyleSheet("color: #999999; font-size: 9px;")
        self.lbl_ml_project.setWordWrap(True)
        group_layout.addWidget(self.lbl_ml_project)

        # Training data stats
        self.lbl_ml_data = QLabel("Labels: 0 positive, 0 negative")
        self.lbl_ml_data.setStyleSheet("color: #999999;")
        group_layout.addWidget(self.lbl_ml_data)

        # Train button
        self.btn_ml_train = QPushButton("Export && Train")
        self.btn_ml_train.setStyleSheet("background-color: #2d7d46;")
        self.btn_ml_train.setToolTip(
            "Export labeled data as spectrogram tiles and train\n"
            "a YOLOv8-nano model. Requires accepted + negative labels.\n"
            "Training stops automatically when loss plateaus."
        )
        self.btn_ml_train.clicked.connect(self._ml_train)
        self.btn_ml_train.setEnabled(False)
        group_layout.addWidget(self.btn_ml_train)

        # Progress
        self.ml_progress = QProgressBar()
        self.ml_progress.setValue(0)
        self.ml_progress.setVisible(False)
        group_layout.addWidget(self.ml_progress)

        self.lbl_ml_status = QLabel("")
        self.lbl_ml_status.setStyleSheet("color: #999999; font-size: 9px;")
        group_layout.addWidget(self.lbl_ml_status)

        # Current model info
        self.lbl_ml_model = QLabel("No trained model")
        self.lbl_ml_model.setStyleSheet("color: #aaaaaa; font-size: 9px;")
        self.lbl_ml_model.setWordWrap(True)
        group_layout.addWidget(self.lbl_ml_model)

        # Run ML Detection buttons
        run_row = QHBoxLayout()
        run_row.setSpacing(2)

        self.btn_ml_detect_current = QPushButton("Detect Current")
        self.btn_ml_detect_current.setToolTip(
            "Run YOLO detection on the current file.\n"
            "Results appear as pending detections."
        )
        self.btn_ml_detect_current.clicked.connect(self._ml_detect_current)
        self.btn_ml_detect_current.setEnabled(False)
        run_row.addWidget(self.btn_ml_detect_current)

        self.btn_ml_detect_all = QPushButton("Detect All")
        self.btn_ml_detect_all.setToolTip(
            "Run YOLO detection on all files in the file list.\n"
            "Batch processes with progress reporting."
        )
        self.btn_ml_detect_all.clicked.connect(self._ml_detect_all)
        self.btn_ml_detect_all.setEnabled(False)
        run_row.addWidget(self.btn_ml_detect_all)

        group_layout.addLayout(run_row)

        group.setLayout(group_layout)
        layout.addWidget(group)

        # Initialize YOLO project state
        self._yolo_project_config = None
        self._yolo_model_path = None

    def _on_gpu_toggle(self, checked):
        """Handle GPU acceleration checkbox toggle."""
        if checked:
            self._show_gpu_detection_dialog()
        else:
            self.lbl_gpu_status.setText("")
            self._selected_gpu_device = "auto"

    def _show_gpu_detection_dialog(self):
        """Show dialog listing detected GPU devices."""
        try:
            from fnt.usv.usv_detector.gpu_utils import detect_available_devices
        except ImportError:
            QMessageBox.warning(self, "PyTorch Not Available",
                "PyTorch is required for GPU acceleration.\n\n"
                "Install with: pip install torch")
            self.chk_gpu_accel.blockSignals(True)
            self.chk_gpu_accel.setChecked(False)
            self.chk_gpu_accel.blockSignals(False)
            return

        devices = detect_available_devices()
        gpu_devices = [d for d in devices if d['type'] != 'cpu']

        if not gpu_devices:
            QMessageBox.warning(self, "No GPU Found",
                "No compatible GPU detected.\n\n"
                "Requires:\n"
                "  \u2022 NVIDIA GPU with CUDA support, or\n"
                "  \u2022 Apple Silicon with MPS support\n\n"
                "Processing will use CPU.")
            self.chk_gpu_accel.blockSignals(True)
            self.chk_gpu_accel.setChecked(False)
            self.chk_gpu_accel.blockSignals(False)
            return

        dialog = QDialog(self)
        dialog.setWindowTitle("GPU Acceleration")
        dialog.setMinimumWidth(420)
        layout = QVBoxLayout()

        layout.addWidget(QLabel("Detected compute devices:"))
        layout.addSpacing(4)

        for dev in devices:
            parts = []
            if dev['type'] != 'cpu':
                parts.append("\u2705")  # Green checkmark
            else:
                parts.append("   ")
            parts.append(f"{dev['name']}  ({dev['type'].upper()})")
            if dev.get('vram_mb'):
                parts.append(f" \u2014 {dev['vram_mb']:,} MB VRAM")
            lbl = QLabel("".join(parts))
            if dev['type'] != 'cpu':
                lbl.setStyleSheet("color: #4CAF50; font-weight: bold;")
            else:
                lbl.setStyleSheet("color: #999999;")
            layout.addWidget(lbl)

        layout.addSpacing(8)

        # Device selection
        sel_row = QHBoxLayout()
        sel_row.addWidget(QLabel("Use device:"))
        combo = QComboBox()
        combo.addItem("Auto (best available)", "auto")
        for dev in gpu_devices:
            label = f"{dev['name']} ({dev['type'].upper()})"
            if dev.get('vram_mb'):
                label += f" \u2014 {dev['vram_mb']:,} MB"
            combo.addItem(label, dev['device'])
        sel_row.addWidget(combo, 1)
        layout.addLayout(sel_row)

        layout.addSpacing(8)

        # OK / Cancel
        btn_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btn_box.accepted.connect(dialog.accept)
        btn_box.rejected.connect(dialog.reject)
        layout.addWidget(btn_box)

        dialog.setLayout(layout)

        if dialog.exec_() == QDialog.Accepted:
            self._selected_gpu_device = combo.currentData()
            dev_name = combo.currentText()
            self.lbl_gpu_status.setText(f"\u26a1 {dev_name}")
            self.lbl_gpu_status.setStyleSheet("color: #4CAF50; font-size: 9px;")
        else:
            # User cancelled — uncheck
            self.chk_gpu_accel.blockSignals(True)
            self.chk_gpu_accel.setChecked(False)
            self.chk_gpu_accel.blockSignals(False)
            self.lbl_gpu_status.setText("")

    def _toggle_advanced_options(self):
        """Toggle visibility of advanced DSP options."""
        visible = not self.advanced_options_widget.isVisible()
        self.advanced_options_widget.setVisible(visible)
        arrow = "▼" if visible else "▶"
        self.btn_advanced_toggle.setText(f"{arrow} Advanced Options")

    def _on_species_profile_changed(self, profile_name):
        """Handle species profile dropdown change."""
        preset = self.SPECIES_PROFILES.get(profile_name)

        if preset is None:
            # Manual mode — re-enable all DSP parameter widgets
            for w in self._dsp_param_widgets:
                w.setEnabled(True)
                w.setStyleSheet("")  # Clear locked styling
            # Respect freq_samples checkbox state
            self.spin_freq_samples.setEnabled(self.chk_freq_samples.isChecked())
        else:
            # Block signals on all spinboxes to avoid cascading updates
            spinboxes = [
                self.spin_min_freq, self.spin_max_freq, self.spin_threshold,
                self.spin_min_dur, self.spin_max_dur,
                self.spin_max_bw, self.spin_tonality, self.spin_min_call_freq,
                self.spin_freq_gap,
                self.spin_min_gap, self.spin_noise_pct,
                self.spin_nperseg, self.spin_noverlap,
            ]
            for s in spinboxes:
                s.blockSignals(True)

            self.spin_min_freq.setValue(preset['min_freq_hz'])
            self.spin_max_freq.setValue(preset['max_freq_hz'])
            self.spin_threshold.setValue(preset['energy_threshold_db'])
            self.spin_min_dur.setValue(preset['min_duration_ms'])
            self.spin_max_dur.setValue(preset['max_duration_ms'])
            self.spin_max_bw.setValue(preset['max_bandwidth_hz'])
            self.spin_tonality.setValue(preset['min_tonality'])
            self.spin_min_call_freq.setValue(preset['min_call_freq_hz'])
            self.spin_min_gap.setValue(preset['min_gap_ms'])
            self.spin_noise_pct.setValue(preset['noise_percentile'])
            self.spin_nperseg.setValue(preset['nperseg'])
            self.spin_noverlap.setValue(preset['noverlap'])
            self.chk_harmonic_filter.setChecked(preset.get('harmonic_filter', True))
            self.spin_freq_gap.setValue(preset.get('min_freq_gap_hz', 5000))

            for s in spinboxes:
                s.blockSignals(False)

            # Disable all DSP parameter widgets with strong visual indicator
            locked_style = "background-color: #1a1a1a; color: #555555;"
            for w in self._dsp_param_widgets:
                w.setEnabled(False)
                w.setStyleSheet(locked_style)

    def _setup_shortcuts(self):
        """Set up global keyboard shortcuts using QShortcut."""
        # Detection navigation: B = back, N = next
        sc_next = QShortcut(QKeySequence(Qt.Key_N), self)
        sc_next.setContext(Qt.ApplicationShortcut)
        sc_next.activated.connect(self._shortcut_next_detection)

        sc_prev = QShortcut(QKeySequence(Qt.Key_B), self)
        sc_prev.setContext(Qt.ApplicationShortcut)
        sc_prev.activated.connect(self._shortcut_prev_detection)

        # Spectrogram panning: Left/Right arrow keys
        sc_pan_right = QShortcut(QKeySequence(Qt.Key_Right), self)
        sc_pan_right.setContext(Qt.ApplicationShortcut)
        sc_pan_right.activated.connect(self._shortcut_pan_right)

        sc_pan_left = QShortcut(QKeySequence(Qt.Key_Left), self)
        sc_pan_left.setContext(Qt.ApplicationShortcut)
        sc_pan_left.activated.connect(self._shortcut_pan_left)

        # Window size: Up = zoom in (smaller window), Down = zoom out (larger window)
        sc_zoom_in = QShortcut(QKeySequence(Qt.Key_Up), self)
        sc_zoom_in.setContext(Qt.ApplicationShortcut)
        sc_zoom_in.activated.connect(self._shortcut_zoom_in)

        sc_zoom_out = QShortcut(QKeySequence(Qt.Key_Down), self)
        sc_zoom_out.setContext(Qt.ApplicationShortcut)
        sc_zoom_out.activated.connect(self._shortcut_zoom_out)

        # P = add new pending USV detection
        sc_add_usv = QShortcut(QKeySequence(Qt.Key_P), self)
        sc_add_usv.setContext(Qt.ApplicationShortcut)
        sc_add_usv.activated.connect(self._shortcut_add_usv)

        # S = skip detection (advance without changing status)
        sc_skip = QShortcut(QKeySequence(Qt.Key_S), self)
        sc_skip.setContext(Qt.ApplicationShortcut)
        sc_skip.activated.connect(self._shortcut_skip)

    def _shortcut_next_detection(self):
        """Handle N key shortcut — next detection."""
        focus = QApplication.focusWidget()
        if isinstance(focus, (QSpinBox, QDoubleSpinBox, QComboBox)):
            return
        if self.btn_next_det.isEnabled():
            self._flash_button(self.btn_next_det)
            self.next_detection()

    def _shortcut_prev_detection(self):
        """Handle B key shortcut — previous detection."""
        focus = QApplication.focusWidget()
        if isinstance(focus, (QSpinBox, QDoubleSpinBox, QComboBox)):
            return
        if self.btn_prev_det.isEnabled():
            self._flash_button(self.btn_prev_det)
            self.prev_detection()

    def _shortcut_pan_right(self):
        """Handle Right arrow — pan spectrogram right."""
        focus = QApplication.focusWidget()
        if isinstance(focus, (QSpinBox, QDoubleSpinBox, QComboBox)):
            return
        self.pan_right()

    def _shortcut_pan_left(self):
        """Handle Left arrow — pan spectrogram left."""
        focus = QApplication.focusWidget()
        if isinstance(focus, (QSpinBox, QDoubleSpinBox, QComboBox)):
            return
        self.pan_left()

    def _shortcut_zoom_out(self):
        """Handle Up arrow — increase window size (zoom out)."""
        focus = QApplication.focusWidget()
        if isinstance(focus, (QSpinBox, QDoubleSpinBox, QComboBox)):
            return
        self.zoom_out()

    def _shortcut_zoom_in(self):
        """Handle Down arrow — decrease window size (zoom in)."""
        focus = QApplication.focusWidget()
        if isinstance(focus, (QSpinBox, QDoubleSpinBox, QComboBox)):
            return
        self.zoom_in()

    def _shortcut_add_usv(self):
        """Handle P key — add new pending USV detection."""
        focus = QApplication.focusWidget()
        if isinstance(focus, (QSpinBox, QDoubleSpinBox, QComboBox)):
            return
        if self.btn_add_usv.isEnabled():
            self._flash_button(self.btn_add_usv)
            self.add_new_usv()

    def _shortcut_skip(self):
        """Handle S key — skip to next detection without changing status."""
        focus = QApplication.focusWidget()
        if isinstance(focus, (QSpinBox, QDoubleSpinBox, QComboBox)):
            return
        if self.btn_skip.isEnabled():
            self._flash_button(self.btn_skip)
            self.skip_detection()

    def _setup_pan_timers(self):
        """Set up press-and-hold repeat timers for pan buttons."""
        self._pan_left_timer = QTimer(self)
        self._pan_left_timer.setInterval(100)  # Repeat every 100ms while held
        self._pan_left_timer.timeout.connect(self.pan_left)

        self._pan_right_timer = QTimer(self)
        self._pan_right_timer.setInterval(100)
        self._pan_right_timer.timeout.connect(self.pan_right)

        # Connect press/release events
        self.btn_pan_left.pressed.connect(self._on_pan_left_pressed)
        self.btn_pan_left.released.connect(self._pan_left_timer.stop)
        self.btn_pan_right.pressed.connect(self._on_pan_right_pressed)
        self.btn_pan_right.released.connect(self._pan_right_timer.stop)

        # Disconnect the old clicked signals to avoid double-firing
        self.btn_pan_left.clicked.disconnect(self.pan_left)
        self.btn_pan_right.clicked.disconnect(self.pan_right)

    def _on_pan_left_pressed(self):
        """Handle pan left button press - immediate pan + start repeat timer."""
        self.pan_left()
        self._pan_left_timer.start()

    def _on_pan_right_pressed(self):
        """Handle pan right button press - immediate pan + start repeat timer."""
        self.pan_right()
        self._pan_right_timer.start()

    def _create_arrow_images(self):
        """Create small arrow PNG images for spinbox/combobox buttons."""
        import tempfile
        self._arrow_dir = tempfile.mkdtemp(prefix='usv_arrows_')

        # Up arrow (white triangle on transparent bg)
        up_img = QImage(9, 6, QImage.Format_ARGB32)
        up_img.fill(QColor(0, 0, 0, 0))
        painter = QPainter(up_img)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor('#cccccc'))
        painter.drawPolygon(QPolygonF([
            QPointF(4.5, 0.5), QPointF(8.5, 5.5), QPointF(0.5, 5.5)
        ]))
        painter.end()
        self._up_arrow_path = os.path.join(self._arrow_dir, 'up.png')
        up_img.save(self._up_arrow_path)

        # Down arrow
        down_img = QImage(9, 6, QImage.Format_ARGB32)
        down_img.fill(QColor(0, 0, 0, 0))
        painter = QPainter(down_img)
        painter.setRenderHint(QPainter.Antialiasing)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QColor('#cccccc'))
        painter.drawPolygon(QPolygonF([
            QPointF(4.5, 5.5), QPointF(0.5, 0.5), QPointF(8.5, 0.5)
        ]))
        painter.end()
        self._down_arrow_path = os.path.join(self._arrow_dir, 'down.png')
        down_img.save(self._down_arrow_path)

    def _apply_styles(self):
        """Apply dark theme styles."""
        self._create_arrow_images()
        up = self._up_arrow_path.replace('\\', '/')
        down = self._down_arrow_path.replace('\\', '/')
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
            QSpinBox::up-button, QDoubleSpinBox::up-button {
                subcontrol-origin: border;
                subcontrol-position: top right;
                width: 16px;
                border-left: 1px solid #555555;
                border-bottom: 1px solid #555555;
                border-top-right-radius: 3px;
                background-color: #404040;
            }
            QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover {
                background-color: #505050;
            }
            QSpinBox::up-button:pressed, QDoubleSpinBox::up-button:pressed {
                background-color: #606060;
            }
            QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {
                image: url(UP_ARROW_PATH);
                width: 9px;
                height: 6px;
            }
            QSpinBox::down-button, QDoubleSpinBox::down-button {
                subcontrol-origin: border;
                subcontrol-position: bottom right;
                width: 16px;
                border-left: 1px solid #555555;
                border-top: 1px solid #555555;
                border-bottom-right-radius: 3px;
                background-color: #404040;
            }
            QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {
                background-color: #505050;
            }
            QSpinBox::down-button:pressed, QDoubleSpinBox::down-button:pressed {
                background-color: #606060;
            }
            QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {
                image: url(DOWN_ARROW_PATH);
                width: 9px;
                height: 6px;
            }
            QComboBox::drop-down {
                subcontrol-origin: padding;
                subcontrol-position: center right;
                width: 18px;
                border-left: 1px solid #555555;
                border-top-right-radius: 3px;
                border-bottom-right-radius: 3px;
                background-color: #404040;
            }
            QComboBox::drop-down:hover {
                background-color: #505050;
            }
            QComboBox::down-arrow {
                image: url(DOWN_ARROW_PATH);
                width: 9px;
                height: 6px;
            }
            QComboBox QAbstractItemView {
                background-color: #1e1e1e;
                color: #cccccc;
                selection-background-color: #0078d4;
                border: 1px solid #3f3f3f;
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
                background-color: #1e1e1e;
                width: 12px;
                border: 1px solid #3f3f3f;
                border-radius: 3px;
            }
            QScrollBar::handle:vertical {
                background-color: #0078d4;
                border-radius: 2px;
                min-height: 20px;
                margin: 1px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #106ebe;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background-color: #1e1e1e;
            }
            QScrollBar:horizontal {
                background-color: #1e1e1e;
                height: 14px;
                border: 1px solid #3f3f3f;
                border-radius: 3px;
            }
            QScrollBar::handle:horizontal {
                background-color: #0078d4;
                border-radius: 2px;
                min-width: 20px;
                margin: 1px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #106ebe;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background-color: #1e1e1e;
            }
            QCheckBox {
                spacing: 5px;
            }
            QCheckBox::indicator {
                width: 14px;
                height: 14px;
            }
        """.replace('UP_ARROW_PATH', up).replace('DOWN_ARROW_PATH', down))

    # =========================================================================
    # File Management
    # =========================================================================

    def add_folder(self):
        """Add all WAV files from a folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return

        wav_files = sorted(list(Path(folder).glob("*.wav")) + list(Path(folder).glob("*.WAV")))
        if not wav_files:
            QMessageBox.warning(self, "No Files", "No WAV files found in folder.")
            return

        self._add_audio_files([str(f) for f in wav_files])

    def add_files(self):
        """Add individual WAV files."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio Files", "",
            "WAV Files (*.wav *.WAV);;All Files (*.*)"
        )
        if not files:
            return

        self._add_audio_files(files)

    def _add_audio_files(self, new_files):
        """Add files to the import list, skipping duplicates with a warning."""
        duplicates = [f for f in new_files if f in self.audio_files]
        to_add = [f for f in new_files if f not in self.audio_files]

        if duplicates and not to_add:
            QMessageBox.information(
                self, "Already Imported",
                f"All {len(duplicates)} file{'s' if len(duplicates) != 1 else ''} "
                "are already imported. No new files added.")
            return

        if duplicates:
            self.status_bar.showMessage(
                f"Added {len(to_add)} new file{'s' if len(to_add) != 1 else ''}, "
                f"skipped {len(duplicates)} already imported")
        else:
            self.status_bar.showMessage(
                f"Added {len(to_add)} file{'s' if len(to_add) != 1 else ''}")

        had_files = len(self.audio_files) > 0
        for f in to_add:
            self.audio_files.append(f)

        # Scan new files for existing CSV detections
        self._scan_existing_csvs()

        if had_files:
            # Already have files loaded — just refresh the list display
            self._refresh_file_list_items_full()
        else:
            # First files being added — do full list build + load first file
            self._update_file_list()

    def _scan_existing_csvs(self):
        """Pre-scan for existing CSV detection files to show counts in file list."""
        for filepath in self.audio_files:
            if filepath in self.all_detections:
                continue
            base = Path(filepath).stem
            parent = Path(filepath).parent
            for suffix in ['_FNT_CAD_detections', '_FNT_DAD_detections', '_cad', '_dad', '_usv_dsp', '_usv_rf', '_usv_yolo', '_usv_detections']:
                csv_path = parent / f"{base}{suffix}.csv"
                if csv_path.exists():
                    try:
                        df = pd.read_csv(csv_path)
                        if 'status' not in df.columns:
                            df['status'] = 'pending'
                        self._ensure_freq_bounds(df)
                        self.all_detections[filepath] = df
                        self.detection_sources[filepath] = suffix.lstrip('_')
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
        self.detection_sources = {}
        self.dsp_queue = []
        self.undo_stack.clear()
        self._update_file_list()
        self.spectrogram.set_audio_data(None, None)
        self.spectrogram.set_detections([], -1)
        self.waveform_overview.set_audio_data(None, None)
        self._update_ui_state()

    def _update_file_list(self):
        """Update file list display. Blocks signals to prevent
        _store_current_detections from overwriting freshly loaded data."""
        self.file_list.blockSignals(True)
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
        self.btn_add_all_to_queue.setEnabled(n > 0)

        if n > 0 and self.current_file_idx < n:
            self.file_list.setCurrentRow(self.current_file_idx)
        self.file_list.blockSignals(False)

        if n > 0 and self.current_file_idx < n:
            self._load_current_file()

        self._update_ui_state()

    def _refresh_file_list_items_full(self):
        """Rebuild file list widget without reloading the current file.

        Used when adding more files to an already-populated list.
        """
        self.file_list.blockSignals(True)
        self.file_list.clear()

        for filepath in self.audio_files:
            filename = os.path.basename(filepath)
            parts = []
            if filepath in self.dsp_queue:
                parts.append("\u2713")
            parts.append(filename)
            if filepath in self.all_detections:
                n_det = len(self.all_detections[filepath])
                parts.append(f"({n_det})")
            item = QListWidgetItem(" ".join(parts))
            item.setData(Qt.UserRole, filepath)
            self.file_list.addItem(item)

        n = len(self.audio_files)
        self.lbl_file_num.setText(f"File {self.current_file_idx + 1 if n > 0 else 0}/{n}")
        self.btn_prev_file.setEnabled(self.current_file_idx > 0)
        self.btn_next_file.setEnabled(self.current_file_idx < n - 1)
        self.btn_add_to_queue.setEnabled(n > 0)
        self.btn_add_all_to_queue.setEnabled(n > 0)

        if n > 0 and self.current_file_idx < n:
            self.file_list.setCurrentRow(self.current_file_idx)
        self.file_list.blockSignals(False)

        self._update_file_navigation()

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
            # Only change list selection — on_file_selected will handle
            # storing old detections and updating current_file_idx
            self.file_list.setCurrentRow(self.current_file_idx - 1)

    def next_file(self):
        """Go to next file."""
        if self.current_file_idx < len(self.audio_files) - 1:
            # Only change list selection — on_file_selected will handle
            # storing old detections and updating current_file_idx
            self.file_list.setCurrentRow(self.current_file_idx + 1)

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
        """Load the current audio file with progress indication."""
        if not self.audio_files or self.current_file_idx >= len(self.audio_files):
            return

        filepath = self.audio_files[self.current_file_idx]
        self.status_bar.showMessage(f"Loading {os.path.basename(filepath)}...")
        # Show a busy cursor during load
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()

        try:
            # Load audio
            if HAS_SOUNDFILE:
                try:
                    self.audio_data, self.sample_rate = sf.read(filepath, dtype='float32')
                except Exception:
                    self.audio_data, self.sample_rate = self._load_with_ffmpeg(filepath)
            else:
                self.audio_data, self.sample_rate = self._load_with_ffmpeg(filepath)

            if self.audio_data.ndim > 1:
                self.audio_data = np.mean(self.audio_data, axis=1)

            # Preserve view settings if we already had a file loaded
            has_previous = self.spectrogram.total_duration > 0
            self.spectrogram.set_audio_data(self.audio_data, self.sample_rate,
                                            preserve_view=has_previous)
            self.waveform_overview.set_audio_data(self.audio_data, self.sample_rate)

            # Sync freq spinners with spectrogram's (possibly preserved) values
            self.spin_display_min_freq.blockSignals(True)
            self.spin_display_max_freq.blockSignals(True)
            self.spin_display_min_freq.setValue(self.spectrogram.min_freq)
            self.spin_display_max_freq.setValue(self.spectrogram.max_freq)
            self.spin_display_min_freq.blockSignals(False)
            self.spin_display_max_freq.blockSignals(False)

            # Sync the window spinner with the preserved view
            if has_previous:
                view_start, view_end = self.spectrogram.get_view_range()
                self.spin_view_window.blockSignals(True)
                self.spin_view_window.setValue(view_end - view_start)
                self.spin_view_window.blockSignals(False)

            # Load detections if available
            if filepath in self.all_detections:
                self.detections_df = self.all_detections[filepath].copy()
                # Ensure legacy data has freq bounds
                self._ensure_freq_bounds(self.detections_df)
                self.current_detection_idx = 0
            else:
                # Try to load from CSV
                self._try_load_detections(filepath)

            # Clear undo stack on file switch
            self.undo_stack.clear()
            self.btn_undo.setEnabled(False)

            self._update_display()
            self.status_bar.showMessage(f"Loaded: {os.path.basename(filepath)}")

        except Exception as e:
            self.status_bar.showMessage(f"Error loading file: {e}")
            self.audio_data = None
        finally:
            QApplication.restoreOverrideCursor()

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
        for suffix in ['_FNT_CAD_detections', '_FNT_DAD_detections', '_cad', '_dad', '_usv_dsp', '_usv_rf', '_usv_yolo', '_usv_detections']:
            csv_path = parent / f"{base}{suffix}.csv"
            if csv_path.exists():
                try:
                    self.detections_df = pd.read_csv(csv_path)
                    if 'status' not in self.detections_df.columns:
                        self.detections_df['status'] = 'pending'
                    # Handle legacy CSVs missing min_freq_hz/max_freq_hz
                    self._ensure_freq_bounds(self.detections_df)
                    self.all_detections[filepath] = self.detections_df.copy()
                    # Track source suffix for correct auto-save naming
                    self.detection_sources[filepath] = suffix.lstrip('_')
                    self.current_detection_idx = 0
                    return
                except Exception:
                    pass

        self.detections_df = None
        self.current_detection_idx = 0

    @staticmethod
    def _ensure_freq_bounds(df):
        """Ensure min_freq_hz and max_freq_hz columns exist. Compute from
        mean_freq_hz and freq_bandwidth_hz if available, else use defaults."""
        if 'min_freq_hz' not in df.columns or df['min_freq_hz'].isna().all():
            if 'mean_freq_hz' in df.columns and 'freq_bandwidth_hz' in df.columns:
                df['min_freq_hz'] = df['mean_freq_hz'] - df['freq_bandwidth_hz'] / 2
                df['min_freq_hz'] = df['min_freq_hz'].clip(lower=0)
            else:
                df['min_freq_hz'] = 25000
        if 'max_freq_hz' not in df.columns or df['max_freq_hz'].isna().all():
            if 'mean_freq_hz' in df.columns and 'freq_bandwidth_hz' in df.columns:
                df['max_freq_hz'] = df['mean_freq_hz'] + df['freq_bandwidth_hz'] / 2
            else:
                df['max_freq_hz'] = 65000
        # Fill any remaining NaNs
        df['min_freq_hz'] = df['min_freq_hz'].fillna(25000)
        df['max_freq_hz'] = df['max_freq_hz'].fillna(65000)

    # =========================================================================
    # DSP Detection
    # =========================================================================

    def _file_has_detections(self, filepath):
        """Check if a file has existing detections (in memory or on disk)."""
        if filepath in self.all_detections and len(self.all_detections[filepath]) > 0:
            return True
        # Check for CSV on disk
        base = Path(filepath).stem
        parent = Path(filepath).parent
        for suffix in ['_FNT_CAD_detections', '_FNT_DAD_detections', '_cad', '_dad', '_usv_dsp', '_usv_rf', '_usv_yolo', '_usv_detections']:
            if (parent / f"{base}{suffix}.csv").exists():
                return True
        return False

    def add_to_queue(self):
        """Add current file to DSP queue."""
        if not self.audio_files or self.current_file_idx >= len(self.audio_files):
            return

        filepath = self.audio_files[self.current_file_idx]
        if filepath not in self.dsp_queue:
            self.dsp_queue.append(filepath)
            self._update_queue_display()
            self.status_bar.showMessage(f"Added 1 file to queue")

    def add_all_to_queue(self):
        """Add all imported files to DSP queue."""
        if not self.audio_files:
            return

        files_to_add = [f for f in self.audio_files if f not in self.dsp_queue]
        if not files_to_add:
            self.status_bar.showMessage("All files already in queue")
            return

        for filepath in files_to_add:
            self.dsp_queue.append(filepath)

        self._update_queue_display()
        self.status_bar.showMessage(f"Added {len(files_to_add)} file{'s' if len(files_to_add) != 1 else ''} to queue")

    def clear_queue(self):
        """Clear DSP queue."""
        self.dsp_queue = []
        self._update_queue_display()

    def _update_queue_display(self):
        """Update queue display and file list checkmarks."""
        n = len(self.dsp_queue)
        self.lbl_queue.setText(f"Queue: {n} file{'s' if n != 1 else ''}")
        self.btn_run_dsp.setEnabled(n > 0)
        if n > 0:
            self.btn_run_dsp.setText(f"Run DSP Detection ({n} file{'s' if n != 1 else ''})")
        else:
            self.btn_run_dsp.setText("Run DSP Detection")
        # Update file list to show/hide checkmarks
        self._refresh_file_list_items()

    def run_dsp_detection(self):
        """Run DSP detection on queued files."""
        if not self.dsp_queue:
            return

        # Check which queued files already have detections
        files_with_dets = [f for f in self.dsp_queue if self._file_has_detections(f)]
        files_without_dets = [f for f in self.dsp_queue if f not in files_with_dets]
        n_with = len(files_with_dets)
        n_without = len(files_without_dets)
        n_total = len(self.dsp_queue)

        if n_with > 0:
            msg = (f"Queue: {n_total} file{'s' if n_total != 1 else ''}\n"
                   f"  - {n_with} file{'s' if n_with != 1 else ''} with existing detections\n"
                   f"  - {n_without} file{'s' if n_without != 1 else ''} without detections\n\n"
                   "How would you like to proceed?")
            box = QMessageBox(self)
            box.setWindowTitle("Existing Detections Found")
            box.setText(msg)
            btn_overwrite = box.addButton("Overwrite All", QMessageBox.AcceptRole)
            btn_skip = box.addButton("Skip Existing", QMessageBox.ActionRole)
            box.addButton("Cancel", QMessageBox.RejectRole)
            box.exec_()

            clicked = box.clickedButton()
            if clicked == btn_skip:
                if not files_without_dets:
                    self.status_bar.showMessage("All queued files already have detections — nothing to process")
                    return
                self.dsp_queue = files_without_dets
            elif clicked == btn_overwrite:
                pass  # Keep full queue
            else:
                return  # Cancel

        # Gather config
        config = {
            'min_freq_hz': self.spin_min_freq.value(),
            'max_freq_hz': self.spin_max_freq.value(),
            'energy_threshold_db': self.spin_threshold.value(),
            'min_duration_ms': self.spin_min_dur.value(),
            'max_duration_ms': self.spin_max_dur.value(),
            'max_bandwidth_hz': self.spin_max_bw.value(),
            'min_tonality': self.spin_tonality.value(),
            'min_call_freq_hz': self.spin_min_call_freq.value(),
            'harmonic_filter': self.chk_harmonic_filter.isChecked(),
            'min_freq_gap_hz': self.spin_freq_gap.value(),
            'min_gap_ms': self.spin_min_gap.value(),
            'noise_percentile': self.spin_noise_pct.value(),
            'nperseg': self.spin_nperseg.value(),
            'noverlap': self.spin_noverlap.value(),
            'freq_samples': self.spin_freq_samples.value() if self.chk_freq_samples.isChecked() else 0,
            'gpu_enabled': self.chk_gpu_accel.isChecked(),
            'gpu_device': getattr(self, '_selected_gpu_device', 'auto'),
        }

        # Start worker
        self.dsp_worker = DSPDetectionWorker(self.dsp_queue.copy(), config)
        self.dsp_worker.progress.connect(self._on_dsp_progress)
        self.dsp_worker.file_progress.connect(self._on_dsp_file_progress)
        self.dsp_worker.file_complete.connect(self._on_dsp_file_complete)
        self.dsp_worker.all_complete.connect(self._on_dsp_complete)
        self.dsp_worker.error.connect(self._on_dsp_error)

        self.dsp_progress.setVisible(True)
        self.dsp_progress.setValue(0)
        self.dsp_file_progress.setVisible(True)
        self.dsp_file_progress.setValue(0)
        self.btn_run_dsp.setEnabled(False)
        self.btn_run_dsp.setText("Running...")
        self.btn_stop_dsp.setVisible(True)

        self.dsp_worker.start()

    def stop_dsp_detection(self):
        """Stop DSP detection. The file currently being processed is discarded."""
        if hasattr(self, 'dsp_worker') and self.dsp_worker is not None:
            self.dsp_worker.stop()
            self.lbl_dsp_status.setText("Stopping after current file...")
            self.btn_stop_dsp.setEnabled(False)

    def _on_dsp_progress(self, filename, current, total):
        """Handle DSP batch progress update (file-level)."""
        self.dsp_progress.setValue(int(current / total * 100))
        self.lbl_dsp_status.setText(f"Processing ({current + 1}/{total}): {filename}")
        self.dsp_file_progress.setValue(0)

    def _on_dsp_file_progress(self, fraction):
        """Handle per-file progress update (chunk-level within a file)."""
        self.dsp_file_progress.setValue(int(fraction * 100))

    def _on_dsp_file_complete(self, filename, filepath, detections, n_detections):
        """Handle DSP file completion — write CSV immediately."""
        self.dsp_file_progress.setValue(100)
        self.lbl_dsp_status.setText(f"{filename}: {n_detections} detections")

        # Standard columns for detection CSVs
        std_columns = ['call_number', 'start_seconds', 'stop_seconds', 'duration_ms',
                        'min_freq_hz', 'max_freq_hz', 'peak_freq_hz', 'freq_bandwidth_hz',
                        'max_power_db', 'mean_power_db', 'status', 'source']

        # Build DataFrame from detection list
        if isinstance(detections, pd.DataFrame):
            df = detections.copy()
        else:
            df = pd.DataFrame(detections)

        if len(df) > 0:
            if 'status' not in df.columns:
                df['status'] = 'pending'
            if 'source' not in df.columns:
                df['source'] = 'dsp'
            if 'call_number' not in df.columns:
                df.insert(0, 'call_number', range(1, len(df) + 1))
        else:
            df = pd.DataFrame(columns=std_columns)

        # Store in memory
        self.all_detections[filepath] = df
        self.detection_sources[filepath] = 'usv_dsp'

        # Write CSV immediately (crash-safe — results are on disk per file)
        base = Path(filepath).stem
        parent = Path(filepath).parent
        csv_path = parent / f"{base}_FNT_CAD_detections.csv"
        try:
            df.to_csv(csv_path, index=False)
        except Exception as e:
            self.status_bar.showMessage(f"Error saving CSV for {filename}: {e}")

        # Update file list item to show detection count
        if filepath in self.audio_files:
            idx = self.audio_files.index(filepath)
            item = self.file_list.item(idx)
            if item:
                parts = []
                if filepath in self.dsp_queue:
                    parts.append("\u2713")
                parts.append(filename)
                parts.append(f"({len(df)})")
                self.file_list.blockSignals(True)
                item.setText(" ".join(parts))
                self.file_list.blockSignals(False)

        # If this is the currently displayed file, update the view
        if (self.audio_files and self.current_file_idx < len(self.audio_files)
                and self.audio_files[self.current_file_idx] == filepath):
            self.detections_df = df.copy()
            self._ensure_freq_bounds(self.detections_df)
            self.current_detection_idx = 0
            self._load_current_file()

    def _on_dsp_complete(self, results):
        """Handle DSP batch completion. CSVs already written per-file."""
        was_stopped = hasattr(self, 'dsp_worker') and self.dsp_worker._stop_requested

        # Update UI
        self.dsp_progress.setValue(100)
        self.dsp_progress.setVisible(False)
        self.dsp_file_progress.setVisible(False)
        self.btn_stop_dsp.setVisible(False)
        self.btn_stop_dsp.setEnabled(True)
        self.btn_run_dsp.setEnabled(True)
        self.btn_run_dsp.setText("Run DSP Detection")

        total_det = sum(len(d) for d in results.values())
        if was_stopped:
            self.lbl_dsp_status.setText(f"Stopped. {len(results)} file{'s' if len(results) != 1 else ''} completed, {total_det} detections")
            self.status_bar.showMessage(f"DSP detection stopped. {len(results)} files completed.")
        else:
            self.lbl_dsp_status.setText(f"Complete: {total_det} total detections")
            self.status_bar.showMessage(f"DSP detection complete: {total_det} detections in {len(results)} files")

        # Clear queue and refresh display
        self.dsp_queue = []
        self._update_queue_display()

        self._refresh_file_list_items()
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
            self.waveform_overview.set_detections([])
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

        # Update jump spinner
        self.spin_jump.blockSignals(True)
        self.spin_jump.setRange(1, max(1, n_det))
        self.spin_jump.setValue(self.current_detection_idx + 1)
        self.spin_jump.blockSignals(False)

        # Update waveform overview detection ticks
        det_times = list(zip(
            self.detections_df['start_seconds'].tolist(),
            self.detections_df['stop_seconds'].tolist()
        ))
        self.waveform_overview.set_detections(det_times)

        # Update spectrogram view
        self._update_spectrogram_view()
        self._update_progress()
        self._update_button_counts()
        self._update_statistics()

        # Refresh ML label counts (enables train button when enough labels exist)
        if self._yolo_project_config is not None:
            self._update_ml_state()

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
        self.waveform_overview.set_view_range(view_start, view_end)

        # Update detection boxes (reuses shared method to include contour data)
        self._update_detection_boxes()

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

    def prev_detection(self):
        """Go to previous detection by time (respects filter)."""
        if self.detections_df is None:
            return
        filtered = self._get_filtered_indices()
        if not filtered:
            return
        # Get current detection's start time
        current_time = self.detections_df.iloc[self.current_detection_idx]['start_seconds']
        # Find all filtered detections that start before current, sorted by time
        earlier = [(i, self.detections_df.iloc[i]['start_seconds']) for i in filtered
                   if self.detections_df.iloc[i]['start_seconds'] < current_time]
        if not earlier:
            # Wrap: also check detections at same time but different index
            earlier = [(i, self.detections_df.iloc[i]['start_seconds']) for i in filtered
                       if i != self.current_detection_idx]
            if not earlier:
                return
            # Go to the last one temporally (wrap around)
            earlier.sort(key=lambda x: x[1])
            self.current_detection_idx = earlier[-1][0]
        else:
            # Go to the nearest earlier detection (latest start time before current)
            earlier.sort(key=lambda x: x[1])
            self.current_detection_idx = earlier[-1][0]
        self._update_display()

    def next_detection(self):
        """Go to next detection by time (respects filter)."""
        if self.detections_df is None:
            return
        filtered = self._get_filtered_indices()
        if not filtered:
            return
        # Get current detection's start time
        current_time = self.detections_df.iloc[self.current_detection_idx]['start_seconds']
        # Find all filtered detections that start after current, sorted by time
        later = [(i, self.detections_df.iloc[i]['start_seconds']) for i in filtered
                 if self.detections_df.iloc[i]['start_seconds'] > current_time]
        if not later:
            # Wrap: also check detections at same time but different index
            later = [(i, self.detections_df.iloc[i]['start_seconds']) for i in filtered
                     if i != self.current_detection_idx]
            if not later:
                return
            # Go to the first one temporally (wrap around)
            later.sort(key=lambda x: x[1])
            self.current_detection_idx = later[0][0]
        else:
            # Go to the nearest later detection (earliest start time after current)
            later.sort(key=lambda x: x[1])
            self.current_detection_idx = later[0][0]
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
        """Update detection boxes on spectrogram (includes peak_freq contour data)."""
        if self.detections_df is None or len(self.detections_df) == 0:
            self.spectrogram.set_detections([], -1)
            return

        detections = []
        for _, row in self.detections_df.iterrows():
            min_freq = row['min_freq_hz'] if 'min_freq_hz' in row and not pd.isna(row['min_freq_hz']) else 20000
            max_freq = row['max_freq_hz'] if 'max_freq_hz' in row and not pd.isna(row['max_freq_hz']) else 80000
            status = row['status'] if 'status' in row and not pd.isna(row['status']) else 'pending'

            det = {
                'start_seconds': row['start_seconds'],
                'stop_seconds': row['stop_seconds'],
                'min_freq_hz': min_freq,
                'max_freq_hz': max_freq,
                'status': status,
            }
            # Pass through peak_freq_N contour columns only if freq sampling is enabled
            if self.chk_freq_samples.isChecked():
                for col in row.index:
                    if col.startswith('peak_freq_') and col != 'peak_freq_hz':
                        val = row[col]
                        if not pd.isna(val):
                            det[col] = val
            detections.append(det)

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
        self.waveform_overview.set_view_range(view_start, view_end)

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
        self.waveform_overview.set_view_range(new_start, new_end)

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
        self.waveform_overview.set_view_range(new_start, new_end)

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
        self.waveform_overview.set_view_range(new_start, new_end)

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
        self.waveform_overview.set_view_range(new_start, new_end)

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
        old_status = self.detections_df.at[self.current_detection_idx, 'status']
        self.undo_stack.append(('label', self.current_detection_idx, old_status))
        self.btn_undo.setEnabled(True)
        self.detections_df.at[self.current_detection_idx, 'status'] = 'accepted'
        self._store_current_detections()
        self._update_display()
        self._update_button_counts()
        self._auto_advance()

    def reject_detection(self):
        """Mark current detection as rejected."""
        if self.detections_df is None:
            return
        old_status = self.detections_df.at[self.current_detection_idx, 'status']
        self.undo_stack.append(('label', self.current_detection_idx, old_status))
        self.btn_undo.setEnabled(True)
        self.detections_df.at[self.current_detection_idx, 'status'] = 'rejected'
        self._store_current_detections()
        self._update_display()
        self._update_button_counts()
        self._auto_advance()

    def mark_negative(self):
        """Mark current detection as negative (background) training data.

        Expands the box to full frequency range (0 to Nyquist) so the
        entire time span is labeled as 'no USV here'.
        """
        if self.detections_df is None or len(self.detections_df) == 0:
            return
        if self.current_detection_idx >= len(self.detections_df):
            return
        old_status = self.detections_df.at[self.current_detection_idx, 'status']
        self.undo_stack.append(('label', self.current_detection_idx, old_status))
        self.btn_undo.setEnabled(True)
        self.detections_df.at[self.current_detection_idx, 'status'] = 'negative'
        # Expand to full frequency range
        nyquist = self.sample_rate / 2 if self.sample_rate else 125000
        self.detections_df.at[self.current_detection_idx, 'min_freq_hz'] = 0
        self.detections_df.at[self.current_detection_idx, 'max_freq_hz'] = nyquist
        self.detections_df.at[self.current_detection_idx, 'freq_bandwidth_hz'] = nyquist
        self._store_current_detections()
        self._update_display()
        self._update_button_counts()
        self._auto_advance()

    def skip_detection(self):
        """Skip to next detection by time without changing its status."""
        if self.detections_df is None:
            return
        current_time = self.detections_df.iloc[self.current_detection_idx]['start_seconds']
        # Find temporally next detection (any status)
        all_dets = [(i, self.detections_df.iloc[i]['start_seconds'])
                    for i in range(len(self.detections_df))]
        all_dets.sort(key=lambda x: x[1])
        for idx, t in all_dets:
            if t > current_time:
                self.current_detection_idx = idx
                self._update_display()
                return
        # At the last detection temporally — about to wrap
        # If fully curated, prompt to move to next file
        n_pending = (self.detections_df['status'] == 'pending').sum()
        if n_pending == 0:
            self._update_display()
            self._prompt_next_file()
            return
        # Wrap to earliest
        self.current_detection_idx = all_dets[0][0]
        self._update_display()

    def _update_button_counts(self):
        """Update Accept/Reject/Noise button labels with counts."""
        if self.detections_df is None or len(self.detections_df) == 0:
            self.btn_accept.setText("Accept")
            self.btn_reject.setText("Reject")
            return

        counts = self.detections_df['status'].value_counts()
        n_accepted = counts.get('accepted', 0)
        n_rejected = counts.get('rejected', 0)
        n_negative = counts.get('negative', 0)

        self.btn_accept.setText(f"Accept ({n_accepted})")
        self.btn_reject.setText(f"Reject ({n_rejected})")
        # Show negative count in status bar if any exist
        if n_negative > 0:
            self.status_bar.showMessage(
                f"Detections: {n_accepted} accepted, {n_rejected} rejected, "
                f"{n_negative} negative, {len(self.detections_df) - n_accepted - n_rejected - n_negative} pending"
            )

    def add_new_usv(self):
        """Add a new USV detection at view center."""
        if self.audio_data is None:
            return

        view_start, view_end = self.spectrogram.get_view_range()
        view_center = (view_start + view_end) / 2
        view_duration = view_end - view_start

        new_duration = min(0.03, view_duration * 0.05)
        new_start = view_center - new_duration / 2
        new_stop = view_center + new_duration / 2

        # Small square box centered at 40 kHz (typical USV range)
        freq_center = 40000
        freq_half = 4000  # ±4 kHz = 8 kHz span
        new_row = {
            'start_seconds': new_start,
            'stop_seconds': new_stop,
            'duration_ms': new_duration * 1000,
            'min_freq_hz': freq_center - freq_half,
            'max_freq_hz': freq_center + freq_half,
            'peak_freq_hz': freq_center,
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

    def delete_all_labels(self):
        """Delete ALL detections (pending + accepted + rejected) for the current file."""
        if self.detections_df is None or len(self.detections_df) == 0:
            return

        n_total = len(self.detections_df)
        reply = QMessageBox.warning(
            self, "Warning",
            f"This action will delete all {n_total} pending and approved labels "
            "in this file.\n\nAre you sure you want to continue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Remove the CSV file on disk so labels don't reload
        if self.audio_files and self.current_file_idx < len(self.audio_files):
            filepath = self.audio_files[self.current_file_idx]
            base = Path(filepath).stem
            parent = Path(filepath).parent
            source = self.detection_sources.get(filepath, 'usv_dsp')
            csv_path = parent / f"{base}_{source}.csv"
            if csv_path.exists():
                csv_path.unlink()

        self.detections_df = None
        self.current_detection_idx = 0
        self._store_current_detections()
        self._update_display()
        self._update_ui_state()
        self.status_bar.showMessage(f"Deleted all {n_total} detections")

    def _auto_advance(self):
        """Auto-advance to next pending detection by time, with wrap-around.

        After accept/reject, finds the temporally next pending detection.
        If none exist after the current position, wraps to the beginning.
        If ALL detections have been curated, prompts to move to next file.
        """
        if self.detections_df is None:
            return

        current_time = self.detections_df.iloc[self.current_detection_idx]['start_seconds']

        # Build list of all pending detections with their times
        pending = []
        for i in range(len(self.detections_df)):
            if self.detections_df.iloc[i]['status'] == 'pending':
                pending.append((i, self.detections_df.iloc[i]['start_seconds']))

        if not pending:
            # All detections have been curated — prompt for next file
            self._update_display()
            self._prompt_next_file()
            return

        # Sort by start time
        pending.sort(key=lambda x: x[1])

        # Find next pending after current time
        for idx, t in pending:
            if t > current_time:
                self.current_detection_idx = idx
                self._update_display()
                return

        # Wrap around — take the earliest pending detection
        self.current_detection_idx = pending[0][0]
        self._update_display()

    def _prompt_next_file(self):
        """Show dialog when all detections are curated, asking to move to next file."""
        box = QMessageBox(self)
        box.setWindowTitle("File Complete")
        box.setText("All detections in this file have been curated.\n\nMove to the next file?")
        btn_yes = box.addButton("Yes (Press Enter)", QMessageBox.AcceptRole)
        box.addButton("No", QMessageBox.RejectRole)
        box.setDefaultButton(btn_yes)
        box.exec_()
        if box.clickedButton() == btn_yes:
            self._advance_to_next_file_with_detections()

    def _advance_to_next_file_with_detections(self):
        """Auto-switch to the next file that has detections."""
        if not self.audio_files:
            return

        # Save current file's detections first
        self._store_current_detections()

        # Search forward through files for one with detections
        for offset in range(1, len(self.audio_files)):
            next_idx = (self.current_file_idx + offset) % len(self.audio_files)
            filepath = self.audio_files[next_idx]

            has_dets = filepath in self.all_detections and len(self.all_detections[filepath]) > 0
            if not has_dets:
                # Check disk
                has_dets = self._file_has_detections(filepath)

            if has_dets:
                self.file_list.setCurrentRow(next_idx)
                # Jump to first pending detection by time, or first detection if none pending
                if self.detections_df is not None and len(self.detections_df) > 0:
                    pending = [(i, self.detections_df.iloc[i]['start_seconds'])
                               for i in range(len(self.detections_df))
                               if self.detections_df.iloc[i]['status'] == 'pending']
                    if pending:
                        pending.sort(key=lambda x: x[1])
                        self.current_detection_idx = pending[0][0]
                    else:
                        time_sorted = self.detections_df['start_seconds'].argsort()
                        self.current_detection_idx = time_sorted.iloc[0]
                    self._update_display()
                self.status_bar.showMessage(
                    f"Advanced to {os.path.basename(filepath)}")
                return

        self.status_bar.showMessage("No more files with detections")

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
        """Save current detections to CSV file, preserving original naming."""
        if self.detections_df is None or len(self.detections_df) == 0:
            return

        # Refresh call_number column before saving
        self.detections_df['call_number'] = range(1, len(self.detections_df) + 1)
        # Ensure call_number is the first column
        cols = self.detections_df.columns.tolist()
        if 'call_number' in cols:
            cols.remove('call_number')
            cols.insert(0, 'call_number')
            self.detections_df = self.detections_df[cols]

        base = Path(filepath).stem
        parent = Path(filepath).parent
        # Use the tracked source suffix, default to _FNT_CAD_detections
        source = self.detection_sources.get(filepath, 'FNT_CAD_detections')
        csv_path = parent / f"{base}_{source}.csv"

        try:
            self.detections_df.to_csv(csv_path, index=False)
        except Exception as e:
            self.status_bar.showMessage(f"Error saving CSV: {e}")

    # =========================================================================
    # Playback
    # =========================================================================

    def on_speed_changed(self):
        """Handle playback speed slider change."""
        idx = self.slider_speed.value()
        speed = self._speed_values[idx]
        self.use_heterodyne = False
        self.playback_speed = speed
        self.lbl_speed.setText(f"{speed}x")

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
                # Resample to a standard output rate for smooth playback.
                # Playing at speed S means the output should have
                # (original_duration / S) seconds of audio at the output rate.
                output_sr = 44100
                # Number of output samples = original_duration / speed * output_sr
                original_duration = len(segment) / self.sample_rate
                output_duration = original_duration / self.playback_speed
                n_output_samples = int(output_duration * output_sr)
                if n_output_samples < 100:
                    return
                segment = signal.resample(segment, n_output_samples).astype(np.float32)
                play_sr = output_sr

            sd.play(segment, play_sr)
            self.is_playing = True
            self.btn_play.setText("Playing...")

            # Track playback position for the moving line.
            # Use a small latency offset to account for audio output buffering.
            import time as _time
            self._playback_latency = 0.15  # seconds of estimated output latency
            self._playback_start_time = _time.time()
            self._playback_start_s = start_s
            self._playback_end_s = stop_s
            self._playback_timer.start()

            # Disable pan/zoom/navigation controls during playback
            self._set_playback_controls_enabled(False)

        except Exception as e:
            self.status_bar.showMessage(f"Playback error: {e}")

    def stop_playback(self):
        """Stop playback."""
        if HAS_SOUNDDEVICE:
            sd.stop()
        self.is_playing = False
        self.btn_play.setText("Play")
        self._playback_timer.stop()
        self.spectrogram.playback_position = None
        self.spectrogram.update()
        # Re-enable controls
        self._set_playback_controls_enabled(True)

    def _update_playback_position(self):
        """Timer callback to update the playback position line."""
        import time as _time
        if not self.is_playing or self._playback_start_time is None:
            self.stop_playback()
            return

        # Subtract latency offset so the line matches when audio is heard
        elapsed = _time.time() - self._playback_start_time - self._playback_latency
        elapsed = max(0.0, elapsed)
        # Convert wall-clock elapsed to audio time using playback speed
        audio_elapsed = elapsed * self.playback_speed
        current_pos = self._playback_start_s + audio_elapsed

        if current_pos >= self._playback_end_s:
            # Playback finished
            self.stop_playback()
            return

        self.spectrogram.playback_position = current_pos
        self.spectrogram.update()

    def _set_playback_controls_enabled(self, enabled):
        """Enable or disable pan/zoom/navigation controls during playback."""
        self.btn_pan_left.setEnabled(enabled)
        self.btn_pan_right.setEnabled(enabled)
        self.btn_zoom_in.setEnabled(enabled)
        self.btn_zoom_out.setEnabled(enabled)
        self.spin_view_window.setEnabled(enabled)
        self.time_scrollbar.setEnabled(enabled)
        self.btn_prev_det.setEnabled(enabled)
        self.btn_next_det.setEnabled(enabled)
        self.btn_prev_file.setEnabled(enabled)
        self.btn_next_file.setEnabled(enabled)
        self.slider_speed.setEnabled(enabled)
        self.file_list.setEnabled(enabled)

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
        """Open folder containing current file (cross-platform)."""
        if self.audio_files and self.current_file_idx < len(self.audio_files):
            folder = Path(self.audio_files[self.current_file_idx]).parent
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(folder)))

    # =========================================================================
    # Machine Learning
    # =========================================================================

    # =========================================================================
    # ML Detection (YOLO)
    # =========================================================================

    def _ml_open_project(self):
        """Open or create a YOLO ML detection project."""
        folder = QFileDialog.getExistingDirectory(
            self, "Select or Create YOLO Project Directory"
        )
        if not folder:
            return

        try:
            from fnt.usv.usv_detector.yolo_detector import (
                YOLOProjectConfig, create_project
            )
        except ImportError as e:
            QMessageBox.warning(self, "Import Error", f"Failed to import YOLO module:\n{e}")
            return

        config_path = os.path.join(folder, 'project_config.json')
        if os.path.exists(config_path):
            # Load existing project
            self._yolo_project_config = YOLOProjectConfig.load(config_path)
            self.lbl_ml_project.setText(os.path.basename(folder))
            # Check for latest model
            if self._yolo_project_config.models:
                latest = self._yolo_project_config.models[-1]
                model_path = latest.get('path', '')
                if os.path.exists(model_path):
                    self._yolo_model_path = model_path
                    self.lbl_ml_model.setText(f"Model: {latest.get('name', 'unknown')}")
            self.status_bar.showMessage(f"Opened YOLO project: {folder}")
        else:
            # Create new project
            self._yolo_project_config = create_project(folder)
            self.lbl_ml_project.setText(os.path.basename(folder))
            self.status_bar.showMessage(f"Created new YOLO project: {folder}")

        self._update_ml_state()

    def _update_ml_state(self):
        """Update ML section UI based on current state."""
        from fnt.usv.usv_detector.yolo_detector import get_training_data_counts

        # Count labeled data
        counts = get_training_data_counts(self.all_detections)
        n_pos = counts['n_positive']
        n_neg = counts['n_negative']
        self.lbl_ml_data.setText(f"Labels: {n_pos} positive, {n_neg} negative")

        has_project = self._yolo_project_config is not None
        has_labels = n_pos >= 1 and n_neg >= 1
        has_model = self._yolo_model_path is not None and os.path.exists(str(self._yolo_model_path or ''))
        has_files = len(self.audio_files) > 0

        # Train button: needs project + at least 1 positive + 1 negative
        self.btn_ml_train.setEnabled(has_project and has_labels)
        if has_project and not has_labels:
            if n_neg == 0:
                self.lbl_ml_status.setText("Add negative examples (X key) to enable training")
            elif n_pos == 0:
                self.lbl_ml_status.setText("Add accepted labels (A key) to enable training")

        # Detect buttons: needs project + trained model + files
        self.btn_ml_detect_current.setEnabled(has_model and has_files)
        self.btn_ml_detect_all.setEnabled(has_model and has_files)

    def _ml_train(self):
        """Export training data and train YOLO model."""
        if self._yolo_project_config is None:
            return

        try:
            from fnt.usv.usv_detector.yolo_detector import (
                export_training_data, write_yolo_dataset_yaml,
                train_yolo_model, get_training_data_counts
            )
        except ImportError as e:
            QMessageBox.warning(self, "Import Error", str(e))
            return

        config = self._yolo_project_config
        counts = get_training_data_counts(self.all_detections)
        n_pos = counts['n_positive']
        n_neg = counts['n_negative']

        if n_pos < 1 or n_neg < 1:
            QMessageBox.warning(
                self, "Insufficient Labels",
                f"Need at least 1 positive and 1 negative label.\n"
                f"Currently: {n_pos} positive, {n_neg} negative.\n\n"
                f"Use A key to accept USV calls, X key to mark background."
            )
            return

        # Confirm
        reply = QMessageBox.question(
            self, "Train YOLO Model",
            f"Training data: {n_pos} positive, {n_neg} negative from {counts['n_files_with_labels']} files.\n\n"
            f"This will export spectrogram tiles and train a YOLOv8 model.\n"
            f"Training stops automatically when loss plateaus.\n"
            f"Continue?",
            QMessageBox.Yes | QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return

        self.btn_ml_train.setEnabled(False)
        self.ml_progress.setVisible(True)
        self.ml_progress.setValue(0)
        self.lbl_ml_status.setText("Exporting training data...")
        QApplication.processEvents()

        # Export training data
        dataset_dir = os.path.join(config.project_dir, 'datasets', 'train')
        try:
            export_stats = export_training_data(
                self.audio_files, self.all_detections, dataset_dir, config,
                progress_callback=lambda msg, cur, tot: (
                    self.lbl_ml_status.setText(msg),
                    self.ml_progress.setValue(int(cur / max(tot, 1) * 30)),
                    QApplication.processEvents(),
                )
            )
        except Exception as e:
            QMessageBox.critical(self, "Export Error", f"Failed to export training data:\n{e}")
            self.btn_ml_train.setEnabled(True)
            self.ml_progress.setVisible(False)
            return

        self.lbl_ml_status.setText(
            f"Exported {export_stats['n_tiles']} tiles "
            f"({export_stats['n_positive']} positive, {export_stats['n_negative']} negative)"
        )
        self.ml_progress.setValue(30)
        QApplication.processEvents()

        # Write dataset YAML
        yaml_path = write_yolo_dataset_yaml(config.project_dir, dataset_dir)

        # Determine model output dir
        model_name = f"fntUSVStudioModel_n={n_pos}"
        model_dir = os.path.join(config.project_dir, 'models', model_name)

        # Get previous model for fine-tuning
        prev_weights = None
        if config.models:
            prev_path = config.models[-1].get('path', '')
            if os.path.exists(prev_path):
                prev_weights = prev_path

        self.lbl_ml_status.setText("Training YOLO model...")
        self.ml_progress.setValue(35)
        QApplication.processEvents()

        # Train in a worker thread
        self._yolo_train_worker = YOLOTrainingWorker(
            yaml_path, model_dir, model_name,
            pretrained_weights=prev_weights,
        )
        self._yolo_train_worker.progress.connect(self._on_ml_train_progress)
        self._yolo_train_worker.complete.connect(
            lambda path: self._on_ml_train_complete(path, model_name, n_pos, n_neg)
        )
        self._yolo_train_worker.error.connect(self._on_ml_train_error)
        self._yolo_train_worker.start()

    def _on_ml_train_progress(self, message):
        """Handle YOLO training progress."""
        self.lbl_ml_status.setText(message)
        # Estimate progress in 35-95% range
        current = self.ml_progress.value()
        if current < 95:
            self.ml_progress.setValue(current + 1)

    def _on_ml_train_complete(self, model_path, model_name, n_pos, n_neg):
        """Handle YOLO training completion."""
        self.ml_progress.setValue(100)
        self.ml_progress.setVisible(False)
        self.btn_ml_train.setEnabled(True)

        self._yolo_model_path = model_path
        self.lbl_ml_model.setText(f"Model: {model_name}")
        self.lbl_ml_status.setText(f"Training complete! Model: {model_name}")
        self.status_bar.showMessage(f"YOLO model trained: {model_path}")

        # Update project config with new model
        if self._yolo_project_config is not None:
            self._yolo_project_config.models.append({
                'name': model_name,
                'n_positive': n_pos,
                'n_negative': n_neg,
                'path': model_path,
                'date': datetime.now().isoformat(),
            })
            self._yolo_project_config.save()

        self._update_ml_state()

    def _on_ml_train_error(self, error_msg):
        """Handle YOLO training error."""
        self.ml_progress.setVisible(False)
        self.btn_ml_train.setEnabled(True)
        self.lbl_ml_status.setText(f"Training failed")
        QMessageBox.critical(self, "Training Error", f"YOLO training failed:\n{error_msg}")

    def _ml_detect_current(self):
        """Run YOLO detection on current file."""
        if not self._yolo_model_path or not self.audio_files:
            return
        filepath = self.audio_files[self.current_file_idx]
        self._ml_detect_files([filepath])

    def _ml_detect_all(self):
        """Run YOLO detection on all files."""
        if not self._yolo_model_path or not self.audio_files:
            return
        self._ml_detect_files(list(self.audio_files))

    def _ml_detect_files(self, files):
        """Run YOLO inference on a list of files."""
        if not self._yolo_model_path or not self._yolo_project_config:
            return

        self.btn_ml_detect_current.setEnabled(False)
        self.btn_ml_detect_all.setEnabled(False)
        self.ml_progress.setVisible(True)
        self.ml_progress.setValue(0)
        self.lbl_ml_status.setText("Running ML detection...")

        self._yolo_infer_worker = YOLOInferenceWorker(
            files, self._yolo_model_path, self._yolo_project_config,
        )
        self._yolo_infer_worker.progress.connect(self._on_ml_infer_progress)
        self._yolo_infer_worker.file_complete.connect(self._on_ml_file_complete)
        self._yolo_infer_worker.all_complete.connect(self._on_ml_infer_complete)
        self._yolo_infer_worker.error.connect(self._on_ml_infer_error)
        self._yolo_infer_worker.start()

    def _on_ml_infer_progress(self, filename, current, total):
        """Handle ML inference progress."""
        self.ml_progress.setValue(int(current / total * 100))
        self.lbl_ml_status.setText(f"Detecting: {filename}")

    def _on_ml_file_complete(self, filename, filepath, detections, n_detections):
        """Handle ML file completion — store results and write CSV."""
        std_columns = ['call_number', 'start_seconds', 'stop_seconds', 'duration_ms',
                        'min_freq_hz', 'max_freq_hz', 'peak_freq_hz', 'freq_bandwidth_hz',
                        'max_power_db', 'mean_power_db', 'status', 'source']

        if isinstance(detections, pd.DataFrame):
            df = detections.copy()
        else:
            df = pd.DataFrame(detections)

        if len(df) > 0:
            if 'status' not in df.columns:
                df['status'] = 'pending'
            if 'source' not in df.columns:
                df['source'] = 'ml'
            if 'call_number' not in df.columns:
                df.insert(0, 'call_number', range(1, len(df) + 1))
        else:
            df = pd.DataFrame(columns=std_columns)

        self.all_detections[filepath] = df
        self.detection_sources[filepath] = 'usv_yolo'

        # Write CSV immediately
        base = Path(filepath).stem
        parent = Path(filepath).parent
        csv_path = parent / f"{base}_usv_yolo.csv"
        try:
            df.to_csv(csv_path, index=False)
        except Exception as e:
            self.status_bar.showMessage(f"Error saving CSV for {filename}: {e}")

        # Update file list
        if filepath in self.audio_files:
            idx = self.audio_files.index(filepath)
            item = self.file_list.item(idx)
            if item:
                self.file_list.blockSignals(True)
                item.setText(f"{filename} ({len(df)})")
                self.file_list.blockSignals(False)

        # Update current view if this is the displayed file
        if (self.audio_files and self.current_file_idx < len(self.audio_files)
                and self.audio_files[self.current_file_idx] == filepath):
            self.detections_df = df.copy()
            self._ensure_freq_bounds(self.detections_df)
            self.current_detection_idx = 0
            self._load_current_file()

    def _on_ml_infer_complete(self, results):
        """Handle ML inference batch completion."""
        self.ml_progress.setValue(100)
        self.ml_progress.setVisible(False)
        self.btn_ml_detect_current.setEnabled(True)
        self.btn_ml_detect_all.setEnabled(True)

        total_det = sum(len(d) for d in results.values() if isinstance(d, (list, pd.DataFrame)))
        self.lbl_ml_status.setText(f"Complete: {total_det} detections in {len(results)} files")
        self.status_bar.showMessage(f"ML detection complete: {total_det} detections")

        self._refresh_file_list_items()
        self._load_current_file()

    def _on_ml_infer_error(self, filename, error):
        """Handle ML inference error."""
        self.lbl_ml_status.setText(f"Error: {filename} - {error}")

    # =========================================================================
    # UI State Management
    # =========================================================================

    def _update_ui_state(self):
        """Update UI enabled states based on current state."""
        has_files = len(self.audio_files) > 0
        has_audio = self.audio_data is not None
        has_det = self.detections_df is not None and len(self.detections_df) > 0
        # Input
        self.btn_add_to_queue.setEnabled(has_files)
        self.btn_add_all_to_queue.setEnabled(has_files)

        # Detection navigation
        self.btn_prev_det.setEnabled(bool(has_det and self.current_detection_idx > 0))
        self.btn_next_det.setEnabled(bool(has_det and self.current_detection_idx < len(self.detections_df) - 1) if has_det else False)

        # Labeling
        self.btn_accept.setEnabled(bool(has_det))
        self.btn_reject.setEnabled(bool(has_det))
        self.btn_skip.setEnabled(bool(has_det))
        self.btn_add_usv.setEnabled(bool(has_audio))
        self.btn_delete.setEnabled(bool(has_det))

        has_pending = bool(has_det and (self.detections_df['status'] == 'pending').any())
        self.btn_delete_pending.setEnabled(has_pending)
        self.btn_delete_all_labels.setEnabled(bool(has_det))
        self.btn_accept_all_pending.setEnabled(has_pending)
        self.btn_reject_all_pending.setEnabled(has_pending)

        # Playback
        self.btn_play.setEnabled(bool(has_audio and HAS_SOUNDDEVICE))
        self.btn_stop.setEnabled(bool(has_audio and HAS_SOUNDDEVICE))

        # Open folder
        self.btn_open_folder.setEnabled(bool(has_files))

        # ML — update training data counts if project is open
        if self._yolo_project_config is not None:
            try:
                self._update_ml_state()
            except Exception:
                pass

    def keyPressEvent(self, event):
        """Handle keyboard shortcuts. Skips when text-input widgets have focus."""
        # Don't intercept if a text-input widget has focus
        focus = QApplication.focusWidget()
        if isinstance(focus, (QSpinBox, QDoubleSpinBox, QComboBox)):
            super().keyPressEvent(event)
            return

        key = event.key()
        modifiers = event.modifiers()

        # Ctrl+Z = Undo
        if key == Qt.Key_Z and (modifiers & Qt.ControlModifier):
            self.undo_action()
            return

        if key == Qt.Key_A and self.btn_accept.isEnabled():
            self._flash_button(self.btn_accept)
            self.accept_detection()
        elif key == Qt.Key_R and self.btn_reject.isEnabled():
            self._flash_button(self.btn_reject)
            self.reject_detection()
        elif key == Qt.Key_X:
            self.mark_negative()
        elif key == Qt.Key_D and self.btn_delete.isEnabled():
            self._flash_button(self.btn_delete)
            self.delete_current()
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

    # =========================================================================
    # Undo
    # =========================================================================

    def undo_action(self):
        """Undo the last labeling action."""
        if not self.undo_stack or self.detections_df is None:
            return

        action = self.undo_stack.pop()
        if action[0] == 'label':
            _, idx, old_status = action
            if 0 <= idx < len(self.detections_df):
                self.detections_df.at[idx, 'status'] = old_status
                self.current_detection_idx = idx
                self._store_current_detections()
                self._update_display()
                self._update_button_counts()
                self.status_bar.showMessage(f"Undid label change on detection {idx + 1}")
        elif action[0] == 'batch_label':
            _, indices, old_status = action
            for idx in indices:
                if 0 <= idx < len(self.detections_df):
                    self.detections_df.at[idx, 'status'] = old_status
            self._store_current_detections()
            self._update_display()
            self._update_button_counts()
            self.status_bar.showMessage(f"Undid batch label change on {len(indices)} detections")

        self.btn_undo.setEnabled(len(self.undo_stack) > 0)

    # =========================================================================
    # Batch Labeling
    # =========================================================================

    def accept_all_pending(self):
        """Accept all pending detections."""
        if self.detections_df is None:
            return
        mask = self.detections_df['status'] == 'pending'
        n = mask.sum()
        if n == 0:
            self.status_bar.showMessage("No pending detections")
            return
        reply = QMessageBox.question(
            self, "Accept All Pending",
            f"Accept all {n} pending detections?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        # Push batch undo
        self.undo_stack.append(('batch_label', list(self.detections_df[mask].index), 'pending'))
        self.btn_undo.setEnabled(True)
        self.detections_df.loc[mask, 'status'] = 'accepted'
        self._store_current_detections()
        self._update_display()
        self._update_button_counts()
        self.status_bar.showMessage(f"Accepted {n} detections")

    def reject_all_pending(self):
        """Reject all pending detections."""
        if self.detections_df is None:
            return
        mask = self.detections_df['status'] == 'pending'
        n = mask.sum()
        if n == 0:
            self.status_bar.showMessage("No pending detections")
            return
        reply = QMessageBox.question(
            self, "Reject All Pending",
            f"Reject all {n} pending detections?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        self.undo_stack.append(('batch_label', list(self.detections_df[mask].index), 'pending'))
        self.btn_undo.setEnabled(True)
        self.detections_df.loc[mask, 'status'] = 'rejected'
        self._store_current_detections()
        self._update_display()
        self._update_button_counts()
        self.status_bar.showMessage(f"Rejected {n} detections")

    # =========================================================================
    # Filter / Jump
    # =========================================================================

    def on_filter_changed(self):
        """Handle filter combo change."""
        self.filter_status = self.combo_filter.currentData()
        self._update_display()

    def on_jump_to_detection(self):
        """Jump to a specific detection number."""
        idx = self.spin_jump.value() - 1
        if self.detections_df is not None and 0 <= idx < len(self.detections_df):
            self.current_detection_idx = idx
            self._update_display()

    def _get_filtered_indices(self):
        """Get detection indices matching current filter."""
        if self.detections_df is None or len(self.detections_df) == 0:
            return []
        if self.filter_status == 'all':
            return list(range(len(self.detections_df)))
        return list(self.detections_df.index[self.detections_df['status'] == self.filter_status])

    # =========================================================================
    # Wheel Zoom / Overview Navigation / Display Frequency
    # =========================================================================

    def on_wheel_zoom(self, factor, center_time):
        """Handle mouse wheel zoom on spectrogram."""
        current_start, current_end = self.spectrogram.get_view_range()
        current_window = current_end - current_start
        new_window = current_window / factor
        total_dur = self.spectrogram.get_total_duration()
        new_window = max(0.05, min(total_dur if total_dur > 0 else 600, new_window))

        # Maintain zoom center at mouse position
        ratio = (center_time - current_start) / current_window if current_window > 0 else 0.5
        new_start = center_time - ratio * new_window
        new_end = new_start + new_window

        new_start = max(0, new_start)
        new_end = min(total_dur, new_end)

        self.spin_view_window.blockSignals(True)
        self.spin_view_window.setValue(new_end - new_start)
        self.spin_view_window.blockSignals(False)

        self.spectrogram.set_view_range(new_start, new_end)
        self._update_scrollbar()
        self._update_detection_boxes()
        self.waveform_overview.set_view_range(new_start, new_end)

    def on_overview_clicked(self, center_time):
        """Handle click on waveform overview to navigate."""
        window = self.spin_view_window.value()
        total_dur = self.spectrogram.get_total_duration()

        view_start = max(0, center_time - window / 2)
        view_end = min(total_dur, center_time + window / 2)

        self.spectrogram.set_view_range(view_start, view_end)
        self._update_scrollbar()
        self._update_detection_boxes()
        self.waveform_overview.set_view_range(view_start, view_end)

    def on_display_freq_changed(self):
        """Handle display frequency range change."""
        min_f = self.spin_display_min_freq.value()
        max_f = self.spin_display_max_freq.value()
        if max_f <= min_f:
            return
        self.spectrogram.min_freq = min_f
        self.spectrogram.max_freq = max_f
        # Force spectrogram recompute
        self.spectrogram.cached_view_start = None
        self.spectrogram.cached_view_end = None
        self.spectrogram.spec_image = None
        if self.spectrogram.total_duration > 0:
            self.spectrogram._compute_view_spectrogram()
        self.spectrogram.update()

    def on_colormap_changed(self, name):
        """Handle colormap selection change."""
        self.spectrogram.set_colormap(name)

    # =========================================================================
    # Export
    # =========================================================================

    # =========================================================================
    # Detection Statistics
    # =========================================================================

    def _update_statistics(self):
        """Update detection statistics label."""
        if self.detections_df is None or len(self.detections_df) == 0:
            self.lbl_stats.setText("")
            return

        n = len(self.detections_df)
        durations = self.detections_df['stop_seconds'] - self.detections_df['start_seconds']
        dur_ms = durations * 1000
        total_dur = self.spectrogram.get_total_duration()
        cpm = n / (total_dur / 60) if total_dur > 0 else 0

        parts = [f"{n} calls | {cpm:.1f}/min"]
        parts.append(f"Dur: {dur_ms.mean():.1f}ms (SD {dur_ms.std():.1f})")

        if 'peak_freq_hz' in self.detections_df.columns:
            pf = self.detections_df['peak_freq_hz'].dropna()
            if len(pf) > 0:
                parts.append(f"Peak F: {pf.mean()/1000:.1f}kHz")

        self.lbl_stats.setText(" | ".join(parts))

    # =========================================================================
    # Settings Persistence
    # =========================================================================

    def _restore_settings(self):
        """Restore window geometry and preferences from QSettings."""
        settings = QSettings("FNT", "USVStudio")
        geometry = settings.value("geometry")
        if geometry:
            self.restoreGeometry(geometry)
        state = settings.value("windowState")
        if state:
            self.restoreState(state)
        # Restore view preferences
        view_window = settings.value("view_window", 2.0, type=float)
        self.spin_view_window.setValue(view_window)
        colormap = settings.value("colormap", "viridis", type=str)
        idx = self.combo_colormap.findText(colormap)
        if idx >= 0:
            self.combo_colormap.setCurrentIndex(idx)

    def closeEvent(self, event):
        """Save settings on close."""
        settings = QSettings("FNT", "USVStudio")
        settings.setValue("geometry", self.saveGeometry())
        settings.setValue("windowState", self.saveState())
        settings.setValue("view_window", self.spin_view_window.value())
        settings.setValue("colormap", self.combo_colormap.currentText())
        super().closeEvent(event)


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
