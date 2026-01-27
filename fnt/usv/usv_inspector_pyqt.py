"""
USV Inspector - Visual and auditory ground-truthing tool for USV detections.

Features:
- Left panel layout with controls, enlarged spectrogram on right
- PyQt-based navigation (no matplotlib toolbar)
- Scrollbar for panning along time axis
- Drawable bounding boxes for adjusting detection boundaries
- Audio playback with slow-down and heterodyne options
"""

import os
import sys
import tempfile
import numpy as np
import pandas as pd
from pathlib import Path
from scipy import signal

from PyQt5.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox, QGroupBox,
    QDoubleSpinBox, QSpinBox, QComboBox, QProgressBar, QStatusBar,
    QSplitter, QFrame, QSizePolicy, QScrollBar, QSlider
)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QRectF, QPointF
from PyQt5.QtGui import QFont, QPainter, QPen, QColor, QBrush, QImage, QPixmap

# Try importing audio libraries
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


class SpectrogramWidget(QWidget):
    """Custom widget for displaying spectrogram with detection boxes.

    Uses lazy spectrogram computation - only computes for visible window.
    """

    # Signals
    detection_selected = pyqtSignal(int)  # index of clicked detection
    box_adjusted = pyqtSignal(int, float, float, float, float)  # idx, start_s, stop_s, min_freq, max_freq

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(400, 300)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMouseTracking(True)

        # Raw audio data (for lazy spectrogram computation)
        self.audio_data = None
        self.sample_rate = None
        self.total_duration = 0.0

        # Cached spectrogram for current view
        self.spec_image = None  # QImage of spectrogram
        self.cached_view_start = None
        self.cached_view_end = None

        # View state
        self.view_start = 0.0  # Start time in seconds
        self.view_end = 10.0   # End time in seconds
        self.min_freq = 0      # Hz
        self.max_freq = 125000 # Hz

        # Detections
        self.detections = []  # List of dicts with start_seconds, stop_seconds, min_freq_hz, max_freq_hz, status
        self.current_detection_idx = -1

        # Interaction state
        self.drag_mode = None  # None, 'resize_left', 'resize_right', 'resize_top', 'resize_bottom', 'move', 'draw'
        self.drag_start = None
        self.drag_start_box = None  # Original box coords when starting move
        self.drag_detection_idx = None
        self.hover_edge = None  # Which edge is being hovered

        # Drawing new box
        self.is_drawing = False
        self.draw_start = None
        self.draw_current = None

        # Colormap lookup table (vectorized)
        self.colormap_lut = self._create_colormap_lut()

    def _create_colormap_lut(self):
        """Create a viridis-like colormap as numpy lookup table (vectorized)."""
        # Simplified viridis colors
        colors = np.array([
            [68, 1, 84],      # dark purple
            [72, 40, 120],    # purple
            [62, 74, 137],    # blue-purple
            [49, 104, 142],   # blue
            [38, 130, 142],   # teal
            [31, 158, 137],   # green-teal
            [53, 183, 121],   # green
            [109, 205, 89],   # light green
            [180, 222, 44],   # yellow-green
            [253, 231, 37],   # yellow
        ], dtype=np.float32)

        # Create 256-color lookup table using vectorized interpolation
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
        """Set raw audio data for lazy spectrogram computation."""
        self.audio_data = audio_data
        self.sample_rate = sample_rate

        if audio_data is not None and sample_rate is not None and len(audio_data) > 0:
            self.total_duration = len(audio_data) / sample_rate
            self.view_start = 0
            self.view_end = min(2.0, self.total_duration)  # Start with 2s view
            self.max_freq = min(125000, sample_rate / 2)
        else:
            self.total_duration = 0
            self.view_start = 0
            self.view_end = 10.0

        # Clear cache
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

    def set_view_range(self, start_s, end_s):
        """Set the visible time range."""
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
        """Compute spectrogram only for the current view window (lazy computation).

        Uses downsampling for large view windows to maintain responsiveness.
        """
        if self.audio_data is None or self.sample_rate is None:
            self.spec_image = None
            return

        # Check if we can use cached result
        # Only use cache if: EXACT same view range (or very close)
        # This ensures spectrogram always matches displayed time range
        current_window = self.view_end - self.view_start
        if (self.cached_view_start is not None and
            self.cached_view_end is not None and
            self.spec_image is not None):
            cached_window = self.cached_view_end - self.cached_view_start
            # Check if zoom level is similar (within 5%)
            zoom_similar = abs(current_window - cached_window) / (cached_window + 1e-10) < 0.05
            # Check if view position is nearly identical (within 1% of window)
            # This prevents stale spectrograms when navigating between close detections
            position_tolerance = current_window * 0.01
            same_position = (abs(self.view_start - self.cached_view_start) < position_tolerance and
                            abs(self.view_end - self.cached_view_end) < position_tolerance)
            if zoom_similar and same_position:
                return  # Use cached spectrogram

        # Extract audio segment for current view (with padding for smooth edges)
        pad_time = 0.1  # 100ms padding
        start_time = max(0, self.view_start - pad_time)
        end_time = min(self.total_duration, self.view_end + pad_time)

        start_sample = int(start_time * self.sample_rate)
        end_sample = int(end_time * self.sample_rate)

        segment = self.audio_data[start_sample:end_sample]

        if len(segment) < 512:
            self.spec_image = None
            return

        # For large views (>30s), downsample to keep computation fast
        # Target: ~2 million samples max for reasonable performance
        view_duration = end_time - start_time
        effective_sr = self.sample_rate

        if len(segment) > 2_000_000:
            # Downsample by taking every Nth sample
            downsample_factor = int(np.ceil(len(segment) / 2_000_000))
            segment = segment[::downsample_factor]
            effective_sr = self.sample_rate / downsample_factor

        # Adjust spectrogram parameters based on effective sample rate
        # Keep nperseg reasonable for the effective sample rate
        nperseg = min(512, len(segment) // 10)
        nperseg = max(128, nperseg)  # At least 128
        noverlap = int(nperseg * 0.75)
        nfft = max(nperseg, 512)

        frequencies, times, Sxx = signal.spectrogram(
            segment, fs=effective_sr,
            nperseg=nperseg, noverlap=noverlap, nfft=nfft, window='hann'
        )

        # Adjust times to absolute positions
        times = times + start_time

        # Convert to dB
        spec_db = 10 * np.log10(Sxx + 1e-10)

        # Find time indices for exact view
        time_mask = (times >= self.view_start) & (times <= self.view_end)
        if not np.any(time_mask):
            self.spec_image = None
            return

        view_spec = spec_db[:, time_mask]

        # Normalize to 0-255
        vmin = np.percentile(view_spec, 5)
        vmax = np.percentile(view_spec, 99)
        normalized = np.clip((view_spec - vmin) / (vmax - vmin + 1e-10), 0, 1)
        indices = (normalized * 255).astype(np.uint8)

        # Flip vertically so low frequencies are at bottom
        indices = np.flipud(indices)

        # Apply colormap using vectorized lookup (FAST)
        rgb_data = self.colormap_lut[indices]

        # Create QImage
        height, width = indices.shape
        self.spec_image = QImage(
            rgb_data.data, width, height, width * 3,
            QImage.Format_RGB888
        ).copy()  # Copy to ensure data persists

        # Cache the view range
        self.cached_view_start = self.view_start
        self.cached_view_end = self.view_end

    def paintEvent(self, event):
        """Paint the spectrogram and detection boxes."""
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)

        # Background
        painter.fillRect(self.rect(), QColor(30, 30, 30))

        if self.spec_image is None:
            # No data message
            painter.setPen(QColor(150, 150, 150))
            painter.drawText(self.rect(), Qt.AlignCenter, "No spectrogram data")
            return

        # Draw spectrogram
        spec_rect = self._get_spec_rect()
        scaled_image = self.spec_image.scaled(
            int(spec_rect.width()), int(spec_rect.height()),
            Qt.IgnoreAspectRatio, Qt.SmoothTransformation
        )
        painter.drawImage(spec_rect.topLeft(), scaled_image)

        # Draw detection boxes
        for i, det in enumerate(self.detections):
            self._draw_detection_box(painter, det, i, spec_rect)

        # Draw currently being drawn box
        if self.is_drawing and self.draw_start and self.draw_current:
            self._draw_temp_box(painter, spec_rect)

        # Draw axes labels
        self._draw_axes(painter, spec_rect)

    def _get_spec_rect(self):
        """Get the rectangle for the spectrogram area (leaving space for axes)."""
        margin_left = 50
        margin_right = 10
        margin_top = 10
        margin_bottom = 40  # Increased for "Time (s)" label visibility
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
            return spec_rect.bottom()
        t = (freq_hz - self.min_freq) / (self.max_freq - self.min_freq)
        return spec_rect.bottom() - t * spec_rect.height()

    def _y_to_freq(self, y, spec_rect):
        """Convert y coordinate to frequency."""
        if spec_rect.height() <= 0:
            return self.min_freq
        t = (spec_rect.bottom() - y) / spec_rect.height()
        return self.min_freq + t * (self.max_freq - self.min_freq)

    def _draw_detection_box(self, painter, det, idx, spec_rect):
        """Draw a single detection box."""
        start_s = det.get('start_seconds', 0)
        stop_s = det.get('stop_seconds', 0)

        # Use full frequency range if not specified
        min_freq = det.get('min_freq_hz', self.min_freq)
        max_freq = det.get('max_freq_hz', self.max_freq)

        # Check if in view
        if stop_s < self.view_start or start_s > self.view_end:
            return

        # Calculate box coordinates
        x1 = self._time_to_x(start_s, spec_rect)
        x2 = self._time_to_x(stop_s, spec_rect)
        y1 = self._freq_to_y(max_freq, spec_rect)
        y2 = self._freq_to_y(min_freq, spec_rect)

        # Clamp to spec_rect
        x1 = max(spec_rect.left(), min(spec_rect.right(), x1))
        x2 = max(spec_rect.left(), min(spec_rect.right(), x2))

        # Determine style based on status and selection
        is_current = (idx == self.current_detection_idx)
        status = det.get('status', 'pending')

        if is_current:
            color = QColor(255, 255, 255)
            width = 2
        elif status == 'accepted':
            color = QColor(16, 124, 16)
            width = 1.5
        elif status == 'rejected':
            color = QColor(209, 52, 56)
            width = 1.5
        else:
            color = QColor(0, 120, 212)
            width = 1

        pen = QPen(color, width)
        if not is_current:
            pen.setStyle(Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(QRectF(x1, y1, x2 - x1, y2 - y1))

    def _draw_temp_box(self, painter, spec_rect):
        """Draw the box currently being drawn."""
        x1 = self._time_to_x(self.draw_start[0], spec_rect)
        y1 = self._freq_to_y(self.draw_start[1], spec_rect)
        x2 = self._time_to_x(self.draw_current[0], spec_rect)
        y2 = self._freq_to_y(self.draw_current[1], spec_rect)

        pen = QPen(QColor(255, 255, 0), 2, Qt.DashLine)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawRect(QRectF(min(x1, x2), min(y1, y2), abs(x2 - x1), abs(y2 - y1)))

    def _draw_axes(self, painter, spec_rect):
        """Draw axis labels."""
        painter.setPen(QColor(200, 200, 200))
        font = QFont("Arial", 8)
        painter.setFont(font)

        # Time axis
        n_time_ticks = 5
        for i in range(n_time_ticks + 1):
            t = self.view_start + (self.view_end - self.view_start) * i / n_time_ticks
            x = self._time_to_x(t, spec_rect)
            painter.drawText(
                int(x - 20), int(spec_rect.bottom() + 15),
                40, 15, Qt.AlignCenter, f"{t:.2f}"
            )

        # Time label
        painter.drawText(
            int(spec_rect.center().x() - 30), int(spec_rect.bottom() + 25),
            60, 15, Qt.AlignCenter, "Time (s)"
        )

        # Frequency axis
        n_freq_ticks = 5
        for i in range(n_freq_ticks + 1):
            freq = self.min_freq + (self.max_freq - self.min_freq) * i / n_freq_ticks
            y = self._freq_to_y(freq, spec_rect)
            painter.drawText(
                0, int(y - 7), 45, 14, Qt.AlignRight, f"{freq/1000:.0f}"
            )

        # Frequency label (rotated would be better but keep simple)
        painter.drawText(5, int(spec_rect.center().y() - 20), "kHz")

    def mousePressEvent(self, event):
        """Handle mouse press for box selection/adjustment."""
        if event.button() != Qt.LeftButton:
            return

        spec_rect = self._get_spec_rect()
        pos = event.pos()

        if not spec_rect.contains(pos):
            return

        time_s = self._x_to_time(pos.x(), spec_rect)
        freq_hz = self._y_to_freq(pos.y(), spec_rect)

        # Check if clicking on a detection edge for resizing or inside for moving
        edge, det_idx = self._find_edge_at_pos(pos, spec_rect)

        if edge:
            self.drag_mode = edge
            self.drag_detection_idx = det_idx
            self.drag_start = (time_s, freq_hz)

            # Store original box coords for move operation
            if edge == 'move' and det_idx is not None and det_idx < len(self.detections):
                det = self.detections[det_idx]
                self.drag_start_box = (
                    det.get('start_seconds', 0),
                    det.get('stop_seconds', 0),
                    det.get('min_freq_hz', self.min_freq),
                    det.get('max_freq_hz', self.max_freq)
                )
        else:
            # Check if clicking inside a detection (but not current one - that's handled by edge detection)
            det_idx = self._find_detection_at_pos(pos, spec_rect)
            if det_idx >= 0:
                self.detection_selected.emit(det_idx)
            else:
                # Start drawing new box
                self.is_drawing = True
                self.draw_start = (time_s, freq_hz)
                self.draw_current = (time_s, freq_hz)

        self.update()

    def mouseMoveEvent(self, event):
        """Handle mouse move for dragging/drawing."""
        spec_rect = self._get_spec_rect()
        pos = event.pos()

        if self.is_drawing:
            time_s = self._x_to_time(pos.x(), spec_rect)
            freq_hz = self._y_to_freq(pos.y(), spec_rect)
            self.draw_current = (time_s, freq_hz)
            self.update()
        elif self.drag_mode and self.drag_detection_idx is not None:
            self._handle_drag(pos, spec_rect)
        else:
            # Update cursor based on edge hover
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

        if self.is_drawing and self.draw_start and self.draw_current:
            # Emit new box
            start_t = min(self.draw_start[0], self.draw_current[0])
            end_t = max(self.draw_start[0], self.draw_current[0])
            min_f = min(self.draw_start[1], self.draw_current[1])
            max_f = max(self.draw_start[1], self.draw_current[1])

            if abs(end_t - start_t) > 0.001 and abs(max_f - min_f) > 100:
                # Valid box - emit adjustment for current detection
                if self.current_detection_idx >= 0:
                    self.box_adjusted.emit(
                        self.current_detection_idx,
                        start_t, end_t, min_f, max_f
                    )

        self.is_drawing = False
        self.draw_start = None
        self.draw_current = None
        self.drag_mode = None
        self.drag_detection_idx = None
        self.update()

    def _find_edge_at_pos(self, pos, spec_rect, threshold=8):
        """Find if position is near a detection edge or inside a box."""
        for i, det in enumerate(self.detections):
            start_s = det.get('start_seconds', 0)
            stop_s = det.get('stop_seconds', 0)
            min_freq = det.get('min_freq_hz', self.min_freq)
            max_freq = det.get('max_freq_hz', self.max_freq)

            x1 = self._time_to_x(start_s, spec_rect)
            x2 = self._time_to_x(stop_s, spec_rect)
            y1 = self._freq_to_y(max_freq, spec_rect)
            y2 = self._freq_to_y(min_freq, spec_rect)

            # Check edges first (priority over move)
            if abs(pos.x() - x1) < threshold and y1 <= pos.y() <= y2:
                return 'resize_left', i
            if abs(pos.x() - x2) < threshold and y1 <= pos.y() <= y2:
                return 'resize_right', i
            if abs(pos.y() - y1) < threshold and x1 <= pos.x() <= x2:
                return 'resize_top', i
            if abs(pos.y() - y2) < threshold and x1 <= pos.x() <= x2:
                return 'resize_bottom', i

            # Check if inside box (for moving) - only for current detection
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
        """Handle dragging for resize or move."""
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
            # Move entire box - calculate delta from drag start
            delta_t = time_s - self.drag_start[0]
            delta_f = freq_hz - self.drag_start[1]

            orig_start, orig_stop, orig_min_f, orig_max_f = self.drag_start_box
            box_duration = orig_stop - orig_start
            box_freq_range = orig_max_f - orig_min_f

            # Apply delta, keeping box within bounds
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

        self.box_adjusted.emit(
            self.drag_detection_idx,
            start_s, stop_s, min_freq, max_freq
        )


class USVInspectorWindow(QMainWindow):
    """Main window for USV Inspector tool."""

    def __init__(self):
        super().__init__()

        # Data storage
        self.detection_files = []
        self.current_file_idx = 0
        self.detections_df = None
        self.current_detection_idx = 0
        self.wav_path = None
        self.audio_data = None
        self.sample_rate = None

        # View settings
        self.view_window_s = 2.0

        # Playback state
        self.is_playing = False
        self.playback_speed = 0.1
        self.use_heterodyne = False

        self.init_ui()

    def init_ui(self):
        """Initialize the user interface."""
        self.setWindowTitle("USV Inspector - FNT")
        self.setGeometry(100, 100, 1400, 800)
        self.setMinimumSize(1000, 600)

        # Apply compact dark theme
        self.setStyleSheet("""
            QMainWindow { background-color: #2b2b2b; color: #cccccc; }
            QWidget { background-color: #2b2b2b; color: #cccccc; font-family: Arial; font-size: 11px; }
            QLabel { color: #cccccc; background-color: transparent; }
            QPushButton {
                background-color: #0078d4; color: white; border: none;
                padding: 4px 8px; border-radius: 3px; font-weight: bold;
                min-height: 18px; font-size: 10px;
            }
            QPushButton:hover { background-color: #106ebe; }
            QPushButton:pressed { background-color: #005a9e; }
            QPushButton:disabled { background-color: #3f3f3f; color: #888888; }
            QPushButton#accept_btn { background-color: #107c10; }
            QPushButton#accept_btn:hover { background-color: #0e6b0e; }
            QPushButton#reject_btn { background-color: #d13438; }
            QPushButton#reject_btn:hover { background-color: #a4262c; }
            QPushButton#small_btn { padding: 2px 6px; min-height: 16px; min-width: 24px; }
            QGroupBox {
                font-weight: bold; border: 1px solid #3f3f3f; border-radius: 3px;
                margin-top: 8px; padding-top: 6px; color: #cccccc; font-size: 10px;
            }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 3px; }
            QDoubleSpinBox, QSpinBox, QComboBox {
                padding: 2px 4px; border: 1px solid #3f3f3f; border-radius: 2px;
                background-color: #1e1e1e; color: #cccccc; min-width: 60px; font-size: 10px;
            }
            QProgressBar {
                border: 1px solid #3f3f3f; border-radius: 3px; text-align: center;
                background-color: #1e1e1e; color: #cccccc; max-height: 16px; font-size: 9px;
            }
            QProgressBar::chunk { background-color: #0078d4; border-radius: 2px; }
            QStatusBar { background-color: #1e1e1e; color: #cccccc; border-top: 1px solid #3f3f3f; font-size: 10px; }
            QScrollBar:horizontal {
                border: none; background: #1e1e1e; height: 14px; margin: 0;
            }
            QScrollBar::handle:horizontal {
                background: #0078d4; min-width: 30px; border-radius: 3px;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal { width: 0; }
        """)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Left panel (controls)
        left_panel = QWidget()
        left_panel.setFixedWidth(220)
        left_layout = QVBoxLayout(left_panel)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(3)

        # Input selection
        self._create_input_section(left_layout)

        # Navigation
        self._create_navigation_section(left_layout)

        # Current detection info
        self._create_detection_section(left_layout)

        # Actions
        self._create_action_section(left_layout)

        # Playback
        self._create_playback_section(left_layout)

        # Progress
        self._create_progress_section(left_layout)

        left_layout.addStretch()
        main_layout.addWidget(left_panel)

        # Right panel (spectrogram)
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(3)

        # Spectrogram widget
        self.spectrogram = SpectrogramWidget()
        self.spectrogram.detection_selected.connect(self.on_detection_clicked)
        self.spectrogram.box_adjusted.connect(self.on_box_adjusted)
        right_layout.addWidget(self.spectrogram, 1)

        # Navigation bar below spectrogram
        nav_bar = QWidget()
        nav_layout = QHBoxLayout(nav_bar)
        nav_layout.setContentsMargins(50, 0, 10, 0)
        nav_layout.setSpacing(5)

        # Pan left button
        self.btn_pan_left = QPushButton("<")
        self.btn_pan_left.setObjectName("small_btn")
        self.btn_pan_left.setFixedWidth(24)
        self.btn_pan_left.clicked.connect(self.pan_left)
        nav_layout.addWidget(self.btn_pan_left)

        # Time scrollbar
        self.time_scrollbar = QScrollBar(Qt.Horizontal)
        self.time_scrollbar.setMinimum(0)
        self.time_scrollbar.setMaximum(1000)
        self.time_scrollbar.valueChanged.connect(self.on_scrollbar_changed)
        nav_layout.addWidget(self.time_scrollbar, 1)

        # Pan right button
        self.btn_pan_right = QPushButton(">")
        self.btn_pan_right.setObjectName("small_btn")
        self.btn_pan_right.setFixedWidth(24)
        self.btn_pan_right.clicked.connect(self.pan_right)
        nav_layout.addWidget(self.btn_pan_right)

        # View window controls
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
        self.spin_view_window.setFixedWidth(70)
        self.spin_view_window.valueChanged.connect(self.on_view_window_changed)
        nav_layout.addWidget(self.spin_view_window)

        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setObjectName("small_btn")
        self.btn_zoom_in.setFixedWidth(24)
        self.btn_zoom_in.clicked.connect(self.zoom_in)
        nav_layout.addWidget(self.btn_zoom_in)

        right_layout.addWidget(nav_bar)
        main_layout.addWidget(right_panel, 1)

        # Status bar
        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage("Load detection CSV files to begin")

    def _create_input_section(self, layout):
        """Create input file selection section."""
        group = QGroupBox("Input")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(2)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(2)

        self.btn_add_folder = QPushButton("Folder")
        self.btn_add_folder.clicked.connect(self.add_folder)
        btn_row.addWidget(self.btn_add_folder)

        self.btn_add_files = QPushButton("Files")
        self.btn_add_files.clicked.connect(self.add_files)
        btn_row.addWidget(self.btn_add_files)

        self.btn_clear = QPushButton("Clear")
        self.btn_clear.clicked.connect(self.clear_files)
        self.btn_clear.setStyleSheet("background-color: #5c5c5c;")
        btn_row.addWidget(self.btn_clear)

        group_layout.addLayout(btn_row)

        self.lbl_files = QLabel("No files loaded")
        self.lbl_files.setStyleSheet("color: #999999; font-style: italic; font-size: 9px;")
        group_layout.addWidget(self.lbl_files)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _create_navigation_section(self, layout):
        """Create navigation controls."""
        group = QGroupBox("Navigation")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(2)

        # File navigation
        file_row = QHBoxLayout()
        file_row.setSpacing(2)

        self.btn_prev_file = QPushButton("<<")
        self.btn_prev_file.setObjectName("small_btn")
        self.btn_prev_file.clicked.connect(self.prev_file)
        self.btn_prev_file.setEnabled(False)
        file_row.addWidget(self.btn_prev_file)

        self.lbl_file_num = QLabel("File 0/0")
        self.lbl_file_num.setAlignment(Qt.AlignCenter)
        file_row.addWidget(self.lbl_file_num, 1)

        self.btn_next_file = QPushButton(">>")
        self.btn_next_file.setObjectName("small_btn")
        self.btn_next_file.clicked.connect(self.next_file)
        self.btn_next_file.setEnabled(False)
        file_row.addWidget(self.btn_next_file)

        group_layout.addLayout(file_row)

        # Detection navigation
        det_row = QHBoxLayout()
        det_row.setSpacing(2)

        self.btn_prev = QPushButton("<")
        self.btn_prev.setObjectName("small_btn")
        self.btn_prev.clicked.connect(self.prev_detection)
        self.btn_prev.setEnabled(False)
        det_row.addWidget(self.btn_prev)

        self.lbl_det_num = QLabel("Det 0/0")
        self.lbl_det_num.setAlignment(Qt.AlignCenter)
        det_row.addWidget(self.lbl_det_num, 1)

        self.btn_next = QPushButton(">")
        self.btn_next.setObjectName("small_btn")
        self.btn_next.clicked.connect(self.next_detection)
        self.btn_next.setEnabled(False)
        det_row.addWidget(self.btn_next)

        group_layout.addLayout(det_row)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _create_detection_section(self, layout):
        """Create current detection info section."""
        group = QGroupBox("Current Detection")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(2)

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
        group_layout.addLayout(time_row)

        stop_row = QHBoxLayout()
        stop_row.setSpacing(2)
        stop_row.addWidget(QLabel("Stop:"))
        self.spin_stop = QDoubleSpinBox()
        self.spin_stop.setDecimals(4)
        self.spin_stop.setRange(0, 9999)
        self.spin_stop.setSuffix(" s")
        self.spin_stop.valueChanged.connect(self.on_time_changed)
        stop_row.addWidget(self.spin_stop, 1)
        group_layout.addLayout(stop_row)

        # Frequency (new)
        freq_row = QHBoxLayout()
        freq_row.setSpacing(2)
        freq_row.addWidget(QLabel("Min F:"))
        self.spin_min_freq = QSpinBox()
        self.spin_min_freq.setRange(0, 150000)
        self.spin_min_freq.setSingleStep(1000)
        self.spin_min_freq.setSuffix(" Hz")
        self.spin_min_freq.valueChanged.connect(self.on_freq_changed)
        freq_row.addWidget(self.spin_min_freq, 1)
        group_layout.addLayout(freq_row)

        maxf_row = QHBoxLayout()
        maxf_row.setSpacing(2)
        maxf_row.addWidget(QLabel("Max F:"))
        self.spin_max_freq = QSpinBox()
        self.spin_max_freq.setRange(0, 150000)
        self.spin_max_freq.setSingleStep(1000)
        self.spin_max_freq.setSuffix(" Hz")
        self.spin_max_freq.valueChanged.connect(self.on_freq_changed)
        maxf_row.addWidget(self.spin_max_freq, 1)
        group_layout.addLayout(maxf_row)

        # Info
        self.lbl_det_info = QLabel("Peak: -- Hz | Dur: -- ms")
        self.lbl_det_info.setStyleSheet("color: #999999; font-size: 9px;")
        group_layout.addWidget(self.lbl_det_info)

        # Status
        status_row = QHBoxLayout()
        status_row.addWidget(QLabel("Status:"))
        self.lbl_status = QLabel("--")
        self.lbl_status.setStyleSheet("font-weight: bold;")
        status_row.addWidget(self.lbl_status)
        status_row.addStretch()
        group_layout.addLayout(status_row)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _create_action_section(self, layout):
        """Create action buttons."""
        group = QGroupBox("Actions")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(2)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(2)

        self.btn_accept = QPushButton("Accept")
        self.btn_accept.setObjectName("accept_btn")
        self.btn_accept.clicked.connect(self.accept_detection)
        self.btn_accept.setEnabled(False)
        btn_row.addWidget(self.btn_accept)

        self.btn_reject = QPushButton("Reject")
        self.btn_reject.setObjectName("reject_btn")
        self.btn_reject.clicked.connect(self.reject_detection)
        self.btn_reject.setEnabled(False)
        btn_row.addWidget(self.btn_reject)

        group_layout.addLayout(btn_row)

        # Add USV button
        btn_row2 = QHBoxLayout()
        btn_row2.setSpacing(2)

        self.btn_add_usv = QPushButton("+ Add USV")
        self.btn_add_usv.setStyleSheet("background-color: #6b4c9a;")  # Purple
        self.btn_add_usv.clicked.connect(self.add_new_usv)
        self.btn_add_usv.setEnabled(False)
        btn_row2.addWidget(self.btn_add_usv)

        self.btn_delete_usv = QPushButton("Delete")
        self.btn_delete_usv.setStyleSheet("background-color: #5c5c5c;")
        self.btn_delete_usv.clicked.connect(self.delete_current_usv)
        self.btn_delete_usv.setEnabled(False)
        btn_row2.addWidget(self.btn_delete_usv)

        group_layout.addLayout(btn_row2)

        # Delete pending button (for cleanup before training)
        self.btn_delete_pending = QPushButton("Delete All Pending")
        self.btn_delete_pending.setStyleSheet("background-color: #8b4513;")  # Brown
        self.btn_delete_pending.clicked.connect(self.delete_all_pending)
        self.btn_delete_pending.setEnabled(False)
        group_layout.addWidget(self.btn_delete_pending)

        # Draw box instruction
        draw_label = QLabel("Drag edges to resize, drag center to move")
        draw_label.setStyleSheet("color: #999999; font-size: 8px; font-style: italic;")
        draw_label.setWordWrap(True)
        group_layout.addWidget(draw_label)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _create_playback_section(self, layout):
        """Create playback controls."""
        group = QGroupBox("Playback")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(2)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(2)

        self.btn_play = QPushButton("Play")
        self.btn_play.clicked.connect(self.toggle_playback)
        self.btn_play.setEnabled(False)
        btn_row.addWidget(self.btn_play)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.clicked.connect(self.stop_playback)
        self.btn_stop.setEnabled(False)
        self.btn_stop.setStyleSheet("background-color: #5c5c5c;")
        btn_row.addWidget(self.btn_stop)

        group_layout.addLayout(btn_row)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(2)
        mode_row.addWidget(QLabel("Mode:"))
        self.combo_mode = QComboBox()
        self.combo_mode.addItem("0.01x", 0.01)
        self.combo_mode.addItem("0.05x", 0.05)
        self.combo_mode.addItem("0.1x", 0.1)
        self.combo_mode.addItem("0.2x", 0.2)
        self.combo_mode.addItem("0.5x", 0.5)
        self.combo_mode.addItem("1x", 1.0)
        self.combo_mode.addItem("Hetero", "heterodyne")
        self.combo_mode.setCurrentIndex(2)  # Default to 0.1x
        self.combo_mode.currentIndexChanged.connect(self.on_playback_mode_changed)
        mode_row.addWidget(self.combo_mode, 1)
        group_layout.addLayout(mode_row)

        if not HAS_SOUNDDEVICE:
            warn = QLabel("sounddevice not installed")
            warn.setStyleSheet("color: #d13438; font-size: 8px;")
            group_layout.addWidget(warn)

        group.setLayout(group_layout)
        layout.addWidget(group)

    def _create_progress_section(self, layout):
        """Create progress and save section."""
        group = QGroupBox("Progress")
        group_layout = QVBoxLayout()
        group_layout.setSpacing(2)

        self.lbl_progress = QLabel("0/0 reviewed")
        group_layout.addWidget(self.lbl_progress)

        self.progress_bar = QProgressBar()
        self.progress_bar.setValue(0)
        group_layout.addWidget(self.progress_bar)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(2)

        self.btn_save = QPushButton("Save")
        self.btn_save.clicked.connect(self.save_inspected)
        self.btn_save.setEnabled(False)
        btn_row.addWidget(self.btn_save)

        self.btn_open = QPushButton("Open Folder")
        self.btn_open.clicked.connect(self.open_output_folder)
        self.btn_open.setEnabled(False)
        self.btn_open.setStyleSheet("background-color: #5c5c5c;")
        btn_row.addWidget(self.btn_open)

        group_layout.addLayout(btn_row)

        # Export and train buttons
        ml_row = QHBoxLayout()
        ml_row.setSpacing(2)

        self.btn_export_training = QPushButton("Export Data")
        self.btn_export_training.setStyleSheet("background-color: #2d7d46;")  # Green
        self.btn_export_training.clicked.connect(self.export_training_data)
        self.btn_export_training.setEnabled(False)
        self.btn_export_training.setToolTip("Export labeled detections for ML training")
        ml_row.addWidget(self.btn_export_training)

        self.btn_train_model = QPushButton("Train Model")
        self.btn_train_model.setStyleSheet("background-color: #8b5a2b;")  # Brown
        self.btn_train_model.clicked.connect(self.train_classifier)
        self.btn_train_model.setEnabled(False)
        self.btn_train_model.setToolTip("Train Random Forest classifier on labeled data")
        ml_row.addWidget(self.btn_train_model)

        group_layout.addLayout(ml_row)

        group.setLayout(group_layout)
        layout.addWidget(group)

    # --- File Management ---

    def add_folder(self):
        """Add detection CSVs from folder."""
        folder = QFileDialog.getExistingDirectory(self, "Select Folder")
        if not folder:
            return

        csv_files = list(Path(folder).glob("*_usv_detections.csv"))
        if not csv_files:
            QMessageBox.warning(self, "No Files", "No *_usv_detections.csv files found.")
            return

        self.detection_files.extend([str(f) for f in csv_files])
        self.update_file_list()

    def add_files(self):
        """Add individual detection files."""
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Detection CSVs", "", "CSV Files (*.csv)"
        )
        if files:
            self.detection_files.extend(files)
            self.update_file_list()

    def clear_files(self):
        """Clear all files."""
        self.detection_files = []
        self.current_file_idx = 0
        self.detections_df = None
        self.current_detection_idx = 0
        self.wav_path = None
        self.audio_data = None
        self.update_file_list()
        self.spectrogram.set_audio_data(None, None)
        self.spectrogram.set_detections([], -1)

    def update_file_list(self):
        """Update file list display."""
        n = len(self.detection_files)
        if n == 0:
            self.lbl_files.setText("No files loaded")
            self.disable_controls()
        else:
            self.lbl_files.setText(f"{n} file(s) loaded")
            self.load_current_file()

    def load_current_file(self):
        """Load current detection file."""
        if not self.detection_files:
            return

        csv_path = self.detection_files[self.current_file_idx]

        try:
            self.detections_df = pd.read_csv(csv_path)

            # Add columns if missing
            if 'status' not in self.detections_df.columns:
                self.detections_df['status'] = 'pending'
            if 'min_freq_hz' not in self.detections_df.columns:
                self.detections_df['min_freq_hz'] = 20000  # Default min
            if 'max_freq_hz' not in self.detections_df.columns:
                # Use peak_freq as estimate if available
                if 'peak_freq_hz' in self.detections_df.columns:
                    self.detections_df['max_freq_hz'] = self.detections_df['peak_freq_hz'] + 10000
                else:
                    self.detections_df['max_freq_hz'] = 80000

            # Find WAV file
            self.wav_path = self.find_wav_file(csv_path)
            if self.wav_path and os.path.exists(self.wav_path):
                self.load_audio(self.wav_path)
            else:
                self.audio_data = None
                self.status_bar.showMessage(f"Warning: WAV not found for {os.path.basename(csv_path)}")

            self.current_detection_idx = 0
            self.update_display()
            self.enable_controls()

            n = len(self.detection_files)
            self.lbl_file_num.setText(f"File {self.current_file_idx + 1}/{n}")
            self.btn_prev_file.setEnabled(self.current_file_idx > 0)
            self.btn_next_file.setEnabled(self.current_file_idx < n - 1)

            self.status_bar.showMessage(f"Loaded: {os.path.basename(csv_path)}")

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load:\n{str(e)}")

    def find_wav_file(self, csv_path):
        """Find associated WAV file."""
        csv_name = os.path.basename(csv_path)
        csv_dir = os.path.dirname(csv_path)

        if csv_name.endswith("_usv_detections.csv"):
            base = csv_name[:-19]
        else:
            base = os.path.splitext(csv_name)[0]

        for ext in [".wav", ".WAV"]:
            path = os.path.join(csv_dir, base + ext)
            if os.path.exists(path):
                return path
        return None

    def load_audio(self, wav_path):
        """Load audio file."""
        try:
            if HAS_SOUNDFILE:
                try:
                    self.audio_data, self.sample_rate = sf.read(wav_path, dtype='float32')
                except:
                    self.audio_data, self.sample_rate = self._load_with_ffmpeg(wav_path)
            else:
                self.audio_data, self.sample_rate = self._load_with_ffmpeg(wav_path)

            if self.audio_data.ndim > 1:
                self.audio_data = np.mean(self.audio_data, axis=1)

            # Pass audio to spectrogram widget for lazy computation
            self.spectrogram.set_audio_data(self.audio_data, self.sample_rate)

        except Exception as e:
            self.status_bar.showMessage(f"Audio load error: {str(e)}")
            self.audio_data = None

    def _load_with_ffmpeg(self, filepath):
        """Load audio via ffmpeg."""
        import subprocess

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
                'ffmpeg', '-i', filepath, '-f', 'f32le',
                '-acodec', 'pcm_f32le', '-ac', '1', '-y', temp_path
            ]
            subprocess.run(convert_cmd, capture_output=True, check=True)
            audio = np.fromfile(temp_path, dtype=np.float32)
        finally:
            if os.path.exists(temp_path):
                os.unlink(temp_path)

        return audio, sr

    # Note: compute_spectrogram removed - now using lazy computation in SpectrogramWidget

    # --- Display ---

    def update_display(self):
        """Update all display elements."""
        if self.detections_df is None or len(self.detections_df) == 0:
            return

        n_det = len(self.detections_df)
        det = self.detections_df.iloc[self.current_detection_idx]

        # Navigation label
        self.lbl_det_num.setText(f"Det {self.current_detection_idx + 1}/{n_det}")

        # Time spinboxes
        self.spin_start.blockSignals(True)
        self.spin_stop.blockSignals(True)
        self.spin_start.setValue(det['start_seconds'])
        self.spin_stop.setValue(det['stop_seconds'])
        self.spin_start.blockSignals(False)
        self.spin_stop.blockSignals(False)

        # Frequency spinboxes
        self.spin_min_freq.blockSignals(True)
        self.spin_max_freq.blockSignals(True)
        self.spin_min_freq.setValue(int(det.get('min_freq_hz', 20000)))
        self.spin_max_freq.setValue(int(det.get('max_freq_hz', 80000)))
        self.spin_min_freq.blockSignals(False)
        self.spin_max_freq.blockSignals(False)

        # Info
        peak = det.get('peak_freq_hz', 0)
        dur = det.get('duration_ms', 0)
        self.lbl_det_info.setText(f"Peak: {peak:.0f} Hz | Dur: {dur:.1f} ms")

        # Status
        status = det.get('status', 'pending')
        self.lbl_status.setText(status.capitalize())
        if status == 'accepted':
            self.lbl_status.setStyleSheet("font-weight: bold; color: #107c10;")
        elif status == 'rejected':
            self.lbl_status.setStyleSheet("font-weight: bold; color: #d13438;")
        else:
            self.lbl_status.setStyleSheet("font-weight: bold; color: #999999;")

        # Progress
        self.update_progress()

        # Update spectrogram view
        self.update_spectrogram_view()

        # Navigation buttons
        self.btn_prev.setEnabled(self.current_detection_idx > 0)
        self.btn_next.setEnabled(self.current_detection_idx < n_det - 1)

    def update_progress(self):
        """Update progress display."""
        if self.detections_df is None:
            return

        total = len(self.detections_df)
        reviewed = len(self.detections_df[self.detections_df['status'] != 'pending'])
        self.lbl_progress.setText(f"{reviewed}/{total} reviewed")
        self.progress_bar.setMaximum(total)
        self.progress_bar.setValue(reviewed)

    def update_spectrogram_view(self):
        """Update spectrogram with current detection centered."""
        if self.detections_df is None:
            return

        det = self.detections_df.iloc[self.current_detection_idx]
        det_start = det['start_seconds']
        det_stop = det['stop_seconds']
        det_center = (det_start + det_stop) / 2

        # Calculate view window
        window = self.spin_view_window.value()
        view_start = det_center - window / 2
        view_end = det_center + window / 2

        # Clamp to audio bounds
        total_dur = self.spectrogram.get_total_duration()
        if total_dur > 0:
            view_start = max(0, view_start)
            view_end = min(total_dur, view_end)

        self.spectrogram.set_view_range(view_start, view_end)

        # Update scrollbar
        self._update_scrollbar()

        # Update detections on spectrogram
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

        # Map center to scrollbar position (0-1000)
        pos = int(view_center / total_dur * 1000)
        self.time_scrollbar.blockSignals(True)
        self.time_scrollbar.setValue(pos)
        self.time_scrollbar.blockSignals(False)

    def enable_controls(self):
        """Enable controls."""
        has_det = self.detections_df is not None and len(self.detections_df) > 0
        has_audio = self.audio_data is not None
        has_pending = (has_det and
                       (self.detections_df['status'] == 'pending').any())
        has_labeled = (has_det and
                       (self.detections_df['status'].isin(['accepted', 'rejected'])).any())
        # Need at least 10 labeled samples to train
        n_labeled = 0
        if has_det:
            n_labeled = (self.detections_df['status'].isin(['accepted', 'rejected'])).sum()
        can_train = has_labeled and has_audio and n_labeled >= 10

        self.btn_accept.setEnabled(has_det)
        self.btn_reject.setEnabled(has_det)
        self.btn_add_usv.setEnabled(has_audio)  # Can add USV if audio is loaded
        self.btn_delete_usv.setEnabled(has_det)
        self.btn_delete_pending.setEnabled(has_pending)
        self.btn_export_training.setEnabled(has_labeled and has_audio)
        self.btn_train_model.setEnabled(can_train)
        self.btn_save.setEnabled(has_det or has_audio)  # Can save even with manual additions
        self.btn_open.setEnabled(has_det or has_audio)
        self.btn_play.setEnabled(has_audio and HAS_SOUNDDEVICE)
        self.btn_stop.setEnabled(has_audio and HAS_SOUNDDEVICE)

    def disable_controls(self):
        """Disable controls."""
        self.btn_prev.setEnabled(False)
        self.btn_next.setEnabled(False)
        self.btn_prev_file.setEnabled(False)
        self.btn_next_file.setEnabled(False)
        self.btn_accept.setEnabled(False)
        self.btn_reject.setEnabled(False)
        self.btn_add_usv.setEnabled(False)
        self.btn_delete_usv.setEnabled(False)
        self.btn_delete_pending.setEnabled(False)
        self.btn_export_training.setEnabled(False)
        self.btn_train_model.setEnabled(False)
        self.btn_save.setEnabled(False)
        self.btn_open.setEnabled(False)
        self.btn_play.setEnabled(False)
        self.btn_stop.setEnabled(False)

    # --- Navigation ---

    def prev_detection(self):
        if self.current_detection_idx > 0:
            self.current_detection_idx -= 1
            self.update_display()

    def next_detection(self):
        if self.detections_df is not None and self.current_detection_idx < len(self.detections_df) - 1:
            self.current_detection_idx += 1
            self.update_display()

    def prev_file(self):
        if self.current_file_idx > 0:
            self.save_inspected(show_message=False)
            self.current_file_idx -= 1
            self.load_current_file()

    def next_file(self):
        if self.current_file_idx < len(self.detection_files) - 1:
            self.save_inspected(show_message=False)
            self.current_file_idx += 1
            self.load_current_file()

    def on_detection_clicked(self, idx):
        """Handle click on detection in spectrogram."""
        if 0 <= idx < len(self.detections_df):
            self.current_detection_idx = idx
            self.update_display()

    def on_box_adjusted(self, idx, start_s, stop_s, min_freq, max_freq):
        """Handle box adjustment from spectrogram."""
        if self.detections_df is None or idx < 0 or idx >= len(self.detections_df):
            return

        self.detections_df.at[idx, 'start_seconds'] = start_s
        self.detections_df.at[idx, 'stop_seconds'] = stop_s
        self.detections_df.at[idx, 'min_freq_hz'] = min_freq
        self.detections_df.at[idx, 'max_freq_hz'] = max_freq
        self.detections_df.at[idx, 'duration_ms'] = (stop_s - start_s) * 1000

        if idx == self.current_detection_idx:
            # Update UI without re-centering the view (important during drag!)
            self._update_detection_info_only()

        # Always update the detection boxes on spectrogram
        self._update_detection_boxes()

    def _update_detection_info_only(self):
        """Update detection info panel without changing spectrogram view."""
        if self.detections_df is None or len(self.detections_df) == 0:
            return

        det = self.detections_df.iloc[self.current_detection_idx]

        # Update spinboxes without triggering callbacks
        self.spin_start.blockSignals(True)
        self.spin_stop.blockSignals(True)
        self.spin_min_freq.blockSignals(True)
        self.spin_max_freq.blockSignals(True)

        self.spin_start.setValue(det['start_seconds'])
        self.spin_stop.setValue(det['stop_seconds'])
        self.spin_min_freq.setValue(int(det.get('min_freq_hz', 20000)))
        self.spin_max_freq.setValue(int(det.get('max_freq_hz', 80000)))

        self.spin_start.blockSignals(False)
        self.spin_stop.blockSignals(False)
        self.spin_min_freq.blockSignals(False)
        self.spin_max_freq.blockSignals(False)

        # Update info label
        peak = det.get('peak_freq_hz', 0)
        dur = det.get('duration_ms', 0)
        self.lbl_det_info.setText(f"Peak: {peak:.0f} Hz | Dur: {dur:.1f} ms")

    def _update_detection_boxes(self):
        """Update detection boxes on spectrogram without changing view."""
        if self.detections_df is None:
            return

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

    # --- View Controls ---

    def on_scrollbar_changed(self, value):
        """Handle scrollbar change."""
        total_dur = self.spectrogram.get_total_duration()
        if total_dur <= 0:
            return

        # Map scrollbar (0-1000) to center position
        center = value / 1000.0 * total_dur
        window = self.spin_view_window.value()

        view_start = max(0, center - window / 2)
        view_end = min(total_dur, center + window / 2)

        self.spectrogram.set_view_range(view_start, view_end)

    def pan_left(self):
        """Pan view left."""
        view_start, view_end = self.spectrogram.get_view_range()
        window = view_end - view_start
        shift = window * 0.3

        new_start = max(0, view_start - shift)
        new_end = new_start + window

        self.spectrogram.set_view_range(new_start, new_end)
        self._update_scrollbar()

    def pan_right(self):
        """Pan view right."""
        view_start, view_end = self.spectrogram.get_view_range()
        window = view_end - view_start
        shift = window * 0.3
        total = self.spectrogram.get_total_duration()

        new_end = min(total, view_end + shift)
        new_start = new_end - window

        self.spectrogram.set_view_range(new_start, new_end)
        self._update_scrollbar()

    def zoom_in(self):
        """Zoom in (decrease view window) maintaining current center."""
        current_start, current_end = self.spectrogram.get_view_range()
        center = (current_start + current_end) / 2
        new_window = max(0.1, (current_end - current_start) / 1.5)

        # Update spinbox without triggering re-center
        self.spin_view_window.blockSignals(True)
        self.spin_view_window.setValue(new_window)
        self.spin_view_window.blockSignals(False)

        # Set view range directly
        total_dur = self.spectrogram.get_total_duration()
        new_start = max(0, center - new_window / 2)
        new_end = min(total_dur, center + new_window / 2)
        self.spectrogram.set_view_range(new_start, new_end)
        self._update_scrollbar()

    def zoom_out(self):
        """Zoom out (increase view window) maintaining current center."""
        current_start, current_end = self.spectrogram.get_view_range()
        center = (current_start + current_end) / 2
        total_dur = self.spectrogram.get_total_duration()
        new_window = min(total_dur if total_dur > 0 else 600, (current_end - current_start) * 1.5)

        # Update spinbox without triggering re-center
        self.spin_view_window.blockSignals(True)
        self.spin_view_window.setValue(new_window)
        self.spin_view_window.blockSignals(False)

        # Set view range directly
        new_start = max(0, center - new_window / 2)
        new_end = min(total_dur, center + new_window / 2)
        self.spectrogram.set_view_range(new_start, new_end)
        self._update_scrollbar()

    def on_view_window_changed(self):
        """Handle view window change from spinbox."""
        self.view_window_s = self.spin_view_window.value()
        self.update_spectrogram_view()

    # --- Time/Freq Adjustment ---

    def on_time_changed(self):
        """Handle time spinbox change."""
        if self.detections_df is None:
            return

        start = self.spin_start.value()
        stop = self.spin_stop.value()

        self.detections_df.at[self.current_detection_idx, 'start_seconds'] = start
        self.detections_df.at[self.current_detection_idx, 'stop_seconds'] = stop
        self.detections_df.at[self.current_detection_idx, 'duration_ms'] = (stop - start) * 1000

        self.update_spectrogram_view()

    def on_freq_changed(self):
        """Handle frequency spinbox change."""
        if self.detections_df is None:
            return

        min_f = self.spin_min_freq.value()
        max_f = self.spin_max_freq.value()

        self.detections_df.at[self.current_detection_idx, 'min_freq_hz'] = min_f
        self.detections_df.at[self.current_detection_idx, 'max_freq_hz'] = max_f

        self.update_spectrogram_view()

    # --- Actions ---

    def accept_detection(self):
        """Mark as accepted."""
        if self.detections_df is None:
            return
        self.detections_df.at[self.current_detection_idx, 'status'] = 'accepted'
        self.update_display()
        self.auto_advance()

    def reject_detection(self):
        """Mark as rejected."""
        if self.detections_df is None:
            return
        self.detections_df.at[self.current_detection_idx, 'status'] = 'rejected'
        self.update_display()
        self.auto_advance()

    def add_new_usv(self):
        """Add a new USV detection at the center of the current view."""
        if self.audio_data is None:
            return

        # Get current view range
        view_start, view_end = self.spectrogram.get_view_range()
        view_center = (view_start + view_end) / 2
        view_duration = view_end - view_start

        # Create a new detection at center of view
        # Default size: 10% of view width, typical USV frequency range
        new_duration = min(0.05, view_duration * 0.1)  # 50ms or 10% of view
        new_start = view_center - new_duration / 2
        new_stop = view_center + new_duration / 2

        new_row = {
            'start_seconds': new_start,
            'stop_seconds': new_stop,
            'duration_ms': new_duration * 1000,
            'min_freq_hz': 30000,  # Typical USV range
            'max_freq_hz': 60000,
            'peak_freq_hz': 45000,
            'status': 'pending',
            'source': 'manual'  # Mark as manually added
        }

        # Create or append to dataframe
        if self.detections_df is None:
            self.detections_df = pd.DataFrame([new_row])
        else:
            self.detections_df = pd.concat([self.detections_df, pd.DataFrame([new_row])],
                                           ignore_index=True)

        # Select the new detection
        self.current_detection_idx = len(self.detections_df) - 1
        self.update_display()
        self.enable_controls()
        self.status_bar.showMessage(f"Added new USV detection (adjust box to fit)")

    def delete_current_usv(self):
        """Delete the current USV detection."""
        if self.detections_df is None or len(self.detections_df) == 0:
            return

        # Remove current detection
        self.detections_df = self.detections_df.drop(self.current_detection_idx).reset_index(drop=True)

        # Adjust current index
        if len(self.detections_df) == 0:
            self.current_detection_idx = 0
            self.detections_df = None
        elif self.current_detection_idx >= len(self.detections_df):
            self.current_detection_idx = len(self.detections_df) - 1

        self.update_display()
        self.enable_controls()
        self.status_bar.showMessage("Deleted USV detection")

    def delete_all_pending(self):
        """Delete all detections with 'pending' status.

        Useful for cleaning up before using labeled data for training,
        keeping only accepted (USV) and rejected (noise) examples.
        """
        if self.detections_df is None or len(self.detections_df) == 0:
            return

        # Count pending before deletion
        pending_mask = self.detections_df['status'] == 'pending'
        n_pending = pending_mask.sum()

        if n_pending == 0:
            self.status_bar.showMessage("No pending detections to delete")
            return

        # Confirm with user
        from PyQt5.QtWidgets import QMessageBox
        reply = QMessageBox.question(
            self, "Delete Pending",
            f"Delete {n_pending} pending detections?\n\n"
            f"This keeps only accepted and rejected detections for training.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )

        if reply != QMessageBox.Yes:
            return

        # Remove pending detections
        self.detections_df = self.detections_df[~pending_mask].reset_index(drop=True)

        # Handle empty result
        if len(self.detections_df) == 0:
            self.detections_df = None
            self.current_detection_idx = 0
        else:
            # Clamp current index
            self.current_detection_idx = min(self.current_detection_idx,
                                             len(self.detections_df) - 1)

        self.update_display()
        self.enable_controls()

        n_remaining = len(self.detections_df) if self.detections_df is not None else 0
        self.status_bar.showMessage(f"Deleted {n_pending} pending | {n_remaining} labeled detections remain")

    def auto_advance(self):
        """Advance to next detection."""
        if self.detections_df is not None and self.current_detection_idx < len(self.detections_df) - 1:
            self.current_detection_idx += 1
            self.update_display()

    # --- Training Data Export ---

    def export_training_data(self):
        """Export labeled detections as training data for ML models.

        Creates a folder with:
        - metadata.csv: detection info, labels, and extracted features
        - patches/: spectrogram images for each detection (PNG)
        - audio_clips/: audio segments for each detection (WAV)
        """
        if self.detections_df is None or self.audio_data is None:
            return

        # Filter to only labeled detections
        labeled_mask = self.detections_df['status'].isin(['accepted', 'rejected'])
        labeled_df = self.detections_df[labeled_mask].copy()

        if len(labeled_df) == 0:
            QMessageBox.warning(self, "No Data",
                               "No accepted or rejected detections to export.")
            return

        # Ask user for export folder
        folder = QFileDialog.getExistingDirectory(
            self, "Select Export Location",
            str(Path.home())
        )
        if not folder:
            return

        # Create project folder
        project_name = "fnt_usv_training_data"
        if self.wav_path:
            base_name = Path(self.wav_path).stem
            project_name = f"fnt_usv_training_{base_name}"

        export_dir = Path(folder) / project_name
        patches_dir = export_dir / "patches"
        audio_dir = export_dir / "audio_clips"

        export_dir.mkdir(exist_ok=True)
        patches_dir.mkdir(exist_ok=True)
        audio_dir.mkdir(exist_ok=True)

        self.status_bar.showMessage("Exporting training data...")
        QApplication.processEvents()

        # Extract features and save patches for each detection
        features_list = []
        n_total = len(labeled_df)

        for i, (idx, row) in enumerate(labeled_df.iterrows()):
            self.status_bar.showMessage(f"Exporting {i+1}/{n_total}...")
            QApplication.processEvents()

            # Extract features
            features = self._extract_detection_features(row)
            features['original_index'] = idx
            features['label'] = 'usv' if row['status'] == 'accepted' else 'noise'
            features['status'] = row['status']

            # Generate unique ID
            det_id = f"{features['label']}_{i:04d}"
            features['detection_id'] = det_id

            # Save spectrogram patch
            patch_path = patches_dir / f"{det_id}.png"
            self._save_spectrogram_patch(row, patch_path)
            features['patch_file'] = f"patches/{det_id}.png"

            # Save audio clip
            audio_path = audio_dir / f"{det_id}.wav"
            self._save_audio_clip(row, audio_path)
            features['audio_file'] = f"audio_clips/{det_id}.wav"

            features_list.append(features)

        # Create metadata DataFrame and save
        metadata_df = pd.DataFrame(features_list)

        # Reorder columns for clarity
        first_cols = ['detection_id', 'label', 'status', 'start_seconds', 'stop_seconds',
                      'duration_ms', 'min_freq_hz', 'max_freq_hz', 'peak_freq_hz']
        other_cols = [c for c in metadata_df.columns if c not in first_cols]
        metadata_df = metadata_df[first_cols + other_cols]

        metadata_df.to_csv(export_dir / "metadata.csv", index=False)

        # Save a summary
        n_usv = (metadata_df['label'] == 'usv').sum()
        n_noise = (metadata_df['label'] == 'noise').sum()

        summary = f"""FNT USV Training Data Export
============================
Source: {self.wav_path or 'Unknown'}
Sample Rate: {self.sample_rate} Hz
Total Duration: {len(self.audio_data) / self.sample_rate:.1f} s

Labeled Detections: {len(metadata_df)}
  - USV (accepted): {n_usv}
  - Noise (rejected): {n_noise}

Files:
  - metadata.csv: Features and labels for each detection
  - patches/: {len(metadata_df)} spectrogram images (128x128 PNG)
  - audio_clips/: {len(metadata_df)} audio segments (WAV)

Feature Columns:
  - duration_ms: Detection duration in milliseconds
  - bandwidth_hz: Frequency range (max - min)
  - center_freq_hz: Center frequency
  - spectral_centroid_hz: Power-weighted mean frequency
  - spectral_bandwidth_hz: Power-weighted frequency std
  - spectral_flatness: Measure of tonality (0=tonal, 1=noisy)
  - rms_power: Root mean square power
  - peak_power_db: Peak power in dB
  - freq_modulation_rate: Frequency change rate (Hz/s)
  - zero_crossing_rate: Rate of sign changes in waveform
"""
        with open(export_dir / "README.txt", 'w') as f:
            f.write(summary)

        self.status_bar.showMessage(f"Exported {len(metadata_df)} detections to {export_dir}")
        QMessageBox.information(
            self, "Export Complete",
            f"Exported {n_usv} USV and {n_noise} noise examples to:\n{export_dir}"
        )

    def _extract_detection_features(self, row):
        """Extract features from a detection for ML training.

        Returns dict of features useful for classification.
        """
        start_s = row['start_seconds']
        stop_s = row['stop_seconds']
        min_freq = row.get('min_freq_hz', 20000)
        max_freq = row.get('max_freq_hz', 80000)

        # Get audio segment
        start_sample = int(start_s * self.sample_rate)
        stop_sample = int(stop_s * self.sample_rate)
        segment = self.audio_data[start_sample:stop_sample]

        if len(segment) < 10:
            segment = np.zeros(100)  # Fallback for very short segments

        features = {
            'start_seconds': start_s,
            'stop_seconds': stop_s,
            'duration_ms': (stop_s - start_s) * 1000,
            'min_freq_hz': min_freq,
            'max_freq_hz': max_freq,
            'peak_freq_hz': row.get('peak_freq_hz', (min_freq + max_freq) / 2),
        }

        # Bandwidth and center frequency
        features['bandwidth_hz'] = max_freq - min_freq
        features['center_freq_hz'] = (min_freq + max_freq) / 2

        # Time-domain features
        features['rms_power'] = float(np.sqrt(np.mean(segment ** 2)))
        features['peak_power_db'] = float(20 * np.log10(np.max(np.abs(segment)) + 1e-10))
        features['zero_crossing_rate'] = float(np.sum(np.diff(np.sign(segment)) != 0) / len(segment))

        # Spectral features (from FFT)
        if len(segment) >= 256:
            # Compute power spectrum
            n_fft = min(1024, len(segment))
            freqs = np.fft.rfftfreq(n_fft, 1.0 / self.sample_rate)
            fft_mag = np.abs(np.fft.rfft(segment[:n_fft]))
            power = fft_mag ** 2

            # Filter to frequency range of interest
            freq_mask = (freqs >= min_freq * 0.5) & (freqs <= max_freq * 1.5)
            if np.any(freq_mask):
                freqs_roi = freqs[freq_mask]
                power_roi = power[freq_mask]
                power_roi = power_roi / (np.sum(power_roi) + 1e-10)  # Normalize

                # Spectral centroid (power-weighted mean frequency)
                features['spectral_centroid_hz'] = float(np.sum(freqs_roi * power_roi))

                # Spectral bandwidth (power-weighted std)
                centroid = features['spectral_centroid_hz']
                features['spectral_bandwidth_hz'] = float(
                    np.sqrt(np.sum(((freqs_roi - centroid) ** 2) * power_roi))
                )

                # Spectral flatness (geometric mean / arithmetic mean)
                # High = noisy, Low = tonal
                geo_mean = np.exp(np.mean(np.log(power_roi + 1e-10)))
                arith_mean = np.mean(power_roi)
                features['spectral_flatness'] = float(geo_mean / (arith_mean + 1e-10))
            else:
                features['spectral_centroid_hz'] = features['center_freq_hz']
                features['spectral_bandwidth_hz'] = features['bandwidth_hz'] / 2
                features['spectral_flatness'] = 0.5
        else:
            features['spectral_centroid_hz'] = features['center_freq_hz']
            features['spectral_bandwidth_hz'] = features['bandwidth_hz'] / 2
            features['spectral_flatness'] = 0.5

        # Frequency modulation rate (crude estimate from spectrogram)
        # Positive = upward sweep, Negative = downward sweep
        try:
            if len(segment) >= 512:
                f, t, Sxx = signal.spectrogram(segment, fs=self.sample_rate,
                                                nperseg=256, noverlap=200)
                # Find peak frequency at each time point
                freq_mask = (f >= min_freq * 0.8) & (f <= max_freq * 1.2)
                if np.any(freq_mask) and Sxx.shape[1] > 1:
                    Sxx_roi = Sxx[freq_mask, :]
                    f_roi = f[freq_mask]
                    peak_freqs = f_roi[np.argmax(Sxx_roi, axis=0)]
                    # Linear fit to get modulation rate
                    if len(peak_freqs) > 2:
                        slope, _ = np.polyfit(np.arange(len(peak_freqs)), peak_freqs, 1)
                        dt = (stop_s - start_s) / len(peak_freqs)
                        features['freq_modulation_rate'] = float(slope / dt) if dt > 0 else 0.0
                    else:
                        features['freq_modulation_rate'] = 0.0
                else:
                    features['freq_modulation_rate'] = 0.0
            else:
                features['freq_modulation_rate'] = 0.0
        except:
            features['freq_modulation_rate'] = 0.0

        return features

    def _save_spectrogram_patch(self, row, filepath, size=(128, 128)):
        """Save a spectrogram patch for a detection as PNG."""
        start_s = row['start_seconds']
        stop_s = row['stop_seconds']
        min_freq = row.get('min_freq_hz', 20000)
        max_freq = row.get('max_freq_hz', 80000)

        # Add padding around detection (20% on each side)
        duration = stop_s - start_s
        freq_range = max_freq - min_freq
        pad_t = duration * 0.2
        pad_f = freq_range * 0.2

        view_start = max(0, start_s - pad_t)
        view_end = min(len(self.audio_data) / self.sample_rate, stop_s + pad_t)

        # Extract audio segment
        start_sample = int(view_start * self.sample_rate)
        stop_sample = int(view_end * self.sample_rate)
        segment = self.audio_data[start_sample:stop_sample]

        if len(segment) < 256:
            # Create blank image for very short segments
            img = QImage(size[0], size[1], QImage.Format_RGB888)
            img.fill(QColor(30, 30, 30))
            img.save(str(filepath))
            return

        # Compute spectrogram
        nperseg = min(256, len(segment) // 4)
        nperseg = max(64, nperseg)
        noverlap = int(nperseg * 0.75)

        f, t, Sxx = signal.spectrogram(segment, fs=self.sample_rate,
                                        nperseg=nperseg, noverlap=noverlap)

        # Filter to frequency range
        freq_min_view = max(0, min_freq - pad_f)
        freq_max_view = min(self.sample_rate / 2, max_freq + pad_f)
        freq_mask = (f >= freq_min_view) & (f <= freq_max_view)

        if not np.any(freq_mask):
            freq_mask = np.ones(len(f), dtype=bool)

        Sxx_roi = Sxx[freq_mask, :]

        # Convert to dB and normalize
        spec_db = 10 * np.log10(Sxx_roi + 1e-10)
        vmin = np.percentile(spec_db, 5)
        vmax = np.percentile(spec_db, 99)
        normalized = np.clip((spec_db - vmin) / (vmax - vmin + 1e-10), 0, 1)
        indices = (normalized * 255).astype(np.uint8)
        indices = np.flipud(indices)  # Low freq at bottom

        # Apply colormap
        rgb_data = self.spectrogram.colormap_lut[indices]

        # Create QImage and resize
        h, w = indices.shape
        img = QImage(rgb_data.data, w, h, w * 3, QImage.Format_RGB888).copy()
        img = img.scaled(size[0], size[1], Qt.IgnoreAspectRatio, Qt.SmoothTransformation)
        img.save(str(filepath))

    def _save_audio_clip(self, row, filepath):
        """Save audio clip for a detection as WAV."""
        start_s = row['start_seconds']
        stop_s = row['stop_seconds']

        # Add small padding
        pad = 0.01  # 10ms
        start_s = max(0, start_s - pad)
        stop_s = min(len(self.audio_data) / self.sample_rate, stop_s + pad)

        start_sample = int(start_s * self.sample_rate)
        stop_sample = int(stop_s * self.sample_rate)
        segment = self.audio_data[start_sample:stop_sample]

        # Save as WAV
        try:
            from scipy.io import wavfile
            # Convert to int16 for WAV
            segment_int = (segment * 32767).astype(np.int16)
            wavfile.write(str(filepath), self.sample_rate, segment_int)
        except Exception as e:
            # Fallback: save as raw float32
            segment.astype(np.float32).tofile(str(filepath).replace('.wav', '.raw'))

    def train_classifier(self):
        """Train a Random Forest classifier on current labeled data."""
        if self.detections_df is None or self.audio_data is None:
            return

        # Check for sklearn
        try:
            from fnt.usv.usv_classifier import USVClassifier, USVClassifierConfig
        except ImportError:
            QMessageBox.warning(
                self, "Missing Dependencies",
                "scikit-learn and joblib are required for training.\n\n"
                "Install with: pip install scikit-learn joblib"
            )
            return

        # Filter to labeled detections
        labeled_mask = self.detections_df['status'].isin(['accepted', 'rejected'])
        labeled_df = self.detections_df[labeled_mask].copy()

        n_usv = (labeled_df['status'] == 'accepted').sum()
        n_noise = (labeled_df['status'] == 'rejected').sum()

        if len(labeled_df) < 10:
            QMessageBox.warning(
                self, "Insufficient Data",
                f"Need at least 10 labeled samples to train.\n"
                f"Currently have: {len(labeled_df)}"
            )
            return

        if n_usv < 3 or n_noise < 3:
            QMessageBox.warning(
                self, "Imbalanced Data",
                f"Need at least 3 samples of each class.\n"
                f"USV (accepted): {n_usv}\n"
                f"Noise (rejected): {n_noise}"
            )
            return

        # Ask user for save location
        folder = QFileDialog.getExistingDirectory(
            self, "Select Model Save Location",
            str(Path.home())
        )
        if not folder:
            return

        # Create model folder name
        model_name = "fnt_usv_classifier"
        if self.wav_path:
            base_name = Path(self.wav_path).stem
            model_name = f"fnt_usv_classifier_{base_name}"

        model_dir = Path(folder) / model_name

        self.status_bar.showMessage("Extracting features...")
        QApplication.processEvents()

        # Extract features for all labeled detections
        features_list = []
        for idx, row in labeled_df.iterrows():
            features = self._extract_detection_features(row)
            features['label'] = 'usv' if row['status'] == 'accepted' else 'noise'
            features_list.append(features)

        features_df = pd.DataFrame(features_list)

        self.status_bar.showMessage("Training classifier...")
        QApplication.processEvents()

        # Train classifier
        try:
            classifier = USVClassifier()
            metrics = classifier.train(features_df, features_df['label'])

            # Save model
            classifier.save(str(model_dir))

            # Show results
            report = (
                f"Training Complete!\n\n"
                f"Samples: {len(labeled_df)} ({n_usv} USV, {n_noise} noise)\n\n"
                f"Performance (test set):\n"
                f"  Accuracy:  {metrics['accuracy']:.1%}\n"
                f"  Precision: {metrics['precision']:.1%}\n"
                f"  Recall:    {metrics['recall']:.1%}\n"
                f"  F1 Score:  {metrics['f1']:.1%}\n\n"
                f"Cross-validation: {metrics['cv_mean']:.1%} (+/- {metrics['cv_std']:.1%})\n\n"
                f"Top Features:\n"
            )

            # Add top 5 features
            sorted_features = sorted(
                metrics['feature_importances'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            for name, imp in sorted_features:
                report += f"  {name}: {imp:.3f}\n"

            report += f"\nModel saved to:\n{model_dir}"

            self.status_bar.showMessage(f"Model saved to {model_dir}")
            QMessageBox.information(self, "Training Complete", report)

        except Exception as e:
            self.status_bar.showMessage(f"Training failed: {str(e)}")
            QMessageBox.critical(self, "Training Error", f"Training failed:\n{str(e)}")

    # --- Playback ---

    def on_playback_mode_changed(self):
        """Handle playback mode change."""
        mode = self.combo_mode.currentData()
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
        """Play currently visible spectrogram region."""
        if not HAS_SOUNDDEVICE or self.audio_data is None:
            return

        # Play what's visible on the spectrogram
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
                play_sr = self.sample_rate
            else:
                play_sr = int(self.sample_rate * self.playback_speed)

            self.is_playing = True
            self.btn_play.setText("Pause")
            sd.play(segment, play_sr)

            dur_ms = int(len(segment) / play_sr * 1000) + 100
            QTimer.singleShot(dur_ms, self._on_playback_done)

        except Exception as e:
            self.status_bar.showMessage(f"Playback error: {str(e)}")
            self.is_playing = False

    def stop_playback(self):
        """Stop playback."""
        if HAS_SOUNDDEVICE:
            sd.stop()
        self.is_playing = False
        self.btn_play.setText("Play")

    def _on_playback_done(self):
        """Playback finished."""
        self.is_playing = False
        self.btn_play.setText("Play")

    def _heterodyne(self, audio, carrier=40000):
        """Apply heterodyne."""
        t = np.arange(len(audio)) / self.sample_rate
        carrier_sig = np.cos(2 * np.pi * carrier * t)
        mixed = audio * carrier_sig

        sos = signal.butter(4, 15000, btype='low', fs=self.sample_rate, output='sos')
        filtered = signal.sosfilt(sos, mixed)
        filtered = filtered / (np.max(np.abs(filtered)) + 1e-10)
        return filtered

    # --- Save ---

    def save_inspected(self, show_message=True):
        """Save inspected file."""
        if self.detections_df is None or not self.detection_files:
            return

        csv_path = self.detection_files[self.current_file_idx]
        csv_dir = os.path.dirname(csv_path)
        csv_name = os.path.basename(csv_path)

        if csv_name.endswith("_usv_detections.csv"):
            out_name = csv_name.replace("_usv_detections.csv", "_inspected.csv")
        else:
            out_name = csv_name.replace(".csv", "_inspected.csv")

        out_path = os.path.join(csv_dir, out_name)

        try:
            self.detections_df.to_csv(out_path, index=False)
            if show_message:
                QMessageBox.information(self, "Saved", f"Saved to:\n{out_path}")
            self.status_bar.showMessage(f"Saved: {out_name}")
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Save failed:\n{str(e)}")

    def open_output_folder(self):
        """Open output folder."""
        if not self.detection_files:
            return

        folder = os.path.dirname(self.detection_files[self.current_file_idx])
        try:
            if sys.platform == 'darwin':
                os.system(f'open "{folder}"')
            elif sys.platform == 'win32':
                os.startfile(folder)
            else:
                os.system(f'xdg-open "{folder}"')
        except:
            pass


def main():
    app = QApplication(sys.argv)
    window = USVInspectorWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
