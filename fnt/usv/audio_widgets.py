"""
Shared audio visualization widgets for USV detection tools.

Contains SpectrogramWidget and WaveformOverviewWidget, used by both
the Classic Audio Detector and the Deep Audio Detector.
"""

import math

import numpy as np
from PyQt5.QtCore import Qt, pyqtSignal, QRectF
from PyQt5.QtGui import QFont, QColor, QPainter, QImage, QPen, QBrush
from PyQt5.QtWidgets import QWidget, QSizePolicy
from scipy import signal


# =============================================================================
# Spectrogram Widget
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

        # Filter overlay state
        self._filter_overlay_active = False
        self._filter_overlay_image = None
        self._filter_params = None
        self._raw_spec_db = None
        self._raw_spec_freqs = None

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
        if self._filter_overlay_active:
            self._compute_filter_overlay()
        self.update()

    def set_filter_overlay(self, active, params=None):
        """Toggle the DSP filter overlay on/off.

        Args:
            active: Whether the overlay is visible.
            params: Dict with keys: min_freq_hz, max_freq_hz,
                    noise_percentile, energy_threshold_db.
        """
        self._filter_overlay_active = active
        if params is not None:
            self._filter_params = params
        if active and self._raw_spec_db is not None:
            self._compute_filter_overlay()
        elif not active:
            self._filter_overlay_image = None
        self.update()

    def _compute_filter_overlay(self):
        """Compute the filtered spectrogram image showing what survives
        frequency masking, noise floor subtraction, and energy thresholding.
        """
        if self._raw_spec_db is None or self._filter_params is None:
            self._filter_overlay_image = None
            return

        spec = self._raw_spec_db.copy()  # (n_freq, n_time) in dB
        freqs = self._raw_spec_freqs
        p = self._filter_params

        min_f = p.get('min_freq_hz', 0)
        max_f = p.get('max_freq_hz', 200000)
        noise_pct = p.get('noise_percentile', 25.0)
        thresh_db = p.get('energy_threshold_db', 10.0)

        # 1. Frequency band mask — mark out-of-band rows as dead
        freq_in_band = (freqs >= min_f) & (freqs <= max_f)
        dead = np.zeros(spec.shape, dtype=bool)
        dead[~freq_in_band, :] = True

        # 2. Noise floor subtraction for in-band rows
        in_band_data = spec[freq_in_band, :]
        if in_band_data.size > 0:
            noise_floor = np.percentile(in_band_data, noise_pct,
                                        axis=1, keepdims=True)
            flattened = in_band_data - noise_floor

            # 3. Energy threshold — mark sub-threshold pixels as dead
            sub_threshold = flattened < thresh_db
            # Map back to full array indices
            in_band_indices = np.where(freq_in_band)[0]
            for local_i, global_i in enumerate(in_band_indices):
                dead[global_i, sub_threshold[local_i, :]] = True

        # 4. Render: surviving pixels use original dB values, dead → black
        # Normalize surviving pixels
        surviving = spec[~dead]
        if surviving.size == 0:
            # Nothing survives — all black
            h, w = spec.shape
            black = np.ascontiguousarray(np.zeros((h, w, 3), dtype=np.uint8))
            self._filter_overlay_image = QImage(
                black.data, w, h, w * 3, QImage.Format_RGB888
            ).copy()
            return

        vmin = np.percentile(surviving, 5)
        vmax = np.percentile(surviving, 99)
        normalized = np.clip((spec - vmin) / (vmax - vmin + 1e-10), 0, 1)
        indices = (normalized * 255).astype(np.uint8)

        # Flip vertically (freq increases upward)
        indices = np.flipud(indices)
        dead_flipped = np.flipud(dead)

        # Apply colormap, then black out dead pixels
        rgb = np.ascontiguousarray(self.colormap_lut[indices])
        rgb[dead_flipped] = [0, 0, 0]

        h, w = indices.shape
        self._filter_overlay_image = QImage(
            rgb.data, w, h, w * 3, QImage.Format_RGB888
        ).copy()

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

        # Cache raw dB data for filter overlay (before normalization)
        self._raw_spec_db = view_spec.copy()
        self._raw_spec_freqs = frequencies[freq_mask].copy()

        # Normalize and apply colormap
        vmin = np.percentile(view_spec, 5)
        vmax = np.percentile(view_spec, 99)
        normalized = np.clip((view_spec - vmin) / (vmax - vmin + 1e-10), 0, 1)
        indices = (normalized * 255).astype(np.uint8)
        indices = np.flipud(indices)

        rgb_data = np.ascontiguousarray(self.colormap_lut[indices])

        height, width = indices.shape
        self.spec_image = QImage(
            rgb_data.data, width, height, width * 3,
            QImage.Format_RGB888
        ).copy()

        self.cached_view_start = self.view_start
        self.cached_view_end = self.view_end
        self.cached_min_freq = self.min_freq
        self.cached_max_freq = self.max_freq

        # Recompute filter overlay if active
        if self._filter_overlay_active:
            self._compute_filter_overlay()

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
        display_image = (self._filter_overlay_image
                         if self._filter_overlay_active
                         and self._filter_overlay_image
                         else self.spec_image)
        scaled_image = display_image.scaled(
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
        elif status == 'harmonic':
            color = QColor(180, 100, 255, 180)  # Purple for harmonics
            label = "H"
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

        # Draw frequency contour dots+lines when peak_freq samples exist
        # (for accepted, pending, and harmonic detections)
        if status in ('accepted', 'pending', 'harmonic') and det.get('peak_freq_1'):
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
