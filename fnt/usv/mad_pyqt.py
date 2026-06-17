"""MAD (Mask Audio Detector) — full shell.

PyQt5 entry point for mask-based segmentation labeling / training /
inference. Parallels CAD/DAD in ergonomics:

* Project scaffolding + file navigation + spectrogram view
* Playback (Play/Stop, slow-down slider, Space)
* CAD/DAD keyboard shortcuts (←/→ pan, ↑/↓ zoom, N/B file prev-next)
* Brush / Eraser painting of pixel-level USV masks, with sibling-PNG
  persistence (``<base>_FNT_MAD_labels.png``)

Training (Phase 3) and blob-review inference (Phase 4) land later via
the Predict menu.

Run directly:
    python fnt/usv/mad_pyqt.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import List, Optional

import numpy as np
from PyQt5.QtCore import (
    Qt, QSettings, QThread, QTimer, QRectF, QPointF, pyqtSignal,
)
from PyQt5.QtGui import (
    QImage, QKeySequence, QPainter, QPen, QColor, QBrush, QPolygonF,
)
from PyQt5.QtWidgets import (
    QAction, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
    QGroupBox, QHBoxLayout, QInputDialog, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QRadioButton, QScrollArea, QScrollBar, QShortcut, QSizePolicy, QSlider,
    QSpinBox, QSplitter, QStatusBar, QTabWidget, QTextEdit, QVBoxLayout,
    QWidget,
)
from scipy import signal

from fnt.usv.audio_widgets import SpectrogramWidget, WaveformOverviewWidget
from fnt.usv.usv_detector.mad_labels import (
    pred_csv_sibling_path, pred_mask_sibling_path,
)
from fnt.usv.usv_detector.mad_project import (
    MADProjectConfig, PROJECT_INFO_FILENAME, create_mad_project,
)
from fnt.usv.usv_detector.spectrogram import load_audio

try:
    import sounddevice as sd
    HAS_SOUNDDEVICE = True
except Exception:
    HAS_SOUNDDEVICE = False

try:
    from PIL import Image as PILImage
    HAS_PIL = True
    # Same rationale as mad_labels.py — our own sibling PNGs can be huge.
    PILImage.MAX_IMAGE_PIXELS = None
except Exception:
    HAS_PIL = False

try:
    import cv2
    HAS_CV2 = True
except Exception:
    HAS_CV2 = False


RECENT_PROJECTS_KEY = "mad/recent_projects"
MAX_RECENT_PROJECTS = 8

# Per-pixel mask values
MASK_UNLABELED = 0
MASK_POSITIVE = 1   # user-painted target
MASK_NEGATIVE = 2   # auto-assigned inside committed time bands

LABEL_SUFFIX = "_FNT_MAD_labels.png"


# ======================================================================
# Paint-capable spectrogram subclass
# ======================================================================
class MADSpectrogramWidget(SpectrogramWidget):
    """SpectrogramWidget with brush/eraser painting over a spec-pixel mask.

    The mask lives in the full-file spec-pixel grid — its shape is
    ``(n_freq_bins, n_time_frames)`` derived from the project's
    ``nperseg / noverlap / nfft`` params. Painting maps the cursor's
    screen coordinates through ``_x_to_time`` / ``_y_to_freq`` down to
    the spec-pixel grid, so paint strokes remain consistent under
    zoom / pan and on different widget sizes.
    """

    # Emitted once per paint stroke (on mouse release) so the main
    # window can auto-save the sibling PNG.
    stroke_committed = pyqtSignal()
    # Emitted when SAM prompt points change so the main window can run a
    # (debounced) SAM2 prediction off the UI thread.
    sam_points_changed = pyqtSignal()
    # Emitted when the scroll wheel changes the brush radius (paint modes),
    # so the main window can keep its brush-radius spin box in sync.
    brush_radius_changed = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.paint_mode: Optional[str] = None  # 'brush' | 'eraser' | 'sam' | None
        self.brush_radius_px = 6              # in spec-pixel units

        # Confirmed positives for the current file (uint8 {0,1}), rebuilt on
        # load from the saved example store. Rendered magenta.
        self.mask: Optional[np.ndarray] = None
        # Pending (unconfirmed) mask being built by brush / eraser / SAM
        # (uint8 {0,1}). Rendered pale-yellow fill + outline until the user
        # confirms it with Enter.
        self._pending: Optional[np.ndarray] = None
        self.n_freq_bins: Optional[int] = None
        self.n_time_frames: Optional[int] = None
        self.hop: Optional[int] = None
        self.nperseg: Optional[int] = None
        self.noverlap: Optional[int] = None
        self.nfft: Optional[int] = None

        # SAM2 prompt points, in full-file spec-pixel coords (t_idx, f_idx).
        # The SAM proposal is written into ``self._pending``.
        self._sam_pos_pts: List[tuple] = []
        self._sam_neg_pts: List[tuple] = []

        self._mask_dirty = False
        self._painting = False
        self._last_paint_idx: Optional[tuple] = None
        # Accumulated wheel delta for slow, fine brush-radius adjustment.
        self._wheel_accum = 0

        # Cursor preview: screen-space last mouse pos for drawing the
        # brush-radius circle around the cursor in paint mode.
        self._cursor_pos = None
        self.setMouseTracking(True)

        self.mask_alpha = 0.45
        # Render mode: 'spec' | 'overlay' | 'mask_only'
        self.view_mode: str = 'overlay'
        # Predicted prob mask (float32 in [0,1]) + per-blob bbox list.
        self.pred_mask: Optional[np.ndarray] = None
        self.pred_blobs: List[dict] = []   # pixel-index bboxes
        self.pred_highlight_idx: Optional[int] = None

    # --- mask lifecycle -------------------------------------------------
    def init_mask(self, audio_len: int, sample_rate: int,
                  nperseg: int, noverlap: int, nfft: int) -> None:
        self.hop = max(1, nperseg - noverlap)
        self.nperseg = nperseg
        self.noverlap = noverlap
        self.nfft = nfft
        self.n_freq_bins = nfft // 2 + 1
        self.n_time_frames = max(
            1, (audio_len - nperseg) // self.hop + 1
        )
        self.mask = np.zeros(
            (self.n_freq_bins, self.n_time_frames), dtype=np.uint8
        )
        self._pending = np.zeros_like(self.mask)
        self._mask_dirty = False
        self.clear_sam_prompts()
        self.update()

    def set_mask(self, arr: np.ndarray) -> None:
        """Replace the confirmed mask (e.g. reconstructed from examples)."""
        if (self.n_freq_bins is None or self.n_time_frames is None):
            self.mask = arr.copy()
            return
        want_shape = (self.n_freq_bins, self.n_time_frames)
        if arr.shape == want_shape:
            self.mask = arr.astype(np.uint8, copy=False)
        else:
            padded = np.zeros(want_shape, dtype=np.uint8)
            h = min(arr.shape[0], want_shape[0])
            w = min(arr.shape[1], want_shape[1])
            padded[:h, :w] = arr[:h, :w]
            self.mask = padded
        self._pending = np.zeros_like(self.mask)
        self._mask_dirty = False
        self.update()

    def get_mask(self) -> Optional[np.ndarray]:
        return self.mask

    def is_mask_dirty(self) -> bool:
        return self._mask_dirty

    def clear_mask(self) -> None:
        if self.mask is None:
            return
        self.mask[:] = 0
        self._mask_dirty = True
        self.update()

    # --- predictions ---------------------------------------------------
    def set_view_mode(self, mode: str) -> None:
        if mode not in ('spec', 'overlay', 'mask_only'):
            mode = 'overlay'
        self.view_mode = mode
        self.update()

    def set_predicted_mask(self, arr: Optional[np.ndarray]) -> None:
        if arr is None:
            self.pred_mask = None
        else:
            self.pred_mask = arr.astype(np.float32, copy=False)
        self.update()

    def set_predicted_blobs(self, blobs: List[dict],
                            highlight_idx: Optional[int] = None) -> None:
        self.pred_blobs = list(blobs) if blobs else []
        self.pred_highlight_idx = highlight_idx
        self.update()

    def set_blob_highlight(self, idx: Optional[int]) -> None:
        self.pred_highlight_idx = idx
        self.update()

    # --- paint mode ----------------------------------------------------
    def set_paint_mode(self, mode: Optional[str]) -> None:
        if self.paint_mode == 'sam' and mode != 'sam':
            # Leaving SAM mode drops the prompt points but keeps the pending
            # mask, so the SAM proposal can be refined with the brush.
            self.clear_sam_prompts()
        self.paint_mode = mode
        if mode in ('brush', 'eraser'):
            # Hide the OS cursor — the paintEvent draws a brush-radius
            # circle instead, which doubles as a visual size indicator.
            self.setCursor(Qt.BlankCursor)
        elif mode == 'sam':
            self.setCursor(Qt.CrossCursor)
            self._cursor_pos = None
        else:
            self.setCursor(Qt.ArrowCursor)
            self._cursor_pos = None
        self.update()

    def set_brush_radius(self, r: int) -> None:
        self.brush_radius_px = max(1, int(r))

    # --- SAM2-assisted labeling ---------------------------------------
    def _visible_spec_bounds(self):
        """Return (t_start, t_end, f_start, f_end) spec-pixel bounds for
        the current view, or None if the mask grid isn't initialized."""
        if (self.hop is None or self.n_time_frames is None or
                self.n_freq_bins is None or self.sample_rate is None):
            return None
        t_start = int(self.view_start * self.sample_rate / self.hop)
        t_end = int(self.view_end * self.sample_rate / self.hop) + 1
        t_start = max(0, t_start)
        t_end = min(self.n_time_frames, t_end)
        nyq = self.sample_rate / 2.0
        f_start = int(self.min_freq / nyq * self.n_freq_bins)
        f_end = int(self.max_freq / nyq * self.n_freq_bins) + 1
        f_start = max(0, f_start)
        f_end = min(self.n_freq_bins, f_end)
        return t_start, t_end, f_start, f_end

    def render_sam_image(self):
        """Render the visible time window as an RGB image on the project's
        spec-pixel grid for SAM2.

        Returns ``(rgb, t_off)`` where ``rgb`` is an ``(n_freq_bins, W, 3)``
        uint8 array (row 0 = lowest frequency, i.e. NOT vertically flipped,
        so it shares the orientation of ``self.mask``), and ``t_off`` is the
        full-grid time-frame index of column 0. Returns ``None`` if no
        spectrogram is available.

        The segment starts exactly at ``t_off * hop`` so spectrogram column
        ``j`` corresponds to full-grid frame ``t_off + j``; the mask SAM2
        returns therefore drops straight onto ``self.mask``.
        """
        if (self.audio_data is None or self.mask is None or
                self.nperseg is None or self.nfft is None):
            return None
        bounds = self._visible_spec_bounds()
        if bounds is None:
            return None
        t_start, t_end, _f0, _f1 = bounds
        if t_end <= t_start:
            return None
        start_sample = t_start * self.hop
        end_sample = min(len(self.audio_data),
                         (t_end - 1) * self.hop + self.nperseg)
        segment = self.audio_data[start_sample:end_sample]
        if len(segment) < self.nperseg:
            return None
        noverlap = min(self.noverlap, self.nperseg - 1)
        _f, _t, Sxx = signal.spectrogram(
            segment, fs=self.sample_rate, nperseg=self.nperseg,
            noverlap=noverlap, nfft=self.nfft, window='hann',
        )
        spec_db = 10.0 * np.log10(Sxx + 1e-10)  # (n_freq_bins, ncols)
        vmin = np.percentile(spec_db, 5)
        vmax = np.percentile(spec_db, 99)
        norm = np.clip((spec_db - vmin) / (vmax - vmin + 1e-10), 0, 1)
        idx = (norm * 255).astype(np.uint8)
        rgb = np.ascontiguousarray(self.colormap_lut[idx])  # row 0 = low freq
        return rgb, t_start

    def add_sam_point(self, pos, positive: bool) -> None:
        """Record a SAM prompt point from a screen-space QPoint."""
        spec_rect = self._get_spec_rect()
        if not spec_rect.contains(pos):
            return
        idx = self._screen_to_spec_idx(pos.x(), pos.y(), spec_rect)
        if idx is None:
            return
        (self._sam_pos_pts if positive else self._sam_neg_pts).append(idx)
        self.sam_points_changed.emit()
        self.update()

    def get_sam_prompts(self):
        """Return (positive_pts, negative_pts) in full-grid (t_idx, f_idx)."""
        return list(self._sam_pos_pts), list(self._sam_neg_pts)

    def has_sam_prompts(self) -> bool:
        return bool(self._sam_pos_pts or self._sam_neg_pts)

    def set_sam_preview(self, mask: Optional[np.ndarray], t_off: int) -> None:
        """Write a SAM-proposed boolean mask crop into the pending buffer
        (replacing it), or clear pending when ``mask`` is None."""
        if self._pending is None:
            return
        self._pending[:] = 0
        if mask is not None:
            pv = np.asarray(mask) > 0
            h, w = pv.shape
            t0 = max(0, int(t_off))
            t1 = min(self.n_time_frames, int(t_off) + w)
            f0, f1 = 0, min(self.n_freq_bins, h)
            if t1 > t0 and f1 > f0:
                sub = pv[(f0):(f1), (t0 - int(t_off)):(t1 - int(t_off))]
                self._pending[f0:f1, t0:t1][sub] = 1
        self.update()

    def clear_sam_prompts(self) -> None:
        """Clear SAM prompt points only (pending mask is preserved)."""
        self._sam_pos_pts = []
        self._sam_neg_pts = []
        self.update()

    # --- pending mask lifecycle ---------------------------------------
    def has_pending(self) -> bool:
        return self._pending is not None and bool(self._pending.any())

    def get_pending(self) -> Optional[np.ndarray]:
        return self._pending

    def clear_pending(self) -> None:
        if self._pending is not None:
            self._pending[:] = 0
        self.update()

    def pending_bbox(self):
        """Return (f0, f1, t0, t1) [half-open] of pending pixels, or None."""
        if self._pending is None or not self._pending.any():
            return None
        fs = np.where(self._pending.any(axis=1))[0]
        ts = np.where(self._pending.any(axis=0))[0]
        return int(fs[0]), int(fs[-1]) + 1, int(ts[0]), int(ts[-1]) + 1

    def confirm_pending(self) -> bool:
        """Merge the pending mask into the confirmed buffer and clear pending."""
        if self.mask is None or self._pending is None or not self._pending.any():
            return False
        self.mask[self._pending > 0] = MASK_POSITIVE
        self._pending[:] = 0
        self._sam_pos_pts = []
        self._sam_neg_pts = []
        self.update()
        return True

    # --- coord mapping -------------------------------------------------
    def _screen_to_spec_idx(self, x, y, spec_rect):
        if (self.hop is None or self.n_time_frames is None or
                self.n_freq_bins is None or self.sample_rate is None):
            return None
        time_s = self._x_to_time(x, spec_rect)
        freq_hz = self._y_to_freq(y, spec_rect)
        t_idx = int(time_s * self.sample_rate / self.hop)
        f_idx = int(freq_hz / (self.sample_rate / 2) * self.n_freq_bins)
        t_idx = max(0, min(self.n_time_frames - 1, t_idx))
        f_idx = max(0, min(self.n_freq_bins - 1, f_idx))
        return t_idx, f_idx

    def _stamp(self, t_idx: int, f_idx: int) -> None:
        # Brush/eraser edit the PENDING mask (confirmed via Enter), not the
        # confirmed buffer.
        if self._pending is None:
            return
        value = 1 if self.paint_mode == 'brush' else 0
        r = self.brush_radius_px
        t0 = max(0, t_idx - r)
        t1 = min(self.n_time_frames, t_idx + r + 1)
        f0 = max(0, f_idx - r)
        f1 = min(self.n_freq_bins, f_idx + r + 1)
        if t1 <= t0 or f1 <= f0:
            return
        tt, ff = np.meshgrid(
            np.arange(t0, t1), np.arange(f0, f1), indexing='xy'
        )
        disk = (tt - t_idx) ** 2 + (ff - f_idx) ** 2 <= r ** 2
        self._pending[f0:f1, t0:t1][disk] = value

    def _stamp_line(self, a_idx, b_idx):
        """Stamp a line of brush dots between two spec-pixel coords."""
        if a_idx is None or b_idx is None:
            return
        at, af = a_idx
        bt, bf = b_idx
        n_steps = max(abs(bt - at), abs(bf - af)) + 1
        for i in range(n_steps + 1):
            alpha = i / max(1, n_steps)
            ti = int(at + (bt - at) * alpha)
            fi = int(af + (bf - af) * alpha)
            self._stamp(ti, fi)

    # --- mouse events --------------------------------------------------
    def mousePressEvent(self, event):
        if self.paint_mode == 'sam':
            if event.button() == Qt.LeftButton:
                self.add_sam_point(event.pos(), positive=True)
                return
            if event.button() == Qt.RightButton:
                self.add_sam_point(event.pos(), positive=False)
                return
        if (self.paint_mode in ('brush', 'eraser') and
                event.button() == Qt.LeftButton):
            spec_rect = self._get_spec_rect()
            if not spec_rect.contains(event.pos()):
                return
            idx = self._screen_to_spec_idx(
                event.pos().x(), event.pos().y(), spec_rect
            )
            if idx is None:
                return
            self._painting = True
            self._stamp(*idx)
            self._last_paint_idx = idx
            self._cursor_pos = event.pos()
            self.update()
            return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        # Track cursor for the preview circle even when no button is held.
        if self.paint_mode in ('brush', 'eraser'):
            self._cursor_pos = event.pos()
            self.update()
        if self._painting:
            spec_rect = self._get_spec_rect()
            idx = self._screen_to_spec_idx(
                event.pos().x(), event.pos().y(), spec_rect
            )
            if idx is not None:
                self._stamp_line(self._last_paint_idx, idx)
                self._last_paint_idx = idx
                self.update()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._painting and event.button() == Qt.LeftButton:
            self._painting = False
            self._last_paint_idx = None
            self.stroke_committed.emit()
            return
        super().mouseReleaseEvent(event)

    def leaveEvent(self, event):
        self._cursor_pos = None
        if self.paint_mode in ('brush', 'eraser'):
            self.update()
        super().leaveEvent(event)

    def wheelEvent(self, event):
        # In paint modes the wheel fine-tunes the brush radius instead of
        # zooming. One notch (120 units) = ±1 px, accumulated so trackpads
        # adjust at the same slow rate — easy to tune mid-stroke.
        if self.paint_mode in ('brush', 'eraser'):
            self._wheel_accum += event.angleDelta().y()
            changed = False
            while self._wheel_accum >= 120:
                self._wheel_accum -= 120
                self.brush_radius_px = min(64, self.brush_radius_px + 1)
                changed = True
            while self._wheel_accum <= -120:
                self._wheel_accum += 120
                self.brush_radius_px = max(1, self.brush_radius_px - 1)
                changed = True
            if changed:
                self.brush_radius_changed.emit(self.brush_radius_px)
                self.update()
            event.accept()
            return
        super().wheelEvent(event)

    # --- overlay render -----------------------------------------------
    def paintEvent(self, event):
        super().paintEvent(event)
        if (self.mask is None or self.spec_image is None or
                self.hop is None or self.sample_rate is None):
            return

        spec_rect = self._get_spec_rect()

        # Visible time-frame slice
        t_start = int(self.view_start * self.sample_rate / self.hop)
        t_end = int(self.view_end * self.sample_rate / self.hop) + 1
        t_start = max(0, t_start)
        t_end = min(self.n_time_frames, t_end)

        # Visible freq-bin slice
        nyq = self.sample_rate / 2.0
        f_start = int(self.min_freq / nyq * self.n_freq_bins)
        f_end = int(self.max_freq / nyq * self.n_freq_bins) + 1
        f_start = max(0, f_start)
        f_end = min(self.n_freq_bins, f_end)

        if t_end <= t_start or f_end <= f_start:
            return

        h = f_end - f_start
        w = t_end - t_start
        rgba = np.zeros((h, w, 4), dtype=np.uint8)

        pend_view = (self._pending[f_start:f_end, t_start:t_end] > 0
                     if self._pending is not None else None)
        if self.view_mode == 'mask_only':
            # Solid dark fill so the spec doesn't show through.
            rgba[:] = (32, 32, 32, 255)
            view_mask = self.mask[f_start:f_end, t_start:t_end]
            rgba[view_mask == MASK_POSITIVE] = (255, 0, 220, 255)
            if pend_view is not None:
                rgba[pend_view] = (255, 230, 90, 255)
        elif self.view_mode == 'overlay':
            view_mask = self.mask[f_start:f_end, t_start:t_end]
            pos_alpha = int(self.mask_alpha * 255)
            # Confirmed positives = magenta.
            rgba[view_mask == MASK_POSITIVE] = (255, 0, 220, pos_alpha)
            # Pending (unconfirmed) = pale-yellow fill (outline drawn below).
            if pend_view is not None:
                rgba[pend_view] = (255, 230, 90, max(60, pos_alpha - 50))
        # else: 'spec' — leave rgba all-zero so only the spec shows.

        # Predicted blob mask shading (cyan, low alpha) if present.
        if (self.view_mode != 'spec' and self.pred_mask is not None and
                self.pred_mask.shape == self.mask.shape):
            pview = self.pred_mask[f_start:f_end, t_start:t_end]
            if pview.size > 0:
                active = pview > 0
                cyan_alpha = (pview[active] * 180).astype(np.uint8)
                if active.any():
                    rgba[active, 0] = np.where(
                        rgba[active, 3] > 0, rgba[active, 0], 0
                    )
                    rgba[active, 1] = np.where(
                        rgba[active, 3] > 0, rgba[active, 1], 220
                    )
                    rgba[active, 2] = np.where(
                        rgba[active, 3] > 0, rgba[active, 2], 255
                    )
                    rgba[active, 3] = np.maximum(rgba[active, 3], cyan_alpha)

        rgba = np.flipud(rgba).copy()
        qimg = QImage(rgba.data, w, h, 4 * w, QImage.Format_RGBA8888).copy()
        scaled = qimg.scaled(
            int(spec_rect.width()), int(spec_rect.height()),
            Qt.IgnoreAspectRatio, Qt.FastTransformation,
        )
        painter = QPainter(self)
        painter.drawImage(spec_rect.topLeft(), scaled)

        def _t_to_x(t):
            frac = (t - t_start) / max(1, (t_end - t_start))
            return spec_rect.left() + frac * spec_rect.width()

        def _f_to_y(f):
            frac = (f - f_start) / max(1, (f_end - f_start))
            return spec_rect.bottom() - frac * spec_rect.height()

        # Thin yellow outline tracing the closed pending mask (MT-style).
        if (HAS_CV2 and self._pending is not None and pend_view is not None
                and pend_view.any()):
            cnts, _ = cv2.findContours(
                pend_view.astype(np.uint8), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            pen = QPen(QColor(255, 225, 60))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            for cnt in cnts:
                if len(cnt) < 2:
                    continue
                poly = QPolygonF()
                for pt in cnt[:, 0, :]:
                    # contour coords are in the visible slice (x=col, y=row)
                    poly.append(QPointF(
                        _t_to_x(t_start + int(pt[0])),
                        _f_to_y(f_start + int(pt[1])),
                    ))
                painter.drawPolygon(poly)

        # SAM prompt-point markers (green = positive, red = negative).
        if self.paint_mode == 'sam' and (self._sam_pos_pts or
                                         self._sam_neg_pts):
            marks = ([(p, QColor(40, 255, 40)) for p in self._sam_pos_pts] +
                     [(p, QColor(255, 60, 60)) for p in self._sam_neg_pts])
            for (t_idx, f_idx), col in marks:
                if not (t_start <= t_idx < t_end and f_start <= f_idx < f_end):
                    continue
                cx = _t_to_x(t_idx)
                cy = _f_to_y(f_idx)
                pen = QPen(QColor(0, 0, 0))
                pen.setWidth(1)
                painter.setPen(pen)
                painter.setBrush(QBrush(col))
                painter.drawEllipse(QRectF(cx - 4, cy - 4, 8, 8))

        # Draw highlighted predicted-blob bbox (if any) in bright green.
        if (self.pred_blobs and self.pred_highlight_idx is not None and
                0 <= self.pred_highlight_idx < len(self.pred_blobs)):
            b = self.pred_blobs[self.pred_highlight_idx]
            t0 = int(b['t_start']); t1 = int(b['t_end_exclusive'])
            f0 = int(b['f_low']);   f1 = int(b['f_high_exclusive'])
            if t0 < t_end and t1 > t_start and f0 < f_end and f1 > f_start:
                # Map to screen rect — use _x_to_time/_y_to_freq inverses via
                # linear interpolation into spec_rect.
                def t_to_x(t):
                    frac = (t - t_start) / max(1, (t_end - t_start))
                    return spec_rect.left() + frac * spec_rect.width()

                def f_to_y(f):
                    # Freq grows up; screen y grows down.
                    frac = (f - f_start) / max(1, (f_end - f_start))
                    return spec_rect.bottom() - frac * spec_rect.height()

                x0 = t_to_x(max(t0, t_start))
                x1 = t_to_x(min(t1, t_end))
                y1 = f_to_y(max(f0, f_start))
                y0 = f_to_y(min(f1, f_end))
                pen = QPen(QColor(80, 255, 100, 255))
                pen.setWidth(2)
                painter.setPen(pen)
                painter.setBrush(Qt.NoBrush)
                painter.drawRect(QRectF(x0, y0, x1 - x0, y1 - y0))

        # Brush / eraser cursor preview circle.
        if (self.paint_mode in ('brush', 'eraser') and
                self._cursor_pos is not None and
                spec_rect.contains(self._cursor_pos)):
            # Convert brush radius (spec-pixel units) to screen pixels.
            px_per_tframe = (spec_rect.width() /
                             max(1, (t_end - t_start)))
            px_per_fbin = (spec_rect.height() /
                           max(1, (f_end - f_start)))
            # Use the geometric mean so the circle stays visually round
            # regardless of anisotropic zoom.
            radius_screen = self.brush_radius_px * (
                (px_per_tframe * px_per_fbin) ** 0.5
            )
            radius_screen = max(2.0, radius_screen)
            color = (QColor(255, 255, 255, 220)
                     if self.paint_mode == 'brush'
                     else QColor(255, 80, 80, 220))
            pen = QPen(color)
            pen.setWidth(1)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            cx = self._cursor_pos.x()
            cy = self._cursor_pos.y()
            painter.drawEllipse(
                QRectF(cx - radius_screen, cy - radius_screen,
                       2 * radius_screen, 2 * radius_screen)
            )
            # Tiny dot at the exact center.
            painter.drawPoint(self._cursor_pos)
        painter.end()


# ======================================================================
# Helpers
# ======================================================================
def _list_wavs_in_folder(folder: str) -> List[str]:
    """Non-recursive: just .wav files directly inside ``folder``."""
    root = Path(folder)
    if not root.exists() or not root.is_dir():
        return []
    out: List[str] = []
    for p in sorted(root.iterdir()):
        if not p.is_file():
            continue
        if p.suffix.lower() != ".wav":
            continue
        if p.name.startswith("."):
            continue
        out.append(str(p))
    return out


def _mask_sibling_path(wav_path: str) -> str:
    stem = Path(wav_path).stem
    return str(Path(wav_path).with_name(stem + LABEL_SUFFIX))


def _save_mask_png(path: str, mask: np.ndarray) -> None:
    if not HAS_PIL:
        raise RuntimeError(
            "PIL/Pillow is required to save MAD label PNGs. "
            "pip install pillow"
        )
    PILImage.fromarray(mask.astype(np.uint8), mode='L').save(path)


def _load_mask_png(path: str) -> np.ndarray:
    if not HAS_PIL:
        raise RuntimeError(
            "PIL/Pillow is required to load MAD label PNGs. "
            "pip install pillow"
        )
    img = PILImage.open(path).convert('L')
    return np.array(img, dtype=np.uint8)


# ======================================================================
# Training dialog + worker thread
# ======================================================================
class RunTrainingDialog(QDialog):
    """Modal dialog for configuring and launching a U-Net training run."""

    def __init__(self, parent, project_dir: str, wav_paths: List[str],
                 spec_params: dict):
        super().__init__(parent)
        self.setWindowTitle("Run MAD Training")
        self.setMinimumWidth(460)
        self.project_dir = project_dir
        self.wav_paths = list(wav_paths)
        self.spec_params = spec_params

        form = QFormLayout()

        self.spin_epochs = QSpinBox()
        self.spin_epochs.setRange(1, 500)
        self.spin_epochs.setValue(100)
        self.spin_epochs.setToolTip(
            "Upper bound on epochs. Training may stop early (SLEAP-style) "
            "once the validation loss plateaus — see patience below."
        )
        form.addRow("Max epochs:", self.spin_epochs)

        self.spin_patience = QSpinBox()
        self.spin_patience.setRange(0, 200)
        self.spin_patience.setValue(8)
        self.spin_patience.setToolTip(
            "Stop training when validation loss fails to improve for this "
            "many consecutive epochs. 0 = disable early stopping."
        )
        form.addRow("Early-stop patience:", self.spin_patience)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 64)
        self.spin_batch.setValue(8)
        form.addRow("Batch size:", self.spin_batch)

        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setDecimals(6)
        self.spin_lr.setRange(1e-6, 1.0)
        self.spin_lr.setSingleStep(1e-4)
        self.spin_lr.setValue(1e-3)
        form.addRow("Learning rate:", self.spin_lr)

        self.spin_val = QDoubleSpinBox()
        self.spin_val.setRange(0.0, 0.9)
        self.spin_val.setSingleStep(0.05)
        self.spin_val.setValue(0.20)
        form.addRow("Validation fraction:", self.spin_val)

        self.combo_encoder = QComboBox()
        self.combo_encoder.addItems([
            "resnet18", "resnet34", "resnet50",
            "efficientnet-b0", "mobilenet_v2",
        ])
        form.addRow("Encoder:", self.combo_encoder)

        self.combo_device = QComboBox()
        self.combo_device.addItems(["auto", "cuda", "mps", "cpu"])
        form.addRow("Device:", self.combo_device)

        self.spin_overlap = QDoubleSpinBox()
        self.spin_overlap.setRange(0.0, 0.9)
        self.spin_overlap.setSingleStep(0.05)
        self.spin_overlap.setValue(0.25)
        form.addRow("Tile overlap fraction:", self.spin_overlap)

        from PyQt5.QtWidgets import QLineEdit
        self.txt_run_name = QLineEdit()
        self.txt_run_name.setPlaceholderText("(auto: unet_YYYYMMDD_HHMMSS)")
        form.addRow("Run name:", self.txt_run_name)

        vbox = QVBoxLayout(self)
        vbox.addLayout(form)

        info = QLabel(
            f"Training files: {len(self.wav_paths)} wav(s) — only those with\n"
            "painted labels contribute tiles."
        )
        info.setStyleSheet("color: #888888; font-size: 10px;")
        vbox.addWidget(info)

        # Post-training inference option.
        post_row = QHBoxLayout()
        self.chk_post_inference = QCheckBox("Run inference after training on:")
        self.chk_post_inference.setChecked(True)
        self.combo_post_scope = QComboBox()
        self.combo_post_scope.addItems([
            "current file", "all files in project",
        ])
        self.combo_post_scope.setEnabled(True)
        self.chk_post_inference.toggled.connect(
            self.combo_post_scope.setEnabled
        )
        post_row.addWidget(self.chk_post_inference)
        post_row.addWidget(self.combo_post_scope, 1)
        vbox.addLayout(post_row)

        post_note = QLabel(
            "Inference runs only on time regions without painted labels, "
            "so existing annotations are preserved."
        )
        post_note.setStyleSheet("color: #888888; font-size: 10px;")
        post_note.setWordWrap(True)
        vbox.addWidget(post_note)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.button(QDialogButtonBox.Ok).setText("Start Training")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        vbox.addWidget(buttons)

    def post_inference_requested(self) -> bool:
        return self.chk_post_inference.isChecked()

    def post_inference_scope(self) -> str:
        """Return 'current' or 'all'."""
        return ('current'
                if self.combo_post_scope.currentIndex() == 0 else 'all')

    def build_config(self):
        from fnt.usv.usv_detector.mad_training import UNetTrainingConfig
        return UNetTrainingConfig(
            project_dir=self.project_dir,
            run_name=self.txt_run_name.text().strip(),
            encoder_name=self.combo_encoder.currentText(),
            n_epochs=self.spin_epochs.value(),
            early_stop_patience=self.spin_patience.value(),
            batch_size=self.spin_batch.value(),
            learning_rate=self.spin_lr.value(),
            val_fraction=self.spin_val.value(),
            device=self.combo_device.currentText(),
            nperseg=self.spec_params['nperseg'],
            noverlap=self.spec_params['noverlap'],
            nfft=self.spec_params['nfft'],
            db_min=self.spec_params['db_min'],
            db_max=self.spec_params['db_max'],
            tile_overlap_fraction=self.spin_overlap.value(),
            wav_paths=list(self.wav_paths),
        )


class MADTrainingWorker(QThread):
    progress_signal = pyqtSignal(int, int, dict)
    finished_signal = pyqtSignal(dict)
    error_signal = pyqtSignal(str)

    def __init__(self, cfg, parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self._stop = False

    def request_stop(self):
        self._stop = True

    def run(self):
        try:
            from fnt.usv.usv_detector.mad_training import train_unet
            summary = train_unet(
                self.cfg,
                progress=lambda e, n, m: self.progress_signal.emit(e, n, m),
                should_stop=lambda: self._stop,
            )
            self.finished_signal.emit(summary)
        except Exception as e:
            import traceback
            self.error_signal.emit(f"{e}\n\n{traceback.format_exc()}")


# ======================================================================
# Inference dialog + worker thread
# ======================================================================
class RunInferenceDialog(QDialog):
    def __init__(self, parent, project_dir: str, wav_paths: List[str],
                 current_wav: Optional[str], default_model_path: Optional[str]):
        super().__init__(parent)
        self.setWindowTitle("Run MAD Inference")
        self.setMinimumWidth(520)
        self.project_dir = project_dir
        self.wav_paths = list(wav_paths)
        self.current_wav = current_wav

        vbox = QVBoxLayout(self)

        from PyQt5.QtWidgets import QLineEdit
        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.txt_model = QLineEdit(default_model_path or "")
        self.txt_model.setPlaceholderText("Path to weights.pt")
        model_row.addWidget(self.txt_model, 1)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._pick_model)
        model_row.addWidget(btn_browse)
        vbox.addLayout(model_row)

        form = QFormLayout()

        self.spin_threshold = QDoubleSpinBox()
        self.spin_threshold.setRange(0.01, 0.99)
        self.spin_threshold.setSingleStep(0.05)
        self.spin_threshold.setValue(0.5)
        form.addRow("Probability threshold:", self.spin_threshold)

        self.spin_min_blob = QSpinBox()
        self.spin_min_blob.setRange(1, 10000)
        self.spin_min_blob.setValue(8)
        form.addRow("Min blob pixels:", self.spin_min_blob)

        self.combo_device = QComboBox()
        self.combo_device.addItems(["auto", "cuda", "mps", "cpu"])
        form.addRow("Device:", self.combo_device)

        vbox.addLayout(form)

        scope = QGroupBox("Scope")
        sv = QVBoxLayout()
        self.rb_current = QRadioButton(
            f"Current file only ({Path(current_wav).name if current_wav else '—'})"
        )
        self.rb_all = QRadioButton(f"All files in project ({len(self.wav_paths)})")
        if current_wav:
            self.rb_current.setChecked(True)
        else:
            self.rb_all.setChecked(True)
            self.rb_current.setEnabled(False)
        sv.addWidget(self.rb_current)
        sv.addWidget(self.rb_all)
        scope.setLayout(sv)
        vbox.addWidget(scope)

        opts_row = QHBoxLayout()
        self.chk_save_mask = QCheckBox("Save prediction PNG")
        self.chk_save_mask.setChecked(True)
        opts_row.addWidget(self.chk_save_mask)
        self.chk_save_csv = QCheckBox("Save blob CSV")
        self.chk_save_csv.setChecked(True)
        opts_row.addWidget(self.chk_save_csv)
        vbox.addLayout(opts_row)

        self.chk_preserve = QCheckBox(
            "Preserve user-painted labels (skip time regions already labeled)"
        )
        self.chk_preserve.setChecked(True)
        self.chk_preserve.setToolTip(
            "When enabled, the model's probability mask is zeroed in any\n"
            "time column that already contains a manually-painted label.\n"
            "Manual annotations are never overwritten by inference."
        )
        vbox.addWidget(self.chk_preserve)

        buttons = QDialogButtonBox(
            QDialogButtonBox.Ok | QDialogButtonBox.Cancel
        )
        buttons.button(QDialogButtonBox.Ok).setText("Run Inference")
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        vbox.addWidget(buttons)

    def _pick_model(self):
        models_root = os.path.join(self.project_dir, 'models')
        start = models_root if os.path.isdir(models_root) else self.project_dir
        path, _ = QFileDialog.getOpenFileName(
            self, "Select trained U-Net weights", start,
            "PyTorch weights (*.pt *.pth);;All Files (*)"
        )
        if path:
            self.txt_model.setText(path)

    def selected_wavs(self) -> List[str]:
        if self.rb_current.isChecked() and self.current_wav:
            return [self.current_wav]
        return list(self.wav_paths)

    def build_config(self):
        from fnt.usv.usv_detector.mad_inference import MADInferenceConfig
        return MADInferenceConfig(
            model_path=self.txt_model.text().strip(),
            threshold=self.spin_threshold.value(),
            min_blob_pixels=self.spin_min_blob.value(),
            device=self.combo_device.currentText(),
            save_mask_png=self.chk_save_mask.isChecked(),
            save_blob_csv=self.chk_save_csv.isChecked(),
            preserve_labels=self.chk_preserve.isChecked(),
        )


class MADInferenceWorker(QThread):
    progress_signal = pyqtSignal(int, int, str, str, int, int)
    finished_signal = pyqtSignal(list)
    error_signal = pyqtSignal(str)

    def __init__(self, cfg, wav_paths: List[str], parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.wav_paths = wav_paths
        self._stop = False

    def request_stop(self):
        self._stop = True

    def run(self):
        try:
            from fnt.usv.usv_detector.mad_inference import run_inference_on_files
            results = run_inference_on_files(
                self.wav_paths, self.cfg,
                progress=lambda *a: self.progress_signal.emit(*a),
                should_stop=lambda: self._stop,
            )
            self.finished_signal.emit(results)
        except Exception as e:
            import traceback
            self.error_signal.emit(f"{e}\n\n{traceback.format_exc()}")


# ======================================================================
# Progress dialog (shared)
# ======================================================================
class MADRunProgressDialog(QDialog):
    """Non-modal-ish progress reporter used by both training and inference.

    When ``show_plot=True`` (set by the training path), the dialog also
    embeds a live matplotlib canvas plotting per-batch training loss and
    per-epoch validation loss — SLEAP / DAD style.
    """

    cancel_requested = pyqtSignal()

    def __init__(self, parent, title: str, show_plot: bool = False):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumWidth(520)
        self.setWindowModality(Qt.NonModal)

        vbox = QVBoxLayout(self)

        self.lbl_stage = QLabel("Starting…")
        self.lbl_stage.setWordWrap(True)
        vbox.addWidget(self.lbl_stage)

        self.pb_main = QProgressBar()
        self.pb_main.setRange(0, 100)
        vbox.addWidget(self.pb_main)

        self.pb_sub = QProgressBar()
        self.pb_sub.setRange(0, 100)
        vbox.addWidget(self.pb_sub)

        # Optional live loss plot.
        self._plot = None
        self._batches_x: list = []
        self._batch_losses: list = []
        self._val_epoch_x: list = []
        self._val_losses: list = []
        self._train_epoch_x: list = []
        self._train_losses: list = []
        if show_plot:
            self._init_loss_plot(vbox)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(140)
        vbox.addWidget(self.log)

        btn_row = QHBoxLayout()
        btn_row.addStretch()
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.clicked.connect(self._on_cancel)
        btn_row.addWidget(self.btn_cancel)
        self.btn_close = QPushButton("Close")
        self.btn_close.setEnabled(False)
        self.btn_close.clicked.connect(self.accept)
        btn_row.addWidget(self.btn_close)
        vbox.addLayout(btn_row)

        if show_plot:
            self.setMinimumSize(720, 680)

    def _init_loss_plot(self, parent_layout):
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import (
                FigureCanvasQTAgg as FigureCanvas,
            )
        except Exception:
            from PyQt5.QtWidgets import QLabel as _QL
            msg = _QL("matplotlib not installed — live plot disabled.")
            msg.setStyleSheet("color: #888888; font-size: 10px;")
            parent_layout.addWidget(msg)
            return

        self._figure = Figure(figsize=(6.2, 3.2), tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setMinimumHeight(220)
        parent_layout.addWidget(self._canvas)

        self._ax = self._figure.add_subplot(1, 1, 1)
        self._ax.set_xlabel("Batch")
        self._ax.set_ylabel("Loss")
        self._ax.grid(alpha=0.3)

        (self._line_batch,) = self._ax.plot(
            [], [], color='#3b82f6', linewidth=0.9, alpha=0.75,
            label='batch train',
        )
        (self._line_train,) = self._ax.plot(
            [], [], color='#1d4ed8', marker='o', linewidth=1.5,
            label='epoch train',
        )
        (self._line_val,) = self._ax.plot(
            [], [], color='#dc2626', marker='s', linewidth=1.5,
            label='epoch val',
        )
        self._ax.legend(loc='upper right', fontsize=8)
        self._plot = True
        self._canvas.draw_idle()

    # --- live-plot public API (GUI thread only) -----------------------
    def plot_batch(self, global_batch: int, loss: float):
        if not self._plot:
            return
        self._batches_x.append(int(global_batch))
        self._batch_losses.append(float(loss))
        self._line_batch.set_data(self._batches_x, self._batch_losses)
        self._redraw_plot()

    def plot_epoch(self, global_batch: int, train_loss: float, val_loss: float):
        if not self._plot:
            return
        self._train_epoch_x.append(int(global_batch))
        self._train_losses.append(float(train_loss))
        self._val_epoch_x.append(int(global_batch))
        self._val_losses.append(float(val_loss))
        self._line_train.set_data(self._train_epoch_x, self._train_losses)
        self._line_val.set_data(self._val_epoch_x, self._val_losses)
        self._redraw_plot()

    def _redraw_plot(self):
        if not self._plot:
            return
        try:
            self._ax.relim()
            self._ax.autoscale_view()
            self._canvas.draw_idle()
        except Exception:
            pass

    def _on_cancel(self):
        self.btn_cancel.setEnabled(False)
        self.lbl_stage.setText("Cancelling…")
        self.cancel_requested.emit()

    def set_stage(self, text: str):
        self.lbl_stage.setText(text)

    def set_main(self, v, n):
        n_i = max(1, int(n))
        self.pb_main.setMaximum(n_i)
        self.pb_main.setValue(int(max(0, min(v, n_i))))

    def set_sub(self, v, n):
        n_i = max(1, int(n))
        self.pb_sub.setMaximum(n_i)
        self.pb_sub.setValue(int(max(0, min(v, n_i))))

    def append(self, text: str):
        self.log.append(text)

    def mark_done(self, ok: bool = True):
        self.btn_cancel.setEnabled(False)
        self.btn_close.setEnabled(True)
        self.lbl_stage.setText("Done." if ok else "Finished with errors.")


# ======================================================================
# Embeddable run panel (inline training/inference progress + live plot)
# ======================================================================
class MADRunPanel(QWidget):
    """Inline progress reporter embedded in the Mask / Inference tabs.

    Exposes the same surface that ``_start_training`` / ``_start_inference``
    drive on :class:`MADRunProgressDialog` (``set_stage`` / ``set_main`` /
    ``set_sub`` / ``append`` / ``plot_batch`` / ``plot_epoch`` / ``mark_done``
    + a ``cancel_requested`` signal), so those launchers work unchanged whether
    the reporter is a modal dialog or this inline widget.
    """

    cancel_requested = pyqtSignal()
    run_finished = pyqtSignal(bool)   # ok — lets the section re-enable its button

    def __init__(self, parent=None, show_plot: bool = False):
        super().__init__(parent)
        self._show_plot = show_plot
        self._plot = None
        self._batches_x: list = []
        self._batch_losses: list = []
        self._val_epoch_x: list = []
        self._val_losses: list = []
        self._train_epoch_x: list = []
        self._train_losses: list = []

        vbox = QVBoxLayout(self)
        vbox.setContentsMargins(0, 0, 0, 0)
        vbox.setSpacing(3)

        self.lbl_stage = QLabel("Idle.")
        self.lbl_stage.setWordWrap(True)
        self.lbl_stage.setStyleSheet("font-size: 10px;")
        vbox.addWidget(self.lbl_stage)

        self.pb_main = QProgressBar()
        self.pb_main.setRange(0, 100)
        self.pb_main.setTextVisible(False)
        self.pb_main.setMaximumHeight(10)
        vbox.addWidget(self.pb_main)

        self.pb_sub = QProgressBar()
        self.pb_sub.setRange(0, 100)
        self.pb_sub.setTextVisible(False)
        self.pb_sub.setMaximumHeight(8)
        vbox.addWidget(self.pb_sub)

        if show_plot:
            self._init_loss_plot(vbox)

        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMaximumHeight(110)
        vbox.addWidget(self.log)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setEnabled(False)
        self.btn_stop.clicked.connect(self._on_stop)
        vbox.addWidget(self.btn_stop)

    def _init_loss_plot(self, parent_layout):
        try:
            from matplotlib.figure import Figure
            from matplotlib.backends.backend_qt5agg import (
                FigureCanvasQTAgg as FigureCanvas,
            )
        except Exception:
            msg = QLabel("matplotlib not installed — live plot disabled.")
            msg.setStyleSheet("color: #888888; font-size: 10px;")
            parent_layout.addWidget(msg)
            return
        self._figure = Figure(figsize=(4.0, 2.2), tight_layout=True)
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setMinimumHeight(170)
        parent_layout.addWidget(self._canvas)
        self._ax = self._figure.add_subplot(1, 1, 1)
        self._ax.set_xlabel("Batch", fontsize=8)
        self._ax.set_ylabel("Loss", fontsize=8)
        self._ax.tick_params(labelsize=7)
        self._ax.grid(alpha=0.3)
        (self._line_batch,) = self._ax.plot(
            [], [], color='#3b82f6', linewidth=0.9, alpha=0.75, label='batch')
        (self._line_train,) = self._ax.plot(
            [], [], color='#1d4ed8', marker='o', markersize=3,
            linewidth=1.3, label='train')
        (self._line_val,) = self._ax.plot(
            [], [], color='#dc2626', marker='s', markersize=3,
            linewidth=1.3, label='val')
        self._ax.legend(loc='upper right', fontsize=7)
        self._plot = True
        self._canvas.draw_idle()

    # --- run lifecycle -------------------------------------------------
    def start_run(self):
        """Reset state and arm the Stop button for a new run."""
        self.log.clear()
        self.btn_stop.setEnabled(True)
        self.pb_main.setValue(0)
        self.pb_sub.setValue(0)
        for lst in (self._batches_x, self._batch_losses, self._val_epoch_x,
                    self._val_losses, self._train_epoch_x, self._train_losses):
            lst.clear()
        if self._plot:
            self._line_batch.set_data([], [])
            self._line_train.set_data([], [])
            self._line_val.set_data([], [])
            self._redraw_plot()

    # --- reporter API (GUI thread only) -------------------------------
    def set_stage(self, text: str):
        self.lbl_stage.setText(text)

    def set_main(self, v, n):
        n_i = max(1, int(n))
        self.pb_main.setMaximum(n_i)
        self.pb_main.setValue(int(max(0, min(v, n_i))))

    def set_sub(self, v, n):
        n_i = max(1, int(n))
        self.pb_sub.setMaximum(n_i)
        self.pb_sub.setValue(int(max(0, min(v, n_i))))

    def append(self, text: str):
        self.log.append(text)

    def plot_batch(self, global_batch: int, loss: float):
        if not self._plot:
            return
        self._batches_x.append(int(global_batch))
        self._batch_losses.append(float(loss))
        self._line_batch.set_data(self._batches_x, self._batch_losses)
        self._redraw_plot()

    def plot_epoch(self, global_batch: int, train_loss: float, val_loss: float):
        if not self._plot:
            return
        self._train_epoch_x.append(int(global_batch))
        self._train_losses.append(float(train_loss))
        self._val_epoch_x.append(int(global_batch))
        self._val_losses.append(float(val_loss))
        self._line_train.set_data(self._train_epoch_x, self._train_losses)
        self._line_val.set_data(self._val_epoch_x, self._val_losses)
        self._redraw_plot()

    def _redraw_plot(self):
        if not self._plot:
            return
        try:
            self._ax.relim()
            self._ax.autoscale_view()
            self._canvas.draw_idle()
        except Exception:
            pass

    def mark_done(self, ok: bool = True):
        self.btn_stop.setEnabled(False)
        self.run_finished.emit(bool(ok))

    def _on_stop(self):
        self.btn_stop.setEnabled(False)
        self.lbl_stage.setText("Stopping…")
        self.cancel_requested.emit()


# ======================================================================
# SAM2-assisted labeling worker threads
# ======================================================================
class MADSamLoadWorker(QThread):
    """Loads the SAM2 image model off the UI thread (heavy torch import)."""
    done_signal = pyqtSignal()
    error_signal = pyqtSignal(str)

    def __init__(self, segmenter, parent=None):
        super().__init__(parent)
        self.segmenter = segmenter

    def run(self):
        try:
            self.segmenter.load_model()
            self.done_signal.emit()
        except Exception as e:
            import traceback
            self.error_signal.emit(f"{e}\n\n{traceback.format_exc()}")


class MADSamPredictWorker(QThread):
    """Runs one SAM2 prediction. If ``image_rgb`` is given, the image
    embedding is (re)computed first; otherwise the cached embedding from a
    previous prediction is reused (fast path for adding more points)."""
    done_signal = pyqtSignal(object, int)   # (bool mask or None, t_off)
    error_signal = pyqtSignal(str)

    def __init__(self, segmenter, image_rgb, pos_pts, neg_pts, t_off,
                 parent=None):
        super().__init__(parent)
        self.segmenter = segmenter
        self.image_rgb = image_rgb
        self.pos_pts = pos_pts
        self.neg_pts = neg_pts
        self.t_off = t_off

    def run(self):
        try:
            if self.image_rgb is not None:
                self.segmenter.set_image(self.image_rgb)
            if not self.pos_pts:
                self.done_signal.emit(None, self.t_off)
                return
            mask, _score = self.segmenter.predict_mask(
                self.pos_pts, self.neg_pts or None
            )
            self.done_signal.emit(mask, self.t_off)
        except Exception as e:
            import traceback
            self.error_signal.emit(f"{e}\n\n{traceback.format_exc()}")


# ======================================================================
# Main window
# ======================================================================
class MADMainWindow(QMainWindow):
    BASE_TITLE = "FNT Mask Audio Detector"

    def __init__(self):
        super().__init__()
        self.setWindowTitle(self.BASE_TITLE)
        self.setMinimumSize(1100, 750)
        self.resize(1400, 900)

        self._project: Optional[MADProjectConfig] = None
        self.audio_files: List[str] = []
        self.current_file_idx: int = 0
        self.audio_data: Optional[np.ndarray] = None
        self.sample_rate: Optional[int] = None

        # Playback state
        self.is_playing = False
        self.playback_speed = 1.0
        self._speed_values = [0.01, 0.02, 0.03, 0.04, 0.05, 0.1, 0.2, 0.5, 1.0]
        self._playback_start_time = None
        self._playback_start_s = None
        self._playback_end_s = None
        self._playback_latency = 0.15
        self._playback_timer = QTimer(self)
        self._playback_timer.setInterval(30)
        self._playback_timer.timeout.connect(self._update_playback_position)

        self._settings = QSettings("FNT", "MAD")

        # Blob-review state
        self._pred_blobs: List[dict] = []
        self._pred_csv_path: Optional[str] = None
        self._pred_wav_path: Optional[str] = None
        self._pred_idx: Optional[int] = None

        # SAM2-assisted labeling state
        self._sam_ckpt: Optional[str] = None
        self._sam_cfg_name: Optional[str] = None
        self._sam_segmenter = None
        self._sam_ready = False
        self._sam_load_worker: Optional[MADSamLoadWorker] = None
        self._sam_predict_worker: Optional[MADSamPredictWorker] = None
        self._sam_predict_pending = False
        self._sam_img_sig = None
        self._sam_last_t_off = None

        self._setup_menu_bar()
        self._build_ui()
        self._setup_shortcuts()
        self._update_project_state()
        self._update_paint_buttons_enabled()
        self._update_playback_buttons_enabled()

    # ==================================================================
    # UI construction
    # ==================================================================
    def _build_ui(self):
        central = QWidget()
        self.setCentralWidget(central)

        main_layout = QHBoxLayout(central)
        main_layout.setContentsMargins(5, 5, 5, 5)
        main_layout.setSpacing(5)

        # Live training graph — created before the tabs so the Train section
        # can wire its signals; it's placed in the right preview area below and
        # only shown while training runs (mirrors Mask Tracker).
        self.train_panel = MADRunPanel(show_plot=True)

        # ---------- Left panel: workflow-stage tabs over a shared canvas ----
        # Mask tab = label + train; Inference tab = run + blob review. Both
        # act on the single spectrogram canvas in the right panel. (A future
        # "Class" tab for call-type classification slots in here.)
        _fm = self.fontMetrics()
        _min_w = max(360, _fm.averageCharWidth() * 56 + 40)
        _max_w = max(460, _fm.averageCharWidth() * 74 + 40)

        self.left_tabs = QTabWidget()
        self.left_tabs.setMinimumWidth(_min_w)
        self.left_tabs.setMaximumWidth(_max_w)
        self.left_tabs.currentChanged.connect(self._on_left_tab_changed)

        def _make_tab(section_builders):
            scroll = QScrollArea()
            scroll.setWidgetResizable(True)
            scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
            page = QWidget()
            page_layout = QVBoxLayout(page)
            page_layout.setContentsMargins(5, 5, 5, 5)
            page_layout.setSpacing(8)
            for build in section_builders:
                build(page_layout)
            page_layout.addStretch()
            scroll.setWidget(page)
            return scroll

        mask_tab = _make_tab([
            self._create_project_section,
            self._create_training_data_section,
            self._create_paint_tools_section,
            self._create_training_section,
        ])
        inference_tab = _make_tab([
            self._create_inference_section,
            self._create_review_section,
        ])
        self.left_tabs.addTab(mask_tab, "Mask")
        self.left_tabs.addTab(inference_tab, "Inference")
        self._inference_tab_index = 1

        # ---------- Right panel ----------
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(2)

        self.spectrogram = MADSpectrogramWidget()
        self.spectrogram.zoom_requested.connect(self._on_wheel_zoom)
        self.spectrogram.stroke_committed.connect(self._on_stroke_committed)
        self.spectrogram.sam_points_changed.connect(self._on_sam_points_changed)
        self.spectrogram.brush_radius_changed.connect(
            self._on_brush_radius_scrolled
        )
        right_layout.addWidget(self.spectrogram, 1)

        # Live training graph occupies the same area as the spectrogram while
        # a run is active (hidden otherwise).
        self.train_panel.setVisible(False)
        right_layout.addWidget(self.train_panel, 1)

        self.waveform_overview = WaveformOverviewWidget()
        self.waveform_overview.view_changed.connect(self._on_overview_clicked)
        right_layout.addWidget(self.waveform_overview)

        # Scrollbar + pan row
        scroll_bar_row = QWidget()
        self._scroll_bar_row = scroll_bar_row
        scroll_layout = QHBoxLayout(scroll_bar_row)
        scroll_layout.setContentsMargins(5, 2, 5, 2)
        scroll_layout.setSpacing(2)

        self.btn_pan_left = QPushButton("<")
        self.btn_pan_left.setFixedWidth(24)
        self.btn_pan_left.setToolTip("Pan left in time (←)")
        self.btn_pan_left.clicked.connect(self._pan_left)
        scroll_layout.addWidget(self.btn_pan_left)

        self.time_scrollbar = QScrollBar(Qt.Horizontal)
        self.time_scrollbar.setMinimum(0)
        self.time_scrollbar.setMaximum(1000)
        self.time_scrollbar.setToolTip("Scroll through the recording timeline")
        self.time_scrollbar.valueChanged.connect(self._on_scrollbar_changed)
        scroll_layout.addWidget(self.time_scrollbar, 1)

        self.btn_pan_right = QPushButton(">")
        self.btn_pan_right.setFixedWidth(24)
        self.btn_pan_right.setToolTip("Pan right in time (→)")
        self.btn_pan_right.clicked.connect(self._pan_right)
        scroll_layout.addWidget(self.btn_pan_right)

        right_layout.addWidget(scroll_bar_row)

        # Controls row
        controls_bar = QWidget()
        self._controls_bar = controls_bar
        controls_layout = QHBoxLayout(controls_bar)
        controls_layout.setContentsMargins(5, 2, 5, 2)
        controls_layout.setSpacing(4)

        controls_layout.addWidget(QLabel("Window:"))
        self.btn_zoom_out = QPushButton("-")
        self.btn_zoom_out.setFixedWidth(24)
        self.btn_zoom_out.setToolTip("Zoom out (↓)")
        self.btn_zoom_out.clicked.connect(self._zoom_out)
        controls_layout.addWidget(self.btn_zoom_out)

        self.spin_view_window = QDoubleSpinBox()
        self.spin_view_window.setRange(0.1, 600.0)
        self.spin_view_window.setValue(2.0)
        self.spin_view_window.setSuffix(" s")
        self.spin_view_window.setFixedWidth(80)
        self.spin_view_window.setToolTip("Time window duration (seconds)")
        self.spin_view_window.valueChanged.connect(self._on_view_window_changed)
        controls_layout.addWidget(self.spin_view_window)

        self.btn_zoom_in = QPushButton("+")
        self.btn_zoom_in.setFixedWidth(24)
        self.btn_zoom_in.setToolTip("Zoom in (↑)")
        self.btn_zoom_in.clicked.connect(self._zoom_in)
        controls_layout.addWidget(self.btn_zoom_in)

        sep1 = QFrame(); sep1.setFrameShape(QFrame.VLine)
        sep1.setStyleSheet("color: #3f3f3f;")
        controls_layout.addWidget(sep1)

        controls_layout.addWidget(QLabel("Freq:"))
        self.spin_display_min_freq = QSpinBox()
        self.spin_display_min_freq.setRange(0, 200000)
        self.spin_display_min_freq.setSingleStep(5000)
        self.spin_display_min_freq.setValue(0)
        self.spin_display_min_freq.setSuffix(" Hz")
        self.spin_display_min_freq.setFixedWidth(90)
        self.spin_display_min_freq.valueChanged.connect(self._on_display_freq_changed)
        controls_layout.addWidget(self.spin_display_min_freq)

        controls_layout.addWidget(QLabel("-"))
        self.spin_display_max_freq = QSpinBox()
        self.spin_display_max_freq.setRange(1000, 250000)
        self.spin_display_max_freq.setSingleStep(5000)
        self.spin_display_max_freq.setValue(125000)
        self.spin_display_max_freq.setSuffix(" Hz")
        self.spin_display_max_freq.setFixedWidth(90)
        self.spin_display_max_freq.valueChanged.connect(self._on_display_freq_changed)
        controls_layout.addWidget(self.spin_display_max_freq)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.VLine)
        sep2.setStyleSheet("color: #3f3f3f;")
        controls_layout.addWidget(sep2)

        controls_layout.addWidget(QLabel("Color Map:"))
        self.combo_colormap = QComboBox()
        self.combo_colormap.addItems(['viridis', 'magma', 'inferno', 'grayscale'])
        self.combo_colormap.setFixedWidth(90)
        self.combo_colormap.currentTextChanged.connect(self._on_colormap_changed)
        controls_layout.addWidget(self.combo_colormap)

        sep3 = QFrame(); sep3.setFrameShape(QFrame.VLine)
        sep3.setStyleSheet("color: #3f3f3f;")
        controls_layout.addWidget(sep3)

        # Playback
        self.btn_play = QPushButton("Play")
        self.btn_play.setToolTip("Play the visible window (Space)")
        self.btn_play.clicked.connect(self._toggle_playback)
        controls_layout.addWidget(self.btn_play)

        self.btn_stop = QPushButton("Stop")
        self.btn_stop.setStyleSheet("background-color: #5c5c5c;")
        self.btn_stop.setToolTip("Stop playback")
        self.btn_stop.clicked.connect(self._stop_playback)
        controls_layout.addWidget(self.btn_stop)

        controls_layout.addWidget(QLabel("Speed:"))
        self.slider_speed = QSlider(Qt.Horizontal)
        self.slider_speed.setRange(0, len(self._speed_values) - 1)
        self.slider_speed.setValue(len(self._speed_values) - 1)
        self.slider_speed.setFixedWidth(100)
        self.slider_speed.setToolTip(
            "Playback speed multiplier. Lower values slow audio down so\n"
            "ultrasonic frequencies shift into the audible range."
        )
        self.slider_speed.valueChanged.connect(self._on_speed_changed)
        controls_layout.addWidget(self.slider_speed)

        self.lbl_speed = QLabel("1.0x")
        self.lbl_speed.setFixedWidth(40)
        controls_layout.addWidget(self.lbl_speed)

        if not HAS_SOUNDDEVICE:
            lbl_warn = QLabel("No audio")
            lbl_warn.setStyleSheet("color: #d13438; font-size: 9px;")
            controls_layout.addWidget(lbl_warn)

        controls_layout.addStretch()
        right_layout.addWidget(controls_bar)

        main_layout.addWidget(self.left_tabs)
        main_layout.addWidget(right_panel, 1)

        self.status_bar = QStatusBar()
        self.setStatusBar(self.status_bar)
        self.status_bar.showMessage(
            "Welcome to Mask Audio Detector — create or open a project to begin"
        )

    # ------------------------------------------------------------------
    # Left-panel section builders
    # ------------------------------------------------------------------
    def _create_project_section(self, layout):
        group = QGroupBox("1. Project")
        vbox = QVBoxLayout()
        vbox.setSpacing(4)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        self.btn_new_project = QPushButton("New Project…")
        self.btn_new_project.clicked.connect(self._menu_new_project)
        btn_row.addWidget(self.btn_new_project)

        self.btn_open_project = QPushButton("Open Project…")
        self.btn_open_project.clicked.connect(self._menu_open_project)
        btn_row.addWidget(self.btn_open_project)
        vbox.addLayout(btn_row)

        self.lbl_project_name = QLabel("No project — click 'New Project…'")
        self.lbl_project_name.setStyleSheet("color: #999999; font-size: 10px;")
        self.lbl_project_name.setWordWrap(True)
        vbox.addWidget(self.lbl_project_name)

        self.lbl_source_folders = QLabel("No source folders")
        self.lbl_source_folders.setStyleSheet("color: #777777; font-size: 9px;")
        self.lbl_source_folders.setWordWrap(True)
        vbox.addWidget(self.lbl_source_folders)

        self.lbl_model_info = QLabel("No trained model")
        self.lbl_model_info.setStyleSheet("color: #999999; font-size: 10px;")
        self.lbl_model_info.setWordWrap(True)
        vbox.addWidget(self.lbl_model_info)

        group.setLayout(vbox)
        layout.addWidget(group)

    def _create_training_data_section(self, layout):
        group = QGroupBox("2. Training Data")
        vbox = QVBoxLayout()
        vbox.setSpacing(4)

        add_row = QHBoxLayout()
        add_row.setSpacing(4)
        self.btn_add_folder = QPushButton("Add Folder…")
        self.btn_add_folder.setToolTip(
            "Add .wav files directly inside a folder (non-recursive)."
        )
        self.btn_add_folder.clicked.connect(self._menu_add_folder)
        self.btn_add_folder.setEnabled(False)
        add_row.addWidget(self.btn_add_folder)

        self.btn_add_files = QPushButton("Add Files…")
        self.btn_add_files.clicked.connect(self._add_audio_files)
        self.btn_add_files.setEnabled(False)
        add_row.addWidget(self.btn_add_files)
        vbox.addLayout(add_row)

        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(180)
        self.file_list.currentRowChanged.connect(self._on_file_selected)
        vbox.addWidget(self.file_list)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(2)
        self.btn_prev_file = QPushButton("< Prev")
        self.btn_prev_file.setToolTip("Previous file (B)")
        self.btn_prev_file.clicked.connect(self._prev_file)
        self.btn_prev_file.setEnabled(False)
        nav_row.addWidget(self.btn_prev_file)

        self.lbl_file_num = QLabel("File 0/0")
        self.lbl_file_num.setAlignment(Qt.AlignCenter)
        nav_row.addWidget(self.lbl_file_num, 1)

        self.btn_next_file = QPushButton("Next >")
        self.btn_next_file.setToolTip("Next file (N)")
        self.btn_next_file.clicked.connect(self._next_file)
        self.btn_next_file.setEnabled(False)
        nav_row.addWidget(self.btn_next_file)
        vbox.addLayout(nav_row)

        self.lbl_data_summary = QLabel("No files loaded")
        self.lbl_data_summary.setStyleSheet("color: #999999; font-size: 10px;")
        vbox.addWidget(self.lbl_data_summary)

        group.setLayout(vbox)
        layout.addWidget(group)

    def _create_paint_tools_section(self, layout):
        group = QGroupBox("3. Labeling Tools")
        vbox = QVBoxLayout()
        vbox.setSpacing(4)

        # SAM2-assisted labeling is the primary tool — listed first.
        sam_row = QHBoxLayout()
        sam_row.setSpacing(2)
        self.btn_sam = QPushButton("SAM Labeling (G)")
        self.btn_sam.setToolTip(
            "SAM2-assisted labeling — left-click a call to propose a mask,\n"
            "right-click to add a negative point. Enter accepts, Esc clears.\n"
            "First use offers any SAM2 models in LocalModels/, or downloads "
            "one.\nShortcut: G"
        )
        self.btn_sam.setCheckable(True)
        self.btn_sam.clicked.connect(self._on_sam_clicked)
        self.btn_sam.setEnabled(False)
        sam_row.addWidget(self.btn_sam)
        sam_row.addStretch()
        vbox.addLayout(sam_row)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(2)

        self.btn_paint = QPushButton("Manual Labeling (B)")
        self.btn_paint.setToolTip(
            "Manually paint target USV pixels with the brush "
            "(left-click + drag).\n"
            "Scroll the wheel over the spectrogram to fine-tune brush size.\n"
            "Shortcut: B"
        )
        self.btn_paint.setCheckable(True)
        self.btn_paint.clicked.connect(self._on_brush_clicked)
        self.btn_paint.setEnabled(False)
        mode_row.addWidget(self.btn_paint)

        self.btn_erase = QPushButton("Eraser (E)")
        self.btn_erase.setToolTip(
            "Erase painted pixels (scroll to resize)\n"
            "Shortcut: E"
        )
        self.btn_erase.setCheckable(True)
        self.btn_erase.clicked.connect(self._on_eraser_clicked)
        self.btn_erase.setEnabled(False)
        mode_row.addWidget(self.btn_erase)

        self.btn_clear_mask = QPushButton("Clear")
        self.btn_clear_mask.setToolTip("Clear all paint on this file")
        self.btn_clear_mask.clicked.connect(self._on_clear_clicked)
        self.btn_clear_mask.setEnabled(False)
        mode_row.addWidget(self.btn_clear_mask)

        vbox.addLayout(mode_row)

        brush_row = QHBoxLayout()
        brush_row.setSpacing(2)
        brush_row.addWidget(QLabel("Brush radius:"))
        self.spin_brush_radius = QSpinBox()
        self.spin_brush_radius.setRange(1, 64)
        self.spin_brush_radius.setValue(6)
        self.spin_brush_radius.setSuffix(" px")
        self.spin_brush_radius.setToolTip(
            "Brush radius in spectrogram pixels (time-frame × freq-bin)"
        )
        self.spin_brush_radius.valueChanged.connect(
            lambda v: self.spectrogram.set_brush_radius(v)
        )
        brush_row.addWidget(self.spin_brush_radius)
        brush_row.addStretch()
        vbox.addLayout(brush_row)

        # Paint strokes auto-save to the sibling PNG (see mouseReleaseEvent).
        self.lbl_mask_status = QLabel("No mask")
        self.lbl_mask_status.setStyleSheet("color: #999999; font-size: 9px;")
        self.lbl_mask_status.setWordWrap(True)
        self.lbl_mask_status.setMinimumWidth(0)
        self.lbl_mask_status.setSizePolicy(
            QSizePolicy.Ignored, QSizePolicy.Preferred
        )
        vbox.addWidget(self.lbl_mask_status)

        # --- View mode (merged in from the former "Mask View" section) ---
        view_row = QHBoxLayout()
        view_row.setSpacing(2)
        view_row.addWidget(QLabel("View:"))
        self.btn_view_spec = QPushButton("Spec")
        self.btn_view_spec.setCheckable(True)
        self.btn_view_spec.setToolTip("Spectrogram only — hide mask overlay")
        self.btn_view_spec.clicked.connect(
            lambda: self._set_view_mode('spec')
        )
        view_row.addWidget(self.btn_view_spec)

        self.btn_view_overlay = QPushButton("Spec + Mask")
        self.btn_view_overlay.setCheckable(True)
        self.btn_view_overlay.setChecked(True)
        self.btn_view_overlay.setToolTip(
            "Show spectrogram with mask overlay (default)"
        )
        self.btn_view_overlay.clicked.connect(
            lambda: self._set_view_mode('overlay')
        )
        view_row.addWidget(self.btn_view_overlay)

        self.btn_view_mask = QPushButton("Mask (M)")
        self.btn_view_mask.setCheckable(True)
        self.btn_view_mask.setToolTip(
            "Hide spectrogram — show confirmed + pending masks only\n"
            "Shortcut: M (toggles back to Spec + Mask)"
        )
        self.btn_view_mask.clicked.connect(
            lambda: self._set_view_mode('mask_only')
        )
        view_row.addWidget(self.btn_view_mask)
        view_row.addStretch()
        vbox.addLayout(view_row)

        hint = QLabel(
            "Yellow = pending (Enter to confirm, pick a class) · Magenta = "
            "confirmed call · Cyan = model prediction. Confirmed calls save as "
            "self-contained training examples."
        )
        hint.setStyleSheet("color: #888888; font-size: 9px; font-style: italic;")
        hint.setWordWrap(True)
        vbox.addWidget(hint)

        group.setLayout(vbox)
        layout.addWidget(group)

    def _set_view_mode(self, mode: str):
        self.btn_view_spec.setChecked(mode == 'spec')
        self.btn_view_overlay.setChecked(mode == 'overlay')
        self.btn_view_mask.setChecked(mode == 'mask_only')
        self.spectrogram.set_view_mode(mode)

    def _create_review_section(self, layout):
        group = QGroupBox("Blob Review")
        vbox = QVBoxLayout()
        vbox.setSpacing(4)

        self.btn_load_predictions = QPushButton("Load Predictions for This File")
        self.btn_load_predictions.setToolTip(
            "Load sibling *_FNT_MAD_predictions.csv/.png for the current wav"
        )
        self.btn_load_predictions.clicked.connect(self._load_predictions_for_current)
        self.btn_load_predictions.setEnabled(False)
        vbox.addWidget(self.btn_load_predictions)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(2)
        self.btn_prev_blob = QPushButton("< Prev")
        self.btn_prev_blob.setToolTip("Previous blob")
        self.btn_prev_blob.clicked.connect(self._prev_blob)
        self.btn_prev_blob.setEnabled(False)
        nav_row.addWidget(self.btn_prev_blob)

        self.lbl_blob_idx = QLabel("Blob 0/0")
        self.lbl_blob_idx.setAlignment(Qt.AlignCenter)
        nav_row.addWidget(self.lbl_blob_idx, 1)

        self.btn_next_blob = QPushButton("Next >")
        self.btn_next_blob.setToolTip("Next blob")
        self.btn_next_blob.clicked.connect(self._next_blob)
        self.btn_next_blob.setEnabled(False)
        nav_row.addWidget(self.btn_next_blob)
        vbox.addLayout(nav_row)

        row = QHBoxLayout()
        row.setSpacing(2)
        self.btn_accept_blob = QPushButton("Accept (A)")
        self.btn_accept_blob.setToolTip("Mark blob as accepted — writes into paint mask")
        self.btn_accept_blob.clicked.connect(lambda: self._review_current_blob('accepted'))
        self.btn_accept_blob.setEnabled(False)
        row.addWidget(self.btn_accept_blob)

        self.btn_reject_blob = QPushButton("Reject (R)")
        self.btn_reject_blob.setToolTip("Mark blob as rejected")
        self.btn_reject_blob.clicked.connect(lambda: self._review_current_blob('rejected'))
        self.btn_reject_blob.setEnabled(False)
        row.addWidget(self.btn_reject_blob)

        self.btn_skip_blob = QPushButton("Skip (S)")
        self.btn_skip_blob.setToolTip("Mark blob as skipped — keeps pending decisions later")
        self.btn_skip_blob.clicked.connect(lambda: self._review_current_blob('skipped'))
        self.btn_skip_blob.setEnabled(False)
        row.addWidget(self.btn_skip_blob)
        vbox.addLayout(row)

        self.lbl_blob_info = QLabel("No predicted blobs yet")
        self.lbl_blob_info.setStyleSheet("color: #888888; font-size: 9px;")
        self.lbl_blob_info.setWordWrap(True)
        vbox.addWidget(self.lbl_blob_info)

        group.setLayout(vbox)
        layout.addWidget(group)

    def _create_training_section(self, layout):
        from PyQt5.QtWidgets import QLineEdit
        group = QGroupBox("4. Train Model")
        vbox = QVBoxLayout()
        vbox.setSpacing(4)
        form = QFormLayout()

        self.combo_arch = QComboBox()
        for label, key in (
            ("U-Net (baseline)", "unet"),
            ("U-Net++ (fine detail)", "unetpp"),
            ("HRNet U-Net (crisp thin outlines)", "hrnet"),
            ("MA-Net (attention)", "manet"),
        ):
            self.combo_arch.addItem(label, key)
        self.combo_arch.setToolTip(
            "Segmentation architecture (all semantic, via "
            "segmentation_models_pytorch). HRNet keeps high-resolution\n"
            "features for the finest USV contours; it uses its own encoder."
        )
        self.combo_arch.currentIndexChanged.connect(self._on_arch_changed)
        form.addRow("Architecture:", self.combo_arch)

        self.combo_train_encoder = QComboBox()
        self.combo_train_encoder.addItems([
            "resnet18", "resnet34", "resnet50",
            "efficientnet-b0", "mobilenet_v2",
        ])
        form.addRow("Encoder:", self.combo_train_encoder)

        self.spin_train_epochs = QSpinBox()
        self.spin_train_epochs.setRange(1, 500)
        self.spin_train_epochs.setValue(100)
        form.addRow("Max epochs:", self.spin_train_epochs)

        self.spin_train_patience = QSpinBox()
        self.spin_train_patience.setRange(0, 200)
        self.spin_train_patience.setValue(8)
        self.spin_train_patience.setToolTip("0 = disable early stopping")
        form.addRow("Early-stop patience:", self.spin_train_patience)

        self.spin_train_batch = QSpinBox()
        self.spin_train_batch.setRange(1, 64)
        self.spin_train_batch.setValue(8)
        form.addRow("Batch size:", self.spin_train_batch)

        self.spin_train_lr = QDoubleSpinBox()
        self.spin_train_lr.setDecimals(6)
        self.spin_train_lr.setRange(1e-6, 1.0)
        self.spin_train_lr.setSingleStep(1e-4)
        self.spin_train_lr.setValue(1e-3)
        form.addRow("Learning rate:", self.spin_train_lr)

        self.spin_train_val = QDoubleSpinBox()
        self.spin_train_val.setRange(0.0, 0.9)
        self.spin_train_val.setSingleStep(0.05)
        self.spin_train_val.setValue(0.20)
        form.addRow("Validation fraction:", self.spin_train_val)

        self.combo_train_device = QComboBox()
        self.combo_train_device.addItems(["auto", "cuda", "mps", "cpu"])
        form.addRow("Device:", self.combo_train_device)

        self.txt_train_run = QLineEdit()
        self.txt_train_run.setPlaceholderText("(auto: unet_YYYYMMDD_HHMMSS)")
        form.addRow("Run name:", self.txt_train_run)

        vbox.addLayout(form)

        post_row = QHBoxLayout()
        self.chk_post_inference = QCheckBox("Run inference after, on:")
        self.chk_post_inference.setChecked(True)
        self.combo_post_scope = QComboBox()
        self.combo_post_scope.addItems(["current file", "all files"])
        self.chk_post_inference.toggled.connect(self.combo_post_scope.setEnabled)
        post_row.addWidget(self.chk_post_inference)
        post_row.addWidget(self.combo_post_scope, 1)
        vbox.addLayout(post_row)

        self.btn_train = QPushButton("Train")
        self.btn_train.setToolTip(
            "Train on every confirmed call. The live loss graph appears in the\n"
            "spectrogram area while training runs."
        )
        self.btn_train.clicked.connect(self._on_inline_train)
        self.btn_train.setEnabled(False)
        vbox.addWidget(self.btn_train)

        # The live training graph (self.train_panel) lives in the right-hand
        # preview area (built in _build_ui) and is only shown while training
        # runs — mirroring Mask Tracker. Here we just wire its signals.
        self.train_panel.run_finished.connect(self._on_training_finished)
        self.train_panel.cancel_requested.connect(
            lambda: self.status_bar.showMessage("Stopping training…")
        )

        group.setLayout(vbox)
        layout.addWidget(group)

    def _create_inference_section(self, layout):
        from PyQt5.QtWidgets import QLineEdit
        group = QGroupBox("Run Inference")
        vbox = QVBoxLayout()
        vbox.setSpacing(4)

        model_row = QHBoxLayout()
        model_row.addWidget(QLabel("Model:"))
        self.txt_infer_model = QLineEdit()
        self.txt_infer_model.setPlaceholderText("(latest trained run)")
        model_row.addWidget(self.txt_infer_model, 1)
        btn_browse = QPushButton("Browse…")
        btn_browse.clicked.connect(self._pick_infer_model)
        model_row.addWidget(btn_browse)
        vbox.addLayout(model_row)

        form = QFormLayout()
        self.spin_infer_threshold = QDoubleSpinBox()
        self.spin_infer_threshold.setRange(0.01, 0.99)
        self.spin_infer_threshold.setSingleStep(0.05)
        self.spin_infer_threshold.setValue(0.5)
        form.addRow("Probability threshold:", self.spin_infer_threshold)

        self.spin_infer_min_blob = QSpinBox()
        self.spin_infer_min_blob.setRange(1, 10000)
        self.spin_infer_min_blob.setValue(8)
        form.addRow("Min blob pixels:", self.spin_infer_min_blob)

        self.combo_infer_device = QComboBox()
        self.combo_infer_device.addItems(["auto", "cuda", "mps", "cpu"])
        form.addRow("Device:", self.combo_infer_device)
        vbox.addLayout(form)

        scope = QGroupBox("Scope")
        sv = QVBoxLayout()
        self.rb_infer_current = QRadioButton("Current file only")
        self.rb_infer_current.setChecked(True)
        self.rb_infer_all = QRadioButton("All files in project")
        sv.addWidget(self.rb_infer_current)
        sv.addWidget(self.rb_infer_all)
        scope.setLayout(sv)
        vbox.addWidget(scope)

        self.chk_infer_save_mask = QCheckBox("Save prediction PNG")
        self.chk_infer_save_mask.setChecked(True)
        vbox.addWidget(self.chk_infer_save_mask)
        self.chk_infer_save_csv = QCheckBox("Save blob CSV")
        self.chk_infer_save_csv.setChecked(True)
        vbox.addWidget(self.chk_infer_save_csv)
        self.chk_infer_preserve = QCheckBox(
            "Preserve painted labels (skip labeled time regions)"
        )
        self.chk_infer_preserve.setChecked(True)
        vbox.addWidget(self.chk_infer_preserve)

        self.btn_infer_run = QPushButton("Run Inference")
        self.btn_infer_run.clicked.connect(self._on_inline_infer)
        self.btn_infer_run.setEnabled(False)
        vbox.addWidget(self.btn_infer_run)

        self.infer_panel = MADRunPanel(show_plot=False)
        self.infer_panel.run_finished.connect(
            lambda ok: self.btn_infer_run.setEnabled(self._project is not None)
        )
        vbox.addWidget(self.infer_panel)

        group.setLayout(vbox)
        layout.addWidget(group)

    # --- inline training/inference handlers ---------------------------
    def _on_left_tab_changed(self, _idx: int):
        # No-op hook for now; review hotkeys gate on the active tab.
        pass

    def _on_arch_changed(self, _idx: int = 0):
        # HRNet brings its own encoder, so the encoder picker is irrelevant.
        is_hrnet = self.combo_arch.currentData() == "hrnet"
        self.combo_train_encoder.setEnabled(not is_hrnet)

    def _build_inline_train_config(self):
        from fnt.usv.usv_detector.mad_training import UNetTrainingConfig
        sp = self._spec_params()
        return UNetTrainingConfig(
            project_dir=self._project.project_dir,
            run_name=self.txt_train_run.text().strip(),
            model_arch=self.combo_arch.currentData(),
            encoder_name=self.combo_train_encoder.currentText(),
            n_epochs=self.spin_train_epochs.value(),
            early_stop_patience=self.spin_train_patience.value(),
            batch_size=self.spin_train_batch.value(),
            learning_rate=self.spin_train_lr.value(),
            val_fraction=self.spin_train_val.value(),
            device=self.combo_train_device.currentText(),
            nperseg=sp['nperseg'], noverlap=sp['noverlap'], nfft=sp['nfft'],
            db_min=sp['db_min'], db_max=sp['db_max'],
            training_data_dir=self._project.training_data_dir,
        )

    def _on_inline_train(self):
        if self._project is None:
            QMessageBox.information(self, "No project", "Open a MAD project first.")
            return
        from fnt.usv.usv_detector.mad_examples import count_examples
        n_examples = count_examples(self._project.training_data_dir)
        if n_examples == 0:
            QMessageBox.warning(
                self, "No training examples",
                "No confirmed calls yet. Label a call (brush or SAM) and press "
                "Enter to confirm it before training."
            )
            return
        cfg = self._build_inline_train_config()
        post_scope = None
        if self.chk_post_inference.isChecked():
            post_scope = ('current'
                          if self.combo_post_scope.currentIndex() == 0 else 'all')
        self.btn_train.setEnabled(False)
        self._set_training_view(True)
        self.train_panel.start_run()
        self._start_training(cfg, post_inference_scope=post_scope,
                             reporter=self.train_panel)

    def _set_training_view(self, active: bool):
        """Swap the right-hand preview between the spectrogram and the live
        training graph (mirrors Mask Tracker: the graph replaces the preview
        while a run is active)."""
        self.train_panel.setVisible(active)
        for w in (self.spectrogram, self.waveform_overview,
                  self._scroll_bar_row, self._controls_bar):
            w.setVisible(not active)

    def _on_training_finished(self, ok: bool):
        self.btn_train.setEnabled(self._project is not None)
        self._set_training_view(False)

    def _default_model_path(self) -> Optional[str]:
        if self._project and self._project.models:
            last = self._project.models[-1]
            return last.get('path') if isinstance(last, dict) else str(last)
        return None

    def _pick_infer_model(self):
        root = (os.path.join(self._project.project_dir, 'models')
                if self._project else os.path.expanduser("~"))
        start = root if os.path.isdir(root) else os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(
            self, "Select trained weights", start,
            "PyTorch weights (*.pt *.pth);;All Files (*)"
        )
        if path:
            self.txt_infer_model.setText(path)

    def _on_inline_infer(self):
        if self._project is None:
            QMessageBox.information(self, "No project", "Open a MAD project first.")
            return
        if not self.audio_files:
            QMessageBox.warning(self, "No files", "Project has no wav files.")
            return
        model = self.txt_infer_model.text().strip() or self._default_model_path()
        if not model or not os.path.isfile(model):
            QMessageBox.warning(
                self, "Model missing",
                "Train a model or select a valid weights.pt file first."
            )
            return
        if self.rb_infer_current.isChecked():
            current = (self.audio_files[self.current_file_idx]
                       if self.audio_files else None)
            wavs = [current] if current else []
        else:
            wavs = list(self.audio_files)
        if not wavs:
            return
        from fnt.usv.usv_detector.mad_inference import MADInferenceConfig
        cfg = MADInferenceConfig(
            model_path=model,
            threshold=self.spin_infer_threshold.value(),
            min_blob_pixels=self.spin_infer_min_blob.value(),
            device=self.combo_infer_device.currentText(),
            save_mask_png=self.chk_infer_save_mask.isChecked(),
            save_blob_csv=self.chk_infer_save_csv.isChecked(),
            preserve_labels=self.chk_infer_preserve.isChecked(),
            training_data_dir=self._project.training_data_dir,
        )
        self.btn_infer_run.setEnabled(False)
        self.infer_panel.start_run()
        self._start_inference(cfg, wavs, reporter=self.infer_panel)

    # ------------------------------------------------------------------
    # Menu bar
    # ------------------------------------------------------------------
    def _setup_menu_bar(self):
        menubar = self.menuBar()
        menubar.setNativeMenuBar(False)

        file_menu = menubar.addMenu("&File")

        act_new = QAction("&New Project…", self)
        act_new.setShortcut(QKeySequence.New)
        act_new.triggered.connect(self._menu_new_project)
        file_menu.addAction(act_new)

        act_open = QAction("&Open Project…", self)
        act_open.setShortcut(QKeySequence.Open)
        act_open.triggered.connect(self._menu_open_project)
        file_menu.addAction(act_open)

        self.menu_recent = file_menu.addMenu("Open &Recent")
        self._rebuild_recent_menu()

        file_menu.addSeparator()

        self.act_add_folder = QAction("&Add Folder…", self)
        self.act_add_folder.setShortcut("Ctrl+Shift+O")
        self.act_add_folder.triggered.connect(self._menu_add_folder)
        self.act_add_folder.setEnabled(False)
        file_menu.addAction(self.act_add_folder)

        self.act_add_files = QAction("Add &Files…", self)
        self.act_add_files.triggered.connect(self._add_audio_files)
        self.act_add_files.setEnabled(False)
        file_menu.addAction(self.act_add_files)

        file_menu.addSeparator()

        self.act_close_project = QAction("&Close Project", self)
        self.act_close_project.triggered.connect(self._menu_close_project)
        self.act_close_project.setEnabled(False)
        file_menu.addAction(self.act_close_project)

        labels_menu = menubar.addMenu("&Labels")
        act_save_mask = QAction("&Save Current Mask", self)
        act_save_mask.setShortcut("Ctrl+S")
        act_save_mask.triggered.connect(self._save_current_mask)
        labels_menu.addAction(act_save_mask)

        predict_menu = menubar.addMenu("&Predict")
        self.act_run_training = QAction("Run &Training…", self)
        self.act_run_training.setShortcut("Ctrl+T")
        self.act_run_training.triggered.connect(self._menu_run_training)
        self.act_run_training.setEnabled(False)
        predict_menu.addAction(self.act_run_training)

        self.act_run_inference = QAction("Run &Inference…", self)
        self.act_run_inference.setShortcut("Ctrl+I")
        self.act_run_inference.triggered.connect(self._menu_run_inference)
        self.act_run_inference.setEnabled(False)
        predict_menu.addAction(self.act_run_inference)

        predict_menu.addSeparator()
        self.act_load_pred = QAction("&Load Predictions for Current File", self)
        self.act_load_pred.triggered.connect(self._load_predictions_for_current)
        self.act_load_pred.setEnabled(False)
        predict_menu.addAction(self.act_load_pred)

        self.act_clear_pred = QAction("&Clear Predictions Overlay", self)
        self.act_clear_pred.triggered.connect(self._clear_predictions)
        self.act_clear_pred.setEnabled(False)
        predict_menu.addAction(self.act_clear_pred)

    # ------------------------------------------------------------------
    # Keyboard shortcuts (CAD/DAD parity)
    # ------------------------------------------------------------------
    def _setup_shortcuts(self):
        def make(key, slot):
            sc = QShortcut(QKeySequence(key), self)
            sc.setContext(Qt.ApplicationShortcut)
            sc.activated.connect(slot)
            return sc

        make(Qt.Key_Left, self._shortcut_pan_left)
        make(Qt.Key_Right, self._shortcut_pan_right)
        make(Qt.Key_Up, self._shortcut_zoom_in)
        make(Qt.Key_Down, self._shortcut_zoom_out)
        make(Qt.Key_N, self._shortcut_next_file)
        make(Qt.Key_P, self._shortcut_prev_file)
        make(Qt.Key_B, self._shortcut_toggle_brush)
        make(Qt.Key_E, self._shortcut_toggle_eraser)
        make(Qt.Key_M, self._shortcut_mask_view)
        make(Qt.Key_G, self._shortcut_toggle_sam)
        make(Qt.Key_Return, self._confirm_pending)
        make(Qt.Key_Enter, self._confirm_pending)
        make(Qt.Key_Escape, self._clear_pending)
        make(Qt.Key_Space, self._shortcut_toggle_playback)
        make("[", self._shortcut_brush_smaller)
        make("]", self._shortcut_brush_bigger)
        make("V", self._shortcut_cycle_view)
        make("A", lambda: self._shortcut_review('accepted'))
        make("R", lambda: self._shortcut_review('rejected'))
        make("S", lambda: self._shortcut_review('skipped'))

    def _focus_is_edit(self):
        focus = QApplication.focusWidget()
        return isinstance(focus, (QSpinBox, QDoubleSpinBox, QComboBox))

    def _shortcut_pan_left(self):
        if not self._focus_is_edit():
            self._pan_left()

    def _shortcut_pan_right(self):
        if not self._focus_is_edit():
            self._pan_right()

    def _shortcut_zoom_in(self):
        if not self._focus_is_edit():
            self._zoom_in()

    def _shortcut_zoom_out(self):
        if not self._focus_is_edit():
            self._zoom_out()

    def _shortcut_next_file(self):
        if not self._focus_is_edit() and self.btn_next_file.isEnabled():
            self._next_file()

    def _shortcut_prev_file(self):
        if not self._focus_is_edit() and self.btn_prev_file.isEnabled():
            self._prev_file()

    def _shortcut_toggle_playback(self):
        if self._focus_is_edit():
            return
        if self.btn_play.isEnabled() or self.is_playing:
            self._toggle_playback()

    def _shortcut_toggle_brush(self):
        if self._focus_is_edit():
            return
        if not self.btn_paint.isEnabled():
            return
        new_state = not self.btn_paint.isChecked()
        self.btn_paint.setChecked(new_state)
        self._on_brush_clicked(new_state)

    def _shortcut_toggle_eraser(self):
        if self._focus_is_edit():
            return
        if not self.btn_erase.isEnabled():
            return
        new_state = not self.btn_erase.isChecked()
        self.btn_erase.setChecked(new_state)
        self._on_eraser_clicked(new_state)

    def _shortcut_mask_view(self):
        # M toggles the Mask-only view (back to Spec + Mask).
        if self._focus_is_edit():
            return
        mode = ('overlay' if self.spectrogram.view_mode == 'mask_only'
                else 'mask_only')
        self._set_view_mode(mode)

    def _shortcut_toggle_sam(self):
        if self._focus_is_edit():
            return
        if not self.btn_sam.isEnabled():
            return
        new_state = not self.btn_sam.isChecked()
        self.btn_sam.setChecked(new_state)
        self._on_sam_clicked(new_state)

    def _shortcut_brush_smaller(self):
        if self._focus_is_edit():
            return
        self.spin_brush_radius.setValue(max(1, self.spin_brush_radius.value() - 1))

    def _shortcut_brush_bigger(self):
        if self._focus_is_edit():
            return
        self.spin_brush_radius.setValue(min(64, self.spin_brush_radius.value() + 1))

    def _shortcut_cycle_view(self):
        if self._focus_is_edit():
            return
        order = ['overlay', 'spec', 'mask_only']
        try:
            idx = order.index(self.spectrogram.view_mode)
        except ValueError:
            idx = 0
        self._set_view_mode(order[(idx + 1) % len(order)])

    def _shortcut_review(self, status: str):
        if self._focus_is_edit():
            return
        # Blob review lives in the Inference tab — don't act from the Mask tab.
        if (hasattr(self, 'left_tabs') and
                self.left_tabs.currentIndex() != self._inference_tab_index):
            return
        if not self.btn_accept_blob.isEnabled():
            return
        self._review_current_blob(status)

    # ==================================================================
    # Project lifecycle
    # ==================================================================
    def _menu_new_project(self):
        name, ok = QInputDialog.getText(
            self, "New MAD Project", "Project name:", text="mad_v1"
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        parent_dir = QFileDialog.getExistingDirectory(
            self, f"Choose where to save '{name}'", os.path.expanduser("~")
        )
        if not parent_dir:
            return
        # Re-prompt for a different name as long as the target folder exists,
        # pre-filling a sensible non-colliding suggestion. The user can keep
        # editing or Cancel out entirely.
        project_dir = os.path.join(parent_dir, name)
        while os.path.exists(project_dir):
            suggestion = self._suggest_project_name(parent_dir, name)
            new_name, ok = QInputDialog.getText(
                self, "Project name already exists",
                f"A project folder named '{name}' already exists in:\n"
                f"{parent_dir}\n\nEnter a different project name:",
                text=suggestion,
            )
            if not ok or not new_name.strip():
                return
            name = new_name.strip()
            project_dir = os.path.join(parent_dir, name)
        try:
            cfg = create_mad_project(project_dir)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create project:\n{e}")
            return
        self._close_project(silent=True)
        self._activate_project(cfg)
        self._remember_recent(project_dir)
        self.status_bar.showMessage(f"Created project: {project_dir}")

    @staticmethod
    def _suggest_project_name(parent_dir: str, base: str) -> str:
        """Return the first ``base``-derived name with no existing folder in
        ``parent_dir``. A trailing number is incremented (``mad_v1`` →
        ``mad_v2``); otherwise ``_v2``, ``_v3``… is appended."""
        import re
        m = re.match(r'^(.*?)(\d+)$', base)
        if m:
            stem, start = m.group(1), int(m.group(2)) + 1
        else:
            stem, start = f"{base}_v", 2
        n = start
        while os.path.exists(os.path.join(parent_dir, f"{stem}{n}")):
            n += 1
        return f"{stem}{n}"

    def _menu_open_project(self):
        config_path, _ = QFileDialog.getOpenFileName(
            self, f"Open MAD project — select {PROJECT_INFO_FILENAME}",
            os.path.expanduser("~"),
            f"Project Info ({PROJECT_INFO_FILENAME});;All Files (*)",
        )
        if not config_path:
            return
        if os.path.basename(config_path) != PROJECT_INFO_FILENAME:
            QMessageBox.warning(
                self, "Not a MAD project",
                f"Expected {PROJECT_INFO_FILENAME} but got:\n"
                f"{os.path.basename(config_path)}"
            )
            return
        self._open_project_from_path(config_path)

    def _open_project_from_path(self, config_path: str):
        try:
            cfg = MADProjectConfig.load(config_path)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open project:\n{e}")
            return
        self._close_project(silent=True)
        self._activate_project(cfg)
        self._remember_recent(cfg.project_dir)

    def _menu_close_project(self):
        if self._project is None:
            return
        self._close_project(silent=False)

    def _menu_add_folder(self):
        if self._project is None:
            return
        folder = QFileDialog.getExistingDirectory(
            self, "Add folder of .wav files (non-recursive)",
            os.path.expanduser("~")
        )
        if not folder:
            return
        if folder in self._project.source_folders:
            QMessageBox.information(
                self, "Already added",
                f"'{folder}' is already part of this project."
            )
            return
        self._project.source_folders.append(folder)
        try:
            self._project.save()
        except Exception:
            pass
        added = self._rescan_project_wavs()
        self.status_bar.showMessage(
            f"Added folder: {folder} (+{added} new wavs)"
        )

    def _activate_project(self, cfg: MADProjectConfig):
        self._project = cfg
        self.setWindowTitle(f"{self.BASE_TITLE} — {cfg.project_name}")
        self.lbl_project_name.setText(f"Project: {cfg.project_name}")
        self.lbl_project_name.setStyleSheet("color: #4CAF50; font-size: 11px;")
        self.btn_add_folder.setEnabled(True)
        self.btn_add_files.setEnabled(True)
        self.act_add_folder.setEnabled(True)
        self.act_add_files.setEnabled(True)
        self.act_close_project.setEnabled(True)
        self.act_run_training.setEnabled(True)
        self.act_run_inference.setEnabled(True)
        self.act_load_pred.setEnabled(True)
        self.act_clear_pred.setEnabled(True)
        self.btn_train.setEnabled(True)
        self.btn_infer_run.setEnabled(True)
        self.txt_infer_model.setText(self._default_model_path() or "")
        self._rescan_project_wavs()
        self._update_source_folders_label()
        self._update_model_info_label()
        self.status_bar.showMessage(f"Opened project: {cfg.project_dir}")

    def _close_project(self, silent: bool = False):
        if self._project is None:
            return
        # Save the current file's mask if dirty before tearing down.
        self._auto_save_mask_if_dirty()
        if not silent:
            self.status_bar.showMessage(
                f"Closed project: {self._project.project_name}"
            )
        self._project = None
        self.audio_files = []
        self.current_file_idx = 0
        self.audio_data = None
        self.sample_rate = None
        self._stop_playback()

        self.file_list.blockSignals(True)
        self.file_list.clear()
        self.file_list.blockSignals(False)

        self.setWindowTitle(self.BASE_TITLE)
        self.lbl_project_name.setText("No project — click 'New Project…'")
        self.lbl_project_name.setStyleSheet("color: #999999; font-size: 10px;")
        self.lbl_source_folders.setText("No source folders")
        self.btn_add_folder.setEnabled(False)
        self.btn_add_files.setEnabled(False)
        self.act_add_folder.setEnabled(False)
        self.act_add_files.setEnabled(False)
        self.act_close_project.setEnabled(False)
        self.act_run_training.setEnabled(False)
        self.act_run_inference.setEnabled(False)
        self.act_load_pred.setEnabled(False)
        self.act_clear_pred.setEnabled(False)
        self.btn_train.setEnabled(False)
        self.btn_infer_run.setEnabled(False)
        self.spectrogram.mask = None
        self.spectrogram.set_audio_data(None, None)
        self.waveform_overview.set_audio_data(None, None)
        self._clear_predictions()
        self._update_project_state()
        self._update_paint_buttons_enabled()
        self._update_playback_buttons_enabled()
        self._update_model_info_label()

    # ==================================================================
    # File management
    # ==================================================================
    def _add_audio_files(self):
        if self._project is None:
            return
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio Files", os.path.expanduser("~"),
            "WAV Files (*.wav *.WAV);;All Files (*.*)"
        )
        if not files:
            return
        to_add = [f for f in files if f not in self.audio_files]
        dup = len(files) - len(to_add)
        for f in to_add:
            self.audio_files.append(f)
        if to_add:
            if self.current_file_idx >= len(self.audio_files):
                self.current_file_idx = 0
            self._refresh_file_list()
            self.file_list.blockSignals(True)
            self.file_list.setCurrentRow(self.current_file_idx)
            self.file_list.blockSignals(False)
            self._load_current_file()
        self._update_project_state()
        msg = f"Added {len(to_add)} file(s)"
        if dup:
            msg += f", skipped {dup} already imported"
        self.status_bar.showMessage(msg)

    def _rescan_project_wavs(self) -> int:
        if self._project is None:
            return 0
        wavs: List[str] = []
        for folder in self._project.source_folders:
            wavs.extend(_list_wavs_in_folder(folder))
        existing = set(self.audio_files)
        added = 0
        for w in wavs:
            if w not in existing:
                self.audio_files.append(w)
                existing.add(w)
                added += 1

        if self.audio_files:
            self.current_file_idx = min(
                self.current_file_idx, len(self.audio_files) - 1
            )
            self._refresh_file_list()
            last = self._project.last_opened_file
            if last and last in self.audio_files:
                self.current_file_idx = self.audio_files.index(last)
            self.file_list.blockSignals(True)
            self.file_list.setCurrentRow(self.current_file_idx)
            self.file_list.blockSignals(False)
            self._load_current_file()
        else:
            self.file_list.blockSignals(True)
            self.file_list.clear()
            self.file_list.blockSignals(False)

        self._update_source_folders_label()
        self._update_project_state()
        return added

    def _refresh_file_list(self):
        self.file_list.blockSignals(True)
        self.file_list.clear()
        for fp in self.audio_files:
            item = QListWidgetItem(os.path.basename(fp))
            item.setData(Qt.UserRole, fp)
            item.setToolTip(fp)
            self.file_list.addItem(item)
        self.file_list.blockSignals(False)

    def _update_source_folders_label(self):
        cfg = self._project
        if cfg is None or not cfg.source_folders:
            self.lbl_source_folders.setText(
                "No source folders (use 'Add Folder…')"
            )
            return
        names = [
            os.path.basename(os.path.normpath(f)) or f
            for f in cfg.source_folders
        ]
        n = len(names)
        self.lbl_source_folders.setText(
            f"{n} folder(s): " + ", ".join(names[:3]) + (" …" if n > 3 else "")
        )

    def _update_project_state(self):
        n = len(self.audio_files)
        self.btn_prev_file.setEnabled(n > 0 and self.current_file_idx > 0)
        self.btn_next_file.setEnabled(n > 0 and self.current_file_idx < n - 1)
        self.lbl_file_num.setText(
            f"File {self.current_file_idx + 1}/{n}" if n else "File 0/0"
        )
        if n == 0:
            self.lbl_data_summary.setText(
                "No files loaded" if self._project is None
                else "Project opened — add files or a folder"
            )
        else:
            self.lbl_data_summary.setText(f"{n} file(s) in project")

    def _on_file_selected(self, row: int):
        if 0 <= row < len(self.audio_files):
            if row != self.current_file_idx:
                self._auto_save_mask_if_dirty()
                self.current_file_idx = row
                self._load_current_file()
                self._update_project_state()

    def _prev_file(self):
        if self.current_file_idx > 0:
            self.file_list.setCurrentRow(self.current_file_idx - 1)

    def _next_file(self):
        if self.current_file_idx < len(self.audio_files) - 1:
            self.file_list.setCurrentRow(self.current_file_idx + 1)

    def _load_current_file(self):
        if not self.audio_files or self.current_file_idx >= len(self.audio_files):
            return
        self._stop_playback()
        # Wipe any prediction overlay from the previous file — a new
        # file's predictions must be explicitly reloaded.
        self._clear_predictions()
        filepath = self.audio_files[self.current_file_idx]
        self.status_bar.showMessage(f"Loading {os.path.basename(filepath)}…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        QApplication.processEvents()
        try:
            audio, sr = load_audio(filepath)
            if audio.ndim > 1:
                audio = np.mean(audio, axis=1)
            self.audio_data = audio.astype(np.float32, copy=False)
            self.sample_rate = int(sr)
            has_previous = self.spectrogram.total_duration > 0
            self.spectrogram.set_audio_data(
                self.audio_data, self.sample_rate, preserve_view=has_previous
            )
            self.waveform_overview.set_audio_data(
                self.audio_data, self.sample_rate
            )
            self.waveform_overview.set_view_range(
                self.spectrogram.view_start, self.spectrogram.view_end
            )
            self._init_or_load_mask_for_current_file()
            self._sync_scrollbar_from_view()
            if self._project is not None:
                self._project.last_opened_file = filepath
                try:
                    self._project.save()
                except Exception:
                    pass
            self.status_bar.showMessage(
                f"{os.path.basename(filepath)}  |  "
                f"{len(self.audio_data) / self.sample_rate:.2f}s @ "
                f"{self.sample_rate} Hz"
            )
        except Exception as e:
            QMessageBox.warning(
                self, "Load error",
                f"{os.path.basename(filepath)}:\n{e}"
            )
            self.status_bar.showMessage(
                f"Failed to load {os.path.basename(filepath)}"
            )
        finally:
            QApplication.restoreOverrideCursor()
        self._update_paint_buttons_enabled()
        self._update_playback_buttons_enabled()

    def _init_or_load_mask_for_current_file(self):
        """Initialize the spec-pixel grid, then rebuild the confirmed mask for
        this file from the saved example store (no sibling PNG / WAV needed)."""
        if (self._project is None or self.audio_data is None or
                self.sample_rate is None):
            return
        cfg = self._project
        self.spectrogram.init_mask(
            audio_len=len(self.audio_data),
            sample_rate=self.sample_rate,
            nperseg=cfg.nperseg, noverlap=cfg.noverlap, nfft=cfg.nfft,
        )
        from fnt.usv.usv_detector.mad_examples import reconstruct_file_mask
        wav_name = os.path.basename(self.audio_files[self.current_file_idx])
        grid = (self.spectrogram.n_freq_bins, self.spectrogram.n_time_frames)
        try:
            confirmed = reconstruct_file_mask(cfg.training_data_dir, wav_name, grid)
        except Exception:
            confirmed = None
        if confirmed is not None and confirmed.any():
            self.spectrogram.set_mask(confirmed)
            self.lbl_mask_status.setText(
                f"Loaded confirmed calls for {wav_name}"
            )
        else:
            self.lbl_mask_status.setText(
                "No labels yet — paint or SAM a call, then Enter to confirm"
            )

    def _save_current_mask(self):
        # Labels are now saved per-call as training examples on confirm (Enter);
        # there is no per-file mask to write. Kept so the Ctrl+S menu/no-ops.
        self.status_bar.showMessage(
            "Labels are saved automatically when you confirm a call (Enter)."
        )

    def _auto_save_mask_if_dirty(self):
        # No-op: confirmed calls persist as examples at confirm time.
        return

    def _on_stroke_committed(self):
        # A brush stroke just ended; nothing to persist until the user
        # confirms the pending mask with Enter.
        return

    # ==================================================================
    # Paint tools
    # ==================================================================
    def _update_paint_buttons_enabled(self):
        has_audio = self.audio_data is not None
        for btn in (self.btn_paint, self.btn_erase, self.btn_clear_mask,
                    self.btn_sam):
            btn.setEnabled(has_audio)
        self.spin_brush_radius.setEnabled(has_audio)
        self.btn_load_predictions.setEnabled(has_audio)
        if not has_audio:
            self.btn_paint.setChecked(False)
            self.btn_erase.setChecked(False)
            self.btn_sam.setChecked(False)
            self.spectrogram.set_paint_mode(None)
            self.lbl_mask_status.setText("No mask")

    def _on_brush_clicked(self, checked: bool):
        if checked:
            self.btn_erase.setChecked(False)
            self.btn_sam.setChecked(False)
            self.spectrogram.set_paint_mode('brush')
            self.status_bar.showMessage(
                "Brush mode — left-click + drag over target pixels"
            )
        else:
            self.spectrogram.set_paint_mode(None)
            self.status_bar.showMessage("Paint mode off")

    def _on_eraser_clicked(self, checked: bool):
        if checked:
            self.btn_paint.setChecked(False)
            self.btn_sam.setChecked(False)
            self.spectrogram.set_paint_mode('eraser')
            self.status_bar.showMessage(
                "Eraser mode — left-click + drag to clear painted pixels"
            )
        else:
            self.spectrogram.set_paint_mode(None)
            self.status_bar.showMessage("Paint mode off")

    # ==================================================================
    # SAM2-assisted labeling
    # ==================================================================
    def _pick_sam_checkpoint(self) -> bool:
        """Choose a SAM2 checkpoint, mirroring Mask Tracker.

        Offers any checkpoints already present in ``LocalModels/`` (plus the
        download option); if none are found, prompts to download one (saved
        to ``LocalModels/``). Sets ``self._sam_ckpt`` / ``self._sam_cfg_name``
        and returns True on success.
        """
        try:
            from fnt.videoTracking.sam2_checkpoint_manager import (
                SAM2_CHECKPOINTS, LOCAL_MODELS_DIR, _LEGACY_DIRS,
                _find_existing_checkpoints,
            )
        except Exception as e:
            QMessageBox.critical(
                self, "SAM2 unavailable",
                f"Could not load the SAM2 checkpoint manager:\n{e}"
            )
            return False

        found = _find_existing_checkpoints(LOCAL_MODELS_DIR, *_LEGACY_DIRS)
        if found:
            names = list(found.keys())
            descriptions = [
                f"{n} ({SAM2_CHECKPOINTS.get(n, {}).get('size_mb', '?')} MB)"
                for n in names
            ]
            download_label = "Download additional model…"
            descriptions.append(download_label)
            choice, ok = QInputDialog.getItem(
                self, "Select SAM2 Model",
                "Found existing SAM2 model(s) in LocalModels. Select one:",
                descriptions, 0, False,
            )
            if not ok:
                return False
            if choice == download_label:
                return self._download_sam_checkpoint()
            name = names[descriptions.index(choice)]
            self._sam_ckpt = str(found[name])
            self._sam_cfg_name = SAM2_CHECKPOINTS.get(name, {}).get("config")
            return True

        reply = QMessageBox.question(
            self, "SAM2 Model Required",
            "No SAM2 model found in LocalModels. Download one?\n\n"
            "(Requires the sam2 package:\n"
            "  pip install git+https://github.com/facebookresearch/sam2.git)",
            QMessageBox.Yes | QMessageBox.No,
        )
        if reply == QMessageBox.Yes:
            return self._download_sam_checkpoint()
        return False

    def _download_sam_checkpoint(self) -> bool:
        """Open the shared SAM2 download dialog (saves to LocalModels)."""
        from fnt.videoTracking.sam2_checkpoint_manager import get_sam2_checkpoint
        ckpt_path, cfg_name = get_sam2_checkpoint(parent=self)
        if not ckpt_path:
            return False
        self._sam_ckpt = str(ckpt_path)
        self._sam_cfg_name = cfg_name
        return True

    def _on_sam_clicked(self, checked: bool):
        if not checked:
            self.spectrogram.set_paint_mode(None)
            self.status_bar.showMessage("SAM mode off")
            return
        if not self._ensure_sam_model():
            self.btn_sam.setChecked(False)
            return
        self.btn_paint.setChecked(False)
        self.btn_erase.setChecked(False)
        self.spectrogram.set_paint_mode('sam')
        self.status_bar.showMessage(
            "SAM mode — left-click a call (right-click = negative), "
            "Enter to accept, Esc to clear"
        )

    def _ensure_sam_model(self) -> bool:
        """Make sure a SAM2 segmenter exists and is loading/loaded. Returns
        False if the user declined to choose/download a checkpoint."""
        if self._sam_segmenter is not None:
            return True  # already created (loading or loaded)
        if not self._sam_ckpt:
            if not self._pick_sam_checkpoint():
                return False
        if self._sam_segmenter is None:
            try:
                from fnt.videoTracking.mask_tracker_annotator import (
                    SAM2ImageSegmenter,
                )
            except Exception as e:
                QMessageBox.critical(
                    self, "SAM2 unavailable",
                    f"Could not import the SAM2 backend:\n{e}\n\n"
                    "Install the SAM2 dependencies (torch, sam2)."
                )
                return False
            self._sam_segmenter = SAM2ImageSegmenter(
                self._sam_ckpt, self._sam_cfg_name
            )
            self._sam_ready = False
            self._sam_img_sig = None
            self._sam_load_worker = MADSamLoadWorker(self._sam_segmenter, self)
            self._sam_load_worker.done_signal.connect(self._on_sam_loaded)
            self._sam_load_worker.error_signal.connect(self._on_sam_error)
            self.status_bar.showMessage("Loading SAM2 model…")
            self._sam_load_worker.start()
        return True

    def _on_sam_loaded(self):
        self._sam_ready = True
        self.status_bar.showMessage("SAM2 model ready — click a call to label")
        # If the user already placed points while the model loaded, predict now.
        if (self.spectrogram.paint_mode == 'sam' and
                self.spectrogram.has_sam_prompts()):
            self._run_sam_predict()

    def _on_sam_error(self, msg: str):
        self._sam_ready = False
        self._sam_segmenter = None
        self.spectrogram.set_paint_mode(None)
        self.btn_sam.setChecked(False)
        QMessageBox.critical(self, "SAM2 error", msg)

    def _view_signature(self):
        """Identity of the currently rendered SAM image — when this changes
        the image embedding must be recomputed."""
        sg = self.spectrogram
        return (self.current_file_idx, round(sg.view_start, 4),
                round(sg.view_end, 4), sg.min_freq, sg.max_freq)

    def _on_sam_points_changed(self):
        if self.spectrogram.paint_mode != 'sam':
            return
        if not self.spectrogram.has_sam_prompts():
            self.spectrogram.set_sam_preview(None, 0)
            return
        self._run_sam_predict()

    def _run_sam_predict(self):
        if not self._sam_ready or self._sam_segmenter is None:
            # Model still loading; the next point change will retry.
            return
        if self._sam_predict_worker is not None and \
                self._sam_predict_worker.isRunning():
            self._sam_predict_pending = True
            return
        pos_pts, neg_pts = self.spectrogram.get_sam_prompts()
        if not pos_pts:
            self.spectrogram.set_sam_preview(None, 0)
            return
        sig = self._view_signature()
        reuse = (sig == self._sam_img_sig and self._sam_last_t_off is not None)
        if reuse:
            # View hasn't moved — reuse the cached embedding, skip rendering.
            image_rgb = None
            t_off = self._sam_last_t_off
        else:
            rendered = self.spectrogram.render_sam_image()
            if rendered is None:
                return
            image_rgb, t_off = rendered
            self._sam_img_sig = sig
            self._sam_last_t_off = t_off
        # Convert prompts from full-grid (t, f) to crop coords (x=t-t_off, y=f).
        pos_crop = [(int(t - t_off), int(f)) for (t, f) in pos_pts]
        neg_crop = [(int(t - t_off), int(f)) for (t, f) in neg_pts]
        self._sam_predict_worker = MADSamPredictWorker(
            self._sam_segmenter, image_rgb, pos_crop, neg_crop, t_off, self
        )
        self._sam_predict_worker.done_signal.connect(self._on_sam_predicted)
        self._sam_predict_worker.error_signal.connect(self._on_sam_error)
        self._sam_predict_worker.start()

    def _on_sam_predicted(self, mask, t_off: int):
        if self.spectrogram.paint_mode == 'sam':
            self.spectrogram.set_sam_preview(mask, t_off)
        if self._sam_predict_pending:
            self._sam_predict_pending = False
            self._run_sam_predict()

    def _confirm_pending(self):
        """Enter — confirm the pending mask (manual or SAM): ask for a class,
        save it as a self-contained training example, then merge it into the
        confirmed buffer."""
        sg = self.spectrogram
        if not sg.has_pending() or self._project is None or \
                self.audio_data is None:
            return
        classes = list(self._project.classes) or ["USV"]
        last = self._project.last_class or classes[0]
        if last not in classes:
            classes = [last] + classes
        cur_idx = classes.index(last)
        name, ok = QInputDialog.getItem(
            self, "Confirm call", "Call type (Enter to reuse):",
            classes, cur_idx, True,
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        if name not in self._project.classes:
            self._project.classes.append(name)
        self._project.last_class = name
        try:
            self._save_pending_as_example(name)
        except Exception as e:
            QMessageBox.critical(self, "Save failed",
                                 f"Could not save training example:\n{e}")
            return
        try:
            self._project.save()
        except Exception:
            pass
        sg.confirm_pending()
        self.lbl_mask_status.setText(f"Confirmed call — class '{name}'")
        self.status_bar.showMessage(
            f"Saved training example (class '{name}') — label the next call"
        )

    def _save_pending_as_example(self, class_name: str) -> None:
        from datetime import datetime
        from scipy import signal as _signal
        from fnt.usv.usv_detector.mad_dataset import spec_to_image
        from fnt.usv.usv_detector.mad_examples import save_example

        sg = self.spectrogram
        cfg = self._project
        bbox = sg.pending_bbox()
        if bbox is None:
            return
        f0, f1, t0, t1 = bbox
        hop = sg.hop
        nperseg, nfft = cfg.nperseg, cfg.nfft
        n_time, n_freq = sg.n_time_frames, sg.n_freq_bins
        sr = self.sample_rate
        margin = 64  # ~quarter tile of time-frame context around the call
        pt0 = max(0, t0 - margin)
        pt1 = min(n_time, t1 + margin)

        start_sample = pt0 * hop
        end_sample = min(len(self.audio_data), (pt1 - 1) * hop + nperseg)
        segment = self.audio_data[start_sample:end_sample]
        if len(segment) < nperseg:
            raise RuntimeError("call window too short to compute a patch")
        noverlap = min(cfg.noverlap, nperseg - 1)
        _f, _t, Sxx = _signal.spectrogram(
            segment, fs=sr, nperseg=nperseg, noverlap=noverlap,
            nfft=nfft, window='hann',
        )
        spec_db = 10.0 * np.log10(Sxx + 1e-10)
        spec_patch = spec_to_image(spec_db, cfg.db_min, cfg.db_max)
        W = spec_patch.shape[1]

        # Target = all confirmed positives ∪ this pending mask, within the
        # patch window — so adjacent confirmed calls aren't marked negative.
        combined = np.maximum(sg.mask, sg.get_pending())
        mask_patch = np.zeros((n_freq, W), dtype=np.uint8)
        src = combined[:, pt0:min(n_time, pt0 + W)]
        mask_patch[:, :src.shape[1]] = src[:n_freq, :]

        df = (sr / 2.0) / (nfft // 2)   # Hz per freq bin
        dt = hop / float(sr)            # seconds per time frame
        wav_name = os.path.basename(self.audio_files[self.current_file_idx])
        meta = {
            'class': class_name,
            'source_wav': wav_name,
            'patch_t_off': int(pt0), 'patch_f_off': 0,
            't_start_s': round(t0 * dt, 6), 't_stop_s': round(t1 * dt, 6),
            'patch_t0_s': round(pt0 * dt, 6), 'patch_t1_s': round(pt1 * dt, 6),
            'f_low_hz': round(f0 * df, 2), 'f_high_hz': round(f1 * df, 2),
            'patch_t_frames': int(W), 'f_bins': int(n_freq),
            'nperseg': nperseg, 'noverlap': cfg.noverlap, 'nfft': nfft,
            'sample_rate': int(sr),
            'db_min': cfg.db_min, 'db_max': cfg.db_max,
            'created': datetime.now().isoformat(timespec='seconds'),
        }
        save_example(cfg.training_data_dir, spec_patch, mask_patch, meta)

    def _clear_pending(self):
        sg = self.spectrogram
        if sg.has_pending() or sg.has_sam_prompts():
            sg.clear_pending()
            sg.clear_sam_prompts()
            self.status_bar.showMessage("Pending mask cleared")

    def _on_clear_clicked(self):
        # Discards the in-progress pending mask. Confirmed calls are saved
        # examples and are not affected (delete them in the training_data
        # folder if needed).
        if not self.spectrogram.has_pending():
            self.status_bar.showMessage("Nothing pending to clear.")
            return
        self.spectrogram.clear_pending()
        self.spectrogram.clear_sam_prompts()
        self.lbl_mask_status.setText("Pending mask cleared")

    def _on_brush_radius_scrolled(self, value: int):
        # Reflect a wheel-driven radius change in the spin box without
        # re-triggering set_brush_radius (the widget already applied it).
        self.spin_brush_radius.blockSignals(True)
        self.spin_brush_radius.setValue(value)
        self.spin_brush_radius.blockSignals(False)

    # ==================================================================
    # Spectrogram view controls
    # ==================================================================
    def _on_view_window_changed(self, value: float):
        if self.spectrogram.total_duration <= 0:
            return
        center = (self.spectrogram.view_start + self.spectrogram.view_end) / 2
        half = value / 2
        self.spectrogram.view_start = max(0.0, center - half)
        self.spectrogram.view_end = min(
            self.spectrogram.total_duration, self.spectrogram.view_start + value
        )
        self._invalidate_spec_cache()
        self._sync_scrollbar_from_view()

    def _on_display_freq_changed(self, _value: int = 0):
        if self.spin_display_min_freq.value() >= self.spin_display_max_freq.value():
            return
        self.spectrogram.min_freq = self.spin_display_min_freq.value()
        self.spectrogram.max_freq = self.spin_display_max_freq.value()
        self._invalidate_spec_cache()

    def _on_colormap_changed(self, name: str):
        self.spectrogram.set_colormap(name)

    def _on_wheel_zoom(self, factor: float, center_time: float):
        if self.spectrogram.total_duration <= 0:
            return
        new_window = max(0.1, min(600.0, self.spin_view_window.value() * factor))
        self.spin_view_window.blockSignals(True)
        self.spin_view_window.setValue(new_window)
        self.spin_view_window.blockSignals(False)
        half = new_window / 2
        self.spectrogram.view_start = max(0.0, center_time - half)
        self.spectrogram.view_end = min(
            self.spectrogram.total_duration,
            self.spectrogram.view_start + new_window,
        )
        self._invalidate_spec_cache()
        self._sync_scrollbar_from_view()

    def _zoom_in(self):
        self.spin_view_window.setValue(max(0.1, self.spin_view_window.value() / 2))

    def _zoom_out(self):
        self.spin_view_window.setValue(min(600.0, self.spin_view_window.value() * 2))

    def _pan_left(self):
        self._pan_by(-self.spin_view_window.value() / 4)

    def _pan_right(self):
        self._pan_by(self.spin_view_window.value() / 4)

    def _pan_by(self, delta_s: float):
        if self.spectrogram.total_duration <= 0:
            return
        window = self.spectrogram.view_end - self.spectrogram.view_start
        new_start = max(
            0.0,
            min(self.spectrogram.total_duration - window,
                self.spectrogram.view_start + delta_s),
        )
        self.spectrogram.view_start = new_start
        self.spectrogram.view_end = new_start + window
        self._invalidate_spec_cache()
        self._sync_scrollbar_from_view()

    def _on_scrollbar_changed(self, value: int):
        if self.spectrogram.total_duration <= 0:
            return
        window = self.spectrogram.view_end - self.spectrogram.view_start
        max_start = max(0.0, self.spectrogram.total_duration - window)
        new_start = (value / 1000.0) * max_start
        if abs(new_start - self.spectrogram.view_start) < 1e-6:
            return
        self.spectrogram.view_start = new_start
        self.spectrogram.view_end = new_start + window
        self._invalidate_spec_cache()

    def _sync_scrollbar_from_view(self):
        if self.spectrogram.total_duration <= 0:
            return
        window = self.spectrogram.view_end - self.spectrogram.view_start
        max_start = max(1e-9, self.spectrogram.total_duration - window)
        frac = self.spectrogram.view_start / max_start
        value = int(max(0.0, min(1.0, frac)) * 1000)
        self.time_scrollbar.blockSignals(True)
        self.time_scrollbar.setValue(value)
        self.time_scrollbar.blockSignals(False)

    def _on_overview_clicked(self, start: float, end: float):
        if self.spectrogram.total_duration <= 0:
            return
        self.spectrogram.view_start = max(0.0, start)
        self.spectrogram.view_end = min(self.spectrogram.total_duration, end)
        self._invalidate_spec_cache()
        self.spin_view_window.blockSignals(True)
        self.spin_view_window.setValue(
            self.spectrogram.view_end - self.spectrogram.view_start
        )
        self.spin_view_window.blockSignals(False)
        self._sync_scrollbar_from_view()

    def _invalidate_spec_cache(self):
        self.spectrogram.cached_view_start = None
        self.spectrogram.cached_view_end = None
        if self.spectrogram.total_duration > 0:
            self.spectrogram._compute_view_spectrogram()
        else:
            self.spectrogram.spec_image = None
        self.spectrogram.update()
        # Keep the overview's viewport highlight in sync with the zoom window
        # (single chokepoint — every zoom/pan path routes through here).
        self.waveform_overview.set_view_range(
            self.spectrogram.view_start, self.spectrogram.view_end
        )

    # ==================================================================
    # Playback (mirrors CAD/DAD)
    # ==================================================================
    def _update_playback_buttons_enabled(self):
        has_audio = self.audio_data is not None and HAS_SOUNDDEVICE
        self.btn_play.setEnabled(has_audio)
        self.btn_stop.setEnabled(has_audio)
        self.slider_speed.setEnabled(has_audio)

    def _on_speed_changed(self, idx: int):
        speed = self._speed_values[idx]
        self.playback_speed = speed
        self.lbl_speed.setText(f"{speed}x")

    def _toggle_playback(self):
        if self.is_playing:
            self._stop_playback()
        else:
            self._play_visible()

    def _play_visible(self):
        if not HAS_SOUNDDEVICE or self.audio_data is None:
            return
        start_s, stop_s = self.spectrogram.get_view_range()
        total_duration = len(self.audio_data) / self.sample_rate
        start_s = max(0, start_s)
        stop_s = min(total_duration, stop_s)
        start_sample = int(start_s * self.sample_rate)
        stop_sample = int(stop_s * self.sample_rate)
        segment = self.audio_data[start_sample:stop_sample].copy()
        if len(segment) < 100:
            return
        try:
            output_sr = 44100
            original_duration = len(segment) / self.sample_rate
            output_duration = original_duration / self.playback_speed
            n_output_samples = int(output_duration * output_sr)
            if n_output_samples < 100:
                return
            segment = signal.resample(segment, n_output_samples).astype(np.float32)
            sd.play(segment, output_sr)
            self.is_playing = True
            self.btn_play.setText("Playing…")
            import time as _time
            self._playback_start_time = _time.time()
            self._playback_start_s = start_s
            self._playback_end_s = stop_s
            self._playback_timer.start()
        except Exception as e:
            self.status_bar.showMessage(f"Playback error: {e}")

    def _stop_playback(self):
        if HAS_SOUNDDEVICE:
            try:
                sd.stop()
            except Exception:
                pass
        self.is_playing = False
        self.btn_play.setText("Play")
        self._playback_timer.stop()
        self.spectrogram.playback_position = None
        self.spectrogram.update()

    def _update_playback_position(self):
        import time as _time
        if not self.is_playing or self._playback_start_time is None:
            self._stop_playback()
            return
        elapsed = (
            _time.time() - self._playback_start_time - self._playback_latency
        )
        elapsed = max(0.0, elapsed)
        audio_elapsed = elapsed * self.playback_speed
        current_pos = self._playback_start_s + audio_elapsed
        if current_pos >= self._playback_end_s:
            self._stop_playback()
            return
        self.spectrogram.playback_position = current_pos
        self.spectrogram.update()

    # ==================================================================
    # Recent projects
    # ==================================================================
    def _remember_recent(self, project_dir: str):
        recent = self._settings.value(RECENT_PROJECTS_KEY, [], type=list) or []
        recent = [p for p in recent if p != project_dir]
        recent.insert(0, project_dir)
        recent = recent[:MAX_RECENT_PROJECTS]
        self._settings.setValue(RECENT_PROJECTS_KEY, recent)
        self._rebuild_recent_menu()

    def _rebuild_recent_menu(self):
        self.menu_recent.clear()
        recent = self._settings.value(RECENT_PROJECTS_KEY, [], type=list) or []
        recent = [p for p in recent if os.path.isdir(p)]
        if not recent:
            empty = QAction("(none)", self)
            empty.setEnabled(False)
            self.menu_recent.addAction(empty)
            return
        for p in recent:
            act = QAction(os.path.basename(os.path.normpath(p)) or p, self)
            act.setToolTip(p)
            act.triggered.connect(
                lambda _checked=False, path=p: self._open_recent(path)
            )
            self.menu_recent.addAction(act)

    def _open_recent(self, project_dir: str):
        info_path = os.path.join(project_dir, PROJECT_INFO_FILENAME)
        if not os.path.isfile(info_path):
            QMessageBox.warning(
                self, "Project not found",
                f"{PROJECT_INFO_FILENAME} missing in:\n{project_dir}"
            )
            return
        self._open_project_from_path(info_path)

    # ==================================================================
    # Training / Inference launch
    # ==================================================================
    def _labeled_wavs(self) -> List[str]:
        """Only wavs with sibling _FNT_MAD_labels.png."""
        out: List[str] = []
        for w in self.audio_files:
            png = _mask_sibling_path(w)
            if os.path.isfile(png):
                out.append(w)
        return out

    def _spec_params(self) -> dict:
        cfg = self._project
        if cfg is None:
            return dict(nperseg=512, noverlap=384, nfft=1024,
                        db_min=-100.0, db_max=-20.0)
        return dict(
            nperseg=cfg.nperseg, noverlap=cfg.noverlap, nfft=cfg.nfft,
            db_min=cfg.db_min, db_max=cfg.db_max,
        )

    def _menu_run_training(self):
        # Training lives inline in the Mask tab — route the menu/shortcut there
        # so there's a single code path using the current architecture/params.
        self.left_tabs.setCurrentIndex(0)
        self._on_inline_train()

    def _start_training(self, cfg, post_inference_scope: Optional[str] = None,
                        reporter=None):
        # ``reporter`` lets the inline Mask-tab panel reuse this launcher; when
        # None we fall back to the standalone modal progress dialog.
        owns_modal = reporter is None
        progress = reporter or MADRunProgressDialog(
            self, "MAD Training", show_plot=True
        )
        progress.set_stage(
            f"Training U-Net ({cfg.encoder_name}, up to {cfg.n_epochs} epochs, "
            f"patience={cfg.early_stop_patience})…"
        )
        progress.append(f"Run dir: {cfg.resolve_run_dir()}")
        progress.append(f"Files: {len(cfg.wav_paths)} labeled wav(s)")

        worker = MADTrainingWorker(cfg, parent=self)

        def on_progress(epoch: int, total: int, metrics: dict):
            status = metrics.get('status', '')
            if status == 'collecting_tiles':
                progress.set_stage(
                    f"Collecting tiles from {metrics.get('file_name', '…')} "
                    f"({metrics.get('file_i', 0)}/{metrics.get('file_n', 0)})"
                )
                progress.set_main(metrics.get('file_i', 0),
                                  max(1, metrics.get('file_n', 1)))
                progress.set_sub(0, 1)
            elif status == 'batch':
                bi = metrics.get('batch_i', 0)
                bn = max(1, metrics.get('batches_per_epoch', 1))
                progress.set_sub(bi, bn)
                progress.plot_batch(
                    metrics.get('global_batch', 0),
                    metrics.get('batch_loss', float('nan')),
                )
            elif status == 'training':
                pstr = (f" (no-improve {metrics.get('epochs_without_improvement', 0)}"
                        f"/{metrics.get('patience', 0)})"
                        if metrics.get('patience', 0) else "")
                progress.set_stage(
                    f"Epoch {epoch}/{total} — "
                    f"train_loss={metrics.get('train_loss', 0):.4f}  "
                    f"val_loss={metrics.get('val_loss', 0):.4f}  "
                    f"val_dice={metrics.get('val_dice', 0):.3f}"
                    f"{pstr}"
                )
                progress.set_main(epoch, total)
                progress.append(
                    f"  epoch {epoch}: "
                    f"train={metrics.get('train_loss', 0):.4f} "
                    f"val={metrics.get('val_loss', 0):.4f} "
                    f"dice={metrics.get('val_dice', 0):.3f}"
                )
                progress.plot_epoch(
                    metrics.get('global_batch', 0),
                    metrics.get('train_loss', float('nan')),
                    metrics.get('val_loss', float('nan')),
                )
            elif status == 'early_stop':
                progress.append(
                    f"  Early stop at epoch {epoch} — best_val_loss="
                    f"{metrics.get('best_val_loss', 0):.4f}"
                )
            elif status == 'done':
                progress.set_main(total, total)

        def on_finished(summary: dict):
            if summary.get('early_stopped'):
                progress.append(
                    "\nEarly stopped — val_loss plateaued for "
                    f"{cfg.early_stop_patience} epoch(s)."
                )
            progress.append(
                f"\nFinished. best_val_loss={summary.get('best_val_loss', 0):.4f}\n"
                f"Model: {summary.get('model_path')}"
            )
            progress.mark_done(ok=True)
            self.status_bar.showMessage(
                f"Training complete — {summary.get('model_path')}"
            )
            if self._project is not None:
                mpath = summary.get('model_path')
                if mpath:
                    from datetime import datetime as _dt
                    models = list(self._project.models or [])
                    paths = {m.get('path') for m in models
                             if isinstance(m, dict)}
                    if mpath not in paths:
                        models.append({
                            'name': Path(mpath).parent.name,
                            'arch': getattr(cfg, 'model_arch', 'unet'),
                            'encoder': cfg.encoder_name,
                            'path': mpath,
                            'date': _dt.now().isoformat(timespec='seconds'),
                            'best_val_loss': summary.get('best_val_loss'),
                            'n_train_tiles': summary.get('n_train_tiles'),
                            'n_val_tiles': summary.get('n_val_tiles'),
                        })
                        self._project.models = models
                        try:
                            self._project.save()
                        except Exception:
                            pass
                        self._update_model_info_label()
                    # Point the Inference tab at the freshly trained model
                    # unless the user has typed a custom path.
                    if (hasattr(self, 'txt_infer_model') and
                            not self.txt_infer_model.text().strip()):
                        self.txt_infer_model.setText(mpath)

            # Post-training inference chain (on the just-saved weights).
            if post_inference_scope and summary.get('model_path'):
                QTimer.singleShot(
                    200,
                    lambda: self._run_post_training_inference(
                        summary['model_path'], post_inference_scope,
                    ),
                )

        def on_error(msg: str):
            progress.append("\nERROR:\n" + msg)
            progress.mark_done(ok=False)

        worker.progress_signal.connect(on_progress)
        worker.finished_signal.connect(on_finished)
        worker.error_signal.connect(on_error)
        progress.cancel_requested.connect(worker.request_stop)
        worker.start()
        if owns_modal:
            progress.exec_()

    def _menu_run_inference(self):
        if self._project is None:
            QMessageBox.information(self, "No project", "Open a MAD project first.")
            return
        if not self.audio_files:
            QMessageBox.warning(self, "No files", "Project has no wav files.")
            return
        current = (self.audio_files[self.current_file_idx]
                   if self.audio_files else None)
        default_model = None
        if self._project.models:
            # Pick the most recent entry (models is a list of dicts).
            last = self._project.models[-1]
            if isinstance(last, dict):
                default_model = last.get('path')
            else:
                default_model = str(last)
        dlg = RunInferenceDialog(
            self, self._project.project_dir, self.audio_files, current,
            default_model,
        )
        if dlg.exec_() != QDialog.Accepted:
            return
        cfg = dlg.build_config()
        if not cfg.model_path or not os.path.isfile(cfg.model_path):
            QMessageBox.warning(
                self, "Model missing",
                "Select a valid weights.pt file before running inference."
            )
            return
        self._start_inference(cfg, dlg.selected_wavs())

    def _run_post_training_inference(self, model_path: str, scope: str):
        """Kick off inference automatically after a training run completes.

        ``scope`` ∈ {'current', 'all'}. Inference is applied only to time
        regions without painted labels — see ``run_inference_on_files``.
        """
        if not self.audio_files:
            return
        if scope == 'current':
            current = self.audio_files[self.current_file_idx]
            wavs = [current] if current else []
        else:
            wavs = list(self.audio_files)
        if not wavs:
            return
        from fnt.usv.usv_detector.mad_inference import MADInferenceConfig
        cfg = MADInferenceConfig(
            model_path=model_path,
            threshold=0.5,
            min_blob_pixels=8,
            device='auto',
            save_mask_png=True,
            save_blob_csv=True,
            preserve_labels=True,
            training_data_dir=(self._project.training_data_dir
                               if self._project else ""),
        )
        self._start_inference(cfg, wavs)

    def _start_inference(self, cfg, wav_paths: List[str], reporter=None):
        owns_modal = reporter is None
        progress = reporter or MADRunProgressDialog(self, "MAD Inference")
        progress.set_stage(f"Running inference on {len(wav_paths)} file(s)…")
        progress.append(f"Model: {cfg.model_path}")
        progress.append(f"Threshold: {cfg.threshold}  "
                        f"Min blob: {cfg.min_blob_pixels}px")

        worker = MADInferenceWorker(cfg, wav_paths, parent=self)

        def on_progress(file_i, file_n, wav_name, stage, si, sn):
            progress.set_stage(
                f"File {file_i + 1}/{file_n}: {wav_name} — {stage} ({si}/{sn})"
            )
            progress.set_main(file_i + (si / max(1, sn)), file_n)
            progress.set_sub(si, sn)

        def on_finished(results):
            total = sum(r.get('n_blobs', 0) for r in results if 'n_blobs' in r)
            errors = [r for r in results if 'error' in r]
            progress.append(
                f"\nFinished. {len(results)} file(s), "
                f"{total} blob(s) total, {len(errors)} error(s)."
            )
            for r in results[:20]:
                if 'error' in r:
                    progress.append(f"  ERR {r.get('wav_path')}: {r['error']}")
                else:
                    progress.append(
                        f"  {Path(r['wav_path']).name}: {r['n_blobs']} blob(s)"
                    )
            progress.mark_done(ok=(len(errors) == 0))
            self.status_bar.showMessage(
                f"Inference complete — {len(results)} file(s), {total} blob(s)"
            )
            # Auto-load predictions for the current file if we just ran on it.
            if self.audio_files:
                cur = self.audio_files[self.current_file_idx]
                if cur in wav_paths:
                    self._load_predictions_for_current()

        def on_error(msg: str):
            progress.append("\nERROR:\n" + msg)
            progress.mark_done(ok=False)

        worker.progress_signal.connect(on_progress)
        worker.finished_signal.connect(on_finished)
        worker.error_signal.connect(on_error)
        progress.cancel_requested.connect(worker.request_stop)
        worker.start()
        if owns_modal:
            progress.exec_()

    # ==================================================================
    # Blob review
    # ==================================================================
    def _load_predictions_for_current(self):
        if not self.audio_files:
            return
        wav = self.audio_files[self.current_file_idx]
        csv_path = pred_csv_sibling_path(wav)
        png_path = pred_mask_sibling_path(wav)

        blobs: List[dict] = []
        if os.path.isfile(csv_path):
            try:
                from fnt.usv.usv_detector.mad_inference import read_blob_csv
                rows = read_blob_csv(csv_path)
                # Convert seconds/Hz back to pixel indices for overlay + jump.
                cfg = self._project
                if cfg is None or self.sample_rate is None:
                    return
                dt = (cfg.nperseg - cfg.noverlap) / float(self.sample_rate)
                df = (self.sample_rate / 2.0) / (cfg.nfft // 2)
                for r in rows:
                    blobs.append({
                        'blob_id': r['blob_id'],
                        't_start': int(round(r['start_s'] / dt)),
                        't_end_exclusive': int(round(r['stop_s'] / dt)),
                        'f_low': int(round(r['min_freq_hz'] / df)),
                        'f_high_exclusive': int(round(r['max_freq_hz'] / df)),
                        'score': r['score'],
                        'area_pixels': r['area_pixels'],
                        'start_s': r['start_s'],
                        'stop_s': r['stop_s'],
                        'min_freq_hz': r['min_freq_hz'],
                        'max_freq_hz': r['max_freq_hz'],
                        'status': r.get('status', 'pending'),
                    })
            except Exception as e:
                QMessageBox.warning(
                    self, "Load predictions failed",
                    f"{Path(csv_path).name}:\n{e}"
                )
                return
        else:
            QMessageBox.information(
                self, "No predictions",
                f"No sibling predictions CSV next to:\n{Path(wav).name}\n\n"
                "Run inference first."
            )
            return

        # Optional: load the binary prediction PNG for overlay.
        pred_mask = None
        if os.path.isfile(png_path):
            try:
                from fnt.usv.usv_detector.mad_labels import load_mask_png
                arr = load_mask_png(png_path)
                pred_mask = (arr.astype(np.float32) / 255.0)
            except Exception:
                pred_mask = None
        self.spectrogram.set_predicted_mask(pred_mask)

        self._pred_blobs = blobs
        self._pred_csv_path = csv_path
        self._pred_wav_path = wav
        self._pred_idx = 0 if blobs else None
        self.spectrogram.set_predicted_blobs(blobs, self._pred_idx)
        self._update_blob_review_widgets()
        if blobs:
            self._jump_to_blob(self._pred_idx)
        self.status_bar.showMessage(
            f"Loaded {len(blobs)} predicted blob(s) for "
            f"{Path(wav).name}"
        )

    def _clear_predictions(self):
        self._pred_blobs = []
        self._pred_csv_path = None
        self._pred_wav_path = None
        self._pred_idx = None
        self.spectrogram.set_predicted_mask(None)
        self.spectrogram.set_predicted_blobs([], None)
        self._update_blob_review_widgets()

    def _update_blob_review_widgets(self):
        blobs = getattr(self, '_pred_blobs', []) or []
        n = len(blobs)
        idx = getattr(self, '_pred_idx', None)
        enabled = n > 0
        self.btn_prev_blob.setEnabled(enabled and idx is not None and idx > 0)
        self.btn_next_blob.setEnabled(enabled and idx is not None and idx < n - 1)
        for b in (self.btn_accept_blob, self.btn_reject_blob, self.btn_skip_blob):
            b.setEnabled(enabled and idx is not None)
        self.lbl_blob_idx.setText(
            f"Blob {(idx + 1) if idx is not None else 0}/{n}" if n else "Blob 0/0"
        )
        if enabled and idx is not None:
            b = blobs[idx]
            self.lbl_blob_info.setText(
                f"#{b['blob_id']}  "
                f"{b['start_s']:.3f}–{b['stop_s']:.3f}s  "
                f"{b['min_freq_hz']/1000:.1f}–{b['max_freq_hz']/1000:.1f} kHz\n"
                f"score={b['score']:.3f}  area={b['area_pixels']}px  "
                f"status={b.get('status', 'pending')}"
            )
        else:
            self.lbl_blob_info.setText(
                "No predicted blobs loaded — use Predict → Run Inference"
            )

    def _prev_blob(self):
        blobs = getattr(self, '_pred_blobs', []) or []
        if not blobs or self._pred_idx is None:
            return
        if self._pred_idx > 0:
            self._pred_idx -= 1
            self.spectrogram.set_blob_highlight(self._pred_idx)
            self._update_blob_review_widgets()
            self._jump_to_blob(self._pred_idx)

    def _next_blob(self):
        blobs = getattr(self, '_pred_blobs', []) or []
        if not blobs or self._pred_idx is None:
            return
        if self._pred_idx < len(blobs) - 1:
            self._pred_idx += 1
            self.spectrogram.set_blob_highlight(self._pred_idx)
            self._update_blob_review_widgets()
            self._jump_to_blob(self._pred_idx)

    def _jump_to_blob(self, idx: int):
        blobs = getattr(self, '_pred_blobs', []) or []
        if not blobs or idx is None or idx >= len(blobs):
            return
        if self.spectrogram.total_duration <= 0:
            return
        b = blobs[idx]
        # Center ~2 blob widths worth of view.
        dur = max(0.05, float(b['stop_s']) - float(b['start_s']))
        window = max(self.spin_view_window.value(), dur * 3.0)
        center = 0.5 * (float(b['start_s']) + float(b['stop_s']))
        start = max(0.0, center - window / 2)
        end = min(self.spectrogram.total_duration, start + window)
        self.spectrogram.view_start = start
        self.spectrogram.view_end = end
        self.spin_view_window.blockSignals(True)
        self.spin_view_window.setValue(end - start)
        self.spin_view_window.blockSignals(False)
        self._invalidate_spec_cache()
        self._sync_scrollbar_from_view()

    def _review_current_blob(self, status: str):
        blobs = getattr(self, '_pred_blobs', []) or []
        if not blobs or self._pred_idx is None:
            return
        b = blobs[self._pred_idx]
        b['status'] = status
        # If 'accepted', stamp blob bbox into the paint mask so it
        # becomes a positive label. Useful for iterating the model.
        if status == 'accepted' and self.spectrogram.mask is not None:
            mask = self.spectrogram.mask
            t0 = max(0, int(b['t_start']))
            t1 = min(mask.shape[1], int(b['t_end_exclusive']))
            f0 = max(0, int(b['f_low']))
            f1 = min(mask.shape[0], int(b['f_high_exclusive']))
            if t1 > t0 and f1 > f0:
                # Only paint where prediction actually fired — fall back
                # to bbox if no prob mask available.
                pm = self.spectrogram.pred_mask
                region_pm = pm[f0:f1, t0:t1] if pm is not None else None
                if region_pm is not None and region_pm.size > 0:
                    fill = region_pm > 0.0
                    self.spectrogram.mask[f0:f1, t0:t1][fill] = MASK_POSITIVE
                else:
                    self.spectrogram.mask[f0:f1, t0:t1] = MASK_POSITIVE
                self.spectrogram._mask_dirty = True
        # Persist review status back to sibling CSV.
        self._persist_review_statuses()
        # Auto-advance.
        if self._pred_idx < len(blobs) - 1:
            self._pred_idx += 1
            self.spectrogram.set_blob_highlight(self._pred_idx)
            self._jump_to_blob(self._pred_idx)
        self._update_blob_review_widgets()
        self.spectrogram.update()

    def _persist_review_statuses(self):
        csv_path = getattr(self, '_pred_csv_path', None)
        if not csv_path:
            return
        try:
            from fnt.usv.usv_detector.mad_inference import write_blob_csv
            rows = []
            for b in (self._pred_blobs or []):
                rows.append({
                    'blob_id': b['blob_id'],
                    'start_s': b['start_s'],
                    'stop_s': b['stop_s'],
                    'min_freq_hz': b['min_freq_hz'],
                    'max_freq_hz': b['max_freq_hz'],
                    'area_pixels': b['area_pixels'],
                    'score': b['score'],
                    'status': b.get('status', 'pending'),
                })
            write_blob_csv(csv_path, rows)
        except Exception:
            pass

    # ==================================================================
    # Model info label
    # ==================================================================
    def _update_model_info_label(self):
        if self._project is None or not self._project.models:
            self.lbl_model_info.setText("No trained model")
            return
        latest = self._project.models[-1]
        path = latest.get('path') if isinstance(latest, dict) else str(latest)
        n = len(self._project.models)
        self.lbl_model_info.setText(
            f"Latest model ({n} total): "
            f"{Path(path).parent.name}/{Path(path).name}"
        )

    # ==================================================================
    # Close
    # ==================================================================
    def closeEvent(self, event):
        self._auto_save_mask_if_dirty()
        self._stop_playback()
        super().closeEvent(event)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    win = MADMainWindow()
    win.show()
    if QApplication.instance() is app:
        sys.exit(app.exec_())


if __name__ == "__main__":
    main()
