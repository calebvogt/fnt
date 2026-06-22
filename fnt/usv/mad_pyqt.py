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
from typing import Dict, List, Optional

import numpy as np
from PyQt5.QtCore import (
    Qt, QEvent, QObject, QSettings, QThread, QTimer, QRectF, QPointF, pyqtSignal,
)
from PyQt5.QtGui import (
    QImage, QKeySequence, QPainter, QPen, QColor, QBrush, QPolygonF,
)
from PyQt5.QtWidgets import (
    QAction, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
    QGroupBox, QHBoxLayout, QHeaderView, QInputDialog, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QRadioButton, QScrollArea, QScrollBar, QShortcut, QSizePolicy, QSlider,
    QSpinBox, QSplitter, QStatusBar, QTabWidget, QTextEdit, QTreeWidget,
    QTreeWidgetItem, QVBoxLayout, QWidget,
)
from scipy import signal

from fnt.usv.audio_widgets import SpectrogramWidget, WaveformOverviewWidget
from fnt.usv.usv_detector.mad_labels import pred_csv_sibling_path
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


class _WheelEater(QObject):
    """Event filter that swallows mouse-wheel / trackpad-swipe events so a
    widget can't be changed by scrolling (only by clicking / typing)."""

    def eventFilter(self, obj, event):
        if event.type() == QEvent.Wheel:
            return True
        return False


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
    # Right-click on an accepted annotation → (annotation index, globalPos).
    request_context_menu = pyqtSignal(int, object)
    # A vertex-edit finished for the given annotation index.
    annotation_edit_finished = pyqtSignal(int)
    # Left-click selected an annotation (no labeling tool engaged).
    annotation_clicked = pyqtSignal(int)
    # A rubber-band drag selected a set of annotations (indices; [] = cleared).
    annotations_selected = pyqtSignal(list)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.paint_mode: Optional[str] = None  # 'brush' | 'eraser' | 'sam' | None
        self.brush_radius_px = 3              # in spec-pixel units

        # Accepted annotations for the current file. Each is a dict
        # {id, category, f0, f1, t0, t1, mask(bool local crop)} — the blue
        # masks. ``self.mask`` is their union (uint8 {0,1}) kept for fill
        # rendering, save-target and inference-preserve compatibility.
        self.annotations: List[dict] = []
        # Vertex-edit state (Phase F): index of the annotation being edited and
        # its polygon points in full-grid (t, f) coords.
        self._editing_ann_idx: Optional[int] = None
        self._edit_points: List[tuple] = []
        self._drag_vertex: Optional[int] = None
        self.mask: Optional[np.ndarray] = None
        # Pending (unconfirmed) mask accumulated by brush / eraser / SAM
        # (uint8 {0,1}). Rendered yellow until the user confirms with Enter.
        self._pending: Optional[np.ndarray] = None
        # Undo stack of pending additions — each entry is (f_idx[], t_idx[])
        # of pixels newly set by one SAM click or brush stroke.
        self._pending_stack: List[tuple] = []
        self._stroke_fset: Optional[set] = None  # collects a stroke's new px
        self._sam_click: Optional[tuple] = None  # last SAM click (t_idx, f_idx)
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
        # Predicted prob mask (float32 in [0,1]).
        self.pred_mask: Optional[np.ndarray] = None
        self._selected_ann_idx: Optional[int] = None
        # Rubber-band multi-select (drag a box over masks with no tool active).
        self._selected_set: set = set()
        self._rubber_start = None
        self._rubber_cur = None

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
        self.annotations = []
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

    def add_sam_point(self, pos, positive: bool = True) -> None:
        """Each left-click is an independent single-point SAM prompt — predict,
        keep one connected component, and add it to the pending buffer."""
        spec_rect = self._get_spec_rect()
        if not spec_rect.contains(pos):
            return
        idx = self._screen_to_spec_idx(pos.x(), pos.y(), spec_rect)
        if idx is None:
            return
        self._sam_pos_pts = [idx]
        self._sam_neg_pts = []
        self._sam_click = idx
        self.sam_points_changed.emit()

    def get_sam_prompts(self):
        """Return (positive_pts, negative_pts) in full-grid (t_idx, f_idx)."""
        return list(self._sam_pos_pts), list(self._sam_neg_pts)

    def has_sam_prompts(self) -> bool:
        return bool(self._sam_pos_pts or self._sam_neg_pts)

    def add_sam_component(self, mask: Optional[np.ndarray], t_off: int) -> bool:
        """Take the single connected component of a SAM proposal under the
        click (else the largest) and OR it into the pending buffer. Returns
        True if pixels were added."""
        self._sam_pos_pts = []
        self._sam_neg_pts = []
        if self._pending is None or mask is None:
            return False
        from scipy import ndimage
        m = np.asarray(mask) > 0
        if not m.any():
            return False
        lbl, n = ndimage.label(m)
        if n == 0:
            return False
        chosen = 0
        if self._sam_click is not None:
            ct = int(self._sam_click[0]) - int(t_off)
            cf = int(self._sam_click[1])
            if 0 <= cf < m.shape[0] and 0 <= ct < m.shape[1]:
                chosen = int(lbl[cf, ct])
        if chosen == 0:  # click not on a blob → largest component
            sizes = ndimage.sum(np.ones_like(lbl), lbl, range(1, n + 1))
            chosen = int(np.argmax(sizes)) + 1
        cc = lbl == chosen
        fcoords, tcoords = np.where(cc)
        tg = tcoords + int(t_off)
        fg = fcoords
        valid = ((tg >= 0) & (tg < self.n_time_frames) &
                 (fg >= 0) & (fg < self.n_freq_bins))
        tg, fg = tg[valid], fg[valid]
        if tg.size == 0:
            return False
        newly = self._pending[fg, tg] == 0
        self._pending[fg, tg] = 1
        nf, nt = fg[newly], tg[newly]
        if nf.size:
            self._pending_stack.append((nf, nt))
        self.update()
        return True

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
        self._pending_stack = []
        self.update()

    def undo_pending(self) -> bool:
        """Remove the most recent SAM click / brush stroke from pending."""
        if not self._pending_stack or self._pending is None:
            return False
        fs, ts = self._pending_stack.pop()
        self._pending[fs, ts] = 0
        self.update()
        return True

    def pending_components(self):
        """Connected components of the pending mask, as
        ``(f0, f1, t0, t1, local_bool)`` tuples."""
        if self._pending is None or not self._pending.any():
            return []
        from scipy import ndimage
        lbl, n = ndimage.label(self._pending > 0)
        out = []
        for k in range(1, n + 1):
            cc = lbl == k
            fs = np.where(cc.any(axis=1))[0]
            ts = np.where(cc.any(axis=0))[0]
            f0, f1 = int(fs[0]), int(fs[-1]) + 1
            t0, t1 = int(ts[0]), int(ts[-1]) + 1
            out.append((f0, f1, t0, t1,
                        np.ascontiguousarray(cc[f0:f1, t0:t1])))
        return out

    def pending_bbox(self):
        """Return (f0, f1, t0, t1) [half-open] of pending pixels, or None."""
        if self._pending is None or not self._pending.any():
            return None
        fs = np.where(self._pending.any(axis=1))[0]
        ts = np.where(self._pending.any(axis=0))[0]
        return int(fs[0]), int(fs[-1]) + 1, int(ts[0]), int(ts[-1]) + 1

    # --- accepted-annotation lifecycle --------------------------------
    def _rebuild_confirmed_mask(self) -> None:
        """Recompute the union buffer from the annotation list."""
        if self.mask is None:
            return
        self.mask[:] = 0
        for ann in self.annotations:
            f0, f1, t0, t1 = ann['f0'], ann['f1'], ann['t0'], ann['t1']
            sub = ann['mask']
            self.mask[f0:f1, t0:t1][sub] = MASK_POSITIVE

    def set_annotations(self, anns: List[dict]) -> None:
        self.annotations = list(anns)
        self._rebuild_confirmed_mask()
        self.update()

    def add_annotation(self, ann: dict) -> None:
        self.annotations.append(ann)
        if self.mask is not None:
            f0, f1, t0, t1 = ann['f0'], ann['f1'], ann['t0'], ann['t1']
            self.mask[f0:f1, t0:t1][ann['mask']] = MASK_POSITIVE
        self.update()

    def remove_annotation(self, idx: int) -> Optional[dict]:
        if not (0 <= idx < len(self.annotations)):
            return None
        ann = self.annotations.pop(idx)
        self._rebuild_confirmed_mask()
        self.update()
        return ann

    def annotation_at(self, t_idx: int, f_idx: int) -> Optional[int]:
        """Return the index of the topmost accepted annotation whose mask
        covers (t_idx, f_idx), or None."""
        for idx in range(len(self.annotations) - 1, -1, -1):
            ann = self.annotations[idx]
            if (ann['t0'] <= t_idx < ann['t1'] and
                    ann['f0'] <= f_idx < ann['f1']):
                lf = f_idx - ann['f0']
                lt = t_idx - ann['t0']
                if ann['mask'][lf, lt]:
                    return idx
        return None

    # --- vertex editing (Phase F) -------------------------------------
    def _grid_to_screen(self, t, f, spec_rect, bounds):
        t_start, t_end, f_start, f_end = bounds
        x = spec_rect.left() + (t - t_start) / max(1, t_end - t_start) * \
            spec_rect.width()
        y = spec_rect.bottom() - (f - f_start) / max(1, f_end - f_start) * \
            spec_rect.height()
        return x, y

    def start_edit(self, ann_idx: int) -> None:
        if not (0 <= ann_idx < len(self.annotations)) or not HAS_CV2:
            return
        ann = self.annotations[ann_idx]
        m = ann['mask'].astype(np.uint8)
        cnts, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not cnts:
            return
        cnt = max(cnts, key=cv2.contourArea)
        eps = max(1.0, 0.01 * cv2.arcLength(cnt, True))
        approx = cv2.approxPolyDP(cnt, eps, True)
        self._edit_points = [
            (ann['t0'] + int(p[0]), ann['f0'] + int(p[1]))
            for p in approx[:, 0, :]
        ]
        if len(self._edit_points) < 3:
            self._edit_points = []
            return
        self._editing_ann_idx = ann_idx
        self._drag_vertex = None
        self.set_paint_mode(None)
        self.update()

    def _find_edit_vertex(self, pos, spec_rect):
        bounds = self._visible_spec_bounds()
        if bounds is None:
            return None
        best, bestd = None, 12.0
        for i, (tt, ff) in enumerate(self._edit_points):
            sx, sy = self._grid_to_screen(tt, ff, spec_rect, bounds)
            d = ((sx - pos.x()) ** 2 + (sy - pos.y()) ** 2) ** 0.5
            if d < bestd:
                bestd, best = d, i
        return best

    def _insert_edit_vertex(self, pos, spec_rect):
        idx = self._screen_to_spec_idx(pos.x(), pos.y(), spec_rect)
        if idx is None or len(self._edit_points) < 2:
            return
        ct, cf = idx
        pts = self._edit_points
        best_i, best_d = 0, float('inf')
        for i in range(len(pts)):
            ax, ay = pts[i]
            bx, by = pts[(i + 1) % len(pts)]
            mx, my = (ax + bx) / 2.0, (ay + by) / 2.0
            d = (mx - ct) ** 2 + (my - cf) ** 2
            if d < best_d:
                best_d, best_i = d, i
        pts.insert(best_i + 1, (ct, cf))
        self._drag_vertex = best_i + 1
        self.update()

    def finish_edit(self):
        if self._editing_ann_idx is None:
            return
        ai = self._editing_ann_idx
        ann = self.annotations[ai]
        pts = self._edit_points
        ts = [p[0] for p in pts]
        fs = [p[1] for p in pts]
        t0 = max(0, min(ts))
        t1 = min(self.n_time_frames, max(ts) + 1)
        f0 = max(0, min(fs))
        f1 = min(self.n_freq_bins, max(fs) + 1)
        local = np.zeros((max(1, f1 - f0), max(1, t1 - t0)), np.uint8)
        if HAS_CV2 and len(pts) >= 3:
            poly = np.array([[t - t0, f - f0] for (t, f) in pts], dtype=np.int32)
            cv2.fillPoly(local, [poly], 1)
        ann['t0'], ann['t1'], ann['f0'], ann['f1'] = t0, t1, f0, f1
        ann['mask'] = local.astype(bool)
        self._editing_ann_idx = None
        self._edit_points = []
        self._drag_vertex = None
        self._rebuild_confirmed_mask()
        self.annotation_edit_finished.emit(ai)
        self.update()

    def cancel_edit(self):
        self._editing_ann_idx = None
        self._edit_points = []
        self._drag_vertex = None
        self.update()

    def is_editing(self) -> bool:
        return self._editing_ann_idx is not None

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
        region = self._pending[f0:f1, t0:t1]
        if value == 1 and self._stroke_fset is not None:
            newly = disk & (region == 0)
            if newly.any():
                lf, lt = np.where(newly)
                for a, b in zip(lf, lt):
                    self._stroke_fset.add((f0 + int(a), t0 + int(b)))
        region[disk] = value

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
        self._selected_ann_idx = None
        self._selected_set = set()
        self._rubber_start = None
        self._rubber_cur = None
        # Grab keyboard focus so arrow keys revert to preview pan/zoom (away
        # from the detections list).
        self.setFocus(Qt.MouseFocusReason)
        spec_rect = self._get_spec_rect()
        # --- vertex-edit mode (Phase F) ---
        if self._editing_ann_idx is not None:
            if event.button() == Qt.LeftButton:
                vi = self._find_edit_vertex(event.pos(), spec_rect)
                if vi is not None:
                    self._drag_vertex = vi
                    return
                # click away from any vertex → finish edit
                self.finish_edit()
                return
            if event.button() == Qt.RightButton:
                self._insert_edit_vertex(event.pos(), spec_rect)
                return
            return

        # --- right-click → context menu on an accepted annotation ---
        if event.button() == Qt.RightButton:
            idx = self._screen_to_spec_idx(
                event.pos().x(), event.pos().y(), spec_rect)
            if idx is not None:
                ai = self.annotation_at(idx[0], idx[1])
                if ai is not None:
                    self.request_context_menu.emit(ai, event.globalPos())
            return

        if self.paint_mode == 'sam':
            if event.button() == Qt.LeftButton:
                self.add_sam_point(event.pos(), positive=True)
                return

        if (self.paint_mode in ('brush', 'eraser') and
                event.button() == Qt.LeftButton):
            if not spec_rect.contains(event.pos()):
                return
            idx = self._screen_to_spec_idx(
                event.pos().x(), event.pos().y(), spec_rect
            )
            if idx is None:
                return
            self._painting = True
            self._stroke_fset = set() if self.paint_mode == 'brush' else None
            self._stamp(*idx)
            self._last_paint_idx = idx
            self._cursor_pos = event.pos()
            self.update()
            return

        # No tool engaged: left-click an existing mask to select it (white
        # outline); drag over empty space to rubber-band multiple masks.
        if event.button() == Qt.LeftButton and self.paint_mode is None:
            idx = self._screen_to_spec_idx(
                event.pos().x(), event.pos().y(), spec_rect)
            if idx is not None:
                ai = self.annotation_at(idx[0], idx[1])
                if ai is not None:
                    self._selected_ann_idx = ai
                    self.update()
                    self.annotation_clicked.emit(ai)
                    return
            # Missed any mask → begin a rubber-band box selection.
            if spec_rect.contains(event.pos()):
                self._rubber_start = event.pos()
                self._rubber_cur = event.pos()
                return
        super().mousePressEvent(event)

    def mouseMoveEvent(self, event):
        if self._editing_ann_idx is not None and self._drag_vertex is not None:
            idx = self._screen_to_spec_idx(
                event.pos().x(), event.pos().y(), self._get_spec_rect())
            if idx is not None:
                self._edit_points[self._drag_vertex] = (idx[0], idx[1])
                self.update()
            return
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
        if self._rubber_start is not None:
            self._rubber_cur = event.pos()
            self.update()
            return
        super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        if self._editing_ann_idx is not None and self._drag_vertex is not None:
            self._drag_vertex = None
            return
        if self._rubber_start is not None and event.button() == Qt.LeftButton:
            self._finish_rubber_band(event.pos())
            return
        if self._painting and event.button() == Qt.LeftButton:
            self._painting = False
            self._last_paint_idx = None
            # Push the stroke's newly-painted pixels onto the undo stack.
            if self._stroke_fset:
                fs = np.array([p[0] for p in self._stroke_fset], dtype=int)
                ts = np.array([p[1] for p in self._stroke_fset], dtype=int)
                self._pending_stack.append((fs, ts))
            self._stroke_fset = None
            self.stroke_committed.emit()
            return
        super().mouseReleaseEvent(event)

    def _finish_rubber_band(self, end_pos):
        """Select every annotation whose bbox intersects the drag rectangle."""
        start = self._rubber_start
        self._rubber_start = None
        self._rubber_cur = None
        spec_rect = self._get_spec_rect()
        a = self._screen_to_spec_idx(start.x(), start.y(), spec_rect)
        b = self._screen_to_spec_idx(end_pos.x(), end_pos.y(), spec_rect)
        # A tiny drag = a click on empty space → clear the selection.
        if (a is None or b is None or
                (abs(end_pos.x() - start.x()) < 4 and
                 abs(end_pos.y() - start.y()) < 4)):
            self._selected_set = set()
            self.annotations_selected.emit([])
            self.update()
            return
        t_lo, t_hi = sorted((a[0], b[0]))
        f_lo, f_hi = sorted((a[1], b[1]))
        hits = []
        for i, ann in enumerate(self.annotations):
            if (ann['t1'] > t_lo and ann['t0'] < t_hi and
                    ann['f1'] > f_lo and ann['f0'] < f_hi):
                hits.append(i)
        self._selected_set = set(hits)
        self.update()
        self.annotations_selected.emit(sorted(hits))

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
        # Outside paint mode the wheel does nothing — zoom/pan is keyboard-only
        # (↑/↓ zoom, ←/→ pan). Trackpad two-finger swipes were too easy to
        # over-zoom and stall the render, so they're consumed here.
        event.accept()

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
        # Build a per-pixel type map: 0=none, 1=confirmed, 2=prediction.
        type_map = np.zeros((h, w), dtype=np.uint8)
        view_mask = self.mask[f_start:f_end, t_start:t_end]
        type_map[view_mask == MASK_POSITIVE] = 1
        for ann in self.annotations:
            if ann.get('status') == 'prediction':
                af0 = max(ann['f0'], f_start); af1 = min(ann['f1'], f_end)
                at0 = max(ann['t0'], t_start); at1 = min(ann['t1'], t_end)
                if af1 > af0 and at1 > at0:
                    lf0, lt0 = af0 - f_start, at0 - t_start
                    mf0, mt0 = af0 - ann['f0'], at0 - ann['t0']
                    mh, mw = af1 - af0, at1 - at0
                    sub = ann['mask'][mf0:mf0+mh, mt0:mt0+mw]
                    type_map[lf0:lf0+mh, lt0:lt0+mw][sub > 0] = 2

        if self.view_mode == 'mask_only':
            rgba[:] = (32, 32, 32, 255)
            rgba[type_map == 1] = (40, 130, 255, 255)
            rgba[type_map == 2] = (255, 230, 90, 255)
            if pend_view is not None:
                rgba[pend_view] = (255, 230, 90, 255)
        elif self.view_mode == 'overlay':
            pos_alpha = int(self.mask_alpha * 255)
            rgba[type_map == 1] = (40, 130, 255, pos_alpha)
            rgba[type_map == 2] = (255, 230, 90, max(60, pos_alpha - 50))
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

        # (SAM prompt clicks are not drawn — only the proposed mask is shown.)

        # Accepted-annotation outlines + class labels (blue) above each mask.
        if self.view_mode != 'spec' and self.annotations and HAS_CV2:
            for ai, ann in enumerate(self.annotations):
                if ai == self._editing_ann_idx:
                    continue
                if (ann['t1'] <= t_start or ann['t0'] >= t_end or
                        ann['f1'] <= f_start or ann['f0'] >= f_end):
                    continue
                is_pred = ann.get('status') == 'prediction'
                is_sel = (ai == self._selected_ann_idx)
                if is_sel:
                    # Selected detection: mask outline AND label both white so
                    # tiny detections are easy to spot.
                    outline_color = QColor(255, 255, 255)
                    label_color = QColor(255, 255, 255)
                else:
                    outline_color = QColor(255, 225, 60) if is_pred else QColor(60, 150, 255)
                    label_color = QColor(255, 230, 100) if is_pred else QColor(120, 200, 255)
                # Thin outline around the mask contour.
                lf0 = max(ann['f0'], f_start) - f_start
                lf1 = min(ann['f1'], f_end) - f_start
                lt0 = max(ann['t0'], t_start) - t_start
                lt1 = min(ann['t1'], t_end) - t_start
                if lf1 > lf0 and lt1 > lt0:
                    view_ann = np.zeros((h, w), dtype=np.uint8)
                    msk = ann['mask']
                    af0 = max(ann['f0'], f_start) - ann['f0']
                    af1 = af0 + (lf1 - lf0)
                    at0 = max(ann['t0'], t_start) - ann['t0']
                    at1 = at0 + (lt1 - lt0)
                    if af1 <= msk.shape[0] and at1 <= msk.shape[1]:
                        view_ann[lf0:lf1, lt0:lt1] = msk[af0:af1, at0:at1].astype(np.uint8)
                    cnts, _ = cv2.findContours(
                        view_ann, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    pen = QPen(outline_color)
                    pen.setWidth(2)
                    painter.setPen(pen)
                    painter.setBrush(Qt.NoBrush)
                    for cnt in cnts:
                        if len(cnt) < 2:
                            continue
                        poly = QPolygonF()
                        for pt in cnt[:, 0, :]:
                            poly.append(QPointF(
                                _t_to_x(t_start + int(pt[0])),
                                _f_to_y(f_start + int(pt[1])),
                            ))
                        painter.drawPolygon(poly)
                # Class label centered horizontally over the mask, just above it.
                painter.setPen(QPen(label_color))
                f = painter.font()
                f.setPointSize(8)
                painter.setFont(f)
                label = ann.get('category', '') or '?'
                if is_pred:
                    label = f"({label})"
                cx = _t_to_x((max(ann['t0'], t_start) + min(ann['t1'], t_end)) / 2.0)
                tw = painter.fontMetrics().horizontalAdvance(label)
                lx = cx - tw / 2.0
                ly = _f_to_y(min(ann['f1'], f_end)) - 3
                painter.drawText(QPointF(lx, ly), label)

            # White highlight outline on the selected annotation(s) — the
            # single click-selected one plus any rubber-band multi-selection.
            highlight = set(self._selected_set)
            if self._selected_ann_idx is not None:
                highlight.add(self._selected_ann_idx)
            for sel in highlight:
                if not (0 <= sel < len(self.annotations)):
                    continue
                sa = self.annotations[sel]
                if (sa['t1'] <= t_start or sa['t0'] >= t_end or
                        sa['f1'] <= f_start or sa['f0'] >= f_end):
                    continue
                lf0 = max(sa['f0'], f_start) - f_start
                lf1 = min(sa['f1'], f_end) - f_start
                lt0 = max(sa['t0'], t_start) - t_start
                lt1 = min(sa['t1'], t_end) - t_start
                if lf1 > lf0 and lt1 > lt0:
                    view_sel = np.zeros((h, w), dtype=np.uint8)
                    smsk = sa['mask']
                    sf0 = max(sa['f0'], f_start) - sa['f0']
                    sf1 = sf0 + (lf1 - lf0)
                    st0 = max(sa['t0'], t_start) - sa['t0']
                    st1 = st0 + (lt1 - lt0)
                    if sf1 <= smsk.shape[0] and st1 <= smsk.shape[1]:
                        view_sel[lf0:lf1, lt0:lt1] = smsk[sf0:sf1, st0:st1].astype(np.uint8)
                    scnts, _ = cv2.findContours(
                        view_sel, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    pen = QPen(QColor(255, 255, 255))
                    pen.setWidth(3)
                    painter.setPen(pen)
                    painter.setBrush(Qt.NoBrush)
                    for cnt in scnts:
                        if len(cnt) < 2:
                            continue
                        poly = QPolygonF()
                        for pt in cnt[:, 0, :]:
                            poly.append(QPointF(
                                _t_to_x(t_start + int(pt[0])),
                                _f_to_y(f_start + int(pt[1])),
                            ))
                        painter.drawPolygon(poly)

        # Blue draggable outline + vertices for the annotation being edited.
        if (self._editing_ann_idx is not None and
                0 <= self._editing_ann_idx < len(self.annotations) and
                self._edit_points):
            pen = QPen(QColor(60, 150, 255))
            pen.setWidth(2)
            painter.setPen(pen)
            painter.setBrush(Qt.NoBrush)
            poly = QPolygonF()
            for (tt, ff) in self._edit_points:
                poly.append(QPointF(_t_to_x(tt), _f_to_y(ff)))
            painter.drawPolygon(poly)
            painter.setBrush(QBrush(QColor(60, 150, 255)))
            for (tt, ff) in self._edit_points:
                cx, cy = _t_to_x(tt), _f_to_y(ff)
                painter.drawEllipse(QRectF(cx - 4, cy - 4, 8, 8))

        # Rubber-band selection rectangle (dotted) while dragging.
        if self._rubber_start is not None and self._rubber_cur is not None:
            pen = QPen(QColor(255, 255, 255))
            pen.setStyle(Qt.DashLine)
            pen.setWidth(1)
            painter.setPen(pen)
            painter.setBrush(QBrush(QColor(255, 255, 255, 30)))
            painter.drawRect(QRectF(self._rubber_start, self._rubber_cur))

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
        self.spin_epochs.setSingleStep(10)
        self.spin_epochs.setToolTip(
            "Upper bound on epochs. Training may stop early (SLEAP-style) "
            "once the validation loss plateaus — see patience below."
        )
        form.addRow("Max epochs:", self.spin_epochs)

        self.spin_patience = QSpinBox()
        self.spin_patience.setRange(0, 200)
        self.spin_patience.setValue(8)
        self.spin_patience.setSingleStep(10)
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
            "so existing detections are preserved."
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

        self.chk_save_csv = QCheckBox("Save blob CSV")
        self.chk_save_csv.setChecked(True)
        vbox.addWidget(self.chk_save_csv)

        self.chk_preserve = QCheckBox(
            "Preserve user-painted labels (skip time regions already labeled)"
        )
        self.chk_preserve.setChecked(True)
        self.chk_preserve.setToolTip(
            "When enabled, the model's probability mask is zeroed in any\n"
            "time column that already contains a manually-painted label.\n"
            "Manual detections are never overwritten by inference."
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

        if show_plot:
            self._init_loss_plot(vbox)

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
        # Dark theme matching Mask Tracker's training graph.
        self._figure = Figure(figsize=(4.0, 2.4), tight_layout=True)
        self._figure.patch.set_facecolor("#1e1e1e")
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setMinimumHeight(180)
        parent_layout.addWidget(self._canvas)
        self._ax = self._figure.add_subplot(1, 1, 1)
        self._style_axes()
        (self._line_batch,) = self._ax.plot(
            [], [], color='#5a8dee', linewidth=0.9, alpha=0.55, label='batch')
        (self._line_train,) = self._ax.plot(
            [], [], color='#2979ff', marker='o', markersize=3,
            linewidth=1.8, label='train')
        (self._line_val,) = self._ax.plot(
            [], [], color='#ff9800', marker='s', markersize=3,
            linewidth=1.8, label='val')
        self._style_legend()
        self._plot = True
        self._canvas.draw_idle()

    def _style_axes(self):
        self._ax.set_facecolor("#1e1e1e")
        self._ax.set_title("Segmentation Training", color="#cccccc", fontsize=10)
        self._ax.set_xlabel("Batch", color="#cccccc", fontsize=9)
        self._ax.set_ylabel("Loss (log)", color="#cccccc", fontsize=9)
        self._ax.set_yscale("log")
        self._ax.tick_params(colors="#999999", labelsize=7)
        for spine in self._ax.spines.values():
            spine.set_color("#555555")
        self._ax.grid(alpha=0.18, color="#888888")

    def _style_legend(self):
        self._ax.legend(loc='upper right', fontsize=7, facecolor="#2b2b2b",
                        edgecolor="#555555", labelcolor="#cccccc", framealpha=0.9)

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

        # Prediction review state — index of the current prediction annotation
        # within spectrogram.annotations (only entries with status='prediction').
        self._pred_review_idx: Optional[int] = None

        # Deployment (Inference tab) file queue — new files to run a trained
        # model on, decoupled from the training file list.
        self.deploy_files: List[str] = []
        self._deploy_file_idx: Optional[int] = None
        # Which context the unified detection review acts in:
        #   'train'  → accept saves a training example
        #   'deploy' → accept/reject writes status to the output CSV
        self._review_mode: str = 'train'

        # Rubber-band multi-selection: stable ids of box-selected detections.
        self._box_sel_ids: List = []

        # Session-level class tracking (used when no project is open).
        self._session_classes: List[str] = ["USV"]
        self._session_last_class: str = "USV"

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
        # Arrow keys pan/zoom the spectrogram even when another widget holds
        # focus (see eventFilter).
        QApplication.instance().installEventFilter(self)
        self._update_project_state()
        self._update_paint_buttons_enabled()
        self._update_playback_buttons_enabled()

        # No startup dialog — user loads wavs freely, project created on demand.

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
            self._create_training_data_section,
            self._create_paint_tools_section,
            self._create_annotation_list_section,
            self._create_training_section,
            self._create_session_log_section,
        ])
        inference_tab = _make_tab([
            self._create_deploy_model_section,
            self._create_deploy_queue_section,
            self._create_inference_section,
            self._create_deploy_review_section,
        ])
        self.left_tabs.addTab(mask_tab, "Label && Train")
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
        self.spectrogram.request_context_menu.connect(
            self._on_annotation_context_menu
        )
        self.spectrogram.annotation_edit_finished.connect(
            self._on_annotation_edit_finished
        )
        self.spectrogram.annotation_clicked.connect(
            self._on_annotation_clicked
        )
        self.spectrogram.annotations_selected.connect(
            self._on_box_selection
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
        self.spin_view_window.setRange(0.1, 300.0)
        self.spin_view_window.setValue(2.0)
        self.spin_view_window.setSuffix(" s")
        self.spin_view_window.setFixedWidth(80)
        self.spin_view_window.setToolTip(
            "Time window duration (seconds). Use ↑/↓ to zoom; scroll-wheel "
            "adjustment is disabled to avoid accidental over-zoom.")
        self.spin_view_window.valueChanged.connect(self._on_view_window_changed)
        # Block trackpad-swipe / wheel from changing the time window or the
        # display frequency range (kept as a single shared filter).
        self._wheel_eater = _WheelEater(self)
        self.spin_view_window.installEventFilter(self._wheel_eater)
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
        self.spin_display_min_freq.setToolTip(
            "Lowest frequency shown on the spectrogram (display only — does not "
            "change the audio, labels, or model).")
        self.spin_display_min_freq.valueChanged.connect(self._on_display_freq_changed)
        self.spin_display_min_freq.installEventFilter(self._wheel_eater)
        controls_layout.addWidget(self.spin_display_min_freq)

        controls_layout.addWidget(QLabel("-"))
        self.spin_display_max_freq = QSpinBox()
        self.spin_display_max_freq.setRange(1000, 250000)
        self.spin_display_max_freq.setSingleStep(5000)
        self.spin_display_max_freq.setValue(125000)
        self.spin_display_max_freq.setSuffix(" Hz")
        self.spin_display_max_freq.setFixedWidth(90)
        self.spin_display_max_freq.setToolTip(
            "Highest frequency shown on the spectrogram (display only). Set near "
            "the Nyquist limit (½ the sample rate) to see the full band.")
        self.spin_display_max_freq.valueChanged.connect(self._on_display_freq_changed)
        self.spin_display_max_freq.installEventFilter(self._wheel_eater)
        controls_layout.addWidget(self.spin_display_max_freq)

        sep2 = QFrame(); sep2.setFrameShape(QFrame.VLine)
        sep2.setStyleSheet("color: #3f3f3f;")
        controls_layout.addWidget(sep2)

        controls_layout.addWidget(QLabel("Color Map:"))
        self.combo_colormap = QComboBox()
        self.combo_colormap.addItems(
            ['grayscale_inv', 'grayscale', 'viridis', 'magma', 'inferno']
        )
        self.combo_colormap.setFixedWidth(110)
        self.combo_colormap.setToolTip(
            "Spectrogram color palette (display only). 'viridis/magma/inferno' "
            "are perceptually uniform; 'grayscale_inv' shows loud = dark.")
        # Default to inverted grayscale (soft = white, loud = dark).
        self.combo_colormap.setCurrentText('viridis')
        self.spectrogram.set_colormap('viridis')
        self.combo_colormap.currentTextChanged.connect(self._on_colormap_changed)
        controls_layout.addWidget(self.combo_colormap)

        sep3 = QFrame(); sep3.setFrameShape(QFrame.VLine)
        sep3.setStyleSheet("color: #3f3f3f;")
        controls_layout.addWidget(sep3)

        # Playback — Play toggles to Stop while audio is playing.
        self.btn_play = QPushButton("Play")
        self.btn_play.setToolTip("Play / stop the visible window (Space)")
        self.btn_play.clicked.connect(self._toggle_playback)
        controls_layout.addWidget(self.btn_play)

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

        # Prevent left-panel controls from stealing keyboard focus so arrow
        # keys always route to spectrogram pan/zoom via eventFilter. The
        # detections list is exempt: clicking it gives it focus so Up/Down
        # navigate detections (see eventFilter); clicking the spectrogram
        # takes focus back so arrows control the preview again.
        from PyQt5.QtWidgets import QAbstractButton, QAbstractSlider
        for child in self.left_tabs.findChildren(QWidget):
            if isinstance(child, (QAbstractButton, QListWidget, QTreeWidget)):
                child.setFocusPolicy(Qt.NoFocus)
        # Lists stay NoFocus (from the blanket above): arrow keys are fully
        # dedicated to the preview's pan/zoom, never list navigation.
        self.spectrogram.setFocusPolicy(Qt.ClickFocus)

    # ------------------------------------------------------------------
    # Left-panel section builders
    # ------------------------------------------------------------------
    def _create_training_data_section(self, layout):
        group = QGroupBox("1. Add Audio Recordings")
        vbox = QVBoxLayout()
        vbox.setSpacing(4)

        # Compact project status (New/Open live in the File menu).
        self.lbl_project_name = QLabel("No project — add audio files to begin")
        self.lbl_project_name.setStyleSheet("color: #999999; font-size: 10px;")
        self.lbl_project_name.setWordWrap(True)
        vbox.addWidget(self.lbl_project_name)

        self.lbl_source_folders = QLabel("No source folders")
        self.lbl_source_folders.setStyleSheet("color: #777777; font-size: 9px;")
        self.lbl_source_folders.setWordWrap(True)
        vbox.addWidget(self.lbl_source_folders)

        self.lbl_model_info = QLabel("No trained model")
        self.lbl_model_info.setStyleSheet("color: #999999; font-size: 9px;")
        self.lbl_model_info.setWordWrap(True)
        vbox.addWidget(self.lbl_model_info)

        # Buttons created (unused but referenced by older code paths/tests).
        self.btn_new_project = QPushButton("New Project…")
        self.btn_new_project.clicked.connect(self._menu_new_project)
        self.btn_new_project.setVisible(False)
        self.btn_open_project = QPushButton("Open Project…")
        self.btn_open_project.clicked.connect(self._menu_open_project)
        self.btn_open_project.setVisible(False)

        add_row = QHBoxLayout()
        add_row.setSpacing(4)
        self.btn_add_folder = QPushButton("Add Folder…")
        self.btn_add_folder.setToolTip(
            "Add .wav files directly inside a folder (non-recursive). Works "
            "with or without a project; folders persist into the project."
        )
        self.btn_add_folder.clicked.connect(self._menu_add_folder)
        add_row.addWidget(self.btn_add_folder)

        self.btn_add_files = QPushButton("Add Files…")
        self.btn_add_files.setToolTip(
            "Add individual .wav files. Works without a project; persists into "
            "the project when one is open."
        )
        self.btn_add_files.clicked.connect(self._add_audio_files)
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

    # Solid fill for tool buttons: gray when off, blue when active. Styling
    # both states avoids the translucent native "checked" overlay on macOS.
    _ACTIVE_TOOL_QSS = (
        "QPushButton { background-color: #4a4a4a; color: #e0e0e0; border: none; "
        "border-radius: 5px; padding: 5px 10px; }"
        "QPushButton:checked { background-color: #2d6cdf; color: white; "
        "font-weight: bold; }"
        "QPushButton:disabled { background-color: #3a3a3a; color: #777; }"
    )

    def _create_paint_tools_section(self, layout):
        group = QGroupBox("2. Labeling Tools")
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
        self.btn_sam.setStyleSheet(self._ACTIVE_TOOL_QSS)
        self.btn_sam.clicked.connect(self._on_sam_clicked)
        self.btn_sam.setEnabled(False)
        sam_row.addWidget(self.btn_sam)
        sam_row.addStretch()
        vbox.addLayout(sam_row)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(2)

        self.btn_paint = QPushButton("Brush (B)")
        self.btn_paint.setToolTip(
            "Manually paint target USV pixels with the brush "
            "(left-click + drag).\n"
            "Scroll the wheel over the spectrogram to fine-tune brush size."
        )
        self.btn_paint.setCheckable(True)
        self.btn_paint.setStyleSheet(self._ACTIVE_TOOL_QSS)
        self.btn_paint.clicked.connect(self._on_brush_clicked)
        self.btn_paint.setEnabled(False)
        mode_row.addWidget(self.btn_paint)

        self.btn_erase = QPushButton("Eraser (E)")
        self.btn_erase.setToolTip(
            "Erase painted pixels (scroll to resize)\n"
            "Shortcut: E"
        )
        self.btn_erase.setCheckable(True)
        self.btn_erase.setStyleSheet(self._ACTIVE_TOOL_QSS)
        self.btn_erase.clicked.connect(self._on_eraser_clicked)
        self.btn_erase.setEnabled(False)
        mode_row.addWidget(self.btn_erase)

        self.btn_clear_mask = QPushButton("Clear")
        self.btn_clear_mask.setToolTip("Discard the current pending (yellow) masks")
        self.btn_clear_mask.clicked.connect(self._on_clear_clicked)
        self.btn_clear_mask.setEnabled(False)
        mode_row.addWidget(self.btn_clear_mask)

        self.btn_undo = QPushButton("Undo (U)")
        self.btn_undo.setToolTip(
            "Undo the last dropped mask — a pending blob, or the last "
            "confirmed detection.\nShortcut: U"
        )
        self.btn_undo.clicked.connect(self._undo_last)
        self.btn_undo.setEnabled(False)
        mode_row.addWidget(self.btn_undo)

        vbox.addLayout(mode_row)

        brush_row = QHBoxLayout()
        brush_row.setSpacing(2)
        brush_row.addWidget(QLabel("Brush radius:"))
        self.spin_brush_radius = QSpinBox()
        self.spin_brush_radius.setRange(1, 64)
        self.spin_brush_radius.setValue(3)
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

        # --- View mode: one button cycles Spec + Mask → Spec → Mask (V key) ---
        view_row = QHBoxLayout()
        view_row.setSpacing(2)
        view_row.addWidget(QLabel("View:"))
        self.btn_view_cycle = QPushButton("Spec + Mask")
        self.btn_view_cycle.setToolTip(
            "Cycle the preview: Spec + Mask → Spec only → Mask only.\n"
            "Shortcut: V")
        self.btn_view_cycle.clicked.connect(self._shortcut_cycle_view)
        view_row.addWidget(self.btn_view_cycle)
        view_row.addStretch()
        vbox.addLayout(view_row)

        hint = QLabel(
            "Yellow = pending / prediction (Enter to confirm, pick a class) · "
            "Blue = confirmed call. Confirmed calls save as self-contained "
            "training examples."
        )
        hint.setStyleSheet("color: #888888; font-size: 9px; font-style: italic;")
        hint.setWordWrap(True)
        vbox.addWidget(hint)

        group.setLayout(vbox)
        layout.addWidget(group)

    def _set_view_mode(self, mode: str):
        labels = {'overlay': 'Spec + Mask', 'spec': 'Spec only',
                  'mask_only': 'Mask only'}
        if hasattr(self, 'btn_view_cycle'):
            self.btn_view_cycle.setText(labels.get(mode, 'Spec + Mask'))
        self.spectrogram.set_view_mode(mode)

    def _create_training_section(self, layout):
        group = QGroupBox("3. Train Model")
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
            "<b>Segmentation architecture</b> (binary U-Net family, via "
            "segmentation_models_pytorch).<br>"
            "• <b>U-Net</b> — classic encoder/decoder; fast, robust baseline.<br>"
            "• <b>U-Net++</b> — nested dense skip connections; recovers finer "
            "detail at a little more compute.<br>"
            "• <b>HRNet U-Net</b> — keeps high-resolution features throughout "
            "(its own HRNet encoder, so the Encoder picker is ignored); best for "
            "crisp, thin USV contours, slowest.<br>"
            "• <b>MA-Net</b> — attention-gated decoder; emphasizes salient "
            "regions, can help on cluttered spectrograms."
        )
        self.combo_arch.currentIndexChanged.connect(self._on_arch_changed)
        form.addRow("Architecture:", self.combo_arch)

        self.combo_train_encoder = QComboBox()
        self.combo_train_encoder.addItems([
            "resnet18", "resnet34", "resnet50",
            "efficientnet-b0", "mobilenet_v2",
        ])
        self.combo_train_encoder.setToolTip(
            "<b>Encoder backbone</b> that extracts features (ImageNet-pretrained)."
            "<br>Bigger = more capacity but slower and more data-hungry:<br>"
            "• <b>resnet18</b> — lightest, good default for small label sets.<br>"
            "• <b>resnet34 / resnet50</b> — deeper, more capacity.<br>"
            "• <b>efficientnet-b0</b> — accuracy-per-FLOP efficient.<br>"
            "• <b>mobilenet_v2</b> — fastest/lightest, for CPU or quick iterations."
            "<br><i>Ignored when Architecture is HRNet.</i>"
        )
        form.addRow("Encoder:", self.combo_train_encoder)

        self.spin_train_epochs = QSpinBox()
        self.spin_train_epochs.setRange(1, 500)
        self.spin_train_epochs.setValue(100)
        self.spin_train_epochs.setSingleStep(10)
        self.spin_train_epochs.setToolTip(
            "<b>Maximum training epochs</b> (full passes over the labeled "
            "examples). Training may stop earlier via early stopping. Higher = "
            "more chances to converge, but longer runs and more overfitting risk."
        )
        form.addRow("Max epochs:", self.spin_train_epochs)

        self.spin_train_patience = QSpinBox()
        self.spin_train_patience.setRange(0, 200)
        self.spin_train_patience.setValue(8)
        self.spin_train_patience.setSingleStep(10)
        self.spin_train_patience.setToolTip(
            "<b>Early-stop patience</b>: stop if validation loss doesn't improve "
            "for this many consecutive epochs (the best checkpoint is kept). "
            "Guards against overfitting and saves time. <b>0 disables</b> early "
            "stopping (always train the full Max epochs)."
        )
        form.addRow("Early-stop patience:", self.spin_train_patience)

        self.spin_train_batch = QSpinBox()
        self.spin_train_batch.setRange(1, 64)
        self.spin_train_batch.setValue(8)
        self.spin_train_batch.setToolTip(
            "<b>Batch size</b>: number of spectrogram tiles per gradient step. "
            "Larger = smoother gradients and faster epochs but more GPU/RAM; "
            "lower it if you hit out-of-memory errors."
        )
        form.addRow("Batch size:", self.spin_train_batch)

        self.spin_train_lr = QDoubleSpinBox()
        self.spin_train_lr.setDecimals(6)
        self.spin_train_lr.setRange(1e-6, 1.0)
        self.spin_train_lr.setSingleStep(1e-4)
        self.spin_train_lr.setValue(1e-3)
        self.spin_train_lr.setToolTip(
            "<b>Learning rate</b> for the Adam optimizer — the step size for "
            "weight updates. Too high → unstable/diverging loss; too low → very "
            "slow convergence. 1e-3 is a sensible default."
        )
        form.addRow("Learning rate:", self.spin_train_lr)

        self.spin_train_val = QDoubleSpinBox()
        self.spin_train_val.setRange(0.0, 0.9)
        self.spin_train_val.setSingleStep(0.05)
        self.spin_train_val.setValue(0.20)
        self.spin_train_val.setToolTip(
            "<b>Validation fraction</b>: share of labeled examples held out "
            "(not trained on) to measure generalization and drive early "
            "stopping. 0.20 = 20% held out. Set higher for a more reliable "
            "validation signal, lower to train on more data."
        )
        form.addRow("Validation fraction:", self.spin_train_val)

        self.combo_train_device = QComboBox()
        self.combo_train_device.addItems(["auto", "cuda", "mps", "cpu"])
        self.combo_train_device.setToolTip(
            "<b>Compute device</b>: <b>auto</b> picks the best available "
            "(CUDA &gt; MPS &gt; CPU). <b>cuda</b> = NVIDIA GPU, <b>mps</b> = "
            "Apple-Silicon GPU, <b>cpu</b> = portable but much slower."
        )
        form.addRow("Device:", self.combo_train_device)

        vbox.addLayout(form)

        post_row = QHBoxLayout()
        self.chk_post_inference = QCheckBox("Run inference after, on:")
        self.chk_post_inference.setChecked(True)
        self.chk_post_inference.setToolTip(
            "When checked, automatically run the freshly trained model right "
            "after training and load its predictions for review — the "
            "label → train → predict → correct active-learning loop."
        )
        self.combo_post_scope = QComboBox()
        self.combo_post_scope.addItems(["current file", "all files"])
        self.combo_post_scope.setToolTip(
            "Scope of the post-training inference: just the <b>current file</b> "
            "(fast) or <b>all files</b> in the project."
        )
        self.chk_post_inference.toggled.connect(self.combo_post_scope.setEnabled)
        post_row.addWidget(self.chk_post_inference)
        post_row.addWidget(self.combo_post_scope, 1)
        vbox.addLayout(post_row)

        self.btn_train = QPushButton("Train")
        self.btn_train.setToolTip(
            "Train a segmentation model on every <b>confirmed</b> call (the "
            "blue detections / saved examples). A live loss graph replaces the "
            "spectrogram while training runs; the model is saved into the "
            "project's models/ folder. Needs an open project."
        )
        self.btn_train.clicked.connect(self._on_inline_train)
        self.btn_train.setEnabled(False)
        vbox.addWidget(self.btn_train)

        # Inline post-training inference progress (replaces the old popup) —
        # shown below the Train button only while auto-inference runs.
        self.train_infer_panel = MADRunPanel(show_plot=False)
        self.train_infer_panel.setVisible(False)
        self.train_infer_panel.run_finished.connect(
            lambda ok: self.train_infer_panel.setVisible(False))
        vbox.addWidget(self.train_infer_panel)

        # The live training graph (self.train_panel) lives in the right-hand
        # preview area (built in _build_ui) and is only shown while training
        # runs — mirroring Mask Tracker. Here we just wire its signals.
        self.train_panel.run_finished.connect(self._on_training_finished)
        self.train_panel.cancel_requested.connect(
            lambda: self.status_bar.showMessage("Stopping training…")
        )

        group.setLayout(vbox)
        layout.addWidget(group)

    def _create_session_log_section(self, layout):
        group = QGroupBox("Session Logs")
        vbox = QVBoxLayout()
        vbox.setSpacing(4)
        self.session_log = QTextEdit()
        self.session_log.setReadOnly(True)
        self.session_log.setMaximumHeight(140)
        self.session_log.setStyleSheet(
            "font-family: monospace; font-size: 10px;")
        self.session_log.setToolTip(
            "Pithy log of your actions and training/inference output. "
            "Copy it to share when reporting an issue."
        )
        vbox.addWidget(self.session_log)
        self.btn_copy_log = QPushButton("Copy Output to Clipboard")
        self.btn_copy_log.setToolTip("Copy the full session log to the clipboard")
        self.btn_copy_log.setFocusPolicy(Qt.NoFocus)
        self.btn_copy_log.clicked.connect(self._copy_session_log)
        vbox.addWidget(self.btn_copy_log)
        group.setLayout(vbox)
        layout.addWidget(group)
        self._log("MAD session started")

    def _log(self, msg: str):
        """Append a timestamped, pithy line to the Session Logs panel."""
        if not hasattr(self, 'session_log'):
            return
        from datetime import datetime as _dt
        self.session_log.append(f"[{_dt.now().strftime('%H:%M:%S')}] {msg}")

    def _copy_session_log(self):
        QApplication.clipboard().setText(self.session_log.toPlainText())
        self.status_bar.showMessage("Session log copied to clipboard")

    def _create_annotation_list_section(self, layout):
        group = QGroupBox("Detections")
        vbox = QVBoxLayout()
        vbox.setSpacing(4)

        filter_row = QHBoxLayout()
        filter_row.setSpacing(4)
        filter_row.addWidget(QLabel("Show:"))
        self.combo_det_filter = QComboBox()
        self.combo_det_filter.addItems(["All", "Pending", "Confirmed"])
        self.combo_det_filter.setToolTip(
            "Filter detections: All, Pending (yellow predictions), or "
            "Confirmed (blue, saved as training examples)."
        )
        self.combo_det_filter.currentIndexChanged.connect(
            lambda _i: self._refresh_annotation_list()
        )
        filter_row.addWidget(self.combo_det_filter, 1)
        vbox.addLayout(filter_row)

        self.annotation_list = QTreeWidget()
        self.annotation_list.setMaximumHeight(200)
        self.annotation_list.setToolTip(
            "Detections for the current file. Blue = confirmed, "
            "Yellow = prediction (pending review). Click to jump."
        )
        cols = ["", "Time", "Class", "Dur", "kHz", "Px", "Score"]
        self.annotation_list.setHeaderLabels(cols)
        self.annotation_list.setRootIsDecorated(False)
        self.annotation_list.setAllColumnsShowFocus(True)
        self.annotation_list.setSortingEnabled(True)
        hdr = self.annotation_list.header()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        for c in range(1, len(cols)):
            hdr.setSectionResizeMode(c, QHeaderView.Interactive)
        hdr.setStretchLastSection(True)
        hdr.setDefaultSectionSize(50)
        self.annotation_list.setStyleSheet(
            "QTreeWidget::item:selected { background-color: #4a4a55; }"
        )
        self.annotation_list.currentItemChanged.connect(
            self._on_annotation_list_selected
        )
        vbox.addWidget(self.annotation_list)

        count_row = QHBoxLayout()
        count_row.setSpacing(4)
        self.lbl_annotation_count = QLabel("0 detections")
        self.lbl_annotation_count.setStyleSheet(
            "color: #999999; font-size: 9px;")
        count_row.addWidget(self.lbl_annotation_count, 1)
        self.btn_delete_selected = QPushButton("Delete (D)")
        self.btn_delete_selected.setToolTip(
            "Delete the selected detection — like Reject but does NOT advance "
            "(view stays, nothing stays selected). Also via right-click → "
            "Delete.\nShortcut: D")
        self.btn_delete_selected.setFocusPolicy(Qt.NoFocus)
        self.btn_delete_selected.clicked.connect(self._delete_selected_annotation)
        count_row.addWidget(self.btn_delete_selected)
        vbox.addLayout(count_row)

        # Prediction review controls (CAD-style).
        review_lbl = QLabel("Review Predictions:")
        review_lbl.setStyleSheet("font-weight: bold; font-size: 10px; margin-top: 4px;")
        vbox.addWidget(review_lbl)
        self._build_review_controls(
            vbox, "Accept this prediction — saves as a training example")

        group.setLayout(vbox)
        layout.addWidget(group)

    @staticmethod
    def _review_btn_qss(bg: str, hover: str) -> str:
        """Rounded colored-button style matching the default (Skip) button."""
        return (
            f"QPushButton {{ background-color: {bg}; color: white; "
            f"border: none; border-radius: 6px; padding: 5px 10px; }}"
            f"QPushButton:hover {{ background-color: {hover}; }}"
            f"QPushButton:disabled {{ background-color: #3a3a3a; color: #888; }}"
        )

    def _build_review_controls(self, vbox, accept_tip: str):
        """Build one set of CAD-style prediction-review controls (prev/next,
        accept/reject/skip, accept-all/reject-all) and register it so
        ``_update_pred_review_widgets`` keeps every set in sync. Used by both
        the Mask tab (train review) and the Inference tab (deploy review)."""
        if not hasattr(self, '_review_sets'):
            self._review_sets = []
        s = {}
        nav = QHBoxLayout()
        nav.setSpacing(2)
        s['prev'] = QPushButton("Back (B)")
        s['prev'].setToolTip(
            "Go to the previous prediction (centers + white-highlights it).\n"
            "Shortcut: B")
        s['prev'].clicked.connect(self._pred_prev)
        nav.addWidget(s['prev'])
        s['nav_lbl'] = QLabel("0/0 predictions")
        s['nav_lbl'].setAlignment(Qt.AlignCenter)
        s['nav_lbl'].setStyleSheet("font-size: 10px;")
        nav.addWidget(s['nav_lbl'], 1)
        s['next'] = QPushButton("Next (N)")
        s['next'].setToolTip(
            "Go to the next prediction (centers + white-highlights it).\n"
            "Shortcut: N")
        s['next'].clicked.connect(self._pred_next)
        nav.addWidget(s['next'])
        vbox.addLayout(nav)

        row = QHBoxLayout()
        row.setSpacing(2)
        s['accept'] = QPushButton("Accept (A)")
        s['accept'].setToolTip(accept_tip + "<br>Shortcut: A")
        s['accept'].setStyleSheet(self._review_btn_qss("#2d7a3a", "#379247"))
        s['accept'].clicked.connect(self._accept_current_pred)
        row.addWidget(s['accept'])
        s['reject'] = QPushButton("Reject (R)")
        s['reject'].setToolTip(
            "Discard this detection — removes the mask and its list entry. "
            "In deploy mode also marks it 'rejected' in the output CSV.<br>"
            "Shortcut: R")
        s['reject'].setStyleSheet(self._review_btn_qss("#8a2c2c", "#a83636"))
        s['reject'].clicked.connect(self._reject_current_pred)
        row.addWidget(s['reject'])
        s['skip'] = QPushButton("Skip (S)")
        s['skip'].setToolTip(
            "Leave this prediction pending and advance to the next one (no "
            "decision recorded).<br>Shortcut: S")
        s['skip'].setStyleSheet(self._review_btn_qss("#5a5a5a", "#6a6a6a"))
        s['skip'].clicked.connect(self._skip_current_pred)
        row.addWidget(s['skip'])
        # Equal widths so the three review buttons line up uniformly.
        for k in ('accept', 'reject', 'skip'):
            s[k].setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        vbox.addLayout(row)

        bulk = QHBoxLayout()
        bulk.setSpacing(2)
        s['accept_all'] = QPushButton("Accept All")
        s['accept_all'].setToolTip(
            "Accept every pending prediction for this file at once "
            "(train mode → saved as training examples; deploy mode → marked "
            "'accepted' in the output CSV).")
        s['accept_all'].clicked.connect(self._accept_all_preds)
        bulk.addWidget(s['accept_all'])
        s['reject_all'] = QPushButton("Reject All")
        s['reject_all'].setToolTip(
            "Discard every pending prediction for this file at once "
            "(deploy mode also marks them 'rejected' in the output CSV).")
        s['reject_all'].clicked.connect(self._reject_all_preds)
        bulk.addWidget(s['reject_all'])
        s['clear_all'] = QPushButton("Clear All")
        s['clear_all'].setToolTip(
            "Remove ALL annotations from the project (confirmed examples + "
            "pending predictions). Asks for confirmation.")
        s['clear_all'].clicked.connect(self._clear_all_detections)
        bulk.addWidget(s['clear_all'])
        vbox.addLayout(bulk)

        for w in s.values():
            if isinstance(w, QPushButton):
                w.setEnabled(False)
                w.setFocusPolicy(Qt.NoFocus)
        self._review_sets.append(s)
        return s

    def _clear_all_detections(self):
        """Delete every annotation in the project — confirmed training examples
        on disk plus any pending predictions — after a confirmation prompt."""
        from fnt.usv.usv_detector.mad_examples import (
            count_examples, iter_examples, delete_example)
        n_examples = (count_examples(self._project.training_data_dir)
                      if self._project else 0)
        n_pred = len(self._pred_indices())
        if n_examples == 0 and n_pred == 0:
            self.status_bar.showMessage("No annotations to clear.")
            return
        reply = QMessageBox.question(
            self, "Clear All Annotations",
            f"This will permanently remove ALL annotations from the project:\n\n"
            f"  • {n_examples} confirmed training example(s)\n"
            f"  • {n_pred} pending prediction(s)\n\n"
            f"This cannot be undone. Are you sure you want to proceed?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        if reply != QMessageBox.Yes:
            return
        deleted = 0
        if self._project is not None:
            for ex in list(iter_examples(self._project.training_data_dir)):
                try:
                    delete_example(self._project.training_data_dir,
                                   ex['meta'].get('id'))
                    deleted += 1
                except Exception:
                    pass
        sg = self.spectrogram
        sg.annotations = []
        sg._rebuild_confirmed_mask()
        sg.update()
        self._pred_review_idx = None
        self._refresh_annotation_list()
        self._update_pred_review_widgets()
        self._log(f"Clear All — deleted {deleted} example(s), "
                  f"{n_pred} pending prediction(s)")
        self.status_bar.showMessage(
            f"Cleared all annotations ({deleted} example(s) deleted)")

    def _refresh_annotation_list(self):
        if not hasattr(self, 'annotation_list'):
            return
        self.spectrogram._selected_ann_idx = None
        flt = (self.combo_det_filter.currentText()
               if hasattr(self, 'combo_det_filter') else "All")
        self.annotation_list.blockSignals(True)
        self.annotation_list.setSortingEnabled(False)
        self.annotation_list.clear()
        cur_wav = ''
        if self.audio_files and 0 <= self.current_file_idx < len(self.audio_files):
            cur_wav = os.path.basename(self.audio_files[self.current_file_idx])
        sp = self._spec_params()
        sr = self.sample_rate or 1
        dt = self.spectrogram.hop / float(sr) if self.spectrogram.hop else 0
        df = (sr / 2.0) / (sp['nfft'] // 2) if sp.get('nfft') else 0
        entries = []
        if dt:
            for ann in self.spectrogram.annotations:
                is_pred = ann.get('status') == 'prediction'
                if is_pred and flt == "Confirmed":
                    continue
                if not is_pred and flt == "Pending":
                    continue
                entries.append(ann)
        entries.sort(key=lambda a: a['t0'])
        n_confirmed, n_pred = 0, 0
        for ann in entries:
            is_pred = ann.get('status') == 'prediction'
            t0 = ann['t0'] * dt
            t1 = ann['t1'] * dt
            dur = max(0.0, t1 - t0)
            durs = f"{dur:.2f}s" if dur >= 1.0 else f"{dur * 1000:.0f}ms"
            cls = ann.get('category', '')
            freq_lo = ann['f0'] * df / 1000 if df else 0
            freq_hi = ann['f1'] * df / 1000 if df else 0
            mask = ann.get('mask')
            pixels = int(mask.sum()) if mask is not None else 0
            score = ann.get('score', 0)
            icon = "○" if is_pred else "●"
            score_s = f"{score:.2f}" if is_pred and score else ""
            item = QTreeWidgetItem([
                icon, f"{t0:.2f}s", cls, durs,
                f"{freq_lo:.0f}-{freq_hi:.0f}",
                str(pixels), score_s,
            ])
            item.setData(0, Qt.UserRole, (cur_wav, t0, ann.get('id', ''),
                                          'prediction' if is_pred else 'confirmed'))
            color = QColor(255, 230, 90) if is_pred else QColor(90, 160, 255)
            for c in range(self.annotation_list.columnCount()):
                item.setForeground(c, color)
            self.annotation_list.addTopLevelItem(item)
            if is_pred:
                n_pred += 1
            else:
                n_confirmed += 1
        self.annotation_list.setSortingEnabled(True)
        self.annotation_list.blockSignals(False)
        parts = []
        if n_confirmed:
            parts.append(f"{n_confirmed} confirmed")
        if n_pred:
            parts.append(f"{n_pred} prediction(s)")
        self.lbl_annotation_count.setText(
            f"{n_confirmed + n_pred} detection(s)" +
            (f" ({', '.join(parts)})" if len(parts) > 1 else "")
        )
        self._update_pred_review_widgets()
        self._update_train_button_count()
        self._update_file_list_counts()

    def _update_train_button_count(self):
        """Show the confirmed-example count on the Train button so the user
        knows how much data the next run will use."""
        if not hasattr(self, 'btn_train'):
            return
        n = 0
        if self._project is not None:
            try:
                from fnt.usv.usv_detector.mad_examples import count_examples
                n = count_examples(self._project.training_data_dir)
            except Exception:
                n = 0
        self.btn_train.setText(f"Train ({n} examples)" if n else "Train")

    def _on_annotation_list_selected(self, current, _previous=None):
        """Single-click in detections list: jump to the detection and highlight it."""
        if current is None:
            self.spectrogram._selected_ann_idx = None
            self.spectrogram.update()
            return
        data = current.data(0, Qt.UserRole)
        if not data:
            return
        wav, t0, eid, status = data
        target_idx = None
        for i, p in enumerate(self.audio_files):
            if os.path.basename(p) == wav:
                target_idx = i
                break
        if target_idx is None:
            return
        if target_idx != self.current_file_idx:
            self.file_list.setCurrentRow(target_idx)
        if self.spectrogram.total_duration > 0:
            window = self.spectrogram.view_end - self.spectrogram.view_start
            self.spectrogram.view_start = max(0.0, t0 - window / 2)
            self.spectrogram.view_end = min(
                self.spectrogram.total_duration,
                self.spectrogram.view_start + window)
            self._invalidate_spec_cache()
            self._sync_scrollbar_from_view()
        # Highlight the matching annotation with a white outline.
        sel = None
        for ai, ann in enumerate(self.spectrogram.annotations):
            if ann.get('id') == eid:
                sel = ai
                break
        self.spectrogram._selected_ann_idx = sel
        self.spectrogram.update()
        # If the selected detection is a pending prediction, sync the review
        # pointer to it so Accept/Reject act on THIS mask (not a stale one).
        if sel is not None and status == 'prediction':
            preds = self._pred_indices()
            if sel in preds:
                self._pred_review_idx = preds.index(sel)
                self._update_pred_review_widgets()

    # --- prediction review (CAD-style accept / reject / skip) -----------
    def _pred_indices(self) -> List[int]:
        """Annotation indices with status='prediction', ordered by the
        detection table's current visual sort so Back/Next follow whatever
        column the user clicked."""
        if hasattr(self, 'annotation_list'):
            ordered_ids = []
            for i in range(self.annotation_list.topLevelItemCount()):
                data = self.annotation_list.topLevelItem(i).data(0, Qt.UserRole)
                if data and data[3] == 'prediction':
                    ordered_ids.append(data[2])
            if ordered_ids:
                id_to_ann = {a.get('id'): idx
                             for idx, a in enumerate(self.spectrogram.annotations)
                             if a.get('status') == 'prediction'}
                return [id_to_ann[eid] for eid in ordered_ids
                        if eid in id_to_ann]
        return [i for i, a in enumerate(self.spectrogram.annotations)
                if a.get('status') == 'prediction']

    def _update_pred_review_widgets(self):
        preds = self._pred_indices()
        n = len(preds)
        has = n > 0
        cur = self._pred_review_idx
        cur_valid = cur is not None and 0 <= cur < n
        nav_text = (f"{cur + 1}/{n} predictions" if cur_valid
                    else f"0/{n} predictions")
        # Clear All is available whenever the project may hold annotations or
        # there are pending predictions (the handler re-checks for emptiness).
        can_clear = has or self._project is not None
        for s in getattr(self, '_review_sets', []):
            s['prev'].setEnabled(has and cur_valid and cur > 0)
            s['next'].setEnabled(has and cur_valid and cur < n - 1)
            for k in ('accept', 'reject', 'skip'):
                s[k].setEnabled(has and cur_valid)
            s['accept_all'].setEnabled(has)
            s['reject_all'].setEnabled(has)
            s['clear_all'].setEnabled(can_clear)
            s['nav_lbl'].setText(nav_text)

    def _jump_to_pred(self, pred_idx: int):
        preds = self._pred_indices()
        if not preds or pred_idx < 0 or pred_idx >= len(preds):
            return
        self._pred_review_idx = pred_idx
        ann_idx = preds[pred_idx]
        ann = self.spectrogram.annotations[ann_idx]
        if self.spectrogram.total_duration > 0:
            dt = self.spectrogram.hop / float(self.sample_rate) if self.sample_rate else 1
            t_center = (ann['t0'] + ann['t1']) / 2.0 * dt
            window = self.spectrogram.view_end - self.spectrogram.view_start
            self.spectrogram.view_start = max(0.0, t_center - window / 2)
            self.spectrogram.view_end = min(
                self.spectrogram.total_duration,
                self.spectrogram.view_start + window)
            self._invalidate_spec_cache()
            self._sync_scrollbar_from_view()
        self.spectrogram._selected_ann_idx = ann_idx
        self.spectrogram.update()
        self._select_list_row_for_id(ann.get('id'))
        self._update_pred_review_widgets()

    def _select_list_row_for_id(self, eid):
        """Highlight the detections-list row whose example/prediction id matches,
        so list selection follows Prev/Next prediction navigation."""
        if eid is None or not hasattr(self, 'annotation_list'):
            return
        self.annotation_list.blockSignals(True)
        for i in range(self.annotation_list.topLevelItemCount()):
            item = self.annotation_list.topLevelItem(i)
            data = item.data(0, Qt.UserRole)
            if data and data[2] == eid:
                self.annotation_list.setCurrentItem(item)
                self.annotation_list.scrollToItem(item)
                break
        self.annotation_list.blockSignals(False)

    def _pred_prev(self):
        if self._pred_review_idx is not None and self._pred_review_idx > 0:
            self._jump_to_pred(self._pred_review_idx - 1)

    def _pred_next(self):
        preds = self._pred_indices()
        if (self._pred_review_idx is not None and
                self._pred_review_idx < len(preds) - 1):
            self._jump_to_pred(self._pred_review_idx + 1)

    def _accept_current_pred(self):
        if self._apply_to_box_selection('accept'):
            return
        preds = self._pred_indices()
        if not preds or self._pred_review_idx is None:
            return
        if self._pred_review_idx >= len(preds):
            return
        ann_idx = preds[self._pred_review_idx]
        self._log(f"Accept prediction {self._pred_describe(ann_idx)} "
                  f"[{self._review_mode}]")
        if self._review_mode == 'deploy':
            self._write_deploy_status(self.spectrogram.annotations[ann_idx],
                                      'accepted')
            self.spectrogram.remove_annotation(ann_idx)
        else:
            self._accept_prediction(ann_idx)
        self._after_review_decision()

    def _reject_current_pred(self):
        if self._apply_to_box_selection('reject'):
            return
        preds = self._pred_indices()
        if not preds or self._pred_review_idx is None:
            return
        if self._pred_review_idx >= len(preds):
            return
        ann_idx = preds[self._pred_review_idx]
        self._log(f"Reject prediction {self._pred_describe(ann_idx)} "
                  f"[{self._review_mode}]")
        if self._review_mode == 'deploy':
            self._write_deploy_status(self.spectrogram.annotations[ann_idx],
                                      'rejected')
        # Reject removes the mask AND its detection-list entry.
        self.spectrogram.remove_annotation(ann_idx)
        self._after_review_decision()

    def _pred_describe(self, ann_idx: int) -> str:
        """Short '#i @ t.tts' description of an annotation for the log."""
        if not (0 <= ann_idx < len(self.spectrogram.annotations)):
            return "?"
        ann = self.spectrogram.annotations[ann_idx]
        t = "?"
        if self.sample_rate and self.spectrogram.hop:
            dt = self.spectrogram.hop / float(self.sample_rate)
            t = f"{ann.get('t0', 0) * dt:.2f}s"
        cur = (self._pred_review_idx + 1) if self._pred_review_idx is not None else 0
        return f"{cur}/{len(self._pred_indices())} @ {t}"

    def _after_review_decision(self):
        """Shared tail for accept/reject: rebuild the list and advance to (and
        select) the next pending prediction so list + preview stay in sync."""
        self._refresh_annotation_list()
        preds = self._pred_indices()
        if preds:
            self._pred_review_idx = min(self._pred_review_idx or 0, len(preds) - 1)
            self._jump_to_pred(self._pred_review_idx)
        else:
            self._pred_review_idx = None
            self._update_pred_review_widgets()
            self.status_bar.showMessage("All predictions reviewed")

    def _write_deploy_status(self, ann: dict, status: str):
        """Persist an accept/reject decision for a deployment prediction back
        into its sibling output CSV (matched by blob_id)."""
        csv_path = ann.get('csv_path')
        bid = ann.get('blob_id')
        if not csv_path or bid is None or not os.path.isfile(csv_path):
            return
        try:
            from fnt.usv.usv_detector.mad_inference import (
                read_blob_csv, write_blob_csv)
            rows = read_blob_csv(csv_path)
            for r in rows:
                if r.get('blob_id') == bid:
                    r['status'] = status
            write_blob_csv(csv_path, rows)
        except Exception:
            pass

    def _skip_current_pred(self):
        preds = self._pred_indices()
        if not preds or self._pred_review_idx is None:
            return
        self._log("Skip prediction")
        if self._pred_review_idx < len(preds) - 1:
            self._jump_to_pred(self._pred_review_idx + 1)
        else:
            self.status_bar.showMessage("No more predictions to review")

    def _accept_prediction(self, ann_idx: int):
        """Accept a prediction annotation — save as training example."""
        sg = self.spectrogram
        if not (0 <= ann_idx < len(sg.annotations)):
            return
        ann = sg.annotations[ann_idx]
        if self.audio_data is None or not self.audio_files:
            return
        if self._project is not None:
            cls_name = ann.get('category') or self._project.last_class or 'USV'
        else:
            cls_name = ann.get('category') or self._session_last_class or 'USV'
        comp = (ann['f0'], ann['f1'], ann['t0'], ann['t1'], ann['mask'])
        try:
            ex_id = self._save_component_example(cls_name, comp)
        except Exception as e:
            self.status_bar.showMessage(f"Failed to save: {e}")
            return
        ann['status'] = 'accepted'
        ann['id'] = ex_id
        sg._rebuild_confirmed_mask()
        sg.update()
        self._refresh_annotation_list()
        self.status_bar.showMessage(f"Accepted prediction as '{cls_name}'")

    def _accept_all_preds(self):
        preds = self._pred_indices()
        if not preds:
            return
        self._log(f"Accept All — {len(preds)} prediction(s) [{self._review_mode}]")
        if self._review_mode == 'deploy':
            for ann_idx in preds:
                self._write_deploy_status(
                    self.spectrogram.annotations[ann_idx], 'accepted')
            for ann_idx in sorted(preds, reverse=True):
                self.spectrogram.annotations.pop(ann_idx)
            self.spectrogram._rebuild_confirmed_mask()
            self.spectrogram.update()
            self._pred_review_idx = None
            self._refresh_annotation_list()
            self._update_pred_review_widgets()
            self.status_bar.showMessage(f"Accepted {len(preds)} prediction(s)")
            return
        if self.audio_data is None or not self.audio_files:
            return
        if self._project is not None:
            cls_name = self._project.last_class or 'USV'
        else:
            cls_name = self._session_last_class or 'USV'
        n = 0
        for ann_idx in preds:
            ann = self.spectrogram.annotations[ann_idx]
            comp = (ann['f0'], ann['f1'], ann['t0'], ann['t1'], ann['mask'])
            try:
                ex_id = self._save_component_example(cls_name, comp)
                ann['status'] = 'accepted'
                ann['id'] = ex_id
                n += 1
            except Exception:
                continue
        self.spectrogram._rebuild_confirmed_mask()
        self.spectrogram.update()
        self._pred_review_idx = None
        self._refresh_annotation_list()
        self._update_pred_review_widgets()
        self.status_bar.showMessage(f"Accepted {n} prediction(s) as '{cls_name}'")

    def _reject_all_preds(self):
        preds = self._pred_indices()
        if not preds:
            return
        self._log(f"Reject All — {len(preds)} prediction(s) [{self._review_mode}]")
        if self._review_mode == 'deploy':
            for ann_idx in preds:
                self._write_deploy_status(
                    self.spectrogram.annotations[ann_idx], 'rejected')
        for ann_idx in sorted(preds, reverse=True):
            self.spectrogram.annotations.pop(ann_idx)
        self.spectrogram._rebuild_confirmed_mask()
        self.spectrogram.update()
        self._pred_review_idx = None
        self._refresh_annotation_list()
        self._update_pred_review_widgets()
        self.status_bar.showMessage(f"Rejected {len(preds)} prediction(s)")

    def _load_predictions_as_annotations(self, wav: Optional[str] = None):
        """Load inference predictions for ``wav`` (default: current training
        file) and turn each blob into a yellow prediction annotation."""
        if wav is None:
            if not self.audio_files:
                return
            wav = self.audio_files[self.current_file_idx]
        csv_path = pred_csv_sibling_path(wav)
        if not os.path.isfile(csv_path):
            return
        try:
            from fnt.usv.usv_detector.mad_inference import read_blob_csv
            rows = read_blob_csv(csv_path)
        except Exception:
            return
        sp = self._spec_params()
        if self.sample_rate is None:
            return
        dt = (sp['nperseg'] - sp['noverlap']) / float(self.sample_rate)
        df = (self.sample_rate / 2.0) / (sp['nfft'] // 2)
        # Load probability mask from h5 for extracting per-blob masks.
        pred_mask = None
        try:
            from fnt.usv.usv_detector.fnt_mask_store import (
                masks_sibling_path, read_prob,
            )
            pred_mask = read_prob(masks_sibling_path(wav))
        except Exception:
            pass
        threshold = 0.5
        sg = self.spectrogram
        sg.annotations = [a for a in sg.annotations
                          if a.get('status') != 'prediction']
        wav_name = os.path.basename(wav)
        n_added = 0
        for r in rows:
            if r.get('status') in ('accepted', 'rejected'):
                continue
            t0 = int(round(r['start_s'] / dt))
            t1 = int(round(r['stop_s'] / dt))
            f0 = int(round(r['min_freq_hz'] / df))
            f1 = int(round(r['max_freq_hz'] / df))
            if t1 <= t0 or f1 <= f0:
                continue
            if (pred_mask is not None and
                    f1 <= pred_mask.shape[0] and t1 <= pred_mask.shape[1]):
                blob_region = pred_mask[f0:f1, t0:t1] >= threshold
            else:
                blob_region = np.ones((f1 - f0, t1 - t0), dtype=bool)
            if not blob_region.any():
                continue
            sg.annotations.append({
                'id': f'pred_{n_added}',
                'category': (self._project.last_class if self._project else
                             self._session_last_class) or 'USV',
                'f0': f0, 'f1': f1, 't0': t0, 't1': t1,
                'mask': np.ascontiguousarray(blob_region),
                'status': 'prediction',
                'source_wav': wav_name,
                'score': r.get('score', 0),
                'blob_id': r.get('blob_id'),
                'csv_path': csv_path,
            })
            n_added += 1
        sg._rebuild_confirmed_mask()
        sg.update()
        self._pred_review_idx = 0 if n_added > 0 else None
        self._refresh_annotation_list()
        self._update_pred_review_widgets()
        if n_added:
            self._jump_to_pred(0)
            self._log(f"Loaded {n_added} prediction(s) for "
                      f"{os.path.basename(str(wav))}")
        return n_added

    def _create_deploy_model_section(self, layout):
        group = QGroupBox("1. Model")
        vbox = QVBoxLayout()
        vbox.setSpacing(4)

        model_row = QHBoxLayout()
        model_row.setSpacing(2)
        self.combo_deploy_model = QComboBox()
        self.combo_deploy_model.setToolTip(
            "Trained models in this project's models/ folder. Pick one to "
            "apply to the file queue below."
        )
        self.combo_deploy_model.currentIndexChanged.connect(
            self._on_deploy_model_changed
        )
        model_row.addWidget(self.combo_deploy_model, 1)
        self.btn_deploy_refresh = QPushButton("Refresh")
        self.btn_deploy_refresh.setToolTip("Rescan the project's models/ folder")
        self.btn_deploy_refresh.clicked.connect(self._refresh_deploy_models)
        model_row.addWidget(self.btn_deploy_refresh)
        self.btn_deploy_browse = QPushButton("Browse…")
        self.btn_deploy_browse.setToolTip("Pick a weights.pt outside this project")
        self.btn_deploy_browse.clicked.connect(self._browse_deploy_model)
        model_row.addWidget(self.btn_deploy_browse)
        vbox.addLayout(model_row)

        self.lbl_deploy_model_info = QLabel("No model selected")
        self.lbl_deploy_model_info.setStyleSheet(
            "color: #999999; font-size: 9px;")
        self.lbl_deploy_model_info.setWordWrap(True)
        vbox.addWidget(self.lbl_deploy_model_info)

        group.setLayout(vbox)
        layout.addWidget(group)

    def _create_deploy_queue_section(self, layout):
        group = QGroupBox("2. File Queue")
        vbox = QVBoxLayout()
        vbox.setSpacing(4)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(2)
        self.btn_deploy_add_files = QPushButton("Add Files…")
        self.btn_deploy_add_files.setToolTip("Add .wav files to the inference queue")
        self.btn_deploy_add_files.clicked.connect(self._add_deploy_files)
        btn_row.addWidget(self.btn_deploy_add_files)
        self.btn_deploy_add_folder = QPushButton("Add Folder…")
        self.btn_deploy_add_folder.setToolTip(
            "Add all .wav files in a folder (non-recursive)")
        self.btn_deploy_add_folder.clicked.connect(self._add_deploy_folder)
        btn_row.addWidget(self.btn_deploy_add_folder)
        self.btn_deploy_clear = QPushButton("Clear")
        self.btn_deploy_clear.setToolTip("Empty the inference queue")
        self.btn_deploy_clear.clicked.connect(self._clear_deploy_files)
        btn_row.addWidget(self.btn_deploy_clear)
        vbox.addLayout(btn_row)

        self.deploy_list = QListWidget()
        self.deploy_list.setMaximumHeight(150)
        self.deploy_list.setToolTip(
            "Files to run inference on. Click a row to preview it.")
        self.deploy_list.currentRowChanged.connect(self._on_deploy_file_selected)
        vbox.addWidget(self.deploy_list)

        self.lbl_deploy_queue = QLabel("0 files queued")
        self.lbl_deploy_queue.setStyleSheet("color: #999999; font-size: 9px;")
        vbox.addWidget(self.lbl_deploy_queue)

        group.setLayout(vbox)
        layout.addWidget(group)

    def _create_inference_section(self, layout):
        group = QGroupBox("3. Run Inference")
        vbox = QVBoxLayout()
        vbox.setSpacing(4)

        form = QFormLayout()
        self.spin_infer_threshold = QDoubleSpinBox()
        self.spin_infer_threshold.setRange(0.01, 0.99)
        self.spin_infer_threshold.setSingleStep(0.05)
        self.spin_infer_threshold.setValue(0.5)
        self.spin_infer_threshold.setToolTip(
            "<b>Probability threshold</b>: a pixel is called positive when the "
            "model's sigmoid output exceeds this value. Lower → more (and "
            "larger) detections, more false positives; higher → fewer, "
            "higher-confidence detections."
        )
        form.addRow("Probability threshold:", self.spin_infer_threshold)

        self.spin_infer_min_blob = QSpinBox()
        self.spin_infer_min_blob.setRange(1, 10000)
        self.spin_infer_min_blob.setValue(8)
        self.spin_infer_min_blob.setToolTip(
            "<b>Minimum blob size</b> in pixels: connected components smaller "
            "than this are discarded as noise. Raise to suppress speckle, lower "
            "to keep very short/faint calls."
        )
        form.addRow("Min blob pixels:", self.spin_infer_min_blob)

        self.combo_infer_device = QComboBox()
        self.combo_infer_device.addItems(["auto", "cuda", "mps", "cpu"])
        self.combo_infer_device.setToolTip(
            "<b>Compute device</b> for inference: <b>auto</b> picks the best "
            "available (CUDA &gt; MPS &gt; CPU). cuda = NVIDIA GPU, mps = "
            "Apple-Silicon GPU, cpu = portable but slower."
        )
        form.addRow("Device:", self.combo_infer_device)
        vbox.addLayout(form)

        self.chk_infer_save_csv = QCheckBox("Save blob CSV")
        self.chk_infer_save_csv.setChecked(True)
        self.chk_infer_save_csv.setToolTip(
            "Write one row per detected call (time, frequency, score, status) to "
            "a sibling <i>*_FNT_MAD_predictions.csv</i> — the analysis output and "
            "the file Accept/Reject decisions are saved back into."
        )
        vbox.addWidget(self.chk_infer_save_csv)
        self.chk_infer_preserve = QCheckBox(
            "Preserve painted labels (skip labeled time regions)"
        )
        self.chk_infer_preserve.setChecked(True)
        self.chk_infer_preserve.setToolTip(
            "Zero out the model's output in any time column you've already "
            "hand-labeled, so inference never overwrites your manual "
            "annotations. Leave on during the human-in-the-loop cycle."
        )
        vbox.addWidget(self.chk_infer_preserve)

        self.btn_infer_run = QPushButton("Run Inference")
        self.btn_infer_run.setToolTip(
            "Run the selected model on every file in the queue, writing sibling "
            "prediction CSV outputs. Then review the detections below."
        )
        self.btn_infer_run.clicked.connect(self._on_deploy_infer)
        self.btn_infer_run.setEnabled(False)
        vbox.addWidget(self.btn_infer_run)

        self.infer_panel = MADRunPanel(show_plot=False)
        self.infer_panel.run_finished.connect(
            lambda ok: self.btn_infer_run.setEnabled(self._project is not None)
        )
        vbox.addWidget(self.infer_panel)

        group.setLayout(vbox)
        layout.addWidget(group)

    def _create_deploy_review_section(self, layout):
        group = QGroupBox("4. Detections")
        vbox = QVBoxLayout()
        vbox.setSpacing(4)
        hint = QLabel(
            "Select a queued file above to review its detections here. "
            "Accept keeps the call; Reject discards it. Decisions are written "
            "back to the file's predictions CSV."
        )
        hint.setStyleSheet("color: #888888; font-size: 9px; font-style: italic;")
        hint.setWordWrap(True)
        vbox.addWidget(hint)
        self._build_review_controls(
            vbox, "Accept this detection (kept in the output CSV)")
        group.setLayout(vbox)
        layout.addWidget(group)

    # --- inline training/inference handlers ---------------------------
    def _on_left_tab_changed(self, idx: int):
        # The two tabs share one spectrogram canvas but operate on different
        # file sets, so reload the active tab's current file and set the
        # review mode that accept/reject acts in. (Guard: this fires during
        # tab construction, before later attributes exist.)
        if not hasattr(self, '_inference_tab_index'):
            return
        if idx == self._inference_tab_index:
            self._review_mode = 'deploy'
            if self._project is None or self.combo_deploy_model.count() == 0:
                QMessageBox.information(
                    self, "Inference",
                    "To run inference, open a project that contains trained "
                    "models.\n\nUse File → Open Project to load one, or train a "
                    "model on the Label && Train tab first."
                )
            if (self._deploy_file_idx is not None and
                    0 <= self._deploy_file_idx < len(self.deploy_files)):
                self._load_deploy_file(self.deploy_files[self._deploy_file_idx])
        else:
            self._review_mode = 'train'
            if (self.audio_files and
                    0 <= self.current_file_idx < len(self.audio_files)):
                self._load_current_file()

    def _on_arch_changed(self, _idx: int = 0):
        # HRNet brings its own encoder, so the encoder picker is irrelevant.
        is_hrnet = self.combo_arch.currentData() == "hrnet"
        self.combo_train_encoder.setEnabled(not is_hrnet)

    def _build_inline_train_config(self):
        from fnt.usv.usv_detector.mad_training import UNetTrainingConfig
        sp = self._spec_params()
        return UNetTrainingConfig(
            project_dir=self._project.project_dir,
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
            # Training needs a project to save the model into — offer to make
            # one now rather than blocking the user.
            box = QMessageBox(self)
            box.setWindowTitle("No project")
            box.setIcon(QMessageBox.Question)
            box.setText("Training needs a project to store the model.")
            box.setInformativeText("Create a new project or open an existing one?")
            new_btn = box.addButton("Create New Project…", QMessageBox.AcceptRole)
            open_btn = box.addButton("Open Project…", QMessageBox.AcceptRole)
            box.addButton("Cancel", QMessageBox.RejectRole)
            box.setDefaultButton(new_btn)
            box.exec_()
            if box.clickedButton() is new_btn:
                self._menu_new_project()
            elif box.clickedButton() is open_btn:
                self._menu_open_project()
            if self._project is None:
                return
        self._consolidate_sibling_examples()
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
        self._log(f"Train START — {n_examples} examples, arch="
                  f"{cfg.model_arch}, encoder={cfg.encoder_name}, "
                  f"epochs={cfg.n_epochs}, batch={cfg.batch_size}")
        self.btn_train.setEnabled(False)
        self._set_training_view(True)
        self.train_panel.start_run()
        self._start_training(cfg, post_inference_scope=post_scope,
                             reporter=self.train_panel)

    def _consolidate_sibling_examples(self):
        """Move any per-wav h5 sibling examples into the project's consolidated
        ``training_data.h5`` so the training pipeline finds them."""
        if self._project is None:
            return
        from fnt.usv.usv_detector.fnt_mask_store import (
            masks_sibling_path, td_iter_examples, td_count, td_delete,
        )
        from fnt.usv.usv_detector.mad_examples import save_example
        td_dir = self._project.training_data_dir
        n = 0
        for fp in self.audio_files:
            h5 = masks_sibling_path(fp)
            if td_count(h5) == 0:
                continue
            for ex in list(td_iter_examples(h5)):
                meta = ex['meta']
                eid = meta.get('id', '')
                try:
                    save_example(td_dir, ex['spec'], ex['mask'], meta, eid)
                    td_delete(h5, eid)
                    n += 1
                except Exception:
                    continue
        if n:
            self._log(f"Consolidated {n} per-wav label(s) into project store")

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

    # --- deployment: model dropdown -----------------------------------
    def _refresh_deploy_models(self):
        """Populate the model dropdown from <project>/models/ (dirs with
        a weights.pt). Preserves the current selection where possible."""
        if not hasattr(self, 'combo_deploy_model'):
            return
        prev = self._selected_deploy_model_path()
        self.combo_deploy_model.blockSignals(True)
        self.combo_deploy_model.clear()
        found = []
        if self._project is not None:
            models_root = os.path.join(self._project.project_dir, 'models')
            if os.path.isdir(models_root):
                for name in sorted(os.listdir(models_root)):
                    d = os.path.join(models_root, name)
                    w = os.path.join(d, 'weights.pt')
                    if os.path.isdir(d) and os.path.isfile(w):
                        found.append((name, w))
        for name, w in found:
            self.combo_deploy_model.addItem(name, w)
        # Restore prior selection if it survived the rescan.
        if prev:
            for i in range(self.combo_deploy_model.count()):
                if self.combo_deploy_model.itemData(i) == prev:
                    self.combo_deploy_model.setCurrentIndex(i)
                    break
        elif found:
            self.combo_deploy_model.setCurrentIndex(self.combo_deploy_model.count() - 1)
        self.combo_deploy_model.blockSignals(False)
        self._on_deploy_model_changed()

    def _browse_deploy_model(self):
        root = (os.path.join(self._project.project_dir, 'models')
                if self._project else os.path.expanduser("~"))
        start = root if os.path.isdir(root) else os.path.expanduser("~")
        path, _ = QFileDialog.getOpenFileName(
            self, "Select trained weights", start,
            "PyTorch weights (*.pt *.pth);;All Files (*)"
        )
        if path:
            label = os.path.basename(os.path.dirname(path)) or os.path.basename(path)
            self.combo_deploy_model.addItem(label, path)
            self.combo_deploy_model.setCurrentIndex(
                self.combo_deploy_model.count() - 1)

    def _selected_deploy_model_path(self) -> Optional[str]:
        if not hasattr(self, 'combo_deploy_model'):
            return None
        data = self.combo_deploy_model.currentData()
        return str(data) if data else None

    def _on_deploy_model_changed(self, _i: int = 0):
        path = self._selected_deploy_model_path()
        if path and os.path.isfile(path):
            self.lbl_deploy_model_info.setText(
                os.path.basename(os.path.dirname(path)))
        else:
            self.lbl_deploy_model_info.setText("No model selected")

    # --- deployment: file queue ---------------------------------------
    def _add_deploy_files(self):
        paths, _ = QFileDialog.getOpenFileNames(
            self, "Add .wav files", os.path.expanduser("~"),
            "WAV files (*.wav);;All Files (*)"
        )
        self._append_deploy_files(paths)

    def _add_deploy_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Add folder of .wav files", os.path.expanduser("~")
        )
        if not folder:
            return
        wavs = [os.path.join(folder, f) for f in sorted(os.listdir(folder))
                if f.lower().endswith('.wav')]
        self._append_deploy_files(wavs)

    def _append_deploy_files(self, paths):
        added = 0
        for p in paths or []:
            if p and os.path.isfile(p) and p not in self.deploy_files:
                self.deploy_files.append(p)
                added += 1
        if added:
            self._refresh_deploy_queue()

    def _clear_deploy_files(self):
        self.deploy_files = []
        self._deploy_file_idx = None
        self._refresh_deploy_queue()

    def _refresh_deploy_queue(self):
        if not hasattr(self, 'deploy_list'):
            return
        self.deploy_list.blockSignals(True)
        self.deploy_list.clear()
        for p in self.deploy_files:
            self.deploy_list.addItem(os.path.basename(p))
        self.deploy_list.blockSignals(False)
        self.lbl_deploy_queue.setText(f"{len(self.deploy_files)} file(s) queued")

    def _on_deploy_file_selected(self, row: int):
        if 0 <= row < len(self.deploy_files):
            self._deploy_file_idx = row
            self._load_deploy_file(self.deploy_files[row])

    def _on_deploy_infer(self):
        model = self._selected_deploy_model_path() or self._default_model_path()
        if not model or not os.path.isfile(model):
            QMessageBox.warning(
                self, "Model missing",
                "Select a trained model (or train one) before running inference."
            )
            return
        wavs = list(self.deploy_files)
        if not wavs:
            QMessageBox.warning(
                self, "Empty queue",
                "Add files to the inference queue first."
            )
            return
        from fnt.usv.usv_detector.mad_inference import MADInferenceConfig
        cfg = MADInferenceConfig(
            model_path=model,
            threshold=self.spin_infer_threshold.value(),
            min_blob_pixels=self.spin_infer_min_blob.value(),
            device=self.combo_infer_device.currentText(),
            save_blob_csv=self.chk_infer_save_csv.isChecked(),
            preserve_labels=self.chk_infer_preserve.isChecked(),
            training_data_dir=(self._project.training_data_dir
                               if self._project else ""),
        )
        self.btn_infer_run.setEnabled(False)
        self.infer_panel.start_run()
        self._start_inference(cfg, wavs, reporter=self.infer_panel)

    def _load_deploy_file(self, filepath: str):
        """Preview a queued deployment file in the shared spectrogram.

        Loads audio + grid only; deploy-mode prediction correction is added in
        Phase 3. Does not touch the training example store."""
        if not filepath or not os.path.isfile(filepath):
            return
        self._stop_playback()
        self._clear_predictions()
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
            self.waveform_overview.set_audio_data(self.audio_data, self.sample_rate)
            self.waveform_overview.set_view_range(
                self.spectrogram.view_start, self.spectrogram.view_end
            )
            n_pred = 0
            if self._project is not None:
                cfg = self._project
                self.spectrogram.init_mask(
                    audio_len=len(self.audio_data), sample_rate=self.sample_rate,
                    nperseg=cfg.nperseg, noverlap=cfg.noverlap, nfft=cfg.nfft,
                )
                self.spectrogram.set_annotations([])
                # Deploy-mode review: load this file's predictions for correction.
                n_pred = self._load_predictions_as_annotations(wav=filepath) or 0
            self._sync_scrollbar_from_view()
            suffix = f"  |  {n_pred} prediction(s)" if n_pred else ""
            self.status_bar.showMessage(
                f"{os.path.basename(filepath)}  |  "
                f"{len(self.audio_data) / self.sample_rate:.2f}s @ "
                f"{self.sample_rate} Hz{suffix}"
            )
        except Exception as e:
            QMessageBox.warning(
                self, "Load error", f"{os.path.basename(filepath)}:\n{e}")
        finally:
            QApplication.restoreOverrideCursor()
        self._update_paint_buttons_enabled()
        self._update_playback_buttons_enabled()

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
        file_menu.addAction(self.act_add_folder)

        self.act_add_files = QAction("Add &Files…", self)
        self.act_add_files.triggered.connect(self._add_audio_files)
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
        labels_menu.addSeparator()
        act_sam_model = QAction("Choose SAM2 &Model…", self)
        act_sam_model.setToolTip("Pick or switch the SAM2 checkpoint used for labeling")
        act_sam_model.triggered.connect(self._change_sam_model)
        labels_menu.addAction(act_sam_model)

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
        self.act_load_pred.triggered.connect(self._load_predictions_as_annotations)
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

        # Arrow keys (pan/zoom) are handled by an application event filter
        # instead of QShortcuts, so they work even when a scrollbar, list, or
        # button holds focus and would otherwise swallow the arrow key. Up/Down
        # are fully dedicated to zoom; B/N navigate predictions instead.
        make(Qt.Key_B, self._shortcut_pred_prev)   # Back through predictions
        make(Qt.Key_N, self._shortcut_pred_next)   # Next prediction
        make(Qt.Key_E, self._shortcut_toggle_eraser)
        make(Qt.Key_M, self._shortcut_mask_view)
        make(Qt.Key_G, self._shortcut_toggle_sam)
        make(Qt.Key_U, self._undo_last)
        make(Qt.Key_Return, self._confirm_pending)
        make(Qt.Key_Enter, self._confirm_pending)
        make(Qt.Key_Escape, self._clear_pending)
        make(Qt.Key_Space, self._shortcut_toggle_playback)
        make("[", self._shortcut_brush_smaller)
        make("]", self._shortcut_brush_bigger)
        make("V", self._shortcut_cycle_view)
        make("D", self._delete_selected_annotation)
        make("A", lambda: self._shortcut_review('accepted'))
        make("R", lambda: self._shortcut_review('rejected'))
        make("S", lambda: self._shortcut_review('skipped'))

    def _shortcut_pred_prev(self):
        if not self._focus_is_edit():
            self._pred_prev()

    def _shortcut_pred_next(self):
        if not self._focus_is_edit():
            self._pred_next()

    def _focus_is_edit(self):
        focus = QApplication.focusWidget()
        return isinstance(focus, (QSpinBox, QDoubleSpinBox, QComboBox))

    def eventFilter(self, obj, event):
        # Route arrow keys to spectrogram pan/zoom regardless of which control
        # has focus (scrollbars, list, buttons all otherwise eat arrow keys).
        if event.type() in (QEvent.KeyPress, QEvent.ShortcutOverride) and \
                self.isActiveWindow():
            key = event.key()
            if key in (Qt.Key_Left, Qt.Key_Right, Qt.Key_Up, Qt.Key_Down):
                from PyQt5.QtWidgets import QAbstractSpinBox, QLineEdit
                fw = QApplication.focusWidget()
                if isinstance(fw, (QAbstractSpinBox, QLineEdit, QComboBox,
                                   QTextEdit)):
                    return False
                if event.type() == QEvent.ShortcutOverride:
                    event.accept()
                    return True
                if self.spectrogram.total_duration <= 0:
                    return False
                if key == Qt.Key_Left:
                    self._pan_left()
                elif key == Qt.Key_Right:
                    self._pan_right()
                elif key == Qt.Key_Up:
                    self._zoom_in()
                elif key == Qt.Key_Down:
                    self._zoom_out()
                return True
        return super().eventFilter(obj, event)

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
        # Spec + Mask → Mask only → Spec only → repeat
        order = ['overlay', 'mask_only', 'spec']
        try:
            idx = order.index(self.spectrogram.view_mode)
        except ValueError:
            idx = 0
        self._set_view_mode(order[(idx + 1) % len(order)])

    def _shortcut_review(self, status: str):
        if self._focus_is_edit():
            return
        # A/R/S route to the unified prediction review on the Detections panel.
        preds = self._pred_indices()
        if preds and self._pred_review_idx is not None:
            if status == 'accepted':
                self._accept_current_pred()
            elif status == 'rejected':
                self._reject_current_pred()
            elif status == 'skipped':
                self._skip_current_pred()

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
        folder = QFileDialog.getExistingDirectory(
            self, "Add folder of .wav files (non-recursive)",
            os.path.expanduser("~")
        )
        if not folder:
            return
        # Persist the folder into the project (if one is open) so its wavs
        # reappear on reopen; without a project they load for this session.
        if self._project is not None and folder not in self._project.source_folders:
            self._project.source_folders.append(folder)
            try:
                self._project.save()
            except Exception:
                pass
            self._update_source_folders_label()
        wavs = _list_wavs_in_folder(folder)
        added = self._append_audio_paths(wavs)  # folder persists via source_folders
        self.status_bar.showMessage(f"Added folder: {folder} (+{added} new wavs)")

    def _append_audio_paths(self, paths, persist_files: bool = False) -> int:
        """Add new on-disk wav paths to the session list, load the current
        file, and (optionally) persist them into the project's ``audio_files``.
        Returns the count actually added."""
        existing = set(self.audio_files)
        to_add = [p for p in paths
                  if p and p not in existing and os.path.isfile(p)]
        if not to_add:
            return 0
        self.audio_files.extend(to_add)
        if persist_files and self._project is not None:
            have = set(self._project.audio_files)
            self._project.audio_files.extend(
                p for p in to_add if p not in have
            )
            try:
                self._project.save()
            except Exception:
                pass
        if self.current_file_idx >= len(self.audio_files):
            self.current_file_idx = 0
        self._refresh_file_list()
        self.file_list.blockSignals(True)
        self.file_list.setCurrentRow(self.current_file_idx)
        self.file_list.blockSignals(False)
        self._load_current_file()
        self._update_project_state()
        return len(to_add)

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
        self._offer_training_store_migration()
        self._refresh_deploy_models()
        self._rescan_project_wavs()
        self._update_source_folders_label()
        self._update_model_info_label()
        self._update_train_button_count()
        self.status_bar.showMessage(f"Opened project: {cfg.project_dir}")

    def _offer_training_store_migration(self):
        """If the project's training examples are still legacy PNG/JSON triplets,
        offer a one-time consolidation into training_data.h5 (originals archived
        to legacy_pre_h5/)."""
        if self._project is None:
            return
        try:
            from fnt.usv.usv_detector.mad_examples import (
                has_legacy_examples, migrate_legacy_to_h5)
            ddir = self._project.training_data_dir
            if not has_legacy_examples(ddir):
                return
            reply = QMessageBox.question(
                self, "Upgrade training data?",
                "This project's training examples use the older per-file "
                "format. Upgrade them to the consolidated training_data.h5 "
                "store now?\n\nThe originals are archived to a 'legacy_pre_h5' "
                "folder. Strongly recommended — older files are not guaranteed "
                "to work with this version.",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes)
            if reply != QMessageBox.Yes:
                return
            n = migrate_legacy_to_h5(ddir)
            QMessageBox.information(
                self, "Upgrade complete",
                f"Migrated {n} training example(s) to training_data.h5.")
        except Exception as e:
            self.status_bar.showMessage(f"Training-store migration failed: {e}")

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
        self.lbl_project_name.setText("No project (files load for this session)")
        self.lbl_project_name.setStyleSheet("color: #999999; font-size: 10px;")
        self.lbl_source_folders.setText("No source folders")
        # Add Folder / Add Files stay enabled — loading audio doesn't require
        # a project.
        self.act_close_project.setEnabled(False)
        self.act_run_training.setEnabled(False)
        self.act_run_inference.setEnabled(False)
        self.act_load_pred.setEnabled(False)
        self.act_clear_pred.setEnabled(False)
        self.btn_train.setEnabled(False)
        self.btn_train.setText("Train")
        self.btn_infer_run.setEnabled(False)
        self.spectrogram.mask = None
        self.spectrogram.set_audio_data(None, None)
        self.waveform_overview.set_audio_data(None, None)
        self._clear_predictions()
        self._clear_deploy_files()
        self._refresh_deploy_models()
        self._update_project_state()
        self._update_paint_buttons_enabled()
        self._update_playback_buttons_enabled()
        self._update_model_info_label()

    # ==================================================================
    # File management
    # ==================================================================
    def _add_audio_files(self):
        files, _ = QFileDialog.getOpenFileNames(
            self, "Select Audio Files", os.path.expanduser("~"),
            "WAV Files (*.wav *.WAV);;All Files (*.*)"
        )
        if not files:
            return
        added = self._append_audio_paths(files, persist_files=True)
        dup = len(files) - added
        msg = f"Added {added} file(s)"
        if dup:
            msg += f", skipped {dup} already loaded"
        self.status_bar.showMessage(msg)

    def _rescan_project_wavs(self) -> int:
        if self._project is None:
            return 0
        wavs: List[str] = []
        for folder in self._project.source_folders:
            wavs.extend(_list_wavs_in_folder(folder))
        # Individually added files persisted on the project (still on disk).
        for f in self._project.audio_files:
            if os.path.isfile(f):
                wavs.append(f)
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
        self._update_file_list_counts()

    def _update_file_list_counts(self):
        """Append (confirmed / total) detection counts to each file list item."""
        if not hasattr(self, 'file_list'):
            return
        confirmed_counts: Dict[str, int] = {}
        if self._project is not None:
            try:
                from fnt.usv.usv_detector.mad_examples import iter_examples
                for ex in iter_examples(self._project.training_data_dir):
                    wav = os.path.basename(
                        str(ex['meta'].get('source_wav', '')))
                    confirmed_counts[wav] = confirmed_counts.get(wav, 0) + 1
            except Exception:
                pass
        # Also count from per-wav h5 siblings (project-free labels).
        try:
            from fnt.usv.usv_detector.fnt_mask_store import (
                masks_sibling_path, td_count,
            )
            for fp in self.audio_files:
                h5 = masks_sibling_path(fp)
                n = td_count(h5)
                if n > 0:
                    bn = os.path.basename(fp)
                    confirmed_counts[bn] = confirmed_counts.get(bn, 0) + n
        except Exception:
            pass
        pred_counts: Dict[str, int] = {}
        try:
            from fnt.usv.usv_detector.mad_inference import read_blob_csv
            for fp in self.audio_files:
                csv_path = pred_csv_sibling_path(fp)
                if not os.path.isfile(csv_path):
                    continue
                rows = read_blob_csv(csv_path)
                n_pending = sum(1 for r in rows
                                if r.get('status') not in
                                ('accepted', 'rejected'))
                if n_pending > 0:
                    pred_counts[os.path.basename(fp)] = n_pending
        except Exception:
            pass
        self.file_list.blockSignals(True)
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            fp = item.data(Qt.UserRole)
            bn = os.path.basename(fp)
            n_conf = confirmed_counts.get(bn, 0)
            n_pred = pred_counts.get(bn, 0)
            total = n_conf + n_pred
            if total > 0:
                item.setText(f"{bn}  ({n_conf}/{total})")
            else:
                item.setText(bn)
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
            self.lbl_data_summary.setText("No files loaded — add files or a folder")
        else:
            self.lbl_data_summary.setText(f"{n} file(s) loaded")

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
            self._log(f"Open file {self.current_file_idx + 1}/"
                      f"{len(self.audio_files)}: {os.path.basename(filepath)}")
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
        this file from the saved example store and/or per-wav h5 sibling."""
        if self.audio_data is None or self.sample_rate is None:
            return
        sp = self._spec_params()
        self.spectrogram.init_mask(
            audio_len=len(self.audio_data),
            sample_rate=self.sample_rate,
            nperseg=sp['nperseg'], noverlap=sp['noverlap'], nfft=sp['nfft'],
        )
        from fnt.usv.usv_detector.mad_examples import iter_file_annotations
        wav_path = self.audio_files[self.current_file_idx]
        wav_name = os.path.basename(wav_path)
        grid = (self.spectrogram.n_freq_bins, self.spectrogram.n_time_frames)
        anns = []
        # Load from project training store if available.
        if self._project is not None:
            try:
                anns = list(iter_file_annotations(
                    self._project.training_data_dir, wav_name, grid))
            except Exception:
                anns = []
        # Also load from per-wav h5 sibling (project-free labels).
        try:
            anns.extend(self._load_sibling_h5_annotations(wav_path, grid))
        except Exception:
            pass
        # Deduplicate by id.
        seen_ids = set()
        unique = []
        for a in anns:
            aid = a.get('id')
            if aid and aid in seen_ids:
                continue
            if aid:
                seen_ids.add(aid)
            unique.append(a)
        anns = unique
        self.spectrogram.set_annotations(anns)
        n_pred = self._load_predictions_as_annotations() or 0
        if anns or n_pred:
            parts = []
            if anns:
                parts.append(f"{len(anns)} confirmed")
            if n_pred:
                parts.append(f"{n_pred} prediction(s)")
            self.lbl_mask_status.setText(
                f"Loaded {', '.join(parts)} for {wav_name}"
            )
        else:
            self.lbl_mask_status.setText(
                "No labels yet — paint or SAM calls, then Enter to confirm"
            )
        self._refresh_annotation_list()

    def _load_sibling_h5_annotations(self, wav_path, grid):
        """Load confirmed examples stored in the per-wav ``_FNT_masks.h5``
        (project-free labels). Returns a list of annotation dicts."""
        from fnt.usv.usv_detector.fnt_mask_store import (
            masks_sibling_path, td_iter_examples,
        )
        h5_path = masks_sibling_path(wav_path)
        if not os.path.isfile(h5_path):
            return []
        n_freq, n_time = grid
        wav_name = os.path.basename(wav_path)
        out = []
        for ex in td_iter_examples(h5_path):
            meta = ex['meta']
            if meta.get('source_wav') and meta['source_wav'] != wav_name:
                continue
            m = ex['mask'] > 0
            if not m.any():
                continue
            t_off = int(meta.get('patch_t_off') or 0)
            f_off = int(meta.get('patch_f_off') or 0)
            fs = np.where(m.any(axis=1))[0]
            ts = np.where(m.any(axis=0))[0]
            lf0, lf1 = int(fs[0]), int(fs[-1]) + 1
            lt0, lt1 = int(ts[0]), int(ts[-1]) + 1
            f0 = max(0, f_off + lf0)
            f1 = min(n_freq, f_off + lf1)
            t0 = max(0, t_off + lt0)
            t1 = min(n_time, t_off + lt1)
            if f1 <= f0 or t1 <= t0:
                continue
            local = m[lf0:lf0 + (f1 - f0), lt0:lt0 + (t1 - t0)]
            out.append({
                'id': meta.get('id', ''),
                'category': meta.get('class', ''),
                'f0': f0, 'f1': f1, 't0': t0, 't1': t1,
                'mask': np.ascontiguousarray(local),
            })
        return out

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
                    self.btn_sam, self.btn_undo):
            btn.setEnabled(has_audio)
        self.spin_brush_radius.setEnabled(has_audio)
        if not has_audio:
            self.btn_paint.setChecked(False)
            self.btn_erase.setChecked(False)
            self.btn_sam.setChecked(False)
            self.spectrogram.set_paint_mode(None)
            self.lbl_mask_status.setText("No mask")

    def _show_labeling_instructions(self):
        """Show labeling best-practices once per user (QSettings-gated)."""
        key = "mad/labeling_instructions_shown"
        if self._settings.value(key, False, type=bool):
            return
        self._settings.setValue(key, True)
        QMessageBox.information(
            self, "Labeling Tips",
            "<b>Tips for effective labeling:</b><br><br>"
            "1. <b>Label all calls in the same time window.</b> When you label "
            "a call, the spectrogram patch around it is used for training. Any "
            "unlabeled call in the same time window will be treated as negative "
            "(non-call) by the model, which can confuse training.<br><br>"
            "2. <b>Use SAM (G) for fast labeling.</b> Click each call to "
            "generate a mask, then press Enter to confirm them all at once.<br><br>"
            "3. <b>Use the brush (B) for fine corrections</b> and the eraser "
            "(E) to remove false strokes.<br><br>"
            "4. <b>Human-in-the-loop:</b> Label a few calls → Train → Run "
            "inference → Accept/reject predictions → Train again. Each cycle "
            "improves the model."
        )

    def _on_brush_clicked(self, checked: bool):
        if checked:
            self._show_labeling_instructions()
            self.btn_erase.setChecked(False)
            self.btn_sam.setChecked(False)
            self.spectrogram.set_paint_mode('brush')
            self.status_bar.showMessage(
                "Brush mode — left-click + drag over target pixels"
            )
            self._log("Tool: Brush ON")
        else:
            self.spectrogram.set_paint_mode(None)
            self.status_bar.showMessage("Paint mode off")
            self._log("Tool: brush OFF")

    def _on_eraser_clicked(self, checked: bool):
        if checked:
            self.btn_paint.setChecked(False)
            self.btn_sam.setChecked(False)
            self.spectrogram.set_paint_mode('eraser')
            self.status_bar.showMessage(
                "Eraser mode — left-click + drag to clear painted pixels"
            )
            self._log("Tool: Eraser ON")
        else:
            self.spectrogram.set_paint_mode(None)
            self.status_bar.showMessage("Paint mode off")
            self._log("Tool: eraser OFF")

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
            self._log("Tool: SAM OFF")
            return
        if not self._ensure_sam_model():
            self.btn_sam.setChecked(False)
            return
        self._show_labeling_instructions()
        self.btn_paint.setChecked(False)
        self.btn_erase.setChecked(False)
        self.spectrogram.set_paint_mode('sam')
        self._log("Tool: SAM ON")
        self.status_bar.showMessage(
            "SAM mode — click each call (yellow), then Enter to assign a class; "
            "Esc clears, U undoes"
        )

    def _change_sam_model(self):
        """Pick/switch the SAM2 checkpoint, discarding any loaded model."""
        # Discard the current model so a fresh one is loaded.
        self._sam_segmenter = None
        self._sam_ready = False
        self._sam_img_sig = None
        self._sam_last_t_off = None
        self._sam_ckpt = None
        self._sam_cfg_name = None
        if not self._pick_sam_checkpoint():
            return
        self.status_bar.showMessage(
            f"Loading SAM2 model: {os.path.basename(self._sam_ckpt)}…")
        self._ensure_sam_model()

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
            return
        if not self._sam_ready:
            self.status_bar.showMessage(
                "SAM model still loading — your click will run once it's ready…")
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
        self.status_bar.showMessage("SAM predicting…")
        self._sam_predict_worker.start()

    def _on_sam_predicted(self, mask, t_off: int):
        added = False
        if self.spectrogram.paint_mode == 'sam' and mask is not None:
            added = self.spectrogram.add_sam_component(mask, t_off)
        if added:
            self.status_bar.showMessage(
                "Mask added (yellow) — keep clicking, then Enter to assign a class")
        else:
            self.status_bar.showMessage(
                "SAM found no mask at that click — try clicking on the call")
        if self._sam_predict_pending:
            self._sam_predict_pending = False
            self._run_sam_predict()

    def _confirm_pending(self):
        """Enter — confirm the pending mask (manual or SAM): ask for a class,
        save it as a self-contained training example, then merge it into the
        confirmed buffer."""
        sg = self.spectrogram
        if sg.is_editing():
            sg.finish_edit()
            return
        comps = sg.pending_components()
        if not comps:
            return
        if self.audio_data is None or not self.audio_files:
            return
        # One-time info when saving labels without a project.
        if self._project is None:
            self._show_first_label_info()
        if self._project is not None:
            classes = list(self._project.classes) or ["USV"]
            last = self._project.last_class or classes[0]
        else:
            classes = list(self._session_classes) if self._session_classes else ["USV"]
            last = self._session_last_class or classes[0]
        if last not in classes:
            classes = [last] + classes
        name, ok = QInputDialog.getItem(
            self, "Confirm calls",
            f"Class for {len(comps)} mask(s) (Enter to reuse):",
            classes, classes.index(last), True,
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        if self._project is not None:
            if name not in self._project.classes:
                self._project.classes.append(name)
            self._project.last_class = name
        else:
            if name not in self._session_classes:
                self._session_classes.append(name)
            self._session_last_class = name
        saved = 0
        for comp in comps:
            try:
                ex_id = self._save_component_example(name, comp)
            except Exception:
                continue
            f0, f1, t0, t1, local = comp
            sg.add_annotation({
                'id': ex_id, 'category': name,
                'f0': f0, 'f1': f1, 't0': t0, 't1': t1,
                'mask': local.astype(bool),
            })
            saved += 1
        if self._project is not None:
            try:
                self._project.save()
            except Exception:
                pass
        sg.clear_pending()
        self._refresh_annotation_list()
        self.lbl_mask_status.setText(f"Confirmed {saved} call(s) — '{name}'")
        self.status_bar.showMessage(
            f"Saved {saved} example(s) (class '{name}') — label the next batch"
        )
        self._log(f"Confirmed {saved} call(s) as '{name}' (Enter)")

    def _show_first_label_info(self):
        """One-time dialog explaining that confirming labels creates .h5 files
        next to the audio. Shown once per user (QSettings-gated)."""
        key = "mad/first_label_info_shown"
        if self._settings.value(key, False, type=bool):
            return
        self._settings.setValue(key, True)
        QMessageBox.information(
            self, "Saving Labels",
            "Confirmed labels are saved as <b>.h5 files</b> next to your "
            "audio files (one per .wav).<br><br>"
            "These files are automatically reloaded when you open the same "
            "audio again. No project is needed for labeling.<br><br>"
            "When you're ready to <b>train a model</b>, you'll be asked to "
            "choose a project directory."
        )

    def _save_component_example(self, class_name: str, comp) -> str:
        """Save one connected-component mask as a self-contained example.
        ``comp`` is ``(f0, f1, t0, t1, local_bool)``. Returns the example id.

        With a project: saves to ``training_data.h5`` (consolidated store).
        Without: saves to the per-wav ``_FNT_masks.h5`` sibling so labels
        survive across sessions without a project.
        """
        from datetime import datetime
        from scipy import signal as _signal
        from fnt.usv.usv_detector.mad_dataset import spec_to_image

        sg = self.spectrogram
        cfg = self._project
        sp = self._spec_params()
        f0, f1, t0, t1, local = comp
        hop = sg.hop
        nperseg = sp['nperseg']
        nfft = sp['nfft']
        noverlap_val = sp['noverlap']
        n_time, n_freq = sg.n_time_frames, sg.n_freq_bins
        sr = self.sample_rate
        margin = 64
        pt0 = max(0, t0 - margin)
        pt1 = min(n_time, t1 + margin)

        start_sample = pt0 * hop
        end_sample = min(len(self.audio_data), (pt1 - 1) * hop + nperseg)
        segment = self.audio_data[start_sample:end_sample]
        if len(segment) < nperseg:
            raise RuntimeError("call window too short to compute a patch")
        noverlap_safe = min(noverlap_val, nperseg - 1)
        _f, _t, Sxx = _signal.spectrogram(
            segment, fs=sr, nperseg=nperseg, noverlap=noverlap_safe,
            nfft=nfft, window='hann',
        )
        spec_db = 10.0 * np.log10(Sxx + 1e-10)
        spec_patch = spec_to_image(spec_db, sp['db_min'], sp['db_max'])
        W = spec_patch.shape[1]

        mask_patch = np.zeros((n_freq, W), dtype=np.uint8)
        lt0 = t0 - pt0
        tw = min(t1 - t0, W - lt0)
        if tw > 0:
            mask_patch[f0:f1, lt0:lt0 + tw] = local[:, :tw].astype(np.uint8)

        df = (sr / 2.0) / (nfft // 2)
        dt = hop / float(sr)
        wav_name = os.path.basename(self.audio_files[self.current_file_idx])
        wav_path = self.audio_files[self.current_file_idx]
        meta = {
            'class': class_name,
            'source_wav': wav_name,
            'patch_t_off': int(pt0), 'patch_f_off': 0,
            't_start_s': round(t0 * dt, 6), 't_stop_s': round(t1 * dt, 6),
            'patch_t0_s': round(pt0 * dt, 6), 'patch_t1_s': round(pt1 * dt, 6),
            'f_low_hz': round(f0 * df, 2), 'f_high_hz': round(f1 * df, 2),
            'patch_t_frames': int(W), 'f_bins': int(n_freq),
            'nperseg': nperseg, 'noverlap': noverlap_val, 'nfft': nfft,
            'sample_rate': int(sr),
            'db_min': sp['db_min'], 'db_max': sp['db_max'],
            'created': datetime.now().isoformat(timespec='seconds'),
        }
        if cfg is not None:
            from fnt.usv.usv_detector.mad_examples import save_example
            ex_id = save_example(cfg.training_data_dir, spec_patch, mask_patch,
                                 meta)
            self._ensure_wav_in_project()
        else:
            import uuid
            from fnt.usv.usv_detector.fnt_mask_store import (
                masks_sibling_path, set_grid_attrs, td_save_example,
            )
            ex_id = f"{os.path.splitext(wav_name)[0]}_{uuid.uuid4().hex[:10]}"
            h5_path = masks_sibling_path(wav_path)
            set_grid_attrs(h5_path, sample_rate=sr, nperseg=nperseg,
                           noverlap=noverlap_val, nfft=nfft,
                           n_freq_bins=n_freq, n_time_frames=n_time)
            td_save_example(h5_path, spec_patch, mask_patch, meta, ex_id)
        return ex_id

    def _ensure_wav_in_project(self):
        """Copy the current WAV into the project's recordings/ folder if it
        is not already there."""
        if self._project is None or not self.audio_files:
            return
        import shutil
        wav_path = self.audio_files[self.current_file_idx]
        rdir = self._project.recordings_dir
        dest = os.path.join(rdir, os.path.basename(wav_path))
        if os.path.isfile(dest):
            return
        try:
            os.makedirs(rdir, exist_ok=True)
            shutil.copy2(wav_path, dest)
        except Exception:
            pass

    def _clear_pending(self):
        sg = self.spectrogram
        if sg.is_editing():
            sg.cancel_edit()
            self.status_bar.showMessage("Shape edit cancelled")
            return
        if sg.has_pending() or sg.has_sam_prompts():
            sg.clear_pending()
            sg.clear_sam_prompts()
            self.status_bar.showMessage("Pending masks cleared")

    # --- undo / context menu / shape-edit (Phases D-F) ----------------
    def _undo_last(self):
        sg = self.spectrogram
        if sg.undo_pending():
            self.status_bar.showMessage("Removed last pending mask")
            self._log("Undo: removed last pending mask")
            return
        if sg.annotations:
            ann = sg.remove_annotation(len(sg.annotations) - 1)
            if ann and self._project is not None:
                from fnt.usv.usv_detector.mad_examples import delete_example
                try:
                    delete_example(self._project.training_data_dir, ann['id'])
                except Exception:
                    pass
            self._refresh_annotation_list()
            self.status_bar.showMessage("Removed last annotation")
            self._log("Undo: removed last confirmed detection")

    def _on_annotation_context_menu(self, ann_idx, global_pos):
        from PyQt5.QtWidgets import QMenu
        sg = self.spectrogram
        if not (0 <= ann_idx < len(sg.annotations)):
            return
        menu = QMenu(self)
        a_class = menu.addAction("Edit class…")
        a_shape = menu.addAction("Edit shape")
        menu.addSeparator()
        a_del = menu.addAction("Delete")
        act = menu.exec_(global_pos)
        if act == a_class:
            self._edit_annotation_class(ann_idx)
        elif act == a_shape:
            sg.start_edit(ann_idx)
            self.status_bar.showMessage(
                "Editing shape — drag vertices, right-click to add, "
                "Enter to finish, Esc to cancel"
            )
        elif act == a_del:
            self._delete_annotation(ann_idx)

    def _edit_annotation_class(self, ann_idx):
        sg = self.spectrogram
        if not (0 <= ann_idx < len(sg.annotations)):
            return
        ann = sg.annotations[ann_idx]
        if self._project is not None:
            classes = list(self._project.classes) or ["USV"]
            cur = ann.get('category') or self._project.last_class or classes[0]
        else:
            classes = list(self._session_classes) if self._session_classes else ["USV"]
            cur = ann.get('category') or self._session_last_class or classes[0]
        if cur not in classes:
            classes = [cur] + classes
        name, ok = QInputDialog.getItem(
            self, "Edit class", "Class:", classes, classes.index(cur), True)
        if not ok or not name.strip():
            return
        name = name.strip()
        ann['category'] = name
        if self._project is not None:
            if name not in self._project.classes:
                self._project.classes.append(name)
            self._project.last_class = name
            from fnt.usv.usv_detector.mad_examples import update_example_class
            try:
                update_example_class(self._project.training_data_dir,
                                     ann['id'], name)
                self._project.save()
            except Exception:
                pass
        else:
            if name not in self._session_classes:
                self._session_classes.append(name)
            self._session_last_class = name
            try:
                from fnt.usv.usv_detector.fnt_mask_store import (
                    masks_sibling_path, td_update_class,
                )
                h5 = masks_sibling_path(
                    self.audio_files[self.current_file_idx])
                td_update_class(h5, ann['id'], name)
            except Exception:
                pass
        self._refresh_annotation_list()
        sg.update()

    def _on_annotation_clicked(self, ai):
        """A mask was left-clicked in the preview (no tool engaged): select it
        in the list and, if it's a pending prediction, point review at it."""
        self._box_sel_ids = []   # single click clears any box multi-selection
        sg = self.spectrogram
        if not (0 <= ai < len(sg.annotations)):
            return
        ann = sg.annotations[ai]
        self._select_list_row_for_id(ann.get('id'))
        if ann.get('status') == 'prediction':
            preds = self._pred_indices()
            if ai in preds:
                self._pred_review_idx = preds.index(ai)
                self._update_pred_review_widgets()

    def _on_box_selection(self, indices):
        """Rubber-band drag selected a set of detections — remember their ids
        (stable across list refresh) so A/R/D act on the whole group."""
        sg = self.spectrogram
        self._box_sel_ids = [sg.annotations[i].get('id')
                             for i in indices if 0 <= i < len(sg.annotations)]
        n = len(self._box_sel_ids)
        if n:
            self.status_bar.showMessage(
                f"{n} detection(s) selected — A=accept, R=reject, D=delete all")
            self._log(f"Box-selected {n} detection(s)")
        else:
            self.status_bar.showMessage("Selection cleared")

    def _box_sel_indices(self):
        """Resolve the stored box-selection ids to current annotation indices."""
        sg = self.spectrogram
        id_to_idx = {a.get('id'): i for i, a in enumerate(sg.annotations)}
        return [id_to_idx[x] for x in self._box_sel_ids if x in id_to_idx]

    def _apply_to_box_selection(self, action):
        """Apply accept/reject/delete to every box-selected detection at once.
        Returns True if a selection existed and was handled."""
        idxs = self._box_sel_indices()
        if not idxs:
            return False
        sg = self.spectrogram
        n_done = 0
        # Process high→low so earlier removals don't shift later indices.
        for i in sorted(idxs, reverse=True):
            if not (0 <= i < len(sg.annotations)):
                continue
            ann = sg.annotations[i]
            is_pred = ann.get('status') == 'prediction'
            if action == 'accept':
                if is_pred:
                    if self._review_mode == 'deploy':
                        self._write_deploy_status(ann, 'accepted')
                        sg.remove_annotation(i)
                    else:
                        self._accept_prediction(i)
                    n_done += 1
            elif action == 'reject':
                if is_pred:
                    if self._review_mode == 'deploy':
                        self._write_deploy_status(ann, 'rejected')
                    sg.remove_annotation(i)
                    n_done += 1
            elif action == 'delete':
                if is_pred and self._review_mode == 'deploy':
                    self._write_deploy_status(ann, 'rejected')
                self._delete_annotation(i)
                n_done += 1
        self._box_sel_ids = []
        sg._selected_set = set()
        sg._selected_ann_idx = None
        sg._rebuild_confirmed_mask()
        sg.update()
        self._pred_review_idx = None
        self._refresh_annotation_list()
        self._update_pred_review_widgets()
        self._log(f"{action.capitalize()} {n_done} box-selected detection(s)")
        self.status_bar.showMessage(f"{action.capitalize()}ed {n_done} detection(s)")
        return True

    def _delete_selected_annotation(self):
        """Delete the selected detection (Delete button / D key).

        Like Reject, but it does NOT auto-advance: the view doesn't move and no
        detection stays selected afterwards. For a deploy-mode prediction it
        still records 'rejected' in the output CSV (matching Reject)."""
        if self._focus_is_edit():
            return
        if self._apply_to_box_selection('delete'):
            return
        sel = self.spectrogram._selected_ann_idx
        if sel is None or not (0 <= sel < len(self.spectrogram.annotations)):
            self.status_bar.showMessage("No detection selected — click one first.")
            return
        ann = self.spectrogram.annotations[sel]
        is_pred = ann.get('status') == 'prediction'
        if is_pred and self._review_mode == 'deploy':
            self._write_deploy_status(ann, 'rejected')
        self.spectrogram._selected_ann_idx = None
        self._delete_annotation(sel)
        self._log("Deleted prediction" if is_pred else "Deleted confirmed detection")
        self.status_bar.showMessage("Deleted detection (D)")

    def _delete_annotation(self, ann_idx):
        sg = self.spectrogram
        ann = sg.remove_annotation(ann_idx)
        if ann:
            aid = ann.get('id', '')
            if self._project is not None:
                from fnt.usv.usv_detector.mad_examples import delete_example
                try:
                    delete_example(self._project.training_data_dir, aid)
                except Exception:
                    pass
            if self.audio_files and 0 <= self.current_file_idx < len(self.audio_files):
                try:
                    from fnt.usv.usv_detector.fnt_mask_store import (
                        masks_sibling_path, td_delete,
                    )
                    h5 = masks_sibling_path(
                        self.audio_files[self.current_file_idx])
                    td_delete(h5, aid)
                except Exception:
                    pass
        self._refresh_annotation_list()

    def _on_annotation_edit_finished(self, ann_idx):
        sg = self.spectrogram
        if (self.audio_data is None or not self.audio_files or
                not (0 <= ann_idx < len(sg.annotations))):
            return
        ann = sg.annotations[ann_idx]
        old_id = ann.get('id')
        comp = (ann['f0'], ann['f1'], ann['t0'], ann['t1'], ann['mask'])
        try:
            new_id = self._save_component_example(
                ann.get('category') or 'USV', comp)
        except Exception:
            return
        if old_id and old_id != new_id:
            if self._project is not None:
                from fnt.usv.usv_detector.mad_examples import delete_example
                try:
                    delete_example(self._project.training_data_dir, old_id)
                except Exception:
                    pass
            try:
                from fnt.usv.usv_detector.fnt_mask_store import (
                    masks_sibling_path, td_delete,
                )
                h5 = masks_sibling_path(
                    self.audio_files[self.current_file_idx])
                td_delete(h5, old_id)
            except Exception:
                pass
        ann['id'] = new_id
        if self._project is not None:
            try:
                self._project.save()
            except Exception:
                pass
        self._refresh_annotation_list()
        self.status_bar.showMessage("Updated mask shape")

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
        self._log("Cleared pending mask(s)")

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
        new_window = max(0.1, min(300.0, self.spin_view_window.value() * factor))
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
        self.spin_view_window.setValue(min(300.0, self.spin_view_window.value() * 2))

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
            self.btn_play.setText("Stop")
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
            self._log(
                f"Train DONE — best_val_loss="
                f"{summary.get('best_val_loss', 0):.4f}, "
                f"epochs={summary.get('n_epochs_run', '?')}, "
                f"model={Path(str(summary.get('model_path', ''))).parent.name}"
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
                    # Surface the freshly trained model in the deploy dropdown.
                    self._refresh_deploy_models()

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
            save_blob_csv=True,
            preserve_labels=True,
            training_data_dir=(self._project.training_data_dir
                               if self._project else ""),
        )
        self.train_infer_panel.setVisible(True)
        self.train_infer_panel.start_run()
        self._start_inference(cfg, wavs, reporter=self.train_infer_panel)

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
            # Auto-load predictions for review.
            if self._review_mode == 'deploy' and self.deploy_files:
                # Deployment run: open the first queued file for correction.
                self._deploy_file_idx = 0
                self.deploy_list.blockSignals(True)
                self.deploy_list.setCurrentRow(0)
                self.deploy_list.blockSignals(False)
                self._load_deploy_file(self.deploy_files[0])
                self.status_bar.showMessage(
                    f"Inference done — review detections (Accept (A) / "
                    f"Reject (R) / Skip (S))")
            elif self.audio_files:
                cur = self.audio_files[self.current_file_idx]
                if cur in wav_paths:
                    n = self._load_predictions_as_annotations()
                    if n:
                        self.left_tabs.setCurrentIndex(0)
                        self.status_bar.showMessage(
                            f"Loaded {n} prediction(s) — Accept (A) / "
                            f"Reject (R) / Skip (S)")

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
    # Predictions
    # ==================================================================
    def _clear_predictions(self):
        """Drop the probability overlay and any pending prediction detections."""
        self.spectrogram.set_predicted_mask(None)
        sg = self.spectrogram
        sg.annotations = [a for a in sg.annotations
                          if a.get('status') != 'prediction']
        sg._rebuild_confirmed_mask()
        sg.update()
        self._pred_review_idx = None
        self._refresh_annotation_list()
        self._update_pred_review_widgets()

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
