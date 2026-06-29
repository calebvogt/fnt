"""MAD (Mask Audio Detector) — full shell.

PyQt5 entry point for mask-based segmentation labeling, training, and
inference. Two-tab architecture: Label & Train (build the project) and
Inference (deploy a trained model). Per-wav h5 storage, project-free
startup, CAD-style keyboard shortcuts.

Run directly:
    python fnt/usv/mad_pyqt.py
"""
from __future__ import annotations

import os
import sys
from pathlib import Path
from typing import Dict, List, Optional

# Quiet the noisy Hugging Face cache symlink warning on Windows (symlinks need
# Developer Mode/admin; caching still works fine without them). Set before any
# huggingface_hub import triggers the encoder/SAM2 download.
os.environ.setdefault("HF_HUB_DISABLE_SYMLINKS_WARNING", "1")

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
        # Separate accumulator for scroll-wheel zoom (outside paint mode), so a
        # touchpad's many small deltas step at the same measured rate as a mouse
        # wheel notch (120 units = one zoom step) rather than over-zooming.
        self._zoom_accum = 0

        # Cursor preview: screen-space last mouse pos for drawing the
        # brush-radius circle around the cursor in paint mode.
        self._cursor_pos = None
        self.setMouseTracking(True)

        self.mask_alpha = 0.45
        # Render mode: 'spec' | 'overlay' | 'mask_only'
        self.view_mode: str = 'overlay'

        # Inference loading overlay
        self._infer_loading = False
        self._infer_progress = 0.0
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
        ``(f0, f1, t0, t1, local_bool)`` tuples.

        The pending buffer spans the whole (huge) spectrogram grid, so labeling
        it directly is very slow. Instead we bound the work to just the painted
        region: the bbox comes from the stroke history (``_pending_stack``,
        already tracked for undo) — no full-grid scan — and only that small crop
        is labeled."""
        p = self._pending
        if p is None:
            return []
        from scipy import ndimage
        if self._pending_stack:
            fcat = np.concatenate([s[0] for s in self._pending_stack])
            tcat = np.concatenate([s[1] for s in self._pending_stack])
            bf0, bf1 = int(fcat.min()), int(fcat.max()) + 1
            bt0, bt1 = int(tcat.min()), int(tcat.max()) + 1
        else:
            # No stroke history (shouldn't normally happen) — fall back to a
            # full scan so behavior stays correct.
            rows = np.where(p.any(axis=1))[0]
            if rows.size == 0:
                return []
            cols = np.where(p.any(axis=0))[0]
            bf0, bf1 = int(rows[0]), int(rows[-1]) + 1
            bt0, bt1 = int(cols[0]), int(cols[-1]) + 1
        sub = p[bf0:bf1, bt0:bt1] > 0
        if not sub.any():
            return []
        lbl, n = ndimage.label(sub)
        out = []
        for k, sl in enumerate(ndimage.find_objects(lbl), start=1):
            if sl is None:
                continue
            fs, ts = sl  # slices within the crop
            cc = lbl[fs, ts] == k
            out.append((bf0 + fs.start, bf0 + fs.stop,
                        bt0 + ts.start, bt0 + ts.stop,
                        np.ascontiguousarray(cc)))
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
        # Clear only this annotation's region instead of zeroing the whole
        # (multi-hundred-MB) grid and re-stamping every annotation. Then re-stamp
        # any other annotation overlapping the cleared box so shared pixels stay.
        if self.mask is not None and ann:
            f0, f1, t0, t1 = ann['f0'], ann['f1'], ann['t0'], ann['t1']
            self.mask[f0:f1, t0:t1] = 0
            for other in self.annotations:
                of0, of1, ot0, ot1 = (other['f0'], other['f1'],
                                      other['t0'], other['t1'])
                if of1 > f0 and of0 < f1 and ot1 > t0 and ot0 < t1:
                    self.mask[of0:of1, ot0:ot1][other['mask']] = MASK_POSITIVE
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
        # Outside paint mode: scroll zooms the spectrogram, centered on the
        # cursor. Deltas accumulate to one notch (120) per zoom step so a
        # trackpad steps at the same measured rate as a mouse wheel instead of
        # over-zooming (↑/↓ keys still zoom; ←/→ pan).
        if self.total_duration <= 0:
            event.accept()
            return
        spec_rect = self._get_spec_rect()
        if not spec_rect.contains(event.pos()):
            event.accept()
            return
        self._zoom_accum += event.angleDelta().y()
        center_time = self._x_to_time(event.pos().x(), spec_rect)
        while self._zoom_accum >= 120:      # scroll up → zoom in
            self._zoom_accum -= 120
            self.zoom_requested.emit(1.0 / 1.25, center_time)
        while self._zoom_accum <= -120:     # scroll down → zoom out
            self._zoom_accum += 120
            self.zoom_requested.emit(1.25, center_time)
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

        # Zoom-out guard: when the view spans far more time frames than the
        # spectrogram has on-screen pixels, building the overlay image at full
        # frame resolution wastes hundreds of MB (and a flipud copy) on every
        # repaint, then throws it away in the down-scale — the cause of the
        # freeze past ~15 s. Instead we max-pool the per-pixel maps down to the
        # display pixel width (max-pool, not stride-sample, so even 1-frame
        # calls survive) and let Qt scale the small image back up. When zoomed
        # in (w <= pixels) t_stride is 1 and this is a no-op.
        spec_w_px = max(1, int(spec_rect.width()))
        t_stride = max(1, w // spec_w_px)
        overlays_visible = t_stride <= 4   # per-call outlines/labels readable?
        if t_stride > 1:
            _seg = np.arange(0, w, t_stride)
            w_img = int(_seg.shape[0])

            def _ds(a):
                return np.maximum.reduceat(a, _seg, axis=1)
        else:
            w_img = w

            def _ds(a):
                return a

        pend_view = (self._pending[f_start:f_end, t_start:t_end] > 0
                     if self._pending is not None else None)
        # Build a per-pixel type map: 0=none, 1=confirmed, 2=prediction,
        # 3=rejected (a recorded human "no" — kept visible, shaded red).
        type_map = np.zeros((h, w), dtype=np.uint8)
        view_mask = self.mask[f_start:f_end, t_start:t_end]
        type_map[view_mask == MASK_POSITIVE] = 1
        for ann in self.annotations:
            st = ann.get('status')
            code = 2 if st == 'prediction' else (3 if st == 'rejected' else 0)
            if not code:
                continue
            af0 = max(ann['f0'], f_start); af1 = min(ann['f1'], f_end)
            at0 = max(ann['t0'], t_start); at1 = min(ann['t1'], t_end)
            if af1 > af0 and at1 > at0:
                lf0, lt0 = af0 - f_start, at0 - t_start
                mf0, mt0 = af0 - ann['f0'], at0 - ann['t0']
                mh, mw = af1 - af0, at1 - at0
                sub = ann['mask'][mf0:mf0+mh, mt0:mt0+mw]
                type_map[lf0:lf0+mh, lt0:lt0+mw][sub > 0] = code

        # Down-pool the maps to render width, then colour the small image.
        tm = _ds(type_map)
        pv = (_ds(pend_view.astype(np.uint8)).astype(bool)
              if pend_view is not None else None)
        rgba = np.zeros((h, w_img, 4), dtype=np.uint8)

        if self.view_mode == 'mask_only':
            rgba[:] = (32, 32, 32, 255)
            rgba[tm == 1] = (40, 200, 90, 255)
            rgba[tm == 2] = (255, 230, 90, 255)
            rgba[tm == 3] = (255, 70, 70, 255)
            if pv is not None:
                rgba[pv] = (255, 230, 90, 255)
        elif self.view_mode == 'overlay':
            pos_alpha = int(self.mask_alpha * 255)
            rgba[tm == 1] = (40, 200, 90, pos_alpha)
            rgba[tm == 2] = (255, 230, 90, max(60, pos_alpha - 50))
            rgba[tm == 3] = (255, 70, 70, max(60, pos_alpha - 50))
            if pv is not None:
                rgba[pv] = (255, 230, 90, max(60, pos_alpha - 50))
        # else: 'spec' — leave rgba all-zero so only the spec shows.

        # Predicted blob mask shading (cyan, low alpha) if present.
        if (self.view_mode != 'spec' and self.pred_mask is not None and
                self.pred_mask.shape == self.mask.shape):
            pview = self.pred_mask[f_start:f_end, t_start:t_end]
            if pview.size > 0:
                pview = _ds(pview)
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
        qimg = QImage(rgba.data, w_img, h, 4 * w_img,
                      QImage.Format_RGBA8888).copy()
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

        # Dotted yellow outline tracing the actively-drawn (SAM/Paint) pending
        # mask — dotted distinguishes it from inference predictions (solid
        # yellow), so you can tell what you're labeling vs reviewing.
        if (HAS_CV2 and overlays_visible and self._pending is not None
                and pend_view is not None and pend_view.any()):
            cnts, _ = cv2.findContours(
                pend_view.astype(np.uint8), cv2.RETR_EXTERNAL,
                cv2.CHAIN_APPROX_SIMPLE,
            )
            pen = QPen(QColor(255, 225, 60))
            pen.setWidth(2)
            pen.setStyle(Qt.DotLine)
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

        # Accepted-annotation outlines + class labels above each mask. Skipped
        # when zoomed out so far each call is only a pixel or two wide — the
        # outlines/labels would be illegible clutter and (pre-fix) very slow.
        if (self.view_mode != 'spec' and self.annotations and HAS_CV2
                and overlays_visible):
            for ai, ann in enumerate(self.annotations):
                if ai == self._editing_ann_idx:
                    continue
                if (ann['t1'] <= t_start or ann['t0'] >= t_end or
                        ann['f1'] <= f_start or ann['f0'] >= f_end):
                    continue
                is_pred = ann.get('status') == 'prediction'
                is_rej = ann.get('status') == 'rejected'
                is_sel = (ai == self._selected_ann_idx)
                if is_sel:
                    # Selected detection: mask outline AND label both white so
                    # tiny detections are easy to spot.
                    outline_color = QColor(255, 255, 255)
                    label_color = QColor(255, 255, 255)
                elif is_rej:
                    outline_color = QColor(255, 80, 80)
                    label_color = QColor(255, 110, 110)
                else:
                    outline_color = (QColor(255, 225, 60) if is_pred
                                     else QColor(60, 210, 110))   # green confirmed
                    label_color = (QColor(255, 230, 100) if is_pred
                                   else QColor(130, 235, 150))
                # Thin outline around the mask contour.
                lf0 = max(ann['f0'], f_start) - f_start
                lf1 = min(ann['f1'], f_end) - f_start
                lt0 = max(ann['t0'], t_start) - t_start
                lt1 = min(ann['t1'], t_end) - t_start
                if lf1 > lf0 and lt1 > lt0:
                    # Allocate only the call's bounding box (not the whole
                    # view) — at wide zoom a full-view array per annotation was
                    # gigabytes of allocations and the main source of freeze.
                    bh, bw = lf1 - lf0, lt1 - lt0
                    view_ann = np.zeros((bh, bw), dtype=np.uint8)
                    msk = ann['mask']
                    af0 = max(ann['f0'], f_start) - ann['f0']
                    af1 = af0 + bh
                    at0 = max(ann['t0'], t_start) - ann['t0']
                    at1 = at0 + bw
                    if af1 <= msk.shape[0] and at1 <= msk.shape[1]:
                        view_ann[:, :] = msk[af0:af1, at0:at1].astype(np.uint8)
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
                            # contour coords are relative to the bbox crop
                            poly.append(QPointF(
                                _t_to_x(t_start + lt0 + int(pt[0])),
                                _f_to_y(f_start + lf0 + int(pt[1])),
                            ))
                        painter.drawPolygon(poly)
                # Class label centered horizontally over the mask, just above it.
                painter.setPen(QPen(label_color))
                f = painter.font()
                f.setPointSize(8)
                painter.setFont(f)
                label = ann.get('category', '') or '?'
                if is_rej:
                    label = "Reject"
                elif is_pred:
                    label = f"({label})"
                cx = _t_to_x((max(ann['t0'], t_start) + min(ann['t1'], t_end)) / 2.0)
                tw = painter.fontMetrics().horizontalAdvance(label)
                lx = cx - tw / 2.0
                ly = _f_to_y(min(ann['f1'], f_end)) - 3
                painter.drawText(QPointF(lx, ly), label)

        # White highlight outline on the selected annotation(s) — the single
        # click-selected one plus any rubber-band multi-selection. Drawn even
        # when zoomed out (per-call overlays hidden) so the selection stays
        # visible; bbox-sized alloc keeps it cheap.
        if self.view_mode != 'spec' and HAS_CV2 and self.annotations:
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
                    bh, bw = lf1 - lf0, lt1 - lt0
                    view_sel = np.zeros((bh, bw), dtype=np.uint8)
                    smsk = sa['mask']
                    sf0 = max(sa['f0'], f_start) - sa['f0']
                    sf1 = sf0 + bh
                    st0 = max(sa['t0'], t_start) - sa['t0']
                    st1 = st0 + bw
                    if sf1 <= smsk.shape[0] and st1 <= smsk.shape[1]:
                        view_sel[:, :] = smsk[sf0:sf1, st0:st1].astype(np.uint8)
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
                                _t_to_x(t_start + lt0 + int(pt[0])),
                                _f_to_y(f_start + lf0 + int(pt[1])),
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

        # Inference loading overlay
        if self._infer_loading:
            painter.setOpacity(0.55)
            painter.fillRect(spec_rect, QColor(0, 0, 0))
            painter.setOpacity(1.0)
            bar_h = 6
            bar_w = spec_rect.width() * 0.6
            bar_x = spec_rect.center().x() - bar_w / 2
            bar_y = spec_rect.center().y() + 14
            painter.setPen(Qt.NoPen)
            painter.setBrush(QColor(80, 80, 80))
            painter.drawRoundedRect(QRectF(bar_x, bar_y, bar_w, bar_h), 3, 3)
            fill_w = bar_w * max(0.0, min(1.0, self._infer_progress))
            if fill_w > 0:
                painter.setBrush(QColor(80, 200, 255))
                painter.drawRoundedRect(
                    QRectF(bar_x, bar_y, fill_w, bar_h), 3, 3)
            painter.setPen(QColor(255, 255, 255))
            f = painter.font()
            f.setPointSize(11)
            painter.setFont(f)
            painter.drawText(
                spec_rect, Qt.AlignCenter, "Running inference…")

        # Constant center reference line — a faint, dotted vertical line at the
        # horizontal middle of the spectrogram. Auto-advance centers the
        # selected call on this line, giving a fixed marker to read against.
        cx_mid = int(spec_rect.center().x())
        pen = QPen(QColor(255, 255, 255, 150))
        pen.setStyle(Qt.DashLine)
        pen.setWidth(2)
        painter.setPen(pen)
        painter.setBrush(Qt.NoBrush)
        painter.drawLine(cx_mid, int(spec_rect.top()),
                         cx_mid, int(spec_rect.bottom()))

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
        self.spin_patience.setValue(20)
        self.spin_patience.setSingleStep(10)
        self.spin_patience.setToolTip(
            "Stop training when validation loss fails to improve for this "
            "many consecutive epochs. 0 = disable early stopping."
        )
        form.addRow("Early-stop patience:", self.spin_patience)

        self.spin_batch = QSpinBox()
        self.spin_batch.setRange(1, 64)
        self.spin_batch.setValue(8)
        self.spin_batch.setToolTip(
            "How many spectrogram tiles the GPU processes at once. Larger = "
            "faster training but more memory; lower this if you hit out-of-"
            "memory errors. Has no effect on final accuracy."
        )
        form.addRow("Batch size:", self.spin_batch)

        self.spin_lr = QDoubleSpinBox()
        self.spin_lr.setDecimals(6)
        self.spin_lr.setRange(1e-6, 1.0)
        self.spin_lr.setSingleStep(1e-4)
        self.spin_lr.setValue(1e-3)
        self.spin_lr.setToolTip(
            "Step size for weight updates. The 1e-3 default is a safe starting "
            "point. Too high → unstable / diverging loss; too low → very slow "
            "training. Usually leave as-is."
        )
        form.addRow("Learning rate:", self.spin_lr)

        self.spin_val = QDoubleSpinBox()
        self.spin_val.setRange(0.0, 0.9)
        self.spin_val.setSingleStep(0.05)
        self.spin_val.setValue(0.20)
        self.spin_val.setToolTip(
            "Fraction of your labeled tiles held out (not trained on) to gauge "
            "generalization and drive early stopping. 0.20 = 20% held out for "
            "validation, 80% used for training."
        )
        form.addRow("Validation fraction:", self.spin_val)

        self.combo_encoder = QComboBox()
        self.combo_encoder.addItems([
            "resnet18", "resnet34", "resnet50",
            "efficientnet-b0", "mobilenet_v2",
        ])
        self.combo_encoder.setToolTip(
            "The pretrained backbone of the U-Net (the part that reads the "
            "spectrogram). Bigger encoders (resnet50) can be more accurate but "
            "are slower and need more data; resnet18 is a fast, solid default. "
            "The U-Net does pixel-by-pixel call/no-call segmentation."
        )
        form.addRow("Encoder:", self.combo_encoder)

        self.combo_device = QComboBox()
        self.combo_device.addItems(["auto", "cuda", "mps", "cpu"])
        self.combo_device.setToolTip(
            "Compute device. 'auto' picks CUDA (NVIDIA) or MPS (Apple Silicon) "
            "if available, else CPU. CPU works but is much slower."
        )
        form.addRow("Device:", self.combo_device)

        self.spin_overlap = QDoubleSpinBox()
        self.spin_overlap.setRange(0.0, 0.9)
        self.spin_overlap.setSingleStep(0.05)
        self.spin_overlap.setValue(0.25)
        self.spin_overlap.setToolTip(
            "Inference slides a fixed-width window across the recording; this "
            "is how much neighboring windows overlap. More overlap (0.25 = 25%) "
            "blends tile seams so calls spanning a boundary aren't cut, at the "
            "cost of slightly slower inference."
        )
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
        self.spin_threshold.setToolTip(
            "For every spectrogram pixel the model outputs a 0–1 probability of "
            "being part of a call. Pixels at or above this cutoff are kept and "
            "grouped into call detections. Lower = more (and fainter) calls but "
            "more false positives; higher = fewer, higher-confidence calls.\n\n"
            "Note: the cutoff is baked in at inference time — MAD does not store "
            "the full probability map, so to try a different threshold just "
            "re-run inference (it's fast)."
        )
        form.addRow("Probability threshold:", self.spin_threshold)

        self.spin_min_blob = QSpinBox()
        self.spin_min_blob.setRange(1, 10000)
        self.spin_min_blob.setValue(8)
        self.spin_min_blob.setToolTip(
            "Discard detected blobs smaller than this many pixels. Filters out "
            "tiny specks of noise that cross the threshold. Raise it if you get "
            "lots of pinpoint false detections; lower it to catch very short "
            "calls."
        )
        form.addRow("Min blob pixels:", self.spin_min_blob)

        self.combo_device = QComboBox()
        self.combo_device.addItems(["auto", "cuda", "mps", "cpu"])
        self.combo_device.setToolTip(
            "Compute device. 'auto' picks CUDA (NVIDIA) or MPS (Apple Silicon) "
            "if available, else CPU."
        )
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
    device_signal = pyqtSignal(str)      # resolved compute device
    file_done_signal = pyqtSignal(dict)  # per-file summary (incl. timing)

    def __init__(self, cfg, wav_paths: List[str], parent=None):
        super().__init__(parent)
        self.cfg = cfg
        self.wav_paths = wav_paths
        self._stop = False
        import threading
        # Event is *set* when running, *cleared* when paused.
        self._resumed = threading.Event()
        self._resumed.set()

    def request_stop(self):
        self._stop = True
        self._resumed.set()  # unblock a paused worker so it can exit cleanly

    def pause(self):
        self._resumed.clear()

    def resume(self):
        self._resumed.set()

    def is_paused(self) -> bool:
        return not self._resumed.is_set()

    def _wait_if_paused(self):
        """Block (off the UI thread) while paused, returning promptly on stop.
        Called between inference tiles/files so pause takes effect almost
        immediately, and the partially-done work simply waits in memory."""
        while not self._resumed.wait(timeout=0.1):
            if self._stop:
                return

    def run(self):
        try:
            from fnt.usv.usv_detector.mad_inference import run_inference_on_files
            results = run_inference_on_files(
                self.wav_paths, self.cfg,
                progress=lambda *a: self.progress_signal.emit(*a),
                should_stop=lambda: self._stop,
                wait_if_paused=self._wait_if_paused,
                on_device=lambda d: self.device_signal.emit(d),
                on_file_done=lambda s: self.file_done_signal.emit(s),
            )
            self.finished_signal.emit(results)
        except Exception as e:
            import traceback
            self.error_signal.emit(f"{e}\n\n{traceback.format_exc()}")


class MADViewInferenceWorker(QThread):
    """Run inference on a time-slice of audio (the visible view)."""
    progress_signal = pyqtSignal(float)   # 0.0–1.0
    finished_signal = pyqtSignal(list, float, float)  # rows, view_start, view_end
    error_signal = pyqtSignal(str)

    def __init__(self, model_path: str, audio_segment: np.ndarray,
                 sample_rate: int, view_start: float, view_end: float,
                 spec_params: dict, threshold: float = 0.5,
                 min_blob_pixels: int = 8, device: str = 'auto',
                 parent=None):
        super().__init__(parent)
        self.model_path = model_path
        self.audio_segment = audio_segment
        self.sample_rate = sample_rate
        self.view_start = view_start
        self.view_end = view_end
        self.spec_params = spec_params
        self.threshold = threshold
        self.min_blob_pixels = min_blob_pixels
        self.device = device
        self.prob_mask: Optional[np.ndarray] = None
        self.frame_offset: int = 0

    def run(self):
        try:
            from fnt.usv.usv_detector.mad_inference import (
                load_model, infer_probability_mask,
                extract_blobs, blobs_to_rows,
            )
            from fnt.usv.usv_detector.mad_dataset import compute_full_spec_image

            self.progress_signal.emit(0.1)
            model, ckpt, device = load_model(self.model_path, self.device)

            sp = self.spec_params
            hop = sp['nperseg'] - sp['noverlap']
            self.frame_offset = int(self.view_start * self.sample_rate / hop)

            self.progress_signal.emit(0.2)
            spec = compute_full_spec_image(
                self.audio_segment.astype(np.float32), self.sample_rate,
                nperseg=sp['nperseg'], noverlap=sp['noverlap'],
                nfft=sp['nfft'], db_min=sp['db_min'], db_max=sp['db_max'],
            )

            tile_freq = int(ckpt.get('tile_freq_bins', 512))
            tile_time = int(ckpt.get('tile_time_frames', 256))

            self.progress_signal.emit(0.3)
            prob = infer_probability_mask(
                model, spec,
                tile_freq_bins=tile_freq,
                tile_time_frames=tile_time,
                overlap_fraction=0.25,
                device=device,
                progress=lambda i, n: self.progress_signal.emit(
                    0.3 + 0.6 * (i / max(1, n))),
            )
            self.prob_mask = prob

            self.progress_signal.emit(0.92)
            blobs = extract_blobs(prob, threshold=self.threshold,
                                  min_blob_pixels=self.min_blob_pixels,
                                  include_mask=True, spec=spec)
            rows = blobs_to_rows(blobs, nperseg=sp['nperseg'],
                                 noverlap=sp['noverlap'],
                                 nfft=sp['nfft'], sr=self.sample_rate,
                                 db_min=sp.get('db_min'), db_max=sp.get('db_max'),
                                 spec=spec)

            # Offset times to file coordinates
            for r in rows:
                r['start_s'] += self.view_start
                r['stop_s'] += self.view_start

            self.progress_signal.emit(1.0)
            self.finished_signal.emit(rows, self.view_start, self.view_end)
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
    per-epoch validation loss — SLEAP style.
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

    def __init__(self, parent=None, show_plot: bool = False,
                 external_log: QTextEdit = None):
        super().__init__(parent)
        self._show_plot = show_plot
        self._external_log = external_log
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

        if external_log is None:
            self.log = QTextEdit()
            self.log.setReadOnly(True)
            self.log.setMaximumHeight(110)
            vbox.addWidget(self.log)
        else:
            self.log = external_log

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
        self._figure = Figure(tight_layout=True)
        self._figure.patch.set_facecolor("#1e1e1e")
        self._canvas = FigureCanvas(self._figure)
        self._canvas.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        parent_layout.addWidget(self._canvas, 1)
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
        if self._external_log is None:
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


class MADTrainGraphDialog(QDialog):
    """Non-modal window that hosts the live training graph so the spectrogram
    stays usable during a run. Closing it while training is active defers to the
    main window (keep running in the background vs stop)."""

    def __init__(self, main):
        super().__init__(main)
        self._main = main
        self.setModal(False)
        self.setWindowTitle("Segmentation Training")
        self.setWindowFlags(self.windowFlags()
                            | Qt.WindowMinimizeButtonHint
                            | Qt.WindowMaximizeButtonHint)
        self.resize(900, 600)

    def closeEvent(self, event):
        self._main._on_train_dialog_close(event)


# ======================================================================
# Main window
# ======================================================================
class MADMainWindow(QMainWindow):
    BASE_TITLE = "FNT Mask Audio Detector"

    @staticmethod
    def _apply_dark_theme():
        """Force a consistent dark look on every OS/platform. Without this MAD
        inherits the native theme — dark on macOS, light on Windows — so we pin
        the Fusion style + a dark palette, which themes all standard widgets
        (inputs, lists, menus, scrollbars) uniformly."""
        app = QApplication.instance()
        if app is None:
            return
        try:
            app.setStyle("Fusion")
        except Exception:
            pass
        from PyQt5.QtGui import QPalette
        p = QPalette()
        p.setColor(QPalette.Window, QColor(43, 43, 43))
        p.setColor(QPalette.WindowText, QColor(220, 220, 220))
        p.setColor(QPalette.Base, QColor(30, 30, 30))
        p.setColor(QPalette.AlternateBase, QColor(43, 43, 43))
        p.setColor(QPalette.ToolTipBase, QColor(30, 30, 30))
        p.setColor(QPalette.ToolTipText, QColor(220, 220, 220))
        p.setColor(QPalette.Text, QColor(220, 220, 220))
        p.setColor(QPalette.Button, QColor(53, 53, 53))
        p.setColor(QPalette.ButtonText, QColor(220, 220, 220))
        p.setColor(QPalette.BrightText, QColor(255, 80, 80))
        p.setColor(QPalette.Link, QColor(0, 120, 212))
        p.setColor(QPalette.Highlight, QColor(0, 120, 212))
        p.setColor(QPalette.HighlightedText, QColor(255, 255, 255))
        for role in (QPalette.WindowText, QPalette.Text, QPalette.ButtonText):
            p.setColor(QPalette.Disabled, role, QColor(120, 120, 120))
        app.setPalette(p)

    def __init__(self):
        super().__init__()
        self._apply_dark_theme()
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

        # When True (default), Accept/Reject jumps to the next pending detection
        # automatically. When False, the user steps through with Back (B) / Next
        # (N) and the view stays put after each decision.
        self._auto_advance: bool = True

        # Count of pending detections the user has decided on for the current
        # file (reset on file load). Drives the "all reviewed — next file?" prompt.
        self._reviewed_count: int = 0

        # Undo stack of review actions (accept/reject/delete) for the current
        # file — each entry snapshots the annotations + CSV + crops so Cmd/Ctrl+Z
        # can reverse the last few. Reset on file change.
        self._undo_stack: List[dict] = []

        # Training Data list (project recordings the model trains on). Backed
        # by deploy_files/deploy_list (repurposed from the old inference queue).
        self.deploy_files: List[str] = []
        self._deploy_file_idx: Optional[int] = None
        self._infer_model_project_dir: Optional[str] = None
        # Which list owns the single preview: 'session' or 'training'. Drives
        # _review_mode (training → accept saves a training example; session →
        # accept/reject only writes status to the sibling CSV).
        self._active_source: str = 'session'
        self._review_mode: str = 'deploy'

        # Rubber-band multi-selection: stable ids of box-selected detections.
        self._box_sel_ids: List = []

        # Session-level class tracking (used when no project is open).
        self._session_classes: List[str] = ["USV"]
        self._session_last_class: str = "USV"
        # Persisted per-file total annotation count cache.
        self._file_count_cache: Dict[str, int] = {}

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

        # Single session log — created early so the train/infer panels can
        # write to it. One copyable log lives at the bottom of the column.
        self.session_log = QTextEdit()
        self.session_log.setReadOnly(True)
        self.session_log.setMaximumHeight(140)
        self.session_log.setStyleSheet(
            "font-size: 9px; font-family: monospace; background: #1e1e1e;")

        # Live training graph — created before the tabs so the Train section
        # can wire its signals; it's placed in the right preview area below and
        # only shown while training runs (mirrors Mask Tracker).
        self.train_panel = MADRunPanel(show_plot=True,
                                       external_log=self.session_log)

        # ---------- Left panel: workflow-stage tabs over a shared canvas ----
        # Mask tab = label + train; Inference tab = run + blob review. Both
        # act on the single spectrogram canvas in the right panel. (A future
        # "Class" tab for call-type classification slots in here.)
        _fm = self.fontMetrics()
        _min_w = max(360, _fm.averageCharWidth() * 56 + 40)
        _max_w = max(460, _fm.averageCharWidth() * 74 + 40)

        # Single scrolled left column (no tabs): everything acts on the one
        # shared spectrogram canvas. Order top→bottom mirrors the workflow:
        # open project → load session audio → curate Training Data → review
        # detections → label → train/infer → logs.
        self.left_column = QScrollArea()
        self.left_column.setWidgetResizable(True)
        self.left_column.setMinimumWidth(_min_w)
        self.left_column.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        self.left_column.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        _page = QWidget()
        _page_layout = QVBoxLayout(_page)
        _page_layout.setContentsMargins(5, 5, 5, 5)
        _page_layout.setSpacing(8)
        for build in (
            self._create_session_audio_section,
            self._create_paint_tools_section,        # Labeling Tools
            self._create_annotation_list_section,    # Detections
            self._create_training_data_list_section, # Training Data
            self._create_model_section,
            self._create_session_log_section,
        ):
            build(_page_layout)
        _page_layout.addStretch()
        self.left_column.setWidget(_page)

        # Persistent project indicator pinned to the bottom-left of the window
        # (below the scrolling column). New/Open Project live in the File menu,
        # so the old top-of-column Project group is gone.
        self.lbl_project_status = QLabel("No project loaded")
        self.lbl_project_status.setStyleSheet(
            "color: #999999; font-size: 10px; padding: 3px 6px;")
        self._left_container = QWidget()
        _lc = QVBoxLayout(self._left_container)
        _lc.setContentsMargins(0, 0, 0, 0)
        _lc.setSpacing(0)
        _lc.addWidget(self.left_column, 1)
        _lc.addWidget(self.lbl_project_status)

        # Disable training/inference sections until a project is open.
        self._set_train_sections_enabled(False)

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

        # The live training graph lives in its own floating window
        # (_train_dialog) so the spectrogram stays visible/usable while a run is
        # in progress — see _show_training_dialog. Not added to this layout.
        self._train_dialog = None
        self._training_active = False

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
            "Time window duration (seconds). Zoom with ↑/↓ or by scrolling the "
            "mouse wheel over the spectrogram (centered on the cursor).")
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
        self.btn_play = QPushButton("Play (Space)")
        self.btn_play.setToolTip("Play / stop the visible window (Space)")
        self.btn_play.clicked.connect(self._toggle_playback)
        controls_layout.addWidget(self.btn_play)

        controls_layout.addWidget(QLabel("Speed (+/-):"))
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

        splitter = QSplitter(Qt.Horizontal)
        splitter.addWidget(self._left_container)
        splitter.addWidget(right_panel)
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        main_layout.addWidget(splitter)

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
        for child in self.left_column.findChildren(QWidget):
            if isinstance(child, (QAbstractButton, QListWidget, QTreeWidget)):
                child.setFocusPolicy(Qt.NoFocus)
        # Lists stay NoFocus (from the blanket above): arrow keys are fully
        # dedicated to the preview's pan/zoom, never list navigation.
        self.spectrogram.setFocusPolicy(Qt.ClickFocus)

    # ------------------------------------------------------------------
    # Left-panel section builders
    # ------------------------------------------------------------------
    def _create_session_audio_section(self, layout):
        group = QGroupBox("Load Session Audio")
        self._grp_session_audio = group
        vbox = QVBoxLayout()
        vbox.setSpacing(4)

        add_row = QHBoxLayout()
        add_row.setSpacing(4)
        self.btn_add_folder = QPushButton("Add Folder…")
        self.btn_add_folder.setToolTip(
            "Load every .wav directly inside a folder (non-recursive) into the "
            "working session. Loading does NOT copy files into the project — "
            "labels/predictions save next to the original audio. Use 'Copy to "
            "Training Data' to add a file to the project's training set."
        )
        self.btn_add_folder.clicked.connect(self._menu_add_folder)
        add_row.addWidget(self.btn_add_folder)

        self.btn_add_files = QPushButton("Add Files…")
        self.btn_add_files.setToolTip(
            "Load individual .wav files into the working session. Does not copy "
            "them into the project — labels save next to the source audio."
        )
        self.btn_add_files.clicked.connect(self._add_audio_files)
        add_row.addWidget(self.btn_add_files)
        vbox.addLayout(add_row)

        self.file_list = QListWidget()
        self.file_list.setMaximumHeight(180)
        self.file_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.file_list.currentRowChanged.connect(self._on_file_selected)
        self.file_list.itemSelectionChanged.connect(self._sync_list_buttons)
        vbox.addWidget(self.file_list)

        btn_row = QHBoxLayout()
        btn_row.setSpacing(4)
        self.btn_remove_files = QPushButton("Remove File(s)")
        self.btn_remove_files.setToolTip(
            "Unload the selected files from the session list. Does NOT delete "
            "anything from disk — the .wav and any sibling csv/h5 stay put. "
            "Use Shift/Cmd+click to select multiple."
        )
        self.btn_remove_files.clicked.connect(self._remove_selected_files)
        self.btn_remove_files.setEnabled(False)
        btn_row.addWidget(self.btn_remove_files)

        self.btn_copy_to_training = QPushButton("Copy File(s) → Training Data")
        self.btn_copy_to_training.setToolTip(
            "Copy the selected file(s) — the .wav plus its sibling csv/h5 "
            "labels/predictions — into the project's Training Data set. The "
            "model trains on the Training Data set. Copies are independent "
            "snapshots; re-copying a file already present asks before "
            "overwriting. Needs an open project."
        )
        self.btn_copy_to_training.clicked.connect(self._copy_to_training_data)
        self.btn_copy_to_training.setEnabled(False)
        btn_row.addWidget(self.btn_copy_to_training)
        vbox.addLayout(btn_row)

        nav_row = QHBoxLayout()
        nav_row.setSpacing(2)
        self.btn_prev_file = QPushButton("< Prev")
        self.btn_prev_file.setToolTip("Previous file")
        self.btn_prev_file.clicked.connect(self._prev_file)
        self.btn_prev_file.setEnabled(False)
        nav_row.addWidget(self.btn_prev_file)

        self.lbl_file_num = QLabel("File 0/0")
        self.lbl_file_num.setAlignment(Qt.AlignCenter)
        nav_row.addWidget(self.lbl_file_num, 1)

        self.btn_next_file = QPushButton("Next >")
        self.btn_next_file.setToolTip("Next file")
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
        group = QGroupBox("Labeling Tools")
        self._grp_labeling_tools = group
        vbox = QVBoxLayout()
        vbox.setSpacing(4)

        # Row 1: the three drawing tools — SAM (primary), Brush, Eraser.
        tools_row = QHBoxLayout()
        tools_row.setSpacing(2)
        self.btn_sam = QPushButton("SAM (M)")
        self.btn_sam.setToolTip(
            "SAM2-assisted labeling — left-click a call to propose a mask,\n"
            "right-click to add a negative point. Enter accepts, Esc clears.\n"
            "First use offers any SAM2 models in LocalModels/, or downloads "
            "one.\nShortcut: M"
        )
        self.btn_sam.setCheckable(True)
        self.btn_sam.setStyleSheet(self._ACTIVE_TOOL_QSS)
        self.btn_sam.clicked.connect(self._on_sam_clicked)
        self.btn_sam.setEnabled(False)
        tools_row.addWidget(self.btn_sam)

        # Small "…" button to switch the SAM2 checkpoint at any time (the model
        # picker otherwise only appears on first use).
        self.btn_sam_model = QPushButton("…")
        self.btn_sam_model.setToolTip(
            "Change the SAM2 model (switch between checkpoints, e.g. tiny ↔ "
            "large, or download another).")
        self.btn_sam_model.setStyleSheet(self._ACTIVE_TOOL_QSS)
        self.btn_sam_model.setFixedWidth(28)
        self.btn_sam_model.setEnabled(False)
        self.btn_sam_model.clicked.connect(self._change_sam_model)
        tools_row.addWidget(self.btn_sam_model)

        self.btn_paint = QPushButton("Paint (P)")
        self.btn_paint.setToolTip(
            "Manually paint target USV pixels with the brush "
            "(left-click + drag).\n"
            "Scroll the wheel over the spectrogram to fine-tune brush size.\n"
            "Shortcut: P"
        )
        self.btn_paint.setCheckable(True)
        self.btn_paint.setStyleSheet(self._ACTIVE_TOOL_QSS)
        self.btn_paint.clicked.connect(self._on_brush_clicked)
        self.btn_paint.setEnabled(False)
        tools_row.addWidget(self.btn_paint)

        self.btn_erase = QPushButton("Eraser (E)")
        self.btn_erase.setToolTip(
            "Erase painted pixels (scroll to resize)\n"
            "Shortcut: E"
        )
        self.btn_erase.setCheckable(True)
        self.btn_erase.setStyleSheet(self._ACTIVE_TOOL_QSS)
        self.btn_erase.clicked.connect(self._on_eraser_clicked)
        self.btn_erase.setEnabled(False)
        tools_row.addWidget(self.btn_erase)
        vbox.addLayout(tools_row)

        # Row 2: discard actions — Clear and Undo.
        mode_row = QHBoxLayout()
        mode_row.setSpacing(2)
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
        self.btn_view_cycle = QPushButton("Spec + Mask (V)")
        self.btn_view_cycle.setToolTip(
            "Toggle the preview: Spec + Mask ↔ Spec only.\nShortcut: V")
        self.btn_view_cycle.clicked.connect(self._shortcut_cycle_view)
        view_row.addWidget(self.btn_view_cycle)
        view_row.addStretch()
        vbox.addLayout(view_row)

        hint = QLabel(
            "Yellow = pending (solid = prediction, dotted = you're drawing; "
            "Enter to confirm) · Green = confirmed · Red = rejected. Confirmed "
            "calls save as self-contained training examples."
        )
        hint.setStyleSheet("color: #888888; font-size: 9px; font-style: italic;")
        hint.setWordWrap(True)
        vbox.addWidget(hint)

        # Quick inference: button + model selector
        infer_row = QHBoxLayout()
        infer_row.setSpacing(4)
        self.btn_quick_infer = QPushButton("Quick Inference (Q)")
        self.btn_quick_infer.setToolTip(
            "Run inference on the current visible view using the selected model.\n"
            "Predictions appear as yellow annotations. Shortcut: I"
        )
        self.btn_quick_infer.clicked.connect(self._quick_inference_current_file)
        self.btn_quick_infer.setEnabled(False)
        infer_row.addWidget(self.btn_quick_infer)
        self.combo_quick_infer_model = QComboBox()
        self.combo_quick_infer_model.setSizePolicy(
            QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.combo_quick_infer_model.setToolTip(
            "Select which trained model to use for quick inference")
        infer_row.addWidget(self.combo_quick_infer_model, 1)
        vbox.addLayout(infer_row)

        group.setLayout(vbox)
        layout.addWidget(group)

    def _refresh_quick_infer_models(self):
        """Populate the quick-inference model dropdown from project models/."""
        if not hasattr(self, 'combo_quick_infer_model'):
            return
        prev = self.combo_quick_infer_model.currentData()
        self.combo_quick_infer_model.blockSignals(True)
        self.combo_quick_infer_model.clear()
        if self._project is None:
            self.combo_quick_infer_model.blockSignals(False)
            return
        models_root = os.path.join(self._project.project_dir, 'models')
        if not os.path.isdir(models_root):
            self.combo_quick_infer_model.blockSignals(False)
            return
        for name in sorted(os.listdir(models_root)):
            w = os.path.join(models_root, name, 'weights.pt')
            if os.path.isfile(w):
                self.combo_quick_infer_model.addItem(name, w)
        if self.combo_quick_infer_model.count() > 0:
            if prev:
                for i in range(self.combo_quick_infer_model.count()):
                    if self.combo_quick_infer_model.itemData(i) == prev:
                        self.combo_quick_infer_model.setCurrentIndex(i)
                        break
                else:
                    self.combo_quick_infer_model.setCurrentIndex(
                        self.combo_quick_infer_model.count() - 1)
            else:
                self.combo_quick_infer_model.setCurrentIndex(
                    self.combo_quick_infer_model.count() - 1)
        self.combo_quick_infer_model.blockSignals(False)
        has_model = self.combo_quick_infer_model.count() > 0
        self.btn_quick_infer.setEnabled(has_model)

    def _set_view_mode(self, mode: str):
        if mode == 'mask_only':  # retired view — fall back to overlay
            mode = 'overlay'
        labels = {'overlay': 'Spec + Mask (V)', 'spec': 'Spec only (V)'}
        if hasattr(self, 'btn_view_cycle'):
            self.btn_view_cycle.setText(labels.get(mode, 'Spec + Mask (V)'))
        self.spectrogram.set_view_mode(mode)

    def _create_training_data_list_section(self, layout):
        group = QGroupBox("Training Data")
        self._grp_training_list = group
        vbox = QVBoxLayout()
        vbox.setSpacing(4)

        hint = QLabel(
            "Recordings copied into the project — the model trains on these. "
            "Add files with 'Copy File(s) → Training Data' above. Click a row "
            "to preview/review it (edits modify the training copy).")
        hint.setStyleSheet("color: #999999; font-size: 9px;")
        hint.setWordWrap(True)
        vbox.addWidget(hint)

        # Repurposed from the old inference queue: this is now the project's
        # Training Data set. deploy_files/deploy_list back it.
        self.deploy_list = QListWidget()
        self.deploy_list.setMaximumHeight(150)
        self.deploy_list.setSelectionMode(QListWidget.ExtendedSelection)
        self.deploy_list.setToolTip(
            "Training Data recordings (stored in the project). Click to preview "
            "and review; edits modify this training copy's csv/h5.")
        self.deploy_list.currentRowChanged.connect(self._on_deploy_file_selected)
        self.deploy_list.itemSelectionChanged.connect(self._sync_list_buttons)
        vbox.addWidget(self.deploy_list)

        self.btn_remove_training = QPushButton("Remove from Training Data")
        self.btn_remove_training.setToolTip(
            "Delete the selected recording(s) from the project's Training Data "
            "(removes the copied wav + csv/h5 from the project). Does not touch "
            "the original session source files. Shift/Cmd+click for multiple.")
        self.btn_remove_training.clicked.connect(self._remove_from_training_data)
        self.btn_remove_training.setEnabled(False)
        vbox.addWidget(self.btn_remove_training)

        self.lbl_deploy_queue = QLabel("0 files in training set")
        self.lbl_deploy_queue.setStyleSheet("color: #999999; font-size: 9px;")
        vbox.addWidget(self.lbl_deploy_queue)

        group.setLayout(vbox)
        layout.addWidget(group)

    def _create_model_section(self, layout):
        group = QGroupBox("Model Training && Inference")
        self._grp_train_model = group
        vbox = QVBoxLayout()
        vbox.setSpacing(6)

        def _sub(title):
            lbl = QLabel(title)
            lbl.setStyleSheet(
                "font-weight: bold; font-size: 10px; margin-top: 4px;")
            vbox.addWidget(lbl)

        # ---- Model picker (inference source; trained models land here) ----
        _sub("Model")
        model_row = QHBoxLayout()
        model_row.setSpacing(2)
        self.combo_deploy_model = QComboBox()
        self.combo_deploy_model.setToolTip(
            "Trained models available for inference. Auto-loaded from the open "
            "project (newest first). Use the buttons below to pull models from "
            "another project or a loose weights file.")
        self.combo_deploy_model.currentIndexChanged.connect(
            self._on_deploy_model_changed)
        model_row.addWidget(self.combo_deploy_model, 1)
        self.btn_deploy_refresh = QPushButton("Refresh")
        self.btn_deploy_refresh.setToolTip(
            "Rescan the project and select the latest model trained here.")
        self.btn_deploy_refresh.clicked.connect(
            lambda: self._refresh_deploy_models(select_latest=True))
        model_row.addWidget(self.btn_deploy_refresh)
        vbox.addLayout(model_row)

        src_row = QHBoxLayout()
        src_row.setSpacing(2)
        self.btn_deploy_load_project = QPushButton("Load Project Models…")
        self.btn_deploy_load_project.setToolTip(
            "Pick another MAD project folder to load its trained models into "
            "the dropdown above.")
        self.btn_deploy_load_project.clicked.connect(self._browse_infer_project)
        src_row.addWidget(self.btn_deploy_load_project)
        self.btn_deploy_browse = QPushButton("Browse Weights…")
        self.btn_deploy_browse.setToolTip(
            "Pick an individual weights.pt file from anywhere on disk.")
        self.btn_deploy_browse.clicked.connect(self._browse_deploy_model)
        src_row.addWidget(self.btn_deploy_browse)
        vbox.addLayout(src_row)

        self.lbl_deploy_model_info = QLabel("No model selected")
        self.lbl_deploy_model_info.setStyleSheet(
            "color: #999999; font-size: 9px;")
        self.lbl_deploy_model_info.setWordWrap(True)
        vbox.addWidget(self.lbl_deploy_model_info)

        # ---- Train a new model (config collapses under a toggle) ----
        self.chk_train_new = QCheckBox("Train a new model")
        self.chk_train_new.setToolTip(
            "Show the training configuration to train a fresh model on the "
            "Training Data set. The trained model is added to the dropdown "
            "above and selected automatically.")
        self.chk_train_new.toggled.connect(self._on_train_new_toggled)
        vbox.addWidget(self.chk_train_new)

        self._train_config_widget = QWidget()
        tc = QVBoxLayout(self._train_config_widget)
        tc.setContentsMargins(0, 0, 0, 0)
        tc.setSpacing(4)
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
        self.spin_train_patience.setValue(20)
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

        tc.addLayout(form)

        post_note = QLabel(
            "Training runs from the single button below. Tick a Run Inference "
            "target to also predict on the new model when training finishes.")
        post_note.setStyleSheet("color: #999999; font-size: 9px;")
        post_note.setWordWrap(True)
        tc.addWidget(post_note)

        self._train_config_widget.setVisible(False)  # revealed by the toggle
        vbox.addWidget(self._train_config_widget)

        # The live training graph (self.train_panel) lives in the right-hand
        # preview area (built in _build_ui) and is only shown while training
        # runs — mirroring Mask Tracker. Here we just wire its signals.
        self.train_panel.run_finished.connect(self._on_training_finished)
        self.train_panel.cancel_requested.connect(
            lambda: self.status_bar.showMessage("Stopping training…")
        )

        # ---- Run inference ----
        _sub("Run Inference")
        scope_hint = QLabel("Run the selected model on:")
        scope_hint.setStyleSheet("font-size: 9px; color: #bbbbbb;")
        vbox.addWidget(scope_hint)

        sess_row = QHBoxLayout()
        sess_row.setSpacing(4)
        self.chk_scope_session = QCheckBox("Session data")
        self.chk_scope_session.setToolTip(
            "Run inference on files loaded in Load Session Audio. Predictions "
            "save as csv/h5 next to each source recording.")
        self.chk_scope_session.toggled.connect(self._update_run_button)
        sess_row.addWidget(self.chk_scope_session)
        self.combo_scope_session = QComboBox()
        self.combo_scope_session.addItems(["All", "Current file"])
        self.combo_scope_session.setToolTip(
            "All session files, or just the currently previewed session file.")
        self.combo_scope_session.currentIndexChanged.connect(
            self._update_run_button)
        sess_row.addWidget(self.combo_scope_session, 1)
        vbox.addLayout(sess_row)

        train_row = QHBoxLayout()
        train_row.setSpacing(4)
        self.chk_scope_training = QCheckBox("Training data")
        self.chk_scope_training.setToolTip(
            "Run inference on the project's Training Data recordings. "
            "Predictions save into the training copy's csv/h5.")
        self.chk_scope_training.toggled.connect(self._update_run_button)
        train_row.addWidget(self.chk_scope_training)
        self.combo_scope_training = QComboBox()
        self.combo_scope_training.addItems(["All", "Current file"])
        self.combo_scope_training.setToolTip(
            "All training-data files, or just the currently previewed one.")
        self.combo_scope_training.currentIndexChanged.connect(
            self._update_run_button)
        train_row.addWidget(self.combo_scope_training, 1)
        vbox.addLayout(train_row)

        iform = QFormLayout()
        self.spin_infer_threshold = QDoubleSpinBox()
        self.spin_infer_threshold.setRange(0.01, 0.99)
        self.spin_infer_threshold.setSingleStep(0.05)
        self.spin_infer_threshold.setValue(0.5)
        self.spin_infer_threshold.setToolTip(
            "<b>Probability threshold</b>: a pixel is called positive when the "
            "model's sigmoid output exceeds this value. Lower → more (and "
            "larger) detections; higher → fewer, higher-confidence ones.")
        iform.addRow("Probability threshold:", self.spin_infer_threshold)

        self.spin_infer_min_blob = QSpinBox()
        self.spin_infer_min_blob.setRange(1, 10000)
        self.spin_infer_min_blob.setValue(8)
        self.spin_infer_min_blob.setToolTip(
            "<b>Minimum blob size</b> in pixels: connected components smaller "
            "than this are discarded as noise.")
        iform.addRow("Min blob pixels:", self.spin_infer_min_blob)

        self.combo_infer_device = QComboBox()
        self.combo_infer_device.addItems(["auto", "cuda", "mps", "cpu"])
        self.combo_infer_device.setToolTip(
            "<b>Compute device</b> for inference: <b>auto</b> picks CUDA &gt; "
            "MPS &gt; CPU.")
        iform.addRow("Device:", self.combo_infer_device)
        vbox.addLayout(iform)

        self.chk_infer_preserve = QCheckBox(
            "Preserve painted labels (skip labeled time regions)")
        self.chk_infer_preserve.setChecked(True)
        self.chk_infer_preserve.setToolTip(
            "Skip inference over time ranges you've already hand-labeled on a "
            "file, so the model doesn't re-detect and duplicate confirmed "
            "calls. Only matters on files that carry painted labels.")
        vbox.addWidget(self.chk_infer_preserve)

        # Single context-aware action button: "Run Inference" with an existing
        # model, "Run Training" when Train-a-new-model is on with no target
        # ticked, or "Run Training + Inference" when both apply. Label + enabled
        # state are kept in sync by _update_run_button.
        self.btn_infer_run = QPushButton("Run Inference")
        self.btn_infer_run.setToolTip(
            "Runs whatever the settings above describe — train a new model, "
            "run the selected model on the ticked targets, or both.")
        self.btn_infer_run.clicked.connect(self._on_run_clicked)
        self.btn_infer_run.setEnabled(False)
        vbox.addWidget(self.btn_infer_run)

        self.btn_infer_pause = QPushButton("Pause")
        self.btn_infer_pause.setToolTip(
            "Pause the running inference (finishes the current tile, then "
            "waits). Resume continues where it left off — nothing is lost.")
        self.btn_infer_pause.setEnabled(False)
        self.btn_infer_pause.clicked.connect(self._toggle_infer_pause)
        vbox.addWidget(self.btn_infer_pause)

        self.infer_panel = MADRunPanel(show_plot=False,
                                       external_log=self.session_log)
        self.infer_panel.run_finished.connect(
            lambda ok: self._update_infer_run_enabled())
        vbox.addWidget(self.infer_panel)

        group.setLayout(vbox)
        layout.addWidget(group)

    def _on_train_new_toggled(self, on: bool):
        """Reveal/hide the training-config block under the 'Train a new model'
        checkbox, and relabel the action button."""
        if hasattr(self, '_train_config_widget'):
            self._train_config_widget.setVisible(on)
        self._update_run_button()

    def _on_run_clicked(self):
        """The single action button: re-show the graph while a run is active;
        otherwise train (if 'Train a new model' is on) — which auto-runs the
        ticked inference targets when it finishes — or just run inference."""
        if getattr(self, '_training_active', False):
            self._show_training_dialog()
            return
        if self.chk_train_new.isChecked():
            self._on_inline_train()
        else:
            self._on_deploy_infer()

    def _training_label_count(self) -> int:
        """Total confirmed training-example labels across the Training Data set
        (what the next training run will use)."""
        n = 0
        try:
            from fnt.usv.usv_detector.fnt_mask_store import (
                masks_sibling_path, td_count,
            )
            for fp in self.deploy_files:
                n += td_count(masks_sibling_path(fp))
        except Exception:
            n = 0
        return n

    def _update_run_button(self):
        """Keep the single action button's label + enabled state in sync with
        the Train/Inference settings."""
        if not hasattr(self, 'btn_infer_run'):
            return
        btn = self.btn_infer_run
        if getattr(self, '_training_active', False):
            btn.setText("Training… (show graph)")
            btn.setEnabled(True)
            return
        train = self.chk_train_new.isChecked()
        infer = (self.chk_scope_session.isChecked()
                 or self.chk_scope_training.isChecked())
        if train and infer:
            label = "Run Training + Inference"
        elif train:
            label = "Run Training"
        else:
            label = "Run Inference"
        if train:
            n = self._training_label_count()
            label += f" ({n} label{'s' if n != 1 else ''})"
            enabled = bool(self._project) and n > 0
        else:
            model = self._selected_deploy_model_path()
            enabled = (bool(model and os.path.isfile(model))
                       and bool(self._gather_inference_targets()))
        btn.setText(label)
        btn.setEnabled(enabled)

    def _create_session_log_section(self, layout):
        group = QGroupBox("Session Logs")
        vbox = QVBoxLayout()
        vbox.setSpacing(4)
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
        """Append a timestamped, pithy line to the Session Log panel."""
        from datetime import datetime as _dt
        line = f"[{_dt.now().strftime('%H:%M:%S')}] {msg}"
        w = getattr(self, 'session_log', None)
        if w is not None:
            w.append(line)

    def _copy_session_log(self):
        QApplication.clipboard().setText(self.session_log.toPlainText())
        self.status_bar.showMessage("Session log copied to clipboard")

    def _create_annotation_list_section(self, layout):
        group = QGroupBox("Detections")
        self._grp_detections = group
        vbox = QVBoxLayout()
        vbox.setSpacing(4)

        filter_row = QHBoxLayout()
        filter_row.setSpacing(4)
        filter_row.addWidget(QLabel("Show:"))
        self.combo_det_filter = QComboBox()
        self.combo_det_filter.addItems(["All", "Pending", "Confirmed", "Rejected"])
        self.combo_det_filter.setToolTip(
            "Filter detections: All, Pending (yellow predictions), "
            "Confirmed (blue, saved as training examples), or "
            "Rejected (red, recorded 'no' decisions)."
        )
        self.combo_det_filter.currentIndexChanged.connect(
            lambda _i: self._refresh_annotation_list()
        )
        filter_row.addWidget(self.combo_det_filter, 1)
        vbox.addLayout(filter_row)

        self.annotation_list = QTreeWidget()
        self.annotation_list.setMaximumHeight(200)
        self.annotation_list.setToolTip(
            "Detections for the current file. Green = confirmed, "
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

        self.lbl_annotation_count = QLabel("0 detections")
        self.lbl_annotation_count.setStyleSheet(
            "color: #999999; font-size: 9px;")

        self.btn_delete_selected = QPushButton("Delete (D)")
        self.btn_delete_selected.setToolTip(
            "Delete the selected detection. For a prediction this follows the "
            "Auto-advance toggle (on → jump to next pending; off → stay). Also "
            "via right-click → Delete.\nShortcut: D")
        self.btn_delete_selected.setFocusPolicy(Qt.NoFocus)
        self.btn_delete_selected.clicked.connect(self._delete_selected_annotation)

        # Prediction review controls (CAD-style).
        review_lbl = QLabel("Review Predictions:")
        review_lbl.setStyleSheet("font-weight: bold; font-size: 10px; margin-top: 4px;")
        self._build_review_controls(
            vbox, "Accept this prediction — saves as a training example",
            count_label=self.lbl_annotation_count,
            delete_btn=self.btn_delete_selected,
            review_label=review_lbl)

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

    def _build_review_controls(self, vbox, accept_tip: str, *,
                               count_label=None, delete_btn=None,
                               review_label=None, skip_shortcut=False):
        """Build one set of CAD-style prediction-review controls and register
        it so ``_update_pred_review_widgets`` keeps every set in sync."""
        if not hasattr(self, '_review_sets'):
            self._review_sets = []
        s = {}
        # Nav row: Back (B) | 0/0 predictions | Next (N)
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

        # Auto-advance toggle: on by default (jump to next pending detection
        # after each Accept/Reject); off lets the user step manually with B/N.
        s['auto_adv'] = QCheckBox("Auto-advance after Accept/Reject")
        s['auto_adv'].setChecked(self._auto_advance)
        s['auto_adv'].setToolTip(
            "On: after you Accept or Reject, jump straight to the next pending "
            "detection (fast review).\n"
            "Off: stay on the current spot — use Back (B) / Next (N) to move "
            "through the list yourself.")
        s['auto_adv'].setFocusPolicy(Qt.NoFocus)
        s['auto_adv'].toggled.connect(self._on_auto_advance_toggled)
        vbox.addWidget(s['auto_adv'])

        # Count label (e.g. "0 detections")
        if count_label is not None:
            vbox.addWidget(count_label)

        # Review label header
        if review_label is not None:
            vbox.addWidget(review_label)

        # Row 1: Accept (A) | Reject (R) | Delete (D)
        row1 = QHBoxLayout()
        row1.setSpacing(2)
        s['accept'] = QPushButton("Accept (A)")
        s['accept'].setToolTip(accept_tip + "<br>Shortcut: A")
        s['accept'].setStyleSheet(self._review_btn_qss("#2d7a3a", "#379247"))
        s['accept'].clicked.connect(self._accept_current_pred)
        row1.addWidget(s['accept'])
        s['reject'] = QPushButton("Reject (R)")
        s['reject'].setToolTip(
            "Record a 'no' on this detection: it stays visible shaded red and "
            "labeled 'Reject', and is saved as 'rejected' in the CSV — an audit "
            "trail of what was dismissed. Use Delete (D) to remove it entirely."
            "<br>Shortcut: R")
        s['reject'].setStyleSheet(self._review_btn_qss("#8a2c2c", "#a83636"))
        s['reject'].clicked.connect(self._reject_current_pred)
        row1.addWidget(s['reject'])
        if delete_btn is not None:
            delete_btn.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
            row1.addWidget(delete_btn)
            s['delete'] = delete_btn  # tracked so it enables with a selection
        for k in ('accept', 'reject'):
            s[k].setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        vbox.addLayout(row1)

        # Row 2: Skip (S) | Accept All | Clear All
        row2 = QHBoxLayout()
        row2.setSpacing(2)
        s['skip'] = QPushButton("Skip (S)")
        s['skip'].setToolTip(
            "Leave this prediction pending and advance to the next one (no "
            "decision recorded).<br>Shortcut: S")
        s['skip'].setStyleSheet(self._review_btn_qss("#5a5a5a", "#6a6a6a"))
        s['skip'].clicked.connect(self._skip_current_pred)
        row2.addWidget(s['skip'])
        s['accept_all'] = QPushButton("Accept All")
        s['accept_all'].setToolTip(
            "Accept every pending prediction for this file at once "
            "(train mode → saved as training examples; deploy mode → marked "
            "'accepted' in the output CSV).")
        s['accept_all'].clicked.connect(self._accept_all_preds)
        row2.addWidget(s['accept_all'])
        s['clear_all'] = QPushButton("Clear All")
        s['clear_all'].setToolTip(
            "Bulk-remove detections from this file by status — choose Pending / "
            "Accepted / Rejected in the dialog (default: Pending only).")
        s['clear_all'].clicked.connect(self._clear_all_detections)
        row2.addWidget(s['clear_all'])
        for k in ('skip', 'accept_all', 'clear_all'):
            s[k].setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        vbox.addLayout(row2)

        for w in s.values():
            if isinstance(w, QPushButton):
                w.setEnabled(False)
                w.setFocusPolicy(Qt.NoFocus)
        if delete_btn is not None:
            delete_btn.setEnabled(False)
            delete_btn.setFocusPolicy(Qt.NoFocus)
        self._review_sets.append(s)
        return s

    # --- undo (Cmd/Ctrl+Z) for review actions -------------------------
    def _snapshot_for_undo(self, label: str, crops: bool = True):
        """Capture the current file's review state (annotations + sibling CSV,
        and the per-blob crops only when ``crops`` is True) so the next
        Cmd/Ctrl+Z can restore it. Accept/Reject only touch the CSV, so they skip
        the (slightly costly) crop read; Delete/Clear need it."""
        sg = self.spectrogram
        wav = self._active_review_wav_path()
        snap = {'label': label,
                'anns': [dict(a) for a in sg.annotations],
                'sel_id': (sg.annotations[sg._selected_ann_idx].get('id')
                           if sg._selected_ann_idx is not None
                           and sg._selected_ann_idx < len(sg.annotations)
                           else None),
                'reviewed': self._reviewed_count,
                'csv': None, 'crops': None}
        if wav:
            try:
                from fnt.usv.usv_detector.mad_inference import read_blob_csv
                cp = pred_csv_sibling_path(wav)
                if os.path.isfile(cp):
                    snap['csv'] = (cp, read_blob_csv(cp))
            except Exception:
                pass
            if crops:
                try:
                    from fnt.usv.usv_detector.fnt_mask_store import (
                        masks_sibling_path, read_all_pred_masks)
                    h5 = masks_sibling_path(wav)
                    snap['crops'] = (h5, read_all_pred_masks(h5))
                except Exception:
                    pass
        self._undo_stack.append(snap)
        if len(self._undo_stack) > 15:  # bound the history
            self._undo_stack.pop(0)

    def _undo_review_action(self):
        """Reverse the last review action (accept/reject/delete) on this file."""
        if not self._undo_stack:
            self.status_bar.showMessage("Nothing to undo")
            return
        snap = self._undo_stack.pop()
        sg = self.spectrogram
        # Restore the annotation list.
        sg.annotations = [dict(a) for a in snap['anns']]
        # Restore the sibling CSV.
        if snap.get('csv'):
            cp, rows = snap['csv']
            try:
                from fnt.usv.usv_detector.mad_inference import write_blob_csv
                write_blob_csv(cp, rows)
            except Exception:
                pass
        # Restore the per-blob crops.
        if snap.get('crops'):
            h5, crops = snap['crops']
            try:
                from fnt.usv.usv_detector.fnt_mask_store import write_pred_masks
                items = [{'blob_id': bid, 'mask': c['mask'],
                          'f_off': c['f_off'], 't_off': c['t_off']}
                         for bid, c in crops.items()]
                write_pred_masks(h5, items)
            except Exception:
                pass
        self._reviewed_count = snap.get('reviewed', 0)
        sg._rebuild_confirmed_mask()
        sg.update()
        self._refresh_annotation_list()
        self._reselect_by_id(snap.get('sel_id'))
        self._update_pred_review_widgets()
        self.status_bar.showMessage(f"Undid: {snap['label']}")
        self._log(f"Undo: {snap['label']}")

    def _purge_review_annotation(self, ann: dict):
        """Remove all persisted traces of one detection — its CSV row, stored
        crop, and any saved training example. Does NOT touch the in-memory
        annotation list (the caller rebuilds it)."""
        self._remove_pred_csv_row(ann)
        self._delete_pred_crop(ann)
        aid = ann.get('id')
        if not aid:
            return
        if self._project is not None:
            try:
                from fnt.usv.usv_detector.mad_examples import delete_example
                delete_example(self._project.training_data_dir, aid)
            except Exception:
                pass
        wav = self._active_review_wav_path()
        if wav:
            try:
                from fnt.usv.usv_detector.fnt_mask_store import (
                    masks_sibling_path, td_delete)
                td_delete(masks_sibling_path(wav), aid)
            except Exception:
                pass

    def _clear_all_detections(self):
        """Bulk-remove this file's detections by status. A dialog lets the user
        pick which categories to clear (default: Pending only)."""
        anns = self.spectrogram.annotations
        n_pending = sum(1 for a in anns if a.get('status') == 'prediction')
        n_rejected = sum(1 for a in anns if a.get('status') == 'rejected')
        n_accepted = sum(1 for a in anns
                         if a.get('status') in (None, 'accepted'))
        if n_pending + n_rejected + n_accepted == 0:
            self.status_bar.showMessage("No detections to clear.")
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Clear Detections")
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel("Clear which detections from this file?"))
        chk_pending = QCheckBox(f"Pending ({n_pending})")
        chk_pending.setChecked(True)
        chk_accepted = QCheckBox(f"Accepted ({n_accepted})")
        chk_rejected = QCheckBox(f"Rejected ({n_rejected})")
        for c in (chk_pending, chk_accepted, chk_rejected):
            v.addWidget(c)
        note = QLabel(
            "Removes them entirely — CSV rows and masks are deleted (and, for "
            "accepted, the saved training example). This cannot be undone.")
        note.setWordWrap(True)
        note.setStyleSheet("color: #888888; font-size: 10px;")
        v.addWidget(note)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Clear")
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        if dlg.exec_() != QDialog.Accepted:
            return

        want_pending = chk_pending.isChecked()
        want_accepted = chk_accepted.isChecked()
        want_rejected = chk_rejected.isChecked()
        if not (want_pending or want_accepted or want_rejected):
            return
        self._snapshot_for_undo("Clear All")

        sg = self.spectrogram
        dropped, keep = [], []
        for ann in sg.annotations:
            st = ann.get('status')
            drop = ((st == 'prediction' and want_pending)
                    or (st == 'rejected' and want_rejected)
                    or (st in (None, 'accepted') and want_accepted))
            (dropped if drop else keep).append(ann)
        # Batch the persistence: ONE CSV write + ONE h5 open for all dropped
        # predictions/rejects, instead of a full rewrite per item (was O(N²) and
        # froze on thousands of detections). Training examples (accepted) are
        # deleted by id — usually few.
        self._batch_remove_pred_persistence(dropped)
        for ann in dropped:
            if ann.get('status') in (None, 'accepted'):
                self._purge_training_example(ann)
        removed = len(dropped)
        sg.annotations = keep
        sg._selected_ann_idx = None
        self._pred_review_idx = None
        self._file_count_cache = {}
        sg._rebuild_confirmed_mask()   # one rebuild for the whole operation
        sg.update()
        self._refresh_annotation_list()
        self._update_pred_review_widgets()
        cats = [c for c, want in (("pending", want_pending),
                                  ("accepted", want_accepted),
                                  ("rejected", want_rejected)) if want]
        self._log(f"Cleared {removed} detection(s) [{', '.join(cats)}]")
        self.status_bar.showMessage(f"Cleared {removed} detection(s)")

    def _refresh_annotation_list(self):
        if not hasattr(self, 'annotation_list'):
            return
        self.spectrogram._selected_ann_idx = None

        tree = self.annotation_list
        combo = getattr(self, 'combo_det_filter', None)
        lbl = getattr(self, 'lbl_annotation_count', None)

        if tree is None:
            return

        flt = combo.currentText() if combo else "All"
        tree.blockSignals(True)
        tree.setSortingEnabled(False)
        tree.clear()

        cur_wav = os.path.basename(self._active_wav_path() or '')

        sp = self._spec_params()
        sr = self.sample_rate or 1
        dt = self.spectrogram.hop / float(sr) if self.spectrogram.hop else 0
        df = (sr / 2.0) / (sp['nfft'] // 2) if sp.get('nfft') else 0
        entries = []
        if dt:
            for ann in self.spectrogram.annotations:
                st = ann.get('status')
                is_pred = st == 'prediction'
                is_rej = st == 'rejected'
                if flt == "Pending" and not is_pred:
                    continue
                if flt == "Confirmed" and (is_pred or is_rej):
                    continue
                if flt == "Rejected" and not is_rej:
                    continue
                entries.append(ann)
        entries.sort(key=lambda a: a['t0'])
        n_confirmed, n_pred, n_rej = 0, 0, 0
        ncols = tree.columnCount()
        new_items = []
        for ann in entries:
            st = ann.get('status')
            is_pred = st == 'prediction'
            is_rej = st == 'rejected'
            t0 = ann['t0'] * dt
            t1 = ann['t1'] * dt
            dur = max(0.0, t1 - t0)
            durs = f"{dur:.2f}s" if dur >= 1.0 else f"{dur * 1000:.0f}ms"
            cls = "Reject" if is_rej else ann.get('category', '')
            freq_lo = ann['f0'] * df / 1000 if df else 0
            freq_hi = ann['f1'] * df / 1000 if df else 0
            mask = ann.get('mask')
            pixels = int(mask.sum()) if mask is not None else 0
            score = ann.get('score', 0)
            icon = "○" if is_pred else ("✕" if is_rej else "●")
            score_s = f"{score:.2f}" if is_pred and score else ""
            item = QTreeWidgetItem([
                icon, f"{t0:.2f}s", cls, durs,
                f"{freq_lo:.0f}-{freq_hi:.0f}",
                str(pixels), score_s,
            ])
            kind = ('prediction' if is_pred
                    else 'rejected' if is_rej else 'confirmed')
            item.setData(0, Qt.UserRole, (cur_wav, t0, ann.get('id', ''), kind))
            color = (QColor(255, 230, 90) if is_pred
                     else QColor(255, 90, 90) if is_rej
                     else QColor(80, 210, 120))   # green confirmed
            for c in range(ncols):
                item.setForeground(c, color)
            new_items.append(item)
            if is_pred:
                n_pred += 1
            elif is_rej:
                n_rej += 1
            else:
                n_confirmed += 1
        if new_items:  # one batched insert is far faster than N addTopLevelItem
            tree.addTopLevelItems(new_items)
        tree.setSortingEnabled(True)
        tree.blockSignals(False)
        parts = []
        if n_confirmed:
            parts.append(f"{n_confirmed} confirmed")
        if n_pred:
            parts.append(f"{n_pred} prediction(s)")
        if n_rej:
            parts.append(f"{n_rej} rejected")
        count_text = (
            f"{n_confirmed + n_pred + n_rej} detection(s)" +
            (f" ({', '.join(parts)})" if len(parts) > 1 else "")
        )
        if lbl:
            lbl.setText(count_text)
        self._update_pred_review_widgets()
        self._update_train_button_count()
        self._update_file_list_counts()
        self._update_active_training_count()
        self._update_overview_marks()

    def _update_active_training_count(self):
        """Live-update the active Training Data list row's mask count from the
        in-memory annotations, so clear / delete / add / confirm reflect
        immediately (mirrors the session list's live reconcile)."""
        if getattr(self, '_active_source', 'session') != 'training':
            return
        if not hasattr(self, 'deploy_list'):
            return
        idx = self._deploy_file_idx
        if idx is None or not (0 <= idx < len(self.deploy_files)):
            return
        item = self.deploy_list.item(idx)
        if item is None:
            return
        base = os.path.basename(self.deploy_files[idx])
        n = len(self.spectrogram.annotations)
        if n:
            item.setText(f"✓  {base}  ({n})")
            item.setForeground(QColor(80, 200, 120))
        else:
            item.setText(base)
            item.setData(Qt.ForegroundRole, None)

    def _update_overview_marks(self):
        """Place one status-colored tick per detection on the waveform overview
        strip (yellow = pending, blue = accepted/confirmed, red = rejected), so
        the user can see where calls cluster across the whole file. Rebuilt on
        every list refresh, so adding/accepting/rejecting/deleting updates it."""
        if not hasattr(self, 'waveform_overview'):
            return
        sr = self.sample_rate
        hop = self.spectrogram.hop
        if not sr or not hop:
            self.waveform_overview.set_status_marks([])
            return
        dt = hop / float(sr)
        pending = QColor(255, 225, 60)
        rejected = QColor(255, 80, 80)
        confirmed = QColor(60, 210, 110)   # green
        marks = []
        for ann in self.spectrogram.annotations:
            st = ann.get('status')
            color = (pending if st == 'prediction'
                     else rejected if st == 'rejected' else confirmed)
            cs = (ann['t0'] + ann['t1']) / 2.0 * dt
            marks.append((cs, color))
        self.waveform_overview.set_status_marks(marks)

    def _update_train_button_count(self):
        """Back-compat alias — the training label count now appears on the single
        action button via :meth:`_update_run_button`."""
        self._update_run_button()

    def _on_annotation_list_selected(self, current, _previous=None):
        """Single-click in detections list: jump to the detection and highlight it."""
        if current is None:
            self.spectrogram._selected_ann_idx = None
            self.spectrogram.update()
            return
        self._dismiss_training_view()
        data = current.data(0, Qt.UserRole)
        if not data:
            return
        wav, t0, eid, status = data
        # The detection belongs to the currently-previewed file in the common
        # case (session OR training). Only switch files if it names a DIFFERENT
        # session file; a Training Data file is already the active preview.
        active = self._active_wav_path()
        if not (active and os.path.basename(active) == wav):
            for i, p in enumerate(self.audio_files):
                if os.path.basename(p) == wav:
                    if i != self.current_file_idx:
                        self.file_list.setCurrentRow(i)
                    break
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
        self._box_sel_ids = []  # a single list click clears any box multi-select
        self.spectrogram.update()
        # Accept/Reject/Delete act on the selected detection (any status), so
        # refresh the controls whenever the selection changes.
        self._update_pred_review_widgets()

    # --- prediction review (CAD-style accept / reject / skip) -----------
    def _active_annotation_tree(self) -> Optional[QTreeWidget]:
        if hasattr(self, 'annotation_list'):
            return self.annotation_list
        return None

    def _active_wav_path(self) -> Optional[str]:
        """Path of the file currently shown in the preview — from whichever
        list (session or training) owns the selection, or None."""
        if getattr(self, '_active_source', 'session') == 'training':
            if (self.deploy_files and self._deploy_file_idx is not None and
                    0 <= self._deploy_file_idx < len(self.deploy_files)):
                return self.deploy_files[self._deploy_file_idx]
        else:
            if (self.audio_files and
                    0 <= self.current_file_idx < len(self.audio_files)):
                return self.audio_files[self.current_file_idx]
        return None

    def _clear_session_selection(self):
        lw = getattr(self, 'file_list', None)
        if lw is not None:
            lw.blockSignals(True)
            lw.clearSelection()
            lw.setCurrentRow(-1)
            lw.blockSignals(False)

    def _clear_training_selection(self):
        lw = getattr(self, 'deploy_list', None)
        if lw is not None:
            lw.blockSignals(True)
            lw.clearSelection()
            lw.setCurrentRow(-1)
            lw.blockSignals(False)

    def _update_scope_labels(self):
        """Refresh the 'All (N)' counts on the inference-scope combos."""
        if not hasattr(self, 'combo_scope_session'):
            return
        self.combo_scope_session.setItemText(0, f"All ({len(self.audio_files)})")
        self.combo_scope_training.setItemText(
            0, f"All ({len(self.deploy_files)})")

    def _update_review_buttons_for_source(self):
        """Refresh widgets that depend on which list owns the preview."""
        self._update_scope_labels()
        self._update_pred_review_widgets()
        self._update_run_button()  # "current file" target may have changed

    def _update_train_button_enabled(self):
        """Back-compat alias — see :meth:`_update_run_button`."""
        self._update_run_button()

    def _sync_list_buttons(self):
        """Enable the Session/Training action buttons based on list selection."""
        if hasattr(self, 'btn_remove_files'):
            self.btn_remove_files.setEnabled(
                bool(self.file_list.selectedItems()))
        if hasattr(self, 'btn_copy_to_training'):
            # Enabled whenever a session file is selected; if no project is open
            # yet, clicking offers to create/open one (Training Data lives in it).
            self.btn_copy_to_training.setEnabled(
                bool(self.file_list.selectedItems()))
        if hasattr(self, 'btn_remove_training'):
            self.btn_remove_training.setEnabled(
                bool(self.deploy_list.selectedItems()))

    def _pred_indices(self) -> List[int]:
        """Annotation indices with status='prediction', ordered by the
        detection table's current visual sort so Back/Next follow whatever
        column the user clicked."""
        tree = self._active_annotation_tree()
        if tree is not None:
            ordered_ids = []
            for i in range(tree.topLevelItemCount()):
                data = tree.topLevelItem(i).data(0, Qt.UserRole)
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

    def _review_order(self) -> List[int]:
        """ALL annotation indices in the detection table's display order — the
        order Back/Next walk through (any status, not just pending)."""
        tree = self._active_annotation_tree()
        anns = self.spectrogram.annotations
        if tree is not None:
            id_to_idx = {a.get('id'): i for i, a in enumerate(anns)}
            ordered = []
            for i in range(tree.topLevelItemCount()):
                data = tree.topLevelItem(i).data(0, Qt.UserRole)
                if data and data[2] in id_to_idx:
                    ordered.append(id_to_idx[data[2]])
            if ordered:
                return ordered
        return list(range(len(anns)))

    def _selected_review_pos(self, order=None):
        """Position of the selected annotation within ``_review_order``."""
        if order is None:
            order = self._review_order()
        sel = self.spectrogram._selected_ann_idx
        if sel is None:
            return None
        try:
            return order.index(sel)
        except ValueError:
            return None

    def _center_and_select_ann(self, ann_idx: int):
        """Center the view on, highlight, and list-select one annotation."""
        anns = self.spectrogram.annotations
        if not (0 <= ann_idx < len(anns)):
            return
        ann = anns[ann_idx]
        if self.spectrogram.total_duration > 0 and self.sample_rate:
            dt = self.spectrogram.hop / float(self.sample_rate)
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

    def _select_review_pos(self, pos: int):
        order = self._review_order()
        if not order:
            return
        pos = max(0, min(pos, len(order) - 1))
        self._center_and_select_ann(order[pos])

    def _current_review_ann_idx(self):
        """The annotation an Accept/Reject/Delete acts on. Prefer the detection
        list's highlighted row (what the user sees selected, any status), then
        the spectrogram selection, then the first pending as a last resort."""
        anns = self.spectrogram.annotations
        tree = self._active_annotation_tree()
        if tree is not None:
            item = tree.currentItem()
            if item is not None:
                data = item.data(0, Qt.UserRole)
                if data:
                    eid = data[2]
                    for i, a in enumerate(anns):
                        if a.get('id') == eid:
                            self.spectrogram._selected_ann_idx = i
                            return i
        sel = self.spectrogram._selected_ann_idx
        if sel is not None and 0 <= sel < len(anns):
            return sel
        preds = self._pred_indices()
        if preds:
            self.spectrogram._selected_ann_idx = preds[0]
            return preds[0]
        return None

    def _reselect_by_id(self, eid):
        if eid is None:
            return
        for i, a in enumerate(self.spectrogram.annotations):
            if a.get('id') == eid:
                self._center_and_select_ann(i)
                return

    def _select_next_pending_after_id(self, eid) -> bool:
        """Select the next pending detection after the item with id ``eid`` in
        display order. Returns False if there's no pending one after it."""
        order = self._review_order()
        anns = self.spectrogram.annotations
        start = 0
        for p, ai in enumerate(order):
            if anns[ai].get('id') == eid:
                start = p + 1
                break
        for p in range(start, len(order)):
            if anns[order[p]].get('status') == 'prediction':
                self._select_review_pos(p)
                return True
        return False

    def _update_pred_review_widgets(self):
        order = self._review_order()
        n = len(order)
        pos = self._selected_review_pos(order)
        has_sel = pos is not None
        nav_text = (f"{pos + 1}/{n}" if has_sel else f"0/{n}")
        has_pending = len(self._pred_indices()) > 0
        for s in getattr(self, '_review_sets', []):
            s['prev'].setEnabled(has_sel and pos > 0)
            s['next'].setEnabled(has_sel and pos < n - 1)
            for k in ('accept', 'reject', 'skip'):
                s[k].setEnabled(has_sel)
            if s.get('delete') is not None:
                s['delete'].setEnabled(has_sel)
            s['accept_all'].setEnabled(has_pending)
            s['clear_all'].setEnabled(n > 0)
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
        if eid is None:
            return
        tree = self._active_annotation_tree()
        if tree is None:
            return
        tree.blockSignals(True)
        for i in range(tree.topLevelItemCount()):
            item = tree.topLevelItem(i)
            data = item.data(0, Qt.UserRole)
            if data and data[2] == eid:
                tree.setCurrentItem(item)
                tree.scrollToItem(item)
                break
        tree.blockSignals(False)

    def _on_auto_advance_toggled(self, checked: bool):
        """Keep the shared flag and every review set's checkbox in sync."""
        self._auto_advance = bool(checked)
        for s in getattr(self, '_review_sets', []):
            cb = s.get('auto_adv')
            if cb is not None and cb.isChecked() != self._auto_advance:
                cb.blockSignals(True)
                cb.setChecked(self._auto_advance)
                cb.blockSignals(False)

    def _pred_prev(self):
        """Back: move the selection to the previous detection in the list
        (any status — so you can revisit an already accepted/rejected call)."""
        order = self._review_order()
        if not order:
            return
        pos = self._selected_review_pos(order)
        self._select_review_pos((len(order) - 1) if pos is None else max(0, pos - 1))

    def _pred_next(self):
        """Next: move the selection to the next detection in the list."""
        order = self._review_order()
        if not order:
            return
        pos = self._selected_review_pos(order)
        self._select_review_pos(0 if pos is None
                                else min(len(order) - 1, pos + 1))

    def _accept_current_pred(self):
        if self._apply_to_box_selection('accept'):
            return
        sel = self._current_review_ann_idx()
        if sel is None:
            return
        ann = self.spectrogram.annotations[sel]
        st = ann.get('status')
        if st in (None, 'accepted'):
            self._after_review_decision(ann.get('id'), was_pending=False)
            return  # already accepted — just move on
        self._snapshot_for_undo("Accept", crops=False)
        self._log(f"Accept {self._pred_describe(sel)} [{self._review_mode}]")
        was_pending = st == 'prediction'
        if self._review_mode == 'deploy':
            # Keep the detection visible (blue) and recorded 'accepted' in the
            # CSV — same as the Label tab, minus saving a training example.
            self._write_pred_csv_status(ann, 'accepted')
            ann['status'] = 'accepted'
            self.spectrogram._rebuild_confirmed_mask()
        else:
            self._accept_prediction(sel)
        self._after_review_decision(self.spectrogram.annotations[sel].get('id')
                                    if sel < len(self.spectrogram.annotations)
                                    else ann.get('id'), was_pending)

    def _reject_current_pred(self):
        if self._apply_to_box_selection('reject'):
            return
        sel = self._current_review_ann_idx()
        if sel is None:
            return
        ann = self.spectrogram.annotations[sel]
        st = ann.get('status')
        if st == 'rejected':
            self._after_review_decision(ann.get('id'), was_pending=False)
            return  # already rejected — just move on
        self._snapshot_for_undo("Reject", crops=False)
        self._log(f"Reject {self._pred_describe(sel)} [{self._review_mode}]")
        was_pending = st == 'prediction'
        # Converting a confirmed/accepted call to rejected: drop its saved
        # training example (train mode) so it no longer trains the model.
        if st in (None, 'accepted'):
            self._purge_training_example(ann)
        # Reject is a *recorded* decision: keep the mask visible (red, labeled
        # "Reject") and persist 'rejected' to the CSV. Use Delete to remove it.
        self._write_pred_csv_status(ann, 'rejected')
        ann['status'] = 'rejected'
        self.spectrogram._rebuild_confirmed_mask()
        self.spectrogram.update()
        self._after_review_decision(ann.get('id'), was_pending)

    def _purge_training_example(self, ann: dict):
        """Delete the saved training example backing a confirmed annotation
        (no-op if there isn't one)."""
        aid = ann.get('id')
        if not aid or self._project is None:
            return
        try:
            from fnt.usv.usv_detector.mad_examples import delete_example
            delete_example(self._project.training_data_dir, aid)
        except Exception:
            pass
        wav = self._active_review_wav_path()
        if wav:
            try:
                from fnt.usv.usv_detector.fnt_mask_store import (
                    masks_sibling_path, td_delete)
                td_delete(masks_sibling_path(wav), aid)
            except Exception:
                pass

    def _pred_describe(self, ann_idx: int) -> str:
        """Short '@ t.tts' description of an annotation for the log."""
        if not (0 <= ann_idx < len(self.spectrogram.annotations)):
            return "?"
        ann = self.spectrogram.annotations[ann_idx]
        if self.sample_rate and self.spectrogram.hop:
            dt = self.spectrogram.hop / float(self.sample_rate)
            return f"@ {ann.get('t0', 0) * dt:.2f}s"
        return "?"

    def _after_review_decision(self, decided_id, was_pending: bool):
        """Tail for accept/reject. Auto-advance fast-forwards to the next pending
        detection ONLY when a pending item was just decided (forward review).
        When the user deliberately re-decides an already-accepted/rejected item,
        the selection stays on it so they can see the change. Only a pending →
        decided transition counts toward the 'all reviewed' tally/prompt."""
        if was_pending:
            self._reviewed_count += 1
        self._refresh_annotation_list()
        if self._auto_advance and was_pending:
            if not self._select_next_pending_after_id(decided_id):
                self._reselect_by_id(decided_id)  # nothing left after — stay
        else:
            self._reselect_by_id(decided_id)  # re-decision / manual: stay put
        if was_pending and not self._pred_indices():
            self.status_bar.showMessage("All predictions reviewed")
            self._maybe_prompt_next_file()

    def _maybe_prompt_next_file(self):
        """When the last pending detection in a file has just been reviewed,
        offer to jump to the next file (Yes is the default, so Enter advances)."""
        if self._pred_indices():
            return  # still pending — nothing to prompt
        n = self._reviewed_count
        if n <= 0:
            return  # nothing was actually reviewed this file — no prompt
        if self._review_mode == 'deploy':
            idx = self._deploy_file_idx
            has_next = idx is not None and idx + 1 < len(self.deploy_files)
        else:
            has_next = self.current_file_idx + 1 < len(self.audio_files)
        base = f"All {n} pending mask(s) reviewed."
        if not has_next:
            self.status_bar.showMessage(base + " (last file)")
            return
        box = QMessageBox(self)
        box.setWindowTitle("Review complete")
        box.setText(base + "\n\nMove to the next file?")
        box.setStandardButtons(QMessageBox.Yes | QMessageBox.No)
        box.setDefaultButton(QMessageBox.Yes)
        if box.exec_() == QMessageBox.Yes:
            self._advance_to_next_review_file()

    def _advance_to_next_review_file(self):
        """Select the next file in the active tab's list (loads it for review)."""
        if self._review_mode == 'deploy':
            nxt = (self._deploy_file_idx or 0) + 1
            if 0 <= nxt < len(self.deploy_files):
                self.deploy_list.setCurrentRow(nxt)
        else:
            nxt = self.current_file_idx + 1
            if 0 <= nxt < len(self.audio_files):
                self.file_list.setCurrentRow(nxt)

    @staticmethod
    def _ann_csv_id(ann: dict):
        """The key joining an annotation to its unified-CSV row: blob_id for
        predictions, the stable example id for hand-labels."""
        return ann.get('blob_id') if ann.get('blob_id') is not None \
            else ann.get('id')

    def _pred_csv_path(self, ann: dict):
        """Resolve the unified per-wav CSV path from the file owning the
        preview, so accept/reject/delete writes land in the right place."""
        if self._ann_csv_id(ann) is None:
            return None
        wav = self._active_review_wav_path()
        if not wav:
            return ann.get('csv_path')
        return pred_csv_sibling_path(wav)

    def _upsert_call_csv_row(self, ann: dict, status: str, score=None):
        """Insert or update one call's row in the unified per-wav CSV (matched
        by blob_id/id). Used for hand-labels and for syncing review decisions."""
        csv_path = self._pred_csv_path(ann)
        bid = self._ann_csv_id(ann)
        if not csv_path or bid is None or self.sample_rate is None:
            return
        sp = self._spec_params()
        sr = self.sample_rate or 1
        dt = (sp['nperseg'] - sp['noverlap']) / float(sr)
        df = (sr / 2.0) / (sp['nfft'] // 2) if sp.get('nfft') else 0.0
        msk = ann.get('mask')
        area = int(msk.sum()) if msk is not None else int(ann.get('area_pixels', 0))
        minf = round(ann['f0'] * df, 2)
        maxf = round(ann['f1'] * df, 2)
        row = {
            'blob_id': bid,
            'class': ann.get('category', '') or '',
            'start_s': round(ann['t0'] * dt, 6),
            'stop_s': round(ann['t1'] * dt, 6),
            'min_freq_hz': minf,
            'max_freq_hz': maxf,
            'area_pixels': area,
            'score': float(score if score is not None else ann.get('score', 1.0)),
            'status': status,
            'source': ann.get('source', 'label'),
        }
        # Carry the full quantification set the annotation computed at confirm.
        from fnt.usv.usv_detector.mad_inference import CALL_METRIC_KEYS
        for k in CALL_METRIC_KEYS:
            if ann.get(k) is not None:
                row[k] = ann.get(k)
        row.setdefault('freq_bandwidth_hz', round(maxf - minf, 2))
        try:
            from fnt.usv.usv_detector.mad_inference import (
                read_blob_csv, write_blob_csv)
            rows = read_blob_csv(csv_path) if os.path.isfile(csv_path) else []
            rows = [r for r in rows if str(r.get('blob_id')) != str(bid)]
            rows.append(row)
            write_blob_csv(csv_path, rows)
        except Exception:
            pass

    def _remove_pred_csv_row(self, ann: dict):
        """Delete a call's row from the unified CSV entirely — Delete leaves no
        trace (the crop is dropped separately). Matched by blob_id/id."""
        csv_path = self._pred_csv_path(ann)
        bid = self._ann_csv_id(ann)
        if not csv_path or bid is None or not os.path.isfile(csv_path):
            return
        try:
            from fnt.usv.usv_detector.mad_inference import (
                read_blob_csv, write_blob_csv)
            rows = [r for r in read_blob_csv(csv_path)
                    if str(r.get('blob_id')) != str(bid)]
            write_blob_csv(csv_path, rows)
        except Exception:
            pass

    def _write_pred_csv_status(self, ann: dict, status: str):
        """Persist an accept/reject decision to the unified CSV (matched by
        blob_id/id). For a hand-label whose row may not exist yet, fall back to
        a full upsert so the decision is always recorded."""
        csv_path = self._pred_csv_path(ann)
        bid = self._ann_csv_id(ann)
        if not csv_path or bid is None or not os.path.isfile(csv_path):
            self._upsert_call_csv_row(ann, status)
            return
        try:
            from fnt.usv.usv_detector.mad_inference import (
                read_blob_csv, write_blob_csv)
            rows = read_blob_csv(csv_path)
            found = False
            for r in rows:
                if str(r.get('blob_id')) == str(bid):
                    r['status'] = status
                    found = True
            if not found:
                self._upsert_call_csv_row(ann, status)
                return
            write_blob_csv(csv_path, rows)
        except Exception:
            pass

    def _skip_current_pred(self):
        """Leave the current detection undecided and jump to the next pending
        one after it (no decision recorded)."""
        if not self._pred_indices():
            return
        self._log("Skip prediction")
        sel = self.spectrogram._selected_ann_idx
        sid = (self.spectrogram.annotations[sel].get('id')
               if sel is not None and 0 <= sel < len(self.spectrogram.annotations)
               else None)
        if not self._select_next_pending_after_id(sid):
            self.status_bar.showMessage("No more pending predictions after this")

    def _accept_prediction(self, ann_idx: int):
        """Accept a prediction annotation — save as training example."""
        sg = self.spectrogram
        if not (0 <= ann_idx < len(sg.annotations)):
            return
        ann = sg.annotations[ann_idx]
        if self.audio_data is None or self._active_wav_path() is None:
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
        self._write_pred_csv_status(ann, 'accepted')
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
        self._snapshot_for_undo("Accept All", crops=False)
        self._log(f"Accept All — {len(preds)} prediction(s) [{self._review_mode}]")
        if self._review_mode == 'deploy':
            # Keep accepted detections visible (blue), recorded in the CSV.
            for ann_idx in preds:
                ann = self.spectrogram.annotations[ann_idx]
                self._write_pred_csv_status(ann, 'accepted')
                ann['status'] = 'accepted'
            self.spectrogram._rebuild_confirmed_mask()
            self.spectrogram.update()
            self._pred_review_idx = None
            self._refresh_annotation_list()
            self._update_pred_review_widgets()
            self.status_bar.showMessage(f"Accepted {len(preds)} prediction(s)")
            self._reviewed_count += len(preds)
            self._maybe_prompt_next_file()
            return
        if self.audio_data is None or self._active_wav_path() is None:
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
                self._write_pred_csv_status(ann, 'accepted')
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
        self._reviewed_count += n
        self._maybe_prompt_next_file()

    def _reject_all_preds(self):
        preds = self._pred_indices()
        if not preds:
            return
        self._snapshot_for_undo("Reject All", crops=False)
        self._log(f"Reject All — {len(preds)} prediction(s) [{self._review_mode}]")
        # Recorded decisions: mark each rejected and keep it visible (red).
        for ann_idx in preds:
            ann = self.spectrogram.annotations[ann_idx]
            self._write_pred_csv_status(ann, 'rejected')
            ann['status'] = 'rejected'
        self.spectrogram.update()
        self._pred_review_idx = None
        self._refresh_annotation_list()
        self._update_pred_review_widgets()
        self.status_bar.showMessage(f"Rejected {len(preds)} prediction(s)")
        self._reviewed_count += len(preds)
        self._maybe_prompt_next_file()

    def _migrate_legacy_prob(self, h5_path, sp, rows=None):
        """One-time upgrade of a legacy ``/prob`` file: read the full grid
        *once*, carve per-blob crops, store them, drop the grid to reclaim
        ~1 GB, and return ``(crops_dict, rows)``.

        If ``rows`` (the sibling CSV's blob rows) are supplied, crops are carved
        from each row's existing box and keyed by its ``blob_id`` — so accept/
        reject status stays joined and the display is identical to before. With
        no CSV (inference-preview files), blobs are re-extracted at 0.5.

        Returns ``({}, None)`` on any failure so callers degrade gracefully.
        """
        from fnt.usv.usv_detector.fnt_mask_store import (
            read_prob, write_pred_masks, delete_prob, read_all_pred_masks,
        )
        try:
            pred_mask = read_prob(h5_path)
        except Exception:
            pred_mask = None
        if pred_mask is None:
            return {}, None
        try:
            if rows:
                dt = (sp['nperseg'] - sp['noverlap']) / float(self.sample_rate)
                df = (self.sample_rate / 2.0) / (sp['nfft'] // 2)
                crops = []
                for r in rows:
                    t0 = int(round(r['start_s'] / dt))
                    t1 = int(round(r['stop_s'] / dt))
                    f0 = int(round(r['min_freq_hz'] / df))
                    f1 = min(int(round(r['max_freq_hz'] / df)),
                             pred_mask.shape[0])
                    t1 = min(t1, pred_mask.shape[1])
                    if t1 <= t0 or f1 <= f0:
                        continue
                    sub = pred_mask[f0:f1, t0:t1] >= 0.5
                    crops.append({'blob_id': r.get('blob_id'),
                                  'mask': np.ascontiguousarray(sub),
                                  'f_off': f0, 't_off': t0})
                out_rows = rows
            else:
                from fnt.usv.usv_detector.mad_inference import (
                    extract_blobs, blobs_to_rows,
                )
                blobs = extract_blobs(pred_mask, 0.5, include_mask=True)
                out_rows = blobs_to_rows(blobs, sp['nperseg'], sp['noverlap'],
                                         sp['nfft'], self.sample_rate)
                crops = [
                    {'blob_id': i, 'mask': b['mask'],
                     'f_off': b['f_low'], 't_off': b['t_start']}
                    for i, b in enumerate(blobs)
                ]
            write_pred_masks(h5_path, crops)
            delete_prob(h5_path)  # reclaim ~1 GB; predictions now live as crops
            self._log(f"Upgraded {os.path.basename(h5_path)} to fast per-blob "
                      f"crops ({len(crops)} call(s)); dropped legacy prob grid")
            return read_all_pred_masks(h5_path), out_rows
        except Exception:
            return {}, None

    def _load_predictions_as_annotations(self, wav: Optional[str] = None):
        """Load inference predictions for ``wav`` (default: current training
        file) and turn each blob into a yellow prediction annotation.

        Fast path: read the small per-blob mask crops from the sibling h5
        (``/pred_calls``) and join them to the CSV blob rows by ``blob_id``.
        The multi-GB ``/prob`` grid is never read; a legacy prob-only file is
        migrated to crops once on first open (see :meth:`_migrate_legacy_prob`).
        """
        if wav is None:
            wav = self._active_wav_path()  # session OR training file
            if wav is None:
                return
        self._reviewed_count = 0  # fresh file → reset the reviewed tally
        self._undo_stack = []     # undo history is per-file
        sp = self._spec_params()
        if self.sample_rate is None:
            return
        dt = (sp['nperseg'] - sp['noverlap']) / float(self.sample_rate)
        df = (self.sample_rate / 2.0) / (sp['nfft'] // 2)

        from fnt.usv.usv_detector.fnt_mask_store import (
            masks_sibling_path, read_all_pred_masks, has_pred_masks,
            has_prob, get_prob_blob_count, set_prob_blob_count,
        )
        h5_path = masks_sibling_path(wav)

        csv_path = pred_csv_sibling_path(wav)
        rows = None
        if os.path.isfile(csv_path):
            try:
                from fnt.usv.usv_detector.mad_inference import read_blob_csv
                rows = read_blob_csv(csv_path)
            except Exception:
                pass

        # Cheap known-empty short-circuit: nothing to load and nothing to migrate.
        if (not rows and not has_pred_masks(h5_path)
                and get_prob_blob_count(h5_path) == 0):
            return

        # Per-blob crops (few MB). Migrate a legacy prob file once, carving
        # crops from the CSV's blob boxes when a CSV is present so blob_ids and
        # accept/reject status stay joined.
        crops = read_all_pred_masks(h5_path)
        if not crops and has_prob(h5_path):
            crops, migrated_rows = self._migrate_legacy_prob(h5_path, sp, rows)
            if rows is None:
                rows = migrated_rows

        # CSV absent but crops present (e.g. preview migrated without a CSV):
        # synthesize minimal rows straight from the crops.
        if not rows and crops:
            rows = []
            for bid, c in crops.items():
                h, w = c['mask'].shape
                f_off, t_off = c['f_off'], c['t_off']
                rows.append({
                    'blob_id': int(bid) if str(bid).isdigit() else bid,
                    'start_s': t_off * dt, 'stop_s': (t_off + w) * dt,
                    'min_freq_hz': f_off * df, 'max_freq_hz': (f_off + h) * df,
                    'area_pixels': int(c['mask'].sum()), 'score': 0.0,
                    'status': 'pending',
                })

        if not rows:
            return

        sg = self.spectrogram
        # Re-load review detections from the CSV — drop existing ones first so we
        # don't duplicate. In deploy mode accepted detections live in the CSV
        # too (restored blue here); in train mode they come from the example
        # store, so we leave those (status-less) confirmed annotations alone.
        in_deploy = self._review_mode == 'deploy'
        strip = (('prediction', 'rejected', 'accepted') if in_deploy
                 else ('prediction', 'rejected'))
        sg.annotations = [a for a in sg.annotations
                          if a.get('status') not in strip]
        # Hand-labels live in BOTH the h5 (loaded above as confirmed) and the
        # unified CSV (as 'accepted' rows). Skip any CSV row already represented
        # by a confirmed annotation so it isn't shown twice.
        confirmed_ids = {str(a.get('id')) for a in sg.annotations
                         if a.get('id') is not None}
        wav_name = os.path.basename(wav)
        n_added = 0
        n_pending = 0
        for r in rows:
            st = r.get('status')
            if st == 'deleted':
                continue  # removed entirely — no trace
            if str(r.get('blob_id')) in confirmed_ids:
                continue  # already loaded from the h5 as a confirmed label
            if st == 'accepted':
                # Train mode loads confirmed from the example store; only deploy
                # restores accepted detections (blue) from the CSV.
                if not in_deploy:
                    continue
                ann_status = 'accepted'
            elif st == 'rejected':
                ann_status = 'rejected'
            else:
                ann_status = 'prediction'
            crop = crops.get(str(r.get('blob_id')))
            if crop is not None:
                # Exact stored shape/offset — no rounding, no grid read.
                blob_region = crop['mask']
                f0, t0 = crop['f_off'], crop['t_off']
                f1 = f0 + blob_region.shape[0]
                t1 = t0 + blob_region.shape[1]
            else:
                # No crop (e.g. CSV-only legacy) — fall back to the box rect.
                t0 = int(round(r['start_s'] / dt))
                t1 = int(round(r['stop_s'] / dt))
                f0 = int(round(r['min_freq_hz'] / df))
                f1 = int(round(r['max_freq_hz'] / df))
                if t1 <= t0 or f1 <= f0:
                    continue
                blob_region = np.ones((f1 - f0, t1 - t0), dtype=bool)
            if not blob_region.any():
                continue
            sg.annotations.append({
                'id': f'pred_{n_added}',
                'category': (self._project.last_class if self._project else
                             self._session_last_class) or 'USV',
                'f0': f0, 'f1': f1, 't0': t0, 't1': t1,
                'mask': np.ascontiguousarray(blob_region),
                'status': ann_status,
                'source_wav': wav_name,
                'score': r.get('score', 0),
                'blob_id': r.get('blob_id'),
                'csv_path': csv_path if os.path.isfile(csv_path) else None,
            })
            n_added += 1
            if ann_status == 'prediction':
                n_pending += 1
        # Persist the *pending* count as the authoritative scalar so the file
        # list (which counts non-rejected CSV rows) matches what's reviewable.
        try:
            set_prob_blob_count(h5_path, n_pending)
        except Exception:
            pass
        sg._rebuild_confirmed_mask()
        sg.update()
        self._pred_review_idx = 0 if n_pending > 0 else None
        self._refresh_annotation_list()
        self._update_pred_review_widgets()
        if n_pending:
            self._jump_to_pred(0)
        if n_added:
            self._log(f"Loaded {n_pending} pending + {n_added - n_pending} "
                      f"rejected for {os.path.basename(str(wav))}")
        return n_pending

    # --- inline training/inference handlers ---------------------------
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

    def _apply_latest_training_config(self):
        """Pre-fill the training-option widgets from the most recent model's
        saved config (each run writes ``training_summary.json`` with a
        ``config`` block), so a new run starts from the same settings."""
        if self._project is None or not hasattr(self, 'combo_arch'):
            return
        path = self._latest_model_path()
        if not path:
            return
        summary = os.path.join(os.path.dirname(path), 'training_summary.json')
        if not os.path.isfile(summary):
            return
        try:
            import json
            with open(summary) as f:
                cfg = (json.load(f) or {}).get('config', {}) or {}
        except Exception:
            return
        if not cfg:
            return

        def _set_combo_data(combo, val):
            i = combo.findData(val)
            if i >= 0:
                combo.setCurrentIndex(i)

        def _set_combo_text(combo, val):
            if val is None:
                return
            i = combo.findText(str(val))
            if i >= 0:
                combo.setCurrentIndex(i)

        try:
            if cfg.get('model_arch') is not None:
                _set_combo_data(self.combo_arch, cfg['model_arch'])
            _set_combo_text(self.combo_train_encoder, cfg.get('encoder_name'))
            if cfg.get('n_epochs') is not None:
                self.spin_train_epochs.setValue(int(cfg['n_epochs']))
            if cfg.get('early_stop_patience') is not None:
                self.spin_train_patience.setValue(int(cfg['early_stop_patience']))
            if cfg.get('batch_size') is not None:
                self.spin_train_batch.setValue(int(cfg['batch_size']))
            if cfg.get('learning_rate') is not None:
                self.spin_train_lr.setValue(float(cfg['learning_rate']))
            if cfg.get('val_fraction') is not None:
                self.spin_train_val.setValue(float(cfg['val_fraction']))
            _set_combo_text(self.combo_train_device, cfg.get('device'))
            # Keep the encoder picker's enabled state in sync with HRNet.
            self._on_arch_changed()
        except Exception as e:
            self._log(f"Could not apply latest training config: {e}")

    def _on_inline_train(self):
        if self._project is None:
            return
        # Already training? The Train button doubles as "re-show the graph".
        if self._training_active:
            self._show_training_dialog()
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
        # One-time nudge if a GPU is present but PyTorch can't use it.
        self._show_gpu_setup_dialog(force=False)
        cfg = self._build_inline_train_config()
        # If the user ticked a Run Inference target, auto-run those settings on
        # the freshly-trained model when training finishes (the "Run Training +
        # Inference" path / active-learning loop). With no target ticked this is
        # "Run Training" — training only. Confirm the overwrite up front so the
        # run proceeds unattended; declining just skips the auto-inference.
        infer_wavs = self._gather_inference_targets()
        if infer_wavs and not self._confirm_overwrite_predictions(infer_wavs):
            infer_wavs = []
            self._log("Post-training inference skipped (declined overwrite)")
        self._post_train_infer_wavs = infer_wavs
        self._log(f"Train START — {n_examples} examples, arch="
                  f"{cfg.model_arch}, encoder={cfg.encoder_name}, "
                  f"epochs={cfg.n_epochs}, batch={cfg.batch_size}")
        # The action button doubles as "re-show the graph" while the run is live.
        self._training_active = True
        self._update_run_button()
        self._set_train_config_enabled(False)  # lock config while running
        self._show_training_dialog()
        self.train_panel.start_run()
        self._start_training(cfg, post_inference_wavs=infer_wavs,
                             reporter=self.train_panel)

    def _consolidate_sibling_examples(self):
        """Rebuild the project's consolidated ``training_data.h5`` from the
        Training Data list's per-wav sibling examples — the source of truth.

        Non-destructive to the siblings (the central store is just a training
        cache), and rebuilt fresh each time so removed/edited labels never
        linger. A file therefore only trains the model once it's been copied
        into Training Data."""
        if self._project is None:
            return
        from fnt.usv.usv_detector.fnt_mask_store import (
            masks_sibling_path, td_iter_examples, td_count,
        )
        from fnt.usv.usv_detector.mad_examples import save_example, _store_path
        td_dir = self._project.training_data_dir
        os.makedirs(td_dir, exist_ok=True)
        store = _store_path(td_dir)
        try:
            if os.path.isfile(store):
                os.remove(store)
        except Exception as e:
            self._log(f"could not reset training store: {e}")
        n = 0
        for fp in self.deploy_files:
            h5 = masks_sibling_path(fp)
            if td_count(h5) == 0:
                continue
            for ex in list(td_iter_examples(h5)):
                meta = ex['meta']
                eid = meta.get('id', '')
                try:
                    save_example(td_dir, ex['spec'], ex['mask'], meta, eid)
                    n += 1
                except Exception:
                    continue
        self._log(f"Built training store from {n} Training Data label(s)")

    def _show_training_dialog(self):
        """Show (or re-show) the floating training-graph window. The spectrogram
        in the main window stays fully visible/usable while a run is active."""
        if self._train_dialog is None:
            self._train_dialog = MADTrainGraphDialog(self)
            lay = QVBoxLayout(self._train_dialog)
            lay.setContentsMargins(6, 6, 6, 6)
            self.train_panel.setParent(self._train_dialog)
            self.train_panel.setVisible(True)
            lay.addWidget(self.train_panel)
        self._train_dialog.show()
        self._train_dialog.raise_()
        self._train_dialog.activateWindow()

    def _on_train_dialog_close(self, event):
        """Closing the training window mid-run: keep it in the background (just
        hide) or stop the run — never silently kill a long run."""
        if not self._training_active:
            event.accept()
            return
        box = QMessageBox(self)
        box.setWindowTitle("Training in progress")
        box.setText("A training run is still in progress.")
        box.setInformativeText("Keep it running in the background, or stop it?")
        keep = box.addButton("Keep running", QMessageBox.AcceptRole)
        stop = box.addButton("Stop training", QMessageBox.DestructiveRole)
        cancel = box.addButton("Cancel", QMessageBox.RejectRole)
        box.setDefaultButton(keep)
        box.exec_()
        clicked = box.clickedButton()
        if clicked is cancel:
            event.ignore()
            return
        if clicked is stop:
            try:
                self.train_panel._on_stop()
            except Exception:
                pass
            event.accept()  # hides the window; the run winds down
            return
        # Keep running → just hide the window; reopen via the Train button.
        event.accept()
        self.status_bar.showMessage(
            "Training continues in the background — click "
            "'Training… (show graph)' to reopen the graph.")

    # Kept as a no-op: the graph is now its own window, so there's nothing to
    # swap back when the user clicks a file/detection or changes tabs.
    def _dismiss_training_view(self):
        return

    def _clear_preview(self):
        """Blank the shared spectrogram/waveform canvas (no file shown) and
        clear its detections. Used when entering the Inference tab without a
        deploy file selected, so the Label tab's preview doesn't linger."""
        self._stop_playback()
        self.audio_data = None
        self.sample_rate = None
        self.spectrogram.mask = None
        self.spectrogram.set_audio_data(None, None)
        self.waveform_overview.set_audio_data(None, None)
        self.spectrogram.annotations.clear()
        self.spectrogram._selected_ann_idx = None
        self._pred_review_idx = None
        self.spectrogram.update()
        self._refresh_annotation_list()

    def _on_training_finished(self, ok: bool):
        self._training_active = False
        self._set_train_config_enabled(True)   # unlock config controls
        self._update_run_button()  # restore the action-button label/enabled
        if self._train_dialog is not None:
            self._train_dialog.setWindowTitle(
                "Segmentation Training — complete (close when ready)")
        self._refresh_quick_infer_models()
        if not ok or self._project is None:
            return
        if getattr(self, '_post_train_infer_wavs', None):
            self.infer_panel.run_finished.connect(
                self._post_training_cleanup_after_infer)
        else:
            QTimer.singleShot(300, self._post_training_cleanup_dialog)

    def _post_training_cleanup_after_infer(self, _ok: bool):
        """Slot: fires once after post-training inference finishes."""
        try:
            self.infer_panel.run_finished.disconnect(
                self._post_training_cleanup_after_infer)
        except TypeError:
            pass
        self._scan_all_file_counts()
        QTimer.singleShot(300, self._post_training_cleanup_dialog)

    def _post_training_cleanup_dialog(self):
        """After training, offer to remove files with no labels or only
        pending annotations.  Skipped if every file has confirmed labels."""
        if self._project is None or not self.audio_files:
            return
        from fnt.usv.usv_detector.fnt_mask_store import masks_sibling_path, td_count
        no_annotations: list = []
        pending_only: list = []
        for fp in self.audio_files:
            has_confirmed = False
            try:
                n = td_count(masks_sibling_path(fp))
                if n > 0:
                    has_confirmed = True
            except Exception:
                pass
            if not has_confirmed and self._project is not None:
                try:
                    from fnt.usv.usv_detector.mad_examples import count_examples
                    # quick per-file check not available; use cache
                    pass
                except Exception:
                    pass
            if has_confirmed:
                continue
            has_preds = os.path.isfile(pred_csv_sibling_path(fp))
            if not has_preds:
                # Cheap metadata checks only — never decompress a pixel grid.
                try:
                    from fnt.usv.usv_detector.fnt_mask_store import (
                        has_pred_masks, get_prob_blob_count,
                    )
                    h5 = masks_sibling_path(fp)
                    if has_pred_masks(h5) or (get_prob_blob_count(h5) or 0) > 0:
                        has_preds = True
                except Exception:
                    pass
            if has_preds:
                pending_only.append(fp)
            else:
                no_annotations.append(fp)

        if not no_annotations and not pending_only:
            return

        dlg = QDialog(self)
        dlg.setWindowTitle("Clean Up Project Files")
        layout = QVBoxLayout(dlg)
        layout.addWidget(QLabel(
            "Some recordings have no confirmed labels.\n"
            "Would you like to remove them from the project?"
        ))
        chk_no_ann = None
        chk_pending = None
        if no_annotations:
            chk_no_ann = QCheckBox(
                f"Remove {len(no_annotations)} file(s) with no annotations")
            chk_no_ann.setChecked(True)
            layout.addWidget(chk_no_ann)
        if pending_only:
            chk_pending = QCheckBox(
                f"Remove {len(pending_only)} file(s) with only pending predictions")
            chk_pending.setChecked(False)
            layout.addWidget(chk_pending)
        btn_row = QHBoxLayout()
        btn_ok = QPushButton("Remove Selected")
        btn_cancel = QPushButton("Keep All")
        btn_row.addWidget(btn_ok)
        btn_row.addWidget(btn_cancel)
        layout.addLayout(btn_row)
        btn_ok.clicked.connect(dlg.accept)
        btn_cancel.clicked.connect(dlg.reject)
        if dlg.exec_() != QDialog.Accepted:
            return
        to_remove: list = []
        if chk_no_ann and chk_no_ann.isChecked():
            to_remove.extend(no_annotations)
        if chk_pending and chk_pending.isChecked():
            to_remove.extend(pending_only)
        if not to_remove:
            return
        self._remove_files_by_path(to_remove)

    def _remove_files_by_path(self, paths_to_remove: list):
        """Unload files from the session list. Does NOT delete anything from
        disk — sources and their sibling csv/h5 are left untouched."""
        self._auto_save_mask_if_dirty()
        current_path = (self.audio_files[self.current_file_idx]
                        if self.audio_files else None)
        remove_set = set(paths_to_remove)
        self.audio_files = [p for p in self.audio_files if p not in remove_set]
        if self._project is not None:
            self._project.audio_files = [
                p for p in self._project.audio_files if p not in remove_set
            ]
            try:
                self._project.save()
            except Exception:
                pass
        # Fix the session index.
        if not self.audio_files:
            self.current_file_idx = 0
        elif current_path in remove_set:
            self.current_file_idx = min(
                self.current_file_idx, len(self.audio_files) - 1)
        else:
            try:
                self.current_file_idx = self.audio_files.index(current_path)
            except ValueError:
                self.current_file_idx = 0
        self._refresh_file_list()
        # Only disturb the preview when the session list currently owns it.
        if self._active_source == 'session':
            self.file_list.blockSignals(True)
            self.file_list.setCurrentRow(self.current_file_idx
                                         if self.audio_files else -1)
            self.file_list.blockSignals(False)
            if not self.audio_files:
                self.spectrogram.set_audio_data(None, None)
                self.waveform_overview.set_audio_data(None, None)
                self.spectrogram.annotations.clear()
                self._refresh_annotation_list()
            elif current_path in remove_set:
                self._load_current_file()
        self._update_project_state()
        self.btn_remove_files.setEnabled(bool(self.audio_files))
        self._update_scope_labels()
        n = len(paths_to_remove)
        self.status_bar.showMessage(f"Unloaded {n} file(s)")
        self._log(f"Unloaded {n} file(s) from the session")

    def _default_model_path(self) -> Optional[str]:
        if self._project and self._project.models:
            last = self._project.models[-1]
            return last.get('path') if isinstance(last, dict) else str(last)
        return None

    # --- deployment: model dropdown -----------------------------------
    def _refresh_deploy_models(self, select_latest: bool = False):
        """Populate the model dropdown from <project>/models/ (dirs with a
        weights.pt). With ``select_latest`` (the Refresh button) jump to the
        newest model trained in the current project; otherwise preserve the
        current selection where possible."""
        if not hasattr(self, 'combo_deploy_model'):
            return
        prev = self._selected_deploy_model_path()
        self.combo_deploy_model.blockSignals(True)
        self.combo_deploy_model.clear()
        found = []
        seen_paths: set = set()

        def _scan_models_dir(models_root, tag=''):
            if not os.path.isdir(models_root):
                return
            for name in sorted(os.listdir(models_root)):
                d = os.path.join(models_root, name)
                w = os.path.join(d, 'weights.pt')
                if os.path.isdir(d) and os.path.isfile(w):
                    if w not in seen_paths:
                        seen_paths.add(w)
                        label = f"{tag}{name}" if tag else name
                        found.append((label, w))

        if self._project is not None:
            _scan_models_dir(
                os.path.join(self._project.project_dir, 'models'))
        if (self._infer_model_project_dir and
                (self._project is None or
                 os.path.normpath(self._infer_model_project_dir) !=
                 os.path.normpath(self._project.project_dir))):
            proj_name = os.path.basename(
                os.path.normpath(self._infer_model_project_dir))
            _scan_models_dir(
                os.path.join(self._infer_model_project_dir, 'models'),
                tag=f"[{proj_name}] ")

        for name, w in found:
            self.combo_deploy_model.addItem(name, w)
        # Choose the selection: Refresh → newest model in this project; else
        # keep the prior pick if it survived the rescan; else fall back to last.
        target = self._latest_project_model_index() if select_latest else None
        if target is None and prev:
            for i in range(self.combo_deploy_model.count()):
                if self.combo_deploy_model.itemData(i) == prev:
                    target = i
                    break
        if target is None and found:
            target = self.combo_deploy_model.count() - 1
        if target is not None:
            self.combo_deploy_model.setCurrentIndex(target)
        self.combo_deploy_model.blockSignals(False)
        self._on_deploy_model_changed()
        if select_latest and target is not None:
            self.status_bar.showMessage(
                f"Loaded latest model: {self.combo_deploy_model.currentText()}")

    def _latest_project_model_index(self) -> Optional[int]:
        """Combo index of the newest model in the *current* project. Model dirs
        are timestamp-named and added in sorted order, so the last entry whose
        path lives under the project's models/ is the latest. None if none."""
        if self._project is None:
            return None
        proj_models = os.path.normpath(
            os.path.join(self._project.project_dir, 'models'))
        latest = None
        for i in range(self.combo_deploy_model.count()):
            p = self.combo_deploy_model.itemData(i)
            if p and os.path.normpath(p).startswith(proj_models + os.sep):
                latest = i
        return latest

    def _browse_deploy_model(self):
        root = (self._infer_model_project_dir
                or (os.path.join(self._project.project_dir, 'models')
                    if self._project else None)
                or os.path.expanduser("~"))
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

    def _browse_infer_project(self):
        start = (self._infer_model_project_dir
                 or (self._project.project_dir if self._project else None)
                 or os.path.expanduser("~"))
        folder = QFileDialog.getExistingDirectory(
            self, "Select a MAD project folder", start)
        if not folder:
            return
        models_root = os.path.join(folder, 'models')
        if not os.path.isdir(models_root):
            QMessageBox.warning(
                self, "No models found",
                f"No models/ directory found in:\n{folder}\n\n"
                "Make sure you select a MAD project folder that "
                "contains trained models.",
            )
            return
        self._infer_model_project_dir = folder
        self._refresh_deploy_models()
        proj_name = os.path.basename(os.path.normpath(folder))
        self.status_bar.showMessage(
            f"Loaded models from project: {proj_name}")

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
        self._update_infer_run_enabled()

    def _update_infer_run_enabled(self):
        """Back-compat alias — the single action button is driven by
        :meth:`_update_run_button` now."""
        self._update_run_button()

    def _gather_inference_targets(self) -> List[str]:
        """Wav paths to run inference on, from the Session/Training scope
        checkboxes (All vs Current file each). De-duplicated, order preserved."""
        if not hasattr(self, 'chk_scope_session'):
            return []
        targets: List[str] = []
        if self.chk_scope_session.isChecked():
            if self.combo_scope_session.currentText().startswith("Current"):
                if (self.audio_files and
                        0 <= self.current_file_idx < len(self.audio_files)):
                    targets.append(self.audio_files[self.current_file_idx])
            else:
                targets.extend(self.audio_files)
        if self.chk_scope_training.isChecked():
            if self.combo_scope_training.currentText().startswith("Current"):
                if (self.deploy_files and self._deploy_file_idx is not None and
                        0 <= self._deploy_file_idx < len(self.deploy_files)):
                    targets.append(self.deploy_files[self._deploy_file_idx])
            else:
                targets.extend(self.deploy_files)
        seen, out = set(), []
        for p in targets:
            if p and p not in seen:
                seen.add(p)
                out.append(p)
        return out

    # --- Training Data list (project recordings the model trains on) -----
    def _file_call_count(self, fp) -> Optional[int]:
        """Total masks recorded for a wav — every call in the unified CSV,
        regardless of source (label/prediction) or status. Falls back to the h5
        (confirmed /td + predicted /pred_calls) when there's no CSV. Returns
        None if the file has no masks. Shared by the session + Training lists."""
        from fnt.usv.usv_detector.fnt_mask_store import (
            masks_sibling_path, get_prob_blob_count, td_count,
        )
        from fnt.usv.usv_detector.mad_inference import read_blob_csv
        csv_p = pred_csv_sibling_path(fp)
        if os.path.isfile(csv_p):
            try:
                return len(read_blob_csv(csv_p)) or None
            except Exception:
                pass
        try:
            h5 = masks_sibling_path(fp)
            n = (td_count(h5) or 0) + (get_prob_blob_count(h5) or 0)
            return n or None
        except Exception:
            return None

    def _refresh_deploy_queue(self):
        """Render the Training Data list, showing a green ✓ + the recorded call
        count (labels + predictions) on each file that has any."""
        if not hasattr(self, 'deploy_list'):
            return
        self.deploy_list.blockSignals(True)
        self.deploy_list.clear()
        for p in self.deploy_files:
            base = os.path.basename(p)
            cnt = self._file_call_count(p)
            if cnt:
                item = QListWidgetItem(f"✓  {base}  ({cnt})")
                item.setForeground(QColor(80, 200, 120))
                item.setToolTip("Has labeled/predicted calls — click to review")
            else:
                item = QListWidgetItem(base)
            self.deploy_list.addItem(item)
        self.deploy_list.blockSignals(False)
        self.lbl_deploy_queue.setText(
            f"{len(self.deploy_files)} file(s) in training set")
        self._update_infer_run_enabled()
        self._update_train_button_count()

    def _set_deploy_item_state(self, wav_path, state: str, count=None):
        """Mark a queue row by inference state: 'pending' | 'done' | 'error'.
        Finished files get a green ✓ and the detection count so the user can see
        how many calls were found and click in to QC them as the run progresses."""
        if not hasattr(self, 'deploy_list'):
            return
        try:
            row = self.deploy_files.index(wav_path)
        except ValueError:
            return
        item = self.deploy_list.item(row)
        if item is None:
            return
        base = os.path.basename(wav_path)
        if state == 'done':
            tag = f"  ({count})" if count is not None else ""
            item.setText(f"✓  {base}{tag}")
            item.setForeground(QColor(80, 200, 120))
            item.setToolTip(
                f"Inference done — {count} detection(s); click to review"
                if count is not None else
                "Inference done — click to review detections")
        elif state == 'error':
            item.setText(f"✗  {base}")
            item.setForeground(QColor(220, 90, 90))
            item.setToolTip("Inference failed")
        else:  # pending
            item.setText(base)
            item.setData(Qt.ForegroundRole, None)
            item.setToolTip("")

    def _on_deploy_file_selected(self, row: int):
        if 0 <= row < len(self.deploy_files):
            # Training Data owns the preview now; clear the session selection.
            self._active_source = 'training'
            self._review_mode = 'train'
            self._clear_session_selection()
            self._deploy_file_idx = row
            self._load_deploy_file(self.deploy_files[row])
            self._update_review_buttons_for_source()

    def _clear_deploy_files(self):
        """Empty the Training Data list (used on project close)."""
        self.deploy_files = []
        self._deploy_file_idx = None
        self._refresh_deploy_queue()

    def _prompt_make_project_for_training(self) -> bool:
        """Offer to create or open a project so there's somewhere to put the
        Training Data set. Returns True if a project is open afterward."""
        box = QMessageBox(self)
        box.setIcon(QMessageBox.Question)
        box.setWindowTitle("Project needed")
        box.setText("Training Data is stored inside a MAD project.")
        box.setInformativeText(
            "Create a new project or open an existing one to hold the "
            "training set?")
        b_new = box.addButton("New Project…", QMessageBox.AcceptRole)
        b_open = box.addButton("Open Project…", QMessageBox.AcceptRole)
        box.addButton("Cancel", QMessageBox.RejectRole)
        box.exec_()
        clicked = box.clickedButton()
        if clicked is b_new:
            self._menu_new_project()
        elif clicked is b_open:
            self._menu_open_project()
        return self._project is not None

    def _copy_to_training_data(self):
        """Copy the selected Session-audio file(s) — wav + sibling csv/h5 — into
        the project's Training Data set (recordings/). Independent snapshots:
        re-copying a name already present prompts before overwriting."""
        # Capture the selection up front — creating/opening a project below may
        # rebuild the session list, but the source files stay put on disk.
        rows = sorted({i.row() for i in self.file_list.selectedIndexes()})
        srcs = [self.audio_files[r] for r in rows
                if 0 <= r < len(self.audio_files)]
        if not srcs:
            self.status_bar.showMessage("Select session file(s) to copy first")
            return
        if not self._project:
            if not self._prompt_make_project_for_training():
                return
            # Re-load the session files (closing the old project cleared the
            # list) so they stay visible after the copy.
            self._append_audio_paths(srcs)
        dest_dir = self._project.recordings_dir
        os.makedirs(dest_dir, exist_ok=True)
        copied, overwrite_all, skip_all = 0, False, False
        for src in srcs:
            base = os.path.basename(src)
            dest = os.path.join(dest_dir, base)
            if os.path.exists(dest) and os.path.normpath(dest) != os.path.normpath(src):
                if skip_all:
                    continue
                if not overwrite_all:
                    box = QMessageBox(self)
                    box.setIcon(QMessageBox.Question)
                    box.setWindowTitle("Already in Training Data")
                    box.setText(
                        f"“{base}” is already in the Training Data set.\n\n"
                        "Overwrite the training copy (wav + csv/h5) with the "
                        "current session version?")
                    bt_yes = box.addButton("Overwrite", QMessageBox.YesRole)
                    bt_all = box.addButton("Overwrite All", QMessageBox.YesRole)
                    bt_skip = box.addButton("Skip", QMessageBox.NoRole)
                    box.addButton("Skip All", QMessageBox.NoRole)
                    box.exec_()
                    clicked = box.clickedButton()
                    if clicked is bt_skip:
                        continue
                    if clicked is bt_all:
                        overwrite_all = True
                    elif clicked is not bt_yes:  # Skip All
                        skip_all = True
                        continue
            self._copy_one_to_training(src, dest)
            if dest not in self.deploy_files:
                self.deploy_files.append(dest)
            copied += 1
        if copied:
            self.deploy_files.sort(key=lambda p: os.path.basename(p).lower())
            self._refresh_deploy_queue()
            self._log(f"Copied {copied} file(s) to Training Data")
            self.status_bar.showMessage(
                f"Copied {copied} file(s) to Training Data")
            self._update_train_button_enabled()

    def _copy_one_to_training(self, src: str, dest: str):
        """Copy a wav and its sibling csv/h5 from ``src`` to ``dest``."""
        import shutil
        from fnt.usv.usv_detector.fnt_mask_store import masks_sibling_path
        if os.path.normpath(src) != os.path.normpath(dest):
            shutil.copy2(src, dest)
        # Prediction CSV.
        for src_side, dest_side in (
            (pred_csv_sibling_path(src), pred_csv_sibling_path(dest)),
            (masks_sibling_path(src), masks_sibling_path(dest)),
        ):
            try:
                if (os.path.isfile(src_side) and
                        os.path.normpath(src_side) != os.path.normpath(dest_side)):
                    shutil.copy2(src_side, dest_side)
            except Exception as e:
                self._log(f"copy sibling failed ({os.path.basename(src_side)}): {e}")

    def _remove_from_training_data(self):
        """Delete the selected recording(s) from the project's Training Data —
        the copied wav + its csv/h5. Leaves the original session source alone."""
        rows = sorted({i.row() for i in self.deploy_list.selectedIndexes()},
                      reverse=True)
        victims = [self.deploy_files[r] for r in rows
                   if 0 <= r < len(self.deploy_files)]
        if not victims:
            return
        from fnt.usv.usv_detector.fnt_mask_store import masks_sibling_path
        names = "\n".join(f"   • {os.path.basename(v)}" for v in victims[:10])
        reply = QMessageBox.question(
            self, "Remove from Training Data",
            f"Delete {len(victims)} recording(s) from the project's Training "
            f"Data set?\n\n{names}\n\n"
            "This removes the copied wav + csv/h5 from the project. The "
            "original session files are not touched.",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply != QMessageBox.Yes:
            return
        for v in victims:
            for path in (v, pred_csv_sibling_path(v), masks_sibling_path(v)):
                try:
                    if os.path.isfile(path):
                        os.remove(path)
                except Exception as e:
                    self._log(f"remove failed ({os.path.basename(path)}): {e}")
            try:
                self.deploy_files.remove(v)
            except ValueError:
                pass
        self._deploy_file_idx = None
        self._refresh_deploy_queue()
        self._update_train_button_enabled()
        self._log(f"Removed {len(victims)} file(s) from Training Data")

    def _file_source_label(self, w) -> str:
        """'Session' / 'Training Data' / '' — which list a wav belongs to."""
        wn = os.path.normpath(w)
        if any(os.path.normpath(p) == wn for p in self.deploy_files):
            return "Training Data"
        if any(os.path.normpath(p) == wn for p in self.audio_files):
            return "Session"
        return ""

    def _file_has_predictions(self, w) -> bool:
        """True only if the wav carries actual model *predictions* (not just
        hand-labels): prediction mask crops in the h5, or prediction-source rows
        in the unified CSV. Label-only files return False."""
        from fnt.usv.usv_detector.fnt_mask_store import (
            masks_sibling_path, has_pred_masks,
        )
        if has_pred_masks(masks_sibling_path(w)):
            return True
        csv = pred_csv_sibling_path(w)
        if os.path.isfile(csv):
            try:
                from fnt.usv.usv_detector.mad_inference import read_blob_csv
                for r in read_blob_csv(csv):
                    if (r.get('source') == 'prediction'
                            or isinstance(r.get('blob_id'), int)):
                        return True
            except Exception:
                pass
        return False

    def _confirm_overwrite_predictions(self, wavs) -> bool:
        """If any of ``wavs`` already carry model predictions, warn that
        re-running inference replaces them and resets the accept/reject/delete
        decisions made on those predictions. Hand-labels are NOT affected, so
        label-only files don't trigger the prompt. Returns True to proceed."""
        existing = [w for w in wavs if self._file_has_predictions(w)]
        if not existing:
            return True

        def _line(w):
            src = self._file_source_label(w)
            tag = f"  [{src}]" if src else ""
            return f"   • {os.path.basename(w)}{tag}"
        shown = "\n".join(_line(w) for w in existing[:8])
        more = (f"\n   …and {len(existing) - 8} more"
                if len(existing) > 8 else "")
        reply = QMessageBox.warning(
            self, "Overwrite existing predictions?",
            f"{len(existing)} of {len(wavs)} file(s) already carry model "
            f"predictions from a previous inference run:\n\n{shown}{more}\n\n"
            "Re-running inference on those files will:\n"
            "   • replace their predictions with fresh ones, and\n"
            "   • reset every Accept / Reject / Delete decision you made on "
            "those predictions back to pending.\n\n"
            "Your confirmed (painted / SAM) labels are NOT affected — they're "
            "kept, and 'Preserve painted labels' additionally skips "
            "re-detecting over them. Files with no predictions are untouched.\n\n"
            "Continue?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No,
        )
        return reply == QMessageBox.Yes

    def _on_deploy_infer(self):
        model = self._selected_deploy_model_path() or self._default_model_path()
        if not model or not os.path.isfile(model):
            QMessageBox.warning(
                self, "Model missing",
                "Select a trained model (or train one) before running inference."
            )
            return
        wavs = self._gather_inference_targets()
        if not wavs:
            QMessageBox.warning(
                self, "No target files",
                "Tick 'Session audio' and/or 'Training data' (and make sure "
                "the chosen set isn't empty) before running inference."
            )
            return
        # Warn before clobbering files that already hold detections from a prior
        # run — re-inferring rewrites their predictions and discards any review
        # decisions (accept/reject/delete) made on them.
        if not self._confirm_overwrite_predictions(wavs):
            return
        # One-time nudge if a GPU is present but PyTorch can't use it.
        self._show_gpu_setup_dialog(force=False)
        from fnt.usv.usv_detector.mad_inference import MADInferenceConfig
        cfg = MADInferenceConfig(
            model_path=model,
            threshold=self.spin_infer_threshold.value(),
            min_blob_pixels=self.spin_infer_min_blob.value(),
            device=self.combo_infer_device.currentText(),
            save_blob_csv=True,  # always — it's the summary output + review state
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
            # Predictions live in the sibling CSV/h5 — load them whether or not
            # a project is open. Use the grid params recorded in the h5 (exactly
            # what the predictions were computed with) so the masks align; fall
            # back to the project's / default params when the h5 has none.
            from fnt.usv.usv_detector.fnt_mask_store import (
                masks_sibling_path, get_grid_attrs,
            )
            grid = get_grid_attrs(masks_sibling_path(filepath))
            sp = self._spec_params()
            self.spectrogram.init_mask(
                audio_len=len(self.audio_data), sample_rate=self.sample_rate,
                nperseg=int(grid.get('nperseg', sp['nperseg'])),
                noverlap=int(grid.get('noverlap', sp['noverlap'])),
                nfft=int(grid.get('nfft', sp['nfft'])),
            )
            # Show this Training-Data file's confirmed labels (green) plus any
            # predictions (yellow) so it can be reviewed and edited like a
            # session file.
            gridhw = (self.spectrogram.n_freq_bins,
                      self.spectrogram.n_time_frames)
            self.spectrogram.set_annotations(
                self._load_confirmed_annotations(filepath, gridhw))
            n_pred = self._load_predictions_as_annotations(wav=filepath) or 0
            self._sync_scrollbar_from_view()
            suffix = f"  |  {n_pred} prediction(s)" if n_pred else ""
            self.status_bar.showMessage(
                f"Training Data — {os.path.basename(filepath)}  |  "
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

        help_menu = menubar.addMenu("&Help")
        act_gpu = QAction("Check &GPU / CUDA setup…", self)
        act_gpu.setToolTip("Test whether training/inference can use your GPU")
        act_gpu.triggered.connect(lambda: self._show_gpu_setup_dialog(force=True))
        help_menu.addAction(act_gpu)

    # ------------------------------------------------------------------
    # GPU / CUDA readiness
    # ------------------------------------------------------------------
    @staticmethod
    def _detect_nvidia_gpu() -> Optional[str]:
        """Return an NVIDIA GPU name via nvidia-smi, or None if absent."""
        import subprocess
        try:
            out = subprocess.run(
                ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
                capture_output=True, text=True, timeout=6)
            if out.returncode == 0 and out.stdout.strip():
                return out.stdout.strip().splitlines()[0].strip()
        except Exception:
            pass
        return None

    def _gpu_status(self):
        """Return ``(ready: bool, title, message)`` describing GPU readiness."""
        try:
            import torch
        except Exception:
            return (False, "PyTorch not found",
                    "PyTorch isn't installed in this environment, so training "
                    "and inference can't run at all. Install it (see the CUDA "
                    "command below) in your 'fnt' conda env.")
        ver = getattr(torch, '__version__', '?')
        if torch.cuda.is_available():
            try:
                name = torch.cuda.get_device_name(0)
            except Exception:
                name = "CUDA GPU"
            return (True, "GPU ready ✓",
                    f"PyTorch {ver} can use your GPU:\n\n    {name}\n\n"
                    "Training and inference will run on CUDA (Device = auto).")
        nv = self._detect_nvidia_gpu()
        import sys as _sys
        pyver = f"{_sys.version_info.major}.{_sys.version_info.minor}"
        cuda_cmd = (
            "    conda activate fnt\n"
            "    pip uninstall -y torch torchvision\n"
            "    pip install torch torchvision --index-url "
            "https://download.pytorch.org/whl/cu124\n")
        guidance = (
            "Get the EXACT command for your setup from:\n"
            "    https://pytorch.org/get-started/locally/\n"
            "(choose Stable · Windows · Pip · Python · your CUDA version)\n\n"
            "Or run, in the 'fnt' env:\n\n" + cuda_cmd +
            "\nNotes:\n"
            f"  • Match the CUDA tag to what's offered — cu124/cu126 are current; "
            "an old tag like cu121 may have no wheels (that's the\n"
            "    'Could not find a version… (from versions: none)' error).\n"
            "  • Double-check the URL host is exactly 'download.pytorch.org'.\n"
            f"  • Your Python is {pyver}; the chosen torch must publish wheels "
            "for it.\n"
            "  • You only need a recent NVIDIA driver — not the full CUDA "
            "Toolkit.\n"
            "  • If the GPU install fails, get back to a working CPU setup with:"
            "  pip install torch torchvision")
        if nv:
            return (False, "GPU present, but PyTorch is CPU-only",
                    f"Detected NVIDIA GPU:\n\n    {nv}\n\n"
                    f"…but the installed PyTorch ({ver}) is a CPU-only build, "
                    "so training/inference run on the CPU (slow).\n\n"
                    + guidance + "\n\nRestart MAD afterward and re-check here.")
        return (False, "No NVIDIA GPU detected",
                "No NVIDIA GPU was found (nvidia-smi isn't available), so "
                "training/inference will use the CPU.\n\nIf this machine does "
                "have an NVIDIA GPU, install its driver first, then a CUDA "
                "build of PyTorch:\n\n" + guidance)

    def _show_gpu_setup_dialog(self, force: bool = False):
        """Show the GPU readiness dialog. When ``force`` is False it only shows
        once per session and only when the GPU is NOT ready (used as a one-time
        nudge before a CPU training/inference run)."""
        ready, title, message = self._gpu_status()
        if not force:
            if ready or getattr(self, '_gpu_nudged', False):
                return
            # Only nudge when there's an actionable GPU sitting unused; if the
            # machine genuinely has no NVIDIA GPU, CPU is expected — stay quiet.
            if not self._detect_nvidia_gpu():
                return
            self._gpu_nudged = True
        dlg = QDialog(self)
        dlg.setWindowTitle(title)
        dlg.setMinimumWidth(560)
        v = QVBoxLayout(dlg)
        txt = QTextEdit()
        txt.setReadOnly(True)
        txt.setPlainText(message)
        txt.setStyleSheet("font-family: monospace; font-size: 11px;")
        v.addWidget(txt)
        row = QHBoxLayout()
        btn_copy = QPushButton("Copy commands")
        def _copy():
            cmds = "\n".join(l.strip() for l in message.splitlines()
                             if l.strip().startswith(('conda ', 'pip ')))
            QApplication.clipboard().setText(cmds or message)
            self.status_bar.showMessage("Copied to clipboard")
        btn_copy.clicked.connect(_copy)
        row.addWidget(btn_copy)
        row.addStretch(1)
        bb = QDialogButtonBox(QDialogButtonBox.Close)
        bb.rejected.connect(dlg.reject)
        bb.accepted.connect(dlg.accept)
        row.addWidget(bb)
        v.addLayout(row)
        dlg.exec_()

    # ------------------------------------------------------------------
    # Keyboard shortcuts (CAD parity)
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
        # are fully dedicated to zoom; B/N step Back/Next through detections.
        make(Qt.Key_B, self._shortcut_pred_prev)   # Back (previous detection)
        make(Qt.Key_N, self._shortcut_pred_next)   # Next detection
        make(Qt.Key_P, self._shortcut_toggle_brush)   # Paint (brush) tool
        make(Qt.Key_E, self._shortcut_toggle_eraser)
        make(Qt.Key_M, self._shortcut_toggle_sam)     # SAM labeling tool
        make(Qt.Key_S, self._shortcut_skip)           # Skip current prediction
        make(Qt.Key_U, self._undo_last)
        # Cmd+Z (macOS) / Ctrl+Z (Windows/Linux) — undo the last review action.
        make(QKeySequence.Undo, self._undo_review_action)
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
        make("+", self._shortcut_speed_up)
        make("=", self._shortcut_speed_up)
        make("-", self._shortcut_speed_down)
        make("Q", self._quick_inference_current_file)

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

    def _shortcut_skip(self):
        """S skips the current prediction (review), in either tab."""
        if self._focus_is_edit():
            return
        self._skip_current_pred()

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
        # Spec + Mask ↔ Spec only
        order = ['overlay', 'spec']
        try:
            idx = order.index(self.spectrogram.view_mode)
        except ValueError:
            idx = 0
        self._set_view_mode(order[(idx + 1) % len(order)])

    def _shortcut_review(self, status: str):
        if self._focus_is_edit():
            return
        # A/R/S route to the unified review actions, which act on the selected
        # detection (any status).
        if self.spectrogram._selected_ann_idx is None and not self._pred_indices():
            return
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
        # Choose the location first, then suggest a name based on what's already
        # there (mad_v1 → mad_v2 if it exists), so the user doesn't have to
        # guess a non-colliding name up front.
        parent_dir = QFileDialog.getExistingDirectory(
            self, "Choose where to create the MAD project",
            os.path.expanduser("~")
        )
        if not parent_dir:
            return
        # Pre-fill the first non-colliding name. Default base is mad_v1; if that
        # folder already exists, suggest mad_v2, mad_v3, …
        if os.path.exists(os.path.join(parent_dir, "mad_v1")):
            suggestion = self._suggest_project_name(parent_dir, "mad_v1")
            prompt = (
                f"A 'mad_v1' project already exists in:\n{parent_dir}\n\n"
                f"Suggested name '{suggestion}' — edit if you like:"
            )
        else:
            suggestion = "mad_v1"
            prompt = f"Project name (created in {parent_dir}):"
        name, ok = QInputDialog.getText(
            self, "New MAD Project", prompt, text=suggestion
        )
        if not ok or not name.strip():
            return
        name = name.strip()
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

    def _default_browse_dir(self) -> str:
        """Where Add Files / Add Folder dialogs should open. When a project is
        open, default to its recordings folder (or the project root) so the user
        lands at their data; otherwise the home directory."""
        if self._project is not None:
            for d in (getattr(self._project, 'recordings_dir', None),
                      getattr(self._project, 'project_dir', None)):
                if d and os.path.isdir(d):
                    return d
        return os.path.expanduser("~")

    def _menu_add_folder(self):
        folder = QFileDialog.getExistingDirectory(
            self, "Add folder of .wav files (non-recursive)",
            self._default_browse_dir()
        )
        if not folder:
            return
        # Browse the folder in place for this session only — files are not
        # copied or persisted; a file joins the project once a call is accepted
        # on it. (Re-add the folder next session to keep browsing.)
        wavs = _list_wavs_in_folder(folder)
        added = self._append_audio_paths(wavs)
        self.status_bar.showMessage(
            f"Browsing folder in place: {folder} (+{added} wavs, "
            f"not copied until you accept a call)")

    def _append_audio_paths(self, paths, persist_files: bool = False) -> int:
        """Load wav paths into the session list (in place — never copied into
        the project). Labels/predictions save next to the source audio. Returns
        the count added."""
        existing = set(self.audio_files)
        to_add = [p for p in paths
                  if p and p not in existing and os.path.isfile(p)]
        if not to_add:
            return 0
        self.audio_files.extend(to_add)
        if self.current_file_idx >= len(self.audio_files):
            self.current_file_idx = 0
        self._refresh_file_list()
        self._active_source = 'session'
        self._review_mode = 'deploy'
        self._clear_training_selection()
        self._deploy_file_idx = None
        self.file_list.blockSignals(True)
        self.file_list.setCurrentRow(self.current_file_idx)
        self.file_list.blockSignals(False)
        self._load_current_file()
        self._update_project_state()
        self._update_scope_labels()
        self._sync_list_buttons()  # selection was set with signals blocked
        return len(to_add)

    def _activate_project(self, cfg: MADProjectConfig):
        self._project = cfg
        self.setWindowTitle(f"{self.BASE_TITLE} — {cfg.project_name}")
        self.lbl_project_status.setText(f"Project: {cfg.project_name}")
        self.lbl_project_status.setStyleSheet(
            "color: #4CAF50; font-size: 10px; padding: 3px 6px;")
        self.btn_add_folder.setEnabled(True)
        self.btn_add_files.setEnabled(True)
        self.act_add_folder.setEnabled(True)
        self.act_add_files.setEnabled(True)
        self.act_close_project.setEnabled(True)
        self.act_run_training.setEnabled(True)
        self.act_run_inference.setEnabled(True)
        self.act_load_pred.setEnabled(True)
        self.act_clear_pred.setEnabled(True)
        self._refresh_quick_infer_models()
        self._set_train_sections_enabled(True)
        self._offer_training_store_migration()
        self._refresh_deploy_models()
        self._rescan_project_wavs()         # loads recordings/ → Training Data
        self._apply_latest_training_config()  # prefill train options from last run
        self._update_source_folders_label()
        self._update_model_info_label()
        self._update_train_button_count()
        self._update_train_button_enabled()
        self._update_infer_run_enabled()
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
        self.lbl_project_status.setText("No project loaded")
        self.lbl_project_status.setStyleSheet(
            "color: #999999; font-size: 10px; padding: 3px 6px;")
        # Add Folder / Add Files stay enabled — loading audio doesn't require
        # a project.
        self.act_close_project.setEnabled(False)
        self.act_run_training.setEnabled(False)
        self.act_run_inference.setEnabled(False)
        self.act_load_pred.setEnabled(False)
        self.act_clear_pred.setEnabled(False)
        self._active_source = 'session'
        self._review_mode = 'deploy'
        self._update_run_button()
        self._set_train_sections_enabled(False)
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
            self, "Select Audio Files", self._default_browse_dir(),
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
        """Load the project's Training Data set — every wav in recordings/ —
        into the Training Data list. Session audio is loaded separately by the
        user and is never auto-populated on open."""
        if self._project is None:
            return 0
        rdir = self._project.recordings_dir
        os.makedirs(rdir, exist_ok=True)
        existing = {os.path.normpath(p) for p in self.deploy_files}
        added = 0
        for w in _list_wavs_in_folder(rdir):
            if os.path.normpath(w) not in existing:
                self.deploy_files.append(w)
                existing.add(os.path.normpath(w))
                added += 1
        self.deploy_files.sort(key=lambda p: os.path.basename(p).lower())
        self._refresh_deploy_queue()
        self._update_train_button_enabled()
        self._update_scope_labels()
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
        self._scan_all_file_counts()

    def _scan_all_file_counts(self):
        """Populate ``_file_count_cache`` with the total mask count per session
        file — every call in the unified CSV, regardless of source or status —
        via the shared :meth:`_file_call_count` (same number the Training Data
        list shows). One cheap sibling read per file; the multi-GB probability
        grid is never touched. Call on project open / file-list rebuild, NOT on
        file switch.
        """
        cache: Dict[str, int] = {}
        for fp in self.audio_files:
            cnt = self._file_call_count(fp)
            if cnt:
                cache[os.path.basename(fp)] = cnt
        self._file_count_cache = cache
        self._update_file_list_counts(sync_current=False)

    def _update_file_list_counts(self, sync_current: bool = True):
        """Refresh the file-list labels from the cached counts (fast).

        With ``sync_current`` (the default, used after a file is loaded or its
        calls are edited), the *currently displayed* file's entry is reconciled
        with the live in-memory annotation count so edits show immediately.
        The scan passes ``sync_current=False`` so it renders purely from the
        stored counts and never clobbers a not-yet-loaded current file.
        """
        if not hasattr(self, 'file_list'):
            return
        if not hasattr(self, '_file_count_cache'):
            self._file_count_cache = {}
        # Reconcile the current file only once its annotations are loaded, and
        # only when the session list owns the preview (else the in-memory
        # annotations belong to a Training Data file, not this list).
        if (sync_current and hasattr(self, 'spectrogram')
                and self._active_source == 'session'
                and self.audio_files and self.audio_data is not None):
            cur_wav = os.path.basename(
                self.audio_files[self.current_file_idx])
            n_mem = len(self.spectrogram.annotations)
            if n_mem > 0:
                self._file_count_cache[cur_wav] = n_mem
            elif cur_wav in self._file_count_cache:
                del self._file_count_cache[cur_wav]
        self.file_list.blockSignals(True)
        for i in range(self.file_list.count()):
            item = self.file_list.item(i)
            fp = item.data(Qt.UserRole)
            bn = os.path.basename(fp)
            n = self._file_count_cache.get(bn, 0)
            if n > 0:
                item.setText(f"{bn}  ({n})")
            else:
                item.setText(bn)
        self.file_list.blockSignals(False)

    def _set_train_sections_enabled(self, enabled: bool):
        # Only the project-backed sections are gated on having a project open.
        # Labeling Tools + Detections stay enabled so you can label a loaded
        # session file with no project (labels save to siblings); their buttons
        # follow whether audio is loaded via _update_paint_buttons_enabled.
        for attr in ('_grp_training_list', '_grp_train_model'):
            grp = getattr(self, attr, None)
            if grp is not None:
                grp.setEnabled(enabled)

    def _set_train_config_enabled(self, enabled: bool):
        """Enable/disable the training-config controls while a run is active, so
        it's visually clear training is in progress. The Train button itself is
        left enabled — during a run it doubles as 'show graph'."""
        for name in ('combo_arch', 'combo_train_encoder', 'spin_train_epochs',
                     'spin_train_patience', 'spin_train_batch', 'spin_train_lr',
                     'spin_train_val', 'combo_train_device'):
            w = getattr(self, name, None)
            if w is not None:
                w.setEnabled(enabled)

    def _update_source_folders_label(self):
        # The source-folders label was dropped with the Project group; this is
        # kept as a no-op so existing call sites don't need touching.
        return

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
            # Session list now owns the single preview — drop any Training Data
            # selection and switch review mode (session = CSV-only accept).
            switching = (self._active_source != 'session' or
                         row != self.current_file_idx)
            self._active_source = 'session'
            self._review_mode = 'deploy'
            self._clear_training_selection()
            self._deploy_file_idx = None
            if switching:
                self._dismiss_training_view()
                self._auto_save_mask_if_dirty()
                self.current_file_idx = row
                self._load_current_file()
                self._update_project_state()
                self._update_review_buttons_for_source()

    def _prev_file(self):
        if self.current_file_idx > 0:
            self.file_list.setCurrentRow(self.current_file_idx - 1)

    def _next_file(self):
        if self.current_file_idx < len(self.audio_files) - 1:
            self.file_list.setCurrentRow(self.current_file_idx + 1)

    def _remove_selected_files(self):
        """Unload the selected files from the session list. Nothing is deleted
        from disk — the .wav and any sibling csv/h5 stay where they are."""
        sel = self.file_list.selectedItems()
        if not sel:
            return
        rows = sorted({self.file_list.row(item) for item in sel}, reverse=True)
        removed_paths = [self.audio_files[r] for r in rows]
        self._remove_files_by_path(removed_paths)

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
                f"Session — {os.path.basename(filepath)}  |  "
                f"{len(self.audio_data) / self.sample_rate:.2f}s @ "
                f"{self.sample_rate} Hz"
            )
            self._log(f"Open session file {self.current_file_idx + 1}/"
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
        wav_path = self.audio_files[self.current_file_idx]
        wav_name = os.path.basename(wav_path)
        grid = (self.spectrogram.n_freq_bins, self.spectrogram.n_time_frames)
        anns = self._load_confirmed_annotations(wav_path, grid)
        self.spectrogram.set_annotations(anns)
        # _load_predictions_as_annotations refreshes the list itself when it adds
        # predictions (returns an int); it returns None on its early-out paths
        # (no predictions) without refreshing — only then do we refresh here, so
        # the (potentially 800+ row) tree is rebuilt once per load, not twice.
        result = self._load_predictions_as_annotations()
        n_pred = result or 0
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
        if result is None:
            self._refresh_annotation_list()
            self._update_file_list_counts()

    def _load_confirmed_annotations(self, wav_path, grid):
        """Confirmed (human-labeled) annotations for ``wav_path`` — from the
        project store (by source wav) plus the per-wav h5 sibling — de-duped by
        id. Shared by the session and Training-Data preview load paths."""
        from fnt.usv.usv_detector.mad_examples import iter_file_annotations
        wav_name = os.path.basename(wav_path)
        anns = []
        if self._project is not None:
            try:
                anns = list(iter_file_annotations(
                    self._project.training_data_dir, wav_name, grid))
            except Exception:
                anns = []
        try:
            anns.extend(self._load_sibling_h5_annotations(wav_path, grid))
        except Exception:
            pass
        seen, uniq = set(), []
        for a in anns:
            aid = a.get('id')
            if aid and aid in seen:
                continue
            if aid:
                seen.add(aid)
            # Join key to the unified CSV row for later review/edit/delete.
            if a.get('blob_id') is None and aid is not None:
                a['blob_id'] = aid
            uniq.append(a)
        return uniq

    def _load_sibling_h5_annotations(self, wav_path, grid):
        """Load confirmed examples stored in the per-wav ``_FNT_masks.h5``
        (project-free labels). Returns a list of annotation dicts."""
        from fnt.usv.usv_detector.fnt_mask_store import (
            masks_sibling_path, td_iter_file_examples,
        )
        from fnt.usv.usv_detector.mad_examples import _examples_to_annotations
        h5_path = masks_sibling_path(wav_path)
        if not os.path.isfile(h5_path):
            return []
        wav_name = os.path.basename(wav_path)
        examples = list(td_iter_file_examples(h5_path, wav_name))
        return list(_examples_to_annotations(examples, wav_name, grid))

    def _save_current_mask(self):
        # Labels are now saved per-call as training examples on confirm (Enter);
        # there is no per-file mask to write. Kept so the Ctrl+S menu/no-ops.
        self.status_bar.showMessage(
            "Labels are saved automatically when you confirm a call (Enter)."
        )

    def _auto_save_mask_if_dirty(self):
        # No-op: confirmed calls persist as examples at confirm time.
        return

    def _ask_class_for_confirm(self, n, classes, last):
        """Class picker for confirming masks. Built so a SINGLE Enter accepts —
        an editable QInputDialog combo eats the first Enter on Windows (the user
        had to press Enter twice). Returns the class name, or None if cancelled."""
        dlg = QDialog(self)
        dlg.setWindowTitle("Confirm calls")
        v = QVBoxLayout(dlg)
        v.addWidget(QLabel(f"Class for {n} mask(s) (Enter to confirm):"))
        combo = QComboBox()
        combo.setEditable(True)
        combo.addItems(classes)
        combo.setCurrentIndex(classes.index(last) if last in classes else 0)
        v.addWidget(combo)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        ok_btn = bb.button(QDialogButtonBox.Ok)
        ok_btn.setText("Confirm")
        ok_btn.setDefault(True)
        ok_btn.setAutoDefault(True)
        bb.accepted.connect(dlg.accept)
        bb.rejected.connect(dlg.reject)
        v.addWidget(bb)
        # Pressing Return in the combo's line edit accepts the dialog directly,
        # so one Enter confirms instead of two.
        combo.lineEdit().returnPressed.connect(dlg.accept)
        combo.setFocus()
        combo.lineEdit().selectAll()
        if dlg.exec_() != QDialog.Accepted:
            return None
        return combo.currentText().strip() or None

    def _refresh_pending_status(self):
        """Show how many manually-drawn masks are awaiting confirmation."""
        try:
            n = len(self.spectrogram.pending_components())
        except Exception:
            n = 0
        if n > 0:
            self.lbl_mask_status.setText(
                f"✎ {n} mask(s) drawn — press Enter to confirm")
        else:
            self.lbl_mask_status.setText(
                "No pending masks — paint or SAM, then Enter to confirm")

    def _on_stroke_committed(self):
        # A brush stroke just ended; nothing is persisted until the user
        # confirms with Enter, but show how many masks are pending.
        self._refresh_pending_status()

    # ==================================================================
    # Paint tools
    # ==================================================================
    def _update_paint_buttons_enabled(self):
        has_audio = self.audio_data is not None
        for btn in (self.btn_paint, self.btn_erase, self.btn_clear_mask,
                    self.btn_sam, self.btn_sam_model, self.btn_undo):
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
            "2. <b>Use SAM (M) for fast labeling.</b> Click each call to "
            "generate a mask, then press Enter to confirm them all at once.<br><br>"
            "3. <b>Use the Paint tool (P) for fine corrections</b> and the "
            "eraser (E) to remove false strokes.<br><br>"
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
            self._refresh_pending_status()
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
        if self.audio_data is None or self._active_wav_path() is None:
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
        name = self._ask_class_for_confirm(len(comps), classes, last)
        if not name:
            return
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
            ann = {
                'id': ex_id, 'blob_id': ex_id, 'category': name,
                'f0': f0, 'f1': f1, 't0': t0, 't1': t1,
                'mask': local.astype(bool),
                'source': 'label',
                **getattr(self, '_last_example_stats', {}),
            }
            sg.add_annotation(ann)
            # Record the label in the unified per-wav CSV (status 'accepted').
            self._upsert_call_csv_row(ann, 'accepted', score=1.0)
            saved += 1
        if self._project is not None:
            try:
                self._project.save()
            except Exception:
                pass
        sg.clear_pending()
        self._refresh_annotation_list()
        self._deactivate_labeling_tools()  # turn off SAM/Paint/Eraser on confirm
        self.lbl_mask_status.setText(f"Confirmed {saved} call(s) — '{name}'")
        self.status_bar.showMessage(
            f"Saved {saved} example(s) (class '{name}') — label the next batch"
        )
        self._log(f"Confirmed {saved} call(s) as '{name}' (Enter)")

    def _deactivate_labeling_tools(self):
        """Turn off whichever labeling tool (SAM / Paint / Eraser) is active and
        drop any SAM prompt points — called after confirming a batch of masks."""
        for btn in (self.btn_paint, self.btn_erase, self.btn_sam):
            if btn.isChecked():
                btn.setChecked(False)
        self.spectrogram.set_paint_mode(None)
        try:
            self.spectrogram.clear_sam_prompts()
        except Exception:
            pass

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
        # Full per-call quantification over the labeled pixels — uses the SAME
        # shared function as the prediction path so label/prediction rows are
        # directly comparable in the unified CSV.
        self._last_example_stats = {}
        try:
            from fnt.usv.usv_detector.mad_inference import compute_call_metrics
            tw_m = min(t1 - t0, W - lt0)
            if tw_m > 0:
                # Full-frequency dB columns for the call's time span (per-frame
                # spectral features need the whole column), plus the bbox mask.
                cols_db = spec_db[:, lt0:lt0 + tw_m]
                crop_mask = local[:, :tw_m].astype(bool)
                if cols_db.shape[1] == crop_mask.shape[1]:
                    self._last_example_stats = compute_call_metrics(
                        cols_db, crop_mask, f0, df, dt,
                        sp['db_min'], sp['db_max'])
                    self._last_example_stats['source'] = 'label'
        except Exception:
            self._last_example_stats = {}
        wav_path = (self._active_wav_path() or
                    self.audio_files[self.current_file_idx])
        wav_name = os.path.basename(wav_path)
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
        # Labels always save to the sibling h5 next to the active file (session
        # source or training copy). The project's consolidated training_data.h5
        # is rebuilt from the Training Data list at train time, so a file only
        # trains the model once it's been copied into Training Data.
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
            self._refresh_pending_status()
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
        sg._selected_ann_idx = ai
        self._select_list_row_for_id(ann.get('id'))
        # Accept/Reject/Delete act on the selected detection (any status).
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

    def _batch_remove_pred_persistence(self, anns):
        """Remove the CSV rows + stored crops for many predictions in ONE CSV
        write and ONE h5 open (vs once per item), for fast box-delete."""
        wav = self._active_review_wav_path()
        ids = {self._ann_csv_id(a) for a in anns
               if self._ann_csv_id(a) is not None}
        if not wav or not ids:
            return
        ids_str = {str(b) for b in ids}
        csv_path = pred_csv_sibling_path(wav)
        if os.path.isfile(csv_path):
            try:
                from fnt.usv.usv_detector.mad_inference import (
                    read_blob_csv, write_blob_csv)
                rows = [r for r in read_blob_csv(csv_path)
                        if str(r.get('blob_id')) not in ids_str]
                write_blob_csv(csv_path, rows)
            except Exception:
                pass
        try:
            from fnt.usv.usv_detector.fnt_mask_store import (
                masks_sibling_path, delete_pred_masks)
            delete_pred_masks(masks_sibling_path(wav), ids)
        except Exception:
            pass

    def _apply_to_box_selection(self, action):
        """Apply accept/reject/delete to every box-selected detection at once.
        Returns True if a selection existed and was handled."""
        idxs = self._box_sel_indices()
        if not idxs:
            return False
        self._snapshot_for_undo(f"{action.capitalize()} selection", crops=(action == 'delete'))
        sg = self.spectrogram
        n_done = 0
        if action == 'delete':
            # Batch the persistence (one CSV write + one h5 open), then drop the
            # annotations without a per-item list refresh or full mask rebuild.
            targets = [sg.annotations[i] for i in idxs
                       if 0 <= i < len(sg.annotations)]
            self._batch_remove_pred_persistence(targets)
            for i in sorted(idxs, reverse=True):
                if 0 <= i < len(sg.annotations):
                    self._delete_annotation(i, refresh=False)
                    n_done += 1
        else:
            # Process high→low so earlier removals don't shift later indices.
            for i in sorted(idxs, reverse=True):
                if not (0 <= i < len(sg.annotations)):
                    continue
                ann = sg.annotations[i]
                is_pred = ann.get('status') == 'prediction'
                if action == 'accept':
                    if is_pred:
                        if self._review_mode == 'deploy':
                            self._write_pred_csv_status(ann, 'accepted')
                            ann['status'] = 'accepted'  # keep visible (blue)
                        else:
                            self._accept_prediction(i)
                        n_done += 1
                elif action == 'reject':
                    if is_pred:
                        # Recorded decision: keep visible (red), persist it.
                        self._write_pred_csv_status(ann, 'rejected')
                        ann['status'] = 'rejected'
                        n_done += 1
        self._box_sel_ids = []
        sg._selected_set = set()
        sg._selected_ann_idx = None
        sg.update()
        self._pred_review_idx = None
        self._refresh_annotation_list()
        self._update_pred_review_widgets()
        self._log(f"{action.capitalize()} {n_done} box-selected detection(s)")
        self.status_bar.showMessage(f"{action.capitalize()}ed {n_done} detection(s)")
        return True

    def _delete_selected_annotation(self):
        """Delete the selected detection (Delete button / D key).

        Delete is permanent removal that leaves no trace — distinct from Reject
        (which keeps a visible red record). It removes the CSV row entirely and
        drops the stored crop. For a prediction it follows the Auto-advance
        toggle (on → next pending; off → stay); deleting a *confirmed* detection
        never advances (there's no review cursor on it)."""
        if self._focus_is_edit():
            return
        if self._apply_to_box_selection('delete'):
            return
        sel = self.spectrogram._selected_ann_idx
        if sel is None or not (0 <= sel < len(self.spectrogram.annotations)):
            self.status_bar.showMessage("No detection selected — click one first.")
            return
        ann = self.spectrogram.annotations[sel]
        st = ann.get('status')
        was_pending = st == 'prediction'
        self._snapshot_for_undo("Delete")
        # Capture the next pending detection after this one (by id) for the
        # post-delete advance, plus this item's position to fall back to.
        order = self._review_order()
        pos = order.index(sel) if sel in order else None
        next_pending_id = None
        if pos is not None:
            for p in range(pos + 1, len(order)):
                a = self.spectrogram.annotations[order[p]]
                if a.get('status') == 'prediction':
                    next_pending_id = a.get('id')
                    break
        # Delete = permanent removal, no trace: drop CSV row + crop (no-ops if
        # the detection has neither). _delete_annotation also deletes any saved
        # training example.
        self._remove_pred_csv_row(ann)
        self._delete_pred_crop(ann)
        self.spectrogram._selected_ann_idx = None
        self._delete_annotation(sel)
        self._log("Deleted detection")
        self.status_bar.showMessage("Deleted detection (D)")
        # Advance only when Auto-advance is ON (jump to the next pending). When
        # OFF, stay put — don't move/recenter; just clear the now-stale
        # selection so Back (B) / Next (N) drive navigation.
        if self._auto_advance and next_pending_id is not None:
            self._reselect_by_id(next_pending_id)
        else:
            self.spectrogram._selected_ann_idx = None
            self.spectrogram.update()
            self._update_pred_review_widgets()
        if was_pending:
            self._reviewed_count += 1
            if not self._pred_indices():
                self._maybe_prompt_next_file()

    def _active_review_wav_path(self) -> Optional[str]:
        """The wav whose sibling stores hold the currently-reviewed detections —
        the file owning the preview (session or Training Data)."""
        return self._active_wav_path()

    def _delete_pred_crop(self, ann: dict):
        """Drop a prediction's stored crop from the sibling h5 (best-effort)."""
        bid = ann.get('blob_id')
        wav = self._active_review_wav_path()
        if bid is None or not wav:
            return
        try:
            from fnt.usv.usv_detector.fnt_mask_store import (
                masks_sibling_path, delete_pred_mask,
            )
            delete_pred_mask(masks_sibling_path(wav), bid)
        except Exception:
            pass

    def _delete_annotation(self, ann_idx, refresh: bool = True):
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
            _wav = self._active_wav_path()
            if _wav:
                try:
                    from fnt.usv.usv_detector.fnt_mask_store import (
                        masks_sibling_path, td_delete,
                    )
                    h5 = masks_sibling_path(_wav)
                    td_delete(h5, aid)
                except Exception:
                    pass
            # Remove its row from the unified CSV too (leaves no trace).
            self._remove_pred_csv_row(ann)
        if refresh:  # bulk callers refresh once at the end instead
            self._refresh_annotation_list()

    def _on_annotation_edit_finished(self, ann_idx):
        sg = self.spectrogram
        if (self.audio_data is None or self._active_wav_path() is None or
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
                h5 = masks_sibling_path(self._active_wav_path())
                td_delete(h5, old_id)
            except Exception:
                pass
            # Drop the stale CSV row keyed by the old id.
            self._remove_pred_csv_row({'blob_id': old_id})
        ann['id'] = new_id
        ann['blob_id'] = new_id
        ann.setdefault('source', 'label')
        ann.update(getattr(self, '_last_example_stats', {}))  # refreshed stats
        # Re-record the edited label's box/mask in the unified CSV.
        self._upsert_call_csv_row(ann, ann.get('status') or 'accepted')
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

    def _on_overview_clicked(self, center: float):
        """Recenter the view window on the clicked overview position, keeping the
        current window width. (The overview emits a single center time.)"""
        dur = self.spectrogram.total_duration
        if dur <= 0:
            return
        window = self.spectrogram.view_end - self.spectrogram.view_start
        if window <= 0:
            window = min(dur, 2.0)
        start = max(0.0, center - window / 2.0)
        end = min(dur, start + window)
        start = max(0.0, end - window)  # preserve full width at the right edge
        self.spectrogram.view_start = start
        self.spectrogram.view_end = end
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
    # Playback (mirrors CAD)
    # ==================================================================
    def _update_playback_buttons_enabled(self):
        has_audio = self.audio_data is not None and HAS_SOUNDDEVICE
        self.btn_play.setEnabled(has_audio)
        self.slider_speed.setEnabled(has_audio)

    def _on_speed_changed(self, idx: int):
        speed = self._speed_values[idx]
        self.playback_speed = speed
        self.lbl_speed.setText(f"{speed}x")

    def _shortcut_speed_up(self):
        idx = self.slider_speed.value()
        if idx < self.slider_speed.maximum():
            self.slider_speed.setValue(idx + 1)

    def _shortcut_speed_down(self):
        idx = self.slider_speed.value()
        if idx > 0:
            self.slider_speed.setValue(idx - 1)

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
            self.btn_play.setText("Stop (Space)")
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
        self.btn_play.setText("Play (Space)")
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
        # Training lives inline in the single left column — route the
        # menu/shortcut straight to it.
        self._on_inline_train()

    def _start_training(self, cfg, post_inference_wavs: Optional[List[str]] = None,
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
        # Report the actual training-set size (accepted labels) — training reads
        # the consolidated example store, not cfg.wav_paths, so don't print the
        # misleading "0 labeled wav(s)" / "n=0".
        try:
            from fnt.usv.usv_detector.mad_examples import count_examples
            n_labels = count_examples(cfg.training_data_dir)
        except Exception:
            n_labels = 0
        progress.append(f"Models dir: {os.path.dirname(cfg.resolve_run_dir())}")
        progress.append(f"Training on {n_labels} accepted label(s)")

        worker = MADTrainingWorker(cfg, parent=self)

        def on_progress(epoch: int, total: int, metrics: dict):
            status = metrics.get('status', '')
            if status == 'device':
                desc = metrics.get('device_desc', metrics.get('device', '?'))
                req = metrics.get('requested', 'auto')
                line = f"Training device: {desc}  (requested: {req})"
                progress.set_stage(line)
                progress.append(line)
                self._log(line)
                return
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

            # Post-training inference chain (on the just-saved weights), using
            # the Run Inference settings on the wavs gathered before the run.
            if post_inference_wavs and summary.get('model_path'):
                QTimer.singleShot(
                    200,
                    lambda: self._run_post_training_inference(
                        summary['model_path'], post_inference_wavs,
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
        files = list(self.audio_files) + [
            f for f in self.deploy_files if f not in self.audio_files]
        if not files:
            QMessageBox.warning(
                self, "No files",
                "Load session audio or add Training Data files first.")
            return
        current = self._active_wav_path()
        default_model = None
        if self._project.models:
            # Pick the most recent entry (models is a list of dicts).
            last = self._project.models[-1]
            if isinstance(last, dict):
                default_model = last.get('path')
            else:
                default_model = str(last)
        dlg = RunInferenceDialog(
            self, self._project.project_dir, files, current,
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

    def _run_post_training_inference(self, model_path: str, wavs: List[str]):
        """Run inference automatically after training, on the just-saved
        weights, using the Run Inference section's settings (threshold, min
        blob, device, preserve labels) over the wavs gathered before the run."""
        wavs = [w for w in (wavs or []) if w]
        if not wavs or not model_path:
            return
        from fnt.usv.usv_detector.mad_inference import MADInferenceConfig
        cfg = MADInferenceConfig(
            model_path=model_path,
            threshold=self.spin_infer_threshold.value(),
            min_blob_pixels=self.spin_infer_min_blob.value(),
            device=self.combo_infer_device.currentText(),
            save_blob_csv=True,
            preserve_labels=self.chk_infer_preserve.isChecked(),
            training_data_dir=(self._project.training_data_dir
                               if self._project else ""),
        )
        self.btn_infer_run.setEnabled(False)
        self.infer_panel.start_run()
        self._start_inference(cfg, wavs, reporter=self.infer_panel)

    def _latest_model_path(self) -> Optional[str]:
        """Return the path to the most recent trained model in the project."""
        if self._project is None:
            return None
        models_root = os.path.join(self._project.project_dir, 'models')
        if not os.path.isdir(models_root):
            return None
        candidates = []
        for name in sorted(os.listdir(models_root)):
            w = os.path.join(models_root, name, 'weights.pt')
            if os.path.isfile(w):
                candidates.append(w)
        return candidates[-1] if candidates else None

    def _quick_inference_current_file(self):
        """Run inference on the visible view using the selected model.
        Mapped to the I key. Produces in-memory predictions only."""
        if self._project is None:
            self.status_bar.showMessage(
                "Open a project first to run inference")
            return
        model = (self.combo_quick_infer_model.currentData()
                 if hasattr(self, 'combo_quick_infer_model')
                 and self.combo_quick_infer_model.count() > 0
                 else self._latest_model_path())
        if not model:
            self.status_bar.showMessage(
                "No trained model found — train a model first")
            return
        if self._active_wav_path() is None:
            return
        sg = self.spectrogram
        if sg.audio_data is None or sg.sample_rate is None:
            return

        v_start = sg.view_start
        v_end = sg.view_end
        s0 = max(0, int(v_start * sg.sample_rate))
        s1 = min(len(sg.audio_data), int(v_end * sg.sample_rate))
        if s1 <= s0:
            return
        audio_seg = sg.audio_data[s0:s1].copy()

        sg._infer_loading = True
        sg._infer_progress = 0.0
        sg.update()
        self.btn_quick_infer.setEnabled(False)

        sp = self._spec_params()
        worker = MADViewInferenceWorker(
            model_path=model,
            audio_segment=audio_seg,
            sample_rate=sg.sample_rate,
            view_start=v_start,
            view_end=v_end,
            spec_params=sp,
            parent=self,
        )

        def on_progress(frac):
            sg._infer_progress = frac
            sg.update()

        def on_finished(rows, vs, ve):
            sg._infer_loading = False
            sg.update()
            self.btn_quick_infer.setEnabled(True)
            prob = worker.prob_mask
            frame_off = worker.frame_offset
            self._apply_view_inference_results(rows, vs, ve, prob, frame_off)

        def on_error(msg):
            sg._infer_loading = False
            sg.update()
            self.btn_quick_infer.setEnabled(True)
            self.status_bar.showMessage("Inference failed")
            self._log(f"View inference error: {msg[:200]}")

        worker.progress_signal.connect(on_progress)
        worker.finished_signal.connect(on_finished)
        worker.error_signal.connect(on_error)
        self._view_infer_worker = worker
        worker.start()

    def _apply_view_inference_results(self, rows, view_start, view_end,
                                      prob_mask=None, frame_offset=0):
        """Turn view-inference blob rows into prediction annotations with
        pixel-level masks extracted from the probability map."""
        sg = self.spectrogram
        sp = self._spec_params()
        hop = sp['nperseg'] - sp['noverlap']
        dt = hop / float(self.sample_rate)
        df = (self.sample_rate / 2.0) / (sp['nfft'] // 2)
        threshold = 0.5
        wav_path = (self._active_wav_path() or
                    self.audio_files[self.current_file_idx])
        wav_name = os.path.basename(wav_path)

        # Write probability mask into the h5 sibling (merge into existing
        # full-file mask if present).
        if not rows:
            self.status_bar.showMessage("No detections in current view")
            return

        # Remove existing predictions that overlap the inferred view range
        t0_view = int(round(view_start / dt))
        t1_view = int(round(view_end / dt))
        sg.annotations = [
            a for a in sg.annotations
            if a.get('status') != 'prediction'
            or a['t1'] <= t0_view or a['t0'] >= t1_view
        ]

        n_added = 0
        new_crops = []  # small per-blob masks to persist (NOT the full grid)
        for r in rows:
            t0 = int(round(r['start_s'] / dt))
            t1 = int(round(r['stop_s'] / dt))
            f0 = int(round(r['min_freq_hz'] / df))
            f1 = int(round(r['max_freq_hz'] / df))
            if t1 <= t0 or f1 <= f0:
                continue
            # Extract actual pixel mask from the in-memory (view-sized) prob map.
            if prob_mask is not None:
                lt0 = t0 - frame_offset
                lt1 = t1 - frame_offset
                lt0 = max(0, lt0)
                lt1 = min(prob_mask.shape[1], lt1)
                lf1 = min(prob_mask.shape[0], f1)
                if lt1 > lt0 and lf1 > f0:
                    blob_region = prob_mask[f0:lf1, lt0:lt1] >= threshold
                else:
                    blob_region = np.ones((f1 - f0, t1 - t0), dtype=bool)
            else:
                blob_region = np.ones((f1 - f0, t1 - t0), dtype=bool)
            if not blob_region.any():
                continue
            blob_region = np.ascontiguousarray(blob_region)
            f1 = f0 + blob_region.shape[0]
            t1 = t0 + blob_region.shape[1]
            sg.annotations.append({
                'id': f'vpred_{n_added}',
                'category': (self._project.last_class if self._project else
                             self._session_last_class) or 'USV',
                'f0': f0, 'f1': f1, 't0': t0, 't1': t1,
                'mask': blob_region,
                'status': 'prediction',
                'source_wav': wav_name,
                'score': r.get('score', 0),
            })
            new_crops.append({'mask': blob_region, 'f_off': f0, 't_off': t0})
            n_added += 1

        # Persist the small per-blob crops (merging with any outside this view),
        # not the multi-GB probability grid. A legacy /prob grid, if present, is
        # dropped here so the file is fully on the fast crop path.
        try:
            from fnt.usv.usv_detector.fnt_mask_store import (
                masks_sibling_path, read_all_pred_masks, write_pred_masks,
                set_grid_attrs, delete_prob,
            )
            h5 = masks_sibling_path(wav_path)
            kept = []
            for bid, c in read_all_pred_masks(h5).items():
                cw = c['mask'].shape[1]
                if c['t_off'] + cw <= t0_view or c['t_off'] >= t1_view:
                    kept.append({'blob_id': bid, 'mask': c['mask'],
                                 'f_off': c['f_off'], 't_off': c['t_off']})
            next_id = 1 + max(
                [int(k['blob_id']) for k in kept
                 if str(k['blob_id']).isdigit()] or [-1])
            for nc in new_crops:
                kept.append({'blob_id': next_id, **nc})
                next_id += 1
            n_freq = sp['nfft'] // 2 + 1
            n_time = sg.n_time_frames or 0
            set_grid_attrs(h5, sample_rate=self.sample_rate,
                           nperseg=sp['nperseg'], noverlap=sp['noverlap'],
                           nfft=sp['nfft'], n_freq_bins=n_freq,
                           n_time_frames=n_time)
            write_pred_masks(h5, kept)
            delete_prob(h5)
        except Exception:
            pass
        sg._rebuild_confirmed_mask()
        sg.update()
        self._pred_review_idx = None
        self._refresh_annotation_list()
        if n_added:
            # Keep the view exactly where the user ran inference — do NOT
            # recenter on a prediction (that made the view slide backwards).
            # Just point the review cursor at the first prediction in this view
            # so Back/Next works, leaving the view (and the new masks) in place.
            preds = self._pred_indices()
            for pos, ai in enumerate(preds):
                a = sg.annotations[ai]
                if a['t1'] > t0_view and a['t0'] < t1_view:
                    self._pred_review_idx = pos
                    break
            if self._pred_review_idx is None and preds:
                self._pred_review_idx = 0
            self._log(f"View inference: {n_added} prediction(s) "
                      f"({view_start:.1f}–{view_end:.1f}s)")
        self._update_pred_review_widgets()
        self._update_file_list_counts()
        self.status_bar.showMessage(
            f"Found {n_added} detection(s) in current view")

    def _start_inference(self, cfg, wav_paths: List[str], reporter=None):
        owns_modal = reporter is None
        progress = reporter or MADRunProgressDialog(self, "MAD Inference")
        progress.set_stage(f"Running inference on {len(wav_paths)} file(s)…")
        progress.append(f"Model: {cfg.model_path}")
        progress.append(f"Threshold: {cfg.threshold}  "
                        f"Min blob: {cfg.min_blob_pixels}px")

        worker = MADInferenceWorker(cfg, wav_paths, parent=self)
        self._infer_worker = worker
        if hasattr(self, 'btn_infer_pause'):
            self.btn_infer_pause.setText("Pause")
            self.btn_infer_pause.setEnabled(True)

        # Reset queue markers so completion shows live as the run progresses.
        for w in wav_paths:
            self._set_deploy_item_state(w, 'pending')
        self._infer_counted = set()

        # Human-readable stage labels so the X/Y is obviously *tiles* scanned by
        # the model (not detections — those are counted at the end).
        stage_labels = {
            'spec': 'building spectrogram',
            'infer': 'scanning tiles',
            'blobs': 'extracting detections',
        }

        import time as _t
        scan = {'fi': None, 't0': 0.0, 'i0': 0}

        def on_progress(file_i, file_n, wav_name, stage, si, sn):
            label = stage_labels.get(stage, stage)
            if stage == 'infer':
                now = _t.time()
                if scan['fi'] != file_i or si <= 1:
                    scan['fi'] = file_i
                    scan['t0'] = now
                    scan['i0'] = si
                rate = ''
                dt = now - scan['t0']
                di = si - scan['i0']
                if dt > 1.0 and di > 0:
                    r = di / dt
                    eta = (sn - si) / r if r > 0 else 0
                    rate = f"  (~{r:.1f}/s, ETA {eta:.0f}s)"
                detail = f"{label} {si}/{sn}{rate}"
            else:
                detail = label
            progress.set_stage(
                f"File {file_i + 1}/{file_n}: {wav_name} — {detail}"
            )
            progress.set_main(file_i + (si / max(1, sn)), file_n)
            progress.set_sub(si, sn)
            # Files before the one in progress are finished → green ✓ them (with
            # their detection count) so the user can click in and QC while later
            # files run. Count is read once per file from its just-written h5.
            for j in range(file_i):
                if j < len(wav_paths) and j not in self._infer_counted:
                    self._infer_counted.add(j)
                    cnt = None
                    try:
                        from fnt.usv.usv_detector.fnt_mask_store import (
                            masks_sibling_path, get_prob_blob_count,
                        )
                        cnt = get_prob_blob_count(masks_sibling_path(wav_paths[j]))
                    except Exception:
                        pass
                    self._set_deploy_item_state(wav_paths[j], 'done', cnt)

        def on_finished(results):
            self._infer_worker = None
            if hasattr(self, 'btn_infer_pause'):
                self.btn_infer_pause.setText("Pause")
                self.btn_infer_pause.setEnabled(False)
            total = sum(r.get('n_blobs', 0) for r in results if 'n_blobs' in r)
            errors = [r for r in results if 'error' in r]
            # Final reconcile of queue markers (covers the last file + errors).
            for r in results:
                wp = r.get('wav_path')
                if wp:
                    if 'error' in r:
                        self._set_deploy_item_state(wp, 'error')
                    else:
                        self._set_deploy_item_state(wp, 'done', r.get('n_blobs'))
            # Aggregate timing across the batch (helps spot CPU vs GPU).
            tt = [r['timing'] for r in results if r.get('timing')]
            timing_line = ""
            if tt:
                tot_audio = sum(t.get('audio_dur_s', 0) for t in tt)
                tot_scan = sum(t.get('t_infer', 0) for t in tt)
                tot_wall = sum(t.get('t_total', 0) for t in tt)
                rt = (tot_audio / tot_scan) if tot_scan > 0 else 0
                dev = tt[0].get('device', '?')
                timing_line = (
                    f"\nTiming: {tot_audio:.0f}s audio scanned in "
                    f"{tot_scan:.0f}s ({rt:.2f}× realtime on {dev}); "
                    f"{tot_wall:.0f}s total wall.")
            progress.append(
                f"\nFinished. {len(results)} file(s), "
                f"{total} blob(s) total, {len(errors)} error(s).{timing_line}"
            )
            if timing_line:
                self._log(timing_line.strip())
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
            # Refresh the Training Data status ticks, then auto-open something
            # for review: the currently-previewed file if it was inferred, else
            # the first target file.
            self._refresh_deploy_queue()
            active = self._active_wav_path()
            if active and active in wav_paths:
                n = self._load_predictions_as_annotations(wav=active)
                if n:
                    self.status_bar.showMessage(
                        f"Loaded {n} prediction(s) — Accept (A) / Reject (R) "
                        f"/ Skip (S)")
            elif wav_paths:
                first = wav_paths[0]
                if first in self.deploy_files:
                    self.deploy_list.setCurrentRow(self.deploy_files.index(first))
                elif first in self.audio_files:
                    self.file_list.setCurrentRow(self.audio_files.index(first))

        def on_error(msg: str):
            self._infer_worker = None
            if hasattr(self, 'btn_infer_pause'):
                self.btn_infer_pause.setText("Pause")
                self.btn_infer_pause.setEnabled(False)
            progress.append("\nERROR:\n" + msg)
            progress.mark_done(ok=False)

        def on_device(dev: str):
            desc = dev
            try:
                import torch
                if dev == 'cuda':
                    desc = f"cuda — {torch.cuda.get_device_name(0)}"
                elif dev == 'mps':
                    desc = "mps — Apple GPU"
                else:
                    seen = bool(getattr(torch, 'cuda', None)
                                and torch.cuda.is_available())
                    desc = "cpu" if seen else "cpu (no CUDA GPU seen by PyTorch)"
            except Exception:
                pass
            line = f"Inference device: {desc}  (requested: {cfg.device})"
            progress.append(line)
            self._log(line)

        def on_file_done(summary: dict):
            t = summary.get('timing')
            name = os.path.basename(str(summary.get('wav_path', '')))
            if not t:
                return
            line = (f"  {name}: {t.get('audio_dur_s')}s audio in "
                    f"{t.get('t_total')}s  [spec {t.get('t_spec')}s · "
                    f"scan {t.get('t_infer')}s · blobs {t.get('t_blobs')}s] "
                    f"→ {t.get('realtime_factor')}× realtime on "
                    f"{t.get('device')}")
            progress.append(line)
            self._log(line)

        worker.device_signal.connect(on_device)
        worker.file_done_signal.connect(on_file_done)
        worker.progress_signal.connect(on_progress)
        worker.finished_signal.connect(on_finished)
        worker.error_signal.connect(on_error)
        progress.cancel_requested.connect(worker.request_stop)
        worker.start()
        if owns_modal:
            progress.exec_()

    def _toggle_infer_pause(self):
        """Pause/resume the running batch inference. Pausing suspends work
        between tiles (current progress stays in memory); resuming continues."""
        worker = getattr(self, '_infer_worker', None)
        if worker is None or not worker.isRunning():
            return
        if worker.is_paused():
            worker.resume()
            self.btn_infer_pause.setText("Pause")
            self.infer_panel.set_stage("Resumed — running…")
            self._log("Inference resumed")
            self.status_bar.showMessage("Inference resumed")
        else:
            worker.pause()
            self.btn_infer_pause.setText("Resume")
            self.infer_panel.set_stage("Paused — click Resume to continue.")
            self._log("Inference paused")
            self.status_bar.showMessage("Inference paused")

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
        # The latest-model label was dropped with the Project group; models are
        # shown/selected in the Model Training & Inference dropdown instead.
        return

    # ==================================================================
    # Close
    # ==================================================================
    def closeEvent(self, event):
        self._auto_save_mask_if_dirty()
        self._stop_playback()
        super().closeEvent(event)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    MADMainWindow._apply_dark_theme()  # consistent dark look on all platforms
    win = MADMainWindow()
    win.show()
    if QApplication.instance() is app:
        sys.exit(app.exec_())


if __name__ == "__main__":
    main()
