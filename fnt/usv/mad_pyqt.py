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
from PyQt5.QtCore import Qt, QSettings, QThread, QTimer, QRectF, pyqtSignal
from PyQt5.QtGui import QImage, QKeySequence, QPainter, QPen, QColor, QBrush
from PyQt5.QtWidgets import (
    QAction, QApplication, QCheckBox, QComboBox, QDialog, QDialogButtonBox,
    QDoubleSpinBox, QFileDialog, QFormLayout, QFrame,
    QGroupBox, QHBoxLayout, QInputDialog, QLabel, QListWidget,
    QListWidgetItem, QMainWindow, QMessageBox, QProgressBar, QPushButton,
    QRadioButton, QScrollArea, QScrollBar, QShortcut, QSizePolicy, QSlider,
    QSpinBox, QSplitter, QStatusBar, QTextEdit, QVBoxLayout, QWidget,
)
from scipy import signal

from fnt.usv.audio_widgets import SpectrogramWidget, WaveformOverviewWidget
from fnt.usv.usv_detector.mad_labels import (
    committed_columns, pred_csv_sibling_path, pred_mask_sibling_path,
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

    def __init__(self, parent=None):
        super().__init__(parent)
        self.paint_mode: Optional[str] = None  # 'brush' | 'eraser' | None
        self.brush_radius_px = 6              # in spec-pixel units

        self.mask: Optional[np.ndarray] = None
        self.n_freq_bins: Optional[int] = None
        self.n_time_frames: Optional[int] = None
        self.hop: Optional[int] = None

        self._mask_dirty = False
        self._painting = False
        self._last_paint_idx: Optional[tuple] = None

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
        self.n_freq_bins = nfft // 2 + 1
        self.n_time_frames = max(
            1, (audio_len - nperseg) // self.hop + 1
        )
        self.mask = np.zeros(
            (self.n_freq_bins, self.n_time_frames), dtype=np.uint8
        )
        self._mask_dirty = False
        self.update()

    def set_mask(self, arr: np.ndarray) -> None:
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
        self.paint_mode = mode
        if mode in ('brush', 'eraser'):
            # Hide the OS cursor — the paintEvent draws a brush-radius
            # circle instead, which doubles as a visual size indicator.
            self.setCursor(Qt.BlankCursor)
        else:
            self.setCursor(Qt.ArrowCursor)
            self._cursor_pos = None
        self.update()

    def set_brush_radius(self, r: int) -> None:
        self.brush_radius_px = max(1, int(r))

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
        if self.mask is None:
            return
        value = MASK_POSITIVE if self.paint_mode == 'brush' else MASK_UNLABELED
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
        self.mask[f0:f1, t0:t1][disk] = value
        self._mask_dirty = True

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

        if self.view_mode == 'mask_only':
            # Solid dark fill so the spec doesn't show through; then
            # render positives/negatives at near-full alpha.
            rgba[:] = (32, 32, 32, 255)
            view_mask = self.mask[f_start:f_end, t_start:t_end]
            pos = view_mask == MASK_POSITIVE
            rgba[pos] = (255, 0, 220, 255)
            # Certified negatives derived from committed columns.
            committed = committed_columns(self.mask)
            committed_slice = committed[t_start:t_end]
            neg_full = np.broadcast_to(
                committed_slice[np.newaxis, :], view_mask.shape
            ) & (view_mask == MASK_UNLABELED)
            rgba[neg_full] = (255, 220, 0, 200)
        elif self.view_mode == 'overlay':
            view_mask = self.mask[f_start:f_end, t_start:t_end]
            pos = view_mask == MASK_POSITIVE
            pos_alpha = int(self.mask_alpha * 255)
            rgba[pos] = (255, 0, 220, pos_alpha)
            # Yellow tint over committed columns (derived, not stored).
            committed = committed_columns(self.mask)
            committed_slice = committed[t_start:t_end]
            neg_alpha = max(0, pos_alpha - 90)
            if committed_slice.any():
                col_idx = np.where(committed_slice)[0]
                for c in col_idx:
                    # Dim yellow strip over full freq range of the view.
                    # Skip pixels that already got painted magenta.
                    strip = rgba[:, c, :]
                    not_painted = strip[:, 3] == 0
                    strip[not_painted] = (255, 220, 0, neg_alpha)
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

        # ---------- Left panel (scrollable) ----------
        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        left_scroll.setVerticalScrollBarPolicy(Qt.ScrollBarAsNeeded)
        _fm = self.fontMetrics()
        _min_w = max(360, _fm.averageCharWidth() * 56 + 40)
        _max_w = max(460, _fm.averageCharWidth() * 74 + 40)
        left_scroll.setMinimumWidth(_min_w)
        left_scroll.setMaximumWidth(_max_w)

        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(5, 5, 5, 5)
        left_layout.setSpacing(8)

        self._create_project_section(left_layout)
        self._create_training_data_section(left_layout)
        self._create_paint_tools_section(left_layout)
        self._create_view_section(left_layout)
        self._create_review_section(left_layout)

        left_layout.addStretch()
        left_scroll.setWidget(left_widget)

        # ---------- Right panel ----------
        right_panel = QWidget()
        right_layout = QVBoxLayout(right_panel)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(2)

        self.spectrogram = MADSpectrogramWidget()
        self.spectrogram.zoom_requested.connect(self._on_wheel_zoom)
        self.spectrogram.stroke_committed.connect(self._on_stroke_committed)
        right_layout.addWidget(self.spectrogram, 1)

        self.waveform_overview = WaveformOverviewWidget()
        self.waveform_overview.view_changed.connect(self._on_overview_clicked)
        right_layout.addWidget(self.waveform_overview)

        # Scrollbar + pan row
        scroll_bar_row = QWidget()
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

        main_layout.addWidget(left_scroll)
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
        group = QGroupBox("3. Paint Tools")
        vbox = QVBoxLayout()
        vbox.setSpacing(4)

        mode_row = QHBoxLayout()
        mode_row.setSpacing(2)

        self.btn_paint = QPushButton("Brush (B)")
        self.btn_paint.setToolTip(
            "Paint target USV pixels (left-click + drag)\n"
            "Shortcut: B"
        )
        self.btn_paint.setCheckable(True)
        self.btn_paint.clicked.connect(self._on_brush_clicked)
        self.btn_paint.setEnabled(False)
        mode_row.addWidget(self.btn_paint)

        self.btn_erase = QPushButton("Eraser (E)")
        self.btn_erase.setToolTip(
            "Erase painted pixels\n"
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

        hint = QLabel(
            "Connected painted regions define 'committed' time bands;\n"
            "any unpainted pixel inside those bands becomes certified negative."
        )
        hint.setStyleSheet("color: #888888; font-size: 9px; font-style: italic;")
        hint.setWordWrap(True)
        vbox.addWidget(hint)

        group.setLayout(vbox)
        layout.addWidget(group)

    def _create_view_section(self, layout):
        group = QGroupBox("4. Mask View")
        vbox = QVBoxLayout()
        vbox.setSpacing(4)

        row = QHBoxLayout()
        row.setSpacing(2)
        self.btn_view_spec = QPushButton("Spec")
        self.btn_view_spec.setCheckable(True)
        self.btn_view_spec.setToolTip("Spectrogram only — hide mask overlay")
        self.btn_view_spec.clicked.connect(
            lambda: self._set_view_mode('spec')
        )
        row.addWidget(self.btn_view_spec)

        self.btn_view_overlay = QPushButton("Spec + Mask")
        self.btn_view_overlay.setCheckable(True)
        self.btn_view_overlay.setChecked(True)
        self.btn_view_overlay.setToolTip(
            "Show spectrogram with mask overlay (default)"
        )
        self.btn_view_overlay.clicked.connect(
            lambda: self._set_view_mode('overlay')
        )
        row.addWidget(self.btn_view_overlay)

        self.btn_view_mask = QPushButton("Mask Only")
        self.btn_view_mask.setCheckable(True)
        self.btn_view_mask.setToolTip(
            "Hide spectrogram — show paint + committed-band negatives only"
        )
        self.btn_view_mask.clicked.connect(
            lambda: self._set_view_mode('mask_only')
        )
        row.addWidget(self.btn_view_mask)

        vbox.addLayout(row)

        hint = QLabel(
            "Magenta = positives you painted · Yellow = certified\n"
            "negatives (columns inside committed bands) · Cyan = predicted"
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
        group = QGroupBox("5. Blob Review")
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
        project_dir = os.path.join(parent_dir, name)
        if os.path.exists(project_dir):
            QMessageBox.warning(
                self, "Project exists",
                f"A directory named '{name}' already exists in:\n{parent_dir}"
            )
            return
        try:
            cfg = create_mad_project(project_dir)
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to create project:\n{e}")
            return
        self._close_project(silent=True)
        self._activate_project(cfg)
        self._remember_recent(project_dir)
        self.status_bar.showMessage(f"Created project: {project_dir}")

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
        """Initialize an empty mask at the project's spec-pixel resolution,
        then overwrite it from the sibling PNG if one exists."""
        if (self._project is None or self.audio_data is None or
                self.sample_rate is None):
            return
        cfg = self._project
        self.spectrogram.init_mask(
            audio_len=len(self.audio_data),
            sample_rate=self.sample_rate,
            nperseg=cfg.nperseg, noverlap=cfg.noverlap, nfft=cfg.nfft,
        )
        wav_path = self.audio_files[self.current_file_idx]
        png_path = _mask_sibling_path(wav_path)
        if os.path.isfile(png_path):
            try:
                arr = _load_mask_png(png_path)
                self.spectrogram.set_mask(arr)
                self.lbl_mask_status.setText(
                    f"Loaded: {os.path.basename(png_path)}"
                )
            except Exception as e:
                self.lbl_mask_status.setText(f"Load failed: {e}")
        else:
            self.lbl_mask_status.setText(
                "Empty mask — start painting to label"
            )

    def _save_current_mask(self):
        if (self._project is None or not self.audio_files or
                self.spectrogram.mask is None):
            return
        if not HAS_PIL:
            QMessageBox.critical(
                self, "PIL missing",
                "Pillow is required to save MAD masks. Install with:\n"
                "    pip install pillow"
            )
            return
        wav_path = self.audio_files[self.current_file_idx]
        png_path = _mask_sibling_path(wav_path)
        try:
            _save_mask_png(png_path, self.spectrogram.mask)
        except Exception as e:
            QMessageBox.critical(self, "Save failed", str(e))
            return
        self.spectrogram._mask_dirty = False
        self.lbl_mask_status.setText(f"Saved: {os.path.basename(png_path)}")
        self.status_bar.showMessage(f"Saved mask → {png_path}")

    def _auto_save_mask_if_dirty(self):
        if (self._project is None or not self.audio_files or
                self.spectrogram.mask is None or
                not self.spectrogram.is_mask_dirty() or
                not HAS_PIL):
            return
        try:
            self._save_current_mask()
        except Exception:
            pass

    def _on_stroke_committed(self):
        """A paint stroke just ended — persist the mask to its sibling
        PNG so the user never has to click Save explicitly."""
        self._auto_save_mask_if_dirty()

    # ==================================================================
    # Paint tools
    # ==================================================================
    def _update_paint_buttons_enabled(self):
        has_audio = self.audio_data is not None
        for btn in (self.btn_paint, self.btn_erase, self.btn_clear_mask):
            btn.setEnabled(has_audio)
        self.spin_brush_radius.setEnabled(has_audio)
        self.btn_load_predictions.setEnabled(has_audio)
        if not has_audio:
            self.btn_paint.setChecked(False)
            self.btn_erase.setChecked(False)
            self.spectrogram.set_paint_mode(None)
            self.lbl_mask_status.setText("No mask")

    def _on_brush_clicked(self, checked: bool):
        if checked:
            self.btn_erase.setChecked(False)
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
            self.spectrogram.set_paint_mode('eraser')
            self.status_bar.showMessage(
                "Eraser mode — left-click + drag to clear painted pixels"
            )
        else:
            self.spectrogram.set_paint_mode(None)
            self.status_bar.showMessage("Paint mode off")

    def _on_clear_clicked(self):
        if self.spectrogram.mask is None:
            return
        reply = QMessageBox.question(
            self, "Clear mask",
            "Erase all paint on the current file?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No
        )
        if reply != QMessageBox.Yes:
            return
        self.spectrogram.clear_mask()
        self.lbl_mask_status.setText("Mask cleared (unsaved)")

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
        if self._project is None:
            QMessageBox.information(self, "No project", "Open a MAD project first.")
            return
        self._auto_save_mask_if_dirty()
        labeled = self._labeled_wavs()
        if not labeled:
            QMessageBox.warning(
                self, "No labels",
                "No wav files have sibling _FNT_MAD_labels.png yet.\n"
                "Paint some USV pixels on at least one file first."
            )
            return
        dlg = RunTrainingDialog(
            self, self._project.project_dir, labeled, self._spec_params(),
        )
        if dlg.exec_() != QDialog.Accepted:
            return
        cfg = dlg.build_config()
        post_scope = (dlg.post_inference_scope()
                      if dlg.post_inference_requested() else None)
        self._start_training(cfg, post_inference_scope=post_scope)

    def _start_training(self, cfg, post_inference_scope: Optional[str] = None):
        progress = MADRunProgressDialog(self, "MAD Training", show_plot=True)
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
                            'arch': cfg.encoder_name,
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
        )
        self._start_inference(cfg, wavs)

    def _start_inference(self, cfg, wav_paths: List[str]):
        progress = MADRunProgressDialog(self, "MAD Inference")
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
