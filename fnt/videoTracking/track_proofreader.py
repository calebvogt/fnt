"""Track proofreading for MTT inference output.

IoU/centroid matching cannot preserve identity through a full occlusion: when
two similarly-coloured animals cross, the tracker will sometimes carry track 1
out the far side as track 2. Detection quality does not fix this — the frames
either side are both correct, only the *assignment* between them is wrong — so
every real dataset needs a place to repair identities by hand. This is that
place, the analogue of SLEAP's proofreading mode.

It reads ``<video>_MaskTracker/<video>_tracks.h5`` (see :mod:`track_store`),
so it never re-runs detection: corrections are edits to one column of stored
identities, and the mask pixels are never touched.

Four operations cover essentially all real repairs:

* **Swap from here** — the identity-crossing fix. Exchanges two tracks for
  every frame at or after the playhead.
* **Merge** — one animal that picked up a new id after an occlusion.
* **Delete** — a spurious track (reflection, shadow, hand entering frame).
* **Rename** — give a track a stable id across a session.

Saving rewrites ``/object_id`` in the store and regenerates
``trajectories.csv`` with identical columns, so the corrected CSV is a drop-in
replacement for the original.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional

import cv2
import numpy as np

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QListWidget,
    QListWidgetItem, QSlider, QMessageBox, QGroupBox, QWidget, QSizePolicy,
    QInputDialog, QAbstractItemView, QSplitter,
)
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QPointF
from PyQt5.QtGui import (
    QPainter, QColor, QPen, QBrush, QImage, QPixmap, QFont,
)

from .track_store import TrackMaskReader, trajectories_dataframe

# Distinct, colour-blind-friendly-ish palette for track identities. Identity is
# the whole point of this view, so tracks must stay visually separable.
TRACK_COLORS = [
    (0, 150, 255), (255, 130, 0), (0, 200, 90), (220, 60, 200),
    (255, 210, 0), (0, 210, 210), (255, 90, 90), (140, 120, 255),
    (150, 200, 40), (255, 150, 190),
]


def track_color(obj_id: int) -> QColor:
    r, g, b = TRACK_COLORS[(int(obj_id) - 1) % len(TRACK_COLORS)]
    return QColor(r, g, b)


class TrackTimeline(QWidget):
    """Occupancy matrix: one row per track, time on x, lit where present.

    This is where an identity swap is actually *visible* — a track that ends
    exactly where another begins is the signature of a crossing gone wrong.
    """

    frame_clicked = pyqtSignal(int)
    track_clicked = pyqtSignal(int)

    ROW_H = 18
    LABEL_W = 54

    def __init__(self, parent=None):
        super().__init__(parent)
        self._ids: List[int] = []
        self._grid = np.zeros((0, 0), dtype=bool)
        self._n_frames = 1
        self._playhead = 0
        self._selected: List[int] = []
        self.setMinimumHeight(80)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setMouseTracking(True)

    def set_data(self, ids: List[int], grid: np.ndarray, n_frames: int):
        self._ids, self._grid = ids, grid
        self._n_frames = max(1, int(n_frames))
        self.setMinimumHeight(max(80, len(ids) * self.ROW_H + 22))
        self.updateGeometry()
        self.update()

    def set_playhead(self, frame_idx: int):
        self._playhead = int(frame_idx)
        self.update()

    def set_selected(self, ids: List[int]):
        self._selected = list(ids)
        self.update()

    def _plot_rect(self) -> QRectF:
        return QRectF(self.LABEL_W, 2, max(1, self.width() - self.LABEL_W - 4),
                      max(1, len(self._ids) * self.ROW_H))

    def mousePressEvent(self, event):
        pr = self._plot_rect()
        y = event.y() - pr.top()
        row = int(y // self.ROW_H)
        if 0 <= row < len(self._ids):
            self.track_clicked.emit(self._ids[row])
        if event.x() >= pr.left():
            frac = (event.x() - pr.left()) / pr.width()
            self.frame_clicked.emit(
                int(np.clip(frac, 0, 1) * (self._n_frames - 1))
            )

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(30, 30, 30))
        if not self._ids:
            p.setPen(QColor(120, 120, 120))
            p.drawText(self.rect(), Qt.AlignCenter, "No tracks")
            return

        pr = self._plot_rect()
        n_bins = self._grid.shape[1]
        bin_w = pr.width() / max(1, n_bins)
        f = p.font()
        f.setPixelSize(10)
        p.setFont(f)

        for i, oid in enumerate(self._ids):
            top = pr.top() + i * self.ROW_H
            selected = oid in self._selected
            if selected:
                p.fillRect(QRectF(0, top, self.width(), self.ROW_H),
                           QColor(45, 65, 95))
            p.setPen(QColor(230, 230, 230) if selected else QColor(160, 160, 160))
            p.drawText(QRectF(4, top, self.LABEL_W - 8, self.ROW_H),
                       Qt.AlignVCenter | Qt.AlignRight, str(oid))

            col = track_color(oid)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(col))
            # Merge adjacent lit bins into single spans so a solid track draws
            # as one bar rather than hundreds of hairlines.
            row = self._grid[i]
            j = 0
            while j < n_bins:
                if not row[j]:
                    j += 1
                    continue
                k = j
                while k < n_bins and row[k]:
                    k += 1
                p.drawRect(QRectF(pr.left() + j * bin_w, top + 3,
                                  max(1.0, (k - j) * bin_w), self.ROW_H - 6))
                j = k

        p.setPen(QPen(QColor(70, 70, 70), 1))
        p.drawLine(int(pr.left()), int(pr.bottom() + 1),
                   int(pr.right()), int(pr.bottom() + 1))

        x = pr.left() + (self._playhead / max(1, self._n_frames - 1)) * pr.width()
        p.setPen(QPen(QColor(255, 255, 255), 1.5))
        p.drawLine(QPointF(x, pr.top()), QPointF(x, pr.bottom() + 4))


class MaskPreview(QWidget):
    """Video frame with track masks drawn as filled, id-coloured silhouettes."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame: Optional[np.ndarray] = None
        self._dets: List[Dict] = []
        self._selected: List[int] = []
        self.setMinimumSize(360, 260)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_content(self, frame_rgb, dets, selected):
        self._frame, self._dets, self._selected = frame_rgb, dets, list(selected)
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(24, 24, 24))
        if self._frame is None:
            p.setPen(QColor(120, 120, 120))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "Video unavailable — masks shown on black")

        fh = self._frame.shape[0] if self._frame is not None else (
            max([d["mask"].shape[0] for d in self._dets if d.get("mask") is not None],
                default=480))
        fw = self._frame.shape[1] if self._frame is not None else (
            max([d["mask"].shape[1] for d in self._dets if d.get("mask") is not None],
                default=640))
        scale = min(self.width() / fw, self.height() / fh)
        ox = (self.width() - fw * scale) / 2
        oy = (self.height() - fh * scale) / 2
        target = QRectF(ox, oy, fw * scale, fh * scale)

        if self._frame is not None:
            h, w = self._frame.shape[:2]
            img = QImage(self._frame.data, w, h, 3 * w, QImage.Format_RGB888)
            p.drawPixmap(target, QPixmap.fromImage(img), QRectF(0, 0, w, h))

        f = p.font()
        f.setPixelSize(max(11, int(15 * min(1.5, scale))))
        f.setBold(True)
        p.setFont(f)

        for det in self._dets:
            oid = det["object_id"]
            col = track_color(oid)
            sel = oid in self._selected
            mask = det.get("mask")
            if mask is not None and mask.any():
                # Tint only the mask pixels; RGBA image keeps the animal visible
                # underneath so the user can judge whether the mask is right.
                overlay = np.zeros((mask.shape[0], mask.shape[1], 4), np.uint8)
                overlay[mask] = [col.red(), col.green(), col.blue(),
                                 150 if sel else 90]
                qimg = QImage(overlay.data, mask.shape[1], mask.shape[0],
                              4 * mask.shape[1], QImage.Format_RGBA8888)
                p.drawPixmap(target, QPixmap.fromImage(qimg),
                             QRectF(0, 0, mask.shape[1], mask.shape[0]))

            x1, y1, x2, y2 = det["bbox"]
            r = QRectF(ox + x1 * scale, oy + y1 * scale,
                       (x2 - x1) * scale, (y2 - y1) * scale)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(col, 3 if sel else 1.5,
                          Qt.SolidLine if sel else Qt.DashLine))
            p.drawRect(r)

            label = f"{oid}"
            p.setPen(QPen(QColor(0, 0, 0), 3))
            p.drawText(QPointF(r.left() + 3, r.top() - 4), label)
            p.setPen(QPen(col))
            p.drawText(QPointF(r.left() + 3, r.top() - 4), label)


class TrackProofreaderDialog(QDialog):
    """Review and repair track identities from a saved inference run."""

    _MAX_UNDO = 30

    def __init__(self, tracks_h5: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Track Proofreader — {os.path.basename(tracks_h5)}")
        self.setMinimumSize(940, 680)
        self.resize(1180, 820)
        self.setStyleSheet(
            "QDialog, QWidget { background-color: #2b2b2b; color: #cccccc; }"
            "QGroupBox { border: 1px solid #555; border-radius: 4px; "
            "margin-top: 8px; padding-top: 12px; font-weight: bold; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
            "QPushButton { background-color: #3c3c3c; border: 1px solid #555; "
            "border-radius: 3px; padding: 5px 10px; }"
            "QPushButton:hover { background-color: #4a4a4a; }"
            "QPushButton:disabled { color: #666; background-color: #333; }"
            "QListWidget { background-color: #1e1e1e; border: 1px solid #444; }"
            "QListWidget::item:selected { background-color: #2979ff; color: white; }"
        )

        self.reader = TrackMaskReader(tracks_h5)
        self.tracks_h5 = tracks_h5
        self.output_dir = os.path.dirname(tracks_h5)
        self._ids = self.reader.object_id.copy()
        self._undo: List[np.ndarray] = []
        self._dirty = False

        self._cap = None
        if self.reader.video_path and os.path.exists(self.reader.video_path):
            cap = cv2.VideoCapture(self.reader.video_path)
            if cap.isOpened():
                self._cap = cap
        self._frame_idx = 0

        self._build_ui()
        self._refresh_tracks()
        self._show_frame(0)

    # -- UI -------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)

        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.preview = MaskPreview()
        lv.addWidget(self.preview, 1)

        nav = QHBoxLayout()
        self.btn_prev = QPushButton("◀")
        self.btn_prev.setToolTip("Previous frame (Left arrow; Shift ±10, Ctrl ±100)")
        self.btn_prev.clicked.connect(lambda: self._step(-1))
        nav.addWidget(self.btn_prev)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMinimum(0)
        self.slider.setMaximum(max(0, self._n_frames() - 1))
        self.slider.valueChanged.connect(self._show_frame)
        self.slider.setStyleSheet(
            "QSlider::groove:horizontal { background: #3c3c3c; height: 6px; border-radius: 3px; }"
            "QSlider::handle:horizontal { background: #2979ff; width: 12px; "
            "margin: -4px 0; border-radius: 6px; }"
        )
        nav.addWidget(self.slider, 1)
        self.btn_next = QPushButton("▶")
        self.btn_next.setToolTip("Next frame (Right arrow; Shift ±10, Ctrl ±100)")
        self.btn_next.clicked.connect(lambda: self._step(1))
        nav.addWidget(self.btn_next)
        self.lbl_frame = QLabel("0 / 0")
        self.lbl_frame.setMinimumWidth(110)
        self.lbl_frame.setAlignment(Qt.AlignCenter)
        nav.addWidget(self.lbl_frame)
        lv.addLayout(nav)
        split.addWidget(left)

        right = QWidget()
        right.setMaximumWidth(330)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)

        g_tracks = QGroupBox("Tracks")
        gv = QVBoxLayout(g_tracks)
        self.track_list = QListWidget()
        self.track_list.setSelectionMode(QAbstractItemView.ExtendedSelection)
        self.track_list.setToolTip(
            "Every identity in this run, with the frames it spans.\n"
            "Select one to highlight it; select two to swap or merge them."
        )
        self.track_list.itemSelectionChanged.connect(self._on_selection_changed)
        self.track_list.itemDoubleClicked.connect(self._goto_track_start)
        gv.addWidget(self.track_list)
        self.lbl_sel = QLabel("Select a track")
        self.lbl_sel.setStyleSheet("color: #999; font-size: 10px;")
        self.lbl_sel.setWordWrap(True)
        gv.addWidget(self.lbl_sel)
        rv.addWidget(g_tracks, 1)

        g_fix = QGroupBox("Repair")
        fv = QVBoxLayout(g_fix)
        fv.setSpacing(4)

        self.btn_swap = QPushButton("Swap IDs from this frame →")
        self.btn_swap.setStyleSheet(
            "QPushButton { background-color: #2979ff; color: white; font-weight: bold; }"
            "QPushButton:hover { background-color: #448aff; }"
            "QPushButton:disabled { background-color: #333; color: #666; }"
        )
        self.btn_swap.setToolTip(
            "Exchange the two selected tracks for every frame from the\n"
            "playhead onward — the fix for an identity swap at a crossing.\n"
            "Scrub to the first frame where the labels are wrong, then apply."
        )
        self.btn_swap.clicked.connect(self._swap_from_here)
        fv.addWidget(self.btn_swap)

        self.btn_merge = QPushButton("Merge B into A")
        self.btn_merge.setToolTip(
            "Relabel every detection of the second selected track with the\n"
            "first track's id — for one animal that was given a new id after\n"
            "an occlusion. Applies across the whole video."
        )
        self.btn_merge.clicked.connect(self._merge)
        fv.addWidget(self.btn_merge)

        self.btn_rename = QPushButton("Rename track…")
        self.btn_rename.setToolTip("Give the selected track a different id number.")
        self.btn_rename.clicked.connect(self._rename)
        fv.addWidget(self.btn_rename)

        self.btn_delete = QPushButton("Delete track")
        self.btn_delete.setStyleSheet(
            "QPushButton { background-color: #c62828; color: white; font-weight: bold; }"
            "QPushButton:hover { background-color: #e53935; }"
            "QPushButton:disabled { background-color: #333; color: #666; }"
        )
        self.btn_delete.setToolTip(
            "Drop the selected track from the output — for reflections,\n"
            "shadows, or a hand entering the arena. Masks stay in the store,\n"
            "so this is reversible until you save (and undoable after)."
        )
        self.btn_delete.clicked.connect(self._delete)
        fv.addWidget(self.btn_delete)

        self.btn_undo = QPushButton("Undo")
        self.btn_undo.setToolTip("Undo the last repair (Ctrl+Z).")
        self.btn_undo.clicked.connect(self._undo_last)
        fv.addWidget(self.btn_undo)
        rv.addWidget(g_fix)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        split.setStretchFactor(1, 0)
        root.addWidget(split, 1)

        g_tl = QGroupBox("Track occupancy — click to seek, click a row to select")
        tv = QVBoxLayout(g_tl)
        tv.setContentsMargins(4, 4, 4, 4)
        self.timeline = TrackTimeline()
        self.timeline.frame_clicked.connect(self._seek)
        self.timeline.track_clicked.connect(self._select_track_id)
        self.timeline.setToolTip(
            "One row per identity, time left to right.\n"
            "A track that stops exactly where another starts is the\n"
            "signature of an identity swap at a crossing."
        )
        tv.addWidget(self.timeline)
        root.addWidget(g_tl)

        bottom = QHBoxLayout()
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #999; font-size: 11px;")
        bottom.addWidget(self.lbl_status, 1)
        self.btn_save = QPushButton("Save Corrections")
        self.btn_save.setStyleSheet(
            "QPushButton { background-color: #2e7d32; color: white; "
            "font-weight: bold; padding: 6px 14px; }"
            "QPushButton:hover { background-color: #388e3c; }"
            "QPushButton:disabled { background-color: #333; color: #666; }"
        )
        self.btn_save.setToolTip(
            "Write corrected identities into the track store and regenerate\n"
            "trajectories.csv. The original CSV is kept as\n"
            "trajectories_original.csv the first time you save."
        )
        self.btn_save.clicked.connect(self._save)
        bottom.addWidget(self.btn_save)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.reject)
        bottom.addWidget(btn_close)
        root.addLayout(bottom)

        self._update_buttons()

    # -- helpers --------------------------------------------------------
    def _n_frames(self) -> int:
        return max(self.reader.frame_count,
                   int(self.reader.frame.max()) + 1 if self.reader.frame.size else 1)

    def _live_ids(self) -> List[int]:
        return sorted({int(v) for v in self._ids if v > 0})

    def _selected_ids(self) -> List[int]:
        return [it.data(Qt.UserRole) for it in self.track_list.selectedItems()]

    def _refresh_tracks(self, keep: Optional[List[int]] = None):
        keep = keep if keep is not None else self._selected_ids()
        self.track_list.blockSignals(True)
        self.track_list.clear()
        for oid in self._live_ids():
            sel = self._ids == oid
            frames = self.reader.frame[sel]
            item = QListWidgetItem(
                f"  Track {oid}   ·   frames {int(frames.min())}–{int(frames.max())}"
                f"   ·   {int(sel.sum())} det"
            )
            item.setData(Qt.UserRole, oid)
            item.setForeground(track_color(oid))
            self.track_list.addItem(item)
            if oid in keep:
                item.setSelected(True)
        self.track_list.blockSignals(False)
        self._refresh_timeline()
        self._on_selection_changed()

    def _refresh_timeline(self):
        ids = self._live_ids()
        n_frames = self._n_frames()
        n_bins = max(120, min(1200, self.timeline.width() - TrackTimeline.LABEL_W))
        grid = np.zeros((len(ids), n_bins), dtype=bool)
        b = np.arange(n_bins, dtype=np.int64)
        lo = b * n_frames // n_bins
        hi = np.maximum((b + 1) * n_frames // n_bins, lo + 1)
        for i, oid in enumerate(ids):
            present = np.zeros(n_frames + 1, dtype=np.int32)
            present[self.reader.frame[self._ids == oid]] = 1
            cum = np.concatenate(([0], np.cumsum(present)))
            grid[i] = (cum[np.minimum(hi, n_frames)] - cum[lo]) > 0
        self.timeline.set_data(ids, grid, n_frames)
        self.timeline.set_playhead(self._frame_idx)
        self.timeline.set_selected(self._selected_ids())

    def _push_undo(self):
        self._undo.append(self._ids.copy())
        if len(self._undo) > self._MAX_UNDO:
            self._undo.pop(0)
        self._dirty = True
        self._update_buttons()

    def _update_buttons(self):
        n = len(self._selected_ids())
        self.btn_swap.setEnabled(n == 2)
        self.btn_merge.setEnabled(n == 2)
        self.btn_rename.setEnabled(n == 1)
        self.btn_delete.setEnabled(n >= 1)
        self.btn_undo.setEnabled(bool(self._undo))
        self.btn_save.setEnabled(self._dirty)
        self.lbl_status.setText(
            "Unsaved corrections" if self._dirty else "No changes"
        )

    # -- navigation -----------------------------------------------------
    def _seek(self, frame_idx: int):
        self.slider.setValue(int(np.clip(frame_idx, 0, self._n_frames() - 1)))

    def _step(self, delta: int):
        self._seek(self._frame_idx + delta)

    def _goto_track_start(self, item):
        oid = item.data(Qt.UserRole)
        frames = self.reader.frame[self._ids == oid]
        if frames.size:
            self._seek(int(frames.min()))

    def _select_track_id(self, oid: int):
        for i in range(self.track_list.count()):
            it = self.track_list.item(i)
            it.setSelected(it.data(Qt.UserRole) == oid)

    def _show_frame(self, frame_idx: int):
        self._frame_idx = int(frame_idx)
        frame_rgb = None
        if self._cap is not None:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._frame_idx)
            ret, bgr = self._cap.read()
            if ret:
                frame_rgb = np.ascontiguousarray(
                    cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)
                )

        dets = []
        for row in self.reader.rows_for_frame(self._frame_idx):
            oid = int(self._ids[row])
            if oid <= 0:
                continue
            dets.append({
                "object_id": oid,
                "bbox": self.reader.bbox[row].tolist(),
                "mask": self.reader.mask_for_row(row),
            })
        self.preview.set_content(frame_rgb, dets, self._selected_ids())
        self.timeline.set_playhead(self._frame_idx)
        t = self._frame_idx / max(self.reader.fps, 1e-6)
        self.lbl_frame.setText(
            f"{self._frame_idx} / {self._n_frames() - 1}   ({t:.1f}s)"
        )

    def _on_selection_changed(self):
        sel = self._selected_ids()
        self.timeline.set_selected(sel)
        self.preview.set_content(
            self.preview._frame, self.preview._dets, sel
        )
        if len(sel) == 2:
            self.lbl_sel.setText(
                f"A = track {sel[0]}, B = track {sel[1]}. "
                "Swap exchanges them from the playhead on; Merge relabels all "
                f"of {sel[1]} as {sel[0]}."
            )
        elif len(sel) == 1:
            self.lbl_sel.setText(f"Track {sel[0]} selected.")
        else:
            self.lbl_sel.setText(
                "Select a track. Two tracks enable Swap and Merge."
            )
        self._update_buttons()

    # -- repairs --------------------------------------------------------
    def _swap_from_here(self):
        sel = self._selected_ids()
        if len(sel) != 2:
            return
        a, b = sel
        at = self.reader.frame >= self._frame_idx
        self._push_undo()
        ids = self._ids
        a_rows = at & (ids == a)
        b_rows = at & (ids == b)
        ids[a_rows] = b
        ids[b_rows] = a
        n = int(a_rows.sum() + b_rows.sum())
        self._refresh_tracks(keep=[a, b])
        self._show_frame(self._frame_idx)
        self.lbl_status.setText(
            f"Swapped {a} ↔ {b} from frame {self._frame_idx} "
            f"({n} detections) — unsaved"
        )

    def _merge(self):
        sel = self._selected_ids()
        if len(sel) != 2:
            return
        a, b = sel
        if QMessageBox.question(
            self, "Merge Tracks",
            f"Relabel every detection of track {b} as track {a}?\n\n"
            f"This applies across the whole video and cannot be split again "
            f"except by Undo.",
            QMessageBox.Yes | QMessageBox.Cancel,
        ) != QMessageBox.Yes:
            return
        self._push_undo()
        n = int((self._ids == b).sum())
        self._ids[self._ids == b] = a
        self._refresh_tracks(keep=[a])
        self._show_frame(self._frame_idx)
        self.lbl_status.setText(
            f"Merged track {b} into {a} ({n} detections) — unsaved"
        )

    def _rename(self):
        sel = self._selected_ids()
        if len(sel) != 1:
            return
        old = sel[0]
        new, ok = QInputDialog.getInt(
            self, "Rename Track", f"New id for track {old}:", old, 1, 9999
        )
        if not ok or new == old:
            return
        if new in self._live_ids():
            if QMessageBox.question(
                self, "Id In Use",
                f"Track {new} already exists. Merge track {old} into it?",
                QMessageBox.Yes | QMessageBox.Cancel,
            ) != QMessageBox.Yes:
                return
        self._push_undo()
        self._ids[self._ids == old] = new
        self._refresh_tracks(keep=[new])
        self._show_frame(self._frame_idx)
        self.lbl_status.setText(f"Renamed track {old} → {new} — unsaved")

    def _delete(self):
        sel = self._selected_ids()
        if not sel:
            return
        n = int(np.isin(self._ids, sel).sum())
        if QMessageBox.question(
            self, "Delete Track(s)",
            f"Remove {len(sel)} track(s) ({n} detections) from the output?\n\n"
            f"Track(s): {', '.join(str(s) for s in sel)}\n\n"
            "The masks stay in the store; this only drops them from the CSV. "
            "Undo restores them.",
            QMessageBox.Yes | QMessageBox.Cancel,
        ) != QMessageBox.Yes:
            return
        self._push_undo()
        self._ids[np.isin(self._ids, sel)] = 0   # 0 = deleted
        self._refresh_tracks(keep=[])
        self._show_frame(self._frame_idx)
        self.lbl_status.setText(f"Deleted {len(sel)} track(s) — unsaved")

    def _undo_last(self):
        if not self._undo:
            return
        self._ids = self._undo.pop()
        self._dirty = bool(self._undo)
        self._refresh_tracks()
        self._show_frame(self._frame_idx)
        self.lbl_status.setText("Undid last repair")
        self._update_buttons()

    # -- save -----------------------------------------------------------
    def _save(self):
        csv_path = os.path.join(self.output_dir, "trajectories.csv")
        backup = os.path.join(self.output_dir, "trajectories_original.csv")
        try:
            # Keep the untouched inference output the first time only, so
            # repeated saves can't overwrite the true original with a
            # half-corrected version.
            if os.path.exists(csv_path) and not os.path.exists(backup):
                import shutil
                shutil.copy2(csv_path, backup)

            categories = self._load_categories()
            df = trajectories_dataframe(self.reader, self._ids, categories)
            df.to_csv(csv_path, index=False)
            self.reader.write_object_ids(self._ids)
        except Exception as e:
            QMessageBox.critical(
                self, "Save Failed", f"Could not save corrections:\n\n{e}"
            )
            return

        self._undo.clear()
        self._dirty = False
        self._update_buttons()
        n_tracks = len(self._live_ids())
        self.lbl_status.setText(
            f"Saved — {n_tracks} tracks, {len(df)} rows to trajectories.csv"
        )
        QMessageBox.information(
            self, "Corrections Saved",
            f"Wrote {len(df)} rows across {n_tracks} track(s).\n\n"
            f"CSV: {csv_path}\n"
            f"Store: {os.path.basename(self.tracks_h5)}\n"
            + (f"Original CSV kept as {os.path.basename(backup)}"
               if os.path.exists(backup) else "")
        )

    def _load_categories(self) -> Optional[Dict]:
        """Class names from the model that produced this run, for object_name."""
        import json
        cfg = os.path.join(self.reader.model_dir, "training_config.json")
        if not os.path.exists(cfg):
            return None
        try:
            with open(cfg) as fh:
                return json.load(fh).get("categories") or None
        except Exception:
            return None

    # -- lifecycle ------------------------------------------------------
    def keyPressEvent(self, event):
        k = event.key()
        mods = event.modifiers()
        step = 100 if mods & Qt.ControlModifier else (
            10 if mods & Qt.ShiftModifier else 1)
        if k == Qt.Key_Left:
            self._step(-step)
        elif k == Qt.Key_Right:
            self._step(step)
        elif k == Qt.Key_Z and mods & Qt.ControlModifier:
            self._undo_last()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._refresh_timeline()

    def reject(self):
        if self._dirty:
            r = QMessageBox.question(
                self, "Discard Corrections?",
                "You have unsaved track corrections. Close anyway?",
                QMessageBox.Discard | QMessageBox.Cancel,
            )
            if r != QMessageBox.Discard:
                return
        self._cleanup()
        super().reject()

    def _cleanup(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        try:
            self.reader.close()
        except Exception:
            pass

    def closeEvent(self, event):
        self._cleanup()
        super().closeEvent(event)
