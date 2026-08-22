"""Ethogram review and bout-level analysis for MTT behavior predictions.

The classifier emits one prediction per frame per animal
(``behavior_predictions.csv``). That is the right storage format and the wrong
review format: nobody proofreads behavior a frame at a time, and no paper
reports per-frame labels. Both jobs work on **bouts** — contiguous runs of one
behavior — so this module converts between the two representations and
provides the timeline where bouts are actually inspected and corrected.

The exports are the numbers behavioral papers report: bout counts, total and
mean durations, percentage of session, latency to first occurrence, and the
behavior-to-behavior transition matrix.

Corrections are written back as per-frame rows, so a proofread
``behavior_predictions.csv`` stays a drop-in replacement for the original.
"""
from __future__ import annotations

import csv
import os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from PyQt5.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QWidget,
    QSizePolicy, QSlider, QGroupBox, QComboBox, QMessageBox, QSplitter,
    QTreeWidget, QTreeWidgetItem, QHeaderView, QFileDialog,
)
from PyQt5.QtCore import Qt, pyqtSignal, QRectF, QPointF
from PyQt5.QtGui import QPainter, QColor, QPen, QBrush, QImage, QPixmap

# Shared with the Behavior tab's category editor so a behavior keeps the same
# colour everywhere it appears — category list, video overlay, ethogram.
BEHAVIOR_COLORS = [
    "#e53935", "#43a047", "#1e88e5", "#fb8c00",
    "#8e24aa", "#00acc1", "#ffb300", "#6d4c41",
]
NC_COLOR = "#5c5c5c"
NC_LABEL = "NC"


def behavior_palette(names: List[str]) -> Dict[str, str]:
    """Stable name -> colour map. NC is always neutral grey, never a hue."""
    palette = {}
    i = 0
    for n in names:
        if n == NC_LABEL:
            palette[n] = NC_COLOR
        else:
            palette[n] = BEHAVIOR_COLORS[i % len(BEHAVIOR_COLORS)]
            i += 1
    return palette


@dataclass
class Bout:
    """One contiguous run of a single behavior for one animal."""
    object_id: int
    behavior: str
    start: int            # inclusive
    end: int              # inclusive
    mean_conf: float = 0.0

    @property
    def n_frames(self) -> int:
        return self.end - self.start + 1

    def duration_s(self, fps: float) -> float:
        return self.n_frames / max(fps, 1e-6)


# ---------------------------------------------------------------------------
# Per-frame  <->  bout conversion
# ---------------------------------------------------------------------------

def load_predictions(csv_path: str) -> List[Dict]:
    """Read behavior_predictions.csv into per-frame dicts."""
    rows = []
    with open(csv_path, newline="") as f:
        for r in csv.DictReader(f):
            try:
                rows.append({
                    "frame": int(r["frame"]),
                    "object_id": int(r["object_id"]),
                    "behavior": r["behavior"],
                    "confidence": float(r.get("confidence") or 0.0),
                    "gap": float(r.get("gap") or 0.0),
                })
            except (ValueError, KeyError):
                continue
    return rows


def bouts_from_frames(rows: List[Dict]) -> List[Bout]:
    """Collapse per-frame predictions into bouts.

    A bout breaks on a behavior change *or* a gap in frame numbers — an animal
    that vanishes for 40 frames and returns doing the same thing has not been
    doing it continuously, and merging across that would inflate durations.
    """
    per_obj: Dict[int, List[Dict]] = {}
    for r in rows:
        per_obj.setdefault(r["object_id"], []).append(r)

    bouts: List[Bout] = []
    for oid, obj_rows in per_obj.items():
        obj_rows.sort(key=lambda r: r["frame"])
        start = prev = obj_rows[0]["frame"]
        beh = obj_rows[0]["behavior"]
        confs = [obj_rows[0]["confidence"]]
        for r in obj_rows[1:]:
            contiguous = r["frame"] == prev + 1
            if r["behavior"] != beh or not contiguous:
                bouts.append(Bout(oid, beh, start, prev,
                                  float(np.mean(confs)) if confs else 0.0))
                start, beh, confs = r["frame"], r["behavior"], []
            confs.append(r["confidence"])
            prev = r["frame"]
        bouts.append(Bout(oid, beh, start, prev,
                          float(np.mean(confs)) if confs else 0.0))
    bouts.sort(key=lambda b: (b.object_id, b.start))
    return bouts


def bouts_to_frames(bouts: List[Bout]) -> List[Dict]:
    """Expand bouts back to per-frame rows for saving."""
    rows = []
    for b in bouts:
        for f in range(b.start, b.end + 1):
            rows.append({
                "frame": f,
                "object_id": b.object_id,
                "behavior": b.behavior,
                "confidence": round(b.mean_conf, 4),
                "gap": 0.0,
            })
    rows.sort(key=lambda r: (r["object_id"], r["frame"]))
    return rows


def save_predictions(csv_path: str, rows: List[Dict]):
    fieldnames = ["frame", "object_id", "behavior", "confidence", "gap"]
    with open(csv_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        w.writerows(rows)


# ---------------------------------------------------------------------------
# Bout-level analysis
# ---------------------------------------------------------------------------

def bout_table(bouts: List[Bout], fps: float):
    """One row per bout — the interchange format for external stats tools."""
    import pandas as pd
    return pd.DataFrame([{
        "object_id": b.object_id,
        "behavior": b.behavior,
        "start_frame": b.start,
        "end_frame": b.end,
        "n_frames": b.n_frames,
        "start_s": round(b.start / max(fps, 1e-6), 3),
        "end_s": round((b.end + 1) / max(fps, 1e-6), 3),
        "duration_s": round(b.duration_s(fps), 3),
        "mean_confidence": round(b.mean_conf, 4),
    } for b in bouts])


def bout_summary(bouts: List[Bout], fps: float):
    """Per animal x behavior stats — what actually goes into a results table."""
    import pandas as pd
    if not bouts:
        return pd.DataFrame()

    per_obj_frames: Dict[int, int] = {}
    for b in bouts:
        per_obj_frames[b.object_id] = per_obj_frames.get(b.object_id, 0) + b.n_frames

    grouped: Dict[Tuple[int, str], List[Bout]] = {}
    for b in bouts:
        grouped.setdefault((b.object_id, b.behavior), []).append(b)

    out = []
    for (oid, beh), bs in sorted(grouped.items()):
        frames = sum(b.n_frames for b in bs)
        durs = [b.duration_s(fps) for b in bs]
        total_frames = per_obj_frames.get(oid, 0) or 1
        out.append({
            "object_id": oid,
            "behavior": beh,
            "n_bouts": len(bs),
            "total_frames": frames,
            "total_s": round(frames / max(fps, 1e-6), 3),
            "percent_time": round(100.0 * frames / total_frames, 2),
            "mean_bout_s": round(float(np.mean(durs)), 3),
            "median_bout_s": round(float(np.median(durs)), 3),
            "min_bout_s": round(float(np.min(durs)), 3),
            "max_bout_s": round(float(np.max(durs)), 3),
            # Latency measured from the start of the recording, which is what
            # "latency to first X" means in a behavioral protocol.
            "latency_to_first_s": round(min(b.start for b in bs) / max(fps, 1e-6), 3),
        })
    return pd.DataFrame(out)


def transition_matrix(bouts: List[Bout]):
    """Counts of behavior -> next behavior, per animal and pooled."""
    import pandas as pd
    per_obj: Dict[int, List[Bout]] = {}
    for b in bouts:
        per_obj.setdefault(b.object_id, []).append(b)

    counts: Dict[Tuple[int, str, str], int] = {}
    for oid, bs in per_obj.items():
        bs = sorted(bs, key=lambda b: b.start)
        for a, c in zip(bs, bs[1:]):
            key = (oid, a.behavior, c.behavior)
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return pd.DataFrame()
    return pd.DataFrame([
        {"object_id": o, "from_behavior": f, "to_behavior": t, "count": n}
        for (o, f, t), n in sorted(counts.items())
    ])


# ---------------------------------------------------------------------------
# Timeline widget
# ---------------------------------------------------------------------------

class EthogramTimeline(QWidget):
    """Behavior bouts as coloured spans, one row per animal."""

    bout_clicked = pyqtSignal(object)      # Bout or None
    frame_clicked = pyqtSignal(int)

    ROW_H = 26
    LABEL_W = 54

    def __init__(self, parent=None):
        super().__init__(parent)
        self._bouts: List[Bout] = []
        self._ids: List[int] = []
        self._palette: Dict[str, str] = {}
        self._n_frames = 1
        self._playhead = 0
        self._selected: Optional[Bout] = None
        self.setMinimumHeight(70)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_data(self, bouts, ids, palette, n_frames):
        self._bouts, self._ids = bouts, ids
        self._palette, self._n_frames = palette, max(1, int(n_frames))
        self.setMinimumHeight(max(70, len(ids) * self.ROW_H + 16))
        self.updateGeometry()
        self.update()

    def set_playhead(self, f):
        self._playhead = int(f)
        self.update()

    def set_selected(self, bout):
        self._selected = bout
        self.update()

    def _plot_rect(self) -> QRectF:
        return QRectF(self.LABEL_W, 4,
                      max(1, self.width() - self.LABEL_W - 6),
                      max(1, len(self._ids) * self.ROW_H))

    def _x_of(self, frame: int) -> float:
        pr = self._plot_rect()
        return pr.left() + (frame / max(1, self._n_frames)) * pr.width()

    def mousePressEvent(self, event):
        pr = self._plot_rect()
        row = int((event.y() - pr.top()) // self.ROW_H)
        frame = int(np.clip(
            (event.x() - pr.left()) / pr.width(), 0, 1) * (self._n_frames - 1))
        hit = None
        if 0 <= row < len(self._ids) and event.x() >= pr.left():
            oid = self._ids[row]
            for b in self._bouts:
                if b.object_id == oid and b.start <= frame <= b.end:
                    hit = b
                    break
        self.bout_clicked.emit(hit)
        if event.x() >= pr.left():
            self.frame_clicked.emit(frame)

    def paintEvent(self, event):
        p = QPainter(self)
        p.fillRect(self.rect(), QColor(30, 30, 30))
        if not self._ids:
            p.setPen(QColor(120, 120, 120))
            p.drawText(self.rect(), Qt.AlignCenter, "No behavior predictions")
            return

        pr = self._plot_rect()
        f = p.font()
        f.setPixelSize(10)
        p.setFont(f)
        row_of = {oid: i for i, oid in enumerate(self._ids)}

        for i, oid in enumerate(self._ids):
            top = pr.top() + i * self.ROW_H
            p.setPen(QColor(160, 160, 160))
            p.drawText(QRectF(4, top, self.LABEL_W - 8, self.ROW_H),
                       Qt.AlignVCenter | Qt.AlignRight, str(oid))
            p.fillRect(QRectF(pr.left(), top + 4, pr.width(), self.ROW_H - 8),
                       QColor(22, 22, 22))

        for b in self._bouts:
            i = row_of.get(b.object_id)
            if i is None:
                continue
            top = pr.top() + i * self.ROW_H
            x0 = self._x_of(b.start)
            x1 = self._x_of(b.end + 1)
            col = QColor(self._palette.get(b.behavior, NC_COLOR))
            rect = QRectF(x0, top + 4, max(1.0, x1 - x0), self.ROW_H - 8)
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(col))
            p.drawRect(rect)
            if self._selected is not None and b is self._selected:
                p.setBrush(Qt.NoBrush)
                p.setPen(QPen(QColor(255, 255, 255), 2))
                p.drawRect(rect.adjusted(-1, -1, 1, 1))

        x = self._x_of(self._playhead)
        p.setPen(QPen(QColor(255, 255, 255), 1.5))
        p.drawLine(QPointF(x, pr.top()), QPointF(x, pr.bottom() + 4))


class EthogramDialog(QDialog):
    """Review, correct, and export behavior bouts for one video."""

    _MAX_UNDO = 40

    def __init__(self, behavior_csv: str, tracks_h5: Optional[str] = None,
                 video_path: Optional[str] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Ethogram — {os.path.basename(os.path.dirname(behavior_csv))}")
        self.setMinimumSize(940, 660)
        self.resize(1200, 820)
        self.setStyleSheet(
            "QDialog, QWidget { background-color: #2b2b2b; color: #cccccc; }"
            "QGroupBox { border: 1px solid #555; border-radius: 4px; "
            "margin-top: 8px; padding-top: 12px; font-weight: bold; }"
            "QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }"
            "QPushButton { background-color: #3c3c3c; border: 1px solid #555; "
            "border-radius: 3px; padding: 5px 10px; }"
            "QPushButton:hover { background-color: #4a4a4a; }"
            "QPushButton:disabled { color: #666; background-color: #333; }"
            "QComboBox { background-color: #3c3c3c; border: 1px solid #555; "
            "border-radius: 3px; padding: 3px 6px; }"
            "QTreeWidget { background-color: #1e1e1e; border: 1px solid #444; }"
            "QHeaderView::section { background-color: #333; border: 1px solid #444; "
            "padding: 2px 6px; }"
        )

        self.behavior_csv = behavior_csv
        self.output_dir = os.path.dirname(behavior_csv)
        self._bouts = bouts_from_frames(load_predictions(behavior_csv))
        self._undo: List[List[Bout]] = []
        self._dirty = False
        self._selected: Optional[Bout] = None
        self._frame_idx = 0

        self.fps = 30.0
        self.reader = None
        self._cap = None
        if tracks_h5 and os.path.exists(tracks_h5):
            try:
                from .track_store import TrackMaskReader
                self.reader = TrackMaskReader(tracks_h5)
                self.fps = self.reader.fps
                video_path = video_path or self.reader.video_path
            except Exception:
                self.reader = None
        if video_path and os.path.exists(video_path):
            import cv2
            cap = cv2.VideoCapture(video_path)
            if cap.isOpened():
                self._cap = cap
                if not self.reader:
                    self.fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

        self._build_ui()
        self._refresh_all()
        self._show_frame(0)

    # -- data helpers ---------------------------------------------------
    def _n_frames(self) -> int:
        if self.reader is not None and self.reader.frame_count:
            return self.reader.frame_count
        return max((b.end for b in self._bouts), default=0) + 1

    def _object_ids(self) -> List[int]:
        return sorted({b.object_id for b in self._bouts})

    def _behavior_names(self) -> List[str]:
        names = sorted({b.behavior for b in self._bouts if b.behavior != NC_LABEL})
        if any(b.behavior == NC_LABEL for b in self._bouts):
            names.append(NC_LABEL)
        return names

    # -- UI -------------------------------------------------------------
    def _build_ui(self):
        root = QVBoxLayout(self)
        root.setSpacing(6)

        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)

        left = QWidget()
        lv = QVBoxLayout(left)
        lv.setContentsMargins(0, 0, 0, 0)
        self.preview = _EthogramPreview()
        lv.addWidget(self.preview, 1)

        nav = QHBoxLayout()
        btn_prev = QPushButton("◀")
        btn_prev.setToolTip("Previous frame (Left; Shift ±10, Ctrl ±100)")
        btn_prev.clicked.connect(lambda: self._step(-1))
        nav.addWidget(btn_prev)
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setMaximum(max(0, self._n_frames() - 1))
        self.slider.valueChanged.connect(self._show_frame)
        self.slider.setStyleSheet(
            "QSlider::groove:horizontal { background: #3c3c3c; height: 6px; border-radius: 3px; }"
            "QSlider::handle:horizontal { background: #2979ff; width: 12px; "
            "margin: -4px 0; border-radius: 6px; }"
        )
        nav.addWidget(self.slider, 1)
        btn_next = QPushButton("▶")
        btn_next.setToolTip("Next frame (Right; Shift ±10, Ctrl ±100)")
        btn_next.clicked.connect(lambda: self._step(1))
        nav.addWidget(btn_next)
        self.lbl_frame = QLabel("0 / 0")
        self.lbl_frame.setMinimumWidth(120)
        self.lbl_frame.setAlignment(Qt.AlignCenter)
        nav.addWidget(self.lbl_frame)
        lv.addLayout(nav)
        split.addWidget(left)

        right = QWidget()
        right.setMaximumWidth(430)
        rv = QVBoxLayout(right)
        rv.setContentsMargins(0, 0, 0, 0)

        g_sum = QGroupBox("Bout summary")
        sv = QVBoxLayout(g_sum)
        self.tbl = QTreeWidget()
        self.tbl.setColumnCount(6)
        self.tbl.setHeaderLabels(["Obj", "Behavior", "Bouts", "Total s", "% time", "Mean s"])
        self.tbl.setRootIsDecorated(False)
        self.tbl.header().setSectionResizeMode(1, QHeaderView.Stretch)
        self.tbl.setToolTip(
            "Per animal and behavior: bout count, total time, share of the\n"
            "animal's tracked frames, and mean bout length. Updates live as\n"
            "you correct bouts."
        )
        sv.addWidget(self.tbl)
        rv.addWidget(g_sum, 1)

        g_edit = QGroupBox("Selected bout")
        ev = QVBoxLayout(g_edit)
        ev.setSpacing(4)
        self.lbl_bout = QLabel("Click a bout in the ethogram")
        self.lbl_bout.setWordWrap(True)
        self.lbl_bout.setStyleSheet("color: #cccccc; font-size: 11px;")
        ev.addWidget(self.lbl_bout)

        row = QHBoxLayout()
        row.addWidget(QLabel("Relabel:"))
        self.combo_relabel = QComboBox()
        self.combo_relabel.setToolTip(
            "Change this bout's behavior. Adjacent bouts that end up with the\n"
            "same label are merged automatically, so the ethogram never shows\n"
            "an artificial split."
        )
        row.addWidget(self.combo_relabel, 1)
        self.btn_relabel = QPushButton("Apply")
        self.btn_relabel.clicked.connect(self._relabel)
        row.addWidget(self.btn_relabel)
        ev.addLayout(row)

        self.btn_split = QPushButton("Split at playhead")
        self.btn_split.setToolTip(
            "Cut the selected bout in two at the current frame — use when the\n"
            "classifier ran one behavior into the next."
        )
        self.btn_split.clicked.connect(self._split)
        ev.addWidget(self.btn_split)

        mrow = QHBoxLayout()
        self.btn_merge_prev = QPushButton("◀ Merge prev")
        self.btn_merge_prev.setToolTip(
            "Absorb the preceding bout into this one, taking this bout's label."
        )
        self.btn_merge_prev.clicked.connect(lambda: self._merge(-1))
        mrow.addWidget(self.btn_merge_prev)
        self.btn_merge_next = QPushButton("Merge next ▶")
        self.btn_merge_next.setToolTip(
            "Absorb the following bout into this one, taking this bout's label."
        )
        self.btn_merge_next.clicked.connect(lambda: self._merge(1))
        mrow.addWidget(self.btn_merge_next)
        ev.addLayout(mrow)

        self.btn_undo = QPushButton("Undo")
        self.btn_undo.setToolTip("Undo the last bout edit (Ctrl+Z).")
        self.btn_undo.clicked.connect(self._undo_last)
        ev.addWidget(self.btn_undo)
        rv.addWidget(g_edit)
        split.addWidget(right)
        split.setStretchFactor(0, 1)
        root.addWidget(split, 1)

        g_tl = QGroupBox("Ethogram — click a bout to select, click anywhere to seek")
        tv = QVBoxLayout(g_tl)
        tv.setContentsMargins(4, 4, 4, 4)
        self.timeline = EthogramTimeline()
        self.timeline.bout_clicked.connect(self._on_bout_clicked)
        self.timeline.frame_clicked.connect(self._seek)
        tv.addWidget(self.timeline)
        self.legend = _Legend()
        tv.addWidget(self.legend)
        root.addWidget(g_tl)

        bottom = QHBoxLayout()
        self.lbl_status = QLabel("")
        self.lbl_status.setStyleSheet("color: #999; font-size: 11px;")
        bottom.addWidget(self.lbl_status, 1)
        btn_export = QPushButton("Export Bouts + Stats…")
        btn_export.setToolTip(
            "Write three CSVs next to the predictions:\n"
            "  ethogram_bouts.csv — one row per bout\n"
            "  ethogram_summary.csv — per animal x behavior stats\n"
            "  ethogram_transitions.csv — behavior transition counts"
        )
        btn_export.clicked.connect(self._export)
        bottom.addWidget(btn_export)
        self.btn_save = QPushButton("Save Corrections")
        self.btn_save.setStyleSheet(
            "QPushButton { background-color: #2e7d32; color: white; "
            "font-weight: bold; padding: 6px 14px; }"
            "QPushButton:hover { background-color: #388e3c; }"
            "QPushButton:disabled { background-color: #333; color: #666; }"
        )
        self.btn_save.setToolTip(
            "Rewrite behavior_predictions.csv from the corrected bouts.\n"
            "The original is kept once as behavior_predictions_original.csv."
        )
        self.btn_save.clicked.connect(self._save)
        bottom.addWidget(self.btn_save)
        btn_close = QPushButton("Close")
        btn_close.clicked.connect(self.reject)
        bottom.addWidget(btn_close)
        root.addLayout(bottom)
        self._update_buttons()

    # -- refresh --------------------------------------------------------
    def _refresh_all(self):
        # Drop a selection whose bout no longer exists, so the edit buttons
        # can never act on a bout that has been merged or replaced.
        if self._selected is not None and not any(
                x is self._selected for x in self._bouts):
            self._selected = None
        names = self._behavior_names()
        self._palette = behavior_palette(names)
        ids = self._object_ids()
        self.timeline.set_data(self._bouts, ids, self._palette, self._n_frames())
        self.timeline.set_playhead(self._frame_idx)
        self.timeline.set_selected(self._selected)
        self.legend.set_palette(self._palette)

        cur = self.combo_relabel.currentText()
        self.combo_relabel.blockSignals(True)
        self.combo_relabel.clear()
        self.combo_relabel.addItems(names)
        if cur in names:
            self.combo_relabel.setCurrentText(cur)
        self.combo_relabel.blockSignals(False)

        self._refresh_summary()
        self._update_buttons()

    def _refresh_summary(self):
        df = bout_summary(self._bouts, self.fps)
        self.tbl.clear()
        if df.empty:
            return
        for _, r in df.iterrows():
            it = QTreeWidgetItem([
                str(int(r["object_id"])), str(r["behavior"]),
                str(int(r["n_bouts"])), f"{r['total_s']:.1f}",
                f"{r['percent_time']:.1f}", f"{r['mean_bout_s']:.2f}",
            ])
            it.setForeground(1, QColor(self._palette.get(r["behavior"], NC_COLOR)))
            self.tbl.addTopLevelItem(it)
        for c in (0, 2, 3, 4, 5):
            self.tbl.resizeColumnToContents(c)

    def _update_buttons(self):
        b = self._selected
        self.btn_relabel.setEnabled(b is not None)
        self.combo_relabel.setEnabled(b is not None)
        self.btn_split.setEnabled(
            b is not None and b.start < self._frame_idx <= b.end
        )
        self.btn_merge_prev.setEnabled(b is not None and self._neighbour(-1) is not None)
        self.btn_merge_next.setEnabled(b is not None and self._neighbour(1) is not None)
        self.btn_undo.setEnabled(bool(self._undo))
        self.btn_save.setEnabled(self._dirty)
        if not self._dirty:
            self.lbl_status.setText(f"{len(self._bouts)} bouts")

    def _neighbour(self, direction: int) -> Optional[Bout]:
        """Adjacent bout of the same animal, if it is frame-contiguous."""
        b = self._selected
        if b is None:
            return None
        same = sorted([x for x in self._bouts if x.object_id == b.object_id],
                      key=lambda x: x.start)
        # The selection can outlive the bout it points at (an edit rebuilt the
        # list); treat that as "no neighbour" rather than raising.
        i = next((k for k, x in enumerate(same) if x is b), None)
        if i is None:
            return None
        j = i + direction
        if not (0 <= j < len(same)):
            return None
        other = same[j]
        # Only merge across a genuine boundary, never across a tracking gap.
        if direction < 0 and other.end + 1 != b.start:
            return None
        if direction > 0 and b.end + 1 != other.start:
            return None
        return other

    # -- navigation -----------------------------------------------------
    def _seek(self, f):
        self.slider.setValue(int(np.clip(f, 0, self._n_frames() - 1)))

    def _step(self, d):
        self._seek(self._frame_idx + d)

    def _show_frame(self, f):
        self._frame_idx = int(f)
        frame_rgb = None
        if self._cap is not None:
            import cv2
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, self._frame_idx)
            ret, bgr = self._cap.read()
            if ret:
                frame_rgb = np.ascontiguousarray(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))

        dets = []
        if self.reader is not None:
            for row in self.reader.rows_for_frame(self._frame_idx):
                oid = int(self.reader.object_id[row])
                beh = self._behavior_at(oid, self._frame_idx)
                dets.append({
                    "object_id": oid,
                    "bbox": self.reader.bbox[row].tolist(),
                    "mask": self.reader.mask_for_row(row),
                    "behavior": beh,
                    "color": self._palette.get(beh, NC_COLOR),
                })
        self.preview.set_content(frame_rgb, dets)
        self.timeline.set_playhead(self._frame_idx)
        self.lbl_frame.setText(
            f"{self._frame_idx} / {self._n_frames() - 1}   "
            f"({self._frame_idx / max(self.fps, 1e-6):.1f}s)"
        )
        self._update_buttons()

    def _behavior_at(self, oid: int, frame: int) -> str:
        for b in self._bouts:
            if b.object_id == oid and b.start <= frame <= b.end:
                return b.behavior
        return NC_LABEL

    def _on_bout_clicked(self, bout):
        self._selected = bout
        self.timeline.set_selected(bout)
        if bout is None:
            self.lbl_bout.setText("Click a bout in the ethogram")
        else:
            self.lbl_bout.setText(
                f"Animal {bout.object_id} · {bout.behavior}\n"
                f"frames {bout.start}–{bout.end} "
                f"({bout.n_frames} f, {bout.duration_s(self.fps):.2f}s) · "
                f"mean conf {bout.mean_conf:.0%}"
            )
            if bout.behavior in [self.combo_relabel.itemText(i)
                                 for i in range(self.combo_relabel.count())]:
                self.combo_relabel.setCurrentText(bout.behavior)
        self._update_buttons()

    # -- edits ----------------------------------------------------------
    def _push_undo(self):
        import copy
        self._undo.append(copy.deepcopy(self._bouts))
        if len(self._undo) > self._MAX_UNDO:
            self._undo.pop(0)
        self._dirty = True

    def _coalesce(self):
        """Merge frame-adjacent bouts that share a label.

        Relabelling can leave two touching bouts with the same behavior; as a
        data model that is a single bout, and leaving them split would
        double-count in the summary's bout counts.
        """
        merged: List[Bout] = []
        for b in sorted(self._bouts, key=lambda x: (x.object_id, x.start)):
            if (merged and merged[-1].object_id == b.object_id
                    and merged[-1].behavior == b.behavior
                    and merged[-1].end + 1 == b.start):
                prev = merged[-1]
                n1, n2 = prev.n_frames, b.n_frames
                prev.mean_conf = (prev.mean_conf * n1 + b.mean_conf * n2) / (n1 + n2)
                prev.end = b.end
            else:
                merged.append(b)
        self._bouts = merged

    def _relabel(self):
        b = self._selected
        new = self.combo_relabel.currentText()
        if b is None or not new or new == b.behavior:
            return
        self._push_undo()
        old = b.behavior
        b.behavior = new
        self._coalesce()
        self._selected = self._find_bout(b.object_id, b.start)
        self._refresh_all()
        self._show_frame(self._frame_idx)
        self._on_bout_clicked(self._selected)
        self.lbl_status.setText(
            f"Relabelled animal {b.object_id} {old} → {new} — unsaved")

    def _split(self):
        b = self._selected
        if b is None or not (b.start < self._frame_idx <= b.end):
            return
        self._push_undo()
        cut = self._frame_idx
        tail = Bout(b.object_id, b.behavior, cut, b.end, b.mean_conf)
        b.end = cut - 1
        self._bouts.append(tail)
        self._bouts.sort(key=lambda x: (x.object_id, x.start))
        self._selected = tail
        self._refresh_all()
        self._on_bout_clicked(tail)
        self.lbl_status.setText(f"Split at frame {cut} — unsaved")

    def _merge(self, direction: int):
        b = self._selected
        other = self._neighbour(direction)
        if b is None or other is None:
            return
        self._push_undo()
        n1, n2 = b.n_frames, other.n_frames
        b.mean_conf = (b.mean_conf * n1 + other.mean_conf * n2) / (n1 + n2)
        b.start = min(b.start, other.start)
        b.end = max(b.end, other.end)
        self._bouts.remove(other)
        self._coalesce()
        self._selected = self._find_bout(b.object_id, b.start)
        self._refresh_all()
        self._on_bout_clicked(self._selected)
        self.lbl_status.setText(
            f"Merged into animal {b.object_id} {b.behavior} bout — unsaved")

    def _find_bout(self, oid: int, frame: int) -> Optional[Bout]:
        for x in self._bouts:
            if x.object_id == oid and x.start <= frame <= x.end:
                return x
        return None

    def _undo_last(self):
        if not self._undo:
            return
        self._bouts = self._undo.pop()
        self._dirty = bool(self._undo)
        self._selected = None
        self._refresh_all()
        self._show_frame(self._frame_idx)
        self._on_bout_clicked(None)
        self.lbl_status.setText("Undid last bout edit")

    # -- output ---------------------------------------------------------
    def _save(self):
        backup = os.path.join(self.output_dir, "behavior_predictions_original.csv")
        try:
            if os.path.exists(self.behavior_csv) and not os.path.exists(backup):
                import shutil
                shutil.copy2(self.behavior_csv, backup)
            save_predictions(self.behavior_csv, bouts_to_frames(self._bouts))
        except Exception as e:
            QMessageBox.critical(self, "Save Failed", f"{e}")
            return
        self._undo.clear()
        self._dirty = False
        self._update_buttons()
        self.lbl_status.setText(
            f"Saved {len(self._bouts)} bouts to behavior_predictions.csv")

    def _export(self):
        d = QFileDialog.getExistingDirectory(
            self, "Export Ethogram Stats To", self.output_dir)
        if not d:
            return
        try:
            bout_table(self._bouts, self.fps).to_csv(
                os.path.join(d, "ethogram_bouts.csv"), index=False)
            bout_summary(self._bouts, self.fps).to_csv(
                os.path.join(d, "ethogram_summary.csv"), index=False)
            tm = transition_matrix(self._bouts)
            if not tm.empty:
                tm.to_csv(os.path.join(d, "ethogram_transitions.csv"), index=False)
        except Exception as e:
            QMessageBox.critical(self, "Export Failed", f"{e}")
            return
        QMessageBox.information(
            self, "Exported",
            f"Wrote ethogram_bouts.csv, ethogram_summary.csv"
            + (", ethogram_transitions.csv" if not tm.empty else "")
            + f"\n\nto {d}"
        )
        self.lbl_status.setText(f"Exported bout stats to {d}")

    # -- lifecycle ------------------------------------------------------
    def keyPressEvent(self, event):
        k, m = event.key(), event.modifiers()
        step = 100 if m & Qt.ControlModifier else (10 if m & Qt.ShiftModifier else 1)
        if k == Qt.Key_Left:
            self._step(-step)
        elif k == Qt.Key_Right:
            self._step(step)
        elif k == Qt.Key_Z and m & Qt.ControlModifier:
            self._undo_last()
        elif Qt.Key_1 <= k <= Qt.Key_9 and self._selected is not None:
            i = k - Qt.Key_1
            if i < self.combo_relabel.count():
                self.combo_relabel.setCurrentIndex(i)
                self._relabel()
        else:
            super().keyPressEvent(event)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.timeline.update()

    def reject(self):
        if self._dirty:
            r = QMessageBox.question(
                self, "Discard Corrections?",
                "You have unsaved bout corrections. Close anyway?",
                QMessageBox.Discard | QMessageBox.Cancel)
            if r != QMessageBox.Discard:
                return
        self._cleanup()
        super().reject()

    def _cleanup(self):
        if self._cap is not None:
            self._cap.release()
            self._cap = None
        if self.reader is not None:
            try:
                self.reader.close()
            except Exception:
                pass
            self.reader = None

    def closeEvent(self, event):
        self._cleanup()
        super().closeEvent(event)


class _Legend(QWidget):
    """Colour key for the ethogram."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._palette: Dict[str, str] = {}
        self.setFixedHeight(20)

    def set_palette(self, palette):
        self._palette = palette
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        f = p.font()
        f.setPixelSize(10)
        p.setFont(f)
        x = EthogramTimeline.LABEL_W
        for name, col in self._palette.items():
            p.setPen(Qt.NoPen)
            p.setBrush(QBrush(QColor(col)))
            p.drawRect(QRectF(x, 6, 9, 9))
            p.setPen(QColor(190, 190, 190))
            w = p.fontMetrics().horizontalAdvance(name)
            p.drawText(QPointF(x + 13, 15), name)
            x += 13 + w + 14


class _EthogramPreview(QWidget):
    """Video frame with masks tinted by the animal's current behavior."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self._frame = None
        self._dets: List[Dict] = []
        self.setMinimumSize(360, 250)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_content(self, frame_rgb, dets):
        self._frame, self._dets = frame_rgb, dets
        self.update()

    def paintEvent(self, event):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        p.fillRect(self.rect(), QColor(24, 24, 24))
        if self._frame is None and not self._dets:
            p.setPen(QColor(120, 120, 120))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "No video or masks — ethogram only")
            return

        if self._frame is not None:
            fh, fw = self._frame.shape[:2]
        else:
            masks = [d["mask"] for d in self._dets if d.get("mask") is not None]
            if not masks:
                return
            fh, fw = masks[0].shape[:2]
        scale = min(self.width() / fw, self.height() / fh)
        ox = (self.width() - fw * scale) / 2
        oy = (self.height() - fh * scale) / 2
        target = QRectF(ox, oy, fw * scale, fh * scale)

        if self._frame is not None:
            img = QImage(self._frame.data, fw, fh, 3 * fw, QImage.Format_RGB888)
            p.drawPixmap(target, QPixmap.fromImage(img), QRectF(0, 0, fw, fh))

        f = p.font()
        f.setPixelSize(max(12, int(16 * min(1.5, scale))))
        f.setBold(True)
        p.setFont(f)

        for det in self._dets:
            col = QColor(det.get("color", NC_COLOR))
            mask = det.get("mask")
            if mask is not None and mask.any():
                ov = np.zeros((mask.shape[0], mask.shape[1], 4), np.uint8)
                ov[mask] = [col.red(), col.green(), col.blue(), 120]
                qimg = QImage(ov.data, mask.shape[1], mask.shape[0],
                              4 * mask.shape[1], QImage.Format_RGBA8888)
                p.drawPixmap(target, QPixmap.fromImage(qimg),
                             QRectF(0, 0, mask.shape[1], mask.shape[0]))
            x1, y1, x2, y2 = det["bbox"]
            r = QRectF(ox + x1 * scale, oy + y1 * scale,
                       (x2 - x1) * scale, (y2 - y1) * scale)
            p.setBrush(Qt.NoBrush)
            p.setPen(QPen(col, 2))
            p.drawRect(r)
            txt = f"{det['object_id']}: {det.get('behavior', NC_LABEL)}"
            p.setPen(QPen(QColor(0, 0, 0), 3))
            p.drawText(QPointF(r.left() + 3, r.top() - 5), txt)
            p.setPen(QPen(col))
            p.drawText(QPointF(r.left() + 3, r.top() - 5), txt)
