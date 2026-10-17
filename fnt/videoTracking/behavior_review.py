"""Behaviour review: a focus session over candidate bouts, one key per decision.

Behavioural classification in Mask Tracker proposes bouts (candidate rules now, a
classifier later); a trained observer confirms or corrects them here, and those decisions
are the labels the classifier learns from. The window stands alone -- like MAD's
inference and training windows -- so a review session can be handed over without the mask
labelling GUI.

Input is a bouts CSV with at least ``video``, ``frame0``, ``frame1`` and ``kind`` (the
proposed behaviour). Optional columns: ``focal`` and ``other`` (track ids), ``tracks_csv``
(per-frame keypoints with ``t``/``frame``, ``track``, ``cx``, ``cy``, ``nose_x``,
``nose_y``, ``tb_x``, ``tb_y``) to centre the view and draw noses (red) and tail bases
(blue), and ``bout_id``.

Decisions go to ``<bouts>_review.csv`` beside it after every key press (write-then-
rename), so a session can stop at any point and pick up where it left off.

Keys: A accept, R reject (not this behaviour), 1-9 / 0 relabel from the ethogram,
F flip head and tail of the focal animal (and accept), S skip, Space play/pause,
N / Right next, B / Left back.
"""
from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from typing import Dict, List, Optional

import cv2
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QImage, QPixmap
from PyQt5.QtWidgets import (
    QApplication, QHBoxLayout, QHeaderView, QLabel, QPushButton, QSizePolicy,
    QTableWidget, QTableWidgetItem, QVBoxLayout, QWidget,
)

ETHOGRAM = [
    "immobile", "wheel running", "huddle", "nose-nose sniff", "anogenital sniff",
    "sniff / explore", "self-groom", "chase", "fight / tumble", "other",
]
# Candidate-rule names -> ethogram entries.
KIND_TO_LABEL = {
    "immobile": "immobile", "huddle": "huddle", "nose_nose": "nose-nose sniff",
    "anogenital": "anogenital sniff", "chase": "chase", "head_only": "sniff / explore",
    "wheel": "wheel running", "fight": "fight / tumble",
}
PAD_S = 0.5            # context shown before and after each bout
VIEW = 640             # side of the square view, px
REVIEW_FIELDS = ["bout_id", "kind", "decision", "label", "flipped", "reviewed_at"]


def review_path(bouts_csv: str) -> str:
    return os.path.splitext(bouts_csv)[0] + "_review.csv"


def load_bouts(path: str) -> List[Dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for i, r in enumerate(rows):
        r.setdefault("bout_id", "")
        if not r["bout_id"]:
            r["bout_id"] = str(i)
        for k in ("video", "frame0", "frame1", "kind"):
            if not r.get(k):
                raise ValueError(f"{path} row {i + 1}: no '{k}'")
    return rows


def load_review(path: str) -> Dict[str, Dict]:
    if not os.path.exists(path):
        return {}
    with open(path, newline="", encoding="utf-8") as fh:
        return {r["bout_id"]: r for r in csv.DictReader(fh)}


def save_review(path: str, decisions: Dict[str, Dict]):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, REVIEW_FIELDS, extrasaction="ignore")
        w.writeheader()
        for bid in sorted(decisions, key=lambda b: (len(b), b)):
            w.writerow(decisions[bid])
    for attempt in range(5):
        try:
            os.replace(tmp, path)
            return
        except PermissionError:          # a reader (Excel, a sync client) holds it briefly
            time.sleep(0.2 * (attempt + 1))
    os.replace(tmp, path)


class _Keypoints:
    """Per-frame nose / tail base / centroid by track, from a tracks CSV."""

    def __init__(self, path: str):
        self.rows: Dict[tuple, Dict] = {}
        with open(path, newline="", encoding="utf-8") as fh:
            for r in csv.DictReader(fh):
                try:
                    key = (int(float(r["frame"])), int(float(r["track"])))
                except (KeyError, ValueError):
                    continue
                self.rows[key] = r

    def get(self, frame: int, track) -> Optional[Dict]:
        try:
            return self.rows.get((int(frame), int(float(track))))
        except (TypeError, ValueError):
            return None


def _num(r: Optional[Dict], k: str) -> Optional[float]:
    try:
        v = float(r[k]) if r else None
    except (KeyError, TypeError, ValueError):
        return None
    return None if v is None or np.isnan(v) else v


class BehaviorReviewWindow(QWidget):
    def __init__(self, bouts_csv: str, ethogram: Optional[List[str]] = None):
        super().__init__()
        from .mask_tracker_gui import DARK_STYLESHEET
        self.setStyleSheet(DARK_STYLESHEET)
        self.bouts_csv = bouts_csv
        self.bouts = load_bouts(bouts_csv)
        self.ethogram = list(ethogram or ETHOGRAM)[:10]
        self.out_path = review_path(bouts_csv)
        self.decisions = load_review(self.out_path)
        self._kp_cache: Dict[str, _Keypoints] = {}
        self._frames: List[np.ndarray] = []
        self._in_bout: List[bool] = []
        self._pos = 0
        self._playing = True
        self.idx = 0
        self.setWindowTitle(f"Behavior Review — {os.path.basename(bouts_csv)} ({len(self.bouts)} bouts)")
        self._build()
        self._timer = QTimer(self)
        self._timer.timeout.connect(self._tick)
        first = next((i for i, b in enumerate(self.bouts) if b["bout_id"] not in self.decisions), 0)
        self.show_bout(first)

    # ---------------------------------------------------------------- layout
    def _build(self):
        root = QHBoxLayout(self)
        left = QVBoxLayout()
        self.table = QTableWidget(len(self.bouts), 4)
        self.table.setHorizontalHeaderLabels(["#", "Proposed", "s", "Decision"])
        self.table.verticalHeader().setVisible(False)
        self.table.setSelectionBehavior(QTableWidget.SelectRows)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        hdr = self.table.horizontalHeader()
        for col in (0, 2, 3):
            hdr.setSectionResizeMode(col, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.Stretch)
        self.table.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self.table.setFocusPolicy(Qt.NoFocus)
        for i, b in enumerate(self.bouts):
            dur = (int(b["frame1"]) - int(b["frame0"]) + 1)
            self.table.setItem(i, 0, QTableWidgetItem(b["bout_id"]))
            self.table.setItem(i, 1, QTableWidgetItem(KIND_TO_LABEL.get(b["kind"], b["kind"])))
            secs = _num(b, "dur_s")
            self.table.setItem(i, 2, QTableWidgetItem(f"{secs:.1f} s" if secs else f"{dur} fr"))
            self.table.setItem(i, 3, QTableWidgetItem(""))
            self._show_decision(i)
        self.table.cellClicked.connect(lambda r, _c: self.show_bout(r))
        left.addWidget(self.table, 1)
        keys = "\n".join(f"{(i + 1) % 10}  {name}" for i, name in enumerate(self.ethogram))
        self.lbl_keys = QLabel("Relabel:\n" + keys)
        self.lbl_keys.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        left.addWidget(self.lbl_keys)
        self.lbl_progress = QLabel("")
        left.addWidget(self.lbl_progress)
        lw = QWidget()
        lw.setLayout(left)
        lw.setFixedWidth(360)
        root.addWidget(lw)

        right = QVBoxLayout()
        self.lbl_info = QLabel("")
        self.lbl_info.setStyleSheet("font-size: 14px; font-weight: bold;")
        right.addWidget(self.lbl_info)
        self.view = QLabel()
        self.view.setMinimumSize(VIEW, VIEW)
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right.addWidget(self.view, 1)
        legend = QLabel("red = focal animal's nose · yellow = the other's nose · blue = tail bases · "
                        "yellow border = the bout, grey = 0.5 s either side")
        legend.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        right.addWidget(legend)
        row = QHBoxLayout()
        for text, slot in (("< Back (B)", self.prev_bout), ("Accept (A)", self.accept),
                           ("Reject (R)", self.reject), ("Flip head (F)", self.flip),
                           ("Skip (S)", self.skip), ("Play/Pause (Space)", self.toggle_play),
                           ("Next (N) >", self.next_bout)):
            btn = QPushButton(text)
            btn.setFocusPolicy(Qt.NoFocus)
            btn.clicked.connect(slot)
            row.addWidget(btn)
        right.addLayout(row)
        root.addLayout(right, 1)
        self.resize(1200, 820)

    def _show_decision(self, i: int):
        d = self.decisions.get(self.bouts[i]["bout_id"])
        item = self.table.item(i, 3)
        if not d:
            item.setText("")
            return
        text = {"accept": "✓", "reject": "✗", "relabel": "→ " + d.get("label", ""),
                "skip": "skipped"}.get(d["decision"], d["decision"])
        if d.get("flipped") == "1":
            text += " (flip)"
        item.setText(text)

    # ---------------------------------------------------------------- playback
    def _keypoints(self, b: Dict) -> Optional[_Keypoints]:
        p = b.get("tracks_csv") or ""
        if not p or not os.path.exists(p):
            return None
        if p not in self._kp_cache:
            self._kp_cache[p] = _Keypoints(p)
        return self._kp_cache[p]

    def _load_frames(self, b: Dict):
        cap = cv2.VideoCapture(b["video"])
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        f0, f1 = int(b["frame0"]), int(b["frame1"])
        pad = int(round(PAD_S * fps))
        start = max(0, f0 - pad)
        kp = self._keypoints(b)
        ids = [x for x in (b.get("focal"), b.get("other")) if x not in (None, "")]
        pts = []
        if kp:
            for f in range(f0, f1 + 1):
                for i in ids:
                    r = kp.get(f, i)
                    if _num(r, "cx") is not None:
                        pts.append((_num(r, "cx"), _num(r, "cy")))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames, inb = [], []
        centre, side = None, None
        for f in range(start, f1 + pad + 1):
            ok, img = cap.read()
            if not ok:
                break
            h, w = img.shape[:2]
            if centre is None:
                if pts:
                    centre = np.mean(pts, axis=0)
                    side = int(min(h, w, max(360, 2.6 * np.ptp(np.array(pts), axis=0).max() + 300)))
                elif b.get("cx"):
                    centre, side = np.array([float(b["cx"]), float(b["cy"])]), min(h, w, 720)
                else:
                    centre, side = np.array([w / 2, h / 2]), None
            if kp:
                for i, col in zip(ids, ((0, 0, 255), (0, 200, 255))):
                    r = kp.get(f, i)
                    if _num(r, "nose_x") is not None:
                        cv2.circle(img, (int(_num(r, "nose_x")), int(_num(r, "nose_y"))), 7, col, -1)
                        cv2.circle(img, (int(_num(r, "tb_x")), int(_num(r, "tb_y"))), 6, (255, 128, 0), -1)
            if side:
                x0 = int(np.clip(centre[0] - side / 2, 0, w - side))
                y0 = int(np.clip(centre[1] - side / 2, 0, h - side))
                img = img[y0:y0 + side, x0:x0 + side]
            scale = VIEW / max(img.shape[:2])
            img = cv2.resize(img, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
            frames.append(img)
            inb.append(f0 <= f <= f1)
        cap.release()
        return frames, inb, fps

    def show_bout(self, i: int):
        if not self.bouts:
            return
        self.idx = max(0, min(i, len(self.bouts) - 1))
        b = self.bouts[self.idx]
        self._frames, self._in_bout, fps = self._load_frames(b)
        self._pos = 0
        self._timer.start(int(1000 / fps))
        self._playing = True
        self.table.selectRow(self.idx)
        dur = (int(b["frame1"]) - int(b["frame0"]) + 1) / fps
        who = f"  tracks {b.get('focal', '')}" + (f" → {b['other']}" if b.get("other") else "")
        self.lbl_info.setText(f"#{b['bout_id']}  {KIND_TO_LABEL.get(b['kind'], b['kind'])}  "
                              f"{dur:.1f} s{who}   {os.path.basename(b['video'])} "
                              f"frames {b['frame0']}-{b['frame1']}")
        self._update_progress()
        self._tick()

    def _tick(self):
        if not self._frames:
            self.view.setText("Could not read this bout's frames.")
            return
        img = self._frames[self._pos].copy()
        col = (0, 200, 255) if self._in_bout[self._pos] else (90, 90, 90)
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), col, 6)
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1], QImage.Format_RGB888)
        self.view.setPixmap(QPixmap.fromImage(qimg.copy()))
        if self._playing:
            self._pos = (self._pos + 1) % len(self._frames)

    def toggle_play(self):
        self._playing = not self._playing

    def _update_progress(self):
        n = sum(1 for b in self.bouts if b["bout_id"] in self.decisions)
        self.lbl_progress.setText(f"{n} of {len(self.bouts)} reviewed")

    # ---------------------------------------------------------------- decisions
    def _decide(self, decision: str, label: Optional[str] = None, flipped: bool = False):
        b = self.bouts[self.idx]
        proposed = KIND_TO_LABEL.get(b["kind"], b["kind"])
        self.decisions[b["bout_id"]] = {
            "bout_id": b["bout_id"], "kind": b["kind"], "decision": decision,
            "label": label if label is not None else (proposed if decision == "accept" else ""),
            "flipped": "1" if flipped else "0",
            "reviewed_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        save_review(self.out_path, self.decisions)
        self._show_decision(self.idx)
        self._update_progress()
        self.next_bout()

    def accept(self):
        self._decide("accept")

    def reject(self):
        self._decide("reject")

    def skip(self):
        self._decide("skip")

    def flip(self):
        """Head and tail were swapped on the focal animal; the behaviour itself is right."""
        self._decide("accept", flipped=True)

    def relabel(self, k: int):
        if 0 <= k < len(self.ethogram):
            self._decide("relabel", label=self.ethogram[k])

    def next_bout(self):
        if self.idx < len(self.bouts) - 1:
            self.show_bout(self.idx + 1)

    def prev_bout(self):
        if self.idx > 0:
            self.show_bout(self.idx - 1)

    def keyPressEvent(self, e):
        k = e.key()
        actions = {Qt.Key_A: self.accept, Qt.Key_R: self.reject, Qt.Key_F: self.flip,
                   Qt.Key_S: self.skip, Qt.Key_Space: self.toggle_play,
                   Qt.Key_N: self.next_bout, Qt.Key_Right: self.next_bout,
                   Qt.Key_B: self.prev_bout, Qt.Key_Left: self.prev_bout}
        if k in actions:
            actions[k]()
        elif Qt.Key_0 <= k <= Qt.Key_9:
            n = k - Qt.Key_0
            self.relabel(9 if n == 0 else n - 1)
        else:
            super().keyPressEvent(e)

    def closeEvent(self, e):
        self._timer.stop()
        super().closeEvent(e)


def main(argv=None):
    p = argparse.ArgumentParser(prog="behavior-review",
                                description="Review candidate behaviour bouts, one key per decision.")
    p.add_argument("--bouts", required=True, help="Bouts CSV (video, frame0, frame1, kind, ...).")
    p.add_argument("--ethogram", default="",
                   help="Comma-separated behaviours for keys 1-9, 0 (default: the rodent set).")
    args = p.parse_args(argv)
    try:
        from .mask_tracker_gui import _host_like_startup
        _host_like_startup()
    except Exception:
        pass
    app = QApplication.instance() or QApplication(sys.argv[:1])
    eth = [s.strip() for s in args.ethogram.split(",") if s.strip()] or None
    win = BehaviorReviewWindow(args.bouts, eth)
    win.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
