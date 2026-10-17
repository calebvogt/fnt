"""Behaviour review: a focus session over candidate bouts, one key per decision.

Behavioural classification in Mask Tracker proposes bouts (candidate rules now, a
classifier later); a trained observer confirms or corrects them here, and those decisions
are the labels the classifier learns from. The window stands alone -- like MAD's
inference and training windows -- so a review session can be handed over without the mask
labelling GUI.

Most behaviours belong to one animal or pass between two, so every bout has an **actor**
(the focal animal) and, for social behaviours, a **recipient**. Each is drawn in its own
colour -- mask, spine, nose (dot) and tail base (ring) -- with its role's label floating
beside it: "anogenital investigation ->" on the investigator, "<- receives anogenital
investigation" on the other. Both labels are saved, so the classifier learns each side.

A bout can hold more than one behaviour -- a facial investigation that turns into an
anogenital one -- so it is labelled in **segments**: split it at a frame (X) and give each
part its own label. The timeline under the video shows the parts, their labels and the
playhead. Bouts the rules already split (a ``segments`` column, ``frame:kind;...``) open
pre-split, so the reviewer only confirms or moves the split.

Input is a bouts CSV with at least ``video``, ``frame0``, ``frame1`` and ``kind`` (the
proposed behaviour). Optional columns: ``focal`` and ``other`` (track ids: actor and
recipient), ``tracks_csv`` (per-frame ``frame``, ``track``, ``cx``, ``cy``, ``nose_x``,
``nose_y``, ``tb_x``, ``tb_y``) for keypoints and to centre the view, ``tracks_h5`` (a
Mask Tracker track store) for the masks and spines, ``segments`` and ``bout_id``.

Decisions go to ``<bouts>_review.csv`` beside it after every decision, one row per
segment (write-then-rename), so a session can stop at any point and pick up where it
left off.

Keys: A accept (every segment, with its label), R reject the bout, S skip, 1-9 / 0 label
from the ethogram (a single-segment bout: relabel and move on; several segments: label
the one under the playhead), X split at the playhead, M merge the segment under the
playhead into the one before, W swap actor and recipient, F head and tail are swapped on
the actor (accept and note it), Space play/pause, comma / period step a frame, N / Right
next bout, B / Left back.
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
    "immobile", "wheel running", "huddle", "facial investigation", "anogenital investigation",
    "sniff / explore", "self-groom", "chase", "fight / tumble", "other",
]
# What the recipient is doing when the actor does a social behaviour. Behaviours not
# listed belong to the actor alone.
RECIPROCAL = {
    "facial investigation": "receives facial investigation",
    "anogenital investigation": "receives anogenital investigation",
    "chase": "chased / flees",
    "fight / tumble": "fight / tumble",
    "huddle": "huddle",
}
# Candidate-rule names -> ethogram entries.
KIND_TO_LABEL = {
    "immobile": "immobile", "huddle": "huddle", "nose_nose": "facial investigation",
    "anogenital": "anogenital investigation", "chase": "chase", "head_only": "sniff / explore",
    "wheel": "wheel running", "fight": "fight / tumble", "investigation": "facial investigation",
}
# Okabe-Ito, as BGR: actor orange, recipient sky blue, anyone else grey.
ROLE_COLOURS = {"actor": (0, 159, 230), "recipient": (233, 180, 86), "other": (150, 150, 150)}
# Timeline colours per ethogram entry (BGR).
LABEL_COLOURS = [(0, 159, 230), (233, 180, 86), (115, 158, 0), (66, 228, 240), (178, 114, 0),
                 (0, 94, 213), (167, 121, 204), (60, 60, 200), (40, 40, 140), (140, 140, 140)]
PAD_S = 0.5            # context shown before and after each bout
VIEW = 640             # side of the square view, px
BAR = 34               # timeline height, px
REVIEW_FIELDS = ["bout_id", "seg", "frame0", "frame1", "kind", "decision", "label", "actor",
                 "recipient", "actor_label", "recipient_label", "flipped", "reviewed_at"]


def review_path(bouts_csv: str) -> str:
    return os.path.splitext(bouts_csv)[0] + "_review.csv"


def recipient_label(actor_label: str, recipient) -> str:
    """The recipient's side of the actor's behaviour; empty for a solo behaviour or no
    recipient."""
    if recipient in (None, ""):
        return ""
    return RECIPROCAL.get(actor_label, "")


def parse_segments(text: str, frame0: int, frame1: int) -> List[Dict]:
    """``"f:kind;f:kind"`` (each segment's first frame and proposed kind) -> segments
    covering frame0..frame1. Empty or unreadable text gives one segment, no label."""
    starts = []
    for part in (text or "").split(";"):
        if ":" in part:
            f, kind = part.split(":", 1)
            try:
                starts.append((int(float(f)), KIND_TO_LABEL.get(kind.strip(), kind.strip())))
            except ValueError:
                pass
    starts = sorted((max(frame0, f), k) for f, k in starts if f <= frame1)
    if not starts or starts[0][0] != frame0:
        starts.insert(0, (frame0, starts[0][1] if starts else None))
    segs = []
    for i, (f, k) in enumerate(starts):
        end = starts[i + 1][0] - 1 if i + 1 < len(starts) else frame1
        if end >= f:
            segs.append({"start": f, "end": end, "label": k, "proposed": k})
    return segs


def load_bouts(path: str) -> List[Dict]:
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    for i, r in enumerate(rows):
        if not r.get("bout_id"):
            r["bout_id"] = str(i)
        for k in ("video", "frame0", "frame1", "kind"):
            if not r.get(k):
                raise ValueError(f"{path} row {i + 1}: no '{k}'")
    return rows


def load_review(path: str) -> Dict[str, List[Dict]]:
    """Saved decisions by bout id, each a list of segment rows in order."""
    if not os.path.exists(path):
        return {}
    out: Dict[str, List[Dict]] = {}
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            out.setdefault(r["bout_id"], []).append(r)
    for v in out.values():
        v.sort(key=lambda r: int(r.get("seg") or 0))
    return out


def save_review(path: str, decisions: Dict[str, List[Dict]]):
    tmp = path + ".tmp"
    with open(tmp, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, REVIEW_FIELDS, extrasaction="ignore")
        w.writeheader()
        for bid in sorted(decisions, key=lambda b: (len(b), b)):
            for row in decisions[bid]:
                w.writerow(row)
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


def _same(a, b) -> bool:
    try:
        return a not in (None, "") and b not in (None, "") and int(float(a)) == int(float(b))
    except ValueError:
        return False


def draw_tag(img, text: str, anchor, colour, scale: float = 1.0):
    """A label on a dark plate, above ``anchor`` (x, y), kept inside the image."""
    font = cv2.FONT_HERSHEY_SIMPLEX
    th = max(1, int(round(2 * scale)))
    (tw, tht), base = cv2.getTextSize(text, font, 0.6 * scale, th)
    x = int(np.clip(anchor[0] - tw / 2, 2, img.shape[1] - tw - 2))
    y = int(np.clip(anchor[1], tht + 6, img.shape[0] - 4))
    cv2.rectangle(img, (x - 4, y - tht - 6), (x + tw + 4, y + base), (25, 25, 25), -1)
    cv2.putText(img, text, (x, y - 2), font, 0.6 * scale, colour, th, cv2.LINE_AA)


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
        self._h5_cache: Dict[str, object] = {}
        self._frames: List[np.ndarray] = []
        self._frame_ids: List[int] = []
        self._tags: List[list] = []
        self._pos = 0
        self._playing = True
        self.idx = 0
        self.actor = self.recipient = ""
        self.segments: List[Dict] = []
        self.flipped = False
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
            secs = _num(b, "dur_s")
            who = f"  {b.get('focal', '')}→{b['other']}" if b.get("other") else ""
            segs = parse_segments(b.get("segments", ""), int(b["frame0"]), int(b["frame1"]))
            name = " → ".join(s["label"] or KIND_TO_LABEL.get(b["kind"], b["kind"]) for s in segs)
            self.table.setItem(i, 0, QTableWidgetItem(b["bout_id"]))
            self.table.setItem(i, 1, QTableWidgetItem(name + who))
            self.table.setItem(i, 2, QTableWidgetItem(f"{secs:.1f} s" if secs else f"{dur} fr"))
            self.table.setItem(i, 3, QTableWidgetItem("        "))
            self._show_decision(i)
        self.table.cellClicked.connect(lambda r, _c: self.show_bout(r))
        left.addWidget(self.table, 1)
        keys = "\n".join(f"{(i + 1) % 10}  {name}" for i, name in enumerate(self.ethogram))
        self.lbl_keys = QLabel(
            "Label (actor):\n" + keys +
            "\n\nX  split at the playhead · M  merge with the part before\n"
            ", / .  step a frame · W  swap actor and recipient")
        self.lbl_keys.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        left.addWidget(self.lbl_keys)
        self.lbl_progress = QLabel("")
        left.addWidget(self.lbl_progress)
        lw = QWidget()
        lw.setLayout(left)
        lw.setFixedWidth(400)
        root.addWidget(lw)

        right = QVBoxLayout()
        self.lbl_info = QLabel("")
        self.lbl_info.setStyleSheet("font-size: 14px; font-weight: bold;")
        right.addWidget(self.lbl_info)
        self.view = QLabel()
        self.view.setMinimumSize(VIEW, VIEW + BAR)
        self.view.setAlignment(Qt.AlignCenter)
        self.view.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        right.addWidget(self.view, 1)
        legend = QLabel(
            "<span style='color:#e69f00'>■ actor (focal animal)</span> &nbsp; "
            "<span style='color:#56b4e9'>■ recipient</span> &nbsp; "
            "<span style='color:#969696'>■ others</span> &nbsp; · &nbsp; "
            "● nose &nbsp; ○ tail base &nbsp; — spine &nbsp; · &nbsp; "
            "yellow border = the bout, grey = 0.5 s either side; timeline: the bout's parts")
        legend.setStyleSheet("color: #aaaaaa; font-size: 11px;")
        right.addWidget(legend)
        row = QHBoxLayout()
        for text, slot in (("< Back (B)", self.prev_bout), ("Accept (A)", self.accept),
                           ("Reject (R)", self.reject), ("Split (X)", self.split),
                           ("Merge (M)", self.merge), ("Swap roles (W)", self.swap_roles),
                           ("Head/tail swapped (F)", self.flip), ("Skip (S)", self.skip),
                           ("Play/Pause (Space)", self.toggle_play), ("Next (N) >", self.next_bout)):
            btn = QPushButton(text)
            btn.setFocusPolicy(Qt.NoFocus)
            btn.clicked.connect(slot)
            row.addWidget(btn)
        right.addLayout(row)
        root.addLayout(right, 1)
        self.resize(1400, 900)

    def _proposed(self, b: Optional[Dict] = None) -> str:
        b = b or self.bouts[self.idx]
        return KIND_TO_LABEL.get(b["kind"], b["kind"])

    def _show_decision(self, i: int):
        rows = self.decisions.get(self.bouts[i]["bout_id"])
        item = self.table.item(i, 3)
        if not rows:
            item.setText("")
            return
        d = rows[0]
        if d["decision"] == "reject":
            text = "✗"
        elif d["decision"] == "skip":
            text = "skipped"
        elif len(rows) == 1 and d.get("label") == self._proposed(self.bouts[i]):
            text = "✓"
        else:
            text = "→ " + " | ".join(r.get("label", "") for r in rows)
        b = self.bouts[i]
        if d.get("actor") and not _same(d.get("actor"), b.get("focal")):
            text += " (roles swapped)"
        if d.get("flipped") == "1":
            text += " (head/tail)"
        item.setText(text)

    # ---------------------------------------------------------------- playback
    def _keypoints(self, b: Dict) -> Optional[_Keypoints]:
        p = b.get("tracks_csv") or ""
        if not p or not os.path.exists(p):
            return None
        if p not in self._kp_cache:
            self._kp_cache[p] = _Keypoints(p)
        return self._kp_cache[p]

    def _track_store(self, b: Dict):
        p = b.get("tracks_h5") or ""
        if not p or not os.path.exists(p):
            return None
        if p not in self._h5_cache:
            from .track_store import TrackMaskReader
            self._h5_cache[p] = TrackMaskReader(p)
        return self._h5_cache[p]

    def _role(self, track) -> str:
        if _same(track, self.actor):
            return "actor"
        if _same(track, self.recipient):
            return "recipient"
        return "other"

    def _draw_animals(self, img, frame: int, kp, store, scale: float) -> list:
        """Masks, spines, noses and tail bases in role colours. Returns where the role
        labels go (role, x, y in frame pixels); they are drawn at playback, so a relabelled
        segment shows its new label at once."""
        from .mask_pose import mask_geometry
        dets = store.detections_for_frame(frame) if store is not None else []
        overlay = img.copy()
        for d in dets:
            m = d["mask"]
            if m is None:
                continue
            overlay[m.astype(bool)] = ROLE_COLOURS[self._role(d["object_id"])]
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, dst=img)
        lw = max(1, int(round(2 * scale)))
        tags = []
        for d in dets:
            role = self._role(d["object_id"])
            col = ROLE_COLOURS[role]
            m = d["mask"]
            if m is None:
                continue
            cnts, _ = cv2.findContours(m.astype(np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cv2.drawContours(img, cnts, -1, col, lw)
            if role != "other":
                x1, y1, x2, y2 = [int(v) for v in d["bbox"]]
                g = mask_geometry(m[y1:y2 + 1, x1:x2 + 1], (x1, y1))
                if g is not None:
                    cv2.polylines(img, [g["spine"].astype(np.int32)], False, col, lw)
                tags.append((role, (d["bbox"][0] + d["bbox"][2]) / 2, d["bbox"][1] - 8 * scale))
        for track in (self.actor, self.recipient):
            r = kp.get(frame, track) if kp else None
            if _num(r, "nose_x") is None:
                continue
            col = ROLE_COLOURS[self._role(track)]
            nose = (int(_num(r, "nose_x")), int(_num(r, "nose_y")))
            tb = (int(_num(r, "tb_x")), int(_num(r, "tb_y")))
            cv2.circle(img, nose, int(7 * scale), (20, 20, 20), -1)
            cv2.circle(img, nose, int(5 * scale), col, -1)
            cv2.circle(img, tb, int(7 * scale), (20, 20, 20), lw + 2)
            cv2.circle(img, tb, int(7 * scale), col, lw)
            if store is None:
                tags.append((self._role(track), nose[0], nose[1] - 28 * scale))
        return tags

    def _load_frames(self, b: Dict):
        cap = cv2.VideoCapture(b["video"])
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        f0, f1 = int(b["frame0"]), int(b["frame1"])
        pad = int(round(PAD_S * fps))
        start = max(0, f0 - pad)
        kp = self._keypoints(b)
        store = self._track_store(b)
        ids = [x for x in (self.actor, self.recipient) if x not in (None, "")]
        pts = []
        if kp:
            for f in range(f0, f1 + 1):
                for i in ids:
                    r = kp.get(f, i)
                    if _num(r, "cx") is not None:
                        pts.append((_num(r, "cx"), _num(r, "cy")))
        cap.set(cv2.CAP_PROP_POS_FRAMES, start)
        frames, ids_out, tags_out = [], [], []
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
            scale = (side or max(h, w)) / VIEW          # drawing sizes in view pixels
            tags = self._draw_animals(img, f, kp, store, scale) if (kp or store is not None) else []
            x0 = y0 = 0
            if side:
                x0 = int(np.clip(centre[0] - side / 2, 0, w - side))
                y0 = int(np.clip(centre[1] - side / 2, 0, h - side))
                img = img[y0:y0 + side, x0:x0 + side]
            k = VIEW / max(img.shape[:2])
            img = cv2.resize(img, None, fx=k, fy=k, interpolation=cv2.INTER_AREA)
            frames.append(img)
            ids_out.append(f)
            tags_out.append([(role, (x - x0) * k, (y - y0) * k) for role, x, y in tags])
        cap.release()
        return frames, ids_out, tags_out, fps

    def show_bout(self, i: int, keep_roles: bool = False):
        if not self.bouts:
            return
        self.idx = max(0, min(i, len(self.bouts) - 1))
        b = self.bouts[self.idx]
        f0, f1 = int(b["frame0"]), int(b["frame1"])
        saved = self.decisions.get(b["bout_id"])
        if not keep_roles:
            self.actor, self.recipient = b.get("focal", ""), b.get("other", "")
            self.flipped = False
            if saved and saved[0]["decision"] not in ("reject", "skip") and saved[0].get("frame0"):
                self.actor, self.recipient = saved[0].get("actor", ""), saved[0].get("recipient", "")
                self.segments = [{"start": int(r["frame0"]), "end": int(r["frame1"]), "label": r.get("label") or None,
                                  "proposed": r.get("label") or None} for r in saved]
            else:
                self.segments = parse_segments(b.get("segments", ""), f0, f1)
        self._frames, self._frame_ids, self._tags, fps = self._load_frames(b)
        self._pos = 0
        self._timer.start(int(1000 / fps))
        self._playing = True
        self.table.selectRow(self.idx)
        self._fps = fps
        self._update_info()
        self._update_progress()
        self._tick()

    def _update_info(self):
        b = self.bouts[self.idx]
        dur = (int(b["frame1"]) - int(b["frame0"]) + 1) / getattr(self, "_fps", 15.0)
        parts = " → ".join(self._seg_label(s) for s in self.segments)
        who = f"  actor: track {self.actor}"
        if self.recipient:
            who += f" → recipient: track {self.recipient}"
        self.lbl_info.setText(f"#{b['bout_id']}  {parts}  {dur:.1f} s{who}   "
                              f"{os.path.basename(b['video'])} frames {b['frame0']}-{b['frame1']}")

    def _seg_label(self, seg: Dict) -> str:
        return seg["label"] or self._proposed()

    def _seg_at(self, frame: int) -> int:
        """Index of the segment holding ``frame`` (padding counts to the nearest)."""
        for i, s in enumerate(self.segments):
            if frame <= s["end"]:
                return i
        return len(self.segments) - 1

    def _timeline(self, width: int) -> np.ndarray:
        bar = np.full((BAR, width, 3), 35, np.uint8)
        if not self._frame_ids:
            return bar
        a, z = self._frame_ids[0], self._frame_ids[-1]
        span = max(1, z - a + 1)

        def x(f):
            return int(round((f - a) / span * (width - 1)))

        cur = self._seg_at(self._frame_ids[self._pos])
        for i, s in enumerate(self.segments):
            lab = self._seg_label(s)
            col = LABEL_COLOURS[self.ethogram.index(lab) % len(LABEL_COLOURS)] if lab in self.ethogram else (120, 120, 120)
            x1, x2 = x(s["start"]), x(s["end"] + 1)
            cv2.rectangle(bar, (x1, 4), (max(x1 + 1, x2 - 1), BAR - 5), col, -1)
            if i == cur:
                cv2.rectangle(bar, (x1, 2), (max(x1 + 1, x2 - 1), BAR - 3), (255, 255, 255), 2)
            (tw, _), _ = cv2.getTextSize(lab, cv2.FONT_HERSHEY_SIMPLEX, 0.42, 1)
            if tw < x2 - x1 - 6:
                cv2.putText(bar, lab, (x1 + 4, BAR // 2 + 5), cv2.FONT_HERSHEY_SIMPLEX, 0.42, (20, 20, 20), 1,
                            cv2.LINE_AA)
            if i > 0:
                cv2.line(bar, (x1, 0), (x1, BAR - 1), (255, 255, 255), 2)
        px = x(self._frame_ids[self._pos])
        cv2.line(bar, (px, 0), (px, BAR - 1), (255, 255, 255), 1)
        return bar

    def _tick(self):
        if not self._frames:
            self.view.setText("Could not read this bout's frames.")
            return
        img = self._frames[self._pos].copy()
        b = self.bouts[self.idx]
        f = self._frame_ids[self._pos]
        inb = int(b["frame0"]) <= f <= int(b["frame1"])
        cv2.rectangle(img, (0, 0), (img.shape[1] - 1, img.shape[0] - 1), (0, 200, 255) if inb else (90, 90, 90), 6)
        actor_label = self._seg_label(self.segments[self._seg_at(f)])
        rec_label = recipient_label(actor_label, self.recipient)
        for role, x, y in self._tags[self._pos]:
            text = actor_label if role == "actor" else rec_label
            if not text:
                continue
            text = (text + " →") if (role == "actor" and rec_label) else ("← " + text if role == "recipient" else text)
            draw_tag(img, text, (x, y), ROLE_COLOURS[role])
        img = np.vstack([img, self._timeline(img.shape[1])])
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        qimg = QImage(rgb.data, rgb.shape[1], rgb.shape[0], 3 * rgb.shape[1], QImage.Format_RGB888)
        self.view.setPixmap(QPixmap.fromImage(qimg.copy()))
        if self._playing:
            self._pos = (self._pos + 1) % len(self._frames)

    def toggle_play(self):
        self._playing = not self._playing

    def step(self, n: int):
        self._playing = False
        if self._frames:
            self._pos = (self._pos + n) % len(self._frames)
            self._tick()

    def _update_progress(self):
        n = sum(1 for b in self.bouts if b["bout_id"] in self.decisions)
        self.lbl_progress.setText(f"{n} of {len(self.bouts)} reviewed")

    # ---------------------------------------------------------------- segments
    def split(self):
        """Split the segment under the playhead at the current frame (a frame of the bout,
        not a segment's first)."""
        if not self._frames:
            return
        f = self._frame_ids[self._pos]
        b = self.bouts[self.idx]
        if not (int(b["frame0"]) < f <= int(b["frame1"])):
            return
        i = self._seg_at(f)
        s = self.segments[i]
        if f <= s["start"]:
            return
        self.segments[i:i + 1] = [dict(s, end=f - 1), dict(s, start=f)]
        self._playing = False
        self._update_info()
        self._tick()

    def merge(self):
        """Merge the segment under the playhead into the one before it (the first into the
        next); the earlier segment's label is kept."""
        if len(self.segments) < 2 or not self._frames:
            return
        i = self._seg_at(self._frame_ids[self._pos])
        j = i - 1 if i > 0 else 0
        a, b = self.segments[j], self.segments[j + 1]
        self.segments[j:j + 2] = [dict(a, end=b["end"])]
        self._update_info()
        self._tick()

    # ---------------------------------------------------------------- decisions
    def _decide(self, decision: str):
        b = self.bouts[self.idx]
        now = time.strftime("%Y-%m-%d %H:%M:%S")
        if decision in ("reject", "skip"):
            rows = [{"bout_id": b["bout_id"], "seg": 0, "frame0": b["frame0"], "frame1": b["frame1"],
                     "kind": b["kind"], "decision": decision, "label": "", "actor": self.actor,
                     "recipient": self.recipient, "actor_label": "", "recipient_label": "",
                     "flipped": "0", "reviewed_at": now}]
        else:
            rows = []
            for i, s in enumerate(self.segments):
                lab = self._seg_label(s)
                proposed = s.get("proposed") or self._proposed()
                rows.append({"bout_id": b["bout_id"], "seg": i, "frame0": s["start"], "frame1": s["end"],
                             "kind": b["kind"], "decision": "accept" if lab == proposed else "relabel",
                             "label": lab, "actor": self.actor, "recipient": self.recipient,
                             "actor_label": lab, "recipient_label": recipient_label(lab, self.recipient),
                             "flipped": "1" if self.flipped else "0", "reviewed_at": now})
        self.decisions[b["bout_id"]] = rows
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
        """The behaviour is right but head and tail were swapped on the actor."""
        self.flipped = True
        self._decide("accept")

    def relabel(self, k: int):
        """One segment: relabel the bout and move on. Several: label the segment under the
        playhead and stay, so the others can be labelled before accepting."""
        if not 0 <= k < len(self.ethogram):
            return
        if len(self.segments) == 1:
            self.segments[0]["label"] = self.ethogram[k]
            self._decide("accept")
            return
        f = self._frame_ids[self._pos] if self._frames else int(self.bouts[self.idx]["frame0"])
        self.segments[self._seg_at(f)]["label"] = self.ethogram[k]
        self._update_info()
        self._tick()

    def swap_roles(self):
        """The other animal is the one doing it: exchange actor and recipient and replay."""
        if self.recipient in (None, ""):
            return
        self.actor, self.recipient = self.recipient, self.actor
        self.show_bout(self.idx, keep_roles=True)

    def next_bout(self):
        if self.idx < len(self.bouts) - 1:
            self.show_bout(self.idx + 1)

    def prev_bout(self):
        if self.idx > 0:
            self.show_bout(self.idx - 1)

    def keyPressEvent(self, e):
        k = e.key()
        actions = {Qt.Key_A: self.accept, Qt.Key_R: self.reject, Qt.Key_F: self.flip,
                   Qt.Key_W: self.swap_roles, Qt.Key_S: self.skip, Qt.Key_Space: self.toggle_play,
                   Qt.Key_X: self.split, Qt.Key_M: self.merge,
                   Qt.Key_Comma: lambda: self.step(-1), Qt.Key_Period: lambda: self.step(1),
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
        for store in self._h5_cache.values():
            try:
                store.close()
            except Exception:
                pass
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
