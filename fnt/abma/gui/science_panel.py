"""The inspection column: pick an animal, see what it is doing and why.

A run view that only draws dots answers "where did they go". The question an
in-silico experiment is for is "why did *this* one go there", and ABMA already
computes the answer every step — the policy blends a set of named, competing
drives and then sums them. :mod:`fnt.abma.core.policy` now hands the parts back
and :mod:`fnt.abma.core.record` stores them; this module is where they become
something you can look at.

Four pieces, top to bottom:

``AgentRoster``      one card per animal — colour by sex, live health bar,
                     what it is doing right now, and badges for the things that
                     matter to a manipulation (anosmic, in estrus, dead).
                     Clicking a card selects that animal everywhere.
``DriveBars``        the selected animal's competing drives *at this instant*,
                     as a bar per drive, sorted so the dominant motive is
                     obvious. This is the "why" panel.
``TracePlot``        the same quantities over the last stretch of the run, so a
                     change of behaviour can be seen happening rather than
                     inferred from a single frame.
``CouplingDiagram``  the condition-dynamics table drawn as what it actually is
                     — a graph. Nodes are the condition bars, edges are the
                     rules in ``config.dynamics``, and both light up live.

Everything is painted with QPainter against the frame dicts the engine already
emits, so the panel adds no dependency and no second traversal of the model.
"""
from __future__ import annotations

import math
from collections import deque

from PyQt5.QtCore import Qt, QSize, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen, QBrush, QPainterPath, QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QScrollArea,
    QSizePolicy, QAbstractButton, QGridLayout, QPushButton, QComboBox,
)

_MALE = "#4a90d9"
_FEMALE = "#e0559a"
_INK = "#e6e6e6"
_DIM = "#8a8a8a"
_PANEL = "#1a1a1a"
_LINE = "#3f3f3f"

_ACTIVITY = {0: "resting", 1: "foraging", 2: "roaming", 3: "fleeing",
             4: "mating", 5: "dead"}

#: The named drives, in the order they are shown. Labels are written for
#: someone reading the panel, not for someone who has read policy.py.
DRIVES: list[tuple[str, str, str]] = [
    ("drive_scent_home", "own scent", "#5ec2a0"),
    ("drive_memory", "memory of home", "#7fb3d5"),
    ("drive_home", "home range", "#7fb3d5"),
    ("drive_resource", "food / water", "#e0a23a"),
    ("drive_social", "other animals", "#c58fd6"),
    ("drive_territory", "rival scent", "#d9534f"),
    ("drive_wander", "exploration", "#9aa0a6"),
]

#: Condition bars, for the trace plot and the coupling diagram.
CONDITION: list[tuple[str, str, str]] = [
    ("health", "health", "#43c46a"),
    ("energy", "energy", "#4a90d9"),
    ("hunger", "hunger", "#e0a23a"),
    ("thirst", "thirst", "#3ab0c4"),
    ("stress", "stress", "#d9534f"),
    ("bladder", "bladder", "#c9b458"),
]

#: What the animal is sensing — the input side of the same decision.
SENSED: list[tuple[str, str, str]] = [
    ("scent_own", "own marks here", "#5ec2a0"),
    ("scent_foreign", "rival marks here", "#d9534f"),
    ("recognition_mean", "recognises others", "#c58fd6"),
    ("detection_mean", "detects others", "#7fb3d5"),
    ("need_food", "needs food", "#e0a23a"),
    ("need_water", "needs water", "#3ab0c4"),
]


def _f(frame, key, idx, default=0.0) -> float:
    """One agent's value for ``key``, tolerant of frames that lack it."""
    arr = frame.get(key)
    if arr is None:
        return default
    try:
        return float(arr[idx])
    except (IndexError, TypeError, ValueError):
        return default


# --------------------------------------------------------------------------- #
# Roster
# --------------------------------------------------------------------------- #
class AgentCard(QAbstractButton):
    """One animal in the roster: who it is, and how it is right now."""

    def __init__(self, index: int, parent=None):
        super().__init__(parent)
        self.index = index
        self.setCheckable(True)
        self.setCursor(Qt.PointingHandCursor)
        self.setFixedHeight(46)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.meta: dict = {}
        self.health = 100.0
        self.activity = 2
        self.alive = True
        self.anosmic = False
        self.estrus = False

    def sizeHint(self) -> QSize:
        return QSize(150, 46)

    def set_meta(self, meta: dict) -> None:
        self.meta = meta or {}
        self.anosmic = float(self.meta.get("smell_ability", 1.0)) < 0.5
        self.update()

    def set_live(self, health, activity, alive, anosmic, estrus) -> None:
        self.health, self.activity = health, int(activity)
        self.alive, self.anosmic, self.estrus = bool(alive), anosmic, estrus
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        sex = self.meta.get("sex", "M")
        accent = QColor(_MALE if sex == "M" else _FEMALE)
        if not self.alive:
            accent = QColor("#5a5a5a")

        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#242424" if self.isChecked() else "#1e1e1e"))
        p.drawRoundedRect(0, 1, w - 1, h - 3, 5, 5)
        if self.isChecked():
            p.setPen(QPen(QColor("#d4a017"), 1.6))
            p.setBrush(Qt.NoBrush)
            p.drawRoundedRect(1, 2, w - 3, h - 5, 5, 5)

        # sex/status swatch
        p.setPen(Qt.NoPen)
        p.setBrush(accent)
        p.drawRoundedRect(7, 9, 5, h - 21, 2, 2)

        p.setPen(QColor(_INK if self.alive else _DIM))
        f = p.font()
        f.setPointSize(9)
        f.setBold(True)
        p.setFont(f)
        label = str(self.meta.get("sexid", f"#{self.index}"))
        p.drawText(18, 8, w - 26, 14, Qt.AlignVCenter | Qt.AlignLeft, label)

        f.setBold(False)
        f.setPointSize(8)
        p.setFont(f)
        p.setPen(QColor(_DIM))
        state = _ACTIVITY.get(self.activity, "?") if self.alive else "dead"
        marks = []
        if self.anosmic:
            marks.append("anosmic")
        if self.estrus:
            marks.append("estrus")
        p.drawText(18, 21, w - 26, 12, Qt.AlignVCenter | Qt.AlignLeft,
                   " · ".join([state] + marks))

        # health bar
        bx, bw = 18, w - 26
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#2c2c2c"))
        p.drawRoundedRect(bx, h - 14, bw, 5, 2, 2)
        frac = max(0.0, min(1.0, self.health / 100.0))
        if frac > 0:
            p.setBrush(QColor("#43c46a") if frac > 0.35 else QColor("#d9534f"))
            p.drawRoundedRect(bx, h - 14, int(bw * frac), 5, 2, 2)


class AgentRoster(QScrollArea):
    """The clickable column of animals. Emits the selected row index."""

    selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWidgetResizable(True)
        self.setFrameShape(QFrame.NoFrame)
        self.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        self._host = QWidget()
        self._lay = QVBoxLayout(self._host)
        self._lay.setContentsMargins(2, 2, 2, 2)
        self._lay.setSpacing(3)
        self._lay.addStretch()
        self.setWidget(self._host)
        self.cards: dict[int, AgentCard] = {}

    def set_population(self, meta_list) -> None:
        """Rebuild the roster. Safe to call again when a protocol adds animals."""
        for card in self.cards.values():
            card.setParent(None)
            card.deleteLater()
        self.cards.clear()
        for meta in sorted(meta_list or [], key=lambda m: m.get("index", 0)):
            idx = int(meta.get("index", 0))
            card = AgentCard(idx, self._host)
            card.set_meta(meta)
            card.clicked.connect(lambda _=False, i=idx: self.selected.emit(i))
            self._lay.insertWidget(self._lay.count() - 1, card)
            self.cards[idx] = card

    def set_selected(self, idx) -> None:
        for i, card in self.cards.items():
            card.setChecked(i == idx)

    def update_frame(self, frame: dict) -> None:
        if not frame:
            return
        for i, card in self.cards.items():
            card.set_live(_f(frame, "health", i, 100.0),
                          _f(frame, "activity", i, 2.0),
                          _f(frame, "alive", i, 1.0) > 0.5,
                          _f(frame, "anosmic", i, 0.0) > 0.5,
                          _f(frame, "estrus", i, 0.0) > 0.5)


# --------------------------------------------------------------------------- #
# The "why" panel
# --------------------------------------------------------------------------- #
class DriveBars(QWidget):
    """The selected animal's competing motives, right now.

    Bars are drawn on a shared scale so their *relative* sizes are the message:
    the longest bar is what is currently steering the animal.

    A drive this configuration cannot produce (``home range`` with scent
    marking on, ``own scent`` with it off) is never listed. One that *is* in
    play but happens to be zero this instant — no rival marks nearby yet — keeps
    its row, because a list whose rows appear and vanish as the animal moves is
    unreadable. So a drive is admitted the first time it is non-zero and then
    stays, which also makes "this drive fell to zero" visible instead of
    silent.
    """

    ROW_H = 15

    def __init__(self, parent=None):
        super().__init__(parent)
        self.values: list[tuple[str, float, str]] = []
        self.heading = 0.0
        self._seen: set[str] = {"drive_wander", "drive_social"}
        self.setMinimumHeight(self.ROW_H * len(DRIVES) + 8)

    def reset(self) -> None:
        """Forget which drives are in play — call when the selection changes."""
        self._seen = {"drive_wander", "drive_social"}
        self.values = []
        self.update()

    def set_frame(self, frame: dict, idx: int) -> None:
        rows = []
        for key, label, colour in DRIVES:
            v = _f(frame, key, idx)
            if v > 1e-9:
                self._seen.add(key)
            if key in self._seen:
                rows.append((label, v, colour))
        self.values = rows
        dx, dy = _f(frame, "desired_x", idx), _f(frame, "desired_y", idx)
        self.heading = math.atan2(dy, dx)
        self.setMinimumHeight(self.ROW_H * max(1, len(rows)) + 8)
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        if not self.values:
            p.setPen(QColor(_DIM))
            p.drawText(self.rect(), Qt.AlignCenter, "no active drives")
            return
        f = p.font()
        f.setPointSize(8)
        p.setFont(f)
        label_w, pad = 96, 40
        span = max(1e-6, max(v for _, v, _ in self.values))
        track = max(20, self.width() - label_w - pad)
        for row, (label, value, colour) in enumerate(self.values):
            y = 4 + row * self.ROW_H
            p.setPen(QColor(_DIM))
            p.drawText(0, y, label_w - 6, self.ROW_H,
                       Qt.AlignVCenter | Qt.AlignRight, label)
            p.setPen(Qt.NoPen)
            p.setBrush(QColor("#2a2a2a"))
            p.drawRoundedRect(label_w, y + 3, track, self.ROW_H - 7, 2, 2)
            width = int(track * value / span)
            if width > 0:
                p.setBrush(QColor(colour))
                p.drawRoundedRect(label_w, y + 3, width, self.ROW_H - 7, 2, 2)
            p.setPen(QColor(_DIM))
            p.drawText(label_w + track + 4, y, pad - 6, self.ROW_H,
                       Qt.AlignVCenter | Qt.AlignLeft, f"{value:.2f}")


class TracePlot(QWidget):
    """Rolling multi-series history for one agent.

    Series are supplied as ``(key, label, colour)`` and read from each frame,
    so the same widget plots drives, condition bars or sensory channels
    depending on which set it is given.
    """

    def __init__(self, series, y_range=None, window: int = 240, parent=None):
        super().__init__(parent)
        self.series = list(series)
        self.y_range = y_range           # None = autoscale
        self.window = window
        self.history: dict[str, deque] = {
            key: deque(maxlen=window) for key, _, _ in self.series}
        self.visible = {key: True for key, _, _ in self.series}
        self.setMinimumHeight(96)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def clear(self) -> None:
        for d in self.history.values():
            d.clear()
        self.update()

    def push(self, frame: dict, idx: int) -> None:
        for key, _, _ in self.series:
            self.history[key].append(_f(frame, key, idx))
        self.update()

    def set_visible(self, key: str, on: bool) -> None:
        self.visible[key] = on
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        left, right, top, bottom = 30, 6, 6, 14
        plot_w, plot_h = max(10, w - left - right), max(10, h - top - bottom)

        shown = [(k, lab, c) for k, lab, c in self.series if self.visible[k]]
        values = [v for k, _, _ in shown for v in self.history[k]]
        if self.y_range is not None:
            lo, hi = self.y_range
        elif values:
            lo, hi = min(0.0, min(values)), max(values)
            hi = hi if hi > lo else lo + 1.0
        else:
            lo, hi = 0.0, 1.0

        p.setPen(QPen(QColor("#2e2e2e"), 1))
        f = p.font()
        f.setPointSize(7)
        p.setFont(f)
        for k in range(3):
            frac = k / 2.0
            y = top + plot_h - frac * plot_h
            p.setPen(QPen(QColor("#2e2e2e"), 1))
            p.drawLine(left, int(y), left + plot_w, int(y))
            p.setPen(QColor(_DIM))
            p.drawText(0, int(y) - 7, left - 4, 14,
                       Qt.AlignVCenter | Qt.AlignRight,
                       f"{lo + frac * (hi - lo):.2g}")

        n = max((len(self.history[k]) for k, _, _ in shown), default=0)
        if n < 2:
            p.setPen(QColor(_DIM))
            p.drawText(self.rect(), Qt.AlignCenter, "collecting…")
            return
        for key, _, colour in shown:
            pts = list(self.history[key])
            if len(pts) < 2:
                continue
            path = QPainterPath()
            for i, v in enumerate(pts):
                x = left + plot_w * i / max(1, n - 1)
                y = top + plot_h - plot_h * (v - lo) / max(1e-9, hi - lo)
                path.moveTo(x, y) if i == 0 else path.lineTo(x, y)
            p.setPen(QPen(QColor(colour), 1.4))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)

        p.setPen(QColor(_DIM))
        p.drawText(left, h - bottom, plot_w, bottom,
                   Qt.AlignVCenter | Qt.AlignRight,
                   f"last {n} samples →")


class TraceGroup(QWidget):
    """A trace plot plus a clickable legend that toggles each series."""

    def __init__(self, title: str, series, y_range=None, parent=None):
        super().__init__(parent)
        lay = QVBoxLayout(self)
        lay.setContentsMargins(0, 0, 0, 0)
        lay.setSpacing(3)
        head = QLabel(title.upper())
        head.setStyleSheet(
            f"color:#777; font-size:10px; font-weight:bold;")
        lay.addWidget(head)
        self.plot = TracePlot(series, y_range=y_range)
        lay.addWidget(self.plot)
        legend = QWidget()
        grid = QGridLayout(legend)
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setSpacing(2)
        for i, (key, label, colour) in enumerate(series):
            chip = QPushButton(f"■ {label}")
            chip.setCheckable(True)
            chip.setChecked(True)
            chip.setCursor(Qt.PointingHandCursor)
            chip.setStyleSheet(
                f"QPushButton{{color:{colour}; background:transparent;"
                f"border:none; font-size:9px; text-align:left; padding:0 2px;}}"
                f"QPushButton:!checked{{color:#555;}}")
            chip.toggled.connect(
                lambda on, k=key: self.plot.set_visible(k, on))
            grid.addWidget(chip, i // 2, i % 2)
        lay.addWidget(legend)

    def push(self, frame, idx):
        self.plot.push(frame, idx)

    def clear(self):
        self.plot.clear()


# --------------------------------------------------------------------------- #
# The dynamics table, drawn as the graph it already is
# --------------------------------------------------------------------------- #
class CouplingDiagram(QWidget):
    """``config.dynamics`` as a node-link diagram, lit by live values.

    Each row of the dynamics table is literally an edge — ``source`` drives
    ``target`` with some gain — so the table a user edits as a spreadsheet is a
    graph they never get to see. Drawing it makes the physiology legible:
    which bars feed which, which way the sign goes, and which couplings are
    actually firing at this moment.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.rows: list = []
        self.values: dict[str, float] = {}
        self.setMinimumHeight(150)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

    def set_dynamics(self, rows) -> None:
        self.rows = list(rows or [])
        self.update()

    def set_frame(self, frame: dict, idx: int) -> None:
        self.values = {key: _f(frame, key, idx)
                       for key, _, _ in CONDITION}
        self.values["movement"] = _f(frame, "drive_wander", idx)
        self.values["crowding"] = _f(frame, "neighbours", idx)
        self.update()

    def _layout(self, w: int, h: int) -> dict[str, tuple[float, float]]:
        """Sources on the left, condition bars on the right."""
        targets = [key for key, _, _ in CONDITION]
        sources = []
        for row in self.rows:
            s = getattr(row, "source", None)
            if s and s not in targets and s not in sources:
                sources.append(s)
        pos = {}
        for i, name in enumerate(sources):
            y = 18 + (h - 36) * (i / max(1, len(sources) - 1))
            pos[name] = (52.0, y)
        for i, name in enumerate(targets):
            y = 18 + (h - 36) * (i / max(1, len(targets) - 1))
            pos[name] = (w - 56.0, y)
        return pos

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        if not self.rows:
            p.setPen(QColor(_DIM))
            p.drawText(self.rect(), Qt.AlignCenter,
                       "no condition dynamics defined")
            return
        pos = self._layout(w, h)
        f = p.font()
        f.setPointSize(7)
        p.setFont(f)

        for row in self.rows:
            src, tgt = getattr(row, "source", ""), getattr(row, "target", "")
            if src not in pos or tgt not in pos:
                continue
            x0, y0 = pos[src]
            x1, y1 = pos[tgt]
            gain = float(getattr(row, "gain", 0.0))
            # sign is the readable part: what builds a bar vs what drains it
            colour = QColor("#4a90d9" if gain >= 0 else "#d9534f")
            colour.setAlpha(150)
            width = 0.6 + min(1.8, math.log1p(abs(gain)) / 2.5)
            path = QPainterPath()
            path.moveTo(x0 + 6, y0)
            path.cubicTo((x0 + x1) / 2, y0, (x0 + x1) / 2, y1, x1 - 7, y1)
            p.setPen(QPen(colour, width))
            p.setBrush(Qt.NoBrush)
            p.drawPath(path)

        colours = {key: colour for key, _, colour in CONDITION}
        for name, (x, y) in pos.items():
            value = self.values.get(name)
            base = QColor(colours.get(name, "#7a7a7a"))
            if value is not None and name in colours:
                # fill tracks the live bar, so the diagram animates
                base = QColor(base)
                base.setAlpha(int(70 + 185 * min(1.0, value / 100.0)))
            p.setPen(QPen(QColor("#4a4a4a"), 1))
            p.setBrush(QBrush(base))
            p.drawEllipse(int(x - 5), int(y - 5), 10, 10)
            p.setPen(QColor(_INK if name in colours else _DIM))
            on_left = x < w / 2
            p.drawText(int(x - (44 if on_left else -10)), int(y - 7),
                       44, 14,
                       Qt.AlignVCenter | (Qt.AlignRight if on_left
                                          else Qt.AlignLeft),
                       name)


# --------------------------------------------------------------------------- #
# The whole column
# --------------------------------------------------------------------------- #
class SciencePanel(QWidget):
    """Roster + the selected animal's motives, history and physiology."""

    selected = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumWidth(268)
        self.setStyleSheet(
            f"SciencePanel{{background:{_PANEL};}}"
            f"QLabel{{color:{_INK};}}")
        self._idx = None
        self._pop: dict[int, dict] = {}

        lay = QVBoxLayout(self)
        lay.setContentsMargins(8, 8, 8, 8)
        lay.setSpacing(6)

        head = QLabel("ANIMALS")
        head.setStyleSheet("color:#777; font-size:10px; font-weight:bold;")
        lay.addWidget(head)
        self.roster = AgentRoster()
        self.roster.setMinimumHeight(120)
        self.roster.selected.connect(self._on_pick)
        lay.addWidget(self.roster, 3)

        self.title = QLabel("No animal selected")
        self.title.setStyleSheet("font-size:13px; font-weight:bold;")
        lay.addWidget(self.title)
        self.subtitle = QLabel("Click a card, or an animal in the arena.")
        self.subtitle.setStyleSheet("color:#999; font-size:10px;")
        self.subtitle.setWordWrap(True)
        lay.addWidget(self.subtitle)

        why = QLabel("WHY IT IS MOVING THAT WAY")
        why.setStyleSheet("color:#777; font-size:10px; font-weight:bold;")
        lay.addWidget(why)
        self.drives = DriveBars()
        lay.addWidget(self.drives)

        self.picker = QComboBox()
        self.picker.addItems(["Drives over time", "Condition over time",
                              "What it senses", "Condition dynamics"])
        self.picker.currentIndexChanged.connect(self._show_page)
        lay.addWidget(self.picker)

        self.pages = [
            TraceGroup("drive strength", DRIVES),
            TraceGroup("condition", CONDITION, y_range=(0.0, 100.0)),
            TraceGroup("sensed", SENSED),
        ]
        self.coupling = CouplingDiagram()
        for page in self.pages:
            lay.addWidget(page, 4)
            page.hide()
        lay.addWidget(self.coupling, 4)
        self.coupling.hide()
        self.pages[0].show()

        self.footer = QLabel("")
        self.footer.setStyleSheet("color:#8a8a8a; font-size:10px;")
        self.footer.setWordWrap(True)
        lay.addWidget(self.footer)

    # ---- wiring --------------------------------------------------------- #
    def _show_page(self, index: int) -> None:
        for page in self.pages:
            page.hide()
        self.coupling.hide()
        if index < len(self.pages):
            self.pages[index].show()
        else:
            self.coupling.show()

    def _on_pick(self, idx: int) -> None:
        self.select(idx)
        self.selected.emit(idx)

    def set_population(self, meta_list) -> None:
        self._pop = {int(m["index"]): m for m in (meta_list or [])}
        self.roster.set_population(meta_list)
        if self._idx is not None and self._idx in self._pop:
            self.roster.set_selected(self._idx)
        else:
            self.select(None)

    def set_dynamics(self, rows) -> None:
        self.coupling.set_dynamics(rows)

    def selected_index(self):
        return self._idx

    def select(self, idx) -> None:
        self._idx = idx
        self.roster.set_selected(idx)
        for page in self.pages:
            page.clear()
        self.drives.reset()
        meta = self._pop.get(idx) if idx is not None else None
        if not meta:
            self.title.setText("No animal selected")
            self.subtitle.setText("Click a card, or an animal in the arena.")
            self.footer.setText("")
            return
        colour = _MALE if meta.get("sex") == "M" else _FEMALE
        mark = "♂" if meta.get("sex") == "M" else "♀"
        self.title.setText(f"{meta.get('sexid', idx)}  {mark}")
        self.title.setStyleSheet(
            f"font-size:13px; font-weight:bold; color:{colour};")
        bits = [str(meta.get("species", "")), f"group {meta.get('group', '')}"]
        if meta.get("genotype", "WT") not in ("WT", ""):
            bits.append(str(meta["genotype"]))
        if meta.get("drug", "none") not in ("none", "saline", ""):
            bits.append(f"{meta['drug']} {float(meta.get('dose', 0)):.2g}")
        self.subtitle.setText(" · ".join(b for b in bits if b))
        self.footer.setText(
            f"mass {float(meta.get('mass0', 0)):.0f} g · "
            f"aggression {float(meta.get('aggression', 0)):.2f} · "
            f"sociability {float(meta.get('sociability', 0)):.2f} · "
            f"smell {float(meta.get('smell_ability', 0)):.2f}")

    def update_frame(self, frame: dict) -> None:
        """Feed one frame: the roster always, the detail only if something
        is selected."""
        if not frame:
            return
        self.roster.update_frame(frame)
        idx = self._idx
        if idx is None:
            return
        n = len(frame.get("x", ()))
        if idx >= n:
            return
        self.drives.set_frame(frame, idx)
        for page in self.pages:
            page.push(frame, idx)
        self.coupling.set_frame(frame, idx)
