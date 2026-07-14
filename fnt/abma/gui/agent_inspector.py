"""Live agent inspector — a Pokémon-style stat card for a selected agent.

Shows static attributes (identity, genotype, treatment, innate stats) and live
condition bars (Health, Energy, Hunger, Thirst, Stress) plus counters, updated
each frame during a run. Click an agent in the run canvas to inspect it.
"""
from __future__ import annotations

from collections import deque

from PyQt5.QtCore import Qt, QSize
from PyQt5.QtGui import QColor, QPainter, QPen
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QSizePolicy,
)

_ACTIVITY = {0: "resting", 1: "foraging", 2: "roaming", 3: "fleeing",
             4: "mating", 5: "dead"}
_MALE = "#4a90d9"
_FEMALE = "#e0559a"


class StatBar(QWidget):
    """A labelled 0–100 bar whose fill colour is fixed per stat."""

    def __init__(self, name: str, color: str, parent=None):
        super().__init__(parent)
        self.name = name
        self.color = QColor(color)
        self.value = 0.0
        self.setFixedHeight(20)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)

    def set_value(self, v: float):
        self.value = max(0.0, min(100.0, float(v)))
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        label_w = 58
        # track
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#2a2a2a"))
        p.drawRoundedRect(label_w, 3, w - label_w, h - 6, 3, 3)
        # fill
        fill_w = int((w - label_w) * self.value / 100.0)
        if fill_w > 0:
            p.setBrush(self.color)
            p.drawRoundedRect(label_w, 3, fill_w, h - 6, 3, 3)
        # label + value
        p.setPen(QColor("#cccccc"))
        p.drawText(0, 0, label_w - 4, h, Qt.AlignVCenter | Qt.AlignLeft, self.name)
        p.setPen(QColor("#eeeeee"))
        p.drawText(label_w, 0, w - label_w - 4, h,
                   Qt.AlignVCenter | Qt.AlignRight, f"{self.value:.0f}")


class Sparkline(QWidget):
    """Tiny multi-series history plot (health & energy over time)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.series = {"health": (deque(maxlen=180), "#43c46a"),
                       "energy": (deque(maxlen=180), "#4a90d9")}
        self.setFixedHeight(46)

    def push(self, health, energy):
        self.series["health"][0].append(health)
        self.series["energy"][0].append(energy)
        self.update()

    def clear(self):
        for d, _ in self.series.values():
            d.clear()
        self.update()

    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        w, h = self.width(), self.height()
        p.fillRect(self.rect(), QColor("#161616"))
        for data, color in self.series.values():
            if len(data) < 2:
                continue
            pen = QPen(QColor(color))
            pen.setWidthF(1.5)
            p.setPen(pen)
            n = len(data)
            prev = None
            for k, val in enumerate(data):
                x = w * k / (n - 1)
                y = h - 3 - (h - 6) * max(0.0, min(100.0, val)) / 100.0
                if prev is not None:
                    p.drawLine(int(prev[0]), int(prev[1]), int(x), int(y))
                prev = (x, y)


def _badge(text: str, color: str) -> QLabel:
    lab = QLabel(text)
    lab.setStyleSheet(
        f"background:{color}; color:white; border-radius:6px; padding:1px 6px;"
        "font-size:10px; font-weight:bold;")
    lab.setAlignment(Qt.AlignCenter)
    return lab


class AgentInspector(QFrame):
    """Docked panel that mirrors a selected agent's live stat block."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setFixedWidth(272)
        self.setFrameShape(QFrame.StyledPanel)
        self.setStyleSheet(
            "AgentInspector{background:#1a1a1a; border:1px solid #3f3f3f;"
            "border-radius:6px;}")
        self._pop = {}   # index -> static meta dict
        self._idx = None

        lay = QVBoxLayout(self)
        lay.setContentsMargins(10, 10, 10, 10)
        lay.setSpacing(6)

        self.title = QLabel("Agent inspector")
        self.title.setStyleSheet("font-size:15px; font-weight:bold; color:#fff;")
        lay.addWidget(self.title)
        self.subtitle = QLabel("Click an agent in the arena during a run.")
        self.subtitle.setStyleSheet("color:#999; font-size:11px;")
        self.subtitle.setWordWrap(True)
        lay.addWidget(self.subtitle)

        self.badges = QHBoxLayout()
        self.badges.setSpacing(4)
        self.badges.addStretch()
        lay.addLayout(self.badges)

        self.stats = QLabel("")
        self.stats.setStyleSheet("color:#bbb; font-size:11px;")
        self.stats.setWordWrap(True)
        lay.addWidget(self.stats)

        cond_hdr = QLabel("CONDITION")
        cond_hdr.setStyleSheet("color:#777; font-size:10px; font-weight:bold;")
        lay.addWidget(cond_hdr)
        self.bars = {
            "Health": StatBar("Health", "#43c46a"),
            "Energy": StatBar("Energy", "#4a90d9"),
            "Hunger": StatBar("Hunger", "#e0a23a"),
            "Thirst": StatBar("Thirst", "#3ab0c4"),
            "Stress": StatBar("Stress", "#d9534f"),
        }
        for b in self.bars.values():
            lay.addWidget(b)

        self.spark = Sparkline()
        lay.addWidget(self.spark)

        self.counters = QLabel("")
        self.counters.setStyleSheet("color:#bbb; font-size:11px;")
        self.counters.setWordWrap(True)
        lay.addWidget(self.counters)
        lay.addStretch()

    # ------------------------------------------------------------------ #
    def set_population(self, meta_list):
        self._pop = {m["index"]: m for m in meta_list}

    def selected_index(self):
        return self._idx

    def clear_selection(self):
        self._idx = None
        self.spark.clear()

    def select(self, idx: int):
        self._idx = idx
        self.spark.clear()
        m = self._pop.get(idx)
        if not m:
            return
        color = _MALE if m["sex"] == "M" else _FEMALE
        self.title.setText(f"{m['sexid']}  {'♂' if m['sex']=='M' else '♀'}")
        self.title.setStyleSheet(
            f"font-size:15px; font-weight:bold; color:{color};")
        self.subtitle.setText(f"{m['species']} · group: {m['group']}")
        self._set_badges(m, anosmic=(m["smell_ability"] < 0.5), estrus=False,
                         dead=False)
        self.stats.setText(
            f"mass {m['mass0']:.0f} g · aggression {m['aggression']:.1f} · "
            f"boldness {m['boldness']:.1f} · sociability {m['sociability']:.1f} · "
            f"speed {m['base_speed']:.2f} · explore {m['exploration']:.1f}")

    def _set_badges(self, m, anosmic, estrus, dead):
        while self.badges.count() > 1:  # keep trailing stretch
            item = self.badges.takeAt(0)
            if item.widget():
                item.widget().deleteLater()
        chips = []
        if m.get("genotype", "WT") != "WT":
            chips.append((m["genotype"], "#6a5acd"))
        if m.get("drug", "none") not in ("none", "saline"):
            chips.append((f"{m['drug']} {m['dose']:.1f}", "#b5793a"))
        if anosmic:
            chips.append(("ANOSMIC", "#8a6d3b"))
        if estrus:
            chips.append(("ESTRUS", "#c2557e"))
        if dead:
            chips.append(("DEAD", "#666"))
        for i, (txt, col) in enumerate(chips):
            self.badges.insertWidget(i, _badge(txt, col))

    def update_dynamic(self, st: dict):
        """st: per-agent condition dict with 0–1 bars, mass g, counters, flags."""
        if self._idx is None:
            return
        m = self._pop.get(self._idx, {})
        self.bars["Health"].set_value(st["health"] * 100)
        self.bars["Energy"].set_value(st["energy"] * 100)
        self.bars["Hunger"].set_value(st["hunger"] * 100)
        self.bars["Thirst"].set_value(st["thirst"] * 100)
        self.bars["Stress"].set_value(st["stress"] * 100)
        self.spark.push(st["health"] * 100, st["energy"] * 100)
        self._set_badges(m, anosmic=st.get("anosmic", False),
                         estrus=st.get("estrus", False),
                         dead=not st.get("alive", True))
        act = _ACTIVITY.get(int(st.get("activity", 2)), "?")
        self.counters.setText(
            f"now: {act}  ·  mass {st['mass']:.1f} g\n"
            f"today: {st['dist_today']:.0f} m  ·  fights "
            f"{st['fights_won']}W/{st['fights_lost']}L  ·  matings {st['matings']}")
