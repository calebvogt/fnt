"""Protocol timeline — the visual schedule of mid-experiment manipulations.

A horizontal day axis carrying one marker per :class:`ProtocolEvent` (add or
remove animals, add or remove resources). Direct manipulation, video-game
style: click an empty spot to add an event at that day, click a marker to edit
or delete it, drag a marker to reschedule it. Emits ``changed`` whenever the
protocol list is mutated so the owner can mark the config dirty.
"""
from __future__ import annotations

import copy

from PyQt5.QtCore import Qt, QRectF, pyqtSignal
from PyQt5.QtGui import QColor, QPainter, QPen, QFont
from PyQt5.QtWidgets import (
    QWidget, QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QLabel,
    QComboBox, QDoubleSpinBox, QSpinBox, QLineEdit, QPushButton,
    QDialogButtonBox, QStackedWidget, QToolTip, QSizePolicy,
)

from ..core.config import ProtocolEvent, ResourceObject

# marker colour per event kind
_KIND_COLOR = {
    "add_agents": "#58c470",       # green  — animals in
    "remove_agents": "#e0605a",    # red    — animals out
    "add_resource": "#4aa3e0",     # blue   — resource in
    "remove_resource": "#e0a23a",  # orange — resource out
}
_KIND_GLYPH = {
    "add_agents": "+",
    "remove_agents": "−",
    "add_resource": "+",
    "remove_resource": "−",
}
_KIND_TITLE = {
    "add_agents": "Add animals",
    "remove_agents": "Remove animals",
    "add_resource": "Add resource",
    "remove_resource": "Remove resource",
}


def describe(ev: ProtocolEvent) -> str:
    """One-line human summary of an event (markers, tooltips, validation)."""
    d = f"day {ev.at_day:g}"
    if ev.kind == "add_agents":
        if ev.group is None:
            return f"{d}: add animals (no type set!)"
        return f"{d}: add {ev.group.count} × {ev.group.label}"
    if ev.kind == "remove_agents":
        n = "all" if not ev.count else str(ev.count)
        return f"{d}: remove {n} of '{ev.target or 'all'}'"
    if ev.kind == "add_resource":
        o = ev.object
        if o is None:
            return f"{d}: add resource (unset!)"
        name = o.label or o.kind
        return f"{d}: place {o.kind} '{name}' at ({o.x:.2f}, {o.y:.2f})"
    if ev.kind == "remove_resource":
        return f"{d}: remove resource '{ev.target}'"
    return f"{d}: {ev.kind}"


class ProtocolTimeline(QWidget):
    """The clickable day-axis strip. Owns a list of ProtocolEvent."""

    changed = pyqtSignal()

    _TRACK_H = 34          # height of the axis band
    _R = 9                 # marker radius (px)

    def __init__(self, context_provider, parent=None):
        """``context_provider()`` -> dict with keys:
        groups (list[AgentGroup]), resource_labels (list[str]),
        arena_wh ((w, h)), days (float)."""
        super().__init__(parent)
        self._ctx = context_provider
        self._events: list[ProtocolEvent] = []
        self._days = 10.0
        self._hits: list[tuple[QRectF, int]] = []   # marker rect -> event idx
        self._drag_idx = None
        self._drag_moved = False
        self._press_x = 0
        self.setMouseTracking(True)
        self.setMinimumHeight(84)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.setToolTip("Click an empty spot to schedule a manipulation at "
                        "that day.\nClick a marker to edit it; drag to "
                        "reschedule.")

    # ---- data ---------------------------------------------------------- #
    def set_protocol(self, events: list[ProtocolEvent], days: float) -> None:
        self._events = list(events)
        self._days = max(0.1, float(days))
        self.update()

    def protocol(self) -> list[ProtocolEvent]:
        return list(self._events)

    def set_days(self, days: float) -> None:
        self._days = max(0.1, float(days))
        self.update()

    # ---- geometry ------------------------------------------------------ #
    def _track_rect(self) -> QRectF:
        m = 14
        y = (self.height() - self._TRACK_H) / 2 + 6
        return QRectF(m, y, self.width() - 2 * m, self._TRACK_H)

    def _day_to_x(self, day: float) -> float:
        t = self._track_rect()
        return t.left() + (day / self._days) * t.width()

    def _x_to_day(self, x: float) -> float:
        t = self._track_rect()
        d = (x - t.left()) / max(1e-9, t.width()) * self._days
        return min(self._days, max(0.0, round(d * 4) / 4.0))   # snap to 6 h

    # ---- painting ------------------------------------------------------ #
    def paintEvent(self, _):
        p = QPainter(self)
        p.setRenderHint(QPainter.Antialiasing)
        t = self._track_rect()

        # axis band + day ticks
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#20242b"))
        p.drawRoundedRect(t, 6, 6)
        p.setPen(QPen(QColor("#3a3f47"), 1))
        font = QFont(self.font())
        font.setPointSize(8)
        p.setFont(font)
        n_days = int(self._days) + 1
        lab_every = max(1, n_days // 14)          # avoid label crowding
        for d in range(n_days + 1):
            x = self._day_to_x(d)
            if x > t.right() + 1:
                break
            p.setPen(QPen(QColor("#3a3f47"), 1))
            p.drawLine(int(x), int(t.top() + 4), int(x), int(t.bottom() - 4))
            if d % lab_every == 0 and d < self._days + 0.01:
                p.setPen(QColor("#8a9099"))
                p.drawText(int(x) - 8, int(t.bottom() + 14), f"d{d}")

        # release marker at day 0
        p.setPen(QColor("#8a9099"))
        p.drawText(int(t.left()) - 4, int(t.top()) - 4, "release")

        # event markers (stack overlapping days into little lanes)
        self._hits = []
        occupied: list[tuple[float, int]] = []    # (x, lane)
        mfont = QFont(self.font())
        mfont.setBold(True)
        mfont.setPointSize(10)
        for i, ev in enumerate(self._events):
            x = self._day_to_x(min(ev.at_day, self._days))
            lane = 0
            for ox, ol in occupied:
                if abs(ox - x) < 2.2 * self._R and ol == lane:
                    lane += 1
            occupied.append((x, lane))
            cy = t.center().y() - lane * (self._R * 2 + 2)
            r = self._R
            rect = QRectF(x - r, cy - r, 2 * r, 2 * r)
            col = QColor(_KIND_COLOR.get(ev.kind, "#8a9099"))
            p.setPen(QPen(col.darker(140), 1.5))
            p.setBrush(col)
            p.drawEllipse(rect)
            p.setPen(QColor("#101216"))
            p.setFont(mfont)
            p.drawText(rect, Qt.AlignCenter, _KIND_GLYPH.get(ev.kind, "·"))
            p.setFont(font)
            self._hits.append((rect.adjusted(-2, -2, 2, 2), i))

        if not self._events:
            p.setPen(QColor("#6f767e"))
            p.drawText(t.toRect(), Qt.AlignCenter,
                       "No manipulations — click a day to add one "
                       "(add/remove animals or resources)")
        p.end()

    # ---- interaction --------------------------------------------------- #
    def _hit(self, pos):
        for rect, i in reversed(self._hits):      # top lane wins
            if rect.contains(pos):
                return i
        return None

    def mousePressEvent(self, ev):
        if ev.button() != Qt.LeftButton:
            return
        self._press_x = ev.pos().x()
        self._drag_moved = False
        self._drag_idx = self._hit(ev.pos())

    def mouseMoveEvent(self, ev):
        i = self._hit(ev.pos())
        if self._drag_idx is not None and (ev.buttons() & Qt.LeftButton):
            if abs(ev.pos().x() - self._press_x) > 4:
                self._drag_moved = True
            if self._drag_moved:
                d = self._x_to_day(ev.pos().x())
                self._events[self._drag_idx].at_day = d
                QToolTip.showText(ev.globalPos(), f"day {d:g}", self)
                self.update()
            return
        self.setCursor(Qt.PointingHandCursor if i is not None
                       else Qt.CrossCursor)
        if i is not None:
            QToolTip.showText(ev.globalPos(), describe(self._events[i]), self)

    def mouseReleaseEvent(self, ev):
        if ev.button() != Qt.LeftButton:
            return
        i, moved = self._drag_idx, self._drag_moved
        self._drag_idx = None
        if i is not None and moved:               # finished a drag-reschedule
            self.changed.emit()
            return
        if i is not None:                         # click on a marker: edit
            self._edit_event(i)
        elif self._track_rect().adjusted(0, -8, 0, 8).contains(ev.pos()):
            self._add_event(self._x_to_day(ev.pos().x()))

    # ---- add / edit ----------------------------------------------------- #
    def _add_event(self, day: float):
        dlg = ProtocolEventDialog(self._ctx(), day=day, parent=self)
        if dlg.exec_() == QDialog.Accepted and dlg.result_event is not None:
            self._events.append(dlg.result_event)
            self._events.sort(key=lambda e: e.at_day)
            self.changed.emit()
            self.update()

    def _edit_event(self, i: int):
        dlg = ProtocolEventDialog(self._ctx(), event=self._events[i],
                                  parent=self)
        out = dlg.exec_()
        if out == QDialog.Accepted and dlg.result_event is not None:
            self._events[i] = dlg.result_event
            self._events.sort(key=lambda e: e.at_day)
        elif out == ProtocolEventDialog.DELETED:
            self._events.pop(i)
        else:
            return
        self.changed.emit()
        self.update()


class ProtocolEventDialog(QDialog):
    """Editor for one protocol event; kind-specific fields via a stack."""

    DELETED = 2   # custom done() code for the Delete button

    _KINDS = ["add_agents", "remove_agents", "add_resource", "remove_resource"]

    def __init__(self, ctx: dict, day: float | None = None,
                 event: ProtocolEvent | None = None, parent=None):
        super().__init__(parent)
        self._ctx = ctx
        self.result_event: ProtocolEvent | None = None
        editing = event is not None
        ev = copy.deepcopy(event) if editing else ProtocolEvent(
            at_day=day if day is not None else 1.0)
        self.setWindowTitle("Edit manipulation" if editing
                            else "Schedule a manipulation")
        self.setMinimumWidth(380)
        lay = QVBoxLayout(self)

        form = QFormLayout()
        self.in_kind = QComboBox()
        for k in self._KINDS:
            self.in_kind.addItem(_KIND_TITLE[k], k)
        self.in_day = QDoubleSpinBox()
        self.in_day.setRange(0.0, max(0.25, float(ctx.get("days", 10.0))))
        self.in_day.setSingleStep(0.25)
        self.in_day.setSuffix(" d")
        self.in_day.setValue(ev.at_day)
        self.in_label = QLineEdit(ev.label)
        self.in_label.setPlaceholderText("optional note, e.g. 'introduce "
                                         "novel males'")
        form.addRow("What", self.in_kind)
        form.addRow("When (day)", self.in_day)
        form.addRow("Note", self.in_label)
        lay.addLayout(form)

        # ---- per-kind panels ------------------------------------------- #
        self.stack = QStackedWidget()
        groups = ctx.get("groups", [])
        labels = [g.label for g in groups]
        w, h = ctx.get("arena_wh", (2.0, 2.0))

        # add_agents: pick an agent type + how many
        pa = QWidget()
        fa = QFormLayout(pa)
        self.in_add_type = QComboBox()
        self.in_add_type.addItems(labels)
        self.in_add_count = QSpinBox()
        self.in_add_count.setRange(1, 500)
        self.in_add_count.setValue(ev.group.count if ev.group else 4)
        fa.addRow("Agent type", self.in_add_type)
        fa.addRow("How many", self.in_add_count)
        if not labels:
            hint = QLabel("Define an agent type first "
                          "(Build && Add Agents → + Add Agent Type).")
            hint.setStyleSheet("color:#e0a23a;")
            hint.setWordWrap(True)
            fa.addRow(hint)
        self.stack.addWidget(pa)

        # remove_agents: who + how many
        pr = QWidget()
        fr = QFormLayout(pr)
        self.in_rm_target = QComboBox()
        self.in_rm_target.setEditable(True)
        self.in_rm_target.addItems(["all"] + labels)
        self.in_rm_count = QSpinBox()
        self.in_rm_count.setRange(0, 500)
        self.in_rm_count.setSpecialValueText("all matching")
        fr.addRow("Remove which", self.in_rm_target)
        fr.addRow("How many", self.in_rm_count)
        self.stack.addWidget(pr)

        # add_resource: kind + place + size + name
        po = QWidget()
        fo = QFormLayout(po)
        self.in_res_kind = QComboBox()
        self.in_res_kind.addItems(["food", "water", "nest"])
        self.in_res_x = QDoubleSpinBox()
        self.in_res_x.setRange(0.0, float(w))
        self.in_res_x.setSingleStep(0.1)
        self.in_res_x.setValue(round(w / 2, 2))
        self.in_res_y = QDoubleSpinBox()
        self.in_res_y.setRange(0.0, float(h))
        self.in_res_y.setSingleStep(0.1)
        self.in_res_y.setValue(round(h / 2, 2))
        self.in_res_r = QDoubleSpinBox()
        self.in_res_r.setRange(0.02, 2.0)
        self.in_res_r.setSingleStep(0.05)
        self.in_res_r.setValue(0.15)
        self.in_res_label = QLineEdit()
        self.in_res_label.setPlaceholderText("e.g. chow_B")
        fo.addRow("Resource", self.in_res_kind)
        fo.addRow("x (m)", self.in_res_x)
        fo.addRow("y (m)", self.in_res_y)
        fo.addRow("Radius (m)", self.in_res_r)
        fo.addRow("Name", self.in_res_label)
        self.stack.addWidget(po)

        # remove_resource: which one (by name) or all of a kind
        pd = QWidget()
        fd = QFormLayout(pd)
        self.in_rmres_target = QComboBox()
        self.in_rmres_target.setEditable(True)
        self.in_rmres_target.addItems(
            list(ctx.get("resource_labels", []))
            + ["food", "water", "nest"])
        fd.addRow("Remove which", self.in_rmres_target)
        hint = QLabel("A name removes that object; 'food' / 'water' / 'nest' "
                      "removes every one of that kind.")
        hint.setStyleSheet("color:#8a9099; font-size:10px;")
        hint.setWordWrap(True)
        fd.addRow(hint)
        self.stack.addWidget(pd)

        lay.addWidget(self.stack)
        self.in_kind.currentIndexChanged.connect(self.stack.setCurrentIndex)

        # populate from the event being edited
        self.in_kind.setCurrentIndex(self._KINDS.index(ev.kind)
                                     if ev.kind in self._KINDS else 0)
        self.stack.setCurrentIndex(self.in_kind.currentIndex())
        if ev.kind == "add_agents" and ev.group is not None:
            if ev.group.label in labels:
                self.in_add_type.setCurrentText(ev.group.label)
            self.in_add_count.setValue(ev.group.count)
        elif ev.kind == "remove_agents":
            self.in_rm_target.setCurrentText(ev.target or "all")
            self.in_rm_count.setValue(ev.count)
        elif ev.kind == "add_resource" and ev.object is not None:
            o = ev.object
            self.in_res_kind.setCurrentText(o.kind)
            self.in_res_x.setValue(o.x)
            self.in_res_y.setValue(o.y)
            self.in_res_r.setValue(o.radius)
            self.in_res_label.setText(o.label)
        elif ev.kind == "remove_resource":
            self.in_rmres_target.setCurrentText(ev.target)

        # ---- buttons ---------------------------------------------------- #
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Save" if editing
                                               else "Schedule")
        bb.accepted.connect(self._accept)
        bb.rejected.connect(self.reject)
        if editing:
            b_del = QPushButton("Delete")
            b_del.setStyleSheet("color:#e0605a;")
            b_del.clicked.connect(lambda: self.done(self.DELETED))
            bb.addButton(b_del, QDialogButtonBox.DestructiveRole)
        lay.addWidget(bb)

    def _accept(self):
        kind = self.in_kind.currentData()
        ev = ProtocolEvent(at_day=self.in_day.value(), kind=kind,
                           label=self.in_label.text().strip())
        if kind == "add_agents":
            groups = self._ctx.get("groups", [])
            label = self.in_add_type.currentText()
            src = next((g for g in groups if g.label == label), None)
            if src is None:
                self.in_add_type.setFocus()
                return                       # nothing to add — keep dialog open
            g = copy.deepcopy(src)
            g.count = self.in_add_count.value()
            ev.group = g
        elif kind == "remove_agents":
            ev.target = self.in_rm_target.currentText().strip() or "all"
            ev.count = self.in_rm_count.value()
        elif kind == "add_resource":
            ev.object = ResourceObject(
                kind=self.in_res_kind.currentText(),
                x=self.in_res_x.value(), y=self.in_res_y.value(),
                radius=self.in_res_r.value(),
                label=self.in_res_label.text().strip())
        elif kind == "remove_resource":
            t = self.in_rmres_target.currentText().strip()
            if not t:
                self.in_rmres_target.setFocus()
                return
            ev.target = t
        self.result_event = ev
        self.accept()
