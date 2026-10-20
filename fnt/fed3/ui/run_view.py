"""Running and Stopped: one experiment, its devices, its problems and its data.

Each device has a card: the FED3 drawn with its counts on it, and what it's
doing beside that. The pellet plot runs across the top at full width, the
cards flow below it in as many columns as fit, and the log runs along the
bottom of the window (log_panel.py).
"""
import os
import subprocess
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtWidgets import (
    QComboBox, QFrame, QGridLayout, QHBoxLayout, QInputDialog, QLabel, QMessageBox,
    QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from ..device_sim import parse_iso
from . import theme
from .editor import ProtocolEditor, format_delta
from .fed_graphic import FedGraphic
from .plot import PelletPlot, zones

CARD_WIDTH = 370                # cards reflow into as many columns as fit
TEST_CUE = "tone:4000:200+flash:bar:0:0:30:0:500"


def ago(moment, now):
    if moment is None:
        return "—"
    seconds = (now - moment).total_seconds()
    if seconds < 60:
        return f"{max(int(seconds), 0)} s ago"
    return f"{format_delta(seconds)} ago"


def describe_next(nxt, zone, now):
    """A device's STATUS ``next`` field, in words."""
    if not nxt or nxt == "-":
        return ""
    if nxt.startswith("at="):
        due = parse_iso(nxt[3:])
        if due:
            return (f"next phase {due.astimezone(zone):%a %H:%M} "
                    f"(in {format_delta((due - now).total_seconds())})")
    if nxt.startswith(("rewards=", "pokes=")):
        kind, count = nxt.split("=", 1)
        return f"next phase at {count} {kind}"
    if nxt == "goto":
        return "next phase through FNT or by hand"
    return ""


def describe_health(device, status):
    if device is None or not device["connected"]:
        return "not connected", theme.BAD
    parts, colour = [f"{device['port']}"], theme.MUTED
    if status.get("clock") == "unset":
        parts.append("clock not set")
        colour = theme.WARN
    if status.get("sd") == "err":
        parts.append("SD error")
        colour = theme.BAD
    if status.get("jam") == "1":
        parts.append("jammed")
        colour = theme.BAD
    # A FED3 without a temperature sensor (most have none) reports nan.
    for key, unit in (("batt", "V"), ("temp", "°C"), ("hum", "% RH")):
        if status.get(key, "nan") not in ("", "nan"):
            parts.append(f"{status[key]} {unit}")
    return " · ".join(parts), colour


class DeviceCard(QFrame):
    def __init__(self, logical, colour, engine, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.logical = logical
        self.engine = engine
        self.problem_keys = []
        layout = QVBoxLayout(self)
        layout.setSpacing(4)

        top = QHBoxLayout()
        self.title = QLabel(f"<b style='color:{colour}; font-size:12pt'>{logical}</b>")
        top.addWidget(self.title)
        self.dot = QLabel("●")
        top.addWidget(self.dot)
        self.fed = QLabel()
        top.addWidget(self.fed)
        top.addStretch()
        for text, tip, slot in (
                ("Feed", "Dispense one pellet now", self._feed),
                ("Test cue…", "Play a cue on this device", self._test),
                ("Set clock", "Set this device's clock to this PC's, even if that's a big "
                              "jump (it refuses jumps over 10 minutes during a run "
                              "otherwise)", self._set_clock)):
            button = theme.small(QPushButton(text))
            button.setToolTip(tip)
            button.clicked.connect(slot)
            top.addWidget(button)
        layout.addLayout(top)

        body = QHBoxLayout()
        self.graphic = FedGraphic()
        self.graphic.setToolTip("Left pokes, pellets and right pokes since the run started")
        body.addWidget(self.graphic, 0, Qt.AlignTop)
        facts = QVBoxLayout()
        facts.setSpacing(2)
        self.phase = QLabel()
        self.rule = QLabel()
        self.rule.setObjectName("muted")
        self.progress = QLabel()
        self.last = QLabel()
        self.last.setObjectName("muted")
        self.health = QLabel()
        for label in (self.phase, self.rule, self.progress, self.last, self.health):
            label.setWordWrap(True)
            facts.addWidget(label)
        facts.addStretch()
        body.addLayout(facts, 1)
        layout.addLayout(body)

        self.problem_box = QVBoxLayout()
        layout.addLayout(self.problem_box)

    def _feed(self):
        self.engine.submit("feed", device=self.logical)

    def _test(self):
        cue, ok = QInputDialog.getText(self, "Test a cue", "Cue (as in a schedule):",
                                       text=TEST_CUE)
        if ok and cue:
            self.engine.submit("test_cue", device=self.logical, cue=cue)

    def _set_clock(self):
        self.engine.submit("confirm_time", device=self.logical)

    def update_card(self, device, run, problems, now, zone):
        connected = bool(device and device["connected"])
        self.dot.setStyleSheet(f"color: {theme.GOOD if connected else theme.BAD};")
        status = device["status"] if device else {}
        row = device["last_row"] if device else {}
        health, colour = describe_health(device, status)
        self.health.setText(health)
        self.health.setStyleSheet(f"color: {colour};")
        self.fed.setText(f"FED{device['id']:03d}" if device else "")

        source = row or status
        counts = None
        if source:
            try:
                counts = {"left": int(source.get("left", source.get("l", 0))),
                          "right": int(source.get("right", source.get("r", 0))),
                          "pellets": int(source.get("pellets", 0))}
            except ValueError:
                counts = None
        self.graphic.set_state(counts, connected)

        try:
            index = int(status.get("phase", row.get("phase", 0)))
        except ValueError:
            index = 0
        phases = run["phases"]
        name = phases[index] if index < len(phases) else "?"
        self.phase.setText(f"<b>{name}</b> · phase {index + 1} of {len(phases)}")
        rules = run["rules"].get(self.logical, [])
        self.rule.setText(rules[index] if index < len(rules) else "")
        rewards = (row or status).get("phase_rewards")
        actives = run.get("actives", {}).get(self.logical, [])
        can_earn = index < len(actives) and actives[index] != "none"
        progress = [f"{rewards} rewards this phase"] if rewards is not None and can_earn else []
        if run["state"] == "running":
            nxt = describe_next(status.get("next"), zone, now)
            if nxt:
                progress.append(nxt)
        self.progress.setText(" · ".join(progress))
        if row:
            when = parse_iso(row.get("time", ""))
            what = " ".join(w.replace("_", " ") for w in
                            (row.get("event"), row.get("side"), row.get("outcome")) if w)
            self.last.setText(f"last: {what}, {ago(when, now)}")
        else:
            self.last.setText("last: —")

        mine = [p for p in problems if p["device"] == self.logical]
        keys = [(p["key"], p["active"], p["message"]) for p in mine]
        if keys != self.problem_keys:
            self.problem_keys = keys
            while self.problem_box.count():
                item = self.problem_box.takeAt(0)
                if item.layout():
                    while item.layout().count():
                        sub = item.layout().takeAt(0)
                        if sub.widget():
                            sub.widget().deleteLater()
                elif item.widget():
                    item.widget().deleteLater()
            for problem in mine:
                line = QHBoxLayout()
                state = "" if problem["active"] else " (cleared)"
                label = QLabel(f"⚠ {problem['message']}{state} · "
                               f"{problem['since'].astimezone(zone):%a %H:%M}")
                label.setStyleSheet(f"color: {theme.BAD if problem['active'] else theme.WARN};")
                label.setWordWrap(True)
                line.addWidget(label, 1)
                if not problem["acknowledged"]:
                    ack = theme.small(QPushButton("Acknowledge"))
                    ack.clicked.connect(lambda _=False, k=problem["key"]:
                                        self.engine.submit("acknowledge", key=k))
                    line.addWidget(ack)
                self.problem_box.addLayout(line)
        flagged = any(not p["acknowledged"] for p in mine)
        if self.property("problem") != flagged:
            self.setProperty("problem", flagged)
            self.style().unpolish(self)
            self.style().polish(self)


class CardArea(QWidget):
    """Device cards in as many columns as the width allows, at their own
    height: no scrolling; the plot above gives up the room."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setSizePolicy(QSizePolicy.Preferred, QSizePolicy.Maximum)
        self.grid = QGridLayout(self)
        self.grid.setContentsMargins(0, 0, 0, 0)
        self.cards = []
        self.columns = 0

    def set_cards(self, cards):
        for card in self.cards:
            self.grid.removeWidget(card)
            card.deleteLater()
        self.cards = list(cards)
        self.columns = 0
        self._reflow()

    def _reflow(self):
        columns = max(1, min(len(self.cards) or 1, self.width() // CARD_WIDTH))
        if columns == self.columns:
            return
        self.columns = columns
        for card in self.cards:
            self.grid.removeWidget(card)
        for n, card in enumerate(self.cards):
            self.grid.addWidget(card, n // columns, n % columns)
        for column in range(self.grid.columnCount()):
            self.grid.setColumnStretch(column, 1 if column < columns else 0)

    def resizeEvent(self, event):
        super().resizeEvent(event)
        self._reflow()


class RunView(QWidget):
    def __init__(self, engine, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.cards = {}
        self.run_id = None
        self.snapshot = {}

        layout = QVBoxLayout(self)
        header = QHBoxLayout()
        self.title = QLabel()
        self.title.setObjectName("title")
        header.addWidget(self.title)
        self.state = QLabel()
        header.addWidget(self.state)
        self.elapsed = QLabel()
        header.addWidget(self.elapsed)
        self.problem_count = QLabel()
        header.addWidget(self.problem_count)
        header.addStretch()
        self.edit = theme.quiet(QPushButton("Edit protocol…"))
        self.edit.clicked.connect(self._edit)
        self.advance = theme.quiet(QPushButton("Advance phase"))
        self.advance.clicked.connect(self._advance)
        self.stop = theme.danger(QPushButton("Stop run"))
        self.stop.clicked.connect(self._stop)
        self.open_folder = theme.quiet(QPushButton("Open folder"))
        self.open_folder.clicked.connect(self._open_folder)
        self.close_run = QPushButton("New run")
        self.close_run.clicked.connect(lambda: engine.submit("close"))
        for button in (self.edit, self.advance, self.stop, self.open_folder, self.close_run):
            header.addWidget(button)
        layout.addLayout(header)
        self.timeline = QLabel()
        self.timeline.setObjectName("muted")
        self.timeline.setWordWrap(True)
        layout.addWidget(self.timeline)
        self.banner = QLabel()
        self.banner.setStyleSheet(f"color: {theme.BAD};")
        self.banner.setWordWrap(True)
        layout.addWidget(self.banner)

        # The plot's title, with the time zone its axis is drawn in.
        title = QHBoxLayout()
        title.addStretch()
        title.addWidget(QLabel("Cumulative pellets, in"))
        self.zone_box = QComboBox()
        self.zone_box.setToolTip("The time zone of the plot's time axis")
        self.zone_box.currentIndexChanged.connect(self._zone_changed)
        title.addWidget(self.zone_box)
        title.addStretch()
        layout.addLayout(title)
        self.plot = PelletPlot()
        self.plot.setMinimumHeight(160)
        layout.addWidget(self.plot, 1)
        self.card_area = CardArea()
        layout.addWidget(self.card_area)

    # --- actions ---------------------------------------------------------

    def _plot_zone(self, standard, now):
        """The chosen zone for the plot; fills the choices the first time."""
        choices = zones(standard, now)
        if self.zone_box.count() != len(choices):
            self.zone_box.blockSignals(True)
            self.zone_box.clear()
            for key, label, _ in choices:
                self.zone_box.addItem(label, key)
            self.zone_box.blockSignals(False)
        key = self.zone_box.currentData()
        return next(zone for k, _, zone in choices if k == key)

    def _zone_changed(self):
        self.plot.last_draw = 0.0       # redraw now, not at the next refresh
        if self.snapshot.get("run"):
            self.update_view(self.snapshot)

    def _edit(self):
        run = self.snapshot.get("run")
        if not run:
            return
        editor = ProtocolEditor(run["protocol_text"], self, running=True, t0=run["t0"])
        if editor.exec_():
            self.engine.submit("edit", protocol_text=editor.text())

    def _advance(self):
        if QMessageBox.question(self, "Advance phase",
                                "Move every device in this phase to the next one now?") \
                == QMessageBox.Yes:
            self.engine.submit("advance")

    def _stop(self):
        answer = QMessageBox.question(
            self, "Stop run",
            "Stop this run?\n\nPhases stop changing. Each device keeps running its current "
            "phase (and its daily windows) until the next run starts, and keeps recording: "
            "its rows still reach this run's folder while FNT is open.")
        if answer == QMessageBox.Yes:
            self.engine.submit("stop")

    def _open_folder(self):
        run = self.snapshot.get("run")
        if not run:
            return
        folder = run["folder"]
        if sys.platform.startswith("win"):
            os.startfile(folder)  # noqa: S606 - opening a folder the user owns
        elif sys.platform == "darwin":
            subprocess.Popen(["open", folder])
        else:
            subprocess.Popen(["xdg-open", folder])

    # --- refresh ---------------------------------------------------------

    def update_view(self, snapshot):
        self.snapshot = snapshot
        run = snapshot.get("run")
        if not run:
            return
        zone = snapshot["zone"]
        now = snapshot["now"].astimezone(zone)
        if run["run_id"] != self.run_id:
            self._build_cards(run)
        running = snapshot["state"] == "running"
        self.title.setText(run["name"])
        self.state.setText("● RUNNING" if running else "■ STOPPED")
        self.state.setStyleSheet(f"color: {theme.GOOD if running else theme.MUTED}; font-weight: bold;")
        self.elapsed.setText(f"{format_delta((now - run['t0']).total_seconds())} since "
                             f"{run['t0']:%a %d %b %H:%M} ({run['t0']:%Z})")
        problems = snapshot["problems"]
        open_count = sum(not p["acknowledged"] for p in problems)
        self.problem_count.setText(f"⚠ {open_count} problem{'s' if open_count != 1 else ''}"
                                   if open_count else "")
        self.problem_count.setStyleSheet(f"color: {theme.BAD}; font-weight: bold;")
        for button in (self.edit, self.advance, self.stop):
            button.setVisible(running)
        for button in (self.close_run,):
            button.setVisible(not running)

        devices = {d["id"]: d for d in snapshot["devices"] if d["id"] is not None}
        phases_now = []
        for logical, device_id in run["binding"].items():
            device = devices.get(device_id)
            if logical in self.cards:
                self.cards[logical].update_card(device, run, problems, now, zone)
            if device and device["status"].get("phase"):
                phases_now.append((int(device["status"]["phase"]), device["status"].get("next", "-")))
        self.advance.setVisible(running and any(run["manual"][p] for p, _ in phases_now
                                                if p < len(run["manual"])))
        if not running:
            stopped = next((e[0] for e in reversed(snapshot["events"])
                            if e[2].startswith("run stopped")), None)
            when = f" {stopped.astimezone(zone):%a %d %b %H:%M}" if stopped else ""
            totals = []
            for logical, device_id in run["binding"].items():
                row = (devices.get(device_id) or {}).get("last_row") or {}
                totals.append(f"{logical}: {row.get('pellets', '?')} pellets")
            self.timeline.setText(f"Stopped{when}. The devices keep running their current "
                                  f"phase and recording; their rows are still copied to this "
                                  f"run's folder until they start another run.  ·  "
                                  + "   ".join(totals))
        elif phases_now:
            index, nxt = min(phases_now)
            text = f"Phase {index + 1} of {len(run['phases'])}: {run['phases'][index]}"
            words = describe_next(nxt, zone, now)
            if words:
                text += f"  ·  {words}"
            self.timeline.setText(text)
        general = [p for p in problems if not p["device"] and not p["acknowledged"]]
        self.banner.setText("\n".join(f"⚠ {p['message']}" for p in general))
        self.banner.setVisible(bool(general))
        self.plot.update_plot(run, self._plot_zone(zone, now))

    def _build_cards(self, run):
        self.run_id = run["run_id"]
        self.cards = {logical: DeviceCard(logical, theme.SERIES[n % len(theme.SERIES)], self.engine)
                      for n, logical in enumerate(run["binding"])}
        self.card_area.set_cards(self.cards.values())
