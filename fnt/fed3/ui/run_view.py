"""Running and Stopped: one experiment, its devices, its problems and its data."""
import os
import subprocess
import sys

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QFrame, QGridLayout, QHBoxLayout, QInputDialog, QLabel, QListWidget,
    QListWidgetItem, QMenu, QMessageBox, QPushButton, QScrollArea, QSplitter,
    QVBoxLayout, QWidget,
)

from ..device_sim import parse_iso
from . import theme
from .editor import ProtocolEditor, format_delta
from .plot import PelletPlot


def ago(moment, now):
    if moment is None:
        return "—"
    return f"{format_delta((now - moment).total_seconds())} ago"


class DeviceCard(QFrame):
    def __init__(self, logical, colour, engine, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.logical = logical
        self.engine = engine
        self.problem_keys = []
        layout = QVBoxLayout(self)
        top = QHBoxLayout()
        self.title = QLabel(f"<b style='color:{colour}'>{logical}</b>")
        top.addWidget(self.title)
        self.dot = QLabel("●")
        top.addWidget(self.dot)
        top.addStretch()
        menu_button = theme.quiet(QPushButton("⋯"))
        menu = QMenu(menu_button)
        menu.addAction("Feed one pellet", lambda: engine.submit("feed", device=logical))
        menu.addAction("Test a cue…", self._test)
        menu.addAction("Set its clock now", lambda: engine.submit("confirm_time", device=logical))
        menu_button.setMenu(menu)
        top.addWidget(menu_button)
        layout.addLayout(top)
        self.phase = QLabel()
        self.rule = QLabel()
        self.rule.setObjectName("muted")
        self.counts = QLabel()
        self.counts.setWordWrap(True)
        self.last = QLabel()
        self.last.setObjectName("muted")
        for widget in (self.phase, self.rule, self.counts, self.last):
            layout.addWidget(widget)
        self.problem_box = QVBoxLayout()
        layout.addLayout(self.problem_box)

    def _test(self):
        cue, ok = QInputDialog.getText(self, "Test a cue", "Cue (as in a schedule):",
                                       text="tone:4000:200+flash:bar:0:0:30:0:500")
        if ok and cue:
            self.engine.submit("test_cue", device=self.logical, cue=cue)

    def update_card(self, device, run, problems, now):
        if device is None or not device["connected"]:
            self.dot.setStyleSheet(f"color: {theme.BAD};")
            self.dot.setToolTip("not connected")
        else:
            self.dot.setStyleSheet(f"color: {theme.GOOD};")
            self.dot.setToolTip(f"FED{device['id']:03d} on {device['port']}")
        status = device["status"] if device else {}
        row = device["last_row"] if device else {}
        try:
            index = int(status.get("phase", row.get("phase", 0)))
        except ValueError:
            index = 0
        phases = run["phases"]
        name = phases[index] if index < len(phases) else "?"
        self.phase.setText(f"<b>{name}</b>  ·  phase {index + 1} of {len(phases)}")
        rules = run["rules"].get(self.logical, [])
        self.rule.setText(rules[index] if index < len(rules) else "")
        if row:
            self.counts.setText(f"L {row.get('left', 0)}   R {row.get('right', 0)}   "
                                f"pellets {row.get('pellets', 0)}   "
                                f"rewards this phase {row.get('phase_rewards', 0)}")
            when = parse_iso(row.get("time", ""))
            self.last.setText(f"last: {row.get('event')} {row.get('outcome', '')} {ago(when, now)}")
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
                               f"{problem['since'].astimezone(now.tzinfo):%a %H:%M}")
                label.setStyleSheet(f"color: {theme.BAD if problem['active'] else theme.WARN};")
                label.setWordWrap(True)
                line.addWidget(label, 1)
                if not problem["acknowledged"]:
                    ack = theme.quiet(QPushButton("Acknowledge"))
                    ack.clicked.connect(lambda _=False, k=problem["key"]:
                                        self.engine.submit("acknowledge", key=k))
                    line.addWidget(ack)
                self.problem_box.addLayout(line)
        flagged = any(not p["acknowledged"] for p in mine)
        if self.property("problem") != flagged:
            self.setProperty("problem", flagged)
            self.style().unpolish(self)
            self.style().polish(self)


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
        layout.addWidget(self.timeline)
        self.banner = QLabel()
        self.banner.setStyleSheet(f"color: {theme.BAD};")
        self.banner.setWordWrap(True)
        layout.addWidget(self.banner)

        split = QSplitter(Qt.Horizontal)
        cards_area = QScrollArea()
        cards_area.setWidgetResizable(True)
        cards_area.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        holder = QWidget()
        self.grid = QGridLayout(holder)
        self.grid.setAlignment(Qt.AlignTop)
        cards_area.setWidget(holder)
        split.addWidget(cards_area)
        right = QSplitter(Qt.Vertical)
        self.plot = PelletPlot()
        right.addWidget(self.plot)
        self.log = QListWidget()
        right.addWidget(self.log)
        right.setSizes([420, 200])
        split.addWidget(right)
        split.setSizes([620, 620])
        layout.addWidget(split, 1)

    # --- actions ---------------------------------------------------------

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
            "Stop recording this run?\n\nEach device keeps running its current phase "
            "(and its daily windows) until the next run starts.")
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
                self.cards[logical].update_card(device, run, problems, now)
            if device and device["status"].get("phase"):
                phases_now.append((int(device["status"]["phase"]), device["status"].get("next", "-")))
        self.advance.setVisible(running and any(run["manual"][p] for p, _ in phases_now
                                                if p < len(run["manual"])))
        if not running:
            stopped = next((e[0] for e in reversed(snapshot["events"])
                            if e[2] == "run stopped"), None)
            when = f" {stopped.astimezone(zone):%a %d %b %H:%M}" if stopped else ""
            totals = []
            for logical, device_id in run["binding"].items():
                row = (devices.get(device_id) or {}).get("last_row") or {}
                totals.append(f"{logical}: {row.get('pellets', '?')} pellets")
            self.timeline.setText(f"Stopped{when}. The devices keep running their current "
                                  f"phase until the next run.  ·  " + "   ".join(totals))
        elif phases_now:
            index, nxt = min(phases_now)
            text = f"Phase {index + 1} of {len(run['phases'])}: {run['phases'][index]}"
            if nxt.startswith("at="):
                due = parse_iso(nxt[3:])
                if due:
                    text += f"  ·  next change {due.astimezone(zone):%a %H:%M} (in {format_delta((due - now).total_seconds())})"
            elif nxt.startswith(("rewards=", "pokes=")):
                text += f"  ·  next change at {nxt.replace('=', ' ')}"
            elif nxt == "goto":
                text += "  ·  ends through FNT or by hand"
            self.timeline.setText(text)
        general = [p for p in problems if not p["device"] and not p["acknowledged"]]
        self.banner.setText("\n".join(f"⚠ {p['message']}" for p in general))

        events = snapshot["events"][-60:]
        if self.log.count() != len(events) or (events and self.log.item(0).text()
                                               != self._event_text(events[-1], zone)):
            self.log.clear()
            for event in reversed(events):
                item = QListWidgetItem(self._event_text(event, zone))
                if "problem" in event[2]:
                    item.setForeground(QColor(theme.BAD))
                self.log.addItem(item)
        self.plot.update_plot(run, zone)

    @staticmethod
    def _event_text(event, zone):
        moment, device, text = event
        who = f"{device}: " if device else ""
        return f"{moment.astimezone(zone):%a %H:%M:%S}  {who}{text}"

    def _build_cards(self, run):
        self.run_id = run["run_id"]
        for card in self.cards.values():
            card.deleteLater()
        self.cards = {}
        columns = 1 if len(run["binding"]) < 3 else 2
        for n, logical in enumerate(run["binding"]):
            card = DeviceCard(logical, theme.SERIES[n % len(theme.SERIES)], self.engine)
            self.grid.addWidget(card, n // columns, n % columns)
            self.cards[logical] = card
