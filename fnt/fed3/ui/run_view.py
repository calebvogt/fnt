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
    QComboBox, QFrame, QGridLayout, QHBoxLayout, QInputDialog, QLabel, QMenu, QMessageBox,
    QPushButton, QSizePolicy, QVBoxLayout, QWidget,
)

from .. import protocol as proto
from ..device_sim import parse_iso
from ..run import check_edit
from . import theme
from .editor import ProtocolEditor, format_delta
from .fed_graphic import FedGraphic
from .health import health, readings
from .plot import PelletPlot, zones

CARD_WIDTH = 370                # cards reflow into as many columns as fit
TEST_CUE = "tone:4000:200+flash:bar:0:0:30:0:500"
# What a card's "last:" reports: what the animal and the run did, not housekeeping
# (clock corrections, hourly status rows, boots).
CARD_EVENTS = ("poke", "dispense", "retrieve", "jam", "phase", "run_start", "run_end")


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


def pellet_waiting(rows, status):
    """Whether a pellet sits in the well: from the latest dispense or retrieve
    row, which is live, else from STATUS, which is polled every 30 s."""
    for row in reversed(rows):
        if row.get("event") in ("dispense", "retrieve"):
            return row["event"] == "dispense"
    return status.get("well") == "1"


def card_health(device):
    """A card's last line: the port, the FED3's state and its readings."""
    parts, colour = health(device)
    if device and device["connected"]:
        parts = [device["port"], *parts, *readings(device["status"])]
    return " · ".join(parts), theme.MUTED if colour == theme.GOOD else colour


def card_counts(row):
    """Left pokes, right pokes and pellets from a data row or a STATUS, or None."""
    if not row:
        return None
    try:
        return {"left": int(row.get("left", row.get("l", 0))),
                "right": int(row.get("right", row.get("r", 0))),
                "pellets": int(row.get("pellets", 0))}
    except ValueError:
        return None


class DeviceCard(QFrame):
    def __init__(self, logical, colour, engine, parent=None):
        super().__init__(parent)
        self.setObjectName("card")
        self.logical = logical
        self.engine = engine
        self.problem_keys = []
        self.device = None
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
        feed = theme.small(QPushButton("Feed"))
        feed.setToolTip("Dispense one pellet now. It's recorded as a manual dispense: it "
                        "counts as a pellet, not a reward.")
        feed.clicked.connect(self._feed)
        top.addWidget(feed)
        more = theme.small(QPushButton("More ▾"))
        menu = QMenu(more)
        menu.addAction("Play a cue…", self._test).setToolTip("Play a tone, flash or BNC "
                                                              "pulse on this FED3")
        menu.addAction("Set its clock now…", self._set_clock)
        more.setMenu(menu)
        top.addWidget(more)
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
        cue, ok = QInputDialog.getText(
            self, f"Play a cue on {self.logical}",
            "Cue, written as in a schedule. The default is a 4 kHz tone with a flash:",
            text=TEST_CUE)
        if ok and cue:
            self.engine.submit("test_cue", device=self.logical, cue=cue)

    def _set_clock(self):
        drift = (self.device or {}).get("drift")
        off = (f"Its clock is {abs(drift):.0f} s {'ahead' if drift > 0 else 'behind'}. "
               if drift is not None else "")
        answer = QMessageBox.question(
            self, f"Set {self.logical}'s clock",
            f"{off}Set {self.logical}'s clock to this PC's time now?\n\n"
            "FNT keeps clocks right by itself. This is for a jump over 10 minutes, which "
            "a FED3 refuses during a run until someone confirms it: phases due in the "
            "meantime happen at once.")
        if answer == QMessageBox.Yes:
            self.engine.submit("confirm_time", device=self.logical)

    def update_card(self, device, run, problems, now, zone):
        self.device = device
        connected = bool(device and device["connected"])
        status = device["status"] if device else {}
        rows = run["recent"].get(self.logical, [])     # this run's, kept across restarts
        latest = (device["last_row"] if device else {}) or (rows[-1] if rows else status)
        self.dot.setStyleSheet(f"color: {theme.GOOD if connected else theme.BAD};")
        # The card is titled with the FED3's ID; say so only if it's been renumbered since.
        current = f"FED{device['id']:03d}" if device else ""
        self.fed.setText(f"now {current}" if current and current != self.logical else "")
        health_text, colour = card_health(device)
        self.health.setText(health_text)
        self.health.setStyleSheet(f"color: {colour};")
        self.graphic.set_state(card_counts(latest), connected)
        self._show_phase(run, status, latest, rows, now, zone)
        self._show_last(rows, now)
        self._show_problems([p for p in problems if p["device"] == self.logical], zone)

    def _show_phase(self, run, status, latest, rows, now, zone):
        """The phase, its rule, and progress: rewards, the well, the next change."""
        try:
            index = int(status.get("phase", latest.get("phase", 0)))
        except ValueError:
            index = 0
        phases = run["phases"]
        name = phases[index] if index < len(phases) else "?"
        self.phase.setText(f"<b>{name}</b> · phase {index + 1} of {len(phases)}")
        rules = run["rules"].get(self.logical, [])
        self.rule.setText(rules[index] if index < len(rules) else "")
        actives = run.get("actives", {}).get(self.logical, [])
        rewards = latest.get("phase_rewards")
        progress = []
        if rewards is not None and index < len(actives) and actives[index] != "none":
            progress.append(f"{rewards} rewards this phase")
        if pellet_waiting(rows, status):
            progress.append("a pellet is waiting in the well")
        if run["state"] == "running":
            nxt = describe_next(status.get("next"), zone, now)
            if nxt:
                progress.append(nxt)
        self.progress.setText(" · ".join(progress))
        self.progress.setVisible(bool(progress))

    def _show_last(self, rows, now):
        event = next((r for r in reversed(rows) if r.get("event") in CARD_EVENTS), None)
        if not event:
            self.last.setText("last: —")
            return
        what = " ".join(w.replace("_", " ") for w in
                        (event.get("event"), event.get("side"), event.get("outcome")) if w)
        self.last.setText(f"last: {what}, {ago(parse_iso(event.get('time', '')), now)}")

    def _show_problems(self, mine, zone):
        """This FED3's problems, each with Acknowledge until it's been seen."""
        keys = [(p["key"], p["active"], p["message"]) for p in mine]
        if keys != self.problem_keys:
            self.problem_keys = keys
            self._clear_problems()
            for problem in mine:
                self.problem_box.addLayout(self._problem_line(problem, zone))
        flagged = any(not p["acknowledged"] for p in mine)
        if self.property("problem") != flagged:
            self.setProperty("problem", flagged)
            self.style().unpolish(self)
            self.style().polish(self)

    def _clear_problems(self):
        while self.problem_box.count():
            line = self.problem_box.takeAt(0).layout()
            while line and line.count():
                widget = line.takeAt(0).widget()
                if widget:
                    widget.deleteLater()

    def _problem_line(self, problem, zone):
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
        return line


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
        # As few rows as the width allows, then as even as they can be: 4 cards
        # that fit 3 across go 2 + 2, not 3 + 1.
        count = len(self.cards) or 1
        rows = -(-count // max(1, self.width() // CARD_WIDTH))
        columns = -(-count // rows)
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
        self.pending_edit = None        # (token, text) sent, awaiting the engine's answer
        self.edit_tokens = 0

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
        self.edit.setToolTip("Change the protocol mid-run. Phases already over can't change.")
        self.edit.clicked.connect(self._edit)
        self.advance = theme.quiet(QPushButton("Advance phase"))
        self.advance.clicked.connect(self._advance)
        self.stop = theme.danger(QPushButton("Stop run"))
        self.stop.clicked.connect(self._stop)
        self.open_folder = theme.quiet(QPushButton("Open folder"))
        self.open_folder.setToolTip("Open this run's folder: its data, protocol and log")
        self.open_folder.clicked.connect(self._open_folder)
        self.close_run = QPushButton("New run")
        self.close_run.setToolTip("Back to Setup. This run's FED3s keep recording into its "
                                  "folder until they start another run.")
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

    def _edit(self, text=None):
        run = self.snapshot.get("run")
        if not run:
            return
        old = proto.compile_protocol(proto.load(run["protocol_text"]), seed=run["seed"])

        def edit_check(new_text):
            """Why the running devices can't take this, as the engine would say."""
            current = (self.snapshot.get("run") or run)["current"]
            try:
                check_edit(old, new_text, current, run["seed"])
            except Exception as exc:      # noqa: BLE001 - shown in the editor
                return str(exc)
            return None

        editor = ProtocolEditor(text or run["protocol_text"], self, running=True,
                                t0=run["t0"], edit_check=edit_check)
        if editor.exec_():
            self.edit_tokens += 1
            self.pending_edit = (self.edit_tokens, editor.text())
            self.engine.submit("edit", protocol_text=editor.text(), token=self.edit_tokens)

    def _edit_answered(self, snapshot):
        """The engine refused an edit after all (a device moved on meanwhile):
        say so, and reopen the editor on what was written so nothing is lost."""
        result = snapshot.get("edit_result")
        if not self.pending_edit or not result or result["token"] != self.pending_edit[0]:
            return
        _, text = self.pending_edit
        self.pending_edit = None
        if result["error"]:
            QMessageBox.warning(self, "Edit not applied",
                                f"The edit wasn't applied: {result['error']}.\n\nNothing "
                                "changed on the devices. The editor opens again with your "
                                "changes.")
            self._edit(text)

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
        stopped = parse_iso(run.get("stopped_at") or "")
        if running or not stopped:
            self.elapsed.setText(f"{format_delta((now - run['t0']).total_seconds())} since "
                                 f"{run['t0']:%a %d %b %H:%M} ({run['t0']:%Z})")
        else:
            self.elapsed.setText(f"ran {format_delta((stopped - run['t0']).total_seconds())}"
                                 f", {run['t0']:%a %d %b %H:%M} to "
                                 f"{stopped.astimezone(zone):%a %d %b %H:%M} ({run['t0']:%Z})")
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
            self.timeline.setText("Stopped. Each FED3 stays in its current phase and keeps "
                                  "recording; its rows still reach this run's folder (after "
                                  "its run_end row) until it starts another run.")
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
        self.plot.update_plot(run, self._plot_zone(zone, now), now)
        self._edit_answered(snapshot)

    def _build_cards(self, run):
        self.run_id = run["run_id"]
        self.cards = {logical: DeviceCard(logical, theme.SERIES[n % len(theme.SERIES)], self.engine)
                      for n, logical in enumerate(run["binding"])}
        self.card_area.set_cards(self.cards.values())
