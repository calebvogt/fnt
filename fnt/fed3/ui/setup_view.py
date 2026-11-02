"""Setup: choose a protocol, check the FED3s it names are plugged in, then Start.

Two columns: the protocol and its timeline on the left; on the right one row
per FED3 plugged in, in port order: its port, its ID, its state, and bench
tests (lights to find it, a tone, a pellet).

A FED3's ID is its one name: on its screen, in the protocol (``devices:
[FED001, FED002]``), in its data file and on its card. It's kept on the SD
card and set on the FED3's boot menu, or here in the ID list. A row is a FED3
on a port and stays put.

Nothing here talks to a device directly; it reads the engine's snapshot and
submits commands, like the rest of the window.
"""
import os
import re
import time
from datetime import datetime

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QFileDialog, QGroupBox, QHBoxLayout, QHeaderView, QLabel, QLineEdit,
    QMessageBox,
    QPlainTextEdit, QPushButton, QSplitter, QTableWidget, QTableWidgetItem, QVBoxLayout,
    QWidget,
)

from .. import protocol as proto
from .. import schedule as sch
from . import theme
from .editor import ProtocolEditor

TEST_TONE = "tone:4000:200"
ID_TEXT = re.compile(r"\s*(?:FED)?\s*(\d{1,3})\s*", re.IGNORECASE)   # 7, 007, FED007
PENDING_FOR = 5.0           # seconds a chosen ID shows while the FED3 takes it
STARTING_FOR = 30.0         # seconds Start stays off waiting for pre-flight
PORT, ID, STATUS, TESTS = range(4)
COLUMNS = ["Port", "ID", "Status", "Tests"]


def fed_name(number):
    return f"FED{number:03d}"


def parse_id(text):
    """A typed ID as a number, or None: 7, 007, FED007 and fed7 are all 7."""
    match = ID_TEXT.fullmatch(text or "")
    number = int(match.group(1)) if match else 0
    return number if 1 <= number <= 999 else None


def port_order(port):
    """COM3 before COM10: ports in natural order, so rows don't move."""
    return [int(p) if p.isdigit() else p for p in re.split(r"(\d+)", port)]


def device_health(device):
    """One line: what's ready and what isn't."""
    if not device["connected"]:
        return "not connected", theme.BAD
    status = device["status"]
    if not status:
        return "connected · reading its status…", theme.MUTED
    parts, colour = [], theme.GOOD
    clock = status.get("clock")
    if clock == "ok":
        drift = device.get("drift")
        parts.append("clock ok" if drift is None or abs(drift) < 5 else f"clock {drift:+.0f}s")
    else:
        parts.append("clock not set")
        colour = theme.WARN
    if status.get("sd") == "ok":
        parts.append("SD ok")
    else:
        parts.append("SD error")
        colour = theme.BAD
    if status.get("jam") == "1":
        parts.append("jammed")
        colour = theme.BAD
    parts.append(f"fw {device['firmware']}")
    if device.get("live_run"):
        parts.append(f"still in run {device['run_label']}")
        colour = theme.BAD if colour == theme.BAD else theme.WARN
    return " · ".join(parts), colour


class SetupView(QWidget):
    def __init__(self, engine, settings, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.settings = settings
        self.protocol_text = ""
        self.protocol_path = settings.value("protocol_path", "") or ""
        self.compiled = None
        self.snapshot = {}
        self.rows = None            # (ports shown, ID choices, roles) last built
        self.pending = {}           # port -> (number chosen, when): not confirmed yet
        self.starting = 0.0         # when Start was pressed, until pre-flight answers

        layout = QVBoxLayout(self)
        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)

        # --- protocol ------------------------------------------------------
        group = QGroupBox("Protocol")
        g = QVBoxLayout(group)
        row = QHBoxLayout()
        self.path_label = QLabel("No protocol chosen")
        self.path_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        row.addWidget(self.path_label, 1)
        self.edit_button = None
        for text, tip, slot in (
                ("Open…", "Open a protocol file", self._open),
                ("New…", "Write a new protocol, starting from a template", self._new),
                ("Edit…", "Edit this protocol", self._edit)):
            button = theme.quiet(QPushButton(text))
            button.setToolTip(tip)
            button.clicked.connect(slot)
            row.addWidget(button)
            self.edit_button = button
        g.addLayout(row)
        self.verdict = QLabel()
        self.verdict.setWordWrap(True)
        g.addWidget(self.verdict)
        self.timeline = QPlainTextEdit()
        self.timeline.setReadOnly(True)
        self.timeline.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.timeline.setToolTip("When each phase would start if you pressed Start now, "
                                 "in standard time")
        g.addWidget(self.timeline, 1)
        split.addWidget(group)

        # --- devices -------------------------------------------------------
        group = QGroupBox("FED3s")
        g = QVBoxLayout(group)
        self.hint = QLabel()
        self.hint.setObjectName("muted")
        self.hint.setWordWrap(True)
        g.addWidget(self.hint)
        self.table = QTableWidget(0, len(COLUMNS))
        self.table.setHorizontalHeaderLabels(COLUMNS)
        tips = {
            PORT: "The port the FED3 is plugged into. A row always shows the same FED3.",
            ID: "The FED3's ID: its name in the protocol, its data and its card. Kept on "
                "its SD card; set it on the FED3's boot menu, or type it here and press "
                "Enter. An ID another FED3 has swaps the two.",
            STATUS: "Whether it's ready: clock, SD card, firmware, and any run it's in.",
            TESTS: "Lights it up to find it, plays the reward tone, or dispenses a pellet.",
        }
        for column, tip in tips.items():
            self.table.horizontalHeaderItem(column).setToolTip(tip)
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        for column in (PORT, ID, TESTS):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(STATUS, QHeaderView.Stretch)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        self.table.setFocusPolicy(Qt.NoFocus)
        g.addWidget(self.table, 1)
        self.missing = QLabel()
        self.missing.setWordWrap(True)
        g.addWidget(self.missing)
        self.others = QLabel()
        self.others.setWordWrap(True)
        g.addWidget(self.others)
        self.id_note = QLabel()
        self.id_note.setObjectName("muted")
        self.id_note.setWordWrap(True)
        g.addWidget(self.id_note)
        self.recording = QLabel()
        self.recording.setObjectName("muted")
        self.recording.setWordWrap(True)
        g.addWidget(self.recording)
        split.addWidget(group)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        split.setSizes([480, 720])
        layout.addWidget(split, 1)

        # --- start -----------------------------------------------------------
        row = QHBoxLayout()
        self.folder_label = QLabel()
        self.folder_label.setObjectName("muted")
        self.folder_label.setTextInteractionFlags(Qt.TextSelectableByMouse)
        row.addWidget(self.folder_label, 1)
        folder = theme.quiet(QPushButton("Change folder…"))
        folder.setToolTip("Where new runs are saved. Runs already in the old folder are "
                          "still followed.")
        folder.clicked.connect(self._choose_folder)
        row.addWidget(folder)
        self.start = QPushButton("Start run ▶")
        self.start.setMinimumWidth(160)
        self.start.setToolTip("Checks every FED3 (clock, SD card, schedule), then starts "
                              "them all at the same moment. If any check fails, nothing "
                              "starts.")
        self.start.clicked.connect(self._start)
        row.addWidget(self.start)
        layout.addLayout(row)
        self.start_note = QLabel()
        self.start_note.setAlignment(Qt.AlignRight)
        self.start_note.setWordWrap(True)
        layout.addWidget(self.start_note)

        self._show_folder()
        if self.protocol_path and os.path.exists(self.protocol_path):
            with open(self.protocol_path, encoding="utf-8") as handle:
                self._set_protocol(handle.read(), self.protocol_path, startup=True)
        else:
            self._set_protocol("", "", startup=True)

    # --- protocol --------------------------------------------------------

    def _set_protocol(self, text, path, startup=False):
        self.protocol_text = text
        self.protocol_path = path
        self.settings.setValue("protocol_path", path)
        self.path_label.setText(os.path.basename(path) if path else
                                ("Unsaved protocol" if text else "No protocol chosen"))
        self.path_label.setToolTip(path)
        self.edit_button.setEnabled(bool(text))
        self.compiled, problems = proto.check(text) if text else (None, [])
        errors = [p for p in problems if p.level == "error"]
        if not text:
            self.verdict.setText("Open a protocol file, or write one with New…")
            self.verdict.setStyleSheet(f"color: {theme.MUTED};")
            self.timeline.setPlainText("")
        elif errors:
            self.verdict.setText(f"✗ {errors[0].format('line').replace('line:', 'line ')}"
                                 + (f"  (+{len(errors) - 1} more)" if len(errors) > 1 else "")
                                 + ". Edit… shows where.")
            self.verdict.setStyleSheet(f"color: {theme.BAD};")
            self.timeline.setPlainText("")
        else:
            c = self.compiled
            fnt = [p.name for p in c.phases if p.fnt]
            note = f" · needs FNT open to end: {', '.join(fnt)}" if fnt else ""
            self.verdict.setText(f"✓ {c.protocol.name} · {sch.plural(len(c.phases), 'phase')}"
                                 f" · {sch.plural(len(c.protocol.devices), 'device')}{note}")
            self.verdict.setStyleSheet(f"color: {theme.GOOD};")
            t0 = datetime.now(sch.standard_timezone()).replace(second=0, microsecond=0)
            self.timeline.setPlainText(proto.preview_text(c, t0))
        self.rows = None            # rebuild: what each number plays changed
        self.engine.submit("clear_preflight")
        self.update_view(self.snapshot)

    def _open(self):
        path, _ = QFileDialog.getOpenFileName(self, "Open protocol", self.protocol_path or "",
                                              "Protocol (*.yaml *.yml);;All files (*)")
        if path:
            with open(path, encoding="utf-8") as handle:
                self._set_protocol(handle.read(), path)

    def _new(self):
        self._edit(new=True)

    def _edit(self, new=False):
        text = "" if new else self.protocol_text
        if new or not text:
            from .editor import DEFAULT_TEMPLATE, TEMPLATES
            text = TEMPLATES[DEFAULT_TEMPLATE]
        editor = ProtocolEditor(text, self, path=None if new else self.protocol_path)
        if editor.exec_():
            path = editor.path
            if not path:
                path, _ = QFileDialog.getSaveFileName(self, "Save protocol",
                                                      self.protocol_path or "protocol.yaml",
                                                      "Protocol (*.yaml *.yml)")
            if path:
                with open(path, "w", encoding="utf-8") as handle:
                    handle.write(editor.text())
            self._set_protocol(editor.text(), path or "")

    # --- IDs ----------------------------------------------------------------

    def _used_ids(self):
        """The FED3 IDs the protocol names, as numbers."""
        return [int(name[3:]) for name in self.compiled.protocol.devices] if self.compiled else []

    def binding(self):
        """Protocol name -> FED3 ID: FED007 is the FED3 with ID 7."""
        return {fed_name(n): n for n in self._used_ids()}

    def _colour(self, number):
        roles = self._used_ids()
        return theme.SERIES[roles.index(number) % len(theme.SERIES)] if number in roles \
            else theme.MUTED

    def _connected(self):
        return [d for d in self.snapshot.get("devices", []) if d["connected"] and d["compatible"]]

    def _device(self, port):
        return next((d for d in self._connected() if d["port"] == port), None)

    def _renumber(self, port, box):
        """Enter (or leaving the box): give the FED3 on this row the typed ID."""
        device = self._device(port)
        if not device:
            return
        old = self._shown_id(device)
        number = parse_id(box.text())
        if number is None:
            box.setText(fed_name(old))
            box.setStyleSheet(box.styleSheet() + f" border: 1px solid {theme.BAD};")
            box.setToolTip(f"'{box.text()}' isn't an ID. Type a number from 1 to 999, "
                           "e.g. 7 or FED007.")
            self.id_note.setText("An ID is a number from 1 to 999, e.g. 7 or FED007.")
            return
        box.setText(fed_name(number))
        self.id_note.setText("")
        if number == old:
            return
        now = time.monotonic()
        # An ID another FED3 has: the engine swaps them, so no two share one
        # (their data would be mixed up). Show both at once, and say so.
        other = next((d for d in self._connected()
                      if d["port"] != port and self._shown_id(d) == number), None)
        self.pending[port] = (number, now)
        if other:
            self.pending[other["port"]] = (old, now)
            self.id_note.setText(f"{fed_name(number)} was on {other['port']}, so the two "
                                 f"swapped: {other['port']} is now {fed_name(old)}.")
        self.engine.submit("set_id", port=port, number=number)

    def _shown_id(self, device):
        """A FED3's number, or the one just chosen for it until it takes it."""
        number, when = self.pending.get(device["port"], (None, 0.0))
        if number is None or number == device["id"] or time.monotonic() - when > PENDING_FOR:
            self.pending.pop(device["port"], None)
            return device["id"]
        return number

    # --- the table ---------------------------------------------------------

    def _wanted_rows(self):
        """Connected FED3s in port order, and the IDs the protocol uses."""
        ports = tuple(sorted((d["port"] for d in self._connected()), key=port_order))
        return ports, tuple(self._used_ids())

    def _rebuild(self, wanted):
        ports, _ = wanted
        self.rows = wanted
        self.table.setRowCount(len(ports))
        for row, port in enumerate(ports):
            for column in (PORT, STATUS):
                self.table.setItem(row, column, QTableWidgetItem(""))
            self.table.item(row, PORT).setText(port)
            box = QLineEdit()
            box.setObjectName("id")
            box.setMaxLength(8)
            box.setFixedWidth(84)
            box.setAlignment(Qt.AlignCenter)
            box.setPlaceholderText("FED…")
            box.editingFinished.connect(lambda p=port, b=box: self._renumber(p, b))
            self.table.setCellWidget(row, ID, box)
            tests = QWidget()
            b = QHBoxLayout(tests)
            b.setContentsMargins(2, 0, 2, 0)
            lights = theme.small(QPushButton("Lights"))
            lights.setObjectName("lights")
            lights.setCheckable(True)
            lights.setToolTip("Light this FED3 up to find it; press again to turn them off. "
                              "They go dark when a run starts.")
            lights.toggled.connect(lambda on, p=port: self._lights(p, on))
            b.addWidget(lights)
            for text, tip, slot in (("Tone", "Play the protocol's reward tone", self._tone),
                                    ("Feed", "Dispense one pellet", self._feed)):
                button = theme.small(QPushButton(text))
                button.setToolTip(tip)
                button.clicked.connect(lambda _=False, p=port, s=slot: s(p))
                b.addWidget(button)
            self.table.setCellWidget(row, TESTS, tests)

    def _fill_row(self, row, port, preflight):
        device = self._device(port)
        number = self._shown_id(device) if device else None
        item = self.table.item(row, PORT)
        item.setToolTip(device["description"] if device else "")
        box = self.table.cellWidget(row, ID)
        if number and not box.hasFocus() and box.text() != fed_name(number):
            box.setText(fed_name(number))      # never under someone's typing
        used = number in self._used_ids()
        role = fed_name(number) if used else None
        style = (f"color: {self._colour(number)}; font-weight: bold;" if used
                 else f"color: {theme.MUTED};" if self._used_ids() else "")
        if box.styleSheet() != style and not box.hasFocus():
            box.setStyleSheet(style)
            box.setToolTip(("In this protocol. " if used else
                            "Not in this protocol, so it won't be used. " if self._used_ids()
                            else "")
                           + "Type another ID (e.g. 7 or FED007) and press Enter to "
                             "renumber this FED3; it's saved on its SD card.")
        if role and role in preflight and preflight[role] != "ok":
            text, colour = f"didn't pass pre-flight: {preflight[role]}", theme.BAD
        elif device:
            text, colour = device_health(device)
            if role and preflight.get(role) == "ok":
                text = f"pre-flight ok · {text}"
        else:
            text, colour = "not connected", theme.BAD
        if not role and self._used_ids() and colour == theme.GOOD:
            text, colour = f"not in this protocol · {text}", theme.MUTED
        status = self.table.item(row, STATUS)
        status.setText(text)
        status.setToolTip(text)
        status.setForeground(QColor(colour))
        tests = self.table.cellWidget(row, TESTS)
        lights = tests.findChild(QPushButton, "lights")
        lights.blockSignals(True)
        lights.setChecked(bool(device and device.get("lights")))
        lights.blockSignals(False)

    # --- bench tests -------------------------------------------------------

    def _lights(self, port, on):
        self.engine.submit("lights", device=port, on=on)

    def _feed(self, port):
        self.engine.submit("feed", device=port)

    def _tone(self, port):
        """The tone of the protocol's first reward cue, or a 4 kHz beep."""
        device = self._device(port)
        if not device:
            return
        cue, role = TEST_TONE, fed_name(device["id"])
        if self.compiled and role in self.compiled.schedules:
            reward = self.compiled.schedules[role].phases[0].rule.get("cue.reward", "")
            cue = next((a for a in reward.split("+") if a.startswith("tone:")), TEST_TONE)
        self.engine.submit("test_cue", device=port, cue=cue)

    # --- folder and start ------------------------------------------------

    def runs_root(self):
        return self.settings.value("runs_root", "") or self.engine.runs_root

    def _show_folder(self):
        self.folder_label.setText(f"Runs are saved in {self.runs_root()}")

    def _choose_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Save runs in", self.runs_root())
        if path and path != self.runs_root():
            known = [r for r in (self.settings.value("runs_roots", []) or []) if r]
            if isinstance(known, str):
                known = [known]
            old = self.runs_root()
            if old and old not in known:
                known.append(old)
            self.settings.setValue("runs_roots", known)
            self.settings.setValue("runs_root", path)
            self.engine.submit("set_runs_root", path=path)
            self._show_folder()

    def _bound_devices(self):
        """Connected FED3s the protocol names, by name."""
        roles = self._used_ids()
        return {fed_name(d["id"]): d for d in self._connected() if d["id"] in roles}

    def _start(self):
        if not self.compiled:
            return
        live = [(role, d) for role, d in self._bound_devices().items() if d.get("live_run")]
        replace = False
        if live:
            lines = "\n".join(f"  • {role} ({d['port']}): run {d['run_label']}"
                              for role, d in live)
            box = QMessageBox(QMessageBox.Warning, "End a run that's still going?",
                              "These FED3s are in a run that hasn't been stopped:\n\n"
                              f"{lines}\n\n"
                              "Starting ends that run on them. What they recorded stays on "
                              "their SD cards, but that run gets no more phases or data from "
                              "them.\n\nIf that run should keep going, cancel, and open the "
                              "folder it was saved in with Change folder… so FNT picks it "
                              "back up.", parent=self)
            end = box.addButton("End it and start", QMessageBox.DestructiveRole)
            box.addButton(QMessageBox.Cancel)
            box.setDefaultButton(QMessageBox.Cancel)
            box.exec_()
            if box.clickedButton() is not end:
                return
            replace = True
        os.makedirs(self.runs_root(), exist_ok=True)
        self.engine.submit("start", protocol_text=self.protocol_text, binding=self.binding(),
                           replace=replace)
        self.starting = time.monotonic()
        self.start.setEnabled(False)
        self.start_note.setText("Checking each FED3…")
        self.start_note.setStyleSheet(f"color: {theme.MUTED};")

    # --- refresh ---------------------------------------------------------

    def update_view(self, snapshot):
        self.snapshot = snapshot or {}
        connected = self._connected()
        wanted = self._wanted_rows()
        if wanted != self.rows:
            self._rebuild(wanted)
        preflight = self.snapshot.get("preflight", {})
        if self.starting and (preflight or time.monotonic() - self.starting > STARTING_FOR):
            self.starting = 0.0
        for row, port in enumerate(self.rows[0]):
            self._fill_row(row, port, preflight)

        names = ", ".join(fed_name(n) for n in self._used_ids())
        self.hint.setText((f"This protocol uses {names}. " if names else "")
                          + "One row per FED3 plugged in. To change an ID, type it in the "
                          "box and press Enter, or use the FED3's boot menu. Lights shows "
                          "which FED3 a row is.")
        problems = self._setup_problems(connected)   # IDs shared, even with no protocol
        self.missing.setText("\n".join(problems) if connected else "")
        self.missing.setStyleSheet(f"color: {theme.WARN};")
        recording = sorted({d["run_label"] for d in connected
                            if d.get("run_label") and not d.get("live_run")})
        old = [p["message"] + (f" ({p['device']})" if p["device"] else "")
               for p in self.snapshot.get("problems", [])
               if p["active"] and (p["kind"] == "firmware" or p["kind"].startswith("resume"))]
        text = []
        if old:
            text.append("⚠ " + "\n⚠ ".join(old))
        if not connected:
            text.append("No FED3 found yet. Plug one in; it appears here within a few seconds.")
        self.others.setText("\n".join(text))
        self.others.setStyleSheet(f"color: {theme.WARN};" if old else "")
        self.recording.setText(
            f"Still recording into the stopped run{'s' if len(recording) > 1 else ''} "
            f"{', '.join(recording)}. A FED3 stops when it starts a new run."
            if recording else "")

        ready, why = self._ready(connected)
        self.start.setEnabled(ready and not self.starting)
        if self.starting:
            return
        failed = [r for r, v in preflight.items() if v != "ok"]
        if not ready:
            note, colour = why, theme.MUTED
        elif failed:
            note, colour = (f"Nothing started: {', '.join(failed)} didn't pass pre-flight "
                            "(see its row). Fix it and press Start again."), theme.BAD
        else:
            note, colour = "Ready. Start checks every FED3, then starts them together.", \
                theme.MUTED
        self.start_note.setText(note)
        self.start_note.setStyleSheet(f"color: {colour};")

    def left_setup(self):
        """The run started (or another is showing): Start is done waiting."""
        self.starting = 0.0

    def stage_note(self):
        """A line for the step bar: what Setup is waiting for."""
        if not self.compiled:
            return "Choose a protocol"
        roles = self._used_ids()
        here = len(self._bound_devices())
        return (f"{self.compiled.protocol.name} · {here} of its {len(roles)} "
                f"FED3{'s' if len(roles) != 1 else ''} plugged in")

    def _setup_problems(self, devices):
        """What stops the FED3s plugged in from playing this protocol."""
        out = []
        roles = self._used_ids()
        bound = self._bound_devices()
        absent = [fed_name(n) for n in roles if fed_name(n) not in bound]
        if absent:
            out.append(f"{', '.join(absent)} {'isn' if len(absent) == 1 else 'aren'}'t plugged "
                       "in. Plug in, or give a FED3 that ID.")
        ids = {}
        for device in devices:
            ids.setdefault(device["id"], []).append(device["port"])
        for number, ports in sorted(ids.items()):
            if len(ports) > 1:
                ports = sorted(ports, key=port_order)
                who = (f"{ports[0]} and {ports[1]} are both" if len(ports) == 2
                       else f"{', '.join(ports[:-1])} and {ports[-1]} are all")
                out.append(f"{who} {fed_name(number)}: give "
                           f"{'one' if len(ports) == 2 else 'all but one'} another ID, so "
                           "their data can be told apart.")
        return out

    def _ready(self, devices):
        if not self.protocol_text:
            return False, "Choose a protocol to start."
        if not self.compiled:
            return False, "Fix the protocol's errors to start (Edit… shows where)."
        problems = self._setup_problems(devices)
        if problems:
            return False, ("Not ready: " + problems[0]) if len(problems) == 1 else \
                f"Not ready: {len(problems)} things to sort out, listed under the FED3s."
        return True, ""
