"""Setup: choose a protocol, number the FED3s to match it, then Start.

Two columns: the protocol and its timeline on the left; on the right one row
per FED3 plugged in: its port, its ID (its number, chosen from a list), its
state, and bench tests (lights to find it, a tone, a pellet).

A FED3's number is its place in the protocol: FED001 plays the protocol's
first device, FED002 the second, and so on. To give a FED3 a part, give it
that number.

Nothing here talks to a device directly; it reads the engine's snapshot and
submits commands, like the rest of the window.
"""
import os
from datetime import datetime

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QHeaderView, QLabel,
    QPlainTextEdit, QPushButton, QSplitter, QTableWidget, QTableWidgetItem,
    QVBoxLayout, QWidget,
)

from .. import protocol as proto
from .. import schedule as sch
from . import theme
from .editor import ProtocolEditor

TEST_TONE = "tone:4000:200"
MIN_NUMBERS = 8             # the ID list always offers at least FED001-FED008
PORT, ID, STATUS, TESTS = range(4)


def fed_name(number):
    return f"FED{number:03d}"


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
    if status.get("run") not in (None, "-") and status.get("ended") == "0":
        parts.append(f"in run {status['run']}")
        colour = theme.WARN
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
        self.rows = []              # (port, number) shown, in order

        layout = QVBoxLayout(self)
        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)

        # --- protocol ------------------------------------------------------
        group = QGroupBox("Protocol")
        g = QVBoxLayout(group)
        row = QHBoxLayout()
        self.path_label = QLabel("No protocol chosen")
        row.addWidget(self.path_label, 1)
        for text, slot in (("Open…", self._open), ("New…", self._new), ("Edit…", self._edit)):
            button = theme.quiet(QPushButton(text))
            button.clicked.connect(slot)
            row.addWidget(button)
        g.addLayout(row)
        self.verdict = QLabel()
        self.verdict.setWordWrap(True)
        g.addWidget(self.verdict)
        self.timeline = QPlainTextEdit()
        self.timeline.setReadOnly(True)
        self.timeline.setLineWrapMode(QPlainTextEdit.NoWrap)
        g.addWidget(self.timeline, 1)
        split.addWidget(group)

        # --- devices -------------------------------------------------------
        group = QGroupBox("FED3s")
        g = QVBoxLayout(group)
        self.hint = QLabel()
        self.hint.setObjectName("muted")
        self.hint.setWordWrap(True)
        g.addWidget(self.hint)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Port", "ID", "Status", "Tests"])
        self.table.horizontalHeaderItem(ID).setToolTip(
            "The FED3's number, stored on its SD card. FED001 plays the protocol's first "
            "device, FED002 the second, and so on. Choosing a number another FED3 has "
            "swaps the two.")
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        for column in (PORT, ID, TESTS):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(STATUS, QHeaderView.Stretch)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        self.table.setEditTriggers(QTableWidget.NoEditTriggers)
        g.addWidget(self.table, 1)
        self.missing = QLabel()
        self.missing.setWordWrap(True)
        g.addWidget(self.missing)
        self.others = QLabel()
        self.others.setWordWrap(True)
        g.addWidget(self.others)
        split.addWidget(group)
        split.setStretchFactor(0, 2)
        split.setStretchFactor(1, 3)
        split.setSizes([480, 720])
        layout.addWidget(split, 1)

        # --- start -----------------------------------------------------------
        row = QHBoxLayout()
        self.folder_label = QLabel()
        row.addWidget(self.folder_label, 1)
        folder = theme.quiet(QPushButton("Change folder…"))
        folder.clicked.connect(self._choose_folder)
        row.addWidget(folder)
        self.start = QPushButton("Start run ▶")
        self.start.setMinimumWidth(160)
        self.start.clicked.connect(self._start)
        row.addWidget(self.start)
        layout.addLayout(row)
        self.start_note = QLabel()
        self.start_note.setObjectName("muted")
        self.start_note.setAlignment(Qt.AlignRight)
        layout.addWidget(self.start_note)

        self._show_folder()
        if self.protocol_path and os.path.exists(self.protocol_path):
            with open(self.protocol_path, encoding="utf-8") as handle:
                self._set_protocol(handle.read(), self.protocol_path)
        else:
            self._set_protocol("", "")

    # --- protocol --------------------------------------------------------

    def _set_protocol(self, text, path):
        self.protocol_text = text
        self.protocol_path = path
        self.settings.setValue("protocol_path", path)
        self.path_label.setText(os.path.basename(path) if path else
                                ("Unsaved protocol" if text else "No protocol chosen"))
        self.compiled, problems = proto.check(text) if text else (None, [])
        errors = [p for p in problems if p.level == "error"]
        if not text:
            self.verdict.setText("Open a protocol file, or make one with New…")
            self.verdict.setStyleSheet(f"color: {theme.MUTED};")
            self.timeline.setPlainText("")
        elif errors:
            self.verdict.setText(f"✗ {errors[0].format('line').replace('line:', 'line ')}"
                                 + (f"  (+{len(errors) - 1} more)" if len(errors) > 1 else ""))
            self.verdict.setStyleSheet(f"color: {theme.BAD};")
            self.timeline.setPlainText("")
        else:
            c = self.compiled
            fnt = [p.name for p in c.phases if p.fnt]
            note = f" · needs FNT to end: {', '.join(fnt)}" if fnt else ""
            self.verdict.setText(f"✓ {c.protocol.name} · {sch.plural(len(c.phases), 'phase')}"
                                 f" · {sch.plural(len(c.protocol.devices), 'device')}{note}")
            self.verdict.setStyleSheet(f"color: {theme.GOOD};")
            t0 = datetime.now(sch.standard_timezone()).replace(second=0, microsecond=0)
            self.timeline.setPlainText(proto.preview_text(c, t0))
        self.rows = None            # rebuild: what each number plays changed
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

    # --- numbers and the protocol's devices ----------------------------------

    def _roles(self):
        return list(self.compiled.protocol.devices) if self.compiled else []

    def _plays(self, number):
        """The protocol device a FED3 with this number plays, or None."""
        roles = self._roles()
        return roles[number - 1] if 1 <= number <= len(roles) else None

    def binding(self):
        """Protocol device -> FED3 number: the nth device is FED00n."""
        return {role: n for n, role in enumerate(self._roles(), start=1)}

    def _connected(self):
        return [d for d in self.snapshot.get("devices", []) if d["connected"] and d["compatible"]]

    def _device(self, port):
        return next((d for d in self._connected() if d["port"] == port), None)

    def _numbers(self):
        """The choices in every ID list."""
        taken = [d["id"] for d in self._connected()]
        top = max([MIN_NUMBERS, len(self._roles()), len(taken)] + taken)
        return list(range(1, top + 1))

    def _renumber(self, port, combo):
        device = self._device(port)
        number = combo.currentData()
        if not device or number is None or number == device["id"]:
            return
        # A number another FED3 already has: they swap.
        other = next((d for d in self._connected()
                      if d["id"] == number and d["port"] != port), None)
        self.engine.submit("set_id", port=port, number=number)
        if other:
            self.engine.submit("set_id", port=other["port"], number=device["id"])

    # --- the table ---------------------------------------------------------

    def _wanted_rows(self):
        """Connected FED3s in number order, and the numbers they can be given."""
        devices = sorted(self._connected(), key=lambda d: (d["id"], d["port"]))
        return [(d["port"], d["id"]) for d in devices], self._numbers(), self._roles()

    def _rebuild(self, wanted):
        rows, numbers, _ = wanted
        self.rows = wanted
        self.table.setRowCount(len(rows))
        for row, (port, number) in enumerate(rows):
            for column in (PORT, STATUS):
                self.table.setItem(row, column, QTableWidgetItem(""))
            combo = QComboBox()
            for n in numbers:
                role = self._plays(n)
                combo.addItem(f"{fed_name(n)}" + (f"  ({role})" if role else ""), n)
            combo.setCurrentIndex(max(combo.findData(number), 0))
            combo.currentIndexChanged.connect(lambda _=0, p=port, c=combo: self._renumber(p, c))
            self.table.setCellWidget(row, ID, combo)
            tests = QWidget()
            b = QHBoxLayout(tests)
            b.setContentsMargins(2, 0, 2, 0)
            lights = theme.small(QPushButton("Lights"))
            lights.setObjectName("lights")
            lights.setCheckable(True)
            lights.setToolTip("Light this FED3 up to find it. It goes dark again when a run "
                              "starts.")
            lights.toggled.connect(lambda on, p=port: self._lights(p, on))
            b.addWidget(lights)
            for text, tip, slot in (("Tone", "Play the protocol's reward tone", self._tone),
                                    ("Feed", "Dispense one pellet", self._feed)):
                button = theme.small(QPushButton(text))
                button.setToolTip(tip)
                button.clicked.connect(lambda _=False, p=port, s=slot: s(p))
                b.addWidget(button)
            self.table.setCellWidget(row, TESTS, tests)

    def _fill_row(self, row, port, number, preflight):
        device = self._device(port)
        item = self.table.item(row, PORT)
        item.setText(port)
        item.setToolTip(device["description"] if device else "")
        role = self._plays(number)
        if role and role in preflight and preflight[role] != "ok":
            text, colour = f"pre-flight: {preflight[role]}", theme.BAD
        elif device:
            text, colour = device_health(device)
        else:
            text, colour = "not connected", theme.BAD
        if role:
            text = f"plays {role} · {text}"
        elif self._roles():
            text = f"not in this protocol · {text}"
            colour = theme.MUTED if colour == theme.GOOD else colour
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
        cue, role = TEST_TONE, self._plays(device["id"])
        if self.compiled and role:
            reward = self.compiled.schedules[role].phases[0].rule["cue.reward"]
            cue = next((a for a in reward.split("+") if a.startswith("tone:")), TEST_TONE)
        self.engine.submit("test_cue", device=port, cue=cue)

    # --- folder and start ------------------------------------------------

    def runs_root(self):
        return self.settings.value("runs_root", "") or self.engine.runs_root

    def _show_folder(self):
        self.folder_label.setText(f"Runs are saved in {self.runs_root()}")

    def _choose_folder(self):
        path = QFileDialog.getExistingDirectory(self, "Save runs in", self.runs_root())
        if path:
            self.settings.setValue("runs_root", path)
            self.engine.runs_root = path
            self._show_folder()

    def _start(self):
        if not self.compiled:
            return
        os.makedirs(self.runs_root(), exist_ok=True)
        self.engine.runs_root = self.runs_root()
        self.engine.submit("start", protocol_text=self.protocol_text, binding=self.binding())
        self.start_note.setText("Checking each device…")

    # --- refresh ---------------------------------------------------------

    def update_view(self, snapshot):
        self.snapshot = snapshot or {}
        connected = self._connected()
        wanted = self._wanted_rows()
        if wanted != self.rows:
            self._rebuild(wanted)
        preflight = self.snapshot.get("preflight", {})
        for row, (port, number) in enumerate(self.rows[0]):
            self._fill_row(row, port, number, preflight)

        roles = self._roles()
        if roles:
            self.hint.setText("Every FED3 plugged in is listed. Its ID is its place in the "
                              "protocol: " + ", ".join(f"{fed_name(n)} plays {r}"
                                                      for n, r in enumerate(roles, start=1))
                              + ". Choose an ID to renumber a FED3. Lights shows which is which.")
        else:
            self.hint.setText("Every FED3 plugged in is listed. Choose an ID to renumber a "
                              "FED3. Lights shows which is which.")
        have = {d["id"] for d in connected}
        missing = [f"{fed_name(n)} ({r})" for n, r in enumerate(roles, start=1) if n not in have]
        if missing and connected:
            self.missing.setText(f"Not plugged in or not numbered yet: {', '.join(missing)}")
            self.missing.setStyleSheet(f"color: {theme.WARN};")
        else:
            self.missing.setText("")
        old = [p["message"] + f" ({p['device']})" for p in self.snapshot.get("problems", [])
               if p["kind"] in ("firmware", "duplicate_id")]
        text = []
        if old:
            text.append("⚠ " + "; ".join(old))
        if not connected:
            text.append("No FED3 found yet. Plug one in; it appears here within a few seconds.")
        self.others.setText("\n".join(text))

        ready, why = self._ready(connected)
        self.start.setEnabled(ready)
        if not ready:
            self.start_note.setText(why)
        elif preflight and all(v == "ok" for v in preflight.values()):
            self.start_note.setText("")
        elif not preflight:
            self.start_note.setText("Start checks every device, then starts them together.")

    def _ready(self, devices):
        if not self.compiled:
            return False, "Choose a protocol that passes its checks."
        have = [d["id"] for d in devices]
        missing = [f"{fed_name(n)} ({r})" for n, r in enumerate(self._roles(), start=1)
                   if n not in have]
        if missing:
            return False, f"Needs {', '.join(missing)}: give a FED3 that ID"
        if len(set(have)) < len(have):
            return False, "Two FED3s have the same ID; give one another."
        return True, ""
