"""Setup: choose a protocol and a FED3 for each of its devices, then Start.

Nothing here talks to a device directly; it reads the engine's snapshot and
submits commands, like the rest of the window.
"""
import os
from datetime import datetime

from PyQt5.QtCore import Qt, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QComboBox, QFileDialog, QGroupBox, QHBoxLayout, QHeaderView, QInputDialog,
    QLabel, QMessageBox, QPlainTextEdit, QPushButton, QSizePolicy, QTableWidget,
    QTableWidgetItem, QVBoxLayout, QWidget,
)

from .. import protocol as proto
from .. import schedule as sch
from . import theme
from .editor import ProtocolEditor

NOT_CHOSEN = "— choose a FED3 —"


def device_label(device):
    return f"FED{device['id']:03d}  ({device['port']})"


def device_health(device):
    """One line: what's ready and what isn't."""
    if not device["connected"]:
        return "not connected", theme.BAD
    status = device["status"]
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
    elif status:
        parts.append("SD error")
        colour = theme.BAD
    parts.append(f"fw {device['firmware']}")
    if status.get("run") not in (None, "-") and status.get("ended") == "0":
        parts.append(f"in run {status['run']}")
        colour = theme.WARN
    return " · ".join(parts), colour


class SetupView(QWidget):
    settings_changed = pyqtSignal()

    def __init__(self, engine, settings, parent=None):
        super().__init__(parent)
        self.engine = engine
        self.settings = settings
        self.protocol_text = ""
        self.protocol_path = settings.value("protocol_path", "") or ""
        self.compiled = None
        self.snapshot = {}
        self._combo_items = {}

        layout = QVBoxLayout(self)

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
        g.addWidget(self.verdict)
        self.timeline = QPlainTextEdit()
        self.timeline.setReadOnly(True)
        self.timeline.setFixedHeight(190)
        self.timeline.setLineWrapMode(QPlainTextEdit.NoWrap)
        g.addWidget(self.timeline)
        group.setSizePolicy(group.sizePolicy().horizontalPolicy(), QSizePolicy.Maximum)
        layout.addWidget(group)

        # --- devices -------------------------------------------------------
        group = QGroupBox("Devices")
        g = QVBoxLayout(group)
        hint = QLabel("Choose which FED3 plays each device in the protocol. "
                      "Feed and Test check a device on the bench.")
        hint.setObjectName("muted")
        g.addWidget(hint)
        self.table = QTableWidget(0, 4)
        self.table.setHorizontalHeaderLabels(["Protocol device", "FED3", "Status", ""])
        self.table.verticalHeader().setVisible(False)
        header = self.table.horizontalHeader()
        for column in (0, 1, 3):
            header.setSectionResizeMode(column, QHeaderView.ResizeToContents)
        header.setSectionResizeMode(2, QHeaderView.Stretch)
        self.table.setSelectionMode(QTableWidget.NoSelection)
        g.addWidget(self.table)
        others_row = QHBoxLayout()
        self.others = QLabel()
        self.others.setWordWrap(True)
        others_row.addWidget(self.others, 1)
        renumber = theme.quiet(QPushButton("Renumber a FED3…"))
        renumber.clicked.connect(self._renumber)
        others_row.addWidget(renumber)
        g.addLayout(others_row)
        layout.addWidget(group)

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
            self.verdict.setText(f"✓ {c.protocol.name} · {len(c.phases)} phases{note}")
            self.verdict.setStyleSheet(f"color: {theme.GOOD};")
            t0 = datetime.now(sch.standard_timezone()).replace(second=0, microsecond=0)
            self.timeline.setPlainText(proto.preview_text(c, t0))
        self._rebuild_table()

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
            from .editor import TEMPLATES
            text = next(iter(TEMPLATES.values()))
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

    # --- devices ---------------------------------------------------------

    def _logical_devices(self):
        return self.compiled.protocol.devices if self.compiled else []

    def _rebuild_table(self):
        names = self._logical_devices()
        self.table.setRowCount(len(names))
        self._combo_items = {}
        saved = self.settings.value("binding", {}) or {}
        for row, name in enumerate(names):
            self.table.setItem(row, 0, QTableWidgetItem(name))
            combo = QComboBox()
            combo.addItem(NOT_CHOSEN, None)
            if name in saved:
                combo.addItem(f"FED{int(saved[name]):03d}  (not connected)", int(saved[name]))
                combo.setCurrentIndex(1)
            combo.currentIndexChanged.connect(self._binding_changed)
            self.table.setCellWidget(row, 1, combo)
            self.table.setItem(row, 2, QTableWidgetItem(""))
            bench = QWidget()
            b = QHBoxLayout(bench)
            b.setContentsMargins(0, 0, 0, 0)
            for text, slot in (("Feed", self._feed), ("Test cue", self._test)):
                button = theme.quiet(QPushButton(text))
                button.clicked.connect(lambda _=False, n=name, s=slot: s(n))
                b.addWidget(button)
            self.table.setCellWidget(row, 3, bench)
        self.update_view(self.snapshot)

    def binding(self):
        out = {}
        for row, name in enumerate(self._logical_devices()):
            combo = self.table.cellWidget(row, 1)
            if combo and combo.currentData() is not None:
                out[name] = combo.currentData()
        return out

    def _binding_changed(self):
        self.settings.setValue("binding", self.binding())
        self.update_view(self.snapshot)

    def _chosen_device(self, logical):
        device_id = self.binding().get(logical)
        for device in self.snapshot.get("devices", []):
            if device["connected"] and device["id"] == device_id:
                return device
        return None

    def _feed(self, logical):
        device = self._chosen_device(logical)
        if device:
            self.engine.submit("feed", device=device["port"])

    def _test(self, logical):
        device = self._chosen_device(logical)
        if not device:
            return
        cue = "tone:4000:200+flash:bar:0:0:30:0:500"
        if self.compiled:
            cue = self.compiled.schedules[logical].phases[0].rule["cue.reward"]
            if cue == "none":
                cue = "tone:4000:200+flash:bar:0:0:30:0:500"
        self.engine.submit("test_cue", device=device["port"], cue=cue)

    def _renumber(self):
        devices = [d for d in self.snapshot.get("devices", []) if d["connected"]]
        if not devices:
            QMessageBox.information(self, "Renumber", "No FED3 is connected.")
            return
        labels = [device_label(d) for d in devices]
        label, ok = QInputDialog.getItem(self, "Renumber a FED3", "FED3:", labels, 0, False)
        if not ok:
            return
        device = devices[labels.index(label)]
        number, ok = QInputDialog.getInt(self, "Renumber a FED3",
                                         f"New number for {label}:", device["id"], 1, 999)
        if ok:
            self.engine.submit("set_id", port=device["port"], number=number)

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
        devices = [d for d in self.snapshot.get("devices", []) if d["connected"] and d["compatible"]]
        binding = self.binding()
        preflight = self.snapshot.get("preflight", {})
        for row, name in enumerate(self._logical_devices()):
            combo = self.table.cellWidget(row, 1)
            if combo is None:
                continue
            wanted = [(device_label(d), d["id"]) for d in devices]
            current = combo.currentData()
            if current is not None and current not in [i for _, i in wanted]:
                wanted.append((f"FED{current:03d}  (not connected)", current))
            if self._combo_items.get(name) != wanted:
                combo.blockSignals(True)
                combo.clear()
                combo.addItem(NOT_CHOSEN, None)
                for text, device_id in wanted:
                    combo.addItem(text, device_id)
                index = combo.findData(current)
                combo.setCurrentIndex(index if index >= 0 else 0)
                combo.blockSignals(False)
                self._combo_items[name] = wanted
            device = next((d for d in devices if d["id"] == binding.get(name)), None)
            if name in preflight and preflight[name] != "ok":
                text, colour = f"pre-flight: {preflight[name]}", theme.BAD
            elif device:
                text, colour = device_health(device)
            elif binding.get(name) is not None:
                text, colour = "not connected", theme.BAD
            else:
                text, colour = "", theme.MUTED
            item = self.table.item(row, 2)
            item.setText(text)
            item.setToolTip(text)
            item.setForeground(QColor(colour))

        bound = set(binding.values())
        others = [device_label(d) for d in devices if d["id"] not in bound]
        old = [p["message"] + f" ({p['device']})" for p in self.snapshot.get("problems", [])
               if p["kind"] in ("firmware", "duplicate_id")]
        text = []
        if others:
            text.append("Also connected: " + ", ".join(others))
        if old:
            text.append("⚠ " + "; ".join(old))
        if not devices:
            text.append("No FED3 found yet. Plug one in; it appears here within a few seconds.")
        self.others.setText("\n".join(text))

        ready, why = self._ready(binding, devices)
        self.start.setEnabled(ready)
        if not ready:
            self.start_note.setText(why)
        elif preflight and all(v == "ok" for v in preflight.values()):
            self.start_note.setText("")
        elif not preflight:
            self.start_note.setText("Start checks every device, then starts them together.")

    def _ready(self, binding, devices):
        if not self.compiled:
            return False, "Choose a protocol that passes its checks."
        names = self._logical_devices()
        if len(binding) < len(names):
            return False, "Choose a FED3 for every protocol device."
        if len(set(binding.values())) < len(binding):
            return False, "Each protocol device needs its own FED3."
        connected = {d["id"] for d in devices}
        missing = [n for n, i in binding.items() if i not in connected]
        if missing:
            return False, f"Not connected: {', '.join(missing)}"
        return True, ""
