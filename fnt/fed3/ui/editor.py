"""The protocol editor: the YAML file, checked as you type, with its timeline.

The file is the protocol, so the editor edits the file. A form-based
"Add phase" covers the common case without typing YAML; everything else
(rotation, daily windows, named rules) is written in the text, with problems
reported against their line as you go.
"""
from datetime import datetime

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont, QTextCursor
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QDialogButtonBox, QFileDialog, QFormLayout,
    QHBoxLayout, QLabel, QLineEdit, QListWidget, QListWidgetItem, QMenu,
    QPlainTextEdit, QPushButton, QSpinBox, QSplitter, QVBoxLayout, QWidget,
)

from .. import protocol as proto
from .. import schedule as sch
from . import theme

TEMPLATES = {
    "FR1, both pokes, with a cue": """\
format: fnt-fed3-protocol/1
name: FR1 both pokes
devices: [FED]
phases:
  - name: FR1
    all:
      active: both
      cue:
        reward:
          - tone: {hz: 4000, ms: 200}
          - flash: {target: poke, color: blue, ms: 500}
""",
    "Step up the ratio by rewards": """\
format: fnt-fed3-protocol/1
name: FR1 to FR5, both pokes
devices: [FED]

cues:
  reward:
    - tone: {hz: 4000, ms: 200}
    - flash: {target: poke, color: blue, ms: 500}

rules:
  base: {active: both, cue: {reward: reward}}

phases:
  - name: FR1
    until: {rewards: 50}
    all: {use: base, ratio: 1}
  - name: FR3
    until: {rewards: 100}
    all: {use: base, ratio: 3}
  - name: FR5
    all: {use: base, ratio: 5}
""",
    "Timed feeding": """\
format: fnt-fed3-protocol/1
name: Timed feeding 19:00-21:00
devices: [FED]
phases:
  - name: Restricted
    all: {active: none}
    daily:
      - {from: "19:00", to: "21:00", all: {free_feed: 1}}
""",
    "Four-patch competition": """\
format: fnt-fed3-protocol/1
name: Four-patch competition
description: >
  Two rich and two poor patches that each refill hourly. Which patches are
  rich is reshuffled every day at lights-off.
devices: [P1, P2, P3, P4]

cues:
  reward:
    - tone: {hz: 4000, ms: 200}
    - flash: {target: bar, color: green, ms: 1000}

rules:
  patch:
    active: both
    cue: {reward: reward, depleted: [click]}
  rich: {use: patch, ratio: 1, quota: 30, quota_period: 1h}
  poor: {use: patch, ratio: 5, quota: 10, quota_period: 1h}

phases:
  - name: Habituation
    until: {at: "19:00", day: 3}
    all: {active: none, free_feed: 1}

  - name: Training
    until: [{rewards: 100, devices: all}, {after: 4d}]
    all: patch

  - name: Competition
    until: {after: 14d}
    rotate:
      every: 1d
      at: "19:00"
      mode: shuffle
      assign: {P1: rich, P2: rich, P3: poor, P4: poor}

  - name: Recovery
    all: {active: none, free_feed: 1}

checks:
  no_pellets: 8h
""",
}


class AddPhaseDialog(QDialog):
    """The common settings of one phase, written out as YAML."""

    ENDS = ["after a time", "at a clock time", "after N rewards", "after N pokes",
            "when every device reaches N rewards", "when Advance phase is pressed",
            "never (the last phase)"]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Add a phase")
        form = QFormLayout(self)
        self.name = QLineEdit("New phase")
        form.addRow("Name", self.name)

        self.ends = QComboBox()
        self.ends.addItems(self.ENDS)
        self.amount = QLineEdit("24h")
        ends_row = QHBoxLayout()
        ends_row.addWidget(self.ends)
        ends_row.addWidget(self.amount)
        form.addRow("Ends", ends_row)
        self.ends.currentIndexChanged.connect(self._ends_changed)

        self.active = QComboBox()
        self.active.addItems(["both", "left", "right", "none"])
        form.addRow("Active poke", self.active)
        self.kind = QComboBox()
        self.kind.addItems(["fixed", "random", "progressive"])
        self.ratio = QSpinBox()
        self.ratio.setRange(1, 999)
        ratio_row = QHBoxLayout()
        ratio_row.addWidget(self.kind)
        ratio_row.addWidget(QLabel("ratio"))
        ratio_row.addWidget(self.ratio)
        form.addRow("Ratio", ratio_row)
        self.timeout = QSpinBox()
        self.timeout.setRange(0, 3600)
        self.timeout.setSuffix(" s")
        form.addRow("Timeout after a reward", self.timeout)
        self.quota = QSpinBox()
        self.quota.setRange(0, 9999)
        self.quota.setSpecialValueText("unlimited")
        self.quota_period = QLineEdit("1h")
        quota_row = QHBoxLayout()
        quota_row.addWidget(self.quota)
        quota_row.addWidget(QLabel("rewards per"))
        quota_row.addWidget(self.quota_period)
        form.addRow("Quota", quota_row)
        self.free = QCheckBox("keep a pellet in the well")
        form.addRow("Free feeding", self.free)

        self.tone = QCheckBox("tone")
        self.tone_hz = QSpinBox()
        self.tone_hz.setRange(20, 20000)
        self.tone_hz.setValue(4000)
        self.tone_hz.setSuffix(" Hz")
        self.flash = QCheckBox("flash")
        self.flash_target = QComboBox()
        self.flash_target.addItems(["poke", "bar", "pokes"])
        self.flash_color = QComboBox()
        self.flash_color.addItems(["blue", "green", "red", "white"])
        cue_row = QHBoxLayout()
        for widget in (self.tone, self.tone_hz, self.flash, self.flash_target, self.flash_color):
            cue_row.addWidget(widget)
        form.addRow("On a reward", cue_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        form.addRow(buttons)
        self._ends_changed(0)

    def _ends_changed(self, index):
        defaults = ["24h", "19:00", "100", "500", "100", "", ""]
        self.amount.setText(defaults[index])
        self.amount.setEnabled(index < 5)

    def yaml(self):
        lines = [f"  - name: {self.name.text().strip() or 'New phase'}"]
        amount = self.amount.text().strip()
        until = {
            0: f"{{after: {amount}}}",
            1: f'{{at: "{amount}"}}',
            2: f"{{rewards: {amount}}}",
            3: f"{{pokes: {amount}}}",
            4: f"[{{rewards: {amount}, devices: all}}, {{after: 7d}}]",
            5: "{manual: true}",
        }.get(self.ends.currentIndex())
        if until:
            lines.append(f"    until: {until}")
        rule = [f"active: {self.active.currentText()}"]
        if self.kind.currentText() != "fixed":
            rule.append(f"ratio_kind: {self.kind.currentText()}")
        if self.ratio.value() != 1 and self.kind.currentText() != "progressive":
            rule.append(f"ratio: {self.ratio.value()}")
        if self.timeout.value():
            rule.append(f"timeout: {self.timeout.value()}s")
        if self.quota.value():
            rule.append(f"quota: {self.quota.value()}")
            rule.append(f"quota_period: {self.quota_period.text().strip() or '1h'}")
        if self.free.isChecked():
            rule.append("free_feed: 1")
        actions = []
        if self.tone.isChecked():
            actions.append(f"{{tone: {{hz: {self.tone_hz.value()}, ms: 200}}}}")
        if self.flash.isChecked():
            actions.append(f"{{flash: {{target: {self.flash_target.currentText()}, "
                           f"color: {self.flash_color.currentText()}, ms: 500}}}}")
        if actions:
            rule.append(f"cue: {{reward: [{', '.join(actions)}]}}")
        lines.append(f"    all: {{{', '.join(rule)}}}")
        return "\n".join(lines) + "\n"


class ProtocolEditor(QDialog):
    """Edit a protocol. ``running`` locks it to edits a running run accepts."""

    def __init__(self, text="", parent=None, path=None, running=False, t0=None):
        super().__init__(parent)
        self.setWindowTitle("Edit protocol" if not running else "Edit the running protocol")
        self.resize(1200, 760)
        self.path = path
        self.running = running
        self.t0 = t0
        self.compiled = None
        self.problems = []

        layout = QVBoxLayout(self)
        tools = QHBoxLayout()
        if not running:
            templates = theme.quiet(QPushButton("New from template ▾"))
            menu = QMenu(templates)
            for name in TEMPLATES:
                menu.addAction(name, lambda n=name: self.editor.setPlainText(TEMPLATES[n]))
            templates.setMenu(menu)
            tools.addWidget(templates)
        add = theme.quiet(QPushButton("Add phase…"))
        add.clicked.connect(self._add_phase)
        tools.addWidget(add)
        tools.addStretch()
        if running:
            note = QLabel("Phases that are over can't change. The current phase can, and "
                          "the change applies at once.")
            note.setObjectName("muted")
            tools.addWidget(note)
        layout.addLayout(tools)

        split = QSplitter(Qt.Horizontal)
        self.editor = QPlainTextEdit()
        self.editor.setFont(QFont("Consolas", 10))
        self.editor.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.editor.setPlainText(text)
        split.addWidget(self.editor)

        side = QWidget()
        side_layout = QVBoxLayout(side)
        side_layout.setContentsMargins(0, 0, 0, 0)
        self.verdict = QLabel()
        side_layout.addWidget(self.verdict)
        self.problem_list = QListWidget()
        self.problem_list.itemClicked.connect(self._jump)
        self.problem_list.setMaximumHeight(140)
        side_layout.addWidget(self.problem_list)
        timeline = QLabel("Timeline")
        timeline.setObjectName("muted")
        side_layout.addWidget(timeline)
        self.outline = QPlainTextEdit()
        self.outline.setReadOnly(True)
        self.outline.setFont(QFont("Consolas", 9))
        self.outline.setLineWrapMode(QPlainTextEdit.NoWrap)
        side_layout.addWidget(self.outline)
        split.addWidget(side)
        split.setSizes([620, 580])
        layout.addWidget(split)

        buttons = QHBoxLayout()
        if not running:
            save_as = theme.quiet(QPushButton("Save as…"))
            save_as.clicked.connect(self._save_as)
            buttons.addWidget(save_as)
        buttons.addStretch()
        cancel = theme.quiet(QPushButton("Cancel"))
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        self.ok = QPushButton("Apply to running devices" if running else "Use this protocol")
        self.ok.clicked.connect(self._accept)
        buttons.addWidget(self.ok)
        layout.addLayout(buttons)

        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.setInterval(300)
        self.timer.timeout.connect(self._check)
        self.editor.textChanged.connect(self.timer.start)
        self._check()

    def text(self):
        return self.editor.toPlainText()

    def _check(self):
        self.compiled, self.problems = proto.check(self.text())
        errors = [p for p in self.problems if p.level == "error"]
        self.problem_list.clear()
        for problem in self.problems:
            where = f"line {problem.line}: " if problem.line else ""
            item = QListWidgetItem(f"{where}{problem.level}: {problem.message}")
            item.setData(Qt.UserRole, problem.line)
            item.setForeground(Qt.red if problem.level == "error" else Qt.yellow)
            self.problem_list.addItem(item)
        if errors:
            self.verdict.setText(f"✗ {len(errors)} problem{'s' if len(errors) > 1 else ''}")
            self.verdict.setStyleSheet(f"color: {theme.BAD}; font-weight: bold;")
            self.outline.setPlainText("")
        else:
            c = self.compiled
            self.verdict.setText(f"✓ checks pass · {len(c.phases)} phases · "
                                 f"{len(c.protocol.devices)} device{'s' if len(c.protocol.devices) > 1 else ''}")
            self.verdict.setStyleSheet(f"color: {theme.GOOD}; font-weight: bold;")
            t0 = self.t0 or datetime.now(sch.standard_timezone()).replace(second=0, microsecond=0)
            self.outline.setPlainText(proto.preview_text(c, t0))
        self.ok.setEnabled(not errors)

    def _jump(self, item):
        line = item.data(Qt.UserRole)
        if not line:
            return
        block = self.editor.document().findBlockByLineNumber(line - 1)
        cursor = QTextCursor(block)
        self.editor.setTextCursor(cursor)
        self.editor.setFocus()

    def _add_phase(self):
        dialog = AddPhaseDialog(self)
        if dialog.exec_() != QDialog.Accepted:
            return
        text = self.text()
        snippet = dialog.yaml()
        lines = text.splitlines(keepends=True)
        # Phases end where the next top-level key starts (often checks:).
        in_phases, insert_at = False, len(lines)
        for index, line in enumerate(lines):
            if line.startswith("phases:"):
                in_phases = True
            elif in_phases and line[:1].isalpha():
                insert_at = index
                break
        if lines and not lines[-1].endswith("\n"):
            lines[-1] += "\n"
        lines.insert(insert_at, snippet)
        self.editor.setPlainText("".join(lines))

    def _save_as(self):
        path, _ = QFileDialog.getSaveFileName(self, "Save protocol", self.path or "",
                                              "Protocol (*.yaml *.yml)")
        if path:
            with open(path, "w", encoding="utf-8") as handle:
                handle.write(self.text())
            self.path = path

    def _accept(self):
        self._check()
        if self.ok.isEnabled():
            self.accept()


def format_delta(seconds):
    seconds = max(int(seconds), 0)
    days, rem = divmod(seconds, 86400)
    hours, rem = divmod(rem, 3600)
    minutes = rem // 60
    if days:
        return f"{days}d {hours:02d}h"
    if hours:
        return f"{hours}h {minutes:02d}m"
    return f"{minutes}m"
