"""The log across the bottom of the window: what FNT did and noticed, newest last.

Each line is coloured by what kind of event it is, so a problem or a phase
change stands out from routine traffic. The panel collapses to its header,
which then counts what arrived while it was closed.
"""
import html

from PyQt5.QtCore import pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QPlainTextEdit, QPushButton, QVBoxLayout, QWidget

from . import theme

MAX_LINES = 2000

# Events read "<action>: <detail>", so the action at the start says what kind
# of event it is. First match wins: (colour, bold, actions).
KINDS = [
    (theme.BAD, True, ("problem", "engine error", "can't open", "not stopped",
                       "pre-flight failed")),
    (theme.GOOD, True, ("run started", "run stopped", "run resumed", "run ended")),
    (theme.WARN, False, ("disconnected", "boot", "schedule re-sent", "the current phase")),
    (theme.LOG_PHASE, False, ("phase",)),
    (theme.GOOD, False, ("connected", "pre-flight ok")),
    (theme.LOG_MANUAL, False, ("feed by hand", "test cue", "renumbered", "clock set",
                               "acknowledged", "edit applied")),
]
# Anywhere in a line these mean something went wrong, whatever the action.
FAILURE = ("failed", "ERR,", "error")


def event_style(text):
    """(colour, bold) for one event's text."""
    if not text.startswith("acknowledged") and any(word in text for word in FAILURE):
        return theme.BAD, False
    for colour, bold, actions in KINDS:
        if text.startswith(actions):
            return colour, bold
    return theme.TEXT, False


def event_html(event, zone):
    moment, device, text = event[:3]
    colour, bold = event_style(text)
    body = html.escape(text)
    if bold:
        body = f"<b>{body}</b>"
    who = f"<b>{html.escape(device)}</b>&nbsp; " if device else ""
    return (f"<span style='color:{theme.MUTED}'>{moment.astimezone(zone):%a %H:%M:%S}</span>"
            f"&nbsp;&nbsp;<span style='color:{colour}'>{who}{body}</span>")


class LogPanel(QWidget):
    toggled = pyqtSignal(bool)          # True when opened

    def __init__(self, parent=None):
        super().__init__(parent)
        self.seen = 0                   # the engine's event_count already shown
        self.unread = 0
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.header = QWidget()
        self.header.setObjectName("logheader")
        row = QHBoxLayout(self.header)
        row.setContentsMargins(6, 2, 6, 2)
        self.toggle = QPushButton()
        self.toggle.setObjectName("logtoggle")
        self.toggle.setCheckable(True)
        self.toggle.setChecked(True)
        self.toggle.setToolTip("Show or hide the log")
        self.toggle.toggled.connect(self._toggled)
        row.addWidget(self.toggle)
        self.badge = QLabel()
        self.badge.setObjectName("muted")
        row.addWidget(self.badge)
        row.addStretch()
        layout.addWidget(self.header)

        self.text = QPlainTextEdit()
        self.text.setObjectName("log")
        self.text.setReadOnly(True)
        self.text.setLineWrapMode(QPlainTextEdit.NoWrap)
        self.text.setMaximumBlockCount(MAX_LINES)
        self.text.setFont(QFont("Consolas", 9))
        layout.addWidget(self.text, 1)
        self._show_header()

    def is_open(self):
        return self.toggle.isChecked()

    def set_open(self, opened):
        self.toggle.setChecked(opened)

    def _toggled(self, opened):
        self.text.setVisible(opened)
        if opened:
            self.unread = 0
            self.setMaximumHeight(16777215)
        else:
            self.setMaximumHeight(self.header.sizeHint().height())
        self._show_header()
        self.toggled.emit(opened)

    def _show_header(self):
        self.toggle.setText(f"{'▼' if self.is_open() else '▲'}  Log")
        self.badge.setText("" if self.is_open() or not self.unread
                           else f"{self.unread} new")

    def update_log(self, snapshot):
        total = snapshot.get("event_count", 0)
        if total < self.seen:           # a different engine: start over
            self.seen = 0
            self.text.clear()
        new = total - self.seen
        if new <= 0:
            return
        events = snapshot.get("events", [])
        bar = self.text.verticalScrollBar()
        at_end = bar.value() >= bar.maximum() - 2
        for event in events[-min(new, len(events)):]:
            self.text.appendHtml(event_html(event, snapshot["zone"]))
        self.seen = total
        if at_end:
            bar.setValue(bar.maximum())
        if not self.is_open():
            self.unread += new
            self._show_header()

    def lines(self):
        return self.text.toPlainText().splitlines()
