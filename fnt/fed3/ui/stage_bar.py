"""Where you are in a run: Setup › Running › Stopped, across the top of the window.

The current step is lit, the ones behind it are ticked, and the current one
carries a short note (what Setup is waiting for, how long the run has gone).
"""
from PyQt5.QtWidgets import QHBoxLayout, QLabel, QWidget

from . import theme

STEPS = [("setup", "Setup"), ("running", "Running"), ("stopped", "Stopped")]


class StageBar(QWidget):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setObjectName("stagebar")
        row = QHBoxLayout(self)
        row.setContentsMargins(10, 4, 10, 4)
        row.setSpacing(6)
        self.steps = {}
        for n, (key, name) in enumerate(STEPS):
            if n:
                arrow = QLabel("›")
                arrow.setObjectName("muted")
                row.addWidget(arrow)
            label = QLabel()
            self.steps[key] = (n, name, label)
            row.addWidget(label)
        self.note = QLabel()
        self.note.setObjectName("muted")
        row.addSpacing(12)
        row.addWidget(self.note, 1)
        self.state = None
        self.set_stage("setup")

    def set_stage(self, state, note=""):
        if state != self.state:
            self.state = state
            current = self.steps[state][0]
            for n, name, label in self.steps.values():
                if n < current:
                    text, colour, weight = f"✓ {name}", theme.GOOD, "normal"
                elif n == current:
                    text, colour, weight = f"{n + 1}  {name}", theme.TEXT, "bold"
                else:
                    text, colour, weight = f"{n + 1}  {name}", theme.MUTED, "normal"
                label.setText(text)
                style = f"color: {colour}; font-weight: {weight}; padding: 2px 8px;"
                if n == current:
                    style += (f"background-color: {theme.ACCENT if state != 'stopped' else theme.BORDER};"
                              " border-radius: 9px; color: white;")
                label.setStyleSheet(style)
        self.note.setText(note)
