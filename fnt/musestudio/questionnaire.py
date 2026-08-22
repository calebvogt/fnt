"""Post-session questionnaire.

Short by design — it runs immediately after every probe, so it has to be fast
enough that you actually fill it in.

The items are chosen to be *interpretive*, not decorative. Drowsiness in
particular is not optional: sleep onset lowers alpha and raises theta, so a
session where you got sleepy can look exactly like a session where the stimulus
failed. Without that rating the EEG is ambiguous.

Saved as ``questionnaire.json`` in the recording root.
"""

import json
import os
from datetime import datetime

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QComboBox, QDialog, QGridLayout, QHBoxLayout, QLabel, QPlainTextEdit,
    QPushButton, QSlider, QVBoxLayout,
)

from fnt.musestudio import theme

# (key, question, low-anchor, high-anchor)
SCALES = [
    ("relaxation", "How relaxed did you feel overall?", "tense", "deeply relaxed"),
    ("drowsiness", "How sleepy did you get?  (be honest — this is a confound)",
     "fully alert", "nearly asleep"),
    ("absorption", "How absorbed / 'inward' did the session feel?",
     "distracted", "fully absorbed"),
    ("unusual", "Any unusual or altered quality to the state?",
     "ordinary", "distinctly altered"),
    ("discomfort", "Any physical discomfort or irritation from the sound?",
     "none", "a lot"),
]


class QuestionnaireDialog(QDialog):
    """Subjective ratings collected right after a probe."""

    def __init__(self, session_root, protocol_name="", block_labels=None,
                 parent=None):
        super().__init__(parent)
        self.session_root = session_root
        self.protocol_name = protocol_name
        self.setWindowTitle("Post-session questions")
        self.setStyleSheet(theme.STYLESHEET)
        self.resize(640, 620)

        root = QVBoxLayout(self)
        title = QLabel("How was that?")
        f = QFont()
        f.setPointSize(15)
        f.setBold(True)
        title.setFont(f)
        root.addWidget(title)
        sub = QLabel("Rough impressions are fine — answer quickly and move on.")
        sub.setStyleSheet(f"color: {theme.TEXT_DIM};")
        root.addWidget(sub)

        grid = QGridLayout()
        self.sliders = {}
        for r, (key, question, low, high) in enumerate(SCALES):
            q = QLabel(question)
            q.setWordWrap(True)
            grid.addWidget(q, r * 2, 0, 1, 3)
            slider = QSlider(Qt.Horizontal)
            slider.setRange(0, 10)
            slider.setValue(5)
            slider.setTickPosition(QSlider.TicksBelow)
            slider.setTickInterval(1)
            value = QLabel("5")
            slider.valueChanged.connect(lambda v, lbl=value: lbl.setText(str(v)))
            lo = QLabel(low)
            lo.setStyleSheet(f"color: {theme.TEXT_FAINT};")
            hi = QLabel(high)
            hi.setStyleSheet(f"color: {theme.TEXT_FAINT};")
            hi.setAlignment(Qt.AlignRight)
            row = QHBoxLayout()
            row.addWidget(lo)
            row.addWidget(slider, stretch=1)
            row.addWidget(hi)
            row.addWidget(value)
            grid.addLayout(row, r * 2 + 1, 0, 1, 3)
            self.sliders[key] = slider
        root.addLayout(grid)

        # Which block stood out — helps localize any effect in time.
        root.addWidget(QLabel("Which block felt most distinct?"))
        self.block_combo = QComboBox()
        self.block_combo.addItem("(not sure / none)", "")
        for label in (block_labels or []):
            self.block_combo.addItem(label, label)
        self.block_combo.setToolTip(
            "If one stretch felt different, which one? This tells me where in "
            "the timeline to look first.")
        root.addWidget(self.block_combo)

        root.addWidget(QLabel("Anything else worth noting?"))
        self.notes = QPlainTextEdit()
        self.notes.setPlaceholderText(
            "Interruptions, itching, a moment where something shifted, whether "
            "you could tell the tones apart…")
        self.notes.setMaximumHeight(90)
        root.addWidget(self.notes)

        buttons = QHBoxLayout()
        buttons.addStretch()
        skip = QPushButton("Skip")
        skip.clicked.connect(self.reject)
        buttons.addWidget(skip)
        save = QPushButton("Save")
        save.setProperty("accent", True)
        save.clicked.connect(self._save)
        buttons.addWidget(save)
        root.addLayout(buttons)

    def responses(self):
        return {
            "recorded": datetime.now().isoformat(timespec="seconds"),
            "protocol": self.protocol_name,
            "ratings": {k: s.value() for k, s in self.sliders.items()},
            "most_distinct_block": self.block_combo.currentData() or "",
            "notes": self.notes.toPlainText().strip(),
        }

    def _save(self):
        try:
            if self.session_root and os.path.isdir(self.session_root):
                path = os.path.join(self.session_root, "questionnaire.json")
                with open(path, "w", encoding="utf-8") as f:
                    json.dump(self.responses(), f, indent=2)
        except Exception:
            pass
        self.accept()
