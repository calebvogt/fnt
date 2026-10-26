"""A project's labelling guide: the rules its labels follow, written down once
and shown wherever the project is labelled.

The guide is ``LABELING_GUIDE.md`` in the project folder (Markdown, so it reads
on GitHub and in any editor too). MAD and Mask Tracker show a **Guide** button
in their Labeling Tools: hovering shows the guide, clicking opens it in a
window where it can also be edited and saved. A portable labelling session
carries a copy, so the rules travel with the frames.

Why a file and not a field of the project config: it is prose that people
edit, and an older FNT that rewrites the config with only the fields it knows
cannot drop it.
"""
from __future__ import annotations

import os
from typing import Optional

from PyQt5.QtCore import Qt
from PyQt5.QtGui import QTextDocument
from PyQt5.QtWidgets import (QDialog, QHBoxLayout, QLabel, QMessageBox, QPlainTextEdit,
                             QPushButton, QTextBrowser, QVBoxLayout)

GUIDE_FILENAME = "LABELING_GUIDE.md"

_TEMPLATE = """# Labelling guide

What counts, and how to label it. Everyone labelling this project follows these rules, so the model
learns one consistent thing.

## Label
-

## Do not label
-

## Edge cases
-
"""


def guide_path(project_dir: Optional[str]) -> Optional[str]:
    return os.path.join(project_dir, GUIDE_FILENAME) if project_dir else None


def read_guide(project_dir: Optional[str]) -> str:
    """The guide's Markdown, or '' if the project has none."""
    p = guide_path(project_dir)
    if p and os.path.isfile(p):
        with open(p, encoding="utf-8") as fh:
            return fh.read()
    return ""


def write_guide(project_dir: str, text: str) -> str:
    p = guide_path(project_dir)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8", newline="\n") as fh:
        fh.write(text if text.endswith("\n") else text + "\n")
    os.replace(tmp, p)
    return p


def markdown_to_html(text: str) -> str:
    doc = QTextDocument()
    doc.setMarkdown(text)
    return doc.toHtml()


class GuideDialog(QDialog):
    """The guide, rendered; Edit switches to its Markdown, Save writes it."""

    def __init__(self, project_dir: str, parent=None):
        super().__init__(None)
        self._dir = project_dir
        self._owner = parent
        self.setWindowTitle(f"Labelling guide — {os.path.basename(os.path.normpath(project_dir))}")
        self.setWindowFlags(Qt.Window | Qt.WindowCloseButtonHint)
        self.resize(640, 560)
        v = QVBoxLayout(self)
        self.view = QTextBrowser()
        self.view.setOpenExternalLinks(True)
        self.edit = QPlainTextEdit()
        self.edit.hide()
        v.addWidget(self.view, 1)
        v.addWidget(self.edit, 1)
        row = QHBoxLayout()
        self.lbl_path = QLabel(guide_path(project_dir))
        self.lbl_path.setStyleSheet("color:#999; font-size:10px;")
        self.lbl_path.setTextInteractionFlags(Qt.TextSelectableByMouse)
        row.addWidget(self.lbl_path, 1)
        self.btn_edit = QPushButton("Edit")
        self.btn_edit.clicked.connect(self._toggle_edit)
        row.addWidget(self.btn_edit)
        self.btn_save = QPushButton("Save")
        self.btn_save.clicked.connect(self._save)
        self.btn_save.hide()
        row.addWidget(self.btn_save)
        v.addLayout(row)
        self._load()

    def _load(self):
        text = read_guide(self._dir)
        if text:
            self.view.setMarkdown(text)
        else:
            self.view.setMarkdown("*This project has no labelling guide yet.* Press **Edit** to write one; "
                                  f"it is saved as `{GUIDE_FILENAME}` in the project folder.")
        self.edit.setPlainText(text or _TEMPLATE)

    def _toggle_edit(self):
        editing = self.edit.isHidden()
        self.edit.setVisible(editing)
        self.view.setVisible(not editing)
        self.btn_save.setVisible(editing)
        self.btn_edit.setText("Cancel" if editing else "Edit")
        if not editing:
            self._load()

    def _save(self):
        try:
            write_guide(self._dir, self.edit.toPlainText())
        except OSError as e:
            QMessageBox.warning(self, "Labelling guide", f"Could not save:\n\n{e}")
            return
        self._toggle_edit()
        if self._owner is not None and hasattr(self._owner, "refresh"):
            self._owner.refresh()


class GuideButton(QPushButton):
    """'Guide' in a Labeling Tools box: hover shows the project's guide,
    click opens it (and lets it be edited)."""

    def __init__(self, parent=None):
        super().__init__("Guide", parent)
        self.setFocusPolicy(Qt.NoFocus)
        self._dir: Optional[str] = None
        self._dialog: Optional[GuideDialog] = None
        self.clicked.connect(self.open_guide)
        self.set_project(None)

    def set_project(self, project_dir: Optional[str]):
        self._dir = project_dir
        self._dialog = None
        self.refresh()

    def refresh(self):
        text = read_guide(self._dir)
        self.setEnabled(bool(self._dir))
        self.setText("Guide" if text or not self._dir else "Guide (none)")
        if text:
            self.setToolTip(markdown_to_html(text))
        elif self._dir:
            self.setToolTip("This project has no labelling guide yet: click to write one\n"
                            f"({GUIDE_FILENAME} in the project folder). It shows here, on hover,\n"
                            "for everyone who labels the project.")
        else:
            self.setToolTip("Open a project to see its labelling guide.")

    def open_guide(self):
        if not self._dir:
            return
        if self._dialog is None:
            self._dialog = GuideDialog(self._dir, self)
        self._dialog.show()
        self._dialog.raise_()
        self._dialog.activateWindow()
