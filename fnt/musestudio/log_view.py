"""Session log window — read and copy the diagnostic log."""

import os
from datetime import datetime

from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QApplication, QCheckBox, QDialog, QFileDialog, QHBoxLayout, QLabel,
    QLineEdit, QPlainTextEdit, QPushButton, QVBoxLayout,
)

from fnt.musestudio import theme
from fnt.musestudio.logbuffer import LOG


class LogDialog(QDialog):
    """Live view of the diagnostic log, with copy/save."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("MuseStudio — session logs")
        self.resize(920, 560)
        self.setStyleSheet(theme.STYLESHEET)

        root = QVBoxLayout(self)

        bar = QHBoxLayout()
        self.filter_edit = QLineEdit()
        self.filter_edit.setPlaceholderText("Filter… (e.g. error, lsl, OPTICS)")
        self.filter_edit.setToolTip(
            "Show only lines containing this text. Case-insensitive.\n"
            "Try 'error' for failures, or a stream name to trace one stream.")
        self.filter_edit.textChanged.connect(self._refresh)
        bar.addWidget(self.filter_edit, stretch=1)

        self.errors_only = QCheckBox("Errors only")
        self.errors_only.setToolTip(
            "Show only lines that look like failures (error, exception, "
            "traceback, warning, failed).")
        self.errors_only.toggled.connect(self._refresh)
        bar.addWidget(self.errors_only)

        self.follow = QCheckBox("Follow")
        self.follow.setChecked(True)
        self.follow.setToolTip("Keep scrolling to the newest line as logs arrive.")
        bar.addWidget(self.follow)
        root.addLayout(bar)

        self.view = QPlainTextEdit()
        self.view.setReadOnly(True)
        self.view.setLineWrapMode(QPlainTextEdit.NoWrap)
        font = QFont("Menlo")
        font.setStyleHint(QFont.Monospace)
        font.setPointSize(11)
        self.view.setFont(font)
        root.addWidget(self.view, stretch=1)

        self.status = QLabel("")
        self.status.setStyleSheet(f"color: {theme.TEXT_DIM};")
        root.addWidget(self.status)

        buttons = QHBoxLayout()
        copy_btn = QPushButton("Copy report to clipboard")
        copy_btn.setProperty("accent", True)
        copy_btn.setToolTip(
            "Copy the full log plus a header of platform and package versions — "
            "this is what to paste when reporting a problem.")
        copy_btn.clicked.connect(self.copy_report)
        buttons.addWidget(copy_btn)

        save_btn = QPushButton("Save to file…")
        save_btn.setToolTip("Write the same report to a .txt file.")
        save_btn.clicked.connect(self.save_report)
        buttons.addWidget(save_btn)

        clear_btn = QPushButton("Clear")
        clear_btn.setToolTip("Empty the buffer. Past lines cannot be recovered.")
        clear_btn.clicked.connect(self._clear)
        buttons.addWidget(clear_btn)

        buttons.addStretch()
        close_btn = QPushButton("Close")
        close_btn.clicked.connect(self.accept)
        buttons.addWidget(close_btn)
        root.addLayout(buttons)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(700)
        self._refresh()

    def _visible_lines(self):
        lines = LOG.lines()
        needle = self.filter_edit.text().strip().lower()
        if needle:
            lines = [ln for ln in lines if needle in ln.lower()]
        if self.errors_only.isChecked():
            keys = ("error", "err|", "exception", "traceback", "warn", "failed",
                    "critical")
            lines = [ln for ln in lines if any(k in ln.lower() for k in keys)]
        return lines

    def _refresh(self):
        lines = self._visible_lines()
        scrollbar = self.view.verticalScrollBar()
        at_end = scrollbar.value() >= scrollbar.maximum() - 4
        text = "\n".join(lines)
        if text != self.view.toPlainText():
            self.view.setPlainText(text)
            if self.follow.isChecked() or at_end:
                self.view.verticalScrollBar().setValue(
                    self.view.verticalScrollBar().maximum())
        self.status.setText(f"{len(lines)} shown · {LOG.count()} captured")

    def _clear(self):
        LOG.clear()
        self._refresh()

    def copy_report(self):
        QApplication.clipboard().setText(LOG.report())
        self.status.setText("Report copied to clipboard.")

    def save_report(self):
        default = os.path.join(
            os.path.expanduser("~"),
            f"musestudio_log_{datetime.now():%Y-%m-%d_%H%M%S}.txt")
        path, _ = QFileDialog.getSaveFileName(self, "Save log", default,
                                              "Text files (*.txt)")
        if path:
            with open(path, "w", encoding="utf-8") as f:
                f.write(LOG.report())
            self.status.setText(f"Saved to {path}")
