"""Reusable widgets and dialogs for the FED3 tab.

Extracted from ``fed_widgets.py``, which had grown to hold a custom layout, a
custom-painted device view, a log console and four dialogs alongside the tab
logic itself.
"""

import os
import sys
from datetime import datetime

from PyQt5.QtCore import QRect, QRectF, QSize, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QBrush, QColor, QFont, QPainter
from PyQt5.QtWidgets import (
    QCheckBox, QDialog, QDialogButtonBox, QHBoxLayout, QLabel, QLayout,
    QLineEdit, QPushButton, QScrollArea, QSizePolicy, QTextEdit, QVBoxLayout,
    QWidget,
)

try:
    from PyQt5.QtSvg import QSvgRenderer
except ImportError:
    QSvgRenderer = None


class FlowLayout(QLayout):
    """Left-to-right layout that wraps to the next row when it runs out of width."""

    def __init__(self, parent=None, margin=-1, spacing=-1):
        super().__init__(parent)
        if margin is not None:
            self.setContentsMargins(margin, margin, margin, margin)
        self.setSpacing(spacing)
        self._items = []

    def __del__(self):
        while self.takeAt(0):
            pass

    def addItem(self, item):
        self._items.append(item)

    def removeWidget(self, widget):
        for i in reversed(range(len(self._items))):
            if self._items[i].widget() is widget:
                self.takeAt(i)
                break
        super().removeWidget(widget)

    def count(self):
        return len(self._items)

    def itemAt(self, index):
        return self._items[index] if 0 <= index < len(self._items) else None

    def takeAt(self, index):
        return self._items.pop(index) if 0 <= index < len(self._items) else None

    def expandingDirections(self):
        return Qt.Orientations(Qt.Orientation(0))

    def hasHeightForWidth(self):
        return True

    def heightForWidth(self, width):
        return self._do_layout(QRect(0, 0, width, 0), test_only=True)

    def setGeometry(self, rect):
        super().setGeometry(rect)
        self._do_layout(rect, test_only=False)

    def sizeHint(self):
        return self.minimumSize()

    def minimumSize(self):
        size = QSize()
        for item in self._items:
            size = size.expandedTo(item.minimumSize())
        left, top, right, bottom = self.getContentsMargins()
        return size + QSize(left + right, top + bottom)

    def _do_layout(self, rect, test_only):
        """Place items in rows, then share each row's leftover width among them.

        Placing bare size hints left a ragged right margin — with two device
        cards on a row, most of a third card's width sat empty. Items on a row
        also take a common height, so cards keep a shared baseline whether or
        not one of them has its setup panel open.
        """
        spacing = self.spacing()
        rows, row, used = [], [], 0

        for item in self._items:
            hint = item.sizeHint()
            needed = hint.width() + (spacing if row else 0)
            if row and used + needed > rect.width():
                rows.append((row, used))
                row, used, needed = [], 0, hint.width()
            row.append((item, hint))
            used += needed
        if row:
            rows.append((row, used))

        y = rect.y()
        for row, used in rows:
            height = max(hint.height() for _, hint in row)
            share = max((rect.width() - used) // len(row), 0)
            x = rect.x()
            for item, hint in row:
                width = hint.width() + share
                if not test_only:
                    # Each item keeps its own height so the row shares a top
                    # edge. Handed the row height instead, QWidgetItem centres
                    # a widget whose vertical policy is Fixed — which floated a
                    # collapsed card half-way down beside an expanded one.
                    item.setGeometry(QRect(x, y, width, hint.height()))
                x += width + spacing
            y += height + spacing

        return (y - spacing - rect.y()) if rows else 0


class FEDSvgView(QWidget):
    """The FED3 illustration with live poke and pellet counters painted on top.

    Counters are painted directly rather than being child widgets, which is what
    caused the opaque corners the child-label version had.
    """

    SVG_ASPECT = 163.67577 / 116.04688
    FLASH_MS = 200

    def __init__(self, parent=None):
        super().__init__(parent)
        self._renderer = self._load_renderer()
        self.counts = {"left": 0, "right": 0, "pellet": 0}
        self.is_tracking = False
        self.is_stale = False           # connection lost: dim the artwork
        self._flash = {}                # counter name -> QColor

    @staticmethod
    def _load_renderer():
        if QSvgRenderer is None:
            return None
        try:
            base = sys._MEIPASS      # PyInstaller bundle
            path = os.path.join(base, "fnt", "fed3", "fed3_image.svg")
        except AttributeError:
            path = os.path.join(os.path.dirname(__file__), "fed3_image.svg")
        return QSvgRenderer(path)

    def set_counts(self, counts):
        self.counts = dict(counts)
        self.update()

    def flash(self, counter_name):
        self._flash[counter_name] = QColor(76, 175, 80, 190)
        self.update()
        QTimer.singleShot(self.FLASH_MS, lambda: self._end_flash(counter_name))

    def _end_flash(self, counter_name):
        self._flash.pop(counter_name, None)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.Antialiasing)
        if self.is_stale:
            painter.setOpacity(0.45)

        width, height = self.width(), self.height()
        aspect = width / height if height else 1
        if aspect > self.SVG_ASPECT:
            draw_h, draw_w = height, int(height * self.SVG_ASPECT)
        else:
            draw_w, draw_h = width, int(width / self.SVG_ASPECT)
        x_off, y_off = (width - draw_w) // 2, (height - draw_h) // 2

        if self._renderer is not None and self._renderer.isValid():
            self._renderer.render(painter, QRectF(x_off, y_off, draw_w, draw_h))

        poke_radius = int(draw_w * 0.0953)
        font_size = max(10, int(poke_radius * 0.65))

        # Fractional positions of the pokes and pellet well within the artwork.
        self._draw_overlay(painter, x_off, y_off, font_size,
                           int(draw_w * 0.2501) - poke_radius,
                           int(draw_h * 0.6444) - poke_radius,
                           poke_radius * 2, poke_radius * 2, True, "left")
        self._draw_overlay(painter, x_off, y_off, font_size,
                           int(draw_w * 0.7488) - poke_radius,
                           int(draw_h * 0.6426) - poke_radius,
                           poke_radius * 2, poke_radius * 2, True, "right")
        self._draw_overlay(painter, x_off, y_off, font_size,
                           int(draw_w * 0.4108), int(draw_h * 0.6095),
                           int(draw_w * 0.1772), int(draw_h * 0.1677),
                           False, "pellet")
        painter.end()

    def _draw_overlay(self, painter, x_off, y_off, font_size,
                      x, y, w, h, is_circle, key):
        rect = QRect(x_off + x, y_off + y, w, h)
        painter.setPen(Qt.NoPen)
        painter.setBrush(QBrush(self._flash.get(key, QColor(0, 0, 0, 150))))
        if is_circle:
            painter.drawEllipse(rect)
        else:
            painter.drawRoundedRect(rect, 4, 4)

        if self.is_tracking:
            painter.setPen(QColor(255, 255, 255))
            painter.setFont(QFont("Arial", font_size, QFont.Bold))
            painter.drawText(rect, Qt.AlignCenter, str(self.counts.get(key, 0)))

    def sizeHint(self):
        return QSize(350, 200)

    def minimumSizeHint(self):
        return QSize(250, 150)


class CollapsibleSection(QWidget):
    """A titled disclosure whose header carries the state of what it hides.

    Setup controls and the scheduler are needed on the day an experiment is
    configured and almost never again, but they were holding the top of a window
    that stays open for weeks. Collapsing them is only an improvement if the
    header answers the question you would have opened them to ask, so the
    summary text is part of the header rather than something inside.
    """

    def __init__(self, title, summary="", expanded=False, parent=None):
        super().__init__(parent)
        self.title = title
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.toggle_button = QPushButton()
        self.toggle_button.setCheckable(True)
        self.toggle_button.setChecked(expanded)
        self.toggle_button.setStyleSheet("""
            QPushButton {
                text-align: left; padding: 7px 10px; font-weight: bold;
                background-color: #333333; color: #ffffff;
                border: 1px solid #444444; border-radius: 4px;
            }
            QPushButton:hover { background-color: #3d3d3d; }
            QPushButton:checked { border-bottom-left-radius: 0px;
                                  border-bottom-right-radius: 0px; }
        """)
        self.toggle_button.toggled.connect(self._on_toggle)

        self.container = QWidget()
        self.body = QVBoxLayout(self.container)
        self.body.setContentsMargins(0, 0, 0, 0)
        self.body.setSpacing(0)

        layout.addWidget(self.toggle_button)
        layout.addWidget(self.container)
        self.container.setVisible(expanded)
        self._summary = summary
        self._refresh_text()

    def add_widget(self, widget):
        self.body.addWidget(widget)

    def set_summary(self, summary):
        """Update the state shown beside the title while it is collapsed."""
        if summary != self._summary:
            self._summary = summary
            self._refresh_text()

    def _refresh_text(self):
        arrow = "\u25bc" if self.toggle_button.isChecked() else "\u25b6"
        suffix = f"   \u2014   {self._summary}" if self._summary else ""
        # A header is a label, not a mnemonic target: unescaped, Qt swallowed
        # the ampersand and rendered "Setup & bulk actions" as "Setup _bulk
        # actions". Summaries carry device names, which can contain one too.
        text = f"{arrow} {self.title}{suffix}".replace("&", "&&")
        self.toggle_button.setText(text)

    def _on_toggle(self, checked):
        self.container.setVisible(checked)
        self._refresh_text()


class CollapsibleLogBox(QWidget):
    """Serial monitor: a collapsible log with a command entry line."""

    command_submitted = pyqtSignal(str)
    MAX_BLOCKS = 2000       # a multi-day session would otherwise grow unbounded

    def __init__(self, title="FED Log", parent=None):
        super().__init__(parent)
        self.title = title
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Maximum)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.toggle_button = QPushButton(f"▶ {title}")
        self.toggle_button.setCheckable(True)
        self.toggle_button.setStyleSheet("""
            QPushButton {
                text-align: left; padding: 6px; font-weight: bold;
                background-color: #333333; color: #ffffff;
                border: none; border-top: 1px solid #444444;
            }
            QPushButton:hover { background-color: #444444; }
        """)
        self.toggle_button.toggled.connect(self._on_toggle)

        self.container = QWidget()
        container_layout = QVBoxLayout(self.container)
        container_layout.setContentsMargins(0, 0, 0, 0)
        container_layout.setSpacing(0)

        input_row = QHBoxLayout()
        input_row.setContentsMargins(0, 0, 0, 0)
        input_row.setSpacing(0)
        self.command_input = QLineEdit()
        self.command_input.setPlaceholderText("Send a raw command to all devices")
        self.command_input.setStyleSheet(
            "background-color: #2b2b2b; color: #ffffff; "
            "border: 1px solid #3f3f3f; padding: 4px;")
        self.command_input.returnPressed.connect(self._submit)
        send_btn = QPushButton("Send")
        send_btn.setFixedWidth(60)
        send_btn.clicked.connect(self._submit)
        input_row.addWidget(self.command_input)
        input_row.addWidget(send_btn)

        self.log_text = QTextEdit()
        self.log_text.setReadOnly(True)
        self.log_text.setFixedHeight(150)
        self.log_text.document().setMaximumBlockCount(self.MAX_BLOCKS)
        self.log_text.setStyleSheet(
            "background-color: #1e1e1e; color: #cccccc; border: none; "
            "font-family: monospace;")

        container_layout.addLayout(input_row)
        container_layout.addWidget(self.log_text)

        layout.addWidget(self.toggle_button)
        layout.addWidget(self.container)
        self.container.hide()

    def _on_toggle(self, checked):
        self.toggle_button.setText(f"{'▼' if checked else '▶'} {self.title}")
        self.container.setVisible(checked)

    def append_log(self, text, success=True):
        stamp = datetime.now().strftime("%H:%M:%S")
        color = "#4caf50" if success else "#f44336"
        prefix = "[OK]" if success else "[ERR]"
        for line in text.splitlines():
            if line.strip():
                self.log_text.append(
                    f"<span style='color:#888888;'>[{stamp}]</span> "
                    f"<span style='color:{color};'>{prefix}</span> {line}")
        scrollbar = self.log_text.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def _submit(self):
        command = self.command_input.text().strip()
        if command:
            self.command_submitted.emit(command)
            self.command_input.clear()


class FileSelectorDialog(QDialog):
    """Pick SD-card files to export, from one device or several."""

    def __init__(self, device_files, parent=None, title="Select files to export"):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.resize(520, 420)
        self.setModal(True)

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "Select the CSV logs to copy from the device SD card.\n"
            "Files already mirrored into the session folder are marked."))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)

        self._entries = []      # (device, filename, checkbox)
        for device, files in device_files.items():
            if len(device_files) > 1:
                header = QLabel(f"<b>{device.name}</b>")
                content_layout.addWidget(header)
            # A DeviceMirror owns two stores (session and archive); the offsets
            # live on those, not on the mirror itself.
            mirrored = set()
            if device.mirror is not None:
                mirrored = (set(device.mirror.session.offsets)
                            | set(device.mirror.archive.offsets))
            for filename, size in files:
                label = f"{filename} ({_format_size(size)})"
                if filename in mirrored:
                    label += "  — mirrored"
                checkbox = QCheckBox(("  " if len(device_files) > 1 else "") + label)
                checkbox.setChecked(True)
                content_layout.addWidget(checkbox)
                self._entries.append((device, filename, checkbox))
            content_layout.addSpacing(8)

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

        buttons_row = QHBoxLayout()
        select_all = QPushButton("Select All")
        clear_all = QPushButton("Clear All")
        select_all.clicked.connect(lambda: self._set_all(True))
        clear_all.clicked.connect(lambda: self._set_all(False))
        buttons_row.addWidget(select_all)
        buttons_row.addWidget(clear_all)
        buttons_row.addStretch()
        layout.addLayout(buttons_row)

        buttons = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel,
                                   Qt.Horizontal, self)
        buttons.accepted.connect(self.accept)
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _set_all(self, checked):
        for _, _, checkbox in self._entries:
            checkbox.setChecked(checked)

    def selected(self):
        """``[(device, filename), ...]`` for the ticked entries."""
        return [(device, filename)
                for device, filename, checkbox in self._entries
                if checkbox.isChecked()]


class ResumeSessionDialog(QDialog):
    """Offers to reopen a session that was interrupted."""

    def __init__(self, sessions, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Unfinished recording session found")
        self.resize(600, 320)
        self.setModal(True)
        self.choice = None

        layout = QVBoxLayout(self)
        layout.addWidget(QLabel(
            "FNT did not shut down cleanly during these recordings.\n"
            "Resuming appends to the existing folder and keeps its data, logs "
            "and SD-card mirrors intact.\nStarting fresh leaves them untouched "
            "on disk."))

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        content = QWidget()
        content_layout = QVBoxLayout(content)

        self._buttons = []
        for root, state in sessions:
            devices = state.get("devices") or []
            names = ", ".join(d.get("name", "?") for d in devices) or "no devices"
            pellets = sum((d.get("stats") or {}).get("pellet", 0) for d in devices)
            button = QPushButton(
                f"{os.path.basename(root)}\n"
                f"    {len(devices)} device(s): {names}\n"
                f"    {pellets} pellets recorded · last activity "
                f"{state.get('updated_at', 'unknown')}")
            button.setStyleSheet("text-align: left; padding: 8px;")
            button.clicked.connect(lambda _, r=root: self._pick(r))
            content_layout.addWidget(button)
            self._buttons.append(button)

        content_layout.addStretch()
        scroll.setWidget(content)
        layout.addWidget(scroll)

        buttons = QDialogButtonBox(QDialogButtonBox.Cancel, Qt.Horizontal, self)
        buttons.button(QDialogButtonBox.Cancel).setText("Start a new session")
        buttons.rejected.connect(self.reject)
        layout.addWidget(buttons)

    def _pick(self, root):
        self.choice = root
        self.accept()


def _format_size(size):
    if size < 1024:
        return f"{size} B"
    if size < 1024 * 1024:
        return f"{size / 1024:.1f} KB"
    return f"{size / (1024 * 1024):.1f} MB"
