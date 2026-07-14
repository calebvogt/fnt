"""Live numeric channel table for MuseStudio.

A compact readout that lists every stream's channels with their current value,
updated at a low refresh rate (values, not graphs). Complements the scrolling
plot: the plot shows shape over time, this shows the instantaneous numbers and
the exact column layout being recorded to CSV.
"""

import re

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtWidgets import (
    QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem,
)


def _short_stream(name):
    """"Muse-EEG (00:11...)" -> "EEG" for a compact table column."""
    s = re.sub(r"\s*\(.*\)\s*$", "", name)   # drop trailing " (device_id)"
    return re.sub(r"^Muse[-_]?", "", s) or s


class LiveValuesPanel(QTableWidget):
    """Table of (stream, channel, value) rows, refreshed on a timer."""

    def __init__(self, parent=None, refresh_hz=5):
        super().__init__(0, 3, parent)
        self.setHorizontalHeaderLabels(["Stream", "Channel", "Value"])
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.setFocusPolicy(Qt.NoFocus)
        hdr = self.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.Stretch)

        self._channel_names = {}   # stream -> [names]
        self._rows = {}            # (stream, ch_idx) -> row number
        self._latest = {}          # (stream, ch_idx) -> float

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(int(1000 / refresh_hz))

    def set_channel_names(self, mapping):
        self._channel_names.update(mapping)

    def add_samples(self, stream_name, timestamps, data):
        if data is None or len(data) == 0:
            return
        last = data[-1]
        n = len(last) if np.ndim(last) else 1
        names = self._channel_names.get(stream_name)
        for ch in range(n):
            key = (stream_name, ch)
            if key not in self._rows:
                label = names[ch] if names and ch < len(names) else f"ch{ch}"
                self._add_row(stream_name, label, key)
            self._latest[key] = float(last[ch]) if n > 1 else float(last)

    def _add_row(self, stream_name, channel_label, key):
        row = self.rowCount()
        self.insertRow(row)
        self.setItem(row, 0, QTableWidgetItem(_short_stream(stream_name)))
        self.setItem(row, 1, QTableWidgetItem(channel_label))
        val = QTableWidgetItem("—")
        val.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.setItem(row, 2, val)
        self._rows[key] = row

    def _refresh(self):
        for key, value in self._latest.items():
            row = self._rows.get(key)
            if row is None:
                continue
            self.item(row, 2).setText(f"{value:.2f}")

    def clear_values(self):
        self.setRowCount(0)
        self._rows.clear()
        self._latest.clear()
        self._channel_names.clear()
