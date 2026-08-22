"""Live numeric channel table for MuseStudio.

A compact readout that lists every stream's channels with their current value,
updated at a low refresh rate (values, not graphs). Complements the scrolling
plot: the plot shows shape over time, this shows the instantaneous numbers and
the exact column layout being recorded to CSV.

Each row also carries a checkbox controlling whether that channel is drawn in
the live view, so a 28-channel headband can be reduced to the handful you care
about without losing the numeric readout.
"""

import re

import numpy as np
from PyQt5.QtCore import Qt, QTimer, pyqtSignal
from PyQt5.QtWidgets import (
    QAbstractItemView, QHeaderView, QTableWidget, QTableWidgetItem,
)

from fnt.musestudio.dsp import describe_optics_channel, is_scalp_eeg


def _short_stream(name):
    """"Muse-EEG (00:11...)" -> "EEG" for a compact table column."""
    s = re.sub(r"\s*\(.*\)\s*$", "", name)   # drop trailing " (device_id)"
    return re.sub(r"^Muse[-_]?", "", s) or s


def channel_tooltip(stream, channel):
    """Explain what a channel actually measures (shown on hover)."""
    upper = str(channel).upper()
    optics = describe_optics_channel(channel)
    if optics != str(channel):
        return f"{channel}\n{optics}"
    if "AUX" in upper:
        return (f"{channel}\nAuxiliary electrode input. With nothing connected "
                "to the accessory port this pin floats and reads as noise — it "
                "is excluded from band power, contact and synchrony.")
    if is_scalp_eeg(channel):
        where = {"TP9": "left ear / temporal", "AF7": "left forehead",
                 "AF8": "right forehead", "TP10": "right ear / temporal"}
        for k, v in where.items():
            if k in upper:
                return f"{channel}\nScalp EEG electrode — {v}."
    if "ACC" in upper:
        return f"{channel}\nAccelerometer axis — head movement (g)."
    if "GYRO" in upper:
        return f"{channel}\nGyroscope axis — head rotation rate (deg/s)."
    if "BATT" in upper:
        return f"{channel}\nHeadband battery level."
    return str(channel)


class LiveValuesPanel(QTableWidget):
    """Table of (visible, stream, channel, value) rows, refreshed on a timer."""

    visibility_changed = pyqtSignal(str, list)   # stream, visible channel names

    def __init__(self, parent=None, refresh_hz=5):
        super().__init__(0, 4, parent)
        self.setHorizontalHeaderLabels(["", "Stream", "Channel", "Value"])
        self.verticalHeader().setVisible(False)
        self.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.setSelectionMode(QAbstractItemView.NoSelection)
        self.setFocusPolicy(Qt.NoFocus)
        hdr = self.horizontalHeader()
        hdr.setSectionResizeMode(0, QHeaderView.Fixed)
        self.setColumnWidth(0, 26)
        hdr.setSectionResizeMode(1, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(2, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.Stretch)

        self._channel_names = {}   # stream -> [names]
        self._rows = {}            # (stream, ch_idx) -> row number
        self._latest = {}          # (stream, ch_idx) -> float
        self._hidden = {}          # stream -> set of hidden channel names
        self._paused = False

        self.itemChanged.connect(self._on_item_changed)

        self._timer = QTimer(self)
        self._timer.timeout.connect(self._refresh)
        self._timer.start(int(1000 / refresh_hz))

    def set_paused(self, paused):
        self._paused = bool(paused)

    def set_channel_names(self, mapping):
        self._channel_names.update(mapping)

    def set_visible_channels(self, stream, visible_names):
        """Programmatically set which channels are ticked (e.g. curated view)."""
        names = self._channel_names.get(stream, [])
        self._hidden[stream] = {n for n in names if n not in set(visible_names)}
        self.blockSignals(True)
        for (s, ch), row in self._rows.items():
            if s != stream:
                continue
            item = self.item(row, 0)
            if item is not None:
                name = names[ch] if ch < len(names) else None
                item.setCheckState(
                    Qt.Unchecked if name in self._hidden[stream] else Qt.Checked)
        self.blockSignals(False)

    def visible_channels(self, stream):
        names = self._channel_names.get(stream, [])
        hidden = self._hidden.get(stream, set())
        return [n for n in names if n not in hidden]

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
        self.blockSignals(True)
        self.insertRow(row)
        check = QTableWidgetItem()
        check.setFlags(Qt.ItemIsUserCheckable | Qt.ItemIsEnabled)
        hidden = self._hidden.get(stream_name, set())
        check.setCheckState(Qt.Unchecked if channel_label in hidden else Qt.Checked)
        check.setToolTip("Show this channel in the live signal view.")
        self.setItem(row, 0, check)
        self.setItem(row, 1, QTableWidgetItem(_short_stream(stream_name)))
        name_item = QTableWidgetItem(channel_label)
        name_item.setToolTip(channel_tooltip(stream_name, channel_label))
        self.setItem(row, 2, name_item)
        val = QTableWidgetItem("—")
        val.setTextAlignment(Qt.AlignRight | Qt.AlignVCenter)
        self.setItem(row, 3, val)
        self.blockSignals(False)
        self._rows[key] = row

    def _on_item_changed(self, item):
        if item.column() != 0:
            return
        for (stream, ch), row in self._rows.items():
            if row != item.row():
                continue
            names = self._channel_names.get(stream, [])
            if ch >= len(names):
                return
            name = names[ch]
            hidden = self._hidden.setdefault(stream, set())
            if item.checkState() == Qt.Checked:
                hidden.discard(name)
            else:
                hidden.add(name)
            self.visibility_changed.emit(stream, self.visible_channels(stream))
            return

    def _refresh(self):
        if self._paused or not self.isVisible():
            return
        for key, value in self._latest.items():
            row = self._rows.get(key)
            if row is None:
                continue
            self.item(row, 3).setText(f"{value:.2f}")

    def clear_values(self):
        self.setRowCount(0)
        self._rows.clear()
        self._latest.clear()
        self._channel_names.clear()
        self._hidden.clear()
