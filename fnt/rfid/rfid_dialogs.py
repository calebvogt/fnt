"""Dialogs for the RFID PreProcessing tool.

* :class:`MetadataMappingDialog` - which row is the header, and which columns
  are name, sex, trial, tags, release and trap.
* :class:`ArenaDialog` - paddocks and readers, zones, and the
  ``(reader, antenna) -> zone`` table, with presets and CSV import/export.
* :class:`IdentityDialog` - per-animal start/stop, found-dead handling, and
  lost-tag intervals (:class:`LostTagsDialog`).
* :class:`DataFrameModel` - a read-only table model for the QC views.

They follow the UWB tool's patterns (IdentityAssignmentDialog,
OffAnimalIntervalsDialog) but are copied, not imported: each FNT tool stands
alone.
"""

from __future__ import annotations

import copy

import numpy as np
import pandas as pd
from PyQt5.QtCore import QAbstractTableModel, QDateTime, QModelIndex, Qt
from PyQt5.QtGui import QBrush, QColor
from PyQt5.QtWidgets import (QAbstractItemView, QCheckBox, QComboBox,
                             QDateTimeEdit, QDialog, QDialogButtonBox, QFileDialog,
                             QFormLayout, QGroupBox, QHBoxLayout, QHeaderView,
                             QLabel, QLineEdit, QListWidget, QListWidgetItem,
                             QMessageBox, QPushButton, QSpinBox, QTableWidget,
                             QTableWidgetItem, QTabWidget, QVBoxLayout, QWidget)

from .config.config_manager import (read_antenna_csv, write_antenna_csv,
                                    zones_for_antennas)
from .config.defaults import (ARENA_PRESETS, METADATA_FIELDS, Antenna, Arena,
                              MetadataSpec, Paddock, Zone, animal_settings)
from .core.metadata import (detect_header_row, guess_mapping, header_names,
                            read_grid)

TIME_FMT = "yyyy-MM-dd HH:mm:ss.zzz"
PY_TIME_FMT = "%Y-%m-%d %H:%M:%S.%f"


def to_qdatetime(value) -> QDateTime:
    """A stored time (text or Timestamp) as a QDateTime; invalid if empty."""
    if value is None or (isinstance(value, float) and np.isnan(value)):
        return QDateTime()
    stamp = pd.to_datetime(value, errors="coerce")
    if pd.isna(stamp):
        return QDateTime()
    return QDateTime.fromString(stamp.strftime(PY_TIME_FMT)[:23], TIME_FMT)


def from_qdatetime(dt: QDateTime) -> str:
    return dt.toString(TIME_FMT) if dt.isValid() else ""


def fmt_time(value) -> str:
    stamp = pd.to_datetime(value, errors="coerce")
    return "" if pd.isna(stamp) else stamp.strftime("%Y-%m-%d %H:%M")


def picker(value=None, fallback: QDateTime | None = None) -> QDateTimeEdit:
    e = QDateTimeEdit()
    e.setDisplayFormat(TIME_FMT)
    e.setCalendarPopup(True)
    dt = to_qdatetime(value)
    if not dt.isValid():
        dt = fallback if fallback is not None and fallback.isValid() \
            else QDateTime.currentDateTime()
    e.setDateTime(dt)
    return e


# --------------------------------------------------------------------------- #
# Table model for the QC views
# --------------------------------------------------------------------------- #

class DataFrameModel(QAbstractTableModel):
    """Read-only view of a DataFrame, sortable by clicking a header.

    Values are formatted on demand rather than converted up front, so a
    65,000-row bouts table opens instantly.
    """

    def __init__(self, frame: pd.DataFrame | None = None, parent=None):
        super().__init__(parent)
        self._df = pd.DataFrame() if frame is None else frame.reset_index(drop=True)

    def set_frame(self, frame: pd.DataFrame | None):
        self.beginResetModel()
        self._df = pd.DataFrame() if frame is None else frame.reset_index(drop=True)
        self.endResetModel()

    def frame(self) -> pd.DataFrame:
        return self._df

    def rowCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df)

    def columnCount(self, parent=QModelIndex()):
        return 0 if parent.isValid() else len(self._df.columns)

    def data(self, index, role=Qt.DisplayRole):
        if not index.isValid() or role not in (Qt.DisplayRole, Qt.ToolTipRole):
            return None
        value = self._df.iat[index.row(), index.column()]
        return self.format(value)

    @staticmethod
    def format(value) -> str:
        if value is None:
            return ""
        if isinstance(value, (pd.Timestamp, np.datetime64)):
            stamp = pd.Timestamp(value)
            return "" if pd.isna(stamp) else stamp.strftime(PY_TIME_FMT)[:23]
        if isinstance(value, (float, np.floating)):
            if np.isnan(value):
                return ""
            if float(value).is_integer() and abs(value) < 1e15:
                return f"{int(value):,}" if abs(value) >= 10000 else str(int(value))
            return f"{value:,.3f}" if abs(value) >= 1 else f"{value:.4f}"
        if isinstance(value, (bool, np.bool_)):
            return "yes" if value else ""
        if isinstance(value, (int, np.integer)):
            return f"{int(value):,}" if abs(int(value)) >= 10000 else str(int(value))
        if isinstance(value, str):
            return value
        try:
            if pd.isna(value):
                return ""
        except (TypeError, ValueError):
            pass
        return str(value)

    def headerData(self, section, orientation, role=Qt.DisplayRole):
        if role != Qt.DisplayRole:
            return None
        if orientation == Qt.Horizontal:
            return str(self._df.columns[section]) if section < len(self._df.columns) else None
        return str(section + 1)

    def sort(self, column, order=Qt.AscendingOrder):
        if self._df.empty or column >= len(self._df.columns):
            return
        self.layoutAboutToBeChanged.emit()
        col = self._df.columns[column]
        try:
            self._df = self._df.sort_values(col, ascending=(order == Qt.AscendingOrder),
                                            kind="mergesort", na_position="last")
        except TypeError:
            self._df = self._df.sort_values(col, key=lambda s: s.astype(str),
                                            ascending=(order == Qt.AscendingOrder),
                                            kind="mergesort")
        self._df = self._df.reset_index(drop=True)
        self.layoutChanged.emit()


# --------------------------------------------------------------------------- #
# Metadata mapping
# --------------------------------------------------------------------------- #

FIELD_LABELS = {
    "name": "Name (required)", "sex": "Sex", "trial": "Trial / home paddock",
    "phase": "Phase / status", "code": "Code", "release_date": "Release date",
    "release_time": "Release time", "trap_date": "Trap / removal date",
    "trap_time": "Trap / removal time"}


class MetadataMappingDialog(QDialog):
    """Pick the header row and say which column is which.

    The preview shows the first rows as the spreadsheet stores them, so a
    notes row above the header (the LID workbook has one) is visible and the
    header row can be set by eye when detection guesses wrong.
    """

    PREVIEW_ROWS = 14

    def __init__(self, spec: MetadataSpec, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Metadata columns")
        self.setMinimumSize(900, 640)
        self.spec = copy.deepcopy(spec)
        self.grid = read_grid(spec.path, spec.sheet, max_rows=200)
        self.detected = detect_header_row(self.grid) + 1 if self.grid else 1
        layout = QVBoxLayout(self)

        intro = QLabel(
            "Choose the row that holds the column names (0 = detect; row numbers "
            "are as the spreadsheet shows them), then which column is which. "
            "Tick every column that holds a tag number - an animal can wear "
            "several tags, but a tag belongs to exactly one animal.")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        row = QHBoxLayout()
        row.addWidget(QLabel("Header row:"))
        self.spin_header = QSpinBox()
        self.spin_header.setRange(0, 50)
        self.spin_header.setValue(spec.header_row)
        self.spin_header.setSpecialValueText(f"detect (row {self.detected})")
        self.spin_header.setToolTip("0 detects the header row from the content.")
        self.spin_header.valueChanged.connect(self._reload_headers)
        row.addWidget(self.spin_header)
        row.addStretch()
        layout.addLayout(row)

        self.preview = QTableWidget()
        self.preview.setEditTriggers(QAbstractItemView.NoEditTriggers)
        self.preview.setMinimumHeight(170)
        layout.addWidget(self.preview, 1)

        body = QHBoxLayout()
        form_box = QGroupBox("Columns")
        form = QFormLayout(form_box)
        self.combos = {}
        for fld in METADATA_FIELDS:
            combo = QComboBox()
            combo.setMinimumWidth(200)
            self.combos[fld] = combo
            form.addRow(FIELD_LABELS.get(fld, fld) + ":", combo)
        body.addWidget(form_box, 1)

        tag_box = QGroupBox("Tag columns")
        tag_layout = QVBoxLayout(tag_box)
        self.tag_list = QListWidget()
        tag_layout.addWidget(self.tag_list)
        body.addWidget(tag_box, 1)
        layout.addLayout(body, 1)

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)
        self._reload_headers(first=True)

    def header_index(self) -> int:
        value = self.spin_header.value()
        return (value if value > 0 else self.detected) - 1

    def _reload_headers(self, *_args, first: bool = False):
        idx = self.header_index()
        headers = header_names(self.grid, idx) if self.grid else []
        # preview: the rows around the header, header row highlighted
        rows = self.grid[:self.PREVIEW_ROWS]
        width = max((len(r) for r in rows), default=0)
        self.preview.setRowCount(len(rows))
        self.preview.setColumnCount(width)
        self.preview.setVerticalHeaderLabels([str(i + 1) for i in range(len(rows))])
        for r, values in enumerate(rows):
            for c in range(width):
                v = values[c] if c < len(values) else None
                item = QTableWidgetItem("" if v is None else str(v))
                if r == idx:
                    item.setBackground(QBrush(QColor("#1f5560")))
                self.preview.setItem(r, c, item)

        guess, guess_tags = guess_mapping(headers)
        current = {f: (self.spec.columns.get(f, "") if first else self.combos[f].currentData())
                   for f in METADATA_FIELDS}
        for fld, combo in self.combos.items():
            combo.blockSignals(True)
            combo.clear()
            combo.addItem("(none)", "")
            for h in headers:
                combo.addItem(h, h)
            want = current.get(fld) if current.get(fld) in headers else guess.get(fld, "")
            combo.setCurrentIndex(max(0, combo.findData(want)))
            combo.blockSignals(False)
        ticked = set(self.spec.tag_columns if first else self.tag_columns())
        if not ticked & set(headers):
            ticked = set(guess_tags)
        self.tag_list.clear()
        first_ticked = None
        for h in headers:
            item = QListWidgetItem(h)
            item.setFlags(item.flags() | Qt.ItemIsUserCheckable)
            item.setCheckState(Qt.Checked if h in ticked else Qt.Unchecked)
            self.tag_list.addItem(item)
            if h in ticked and first_ticked is None:
                first_ticked = item
        if first_ticked is not None:
            self.tag_list.scrollToItem(first_ticked, QAbstractItemView.PositionAtTop)

    def tag_columns(self) -> list[str]:
        return [self.tag_list.item(i).text() for i in range(self.tag_list.count())
                if self.tag_list.item(i).checkState() == Qt.Checked]

    def accept(self):
        if not self.combos["name"].currentData():
            QMessageBox.warning(self, "Name column", "Choose the column that holds "
                                "the animal names.")
            return
        if not self.tag_columns():
            QMessageBox.warning(self, "Tag columns", "Tick at least one tag column.")
            return
        super().accept()

    def result_spec(self) -> MetadataSpec:
        spec = copy.deepcopy(self.spec)
        spec.header_row = self.spin_header.value()
        spec.columns = {f: c.currentData() for f, c in self.combos.items() if c.currentData()}
        spec.tag_columns = self.tag_columns()
        return spec


# --------------------------------------------------------------------------- #
# Arena
# --------------------------------------------------------------------------- #

def _cell(table: QTableWidget, r: int, c: int) -> str:
    item = table.item(r, c)
    return item.text().strip() if item else ""


def _set_row(table: QTableWidget, r: int, values: list):
    for c, v in enumerate(values):
        table.setItem(r, c, QTableWidgetItem("" if v is None else str(v)))


class ArenaDialog(QDialog):
    """Edit paddocks, zones and antennas.

    Zones are named by paddock + zone (``A3``); an antenna is keyed by
    ``(reader, antenna)`` and lands in its reader's paddock unless its
    Paddock cell says otherwise. Several antennas per zone is normal.
    """

    PADDOCK_COLS = ["ID", "Name", "Readers", "Trials / home values", "Width (m)",
                    "Length (m)"]
    ZONE_COLS = ["Paddock", "Zone", "X (m)", "Y (m)", "Label"]
    ANTENNA_COLS = ["Reader", "Antenna", "Zone", "Location", "Paddock (optional)",
                    "X (m)", "Y (m)"]

    def __init__(self, arena: Arena, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Arena: paddocks, zones and antennas")
        self.setMinimumSize(980, 700)
        layout = QVBoxLayout(self)

        top = QHBoxLayout()
        top.addWidget(QLabel("Preset:"))
        self.combo_preset = QComboBox()
        self.combo_preset.addItems(list(ARENA_PRESETS))
        top.addWidget(self.combo_preset, 1)
        btn_preset = QPushButton("Apply Preset")
        btn_preset.setToolTip("Replace every table below with the preset.")
        btn_preset.clicked.connect(self.apply_preset)
        top.addWidget(btn_preset)
        layout.addLayout(top)

        intro = QLabel(
            "<b>Paddocks</b>: the ID is the zone prefix (A, B...). <b>Readers</b> are "
            "the reader IDs installed in that paddock. <b>Trials / home values</b> "
            "are the metadata trial values of the animals that live there - a read "
            "of one of them by another paddock's reader is a visit (flagged away).")
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.RichText)
        layout.addWidget(intro)

        tabs = QTabWidget()
        self.tbl_paddocks = self._table(self.PADDOCK_COLS)
        self.tbl_zones = self._table(self.ZONE_COLS)
        self.tbl_antennas = self._table(self.ANTENNA_COLS)
        tabs.addTab(self._with_buttons(self.tbl_paddocks), "Paddocks")
        tabs.addTab(self._with_buttons(self.tbl_zones), "Zones")
        ant_page = self._with_buttons(self.tbl_antennas, extra=[
            ("Import CSV...", self.import_antennas),
            ("Export CSV...", self.export_antennas),
            ("Add Missing Zones", self.add_missing_zones)])
        tabs.addTab(ant_page, "Antennas")
        layout.addWidget(tabs, 1)

        self.lbl_problems = QLabel("")
        self.lbl_problems.setWordWrap(True)
        layout.addWidget(self.lbl_problems)

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)
        self.load(arena)
        for t in (self.tbl_paddocks, self.tbl_zones, self.tbl_antennas):
            t.itemChanged.connect(self._validate)
        self._validate()

    @staticmethod
    def _table(columns):
        t = QTableWidget(0, len(columns))
        t.setHorizontalHeaderLabels(columns)
        t.horizontalHeader().setSectionResizeMode(QHeaderView.Stretch)
        t.verticalHeader().setVisible(False)
        t.setSelectionBehavior(QAbstractItemView.SelectRows)
        return t

    def _with_buttons(self, table, extra=()):
        page = QWidget()
        v = QVBoxLayout(page)
        v.addWidget(table, 1)
        h = QHBoxLayout()
        add = QPushButton("Add Row")
        add.clicked.connect(lambda: self._add_row(table))
        rm = QPushButton("Remove Selected")
        rm.clicked.connect(lambda: self._remove_rows(table))
        h.addWidget(add)
        h.addWidget(rm)
        for label, slot in extra:
            b = QPushButton(label)
            b.clicked.connect(slot)
            h.addWidget(b)
        h.addStretch()
        v.addLayout(h)
        return page

    def _add_row(self, table):
        r = table.rowCount()
        table.insertRow(r)
        # copy the row above: most new rows differ from it in one cell
        if r > 0:
            _set_row(table, r, [_cell(table, r - 1, c) for c in range(table.columnCount())])
        self._validate()

    def _remove_rows(self, table):
        for r in sorted({i.row() for i in table.selectedIndexes()}, reverse=True):
            table.removeRow(r)
        self._validate()

    def load(self, arena: Arena):
        for t in (self.tbl_paddocks, self.tbl_zones, self.tbl_antennas):
            t.blockSignals(True)
            t.setRowCount(0)
        for p in arena.paddocks:
            r = self.tbl_paddocks.rowCount()
            self.tbl_paddocks.insertRow(r)
            _set_row(self.tbl_paddocks, r, [p.paddock_id, p.name, ", ".join(p.readers),
                                            ", ".join(p.trials), p.width_m, p.length_m])
        for z in arena.zones:
            r = self.tbl_zones.rowCount()
            self.tbl_zones.insertRow(r)
            _set_row(self.tbl_zones, r, [z.paddock_id, z.zone, z.x, z.y, z.label])
        for a in arena.antennas:
            r = self.tbl_antennas.rowCount()
            self.tbl_antennas.insertRow(r)
            _set_row(self.tbl_antennas, r, [a.reader, a.antenna, a.zone, a.location,
                                            a.paddock_id, a.x, a.y])
        for t in (self.tbl_paddocks, self.tbl_zones, self.tbl_antennas):
            t.blockSignals(False)

    def arena(self) -> Arena:
        def split(text):
            return [x.strip() for x in text.replace(";", ",").split(",") if x.strip()]
        arena = Arena()
        for r in range(self.tbl_paddocks.rowCount()):
            pid = _cell(self.tbl_paddocks, r, 0)
            if not pid:
                continue
            arena.paddocks.append(Paddock(pid, _cell(self.tbl_paddocks, r, 1),
                                          split(_cell(self.tbl_paddocks, r, 2)),
                                          split(_cell(self.tbl_paddocks, r, 3)),
                                          _cell(self.tbl_paddocks, r, 4),
                                          _cell(self.tbl_paddocks, r, 5)))
        for r in range(self.tbl_zones.rowCount()):
            if not _cell(self.tbl_zones, r, 1):
                continue
            arena.zones.append(Zone(*[_cell(self.tbl_zones, r, c) for c in range(5)]))
        for r in range(self.tbl_antennas.rowCount()):
            vals = [_cell(self.tbl_antennas, r, c) for c in range(7)]
            if not vals[0] and not vals[1]:
                continue
            arena.antennas.append(Antenna(reader=vals[0], antenna=vals[1], zone=vals[2],
                                          location=vals[3], paddock_id=vals[4],
                                          x=vals[5], y=vals[6]))
        return arena

    def _validate(self, *_):
        blocking, warnings = self.arena().problems()
        lines = [f"<span style='color:#e06c75'>&#9888; {p}</span>" for p in blocking]
        lines += [f"<span style='color:#e5c07b'>{w}</span>" for w in warnings]
        a = self.arena()
        summary = (f"{len(a.paddocks)} paddock(s), {len(a.zones)} zone(s), "
                   f"{len(a.antennas)} antenna(s)")
        self.lbl_problems.setText("<br>".join([summary] + lines))

    def apply_preset(self):
        name = self.combo_preset.currentText()
        if self.tbl_antennas.rowCount() and QMessageBox.question(
                self, "Replace arena", f"Replace the current tables with '{name}'?",
                QMessageBox.Yes | QMessageBox.No) != QMessageBox.Yes:
            return
        self.load(ARENA_PRESETS[name]())
        self._validate()

    def import_antennas(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import antenna table", "",
                                              "CSV files (*.csv);;All files (*)")
        if not path:
            return
        try:
            antennas = read_antenna_csv(path)
        except Exception as exc:                          # noqa: BLE001
            QMessageBox.warning(self, "Import failed", str(exc))
            return
        current = self.arena()
        current.antennas = antennas
        current.zones.extend(zones_for_antennas(current, antennas))
        self.load(current)
        self._validate()

    def export_antennas(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export antenna table",
                                              "antennas.csv", "CSV files (*.csv)")
        if path:
            write_antenna_csv(self.arena(), path)

    def add_missing_zones(self):
        current = self.arena()
        new = zones_for_antennas(current, current.antennas)
        current.zones.extend(new)
        self.load(current)
        self._validate()

    def accept(self):
        blocking, _ = self.arena().problems()
        if blocking and QMessageBox.question(
                self, "Arena has problems",
                "The arena cannot be run as it is:\n\n" + "\n".join(blocking)
                + "\n\nKeep it anyway (and fix it later)?",
                QMessageBox.Yes | QMessageBox.No, QMessageBox.No) != QMessageBox.Yes:
            return
        super().accept()


# --------------------------------------------------------------------------- #
# Lost tags and identities
# --------------------------------------------------------------------------- #

class LostTagsDialog(QDialog):
    """One animal's lost-tag intervals: a tag off the animal from a time.

    A tag lying in an entrance tube or under a floor antenna keeps being read
    and places the animal where the tag is. Reads of that tag inside an
    interval are dropped; the animal's other tags are unaffected. Leave
    'Back on' unticked for a tag that never went back on.
    """

    def __init__(self, animal: str, tags: list[str], intervals: list[dict],
                 default_time=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle(f"Lost tags — {animal}")
        self.setMinimumWidth(900)
        self.tags = tags
        self.default_time = to_qdatetime(default_time)
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Periods when a tag was being read but was <b>not on the animal</b> "
            "(found in a tube, under a floor, recovered in the field). Its reads "
            "inside are dropped from every output; the QC report 'Two places at "
            "once' is the usual way to find them.")
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.RichText)
        layout.addWidget(intro)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Tag", "Off from", "Back on", "Note", ""])
        hdr = self.table.horizontalHeader()
        for c in (0, 1, 2, 4):
            hdr.setSectionResizeMode(c, QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(3, QHeaderView.Stretch)
        self.table.verticalHeader().setVisible(False)
        layout.addWidget(self.table, 1)
        for item in intervals or []:
            self._add_row(item)
        add = QPushButton("Add Lost Tag")
        add.clicked.connect(lambda: self._add_row({}))
        layout.addWidget(add)
        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)

    def _add_row(self, item: dict):
        r = self.table.rowCount()
        self.table.insertRow(r)
        combo = QComboBox()
        combo.setEditable(True)
        combo.addItems(self.tags)
        if item.get("tag"):
            idx = combo.findText(str(item["tag"]))
            if idx >= 0:
                combo.setCurrentIndex(idx)
            else:
                combo.setEditText(str(item["tag"]))
        start = picker(item.get("start"), self.default_time)
        back = QWidget()
        bl = QHBoxLayout(back)
        bl.setContentsMargins(0, 0, 0, 0)
        chk = QCheckBox()
        stop = picker(item.get("stop"), start.dateTime().addDays(1))
        chk.setChecked(bool(item.get("stop")))
        stop.setEnabled(chk.isChecked())
        chk.toggled.connect(stop.setEnabled)
        bl.addWidget(chk)
        bl.addWidget(stop)
        note = QLineEdit(item.get("note", ""))
        note.setPlaceholderText("optional, e.g. recovered from B3 entrance")
        rm = QPushButton("Remove")

        def _remove():
            for i in range(self.table.rowCount()):
                if self.table.cellWidget(i, 4) is rm:
                    self.table.removeRow(i)
                    return
        rm.clicked.connect(_remove)
        for c, w in enumerate((combo, start, back, note, rm)):
            self.table.setCellWidget(r, c, w)

    def intervals(self) -> list[dict]:
        out = []
        for r in range(self.table.rowCount()):
            tag = self.table.cellWidget(r, 0).currentText().strip()
            start = from_qdatetime(self.table.cellWidget(r, 1).dateTime())
            back = self.table.cellWidget(r, 2)
            chk = back.findChild(QCheckBox)
            stop_edit = back.findChild(QDateTimeEdit)
            stop = from_qdatetime(stop_edit.dateTime()) if chk.isChecked() else ""
            note = self.table.cellWidget(r, 3).text().strip()
            if tag:
                out.append({"tag": tag, "start": start, "stop": stop, "note": note})
        return out

    def accept(self):
        bad = [i + 1 for i, iv in enumerate(self.intervals())
               if iv["stop"] and iv["stop"] <= iv["start"]]
        if bad:
            QMessageBox.warning(self, "Interval ends before it starts",
                                f"Row(s) {', '.join(map(str, bad))}: 'Back on' must "
                                "be later than 'Off from'.")
            return
        super().accept()


class IdentityDialog(QDialog):
    """Per-animal windows, found-dead handling and lost tags.

    Start / Stop follow the UWB tool's Auto / Manual rule: **Auto** applies
    the run-wide setting (trim to the metadata release / trap time, or not at
    all), **Manual** trims at exactly the time shown. Found dead:

    * **Last entry** - cut after the animal's last entry (a read on an
      entrance antenna after a gap longer than the bout gap). The time, if
      set, is when the body was found; entries after it are ignored.
    * **At time** - drop everything after the time shown.
    """

    COLS = ["Animal", "Sex", "Trial", "Tags", "Release \u2192 trap",
            "Read first \u2192 last", "Start", "Stop", "Found dead", "Lost tags", "Note"]
    PICK_FMT = "yyyy-MM-dd HH:mm:ss"

    def __init__(self, animals: pd.DataFrame, settings: dict, observed: dict | None = None,
                 parent=None):
        super().__init__(parent)
        self.setWindowTitle("Animals: windows, found dead, lost tags")
        self.setMinimumSize(1250, 650)
        self.animals = animals.reset_index(drop=True)
        self.settings = {a: animal_settings(settings.get(a)) for a in self.animals["animal"]}
        self.observed = observed or {}
        layout = QVBoxLayout(self)
        intro = QLabel(
            "Leave Start / Stop on <i>Auto</i> unless an animal has a real "
            "deployment or removal time that differs from the run-wide rule. "
            "Use <b>Found dead → Last entry</b> for a body left in the arena: "
            "it is read like a resident that never leaves. First/last read come "
            "from the last run (blank before one).")
        intro.setWordWrap(True)
        intro.setTextFormat(Qt.RichText)
        layout.addWidget(intro)

        self.table = QTableWidget(len(self.animals), len(self.COLS))
        self.table.setHorizontalHeaderLabels(self.COLS)
        self.table.verticalHeader().setVisible(False)
        hdr = self.table.horizontalHeader()
        hdr.setSectionResizeMode(QHeaderView.ResizeToContents)
        hdr.setSectionResizeMode(len(self.COLS) - 1, QHeaderView.Stretch)
        self.widgets = {}
        for r, row in self.animals.iterrows():
            self._fill_row(r, row)
        layout.addWidget(self.table, 1)

        box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        box.accepted.connect(self.accept)
        box.rejected.connect(self.reject)
        layout.addWidget(box)

    def _fill_row(self, r, row):
        animal = row["animal"]
        s = self.settings[animal]
        first, last = self.observed.get(animal, (None, None))
        tags = [t for t in str(row.get("tags", "")).split(";") if t]

        def span(a, b):
            a, b = pd.to_datetime(a, errors="coerce"), pd.to_datetime(b, errors="coerce")
            fa = "" if pd.isna(a) else a.strftime("%m-%d %H:%M")
            fb = "" if pd.isna(b) else b.strftime("%m-%d %H:%M")
            return f"{fa} \u2192 {fb}" if fa or fb else ""
        # tags differ in their last digits; the full numbers are in the tooltip
        cells = [(animal, ""), (row.get("sex", ""), ""), (row.get("trial", ""), ""),
                 (", ".join("\u2026" + t[-6:] for t in tags), "\n".join(tags)),
                 (span(row.get("release"), row.get("trap")),
                  f"release {fmt_time(row.get('release'))}\ntrap {fmt_time(row.get('trap'))}"),
                 (span(first, last), f"first {fmt_time(first)}\nlast {fmt_time(last)}")]
        for c, (text, tip) in enumerate(cells):
            item = QTableWidgetItem(str(text))
            item.setFlags(item.flags() & ~Qt.ItemIsEditable)
            if tip:
                item.setToolTip(tip)
            self.table.setItem(r, c, item)

        def bound(mode, value, fallback, modes):
            w = QWidget()
            h = QHBoxLayout(w)
            h.setContentsMargins(2, 0, 2, 0)
            combo = QComboBox()
            for label, data in modes:
                combo.addItem(label, data)
            combo.setCurrentIndex(max(0, combo.findData(mode)))
            edit = picker(value, to_qdatetime(fallback))
            edit.setDisplayFormat(self.PICK_FMT)
            edit.setEnabled(combo.currentData() not in ("auto", "none"))
            combo.currentIndexChanged.connect(
                lambda _i: edit.setEnabled(combo.currentData() not in ("auto", "none")))
            h.addWidget(combo)
            h.addWidget(edit)
            return w, combo, edit

        def first_known(*values):
            for v in values:
                if v is not None and not pd.isna(v):
                    return v
            return None
        # what a picker shows before anyone sets it: the metadata time, else
        # the data's own extent, else the release - never "now"
        start_ref = first_known(row.get("release"), first)
        stop_ref = first_known(row.get("trap"), last, row.get("release"))
        start_w, start_c, start_e = bound(s["start_mode"], s["start_time"], start_ref,
                                          [("Auto", "auto"), ("Manual", "manual")])
        stop_w, stop_c, stop_e = bound(s["stop_mode"], s["stop_time"], stop_ref,
                                       [("Auto", "auto"), ("Manual", "manual")])
        dead_w, dead_c, dead_e = bound(s["dead_mode"], s["dead_time"], stop_ref,
                                       [("No", "none"), ("Last entry", "last_entry"),
                                        ("At time", "manual")])
        # for 'last entry' the time is optional ("found at"): keep it editable
        dead_chk = QCheckBox("found at")
        dead_chk.setToolTip("Last entry: ignore entries after this time (when the "
                            "body was found). At time: the cut time.")
        dead_chk.setChecked(bool(s["dead_time"]) or s["dead_mode"] == "manual")
        dead_w.layout().insertWidget(1, dead_chk)

        def _sync_dead(*_):
            mode = dead_c.currentData()
            dead_chk.setEnabled(mode != "none")
            if mode == "manual":
                dead_chk.setChecked(True)
            dead_e.setEnabled(mode != "none" and dead_chk.isChecked())
        dead_c.currentIndexChanged.connect(_sync_dead)
        dead_chk.toggled.connect(_sync_dead)
        _sync_dead()

        lost_btn = QPushButton()
        lost_btn.clicked.connect(lambda _=False, a=animal: self._edit_lost(a))
        note = QLineEdit(s.get("note", ""))
        for c, w in ((6, start_w), (7, stop_w), (8, dead_w), (9, lost_btn), (10, note)):
            self.table.setCellWidget(r, c, w)
        self.widgets[animal] = {"start": (start_c, start_e), "stop": (stop_c, stop_e),
                                "dead": (dead_c, dead_e, dead_chk), "lost": lost_btn,
                                "note": note}
        self._label_lost(animal)

    def _label_lost(self, animal):
        n = len(self.settings[animal]["lost_tags"])
        btn = self.widgets[animal]["lost"]
        btn.setText(f"Lost tags ({n})…" if n else "Lost tags…")
        btn.setStyleSheet("color:#e5c07b; font-weight:bold;" if n else "")

    def _edit_lost(self, animal):
        row = self.animals[self.animals["animal"] == animal].iloc[0]
        tags = [t for t in str(row.get("tags", "")).split(";") if t]
        first, _last = self.observed.get(animal, (None, None))
        dlg = LostTagsDialog(animal, tags, self.settings[animal]["lost_tags"],
                             row.get("release") if not pd.isna(row.get("release")) else first,
                             self)
        if dlg.exec_() == QDialog.Accepted:
            self.settings[animal]["lost_tags"] = dlg.intervals()
            self._label_lost(animal)

    def result_settings(self) -> dict:
        out = {}
        for animal, w in self.widgets.items():
            s = dict(self.settings[animal])
            sc, se = w["start"]
            s["start_mode"] = sc.currentData()
            s["start_time"] = from_qdatetime(se.dateTime()) if s["start_mode"] == "manual" else ""
            tc, te = w["stop"]
            s["stop_mode"] = tc.currentData()
            s["stop_time"] = from_qdatetime(te.dateTime()) if s["stop_mode"] == "manual" else ""
            dc, de, dchk = w["dead"]
            s["dead_mode"] = dc.currentData()
            s["dead_time"] = (from_qdatetime(de.dateTime())
                              if s["dead_mode"] != "none" and dchk.isChecked() else "")
            s["note"] = w["note"].text().strip()
            out[animal] = animal_settings(s)
        return out
