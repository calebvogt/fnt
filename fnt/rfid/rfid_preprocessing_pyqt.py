"""RFID PreProcessing Tool (PyQt5).

Does for passive RFID (Biomark readers) what the UWB PreProcessing tool does
for UWB: load the raw downloads, tie tags to animals, describe the arena,
clean, check, and export analysis-ready tables.

Layout follows the other FNT tools: settings in a scrollable column on the
left, results on the right (a Summary tab plus one tab per QC report and
output table), and a session log under the settings. A run happens in a
background thread (:mod:`fnt.rfid.rfid_worker`); nothing is written until
every stage has succeeded, and nothing is ever written into a download
folder.

Standalone usage:
    python -m fnt.rfid.rfid_preprocessing_pyqt
"""

from __future__ import annotations

import os
import sys
from datetime import datetime

import pandas as pd
from PyQt5.QtCore import QDateTime, Qt, QTime, QUrl
from PyQt5.QtGui import QDesktopServices, QTextCursor
from PyQt5.QtWidgets import (QAbstractItemView, QApplication, QCheckBox, QComboBox,
                             QDateTimeEdit, QDialog, QDoubleSpinBox, QFileDialog,
                             QFormLayout, QGroupBox, QHBoxLayout, QLabel, QLineEdit,
                             QListWidget, QMenu, QMessageBox, QProgressBar,
                             QPushButton, QScrollArea, QSplitter, QTableView,
                             QTabWidget, QTextEdit, QTimeEdit, QVBoxLayout, QWidget)

from .config.config_manager import (ConfigManager, default_name, default_output_dir,
                                    resolve_output_dir)
from .config.defaults import (OPTIONAL_LAYERS, RFIDConfig, animal_settings,
                              get_fnt_version, is_default_animal)
from .core.file_readers import list_download_files, plan_twins
from .core.metadata import (detect_header_row, guess_mapping, header_names,
                            list_sheets, load_animals, read_grid)
from .rfid_dialogs import (ArenaDialog, DataFrameModel, IdentityDialog,
                           MetadataMappingDialog)
from .rfid_worker import RFIDWorker

# (tab title, table key) for the results pane, in display order. The QC
# reports come first: they are what to read before trusting the outputs.
RESULT_TABS = [
    ("Files", "qc_files"),
    ("Animals", "qc_animals"),
    ("Unknown tags", "qc_unknown_tags"),
    ("Unmapped antennas", "qc_unmapped_antennas"),
    ("Cross-paddock", "qc_cross_paddock"),
    ("Two places", "qc_two_places"),
    ("Stationary tags", "qc_stationary_tags"),
    ("Found dead", "qc_found_dead"),
    ("Dropped reads", "qc_dropped_reads"),
    ("Zone holders", "zone_holders"),
    ("Animal-days", "animal_days"),
    ("Co-occupancy", "co_occupancy"),
    ("Bouts", "bouts"),
]

LAYER_LABELS = {
    "gbi": "Group-by-individual matrix (GBI)",
    "edgelist": "Co-presence bouts + edgelist",
    "displacement": "Displacement events (male 1→n→1)",
    "hinde": "Hinde contact indices",
    "sna": "Social networks (needs python-igraph)",
}

POLICY_LABELS = [
    ("Keep as visits to the other paddock (flag away)", "keep"),
    ("Drop them", "drop"),
    ("Fold into the animal's own paddock (legacy)", "fold"),
]

STYLE = """
    QWidget { background-color: #2b2b2b; color: #cccccc; font-family: Arial; }
    QToolTip { background-color: #1e1e1e; color: #dddddd; border: 1px solid #0078d4;
               border-radius: 3px; padding: 4px 6px; font-weight: normal;
               font-size: 11px; }
    QLabel { color: #cccccc; background-color: transparent; }
    QPushButton { background-color: #0078d4; color: white; border: none;
                  padding: 5px 12px; border-radius: 4px; font-weight: bold;
                  min-width: 70px; }
    QPushButton:hover { background-color: #106ebe; }
    QPushButton:pressed { background-color: #005a9e; }
    QPushButton:disabled { background-color: #3f3f3f; color: #666666; }
    QGroupBox { background-color: #1e1e1e; border: 1px solid #3f3f3f;
                border-radius: 6px; margin-top: 12px; padding-top: 12px;
                font-weight: bold; color: #cccccc; }
    QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left;
                       padding: 4px 8px; background-color: #2b2b2b; color: #0078d4; }
    QComboBox, QLineEdit, QSpinBox, QDoubleSpinBox, QDateTimeEdit, QTimeEdit {
        background-color: #1e1e1e; color: #cccccc; border: 1px solid #3f3f3f;
        border-radius: 4px; padding: 3px 6px; }
    QComboBox:disabled, QLineEdit:disabled, QSpinBox:disabled,
    QDoubleSpinBox:disabled, QDateTimeEdit:disabled, QTimeEdit:disabled {
        color: #666666; }
    QListWidget, QTableView, QTableWidget, QTextEdit {
        background-color: #1e1e1e; color: #cccccc; border: 1px solid #3f3f3f;
        gridline-color: #3f3f3f; }
    QHeaderView::section { background-color: #333333; color: #cccccc; padding: 3px;
                           border: 1px solid #3f3f3f; }
    QTabBar::tab { background: #333333; color: #cccccc; padding: 5px 10px; }
    QTabBar::tab:selected { background: #0078d4; color: white; }
    QScrollArea { border: none; }
    QProgressBar { border: 1px solid #3f3f3f; border-radius: 4px; text-align: center; }
    QProgressBar::chunk { background-color: #0078d4; }
"""


class RFIDPreprocessingWindow(QWidget):
    """Main window of the RFID PreProcessing tool."""

    def __init__(self):
        super().__init__()
        self.config = RFIDConfig()
        self.animal_table = None          # metadata as last loaded, for the summaries
        self.last_result = None           # RunResult of the last run
        self.worker: RFIDWorker | None = None
        self._syncing = False
        self.initUI()
        self.apply_config(self.config)

    # ------------------------------------------------------------------ #
    # Layout
    # ------------------------------------------------------------------ #
    def initUI(self):
        self.setWindowTitle("RFID PreProcessing Tool")
        self.setGeometry(60, 60, 1500, 950)
        self.setStyleSheet(STYLE)
        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self.create_settings_panel())
        splitter.addWidget(self.create_results_panel())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([540, 960])
        layout = QVBoxLayout(self)
        layout.addWidget(splitter)

    def create_settings_panel(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)

        # --- configuration --------------------------------------------------
        grp = QGroupBox("Configuration")
        g = QVBoxLayout(grp)
        row = QHBoxLayout()
        for label, slot, tip in (
                ("New", self.new_config, "Start from an empty configuration."),
                ("Load Config...", self.load_config,
                 "Load a saved config, or a previous run's fnt_config.json."),
                ("Save Config...", self.save_config,
                 "Save every setting to a JSON file, e.g. to reuse the arena and "
                 "rules for the next dataset.")):
            b = QPushButton(label)
            b.setToolTip(tip)
            b.clicked.connect(slot)
            row.addWidget(b)
        g.addLayout(row)
        form = QFormLayout()
        self.edit_name = QLineEdit()
        self.edit_name.setPlaceholderText("from the download folder names")
        self.edit_name.setToolTip("Prefix of every output file and of the "
                                  "<name>_FNT_analysis folder.")
        self.edit_name.editingFinished.connect(self._update_output_label)
        form.addRow("Dataset name:", self.edit_name)
        g.addLayout(form)
        layout.addWidget(grp)

        # --- downloads --------------------------------------------------------
        grp = QGroupBox("Reader Downloads")
        g = QVBoxLayout(grp)
        self.list_dirs = QListWidget()
        self.list_dirs.setMaximumHeight(80)
        self.list_dirs.setToolTip("Folders holding Biomark 'Export All Records' files "
                                  "(.xlsx, .xls, .txt, .csv). They are only read.")
        g.addWidget(self.list_dirs)
        row = QHBoxLayout()
        b = QPushButton("Add Folder...")
        b.clicked.connect(self.add_download_dir)
        row.addWidget(b)
        b = QPushButton("Remove")
        b.clicked.connect(self.remove_download_dir)
        row.addWidget(b)
        self.chk_subfolders = QCheckBox("Include subfolders")
        self.chk_subfolders.toggled.connect(self._update_files_label)
        row.addWidget(self.chk_subfolders)
        row.addStretch()
        g.addLayout(row)
        self.lbl_files = self._note_label()
        g.addWidget(self.lbl_files)
        layout.addWidget(grp)

        # --- metadata ----------------------------------------------------------
        grp = QGroupBox("Animals (metadata)")
        g = QVBoxLayout(grp)
        row = QHBoxLayout()
        b = QPushButton("Choose File...")
        b.clicked.connect(self.choose_metadata)
        row.addWidget(b)
        row.addWidget(QLabel("Sheet:"))
        self.combo_sheet = QComboBox()
        self.combo_sheet.currentIndexChanged.connect(self._on_sheet_changed)
        row.addWidget(self.combo_sheet, 1)
        g.addLayout(row)
        self.lbl_metadata = self._note_label("No metadata file")
        g.addWidget(self.lbl_metadata)
        row = QHBoxLayout()
        b = QPushButton("Columns...")
        b.setToolTip("Header row and which column is name, sex, trial, tags, "
                     "release and trap.")
        b.clicked.connect(self.edit_metadata_mapping)
        row.addWidget(b)
        b = QPushButton("Animals && Tags...")
        b.setToolTip("Per-animal start/stop, found-dead handling and lost tags.")
        b.clicked.connect(self.edit_identities)
        row.addWidget(b)
        row.addStretch()
        g.addLayout(row)
        self.lbl_animals = self._note_label("")
        g.addWidget(self.lbl_animals)
        layout.addWidget(grp)

        # --- arena --------------------------------------------------------------
        grp = QGroupBox("Arena")
        g = QVBoxLayout(grp)
        row = QHBoxLayout()
        b = QPushButton("Edit Arena...")
        b.setToolTip("Paddocks and readers, zones, and the antenna table; presets "
                     "and CSV import/export.")
        b.clicked.connect(self.edit_arena)
        row.addWidget(b)
        row.addStretch()
        g.addLayout(row)
        self.lbl_arena = self._note_label("")
        g.addWidget(self.lbl_arena)
        layout.addWidget(grp)

        # --- time -----------------------------------------------------------------
        grp = QGroupBox("Time")
        form = QFormLayout(grp)
        self.combo_tz = QComboBox()
        self.combo_tz.setEditable(True)
        self.combo_tz.addItem("")
        try:
            import pytz
            self.combo_tz.addItems(pytz.common_timezones)
        except ImportError:                                # pragma: no cover
            pass
        self.combo_tz.setToolTip(
            "Readers log local wall-clock time. With a timezone, gaps and durations "
            "are measured correctly across a DST change. Leave blank if the reader "
            "clock never changes (e.g. left on standard time all year).")
        form.addRow("Timezone:", self.combo_tz)
        self.combo_date_order = QComboBox()
        self.combo_date_order.addItem("Month/Day/Year (Biomark default)", "MDY")
        self.combo_date_order.addItem("Day/Month/Year", "DMY")
        form.addRow("Scan dates:", self.combo_date_order)
        row = QHBoxLayout()
        self.chk_auto_origin = QCheckBox("Auto")
        self.chk_auto_origin.setToolTip(
            "Auto: the day-start time on the date of the earliest release in the "
            "metadata (else of the first read).")
        self.chk_auto_origin.toggled.connect(self._sync_origin_enabled)
        row.addWidget(self.chk_auto_origin)
        self.edit_origin = QDateTimeEdit()
        self.edit_origin.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.edit_origin.setCalendarPopup(True)
        row.addWidget(self.edit_origin, 1)
        form.addRow("Day 1 starts:", row)
        self.edit_day_start = QTimeEdit()
        self.edit_day_start.setDisplayFormat("HH:mm:ss")
        self.edit_day_start.setToolTip("Days run from this time to the same time "
                                       "next day (noon-to-noon by default).")
        form.addRow("Day boundary:", self.edit_day_start)
        layout.addWidget(grp)

        # --- cleaning -----------------------------------------------------------
        grp = QGroupBox("Cleaning")
        form = QFormLayout(grp)
        self.combo_policy = QComboBox()
        for label, data in POLICY_LABELS:
            self.combo_policy.addItem(label, data)
        self.combo_policy.setToolTip(
            "Reads of an animal by another paddock's reader. They are real visits "
            "(e.g. through a hole in a shared wall), so the default keeps them in "
            "the paddock that read them.")
        form.addRow("Other paddock's reader:", self.combo_policy)
        self.edit_entrance = QLineEdit()
        self.edit_entrance.setToolTip(
            "Antenna locations that count as an entrance (comma-separated). An "
            "entry = a read on one of these after a gap longer than the bout gap; "
            "the found-dead rule and the stationary-tag report use it.")
        form.addRow("Entrance locations:", self.edit_entrance)
        self.chk_release = QCheckBox("Trim each animal to its metadata release time")
        self.chk_trap = QCheckBox("Trim each animal to its metadata trap time")
        form.addRow(self.chk_release)
        form.addRow(self.chk_trap)
        self.spin_two_place = self._dspin(0.5, 1.0, 0.05, 2)
        self.spin_two_place.setToolTip("Two-places report: each tag's share of its "
                                       "day's reads in its top zone must exceed this.")
        form.addRow("Two-places share:", self.spin_two_place)
        self.spin_stationary = self._dspin(0.1, 1.0, 0.05, 2)
        self.spin_stationary.setToolTip("Stationary-tag report: read in more than "
                                        "this fraction of a day's 24 hours.")
        form.addRow("Stationary hours fraction:", self.spin_stationary)
        layout.addWidget(grp)

        # --- bouts and summaries -----------------------------------------------
        grp = QGroupBox("Bouts && Summaries")
        form = QFormLayout(grp)
        self.spin_gap = self._dspin(1, 3600, 1, 1, " s")
        self.spin_gap.setToolTip("Reads of one animal in one zone no further apart "
                                 "than this form one bout.")
        form.addRow("Bout gap:", self.spin_gap)
        self.spin_min_bout = self._dspin(0, 60, 0.5, 1, " s")
        self.spin_min_bout.setToolTip("Minimum bout duration: a lone read is a bout "
                                      "this long.")
        form.addRow("Minimum bout:", self.spin_min_bout)
        self.combo_rule = QComboBox()
        self.combo_rule.addItem("Gap / zone change / day boundary", "gap_zone_day")
        self.combo_rule.addItem("Legacy R pipeline (reproduce old numbers)", "legacy_r")
        form.addRow("Bout rule:", self.combo_rule)
        self.combo_resolution = QComboBox()
        self.combo_resolution.addItem("Milliseconds", "ms")
        self.combo_resolution.addItem("Whole seconds (legacy)", "s")
        form.addRow("Time resolution:", self.combo_resolution)
        self.spin_holder = self._dspin(0.05, 0.95, 0.05, 2)
        self.spin_holder.setToolTip("A zone is held on a day by the animal with more "
                                    "than this share of that zone's reads of the "
                                    "holder sex; otherwise it is contested.")
        form.addRow("Holder share >", self.spin_holder)
        self.combo_holder_sex = QComboBox()
        self.combo_holder_sex.addItem("Males", "M")
        self.combo_holder_sex.addItem("Females", "F")
        self.combo_holder_sex.addItem("Any sex", "")
        form.addRow("Holders among:", self.combo_holder_sex)
        self.chk_write_reads = QCheckBox("Write the cleaned reads table (large)")
        form.addRow(self.chk_write_reads)
        layers = QGroupBox("Optional layers (legacy, off by default)")
        lv = QVBoxLayout(layers)
        self.layer_checks = {}
        for key in OPTIONAL_LAYERS:
            chk = QCheckBox(LAYER_LABELS.get(key, key))
            self.layer_checks[key] = chk
            lv.addWidget(chk)
        form.addRow(layers)
        layout.addWidget(grp)

        # --- output and run -------------------------------------------------------
        grp = QGroupBox("Output")
        g = QVBoxLayout(grp)
        row = QHBoxLayout()
        self.edit_output = QLineEdit()
        self.edit_output.setToolTip("Blank: <name>_FNT_analysis beside the download "
                                    "folders. Never inside a download folder.")
        self.edit_output.editingFinished.connect(self._update_output_label)
        row.addWidget(self.edit_output, 1)
        b = QPushButton("Browse...")
        b.clicked.connect(self.choose_output_dir)
        row.addWidget(b)
        g.addLayout(row)
        self.lbl_output = self._note_label("")
        g.addWidget(self.lbl_output)
        row = QHBoxLayout()
        self.btn_check = QPushButton("Check (no export)")
        self.btn_check.setToolTip("Run everything and fill the QC tabs, but write "
                                  "nothing.")
        self.btn_check.clicked.connect(lambda: self.start_run(write=False))
        row.addWidget(self.btn_check)
        self.btn_run = QPushButton("Run and Export")
        self.btn_run.setStyleSheet("QPushButton { background-color: #2ea043; }"
                                   "QPushButton:disabled { background-color: #3f3f3f; }")
        self.btn_run.clicked.connect(lambda: self.start_run(write=True))
        row.addWidget(self.btn_run)
        self.btn_cancel = QPushButton("Cancel")
        self.btn_cancel.setEnabled(False)
        self.btn_cancel.clicked.connect(self.cancel_run)
        row.addWidget(self.btn_cancel)
        g.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setValue(0)
        g.addWidget(self.progress)
        self.btn_open = QPushButton("Open Analysis Folder")
        self.btn_open.clicked.connect(self.open_output_dir)
        g.addWidget(self.btn_open)
        layout.addWidget(grp)

        # --- log --------------------------------------------------------------------
        layout.addWidget(QLabel("Session Logs:"))
        self.txt_messages = QTextEdit()
        self.txt_messages.setReadOnly(True)
        self.txt_messages.setMinimumHeight(140)
        self.txt_messages.setStyleSheet("QTextEdit { font-family: monospace; "
                                        "font-size: 9px; }")
        layout.addWidget(self.txt_messages)
        b = QPushButton("Copy Session Logs to Clipboard")
        b.clicked.connect(self.copy_session_logs)
        layout.addWidget(b)
        layout.addStretch()

        # Long combo texts and paths must not widen the column: it scrolls
        # vertically only, as in the UWB tool.
        for combo in panel.findChildren(QComboBox):
            combo.setSizeAdjustPolicy(QComboBox.AdjustToMinimumContentsLengthWithIcon)
            combo.setMinimumContentsLength(10)
        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(470)
        return scroll

    def create_results_panel(self):
        self.tabs = QTabWidget()
        self.txt_summary = QTextEdit()
        self.txt_summary.setReadOnly(True)
        self.txt_summary.setHtml(self._welcome_html())
        self.tabs.addTab(self.txt_summary, "Summary")
        self.models = {}
        self.views = {}
        self.tab_index = {}
        for title, key in RESULT_TABS:
            view = QTableView()
            model = DataFrameModel()
            view.setModel(model)
            view.setSortingEnabled(True)
            view.setSelectionBehavior(QAbstractItemView.SelectRows)
            view.setAlternatingRowColors(False)
            if key in ("qc_two_places", "qc_stationary_tags"):
                view.setContextMenuPolicy(Qt.CustomContextMenu)
                view.customContextMenuRequested.connect(
                    lambda pos, v=view, k=key: self._qc_menu(v, k, pos))
                view.setToolTip("Right-click a row to mark a lost tag or a "
                                "found-dead animal from it.")
            self.models[key] = model
            self.views[key] = view
            self.tab_index[key] = self.tabs.addTab(view, title)
        return self.tabs

    @staticmethod
    def breakable(path: str) -> str:
        """A path that a word-wrapped label can break after each separator.

        QLabel wraps only at break opportunities, and a path has none, so a
        long one would force the whole settings column wider.
        """
        return str(path).replace("/", "/\u200b").replace("\\", "\\\u200b")

    @staticmethod
    def _note_label(text=""):
        lbl = QLabel(text)
        lbl.setWordWrap(True)
        lbl.setTextFormat(Qt.RichText)
        lbl.setStyleSheet("color: #9aa7b4; font-size: 10px;")
        return lbl

    @staticmethod
    def _dspin(lo, hi, step, decimals, suffix=""):
        s = QDoubleSpinBox()
        s.setRange(lo, hi)
        s.setSingleStep(step)
        s.setDecimals(decimals)
        if suffix:
            s.setSuffix(suffix)
        return s

    @staticmethod
    def _welcome_html():
        return ("<h3>RFID PreProcessing</h3>"
                "<ol><li>Add the folder(s) of reader downloads.</li>"
                "<li>Choose the metadata file and check its columns.</li>"
                "<li>Set up the arena (a preset is the quickest start).</li>"
                "<li><b>Check</b> to see the QC reports, set lost tags and "
                "found-dead animals under <i>Animals &amp; Tags</i>, then "
                "<b>Run and Export</b>.</li></ol>"
                "<p>Outputs go to <code>&lt;name&gt;_FNT_analysis</code> beside the "
                "download folders: <code>csvs/</code>, <code>fnt_config.json</code> "
                "(every setting plus provenance) and the message log.</p>")

    # ------------------------------------------------------------------ #
    # Log
    # ------------------------------------------------------------------ #
    def log_message(self, message: str):
        timestamp = datetime.now().strftime("%H:%M:%S")
        text = str(message)
        # consecutive "Reading downloads... n/N" ticks overwrite each other
        if text.startswith("Reading downloads...") and \
                getattr(self, "_last_log", "").startswith("Reading downloads..."):
            cursor = self.txt_messages.textCursor()
            cursor.movePosition(QTextCursor.End)
            cursor.select(QTextCursor.BlockUnderCursor)
            cursor.removeSelectedText()
        self._last_log = text
        self.txt_messages.append(f"[{timestamp}] {text}")
        bar = self.txt_messages.verticalScrollBar()
        bar.setValue(bar.maximum())

    def copy_session_logs(self):
        text = self.txt_messages.toPlainText()
        if not text.strip():
            return
        self.log_message("Copied session logs to clipboard")
        QApplication.clipboard().setText(self.txt_messages.toPlainText())

    # ------------------------------------------------------------------ #
    # Config <-> widgets
    # ------------------------------------------------------------------ #
    def apply_config(self, cfg: RFIDConfig):
        """Show a config in the widgets (the config itself is kept as is)."""
        self._syncing = True
        try:
            self.config = cfg
            self.edit_name.setText(cfg.name)
            self.list_dirs.clear()
            self.list_dirs.addItems(cfg.download_dirs)
            self.chk_subfolders.setChecked(cfg.include_subfolders)
            self.combo_tz.setCurrentText(cfg.timezone)
            self.combo_date_order.setCurrentIndex(
                max(0, self.combo_date_order.findData(cfg.date_order)))
            self.chk_auto_origin.setChecked(not cfg.day_origin)
            origin = pd.to_datetime(cfg.day_origin or None, errors="coerce")
            if pd.isna(origin):
                origin = pd.Timestamp(datetime.now().date()) + pd.Timedelta(hours=12)
            self.edit_origin.setDateTime(QDateTime.fromString(
                origin.strftime("%Y-%m-%d %H:%M:%S"), "yyyy-MM-dd HH:mm:ss"))
            self.edit_day_start.setTime(QTime.fromString(cfg.day_start_time or "12:00:00",
                                                         "HH:mm:ss"))
            self.combo_policy.setCurrentIndex(
                max(0, self.combo_policy.findData(cfg.cross_paddock_policy)))
            self.edit_entrance.setText(", ".join(cfg.entrance_locations))
            self.chk_release.setChecked(cfg.trim_to_release)
            self.chk_trap.setChecked(cfg.trim_to_trap)
            self.spin_two_place.setValue(cfg.two_place_share)
            self.spin_stationary.setValue(cfg.stationary_hour_fraction)
            self.spin_gap.setValue(cfg.bout_gap_s)
            self.spin_min_bout.setValue(cfg.min_bout_s)
            self.combo_rule.setCurrentIndex(max(0, self.combo_rule.findData(cfg.bout_rule)))
            self.combo_resolution.setCurrentIndex(
                max(0, self.combo_resolution.findData(cfg.time_resolution)))
            self.spin_holder.setValue(cfg.holder_share)
            self.combo_holder_sex.setCurrentIndex(
                max(0, self.combo_holder_sex.findData(cfg.holder_sex)))
            self.chk_write_reads.setChecked(cfg.write_reads)
            for key, chk in self.layer_checks.items():
                chk.setChecked(bool(cfg.layers.get(key)))
            self.edit_output.setText(cfg.output_dir)
            self._populate_sheets()
        finally:
            self._syncing = False
        self._sync_origin_enabled()
        self._update_files_label()
        self._update_arena_label()
        self._update_output_label()
        self.refresh_animals(quiet=True)

    def collect_config(self) -> RFIDConfig:
        """Read the widgets back into ``self.config`` and return it."""
        cfg = self.config
        cfg.name = self.edit_name.text().strip()
        cfg.download_dirs = [self.list_dirs.item(i).text()
                             for i in range(self.list_dirs.count())]
        cfg.include_subfolders = self.chk_subfolders.isChecked()
        cfg.timezone = self.combo_tz.currentText().strip()
        cfg.date_order = self.combo_date_order.currentData()
        cfg.day_origin = ("" if self.chk_auto_origin.isChecked() else
                          self.edit_origin.dateTime().toString("yyyy-MM-dd HH:mm:ss"))
        cfg.day_start_time = self.edit_day_start.time().toString("HH:mm:ss")
        cfg.cross_paddock_policy = self.combo_policy.currentData()
        cfg.entrance_locations = [x.strip() for x in self.edit_entrance.text().split(",")
                                  if x.strip()]
        cfg.trim_to_release = self.chk_release.isChecked()
        cfg.trim_to_trap = self.chk_trap.isChecked()
        cfg.two_place_share = self.spin_two_place.value()
        cfg.stationary_hour_fraction = self.spin_stationary.value()
        cfg.bout_gap_s = self.spin_gap.value()
        cfg.min_bout_s = self.spin_min_bout.value()
        cfg.bout_rule = self.combo_rule.currentData()
        cfg.time_resolution = self.combo_resolution.currentData()
        cfg.holder_share = self.spin_holder.value()
        cfg.holder_sex = self.combo_holder_sex.currentData()
        cfg.write_reads = self.chk_write_reads.isChecked()
        cfg.layers = {k: c.isChecked() for k, c in self.layer_checks.items()}
        cfg.output_dir = self.edit_output.text().strip()
        return cfg

    def _sync_origin_enabled(self, *_):
        self.edit_origin.setEnabled(not self.chk_auto_origin.isChecked())

    # ------------------------------------------------------------------ #
    # Config files
    # ------------------------------------------------------------------ #
    def new_config(self):
        if self.worker is not None:
            return
        self.animal_table = None
        self.apply_config(RFIDConfig())
        self.log_message("New configuration")

    def load_config(self):
        path, _ = QFileDialog.getOpenFileName(self, "Load RFID config", "",
                                              "JSON files (*.json);;All files (*)")
        if not path:
            return
        try:
            cfg = ConfigManager.load_from_file(path)
        except Exception as exc:                          # noqa: BLE001
            QMessageBox.warning(self, "Load failed", f"{path}\n\n{exc}")
            return
        self.apply_config(cfg)
        self.log_message(f"Loaded config {path}")

    def save_config(self):
        cfg = self.collect_config()
        start = os.path.join(resolve_output_dir(cfg) or "", "rfid_config.json") \
            if cfg.download_dirs else "rfid_config.json"
        path, _ = QFileDialog.getSaveFileName(self, "Save RFID config", start,
                                              "JSON files (*.json)")
        if not path:
            return
        try:
            ConfigManager.save_to_file(cfg, path)
            self.log_message(f"Saved config {path}")
        except Exception as exc:                          # noqa: BLE001
            QMessageBox.warning(self, "Save failed", str(exc))

    # ------------------------------------------------------------------ #
    # Downloads
    # ------------------------------------------------------------------ #
    def add_download_dir(self):
        start = self.config.download_dirs[-1] if self.config.download_dirs else ""
        folder = QFileDialog.getExistingDirectory(self, "Folder of reader downloads", start)
        if not folder:
            return
        if folder in self.config.download_dirs:
            return
        self.list_dirs.addItem(folder)
        self.collect_config()
        self._update_files_label()
        self._update_output_label()

    def remove_download_dir(self):
        for item in self.list_dirs.selectedItems():
            self.list_dirs.takeItem(self.list_dirs.row(item))
        self.collect_config()
        self._update_files_label()
        self._update_output_label()

    def _update_files_label(self, *_):
        if self._syncing:
            return
        dirs = [self.list_dirs.item(i).text() for i in range(self.list_dirs.count())]
        if not dirs:
            self.lbl_files.setText("No download folder")
            return
        paths = list_download_files(dirs, self.chk_subfolders.isChecked())
        by_ext = {}
        for p in paths:
            ext = os.path.splitext(p)[1].lower()
            by_ext[ext] = by_ext.get(ext, 0) + 1
        twins = sum(1 for v in plan_twins(paths).values() if v)
        parts = ", ".join(f"{n} {e}" for e, n in sorted(by_ext.items()))
        text = f"{len(paths)} candidate file(s): {parts or 'none'}"
        if twins:
            text += (f"<br>{twins} file(s) have a same-name spreadsheet twin; the "
                     f"spreadsheet is used and the twin checked for truncation")
        self.lbl_files.setText(text)

    # ------------------------------------------------------------------ #
    # Metadata
    # ------------------------------------------------------------------ #
    def choose_metadata(self):
        start = os.path.dirname(self.config.metadata.path) if self.config.metadata.path else ""
        path, _ = QFileDialog.getOpenFileName(
            self, "Animal metadata", start,
            "Spreadsheets (*.xlsx *.xlsm *.xls *.csv);;All files (*)")
        if not path:
            return
        self.set_metadata_file(path)

    def set_metadata_file(self, path: str, sheet: str = ""):
        """Point at a metadata file and guess its header row and columns."""
        spec = self.config.metadata
        spec.path = path
        try:
            sheets = list_sheets(path)
        except Exception as exc:                          # noqa: BLE001
            self.log_message(f"Could not open {path}: {exc}")
            sheets = []
        # prefer a sheet literally called "metadata" when there are several
        preferred = next((s for s in sheets if s.lower() == "metadata"), "")
        spec.sheet = sheet or preferred or (sheets[0] if sheets else "")
        spec.header_row = 0
        self._guess_mapping()
        self._populate_sheets()
        self.refresh_animals()

    def _guess_mapping(self):
        spec = self.config.metadata
        try:
            grid = read_grid(spec.path, spec.sheet, max_rows=60)
            headers = header_names(grid, detect_header_row(grid))
            spec.columns, spec.tag_columns = guess_mapping(headers)
        except Exception as exc:                          # noqa: BLE001
            self.log_message(f"Could not read {spec.path}: {exc}")

    def _populate_sheets(self):
        spec = self.config.metadata
        was = self._syncing
        self._syncing = True
        self.combo_sheet.clear()
        try:
            sheets = list_sheets(spec.path) if spec.path and os.path.isfile(spec.path) else []
        except Exception:                                  # noqa: BLE001
            sheets = []
        self.combo_sheet.addItems(sheets)
        if spec.sheet in sheets:
            self.combo_sheet.setCurrentText(spec.sheet)
        self.combo_sheet.setEnabled(bool(sheets))
        self._syncing = was
        self.lbl_metadata.setText(self.breakable(spec.path) if spec.path else "No metadata file")
        self.lbl_metadata.setToolTip(spec.path or "")

    def _on_sheet_changed(self, *_):
        if self._syncing:
            return
        self.config.metadata.sheet = self.combo_sheet.currentText()
        self.config.metadata.header_row = 0
        self._guess_mapping()
        self.refresh_animals()

    def edit_metadata_mapping(self):
        spec = self.config.metadata
        if not spec.path or not os.path.isfile(spec.path):
            QMessageBox.information(self, "Metadata", "Choose the metadata file first.")
            return
        try:
            dlg = MetadataMappingDialog(spec, self)
        except Exception as exc:                          # noqa: BLE001
            QMessageBox.warning(self, "Metadata", f"Could not read {spec.path}:\n{exc}")
            return
        if dlg.exec_() == QDialog.Accepted:
            self.config.metadata = dlg.result_spec()
            self.refresh_animals()

    def refresh_animals(self, quiet: bool = False):
        """Load the metadata with the current mapping, for the summary line."""
        spec = self.config.metadata
        self.animal_table = None
        if not spec.path or not os.path.isfile(spec.path) or not spec.columns.get("name"):
            self.lbl_animals.setText("")
            return
        try:
            table = load_animals(spec)
        except Exception as exc:                          # noqa: BLE001
            self.lbl_animals.setText(f"<span style='color:#e06c75'>&#9888; {exc}</span>")
            if not quiet:
                self.log_message(f"Metadata: {exc}")
            return
        self.animal_table = table
        a = table.animals
        sexes = ", ".join(f"{n} {s}" for s, n in a["sex"].value_counts().items())
        trials = ", ".join(sorted(set(a["trial"]) - {""}))
        names = set(a["animal"])
        custom = sum(1 for k, v in self.config.animals.items()
                     if k in names and not is_default_animal(v))
        text = (f"{len(a)} animals ({sexes}), {len(table.tags)} tags; header on row "
                f"{table.header_row}; tags from {', '.join(spec.tag_columns)}")
        if trials:
            text += f"<br>Trials: {trials}"
        if custom:
            text += f"<br>{custom} animal(s) with custom windows / cuts"
        for w in table.warnings:
            text += f"<br><span style='color:#e5c07b'>{w}</span>"
        self.lbl_animals.setText(text)
        self._update_arena_label()
        if not quiet:
            self.log_message(f"Metadata: {len(a)} animals, {len(table.tags)} tags")

    def edit_identities(self):
        if self.animal_table is None:
            self.refresh_animals()
        if self.animal_table is None:
            QMessageBox.information(self, "Animals", "Load the metadata (and check its "
                                    "columns) first.")
            return
        observed = {}
        if self.last_result is not None and "qc_animals" in self.last_result.tables:
            roster = self.last_result.tables["qc_animals"]
            observed = {a: (f, l) for a, f, l in
                        zip(roster["animal"], roster["first"], roster["last"])}
        dlg = IdentityDialog(self.animal_table.animals, self.config.animals, observed, self)
        if dlg.exec_() == QDialog.Accepted:
            self.config.animals = dlg.result_settings()
            self.refresh_animals(quiet=True)
            n = sum(1 for s in self.config.to_dict()["animals"])
            self.log_message(f"Animal settings updated ({n} with custom windows / cuts)")

    # ------------------------------------------------------------------ #
    # Arena
    # ------------------------------------------------------------------ #
    def edit_arena(self):
        dlg = ArenaDialog(self.config.arena, self)
        if dlg.exec_() == QDialog.Accepted:
            self.config.arena = dlg.arena()
            self._update_arena_label()
            self.log_message("Arena updated")

    def _update_arena_label(self):
        arena = self.config.arena
        parts = []
        for p in arena.paddocks:
            n_z = sum(1 for z in arena.zones if z.paddock_id == p.paddock_id)
            label = p.paddock_id + (f" ({p.name})" if p.name else "")
            readers = ", ".join(p.readers) or "no reader"
            trials = f"; home of {', '.join(p.trials)}" if p.trials else ""
            parts.append(f"{label}: reader {readers}, {n_z} zones{trials}")
        text = "<br>".join(parts) or "No arena yet - Edit Arena... and pick a preset"
        text += f"<br>{len(arena.antennas)} antenna(s)"
        blocking, warnings = arena.problems()
        if arena.paddocks or arena.antennas:
            for p in blocking:
                text += f"<br><span style='color:#e06c75'>&#9888; {p}</span>"
            for w in warnings:
                text += f"<br><span style='color:#e5c07b'>{w}</span>"
        if self.animal_table is not None and len(arena.paddocks) > 1:
            trials = set(self.animal_table.animals["trial"]) - {""}
            homeless = sorted(t for t in trials if not arena.home_paddock(t))
            if homeless:
                text += (f"<br><span style='color:#e5c07b'>Trial value(s) {homeless} "
                         f"match no paddock: add them to a paddock's Trials so "
                         f"visits to the other paddock are flagged</span>")
        self.lbl_arena.setText(text)

    # ------------------------------------------------------------------ #
    # Output
    # ------------------------------------------------------------------ #
    def choose_output_dir(self):
        cfg = self.collect_config()
        start = resolve_output_dir(cfg)
        folder = QFileDialog.getExistingDirectory(self, "Analysis folder",
                                                  os.path.dirname(start) if start else "")
        if folder:
            self.edit_output.setText(folder)
            self._update_output_label()

    def _update_output_label(self, *_):
        if self._syncing:
            return
        cfg = self.collect_config()
        if cfg.output_dir:
            self.lbl_output.setText(f"Outputs: {self.breakable(cfg.output_dir)}")
            return
        default = default_output_dir(cfg)
        self.edit_output.setPlaceholderText(default or "choose download folders first")
        self.lbl_output.setText(f"Outputs: {self.breakable(default)}" if default else "")

    def open_output_dir(self):
        path = (self.last_result.output_dir if self.last_result and
                self.last_result.output_dir else resolve_output_dir(self.collect_config()))
        if path and os.path.isdir(path):
            QDesktopServices.openUrl(QUrl.fromLocalFile(path))
        else:
            self.log_message(f"No analysis folder yet ({path or 'unset'})")

    # ------------------------------------------------------------------ #
    # Run
    # ------------------------------------------------------------------ #
    def start_run(self, write: bool = True):
        if self.worker is not None:
            return
        cfg = self.collect_config()
        if not cfg.name:
            cfg.name = default_name(cfg.download_dirs)
            self.edit_name.setText(cfg.name)
        blocking, _warnings = cfg.problems()
        if blocking:
            QMessageBox.warning(self, "Cannot run yet", "\n".join(blocking))
            return
        self.log_message(f"{'Run and export' if write else 'Check'} started "
                         f"(FNT {get_fnt_version()})")
        self.progress.setValue(0)
        self._set_running(True)
        self.worker = RFIDWorker(cfg, write=write, parent=self)
        self.worker.progress.connect(self._on_progress)
        self.worker.finished_run.connect(self._on_finished)
        self.worker.start()

    def cancel_run(self):
        if self.worker is not None:
            self.worker.cancel()
            self.log_message("Cancelling after the current step...")

    def _set_running(self, running: bool):
        self.btn_run.setEnabled(not running)
        self.btn_check.setEnabled(not running)
        self.btn_cancel.setEnabled(running)

    def _on_progress(self, percent: int, message: str):
        self.progress.setValue(percent)
        self.log_message(message)

    def _on_finished(self, result):
        worker, self.worker = self.worker, None
        if worker is not None:
            worker.wait(2000)
            worker.deleteLater()
        self._set_running(False)
        self.last_result = result
        if result.cancelled:
            self.progress.setValue(0)
            self.log_message("Run cancelled")
            return
        if result.error:
            self.progress.setValue(0)
            self.log_message(f"Run failed: {result.error}")
            self.txt_summary.setHtml(f"<h3>Run failed</h3><p>{result.error}</p>")
            return
        self.progress.setValue(100)
        self.show_result(result)
        if result.written:
            self.log_message(f"Exported {len(result.written)} table(s) to "
                             f"{result.output_dir}")

    def show_result(self, result):
        """Fill the Summary tab and every table tab from a finished run."""
        for title, key in RESULT_TABS:
            frame = result.tables.get(key)
            self.models[key].set_frame(frame if frame is not None else pd.DataFrame())
            n = 0 if frame is None else len(frame)
            self.tabs.setTabText(self.tab_index[key],
                                 f"{title} ({n})" if key.startswith("qc_") else title)
        self.txt_summary.setHtml(self.summary_html(result))
        self.tabs.setCurrentIndex(0)

    @staticmethod
    def summary_html(result) -> str:
        s = result.stats
        d = s.get("dropped", {})
        rows = [
            ("Files used", f"{s.get('files_used', 0)}"),
            ("Rows in those files", f"{s.get('raw_rows', 0):,}"),
            ("Unique reads", f"{s.get('unique_reads', 0):,}"),
            ("Animals / tags in metadata",
             f"{s.get('animals_in_metadata', 0)} / {s.get('tags_in_metadata', 0)}"),
            ("Unknown-tag reads dropped",
             f"{d.get('unknown_tag', 0):,} ({s.get('unknown_tags', 0)} tags)"),
            ("Unmapped-antenna reads dropped", f"{d.get('unmapped_antenna', 0):,}"),
            ("Other paddock's reader", f"{s.get('away_reads', 0):,} kept as away visits"
             if result.config.cross_paddock_policy == "keep"
             else f"{d.get('cross_paddock', 0):,} dropped"
             if result.config.cross_paddock_policy == "drop" else "folded (legacy)"),
            ("Trimmed to windows", f"{d.get('before_start', 0) + d.get('after_stop', 0):,}"),
            ("Lost-tag reads dropped", f"{d.get('lost_tag', 0):,}"),
            ("Found-dead reads dropped", f"{d.get('found_dead', 0):,}"),
            ("Cleaned reads", f"{s.get('cleaned_reads', 0):,} "
                              f"({s.get('animals_read', 0)} animals)"),
            ("Bouts", f"{s.get('bouts', 0):,} ({result.config.bout_rule}, "
                      f"gap {result.config.bout_gap_s:g} s)"),
            ("Animal-hours in zones", f"{s.get('animal_hours_in_zones', 0):,.1f}"),
            ("Day 1 starts", f"{s.get('day_origin', '')}; days "
                             f"{s.get('first_day', '')}–{s.get('last_day', '')}"),
            ("Run time", f"{s.get('run_seconds', 0):.1f} s"),
        ]
        html = ["<h3>Run summary</h3><table cellpadding='3'>"]
        html += [f"<tr><td><b>{k}</b></td><td>{v}</td></tr>" for k, v in rows]
        html.append("</table>")
        if result.written:
            html.append(f"<p>Written to <code>{result.output_dir}</code></p>")
        else:
            html.append("<p><i>Check only - nothing was written.</i></p>")
        n_two = len(result.tables.get("qc_two_places", []))
        n_still = len(result.tables.get("qc_stationary_tags", []))
        if n_two or n_still:
            html.append("<h4>To look at</h4><ul>")
            if n_two:
                html.append(f"<li>{n_two} animal-day(s) with two tags in two places "
                            f"(<i>Two places</i> tab) - a lost tag?</li>")
            if n_still:
                html.append(f"<li>{n_still} tag-day(s) read by one antenna all day "
                            f"(<i>Stationary tags</i> tab) - a body or a dropped tag?"
                            f"</li>")
            html.append("</ul>")
        if result.warnings:
            html.append("<h4>Warnings</h4><ul>")
            html += [f"<li>{w}</li>" for w in result.warnings]
            html.append("</ul>")
        return "".join(html)

    # ------------------------------------------------------------------ #
    # Acting on the QC reports
    # ------------------------------------------------------------------ #
    def day_start(self, day: int):
        """Start of a numbered day in the last run (local wall clock)."""
        origin = self.last_result.origin if self.last_result else None
        if origin is None or pd.isna(origin):
            return None
        return origin + pd.Timedelta(days=int(day) - 1)

    def _qc_menu(self, view, key, pos):
        index = view.indexAt(pos)
        if not index.isValid():
            return
        row = self.models[key].frame().iloc[index.row()]
        animal, day = str(row["animal"]), int(row["day"])
        menu = QMenu(self)
        tags = ([(row["tag_a"], row["zone_a"]), (row["tag_b"], row["zone_b"])]
                if key == "qc_two_places" else [(row["tag"], row["zone_id"])])
        for tag, zone in tags:
            act = menu.addAction(f"Tag {tag} (in {zone}) lost from the start of day {day}")
            act.triggered.connect(lambda _=False, t=str(tag): self.mark_lost_tag(animal, t, day))
        if key == "qc_stationary_tags":
            act = menu.addAction(f"{animal} found dead: cut at last entry")
            act.triggered.connect(lambda _=False: self.mark_found_dead(animal))
        menu.exec_(view.viewport().mapToGlobal(pos))

    def mark_lost_tag(self, animal: str, tag: str, day: int):
        """Add a lost-tag interval from the start of ``day``, open-ended.

        The start of the day is a deliberately coarse choice: it is what the
        report can see. Refine it under Animals & Tags when the field notes
        say exactly when the tag came off.
        """
        start = self.day_start(day)
        if start is None:
            return
        s = animal_settings(self.config.animals.get(animal))
        entry = {"tag": tag, "start": start.strftime("%Y-%m-%d %H:%M:%S"), "stop": "",
                 "note": f"from QC: day {day}"}
        if any(t.get("tag") == tag for t in s["lost_tags"]):
            s["lost_tags"] = [entry if t.get("tag") == tag else t for t in s["lost_tags"]]
        else:
            s["lost_tags"].append(entry)
        self.config.animals[animal] = s
        self.refresh_animals(quiet=True)
        self.log_message(f"{animal}: tag {tag} marked lost from {entry['start']} - "
                         f"run again to apply")

    def mark_found_dead(self, animal: str):
        s = animal_settings(self.config.animals.get(animal))
        s["dead_mode"] = "last_entry"
        self.config.animals[animal] = s
        self.refresh_animals(quiet=True)
        self.log_message(f"{animal}: found dead, cut at last entry - run again to apply")

    def closeEvent(self, event):
        if self.worker is not None:
            self.worker.cancel()
            self.worker.wait(10000)
        super().closeEvent(event)


def main():
    app = QApplication.instance() or QApplication(sys.argv)
    try:
        from fnt.theme import apply_dark_theme
        apply_dark_theme(app)
    except Exception:                                      # pragma: no cover
        pass
    window = RFIDPreprocessingWindow()
    window.show()
    return app.exec_()


if __name__ == "__main__":
    sys.exit(main())
