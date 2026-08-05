import base64
import faulthandler
import io
import os
import re
from collections import OrderedDict
import sys
import sqlite3
import struct
import numpy as np
import pandas as pd
import pytz
import json
import shutil
import xml.etree.ElementTree as ET
from multiprocessing import Pool, cpu_count
import matplotlib
matplotlib.use('Qt5Agg')
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.patches import Polygon as MplPolygon
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg, NavigationToolbar2QT)
from scipy.signal import savgol_filter
from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout,
                             QPushButton, QLabel, QFileDialog, QMessageBox,
                             QGroupBox, QCheckBox, QScrollArea, QComboBox,
                             QSpinBox, QDoubleSpinBox, QFrame, QLineEdit,
                             QDialog, QDialogButtonBox, QFormLayout, QTableWidget,
                             QTableWidgetItem, QHeaderView, QTextEdit, QProgressBar,
                             QDateTimeEdit, QTreeWidget, QTreeWidgetItem,
                             QSplitter, QSlider, QProgressDialog, QGridLayout)
from PyQt5.QtCore import Qt, QThread, pyqtSignal, QTimer, QDateTime
from PyQt5.QtGui import QFont

from fnt.uwb.uwb_preview_canvas import (
    UWBPreview2D, UWBPreview3D, PreviewArena, fit_arena_to_data,
    BUILTIN_ARENAS, HAVE_GL as PREVIEW_HAVE_GL, GL_ERROR as PREVIEW_GL_ERROR)
from fnt.uwb import animation as uwb_animation


# Columns the processing pipeline actually consumes. Wiser tables carry ~20
# columns, most of them unused TEXT fields (zones, alias, groupnames,
# arenaname) that dominate DataFrame memory: reading all of them for one Echo
# tag cost 5.0 s / 387 MB versus 1.6 s / 37 MB for just these five.
#
# This applies only to the *processed* reads (preview and the smoothed /
# downsampled exports). The raw CSV export still does SELECT * so the
# full-fidelity dump of the database is preserved untouched.
PROCESSING_COLUMNS = ("shortid", "timestamp", "location_x", "location_y",
                      "battery_voltage")
# Without these the pipeline cannot run at all, so a table missing any of them
# falls back to SELECT * rather than failing on a missing column.
REQUIRED_COLUMNS = ("shortid", "timestamp", "location_x", "location_y")


def get_fnt_version():
    """Resolve the installed FNT version for stamping into export artifacts.

    Mirrors the main GUI's resolution order: read pyproject.toml when running
    from a source checkout, then fall back to installed package metadata (used
    by the PyInstaller build). Returns 'unknown' if neither is available so
    stamping never breaks an export.
    """
    try:
        import tomllib
        from pathlib import Path
        toml_path = Path(__file__).resolve().parent.parent.parent / 'pyproject.toml'
        if toml_path.exists():
            with open(toml_path, "rb") as f:
                data = tomllib.load(f)
            version = data.get('project', {}).get('version')
            if version:
                return version
    except Exception:
        pass
    try:
        from importlib.metadata import version
        return version("fnt")
    except Exception:
        return "unknown"


def list_visible_files(directory):
    """List filenames in ``directory``, excluding hidden and macOS sidecar files.

    Network/exFAT shares written from macOS accumulate AppleDouble companions
    ('._<name>') and '.DS_Store' entries alongside the real files. These are not
    configuration or data files and must never be auto-discovered — e.g.
    '._EchoConfiguration_2024.11.6.xml' would otherwise shadow the real
    'EchoConfiguration_2024.11.6.xml' (its leading '.' can even sort first, so a
    naive scan picks it). Filtering any name starting with '.' covers AppleDouble
    files, .DS_Store, and Unix dotfiles in one rule.

    Returns [] if the directory cannot be read.
    """
    try:
        return [f for f in os.listdir(directory) if not f.startswith('.')]
    except OSError:
        return []


# Default spatial context layers for exported plots/animation. The user picks
# these per-export in PlotLayersDialog; the choice is stored on the window and
# in the config. Kept separate from the Preview Options 'Show ...' toggles,
# which only affect the live preview.
DEFAULT_PLOT_LAYERS = {'background': True, 'zones': True, 'anchors': True}


def draw_context_layers(ax, layers, *, bg_image=None, bg_extent=None,
                        zones_xml=None, zones_df=None, anchors=None):
    """Draw the background image, zone polygons and anchor markers on ``ax``.

    Gated by ``layers`` (a dict with 'background'/'zones'/'anchors' booleans) so
    every exported spatial figure honours the same per-export choice. Sources
    are passed explicitly because the worker and the main window hold them in
    different forms (loaded PNG vs. XML map image; xml_zones list vs.
    arena_zones DataFrame).
    """
    layers = layers or {}
    if layers.get('background') and bg_image is not None and bg_extent is not None:
        ax.imshow(bg_image, extent=list(bg_extent), origin='upper',
                  aspect='auto', alpha=0.6, zorder=0)
    if layers.get('zones'):
        if zones_xml:
            for z in zones_xml:
                pts = z.get('points')
                if pts is None or len(pts) < 3:
                    continue
                is_bounds = (z.get('name', '') or '').strip().lower() == 'arena'
                ax.add_patch(MplPolygon(
                    pts, closed=True,
                    facecolor='none' if is_bounds else z.get('color', '#888'),
                    edgecolor=z.get('color', '#888'),
                    alpha=1.0 if is_bounds else 0.22,
                    linewidth=1.8 if is_bounds else 1.0, zorder=1))
        elif zones_df is not None and not zones_df.empty:
            for zone_name in zones_df['zone'].unique():
                coords = zones_df[zones_df['zone'] == zone_name][['x', 'y']].values
                if len(coords) >= 3:
                    ax.add_patch(MplPolygon(coords, closed=True, fill=False,
                                            edgecolor='black', linewidth=1.5,
                                            linestyle='--', zorder=1))
    if layers.get('anchors') and anchors:
        ax.scatter([a['x'] for a in anchors], [a['y'] for a in anchors],
                   marker='^', s=40, c='#f2c24f', edgecolors='none', zorder=2)


def processing_select_clause(conn, table_name):
    """SELECT list covering PROCESSING_COLUMNS present in ``table_name``.

    Intersects with the real schema so tables written by different Wiser
    versions still work, and degrades to ``*`` if anything essential is absent.
    """
    try:
        have = {row[1] for row in conn.execute(f"PRAGMA table_info({table_name})")}
    except Exception:
        return "*"
    if not have or any(c not in have for c in REQUIRED_COLUMNS):
        return "*"
    return ", ".join(c for c in PROCESSING_COLUMNS if c in have)


def forward_backward_ewma(series, span):
    """Zero-phase exponential weighted moving average (filtfilt-style cascade).

    The Wiser hardware applies a causal EWMA, which necessarily lags the true
    signal. Post-hoc we know each sample's future, so we run the EWMA forward,
    then run a second EWMA backward over that result. The two passes impose
    equal and opposite group delays, so the lag cancels exactly.

    Reversing with [::-1] changes row order but preserves index labels, so the
    final reversal restores the original ordering and the returned Series still
    aligns correctly when used inside groupby(...).transform(...).
    """
    forward = series.ewm(span=span, adjust=False).mean()
    return forward[::-1].ewm(span=span, adjust=False).mean()[::-1]


def rolling_smooth_xy(data, agg, window_value, time_based):
    """Centred (zero-phase) rolling smoothing of location_x/location_y per tag.

    Shared by the interactive preview/export path and the plot-worker fallback
    so both interpret the smoothing window identically.

    Parameters
    ----------
    data : DataFrame with 'shortid', 'Timestamp', 'location_x', 'location_y'.
    agg : 'mean' (Rolling Average) or 'median' (Rolling Median).
    window_value : the window size. When ``time_based`` it is a duration in
        seconds; otherwise it is a count of consecutive samples.
    time_based : if True the window is a real-time span (e.g. '30s') evaluated
        against each tag's Timestamp, so the degree of smoothing is independent
        of the (irregular, well-under-1 Hz) reporting rate — 30 s of data is
        averaged regardless of how many fixes landed in that interval. If False
        the window is a fixed number of consecutive fixes, so its wall-clock
        span drifts with the reporting rate (the historical behaviour).

    In both modes the window is **centred**, so each output sample uses roughly
    equal amounts of past and future data and smoothing adds no temporal lag.
    Writes 'smoothed_x'/'smoothed_y' back into ``data`` and returns it.
    """
    def _agg(roller):
        return roller.mean() if agg == 'mean' else roller.median()

    if time_based:
        # Time-based rolling needs a monotonic DatetimeIndex, so sort each tag
        # by Timestamp, roll, then scatter the result back to the original rows.
        win = f"{int(window_value)}s"
        data['smoothed_x'] = np.nan
        data['smoothed_y'] = np.nan
        for _, g in data.groupby('shortid'):
            g_sorted = g.sort_values('Timestamp')
            idx = pd.DatetimeIndex(g_sorted['Timestamp'])
            for src, dst in (('location_x', 'smoothed_x'), ('location_y', 'smoothed_y')):
                s = pd.Series(g_sorted[src].to_numpy(), index=idx)
                rolled = _agg(s.rolling(win, center=True, min_periods=1))
                data.loc[g_sorted.index, dst] = rolled.to_numpy()
    else:
        window_size = max(3, int(window_value))  # sample-count floor of 3
        for src, dst in (('location_x', 'smoothed_x'), ('location_y', 'smoothed_y')):
            data[dst] = data.groupby('shortid')[src].transform(
                lambda x: _agg(x.rolling(window=window_size, center=True, min_periods=1)))
    return data


# Shared, math-level description of every smoothing method. Used verbatim as
# the tooltip for BOTH the Export "Smoothing method" combo and the live-preview
# "Smoothing method" combo, so the two are guaranteed identical. Keep it in sync
# with forward_backward_ewma(), rolling_smooth_xy() and apply_smoothing_to_data().
SMOOTHING_METHODS_TOOLTIP = (
    "Smoothing is applied per tag, after threshold filtering. Every method here\n"
    "is symmetric (zero-phase): because this is post-processing, each sample's\n"
    "future is known, so the window is centred and smoothing adds no temporal\n"
    "lag — unlike the causal filter the Wiser hardware must run live.\n"
    "\n"
    "The Smoothing Window / units control sets the window for the two rolling\n"
    "methods; Forward-Backward Exponentially Weighted Moving Average uses it as a sample span; Savitzky-Golay and\n"
    "None ignore it.\n"
    "\n"
    "• None\n"
    "    No smoothing. The raw (already threshold-filtered) fixes are used as-is.\n"
    "\n"
    "• Rolling Average\n"
    "    Centred moving mean — each output is the arithmetic mean of the fixes\n"
    "    inside a window centred on that sample:\n"
    "        out[i] = mean( x[j] : t[j] within the window around t[i] )\n"
    "    With units = Seconds the window is a real-time span (e.g. 30 s), so the\n"
    "    amount of smoothing is independent of the irregular, well-under-1 Hz\n"
    "    reporting rate; with units = Samples it is a fixed count of consecutive\n"
    "    fixes (minimum 3). Partial windows at the ends use whatever samples\n"
    "    exist (min_periods = 1). Good general-purpose smoothing, but sensitive\n"
    "    to any outliers the thresholds missed.\n"
    "\n"
    "• Rolling Median\n"
    "    Identical windowing to Rolling Average but takes the median:\n"
    "        out[i] = median( x[j] within the window around t[i] )\n"
    "    The median ignores a minority of extreme values, so it is far more\n"
    "    robust to residual spikes/outliers, at the cost of slightly blockier\n"
    "    output on smoothly curving paths.\n"
    "\n"
    "• Savitzky-Golay\n"
    "    Fits a low-order polynomial to each centred window by least squares and\n"
    "    evaluates it at the window's centre. Here the window length is\n"
    "    min(31, n) samples (forced odd) with polynomial order 2 — a local\n"
    "    quadratic; the Smoothing Window setting does not apply. Because it\n"
    "    models curvature instead of flattening it, Savitzky-Golay preserves\n"
    "    peaks, turns and the height of sharp features better than averaging.\n"
    "\n"
    "• Forward-Backward Exponentially Weighted Moving Average\n"
    "    A zero-phase exponentially weighted moving average, run as a two-pass\n"
    "    (filtfilt-style) cascade. One pass is the recursive EWMA\n"
    "        y[t] = alpha * x[t] + (1 - alpha) * y[t-1],   alpha = 2 / (span + 1)\n"
    "    which weights the most recent sample most heavily and older samples with\n"
    "    geometrically decaying weight. A single forward pass lags the true path\n"
    "    by about (span - 1) / 2 samples; that forward result is then filtered\n"
    "    again backward, imposing an equal and opposite delay, so the lag cancels\n"
    "    exactly and the output stays centred on the real trajectory. Filtering\n"
    "    twice squares the frequency response, so expect noticeably stronger\n"
    "    smoothing than a single pass at the same span. The span is always in\n"
    "    samples. This is the same filter family the Wiser hardware applies in\n"
    "    real time (where it must stay causal, hence laggy); to match a hardware\n"
    "    filter value F, use span = 2F - 1."
)


class PreviewIndexBuilder(QThread):
    """Build an indexed *copy* of the database for fast preview scrubbing.

    The Wiser database is a primary record, so it is never modified. This
    creates a byte-identical duplicate in the analysis folder and adds one
    index on (shortid, timestamp) — no filtering, no schema changes, no row
    edits. The copy is a derived cache and is safe to delete at any time.

    Worth being precise about the payoff: the index accelerates the preview's
    narrow time-window reads by ~80x, but does **not** speed up export, which
    reads every row for a tag and is a full scan either way.
    """
    progress = pyqtSignal(str)
    # (source_db_path, indexed_copy_path) — the source is carried through so the
    # receiver can tell whether the database it was built for is still current.
    done = pyqtSignal(str, str)
    failed = pyqtSignal(str, str)   # (source_db_path, error)

    INDEX_NAME = "idx_fnt_shortid_ts"

    def __init__(self, src_path, table_name, dst_path, meta_path):
        super().__init__()
        self.src_path = src_path
        self.table_name = table_name
        self.dst_path = dst_path
        self.meta_path = meta_path

    def run(self):
        tmp = self.dst_path + ".partial"
        try:
            self.progress.emit("Copying database…")
            # Write to a temp name first so an interrupted build can never be
            # mistaken for a finished cache.
            if os.path.exists(tmp):
                os.remove(tmp)
            shutil.copy2(self.src_path, tmp)

            self.progress.emit("Building index…")
            conn = sqlite3.connect(tmp)
            conn.execute(f"CREATE INDEX IF NOT EXISTS {self.INDEX_NAME} "
                         f"ON {self.table_name}(shortid, timestamp)")
            conn.commit()
            conn.close()

            os.replace(tmp, self.dst_path)

            st = os.stat(self.src_path)
            from datetime import datetime
            with open(self.meta_path, "w") as f:
                json.dump({
                    "source_path": self.src_path,
                    "source_size": st.st_size,
                    "source_mtime": st.st_mtime,
                    "table_name": self.table_name,
                    "index_name": self.INDEX_NAME,
                    "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                    "note": ("Derived cache for preview scrubbing. Identical to "
                             "the source database plus one index on "
                             "(shortid, timestamp). No data was filtered or "
                             "altered. Safe to delete; it will be rebuilt on "
                             "demand."),
                }, f, indent=4)
            self.done.emit(self.src_path, self.dst_path)
        except Exception as e:
            for p in (tmp,):
                try:
                    if os.path.exists(p):
                        os.remove(p)
                except Exception:
                    pass
            self.failed.emit(self.src_path, str(e))


class PreviewChunkLoader(QThread):
    """Reads one bounded time slice off the UI thread.

    Only the SQL read happens here — measured at ~100% of chunk load cost on an
    unindexed 4.7M-row table — so the widget's filter/smoothing methods stay on
    the main thread where they can safely read their own spinboxes.

    All selected tags are fetched in a single query. One combined scan is ~2.6x
    faster than looping per tag, because without an index every query is a full
    table scan and the per-tag loop pays for one scan each.
    """
    loaded = pyqtSignal(int, object)    # chunk_index, DataFrame
    failed = pyqtSignal(int, str)

    def __init__(self, db_path, table_name, tags, start_ms, end_ms, chunk_index):
        super().__init__()
        self.db_path = db_path
        self.table_name = table_name
        self.tags = list(tags)
        self.start_ms = int(start_ms)
        self.end_ms = int(end_ms)
        self.chunk_index = int(chunk_index)

    def run(self):
        try:
            # The connection must be created in this thread; sqlite3 objects
            # cannot be shared across threads.
            conn = sqlite3.connect(self.db_path)
            cols = processing_select_clause(conn, self.table_name)
            placeholders = ",".join(["?"] * len(self.tags))
            df = pd.read_sql_query(
                f"SELECT {cols} FROM {self.table_name} "
                f"WHERE shortid IN ({placeholders}) AND timestamp BETWEEN ? AND ? "
                f"ORDER BY shortid, timestamp",
                conn, params=self.tags + [self.start_ms, self.end_ms])
            conn.close()
            self.loaded.emit(self.chunk_index, df)
        except Exception as e:
            self.failed.emit(self.chunk_index, str(e))


class ExportConflictDialog(QDialog):
    """Dialog shown when export would overwrite existing files.
    Shows categorized lists of conflicting and new files."""

    # Non-zero so none of these collide with QDialog.Rejected (0), which is
    # what the Cancel button and the window close (X) return via reject(). If
    # SKIP were 0 a cancel would be indistinguishable from "skip existing" and
    # the export would proceed anyway instead of halting.
    SKIP = 1
    OVERWRITE = 2
    NEW_FOLDER = 3

    def __init__(self, conflicting_files, new_files, parent=None):
        """
        Args:
            conflicting_files: list of (filename, subfolder) tuples for files that already exist
            new_files: list of (filename, subfolder) tuples for files that will be newly created
        """
        super().__init__(parent)
        self.setWindowTitle("Export Conflict")
        self.setModal(True)
        self.setMinimumWidth(600)
        self.setMinimumHeight(400)

        layout = QVBoxLayout()

        # Summary message
        num_conflicts = len(conflicting_files)
        num_new = len(new_files)
        total = num_conflicts + num_new
        message = QLabel(
            f"<b>{total}</b> file(s) will be produced. "
            f"<b>{num_conflicts}</b> existing file(s) would be overwritten, "
            f"and <b>{num_new}</b> file(s) are new."
        )
        message.setWordWrap(True)
        layout.addWidget(message)

        # File tree
        tree = QTreeWidget()
        tree.setHeaderHidden(True)
        tree.setRootIsDecorated(True)
        tree.setIndentation(20)
        tree.setStyleSheet("""
            QTreeWidget {
                background-color: #1e1e1e;
                border: 1px solid #3f3f3f;
                border-radius: 4px;
                color: #cccccc;
                font-size: 11px;
            }
            QTreeWidget::item {
                padding: 2px 0px;
            }
        """)

        # Conflicting files section
        if conflicting_files:
            conflict_root = QTreeWidgetItem(tree)
            conflict_root.setText(0, f"⚠ Files that would be overwritten ({num_conflicts})")
            conflict_root.setForeground(0, Qt.yellow)
            conflict_root.setExpanded(True)

            # Group by subfolder
            grouped = {}
            for fname, subfolder in conflicting_files:
                grouped.setdefault(subfolder, []).append(fname)

            for subfolder in sorted(grouped.keys()):
                if subfolder:
                    folder_item = QTreeWidgetItem(conflict_root)
                    folder_item.setText(0, f"📁 {subfolder}/")
                    folder_item.setExpanded(True)
                    parent_item = folder_item
                else:
                    parent_item = conflict_root

                for fname in sorted(grouped[subfolder]):
                    file_item = QTreeWidgetItem(parent_item)
                    file_item.setText(0, fname)

        # New files section
        if new_files:
            new_root = QTreeWidgetItem(tree)
            new_root.setText(0, f"✓ New files to be created ({num_new})")
            new_root.setForeground(0, Qt.green)
            new_root.setExpanded(True)

            # Group by subfolder
            grouped = {}
            for fname, subfolder in new_files:
                grouped.setdefault(subfolder, []).append(fname)

            for subfolder in sorted(grouped.keys()):
                if subfolder:
                    folder_item = QTreeWidgetItem(new_root)
                    folder_item.setText(0, f"📁 {subfolder}/")
                    folder_item.setExpanded(True)
                    parent_item = folder_item
                else:
                    parent_item = new_root

                for fname in sorted(grouped[subfolder]):
                    file_item = QTreeWidgetItem(parent_item)
                    file_item.setText(0, fname)

        layout.addWidget(tree)

        # Question
        question = QLabel("What would you like to do?")
        question.setStyleSheet("font-weight: bold; margin-top: 6px;")
        layout.addWidget(question)

        # Buttons
        btn_layout = QHBoxLayout()

        skip_btn = QPushButton("Skip Existing")
        skip_btn.setToolTip("Only write the new files; leave existing files untouched")
        skip_btn.clicked.connect(lambda: self.done(self.SKIP))
        btn_layout.addWidget(skip_btn)

        overwrite_btn = QPushButton("Overwrite")
        overwrite_btn.setToolTip("Replace all conflicting files and write new files")
        overwrite_btn.clicked.connect(lambda: self.done(self.OVERWRITE))
        btn_layout.addWidget(overwrite_btn)

        new_folder_btn = QPushButton("New Folder")
        new_folder_btn.setToolTip("Create a new timestamped analysis folder instead")
        new_folder_btn.clicked.connect(lambda: self.done(self.NEW_FOLDER))
        btn_layout.addWidget(new_folder_btn)

        cancel_btn = QPushButton("Cancel")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)

        layout.addLayout(btn_layout)
        self.setLayout(layout)


class PlotLayersDialog(QDialog):
    """Choose which spatial context layers exported plots/animation include.

    Shown on export when plots or an animation are requested and at least one
    layer (background image / XML zones / anchors) is available. Each toggle is
    independent — unchecking all draws tag positions only. Options with no data
    are disabled. OK applies the choice; Cancel aborts the export.
    """

    def __init__(self, has_background, has_zones, has_anchors, defaults=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Plot & Animation Layers")
        self.setModal(True)
        defaults = defaults or DEFAULT_PLOT_LAYERS

        layout = QVBoxLayout()
        info = QLabel(
            "Choose the spatial context to draw under the tag trajectories in "
            "exported plots and animations. Unavailable layers are greyed out; "
            "with none checked, only the tag positions are drawn.")
        info.setWordWrap(True)
        layout.addWidget(info)

        def _make(text, available, default, tip):
            cb = QCheckBox(text)
            cb.setEnabled(available)
            cb.setChecked(bool(default) and available)
            if not available:
                cb.setToolTip("Not available for this dataset")
            else:
                cb.setToolTip(tip)
            layout.addWidget(cb)
            return cb

        self.chk_background = _make(
            "Background image", has_background, defaults.get('background', True),
            "Draw the loaded floorplan / site-map image beneath the tracks.")
        self.chk_zones = _make(
            "Zones (from XML)", has_zones, defaults.get('zones', True),
            "Draw the surveyed zone polygons parsed from the site XML.")
        self.chk_anchors = _make(
            "Anchor positions", has_anchors, defaults.get('anchors', True),
            "Draw the UWB anchor/antenna positions as triangles.")

        btn_layout = QHBoxLayout()
        btn_layout.addStretch()
        ok_btn = QPushButton("OK")
        ok_btn.setToolTip("Use these layers for this export")
        ok_btn.clicked.connect(self.accept)
        btn_layout.addWidget(ok_btn)
        cancel_btn = QPushButton("Cancel")
        cancel_btn.setToolTip("Cancel the export")
        cancel_btn.clicked.connect(self.reject)
        btn_layout.addWidget(cancel_btn)
        layout.addLayout(btn_layout)
        self.setLayout(layout)

    def get_layers(self):
        return {
            'background': self.chk_background.isChecked(),
            'zones': self.chk_zones.isChecked(),
            'anchors': self.chk_anchors.isChecked(),
        }


class IdentityAssignmentDialog(QDialog):
    """Dialog for assigning sex, custom identities, and active time windows to tags"""
    def __init__(self, available_tags, existing_identities=None, tag_time_ranges=None, parent=None):
        super().__init__(parent)
        self.available_tags = available_tags
        self.identities = existing_identities if existing_identities else {}
        self.tag_time_ranges = tag_time_ranges if tag_time_ranges else {}
        self.initUI()

    def initUI(self):
        self.setWindowTitle("Assign Tag Identities")
        self.setMinimumWidth(700)

        layout = QVBoxLayout()

        # Instructions
        instructions = QLabel(
            "Assign sex (M/F), IDs, and active time windows. "
            "To merge tags (e.g., lost tag replaced), assign the same ID to both tags."
        )
        instructions.setWordWrap(True)
        layout.addWidget(instructions)

        # Form for each tag — placed inside a scroll area so a long tag list
        # never pushes the OK/Cancel row off-screen (the row stays fixed below).
        form_layout = QFormLayout()
        self.sex_combos = {}
        self.identity_edits = {}
        self.start_edits = {}
        self.stop_edits = {}

        for tag in sorted(self.available_tags):
            # Sex selection
            sex_combo = QComboBox()
            sex_combo.addItems(["M", "F"])

            # Identity text input
            identity_edit = QLineEdit()
            identity_edit.setPlaceholderText(f"e.g., {tag}")

            # Start/Stop time pickers
            start_edit = QDateTimeEdit()
            start_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
            start_edit.setCalendarPopup(True)
            stop_edit = QDateTimeEdit()
            stop_edit.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
            stop_edit.setCalendarPopup(True)

            # Set defaults from tag time ranges
            if tag in self.tag_time_ranges:
                tr = self.tag_time_ranges[tag]
                start_edit.setDateTime(QDateTime.fromString(tr['start'], "yyyy-MM-dd HH:mm:ss"))
                stop_edit.setDateTime(QDateTime.fromString(tr['end'], "yyyy-MM-dd HH:mm:ss"))
                # Set min/max to observed range
                start_edit.setMinimumDateTime(QDateTime.fromString(tr['start'], "yyyy-MM-dd HH:mm:ss"))
                stop_edit.setMaximumDateTime(QDateTime.fromString(tr['end'], "yyyy-MM-dd HH:mm:ss"))

            # Load existing identity values if available
            if tag in self.identities:
                sex_idx = 0 if self.identities[tag].get('sex', 'M') == 'M' else 1
                sex_combo.setCurrentIndex(sex_idx)
                identity_edit.setText(self.identities[tag].get('identity', ''))
                # Restore saved start/stop times
                if 'start_time' in self.identities[tag]:
                    dt = QDateTime.fromString(self.identities[tag]['start_time'], "yyyy-MM-dd HH:mm:ss")
                    if dt.isValid():
                        start_edit.setDateTime(dt)
                if 'stop_time' in self.identities[tag]:
                    dt = QDateTime.fromString(self.identities[tag]['stop_time'], "yyyy-MM-dd HH:mm:ss")
                    if dt.isValid():
                        stop_edit.setDateTime(dt)
            else:
                sex_combo.setCurrentIndex(-1)  # No default selection
                identity_edit.setText("")  # Blank until user configures

            # Layout: Sex + ID on first row, Start/Stop on second row
            tag_widget = QWidget()
            tag_vlayout = QVBoxLayout()
            tag_vlayout.setContentsMargins(0, 0, 0, 0)

            row1 = QHBoxLayout()
            row1.addWidget(QLabel("Sex:"))
            row1.addWidget(sex_combo)
            row1.addWidget(QLabel("ID:"))
            row1.addWidget(identity_edit)
            tag_vlayout.addLayout(row1)

            row2 = QHBoxLayout()
            start_label = QLabel("Start:")
            start_label.setFixedWidth(35)
            row2.addWidget(start_label)
            row2.addWidget(start_edit, 1)
            row2.addSpacing(10)
            stop_label = QLabel("Stop:")
            stop_label.setFixedWidth(35)
            row2.addWidget(stop_label)
            row2.addWidget(stop_edit, 1)
            tag_vlayout.addLayout(row2)

            tag_widget.setLayout(tag_vlayout)

            # Convert DEC to HEX for display
            hex_id = hex(tag).upper().replace('0X', '')
            form_layout.addRow(f"HexID {hex_id}:", tag_widget)

            self.sex_combos[tag] = sex_combo
            self.identity_edits[tag] = identity_edit
            self.start_edits[tag] = start_edit
            self.stop_edits[tag] = stop_edit

        form_container = QWidget()
        form_container.setLayout(form_layout)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_container)
        layout.addWidget(scroll, 1)   # takes the stretch; buttons stay fixed

        # Dialog buttons (outside the scroll area — always visible)
        button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        button_box.accepted.connect(self.accept)
        button_box.rejected.connect(self.reject)
        layout.addWidget(button_box)

        self.setLayout(layout)

        # Open at a height that fits the current screen (tag list scrolls
        # inside) and never let the window exceed it, keeping OK/Cancel reachable.
        try:
            avail = QApplication.primaryScreen().availableGeometry()
            self.setMaximumHeight(avail.height())
            self.resize(760, min(880, int(avail.height() * 0.9)))
        except Exception:
            self.resize(760, 800)

    def get_identities(self):
        """Return the configured identities with start/stop times"""
        result = {}
        for tag in self.available_tags:
            sex = self.sex_combos[tag].currentText()
            identity = self.identity_edits[tag].text().strip()
            if not identity:
                identity = str(tag)
            result[tag] = {
                'sex': sex,
                'identity': identity,
                'start_time': self.start_edits[tag].dateTime().toString("yyyy-MM-dd HH:mm:ss"),
                'stop_time': self.stop_edits[tag].dateTime().toString("yyyy-MM-dd HH:mm:ss"),
            }
        return result


class PlotSaverWorker(QThread):
    """Worker thread for saving plots without blocking the UI"""
    progress = pyqtSignal(str)
    finished = pyqtSignal(bool, str)
    
    def __init__(self, db_path, table_name, selected_tags, downsample, smoothing_method,
                 # `downsample` is vestigial (the downsampling feature was removed);
                 # kept only to preserve this positional signature.
                 plot_types=None, skip_existing=False, rolling_window=10, timezone='US/Mountain',
                 tag_identities=None, use_identities=False, background_image=None,
                 bg_width_meters=None, bg_height_meters=None, csv_path=None, save_svg=False,
                 output_dir=None, plots_dir=None, rolling_window_units='Seconds',
                 plot_layers=None, xml_zones=None, anchor_positions=None,
                 xml_map_image=None, xml_map_extent=None,
                 bg_offset_x=0.0, bg_offset_y=0.0):
        super().__init__()
        self.db_path = db_path
        self.table_name = table_name
        self.csv_path = csv_path
        self.selected_tags = selected_tags
        self.downsample = downsample
        self.smoothing_method = smoothing_method
        self.plot_types = plot_types if plot_types is not None else {
            'daily_paths': True,
            'trajectory_overview': True,
            'battery_levels': True
        }
        self.skip_existing = skip_existing
        self.output_dir = output_dir
        self.plots_dir = plots_dir  # Subfolder for plot output (PNGs/SVGs)
        self.rolling_window = rolling_window
        self.rolling_window_units = rolling_window_units
        self.timezone = timezone
        self.tag_identities = tag_identities if tag_identities else {}
        self.use_identities = use_identities
        self.background_image = background_image
        self.bg_width_meters = bg_width_meters
        self.bg_height_meters = bg_height_meters
        self.bg_offset_x = bg_offset_x
        self.bg_offset_y = bg_offset_y
        self.save_svg = save_svg
        # Spatial context layer choice + sources for the trajectory plots.
        self.plot_layers = plot_layers if plot_layers is not None else dict(DEFAULT_PLOT_LAYERS)
        self.xml_zones = xml_zones or []
        self.anchor_positions = anchor_positions or []
        self.xml_map_image = xml_map_image
        self.xml_map_extent = xml_map_extent

    def _bg_source(self):
        """(image, extent) for the background layer, or (None, None).

        Prefers the user-loaded floorplan PNG, then falls back to the
        XML-embedded site map so the map still shows when no PNG was loaded.
        """
        if self.background_image is not None and self.bg_width_meters is not None:
            x0, y0 = self.bg_offset_x, self.bg_offset_y
            return self.background_image, [x0, x0 + self.bg_width_meters,
                                           y0, y0 + self.bg_height_meters]
        if self.xml_map_image is not None and self.xml_map_extent is not None:
            return self.xml_map_image, list(self.xml_map_extent)
        return None, None

    def save_figure(self, fig, output_path):
        """Save figure as PNG, and optionally also as SVG"""
        fig.savefig(output_path, dpi=150, bbox_inches='tight')
        if self.save_svg:
            svg_path = os.path.splitext(output_path)[0] + '.svg'
            fig.savefig(svg_path, format='svg', bbox_inches='tight')
            self.progress.emit(f"Saved SVG: {os.path.basename(svg_path)}")

    def run(self):
        try:
            # Load from CSV if available (much faster and ensures consistency)
            if self.csv_path and os.path.exists(self.csv_path):
                self.progress.emit("Loading data from CSV...")
                data = pd.read_csv(self.csv_path, low_memory=False)
                
                # Parse Timestamp column (let pandas infer the format automatically)
                # This handles timezone-aware timestamps like "2025-10-13 18:09:10-06:00"
                data['Timestamp'] = pd.to_datetime(data['Timestamp'], format='mixed')
                
                self.progress.emit(f"Loaded {len(data)} records from CSV")
                coord = 'smoothed' if 'smoothed_x' in data.columns else 'raw (unsmoothed)'
                self.progress.emit(f"Plots will use {coord} coordinates")
            else:
                # Fallback: Load from database (old behavior)
                self.progress.emit("Loading data from database...")
                
                # Connect to database
                conn = sqlite3.connect(self.db_path)
                query = (f"SELECT {processing_select_clause(conn, self.table_name)} "
                         f"FROM {self.table_name}")
                data = pd.read_sql_query(query, conn)
                conn.close()
                
                self.progress.emit(f"Loaded {len(data)} records")
                
                # Process data
                data['Timestamp'] = pd.to_datetime(data['timestamp'], unit='ms', origin='unix', utc=True)
                tz = pytz.timezone(self.timezone)
                data['Timestamp'] = data['Timestamp'].dt.tz_convert(tz)
                
                # Convert location to meters
                data['location_x'] *= 0.0254
                data['location_y'] *= 0.0254
                
                data = data.sort_values(by=['shortid', 'Timestamp'])
                
                # Filter to selected tags
                if self.selected_tags:
                    data = data[data['shortid'].isin(self.selected_tags)]
                
                # Apply custom sex and identities if configured
                if self.use_identities and self.tag_identities:
                    data['sex'] = data['shortid'].map(lambda x: self.tag_identities.get(x, {}).get('sex', 'M'))
                    data['identity'] = data['shortid'].map(lambda x: self.tag_identities.get(x, {}).get('identity', f'Tag{x}'))
                else:
                    data['sex'] = 'M'
                    data['identity'] = data['shortid'].apply(lambda x: f'Tag{x}')
                
                # Apply smoothing FIRST (on full resolution data)
                if self.smoothing_method != "None":
                    self.progress.emit("Applying smoothing to full resolution data...")
                    data = self.apply_smoothing(data, self.smoothing_method)

            # Get output directory
            if self.output_dir:
                output_dir = self.output_dir
            else:
                db_dir = os.path.dirname(self.db_path)
                db_filename = os.path.basename(self.db_path)
                db_name = os.path.splitext(db_filename)[0]
                output_dir = os.path.join(db_dir, f"{db_name}_FNT_analysis")

            # Use plots subfolder for all plot output
            plots_dir = self.plots_dir if self.plots_dir else os.path.join(output_dir, 'plots')

            db_name = os.path.splitext(os.path.basename(self.db_path))[0]
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(plots_dir, exist_ok=True)
            
            # Generate and save plots based on selection
            generated_count = 0
            skipped_count = 0
            
            if self.plot_types.get('daily_paths', False):
                result = self.save_daily_paths_per_tag(data, plots_dir, db_name)
                if result:
                    generated_count += result

            if self.plot_types.get('trajectory_overview', False):
                result = self.save_trajectory_overview(data, plots_dir, db_name)
                if result:
                    generated_count += 1
                else:
                    skipped_count += 1

            if self.plot_types.get('battery_levels', False):
                result = self.save_battery_levels(data, plots_dir, db_name)
                if result:
                    generated_count += 1
                else:
                    skipped_count += 1

            if self.plot_types.get('3d_occupancy', False):
                result = self.save_3d_occupancy(data, plots_dir, db_name)
                if result:
                    generated_count += result

            if self.plot_types.get('activity_timeline', False):
                result = self.save_activity_timeline(data, plots_dir, db_name)
                if result:
                    generated_count += 1
                else:
                    skipped_count += 1

            if self.plot_types.get('velocity_distribution', False):
                result = self.save_velocity_distribution(data, plots_dir, db_name)
                if result:
                    generated_count += 1
                else:
                    skipped_count += 1
            
            if self.plot_types.get('cumulative_distance', False):
                result = self.save_cumulative_distance(data, plots_dir, db_name)
                if result:
                    generated_count += 1
                else:
                    skipped_count += 1

            if self.plot_types.get('velocity_timeline', False):
                result = self.save_velocity_timeline(data, plots_dir, db_name)
                if result:
                    generated_count += result

            if self.plot_types.get('actogram', False):
                result = self.save_actogram(data, plots_dir, db_name)
                if result:
                    generated_count += result

            if self.plot_types.get('data_quality', False):
                result = self.save_data_quality(data, plots_dir, db_name)
                if result:
                    generated_count += 1
                else:
                    skipped_count += 1
            
            msg = f"Generated {generated_count} plot(s)"
            if skipped_count > 0:
                msg += f", skipped {skipped_count} existing file(s)"
            msg += f" in {output_dir}"
            self.finished.emit(True, msg)
            
        except Exception as e:
            self.finished.emit(False, f"Error generating plots: {str(e)}")

    def apply_smoothing(self, data, method):
        """Apply smoothing to trajectory data"""
        def apply_savgol_filter(group):
            window_length = min(31, len(group))
            if window_length % 2 == 0:
                window_length -= 1
            polyorder = min(2, window_length - 1)
            if len(group) > polyorder:
                return savgol_filter(group, window_length=window_length, polyorder=polyorder)
            return group
        
        if method == "Savitzky-Golay":
            data['smoothed_x'] = data.groupby('shortid')['location_x'].transform(apply_savgol_filter)
            data['smoothed_y'] = data.groupby('shortid')['location_y'].transform(apply_savgol_filter)
        elif method in ("Rolling Average", "Rolling Median"):
            # Same interpretation as the interactive path: window is seconds
            # (time-based, centred) or a sample count, per the units setting.
            agg = 'mean' if method == "Rolling Average" else 'median'
            time_based = self.rolling_window_units == "Seconds"
            rolling_smooth_xy(data, agg, self.rolling_window, time_based)
        elif method == "Forward-Backward Exponentially Weighted Moving Average":
            # No minimum floor here — a span of 1-2 is legitimate for EWMA
            span = max(1, self.rolling_window)
            data['smoothed_x'] = data.groupby('shortid')['location_x'].transform(
                lambda x: forward_backward_ewma(x, span))
            data['smoothed_y'] = data.groupby('shortid')['location_y'].transform(
                lambda x: forward_backward_ewma(x, span))

        return data
    
    def save_daily_paths_per_tag(self, data, output_dir, db_name):
        """Save daily paths - one PNG per tag with all days
        Returns: number of plots generated"""
        self.progress.emit("Generating daily paths per tag...")
        
        data = data.copy()
        if 'Date' not in data.columns:
            data['Date'] = data['Timestamp'].dt.date
        
        x_col = 'smoothed_x' if 'smoothed_x' in data.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in data.columns else 'location_y'
        
        # Get global min/max coordinates across ALL data
        x_min, x_max = data[x_col].min(), data[x_col].max()
        y_min, y_max = data[y_col].min(), data[y_col].max()
        
        # Background is the loaded PNG, else the XML-embedded site map. If it
        # will be drawn, expand the shared axis limits to include its extent.
        day_bg_image, day_bg_extent = self._bg_source()
        if self.plot_layers.get('background') and day_bg_extent is not None:
            x_min = min(x_min, day_bg_extent[0])
            x_max = max(x_max, day_bg_extent[1])
            y_min = min(y_min, day_bg_extent[2])
            y_max = max(y_max, day_bg_extent[3])

        # Add padding
        x_range = x_max - x_min
        y_range = y_max - y_min
        x_pad = x_range * 0.05 if x_range > 0 else 1
        y_pad = y_range * 0.05 if y_range > 0 else 1

        x_min -= x_pad
        x_max += x_pad
        y_min -= y_pad
        y_max += y_pad

        unique_dates = sorted(data['Date'].unique())
        unique_tags = sorted(data['shortid'].unique())
        
        generated = 0
        
        # Create one plot per tag
        for tag in unique_tags:
            # Generate filename with HexID or sex-identity
            if self.use_identities and tag in self.tag_identities:
                info = self.tag_identities[tag]
                sex = info.get('sex', 'M')
                identity = info.get('identity', str(tag))
                file_suffix = f"{sex}-{identity}"
            else:
                hex_id = hex(tag).upper().replace('0X', '')
                file_suffix = f"HexID{hex_id}"
            
            output_path = os.path.join(output_dir, f'{db_name}_DailyPaths_{file_suffix}.png')
            
            # Check if file exists and overwrite is False
            if self.skip_existing and os.path.exists(output_path):
                self.progress.emit(f"Skipped (exists): {db_name}_DailyPaths_{file_suffix}.png")
                continue
            
            tag_data = data[data['shortid'] == tag]
            num_days = len(unique_dates)
            
            fig = Figure(figsize=(min(4 * num_days, 20), 4))
            
            for day_idx, date in enumerate(unique_dates):
                day_data = tag_data[tag_data['Date'] == date]
                
                ax = fig.add_subplot(1, num_days, day_idx + 1)

                # Spatial context (background/zones/anchors) per the export choice
                draw_context_layers(
                    ax, self.plot_layers,
                    bg_image=day_bg_image, bg_extent=day_bg_extent,
                    zones_xml=self.xml_zones, anchors=self.anchor_positions)

                if not day_data.empty:
                    ax.plot(day_data[x_col], day_data[y_col],
                           linewidth=1.5, alpha=0.8, color='blue')
                    ax.scatter(day_data[x_col].iloc[0], day_data[y_col].iloc[0],
                              c='black', s=50, marker='o', zorder=5)
                    ax.scatter(day_data[x_col].iloc[-1], day_data[y_col].iloc[-1],
                              c='black', s=50, marker='s', zorder=5)
                
                # Apply global axis limits
                ax.set_xlim(x_min, x_max)
                ax.set_ylim(y_min, y_max)
                
                ax.set_xlabel('X (m)', fontsize=9)
                ax.set_ylabel('Y (m)', fontsize=9)
                ax.set_title(f'{date}', fontsize=10)
                ax.grid(True, alpha=0.3)
                ax.set_aspect('equal')
            
            fig.suptitle(f'Daily Paths - {file_suffix}', fontsize=14, fontweight='bold')
            fig.tight_layout()
            
            self.save_figure(fig, output_path)
            plt.close(fig)
            generated += 1
            
            self.progress.emit(f"Saved: {db_name}_DailyPaths_{file_suffix}.png")
        
        return generated
    
    def save_trajectory_overview(self, data, output_dir, db_name):
        """Save trajectory overview
        Returns: True if generated, False if skipped"""
        self.progress.emit("Generating trajectory overview...")
        
        output_path = os.path.join(output_dir, f'{db_name}_TrajectoryOverview.png')
        
        # Check if file exists and overwrite is False
        if self.skip_existing and os.path.exists(output_path):
            self.progress.emit(f"Skipped (exists): {db_name}_TrajectoryOverview.png")
            return False
        
        fig = Figure(figsize=(10, 8))
        ax = fig.add_subplot(111)

        # Spatial context (background/zones/anchors) per the export choice.
        # Background is the loaded PNG, else the XML-embedded site map.
        bg_image, bg_extent = self._bg_source()
        draw_context_layers(
            ax, self.plot_layers,
            bg_image=bg_image, bg_extent=bg_extent,
            zones_xml=self.xml_zones, anchors=self.anchor_positions)

        x_col = 'smoothed_x' if 'smoothed_x' in data.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in data.columns else 'location_y'
        
        # Plot each tag with sex-based coloring
        for tag in data['shortid'].unique():
            tag_data = data[data['shortid'] == tag]
            
            # Determine label and color based on identity configuration
            if self.use_identities and tag in self.tag_identities:
                info = self.tag_identities[tag]
                sex = info.get('sex', 'M')
                identity = info.get('identity', str(tag))
                label = f"{sex}-{identity}"
                color = 'blue' if sex == 'M' else 'red'
            else:
                hex_id = hex(tag).upper().replace('0X', '')
                label = f"HexID {hex_id}"
                color = 'blue'  # Default to blue
            
            ax.plot(tag_data[x_col], tag_data[y_col], 
                   linewidth=1, alpha=0.7, color=color, label=label)
        
        ax.set_xlabel('X Position (m)', fontsize=10)
        ax.set_ylabel('Y Position (m)', fontsize=10)
        ax.set_title('Trajectory Overview - All Tags', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        ax.set_aspect('equal')
        fig.tight_layout()
        
        self.save_figure(fig, output_path)
        plt.close(fig)
        
        self.progress.emit(f"Saved: {db_name}_TrajectoryOverview.png")
        return True
    
    def save_battery_levels(self, data, output_dir, db_name):
        """Save battery levels plot — faceted by tag (one subplot per tag).
        Returns: True if generated, False if skipped or no battery data"""
        self.progress.emit("Generating battery levels (faceted by tag)...")

        output_path = os.path.join(output_dir, f'{db_name}_BatteryLevels.png')

        # Check if file exists and skip_existing is True
        if self.skip_existing and os.path.exists(output_path):
            self.progress.emit(f"Skipped (exists): {db_name}_BatteryLevels.png")
            return False

        # Search for battery column
        battery_col = None
        possible_names = ['battery_voltage', 'vbat', 'battery', 'bat', 'voltage']
        for col_name in possible_names:
            if col_name in data.columns:
                battery_col = col_name
                break

        if battery_col is None:
            self.progress.emit("No battery column found, skipping battery plot")
            return False

        tags = sorted(data['shortid'].unique())
        n_tags = len(tags)

        if n_tags == 0:
            self.progress.emit("No tags found, skipping battery plot")
            return False

        # Size: generous height per facet so many tags remain readable when zoomed
        subplot_height = 2.0
        fig_height = max(6, n_tags * subplot_height + 1.5)
        fig = Figure(figsize=(14, fig_height), dpi=150)

        axes = fig.subplots(n_tags, 1, sharex=True, squeeze=False)
        axes = axes.flatten()

        # Shared y-limits across all facets for easy comparison
        global_min = data[battery_col].min()
        global_max = data[battery_col].max()
        y_margin = (global_max - global_min) * 0.1 if global_max != global_min else 0.1
        y_lo = global_min - y_margin
        y_hi = global_max + y_margin

        for idx, tag in enumerate(tags):
            ax = axes[idx]
            tag_data = data[data['shortid'] == tag]

            # Determine label from identity info
            if self.use_identities and tag in self.tag_identities:
                info = self.tag_identities[tag]
                sex = info.get('sex', 'M')
                identity = info.get('identity', str(tag))
                label = f"{sex}-{identity}"
            else:
                hex_id = hex(tag).upper().replace('0X', '')
                label = f"HexID {hex_id}"

            color = 'tab:blue'
            if self.use_identities and tag in self.tag_identities:
                sex = self.tag_identities[tag].get('sex', 'M')
                color = 'tab:blue' if sex == 'M' else 'tab:red'

            ax.plot(tag_data['Timestamp'], tag_data[battery_col],
                    linewidth=1.0, marker='o', markersize=1, color=color, alpha=0.85)

            ax.set_ylim(y_lo, y_hi)
            ax.set_ylabel(label, fontsize=8, fontweight='bold', rotation=0, labelpad=60, ha='right')
            ax.grid(True, alpha=0.25)
            ax.tick_params(axis='y', labelsize=7)
            ax.tick_params(axis='x', labelsize=7)

            # Only show x-tick labels on the bottom subplot
            if idx < n_tags - 1:
                ax.tick_params(axis='x', labelbottom=False)

        # Bottom axis label
        axes[-1].set_xlabel('Time', fontsize=10)

        fig.suptitle(f'Battery Levels Over Time — {battery_col} (V)', fontsize=13, fontweight='bold', y=1.0)
        fig.autofmt_xdate()
        fig.tight_layout(rect=[0, 0, 1, 0.98])

        self.save_figure(fig, output_path)
        plt.close(fig)

        self.progress.emit(f"Saved: {db_name}_BatteryLevels.png")
        return True
    
    def save_3d_occupancy(self, data, output_dir, db_name):
        """Save 3D occupancy heatmap - one file per tag faceted by day
        Returns: number of plots generated"""
        self.progress.emit("Generating 3D occupancy heatmaps...")
        
        from mpl_toolkits.mplot3d import Axes3D
        
        data = data.copy()
        if 'Date' not in data.columns:
            data['Date'] = data['Timestamp'].dt.date
        
        unique_dates = sorted(data['Date'].unique())
        date_to_day = {date: i+1 for i, date in enumerate(unique_dates)}
        data['Day'] = data['Date'].map(date_to_day)
        
        x_col = 'smoothed_x' if 'smoothed_x' in data.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in data.columns else 'location_y'
        
        unique_tags = sorted(data['shortid'].unique())
        num_days = len(unique_dates)
        generated = 0
        
        # Create one plot per tag with all days
        for tag in unique_tags:
            # Generate filename with HexID or sex-identity
            if self.use_identities and tag in self.tag_identities:
                info = self.tag_identities[tag]
                sex = info.get('sex', 'M')
                identity = info.get('identity', str(tag))
                file_suffix = f"{sex}-{identity}"
            else:
                hex_id = hex(tag).upper().replace('0X', '')
                file_suffix = f"HexID{hex_id}"
            
            output_path = os.path.join(output_dir, f'{db_name}_3D_Occupancy_{file_suffix}.png')
            
            if self.skip_existing and os.path.exists(output_path):
                self.progress.emit(f"Skipped (exists): {db_name}_3D_Occupancy_{file_suffix}.png")
                continue
            
            tag_data = data[data['shortid'] == tag]
            
            # Create subplots for each day
            cols = min(3, num_days)
            rows = (num_days + cols - 1) // cols
            fig = Figure(figsize=(6 * cols, 5 * rows))
            
            for day_idx, day in enumerate(sorted(tag_data['Day'].unique())):
                day_data = tag_data[tag_data['Day'] == day]
                
                ax = fig.add_subplot(rows, cols, day_idx + 1, projection='3d')
                
                # Create 3D histogram
                hist, xedges, yedges = np.histogram2d(
                    day_data[x_col], day_data[y_col], bins=20
                )
                
                xpos, ypos = np.meshgrid(xedges[:-1], yedges[:-1], indexing="ij")
                xpos = xpos.ravel()
                ypos = ypos.ravel()
                zpos = 0
                
                dx = dy = (xedges[1] - xedges[0]) * np.ones_like(zpos)
                dz = hist.ravel()
                
                # Color by sex if available
                if 'sex' in day_data.columns:
                    sex = day_data['sex'].iloc[0]
                    color = 'blue' if sex == 'M' else 'red'
                else:
                    color = 'steelblue'
                
                ax.bar3d(xpos, ypos, zpos, dx, dy, dz, zsort='average', alpha=0.8, color=color)
                ax.set_xlabel('X (m)', fontsize=8)
                ax.set_ylabel('Y (m)', fontsize=8)
                ax.set_zlabel('Count', fontsize=8)
                ax.set_title(f'Day {day}', fontsize=10)
            
            fig.suptitle(f'3D Occupancy - {file_suffix}', fontsize=14, fontweight='bold')
            fig.tight_layout()
            self.save_figure(fig, output_path)
            plt.close(fig)
            generated += 1
            
            self.progress.emit(f"Saved: {db_name}_3D_Occupancy_{file_suffix}.png")
        
        return generated
    
    def save_activity_timeline(self, data, output_dir, db_name):
        """Save activity timeline
        Returns: True if generated, False if skipped"""
        self.progress.emit("Generating activity timeline...")
        
        output_path = os.path.join(output_dir, f'{db_name}_ActivityTimeline.png')
        
        if self.skip_existing and os.path.exists(output_path):
            self.progress.emit(f"Skipped (exists): {db_name}_ActivityTimeline.png")
            return False
        
        fig = Figure(figsize=(12, 6))
        ax = fig.add_subplot(111)
        
        for tag in data['shortid'].unique():
            tag_data = data[data['shortid'] == tag]
            hourly_counts = tag_data.set_index('Timestamp').resample('h').size()
            ax.plot(hourly_counts.index, hourly_counts.values, label=f'Tag {tag}', linewidth=1.5)
        
        ax.set_xlabel('Time', fontsize=10)
        ax.set_ylabel('Data Points per Hour', fontsize=10)
        ax.set_title('Activity Timeline - Data Points Over Time', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.autofmt_xdate()
        fig.tight_layout()
        self.save_figure(fig, output_path)
        plt.close(fig)
        
        self.progress.emit(f"Saved: {db_name}_ActivityTimeline.png")
        return True
    
    def save_velocity_distribution(self, data, output_dir, db_name):
        """Save velocity distribution
        Returns: True if generated, False if skipped"""
        self.progress.emit("Generating velocity distribution...")
        
        output_path = os.path.join(output_dir, f'{db_name}_VelocityDistribution.png')
        
        if self.skip_existing and os.path.exists(output_path):
            self.progress.emit(f"Skipped (exists): {db_name}_VelocityDistribution.png")
            return False
        
        data = data.copy()
        x_col = 'smoothed_x' if 'smoothed_x' in data.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in data.columns else 'location_y'
        
        # Calculate velocity
        data['time_diff'] = data.groupby('shortid')['Timestamp'].diff().dt.total_seconds()
        data['distance'] = np.sqrt(
            (data[x_col] - data.groupby('shortid')[x_col].shift())**2 +
            (data[y_col] - data.groupby('shortid')[y_col].shift())**2
        )
        data['velocity'] = data['distance'] / data['time_diff']
        
        # Filter out unrealistic velocities
        data = data[(data['velocity'] <= 2) | (data['velocity'].isna())]
        
        fig = Figure(figsize=(10, 6))
        ax = fig.add_subplot(111)
        
        for tag in data['shortid'].unique():
            tag_data = data[data['shortid'] == tag]['velocity'].dropna()
            if len(tag_data) > 0:
                # Generate label with HexID or sex-identity
                if self.use_identities and tag in self.tag_identities:
                    info = self.tag_identities[tag]
                    sex = info.get('sex', 'M')
                    identity = info.get('identity', str(tag))
                    label = f"{sex}-{identity}"
                else:
                    hex_id = hex(tag).upper().replace('0X', '')
                    label = f"HexID {hex_id}"
                
                ax.hist(tag_data, bins=50, alpha=0.5, label=label, density=True)
        
        ax.set_xlabel('Velocity (m/s)', fontsize=10)
        ax.set_ylabel('Density', fontsize=10)
        ax.set_title('Velocity Distribution', fontsize=12, fontweight='bold')
        ax.legend(loc='best', fontsize=8)
        ax.grid(True, alpha=0.3)
        fig.tight_layout()
        self.save_figure(fig, output_path)
        plt.close(fig)
        
        self.progress.emit(f"Saved: {db_name}_VelocityDistribution.png")
        return True
    
    def save_cumulative_distance(self, data, output_dir, db_name):
        """Save cumulative distance plots (reset daily)
        Returns: True if generated, False if skipped"""
        self.progress.emit("Generating cumulative distance plots...")
        
        output_path = os.path.join(output_dir, f'{db_name}_CumulativeDistance.png')
        
        if self.skip_existing and os.path.exists(output_path):
            self.progress.emit(f"Skipped (exists): {db_name}_CumulativeDistance.png")
            return False
        
        data = data.copy()
        if 'Date' not in data.columns:
            data['Date'] = data['Timestamp'].dt.date
        
        x_col = 'smoothed_x' if 'smoothed_x' in data.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in data.columns else 'location_y'
        
        # Calculate distance between consecutive points
        data = data.sort_values(['shortid', 'Timestamp'])
        data['distance_step'] = data.groupby('shortid', group_keys=False).apply(
            lambda group: np.sqrt(
                group[x_col].diff()**2 + group[y_col].diff()**2
            ).fillna(0)
        ).reset_index(level=0, drop=True)
        
        # Cumulative distance per day (reset each day)
        data['cumulative_distance'] = data.groupby(['shortid', 'Date'])['distance_step'].cumsum()
        data['time_of_day'] = (data['Timestamp'] - data['Timestamp'].dt.normalize()).dt.total_seconds() / 3600
        
        unique_days = sorted(data['Date'].unique())
        num_days = len(unique_days)
        num_cols = min(4, num_days)
        num_rows = (num_days + num_cols - 1) // num_cols
        
        fig = Figure(figsize=(5 * num_cols, 4 * num_rows))
        
        for i, day in enumerate(unique_days):
            ax = fig.add_subplot(num_rows, num_cols, i + 1)
            day_data = data[data['Date'] == day]
            
            for tag in day_data['shortid'].unique():
                tag_data = day_data[day_data['shortid'] == tag]
                
                # Generate label
                if self.use_identities and tag in self.tag_identities:
                    info = self.tag_identities[tag]
                    sex = info.get('sex', 'M')
                    identity = info.get('identity', str(tag))
                    label = f"{sex}-{identity}"
                else:
                    hex_id = hex(tag).upper().replace('0X', '')
                    label = f"HexID {hex_id}"
                
                ax.plot(tag_data['time_of_day'], tag_data['cumulative_distance'], label=label, alpha=0.7)
            
            ax.set_xlabel('Hour of Day', fontsize=9)
            ax.set_ylabel('Distance (m)', fontsize=9)
            ax.set_title(f'Day {i+1}: {day}', fontsize=10)
            ax.legend(fontsize=7)
            ax.grid(True, alpha=0.3)
        
        fig.suptitle('Cumulative Distance Traveled by Day', fontsize=12, fontweight='bold')
        fig.tight_layout()
        self.save_figure(fig, output_path)
        plt.close(fig)
        
        self.progress.emit(f"Saved: {db_name}_CumulativeDistance.png")
        return True
    
    def save_velocity_timeline(self, data, output_dir, db_name):
        """Save velocity timeline plots
        Returns: number of plots generated"""
        self.progress.emit("Generating velocity timeline plots...")
        
        data = data.copy()
        if 'Date' not in data.columns:
            data['Date'] = data['Timestamp'].dt.date
        
        x_col = 'smoothed_x' if 'smoothed_x' in data.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in data.columns else 'location_y'
        
        # Calculate velocity
        data['time_diff'] = data.groupby('shortid')['Timestamp'].diff().dt.total_seconds()
        data['distance'] = np.sqrt(
            (data[x_col] - data.groupby('shortid')[x_col].shift())**2 +
            (data[y_col] - data.groupby('shortid')[y_col].shift())**2
        )
        data['velocity'] = data['distance'] / data['time_diff']
        data = data[(data['velocity'] <= 2) | (data['velocity'].isna())]
        
        generated = 0
        for tag in data['shortid'].unique():
            # Generate filename
            if self.use_identities and tag in self.tag_identities:
                info = self.tag_identities[tag]
                sex = info.get('sex', 'M')
                identity = info.get('identity', str(tag))
                filename = f'{db_name}_VelocityTimeline_{sex}-{identity}.png'
            else:
                hex_id = hex(tag).upper().replace('0X', '')
                filename = f'{db_name}_VelocityTimeline_HexID{hex_id}.png'
            
            output_path = os.path.join(output_dir, filename)
            
            if self.skip_existing and os.path.exists(output_path):
                self.progress.emit(f"Skipped (exists): {filename}")
                continue
            
            tag_data = data[data['shortid'] == tag].copy()
            
            fig = Figure(figsize=(12, 6))
            ax = fig.add_subplot(111)
            
            ax.plot(tag_data['Timestamp'], tag_data['velocity'], alpha=0.6, linewidth=0.5, color='blue')
            ax.axhline(y=0.1, color='red', linestyle='--', label='Activity threshold (0.1 m/s)')
            
            ax.set_xlabel('Time', fontsize=10)
            ax.set_ylabel('Velocity (m/s)', fontsize=10)
            
            # Generate title
            if self.use_identities and tag in self.tag_identities:
                info = self.tag_identities[tag]
                sex = info.get('sex', 'M')
                identity = info.get('identity', str(tag))
                ax.set_title(f'Velocity Timeline: {sex}-{identity}', fontsize=12, fontweight='bold')
            else:
                hex_id = hex(tag).upper().replace('0X', '')
                ax.set_title(f'Velocity Timeline: HexID {hex_id}', fontsize=12, fontweight='bold')
            
            ax.legend(fontsize=8)
            ax.grid(True, alpha=0.3)
            fig.autofmt_xdate()
            fig.tight_layout()
            self.save_figure(fig, output_path)
            plt.close(fig)
            
            self.progress.emit(f"Saved: {filename}")
            generated += 1
        
        return generated
    
    def save_actogram(self, data, output_dir, db_name):
        """Save circadian actogram plots
        Returns: number of plots generated"""
        self.progress.emit("Generating actogram plots...")
        
        data = data.copy()
        if 'Date' not in data.columns:
            data['Date'] = data['Timestamp'].dt.date
        
        x_col = 'smoothed_x' if 'smoothed_x' in data.columns else 'location_x'
        y_col = 'smoothed_y' if 'smoothed_y' in data.columns else 'location_y'
        
        # Calculate velocity for activity
        data['time_diff'] = data.groupby('shortid')['Timestamp'].diff().dt.total_seconds()
        data['distance'] = np.sqrt(
            (data[x_col] - data.groupby('shortid')[x_col].shift())**2 +
            (data[y_col] - data.groupby('shortid')[y_col].shift())**2
        )
        data['velocity'] = data['distance'] / data['time_diff']
        data = data[(data['velocity'] <= 2) | (data['velocity'].isna())]
        
        # Add hour and day columns
        data['hour'] = data['Timestamp'].dt.hour + data['Timestamp'].dt.minute / 60
        unique_dates = sorted(data['Date'].unique())
        date_to_day = {date: i+1 for i, date in enumerate(unique_dates)}
        data['Day'] = data['Date'].map(date_to_day)
        
        generated = 0
        for tag in data['shortid'].unique():
            # Generate filename
            if self.use_identities and tag in self.tag_identities:
                info = self.tag_identities[tag]
                sex = info.get('sex', 'M')
                identity = info.get('identity', str(tag))
                filename = f'{db_name}_Actogram_{sex}-{identity}.png'
            else:
                hex_id = hex(tag).upper().replace('0X', '')
                filename = f'{db_name}_Actogram_HexID{hex_id}.png'
            
            output_path = os.path.join(output_dir, filename)
            
            if self.skip_existing and os.path.exists(output_path):
                self.progress.emit(f"Skipped (exists): {filename}")
                continue
            
            tag_data = data[data['shortid'] == tag].copy()
            
            # Bin activity by hour and day
            tag_data['active'] = (tag_data['velocity'] > 0.1).astype(int)
            activity_grid = tag_data.groupby(['Day', 'hour'])['active'].sum().unstack(fill_value=0)
            
            fig = Figure(figsize=(12, max(6, len(unique_dates) * 0.3)))
            ax = fig.add_subplot(111)
            
            im = ax.imshow(activity_grid.values, aspect='auto', cmap='YlOrRd', interpolation='nearest')
            ax.set_xlabel('Hour of Day', fontsize=10)
            ax.set_ylabel('Day', fontsize=10)
            ax.set_xticks(range(0, 24, 2))
            ax.set_xticklabels(range(0, 24, 2))
            
            # Generate title
            if self.use_identities and tag in self.tag_identities:
                info = self.tag_identities[tag]
                sex = info.get('sex', 'M')
                identity = info.get('identity', str(tag))
                ax.set_title(f'Circadian Actogram: {sex}-{identity}', fontsize=12, fontweight='bold')
            else:
                hex_id = hex(tag).upper().replace('0X', '')
                ax.set_title(f'Circadian Actogram: HexID {hex_id}', fontsize=12, fontweight='bold')
            
            fig.colorbar(im, ax=ax, label='Activity Count')
            fig.tight_layout()
            self.save_figure(fig, output_path)
            plt.close(fig)
            
            self.progress.emit(f"Saved: {filename}")
            generated += 1
        
        return generated
    
    def save_data_quality(self, data, output_dir, db_name):
        """Save data quality metrics table
        Returns: True if generated, False if skipped"""
        self.progress.emit("Generating data quality metrics...")
        
        output_path = os.path.join(output_dir, f'{db_name}_DataQuality.png')
        
        if self.skip_existing and os.path.exists(output_path):
            self.progress.emit(f"Skipped (exists): {db_name}_DataQuality.png")
            return False
        
        quality_data = []
        for tag in sorted(data['shortid'].unique()):
            tag_data = data[data['shortid'] == tag].sort_values('Timestamp')
            gaps = tag_data['Timestamp'].diff().dt.total_seconds()
            
            # Generate label
            if self.use_identities and tag in self.tag_identities:
                info = self.tag_identities[tag]
                sex = info.get('sex', 'M')
                identity = info.get('identity', str(tag))
                label = f"{sex}-{identity}"
            else:
                hex_id = hex(tag).upper().replace('0X', '')
                label = f"HexID {hex_id}"
            
            median_gap = gaps.median()
            max_gap = gaps.max()
            large_gaps = (gaps > 60).sum()
            
            quality_data.append([
                label,
                f"{median_gap:.2f}s" if pd.notna(median_gap) else "N/A",
                f"{max_gap:.2f}s" if pd.notna(max_gap) else "N/A",
                str(large_gaps)
            ])
        
        fig = Figure(figsize=(10, max(4, len(quality_data) * 0.5)))
        ax = fig.add_subplot(111)
        ax.axis('off')
        
        table = ax.table(cellText=quality_data,
                        colLabels=['Tag', 'Median Gap', 'Max Gap', 'Gaps >60s'],
                        cellLoc='center',
                        loc='center',
                        colWidths=[0.25, 0.25, 0.25, 0.25])
        
        table.auto_set_font_size(False)
        table.set_fontsize(10)
        table.scale(1, 2.5)
        
        # Style header
        for i in range(4):
            table[(0, i)].set_facecolor('#4472C4')
            table[(0, i)].set_text_props(weight='bold', color='white')
        
        ax.set_title('Data Quality Metrics', fontsize=12, fontweight='bold', pad=20)
        fig.tight_layout()
        self.save_figure(fig, output_path)
        plt.close(fig)
        
        self.progress.emit(f"Saved: {db_name}_DataQuality.png")
        return True


class FigurePopup(QDialog):
    """A resizable, zoom/pan-able window wrapping a matplotlib Figure.

    Used for the occupancy heatmaps: the matplotlib navigation toolbar gives
    scroll/box zoom and pan over the raster panels, and Save exports the view.
    """
    def __init__(self, fig, title="Figure", parent=None):
        super().__init__(parent)
        self.setWindowTitle(title)
        self.setMinimumSize(820, 640)
        # Non-modal so the user can keep working with the tool behind it.
        self.setModal(False)
        layout = QVBoxLayout()
        self.canvas = FigureCanvasQTAgg(fig)
        toolbar = NavigationToolbar2QT(self.canvas, self)
        layout.addWidget(toolbar)
        layout.addWidget(self.canvas, 1)
        self.setLayout(layout)


class UWBQuickVisualizationWindow(QWidget):
    # Preview view modes
    VIEW_XML = "XML (site map)"
    VIEW_2D = "2D Idealized"
    VIEW_3D = "3D Idealized"

    def __init__(self):
        super().__init__()
        self.db_path = None
        self.table_name = None
        self.available_tags = []
        self.data = None
        self.worker = None

        # Export control flags
        self.export_cancelled = False
        self.exporting = False
        
        # Tag identity and sex mapping
        self.tag_identities = {}  # {tag_id: {'sex': 'M', 'identity': 'Animal1'}}
        
        # XML configuration and background image
        self.xml_config_path = None
        self.background_image_path = None
        self.background_image = None  # Loaded matplotlib image
        self.xml_scale = None  # Scale from XML in inches/pixel (first Map element)
        self.bg_width_meters = None  # Background image width in meters
        self.bg_height_meters = None  # Background image height in meters
        # Manual background-image transform (option (b)): the loaded floorplan is
        # placed by pixels * bg_scale(in/px), then shifted by (bg_offset_x,
        # bg_offset_y) metres. Defaults come from the XML but the user can nudge
        # them live to match the Wiser server frame — needed when the loaded
        # image is a different render (resolution/decoration) than the one the
        # XML scale was calibrated against.
        self.bg_scale = None          # effective in/px for the loaded image
        self.bg_offset_x = 0.0        # metres: world X of the image's left edge
        self.bg_offset_y = 0.0        # metres: world Y of the image's bottom edge
        self.arena_zones = None  # DataFrame with zone coordinates from XML
        self.anchor_positions = []  # List of dicts: {'shortid': int, 'x': float, 'y': float, 'z': float}
        self.xml_zones = []         # [{name, color, points:(N,2) m}] from the site XML
        self.xml_map_image = None   # site map decoded from the XML
        self.xml_map_extent = None  # (x0, x1, y0, y1) in metres
        # Every embedded <Map>/<Image> render, each with its own pixel dims and
        # scale: Wiser writes several (e.g. 'default' + a decorated 'Dark2d').
        # Used to pick the correct default scale for a loaded external image by
        # matching its resolution.
        self.xml_maps = []          # [{name, scale, w_px, h_px}]

        # Spatial context layers drawn on exported plots/animation. Chosen per
        # export via PlotLayersDialog, persisted in the config, and applied to
        # every spatial output (trajectory plots, animation, occupancy,
        # last-known). Independent of the Preview Options 'Show ...' toggles,
        # which only affect the live preview.
        self.plot_layers = dict(DEFAULT_PLOT_LAYERS)
        self._exported_sitemap = None  # {filename, extent_m} once an export writes it
        self._animation_tags = None    # None = all; else a subset of shortids

        # Preview state. The timeline spans the whole recording but memory
        # holds at most MAX_CACHED_CHUNKS slices — see the streaming chunk
        # engine below.
        self.preview_timer = None
        self.preview_arena = None
        self.preview_tags = []
        self.preview_times = None
        self.preview_hz = 1
        self.preview_x = None           # (frames, tags) smoothed positions
        self.preview_y = None
        self.preview_xf = None          # forward-filled marker positions
        self.preview_yf = None
        self.preview_raw_x = None       # same grid, pre-smoothing
        self.preview_raw_y = None
        self.preview_batt = None        # battery voltage per tag, same grid
        self.preview_colors = None

        self.preview_cache = OrderedDict()   # chunk_index -> frame arrays (LRU)
        self.preview_inflight = {}           # chunk_index -> running loader
        self.preview_current_chunk = None
        self.preview_pending_current = None  # chunk to display once it lands
        self.preview_t0 = None               # first ping, epoch ms
        self.preview_t1 = None               # last ping, epoch ms
        self.preview_playhead_ms = 0
        self._timeline_guard = False         # suppress slider feedback loops
        self._tag_selection_guard = False    # suppress bulk checkbox churn
        self._preview_active = False         # streaming begins once data exists
        self.preview_db_path = None          # indexed copy, or the original
        self.preview_index_builder = None

        self.initUI()
        
    def log_message(self, message):
        """Add a message to the messages window"""
        from datetime import datetime
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.txt_messages.append(f"[{timestamp}] {message}")
        # Auto-scroll to bottom
        self.txt_messages.verticalScrollBar().setValue(
            self.txt_messages.verticalScrollBar().maximum()
        )
        # Also update legacy status label for compatibility
        self.lbl_status.setText(message)

    def copy_session_logs(self):
        """Copy the full session log to the clipboard."""
        text = self.txt_messages.toPlainText()
        QApplication.clipboard().setText(text)
        if text.strip():
            lines = text.count("\n") + 1
            self.log_message(f"Copied session logs to clipboard ({lines} lines)")
        else:
            self.log_message("Session logs are empty — nothing to copy")

    def save_message_log(self, output_dir):
        """Save the message log to a text file"""
        try:
            db_name = os.path.splitext(os.path.basename(self.db_path))[0]
            log_path = os.path.join(output_dir, f"{db_name}_messageLog.txt")
            
            with open(log_path, 'w', encoding='utf-8') as f:
                f.write(self.txt_messages.toPlainText())
            
            self.log_message(f"✓ Message log saved: {os.path.basename(log_path)}")
        except Exception as e:
            self.log_message(f"Warning: Could not save message log: {str(e)}")
    
    def save_run_summary(self, output_dir):
        """Save run summary with filtering statistics to CSV"""
        try:
            from datetime import datetime
            db_name = os.path.splitext(os.path.basename(self.db_path))[0]
            summary_path = os.path.join(output_dir, f"{db_name}_runSummary.csv")
            
            # Collect run parameters
            summary_data = {
                'Parameter': [],
                'Value': []
            }
            
            # General info
            summary_data['Parameter'].append('FNT Version')
            summary_data['Value'].append(get_fnt_version())

            summary_data['Parameter'].append('Run Date')
            summary_data['Value'].append(datetime.now().strftime('%Y-%m-%d %H:%M:%S'))

            summary_data['Parameter'].append('Database')
            summary_data['Value'].append(os.path.basename(self.db_path))
            
            summary_data['Parameter'].append('Table')
            summary_data['Value'].append(self.table_name)
            
            summary_data['Parameter'].append('Timezone')
            summary_data['Value'].append(self.combo_timezone.currentText())
            
            # Selected tags
            selected_tags = [tag for tag, cb in self.tag_checkboxes.items() if cb.isChecked()]
            summary_data['Parameter'].append('Selected Tags')
            summary_data['Value'].append(', '.join([str(t) for t in selected_tags]))
            
            # Thresholding settings (pre-smoothing outlier rejection)
            summary_data['Parameter'].append('Velocity Threshold Enabled')
            summary_data['Value'].append('Yes' if self.chk_velocity_filter.isChecked() else 'No')

            if self.chk_velocity_filter.isChecked():
                summary_data['Parameter'].append('Velocity Threshold (m/s)')
                summary_data['Value'].append(self.spin_velocity_threshold.value())

            summary_data['Parameter'].append('Jump Threshold Enabled')
            summary_data['Value'].append('Yes' if self.chk_jump_filter.isChecked() else 'No')

            if self.chk_jump_filter.isChecked():
                summary_data['Parameter'].append('Jump Threshold (m)')
                summary_data['Value'].append(self.spin_jump_threshold.value())

            summary_data['Parameter'].append('Time Gap Threshold (s)')
            summary_data['Value'].append(self.spin_time_gap.value())
            
            # Smoothing settings
            summary_data['Parameter'].append('Smoothing')
            summary_data['Value'].append(self.combo_smoothing.currentText())

            smoothing_method = self.get_smoothing_method()
            if smoothing_method in ("Rolling Average", "Rolling Median"):
                unit = "s" if self.combo_window_units.currentText() == "Seconds" else "samples"
                summary_data['Parameter'].append('Smoothing Window')
                summary_data['Value'].append(f"{self.spin_rolling_window.value()} {unit}")
            elif smoothing_method == "Forward-Backward Exponentially Weighted Moving Average":
                summary_data['Parameter'].append('EWMA Span (samples)')
                summary_data['Value'].append(self.spin_rolling_window.value())

            # Filtering statistics (if available)
            if hasattr(self, 'filter_stats') and self.filter_stats:
                summary_data['Parameter'].append('')
                summary_data['Value'].append('')
                
                summary_data['Parameter'].append('--- Filtering Statistics (all tags) ---')
                summary_data['Value'].append('')

                summary_data['Parameter'].append('Tags Processed')
                summary_data['Value'].append(self.filter_stats.get('tags_processed', 'N/A'))

                summary_data['Parameter'].append('Initial Data Points')
                summary_data['Value'].append(self.filter_stats.get('initial_count', 'N/A'))
                
                summary_data['Parameter'].append('Points Removed (Velocity)')
                summary_data['Value'].append(self.filter_stats.get('removed_velocity', 0))
                
                summary_data['Parameter'].append('Points Removed (Jump)')
                summary_data['Value'].append(self.filter_stats.get('removed_jump', 0))
                
                summary_data['Parameter'].append('Final Data Points')
                summary_data['Value'].append(self.filter_stats.get('final_count', 'N/A'))
                
                summary_data['Parameter'].append('Percent Filtered')
                summary_data['Value'].append(f"{self.filter_stats.get('percent_filtered', 0):.2f}%")
            
            # Export options
            summary_data['Parameter'].append('')
            summary_data['Value'].append('')
            
            summary_data['Parameter'].append('--- Export Options ---')
            summary_data['Value'].append('')
            
            summary_data['Parameter'].append('Raw CSV Exported')
            summary_data['Value'].append('Yes' if self.chk_export_raw_csv.isChecked() else 'No')

            summary_data['Parameter'].append('Smoothed CSV Exported')
            summary_data['Value'].append('Yes' if self.chk_export_smoothed_csv.isChecked() else 'No')

            summary_data['Parameter'].append('Plots Generated')
            summary_data['Value'].append('Yes' if self.chk_save_plots.isChecked() else 'No')
            
            summary_data['Parameter'].append('Animation Generated')
            summary_data['Value'].append('Yes' if self.chk_save_animation.isChecked() else 'No')

            if self.chk_save_plots.isChecked():
                summary_data['Parameter'].append('Plot Downsampling')
                summary_data['Value'].append(
                    f'{self.spin_plot_downsample_hz.value()} Hz'
                    if self.chk_downsample_plots.isChecked() else 'Off (full resolution)')
            if self.chk_save_animation.isChecked():
                summary_data['Parameter'].append('Animation Downsampling')
                summary_data['Value'].append(
                    f'{self.spin_anim_downsample_hz.value()} Hz'
                    if self.chk_downsample_animation.isChecked() else 'Off (full resolution)')

            summary_data['Parameter'].append('Proximity Detection')
            if self.chk_proximity_detection.isChecked():
                summary_data['Value'].append(f'Yes (threshold: {self.spin_proximity_threshold.value()} m)')
            else:
                summary_data['Value'].append('No')
            
            # Create DataFrame and save
            summary_df = pd.DataFrame(summary_data)
            summary_df.to_csv(summary_path, index=False)
            
            self.log_message(f"✓ Run summary saved: {os.path.basename(summary_path)}")
        except Exception as e:
            self.log_message(f"Warning: Could not save run summary: {str(e)}")
    
    def initUI(self):
        self.setWindowTitle("UWB PreProcessing Tool")
        self.setGeometry(50, 50, 1500, 950)
        
        # Set dark theme style
        self.setStyleSheet("""
            QWidget {
                background-color: #2b2b2b;
                color: #cccccc;
                font-family: Arial;
            }
            QLabel {
                color: #cccccc;
                background-color: transparent;
            }
            QPushButton {
                background-color: #0078d4;
                color: white;
                border: none;
                padding: 8px 16px;
                border-radius: 4px;
                font-weight: bold;
                min-width: 100px;
            }
            QPushButton:hover {
                background-color: #106ebe;
            }
            QPushButton:pressed {
                background-color: #005a9e;
            }
            QPushButton:disabled {
                background-color: #3f3f3f;
                color: #666666;
            }
            QGroupBox {
                background-color: #1e1e1e;
                border: 1px solid #3f3f3f;
                border-radius: 6px;
                margin-top: 12px;
                padding-top: 12px;
                font-weight: bold;
                color: #cccccc;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top left;
                padding: 4px 8px;
                background-color: #2b2b2b;
                color: #0078d4;
            }
            QCheckBox {
                color: #cccccc;
                spacing: 8px;
            }
            QComboBox {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #3f3f3f;
                border-radius: 4px;
                padding: 4px 8px;
                min-width: 100px;
            }
            QScrollArea {
                border: none;
            }
            QScrollBar:vertical {
                background-color: #2b2b2b;
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background-color: #0078d4;
                border-radius: 4px;
                min-height: 20px;
            }
            QScrollBar::handle:vertical:hover {
                background-color: #106ebe;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                height: 0px;
            }
            QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {
                background: none;
            }
            QScrollBar:horizontal {
                background-color: #2b2b2b;
                height: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:horizontal {
                background-color: #0078d4;
                border-radius: 4px;
                min-width: 20px;
            }
            QScrollBar::handle:horizontal:hover {
                background-color: #106ebe;
            }
            QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {
                width: 0px;
            }
            QScrollBar::add-page:horizontal, QScrollBar::sub-page:horizontal {
                background: none;
            }
        """)

        # Settings column on the left, live preview on the right — the same
        # left-controls / right-display layout as the other FNT tools. The
        # preview is always present and simply sits empty until a database is
        # loaded; no data is read until tags exist and the user scrubs/plays.
        self.main_splitter = QSplitter(Qt.Horizontal)
        self.main_splitter.setChildrenCollapsible(False)
        self.main_splitter.setHandleWidth(5)
        # Build the preview pane (canvases) first: the left column's Preview
        # Options group wires signals whose slots touch the canvases, so the
        # canvases must exist before that group is constructed.
        self.preview_panel = self.create_preview_panel()
        settings_panel = self.create_settings_panel()
        self.main_splitter.addWidget(settings_panel)
        self.main_splitter.addWidget(self.preview_panel)
        self.main_splitter.setStretchFactor(0, 0)
        self.main_splitter.setStretchFactor(1, 1)
        self.main_splitter.setSizes([520, 980])

        main_layout = QVBoxLayout()
        main_layout.addWidget(self.main_splitter)
        self.setLayout(main_layout)

        # Scrubbing fires a continuous stream of slider values; only fetch
        # where the user actually stops.
        self.preview_scrub_timer = QTimer(self)
        self.preview_scrub_timer.setSingleShot(True)
        self.preview_scrub_timer.timeout.connect(self._on_scrub_settled)

        # Changing the trail length redraws instantly from cache, but the chunks
        # also carry a lead-in overlap sized to the trail; reloading them keeps
        # the trail continuous across chunk boundaries. Debounced so holding the
        # spinbox arrow doesn't reload on every step.
        self.preview_trail_timer = QTimer(self)
        self.preview_trail_timer.setSingleShot(True)
        self.preview_trail_timer.timeout.connect(self.invalidate_preview_cache)

        # ← / → arrow scrubbing (1 second per step). App-level filter so the
        # keys work regardless of which widget has focus.
        QApplication.instance().installEventFilter(self)

        # Select All/None and config loading toggle every tag checkbox in a
        # burst; coalesce them into one preview rebuild.
        self.preview_tag_timer = QTimer(self)
        self.preview_tag_timer.setSingleShot(True)
        self.preview_tag_timer.timeout.connect(self._apply_tag_selection_change)

        # Anything upstream of the rendered frames invalidates cached chunks,
        # otherwise the pane would keep showing data processed under the old
        # settings — which would quietly defeat the whole point of comparing
        # smoothing methods here.
        for w in (self.combo_smoothing, self.combo_timezone):
            w.currentTextChanged.connect(self.invalidate_preview_cache)
        for w in (self.spin_rolling_window, self.spin_velocity_threshold,
                  self.spin_jump_threshold, self.spin_time_gap):
            w.valueChanged.connect(self.invalidate_preview_cache)
        for w in (self.chk_velocity_filter, self.chk_jump_filter):
            w.stateChanged.connect(self.invalidate_preview_cache)
        
    def _make_downsample_control(self, which):
        """Build a 'Downsample data for <which> to [N] Hz' sub-row.

        Returns (row_widget, checkbox, spinbox). Used to give Save Plots and
        Save Animation each their own independent downsampling control.
        """
        row = QWidget()
        lay = QHBoxLayout()
        lay.setContentsMargins(0, 0, 0, 0)
        chk = QCheckBox(f"Downsample data for {which} to")
        chk.setChecked(True)  # on by default
        chk.setToolTip(
            f"Thin the data used to render {which} to the chosen rate. Does NOT "
            "affect the exported smoothed CSV (which stays full resolution) or "
            "proximity detection.\n"
            "\n"
            "UWB tags can log several Hz during movement, so 1 Hz typically cuts "
            f"{which} time several-fold with no visible change in the tracks. A "
            "temporary working file is written as the render source and deleted "
            "when the export finishes.")
        lay.addWidget(chk)
        spin = QSpinBox()
        spin.setRange(1, 5)
        spin.setValue(1)
        spin.setSuffix(" Hz")
        spin.setFixedWidth(70)
        spin.setToolTip(
            "Target sample rate: keeps the first fix in each 1/Hz-second bin, "
            "per tag.")
        lay.addWidget(spin)
        lay.addStretch()
        row.setLayout(lay)
        chk.toggled.connect(spin.setEnabled)  # spinbox active only when ticked
        return row, chk, spin

    def create_settings_panel(self):
        """Create the settings panel as a single scrollable column"""
        panel = QWidget()
        layout = QVBoxLayout()

        # Database selection
        db_group = QGroupBox("Database Selection")
        db_layout = QVBoxLayout()

        btn_select = QPushButton("Select SQLite Database")
        btn_select.setToolTip("Open a UWB SQLite database file (.db) for preprocessing")
        btn_select.clicked.connect(self.select_database)
        db_layout.addWidget(btn_select)

        self.lbl_db = QLabel("No database selected")
        self.lbl_db.setStyleSheet("color: #666666; font-style: italic;")
        self.lbl_db.setWordWrap(True)
        db_layout.addWidget(self.lbl_db)

        self.lbl_config_status = QLabel("")
        self.lbl_config_status.setStyleSheet("color: #ff4444; font-style: italic; font-size: 9px;")
        self.lbl_config_status.setVisible(False)
        db_layout.addWidget(self.lbl_config_status)

        table_layout = QHBoxLayout()
        table_layout.addWidget(QLabel("Table:"))
        self.combo_table = QComboBox()
        self.combo_table.setEnabled(False)
        self.combo_table.setToolTip("Select which table in the database contains the UWB tracking data")
        self.combo_table.currentTextChanged.connect(self.on_table_selected)
        table_layout.addWidget(self.combo_table)
        db_layout.addLayout(table_layout)

        self.btn_preview_table = QPushButton("Preview Table")
        self.btn_preview_table.setToolTip("Show the first rows of the selected table to verify correct data")
        self.btn_preview_table.clicked.connect(self.preview_table)
        self.btn_preview_table.setEnabled(False)
        db_layout.addWidget(self.btn_preview_table)

        db_group.setLayout(db_layout)
        layout.addWidget(db_group)

        # Timezone
        tz_group = QGroupBox("Timezone")
        tz_group_layout = QVBoxLayout()
        tz_layout = QHBoxLayout()
        tz_layout.addWidget(QLabel("Timezone:"))
        self.combo_timezone = QComboBox()
        self.combo_timezone.setToolTip(
            "Timestamps in the database are stored as UTC epoch milliseconds. "
            "This setting converts them to your local timezone for all outputs, "
            "plots, and the identity dialog time pickers."
        )
        common_timezones = [
            "US/Mountain", "US/Pacific", "US/Central", "US/Eastern",
            "UTC", "Europe/London", "Europe/Paris", "Asia/Tokyo"
        ]
        self.combo_timezone.addItems(common_timezones)
        self.combo_timezone.setCurrentText("US/Mountain")
        tz_layout.addWidget(self.combo_timezone)
        tz_group_layout.addLayout(tz_layout)
        tz_group.setLayout(tz_group_layout)
        layout.addWidget(tz_group)

        # Tag selection
        self.tag_group = QGroupBox("Tag Selection")
        self.tag_layout = QVBoxLayout()
        self.tag_checkboxes = {}

        self.lbl_no_tags = QLabel("Load a database to see available tags")
        self.lbl_no_tags.setStyleSheet("color: #666666; font-style: italic;")
        self.tag_layout.addWidget(self.lbl_no_tags)

        self.tag_group.setLayout(self.tag_layout)
        layout.addWidget(self.tag_group)

        # Preview Options group now sits directly under Tag Selection (above the
        # export controls). It is a checkable "Enable Preview and Playback"
        # group, off by default, and also hosts the background/anchor toggles.
        layout.addWidget(self._build_preview_options_group())

        # Export Options. The velocity/jump thresholds and the smoothing
        # method/window controls lead this group; the CSV / plot / animation
        # options follow below. (These threshold/smoothing controls formerly
        # lived in their own "Smoothing & Filtering Options" group.)
        export_group = QGroupBox("Export Options")
        export_layout = QVBoxLayout()

        velocity_filter_layout = QHBoxLayout()
        self.chk_velocity_filter = QCheckBox("Velocity threshold (remove >")
        self.chk_velocity_filter.setChecked(True)
        self.chk_velocity_filter.setToolTip(
            "Discard fixes whose computed velocity exceeds the threshold. "
            "Catches teleportation artifacts from multipath/reflection errors.\n"
            "\n"
            "Applied BEFORE smoothing, on the raw fixes — velocity is measured "
            "between consecutive unsmoothed positions, and offending fixes are "
            "removed before the smoothing pass runs."
        )
        velocity_filter_layout.addWidget(self.chk_velocity_filter)
        self.spin_velocity_threshold = QDoubleSpinBox()
        self.spin_velocity_threshold.setRange(0.1, 10.0)
        self.spin_velocity_threshold.setValue(2.0)
        self.spin_velocity_threshold.setSuffix(" m/s)")
        self.spin_velocity_threshold.setDecimals(1)
        self.spin_velocity_threshold.setSingleStep(0.1)
        self.spin_velocity_threshold.setToolTip(
            "Points moving faster than this are removed. "
            "2 m/s is a good default for rodents in an arena."
        )
        velocity_filter_layout.addWidget(self.spin_velocity_threshold)
        velocity_filter_layout.addStretch()
        export_layout.addLayout(velocity_filter_layout)

        jump_filter_layout = QHBoxLayout()
        self.chk_jump_filter = QCheckBox("Jump threshold (remove >")
        self.chk_jump_filter.setChecked(True)
        self.chk_jump_filter.setToolTip(
            "Discard fixes where the spatial jump between consecutive samples "
            "exceeds the threshold. Catches single-sample position glitches.\n"
            "\n"
            "Applied BEFORE smoothing, on the raw fixes — the jump distance is "
            "measured between consecutive unsmoothed positions, and offending "
            "fixes are removed before the smoothing pass runs."
        )
        jump_filter_layout.addWidget(self.chk_jump_filter)
        self.spin_jump_threshold = QDoubleSpinBox()
        self.spin_jump_threshold.setRange(0.1, 10.0)
        self.spin_jump_threshold.setValue(2.0)
        self.spin_jump_threshold.setSuffix(" m)")
        self.spin_jump_threshold.setDecimals(1)
        self.spin_jump_threshold.setSingleStep(0.1)
        self.spin_jump_threshold.setToolTip(
            "Consecutive points farther apart than this distance are removed. "
            "2 m is a good default for typical arena sizes."
        )
        jump_filter_layout.addWidget(self.spin_jump_threshold)
        jump_filter_layout.addStretch()
        export_layout.addLayout(jump_filter_layout)

        time_gap_layout = QHBoxLayout()
        time_gap_layout.addWidget(QLabel("Time gap grouping:"))
        self.spin_time_gap = QSpinBox()
        self.spin_time_gap.setRange(5, 300)
        self.spin_time_gap.setValue(30)
        self.spin_time_gap.setSuffix(" sec")
        self.spin_time_gap.setToolTip(
            "Splits data into segments when gaps exceed this duration. "
            "Prevents the velocity/jump thresholds from comparing points across "
            "battery restarts or signal dropouts."
        )
        time_gap_layout.addWidget(self.spin_time_gap)
        time_gap_layout.addStretch()
        export_layout.addLayout(time_gap_layout)

        smoothing_label_layout = QHBoxLayout()
        smoothing_label_layout.addWidget(QLabel("Smoothing method:"))
        self.combo_smoothing = QComboBox()
        self.combo_smoothing.setToolTip(SMOOTHING_METHODS_TOOLTIP)
        self.combo_smoothing.addItems([
            "None (default)", "Forward-Backward Exponentially Weighted Moving Average", "Savitzky-Golay",
            "Rolling Median", "Rolling Average"
        ])
        self.combo_smoothing.setCurrentIndex(0)
        self.combo_smoothing.currentTextChanged.connect(self.on_smoothing_changed)
        smoothing_label_layout.addWidget(self.combo_smoothing)
        export_layout.addLayout(smoothing_label_layout)

        self.rolling_window_layout = QHBoxLayout()
        self.lbl_rolling_window = QLabel("Smoothing Window:")
        self.rolling_window_layout.addWidget(self.lbl_rolling_window)
        self.spin_rolling_window = QSpinBox()
        # Range covers both interpretations: up to 600 s (10 min) of time-based
        # window, or up to 600 samples.
        self.spin_rolling_window.setRange(1, 600)
        self.spin_rolling_window.setValue(30)
        self.spin_rolling_window.setEnabled(False)
        self.spin_rolling_window.setToolTip(
            "Size of the smoothing window. How it is read depends on the units "
            "selector to the right and on the smoothing method:\n"
            "\n"
            "• Rolling Average / Rolling Median, units = Seconds: a real-time "
            "window (e.g. 30 = a 30-second window centred on each fix). The "
            "amount of smoothing is independent of the reporting rate.\n"
            "• Rolling Average / Rolling Median, units = Samples: a fixed count "
            "of consecutive fixes (e.g. 30 = 30 fixes). Its wall-clock span "
            "varies with the reporting rate.\n"
            "• Forward-Backward Exponentially Weighted Moving Average: the span (always in samples), which sets "
            "the decay rate (alpha = 2 / (span + 1)). To match a Wiser hardware "
            "filter value F, use span = 2F - 1.\n"
            "\n"
            "Larger values produce smoother trajectories but lose fine detail. "
            "The window is centred in every mode, so smoothing adds no lag."
        )
        self.rolling_window_layout.addWidget(self.spin_rolling_window)

        # Units selector: does the number above mean seconds or samples? Only
        # meaningful for the rolling methods (EWMA/Savitzky-Golay ignore it), so
        # on_smoothing_changed shows/hides it with the method.
        self.combo_window_units = QComboBox()
        self.combo_window_units.addItems(["Seconds", "Samples"])
        self.combo_window_units.setCurrentText("Seconds")
        self.combo_window_units.setToolTip(
            "How to interpret the Smoothing Window value (Rolling Average / "
            "Rolling Median only):\n"
            "\n"
            "• Seconds (default): a real-time window evaluated on each tag's "
            "timestamps. A value of 30 averages all fixes within a 30-second "
            "window centred on each point, so the smoothing is the same amount "
            "of real time everywhere regardless of how many fixes were "
            "recorded. Robust to the tags' irregular, sub-1 Hz reporting.\n"
            "• Samples: the window is a fixed number of consecutive fixes. A "
            "value of 30 averages 30 fixes; the real-time span it covers grows "
            "and shrinks with the reporting rate, so identical settings smooth "
            "different amounts of time on sparsely- vs densely-sampled tags."
        )
        self.combo_window_units.currentTextChanged.connect(self.invalidate_preview_cache)
        self.rolling_window_layout.addWidget(self.combo_window_units)
        self.rolling_window_layout.addStretch()
        export_layout.addLayout(self.rolling_window_layout)
        # Sync the window control's visibility to the default method now.
        # setCurrentIndex above ran before the signal was connected, so
        # on_smoothing_changed never fired — sync the Smoothing Window's
        # visibility to the initial method (hidden for the None default,
        # shown once a rolling method is chosen).
        self.on_smoothing_changed(self.combo_smoothing.currentText())

        # ---- CSV / plot / animation export options (continue the same group) --
        self.chk_export_raw_csv = QCheckBox("Export Raw CSV")
        self.chk_export_raw_csv.setChecked(False)
        self.chk_export_raw_csv.setToolTip(
            "Dump the raw database table to CSV with no filtering, smoothing, or unit conversion. "
            "Useful as a reference baseline."
        )
        export_layout.addWidget(self.chk_export_raw_csv)

        self.chk_export_smoothed_csv = QCheckBox("Export Smoothed CSV")
        self.chk_export_smoothed_csv.setChecked(True)
        self.chk_export_smoothed_csv.setToolTip(
            "Export filtered and smoothed data at full temporal resolution. "
            "Coordinates are in meters, timestamps in your selected timezone.\n"
            "\n"
            "This is also the data source for plots and animations: when it is "
            "on, they use the smoothed track; when it is off, they fall back to "
            "the unsmoothed (filtered) positions. To downsample, do it in your "
            "own downstream analysis of this CSV."
        )
        export_layout.addWidget(self.chk_export_smoothed_csv)

        self.chk_proximity_detection = QCheckBox("Detect Proximity Bouts")
        self.chk_proximity_detection.setChecked(True)
        self.chk_proximity_detection.setToolTip(
            "Detect when pairs of tags are within the proximity threshold. "
            "Outputs one CSV of aggregated proximity bouts: for each pair, the "
            "contiguous periods spent within the threshold, with start/stop, "
            "duration, mean distance and observation count.\n"
            "\n"
            "Computed from the filtered + smoothed data at full temporal "
            "resolution — NOT the downsampled export — so no fixes are dropped "
            "before pairwise distances are measured.\n"
            "\n"
            "Bouts are threshold-specific: to analyze a different distance, "
            "re-run with a new threshold. The raw per-timestamp pairwise "
            "distances are not exported."
        )
        self.chk_proximity_detection.stateChanged.connect(self.on_proximity_detection_toggled)
        export_layout.addWidget(self.chk_proximity_detection)

        self.proximity_threshold_widget = QWidget()
        prox_layout = QHBoxLayout()
        prox_layout.setContentsMargins(30, 0, 0, 0)
        prox_layout.addWidget(QLabel("Proximity threshold:"))
        self.spin_proximity_threshold = QDoubleSpinBox()
        self.spin_proximity_threshold.setRange(0.01, 10.0)
        self.spin_proximity_threshold.setValue(0.5)
        self.spin_proximity_threshold.setSingleStep(0.05)
        self.spin_proximity_threshold.setDecimals(2)
        self.spin_proximity_threshold.setSuffix(" m")
        self.spin_proximity_threshold.setToolTip(
            "Two tags closer than this distance are considered in sociospatial proximity. "
            "0.5 m is a typical threshold for rodent social contact."
        )
        self.spin_proximity_threshold.setFixedWidth(100)
        prox_layout.addWidget(self.spin_proximity_threshold)
        prox_layout.addStretch()
        self.proximity_threshold_widget.setLayout(prox_layout)
        self.proximity_threshold_widget.setVisible(True)
        export_layout.addWidget(self.proximity_threshold_widget)

        self.chk_save_plots = QCheckBox("Save Plots")
        self.chk_save_plots.setChecked(False)  # off by default; opt in per run
        self.chk_save_plots.stateChanged.connect(self.on_save_plots_toggled)
        self.chk_save_plots.setToolTip(
            "Generate and save visualization plots to the plots/ subfolder. "
            "Built from the filtered + smoothed data. "
            "PNG is always produced; SVG is optional below."
        )
        export_layout.addWidget(self.chk_save_plots)

        self.plot_types_widget = QWidget()
        plot_types_layout = QVBoxLayout()
        plot_types_layout.setContentsMargins(30, 0, 0, 0)
        # Per-plots downsampling of the render source (a sub-option of Save Plots).
        plot_ds_row, self.chk_downsample_plots, self.spin_plot_downsample_hz = \
            self._make_downsample_control("plots")
        plot_types_layout.addWidget(plot_ds_row)
        self.plot_type_checkboxes = {}
        plot_types = [
            ("daily_paths", "Daily Paths per Tag",
             "One plot per tag showing XY trajectory for each day, color-coded by date"),
            ("trajectory_overview", "Trajectory Overview",
             "All selected tags overlaid on a single plot with optional background image"),
            ("battery_levels", "Battery Levels",
             "Battery voltage over time for each tag — useful for detecting low-power dropouts"),
            ("3d_occupancy", "3D Occupancy Heatmap",
             "3D surface plot of spatial occupancy density per tag"),
            ("activity_timeline", "Activity Timeline",
             "Data points per hour across the recording — reveals active vs. inactive periods"),
            ("velocity_distribution", "Velocity Distribution",
             "Histogram of movement speeds per tag to characterize locomotion patterns"),
            ("cumulative_distance", "Cumulative Distance",
             "Total distance traveled over time per tag, reset at midnight each day"),
            ("velocity_timeline", "Velocity Timeline",
             "Velocity over time per tag with an activity threshold line overlay"),
            ("actogram", "Circadian Actogram",
             "Double-plotted 24-hour activity raster showing circadian patterns across days"),
            ("data_quality", "Data Quality Metrics",
             "Summary table of data gaps, sample counts, and coverage statistics per tag")
        ]
        for key, plot_name, plot_desc in plot_types:
            cb = QCheckBox(plot_name)
            cb.setChecked(True)
            cb.setToolTip(plot_desc)
            self.plot_type_checkboxes[key] = cb
            plot_types_layout.addWidget(cb)
        self.plot_types_widget.setLayout(plot_types_layout)
        self.plot_types_widget.setVisible(False)  # hidden until Save Plots is on
        export_layout.addWidget(self.plot_types_widget)

        self.svg_option_widget = QWidget()
        svg_option_layout = QHBoxLayout()
        svg_option_layout.setContentsMargins(30, 0, 0, 0)
        self.chk_save_svg = QCheckBox("Also save as SVG")
        self.chk_save_svg.setChecked(False)
        self.chk_save_svg.setToolTip(
            "Save each plot as a scalable vector graphic in addition to PNG. "
            "Useful for publication-quality figures that can be edited in Illustrator."
        )
        svg_option_layout.addWidget(self.chk_save_svg)
        svg_option_layout.addStretch()
        self.svg_option_widget.setLayout(svg_option_layout)
        self.svg_option_widget.setVisible(False)  # hidden until Save Plots is on
        export_layout.addWidget(self.svg_option_widget)

        self.chk_save_animation = QCheckBox("Save Animation")
        self.chk_save_animation.setChecked(False)
        self.chk_save_animation.stateChanged.connect(self.on_save_animation_toggled)
        self.chk_save_animation.setToolTip(
            "Render an MP4 video of tag trajectories over time, from the "
            "filtered + smoothed data. "
            "Can take a long time for multi-day recordings at high quality."
        )
        export_layout.addWidget(self.chk_save_animation)

        self.animation_options_widget = QWidget()
        animation_options_layout = QVBoxLayout()
        animation_options_layout.setContentsMargins(30, 0, 0, 0)

        # Per-animation downsampling of the render source (a sub-option of
        # Save Animation), independent of the plots downsampling.
        anim_ds_row, self.chk_downsample_animation, self.spin_anim_downsample_hz = \
            self._make_downsample_control("animation")
        animation_options_layout.addWidget(anim_ds_row)

        trail_layout = QHBoxLayout()
        trail_layout.addWidget(QLabel("Trail length (seconds):"))
        self.spin_animation_trail = QSpinBox()
        self.spin_animation_trail.setRange(1, 1000)
        self.spin_animation_trail.setValue(500)
        self.spin_animation_trail.setToolTip(
            "How many seconds of trailing path to draw behind each tag in the animation. "
            "Higher values show more historical movement context."
        )
        trail_layout.addWidget(self.spin_animation_trail)
        animation_options_layout.addLayout(trail_layout)

        color_layout = QHBoxLayout()
        color_layout.addWidget(QLabel("Color by:"))
        self.combo_color_by = QComboBox()
        self.combo_color_by.addItems(["None", "ID", "sex"])
        self.combo_color_by.setCurrentText("None")
        self.combo_color_by.setToolTip(
            "How trajectories are coloured. None: all the same. ID: a unique "
            "colour per tag. sex: blue = male, red = female (needs identities).")
        color_layout.addWidget(self.combo_color_by, 1)
        animation_options_layout.addLayout(color_layout)

        # Which animals to render (default: all configured tags).
        inctag_layout = QHBoxLayout()
        inctag_layout.addWidget(QLabel("Include tags:"))
        self.lbl_anim_tags = QLabel("All configured tags")
        self.lbl_anim_tags.setStyleSheet("color:#aaaaaa; font-style:italic;")
        inctag_layout.addWidget(self.lbl_anim_tags, 1)
        self.btn_anim_tags = QPushButton("Select…")
        self.btn_anim_tags.setFixedWidth(80)
        self.btn_anim_tags.setToolTip("Choose a subset of tags to include in the animation")
        self.btn_anim_tags.clicked.connect(self.select_animation_tags)
        inctag_layout.addWidget(self.btn_anim_tags)
        animation_options_layout.addLayout(inctag_layout)

        # Marker diameter (points) for the rendered video. Mirrors the preview's
        # 'Tag Display Size' so a size dialled in during preview matches the export.
        anim_tagsize_layout = QHBoxLayout()
        anim_tagsize_layout.addWidget(QLabel("Tag Display Size:"))
        self.spin_anim_tag_size = QSpinBox()
        self.spin_anim_tag_size.setRange(2, 40)
        self.spin_anim_tag_size.setValue(10)
        self.spin_anim_tag_size.setSuffix(" pts")
        self.spin_anim_tag_size.setToolTip(
            "Diameter (in points) of each tag's marker in the exported animation. "
            "Match this to the preview's 'Tag Display Size' so the video looks "
            "like what you tuned in the preview window.")
        anim_tagsize_layout.addWidget(self.spin_anim_tag_size)
        anim_tagsize_layout.addStretch(1)
        animation_options_layout.addLayout(anim_tagsize_layout)

        self.chk_show_battery_export = QCheckBox("Display Tag Battery Levels")
        self.chk_show_battery_export.setChecked(False)
        self.chk_show_battery_export.setToolTip(
            "Show each tag's battery voltage in small black text under its ID "
            "label in the exported animation. Off by default.")
        animation_options_layout.addWidget(self.chk_show_battery_export)

        speed_layout = QHBoxLayout()
        speed_layout.addWidget(QLabel("Animation Speed:"))
        self.combo_animation_speed = QComboBox()
        self.combo_animation_speed.addItems(["1x", "5x", "15x", "30x", "60x", "120x", "240x"])
        self.combo_animation_speed.setCurrentText("60x")
        self.combo_animation_speed.setToolTip(
            "How fast real time plays in the video. "
            "80x means 80 seconds of tracking data per 1 second of video."
        )
        speed_layout.addWidget(self.combo_animation_speed)
        animation_options_layout.addLayout(speed_layout)

        fps_layout = QHBoxLayout()
        fps_layout.addWidget(QLabel("FPS:"))
        self.combo_animation_fps = QComboBox()
        self.combo_animation_fps.addItems(["1", "5", "10", "20", "30"])
        self.combo_animation_fps.setCurrentText("20")
        self.combo_animation_fps.setToolTip(
            "Video frames per second. Higher values are smoother but "
            "produce larger files and take longer to render."
        )
        fps_layout.addWidget(self.combo_animation_fps)
        animation_options_layout.addLayout(fps_layout)

        quality_layout = QHBoxLayout()
        quality_layout.addWidget(QLabel("Video Quality:"))
        self.combo_video_quality = QComboBox()
        self.combo_video_quality.addItems(["Draft (Fast)", "Standard", "High Quality"])
        self.combo_video_quality.setCurrentText("High Quality")
        self.combo_video_quality.setToolTip(
            "Render resolution: Draft (75 dpi, ~4x faster), "
            "Standard (100 dpi), High Quality (150 dpi). "
            "Draft is good for quick previews."
        )
        quality_layout.addWidget(self.combo_video_quality)
        animation_options_layout.addLayout(quality_layout)

        self.lbl_estimated_frames = QLabel("Estimated frames: -- (load data first)")
        self.lbl_estimated_frames.setStyleSheet("color: #888; font-size: 10pt; font-style: italic;")
        animation_options_layout.addWidget(self.lbl_estimated_frames)

        self.combo_animation_speed.currentTextChanged.connect(self.update_frame_estimate)
        self.combo_animation_fps.currentTextChanged.connect(self.update_frame_estimate)

        self.chk_daily_animations = QCheckBox("Generate daily animations (one per day)")
        self.chk_daily_animations.setChecked(False)
        self.chk_daily_animations.stateChanged.connect(self.on_daily_animations_toggled)
        self.chk_daily_animations.setToolTip(
            "Create a separate MP4 for each calendar day (midnight to midnight) "
            "instead of one long video for the entire recording."
        )
        animation_options_layout.addWidget(self.chk_daily_animations)

        self.daily_animation_days_widget = QWidget()
        daily_days_layout = QVBoxLayout()
        daily_days_layout.setContentsMargins(20, 5, 0, 5)
        self.daily_animation_day_checkboxes = {}
        self.daily_days_layout_inner = QVBoxLayout()
        daily_days_layout.addLayout(self.daily_days_layout_inner)
        self.daily_animation_days_widget.setLayout(daily_days_layout)
        self.daily_animation_days_widget.setVisible(False)
        animation_options_layout.addWidget(self.daily_animation_days_widget)

        self.animation_options_widget.setLayout(animation_options_layout)
        self.animation_options_widget.setVisible(False)  # hidden until Save Animation is on
        export_layout.addWidget(self.animation_options_widget)

        export_group.setLayout(export_layout)
        layout.addWidget(export_group)

        # Progress bar (hidden by default)
        self.progress_widget = QWidget()
        progress_layout = QVBoxLayout()
        progress_layout.setContentsMargins(0, 5, 0, 5)
        self.lbl_export_progress = QLabel("")
        self.lbl_export_progress.setStyleSheet("color: #00aa00; font-weight: bold;")
        progress_layout.addWidget(self.lbl_export_progress)
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setStyleSheet("""
            QProgressBar {
                border: 1px solid #555555;
                border-radius: 3px;
                text-align: center;
                background-color: #1e1e1e;
            }
            QProgressBar::chunk {
                background-color: #0078d4;
            }
        """)
        progress_layout.addWidget(self.progress_bar)
        self.progress_widget.setLayout(progress_layout)
        self.progress_widget.setVisible(False)
        layout.addWidget(self.progress_widget)

        # Export buttons
        export_buttons_layout = QHBoxLayout()
        self.btn_export = QPushButton("Export")
        self.btn_export.clicked.connect(self.export_data)
        self.btn_export.setEnabled(False)
        self.btn_export.setToolTip(
            "Run the full preprocessing pipeline: filter, smooth, export CSVs, "
            "generate plots/animations, and save config to the output folder."
        )
        self.btn_export.setStyleSheet("padding: 8px; font-size: 11px; font-weight: bold;")
        export_buttons_layout.addWidget(self.btn_export)
        self.btn_stop_export = QPushButton("Stop Export")
        self.btn_stop_export.clicked.connect(self.stop_export)
        self.btn_stop_export.setToolTip("Cancel the current export operation")
        self.btn_stop_export.setStyleSheet("padding: 8px; font-size: 11px; font-weight: bold; background-color: #d41100;")
        self.btn_stop_export.setVisible(False)
        export_buttons_layout.addWidget(self.btn_stop_export)
        layout.addLayout(export_buttons_layout)

        # Session Logs window
        messages_label = QLabel("Session Logs:")
        messages_label.setStyleSheet("font-weight: bold; margin-top: 10px;")
        layout.addWidget(messages_label)

        self.txt_messages = QTextEdit()
        self.txt_messages.setReadOnly(True)
        self.txt_messages.setMaximumHeight(150)
        self.txt_messages.setStyleSheet("""
            QTextEdit {
                background-color: #1e1e1e;
                color: #cccccc;
                border: 1px solid #555555;
                padding: 5px;
                font-family: Consolas, monospace;
                font-size: 9px;
            }
        """)
        layout.addWidget(self.txt_messages)

        self.btn_copy_logs = QPushButton("Copy Session Logs to Clipboard")
        self.btn_copy_logs.setToolTip("Copy the full session log above to the clipboard for pasting into a bug report or notes")
        self.btn_copy_logs.setStyleSheet("padding: 6px; font-size: 10px;")
        self.btn_copy_logs.clicked.connect(self.copy_session_logs)
        layout.addWidget(self.btn_copy_logs)

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color: #666666; font-style: italic; font-size: 10px;")
        self.lbl_status.setVisible(False)
        layout.addWidget(self.lbl_status)

        layout.addStretch()
        panel.setLayout(layout)

        scroll = QScrollArea()
        scroll.setWidget(panel)
        scroll.setWidgetResizable(True)

        return scroll

    # ------------------------------------------------------------------ #
    # Preview pane
    # ------------------------------------------------------------------ #
    def create_preview_panel(self):
        """Right-hand display: the arena map on top, transport controls below.

        Every configuration control lives in the left column's 'Preview
        Options' group (see _build_preview_options_group); this pane is kept
        deliberately spare so the map gets the room.
        """
        panel = QWidget()
        layout = QVBoxLayout()

        # --- render surface (the map) ---
        self.preview_stack = QWidget()
        stack_layout = QVBoxLayout()
        stack_layout.setContentsMargins(0, 0, 0, 0)
        self.preview_canvas_2d = UWBPreview2D(self)
        self.preview_canvas_2d.set_theme("light")   # light is the default view
        stack_layout.addWidget(self.preview_canvas_2d)
        self.preview_canvas_3d = None
        if PREVIEW_HAVE_GL:
            try:
                self.preview_canvas_3d = UWBPreview3D(self)
                self.preview_canvas_3d.setVisible(False)
                stack_layout.addWidget(self.preview_canvas_3d)
            except Exception as e:
                self.preview_canvas_3d = None
                print(f"3D preview unavailable: {e}")
        self.preview_stack.setLayout(stack_layout)
        layout.addWidget(self.preview_stack, 1)

        # --- transport (scrubber only), directly under the map ---
        self.slider_timeline = QSlider(Qt.Horizontal)
        self.slider_timeline.setRange(0, 0)
        self.slider_timeline.setToolTip(
            "Scrub the whole recording. Drag, or use the ← / → arrow keys "
            "(hold to scan) to step back and forth. Nothing is read while you "
            "move — the chunk under the playhead loads once you stop.")
        self.slider_timeline.valueChanged.connect(self.on_timeline_moved)
        layout.addWidget(self.slider_timeline)

        self.lbl_preview_time = QLabel("--")
        self.lbl_preview_time.setStyleSheet("color: #cccccc; font-family: Consolas, monospace; font-size: 10px;")
        self.lbl_preview_time.setAlignment(Qt.AlignCenter)
        layout.addWidget(self.lbl_preview_time)

        panel.setLayout(layout)
        panel.setMinimumWidth(520)
        return panel

    def _build_preview_options_group(self):
        """All preview configuration, as a group for the left settings column.

        The controls formerly crowded under the map now live here so the right
        pane can be just the map plus its transport bar.

        The group is checkable ("Enable Preview") and off by default: loading a
        database does not start streaming until the user opts in. Qt disables
        every child of an unchecked checkable group, so the preview controls
        (and the background/zone/anchor toggles hosted here) stay greyed out
        until the preview is enabled.

        The preview is a streamlined scrubber: drag the timeline to move through
        the recording as a clean 2D top-down view, with optional zone/background/
        anchor overlays and identity labels. No playback, no camera, no arena
        registration — those richer views moved to the UWB Studio.
        """
        group = QGroupBox("Preview")
        v = QVBoxLayout()

        # The view/arena-registration widgets are retained but hidden: the
        # streamlined preview is fixed to the 2D idealized top-down view. They
        # stay instantiated (2D / Auto / zero offset) so the streaming + arena
        # engine keeps working unchanged.
        self._legacy_view_holder = QWidget()
        legacy = QVBoxLayout()
        legacy.setContentsMargins(0, 0, 0, 0)
        self.combo_view_mode = QComboBox()
        self.combo_view_mode.addItems([self.VIEW_XML, self.VIEW_2D, self.VIEW_3D])
        self.combo_view_mode.setCurrentText(self.VIEW_2D)
        self.combo_view_mode.currentTextChanged.connect(self.on_preview_backend_changed)
        legacy.addWidget(self.combo_view_mode)
        self.combo_arena = QComboBox()
        self.combo_arena.addItem("Auto (fit to data)")
        for name in BUILTIN_ARENAS:
            self.combo_arena.addItem(name)
        self.combo_arena.currentTextChanged.connect(self.refresh_preview_arena)
        legacy.addWidget(self.combo_arena)
        self.spin_arena_dx = QDoubleSpinBox(); self.spin_arena_dx.setRange(-500.0, 500.0)
        self.spin_arena_dx.valueChanged.connect(self.refresh_preview_arena)
        legacy.addWidget(self.spin_arena_dx)
        self.spin_arena_dy = QDoubleSpinBox(); self.spin_arena_dy.setRange(-500.0, 500.0)
        self.spin_arena_dy.valueChanged.connect(self.refresh_preview_arena)
        legacy.addWidget(self.spin_arena_dy)
        self._legacy_view_holder.setLayout(legacy)
        self._legacy_view_holder.setVisible(False)
        v.addWidget(self._legacy_view_holder)

        # Background image + anchor display (relocated here from the old
        # Smoothing & Filtering group; these are display concerns for the
        # preview and saved plots).
        bg_buttons_layout = QHBoxLayout()
        self.btn_load_background = QPushButton("Load Background")
        self.btn_load_background.clicked.connect(self.select_background_image)
        self.btn_load_background.setEnabled(False)
        self.btn_load_background.setToolTip(
            "Load a floorplan or arena image to overlay under trajectory plots. "
            "Requires an XML config in the database folder for spatial scaling."
        )
        self.btn_load_background.setStyleSheet("padding: 8px; font-size: 11px;")
        bg_buttons_layout.addWidget(self.btn_load_background)
        self.btn_remove_background = QPushButton("Remove Background")
        self.btn_remove_background.clicked.connect(self.remove_background)
        self.btn_remove_background.setEnabled(False)
        self.btn_remove_background.setToolTip("Clear the loaded background image from all visualizations")
        self.btn_remove_background.setStyleSheet("padding: 8px; font-size: 11px;")
        bg_buttons_layout.addWidget(self.btn_remove_background)
        v.addLayout(bg_buttons_layout)

        self.lbl_background_status = QLabel("No background image loaded")
        self.lbl_background_status.setStyleSheet("color: #666666; font-style: italic; font-size: 9px;")
        self.lbl_background_status.setWordWrap(True)
        v.addWidget(self.lbl_background_status)

        self.chk_show_background = QCheckBox("Show background image")
        self.chk_show_background.setChecked(True)
        self.chk_show_background.setToolTip(
            "Toggle the loaded floorplan/background image in the LIVE PREVIEW "
            "only, without unloading it. Exported plots and animations take "
            "their layers from the dialog shown when you click Export.")
        self.chk_show_background.stateChanged.connect(self.on_show_background_toggled)
        v.addWidget(self.chk_show_background)

        # Manual background transform (option (b)): scale + X/Y origin offset so
        # the loaded image can be nudged live onto the anchor/track frame — the
        # way the Wiser server lets you place a floorplan. Needed when the loaded
        # image is a different render (resolution / decorated margins) than the
        # one the XML scale was calibrated for, so pixels*scale is wrong or the
        # arena content sits inset from the image edges. Defaults seed from the
        # XML; live values feed the preview AND exported plots/animation.
        self._bg_transform_box = QGroupBox("Background alignment")
        self._bg_transform_box.setToolTip(
            "Rescale and reposition the loaded background image so it lines up "
            "with the anchors and tracks. Scale is inches per pixel (larger = "
            "bigger image); offsets shift the image's bottom-left corner in "
            "metres. Defaults come from the site XML. Anchors/tracks use "
            "absolute coordinates and never move.")
        bg_tf = QGridLayout()
        bg_tf.setContentsMargins(6, 2, 6, 2)
        bg_tf.addWidget(QLabel("Scale (in/px):"), 0, 0)
        self.spin_bg_scale = QDoubleSpinBox()
        self.spin_bg_scale.setRange(0.0001, 1000.0)
        self.spin_bg_scale.setDecimals(4)
        self.spin_bg_scale.setSingleStep(0.005)
        self.spin_bg_scale.setToolTip(
            "Inches per pixel. The image's physical size = pixels x this x "
            "0.0254 m. Increase to enlarge the image, decrease to shrink it.")
        self.spin_bg_scale.valueChanged.connect(self.on_bg_transform_changed)
        bg_tf.addWidget(self.spin_bg_scale, 0, 1)
        bg_tf.addWidget(QLabel("Offset X (m):"), 1, 0)
        self.spin_bg_offx = QDoubleSpinBox()
        self.spin_bg_offx.setRange(-1000.0, 1000.0)
        self.spin_bg_offx.setDecimals(2)
        self.spin_bg_offx.setSingleStep(0.10)
        self.spin_bg_offx.setToolTip("Shift the image right (+) / left (-), in metres.")
        self.spin_bg_offx.valueChanged.connect(self.on_bg_transform_changed)
        bg_tf.addWidget(self.spin_bg_offx, 1, 1)
        bg_tf.addWidget(QLabel("Offset Y (m):"), 2, 0)
        self.spin_bg_offy = QDoubleSpinBox()
        self.spin_bg_offy.setRange(-1000.0, 1000.0)
        self.spin_bg_offy.setDecimals(2)
        self.spin_bg_offy.setSingleStep(0.10)
        self.spin_bg_offy.setToolTip("Shift the image up (+) / down (-), in metres.")
        self.spin_bg_offy.valueChanged.connect(self.on_bg_transform_changed)
        bg_tf.addWidget(self.spin_bg_offy, 2, 1)
        self.btn_bg_reset = QPushButton("Reset to XML")
        self.btn_bg_reset.setToolTip(
            "Restore the scale to the XML value (matched to this image's "
            "resolution) and the offset to (0, 0).")
        self.btn_bg_reset.clicked.connect(self.reset_bg_transform)
        bg_tf.addWidget(self.btn_bg_reset, 3, 0, 1, 2)
        self._bg_transform_box.setLayout(bg_tf)
        self._bg_transform_box.setEnabled(False)   # enabled once an image loads
        v.addWidget(self._bg_transform_box)

        self.chk_show_anchors = QCheckBox("Show anchor positions")
        self.chk_show_anchors.setChecked(True)
        self.chk_show_anchors.setEnabled(False)
        self.chk_show_anchors.stateChanged.connect(self.on_show_background_toggled)
        self.chk_show_anchors.setToolTip(
            "Draw UWB anchor/antenna positions as triangles in the LIVE PREVIEW "
            "only. Exported plots/animations take their layers from the dialog "
            "shown when you click Export. Anchors are parsed from the XML config."
        )
        v.addWidget(self.chk_show_anchors)

        self.chk_show_zones = QCheckBox("Show zone coordinates")
        self.chk_show_zones.setChecked(True)
        self.chk_show_zones.setEnabled(False)   # enabled once XML zones are parsed
        self.chk_show_zones.setToolTip(
            "Draw the XML-derived zone polygons over the 2D preview, in their "
            "authored colours. Preview only. Zones are parsed from the site XML.")
        self.chk_show_zones.stateChanged.connect(self.on_show_background_toggled)
        v.addWidget(self.chk_show_zones)

        self.chk_show_tracking = QCheckBox("Show Tag Tracking Data")
        self.chk_show_tracking.setChecked(True)
        self.chk_show_tracking.setToolTip(
            "Draw the tag position markers and trailing tracks in the LIVE "
            "PREVIEW. Turn off to inspect just the arena / background / anchors "
            "without any tracking data on top.")
        self.chk_show_tracking.stateChanged.connect(self.on_show_background_toggled)
        v.addWidget(self.chk_show_tracking)

        # Marker diameter in points. Matches the animation export's Tag Display
        # Size so a size dialled in here reads the same in the rendered video —
        # the preview marker was previously smaller than the export.
        tagsize_row = QHBoxLayout()
        tagsize_row.addWidget(QLabel("Tag Display Size:"))
        self.spin_tag_size = QSpinBox()
        self.spin_tag_size.setRange(2, 40)
        self.spin_tag_size.setValue(10)
        self.spin_tag_size.setSuffix(" pts")
        self.spin_tag_size.setToolTip(
            "Diameter (in points) of each tag's position marker in the preview. "
            "Set the same value in the animation export's 'Tag Display Size' to "
            "make the video markers match what you see here.")
        self.spin_tag_size.valueChanged.connect(self.on_show_background_toggled)
        tagsize_row.addWidget(self.spin_tag_size)
        tagsize_row.addStretch(1)
        v.addLayout(tagsize_row)

        self.chk_show_battery = QCheckBox("Display Tag Battery Levels")
        self.chk_show_battery.setChecked(False)
        self.chk_show_battery.setToolTip(
            "Label each tag in the preview with its ID and, beneath it, the "
            "current battery voltage (small black text). Off by default.")
        self.chk_show_battery.stateChanged.connect(self.on_show_background_toggled)
        v.addWidget(self.chk_show_battery)

        self.chk_preview_dark = QCheckBox("Dark mode")
        self.chk_preview_dark.setChecked(False)   # light by default
        self.chk_preview_dark.setToolTip(
            "Dark background for the preview. Off by default — the light theme "
            "(white background, tan arena floor) is better for figures and "
            "screenshots.")
        self.chk_preview_dark.stateChanged.connect(self.on_preview_theme_changed)
        v.addWidget(self.chk_preview_dark)

        # Preview thresholding — independent of the Export thresholds, so you can
        # see the track with/without each applied AND tune the cutoff live. Each
        # toggle carries its own value spinbox (defaults match Export: 2.0).
        pvel_row = QHBoxLayout()
        self.chk_preview_velocity = QCheckBox("Velocity threshold (remove >")
        self.chk_preview_velocity.setChecked(True)
        self.chk_preview_velocity.setToolTip(
            "Apply a velocity threshold to the PREVIEW track. Toggle to compare "
            "with/without; adjust the value to see the cutoff's effect live.")
        self.chk_preview_velocity.stateChanged.connect(self.invalidate_preview_cache)
        pvel_row.addWidget(self.chk_preview_velocity)
        self.spin_preview_velocity = QDoubleSpinBox()
        self.spin_preview_velocity.setRange(0.1, 10.0)
        self.spin_preview_velocity.setValue(2.0)
        self.spin_preview_velocity.setSuffix(" m/s)")
        self.spin_preview_velocity.setDecimals(1)
        self.spin_preview_velocity.setSingleStep(0.1)
        self.spin_preview_velocity.setToolTip(
            "Remove preview fixes whose speed from the previous fix exceeds this.")
        self.spin_preview_velocity.valueChanged.connect(self.invalidate_preview_cache)
        pvel_row.addWidget(self.spin_preview_velocity)
        pvel_row.addStretch()
        v.addLayout(pvel_row)

        pjump_row = QHBoxLayout()
        self.chk_preview_jump = QCheckBox("Jump threshold (remove >")
        self.chk_preview_jump.setChecked(True)
        self.chk_preview_jump.setToolTip(
            "Apply a distance-jump threshold to the PREVIEW track. Toggle to "
            "compare with/without; adjust the value to see the cutoff's effect live.")
        self.chk_preview_jump.stateChanged.connect(self.invalidate_preview_cache)
        pjump_row.addWidget(self.chk_preview_jump)
        self.spin_preview_jump = QDoubleSpinBox()
        self.spin_preview_jump.setRange(0.1, 10.0)
        self.spin_preview_jump.setValue(2.0)
        self.spin_preview_jump.setSuffix(" m)")
        self.spin_preview_jump.setDecimals(1)
        self.spin_preview_jump.setSingleStep(0.1)
        self.spin_preview_jump.setToolTip(
            "Remove preview fixes that leap more than this distance from the "
            "previous fix.")
        self.spin_preview_jump.valueChanged.connect(self.invalidate_preview_cache)
        pjump_row.addWidget(self.spin_preview_jump)
        pjump_row.addStretch()
        v.addLayout(pjump_row)

        color_row = QHBoxLayout()
        color_row.addWidget(QLabel("Color by:"))
        self.combo_preview_color = QComboBox()
        self.combo_preview_color.addItems(["None", "ID", "Sex"])
        self.combo_preview_color.setCurrentText("None")
        self.combo_preview_color.setToolTip(
            "How preview tags are coloured. None: all the same. ID: a unique "
            "colour per tag. Sex: blue = male, red = female (from identities).")
        self.combo_preview_color.currentTextChanged.connect(self.on_preview_color_changed)
        color_row.addWidget(self.combo_preview_color, 1)
        v.addLayout(color_row)

        # Smoothing method: how the live track is drawn — None (actual fixes) or
        # a smoothing method, so you can compare before committing to export.
        smooth_row = QHBoxLayout()
        smooth_row.addWidget(QLabel("Smoothing method:"))
        self.combo_preview_smoothing = QComboBox()
        self.combo_preview_smoothing.addItems([
            "None", "Forward-Backward Exponentially Weighted Moving Average", "Savitzky-Golay",
            "Rolling Median", "Rolling Average"])
        self.combo_preview_smoothing.setCurrentText("None")
        self.combo_preview_smoothing.setToolTip(
            "Preview only: see how each method looks before committing to one for\n"
            "export. It reads the same Smoothing Window / units set in Export\n"
            "Options, so the preview matches the exported result.\n"
            "\n" + SMOOTHING_METHODS_TOOLTIP)
        self.combo_preview_smoothing.currentTextChanged.connect(self.on_preview_smoothing_changed)
        smooth_row.addWidget(self.combo_preview_smoothing, 1)
        v.addLayout(smooth_row)

        # Smoothing Window for the preview — its own controls (independent of
        # Export Options) so you can see live how the window size and the
        # seconds/samples interpretation change the track. on_preview_smoothing_
        # changed shows/hides these per method (None/Savitzky-Golay expose
        # nothing, EWMA hides the units since its span is always in samples).
        pwin_row = QHBoxLayout()
        self.lbl_preview_window = QLabel("Smoothing Window:")
        pwin_row.addWidget(self.lbl_preview_window)
        self.spin_preview_window = QSpinBox()
        self.spin_preview_window.setRange(1, 600)
        self.spin_preview_window.setValue(30)
        self.spin_preview_window.setToolTip(
            "Size of the preview smoothing window. Read as seconds or samples "
            "per the units selector; for Forward-Backward Exponentially Weighted Moving Average it is the span "
            "(alpha = 2 / (span + 1)). Larger = smoother, less detail.")
        self.spin_preview_window.valueChanged.connect(self.invalidate_preview_cache)
        pwin_row.addWidget(self.spin_preview_window)
        self.combo_preview_window_units = QComboBox()
        self.combo_preview_window_units.addItems(["Seconds", "Samples"])
        self.combo_preview_window_units.setCurrentText("Seconds")
        self.combo_preview_window_units.setToolTip(
            "How to read the preview Smoothing Window (Rolling Average / Median "
            "only): Seconds = a real-time window, independent of the reporting "
            "rate; Samples = a fixed count of consecutive fixes.")
        self.combo_preview_window_units.currentTextChanged.connect(self.invalidate_preview_cache)
        pwin_row.addWidget(self.combo_preview_window_units)
        pwin_row.addStretch()
        v.addLayout(pwin_row)
        # Set initial visibility for the default method (None → hidden).
        self.on_preview_smoothing_changed()

        disp_row = QHBoxLayout()
        disp_row.addWidget(QLabel("Trail:"))
        self.spin_preview_trail = QSpinBox()
        self.spin_preview_trail.setRange(0, 3600)     # up to one hour of path
        self.spin_preview_trail.setValue(60)
        self.spin_preview_trail.setSuffix(" s")
        self.spin_preview_trail.setSingleStep(30)
        self.spin_preview_trail.setToolTip(
            "Seconds of trailing track drawn behind each tag's current position.")
        self.spin_preview_trail.valueChanged.connect(self.on_preview_trail_changed)
        disp_row.addWidget(self.spin_preview_trail)
        disp_row.addStretch()
        v.addLayout(disp_row)

        # Chunk length + binning rate are streaming internals: kept at sensible
        # defaults but hidden so the streamlined preview stays simple.
        self._preview_stream_holder = QWidget()
        stream_l = QHBoxLayout(); stream_l.setContentsMargins(0, 0, 0, 0)
        self.spin_preview_minutes = QSpinBox()
        self.spin_preview_minutes.setRange(1, 60)
        self.spin_preview_minutes.setValue(10)
        self.spin_preview_minutes.valueChanged.connect(self.invalidate_preview_cache)
        stream_l.addWidget(self.spin_preview_minutes)
        self.combo_preview_hz = QComboBox()
        self.combo_preview_hz.addItems(["1 Hz", "2 Hz", "5 Hz"])
        self.combo_preview_hz.currentTextChanged.connect(self.invalidate_preview_cache)
        stream_l.addWidget(self.combo_preview_hz)
        self._preview_stream_holder.setLayout(stream_l)
        self._preview_stream_holder.setVisible(False)
        v.addWidget(self._preview_stream_holder)

        self.lbl_preview_status = QLabel("Load a database and select tags to preview")
        self.lbl_preview_status.setStyleSheet("color: #888888; font-style: italic; font-size: 9px;")
        self.lbl_preview_status.setWordWrap(True)
        v.addWidget(self.lbl_preview_status)

        self.lbl_cache_status = QLabel("")
        self.lbl_cache_status.setStyleSheet("color: #666666; font-size: 9px;")
        v.addWidget(self.lbl_cache_status)

        # The fast index builds/adopts automatically when the preview loads
        # (see ensure_preview_index_db) — no manual button needed. _set_index_status
        # writes into lbl_cache_status when a build is running.
        self.preview_options_body = QWidget()
        self.preview_options_body.setLayout(v)
        outer = QVBoxLayout()
        outer.setContentsMargins(0, 0, 0, 0)
        outer.addWidget(self.preview_options_body)
        group.setLayout(outer)

        # No enable toggle — the preview loads automatically when a database and
        # tags are available (see activate_preview, driven by the tag-selection
        # and load paths). The body is always visible.
        self.grp_preview = group
        return group

    def activate_preview(self):
        """Bring the always-visible preview to life once data is available.

        The pane is shown from startup (empty), mirroring the other FNT tools.
        This is called when a database, table and at least one tag exist.
        Enabling the preview now builds the indexed copy automatically (for the
        fastest possible scrubbing); reads stream from the original meanwhile
        and switch over transparently once the copy is ready.

        The preview loads automatically once a database, table and at least one
        tag are available — there is no enable toggle.
        """
        if not (self.db_path and self.table_name and self.selected_preview_tags()):
            return
        self._preview_active = True
        # Build/adopt the fast index automatically for instant scrubbing.
        self.ensure_preview_index_db()
        self.refresh_preview_arena()
        if self.init_preview_timeline():
            self.preview_playhead_ms = self.preview_t0
            self._sync_timeline_to_playhead()
            self._request_chunk(0, make_current=True)

    def eventFilter(self, obj, event):
        """App-level ← / → scrubbing so the arrows work regardless of focus.

        One second per step — including while held (auto-repeat just steps again
        at the OS repeat rate); no acceleration. The timeline slider's unit is
        one second. Arrows keep their normal behaviour only while a value editor
        (spinbox / text field) is focused, or a combo's dropdown is actually
        open — a merely focused-but-closed combo (e.g. right after picking a
        smoothing option) does NOT block scrubbing.
        """
        from PyQt5.QtCore import QEvent
        from PyQt5.QtWidgets import QAbstractSpinBox, QComboBox, QLineEdit
        if (event.type() == QEvent.KeyPress
                and event.key() in (Qt.Key_Left, Qt.Key_Right)
                and getattr(self, '_preview_active', False)
                and self.slider_timeline.maximum() > 0):
            fw = QApplication.focusWidget()
            editing = isinstance(fw, (QAbstractSpinBox, QLineEdit))
            combo_open = isinstance(fw, QComboBox) and fw.view().isVisible()
            if not editing and not combo_open:
                delta = -1 if event.key() == Qt.Key_Left else 1
                self.slider_timeline.setValue(int(np.clip(
                    self.slider_timeline.value() + delta,
                    self.slider_timeline.minimum(), self.slider_timeline.maximum())))
                return True
        return super().eventFilter(obj, event)

    def on_preview_enabled_toggled(self, enabled):
        """Enable/disable the live preview when the group checkbox is toggled.

        Checking it brings the preview to life if a database, table and tags are
        already loaded; otherwise it waits and the normal tag-selection path
        activates it later. Unchecking it tears the stream down and drops the
        pane back to its empty placeholder.

        The controls also collapse when disabled so the group folds down to just
        its title/checkbox instead of showing a greyed-out wall of options.
        """
        if hasattr(self, 'preview_options_body'):
            self.preview_options_body.setVisible(enabled)
        if enabled:
            if self.db_path and self.table_name and self.selected_preview_tags():
                self.activate_preview()
            else:
                self.lbl_preview_status.setText("Load a database and select tags to preview")
        else:
            self._preview_active = False
            self.stop_preview_playback()
            self.preview_canvas_2d.show_placeholder()
            if self.preview_canvas_3d is not None:
                self.preview_canvas_3d.clear()
            self.lbl_preview_status.setText("Preview disabled — tick 'Enable Preview and Playback' to view data")

    def _preview_backend(self):
        """The canvas currently on screen."""
        if (self.combo_view_mode.currentText() == self.VIEW_3D
                and self.preview_canvas_3d is not None):
            return self.preview_canvas_3d
        return self.preview_canvas_2d

    def on_preview_backend_changed(self):
        mode = self.combo_view_mode.currentText()
        if mode == self.VIEW_3D and self.preview_canvas_3d is None:
            self.log_message(
                f"3D view unavailable ({PREVIEW_GL_ERROR or 'pyqtgraph.opengl missing'}) "
                f"— staying in {self.VIEW_2D}")
            self.combo_view_mode.setCurrentText(self.VIEW_2D)
            return

        use_3d = mode == self.VIEW_3D and self.preview_canvas_3d is not None
        self.preview_canvas_2d.setVisible(not use_3d)
        if self.preview_canvas_3d is not None:
            self.preview_canvas_3d.setVisible(use_3d)

        # Arena source only applies to the idealized views; the XML view takes
        # its geometry entirely from the site config.
        is_xml = mode == self.VIEW_XML
        self.combo_arena.setEnabled(not is_xml)
        self.spin_arena_dx.setEnabled(not is_xml)
        self.spin_arena_dy.setEnabled(not is_xml)

        self.refresh_preview_arena()

    def build_xml_arena(self):
        """Arena built from the site XML: surveyed zones, map image, anchors."""
        pts = [z["points"] for z in self.xml_zones if len(z["points"])]
        if pts:
            allp = np.vstack(pts)
            x0, y0 = float(allp[:, 0].min()), float(allp[:, 1].min())
            x1, y1 = float(allp[:, 0].max()), float(allp[:, 1].max())
        elif self.xml_map_extent:
            x0, x1, y0, y1 = self.xml_map_extent
        else:
            return None

        if self.xml_map_extent:   # keep the whole map in frame
            mx0, mx1, my0, my1 = self.xml_map_extent
            x0, y0 = min(x0, mx0), min(y0, my0)
            x1, y1 = max(x1, mx1), max(y1, my1)

        pad = max(x1 - x0, y1 - y0) * 0.03
        return PreviewArena(
            width=(x1 - x0) + 2 * pad, height=(y1 - y0) + 2 * pad,
            origin_x=x0 - pad, origin_y=y0 - pad,
            anchors=list(self.anchor_positions or []),
            zones=list(self.xml_zones),
            map_image=self.xml_map_image, map_extent=self.xml_map_extent,
            label="XML (site map)")

    def refresh_preview_arena(self):
        """Rebuild arena geometry for the selected view mode and offsets."""
        if not hasattr(self, "combo_arena"):
            return
        mode = self.combo_view_mode.currentText()
        dx = self.spin_arena_dx.value()
        dy = self.spin_arena_dy.value()
        arena = None

        if mode == self.VIEW_XML:
            arena = self.build_xml_arena()
            if arena is None:
                self.log_message(
                    "No zones or map in the site XML — falling back to 2D Idealized")
                self.combo_view_mode.setCurrentText(self.VIEW_2D)
                return

        if arena is None:
            name = self.combo_arena.currentText()
            if name in BUILTIN_ARENAS:
                arena = BUILTIN_ARENAS[name](origin_x=dx, origin_y=dy)
                arena.anchors = list(self.anchor_positions or [])
            else:
                # Frame to the union of everything on screen so the site is
                # never cropped: data, plus the XML zones and map extent when
                # present. (fit_arena_to_data already folds in the anchors.)
                xs_parts, ys_parts = [], []
                if self.preview_x is not None and len(self.preview_x):
                    xs_parts.append(self.preview_x.ravel())
                    ys_parts.append(self.preview_y.ravel())
                for z in (self.xml_zones or []):
                    pts = z.get('points')
                    if pts is not None and len(pts):
                        pts = np.asarray(pts, float)
                        xs_parts.append(pts[:, 0]); ys_parts.append(pts[:, 1])
                if self.xml_map_extent:
                    mx0, mx1, my0, my1 = self.xml_map_extent
                    xs_parts.append(np.array([mx0, mx1])); ys_parts.append(np.array([my0, my1]))
                xs = np.concatenate(xs_parts) if xs_parts else None
                ys = np.concatenate(ys_parts) if ys_parts else None
                arena = fit_arena_to_data(xs, ys, self.anchor_positions)
                arena.origin_x += dx
                arena.origin_y += dy

        if not self.chk_show_anchors.isChecked():
            arena.anchors = []

        # Zone overlay for the streamlined 2D preview: the data-fit arena carries
        # no zones, so inject the XML zones when the user wants them shown. (In
        # the retained XML view the arena already has them; re-setting is a
        # harmless no-op.)
        if getattr(self, 'chk_show_zones', None) is not None and self.chk_show_zones.isChecked() and self.xml_zones:
            arena.zones = list(self.xml_zones)
        elif getattr(self, 'chk_show_zones', None) is not None and not self.chk_show_zones.isChecked():
            arena.zones = []

        self.preview_arena = arena
        self.preview_canvas_2d.set_arena(arena)
        self.preview_canvas_2d.show_anchors = self.chk_show_anchors.isChecked()
        # Background-image mapping (UWB world coords, metres):
        #   • data location_x/y are inches -> metres via *0.0254
        #   • image footprint = pixels * xml_scale(in/px) * 0.0254
        #   • placed at extent [0, W, 0, H] so the image's bottom-left corner
        #     is world (0, 0) — the Wiser reference frame
        #   • drawn origin="upper" (see UWBPreview2D.update_frame): the image is
        #     shown UPRIGHT, north at top, exactly as the file looks
        #   • the canvas expands the view to include the whole image so it is
        #     never clipped by the data-fit arena bounds
        # Honour the show/hide toggle: only hand the canvas an image when the
        # user has the background enabled.
        show_bg = self.background_image is not None and self.chk_show_background.isChecked()
        self.preview_canvas_2d.background_image = self.background_image if show_bg else None
        self.preview_canvas_2d.bg_extent = self._bg_placed_extent() if show_bg else None
        if self.preview_canvas_3d is not None:
            self.preview_canvas_3d.set_arena(arena)
        self.render_preview_frame()

    def auto_register_arena(self):
        """Centre the selected enclosure on the loaded data's bounding box."""
        if self.preview_x is None or not len(self.preview_x):
            QMessageBox.information(self, "No Preview Data",
                                    "Load a preview window first so there is data to register against.")
            return
        name = self.combo_arena.currentText()
        if name not in BUILTIN_ARENAS:
            QMessageBox.information(self, "Auto-Fit",
                                    "Auto-Fit applies to a named enclosure. "
                                    "'Auto (fit to data)' already tracks the data.")
            return

        xs = self.preview_x[np.isfinite(self.preview_x)]
        ys = self.preview_y[np.isfinite(self.preview_y)]
        if not len(xs) or not len(ys):
            return
        data_cx = (float(xs.min()) + float(xs.max())) / 2.0
        data_cy = (float(ys.min()) + float(ys.max())) / 2.0

        base = BUILTIN_ARENAS[name](0.0, 0.0)
        self.spin_arena_dx.setValue(data_cx - base.width / 2.0)
        self.spin_arena_dy.setValue(data_cy - base.height / 2.0)
        self.log_message(f"Arena registered: offset ({self.spin_arena_dx.value():.2f}, "
                         f"{self.spin_arena_dy.value():.2f}) m")

    # -- streaming chunk engine ------------------------------------------- #
    # The timeline spans the whole recording, but memory holds at most
    # MAX_CACHED_CHUNKS slices. Scrubbing is debounced so dragging costs
    # nothing; only where you stop is fetched, on a background thread.
    MAX_CACHED_CHUNKS = 6
    SCRUB_DEBOUNCE_MS = 70

    def preview_time_bounds(self, selected_tags):
        """(min_ts, max_ts) epoch-ms across the selected tags, or None."""
        try:
            conn = sqlite3.connect(self.preview_db_path or self.db_path)
            placeholders = ",".join(["?"] * len(selected_tags))
            row = conn.execute(
                f"SELECT MIN(timestamp), MAX(timestamp) FROM {self.table_name} "
                f"WHERE shortid IN ({placeholders})", selected_tags).fetchone()
            conn.close()
            if row and row[0] is not None:
                return int(row[0]), int(row[1])
        except Exception as e:
            self.log_message(f"Could not read time bounds: {e}")
        return None

    def selected_preview_tags(self):
        return [tag for tag, cb in self.tag_checkboxes.items() if cb.isChecked()]

    def on_tag_selection_changed(self, _state=None):
        """Tag selection changed — refresh the preview for the new tag set.

        Coalesced through a timer because Select All/None and config loading
        toggle every checkbox in a burst, and each one emits separately.
        """
        if self._tag_selection_guard:
            return
        # First tags for a freshly-loaded database bring the preview to life;
        # later changes just refresh it.
        if not self._preview_active:
            if self.db_path and self.table_name and self.selected_preview_tags():
                self.preview_tag_timer.start(150)
            return
        self.preview_tag_timer.start(150)

    def _apply_tag_selection_change(self):
        """Rebuild the timeline and cached chunks for the current tag set."""
        if not (self.db_path and self.table_name):
            return
        if not self._preview_active:
            self.activate_preview()
            return
        tags = self.selected_preview_tags()
        if not tags:
            self.stop_preview_playback()
            self.preview_cache.clear()
            self.preview_x = None
            self._update_cache_label()
            self.lbl_preview_status.setText("No tags selected")
            self.preview_canvas_2d.clear()
            if self.preview_canvas_3d is not None:
                self.preview_canvas_3d.clear()
            return

        # Switching databases with the preview already open lands here (config
        # load re-checks the boxes), and that path never runs the enable
        # handler, so make sure the new database gets its indexed copy.
        if self.preview_db_path is None:
            self.ensure_preview_index_db()

        # The recording bounds are per-tag, so adding or removing a tag can
        # change the span the timeline has to cover.
        keep = self.preview_playhead_ms
        if self.init_preview_timeline():
            if self.preview_t0 <= keep <= self.preview_t1:
                self.preview_playhead_ms = keep     # stay where the user was
            else:
                self.preview_playhead_ms = self.preview_t0
            self._sync_timeline_to_playhead()
        self.invalidate_preview_cache()

    def invalidate_preview_cache(self):
        """Drop cached chunks. Called whenever anything upstream of the frames
        changes — filtering, smoothing, timezone, tag selection, chunk size —
        so the pane can never show data processed with stale settings."""
        self.preview_cache.clear()
        self.preview_x = None
        self._update_cache_label()
        if self._preview_active and self.preview_t0 is not None:
            # Re-fetch whatever the playhead is sitting on under the new settings
            self._request_chunk(self._chunk_index_for(self.preview_playhead_ms),
                                make_current=True)

    def init_preview_timeline(self):
        """Size the timeline slider to the full recording for the selected tags."""
        tags = self.selected_preview_tags()
        if not self.db_path or not self.table_name or not tags:
            return False
        bounds = self.preview_time_bounds(tags)
        if not bounds:
            self.lbl_preview_status.setText("No records for the selected tags")
            return False

        self.preview_t0, self.preview_t1 = bounds
        total_s = max(1, (self.preview_t1 - self.preview_t0) // 1000)
        self._timeline_guard = True
        self.slider_timeline.setRange(0, int(total_s))
        # Arrow-key / hold scrubbing: ~1000 steps across the recording per
        # single press, ~50 per Page. Fine for fine control, fast when held.
        self.slider_timeline.setSingleStep(max(1, int(total_s) // 1000))
        self.slider_timeline.setPageStep(max(1, int(total_s) // 50))
        self._timeline_guard = False
        days = total_s / 86400.0
        self.log_message(
            f"Preview timeline: {days:.1f} days "
            f"({pd.Timestamp(self.preview_t0, unit='ms', tz='UTC').tz_convert(self.combo_timezone.currentText()):%Y-%m-%d %H:%M} "
            f"to {pd.Timestamp(self.preview_t1, unit='ms', tz='UTC').tz_convert(self.combo_timezone.currentText()):%Y-%m-%d %H:%M})")
        return True

    def _chunk_ms(self):
        return self.spin_preview_minutes.value() * 60 * 1000

    def _chunk_index_for(self, ts_ms):
        if self.preview_t0 is None:
            return 0
        return max(0, int((ts_ms - self.preview_t0) // self._chunk_ms()))

    def _chunk_bounds(self, idx):
        c = self._chunk_ms()
        start = self.preview_t0 + idx * c
        return start, start + c

    def _resident_chunk_for(self, ts_ms):
        """Index of the cached chunk whose span contains ``ts_ms``, else None.

        Chunks are fixed-width and index-aligned, so the only candidate is
        ``_chunk_index_for(ts_ms)``; return it only when it is actually cached.
        """
        idx = self._chunk_index_for(ts_ms)
        c = self.preview_cache.get(idx)
        if c is not None and c["t_start"] <= ts_ms <= c["t_end"]:
            return idx
        return None

    def _prefetch_neighbors(self, idx):
        """Warm the chunks on either side so a continuous scrub stays in cache.

        ``_request_chunk`` no-ops for anything already cached or in flight, so
        calling this on every chunk switch is cheap and self-limiting.
        """
        for nb in (idx + 1, idx - 1):
            if (nb >= 0 and nb not in self.preview_cache
                    and nb not in self.preview_inflight):
                self._request_chunk(nb, make_current=False)

    def _update_cache_label(self):
        if not hasattr(self, "lbl_cache_status"):
            return
        n = len(self.preview_cache)
        kb = sum(c["nbytes"] for c in self.preview_cache.values()) / 1024.0
        self.lbl_cache_status.setText(
            f"{n}/{self.MAX_CACHED_CHUNKS} chunks · {kb:.0f} KB" if n else "")

    # -- request / receive ------------------------------------------------- #
    def _request_chunk(self, idx, make_current=False):
        """Fetch a chunk in the background unless it is cached or in flight."""
        if self.preview_t0 is None or idx < 0:
            return
        if idx in self.preview_cache:
            self.preview_cache.move_to_end(idx)
            if make_current:
                self._show_chunk(idx)
            return
        if idx in self.preview_inflight:
            if make_current:
                self.preview_pending_current = idx
            return

        tags = self.selected_preview_tags()
        if not tags:
            return
        start, end = self._chunk_bounds(idx)
        if start > self.preview_t1:
            return
        # Load a lead-in before the chunk so smoothing and forward-fill are
        # warmed and the trailing track is continuous across the boundary — a
        # tag's position and its tail no longer reset when you scrub from one
        # chunk into the next. Bounded by the chunk width: a trail longer than a
        # chunk can't be shown in full anyway (frames are chunk-local).
        lead = min(self.spin_preview_trail.value() * 1000, self._chunk_ms())
        load_start = max(self.preview_t0, start - lead)

        if make_current:
            self.preview_pending_current = idx
            self.lbl_preview_status.setText("Loading…")

        loader = PreviewChunkLoader(self.preview_db_path or self.db_path,
                                    self.table_name, tags, load_start, end, idx)
        loader.loaded.connect(self._on_chunk_loaded)
        loader.failed.connect(self._on_chunk_failed)
        # The inflight dict holds the only strong reference to the thread.
        # Dropping it inside the finished handler can free the QThread during
        # signal emission ("Destroyed while thread is still running"), so defer
        # the drop to the next event-loop turn, once emission has unwound.
        loader.finished.connect(lambda i=idx: QTimer.singleShot(
            0, lambda: self._retire_loader(i)))
        self.preview_inflight[idx] = loader
        loader.start()

    def _retire_loader(self, idx):
        """Release a finished loader. Safe: runs outside the finished signal."""
        loader = self.preview_inflight.get(idx)
        if loader is None:
            return
        if not loader.isFinished():
            # Slot raced ahead of the thread actually stopping; try again.
            QTimer.singleShot(50, lambda: self._retire_loader(idx))
            return
        self.preview_inflight.pop(idx, None)

    def _on_chunk_failed(self, idx, err):
        self.log_message(f"Preview chunk {idx} failed: {err}")
        if self.preview_pending_current == idx:
            self.preview_pending_current = None
            self.lbl_preview_status.setText(f"Load failed: {err}")

    def _on_chunk_loaded(self, idx, df):
        """Process a delivered slice on the main thread and cache it.

        Filtering and smoothing run here rather than in the worker because they
        read the settings widgets directly, and they are a negligible share of
        the cost (the SQL read dominates).
        """
        try:
            frames = self._process_chunk(df, idx) if df is not None and len(df) else None
        except Exception as e:
            self.log_message(f"Preview chunk {idx} processing failed: {e}")
            frames = None

        if frames is not None:
            self.preview_cache[idx] = frames
            self.preview_cache.move_to_end(idx)
            while len(self.preview_cache) > self.MAX_CACHED_CHUNKS:
                self.preview_cache.popitem(last=False)   # evict least-recent
            self._update_cache_label()

        if self.preview_pending_current == idx:
            self.preview_pending_current = None
            if frames is None:
                self._handle_empty_chunk(idx)
            else:
                self._show_chunk(idx)

    def _handle_empty_chunk(self, idx):
        """Chunk had no samples — jump to the next real data.

        Real recordings contain outages (Echo T001 has a ~14 h gap), and the
        timeline is linear in wall-clock time, so a position can land in dead
        air. Rather than showing an empty scene, find the next sample and move
        the playhead there.
        """
        tags = self.selected_preview_tags()
        start, _ = self._chunk_bounds(idx)
        try:
            conn = sqlite3.connect(self.preview_db_path or self.db_path)
            nxt = self._nearest_sample_ts(conn, tags, start)
            conn.close()
        except Exception:
            nxt = None

        if nxt is None or self._chunk_index_for(nxt) == idx:
            self.lbl_preview_status.setText("No samples here — try elsewhere on the timeline.")
            return

        gap_min = abs(nxt - start) / 60000.0
        self.log_message(f"No data at that position — skipped {gap_min:.0f} min to the next samples.")
        self.lbl_preview_status.setText(f"Gap in recording — skipped {gap_min:.0f} min forward.")
        self.preview_playhead_ms = nxt
        self._sync_timeline_to_playhead()
        self._request_chunk(self._chunk_index_for(nxt), make_current=True)

    def get_preview_smoothing_method(self):
        """Smoothing method for the LIVE preview (its own 'Smoothing method'
        selector, independent of the export method). 'None' shows raw fixes."""
        return (self.combo_preview_smoothing.currentText()
                if hasattr(self, 'combo_preview_smoothing') else 'None')

    def _process_chunk(self, df, idx=None):
        """Filter, smooth and bin one slice into frame arrays, or None if empty.

        ``idx`` is the chunk index; when given, the frames' nominal span is set
        from it (not from the data), so the lead-in overlap the loader prepends
        stays outside the scrubbable/containment range.
        """
        tz = pytz.timezone(self.combo_timezone.currentText())
        smoothing_method = self.get_preview_smoothing_method()
        # Preview thresholding toggles are independent of the export ones, so the
        # user can compare the track with/without each threshold live.
        use_vel = self.chk_preview_velocity.isChecked()
        use_jump = self.chk_preview_jump.isChecked()
        do_filter = use_vel or use_jump
        tags = self.selected_preview_tags()

        chunks = []
        for tag, g in df.groupby('shortid', sort=False):
            g = g.copy()
            g['Timestamp'] = pd.to_datetime(
                g['timestamp'], unit='ms', origin='unix', utc=True).dt.tz_convert(tz)
            g['location_x'] *= 0.0254
            g['location_y'] *= 0.0254
            # Keep pre-smoothing fixes so the pane can overlay raw vs smoothed
            g['raw_x'] = g['location_x']
            g['raw_y'] = g['location_y']
            if do_filter:
                # collect_stats=False: preview scrubbing must not overwrite the
                # figures an export writes into runSummary.csv
                g = self.apply_filters_to_data(
                    g, collect_stats=False, velocity=use_vel, jump=use_jump,
                    velocity_thresh=self.spin_preview_velocity.value(),
                    jump_thresh=self.spin_preview_jump.value())
            if smoothing_method != "None" and len(g):
                # Preview uses its own Smoothing Window / units so the effect can
                # be tuned live, independent of the Export Options values.
                g = self.apply_smoothing_to_data(
                    g, smoothing_method,
                    window=self.spin_preview_window.value(),
                    units=self.combo_preview_window_units.currentText())
            if len(g):
                chunks.append(g)
        if not chunks:
            return None

        merged = pd.concat(chunks, ignore_index=True)
        return self._build_preview_frames(merged, tags, idx)

    def _build_preview_frames(self, df, selected_tags, idx=None):
        """Bin a slice onto a fixed time grid: one column per selected tag.

        When ``idx`` is given, t_start/t_end are the chunk's nominal bounds so
        the loader's lead-in overlap is excluded from the resident/scrub range.
        """
        hz = int(self.combo_preview_hz.currentText().split()[0])
        bin_ns = int(1_000_000_000 / hz)
        df = df.copy()
        # as_unit('ns') forces nanoseconds regardless of the datetime resolution
        # (pandas 3.0 defaults to microseconds); a bare astype(int64) would yield
        # microseconds here and make every bin 1000x too wide, collapsing the
        # whole chunk into a single preview frame.
        df['tbin'] = df['Timestamp'].dt.as_unit('ns').astype('int64') // bin_ns

        sx = 'smoothed_x' if 'smoothed_x' in df.columns else 'location_x'
        sy = 'smoothed_y' if 'smoothed_y' in df.columns else 'location_y'

        px = df.pivot_table(index='tbin', columns='shortid', values=sx, aggfunc='first')
        py = df.pivot_table(index='tbin', columns='shortid', values=sy, aggfunc='first')
        rx = df.pivot_table(index='tbin', columns='shortid', values='raw_x', aggfunc='first')
        ry = df.pivot_table(index='tbin', columns='shortid', values='raw_y', aggfunc='first')

        # Reindex onto every selected tag, not just those present in this
        # chunk. A tag that drops out would otherwise lose its column and shift
        # every later tag's colour, so the same animal would change colour
        # between chunks. Absent tags become all-NaN columns, which the
        # canvases hide.
        cols = list(selected_tags)
        px, py = px.reindex(columns=cols), py.reindex(columns=cols)
        rx, ry = rx.reindex(columns=cols), ry.reindex(columns=cols)

        grid = np.arange(px.index.min(), px.index.max() + 1)
        px, py = px.reindex(grid), py.reindex(grid)
        rx, ry = rx.reindex(grid), ry.reindex(grid)

        X = px.to_numpy(float)
        Y = py.to_numpy(float)
        RX = rx.to_numpy(float)
        RY = ry.to_numpy(float)

        # Forward-filled marker positions. UWB tags report asynchronously at
        # well under 1 Hz, so in any single time bin most tags have no fix and
        # would blink out. Carrying each tag's last known position forward keeps
        # a solid, persistent marker; the sparse trail still uses the real
        # fixes. Cells before a tag's first fix stay NaN (nothing to show yet).
        XF = pd.DataFrame(X).ffill().to_numpy()
        YF = pd.DataFrame(Y).ffill().to_numpy()

        # Battery voltage per tag on the same grid, forward-filled so the reading
        # persists between the tag's sparse fixes. None when the table carries no
        # battery column. 'last' per bin (battery varies slowly, so any fix does).
        if 'battery_voltage' in df.columns:
            pb = df.pivot_table(index='tbin', columns='shortid',
                                values='battery_voltage', aggfunc='last')
            pb = pb.reindex(columns=cols).reindex(grid)
            BAT = pd.DataFrame(pb.to_numpy(float)).ffill().to_numpy()
        else:
            BAT = None

        times = pd.to_datetime(grid * bin_ns, utc=True).tz_convert(
            pytz.timezone(self.combo_timezone.currentText()))

        # Nominal chunk bounds when we know the index, so the lead-in overlap the
        # loader prepends is not treated as scrubbable/resident; else fall back
        # to the data span.
        if idx is not None:
            t_start, t_end = self._chunk_bounds(idx)
        else:
            t_start = int(times[0].timestamp() * 1000)
            t_end = int(times[-1].timestamp() * 1000)

        return dict(x=X, y=Y, xf=XF, yf=YF, raw_x=RX, raw_y=RY, batt=BAT,
                    times=times,
                    tags=cols, hz=hz, t_start=t_start, t_end=t_end,
                    nbytes=X.nbytes + Y.nbytes + XF.nbytes + YF.nbytes
                           + RX.nbytes + RY.nbytes
                           + (BAT.nbytes if BAT is not None else 0))

    # Default preview marker colours: blue for male, red for female. Tags with
    # no configured identity default to male (matching the codebase's
    # sex.get(..., 'M') convention), so they show as solid blue until the user
    # assigns sex/identity in Configure Identities.
    _SEX_BLUE = (0.20, 0.55, 0.90, 1.0)
    _SEX_RED = (0.90, 0.25, 0.25, 1.0)
    _NEUTRAL = (0.20, 0.55, 0.90, 1.0)   # single colour when "Color by: None"

    def _preview_tag_colors(self, tags):
        """One RGBA per tag, per the preview 'Color by' selector:
        None → all the same; ID → a unique colour per tag; Sex → blue=M, red=F."""
        mode = self.combo_preview_color.currentText() if hasattr(self, 'combo_preview_color') else 'None'
        if mode == 'Sex':
            out = []
            for tag in tags:
                sex = (self.tag_identities.get(tag, {}) or {}).get('sex', 'M')
                out.append(self._SEX_RED if sex == 'F' else self._SEX_BLUE)
            return np.array(out, dtype=float)
        if mode == 'ID':
            cmap = plt.get_cmap('tab20')
            return np.array([cmap(i % 20) for i in range(len(tags))], dtype=float)
        return np.array([self._NEUTRAL] * len(tags), dtype=float)   # None

    def _preview_tag_labels(self, tags):
        """One display label per tag: 'sex-identity' when configured, else HexID.

        Matches the tag naming used elsewhere so the preview labels read the same
        as the Tag Selection list and the animation.
        """
        labels = []
        for tag in tags:
            info = self.tag_identities.get(tag, {}) or {}
            sex, ident = info.get('sex', ''), info.get('identity', '')
            if sex and ident:
                labels.append(f"{sex}-{ident}")
            else:
                labels.append(f"HexID {hex(tag).upper().replace('0X', '')}")
        return labels

    def on_preview_smoothing_changed(self, *_):
        """Preview smoothing method changed: show/hide the preview Smoothing
        Window controls to match the method, then reload chunks under it.

        Mirrors on_smoothing_changed for the Export combo: None and
        Savitzky-Golay expose no window; the rolling methods expose window +
        units; EWMA exposes the window (as its span) but not the units.
        """
        method = self.combo_preview_smoothing.currentText()
        is_ewma = method == "Forward-Backward Exponentially Weighted Moving Average"
        is_rolling = method in ("Rolling Average", "Rolling Median")
        needs_param = is_ewma or is_rolling
        self.lbl_preview_window.setText("Span (samples):" if is_ewma else "Smoothing Window:")
        self.spin_preview_window.setVisible(needs_param)
        self.lbl_preview_window.setVisible(needs_param)
        self.combo_preview_window_units.setVisible(is_rolling)
        self.invalidate_preview_cache()

    def on_preview_trail_changed(self, *_):
        """Trail length changed: redraw now from cached frames (cheap), and —
        debounced — reload the chunks so their lead-in overlap matches the new
        trail. Without the reload a longer trail would clip at the chunk's
        loaded start; within a chunk the redraw alone already extends the tail."""
        self.render_preview_frame()
        if getattr(self, '_preview_active', False):
            self.preview_trail_timer.start(400)

    def on_preview_color_changed(self, *_):
        """Recolour the preview without reloading chunks (colour is cheap)."""
        if getattr(self, 'preview_tags', None) is not None and len(self.preview_tags):
            self.preview_colors = self._preview_tag_colors(self.preview_tags)
            self.render_preview_frame()

    def _show_chunk(self, idx):
        """Make a cached chunk the active one and render the playhead frame."""
        c = self.preview_cache.get(idx)
        if c is None:
            return
        self.preview_current_chunk = idx
        self.preview_tags = c["tags"]
        self.preview_x, self.preview_y = c["x"], c["y"]
        self.preview_xf, self.preview_yf = c["xf"], c["yf"]
        self.preview_raw_x, self.preview_raw_y = c["raw_x"], c["raw_y"]
        self.preview_batt = c.get("batt")
        self.preview_times = c["times"]
        self.preview_hz = c["hz"]
        self.preview_colors = self._preview_tag_colors(self.preview_tags)

        # Clamp the playhead into this chunk's actual span
        self.preview_playhead_ms = int(np.clip(
            self.preview_playhead_ms, c["t_start"], c["t_end"]))

        finite = int(np.isfinite(c["x"]).sum())
        self.lbl_preview_status.setText(
            f"{len(c['x'])} frames · {len(c['tags'])} tags · {finite:,} fixes · "
            f"{self.combo_preview_smoothing.currentText()}")
        if self.preview_arena is None:
            self.refresh_preview_arena()
        self.render_preview_frame()
        # Warm the adjacent chunks so the next scrub crossing is already cached.
        self._prefetch_neighbors(idx)

    # -- timeline / playhead ----------------------------------------------- #
    def on_timeline_moved(self, value):
        """Slider moved: update the clock immediately, defer the actual read.

        Dragging emits a continuous stream of values; firing a query on each
        would be exactly the behaviour that made the old preview unusable. The
        debounce means only where you *stop* is ever fetched.
        """
        if self._timeline_guard or self.preview_t0 is None:
            return
        self.preview_playhead_ms = self.preview_t0 + int(value) * 1000
        self._update_time_label()
        # If the playhead is inside ANY resident chunk, redraw instantly — it is
        # just an array index, no DB read. Because we prefetch the neighbouring
        # chunks, a continuous hold stays inside cache and the tags keep moving
        # instead of freezing at the old chunk's edge.
        resident = self._resident_chunk_for(self.preview_playhead_ms)
        if resident is not None:
            if resident != self.preview_current_chunk:
                self._show_chunk(resident)   # also warms the new neighbours
            else:
                self.render_preview_frame()
        elif not self.preview_scrub_timer.isActive():
            # Not resident: throttle a background load. Testing isActive() rather
            # than restarting guarantees the timer still fires while the arrow
            # key auto-repeats, so a sustained hold keeps advancing instead of
            # starving the loader until the key is released.
            self.preview_scrub_timer.start(self.SCRUB_DEBOUNCE_MS)

    def _on_scrub_settled(self):
        self._request_chunk(self._chunk_index_for(self.preview_playhead_ms),
                            make_current=True)

    def _sync_timeline_to_playhead(self):
        if self.preview_t0 is None:
            return
        self._timeline_guard = True
        self.slider_timeline.setValue(
            int((self.preview_playhead_ms - self.preview_t0) // 1000))
        self._timeline_guard = False
        self._update_time_label()

    def _update_time_label(self):
        if self.preview_t0 is None:
            return
        ts = pd.Timestamp(self.preview_playhead_ms, unit='ms', tz='UTC').tz_convert(
            pytz.timezone(self.combo_timezone.currentText()))
        cached = "" if self.preview_current_chunk in self.preview_cache else "  (loading…)"
        self.lbl_preview_time.setText(f"{ts:%Y-%m-%d %H:%M:%S}{cached}")

    def _frame_index(self):
        """Index into the active chunk for the current playhead time."""
        if self.preview_times is None or not len(self.preview_times):
            return 0
        target = pd.Timestamp(self.preview_playhead_ms, unit='ms', tz='UTC')
        i = int(np.searchsorted(self.preview_times.tz_convert('UTC').values,
                                np.datetime64(target.tz_localize(None), 'ns')))
        return int(np.clip(i, 0, len(self.preview_times) - 1))

    # -- rendering --------------------------------------------------------- #
    def render_preview_frame(self):
        """Draw the frame the playhead currently points at."""
        if getattr(self, "preview_x", None) is None or not len(self.preview_x):
            return
        idx = self._frame_index()

        # Marker uses the forward-filled position so a tag stays put between its
        # sparse fixes instead of blinking; the trail below uses the real fixes.
        xf = getattr(self, "preview_xf", self.preview_x)
        yf = getattr(self, "preview_yf", self.preview_y)
        x = xf[idx]
        y = yf[idx]
        colors = self.preview_colors

        backend = self._preview_backend()
        # Live marker size (diameter, points); see UWBPreview2D.update_frame.
        if hasattr(self, "spin_tag_size"):
            backend.tag_size = self.spin_tag_size.value()

        # "Show Tag Tracking Data" off: draw an empty frame so only the arena /
        # background / anchors remain (no markers, no trails).
        if hasattr(self, "chk_show_tracking") and not self.chk_show_tracking.isChecked():
            backend.update_frame(np.full_like(np.asarray(x, float), np.nan),
                                 np.full_like(np.asarray(y, float), np.nan),
                                 colors, tracks=None, raw_pts=None)
            self._update_time_label()
            return

        # Build the trailing track: one connected polyline per tag over the
        # trail window (seconds -> frames). Chunks are loaded with a lead-in
        # overlap sized to the trail, so the tail stays continuous across chunk
        # boundaries; a trail longer than a whole chunk still can't be shown in
        # full. Points are the (smoothed) positions; drawing them as a line is
        # the "track" the user wants instead of a dot cloud.
        trail_frames = int(self.spin_preview_trail.value() * self.preview_hz)
        tracks = []
        if trail_frames > 0:
            lo = max(0, idx - trail_frames)
            seg_x = self.preview_x[lo:idx + 1]
            seg_y = self.preview_y[lo:idx + 1]
            for t in range(len(self.preview_tags)):
                tx, ty = seg_x[:, t], seg_y[:, t]
                ok = np.isfinite(tx) & np.isfinite(ty)
                if ok.sum() >= 1:
                    tracks.append((np.column_stack([tx[ok], ty[ok]]),
                                   tuple(colors[t])))

        # ID label + battery voltage under each dot when enabled. Labels are
        # only drawn alongside the battery toggle (the preview is otherwise a
        # clean dot view); the voltage is read from the forward-filled battery
        # grid at the current frame.
        labels = batteries = None
        if getattr(self, "chk_show_battery", None) is not None and self.chk_show_battery.isChecked():
            labels = self._preview_tag_labels(self.preview_tags)
            batt = getattr(self, "preview_batt", None)
            if batt is not None and idx < len(batt):
                batteries = batt[idx]

        backend.update_frame(x, y, colors, tracks=tracks, raw_pts=None,
                             labels=labels, batteries=batteries)
        self._update_time_label()

    # -- transport --------------------------------------------------------- #
    def stop_preview_playback(self):
        """No-op holdover from the old playback engine (kept for call sites).

        Playback was removed; the preview is now scrub-only. If a stray timer
        ever exists it is stopped, but none is created anymore.
        """
        if self.preview_timer is not None:
            self.preview_timer.stop()

    def _context_bg_source(self):
        """(image, extent) for the 'background' layer of main-thread figures.

        Prefers a user-loaded floorplan PNG (scaled via the XML) and falls back
        to the XML-embedded site map. Returns (None, None) if neither exists.
        """
        ext = self._bg_placed_extent()
        if ext is not None:
            return self.background_image, list(ext)
        if self.xml_map_image is not None and self.xml_map_extent is not None:
            return self.xml_map_image, list(self.xml_map_extent)
        return None, None

    def _export_sitemap(self, output_dir, db_name):
        """Write the background/site-map image into the export folder.

        Makes the analysis folder a self-contained data product for the planned
        UWB Studio: it carries the positions (CSV), zones/anchors/identities/scale
        (config) and now the floorplan image too. Prefers a user-loaded PNG
        (copied verbatim), else the XML-embedded site map (decoded to PNG).
        Records filename + meter-extent in self._exported_sitemap for the config;
        leaves it None when there is no background (that is fine — the Studio then
        draws zones only, or bare tracks).
        """
        self._exported_sitemap = None
        bg_image, bg_extent = self._context_bg_source()
        if bg_image is None or bg_extent is None:
            return
        filename = f'{db_name}_sitemap.png'
        dest = os.path.join(output_dir, filename)
        try:
            if self.background_image_path and os.path.exists(self.background_image_path):
                shutil.copyfile(self.background_image_path, dest)
            else:
                plt.imsave(dest, bg_image)
            self._exported_sitemap = {
                'filename': filename,
                'extent_m': [float(v) for v in bg_extent],
            }
            self.log_message(f"✓ Site-map image exported: {filename}")
        except Exception as e:
            self.log_message(f"Warning: could not export site-map image: {e}")

    def _prompt_plot_layers(self):
        """Prompt for the spatial context layers; update self.plot_layers.

        Shared by the Export button and the Occupancy / Last-Known buttons so
        every spatial output asks the same question each time. Returns True to
        proceed, False if the user cancelled. If no layer is available, sets all
        layers off and proceeds without a dialog.

        'Background image' is offered when the user has loaded one (via Load
        Background) or the site XML carries an embedded map (used as a fallback).
        """
        has_background = (self.background_image is not None
                          or self.xml_map_image is not None)
        has_zones = bool(self.xml_zones) or (
            self.arena_zones is not None and not self.arena_zones.empty)
        has_anchors = bool(self.anchor_positions)
        if not (has_background or has_zones or has_anchors):
            self.plot_layers = {'background': False, 'zones': False, 'anchors': False}
            return True
        dlg = PlotLayersDialog(has_background, has_zones, has_anchors,
                               defaults=self.plot_layers, parent=self)
        if dlg.exec_() != QDialog.Accepted:
            return False
        self.plot_layers = dlg.get_layers()
        self.log_message(
            "Plot/animation layers — "
            f"background: {'on' if self.plot_layers['background'] else 'off'}, "
            f"zones: {'on' if self.plot_layers['zones'] else 'off'}, "
            f"anchors: {'on' if self.plot_layers['anchors'] else 'off'}")
        return True

    # -- quick snapshot plot ---------------------------------------------- #
    def plot_last_known_locations(self):
        """Save a labelled snapshot of each selected tag's most recent fix.

        One cheap query per tag (MAX(timestamp)), so it is fast even on the
        unindexed original — and faster still if the indexed copy exists. The
        figure is drawn over the same arena the preview shows (XML zones/map
        when available), then written to the analysis folder.
        """
        if not self.db_path or not self.table_name:
            QMessageBox.warning(self, "No Database", "Select a database and table first")
            return
        tags = self.selected_preview_tags()
        if not tags:
            QMessageBox.warning(self, "No Tags", "Select at least one tag first")
            return

        # Ask which context layers to draw (background/zones/anchors).
        if not self._prompt_plot_layers():
            return

        try:
            tz = pytz.timezone(self.combo_timezone.currentText())
            db = self.preview_db_path or self.db_path
            conn = sqlite3.connect(db)
            placeholders = ",".join(["?"] * len(tags))
            # Last fix per tag: join each tag's MAX(timestamp) back to its row.
            rows = conn.execute(
                f"SELECT d.shortid, d.location_x, d.location_y, d.timestamp "
                f"FROM {self.table_name} d "
                f"JOIN (SELECT shortid, MAX(timestamp) mt FROM {self.table_name} "
                f"      WHERE shortid IN ({placeholders}) GROUP BY shortid) m "
                f"ON d.shortid = m.shortid AND d.timestamp = m.mt",
                tags).fetchall()
            conn.close()
        except Exception as e:
            self.log_message(f"Last-locations query failed: {e}")
            QMessageBox.critical(self, "Query Failed", f"Could not read last locations:\n{e}")
            return

        if not rows:
            QMessageBox.information(self, "No Data", "No fixes found for the selected tags.")
            return

        # Deduplicate on shortid (a tie in MAX could yield two rows) and convert
        # inches -> metres to match every other coordinate in the tool.
        latest = {}
        for shortid, lx, ly, ts in rows:
            if shortid not in latest or ts > latest[shortid][2]:
                latest[shortid] = (lx * 0.0254, ly * 0.0254, ts)

        # Busy cursor for the figure build + save (all on the GUI thread).
        # try/finally guarantees the cursor clears even if drawing/saving raises.
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            fig = Figure(figsize=(8, 8))
            ax = fig.add_subplot(111)
            pal_bg, pal_fg = "#1e1e1e", "#cccccc"
            fig.patch.set_facecolor(pal_bg)
            ax.set_facecolor(pal_bg)

            # Arena context (background/zones/anchors) per the export layer choice.
            xs = [v[0] for v in latest.values()]
            ys = [v[1] for v in latest.values()]
            bg_image, bg_extent = self._context_bg_source()
            draw_context_layers(
                ax, self.plot_layers, bg_image=bg_image, bg_extent=bg_extent,
                zones_xml=self.xml_zones, anchors=self.anchor_positions)

            cmap = plt.get_cmap('tab20')
            for i, (shortid, (x, y, ts)) in enumerate(sorted(latest.items())):
                label = self._tag_display_label(shortid)
                local = pd.Timestamp(int(ts), unit='ms', tz='UTC').tz_convert(tz)
                ax.scatter([x], [y], s=140, color=cmap(i % 20),
                           edgecolors="white", linewidths=1.0, zorder=5)
                ax.annotate(f"{label}\n{local:%m-%d %H:%M}", (x, y),
                            textcoords="offset points", xytext=(8, 6),
                            color=pal_fg, fontsize=8, zorder=6)

            ax.set_aspect("equal")
            ax.set_xlabel("X (m)", color=pal_fg)
            ax.set_ylabel("Y (m)", color=pal_fg)
            ax.tick_params(colors=pal_fg)
            for sp in ax.spines.values():
                sp.set_color("#555555")
            ax.set_title(f"Last known locations · {len(latest)} tags", color=pal_fg)
            fig.tight_layout()

            # Save into the analysis folder, matching the export naming.
            db_dir = os.path.dirname(self.db_path)
            db_name = os.path.splitext(os.path.basename(self.db_path))[0]
            out_dir = os.path.join(db_dir, f"{db_name}_FNT_analysis")
            os.makedirs(out_dir, exist_ok=True)
            png = os.path.join(out_dir, f"{db_name}_LastKnownLocations.png")
            try:
                fig.savefig(png, dpi=150, facecolor=pal_bg, bbox_inches="tight")
                saved = [os.path.basename(png)]
                if self.chk_save_svg.isChecked():
                    svg = os.path.splitext(png)[0] + ".svg"
                    fig.savefig(svg, facecolor=pal_bg, bbox_inches="tight")
                    saved.append(os.path.basename(svg))
                self.log_message(f"✓ Saved last-known-locations plot: {', '.join(saved)}")
                QMessageBox.information(
                    self, "Plot Saved",
                    f"Saved {', '.join(saved)}\n\nin {os.path.basename(out_dir)}/")
            except Exception as e:
                self.log_message(f"Could not save last-locations plot: {e}")
                QMessageBox.critical(self, "Save Failed", str(e))
        finally:
            QApplication.restoreOverrideCursor()

    def _tag_display_label(self, shortid):
        """Identity label for a tag: 'sex-identity' if configured, else HexID."""
        info = self.tag_identities.get(shortid)
        if info:
            sex, ident = info.get("sex", ""), info.get("identity", "")
            if sex and ident:
                return f"{sex}-{ident}"
            if ident:
                return str(ident)
        return f"HexID {hex(shortid).upper().replace('0X', '')}"

    # -- occupancy heatmaps ------------------------------------------------ #
    def plot_occupancy_heatmaps(self):
        """Build per-animal space-use heatmaps over the whole recording.

        Each tag's fixes are binned into a 2-D histogram of % time per arena
        cell, then shown faceted (one panel per animal) in a zoomable window.
        Memory stays flat: each tag is read, filtered, histogrammed and then
        discarded — only the small bin grids are kept.
        """
        if not self.db_path or not self.table_name:
            QMessageBox.warning(self, "No Database", "Select a database and table first")
            return
        tags = self.selected_preview_tags()
        if not tags:
            QMessageBox.warning(self, "No Tags", "Select at least one tag first")
            return

        # Ask which context layers to draw (zones/anchors overlay the heatmap).
        if not self._prompt_plot_layers():
            return

        db = self.preview_db_path or self.db_path
        tz = pytz.timezone(self.combo_timezone.currentText())
        do_filter = self.chk_velocity_filter.isChecked() or self.chk_jump_filter.isChecked()

        progress = QProgressDialog("Building occupancy heatmaps…", "Cancel",
                                   0, len(tags) + 1, self)
        progress.setWindowTitle("Occupancy Heatmaps")
        progress.setMinimumDuration(0)
        progress.setValue(0)
        QApplication.processEvents()

        try:
            conn = sqlite3.connect(db)

            # Common spatial extent so every facet shares one grid. Prefer the
            # anchor footprint (the true arena); UWB data carries stray fixes
            # far outside it, so a raw data min/max would blow up the frame.
            placeholders = ",".join(["?"] * len(tags))
            if self.anchor_positions:
                xs_a = [a["x"] for a in self.anchor_positions]
                ys_a = [a["y"] for a in self.anchor_positions]
                x0, x1 = min(xs_a), max(xs_a)
                y0, y1 = min(ys_a), max(ys_a)
            else:
                # No anchors: use robust percentiles to reject outliers. Sample
                # the data rather than pulling every row for the bounds.
                s = pd.read_sql_query(
                    f"SELECT location_x, location_y FROM {self.table_name} "
                    f"WHERE shortid IN ({placeholders}) "
                    f"ORDER BY timestamp LIMIT 200000", conn, params=tags)
                if len(s) == 0:
                    conn.close(); progress.close()
                    QMessageBox.information(self, "No Data", "No fixes for the selected tags.")
                    return
                x0, x1 = np.percentile(s['location_x'] * 0.0254, [1, 99])
                y0, y1 = np.percentile(s['location_y'] * 0.0254, [1, 99])
            pad = 0.5
            x0, x1 = x0 - pad, x1 + pad
            y0, y1 = y0 - pad, y1 + pad
            # histogram2d range clips out-of-arena outliers automatically
            hist_range = [[x0, x1], [y0, y1]]
            # ~0.15 m cells, capped so huge arenas stay a sane grid size
            nx = int(np.clip(round((x1 - x0) / 0.15), 20, 90))
            ny = int(np.clip(round((y1 - y0) / 0.15), 20, 90))
            xedges = np.linspace(x0, x1, nx + 1)
            yedges = np.linspace(y0, y1, ny + 1)

            heatmaps, labels = [], []
            for i, tag in enumerate(tags):
                if progress.wasCanceled():
                    conn.close(); return
                progress.setLabelText(f"Tag {i + 1}/{len(tags)} — reading & binning…")
                progress.setValue(i)
                QApplication.processEvents()

                td = pd.read_sql_query(
                    f"SELECT timestamp, location_x, location_y FROM {self.table_name} "
                    f"WHERE shortid = ? ORDER BY timestamp", conn, params=(tag,))
                if len(td) == 0:
                    heatmaps.append(None); labels.append(self._tag_display_label(tag))
                    continue

                td['shortid'] = tag
                td['location_x'] *= 0.0254
                td['location_y'] *= 0.0254

                # Respect the tag's active window and the current filters, so
                # the heatmap matches what an export would produce.
                if tag in self.tag_identities:
                    info = self.tag_identities[tag]
                    if 'start_time' in info and 'stop_time' in info:
                        td['Timestamp'] = pd.to_datetime(
                            td['timestamp'], unit='ms', origin='unix', utc=True).dt.tz_convert(tz)
                        start = pd.Timestamp(info['start_time']).tz_localize(tz)
                        stop = pd.Timestamp(info['stop_time']).tz_localize(tz)
                        td = td[(td['Timestamp'] >= start) & (td['Timestamp'] <= stop)]
                if do_filter and len(td) > 0:
                    td['Timestamp'] = pd.to_datetime(
                        td['timestamp'], unit='ms', origin='unix', utc=True).dt.tz_convert(tz)
                    td = self.apply_filters_to_data(td, collect_stats=False)

                labels.append(self._tag_display_label(tag))
                if len(td) == 0:
                    heatmaps.append(None); continue
                # range= clips fixes outside the arena frame
                H, _, _ = np.histogram2d(td['location_x'].to_numpy(),
                                         td['location_y'].to_numpy(),
                                         bins=[xedges, yedges], range=hist_range)
                total = H.sum()
                heatmaps.append((H / total * 100.0) if total else None)
                self.log_message(f"  {labels[-1]}: {int(total):,} fixes binned")
                # Repaint the progress dialog after this tag's heavy read+bin.
                QApplication.processEvents()

            conn.close()
            progress.setValue(len(tags))
            QApplication.processEvents()

            valid = [h for h in heatmaps if h is not None]
            if not valid:
                progress.close()
                QMessageBox.information(self, "No Data",
                                        "No fixes survived filtering for the selected tags.")
                return

            # Shared colour scale: 99th percentile of occupied cells, so a
            # single hot spot doesn't wash out every panel.
            pooled = np.concatenate([h[h > 0].ravel() for h in valid])
            vmax = float(np.percentile(pooled, 99)) if len(pooled) else 1.0

            fig = self._build_heatmap_figure(heatmaps, labels, xedges, yedges, vmax)
            progress.close()

            # Save alongside the other outputs, then show it zoomably.
            db_name = os.path.splitext(os.path.basename(self.db_path))[0]
            out_dir = os.path.join(os.path.dirname(self.db_path), f"{db_name}_FNT_analysis")
            os.makedirs(out_dir, exist_ok=True)
            png = os.path.join(out_dir, f"{db_name}_OccupancyHeatmaps.png")
            try:
                fig.savefig(png, dpi=150, facecolor="white", bbox_inches="tight")
                self.log_message(f"✓ Saved occupancy heatmaps: {os.path.basename(png)}")
            except Exception as e:
                self.log_message(f"Could not save heatmap PNG: {e}")

            self._heatmap_popup = FigurePopup(
                fig, title=f"Occupancy Heatmaps — {db_name}", parent=self)
            self._heatmap_popup.resize(960, 720)
            self._heatmap_popup.show()

        except Exception as e:
            progress.close()
            self.log_message(f"Occupancy heatmap failed: {e}")
            QMessageBox.critical(self, "Heatmap Failed", str(e))

    def _build_heatmap_figure(self, heatmaps, labels, xedges, yedges, vmax):
        """Faceted magma raster of % time per cell, shared colour scale."""
        n = len(heatmaps)
        ncols = 1 if n == 1 else min(3, n)
        nrows = int(np.ceil(n / ncols))
        fig = Figure(figsize=(3.2 * ncols + 1.2, 3.0 * nrows), facecolor="white")
        extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]
        im = None
        for i, (H, lab) in enumerate(zip(heatmaps, labels)):
            ax = fig.add_subplot(nrows, ncols, i + 1)
            ax.set_facecolor("black")
            if H is not None:
                # histogram2d is [x, y]; transpose so rows=y for imshow, and
                # origin="lower" puts +y up.
                im = ax.imshow(H.T, extent=extent, origin="lower", cmap="magma",
                               vmin=0, vmax=vmax, aspect="equal",
                               interpolation="nearest")
            else:
                ax.text(0.5, 0.5, "no data", transform=ax.transAxes,
                        ha="center", va="center", color="#999")
                ax.set_xlim(extent[0], extent[1]); ax.set_ylim(extent[2], extent[3])
            # Overlay chosen context layers. Background is omitted here — the
            # density raster already fills the panel and would hide it.
            draw_context_layers(
                ax, {'background': False,
                     'zones': self.plot_layers.get('zones'),
                     'anchors': self.plot_layers.get('anchors')},
                zones_xml=self.xml_zones, anchors=self.anchor_positions)
            ax.set_title(lab, fontsize=10)
            ax.set_xlabel("X Position (m)", fontsize=8)
            ax.set_ylabel("Y Position (m)", fontsize=8)
            ax.tick_params(labelsize=7)
        if im is not None:
            cbar = fig.colorbar(im, ax=fig.axes, fraction=0.025, pad=0.02)
            cbar.set_label("% time", fontsize=9)
        fig.suptitle("Occupancy (space use) over full recording", fontsize=12)
        return fig

    # -- indexed preview copy ---------------------------------------------- #
    # The original Wiser database is treated as a read-only primary record and
    # is never written to. Fast scrubbing instead uses a derived, indexed copy
    # kept alongside the other analysis outputs.
    def _preview_index_paths(self):
        """(index_dir, indexed_copy_path, provenance_json_path).

        The indexed copy + its provenance JSON live alongside the source
        database (in the DB's own folder), not in the analysis/export folder, so
        the fast-scrubbing index travels with the raw data and is reused across
        exports rather than being duplicated per analysis run.
        """
        db_dir = os.path.dirname(self.db_path)
        db_name = os.path.splitext(os.path.basename(self.db_path))[0]
        return (db_dir,
                os.path.join(db_dir, f"{db_name}_indexed.sqlite"),
                os.path.join(db_dir, f"{db_name}_indexed.json"))

    def _indexed_copy_is_current(self, copy_path, meta_path):
        """Is an existing copy still faithful to the source database?

        Compared on size and mtime. A stale cache is worse than none, so any
        mismatch (or unreadable metadata) fails closed and forces a rebuild.
        """
        if not (os.path.exists(copy_path) and os.path.exists(meta_path)):
            return False
        try:
            with open(meta_path) as f:
                meta = json.load(f)
            st = os.stat(self.db_path)
            return (meta.get("source_size") == st.st_size
                    and abs(meta.get("source_mtime", 0) - st.st_mtime) < 1.0
                    and meta.get("table_name") == self.table_name)
        except Exception:
            return False

    def adopt_preview_index_db(self):
        """Use an existing indexed copy if one is current, else the original.

        Never builds — this runs automatically when the preview activates, and
        writing a multi-GB copy on every database load would be a surprise. The
        'Fast index' button (ensure_preview_index_db) is the explicit opt-in.
        """
        if not self.db_path or not self.table_name:
            return
        self.preview_db_path = self.db_path
        _, copy_path, meta_path = self._preview_index_paths()

        # Sweep any leftover .partial from a build that was interrupted (crash
        # or hard kill) — these can be multi-GB and would otherwise linger. Only
        # safe to remove when no build is currently running.
        if not (self.preview_index_builder is not None
                and self.preview_index_builder.isRunning()):
            for junk in (copy_path + ".partial", copy_path + ".partial-journal"):
                try:
                    if os.path.exists(junk):
                        os.remove(junk)
                        self.log_message(f"Removed stale index build file: {os.path.basename(junk)}")
                except Exception:
                    pass

        if self._indexed_copy_is_current(copy_path, meta_path):
            self.preview_db_path = copy_path
            self._set_index_status("Fast index: ready ✓", "#00aa00")
            self.log_message(f"Using existing indexed copy for preview: {os.path.basename(copy_path)}")
        else:
            self._set_index_status("Fast index: not built (click to speed up scrubbing)", "#cccccc")

    def ensure_preview_index_db(self):
        """Point the preview at an indexed copy, building one if needed.

        Preview stays usable throughout: reads run against the original until
        the copy is ready, then transparently switch over.
        """
        if not self.db_path or not self.table_name:
            return
        self.preview_db_path = self.db_path      # usable immediately

        index_dir, copy_path, meta_path = self._preview_index_paths()
        if self._indexed_copy_is_current(copy_path, meta_path):
            self.preview_db_path = copy_path
            self._set_index_status("Fast index: ready ✓", "#00aa00")
            self.log_message(f"Using indexed copy for preview: {os.path.basename(copy_path)}")
            return

        if self.preview_index_builder is not None and self.preview_index_builder.isRunning():
            return

        if os.path.exists(copy_path):
            self.log_message("Indexed copy is out of date with the source — rebuilding.")

        try:
            os.makedirs(index_dir, exist_ok=True)
        except Exception as e:
            self.log_message(f"Could not create index folder: {e}")
            return

        src_mb = os.path.getsize(self.db_path) / 1e6
        self.log_message(
            f"Building indexed copy for fast scrubbing (~{src_mb * 1.25:.0f} MB "
            f"in {os.path.basename(index_dir)}). The original database is "
            f"not modified.")
        self._set_index_status("Fast index: building…", "#cc9900")

        b = PreviewIndexBuilder(self.db_path, self.table_name, copy_path, meta_path)
        b.progress.connect(self._on_index_progress)
        b.done.connect(self._on_index_ready)
        b.failed.connect(self._on_index_failed)
        self.preview_index_builder = b
        b.start()

    def _index_build_is_current(self, src_path):
        """Is a build result still for the database the tool is showing?

        A build takes seconds; the user can switch databases in that time. Any
        result for a different source must be discarded, or the preview would
        silently read another trial's data.
        """
        return bool(self.db_path) and os.path.abspath(src_path) == os.path.abspath(self.db_path)

    def _on_index_progress(self, msg):
        if self.preview_index_builder is None:
            return
        if not self._index_build_is_current(self.preview_index_builder.src_path):
            return
        self._set_index_status(f"Fast index: {msg}", "#cc9900")

    def _set_index_status(self, text, color):
        # Shown as a small status line under the cache label (no button anymore).
        if hasattr(self, "lbl_cache_status"):
            self.lbl_cache_status.setText(text)
            self.lbl_cache_status.setStyleSheet(f"color: {color}; font-size: 9px;")

    def _on_index_ready(self, src_path, path):
        self.preview_index_builder = None
        if not self._index_build_is_current(src_path):
            # The user moved to another database while this was building. The
            # copy on disk is still valid for its own database and will be
            # picked up if they return to it — it just must not be adopted now.
            self.log_message(
                f"Indexed copy for {os.path.basename(src_path)} finished after "
                f"the database changed — not used for the current selection.")
            return
        self.preview_db_path = path
        mb = os.path.getsize(path) / 1e6
        self._set_index_status("Fast index: ready ✓", "#00aa00")
        self.log_message(
            f"Indexed copy ready ({mb:.0f} MB). Scrubbing is now ~80x faster; "
            f"the original database is untouched.")
        # Cached chunks came from the unindexed original but are byte-identical
        # in content, so they stay valid — only future reads get faster.

    def _on_index_failed(self, src_path, err):
        self.preview_index_builder = None
        if not self._index_build_is_current(src_path):
            return
        self._set_index_status("Fast index: unavailable (using original)", "#cc5555")
        self.log_message(
            f"Could not build indexed copy ({err}). Preview still works, "
            f"reading directly from the original database.")

    def rebuild_preview_index(self):
        """Manual rebuild, e.g. after the source database has changed."""
        if not self.db_path or not self.table_name:
            QMessageBox.warning(self, "No Database", "Select a database and table first")
            return
        _, copy_path, meta_path = self._preview_index_paths()
        for p in (copy_path, meta_path):
            try:
                if os.path.exists(p):
                    os.remove(p)
            except Exception as e:
                self.log_message(f"Could not remove {os.path.basename(p)}: {e}")
        self.preview_db_path = self.db_path
        self.invalidate_preview_cache()
        self.ensure_preview_index_db()

    def _nearest_sample_ts(self, conn, selected_tags, ts):
        """Timestamp of the closest sample at or after ``ts``, else the last before it."""
        placeholders = ",".join(["?"] * len(selected_tags))
        try:
            row = conn.execute(
                f"SELECT MIN(timestamp) FROM {self.table_name} "
                f"WHERE shortid IN ({placeholders}) AND timestamp >= ?",
                list(selected_tags) + [ts]).fetchone()
            if row and row[0] is not None:
                return int(row[0])
            row = conn.execute(
                f"SELECT MAX(timestamp) FROM {self.table_name} "
                f"WHERE shortid IN ({placeholders}) AND timestamp <= ?",
                list(selected_tags) + [ts]).fetchone()
            if row and row[0] is not None:
                return int(row[0])
        except Exception as e:
            self.log_message(f"Could not locate nearest samples: {e}")
        return None
    def reset_to_defaults(self):
        """Reset all settings and state to defaults for a clean database switch."""
        # Clear identity config
        self.tag_identities = {}

        # Clear data state
        self.data = None
        self.available_tags = []

        # Clear background image and XML state
        self.xml_config_path = None
        self.background_image_path = None
        self.background_image = None
        self.xml_scale = None
        self.bg_width_meters = None
        self.bg_height_meters = None
        self.bg_scale = None
        self.bg_offset_x = 0.0
        self.bg_offset_y = 0.0
        self.xml_maps = []
        self.arena_zones = None
        self.anchor_positions = []

        # Drop all cached preview chunks so nothing leaks across databases. The
        # pane stays visible but goes inert until the new database's tags load.
        self._preview_active = False
        self.stop_preview_playback()
        self.preview_cache.clear()
        # Let in-flight reads finish before releasing them. Clearing the dict
        # outright would drop the last reference to a running QThread and abort
        # the process. These are short windowed reads (~5 ms indexed, ~0.4 s
        # unindexed), so the wait is not user-visible.
        for loader in list(self.preview_inflight.values()):
            if loader.isRunning():
                loader.wait(5000)
        self.preview_inflight.clear()
        self.preview_current_chunk = None
        self.preview_pending_current = None
        self.preview_t0 = self.preview_t1 = None
        self.preview_playhead_ms = 0
        self.preview_db_path = None
        self.preview_x = self.preview_y = None
        self.preview_xf = self.preview_yf = None
        self.preview_raw_x = self.preview_raw_y = None
        self.preview_batt = None
        self.preview_times = None
        self.preview_tags = []
        self.preview_arena = None
        self.xml_zones = []
        self.xml_map_image = None
        self.xml_map_extent = None
        self._sync_bg_transform_controls()
        if hasattr(self, 'slider_timeline'):
            self._timeline_guard = True
            self.slider_timeline.setRange(0, 0)
            self._timeline_guard = False
            self.lbl_preview_status.setText("Load a database and select tags to preview")
            self.lbl_preview_time.setText("--")
            self._update_cache_label()
            self._set_index_status("Fast index: not built", "#cccccc")
            self.preview_canvas_2d.clear()
            if self.preview_canvas_3d is not None:
                self.preview_canvas_3d.clear()

        # Reset pending table name
        if hasattr(self, 'pending_table_name'):
            delattr(self, 'pending_table_name')

        # Reset GUI widgets to defaults
        self.combo_timezone.setCurrentText("US/Mountain")
        self.combo_smoothing.setCurrentIndex(0)  # None (default)
        self.spin_rolling_window.setValue(30)
        self.combo_window_units.setCurrentText("Seconds")
        if hasattr(self, 'spin_preview_window'):
            self.spin_preview_window.setValue(30)
            self.combo_preview_window_units.setCurrentText("Seconds")
        if hasattr(self, 'spin_preview_velocity'):
            self.spin_preview_velocity.setValue(2.0)
            self.spin_preview_jump.setValue(2.0)
        self.chk_velocity_filter.setChecked(True)
        self.spin_velocity_threshold.setValue(2.0)
        self.chk_jump_filter.setChecked(True)
        self.spin_jump_threshold.setValue(2.0)
        self.spin_time_gap.setValue(30)
        self.combo_color_by.setCurrentIndex(0)  # None

        # Export options
        self.chk_export_raw_csv.setChecked(False)
        self.chk_export_smoothed_csv.setChecked(True)
        self.chk_proximity_detection.setChecked(True)
        self.spin_proximity_threshold.setValue(0.5)
        self.chk_save_plots.setChecked(False)
        self.chk_downsample_plots.setChecked(True)
        self.spin_plot_downsample_hz.setValue(1)
        self.chk_downsample_animation.setChecked(True)
        self.spin_anim_downsample_hz.setValue(1)
        self.chk_save_svg.setChecked(False)
        for cb in self.plot_type_checkboxes.values():
            cb.setChecked(True)
        self.chk_save_animation.setChecked(False)
        self.spin_animation_trail.setValue(500)
        self.spin_anim_tag_size.setValue(10)
        self.chk_show_battery_export.setChecked(False)
        self.combo_animation_speed.setCurrentText("60x")
        self.combo_animation_fps.setCurrentText("20")

        # Preview display defaults
        self.chk_show_tracking.setChecked(True)
        self.spin_tag_size.setValue(10)
        self.chk_show_battery.setChecked(False)

        # Reset background image label
        if hasattr(self, 'lbl_background_status'):
            self.lbl_background_status.setText("No background image loaded")

        # Reset show anchor positions checkbox
        if hasattr(self, 'chk_show_anchors'):
            self.chk_show_anchors.setChecked(False)

    def select_database(self):
        """Select SQLite database"""
        file_path, _ = QFileDialog.getOpenFileName(
            self, "Select SQLite Database", "",
            "SQLite Files (*.sqlite *.db *.sql);;All Files (*.*)"
        )
        
        if not file_path:
            return
        
        try:
            conn = sqlite3.connect(file_path)
            cursor = conn.cursor()
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
            tables = [row[0] for row in cursor.fetchall()]
            conn.close()
            
            if not tables:
                QMessageBox.warning(self, "No Tables", "No tables found in database")
                return
            
            self.db_path = file_path
            self.lbl_db.setText(f"Selected: {os.path.basename(file_path)}")

            # Reset all settings to defaults before loading new config
            self.reset_to_defaults()

            # Load config BEFORE populating combo_table, so that pending_tag_selection
            # is set before on_table_selected() triggers tag checkbox creation
            config_loaded = self.load_config_if_exists()
            if config_loaded:
                self.lbl_config_status.setText("Loaded previous fnt_config.json file.")
                self.lbl_config_status.setVisible(True)
            else:
                self.lbl_config_status.setVisible(False)

            self.combo_table.clear()
            self.combo_table.addItems(tables)
            self.combo_table.setEnabled(True)
            self.btn_preview_table.setEnabled(True)
            self.btn_load_background.setEnabled(True)  # Enable background image loading

            # Apply saved table name from config (if any), otherwise default to first
            if hasattr(self, 'pending_table_name') and self.pending_table_name:
                index = self.combo_table.findText(self.pending_table_name)
                if index >= 0:
                    self.combo_table.setCurrentIndex(index)
                delattr(self, 'pending_table_name')
            elif len(tables) == 1:
                self.combo_table.setCurrentIndex(0)

            # Check for XML configuration file in the database directory
            self.load_xml_config()

            # Auto-load a background image only if one clearly belongs to this
            # database (name match). Otherwise offer to pick one.
            self.auto_load_background()
            if self.background_image is None:
                self.prompt_background_image()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to open database: {str(e)}")

    def is_wiser_database(self):
        """Heuristic: does this look like a Wiser UWB export?

        True if the site XML has a <Wiser> root, or the table carries the
        Wiser column signature (shortid + location_x/y + timestamp).
        """
        if self.xml_config_path and os.path.exists(self.xml_config_path):
            try:
                root = ET.parse(self.xml_config_path).getroot()
                if root.tag.lower() == "wiser":
                    return True
            except Exception:
                pass
        if self.db_path and self.table_name:
            try:
                conn = sqlite3.connect(self.db_path)
                cols = {r[1].lower() for r in conn.execute(f"PRAGMA table_info({self.table_name})")}
                conn.close()
                return {"shortid", "location_x", "location_y", "timestamp"}.issubset(cols)
            except Exception:
                pass
        return False

    def _find_candidate_backgrounds(self):
        """Visible image files in the DB folder, best name-match candidate first.

        Ranks a filename matching the database name ahead of one matching the
        XML config name, ahead of everything else, so the first entry is the
        most likely floorplan.
        """
        if not self.db_path:
            return []
        db_dir = os.path.dirname(self.db_path)
        exts = ('.png', '.jpg', '.jpeg', '.bmp')
        imgs = [f for f in list_visible_files(db_dir) if f.lower().endswith(exts)]
        if not imgs:
            return []
        db_name = os.path.splitext(os.path.basename(self.db_path))[0].lower()
        xml_name = ''
        if getattr(self, 'xml_config_path', None):
            xml_name = os.path.splitext(os.path.basename(self.xml_config_path))[0].lower()

        def rank(f):
            fname = os.path.splitext(f)[0].lower()
            if db_name and (db_name in fname or fname in db_name):
                return 0
            if xml_name and (xml_name in fname or fname in xml_name):
                return 1
            return 2

        return sorted(imgs, key=rank)

    def _confirm_background_preview(self, image_path):
        """Show ``image_path`` in a preview dialog; return True if the user accepts."""
        from PyQt5.QtGui import QPixmap

        dlg = QDialog(self)
        dlg.setWindowTitle("Use This Background Image?")
        layout = QVBoxLayout(dlg)

        intro = QLabel(f"Detected an image in the database folder:\n\n{os.path.basename(image_path)}")
        intro.setWordWrap(True)
        layout.addWidget(intro)

        preview = QLabel()
        preview.setAlignment(Qt.AlignCenter)
        pix = QPixmap(image_path)
        if not pix.isNull():
            pix = pix.scaled(480, 360, Qt.KeepAspectRatio, Qt.SmoothTransformation)
            preview.setPixmap(pix)
        else:
            preview.setText("(Could not render a preview of this image.)")
        layout.addWidget(preview)

        question = QLabel("Use this image as the arena background?")
        question.setWordWrap(True)
        layout.addWidget(question)

        btn_row = QHBoxLayout()
        btn_no = QPushButton("No")
        btn_yes = QPushButton("Yes, use this image")
        btn_yes.setDefault(True)
        btn_no.clicked.connect(dlg.reject)
        btn_yes.clicked.connect(dlg.accept)
        btn_row.addStretch()
        btn_row.addWidget(btn_no)
        btn_row.addWidget(btn_yes)
        layout.addLayout(btn_row)

        return dlg.exec_() == QDialog.Accepted

    def prompt_background_image(self):
        """Detect a floorplan image in the DB folder, preview it, and confirm use.

        If one or more images are present, the best-matching candidate is shown
        in a preview dialog for the user to accept or decline. Declining (or a
        folder with no images) falls back to the manual file picker.
        """
        if not self.db_path:
            return

        db_dir = os.path.dirname(self.db_path)
        candidates = self._find_candidate_backgrounds()

        if candidates:
            candidate = os.path.join(db_dir, candidates[0])
            self.log_message(f"Detected candidate background image: {candidates[0]}")
            if self._confirm_background_preview(candidate):
                self._apply_background_image(candidate, source="loaded")
            else:
                # User declined the detected image — let them browse for another.
                self.select_background_image()
            return

        # Nothing detected — say so, then offer the manual picker.
        self.log_message("No background image (PNG/JPG) detected in the database folder.")
        reply = QMessageBox.question(
            self, "No Background Image Found",
            "No background image (PNG/JPG) was detected in the database folder.\n\n"
            "Would you like to load one?",
            QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
        if reply == QMessageBox.Yes:
            self.select_background_image()
    
    def load_xml_config(self):
        """Look for XML configuration file in the database directory"""
        if not self.db_path:
            return
        
        db_dir = os.path.dirname(self.db_path)
        
        # Look for .xml files in the same directory (skip hidden/AppleDouble
        # sidecars like '._EchoConfiguration.xml' left by macOS on shares).
        xml_files = [f for f in list_visible_files(db_dir) if f.lower().endswith('.xml')]
        
        if not xml_files:
            self.log_message("No XML configuration file found in database directory")
            QMessageBox.warning(
                self, "XML Configuration Not Found",
                "No XML configuration file found in the database folder.\n\n"
                "Anchor positions and floorplan scale will not be available."
            )
            return
        
        # If multiple XML files, use the first one or one matching config/Config pattern
        xml_file = None
        for f in xml_files:
            if 'config' in f.lower():
                xml_file = f
                break
        if not xml_file:
            xml_file = xml_files[0]
        
        self.xml_config_path = os.path.join(db_dir, xml_file)
        self.log_message(f"Found XML config: {xml_file}")
        
        try:
            self.parse_xml_config()
        except Exception as e:
            self.log_message(f"Warning: Could not parse XML config: {str(e)}")
    
    def parse_xml_config(self):
        """Parse XML configuration file and check for background image"""
        if not self.xml_config_path or not os.path.exists(self.xml_config_path):
            return
        
        try:
            tree = ET.parse(self.xml_config_path)
            root = tree.getroot()
            
            # Extract scale attribute (inches/pixel)
            for elem in root.iter():
                if 'scale' in elem.attrib:
                    try:
                        self.xml_scale = float(elem.attrib['scale'])
                        self.log_message(f"Found XML scale: {self.xml_scale} inches/pixel")
                        break
                    except:
                        pass
            
            # Parse zone coordinates from Zones section
            zones_element = root.find('Zones')
            if zones_element is not None:
                zone_data = []
                self.xml_zones = []
                for zone in zones_element.findall('Zone'):
                    zone_name = zone.get('name')
                    if zone_name is None:
                        continue

                    shape = zone.find('Shape')
                    if shape is None:
                        continue

                    # Keep the polygon intact (with its authored colour) for the
                    # XML site view; the flat zone_data table below stays as-is
                    # for the existing config/plot code paths.
                    poly = []
                    for point in shape.findall('Point'):
                        try:
                            poly.append((float(point.get('x')) * 0.0254,
                                         float(point.get('y')) * 0.0254))
                        except (TypeError, ValueError):
                            continue
                    if len(poly) >= 3:
                        self.xml_zones.append({
                            'name': zone_name,
                            'color': zone.get('color', '#888888'),
                            'points': np.array(poly, float),
                        })

                    for point in shape.findall('Point'):
                        x_str = point.get('x')
                        y_str = point.get('y')
                        
                        if x_str is not None and y_str is not None:
                            try:
                                # Convert coordinates to meters (inches to meters)
                                x_meters = float(x_str) * 0.0254
                                y_meters = float(y_str) * 0.0254
                                zone_data.append({
                                    'zone': zone_name,
                                    'x': x_meters,
                                    'y': y_meters
                                })
                            except ValueError:
                                continue
                
                if zone_data:
                    self.arena_zones = pd.DataFrame(zone_data)
                    num_zones = len(self.arena_zones['zone'].unique())
                    self.log_message(f"Parsed {num_zones} zones with {len(zone_data)} coordinate points from XML")
                    if hasattr(self, 'chk_show_zones'):
                        self.chk_show_zones.setEnabled(True)
                else:
                    self.log_message("No valid zone coordinates found in XML")
            else:
                self.log_message("No Zones section found in XML")

            # Parse anchor positions
            self.anchor_positions = []
            for anchor in root.iter('Anchor'):
                try:
                    shortid = int(anchor.get('shortid', '0'))
                    x_hex = anchor.get('x', '0x0')
                    y_hex = anchor.get('y', '0x0')
                    z_hex = anchor.get('z', '0x0')

                    # Decode IEEE 754 hex-encoded doubles
                    x_inches = struct.unpack('d', struct.pack('Q', int(x_hex, 16)))[0]
                    y_inches = struct.unpack('d', struct.pack('Q', int(y_hex, 16)))[0]
                    z_inches = struct.unpack('d', struct.pack('Q', int(z_hex, 16)))[0]

                    # Convert inches to meters
                    self.anchor_positions.append({
                        'shortid': shortid,
                        'x': x_inches * 0.0254,
                        'y': y_inches * 0.0254,
                        'z': z_inches * 0.0254
                    })
                except (ValueError, struct.error):
                    continue

            if self.anchor_positions:
                self.log_message(f"Parsed {len(self.anchor_positions)} anchor positions from XML")
                self.chk_show_anchors.setEnabled(True)
            else:
                self.log_message("No anchor positions found in XML")

            # Decode the site map. Wiser writes it as a base64 data URI in the
            # Map element's *attribute* (map="data:image/png;base64,..."), not
            # as element text, so pull it from there and fall back to text for
            # any other dialect.
            self.xml_map_image = None
            self.xml_map_extent = None
            self.xml_maps = []
            for elem in root.iter():
                if elem.tag not in ('Map', 'BackgroundImage', 'Image'):
                    continue
                payload = elem.get('map') or elem.get('image') or (elem.text or '')
                payload = payload.strip()
                if len(payload) < 200:
                    continue
                try:
                    b64 = payload.split(',', 1)[1] if ',' in payload else payload
                    img = plt.imread(io.BytesIO(base64.b64decode(b64)), format='png')
                    px_scale = float(elem.get('scale', self.xml_scale or 1.0))
                    h_px, w_px = img.shape[:2]
                    w_m = w_px * px_scale * 0.0254
                    h_m = h_px * px_scale * 0.0254
                    # Record every render so a loaded external image can be
                    # matched to the map it came from (and its correct scale).
                    self.xml_maps.append({
                        'name': elem.get('name', ''),
                        'scale': px_scale, 'w_px': w_px, 'h_px': h_px})
                    if self.xml_map_image is None:
                        # First decodable map drives the XML site view.
                        self.xml_map_image = img
                        # origin='upper' when drawn, so y runs top-down from h_m
                        self.xml_map_extent = (0.0, w_m, 0.0, h_m)
                    map_label = elem.get('name') or ''
                    map_label = f" '{map_label}'" if map_label else ''
                    self.log_message(
                        f"Decoded embedded site map{map_label}: "
                        f"{w_px}x{h_px} px @ {px_scale} in/px -> {w_m:.2f} x {h_m:.2f} m")
                except Exception as e:
                    self.log_message(f"Could not decode embedded map: {e}")

            if self.xml_map_image is None:
                self.log_message("No embedded background image found in XML")
                
        except Exception as e:
            self.log_message(f"Error parsing XML: {str(e)}")
    
    def auto_load_background(self):
        """Silently load a background image whose name clearly matches this DB.

        Only fires on a strong name match (DB name or XML config name); an
        arbitrary lone image (e.g. an ABMA model config PNG) is left for the
        preview/confirm prompt so the user picks deliberately.
        """
        if not self.db_path or self.background_image is not None:
            return  # Already loaded (e.g., from config) or no database

        db_dir = os.path.dirname(self.db_path)
        db_name = os.path.splitext(os.path.basename(self.db_path))[0]

        # Candidates ranked by name match; only accept an actual match here.
        candidates = self._find_candidate_backgrounds()
        xml_name = ''
        if getattr(self, 'xml_config_path', None):
            xml_name = os.path.splitext(os.path.basename(self.xml_config_path))[0].lower()
        selected = None
        for f in candidates:
            fname = os.path.splitext(f)[0].lower()
            if (db_name.lower() in fname or fname in db_name.lower() or
                    (xml_name and (xml_name in fname or fname in xml_name))):
                selected = f
                break

        if not selected:
            return

        self._apply_background_image(os.path.join(db_dir, selected), source="auto-loaded")

    @staticmethod
    def _name_tokens(s):
        """Significant (len>=2) alnum tokens of a name, lowercased.

        'Dark2d' -> ['dark'] ('2'/'d' dropped as too short); 'default' ->
        ['default']. Used to loosely match a map name against a filename.
        """
        return [t for t in re.findall(r'[a-z]+|\d+', (s or '').lower()) if len(t) >= 2]

    def _default_bg_scale(self, w_px, h_px, filename=None):
        """Best XML scale (in/px) for an external image about to be placed.

        Wiser writes several embedded renders, each with its own scale (e.g. a
        plain 'default' map and a decorated 'Dark2d' one at a different
        resolution). Identify which render the loaded image is, in order:

        1. exact pixel-resolution match to an embedded map (most reliable);
        2. unambiguous filename<->map-name match (e.g. an FNT ABMA screenshot
           'ABMA_VoleTerra2D_DarkModel...' -> map 'Dark2d'), for a re-exported
           image whose resolution no longer matches the embedded render;

        then fall back to the first XML scale, then None (1 in/px upstream).
        """
        maps = getattr(self, 'xml_maps', None) or []
        for m in maps:
            if m['w_px'] == w_px and m['h_px'] == h_px:
                return m['scale']
        if filename:
            stem = self._name_tokens(os.path.splitext(os.path.basename(filename))[0])
            fname_norm = ''.join(stem)
            hits = []
            for m in maps:
                toks = self._name_tokens(m.get('name', ''))
                if toks and all(t in fname_norm for t in toks):
                    hits.append(m)
            if len(hits) == 1:   # only trust an unambiguous name match
                return hits[0]['scale']
        return self.xml_scale

    def _recompute_bg_size(self):
        """Set bg_width/height_meters from the current image and bg_scale."""
        if self.background_image is None or not self.bg_scale:
            return
        h_px, w_px = self.background_image.shape[:2]
        self.bg_width_meters = w_px * self.bg_scale * 0.0254
        self.bg_height_meters = h_px * self.bg_scale * 0.0254

    def _bg_placed_extent(self):
        """(x0, x1, y0, y1) metres for the loaded background, honouring offsets.

        The image spans bg_width/height_meters (from pixels * bg_scale) with its
        bottom-left corner placed at (bg_offset_x, bg_offset_y). Returns None
        when there is no sized background image.
        """
        if self.background_image is None or not self.bg_width_meters:
            return None
        x0, y0 = self.bg_offset_x, self.bg_offset_y
        return (x0, x0 + self.bg_width_meters, y0, y0 + self.bg_height_meters)

    def _apply_background_image(self, file_path, source="loaded"):
        """Load ``file_path`` as the arena background and refresh UI/preview.

        Seeds the manual transform (scale + offset) from the XML — matching the
        image resolution to the right embedded map when several exist — then
        computes real-world dimensions, updates the status label and preview,
        and returns True on success. ``source`` is a short verb used in the log
        line ("loaded"/"auto-loaded").
        """
        try:
            self.background_image = plt.imread(file_path)
        except Exception as e:
            self.log_message(f"Error loading background image: {str(e)}")
            self.background_image = None
            self.background_image_path = None
            self.bg_width_meters = None
            self.bg_height_meters = None
            self.lbl_background_status.setText("Error loading background image")
            self.lbl_background_status.setStyleSheet("color: #aa0000; font-style: italic; font-size: 9px;")
            return False

        self.background_image_path = file_path
        name = os.path.basename(file_path)
        img_height_px, img_width_px = self.background_image.shape[:2]

        # Seed the manual transform: default scale from the matching XML map,
        # offsets reset to 0 (bottom-left corner at world origin, the Wiser
        # frame). The user can nudge these live via the Preview controls.
        self.bg_offset_x = 0.0
        self.bg_offset_y = 0.0
        matched = self._default_bg_scale(img_width_px, img_height_px, file_path)
        self.bg_scale = matched if matched else 1.0
        self._recompute_bg_size()
        if matched:
            self.log_message(
                f"✓ Background {source}: {name} "
                f"({self.bg_width_meters:.2f}m x {self.bg_height_meters:.2f}m "
                f"@ {self.bg_scale:.4f} in/px)")
        else:
            self.log_message(
                f"✓ Background {source}: {name} (no XML scale — using 1 in/px; "
                "dimensions may be inaccurate)")

        self.lbl_background_status.setText(f"✓ Background: {name}")
        self.lbl_background_status.setStyleSheet("color: #00aa00; font-style: normal; font-size: 9px;")
        self.btn_remove_background.setEnabled(True)
        self._sync_bg_transform_controls()

        self._log_background_alignment()

        # Switch the live preview to 2D (the only view that shows the
        # background) and redraw so the floorplan appears immediately.
        if self._preview_active:
            if self.combo_view_mode.currentText() != self.VIEW_2D:
                self.combo_view_mode.setCurrentText(self.VIEW_2D)
            self.refresh_preview_arena()
        return True

    def _log_background_alignment(self):
        """Log the numbers that expose an image-vs-tracking scale mismatch.

        The background's physical size is ``img_pixels * xml_scale``, but the
        anchors and zones come from absolute inch coordinates, so they are
        image-independent. If the loaded image's resolution differs from the map
        the XML scale was calibrated against (e.g. a re-rendered floorplan), the
        image ends up the wrong physical size even though its corner still sits
        at (0, 0). Comparing the image extent to the anchor bounding box makes
        that visible and suggests the scale that would fit.
        """
        if self.background_image is None:
            return
        h_px, w_px = self.background_image.shape[:2]
        self.log_message("── Background alignment check ─────────────")
        self.log_message(f"  Image: {w_px} x {h_px} px")
        self.log_message(
            f"  XML scale (first Map): {self.xml_scale} in/px" if self.xml_scale
            else "  XML scale: none — using 1 in/px fallback (likely wrong)")
        if self.bg_scale:
            res_match = any(m['w_px'] == w_px and m['h_px'] == h_px
                            for m in (getattr(self, 'xml_maps', None) or []))
            matched = self._default_bg_scale(w_px, h_px, self.background_image_path)
            note = ""
            if matched and abs(matched - self.bg_scale) < 1e-9 and matched != self.xml_scale:
                note = (" (matched to XML render of same resolution)" if res_match
                        else " (matched to XML render by filename)")
            self.log_message(f"  Effective bg scale: {self.bg_scale:.4f} in/px{note}")
        ext = self._bg_placed_extent()
        if ext:
            self.log_message(
                f"  Image placed extent: x[{ext[0]:.2f}, {ext[1]:.2f}] "
                f"y[{ext[2]:.2f}, {ext[3]:.2f}] m "
                f"(offset {self.bg_offset_x:.2f}, {self.bg_offset_y:.2f})")

        anchors = self.anchor_positions or []
        if anchors:
            xs = [a['x'] for a in anchors]
            ys = [a['y'] for a in anchors]
            ax0, ax1, ay0, ay1 = min(xs), max(xs), min(ys), max(ys)
            span_x, span_y = ax1 - ax0, ay1 - ay0
            self.log_message(
                f"  Anchors: {len(anchors)} — span x[{ax0:.2f}, {ax1:.2f}] "
                f"y[{ay0:.2f}, {ay1:.2f}] m = {span_x:.2f} x {span_y:.2f} m")
            if self.bg_width_meters and span_x > 0 and span_y > 0:
                rx = self.bg_width_meters / span_x
                ry = self.bg_height_meters / span_y
                self.log_message(
                    f"  Image / anchor-span ratio: x={rx:.2f}, y={ry:.2f} "
                    "(≈1.0–1.3 expected; far from 1 = image mis-scaled)")
                if self.bg_scale:
                    # Scale that would make the image span exactly the anchor
                    # bbox (the true arena is a bit larger, so nudge from here).
                    self.log_message(
                        f"  Scale to fit image to anchor span: x≈"
                        f"{self.bg_scale / rx:.4f}, y≈{self.bg_scale / ry:.4f} in/px "
                        f"(current {self.bg_scale:.4f}) — set via the Preview "
                        "'Bg scale' control")
        else:
            self.log_message("  No anchors parsed — cannot cross-check the image scale.")

        if getattr(self, 'xml_map_extent', None) is not None:
            e = self.xml_map_extent
            self.log_message(
                f"  Embedded XML site map extent: {e[1]:.2f} x {e[3]:.2f} m "
                "(what the XML scale actually calibrates)")
        self.log_message("──────────────────────────────────────────")

    def select_background_image(self):
        """Allow user to select a background image via the file picker."""
        file_path, _ = QFileDialog.getOpenFileName(
            self,
            "Select Background Image",
            os.path.dirname(self.db_path) if self.db_path else "",
            "Image Files (*.png *.jpg *.jpeg *.bmp);;All Files (*.*)"
        )

        if file_path and os.path.exists(file_path):
            had_scale = bool(self.xml_scale)
            if self._apply_background_image(file_path, source="loaded") and not had_scale:
                QMessageBox.warning(
                    self, "No Scale Available",
                    "No XML scale has been loaded. The background image dimensions "
                    "may be incorrect.\n\nLoad an XML configuration file first for "
                    "accurate floorplan scaling."
                )

    def remove_background(self):
        """Remove the background image from visualizations"""
        self.background_image = None
        self.background_image_path = None
        self.bg_width_meters = None
        self.bg_height_meters = None
        self.bg_scale = None
        self.bg_offset_x = 0.0
        self.bg_offset_y = 0.0
        self.lbl_background_status.setText("No background image loaded")
        self.lbl_background_status.setStyleSheet("color: #666666; font-style: italic; font-size: 9px;")
        self.btn_remove_background.setEnabled(False)
        self._sync_bg_transform_controls()
        self.log_message("Background image removed")
        if self.preview_canvas_2d is not None:
            self.preview_canvas_2d.background_image = None
            self.preview_canvas_2d.bg_extent = None
        if self._preview_active:
            self.refresh_preview_arena()

    def on_show_background_toggled(self, _state=None):
        """Redraw the preview after a display toggle (background or anchors)."""
        if self._preview_active:
            self.refresh_preview_arena()

    def _sync_bg_transform_controls(self):
        """Push the current scale/offset onto the spinboxes without re-firing.

        Enables the control group only when a background image is loaded. Signals
        are blocked so seeding the widgets does not trigger a redraw loop.
        """
        if not hasattr(self, 'spin_bg_scale'):
            return   # controls not built yet (early call during startup)
        have_img = self.background_image is not None
        for w in (self.spin_bg_scale, self.spin_bg_offx, self.spin_bg_offy):
            w.blockSignals(True)
        self.spin_bg_scale.setValue(self.bg_scale if self.bg_scale else 1.0)
        self.spin_bg_offx.setValue(self.bg_offset_x)
        self.spin_bg_offy.setValue(self.bg_offset_y)
        for w in (self.spin_bg_scale, self.spin_bg_offx, self.spin_bg_offy):
            w.blockSignals(False)
        if hasattr(self, '_bg_transform_box'):
            self._bg_transform_box.setEnabled(have_img)

    def on_bg_transform_changed(self, _value=None):
        """Apply the manual scale + offset from the spinboxes and redraw.

        Re-sizes the background from the new scale, shifts it by the new offset,
        and refreshes the live preview so the user sees the alignment update as
        they nudge. No-op when no image is loaded.
        """
        if self.background_image is None:
            return
        self.bg_scale = self.spin_bg_scale.value()
        self.bg_offset_x = self.spin_bg_offx.value()
        self.bg_offset_y = self.spin_bg_offy.value()
        self._recompute_bg_size()
        if self._preview_active:
            self.refresh_preview_arena()

    def reset_bg_transform(self):
        """Restore scale to the XML default (matched to this image) and offset to 0."""
        if self.background_image is None:
            return
        h_px, w_px = self.background_image.shape[:2]
        matched = self._default_bg_scale(w_px, h_px, self.background_image_path)
        self.bg_scale = matched if matched else 1.0
        self.bg_offset_x = 0.0
        self.bg_offset_y = 0.0
        self._recompute_bg_size()
        self._sync_bg_transform_controls()
        self.log_message(
            f"Background transform reset to XML: scale {self.bg_scale:.4f} in/px, "
            "offset (0.00, 0.00) m")
        if self._preview_active:
            self.refresh_preview_arena()

    def on_preview_theme_changed(self, _state=None):
        """Switch the preview canvases between the dark and light palettes.
        Light is the default; the checkbox opts into dark."""
        theme = "dark" if self.chk_preview_dark.isChecked() else "light"
        self.preview_canvas_2d.set_theme(theme)
        if self.preview_canvas_3d is not None:
            self.preview_canvas_3d.set_theme(theme)
        if self._preview_active:
            self.refresh_preview_arena()
        else:
            self.preview_canvas_2d.show_placeholder()

    def on_table_selected(self, table_name):
        """Handle table selection"""
        if table_name:
            self.table_name = table_name
            self.load_tags_from_table()
    
    def preview_table(self):
        """Preview table data in a dialog"""
        if not self.db_path or not self.table_name:
            return
        
        try:
            # Load first 100 rows for preview
            conn = sqlite3.connect(self.db_path)
            query = f"SELECT * FROM {self.table_name} LIMIT 100"
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Create preview dialog
            from PyQt5.QtWidgets import QDialog, QTableWidget, QTableWidgetItem, QHeaderView
            
            dialog = QDialog(self)
            dialog.setWindowTitle(f"Preview: {self.table_name}")
            dialog.setGeometry(100, 100, 1000, 600)
            
            layout = QVBoxLayout()
            
            # Info label
            info_label = QLabel(f"Showing first 100 rows of {len(df)} columns")
            info_label.setStyleSheet("color: #cccccc; font-weight: bold; padding: 10px;")
            layout.addWidget(info_label)
            
            # Create table widget
            table = QTableWidget()
            table.setRowCount(len(df))
            table.setColumnCount(len(df.columns))
            table.setHorizontalHeaderLabels(df.columns.tolist())
            
            # Populate table
            for i in range(len(df)):
                for j in range(len(df.columns)):
                    item = QTableWidgetItem(str(df.iloc[i, j]))
                    table.setItem(i, j, item)
            
            # Auto-resize columns
            table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeToContents)
            
            # Style table
            table.setStyleSheet("""
                QTableWidget {
                    background-color: #1e1e1e;
                    color: #cccccc;
                    gridline-color: #3f3f3f;
                    border: 1px solid #3f3f3f;
                }
                QHeaderView::section {
                    background-color: #2b2b2b;
                    color: #0078d4;
                    font-weight: bold;
                    padding: 4px;
                    border: 1px solid #3f3f3f;
                }
            """)
            
            layout.addWidget(table)
            
            # Close button
            close_btn = QPushButton("Close")
            close_btn.clicked.connect(dialog.close)
            layout.addWidget(close_btn)
            
            dialog.setLayout(layout)
            dialog.exec_()
            
        except Exception as e:
            QMessageBox.critical(self, "Preview Error", f"Failed to preview table: {str(e)}")
    
    def on_smoothing_changed(self, method):
        """Handle smoothing method change"""
        clean_method = method.replace(" (default)", "")
        is_ewma = clean_method == "Forward-Backward Exponentially Weighted Moving Average"
        is_rolling = clean_method in ("Rolling Average", "Rolling Median")
        # Savitzky-Golay sizes its own window, so only the rolling methods and
        # EWMA expose the parameter spinbox.
        needs_param = is_ewma or is_rolling

        self.lbl_rolling_window.setText("Span (samples):" if is_ewma else "Smoothing Window:")
        self.spin_rolling_window.setEnabled(needs_param)
        self.spin_rolling_window.setVisible(needs_param)
        self.lbl_rolling_window.setVisible(needs_param)
        # The seconds/samples toggle only applies to the rolling windows; the
        # EWMA span is always in samples and Savitzky-Golay has no exposed size.
        self.combo_window_units.setVisible(is_rolling)

    def on_save_plots_toggled(self):
        """Handle save plots checkbox toggle"""
        enabled = self.chk_save_plots.isChecked()
        self.plot_types_widget.setVisible(enabled)
        self.svg_option_widget.setVisible(enabled)
    
    def on_proximity_detection_toggled(self):
        """Handle proximity detection checkbox toggle"""
        enabled = self.chk_proximity_detection.isChecked()
        self.proximity_threshold_widget.setVisible(enabled)

    def on_save_animation_toggled(self):
        """Handle save animation checkbox toggle"""
        enabled = self.chk_save_animation.isChecked()
        self.animation_options_widget.setVisible(enabled)
    
    def update_frame_estimate(self):
        """Update estimated frame count based on animation settings and loaded data"""
        if self.data is None or 'Timestamp' not in self.data.columns:
            self.lbl_estimated_frames.setText("Estimated frames: -- (load data first)")
            return
        source_data = self.data

        try:
            # Get animation parameters
            fps = int(self.combo_animation_fps.currentText())
            speed_text = self.combo_animation_speed.currentText()
            speed_multiplier = int(speed_text.replace('x', ''))

            # Calculate frame interval (real seconds per frame)
            frame_interval = speed_multiplier / fps

            # Get total time span of data
            time_span = (source_data['Timestamp'].max() - source_data['Timestamp'].min()).total_seconds()
            
            # Estimate number of frames
            estimated_frames = int(time_span / frame_interval)
            
            # Format with commas for readability
            frames_formatted = f"{estimated_frames:,}"
            
            # Calculate estimated video duration
            video_duration = estimated_frames / fps
            
            if video_duration >= 60:
                duration_str = f"{video_duration/60:.1f} min"
            else:
                duration_str = f"{video_duration:.1f} sec"
            
            self.lbl_estimated_frames.setText(
                f"Estimated frames: {frames_formatted} (~{duration_str} video @ {fps} FPS)"
            )
        except Exception as e:
            self.lbl_estimated_frames.setText(f"Estimated frames: Error calculating ({str(e)})")
    
    def on_daily_animations_toggled(self):
        """Handle daily animations checkbox toggle"""
        enabled = self.chk_daily_animations.isChecked()
        self.daily_animation_days_widget.setVisible(enabled)
    
    def populate_animation_days_from_list(self, date_strings):
        """Populate day checkboxes from a list of date strings (works without loading full data)"""
        # Clear existing checkboxes
        for cb in self.daily_animation_day_checkboxes.values():
            cb.deleteLater()
        self.daily_animation_day_checkboxes.clear()
        
        if not date_strings:
            return
        
        # Create checkboxes for each day
        for i, date_str in enumerate(date_strings):
            cb = QCheckBox(f"Day {i+1}: {date_str}")
            cb.setChecked(True)  # Default to all days selected
            self.daily_animation_day_checkboxes[date_str] = cb
            self.daily_days_layout_inner.addWidget(cb)
    
    def populate_animation_days(self):
        """Populate day checkboxes based on loaded data (called after preview loads)"""
        # Clear existing checkboxes
        for cb in self.daily_animation_day_checkboxes.values():
            cb.deleteLater()
        self.daily_animation_day_checkboxes.clear()

        if self.data is None or 'Timestamp' not in self.data.columns:
            return

        # Get unique dates
        dates = pd.to_datetime(self.data['Timestamp']).dt.date.unique()
        dates = sorted(dates)
        date_strings = [date.strftime('%Y-%m-%d') for date in dates]
        
        # Use the list-based function
        self.populate_animation_days_from_list(date_strings)
        
        self.log_message(f"Found {len(dates)} unique days in dataset")
    
    def open_identity_dialog(self):
        """Open dialog to assign identities to tags"""
        if not self.available_tags:
            QMessageBox.warning(self, "No Tags", "Please load a database and table first")
            return

        # Get only selected tags
        selected_tags = [tag for tag, cb in self.tag_checkboxes.items() if cb.isChecked()]
        if not selected_tags:
            QMessageBox.warning(self, "No Tags Selected", "Please select at least one tag")
            return

        # Query per-tag time ranges from database
        tag_time_ranges = {}
        try:
            tz = pytz.timezone(self.combo_timezone.currentText())
            conn = sqlite3.connect(self.db_path)
            placeholders = ','.join(['?'] * len(selected_tags))
            query = f"SELECT shortid, MIN(timestamp) as first_ts, MAX(timestamp) as last_ts FROM {self.table_name} WHERE shortid IN ({placeholders}) GROUP BY shortid"
            cursor = conn.execute(query, selected_tags)
            for row in cursor:
                tag_id, first_ts, last_ts = row
                # Convert ms timestamps to timezone-aware datetimes
                first_dt = pd.Timestamp(first_ts, unit='ms', tz='UTC').tz_convert(tz)
                last_dt = pd.Timestamp(last_ts, unit='ms', tz='UTC').tz_convert(tz)
                tag_time_ranges[tag_id] = {
                    'start': first_dt.strftime('%Y-%m-%d %H:%M:%S'),
                    'end': last_dt.strftime('%Y-%m-%d %H:%M:%S'),
                }
            conn.close()
        except Exception as e:
            self.log_message(f"Warning: Could not query tag time ranges: {str(e)}")

        dialog = IdentityAssignmentDialog(selected_tags, self.tag_identities, tag_time_ranges, self)
        if dialog.exec_() == QDialog.Accepted:
            self.tag_identities = dialog.get_identities()
            self.log_message(f"Updated identities for {len(self.tag_identities)} tags")
            # Snapshot the export-details JSON to the analysis folder now, so it
            # exists and reflects the just-applied identities before any export.
            self.write_live_config()
            # Update tag checkbox labels to reflect new identities
            self.update_tag_labels()
            # Recolour preview markers by the newly-assigned sex (blue=M, red=F)
            if self._preview_active and self.preview_tags:
                self.preview_colors = self._preview_tag_colors(self.preview_tags)
                self.render_preview_frame()
            self.lbl_status.setText(f"Identity assignments saved for {len(self.tag_identities)} tags")
            
            # Auto-save configuration to JSON
            if self.db_path:
                db_dir = os.path.dirname(self.db_path)
                db_filename = os.path.basename(self.db_path)
                db_name = os.path.splitext(db_filename)[0]
                output_dir = os.path.join(db_dir, f"{db_name}_FNT_analysis")
                os.makedirs(output_dir, exist_ok=True)
                self.save_config(output_dir)
                self.log_message("✓ Configuration auto-saved to JSON")
    
    def stop_export(self):
        """Cancel ongoing export operations"""
        self.export_cancelled = True
        self.log_message("⚠ Export cancellation requested...")
        
        # Stop worker thread if running
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self.log_message("✗ Plot export cancelled")

        # Drop any temp render CSVs left behind by the cancelled run.
        self._cleanup_plot_working_files()

        # Reset UI
        self.exporting = False
        self.btn_export.setEnabled(True)
        self.btn_stop_export.setVisible(False)
        self.progress_widget.setVisible(False)
        self.progress_bar.setValue(0)
        self.lbl_export_progress.setText("")
    
    def load_tags_from_table(self):
        """Load tags from selected table"""
        if not self.db_path or not self.table_name:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            query = f"SELECT DISTINCT shortid FROM {self.table_name} ORDER BY shortid"
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            self.available_tags = df['shortid'].tolist()
            self.update_tag_selection()

            # Enable export button
            self.btn_export.setEnabled(True)

            # Load unique days for daily animation options
            self.load_unique_days_from_database()

        except Exception as e:
            QMessageBox.critical(self, "Error", f"Failed to load tags: {str(e)}")
    
    def load_unique_days_from_database(self):
        """Load unique days from database without loading full data"""
        if not self.db_path or not self.table_name:
            return
        
        try:
            conn = sqlite3.connect(self.db_path)
            # Query for distinct dates
            query = f"""
                SELECT DISTINCT date(datetime(timestamp/1000, 'unixepoch'), 'localtime') as date
                FROM {self.table_name}
                ORDER BY date
            """
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            if len(df) > 0:
                self.populate_animation_days_from_list(df['date'].tolist())
                self.log_message(f"Found {len(df)} unique days in database")
            
        except Exception as e:
            self.log_message(f"Could not load unique days: {str(e)}")
    
    def update_tag_selection(self):
        """Update tag checkboxes"""
        for cb in self.tag_checkboxes.values():
            cb.deleteLater()
        self.tag_checkboxes.clear()
        
        if self.lbl_no_tags:
            self.lbl_no_tags.deleteLater()
            self.lbl_no_tags = None
        
        # Remove existing buttons if they exist
        if hasattr(self, 'tag_buttons_layout'):
            while self.tag_buttons_layout.count():
                item = self.tag_buttons_layout.takeAt(0)
                if item.widget():
                    item.widget().deleteLater()
            self.tag_layout.removeItem(self.tag_buttons_layout)
        
        if hasattr(self, 'btn_assign_identities'):
            self.btn_assign_identities.deleteLater()
        
        for tag in self.available_tags:
            hex_id = hex(tag).upper().replace('0X', '')
            # Show HexID with identity info only if user has configured it
            if tag in self.tag_identities:
                info = self.tag_identities[tag]
                sex = info.get('sex', '')
                identity = info.get('identity', '')
                if sex and identity:
                    cb = QCheckBox(f"HexID {hex_id} ({sex}, {identity})")
                else:
                    cb = QCheckBox(f"HexID {hex_id}")
            else:
                cb = QCheckBox(f"HexID {hex_id}")
            cb.setChecked(True)
            cb.stateChanged.connect(self.update_identity_button_state)
            # The preview's frame columns and its timeline range both derive
            # from the tag selection, so a change has to rebuild both.
            cb.stateChanged.connect(self.on_tag_selection_changed)
            self.tag_checkboxes[tag] = cb
            self.tag_layout.addWidget(cb)
        
        # Add Select All/None buttons below checkboxes
        self.tag_buttons_layout = QHBoxLayout()
        btn_select_all = QPushButton("Select All")
        btn_select_all.setToolTip("Check all tags for export")
        btn_select_all.setStyleSheet("padding: 6px; font-size: 10px;")
        btn_select_all.clicked.connect(self.select_all_tags)
        btn_select_none = QPushButton("Select None")
        btn_select_none.setToolTip("Uncheck all tags")
        btn_select_none.setStyleSheet("padding: 6px; font-size: 10px;")
        btn_select_none.clicked.connect(self.select_none_tags)
        self.tag_buttons_layout.addWidget(btn_select_all)
        self.tag_buttons_layout.addWidget(btn_select_none)
        self.tag_layout.addLayout(self.tag_buttons_layout)

        # Add Configure Identities button below Select All/None
        self.btn_assign_identities = QPushButton("Configure Identities...")
        self.btn_assign_identities.clicked.connect(self.open_identity_dialog)
        self.btn_assign_identities.setEnabled(False)
        self.btn_assign_identities.setToolTip(
            "Assign sex (M/F), custom IDs, and active time windows to each tag. "
            "Time pickers default to the tag's first/last timestamp in your selected timezone."
        )
        self.btn_assign_identities.setStyleSheet("padding: 6px; font-size: 10px;")
        self.tag_layout.addWidget(self.btn_assign_identities)
        
        # Apply pending tag selection from loaded config
        self.apply_pending_tag_selection()

        # Update identity button state
        self.update_identity_button_state()

        # Tags now exist for this database — bring the live preview to life.
        # Checkbox creation above sets states before their signals are wired,
        # so no on_tag_selection_changed fired; activate explicitly.
        if not self._preview_active and self.selected_preview_tags():
            self.activate_preview()

    def update_identity_button_state(self):
        """Enable Configure Identities button if any tag is selected"""
        any_selected = any(cb.isChecked() for cb in self.tag_checkboxes.values())
        self.btn_assign_identities.setEnabled(any_selected)
    
    def update_tag_labels(self):
        """Update tag checkbox labels to reflect sex and ID information"""
        for tag, cb in self.tag_checkboxes.items():
            hex_id = hex(tag).upper().replace('0X', '')
            if tag in self.tag_identities:
                info = self.tag_identities[tag]
                sex = info.get('sex', '')
                identity = info.get('identity', '')
                if sex and identity:
                    cb.setText(f"HexID {hex_id} ({sex}, {identity})")
                else:
                    cb.setText(f"HexID {hex_id}")
            else:
                cb.setText(f"HexID {hex_id}")
    
    def select_all_tags(self):
        """Select all tags"""
        for cb in self.tag_checkboxes.values():
            cb.setChecked(True)

    def select_none_tags(self):
        """Deselect all tags"""
        for cb in self.tag_checkboxes.values():
            cb.setChecked(False)
    
    def get_smoothing_method(self):
        """Get the current smoothing method name (stripped of UI hints like '(default)')"""
        text = self.combo_smoothing.currentText()
        return text.replace(" (default)", "")

    def apply_smoothing(self):
        """Apply smoothing to self.data"""
        self.data = self.apply_smoothing_to_data(self.data, self.get_smoothing_method())
    
    def apply_smoothing_to_data(self, data, method, window=None, units=None):
        """Apply smoothing to a dataframe (works on any dataframe, not just self.data).

        ``window`` (samples/seconds) and ``units`` ("Seconds"/"Samples")
        override the Export Options spinbox/selector when given, so the live
        preview can drive smoothing from its own independent window controls.
        """
        win = self.spin_rolling_window.value() if window is None else window
        use_seconds = ((self.combo_window_units.currentText() == "Seconds")
                       if units is None else (units == "Seconds"))

        def apply_savgol(group):
            window_length = min(31, len(group))
            if window_length % 2 == 0:
                window_length -= 1
            polyorder = min(2, window_length - 1)
            if len(group) > polyorder:
                return savgol_filter(group, window_length=window_length, polyorder=polyorder)
            return group

        if method == "Savitzky-Golay":
            data['smoothed_x'] = data.groupby('shortid')['location_x'].transform(apply_savgol)
            data['smoothed_y'] = data.groupby('shortid')['location_y'].transform(apply_savgol)
        elif method in ("Rolling Average", "Rolling Median"):
            # The window value is interpreted as seconds (time-based, centred on
            # each tag's Timestamp) or as a fixed sample count, per the units
            # selector next to the spinbox. Time-based is the default and is
            # independent of the irregular, sub-1 Hz reporting rate.
            agg = 'mean' if method == "Rolling Average" else 'median'
            rolling_smooth_xy(data, agg, win, use_seconds)
        elif method == "Forward-Backward Exponentially Weighted Moving Average":
            # Window doubles as the EWMA span. No minimum floor — a span of
            # 1-2 is legitimate (span=1 gives alpha=1, i.e. passthrough).
            span = max(1, win)

            data['smoothed_x'] = data.groupby('shortid')['location_x'].transform(
                lambda x: forward_backward_ewma(x, span))
            data['smoothed_y'] = data.groupby('shortid')['location_y'].transform(
                lambda x: forward_backward_ewma(x, span))

        return data
    
    def apply_filters(self, data):
        """Apply velocity and jump filtering with time window grouping"""
        initial_count = len(data)
        
        # Calculate time differences and group by time gaps (prevents filtering across battery restarts)
        time_gap_threshold = self.spin_time_gap.value()
        data['time_diff'] = data.groupby('shortid')['Timestamp'].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds()
        data['time_diff_s'] = np.ceil(data['time_diff']).astype(int)
        data['tw_group'] = data.groupby('shortid')['time_diff_s'].apply(
            lambda x: (x > time_gap_threshold).cumsum()
        ).reset_index(level=0, drop=True)
        
        # Calculate distance and velocity within time window groups
        data['distance'] = np.sqrt(
            (data['location_x'] - data.groupby(['shortid', 'tw_group'])['location_x'].shift())**2 +
            (data['location_y'] - data.groupby(['shortid', 'tw_group'])['location_y'].shift())**2
        )
        data['velocity'] = data['distance'] / data['time_diff']
        
        # Apply velocity filtering if enabled
        if self.chk_velocity_filter.isChecked():
            velocity_threshold = self.spin_velocity_threshold.value()
            before_velocity = len(data)
            data = data[(data['velocity'] <= velocity_threshold) | (data['velocity'].isna())]
            removed_velocity = before_velocity - len(data)
            if removed_velocity > 0:
                self.log_message(f"  Removed {removed_velocity} points with velocity > {velocity_threshold} m/s")
        
        # Apply jump filtering if enabled
        if self.chk_jump_filter.isChecked():
            jump_threshold = self.spin_jump_threshold.value()
            before_jump = len(data)
            data['is_jump'] = (data['distance'] > jump_threshold)
            data = data[~data['is_jump']]
            removed_jump = before_jump - len(data)
            if removed_jump > 0:
                self.log_message(f"  Removed {removed_jump} points with distance jump > {jump_threshold} m")
        
        # Clean up temporary columns
        data = data.drop(columns=['time_diff', 'time_diff_s', 'tw_group', 'distance', 'velocity'], errors='ignore')
        if 'is_jump' in data.columns:
            data = data.drop(columns=['is_jump'])
        
        final_count = len(data)
        if initial_count != final_count:
            self.log_message(f"  Total filtered: {initial_count - final_count} points ({100*(initial_count-final_count)/initial_count:.1f}%)")
        
        return data
    
    def reset_filter_stats(self):
        """Clear accumulated filter statistics at the start of an export run."""
        self.filter_stats = {}

    def apply_filters_to_data(self, data, collect_stats=True, velocity=None, jump=None,
                              velocity_thresh=None, jump_thresh=None):
        """Apply velocity and jump thresholding with time-window grouping.

        ``velocity``/``jump`` override whether each threshold is applied and
        ``velocity_thresh``/``jump_thresh`` override the cutoff values; when None
        they follow the Export checkboxes/spinboxes. The preview passes its own
        toggles and values so the user can tune thresholding live.
        """
        use_velocity = self.chk_velocity_filter.isChecked() if velocity is None else velocity
        use_jump = self.chk_jump_filter.isChecked() if jump is None else jump
        initial_count = len(data)
        removed_velocity = 0
        removed_jump = 0
        
        # Make explicit copy at start to avoid any SettingWithCopyWarning
        data = data.copy()
        
        # Calculate time differences and group by time gaps (prevents filtering across battery restarts)
        time_gap_threshold = self.spin_time_gap.value()
        data['time_diff'] = data.groupby('shortid')['Timestamp'].diff().fillna(pd.Timedelta(seconds=0)).dt.total_seconds()
        data['time_diff_s'] = np.ceil(data['time_diff']).astype(int)
        data['tw_group'] = data.groupby('shortid')['time_diff_s'].apply(
            lambda x: (x > time_gap_threshold).cumsum()
        ).reset_index(level=0, drop=True)
        
        # Calculate distance and velocity within time window groups
        data['distance'] = np.sqrt(
            (data['location_x'] - data.groupby(['shortid', 'tw_group'])['location_x'].shift())**2 +
            (data['location_y'] - data.groupby(['shortid', 'tw_group'])['location_y'].shift())**2
        )
        data['velocity'] = data['distance'] / data['time_diff']
        
        # Apply velocity filtering if enabled
        if use_velocity:
            velocity_threshold = (self.spin_velocity_threshold.value()
                                  if velocity_thresh is None else velocity_thresh)
            before_velocity = len(data)
            data = data[(data['velocity'] <= velocity_threshold) | (data['velocity'].isna())].copy()
            removed_velocity = before_velocity - len(data)
            if removed_velocity > 0:
                self.log_message(f"  Removed {removed_velocity} points with velocity > {velocity_threshold} m/s")
        else:
            removed_velocity = 0

        # Apply jump filtering if enabled
        if use_jump:
            jump_threshold = (self.spin_jump_threshold.value()
                              if jump_thresh is None else jump_thresh)
            before_jump = len(data)
            data['is_jump'] = (data['distance'] > jump_threshold)
            data = data[~data['is_jump']].copy()
            removed_jump = before_jump - len(data)
            if removed_jump > 0:
                self.log_message(f"  Removed {removed_jump} points with distance jump > {jump_threshold} m")
        else:
            removed_jump = 0
        
        # Clean up temporary columns
        data = data.drop(columns=['time_diff', 'time_diff_s', 'tw_group', 'distance', 'velocity'], errors='ignore')
        if 'is_jump' in data.columns:
            data = data.drop(columns=['is_jump'])
        
        final_count = len(data)
        if initial_count != final_count:
            self.log_message(f"  Total filtered: {initial_count - final_count} points ({100*(initial_count-final_count)/initial_count:.1f}%)")
        
        # Accumulate stats for the run summary. This runs once per tag during
        # export, so the totals must sum across calls — replacing them would
        # report only the last tag while labelling it as the whole run.
        # The preview passes collect_stats=False so scrubbing cannot pollute
        # an export's figures.
        if collect_stats:
            s = getattr(self, 'filter_stats', None) or {}
            self.filter_stats = {
                'initial_count': s.get('initial_count', 0) + initial_count,
                'removed_velocity': s.get('removed_velocity', 0) + removed_velocity,
                'removed_jump': s.get('removed_jump', 0) + removed_jump,
                'final_count': s.get('final_count', 0) + final_count,
                'tags_processed': s.get('tags_processed', 0) + 1,
            }
            tot = self.filter_stats['initial_count']
            self.filter_stats['percent_filtered'] = (
                100 * (tot - self.filter_stats['final_count']) / tot if tot else 0)

        return data

    def xml_config_summary(self):
        """Site metadata parsed from the detected XML, or None if there is none.

        Makes the export record self-contained: it captures the anchor/antenna
        positions, zone polygons, scale and map extent read from the site XML
        (all lengths already converted to meters). The embedded map *image*
        itself is not copied — it lives in the XML file referenced by 'path'.

        Every value is coerced to a plain Python type so json.dump never chokes
        on a numpy scalar.
        """
        if not getattr(self, 'xml_config_path', None):
            return None

        summary = {
            'path': self.xml_config_path,
            'filename': os.path.basename(self.xml_config_path),
            'scale_inches_per_px': (float(self.xml_scale)
                                    if self.xml_scale is not None else None),
            'anchor_positions_m': [
                {'shortid': int(a['shortid']),
                 'x': float(a['x']), 'y': float(a['y']), 'z': float(a['z'])}
                for a in (self.anchor_positions or [])
            ],
            'zones': [
                {'name': z.get('name'),
                 'color': z.get('color'),
                 'points_m': (z['points'].tolist() if hasattr(z.get('points'), 'tolist')
                              else z.get('points'))}
                for z in (getattr(self, 'xml_zones', None) or [])
            ],
        }
        if getattr(self, 'xml_map_extent', None) is not None:
            summary['map_extent_m'] = [float(v) for v in self.xml_map_extent]
        if self.bg_width_meters is not None and self.bg_height_meters is not None:
            summary['background_size_m'] = [float(self.bg_width_meters),
                                            float(self.bg_height_meters)]
        return summary

    def get_config_dict(self):
        """Get current configuration as dictionary"""
        from datetime import datetime
        config = {
            'fnt_version': get_fnt_version(),
            'run_timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S'),
            'database_path': self.db_path,
            'database_name': os.path.basename(self.db_path) if self.db_path else None,
            'table_name': self.table_name,
            'selected_tags': [tag for tag, cb in self.tag_checkboxes.items() if cb.isChecked()],
            'timezone': self.combo_timezone.currentText(),
            'velocity_filter': self.chk_velocity_filter.isChecked(),
            'velocity_threshold': self.spin_velocity_threshold.value(),
            'jump_filter': self.chk_jump_filter.isChecked(),
            'jump_threshold': self.spin_jump_threshold.value(),
            'time_gap': self.spin_time_gap.value(),
            'smoothing_method': self.combo_smoothing.currentText(),
            'rolling_window': self.spin_rolling_window.value(),
            'rolling_window_units': self.combo_window_units.currentText(),
            'show_anchors': self.chk_show_anchors.isChecked(),
            'show_tracking': self.chk_show_tracking.isChecked(),
            'preview_tag_size': self.spin_tag_size.value(),
            'show_battery': self.chk_show_battery.isChecked(),
            'export_raw_csv': self.chk_export_raw_csv.isChecked(),
            'export_smoothed_csv': self.chk_export_smoothed_csv.isChecked(),
            'proximity_detection': self.chk_proximity_detection.isChecked(),
            'proximity_threshold': self.spin_proximity_threshold.value(),
            'save_plots': self.chk_save_plots.isChecked(),
            'downsample_plots': self.chk_downsample_plots.isChecked(),
            'plot_downsample_hz': self.spin_plot_downsample_hz.value(),
            'downsample_animation': self.chk_downsample_animation.isChecked(),
            'anim_downsample_hz': self.spin_anim_downsample_hz.value(),
            'save_svg': self.chk_save_svg.isChecked(),
            'plot_types': {k: cb.isChecked() for k, cb in self.plot_type_checkboxes.items()},
            'save_animation': self.chk_save_animation.isChecked(),
            'animation_trail': self.spin_animation_trail.value(),
            'animation_tag_size': self.spin_anim_tag_size.value(),
            'animation_show_battery': self.chk_show_battery_export.isChecked(),
            'animation_speed': self.combo_animation_speed.currentText(),
            'animation_fps': self.combo_animation_fps.currentText(),
            'color_by': self.combo_color_by.currentText(),
            'video_quality': self.combo_video_quality.currentText(),
            'daily_animations': self.chk_daily_animations.isChecked(),
            'tag_identities': self.tag_identities,
            'plot_layers': dict(self.plot_layers),
            # Site-map image copied into the analysis folder (filename + extent),
            # or None. Written by _export_sitemap during export.
            'sitemap': getattr(self, '_exported_sitemap', None),
            'background_image_path': self.background_image_path,
            # Manual background transform (option (b)) so the alignment nudge
            # survives a reload. Scale is in/px; offset is metres.
            'background_scale_inpx': self.bg_scale,
            'background_offset_m': [float(self.bg_offset_x), float(self.bg_offset_y)],
            'arena_zones': self.arena_zones.to_dict('records') if self.arena_zones is not None else None,
            # Anchor/antenna positions, zones, scale and map extent from the
            # site XML (all in meters); None when no XML was detected.
            'xml_config': self.xml_config_summary()
        }
        return config
    
    def save_config(self, output_dir):
        """Save current configuration to JSON file"""
        config = self.get_config_dict()
        config_path = os.path.join(output_dir, 'fnt_config.json')

        try:
            with open(config_path, 'w') as f:
                json.dump(config, f, indent=4)
            self.log_message(f"Config saved: {os.path.basename(config_path)}")
        except Exception as e:
            self.log_message(f"Warning: Could not save config: {str(e)}")

    def analysis_dir_for_db(self):
        """Path to the '<db>_FNT_analysis' folder for the loaded database.

        This is the same folder the export writes to and that
        load_config_if_exists reads back on open, so a config written here is
        picked up automatically next time the database is loaded. Returns None
        if no database is selected.
        """
        if not self.db_path:
            return None
        db_dir = os.path.dirname(self.db_path)
        db_name = os.path.splitext(os.path.basename(self.db_path))[0]
        return os.path.join(db_dir, f"{db_name}_FNT_analysis")

    def write_live_config(self):
        """Write fnt_config.json to the analysis folder right now.

        Config-only (no CSVs/plots/heavy reads): used to snapshot the current
        settings as soon as the user applies identity assignments, so the
        export-details JSON exists and stays current before a full export runs.
        Creates the analysis folder if it does not exist yet.
        """
        analysis_dir = self.analysis_dir_for_db()
        if not analysis_dir:
            return
        try:
            os.makedirs(analysis_dir, exist_ok=True)
            self.save_config(analysis_dir)
        except Exception as e:
            self.log_message(f"Warning: Could not write live config: {str(e)}")

    def load_config_if_exists(self):
        """Check for existing config file and load it.
        Returns True if a config was loaded, False otherwise."""
        if not self.db_path:
            return False

        db_dir = os.path.dirname(self.db_path)
        db_filename = os.path.basename(self.db_path)
        db_name = os.path.splitext(db_filename)[0]
        analysis_dir = os.path.join(db_dir, f"{db_name}_FNT_analysis")
        config_path = os.path.join(analysis_dir, 'fnt_config.json')

        if not os.path.exists(config_path):
            return False
        
        try:
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            # Load configuration into GUI
            if 'table_name' in config and config['table_name']:
                # Store for deferred application after combo_table is populated
                self.pending_table_name = config['table_name']
                # Also try to set directly if combo is already populated
                index = self.combo_table.findText(config['table_name'])
                if index >= 0:
                    self.combo_table.setCurrentIndex(index)
            
            if 'timezone' in config:
                index = self.combo_timezone.findText(config['timezone'])
                if index >= 0:
                    self.combo_timezone.setCurrentIndex(index)
            
            if 'smoothing_method' in config:
                # Match on the base method name so configs saved under the old
                # labels (e.g. "Rolling Average (default)") still resolve after
                # the "(default)" marker moved to None.
                want = config['smoothing_method'].replace(" (default)", "")
                # The EWMA option's label was spelled out in full; map the old
                # abbreviated label so pre-existing configs still restore it.
                if want == "Forward-Backward EWMA":
                    want = "Forward-Backward Exponentially Weighted Moving Average"
                index = next(
                    (i for i in range(self.combo_smoothing.count())
                     if self.combo_smoothing.itemText(i).replace(" (default)", "") == want),
                    -1)
                if index >= 0:
                    self.combo_smoothing.setCurrentIndex(index)
            
            if 'rolling_window' in config:
                self.spin_rolling_window.setValue(config['rolling_window'])

            if 'rolling_window_units' in config:
                idx = self.combo_window_units.findText(config['rolling_window_units'])
                if idx >= 0:
                    self.combo_window_units.setCurrentIndex(idx)
            
            if 'velocity_filter' in config:
                self.chk_velocity_filter.setChecked(config['velocity_filter'])
            
            if 'velocity_threshold' in config:
                self.spin_velocity_threshold.setValue(config['velocity_threshold'])
            
            if 'jump_filter' in config:
                self.chk_jump_filter.setChecked(config['jump_filter'])
            
            if 'jump_threshold' in config:
                self.spin_jump_threshold.setValue(config['jump_threshold'])
            
            if 'time_gap' in config:
                self.spin_time_gap.setValue(config['time_gap'])
            
            if 'show_tracking' in config:
                self.chk_show_tracking.setChecked(config['show_tracking'])

            if 'preview_tag_size' in config:
                self.spin_tag_size.setValue(config['preview_tag_size'])

            if 'show_battery' in config:
                self.chk_show_battery.setChecked(config['show_battery'])

            if 'animation_tag_size' in config:
                self.spin_anim_tag_size.setValue(config['animation_tag_size'])

            if 'animation_show_battery' in config:
                self.chk_show_battery_export.setChecked(config['animation_show_battery'])

            if 'export_raw_csv' in config:
                self.chk_export_raw_csv.setChecked(config['export_raw_csv'])

            if 'export_smoothed_csv' in config:
                self.chk_export_smoothed_csv.setChecked(config['export_smoothed_csv'])

            # 'export_downsampled_csv' / 'downsample_hz' are obsolete (the
            # downsampled export was removed); ignored if present in old configs.

            if isinstance(config.get('plot_layers'), dict):
                self.plot_layers = {
                    k: bool(config['plot_layers'].get(k, DEFAULT_PLOT_LAYERS[k]))
                    for k in DEFAULT_PLOT_LAYERS}

            if 'proximity_detection' in config:
                self.chk_proximity_detection.setChecked(config['proximity_detection'])

            if 'proximity_threshold' in config:
                self.spin_proximity_threshold.setValue(config['proximity_threshold'])
            
            if 'save_plots' in config:
                self.chk_save_plots.setChecked(config['save_plots'])

            if 'downsample_plots' in config:
                self.chk_downsample_plots.setChecked(config['downsample_plots'])
            if 'plot_downsample_hz' in config:
                self.spin_plot_downsample_hz.setValue(config['plot_downsample_hz'])
            if 'downsample_animation' in config:
                self.chk_downsample_animation.setChecked(config['downsample_animation'])
            if 'anim_downsample_hz' in config:
                self.spin_anim_downsample_hz.setValue(config['anim_downsample_hz'])

            if 'save_svg' in config:
                self.chk_save_svg.setChecked(config['save_svg'])

            if 'plot_types' in config:
                for key, value in config['plot_types'].items():
                    if key in self.plot_type_checkboxes:
                        self.plot_type_checkboxes[key].setChecked(value)
            
            if 'save_animation' in config:
                self.chk_save_animation.setChecked(config['save_animation'])
            
            if 'animation_trail' in config:
                self.spin_animation_trail.setValue(config['animation_trail'])
            
            if 'animation_speed' in config:
                index = self.combo_animation_speed.findText(config['animation_speed'])
                if index >= 0:
                    self.combo_animation_speed.setCurrentIndex(index)
            
            if 'animation_fps' in config:
                index = self.combo_animation_fps.findText(str(config['animation_fps']))
                if index >= 0:
                    self.combo_animation_fps.setCurrentIndex(index)
            
            if 'color_by' in config:
                index = self.combo_color_by.findText(config['color_by'])
                if index >= 0:
                    self.combo_color_by.setCurrentIndex(index)

            if 'video_quality' in config:
                index = self.combo_video_quality.findText(config['video_quality'])
                if index >= 0:
                    self.combo_video_quality.setCurrentIndex(index)

            if 'daily_animations' in config:
                self.chk_daily_animations.setChecked(config['daily_animations'])

            if 'show_anchors' in config:
                self.chk_show_anchors.setChecked(config['show_anchors'])

            if 'tag_identities' in config:
                # Convert string keys back to integers if needed
                self.tag_identities = {}
                for key, value in config['tag_identities'].items():
                    tag_key = int(key) if isinstance(key, str) and key.isdigit() else key
                    self.tag_identities[tag_key] = value
            
            # Load background image if path is saved and file exists
            if 'background_image_path' in config and config['background_image_path']:
                bg_path = config['background_image_path']
                # Ensure XML config is parsed first to get scale
                if not self.xml_scale and self.db_path:
                    db_dir = os.path.dirname(self.db_path)
                    xml_files = [f for f in list_visible_files(db_dir) if f.lower().endswith('.xml')]
                    if xml_files:
                        xml_file = next((f for f in xml_files if 'config' in f.lower()), xml_files[0])
                        self.xml_config_path = os.path.join(db_dir, xml_file)
                        try:
                            self.parse_xml_config()
                        except:
                            pass
                
                # Resolve the saved path (absolute, else relative to the DB dir),
                # then load through the shared helper so scale/offset seeding and
                # the alignment log stay consistent with an interactive load.
                resolved = None
                if os.path.exists(bg_path):
                    resolved = bg_path
                elif os.path.exists(os.path.join(db_dir, os.path.basename(bg_path))):
                    resolved = os.path.join(db_dir, os.path.basename(bg_path))

                if resolved and self._apply_background_image(resolved, source="restored"):
                    # Honour a saved manual transform (overrides the XML default).
                    saved_scale = config.get('background_scale_inpx')
                    saved_off = config.get('background_offset_m')
                    if saved_scale:
                        self.bg_scale = float(saved_scale)
                    if saved_off and len(saved_off) == 2:
                        self.bg_offset_x = float(saved_off[0])
                        self.bg_offset_y = float(saved_off[1])
                    if saved_scale or saved_off:
                        self._recompute_bg_size()
                        self._sync_bg_transform_controls()
                        self.log_message(
                            f"Restored background transform: scale {self.bg_scale:.4f} in/px, "
                            f"offset ({self.bg_offset_x:.2f}, {self.bg_offset_y:.2f}) m")
                elif not resolved:
                    self.log_message(f"Warning: Saved background image not found: {bg_path}")
            
            # Note: selected_tags will be loaded after tags are populated from table
            if 'selected_tags' in config:
                self.pending_tag_selection = config['selected_tags']
            
            # Load zone data if present
            if 'arena_zones' in config and config['arena_zones'] is not None:
                self.arena_zones = pd.DataFrame(config['arena_zones'])
                if not self.arena_zones.empty:
                    num_zones = self.arena_zones['zone'].nunique()
                    num_points = len(self.arena_zones)
                    self.log_message(f"Loaded {num_zones} zones with {num_points} coordinate points from config")
            
            self.log_message(f"Loaded previous configuration from {config_path}")

            # Update tag labels if identities were loaded
            if self.tag_identities and self.tag_checkboxes:
                self.update_tag_labels()

            # --- Migration prompt: move loose files into subfolders ---
            plots_subdir = os.path.join(analysis_dir, 'plots')
            animations_subdir = os.path.join(analysis_dir, 'animations')

            loose_plots = []
            loose_animations = []

            if not os.path.exists(plots_subdir):
                loose_plots = [f for f in list_visible_files(analysis_dir)
                               if os.path.isfile(os.path.join(analysis_dir, f))
                               and f.lower().endswith(('.png', '.svg'))]

            if not os.path.exists(animations_subdir):
                loose_animations = [f for f in list_visible_files(analysis_dir)
                                    if os.path.isfile(os.path.join(analysis_dir, f))
                                    and f.lower().endswith('.mp4')]

            if loose_plots or loose_animations:
                parts = []
                if loose_plots:
                    parts.append(f"{len(loose_plots)} plot file(s)")
                if loose_animations:
                    parts.append(f"{len(loose_animations)} animation file(s)")
                file_desc = " and ".join(parts)

                reply = QMessageBox.question(
                    self, "Update Folder Structure",
                    f"Your analysis folder has {file_desc} in the root directory. "
                    f"Would you like to organize them into subfolders (plots/ and animations/) "
                    f"for better organization?",
                    QMessageBox.Yes | QMessageBox.No, QMessageBox.Yes
                )
                if reply == QMessageBox.Yes:
                    moved = 0
                    if loose_plots:
                        os.makedirs(plots_subdir, exist_ok=True)
                        for plot_file in loose_plots:
                            src = os.path.join(analysis_dir, plot_file)
                            dst = os.path.join(plots_subdir, plot_file)
                            try:
                                shutil.move(src, dst)
                                moved += 1
                            except Exception as move_err:
                                self.log_message(f"Warning: Could not move {plot_file}: {move_err}")
                    if loose_animations:
                        os.makedirs(animations_subdir, exist_ok=True)
                        for anim_file in loose_animations:
                            src = os.path.join(analysis_dir, anim_file)
                            dst = os.path.join(animations_subdir, anim_file)
                            try:
                                shutil.move(src, dst)
                                moved += 1
                            except Exception as move_err:
                                self.log_message(f"Warning: Could not move {anim_file}: {move_err}")
                    self.log_message(f"Migrated {moved} file(s) into subfolders")

            return True

        except Exception as e:
            print(f"Warning: Could not load config: {str(e)}")
            return False
    
    def apply_pending_tag_selection(self):
        """Apply tag selection from loaded config after tags are populated"""
        if hasattr(self, 'pending_tag_selection') and self.pending_tag_selection:
            for tag, cb in self.tag_checkboxes.items():
                cb.setChecked(tag in self.pending_tag_selection)
            delattr(self, 'pending_tag_selection')
    
    def select_animation_tags(self):
        """Pick which tags the animation includes.

        The list is a sub-selection of the tags currently checked in the
        top-level Tag Selection — you cannot add tags here that weren't chosen
        there. Default is all of them.
        """
        from PyQt5.QtWidgets import QDialog, QDialogButtonBox
        tags = self.selected_preview_tags()
        if not tags:
            QMessageBox.information(
                self, "No Tags Selected",
                "Select one or more tags in the Tag Selection section first — "
                "the animation can only include tags chosen there.")
            return
        dlg = QDialog(self)
        dlg.setWindowTitle("Include Tags in Animation")
        lay = QVBoxLayout()
        lay.addWidget(QLabel("Select the tags to include in the animation:"))
        boxes = {}
        for t in tags:
            info = self.tag_identities.get(t, {}) or {}
            hex_id = hex(int(t)).upper().replace('0X', '')
            label = (f"{info['sex']}-{info['identity']}  (HexID {hex_id})"
                     if info else f"HexID {hex_id}")
            cb = QCheckBox(label)
            cb.setChecked(self._animation_tags is None or t in self._animation_tags)
            boxes[t] = cb
            lay.addWidget(cb)

        sel_row = QHBoxLayout()
        btn_all = QPushButton("Select All")
        btn_none = QPushButton("Unselect All")
        btn_all.clicked.connect(lambda: [cb.setChecked(True) for cb in boxes.values()])
        btn_none.clicked.connect(lambda: [cb.setChecked(False) for cb in boxes.values()])
        sel_row.addWidget(btn_all)
        sel_row.addWidget(btn_none)
        lay.addLayout(sel_row)

        btns = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        btns.accepted.connect(dlg.accept)
        btns.rejected.connect(dlg.reject)
        lay.addWidget(btns)
        dlg.setLayout(lay)
        if dlg.exec_() == QDialog.Accepted:
            sel = [t for t, cb in boxes.items() if cb.isChecked()]
            if not sel or len(sel) == len(tags):
                self._animation_tags = None
                self.lbl_anim_tags.setText("All configured tags")
            else:
                self._animation_tags = sel
                self.lbl_anim_tags.setText(f"{len(sel)} of {len(tags)} tags")

    def generate_animation(self, output_dir, total_export_steps=1, current_export_step=1, csv_path=None, animations_dir=None):
        """Generate animation video from tracking data"""
        try:
            if self.export_cancelled:
                return
            
            self.log_message("Preparing animation data...")

            # Resolve animations output directory
            if animations_dir is None:
                animations_dir = os.path.join(output_dir, 'animations')
            os.makedirs(animations_dir, exist_ok=True)

            # Ask user to confirm temp frames location
            default_temp_dir = os.path.join(animations_dir, 'temp_frames')
            msg = QMessageBox(self)
            msg.setWindowTitle("Temporary Frames Location")
            msg.setText(
                f"Animation rendering requires temporary frame files.\n\n"
                f"Default location:\n{default_temp_dir}\n\n"
                f"These files will be deleted after the animation is complete. "
                f"Use the default location, or choose a different folder?"
            )
            use_default_btn = msg.addButton("Use Default", QMessageBox.AcceptRole)
            choose_btn = msg.addButton("Choose Folder...", QMessageBox.ActionRole)
            cancel_btn = msg.addButton("Cancel", QMessageBox.RejectRole)
            msg.exec_()

            clicked = msg.clickedButton()
            if clicked == cancel_btn:
                self.log_message("Animation cancelled by user")
                self.exporting = False
                self.btn_export.setEnabled(True)
                self.btn_stop_export.setVisible(False)
                self.progress_widget.setVisible(False)
                return
            elif clicked == choose_btn:
                custom_dir = QFileDialog.getExistingDirectory(
                    self, "Select Folder for Temporary Animation Frames",
                    os.path.expanduser("~")
                )
                if not custom_dir:
                    self.log_message("Animation cancelled by user")
                    self.exporting = False
                    self.btn_export.setEnabled(True)
                    self.btn_stop_export.setVisible(False)
                    self.progress_widget.setVisible(False)
                    return
                temp_frames_dir = os.path.join(custom_dir, 'temp_animation_frames')
            else:
                temp_frames_dir = default_temp_dir

            os.makedirs(temp_frames_dir, exist_ok=True)
            self.log_message(f"Temp frames location: {temp_frames_dir}")
            
            # Load data from CSV if provided (for consistency with plots)
            if csv_path and os.path.exists(csv_path):
                self.log_message(f"Loading animation data from CSV...")
                anim_data = pd.read_csv(csv_path, low_memory=False)
                # Parse Timestamp column (with mixed format to handle timezone-aware timestamps)
                anim_data['Timestamp'] = pd.to_datetime(anim_data['Timestamp'], format='mixed')
                self.log_message(f"Loaded {len(anim_data)} records from CSV")
            else:
                # Fallback to using self.data
                self.log_message("Using in-memory data for animation...")
                anim_data = self.data.copy()
            
            # Report the coordinate basis, then normalise columns (ID/sex/
            # identities/smoothed-coord fallback) via the shared helper.
            self.log_message(
                "Animation trajectories use "
                f"{'smoothed' if 'smoothed_x' in anim_data.columns else 'raw (unsmoothed)'} coordinates")
            anim_data = uwb_animation.prepare_animation_data(anim_data, self.tag_identities)

            # Restrict to the chosen subset of tags (default: all).
            if self._animation_tags:
                anim_data = anim_data[anim_data['shortid'].isin(self._animation_tags)]
                self.log_message(f"Animation limited to {len(self._animation_tags)} selected tag(s)")

            # Get animation parameters
            trailing_window = self.spin_animation_trail.value()
            fps = int(self.combo_animation_fps.currentText())
            speed_text = self.combo_animation_speed.currentText()
            speed_multiplier = int(speed_text.replace('x', ''))
            color_by = self.combo_color_by.currentText()

            frame_interval = uwb_animation.frame_interval_seconds(speed_multiplier, fps)
            self.log_message(f"Animation: {speed_text} speed at {fps} FPS (each frame = {frame_interval:.2f}s of real time)")
            
            self.log_message("Setting up animation frames...")
            
            # Check if daily animations are requested
            generate_daily = self.chk_daily_animations.isChecked()
            
            if generate_daily:
                # Get selected days
                selected_days = [date_str for date_str, cb in self.daily_animation_day_checkboxes.items() if cb.isChecked()]
                if not selected_days:
                    self.log_message("⚠ No days selected for daily animations")
                    return
                
                self.log_message(f"Generating {len(selected_days)} daily animations...")
                
                # Generate one animation per selected day
                for day_idx, date_str in enumerate(selected_days):
                    if self.export_cancelled:
                        return
                    
                    self.log_message(f"Processing Day {day_idx + 1}/{len(selected_days)}: {date_str}")
                    
                    # Filter data for this specific day (midnight to midnight)
                    # Get timezone from the data to ensure compatible comparison
                    data_tz = anim_data['Timestamp'].dt.tz
                    date_obj = pd.to_datetime(date_str).date()
                    
                    if data_tz is not None:
                        # Create timezone-aware timestamps matching the data's timezone
                        day_start = pd.Timestamp(date_obj, tz=data_tz)
                        day_end = day_start + pd.Timedelta(days=1)
                    else:
                        # Data is timezone-naive
                        day_start = pd.Timestamp(date_obj)
                        day_end = day_start + pd.Timedelta(days=1)
                    
                    day_data = anim_data[(anim_data['Timestamp'] >= day_start) & (anim_data['Timestamp'] < day_end)].copy()
                    
                    if len(day_data) == 0:
                        self.log_message(f"⚠ No data for {date_str}, skipping")
                        continue
                    
                    # Get speed text for display
                    speed_text = self.combo_animation_speed.currentText()
                    
                    # Generate animation for this day
                    video_path = self.create_animation_frames(
                        day_data, temp_frames_dir, frame_interval, trailing_window,
                        fps, color_by, bool(self.tag_identities),
                        total_export_steps, current_export_step,
                        day_suffix=f"_Day{day_idx + 1}_{date_str}",
                        speed_text=speed_text
                    )
                    
                    if video_path and not self.export_cancelled:
                        # Move video to final location with FPS and speed in filename
                        db_filename = os.path.basename(self.db_path)
                        db_name = os.path.splitext(db_filename)[0]
                        speed_text = self.combo_animation_speed.currentText()
                        final_video_path = os.path.join(animations_dir, f"{db_name}_Animation_Day{day_idx + 1}_{date_str}_{fps}fps_{speed_text}.mp4")
                        
                        # Check if file exists and skip if not overwriting
                        if os.path.exists(final_video_path):
                            self.log_message(f"⚠ Day {day_idx + 1} animation already exists, skipping: {os.path.basename(final_video_path)}")
                        elif os.path.exists(video_path):
                            shutil.move(video_path, final_video_path)
                            self.log_message(f"✓ Day {day_idx + 1} animation saved: {final_video_path}")
                
                # Clean up temp frames
                try:
                    shutil.rmtree(temp_frames_dir)
                    self.log_message("✓ Temp frames cleaned up")
                except Exception as e:
                    self.log_message(f"Warning: Could not clean temp frames: {str(e)}")
            else:
                # Generate single animation for all data
                # Clean existing temp frames
                if os.path.exists(temp_frames_dir):
                    for filename in os.listdir(temp_frames_dir):
                        file_path = os.path.join(temp_frames_dir, filename)
                        if os.path.isfile(file_path):
                            os.remove(file_path)
                
                # Create animation
                self.log_message("Generating animation frames (this may take a while)...")
                speed_text = self.combo_animation_speed.currentText()
                video_path = self.create_animation_frames(anim_data, temp_frames_dir, frame_interval, trailing_window, 
                                            fps, color_by, bool(self.tag_identities),
                                            total_export_steps, current_export_step,
                                            speed_text=speed_text)
                
                if video_path and not self.export_cancelled:
                    # Move video to final location with FPS and speed in filename
                    db_filename = os.path.basename(self.db_path)
                    db_name = os.path.splitext(db_filename)[0]
                    speed_text = self.combo_animation_speed.currentText()
                    final_video_path = os.path.join(animations_dir, f"{db_name}_Animation_{fps}fps_{speed_text}.mp4")
                    
                    # Check if file exists and skip if not overwriting
                    if os.path.exists(final_video_path):
                        self.log_message(f"⚠ Animation already exists, skipping: {os.path.basename(final_video_path)}")
                    elif os.path.exists(video_path):
                        shutil.move(video_path, final_video_path)
                        self.log_message(f"✓ Animation saved: {final_video_path}")
                    
                    # Clean up temp frames
                    try:
                        shutil.rmtree(temp_frames_dir)
                        self.log_message("✓ Temp frames cleaned up")
                    except Exception as e:
                        self.log_message(f"Warning: Could not clean temp frames: {str(e)}")
            
            self.log_message("✓ Animation generation complete!")
            
            # Reset UI after animation completes
            self.exporting = False
            self.btn_export.setEnabled(True)
            self.btn_stop_export.setVisible(False)
            self.progress_bar.setValue(100)
            self.lbl_export_progress.setText("All exports complete!")
            QMessageBox.information(self, "Success", "All exports completed successfully!")
            QTimer.singleShot(3000, lambda: self.progress_widget.setVisible(False))
            
        except Exception as e:
            QMessageBox.critical(self, "Animation Error", f"Failed to generate animation: {str(e)}")
            self.log_message(f"✗ Animation generation failed: {str(e)}")
        finally:
            # Animation is the last render stage; drop the temp render CSVs.
            self._cleanup_plot_working_files()

    def create_animation_frames(self, data, output_dir, frame_interval, trailing_window,
                               fps, color_by, use_custom_identities=False,
                               total_export_steps=1, current_export_step=1, day_suffix="", speed_text=""):
        """Render one animation video via the shared ``uwb_animation`` core.

        This is a thin adapter: it gathers the GUI-held settings (quality/DPI,
        layer choice, background/zones/anchors, identities) and wires the tool's
        progress bar, cancellation flag and logger into the pure renderer.
        """
        dpi = uwb_animation.QUALITY_DPI.get(self.combo_video_quality.currentText(), 100)
        self.log_message(f"Using {dpi} DPI for video generation")

        bg_image, bg_extent = self._context_bg_source()
        video_filename = f'animation_temp{day_suffix}.mp4' if day_suffix else 'animation_temp.mp4'
        video_output_path = os.path.join(output_dir, video_filename)

        def _progress(i, total):
            if i % 10 == 0 or i == total - 1:
                pct = int(((current_export_step - 1) + (i + 1) / max(1, total)) / max(1, total_export_steps) * 100)
                self.progress_bar.setValue(pct)
                self.lbl_export_progress.setText(
                    f"Step {current_export_step}/{total_export_steps}: Rendering frame {i+1}/{total}...")
                if i % 50 == 0:
                    self.log_message(f"Rendering frame {i+1}/{total}...")
                QApplication.processEvents()

        return uwb_animation.render_animation(
            data, video_output_path,
            frame_interval=frame_interval, trailing_window=trailing_window, fps=fps,
            dpi=dpi, speed_text=speed_text,
            layers=self.plot_layers, bg_image=bg_image, bg_extent=bg_extent,
            arena_zones=self.arena_zones, anchors=self.anchor_positions,
            tag_identities=self.tag_identities, use_custom_identities=use_custom_identities,
            color_by=color_by, marker_size=self.spin_anim_tag_size.value(),
            show_battery=self.chk_show_battery_export.isChecked(),
            is_cancelled=lambda: self.export_cancelled,
            progress=_progress, log=self.log_message,
        )
    
    def stop_export(self):
        """Cancel ongoing export operations"""
        self.export_cancelled = True
        self.log_message("⚠ Export cancellation requested...")
        
        # Stop worker thread if running
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait()
            self.log_message("✗ Plot export cancelled")

        # Drop any temp render CSVs left behind by the cancelled run.
        self._cleanup_plot_working_files()

        # Reset UI
        self.exporting = False
        self.btn_export.setEnabled(True)
        self.btn_stop_export.setVisible(False)
        self.progress_widget.setVisible(False)
        self.progress_bar.setValue(0)
        self.lbl_export_progress.setText("")
    
    def _cleanup_plot_working_files(self):
        """Delete the temporary downsampled/raw render CSVs written for this run.

        These '<db>_plotdata_*.csv' files exist only so the plot worker and the
        animation have a source to read; they are removed once those consumers
        finish. The exported smoothed CSV deliverable is never in this list.
        """
        for path in getattr(self, '_plot_working_files', []):
            try:
                if path and os.path.exists(path):
                    os.remove(path)
                    self.log_message(f"Removed temporary render file: {os.path.basename(path)}")
            except OSError as e:
                self.log_message(f"Could not remove temp render file {os.path.basename(path)}: {e}")
        self._plot_working_files = []

    def export_data(self):
        """Export data and/or plots based on selected options"""
        if not self.db_path:
            QMessageBox.warning(self, "No Database", "Please select a database first")
            return

        # Note: self.data can be None if this is a fresh export
        # This is OK - plots and animations will load data directly from CSV/database

        # Initialize export state
        self.export_cancelled = False
        self._plot_working_files = []  # temp render CSVs to delete when done
        self.exporting = True
        self.btn_export.setEnabled(False)
        self.btn_stop_export.setVisible(True)
        self.progress_widget.setVisible(True)
        self.progress_bar.setValue(0)
        
        # Gather export settings
        db_dir = os.path.dirname(self.db_path)
        db_filename = os.path.basename(self.db_path)
        db_name = os.path.splitext(db_filename)[0]  # Remove extension

        export_raw_csv = self.chk_export_raw_csv.isChecked()
        export_smoothed_csv = self.chk_export_smoothed_csv.isChecked()
        save_plots = self.chk_save_plots.isChecked()
        save_animation = self.chk_save_animation.isChecked()
        downsample_plots = self.chk_downsample_plots.isChecked()
        plot_downsample_hz = self.spin_plot_downsample_hz.value()
        downsample_animation = self.chk_downsample_animation.isChecked()
        anim_downsample_hz = self.spin_anim_downsample_hz.value()

        detect_proximity = self.chk_proximity_detection.isChecked()
        proximity_threshold = self.spin_proximity_threshold.value()

        if not export_raw_csv and not export_smoothed_csv and not save_plots and not save_animation and not detect_proximity:
            QMessageBox.warning(self, "No Export Selected", "Please select at least one export option (CSV, Plots, or Animation)")
            return

        # --- Spatial layer choice for plots/animation ---
        # Prompt for the context layers when there is spatial output to make.
        # Cancelling the dialog aborts the whole export.
        if save_plots or save_animation:
            if not self._prompt_plot_layers():
                self.log_message("Export cancelled at layer selection.")
                return

        # --- Conflict detection ---
        base_output_dir = os.path.join(db_dir, f"{db_name}_FNT_analysis")
        plots_subdir = os.path.join(base_output_dir, 'plots')
        animations_subdir = os.path.join(base_output_dir, 'animations')

        # --- Build helper to get tag file suffix (matches PlotSaverWorker logic) ---
        def _tag_suffix(tag):
            if self.tag_identities and tag in self.tag_identities:
                info = self.tag_identities[tag]
                sex = info.get('sex', 'M')
                identity = info.get('identity', str(tag))
                return f"{sex}-{identity}"
            else:
                hex_id = hex(tag).upper().replace('0X', '')
                return f"HexID{hex_id}"

        save_svg = self.chk_save_svg.isChecked()

        def _add_with_svg(lst, basename):
            """Append a .png filename and optionally the .svg counterpart."""
            lst.append(basename + '.png')
            if save_svg:
                lst.append(basename + '.svg')

        # Predict which files will be produced (root-level CSVs and behavior files)
        predicted_files = []  # filenames in root output_dir
        if export_raw_csv:
            predicted_files.append(f'{db_name}_raw.csv')
        if export_smoothed_csv:
            predicted_files.append(f'{db_name}_smoothed.csv')
        if detect_proximity:
            predicted_files.append(f'{db_name}_proximity_bouts.csv')

        # Predict animation files (in animations/ subfolder)
        predicted_animation_files = []
        if save_animation:
            fps = int(self.combo_animation_fps.currentText())
            speed_text = self.combo_animation_speed.currentText()
            if self.chk_daily_animations.isChecked():
                selected_days = [d for d, cb in self.daily_animation_day_checkboxes.items() if cb.isChecked()]
                for day_idx, date_str in enumerate(selected_days):
                    predicted_animation_files.append(f'{db_name}_Animation_Day{day_idx + 1}_{date_str}_{fps}fps_{speed_text}.mp4')
            else:
                predicted_animation_files.append(f'{db_name}_Animation_{fps}fps_{speed_text}.mp4')

        # Predict plot files (in plots/ subfolder) — must match PlotSaverWorker filenames exactly
        predicted_plot_files = []
        if save_plots:
            selected_tags = [tag for tag, cb in self.tag_checkboxes.items() if cb.isChecked()]
            plot_types = {k: self.plot_type_checkboxes[k].isChecked() for k in self.plot_type_checkboxes}

            if plot_types.get('daily_paths', False):
                for tag in selected_tags:
                    _add_with_svg(predicted_plot_files, f'{db_name}_DailyPaths_{_tag_suffix(tag)}')
            if plot_types.get('trajectory_overview', False):
                _add_with_svg(predicted_plot_files, f'{db_name}_TrajectoryOverview')
            if plot_types.get('battery_levels', False):
                _add_with_svg(predicted_plot_files, f'{db_name}_BatteryLevels')
            if plot_types.get('3d_occupancy', False):
                for tag in selected_tags:
                    _add_with_svg(predicted_plot_files, f'{db_name}_3D_Occupancy_{_tag_suffix(tag)}')
            if plot_types.get('activity_timeline', False):
                _add_with_svg(predicted_plot_files, f'{db_name}_ActivityTimeline')
            if plot_types.get('velocity_distribution', False):
                _add_with_svg(predicted_plot_files, f'{db_name}_VelocityDistribution')
            if plot_types.get('cumulative_distance', False):
                _add_with_svg(predicted_plot_files, f'{db_name}_CumulativeDistance')
            if plot_types.get('velocity_timeline', False):
                for tag in selected_tags:
                    _add_with_svg(predicted_plot_files, f'{db_name}_VelocityTimeline_{_tag_suffix(tag)}')
            if plot_types.get('actogram', False):
                for tag in selected_tags:
                    _add_with_svg(predicted_plot_files, f'{db_name}_Actogram_{_tag_suffix(tag)}')
            if plot_types.get('data_quality', False):
                _add_with_svg(predicted_plot_files, f'{db_name}_DataQuality')

        # Check for conflicts against existing files and build categorized lists
        # Each entry is (filename, subfolder_label) — subfolder_label is "" for root
        all_conflicting = []
        all_new = []

        for f in predicted_files:
            if os.path.exists(os.path.join(base_output_dir, f)):
                all_conflicting.append((f, ""))
            else:
                all_new.append((f, ""))

        for f in predicted_plot_files:
            if os.path.exists(os.path.join(plots_subdir, f)):
                all_conflicting.append((f, "plots"))
            else:
                all_new.append((f, "plots"))

        for f in predicted_animation_files:
            if os.path.exists(os.path.join(animations_subdir, f)):
                all_conflicting.append((f, "animations"))
            else:
                all_new.append((f, "animations"))

        skip_existing = False
        output_dir = base_output_dir

        if all_conflicting:
            dialog = ExportConflictDialog(all_conflicting, all_new, parent=self)
            result = dialog.exec_()

            if result == ExportConflictDialog.SKIP:
                skip_existing = True
                output_dir = base_output_dir
            elif result == ExportConflictDialog.OVERWRITE:
                skip_existing = False
                output_dir = base_output_dir
                # Delete conflicting files so they get cleanly rewritten
                for fname, subfolder in all_conflicting:
                    if subfolder:
                        fpath = os.path.join(base_output_dir, subfolder, fname)
                    else:
                        fpath = os.path.join(base_output_dir, fname)
                    try:
                        os.remove(fpath)
                    except Exception:
                        pass
            elif result == ExportConflictDialog.NEW_FOLDER:
                from datetime import datetime
                timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
                output_dir = os.path.join(db_dir, f"{db_name}_FNT_analysis_{timestamp}")
                skip_existing = False
            else:
                # User cancelled
                self.exporting = False
                self.btn_export.setEnabled(True)
                self.btn_stop_export.setVisible(False)
                self.progress_widget.setVisible(False)
                return

        # Create output directory and subfolders only as needed
        os.makedirs(output_dir, exist_ok=True)
        plots_dir = os.path.join(output_dir, 'plots')
        if save_plots:
            os.makedirs(plots_dir, exist_ok=True)
        animations_dir = os.path.join(output_dir, 'animations')
        if save_animation:
            os.makedirs(animations_dir, exist_ok=True)

        try:
            # Busy cursor for the whole synchronous prep. The heavy pandas /
            # sqlite calls below run on the GUI thread and cannot be interrupted
            # mid-call, so the window can still briefly stop repainting; the
            # cursor plus the finer-grained processEvents() calls keep it from
            # looking dead. Restored in the finally so every exit path clears it.
            QApplication.setOverrideCursor(Qt.WaitCursor)
            self.log_message(f"Starting export to {output_dir}")
            self.lbl_export_progress.setText("Initializing export...")

            # Calculate total steps for progress
            total_steps = 0
            if export_raw_csv:
                total_steps += 1
            if export_smoothed_csv:
                total_steps += 1
            if save_plots:
                total_steps += 1
            if save_animation:
                total_steps += 1

            current_step = 0

            # Note: Conflict resolution (skip/overwrite/new folder) was handled above

            # Export raw CSV (unprocessed database dump)
            if export_raw_csv:
                if self.export_cancelled:
                    self.stop_export()
                    return

                current_step += 1
                self.lbl_export_progress.setText(f"Step {current_step}/{total_steps}: Exporting raw CSV...")
                self.progress_bar.setValue(int(current_step / total_steps * 100))
                QApplication.processEvents()

                raw_csv_filename = f'{db_name}_raw.csv'
                raw_csv_path = os.path.join(output_dir, raw_csv_filename)
                if skip_existing and os.path.exists(raw_csv_path):
                    self.log_message(f"Skipped (exists): {raw_csv_filename}")
                else:
                    self.log_message("Exporting raw database to CSV...")
                    QApplication.processEvents()
                    conn = sqlite3.connect(self.db_path)
                    raw_data = pd.read_sql_query(f"SELECT * FROM {self.table_name}", conn)
                    conn.close()
                    raw_data.to_csv(raw_csv_path, index=False)
                    self.log_message(f"✓ Raw CSV exported: {raw_csv_filename}")
                    QApplication.processEvents()

            # Prepare processed data (needed for smoothed CSV, plots, animation, behaviors)
            needs_processed_data = export_smoothed_csv or save_plots or save_animation or detect_proximity
            smoothed_data = None
            csv_path = None  # Path to the CSV that plots/animation will use

            if needs_processed_data:
                if self.export_cancelled:
                    self.stop_export()
                    return

                self.log_message("Preparing processed data (per-tag chunked processing)...")

                selected_tags = [tag for tag, cb in self.tag_checkboxes.items() if cb.isChecked()]
                tz = pytz.timezone(self.combo_timezone.currentText())
                smoothing_method = self.get_smoothing_method()
                do_filter = self.chk_velocity_filter.isChecked() or self.chk_jump_filter.isChecked()

                # Filter statistics accumulate across tags; without this reset
                # the run summary would report whatever the last call left
                # behind (a preview chunk, or a previous export).
                self.reset_filter_stats()

                processed_chunks = []
                conn = sqlite3.connect(self.db_path)

                # Read only the columns the pipeline uses. The unused TEXT
                # columns dominate memory, so this is the difference between
                # ~390 MB and ~37 MB of transient DataFrame per tag.
                proc_cols = processing_select_clause(conn, self.table_name)
                if proc_cols != "*":
                    self.log_message(
                        f"Reading columns [{proc_cols}] for processing; "
                        f"the raw CSV still contains every database column.")

                for i, tag in enumerate(selected_tags):
                    if self.export_cancelled:
                        conn.close()
                        self.stop_export()
                        return

                    hex_id = hex(tag).upper().replace('0X', '')
                    self.log_message(f"  Processing tag {hex_id} ({i+1}/{len(selected_tags)})...")
                    QApplication.processEvents()

                    tag_data = pd.read_sql_query(
                        f"SELECT {proc_cols} FROM {self.table_name} WHERE shortid = ?",
                        conn, params=(tag,))

                    if len(tag_data) == 0:
                        continue

                    tag_data['Timestamp'] = pd.to_datetime(tag_data['timestamp'], unit='ms', origin='unix', utc=True)
                    tag_data['Timestamp'] = tag_data['Timestamp'].dt.tz_convert(tz)
                    tag_data['location_x'] *= 0.0254
                    tag_data['location_y'] *= 0.0254
                    tag_data = tag_data.sort_values(by='Timestamp')

                    # Per-tag time trimming
                    if tag in self.tag_identities:
                        info = self.tag_identities[tag]
                        if 'start_time' in info and 'stop_time' in info:
                            start = pd.Timestamp(info['start_time']).tz_localize(tz)
                            stop = pd.Timestamp(info['stop_time']).tz_localize(tz)
                            before = len(tag_data)
                            tag_data = tag_data[(tag_data['Timestamp'] >= start) & (tag_data['Timestamp'] <= stop)]
                            trimmed = before - len(tag_data)
                            if trimmed > 0:
                                self.log_message(f"    Trimmed {trimmed} points outside time window")

                    # Filtering
                    if do_filter and len(tag_data) > 0:
                        tag_data = self.apply_filters_to_data(tag_data)

                    # Smoothing
                    if smoothing_method != "None" and len(tag_data) > 0:
                        tag_data = self.apply_smoothing_to_data(tag_data, smoothing_method)

                    # Let the window repaint / process the Stop button between
                    # tags, right after the heaviest per-tag work.
                    QApplication.processEvents()

                    # Identity mapping
                    if tag in self.tag_identities:
                        tag_data['sex'] = self.tag_identities[tag].get('sex', 'M')
                        tag_data['identity'] = self.tag_identities[tag].get('identity', f'Tag{tag}')
                    else:
                        tag_data['sex'] = 'M'
                        tag_data['identity'] = f'Tag{tag}'

                    processed_chunks.append(tag_data)
                    self.log_message(f"    {len(tag_data)} points after processing")

                conn.close()

                if processed_chunks:
                    smoothed_data = pd.concat(processed_chunks, ignore_index=True)
                    smoothed_data = smoothed_data.sort_values(by=['shortid', 'Timestamp'])
                    self.log_message(f"Total processed: {len(smoothed_data)} points across {len(processed_chunks)} tags")
                    QApplication.processEvents()
                else:
                    self.log_message("Warning: No data after processing")
                    smoothed_data = pd.DataFrame()

                # Export smoothed CSV
                if export_smoothed_csv:
                    current_step += 1
                    self.lbl_export_progress.setText(f"Step {current_step}/{total_steps}: Exporting smoothed CSV...")
                    self.progress_bar.setValue(int(current_step / total_steps * 100))
                    QApplication.processEvents()

                    smoothed_csv_filename = f'{db_name}_smoothed.csv'
                    smoothed_csv_path = os.path.join(output_dir, smoothed_csv_filename)
                    if skip_existing and os.path.exists(smoothed_csv_path):
                        self.log_message(f"Skipped (exists): {smoothed_csv_filename}")
                        csv_path = smoothed_csv_path  # Still use existing for plots/animation
                    else:
                        smoothed_data.to_csv(smoothed_csv_path, index=False)
                        self.log_message(f"✓ Smoothed CSV exported: {smoothed_csv_filename}")
                        csv_path = smoothed_csv_path  # Use smoothed for plots/animation

                # Render sources for plots and animation. Each is independent of
                # the full-resolution smoothed CSV deliverable: it can be
                # downsampled (a big render-cost saver, chosen per-consumer) and
                # holds the smoothed track if the smoothed CSV was written, else
                # the raw (unsmoothed) fixes. (The raw CSV export is a verbatim
                # DB dump in inches with no Timestamp column, so it can't be a
                # plotting source directly.) Temporary working files are tracked
                # and deleted once the renderers finish.
                render_source_cache = {}

                def _build_render_source(downsample, hz):
                    key = (downsample, hz if downsample else None)
                    if key in render_source_cache:
                        return render_source_cache[key]
                    if export_smoothed_csv:
                        base = smoothed_data
                        kind = 'smoothed'
                    else:
                        base = smoothed_data.drop(
                            columns=['smoothed_x', 'smoothed_y'], errors='ignore')
                        kind = 'raw (unsmoothed)'

                    if downsample and not base.empty:
                        # Thin to the target rate: first fix per 1/Hz-second bin
                        # per tag (resolution-safe floor binning).
                        work = base.copy()
                        period = pd.Timedelta(seconds=1.0 / hz)
                        work['time_bin'] = work['Timestamp'].dt.floor(period)
                        work = work.groupby(['shortid', 'time_bin']).first().reset_index()
                        work = work.drop(columns=['time_bin'])
                        fn = f'{db_name}_plotdata_{hz}Hz.csv'
                        path = os.path.join(output_dir, fn)
                        work.to_csv(path, index=False)
                        self._plot_working_files.append(path)
                        self.log_message(
                            f"Render source: {hz} Hz downsampled {kind} data "
                            f"({len(work)} of {len(base)} rows): {fn}")
                        result = path
                    elif export_smoothed_csv:
                        # Full resolution + smoothed CSV written: reuse the
                        # deliverable directly (not a temp file).
                        result = smoothed_csv_path
                    else:
                        # Full resolution, no smoothed CSV: working copy of the
                        # raw (unsmoothed) fixes for the renderers.
                        fn = f'{db_name}_plotdata_raw.csv'
                        path = os.path.join(output_dir, fn)
                        base.to_csv(path, index=False)
                        self._plot_working_files.append(path)
                        self.log_message(
                            f"Render source: raw (unsmoothed) full-resolution fixes: {fn}")
                        result = path
                    render_source_cache[key] = result
                    return result

                plot_csv_path = _build_render_source(downsample_plots, plot_downsample_hz) if save_plots else None
                anim_csv_path = _build_render_source(downsample_animation, anim_downsample_hz) if save_animation else None
            
            # Write the site-map image (if any) so the analysis folder is a
            # self-contained data product, then save the config that references it.
            self._export_sitemap(output_dir, db_name)
            self.save_config(output_dir)

            # Save message log and run summary
            self.save_message_log(output_dir)
            self.save_run_summary(output_dir)
            
            # Detect proximity bouts if requested
            if detect_proximity:
                if self.export_cancelled:
                    self.stop_export()
                    return

                self.log_message("=" * 50)
                self.log_message("Starting proximity detection...")
                self.lbl_export_progress.setText("Detecting proximity events and bouts...")
                QApplication.processEvents()

                try:
                    from fnt.uwb.proximity_detection import detect_proximity_bouts

                    # Warn if tag identities not configured
                    if not self.tag_identities:
                        self.log_message("Warning: Tag identities not configured. "
                                         "Proximity output will use raw shortid values. "
                                         "Configure identities in Tag Configuration for sex-ID labels.")

                    # Use the smoothed full-resolution data (not downsampled)
                    if smoothed_data is not None:
                        prox_data = smoothed_data.copy()
                    elif csv_path and os.path.exists(csv_path):
                        prox_data = pd.read_csv(csv_path, low_memory=False)
                        prox_data['Timestamp'] = pd.to_datetime(prox_data['Timestamp'], format='ISO8601')
                    else:
                        self.log_message("Error: No smoothed data available for proximity detection")
                        prox_data = None

                    if prox_data is not None:
                        # The per-timestamp events frame is computed internally
                        # (bouts are derived from it) but intentionally NOT
                        # written out: bouts is the tool's threshold-based
                        # deliverable. Anyone needing the raw per-second dyadic
                        # distances can re-run at a different threshold or
                        # compute them from the exported smoothed CSV themselves.
                        _events, proximity_bouts = detect_proximity_bouts(
                            prox_data,
                            threshold=proximity_threshold,
                            gap_s=5,
                            tag_identities=self.tag_identities,
                            log_callback=self.log_message
                        )

                        # Export proximity bouts
                        bouts_path = os.path.join(output_dir, f'{db_name}_proximity_bouts.csv')
                        if skip_existing and os.path.exists(bouts_path):
                            self.log_message(f"Skipped (exists): {os.path.basename(bouts_path)}")
                        else:
                            proximity_bouts.to_csv(bouts_path, index=False)
                            self.log_message(f"✓ Exported proximity bouts ({len(proximity_bouts)} bouts): "
                                             f"{os.path.basename(bouts_path)}")

                        self.log_message("✓ Proximity detection complete!")

                except Exception as e:
                    self.log_message(f"Error during proximity detection: {e}")
                    import traceback
                    traceback.print_exc()

                self.log_message("=" * 50)
            
            # Export plots if requested (BEFORE animation, which takes longer)
            if save_plots:
                if self.export_cancelled:
                    self.stop_export()
                    return
                
                current_step += 1
                self.lbl_export_progress.setText(f"Step {current_step}/{total_steps}: Generating plots...")
                self.progress_bar.setValue(int(current_step / total_steps * 100))
                QApplication.processEvents()
                
                selected_tags = [tag for tag, cb in self.tag_checkboxes.items() if cb.isChecked()]

                # Get selected plot types - include ALL plot types
                plot_types = {
                    'daily_paths': self.plot_type_checkboxes['daily_paths'].isChecked(),
                    'trajectory_overview': self.plot_type_checkboxes['trajectory_overview'].isChecked(),
                    'battery_levels': self.plot_type_checkboxes['battery_levels'].isChecked(),
                    '3d_occupancy': self.plot_type_checkboxes['3d_occupancy'].isChecked(),
                    'activity_timeline': self.plot_type_checkboxes['activity_timeline'].isChecked(),
                    'velocity_distribution': self.plot_type_checkboxes['velocity_distribution'].isChecked(),
                    'cumulative_distance': self.plot_type_checkboxes['cumulative_distance'].isChecked(),
                    'velocity_timeline': self.plot_type_checkboxes['velocity_timeline'].isChecked(),
                    'actogram': self.plot_type_checkboxes['actogram'].isChecked(),
                    'data_quality': self.plot_type_checkboxes['data_quality'].isChecked()
                }

                # Get rolling window value
                rolling_window = self.spin_rolling_window.value()

                self.btn_export.setEnabled(False)
                self.log_message("Starting plot generation in background...")

                self.worker = PlotSaverWorker(
                    self.db_path,
                    self.table_name,
                    selected_tags,
                    False,  # downsample: vestigial, feature removed
                    self.get_smoothing_method(),
                    plot_types,
                    skip_existing,
                    rolling_window,
                    self.combo_timezone.currentText(),
                    self.tag_identities,
                    bool(self.tag_identities),  # Use identities if any are configured
                    # Always pass the image; whether it is drawn is decided by the
                    # export layer choice (self.plot_layers), not the preview toggle.
                    self.background_image,
                    self.bg_width_meters,  # Pass scaled width
                    self.bg_height_meters,  # Pass scaled height
                    plot_csv_path,  # plots render source (possibly downsampled)
                    self.chk_save_svg.isChecked(),  # Save SVG copies
                    output_dir,  # Pass output directory
                    plots_dir,  # Pass plots subfolder
                    self.combo_window_units.currentText(),  # seconds vs samples
                    plot_layers=self.plot_layers,
                    xml_zones=self.xml_zones,
                    anchor_positions=self.anchor_positions,
                    xml_map_image=self.xml_map_image,
                    xml_map_extent=self.xml_map_extent,
                    bg_offset_x=self.bg_offset_x,
                    bg_offset_y=self.bg_offset_y,
                )
                self.worker.progress.connect(self.update_status)
                self.worker.finished.connect(lambda success, msg: self.export_finished(success, msg, save_animation, output_dir, total_steps, current_step, anim_csv_path, animations_dir))
                self.worker.start()
            
            # Animation will be started from export_finished() after plots complete
            elif save_animation:
                # If no plots, start animation directly
                if self.export_cancelled:
                    self.stop_export()
                    return
                
                current_step += 1
                self.lbl_export_progress.setText(f"Step {current_step}/{total_steps}: Generating animation...")
                self.progress_bar.setValue(int((current_step - 1) / total_steps * 100))
                QApplication.processEvents()
                
                self.generate_animation(output_dir, total_steps, current_step, anim_csv_path, animations_dir)

            # If no plots or animation, show success message now
            any_csv = export_raw_csv or export_smoothed_csv
            if (any_csv or detect_proximity) and not save_plots and not save_animation:
                self.log_message("✓ Export completed successfully")
                msg = f"Export completed to:\n{output_dir}"
                QMessageBox.information(self, "Success", msg)
                
        except Exception as e:
            QMessageBox.critical(self, "Error", f"Export failed: {str(e)}")
            self.log_message(f"✗ Export failed: {str(e)}")
        finally:
            # Always clear the busy cursor, on success, cancel, or error. The
            # async plot worker (if started) runs in its own thread and does not
            # freeze the UI, so restoring here is correct.
            QApplication.restoreOverrideCursor()

    def update_status(self, message):
        """Update status label and messages window"""
        self.log_message(message)
    
    def export_finished(self, success, message, start_animation=False, output_dir=None, total_steps=1, current_step=1, csv_path=None, animations_dir=None):
        """Handle export completion"""
        if not self.export_cancelled:
            if success:
                self.log_message("✓ Plot export completed successfully")

                # Start animation if requested (plots are now complete)
                if start_animation and output_dir:
                    self.log_message("Starting animation generation...")
                    self.lbl_export_progress.setText(f"Step {current_step + 1}/{total_steps}: Generating animation...")
                    self.progress_bar.setValue(int(current_step / total_steps * 100))
                    QApplication.processEvents()
                    self.generate_animation(output_dir, total_steps, current_step + 1, csv_path, animations_dir)
                    return  # Don't reset UI yet, animation will do that
                else:
                    self.progress_bar.setValue(100)
                    self.lbl_export_progress.setText("Export complete!")
                    QMessageBox.information(self, "Success", "Export completed successfully!")
            else:
                self.log_message(f"✗ Plot export failed: {message}")
                QMessageBox.critical(self, "Error", message)

        # Plots are done and no animation follows (that path returned above and
        # cleans up itself), so drop the temp render CSVs now.
        self._cleanup_plot_working_files()

        # Reset UI state
        self.exporting = False
        self.btn_export.setEnabled(True)
        self.btn_stop_export.setVisible(False)

        # Hide progress after a delay
        QTimer.singleShot(3000, lambda: self.progress_widget.setVisible(False))
    
    def closeEvent(self, event):
        """Handle window close event - stop any ongoing exports"""
        if self.exporting:
            reply = QMessageBox.question(self, 'Export in Progress', 
                                        'An export is in progress. Do you want to cancel it and close?',
                                        QMessageBox.Yes | QMessageBox.No, QMessageBox.No)
            
            if reply == QMessageBox.Yes:
                self.stop_export()
                event.accept()
            else:
                event.ignore()
        else:
            event.accept()
        
        self.log_message("Ready for next operation")


# Kept alive for the whole process so faulthandler's file target isn't GC'd.
_FAULT_LOG = None


def _install_faulthandler():
    """Dump a native-crash traceback to a local log file (and the console).

    A hard crash inside a C extension (numpy/MKL/matplotlib) terminates the
    process with no Python traceback — e.g. the 0xc000001d illegal-instruction
    crash seen on Python 3.14 during plot export. faulthandler installs a fatal-
    signal handler that writes every thread's Python stack on the way down, so a
    recurrence names the exact plot/line (including the export worker thread).

    The trace goes to ~/.fnt/faulthandler_crash.log on the LOCAL disk (never the
    network share, so it survives even if that share is what hiccuped) and the
    path is printed at startup.
    """
    global _FAULT_LOG
    try:
        log_dir = os.path.join(os.path.expanduser("~"), ".fnt")
        os.makedirs(log_dir, exist_ok=True)
        log_path = os.path.join(log_dir, "faulthandler_crash.log")
        _FAULT_LOG = open(log_path, "a", buffering=1, encoding="utf-8")
        from datetime import datetime
        _FAULT_LOG.write(f"\n===== FNT session {datetime.now():%Y-%m-%d %H:%M:%S} =====\n")
        _FAULT_LOG.flush()
        faulthandler.enable(file=_FAULT_LOG, all_threads=True)
        print(f"[FNT] Native-crash log enabled: {log_path}")
    except Exception as e:
        # Never block startup on logging setup — fall back to stderr-only.
        try:
            faulthandler.enable(all_threads=True)
        except Exception:
            pass
        print(f"[FNT] faulthandler on stderr only ({e})")


def main():
    _install_faulthandler()
    app = QApplication(sys.argv)
    window = UWBQuickVisualizationWindow()
    window.show()
    sys.exit(app.exec_())


if __name__ == "__main__":
    main()
