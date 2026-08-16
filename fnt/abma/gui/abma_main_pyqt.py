"""ABMA main window — the PyQt5 tab launched from the FNT 'WIFP' / ABMA tab.

Four-panel workflow: Arena → Population → Experiment → Run. The window collects
an :class:`ExperimentConfig` from the widgets and hands it to the headless
engine, streaming live agent positions back to the canvas during the first trial.
"""
from __future__ import annotations

import copy
import os
import subprocess
import sys
import threading
import time
import traceback

from PyQt5.QtCore import Qt, QThread, QTimer, QPoint, pyqtSignal
from PyQt5.QtGui import QColor, QPixmap, QPainter, QPolygon, QPen
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QTabWidget, QLabel, QPushButton, QDoubleSpinBox, QSpinBox, QComboBox,
    QLineEdit, QCheckBox, QTableWidget, QTableWidgetItem, QGroupBox,
    QTextEdit, QProgressBar, QFileDialog, QMessageBox, QHeaderView,
    QAbstractItemView, QAction, QSplitter, QStackedWidget, QScrollArea,
    QFrame, QDialog, QListWidget, QDialogButtonBox, QSizePolicy,
    QToolButton, QSlider, QInputDialog, QMenu, QApplication,
)

from ..core.config import (
    ExperimentConfig, ArenaConfig, AgentGroup, Genotype, Treatment,
    TraitProfile, ResourceObject, Intervention, Appearance, Coupling,
    default_dynamics, GrassSpec, blank_experiment, default_vole_experiment,
)
from ..core.runner import run_experiment, grid_offsets
from ..core.sampling import parse_spec
from ..core.presets import (
    all_presets, save_user_preset, suggest_preset_name,
)
from ..core.project import Project, list_projects, default_root
from .abma_canvas import ArenaCanvas
from .agent_inspector import AgentInspector
from .protocol_timeline import ProtocolTimeline, describe as describe_event

try:  # 3D live view needs pyqtgraph.opengl (PyOpenGL); fall back if absent
    from .pg_canvas import Arena3DView
    _HAVE_GL = True
    _GL_ERROR = ""
except Exception as _e:  # pragma: no cover - depends on optional PyOpenGL
    Arena3DView = None
    _HAVE_GL = False
    _GL_ERROR = f"{type(_e).__name__}: {_e}"

POP_COLS = ["label", "species", "sex", "count", "genes", "drug", "dose",
            "onset", "mass", "aggression", "boldness", "sociability", "smell",
            "identity", "home_r"]
# attribute columns that accept a scalar OR a distribution spec (e.g. "N(33,3)")
_COL_TRAIT = {
    "mass": "mass", "aggression": "aggression", "boldness": "boldness",
    "sociability": "sociability", "smell": "smell_ability",
    "identity": "identity_signal", "home_r": "home_range_r",
}
_TRAIT_DEFAULT = {
    "mass": 40.0, "aggression": 0.5, "boldness": 0.5, "sociability": 0.5,
    "smell_ability": 1.0, "identity_signal": 1.0, "home_range_r": 0.55,
}
OBJ_COLS = ["kind", "x", "y", "radius", "label"]
IV_COLS = ["at_day", "target", "attribute", "op", "value"]
DYN_COLS = ["source", "target", "effect", "gain", "scale_by", "when",
            "threshold", "note"]
_IV_ATTRS = ["smell_ability", "identity_signal", "aggression", "boldness",
             "sociability", "exploration", "base_speed", "home_range_r",
             "mass", "metabolism"]


# --------------------------------------------------------------------------- #
# Worker
# --------------------------------------------------------------------------- #
class ABMARunWorker(QThread):
    progress = pyqtSignal(float)
    frame = pyqtSignal(dict)
    agents = pyqtSignal(list)   # static per-agent metadata (once, at run start)
    log = pyqtSignal(str)
    done = pyqtSignal(list)
    failed = pyqtSignal(str)

    def __init__(self, config: ExperimentConfig, project_dir: str,
                 analyze: bool = False):
        super().__init__()
        self.config = config
        self.project_dir = project_dir
        self.analyze = analyze
        total_s = config.days * 86400.0
        # ~500 live frames regardless of duration; never finer than one step
        self._frame_interval = max(config.dt, total_s / 500.0)
        self._cancel = threading.Event()
        self._last_emit = 0.0
        self._min_frame_dt = 1.0 / 30.0     # cap live frames at ~30/s (wall-clock)

    def cancel(self):
        """Request a graceful stop; the run loop checks this each step."""
        self._cancel.set()

    def _emit_frame(self, fr):
        # throttle to wall-clock so a fast run can't flood the GUI event queue
        now = time.monotonic()
        if now - self._last_emit >= self._min_frame_dt:
            self._last_emit = now
            self.frame.emit(fr)

    def run(self):
        try:
            res = run_experiment(
                self.config, self.project_dir,
                progress_cb=lambda f: self.progress.emit(float(f)),
                frame_cb=self._emit_frame,
                log_cb=lambda m: self.log.emit(m),
                frame_interval_s=self._frame_interval,
                analyze=self.analyze,
                meta_cb=lambda meta: self.agents.emit(meta),
                cancel_cb=self._cancel.is_set,
            )
            self.done.emit(res)
        except Exception:
            self.failed.emit(traceback.format_exc())


# --------------------------------------------------------------------------- #
# Main window
# --------------------------------------------------------------------------- #
class ABMAWindow(QMainWindow):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ABMA — Animal Behavior Modeling Arena")
        self.resize(1240, 800)
        self.worker = None
        self.project = None                  # the open Project, if any
        self._run_t0 = None
        self._last_frame = None
        self._zones = []
        self._running = False
        self._preview_sim = None
        self._preview_elapsed = 0.0
        self._preview_interval = 60          # ms; slowed for large populations
        self._preview_timer = QTimer(self)
        self._preview_timer.timeout.connect(self._preview_tick)
        # transport / playback state (for reviewing a run's buffered frames)
        self._frames = []
        self._play_idx = 0
        self._playing = False
        self._live_follow = True
        self._play_speed = 1.0
        self._follow_agent = False
        self._play_timer = QTimer(self)
        self._play_timer.timeout.connect(self._play_tick)
        self.setStyleSheet(_DARK_QSS + "\n" + _arrow_qss())
        self._build_menu()

        # unified interactive preview: one 3D GL view + one 2D view (toggle),
        # used for BOTH setup and running.
        self.view_3d = Arena3DView() if _HAVE_GL else None
        self.view_2d = ArenaCanvas()
        self.canvas = self.view_2d          # placement alias (2D handles clicks)

        left = self._build_left_column()
        right = self._build_preview()

        split = QSplitter(Qt.Horizontal)
        split.setChildrenCollapsible(False)
        split.setHandleWidth(5)
        split.addWidget(left)
        split.addWidget(right)
        split.setStretchFactor(0, 0)
        split.setStretchFactor(1, 1)
        split.setSizes([500, 820])
        self.setCentralWidget(split)

        self._load_config(blank_experiment())
        self._set_view(0)
        self._rebuild_preview()

    # ------------------------------------------------------------------ #
    # Live in-editor preview — agents animate as soon as they're added.
    # This is purely visual; nothing is written to disk (only a real Run does).
    # ------------------------------------------------------------------ #
    def _rebuild_preview(self):
        if self._running or not hasattr(self, "view_2d"):
            return
        try:
            cfg = self._collect_config()
        except Exception as e:
            # surface config errors (e.g. a typo in a dynamics/attribute cell)
            # instead of silently freezing the preview
            self._stop_preview()
            if hasattr(self, "status_label"):
                self.status_label.setText(f"⚠ preview paused — {e}")
                self.status_label.setStyleSheet("color:#e0a23a; font-size:11px;")
            return
        if hasattr(self, "status_label") and not self._frames:
            self.status_label.setText("Idle.")
            self.status_label.setStyleSheet("color:#8a9099; font-size:11px;")
        if cfg.total_agents() == 0:
            self._stop_preview()
            return
        from ..core.simulation import Simulation
        pcfg = copy.deepcopy(cfg)
        pcfg.n_trials = 1
        pcfg.enable_mortality = False
        self._preview_sim = Simulation(pcfg, trial_index=0)
        self._preview_elapsed = 0.0
        for v in self._views():
            v.set_arena(cfg.arena)          # single chamber for the preview
        self.inspector.set_population(self._preview_sim.agent_static())
        # entering live preview discards any prior run's review buffer
        self._frames = []
        self._live_follow = True
        if hasattr(self, "scrubber"):
            self.scrubber.blockSignals(True)
            self.scrubber.setRange(0, 0)
            self.scrubber.blockSignals(False)
            self.lbl_time.setText("")
            self.btn_play.setText("⏸")
        # the preview steps on the GUI thread; ease off for big populations
        self._preview_interval = 60 if self._preview_sim.n <= 150 else 120
        self._preview_timer.start(self._preview_interval)

    def _stop_preview(self):
        self._preview_timer.stop()
        self._preview_sim = None

    def _preview_tick(self):
        if self._preview_sim is None:
            return
        sim = self._preview_sim
        sim.step(self._preview_elapsed, 0.4, events=None)   # no records, no combat
        self._preview_elapsed += 0.4
        self._render_live(sim._frame(self._preview_elapsed))

    # ------------------------------------------------------------------ #
    # Layout: left scrolling section column + right preview
    # ------------------------------------------------------------------ #
    def _build_left_column(self):
        wrap = QWidget()
        wlay = QVBoxLayout(wrap)
        wlay.setContentsMargins(0, 0, 0, 0)
        wlay.setSpacing(0)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(490)
        content = QWidget()
        col = QVBoxLayout(content)
        col.setContentsMargins(10, 10, 10, 10)
        col.setSpacing(8)
        self._sections = {}
        for key, title, builder, expanded in [
            ("arena", "1 · Arena", self._build_arena_tab, True),
            ("agents", "2 · Build && Add Agents", self._build_population_tab, True),
            ("experiment", "3 · Experiment", self._build_experiment_tab, False),
            ("run", "4 · Run log", self._build_run_tab, False),
        ]:
            sec = CollapsibleSection(title, builder(), expanded=expanded)
            self._sections[key] = sec
            col.addWidget(sec)
        col.addStretch(1)
        scroll.setWidget(content)
        wlay.addWidget(scroll, 1)

        # ---- persistent action bar (Run is always reachable) ----
        bar = QFrame()
        bar.setObjectName("run_bar")
        bl = QVBoxLayout(bar)
        bl.setContentsMargins(10, 8, 10, 8)
        bl.setSpacing(5)
        btn_row = QHBoxLayout()
        self.btn_run = QPushButton("▶  Run Experiment")
        self.btn_run.setObjectName("accept_btn")
        self.btn_run.clicked.connect(self._on_run)
        self.btn_stop = QPushButton("■")
        self.btn_stop.setObjectName("reject_btn")
        self.btn_stop.setMaximumWidth(44)
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_stop.setEnabled(False)
        btn_row.addWidget(self.btn_run, 1)
        btn_row.addWidget(self.btn_stop)
        bl.addLayout(btn_row)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        self.progress.setMaximumHeight(6)
        self.progress.setTextVisible(False)
        bl.addWidget(self.progress)
        self.status_label = QLabel("Idle.")
        self.status_label.setStyleSheet("color:#8a9099; font-size:11px;")
        bl.addWidget(self.status_label)
        wlay.addWidget(bar)
        return wrap

    def _build_preview(self):
        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(0, 0, 0, 0)
        rlay.setSpacing(4)
        title_row = QHBoxLayout()
        self.preview_title = QLabel(
            "Preview Window  ·  drag to orbit · right-drag (or ⇧+drag) to pan · "
            "scroll to zoom")
        self.preview_title.setObjectName("preview_title")
        title_row.addWidget(self.preview_title)
        title_row.addStretch()
        self.btn_view3d = QPushButton("3D")
        self.btn_view2d = QPushButton("2D")
        for b in (self.btn_view3d, self.btn_view2d):
            b.setCheckable(True)
            b.setMaximumWidth(46)
            title_row.addWidget(b)
        self.btn_view3d.clicked.connect(lambda: self._set_view(0))
        self.btn_view2d.clicked.connect(lambda: self._set_view(1))
        self.btn_view3d.setEnabled(self.view_3d is not None)
        if self.view_3d is None:
            self.btn_view3d.setToolTip(
                "3D view unavailable — install PyOpenGL in this environment:\n"
                "    pip install PyOpenGL\n"
                "then restart FNT.\n\n"
                f"(import error: {_GL_ERROR})")
        rlay.addLayout(title_row)

        self.view_stack = QStackedWidget()
        if self.view_3d is not None:
            self.view_stack.addWidget(self.view_3d)      # index 0 — 3D
            self.view_3d.agent_picked.connect(self._select_agent)
            self.view_3d.object_added.connect(self._on_canvas_add)
        self.view_stack.addWidget(self.view_2d)          # 1 (or 0 without GL)
        self.view_2d.agent_picked.connect(self._select_agent)
        self.view_2d.object_added.connect(self._on_canvas_add)
        self.view_2d.agent_hovered.connect(self._on_agent_hover)
        self.view_2d.object_moved.connect(self._on_object_moved)
        if self.view_3d is not None:
            self.view_3d.agent_hovered.connect(self._on_agent_hover)
            self.view_3d.object_moved.connect(self._on_object_moved)

        rlay.addWidget(self.view_stack, 1)   # preview takes the full width

        # ---- transport / playback bar ----
        tb = QFrame()
        tb.setObjectName("transport")
        tl = QHBoxLayout(tb)
        tl.setContentsMargins(8, 4, 8, 4)
        tl.setSpacing(8)
        self.btn_play = QToolButton()
        self.btn_play.setText("⏸")
        self.btn_play.setToolTip("Play / pause")
        self.btn_play.clicked.connect(self._on_play_pause)
        tl.addWidget(self.btn_play)
        self.scrubber = QSlider(Qt.Horizontal)
        self.scrubber.setRange(0, 0)
        self.scrubber.valueChanged.connect(self._on_scrub)
        tl.addWidget(self.scrubber, 1)
        self.lbl_time = QLabel("")
        self.lbl_time.setStyleSheet("color:#8a9099; font-size:11px;")
        self.lbl_time.setMinimumWidth(110)
        tl.addWidget(self.lbl_time)
        self.in_speed = QComboBox()
        self.in_speed.addItems(["0.5×", "1×", "2×", "4×"])
        self.in_speed.setCurrentText("1×")
        self.in_speed.setToolTip("Replay speed")
        self.in_speed.currentTextChanged.connect(
            lambda t: setattr(self, "_play_speed", float(t.rstrip("×"))))
        tl.addWidget(self.in_speed)
        b = QToolButton()
        b.setText("⌂")
        b.setToolTip("Reset camera (isometric)")
        b.clicked.connect(lambda: self._snap_view("iso"))
        b.setEnabled(self.view_3d is not None)
        tl.addWidget(b)
        # CAD-style snap-view menu (top/bottom/N/S/E/W/iso)
        self.btn_viewcube = QToolButton()
        self.btn_viewcube.setText("◫ View")
        self.btn_viewcube.setToolTip("Snap to a standard view")
        self.btn_viewcube.setPopupMode(QToolButton.InstantPopup)
        vmenu = QMenu(self.btn_viewcube)
        for label, name in [("Isometric", "iso"), ("Top View", "top"),
                            ("Bottom", "bottom"), ("North side", "north"),
                            ("South side", "south"), ("East side", "east"),
                            ("West side", "west")]:
            vmenu.addAction(label, lambda nm=name: self._snap_view(nm))
        self.btn_viewcube.setMenu(vmenu)
        self.btn_viewcube.setEnabled(self.view_3d is not None)
        tl.addWidget(self.btn_viewcube)
        # slow turntable spin around the arena's central axis (3D only)
        self.btn_spin = QToolButton()
        self.btn_spin.setText("⟳")
        self.btn_spin.setToolTip("Slowly spin the camera around the arena")
        self.btn_spin.setCheckable(True)
        self.btn_spin.setEnabled(self.view_3d is not None)
        self.btn_spin.toggled.connect(self._on_toggle_spin)
        # right-click the spin button to pick a speed / reverse direction
        self.btn_spin.setToolTip(
            "Slowly spin the camera around the arena\n"
            "(right-click for speed / direction)")
        self._spin_speed = 12.0          # °/s magnitude; one rev ≈ 30 s
        self._spin_reverse = False
        self.btn_spin.setContextMenuPolicy(Qt.CustomContextMenu)
        self.btn_spin.customContextMenuRequested.connect(self._show_spin_menu)
        tl.addWidget(self.btn_spin)
        self.btn_follow = QToolButton()
        self.btn_follow.setText("◎")
        self.btn_follow.setToolTip("Follow selected agent")
        self.btn_follow.setCheckable(True)
        self.btn_follow.setEnabled(self.view_3d is not None)
        self.btn_follow.toggled.connect(
            lambda c: setattr(self, "_follow_agent", c))
        tl.addWidget(self.btn_follow)
        self.btn_agents = QToolButton()
        self.btn_agents.setText("🐭")
        self.btn_agents.setToolTip("Show / hide agents")
        self.btn_agents.setCheckable(True)
        self.btn_agents.setChecked(True)
        self.btn_agents.toggled.connect(self._on_toggle_agents)
        tl.addWidget(self.btn_agents)
        self._theme_state = "dark"
        self.btn_theme = QToolButton()
        self.btn_theme.setText("🌙")
        self.btn_theme.setToolTip(
            "Arena theme: dark ↔ light (pure white, for slides)")
        self.btn_theme.setCheckable(True)
        self.btn_theme.clicked.connect(self._cycle_theme)
        tl.addWidget(self.btn_theme)
        self.btn_grass = QToolButton()
        self.btn_grass.setText("🌱")
        self.btn_grass.setToolTip("Toggle grass (outdoor field sites)")
        self.btn_grass.setCheckable(True)
        self.btn_grass.setEnabled(False)
        self.btn_grass.toggled.connect(self._on_toggle_grass)
        tl.addWidget(self.btn_grass)
        self._antenna_state = -1        # -1 = off; else antenna-layout index
        self.btn_antenna = QToolButton()
        self.btn_antenna.setText("📡")
        self.btn_antenna.setToolTip("UWB antennas: cycle layouts / off")
        self.btn_antenna.setCheckable(True)
        self.btn_antenna.setEnabled(False)
        self.btn_antenna.clicked.connect(self._cycle_antennas)
        tl.addWidget(self.btn_antenna)
        self._measure_mode = None
        self.btn_measure = QToolButton()
        self.btn_measure.setText("📏")
        self.btn_measure.setCheckable(True)
        self.btn_measure.setToolTip("Measure grid: off → metric → imperial")
        self.btn_measure.clicked.connect(self._cycle_measure)
        tl.addWidget(self.btn_measure)
        self._resource_state = -1       # -1 off, 0 lids-on, 1 lids-off
        self.btn_resources = QToolButton()
        self.btn_resources.setText("💧🌿")
        self.btn_resources.setToolTip(
            "Show / hide water towers & resource zones")
        self.btn_resources.setCheckable(True)
        self.btn_resources.setEnabled(False)
        self.btn_resources.clicked.connect(self._cycle_resources)
        tl.addWidget(self.btn_resources)
        rlay.addWidget(tb)

        # inspector is a floating popup shown on hover / click (frees the preview)
        self.inspector = AgentInspector(self)
        self.inspector.setWindowFlags(
            Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        self.inspector.hide()
        self._inspector_pinned = False
        self._hover_idx = None
        return right

    def _on_toggle_grass(self, on):
        for v in self._views():
            if hasattr(v, "set_grass_enabled"):
                v.set_grass_enabled(on)

    def _on_toggle_agents(self, on):
        for v in self._views():
            if hasattr(v, "set_agents_visible"):
                v.set_agents_visible(on)

    _THEME_ORDER = ["dark", "light"]
    _THEME_ICON = {"dark": "🌙", "light": "☀"}
    _THEME_NAME = {"dark": "dark", "light": "light (pure white, for slides)"}

    def _cycle_theme(self, *_):
        order = self._THEME_ORDER
        self._theme_state = order[(order.index(self._theme_state) + 1) % len(order)]
        self.btn_theme.setText(self._THEME_ICON[self._theme_state])
        self.btn_theme.setChecked(self._theme_state != "dark")
        for v in self._views():
            if hasattr(v, "set_theme"):
                v.set_theme(self._theme_state)
        self.statusBar().showMessage(
            f"Arena theme: {self._THEME_NAME[self._theme_state]}", 2500)

    def _snap_view(self, name):
        v = self._active_view()
        if hasattr(v, "snap_view"):
            v.snap_view(name)
        elif name == "top" and hasattr(v, "top_down"):
            v.top_down()
        elif hasattr(v, "reset_camera"):
            v.reset_camera()

    def _antenna_sets(self):
        if getattr(self, "_antenna_layouts", None):
            return [(l.name, l.antennas) for l in self._antenna_layouts]
        if getattr(self, "_antennas", None):
            return [("antennas", self._antennas)]
        return []

    def _antenna_options(self):
        """Cycle steps: each layout with, then without, its antenna numbers."""
        opts = []
        for i, entry in enumerate(self._antenna_sets()):
            opts.append((i, True, entry[0]))
            opts.append((i, False, f"{entry[0]} — no labels"))
        return opts

    def _apply_antenna_state(self):
        """Push the current antenna layout + label mode to both views."""
        self.btn_antenna.setChecked(self._antenna_state >= 0)
        opts = self._antenna_options()
        if 0 <= self._antenna_state < len(opts):
            idx, labels, _ = opts[self._antenna_state]
        else:
            idx, labels = -1, True
        for v in self._views():
            if hasattr(v, "set_antenna_layout"):
                v.set_antenna_layout(idx, labels)

    def _cycle_antennas(self, *_):
        opts = self._antenna_options()
        if not opts or self._antenna_state >= len(opts) - 1:
            self._antenna_state = -1
        else:
            self._antenna_state += 1
        self._apply_antenna_state()
        msg = "off" if self._antenna_state < 0 else opts[self._antenna_state][2]
        self.statusBar().showMessage(f"UWB antennas: {msg}", 2500)

    def _apply_resource_state(self):
        self.btn_resources.setChecked(self._resource_state >= 0)
        for v in self._views():
            if hasattr(v, "set_resources_mode"):
                v.set_resources_mode(self._resource_state)

    def _cycle_resources(self, *_):
        # zones have no lid, so this is a simple show / hide
        self._resource_state = 0 if self._resource_state < 0 else -1
        self._apply_resource_state()
        msg = "shown" if self._resource_state >= 0 else "off"
        self.statusBar().showMessage(f"Resources: {msg}", 2500)

    def _cycle_measure(self, *_):
        nxt = {None: "metric", "metric": "imperial",
               "imperial": None}[self._measure_mode]
        self._measure_mode = nxt
        self.btn_measure.setChecked(nxt is not None)
        for v in self._views():
            if hasattr(v, "set_measure"):
                v.set_measure(nxt)
        lbl = {None: "off", "metric": "metric (m)",
               "imperial": "imperial (ft)"}[nxt]
        self.statusBar().showMessage(f"Measure grid: {lbl}", 2500)

    def _views(self):
        return [v for v in (self.view_3d, self.view_2d) if v is not None]

    def _active_view(self):
        return self.view_stack.currentWidget()

    def _set_view(self, idx):
        show3d = (idx == 0 and self.view_3d is not None)
        self.btn_view3d.setChecked(show3d)
        self.btn_view2d.setChecked(not show3d)
        self.view_stack.setCurrentIndex(
            0 if show3d else self.view_stack.count() - 1)
        # spin only runs while the 3D view is showing; resume it on return if
        # the toggle is still on, and don't churn a hidden GL widget in 2D.
        if self.view_3d is not None and hasattr(self.view_3d, "set_spin"):
            self.view_3d.set_spin(show3d and self.btn_spin.isChecked())
        if self._last_frame is not None:
            self._push_frame(self._active_view(), self._last_frame)

    def _on_toggle_spin(self, on):
        v = self.view_3d
        if v is not None and hasattr(v, "set_spin"):
            self._apply_spin_speed()
            v.set_spin(on)

    def _apply_spin_speed(self):
        """Push the current speed magnitude + direction to the 3D view."""
        v = self.view_3d
        if v is not None and hasattr(v, "set_spin_speed"):
            v.set_spin_speed(-self._spin_speed if self._spin_reverse
                             else self._spin_speed)

    def _show_spin_menu(self, pos):
        """Right-click menu on the spin button: pick a speed / reverse."""
        if self.view_3d is None:
            return
        menu = QMenu(self.btn_spin)
        # (label, °/s) — revolution time ≈ 360 / (°/s)
        for label, dps in [("Very slow  (~60 s / turn)", 6.0),
                           ("Slow  (~30 s / turn)", 12.0),
                           ("Medium  (~15 s / turn)", 24.0),
                           ("Fast  (~8 s / turn)", 45.0)]:
            act = menu.addAction(label)
            act.setCheckable(True)
            act.setChecked(abs(self._spin_speed - dps) < 1e-6)
            act.triggered.connect(lambda _=False, d=dps: self._set_spin_speed(d))
        menu.addSeparator()
        rev = menu.addAction("Reverse direction")
        rev.setCheckable(True)
        rev.setChecked(self._spin_reverse)
        rev.triggered.connect(self._toggle_spin_reverse)
        menu.exec_(self.btn_spin.mapToGlobal(pos))

    def _set_spin_speed(self, dps):
        self._spin_speed = float(dps)
        self._apply_spin_speed()

    def _toggle_spin_reverse(self, on):
        self._spin_reverse = bool(on)
        self._apply_spin_speed()

    # ------------------------------------------------------------------ #
    # Projects — the unit that carries world + population + protocol + runs
    # ------------------------------------------------------------------ #
    def show_start_dialog(self):
        """Title screen. Called by the launcher, never from __init__ (so the
        window stays constructible headlessly)."""
        dlg = StartDialog(self)
        if dlg.exec_() != QDialog.Accepted or not dlg.choice:
            return
        kind = dlg.choice[0]
        if kind == "new":
            _, preset, name = dlg.choice
            self._apply_preset(preset)
            self._new_project(name)
        else:
            self._open_project(dlg.choice[1])

    def _new_project(self, name):
        try:
            cfg = self._collect_config()
        except Exception as e:
            QMessageBox.warning(self, "Invalid configuration", str(e))
            return
        cfg.name = name
        try:
            self.project = Project.create(name, cfg)
        except Exception as e:
            QMessageBox.critical(self, "Could not create project", str(e))
            return
        self.in_name.setText(name)
        self._update_title()
        self.statusBar().showMessage(f"Project created: {self.project.path}", 6000)

    def _open_project(self, path):
        try:
            proj = Project.load(path)
        except Exception as e:
            QMessageBox.critical(self, "Could not open project", str(e))
            return
        self.project = proj
        if proj.config is not None:
            self._load_config(proj.config)
        self._update_title()
        n = len(proj.runs())
        self.statusBar().showMessage(
            f"Opened '{proj.name}' — {n} run(s) in history", 6000)

    def _save_project(self):
        if self.project is None:
            name, ok = QInputDialog.getText(
                self, "Save as project", "Project name:",
                text=self.in_name.text().strip() or "experiment")
            if not ok or not name.strip():
                return
            self._new_project(name.strip())
            return
        try:
            self.project.config = self._collect_config()
        except Exception as e:
            QMessageBox.warning(self, "Invalid configuration", str(e))
            return
        self.project.save()
        self.statusBar().showMessage(f"Saved {self.project.name}", 4000)

    def _run_study_dialog(self):
        """Sweep one parameter across levels and compare the arms."""
        try:
            base = self._collect_config()
        except Exception as e:
            QMessageBox.warning(self, "Invalid configuration", str(e))
            return
        path, ok = QInputDialog.getText(
            self, "Run study — 1/3: what to vary",
            "Override path ([*] = all groups):",
            text="groups[*].traits.smell_ability")
        if not ok or not path.strip():
            return
        levels, ok = QInputDialog.getText(
            self, "Run study — 2/3: conditions",
            "name=value, comma separated:", text="intact=1.0, anosmic=0.0")
        if not ok or not levels.strip():
            return
        reps, ok = QInputDialog.getInt(
            self, "Run study — 3/3: replicates",
            "Replicates per condition:", 4, 1, 50)
        if not ok:
            return
        try:
            lv = {}
            for part in levels.split(","):
                k, v = part.split("=")
                try:
                    lv[k.strip()] = float(v)
                except ValueError:
                    lv[k.strip()] = v.strip()
            if len(lv) < 2:
                raise ValueError("need at least two conditions")
            from ..core.study import lesion_study, run_study
            study = lesion_study(f"{base.name}_study", base, path.strip(),
                                 lv, replicates=reps)
            study.config_for(0)                 # fail fast on a bad path
        except Exception as e:
            QMessageBox.warning(self, "Could not build study", str(e))
            return

        root = (self.project.path if self.project
                else (self.in_outdir.text().strip() or os.getcwd()))
        sdir = os.path.join(root, "studies",
                            f"{time.strftime('%Y%m%d-%H%M')}_{path.split('.')[-1]}")
        n_runs = len(lv) * reps
        if QMessageBox.question(
                self, "Run study?",
                f"{len(lv)} conditions × {reps} replicates = {n_runs} trials\n"
                f"varying {path}\n\nOutput: {sdir}") != QMessageBox.Yes:
            return
        self._append_log(f"Study: {len(lv)} conditions × {reps} replicates…")
        QApplication.setOverrideCursor(Qt.WaitCursor)
        try:
            res = run_study(study, sdir, log_cb=self._append_log)
        except Exception as e:
            QApplication.restoreOverrideCursor()
            QMessageBox.critical(self, "Study failed", str(e))
            return
        QApplication.restoreOverrideCursor()
        self._project_dir = sdir
        msg = f"Study complete — {n_runs} trials.\n\n{sdir}"
        if res.get("comparison_csv"):
            msg += "\n\nresults/metrics_long.csv  (tidy, for R)\nresults/comparison.csv"
        QMessageBox.information(self, "Study complete", msg)

    def _update_title(self):
        base = "ABMA — Animal Behavior Modeling Arena"
        self.setWindowTitle(f"{self.project.name} — {base}"
                            if self.project else base)

    def _open_preset_dialog(self):
        dlg = ArenaPresetDialog(self)
        if dlg.exec_() == QDialog.Accepted and dlg.chosen():
            self._apply_preset(dlg.chosen())

    def _apply_preset(self, p):
        self._loaded_abbr = p.abbr or "Arena"
        self._loaded_preset = p
        cfgs = p.config_list() if hasattr(p, "config_list") else [(p.name, p.factory)]
        self.cfg_combo.blockSignals(True)
        self.cfg_combo.clear()
        self.cfg_combo.addItems([c[0] for c in cfgs])
        self.cfg_combo.setCurrentIndex(0)
        self.cfg_combo.blockSignals(False)
        self._cfg_row.setVisible(len(cfgs) > 1)
        self._load_config(cfgs[0][1]())
        # a blank arena opens the editor to build from scratch; a named preset
        # stays collapsed behind "Modify".
        self.btn_modify.setChecked(bool(getattr(p, "blank", False)))
        self.statusBar().showMessage(f"Loaded: {p.name}", 4000)

    def _on_config_changed(self, idx):
        """Switch to another named configuration of the loaded enclosure."""
        p = getattr(self, "_loaded_preset", None)
        if p is None or idx < 0:
            return
        cfgs = p.config_list()
        if not (0 <= idx < len(cfgs)):
            return
        name, factory = cfgs[idx]
        self._load_config(factory())
        self.statusBar().showMessage(f"Configuration: {name}", 4000)

    # ------------------------------------------------------------------ #
    # Menu bar
    # ------------------------------------------------------------------ #
    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        preset_menu = m.addMenu("Load &Preset")
        for p in all_presets():
            act = QAction(p.name, self)
            act.setStatusTip(p.description)
            act.triggered.connect(lambda _, pr=p: self._apply_preset(pr))
            preset_menu.addAction(act)
        m.addSeparator()
        for text, shortcut, slot in [
            ("Start screen…", None, self.show_start_dialog),
            ("&Save Project", "Ctrl+Shift+S", self._save_project),
            ("Run &Study (compare conditions)…", None, self._run_study_dialog),
            (None, None, None),
            ("&New (blank experiment)", "Ctrl+N",
             lambda: self._load_config(blank_experiment())),
            (None, None, None),
            ("&Save Config…", "Ctrl+S", self._save_config),
            ("&Open Config…", "Ctrl+O", self._open_config),
            (None, None, None),
            ("Open Output &Folder", None, self._open_output),
            (None, None, None),
            ("&Close", "Ctrl+W", self.close),
        ]:
            if text is None:
                m.addSeparator()
                continue
            act = QAction(text, self)
            if shortcut:
                act.setShortcut(shortcut)
            act.triggered.connect(slot)
            m.addAction(act)

    def _save_config(self):
        try:
            cfg = self._collect_config()
        except Exception as e:
            QMessageBox.warning(self, "Invalid configuration", str(e))
            return
        path, _ = QFileDialog.getSaveFileName(
            self, "Save experiment config", f"{cfg.name}.json", "JSON (*.json)")
        if path:
            cfg.to_json(path)
            self.statusBar().showMessage(f"Saved {path}", 4000)


    def _open_config(self):
        path, _ = QFileDialog.getOpenFileName(
            self, "Open experiment config", "", "JSON (*.json)")
        if not path:
            return
        try:
            self._load_config(ExperimentConfig.from_json(path))
            self.statusBar().showMessage(f"Loaded {path}", 4000)
        except Exception as e:
            QMessageBox.critical(self, "Load failed", str(e))

    # ------------------------------------------------------------------ #
    # Tab 1: Arena
    # ------------------------------------------------------------------ #
    def _build_arena_tab(self):
        w = QWidget()
        left = QVBoxLayout(w)
        left.setContentsMargins(0, 0, 0, 0)
        self._arena_dirty = False
        self._loaded_abbr = "Arena"

        btn_preset = QPushButton("⬗  Load Preset Arena…")
        btn_preset.setObjectName("accept_btn")
        btn_preset.clicked.connect(self._open_preset_dialog)
        left.addWidget(btn_preset)

        # configuration picker — shown only for presets with named variants
        self._cfg_row = QWidget()
        cfgl = QHBoxLayout(self._cfg_row)
        cfgl.setContentsMargins(0, 0, 0, 0)
        cfgl.addWidget(QLabel("Configuration"))
        self.cfg_combo = QComboBox()
        self.cfg_combo.setToolTip("Alternate configurations of this enclosure")
        self.cfg_combo.currentIndexChanged.connect(self._on_config_changed)
        cfgl.addWidget(self.cfg_combo, 1)
        self._cfg_row.setVisible(False)
        left.addWidget(self._cfg_row)

        self.arena_summary = QLabel("No arena loaded.")
        self.arena_summary.setWordWrap(True)
        self.arena_summary.setStyleSheet("color:#8fbfff; font-size:11px;")
        left.addWidget(self.arena_summary)

        act = QHBoxLayout()
        self.btn_modify = QPushButton("Modify arena")
        self.btn_modify.setCheckable(True)
        self.btn_modify.toggled.connect(self._toggle_arena_editor)
        self.btn_save_preset = QPushButton("Save preset…")
        self.btn_save_preset.setEnabled(False)
        self.btn_save_preset.setToolTip("Save the modified arena as a reusable preset")
        self.btn_save_preset.clicked.connect(self._save_preset)
        act.addWidget(self.btn_modify)
        act.addWidget(self.btn_save_preset)
        left.addLayout(act)

        self.btn_photo_grass = QPushButton("🌿  Grass from drone photo…")
        self.btn_photo_grass.setToolTip(
            "Measure ground cover from an overhead photo of this enclosure")
        self.btn_photo_grass.clicked.connect(self._grass_from_photo)
        left.addWidget(self.btn_photo_grass)

        # ---- editing panel (hidden until Modify, or a blank arena is loaded) ----
        self._arena_editor = QWidget()
        ed = QVBoxLayout(self._arena_editor)
        ed.setContentsMargins(0, 4, 0, 0)
        form = QFormLayout()
        form.setLabelAlignment(Qt.AlignLeft)                    # left-aligned labels
        form.setFormAlignment(Qt.AlignLeft | Qt.AlignTop)
        form.setFieldGrowthPolicy(QFormLayout.AllNonFixedFieldsGrow)
        self.in_width = _dspin(0.2, 50, 2.0, 0.1, " m")
        self.in_height = _dspin(0.2, 50, 2.0, 0.1, " m")
        self.in_boundary = QComboBox()
        self.in_boundary.addItems(["reflective", "wrap", "absorbing"])
        for lab, wdg in [("Arena width", self.in_width),
                         ("Arena height", self.in_height),
                         ("Boundary", self.in_boundary)]:
            form.addRow(lab, wdg)
        self.in_snap = _dspin(0.0, 5.0, 0.0, 0.05, " m")
        self.in_snap.setToolTip(
            "Snap while dragging objects (0 = free). Editing aid only — the "
            "simulation itself stays continuous.")
        self.in_snap.valueChanged.connect(lambda _: self._apply_edit_mode())
        form.addRow("Snap to grid", self.in_snap)
        self.in_width.valueChanged.connect(self._on_arena_edit)
        self.in_height.valueChanged.connect(self._on_arena_edit)
        self.in_boundary.currentTextChanged.connect(self._on_arena_edit)
        ed.addLayout(form)
        hint = QLabel("Drag objects in the preview to move them.")
        hint.setStyleSheet("color:#8a9099; font-size:10px;")
        hint.setWordWrap(True)
        ed.addWidget(hint)

        add_box = QGroupBox("Add object (then click the arena)")
        ab = QHBoxLayout(add_box)
        for kind in ("nest", "food", "water"):
            b = QPushButton(f"+ {kind}")
            b.setCheckable(True)
            b.clicked.connect(lambda _, k=kind, btn=b: self._arm(k, btn))
            ab.addWidget(b)
            setattr(self, f"_arm_btn_{kind}", b)
        ed.addWidget(add_box)

        self.obj_table = _table(OBJ_COLS)
        self.obj_table.setMinimumHeight(140)
        self.obj_table.itemChanged.connect(self._on_arena_edit)
        ed.addWidget(self.obj_table, 1)
        b_del = QPushButton("Remove selected object")
        b_del.clicked.connect(
            lambda: (self._del_row(self.obj_table), self._mark_arena_dirty()))
        ed.addWidget(b_del)

        self._arena_editor.setVisible(False)
        left.addWidget(self._arena_editor)
        return w

    def _grass_from_photo(self):
        """Measure ground cover from an overhead photo and apply it as grass."""
        path, _ = QFileDialog.getOpenFileName(
            self, "Select an overhead photo of this enclosure", "",
            "Images (*.jpg *.jpeg *.png *.tif *.tiff)")
        if not path:
            return
        dlg = CornerPickDialog(path, self)
        if dlg.exec_() != QDialog.Accepted or len(dlg.corners()) != 4:
            return
        try:
            from ..core.photo_cover import (analyse_photo, to_cover_map,
                                            suggested_dry_fraction)
            arena = self._arena_from_table()
            res = analyse_photo(path, dlg.corners(), arena.width, arena.height,
                                grid=8)
            spec = self._grass_spec or GrassSpec()
            spec.cover_map = to_cover_map(res["cover"])
            spec.dry_fraction = suggested_dry_fraction(res["mean_green"])
            spec.patchiness = max(spec.patchiness, 0.7)
            self._grass_spec = spec
            self._ground = "grass"
        except Exception as e:
            QMessageBox.warning(self, "Could not analyse photo", str(e))
            return
        self._mark_arena_dirty()
        self._refresh_arena()
        if hasattr(self, "btn_grass"):
            self.btn_grass.setChecked(True)
        QMessageBox.information(
            self, "Ground cover measured",
            f"Live green cover: {100 * res['mean_green']:.1f}%  "
            f"(range {100 * res['min_green']:.0f}–{100 * res['max_green']:.0f}% "
            "by cell)\n"
            f"North {100 * res['north_half']:.1f}%  vs  South "
            f"{100 * res['south_half']:.1f}%\n"
            f"West {100 * res['west_half']:.1f}%  vs  East "
            f"{100 * res['east_half']:.1f}%\n\n"
            f"Applied as an 8×8 cover map with dry_fraction "
            f"{spec.dry_fraction:.2f}.\n"
            "Only live green is measurable from colour; dry thatch is "
            "represented by a uniform floor.")

    def _toggle_arena_editor(self, on):
        self._arena_editor.setVisible(on)
        self.btn_modify.setText("Hide arena editor" if on else "Modify arena")
        self._apply_edit_mode()

    def _apply_edit_mode(self):
        """Objects are only draggable while the arena editor is open."""
        on = self.btn_modify.isChecked() and not self._running
        snap = self.in_snap.value() if hasattr(self, "in_snap") else 0.0
        for v in self._views():
            if hasattr(v, "set_edit_layout"):
                v.set_edit_layout(on, snap)

    def _on_object_moved(self, kind, index, x, y):
        """Live drag from either view -> update the arena and redraw."""
        x = min(max(float(x), 0.0), self.in_width.value())
        y = min(max(float(y), 0.0), self.in_height.value())
        attr = {"hut": "_huts", "water": "_water_towers", "pole": "_poles",
                "zone": "_resource_zones"}.get(kind)
        if attr is not None:
            lst = getattr(self, attr, [])
            if 0 <= index < len(lst):
                lst[index].x, lst[index].y = x, y
        elif kind == "object":                    # lives in the object table
            t = self.obj_table
            if 0 <= index < t.rowCount():
                t.blockSignals(True)
                t.item(index, 1).setText(f"{x:.3f}")
                t.item(index, 2).setText(f"{y:.3f}")
                t.blockSignals(False)
        self._mark_arena_dirty()
        self._refresh_arena()

    def _on_arena_edit(self, *_):
        self._refresh_arena()
        self._mark_arena_dirty()

    def _mark_arena_dirty(self):
        if getattr(self, "_loading", False):
            return
        self._arena_dirty = True
        if hasattr(self, "btn_save_preset"):
            self.btn_save_preset.setEnabled(True)

    def _save_preset(self):
        try:
            cfg = self._collect_config()
        except Exception as e:
            QMessageBox.warning(self, "Can't save preset", str(e))
            return
        default = suggest_preset_name(getattr(self, "_loaded_abbr", "Arena"))
        name, ok = QInputDialog.getText(
            self, "Save arena preset", "Preset name:", text=default)
        if not ok or not name.strip():
            return
        save_user_preset(name.strip(), cfg)
        self._arena_dirty = False
        self.btn_save_preset.setEnabled(False)
        self.statusBar().showMessage(f"Saved preset: {name.strip()}", 4000)

    def _arm(self, kind, btn):
        for k in ("nest", "food", "water"):
            b = getattr(self, f"_arm_btn_{k}")
            if b is not btn:
                b.setChecked(False)
        for v in self._views():
            v.arm_add(kind if btn.isChecked() else None)

    def _on_canvas_add(self, obj):
        self._add_obj_row(obj)
        self._refresh_arena()

    # ------------------------------------------------------------------ #
    # Tab 2: Population
    # ------------------------------------------------------------------ #
    def _build_population_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)
        hint = QLabel("Build one or more agent types, then add copies to the "
                      "arena. Each type has its own look, movement, and biology.")
        hint.setWordWrap(True)
        lay.addWidget(hint)
        self._agent_types = []
        self.cards_layout = QVBoxLayout()
        self.cards_layout.setContentsMargins(0, 0, 0, 0)
        self.cards_layout.setSpacing(6)
        lay.addLayout(self.cards_layout)
        row = QHBoxLayout()
        b_add = QPushButton("＋  Add Agent Type")
        b_add.setObjectName("accept_btn")
        b_add.clicked.connect(self._add_agent_type)
        b_ex = QPushButton("Load example")
        b_ex.clicked.connect(self._load_vole_default)
        row.addWidget(b_add, 1)
        row.addWidget(b_ex)
        lay.addLayout(row)
        self.pop_summary = QLabel()
        self.pop_summary.setStyleSheet("color: #8fbfff; font-size: 12px;")
        lay.addWidget(self.pop_summary)
        return w

    def _load_vole_default(self):
        self._load_config(default_vole_experiment())

    # ---- agent-type cards ---------------------------------------------- #
    def _render_agent_cards(self):
        while self.cards_layout.count():
            it = self.cards_layout.takeAt(0)
            if it.widget():
                it.widget().deleteLater()
        for i, g in enumerate(self._agent_types):
            self.cards_layout.addWidget(self._agent_card(i, g))
        self._update_pop_summary()
        self._rebuild_preview()

    def _agent_card(self, i, g):
        card = QFrame()
        card.setObjectName("agent_card")
        h = QHBoxLayout(card)
        h.setContentsMargins(8, 6, 8, 6)
        sw = QLabel()
        sw.setFixedSize(18, 18)
        col = g.appearance.color or ("#4a90d9" if g.sex == "M" else "#e0559a")
        sw.setStyleSheet(f"background:{col}; border-radius:9px;")
        h.addWidget(sw)
        info = QLabel(f"<b>{g.label}</b>  ·  {g.sex}×{g.count}  ·  "
                      f"{g.appearance.shape}  ·  {g.species}")
        h.addWidget(info, 1)
        for txt, w_, cb in [("Edit", 52, lambda: self._edit_agent_type(i)),
                            ("⧉", 30, lambda: self._dup_agent_type(i)),
                            ("✕", 30, lambda: self._remove_agent_type(i))]:
            b = QPushButton(txt)
            b.setMaximumWidth(w_)
            b.clicked.connect(cb)
            h.addWidget(b)
        return card

    def _add_agent_type(self):
        dlg = AgentBuilderDialog(parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self._agent_types.append(dlg.result_group())
            self._render_agent_cards()

    def _edit_agent_type(self, i):
        dlg = AgentBuilderDialog(self._agent_types[i], parent=self)
        if dlg.exec_() == QDialog.Accepted:
            self._agent_types[i] = dlg.result_group()
            self._render_agent_cards()

    def _dup_agent_type(self, i):
        self._agent_types.insert(i + 1, copy.deepcopy(self._agent_types[i]))
        self._render_agent_cards()

    def _remove_agent_type(self, i):
        del self._agent_types[i]
        self._render_agent_cards()

    # ------------------------------------------------------------------ #
    # Tab 3: Experiment
    # ------------------------------------------------------------------ #
    # fidelity presets -> (dt seconds, record interval seconds)
    _FIDELITY = {"Fast": (5.0, 30.0), "Balanced": (2.0, 10.0),
                 "Fine": (0.5, 2.0)}

    def _build_experiment_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        lay.setContentsMargins(0, 0, 0, 0)

        # ---- essentials (what everyone sets) ----
        ess = QFormLayout()
        ess.setLabelAlignment(Qt.AlignLeft)
        self.in_name = QLineEdit("experiment")
        self.in_name.setToolTip("Names the output project folder")
        self.in_days = _dspin(0.05, 365, 10.0, 0.5, " days")
        self.in_trials = _ispin(1, 100, 1)
        self.in_trials.setToolTip("Replicate chambers, shown side-by-side")
        self.in_fidelity = QComboBox()
        self.in_fidelity.addItems(["Fast", "Balanced", "Fine", "Custom"])
        self.in_fidelity.setToolTip(
            "Simulation resolution: Fast = coarse & quick, Fine = smooth & slow")
        self.in_fidelity.currentTextChanged.connect(self._apply_fidelity)
        ess.addRow("Experiment name", self.in_name)
        ess.addRow("Duration", self.in_days)
        ess.addRow("Replicates", self.in_trials)
        ess.addRow("Resolution", self.in_fidelity)
        lay.addLayout(ess)

        # ---- protocol timeline (timed add/remove of animals & resources) ----
        proto_box = QGroupBox("Experiment protocol")
        pl = QVBoxLayout(proto_box)
        proto_hint = QLabel(
            "Mid-experiment manipulations: introduce or trap out animals, "
            "place or remove food/water. Click the timeline to schedule one.")
        proto_hint.setWordWrap(True)
        proto_hint.setStyleSheet("color:#8a9099; font-size:11px;")
        pl.addWidget(proto_hint)
        self.timeline = ProtocolTimeline(self._protocol_context)
        pl.addWidget(self.timeline)
        prow = QHBoxLayout()
        b_ev = QPushButton("＋ Schedule manipulation…")
        b_ev.clicked.connect(
            lambda: self.timeline._add_event(self.in_days.value() / 2))
        prow.addWidget(b_ev)
        prow.addStretch()
        pl.addLayout(prow)
        lay.addWidget(proto_box)
        self.in_days.valueChanged.connect(self.timeline.set_days)

        # ---- advanced knobs (collapsed by default) ----
        self.in_dt = _dspin(0.1, 60, 2.0, 0.5, " s")
        self.in_rec = _dspin(1, 3600, 10.0, 1.0, " s")
        self.in_seed = _ispin(0, 10 ** 6, 0)
        self.in_daystart = _dspin(0, 23.9, 6.0, 0.5, " h")
        self.in_dayact = _dspin(0, 3, 0.7, 0.1)
        self.in_nightact = _dspin(0, 3, 1.3, 0.1)
        self.in_start = QLineEdit("2025-11-07T18:00:00")
        self.in_variation = _dspin(0, 1, 0.0, 0.05)
        self.in_mortality = QCheckBox("Enable starvation mortality")
        self.in_analyze = QCheckBox("Compute socio-spatial analysis after run")
        self.in_analyze.setChecked(True)
        self.in_parallel = QCheckBox("Run replicates in parallel (no live view)")
        self.in_workers = _ispin(1, 16, 2)
        self.in_espeed = _dspin(0, 1, 0.6, 0.1)
        self.in_rest = _dspin(0, 1, 0.15, 0.05)
        # editing dt/record by hand switches Resolution to Custom
        self.in_dt.valueChanged.connect(self._mark_custom_fidelity)
        self.in_rec.valueChanged.connect(self._mark_custom_fidelity)
        adv_w = QWidget()
        adv = QFormLayout(adv_w)
        adv.setContentsMargins(4, 4, 4, 4)
        for lab, wdg, tip in [
            ("Timestep (dt)", self.in_dt, "Integration step (s) — set by Resolution"),
            ("Record interval", self.in_rec, "Seconds between logged positions"),
            ("Random seed", self.in_seed, "Base seed; replicate i uses seed+i"),
            ("Individual variation", self.in_variation,
             "Per-agent trait jitter SD (0 = identical clones)"),
            ("Day start hour", self.in_daystart, "Local hour the light phase starts"),
            ("Day activity ×", self.in_dayact, "Movement multiplier during day"),
            ("Night activity ×", self.in_nightact, "Movement multiplier at night"),
            ("Release datetime", self.in_start, "ISO timestamp for t=0"),
            ("Energy→speed coupling", self.in_espeed,
             "How much low energy slows movement (0 = speed independent of energy)"),
            ("Rest speed ×", self.in_rest,
             "Speed multiplier when satiated near home (1 = no resting)"),
            ("", self.in_mortality, ""),
            ("", self.in_analyze, ""),
            ("", self.in_parallel, ""),
            ("Parallel workers", self.in_workers, ""),
        ]:
            if tip:
                wdg.setToolTip(tip)
            adv.addRow(lab, wdg)
        lay.addWidget(CollapsibleSection("Advanced ▸ dt, seed, circadian, "
                                         "mortality, parallel", adv_w,
                                         expanded=False))

        # ---- intervention schedule ----
        iv_box = QGroupBox("Intervention schedule")
        ivl = QVBoxLayout(iv_box)
        iv_hint = QLabel(
            "Timed changes to an attribute. target = group label, sexid, or "
            "'all'  ·  op = set / scale / add.\nE.g. induce anosmia on day 3:  "
            "3 · all · smell_ability · scale · 0.")
        iv_hint.setWordWrap(True)
        ivl.addWidget(iv_hint)
        self.iv_table = _table(IV_COLS)
        self.iv_table.horizontalHeader().setSectionResizeMode(
            QHeaderView.Stretch)
        self.iv_table.setMaximumHeight(160)
        ivl.addWidget(self.iv_table)
        ivr = QHBoxLayout()
        b_add = QPushButton("Add intervention")
        b_add.clicked.connect(lambda: self._add_iv_row())
        b_rm = QPushButton("Remove selected")
        b_rm.clicked.connect(lambda: self._del_row(self.iv_table))
        ivr.addWidget(b_add)
        ivr.addWidget(b_rm)
        ivr.addStretch()
        ivl.addLayout(ivr)
        lay.addWidget(iv_box)

        # ---- attribute dynamics (interaction table) ----
        dyn_body = QWidget()
        dynl = QVBoxLayout(dyn_body)
        dynl.setContentsMargins(2, 2, 2, 2)
        dyn_hint = QLabel(
            "How condition variables interact. Each rule: target += gain × "
            "source per hour (or effect=set).\nsources: time · movement · "
            "on_food · on_water · crowding · fed · hydrated · mass · metabolism · "
            "energy/hunger/thirst/stress/health.  targets: energy · hunger · "
            "thirst · stress · health.  scale_by: none/mass/activity/metabolism.  "
            "when: always / source_high / source_low (vs threshold).")
        dyn_hint.setWordWrap(True)
        dyn_hint.setStyleSheet("color:#8a9099; font-size:10px;")
        dynl.addWidget(dyn_hint)
        self.dyn_table = _table(DYN_COLS)
        self.dyn_table.setMinimumHeight(150)
        dynl.addWidget(self.dyn_table)
        dynr = QHBoxLayout()
        for label, cb in [("Add rule", lambda: self._add_dyn_row()),
                          ("Remove selected",
                           lambda: self._del_row(self.dyn_table)),
                          ("Reset to default", self._reset_dynamics)]:
            b = QPushButton(label)
            b.clicked.connect(cb)
            dynr.addWidget(b)
        dynr.addStretch()
        dynl.addLayout(dynr)
        lay.addWidget(CollapsibleSection(
            "Attribute dynamics ▸ how energy, hunger, health… interact",
            dyn_body, expanded=False))

        out_box = QGroupBox("Output")
        ob = QHBoxLayout(out_box)
        self.in_outdir = QLineEdit(
            os.path.join(os.path.expanduser("~"), "ABMA_projects"))
        b = QPushButton("Browse…")
        b.clicked.connect(self._pick_outdir)
        ob.addWidget(QLabel("Project folder:"))
        ob.addWidget(self.in_outdir, 1)
        ob.addWidget(b)
        lay.addWidget(out_box)
        lay.addStretch(1)
        return w

    def _apply_fidelity(self, name):
        preset = self._FIDELITY.get(name)
        if not preset:
            return
        for wdg, val in ((self.in_dt, preset[0]), (self.in_rec, preset[1])):
            wdg.blockSignals(True)
            wdg.setValue(val)
            wdg.blockSignals(False)
        self._rebuild_preview()

    def _mark_custom_fidelity(self, *_):
        if (self.in_dt.value(), self.in_rec.value()) not in self._FIDELITY.values():
            self.in_fidelity.blockSignals(True)
            self.in_fidelity.setCurrentText("Custom")
            self.in_fidelity.blockSignals(False)

    def _set_fidelity_combo(self):
        cur = (self.in_dt.value(), self.in_rec.value())
        name = next((k for k, v in self._FIDELITY.items() if v == cur), "Custom")
        self.in_fidelity.blockSignals(True)
        self.in_fidelity.setCurrentText(name)
        self.in_fidelity.blockSignals(False)

    def _pick_outdir(self):
        d = QFileDialog.getExistingDirectory(self, "Choose project parent folder",
                                             self.in_outdir.text())
        if d:
            self.in_outdir.setText(d)

    def _add_dyn_row(self, c: Coupling | None = None):
        if not isinstance(c, Coupling):
            c = Coupling()
        t = self.dyn_table
        r = t.rowCount()
        t.insertRow(r)
        for i, v in enumerate([c.source, c.target, c.effect, c.gain, c.scale_by,
                               c.only_when, c.threshold, c.note]):
            t.setItem(r, i, QTableWidgetItem(str(v)))

    def _reset_dynamics(self):
        self.dyn_table.setRowCount(0)
        for c in default_dynamics():
            self._add_dyn_row(c)

    def _dynamics_from_table(self) -> list[Coupling]:
        out = []
        t = self.dyn_table
        for r in range(t.rowCount()):
            def cell(c, default=""):
                it = t.item(r, c)
                return it.text().strip() if it and it.text() else default
            if not cell(0) or not cell(1):
                continue
            try:
                out.append(Coupling(
                    source=cell(0, "time"), target=cell(1, "energy"),
                    effect=cell(2, "rate").lower(), gain=float(cell(3, "0")),
                    scale_by=cell(4, "none").lower(),
                    only_when=cell(5, "always").lower(),
                    threshold=float(cell(6, "0.5")), note=cell(7)))
            except ValueError as e:
                raise ValueError(f"Dynamics row {r + 1}: {e}")
        return out

    def _add_iv_row(self, iv: Intervention | None = None):
        if not isinstance(iv, Intervention):
            iv = Intervention()
        t = self.iv_table
        r = t.rowCount()
        t.insertRow(r)
        for c, v in enumerate([iv.at_day, iv.target, iv.attribute, iv.op,
                               iv.value]):
            t.setItem(r, c, QTableWidgetItem(str(v)))

    def _interventions_from_table(self) -> list[Intervention]:
        ivs = []
        t = self.iv_table
        for r in range(t.rowCount()):
            def cell(c, default=""):
                it = t.item(r, c)
                return it.text().strip() if it and it.text() else default
            if not cell(2):
                continue
            try:
                ivs.append(Intervention(
                    at_day=float(cell(0, "0")), target=cell(1, "all"),
                    attribute=cell(2), op=cell(3, "scale").lower(),
                    value=float(cell(4, "0"))))
            except ValueError as e:
                raise ValueError(f"Intervention row {r+1}: {e}")
        return ivs

    # ------------------------------------------------------------------ #
    # Tab 4: Run
    # ------------------------------------------------------------------ #
    def _build_run_tab(self):
        w = QWidget()
        left = QVBoxLayout(w)
        left.setContentsMargins(0, 0, 0, 0)
        self.log = QTextEdit()
        self.log.setReadOnly(True)
        self.log.setMinimumHeight(130)
        left.addWidget(self.log, 1)
        br = QHBoxLayout()
        self.btn_open = QPushButton("Open output folder")
        self.btn_open.clicked.connect(self._open_output)
        self.btn_open.setEnabled(False)
        br.addWidget(self.btn_open)
        b_an = QPushButton("Analyze existing project…")
        b_an.clicked.connect(self._analyze_existing)
        br.addWidget(b_an)
        left.addLayout(br)
        note = QLabel(
            "Output: data/uwb_<trial>_processed.csv matches FNT UWB format — "
            "analyse it with the UWB Proximity/Network tools or your R pipeline. "
            "During a run, hover an agent in the preview to inspect it live.")
        note.setWordWrap(True)
        note.setStyleSheet("color:#8a9099; font-size:11px;")
        left.addWidget(note)
        left.addStretch()
        return w

    def _select_agent(self, idx):
        """Click handler: a hit pins the popup to that agent; -1 unpins/hides."""
        if idx is None or idx < 0:
            self._inspector_pinned = False
            self._hover_idx = None
            self.inspector.hide()
            for v in self._views():
                v.set_selected(None)
        else:
            self._inspector_pinned = True
            self._hover_idx = idx
            for v in self._views():
                v.set_selected(idx)
            self.inspector.select(idx)
            self._show_inspector_popup()
        if self._last_frame is not None:
            self._push_frame(self._active_view(), self._last_frame)

    def _on_agent_hover(self, idx):
        """Hover handler: show the popup for the agent under the cursor."""
        if self._inspector_pinned:
            return
        if idx is not None and idx >= 0:
            if idx != self._hover_idx:
                self._hover_idx = idx
                for v in self._views():
                    v.set_selected(idx)
                self.inspector.select(idx)
                self._show_inspector_popup()
                if self._last_frame is not None:
                    self._push_frame(self._active_view(), self._last_frame)
        elif self._hover_idx is not None:
            self._hover_idx = None
            self.inspector.hide()
            for v in self._views():
                v.set_selected(None)
            if self._last_frame is not None:
                self._push_frame(self._active_view(), self._last_frame)

    def _show_inspector_popup(self):
        from PyQt5.QtGui import QCursor
        p = QCursor.pos()
        self.inspector.move(p.x() + 18, p.y() + 12)
        self.inspector.show()
        self.inspector.raise_()
        if self._last_frame is not None:
            self._update_inspector_dynamic(self._last_frame)

    def _push_frame(self, view, fr):
        view.update_agents(
            fr["x"], fr["y"], fr["sex_m"], heading=fr.get("heading"),
            day=fr.get("day"), hour=fr.get("hour"), is_day=fr.get("is_day"),
            alive=fr.get("alive"), colors=fr.get("color"), sizes=fr.get("size"),
            shapes=fr.get("shape"))

    # ------------------------------------------------------------------ #
    # Config <-> UI
    # ------------------------------------------------------------------ #
    def _protocol_context(self) -> dict:
        """Live context handed to the timeline's event editor."""
        labels = []
        try:
            labels = [o.label for o in self._arena_from_table().objects
                      if o.label]
        except Exception:
            pass
        # resources scheduled for addition are also removable later
        for ev in self.timeline.protocol() if hasattr(self, "timeline") else []:
            if ev.kind == "add_resource" and ev.object is not None \
                    and ev.object.label and ev.object.label not in labels:
                labels.append(ev.object.label)
        return {
            "groups": self._agent_types,
            "resource_labels": labels,
            "arena_wh": (self.in_width.value(), self.in_height.value()),
            "days": self.in_days.value(),
        }

    def _collect_config(self) -> ExperimentConfig:
        arena = self._arena_from_table()
        groups = [g for g in self._agent_types]
        if not groups:
            raise ValueError("Add at least one agent type "
                             "(Build & Add Agents → + Add Agent Type).")
        return ExperimentConfig(
            name=self.in_name.text().strip() or "experiment",
            arena=arena, groups=groups,
            protocol=copy.deepcopy(self.timeline.protocol()),
            interventions=self._interventions_from_table(),
            dynamics=self._dynamics_from_table(),
            days=self.in_days.value(), dt=self.in_dt.value(),
            record_interval=self.in_rec.value(),
            n_trials=self.in_trials.value(), seed=self.in_seed.value(),
            day_start_hour=self.in_daystart.value(),
            day_activity=self.in_dayact.value(),
            night_activity=self.in_nightact.value(),
            start_datetime=self.in_start.text().strip(),
            individual_variation=self.in_variation.value(),
            enable_mortality=self.in_mortality.isChecked(),
            energy_speed_coupling=self.in_espeed.value(),
            rest_speed_factor=self.in_rest.value(),
            parallel=self.in_parallel.isChecked(),
            n_workers=self.in_workers.value(),
            trial_prefix="S",
            policy=copy.deepcopy(self._policy),
        )

    def _load_config(self, cfg: ExperimentConfig):
        self._loading = True          # suppress dirty-tracking during populate
        # policy params have no editor widgets (yet) — carry them through so a
        # tuned config's movement weights survive the GUI round-trip
        self._policy = copy.deepcopy(cfg.policy)
        # arena attributes without a table cell ride along on the window
        self._zones = list(cfg.arena.zones)
        self._poles = list(cfg.arena.poles)
        self._water_towers = list(cfg.arena.water_towers)
        self._resource_zones = list(cfg.arena.resource_zones)
        self._huts = list(cfg.arena.huts)
        self._resource_state = -1       # resources start hidden on load
        self._antennas = list(cfg.arena.antennas)
        self._antenna_layouts = list(cfg.arena.antenna_layouts)
        self._antenna_state = -1        # antennas start hidden on load
        self._ground = cfg.arena.ground
        self._grass_spec = cfg.arena.grass
        self._oriented = cfg.arena.oriented
        self._wall_height = cfg.arena.wall_height
        self._wall_thickness = cfg.arena.wall_thickness
        self.in_name.setText(cfg.name)
        self.in_width.setValue(cfg.arena.width)
        self.in_height.setValue(cfg.arena.height)
        self.in_boundary.setCurrentText(cfg.arena.boundary)
        self.obj_table.blockSignals(True)
        self.obj_table.setRowCount(0)
        for o in cfg.arena.objects:
            self._add_obj_row(o)
        self.obj_table.blockSignals(False)
        self._agent_types = [copy.deepcopy(g) for g in cfg.groups]
        self._render_agent_cards()
        if hasattr(self, "timeline"):
            self.timeline.set_protocol(
                copy.deepcopy(getattr(cfg, "protocol", []) or []), cfg.days)
        if hasattr(self, "iv_table"):
            self.iv_table.setRowCount(0)
            for iv in cfg.interventions:
                self._add_iv_row(iv)
        if hasattr(self, "dyn_table"):
            self.dyn_table.setRowCount(0)
            for c in cfg.dynamics:
                self._add_dyn_row(c)
        self.in_days.setValue(cfg.days)
        self.in_dt.setValue(cfg.dt)
        self.in_rec.setValue(cfg.record_interval)
        self._set_fidelity_combo()
        self.in_trials.setValue(cfg.n_trials)
        self.in_seed.setValue(cfg.seed)
        self.in_daystart.setValue(cfg.day_start_hour)
        self.in_dayact.setValue(cfg.day_activity)
        self.in_nightact.setValue(cfg.night_activity)
        self.in_start.setText(cfg.start_datetime)
        self.in_variation.setValue(cfg.individual_variation)
        self.in_mortality.setChecked(cfg.enable_mortality)
        self.in_espeed.setValue(cfg.energy_speed_coupling)
        self.in_rest.setValue(cfg.rest_speed_factor)
        self._refresh_arena()
        self._update_pop_summary()
        self._loading = False
        # a freshly loaded arena is not yet "modified"
        self._arena_dirty = False
        if hasattr(self, "btn_save_preset"):
            self.btn_save_preset.setEnabled(False)

    # ---- arena table helpers ---- #
    def _add_obj_row(self, o: ResourceObject):
        t = self.obj_table
        r = t.rowCount()
        t.insertRow(r)
        for c, val in enumerate([o.kind, o.x, o.y, o.radius, o.label]):
            t.setItem(r, c, QTableWidgetItem(str(val)))

    def _arena_from_table(self) -> ArenaConfig:
        objs = []
        t = self.obj_table
        for r in range(t.rowCount()):
            try:
                kind = t.item(r, 0).text().strip()
                x = float(t.item(r, 1).text())
                y = float(t.item(r, 2).text())
                rad = float(t.item(r, 3).text())
                lbl = t.item(r, 4).text() if t.item(r, 4) else ""
            except (AttributeError, ValueError):
                continue
            objs.append(ResourceObject(kind, x, y, rad, label=lbl))
        return ArenaConfig(
            width=self.in_width.value(), height=self.in_height.value(),
            boundary=self.in_boundary.currentText(), objects=objs,
            zones=list(getattr(self, "_zones", [])),
            poles=list(getattr(self, "_poles", [])),
            water_towers=list(getattr(self, "_water_towers", [])),
            resource_zones=list(getattr(self, "_resource_zones", [])),
            huts=list(getattr(self, "_huts", [])),
            antennas=list(getattr(self, "_antennas", [])),
            antenna_layouts=list(getattr(self, "_antenna_layouts", [])),
            ground=getattr(self, "_ground", "floor"),
            grass=getattr(self, "_grass_spec", None) or GrassSpec(),
            oriented=getattr(self, "_oriented", False),
            wall_height=getattr(self, "_wall_height", 0.0),
            wall_thickness=getattr(self, "_wall_thickness", 0.005))

    def _refresh_arena(self, *_):
        try:
            arena = self._arena_from_table()
        except Exception:
            return
        for v in self._views():
            v.set_arena(arena)
        if hasattr(self, "arena_summary"):
            n = len(arena.objects_of("nest"))
            f = len(arena.objects_of("food"))
            wt = len(arena.objects_of("water"))
            npoles = len(getattr(arena, "poles", []))
            pole_txt = f", {npoles} poles" if npoles else ""
            self.arena_summary.setText(
                f"Arena {arena.width:g}×{arena.height:g} {arena.units} · "
                f"{arena.boundary} · {n} nests, {f} food, {wt} water{pole_txt}")
        if hasattr(self, "btn_grass"):
            grassy = getattr(arena, "ground", "floor") == "grass"
            self.btn_grass.setEnabled(grassy)
            if not grassy and self.btn_grass.isChecked():
                self.btn_grass.setChecked(False)
        if hasattr(self, "btn_resources"):
            has_res = bool(getattr(arena, "water_towers", [])
                           or getattr(arena, "resource_zones", []))
            self.btn_resources.setEnabled(has_res)
            if not has_res:
                self._resource_state = -1
            self._apply_resource_state()
        if hasattr(self, "btn_antenna"):
            sets = arena.antenna_sets() if hasattr(arena, "antenna_sets") else []
            self.btn_antenna.setEnabled(bool(sets))
            if self._antenna_state >= 2 * len(sets):     # 2 steps per layout
                self._antenna_state = -1
            self._apply_antenna_state()
        self._rebuild_preview()

    def _update_pop_summary(self, *_):
        if not hasattr(self, "pop_summary"):
            return
        groups = getattr(self, "_agent_types", [])
        nf = sum(g.count for g in groups if g.sex == "F")
        nm = sum(g.count for g in groups if g.sex == "M")
        treated = sum(g.count for g in groups
                      if g.treatment.drug not in ("none", "saline"))
        self.pop_summary.setText(
            f"Total: {nf + nm} agents ({nf}F, {nm}M) · {len(groups)} type(s)"
            + (f" · {treated} drug-treated" if treated else ""))

    def _del_row(self, table):
        r = table.currentRow()
        if r >= 0:
            table.removeRow(r)
        if table is self.obj_table:
            self._refresh_arena()

    # ------------------------------------------------------------------ #
    # Run control
    # ------------------------------------------------------------------ #
    def _validate(self, cfg) -> tuple[list[str], str]:
        """Return (warnings, human-readable pre-run summary)."""
        warn = []
        if not cfg.arena.objects_of("food"):
            warn.append("No food source — enable mortality only with care.")
        if not cfg.arena.objects_of("water"):
            warn.append("No water source.")
        if cfg.record_interval < cfg.dt:
            warn.append(f"Record interval ({cfg.record_interval}s) < timestep "
                        f"({cfg.dt}s); positions logged every step.")
        if cfg.enable_mortality and not cfg.arena.objects_of("food"):
            warn.append("Mortality ON with no food: agents will starve.")
        for ev in getattr(cfg, "protocol", []):
            if ev.at_day > cfg.days:
                warn.append(f"Protocol event after the experiment ends "
                            f"({describe_event(ev)}, duration {cfg.days:g} d).")
            if ev.kind == "add_agents" and ev.group is None:
                warn.append(f"Protocol event has no agent type "
                            f"({describe_event(ev)}).")
        n = cfg.total_agents()
        samples = int(cfg.days * 86400 / cfg.record_interval)
        est_rows = samples * n * cfg.n_trials
        if est_rows > 5_000_000:
            warn.append(f"Large output (~{est_rows/1e6:.1f}M rows). Consider a "
                        "coarser record interval or fewer days.")
        nf = sum(g.count for g in cfg.groups if g.sex == "F")
        nm = sum(g.count for g in cfg.groups if g.sex == "M")
        info = (f"<b>{cfg.name}</b><br>"
                f"{n} agents ({nf}F / {nm}M) · {len(cfg.groups)} groups<br>"
                f"{cfg.n_trials} trial(s) × {cfg.days:g} days<br>"
                f"arena {cfg.arena.width:g}×{cfg.arena.height:g} {cfg.arena.units}"
                f" · ~{est_rows:,} trajectory rows total")
        return warn, info

    def _on_run(self):
        try:
            cfg = self._collect_config()
        except Exception as e:
            QMessageBox.warning(self, "Invalid configuration", str(e))
            return
        warn, info = self._validate(cfg)
        msg = info
        if warn:
            msg += "<br><br><b>Warnings:</b><br>• " + "<br>• ".join(warn)
        box = QMessageBox(self)
        box.setWindowTitle("Run experiment?")
        box.setTextFormat(Qt.RichText)
        box.setText(msg)
        box.setStandardButtons(QMessageBox.Ok | QMessageBox.Cancel)
        box.button(QMessageBox.Ok).setText("Run")
        if box.exec_() != QMessageBox.Ok:
            return

        # Runs are append-only: each execution gets its own folder inside the
        # project, so history is never overwritten and every run stays
        # reproducible from its own config.json.
        if self.project is not None:
            self.project.config = cfg          # keep the working config current
            self.project.save()
            project_dir = self.project.new_run_dir()
        else:
            parent = self.in_outdir.text().strip() or os.getcwd()
            project_dir = os.path.join(parent, cfg.name)
            if os.path.exists(os.path.join(project_dir, "data")):
                if QMessageBox.question(
                        self, "Overwrite?",
                        f"{project_dir} already has data. Overwrite trials?"
                ) != QMessageBox.Yes:
                    return
        self._project_dir = project_dir
        self._run_cfg = cfg
        self._run_t0 = time.time()
        self._running = True
        self._stop_preview()             # the real run takes over the views
        self._last_frame = None
        self._frames = []                # fresh buffer for this run
        self._play_idx = 0
        self._live_follow = True
        self._playing = False
        self._play_timer.stop()
        self.scrubber.blockSignals(True)
        self.scrubber.setRange(0, 0)
        self.scrubber.blockSignals(False)
        self.btn_play.setText("⏸")
        self._inspector_pinned = False
        self._hover_idx = None
        self.inspector.hide()
        # lay replicate chambers out in a grid so they can be watched together
        offsets = grid_offsets(cfg.n_trials, cfg.arena.width, cfg.arena.height)
        for v in self._views():
            v.clear_playback()
            v.set_arena(cfg.arena, chambers=offsets)
            v.set_selected(None)
        self.inspector.clear_selection()
        self.log.clear()
        self.progress.setValue(0)
        self.status_label.setText("Starting…")
        self.btn_run.setEnabled(False)
        self.btn_stop.setEnabled(True)
        self.worker = ABMARunWorker(cfg, project_dir,
                                    analyze=self.in_analyze.isChecked())
        self.worker.progress.connect(self._on_progress)
        self.worker.frame.connect(self._on_frame)
        self.worker.agents.connect(self.inspector.set_population)
        self.worker.log.connect(self._append_log)
        self.worker.done.connect(self._on_done)
        self.worker.failed.connect(self._on_failed)
        self.worker.start()

    def _on_progress(self, frac):
        self.progress.setValue(int(frac * 100))
        if self._run_t0 and frac > 0.01:
            elapsed = time.time() - self._run_t0
            eta = elapsed * (1 - frac) / frac
            self.status_label.setText(
                f"{int(frac*100)}% · elapsed {_mmss(elapsed)} · "
                f"ETA {_mmss(eta)}")

    def _on_frame(self, fr):
        # a run frame — buffer it so it can be scrubbed / replayed
        self._frames.append(fr)
        self.scrubber.blockSignals(True)
        self.scrubber.setRange(0, max(0, len(self._frames) - 1))
        if self._live_follow:
            self._play_idx = len(self._frames) - 1
            self.scrubber.setValue(self._play_idx)
        self.scrubber.blockSignals(False)
        if self._live_follow:
            self._display_frame(fr)

    def _render_live(self, fr):
        """Live editor-preview frame (not buffered — nothing to review)."""
        self._display_frame(fr)

    def _render_frame(self, idx):
        if 0 <= idx < len(self._frames):
            self._display_frame(self._frames[idx])

    def _display_frame(self, fr):
        self._last_frame = fr
        self._push_frame(self._active_view(), fr)
        sel = self.inspector.selected_index()
        if self._follow_agent and sel is not None and sel < len(fr["x"]):
            self._active_view().center_on(fr["x"][sel], fr["y"][sel])
        if self.inspector.isVisible():
            self._update_inspector_dynamic(fr)
        d, h = fr.get("day"), fr.get("hour")
        if d is not None and h is not None:
            self.lbl_time.setText(f"day {d} · {int(h):02d}:00")

    def _on_play_pause(self):
        if self._running:                       # live run: toggle follow vs scrub
            self._live_follow = not self._live_follow
            self.btn_play.setText("⏸" if self._live_follow else "▶")
            if self._live_follow and self._frames:
                self._play_idx = len(self._frames) - 1
                self.scrubber.blockSignals(True)
                self.scrubber.setValue(self._play_idx)
                self.scrubber.blockSignals(False)
                self._render_frame(self._play_idx)
        elif self._frames:                      # finished run: replay the buffer
            if self._playing:
                self._playing = False
                self._play_timer.stop()
                self.btn_play.setText("▶")
            else:
                self._playing = True
                if self._play_idx >= len(self._frames) - 1:
                    self._play_idx = 0
                self._play_timer.start(50)
                self.btn_play.setText("⏸")
        else:                                   # live editor preview
            if self._preview_timer.isActive():
                self._preview_timer.stop()
                self.btn_play.setText("▶")
            elif self._preview_sim is not None:
                self._preview_timer.start(self._preview_interval)
                self.btn_play.setText("⏸")

    def _play_tick(self):
        if not self._frames:
            self._play_timer.stop()
            return
        self._play_idx = min(self._play_idx + max(1, int(round(self._play_speed))),
                             len(self._frames) - 1)
        self.scrubber.blockSignals(True)
        self.scrubber.setValue(self._play_idx)
        self.scrubber.blockSignals(False)
        self._render_frame(self._play_idx)
        if self._play_idx >= len(self._frames) - 1:
            self._playing = False
            self._play_timer.stop()
            self.btn_play.setText("▶")

    def _on_scrub(self, idx):
        self._live_follow = False
        self._playing = False
        self._play_timer.stop()
        self.btn_play.setText("▶")
        self._play_idx = idx
        self._render_frame(idx)

    def _update_inspector_dynamic(self, fr):
        idx = self.inspector.selected_index()
        if idx is None or idx >= len(fr["x"]):
            return
        self.inspector.update_dynamic({
            "health": float(fr["health"][idx]),
            "energy": float(fr["energy"][idx]),
            "hunger": float(fr["hunger"][idx]),
            "thirst": float(fr["thirst"][idx]),
            "stress": float(fr["stress"][idx]),
            "mass": float(fr["mass"][idx]),
            "activity": int(fr["activity"][idx]),
            "anosmic": bool(fr["anosmic"][idx]),
            "estrus": bool(fr["estrus"][idx]),
            "alive": bool(fr["alive"][idx]),
            "fights_won": int(fr["fights_won"][idx]),
            "fights_lost": int(fr["fights_lost"][idx]),
            "matings": int(fr["matings"][idx]),
            "dist_today": float(fr["dist_today"][idx]),
        })

    def _append_log(self, msg):
        self.log.append(msg)

    def _on_done(self, results):
        self._append_log(f"\n✓ Completed {len(results)} trial(s).")
        self.progress.setValue(100)
        elapsed = time.time() - self._run_t0 if self._run_t0 else 0
        self.status_label.setText(
            f"Done · {len(results)} trial(s) · {_mmss(elapsed)}")
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self.btn_open.setEnabled(True)
        self._running = False
        # keep the frame buffer so the run can be scrubbed / replayed
        self._live_follow = False
        self._playing = False
        self.btn_play.setText("▶")
        if self._frames:
            self._append_log("Playback ready — scrub the timeline or press ▶ to "
                             "replay. Edit the setup to return to live preview.")
        else:
            self._rebuild_preview()

    def _analyze_existing(self):
        """Run the built-in analysis on an already-generated project folder."""
        d = QFileDialog.getExistingDirectory(
            self, "Choose an ABMA project folder (contains data/)",
            self.in_outdir.text())
        if not d:
            return
        try:
            from ..core.analysis import analyze_experiment
            self.log.append(f"Analysing {d} …")
            out = analyze_experiment(d, log_cb=self._append_log)
            self._project_dir = d
            self.btn_open.setEnabled(True)
            self.status_label.setText("Analysis complete.")
            QMessageBox.information(self, "Analysis complete",
                                    f"Wrote {os.path.basename(out)}")
        except Exception as e:
            QMessageBox.critical(self, "Analysis failed", str(e))

    def _on_failed(self, tb):
        self._append_log("\n✗ ERROR:\n" + tb)
        self.btn_run.setEnabled(True)
        self.btn_stop.setEnabled(False)
        self._running = False
        self._rebuild_preview()
        QMessageBox.critical(self, "Simulation failed", tb.splitlines()[-1])

    def _on_stop(self):
        # cooperative stop: the run loop breaks and closes files cleanly, then
        # emits done/failed (handled by _on_done/_on_failed). No hard terminate.
        if self.worker and self.worker.isRunning():
            self.worker.cancel()
            self._append_log("Stopping…")
            self.btn_stop.setEnabled(False)

    def _open_output(self):
        d = getattr(self, "_project_dir", None)
        if not d or not os.path.isdir(d):
            return
        if sys.platform == "darwin":
            subprocess.run(["open", d])
        elif sys.platform.startswith("win"):
            os.startfile(d)  # type: ignore[attr-defined]
        else:
            subprocess.run(["xdg-open", d])


# --------------------------------------------------------------------------- #
# Preset picker dialog
# --------------------------------------------------------------------------- #
class ArenaPresetDialog(QDialog):
    """A small window to pick a preset arena/experiment (e.g. the OFT)."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Load Preset Arena")
        self.setMinimumWidth(440)
        self._presets = all_presets()      # built-in + user-saved
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Select a preset to load into the preview:"))
        self.list = QListWidget()
        for p in self._presets:
            self.list.addItem(p.name if p.builtin else f"{p.name}  (saved)")
        self.list.setCurrentRow(0)
        self.list.currentRowChanged.connect(self._on_row)
        self.list.itemDoubleClicked.connect(lambda _: self.accept())
        lay.addWidget(self.list)
        self.desc = QLabel(self._presets[0].description if self._presets else "")
        self.desc.setWordWrap(True)
        self.desc.setStyleSheet("color:#8a9099; font-size:11px; padding:2px;")
        lay.addWidget(self.desc)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Load")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _on_row(self, r):
        if 0 <= r < len(self._presets):
            self.desc.setText(self._presets[r].description)

    def chosen(self):
        r = self.list.currentRow()
        return self._presets[r] if 0 <= r < len(self._presets) else None


# --------------------------------------------------------------------------- #
# Collapsible section (accordion-style step)
# --------------------------------------------------------------------------- #
class StartDialog(QDialog):
    """Title screen: start a new project, or reopen an existing one.

    A project is the unit that carries the world, the population, the protocol
    and every run made from it — so "load project" and "reproduce this
    experiment" are the same action.
    """

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("ABMA")
        self.setMinimumWidth(620)
        self.choice = None                  # ("new", preset) | ("open", path)
        lay = QVBoxLayout(self)
        lay.setSpacing(10)

        title = QLabel("Animal Behavior Modeling Arena")
        title.setStyleSheet("font-size:22px; font-weight:bold; color:#e9edf2;")
        lay.addWidget(title)
        sub = QLabel("Build a world, populate it, run the experiment.")
        sub.setStyleSheet("color:#8a9099;")
        lay.addWidget(sub)

        # ---- new project ---------------------------------------------- #
        newbox = QGroupBox("New project")
        nl = QVBoxLayout(newbox)
        nl.addWidget(QLabel("Start from:"))
        self.preset_list = QListWidget()
        self._presets = all_presets()
        for p in self._presets:
            self.preset_list.addItem(
                p.name if p.builtin else f"{p.name}  (saved)")
        self.preset_list.setCurrentRow(0)
        self.preset_list.setMaximumHeight(120)
        self.preset_list.itemDoubleClicked.connect(lambda _: self._new())
        nl.addWidget(self.preset_list)
        b_new = QPushButton("＋  Create project")
        b_new.setObjectName("accept_btn")
        b_new.clicked.connect(self._new)
        nl.addWidget(b_new)
        lay.addWidget(newbox)

        # ---- open existing --------------------------------------------- #
        openbox = QGroupBox("Open project")
        ol = QVBoxLayout(openbox)
        self.proj_list = QListWidget()
        self._rows = list_projects()
        for r in self._rows:
            self.proj_list.addItem(f"{r['name']}   —   {r['summary']}"
                                   f"   ·   {r['modified'][:10]}")
        if self._rows:
            self.proj_list.setCurrentRow(0)
            self.proj_list.itemDoubleClicked.connect(lambda _: self._open())
        else:
            self.proj_list.addItem("No projects yet — create one above.")
            self.proj_list.setEnabled(False)
        self.proj_list.setMaximumHeight(140)
        ol.addWidget(self.proj_list)
        orow = QHBoxLayout()
        b_open = QPushButton("Open selected")
        b_open.setEnabled(bool(self._rows))
        b_open.clicked.connect(self._open)
        b_browse = QPushButton("Browse…")
        b_browse.clicked.connect(self._browse)
        orow.addWidget(b_open)
        orow.addWidget(b_browse)
        orow.addStretch()
        ol.addLayout(orow)
        lay.addWidget(openbox)

        skip = QPushButton("Skip — just open the editor")
        skip.setToolTip("Work without a project (runs won't be kept in history)")
        skip.clicked.connect(self.reject)
        lay.addWidget(skip)

    def _new(self):
        r = self.preset_list.currentRow()
        if not (0 <= r < len(self._presets)):
            return
        name, ok = QInputDialog.getText(
            self, "New project", "Project name:",
            text=self._presets[r].abbr or "experiment")
        if not ok or not name.strip():
            return
        self.choice = ("new", self._presets[r], name.strip())
        self.accept()

    def _open(self):
        r = self.proj_list.currentRow()
        if 0 <= r < len(self._rows):
            self.choice = ("open", self._rows[r]["path"])
            self.accept()

    def _browse(self):
        d = QFileDialog.getExistingDirectory(
            self, "Select an ABMA project folder", default_root())
        if d:
            self.choice = ("open", d)
            self.accept()


class CornerPickDialog(QDialog):
    """Click the four inner corners of the enclosure on an overhead photo."""

    ORDER = ["SW (south-west)", "SE (south-east)",
             "NE (north-east)", "NW (north-west)"]

    def __init__(self, path, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Mark the enclosure corners")
        self.pts = []                       # full-resolution image pixels
        pix = QPixmap(path)
        self._full = (pix.width(), pix.height())
        shown = pix.scaled(1000, 700, Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self._scale = pix.width() / max(1, shown.width())
        self._base = shown
        lay = QVBoxLayout(self)
        self.hint = QLabel()
        self.hint.setStyleSheet("font-weight:bold; color:#8fbfff;")
        lay.addWidget(self.hint)
        self.view = QLabel()
        self.view.setPixmap(shown)
        self.view.mousePressEvent = self._click
        lay.addWidget(self.view)
        row = QHBoxLayout()
        undo = QPushButton("Undo last")
        undo.clicked.connect(self._undo)
        row.addWidget(undo)
        row.addStretch()
        self.bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        self.bb.accepted.connect(self.accept)
        self.bb.rejected.connect(self.reject)
        row.addWidget(self.bb)
        lay.addLayout(row)
        self._refresh()

    def _refresh(self):
        n = len(self.pts)
        self.bb.button(QDialogButtonBox.Ok).setEnabled(n == 4)
        self.hint.setText("All four marked — press OK." if n == 4 else
                          f"Click corner {n + 1} of 4:  {self.ORDER[n]}")
        pm = QPixmap(self._base)
        p = QPainter(pm)
        p.setPen(QPen(QColor("#ffd23f"), 3))
        for i, (fx, fy) in enumerate(self.pts):
            x, y = fx / self._scale, fy / self._scale
            p.drawEllipse(int(x - 7), int(y - 7), 14, 14)
            p.drawText(int(x + 10), int(y - 10), self.ORDER[i][:2])
        if len(self.pts) > 1:
            for a, b in zip(self.pts, self.pts[1:]):
                p.drawLine(int(a[0] / self._scale), int(a[1] / self._scale),
                           int(b[0] / self._scale), int(b[1] / self._scale))
        p.end()
        self.view.setPixmap(pm)

    def _click(self, ev):
        if len(self.pts) < 4:
            self.pts.append((ev.pos().x() * self._scale,
                             ev.pos().y() * self._scale))
            self._refresh()

    def _undo(self):
        if self.pts:
            self.pts.pop()
            self._refresh()

    def corners(self):
        return list(self.pts)


class CollapsibleSection(QWidget):
    """A titled header that expands/collapses its content widget."""

    def __init__(self, title, content, expanded=True, parent=None):
        super().__init__(parent)
        v = QVBoxLayout(self)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(0)
        self.header = QToolButton()
        self.header.setObjectName("section_header")
        self.header.setText(title)
        self.header.setCheckable(True)
        self.header.setChecked(expanded)
        self.header.setToolButtonStyle(Qt.ToolButtonTextBesideIcon)
        self.header.setArrowType(Qt.DownArrow if expanded else Qt.RightArrow)
        self.header.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Fixed)
        self.header.clicked.connect(self._toggle)
        v.addWidget(self.header)
        self.content = content
        self.content.setVisible(expanded)
        v.addWidget(self.content)

    def _toggle(self):
        vis = self.header.isChecked()
        self.content.setVisible(vis)
        self.header.setArrowType(Qt.DownArrow if vis else Qt.RightArrow)

    def set_expanded(self, expanded: bool):
        self.header.setChecked(expanded)
        self._toggle()


# --------------------------------------------------------------------------- #
# Agent Builder dialog
# --------------------------------------------------------------------------- #
class AgentBuilderDialog(QDialog):
    """Build/edit one agent TYPE: appearance, movement, attributes, biology."""

    def __init__(self, group=None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Agent Builder")
        self.setMinimumWidth(470)
        g = group or AgentGroup(label="agent_type", species="mouse", sex="M",
                                count=1)
        self._color = g.appearance.color

        tabs = QTabWidget()
        # ---- Appearance & basics ----
        ap = QWidget()
        apf = QFormLayout(ap)
        self.in_name = QLineEdit(g.label)
        self.in_species = QLineEdit(g.species)
        self.in_sex = QComboBox()
        self.in_sex.addItems(["M", "F"])
        self.in_sex.setCurrentText(g.sex)
        self.in_count = _ispin(1, 999, g.count)
        self.in_shape = QComboBox()
        self.in_shape.addItems(["rodent", "blob", "bird"])
        self.in_shape.setCurrentText(g.appearance.shape)
        self.btn_color = QPushButton()
        self.btn_color.clicked.connect(self._pick_color)
        self._update_color_btn()
        self.in_size = _dspin(0.2, 5.0, g.appearance.size, 0.1)
        for lab, wdg in [("Type name", self.in_name), ("Species", self.in_species),
                         ("Sex", self.in_sex), ("Count", self.in_count),
                         ("Shape", self.in_shape), ("Colour", self.btn_color),
                         ("Size ×", self.in_size)]:
            apf.addRow(lab, wdg)
        tabs.addTab(ap, "Appearance")

        # ---- Movement ----
        mv = QWidget()
        mvf = QFormLayout(mv)
        self._move_fields = {}
        for lab, tname, tip in [
            ("Base speed (m/s)", "base_speed", "locomotor speed when active"),
            ("Wander", "wander", "random-walk drive — exploratory movement"),
            ("Turn rate", "turn_rate", "heading jitter — path tortuosity"),
            ("Home-range r (m)", "home_range_r", "site-fidelity radius")]:
            e = QLineEdit(str(g.dists.get(tname, getattr(g.traits, tname))))
            e.setToolTip(tip)
            self._move_fields[tname] = e
            mvf.addRow(lab, e)
        tabs.addTab(mv, "Movement")

        # ---- Attributes ----
        at = QWidget()
        atf = QFormLayout(at)
        self._attr_fields = {}
        for lab, tname in [("Mass (g)", "mass"), ("Aggression", "aggression"),
                           ("Boldness", "boldness"), ("Sociability", "sociability"),
                           ("Exploration", "exploration"),
                           ("Smell ability", "smell_ability"),
                           ("Identity signal", "identity_signal"),
                           ("Metabolism", "metabolism")]:
            e = QLineEdit(str(g.dists.get(tname, getattr(g.traits, tname))))
            self._attr_fields[tname] = e
            atf.addRow(lab, e)
        h = QLabel("Fixed value or distribution per animal, e.g. mass N(33,3).")
        h.setStyleSheet("color:#8a9099; font-size:10px;")
        h.setWordWrap(True)
        atf.addRow(h)
        tabs.addTab(at, "Attributes")

        # ---- Biology ----
        bio = QWidget()
        bf = QFormLayout(bio)
        self.in_genes = QLineEdit(
            ";".join(f"{k}:{v}" for k, v in (g.genotype.genes or {}).items()))
        self.in_genes.setPlaceholderText("e.g. OXTR:KO;AVPR1A:HET")
        self.in_drug = QComboBox()
        self.in_drug.addItems(["none", "saline", "methimazole"])
        self.in_drug.setCurrentText(g.treatment.drug)
        self.in_dose = _dspin(0, 1, g.treatment.dose, 0.1)
        self.in_onset = _dspin(-30, 60, g.treatment.day_offset, 1)
        for lab, wdg in [("Genes", self.in_genes), ("Drug", self.in_drug),
                         ("Dose", self.in_dose), ("Onset day", self.in_onset)]:
            bf.addRow(lab, wdg)
        tabs.addTab(bio, "Biology")

        lay = QVBoxLayout(self)
        lay.addWidget(tabs)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Save Agent")
        bb.accepted.connect(self._validate_accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _pick_color(self):
        from PyQt5.QtWidgets import QColorDialog
        init = QColor(self._color) if self._color else QColor("#4a90d9")
        c = QColorDialog.getColor(init, self, "Agent colour")
        if c.isValid():
            self._color = c.name()
            self._update_color_btn()

    def _update_color_btn(self):
        if self._color:
            self.btn_color.setText(self._color)
            self.btn_color.setStyleSheet(
                f"background:{self._color}; color:white;")
        else:
            self.btn_color.setText("auto (by sex) — click to set")
            self.btn_color.setStyleSheet("")

    def _validate_accept(self):
        try:
            self.result_group()
        except ValueError as e:
            QMessageBox.warning(self, "Invalid value", str(e))
            return
        self.accept()

    def result_group(self) -> AgentGroup:
        traits = TraitProfile()
        dists = {}

        def apply(tname, raw):
            spec = parse_spec(raw)
            setattr(traits, tname, spec.mean)
            if spec.kind != "fixed":
                dists[tname] = spec.to_str()

        for tname, e in {**self._move_fields, **self._attr_fields}.items():
            apply(tname, e.text())
        genes = {}
        for tok in self.in_genes.text().split(";"):
            if ":" in tok:
                k, v = tok.split(":")
                genes[k.strip()] = v.strip().upper()
        return AgentGroup(
            label=self.in_name.text().strip() or "agent_type",
            species=self.in_species.text().strip() or "mouse",
            sex=self.in_sex.currentText(), count=self.in_count.value(),
            genotype=Genotype(genes),
            treatment=Treatment(self.in_drug.currentText(),
                                self.in_dose.value(), self.in_onset.value()),
            traits=traits,
            appearance=Appearance(self.in_shape.currentText(), self._color or "",
                                  self.in_size.value()),
            dists=dists)


# --------------------------------------------------------------------------- #
# small widget helpers
# --------------------------------------------------------------------------- #
def _mmss(seconds: float) -> str:
    seconds = int(max(0, seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


def _dspin(lo, hi, val, step, suffix=""):
    s = QDoubleSpinBox()
    s.setRange(lo, hi)
    s.setSingleStep(step)
    s.setValue(val)
    s.setDecimals(3)
    if suffix:
        s.setSuffix(suffix)
    return s


def _ispin(lo, hi, val):
    s = QSpinBox()
    s.setRange(lo, hi)
    s.setValue(val)
    return s


def _table(cols):
    t = QTableWidget(0, len(cols))
    t.setHorizontalHeaderLabels(cols)
    t.setSelectionBehavior(QAbstractItemView.SelectRows)
    t.setEditTriggers(QAbstractItemView.AllEditTriggers)
    # Ignore the table's (wide) width hint so it never stretches the left column;
    # it takes the available width and scrolls horizontally when needed.
    t.setSizePolicy(QSizePolicy.Ignored, QSizePolicy.Preferred)
    t.setHorizontalScrollBarPolicy(Qt.ScrollBarAsNeeded)
    return t


_ACCENT = "#2979ff"
_ARROW_QSS_CACHE = None


def _arrow_qss():
    """Generate up/down chevron PNGs at runtime and return QSS referencing them.

    Reliable where QSS triangles / SVG data-URIs / native arrows fail (macOS).
    """
    global _ARROW_QSS_CACHE
    if _ARROW_QSS_CACHE is not None:
        return _ARROW_QSS_CACHE
    import os
    import tempfile
    d = os.path.join(tempfile.gettempdir(), "abma_icons")
    os.makedirs(d, exist_ok=True)
    paths = {}
    for name, up in (("up", True), ("dn", False)):
        pm = QPixmap(12, 8)
        pm.fill(Qt.transparent)
        p = QPainter(pm)
        p.setRenderHint(QPainter.Antialiasing)
        p.setPen(Qt.NoPen)
        p.setBrush(QColor("#dfe3e8"))
        pts = ([QPoint(6, 1), QPoint(10, 6), QPoint(2, 6)] if up
               else [QPoint(2, 2), QPoint(10, 2), QPoint(6, 7)])
        p.drawPolygon(QPolygon(pts))
        p.end()
        fp = os.path.join(d, f"arrow_{name}.png")
        pm.save(fp, "PNG")
        paths[name] = fp.replace("\\", "/")
    _ARROW_QSS_CACHE = (
        f'QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{ '
        f'image: url("{paths["up"]}"); width: 12px; height: 8px; }}\n'
        f'QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{ '
        f'image: url("{paths["dn"]}"); width: 12px; height: 8px; }}')
    return _ARROW_QSS_CACHE


_DARK_QSS = f"""
QMainWindow, QWidget {{ background: #191b1f; color: #d6d9de;
    font-size: 12px; }}
QLabel {{ color: #c4c8cf; background: transparent; }}
QLabel#preview_title {{ color: #8a9099; font-size: 11px; font-weight: bold;
    padding: 4px 6px; }}

QGroupBox {{ border: 1px solid #303338; border-radius: 6px; margin-top: 12px;
    padding: 10px 8px 8px 8px; background: #1d2024; }}
QGroupBox::title {{ subcontrol-origin: margin; left: 10px; padding: 0 4px;
    color: #7f8790; font-weight: bold; }}
QFrame#agent_card {{ background: #24272c; border: 1px solid #34383e;
    border-radius: 6px; }}
QToolButton#section_header {{ background: #24272c; color: #c4c8cf;
    border: 1px solid #303338; border-radius: 6px; padding: 8px 10px;
    font-weight: bold; font-size: 13px; text-align: left; }}
QToolButton#section_header:hover {{ background: #2b2f35; }}
QFrame#run_bar {{ background: #1d2024; border-top: 1px solid #303338; }}
QFrame#transport {{ background: #1d2024; border: 1px solid #303338;
    border-radius: 6px; }}
QFrame#transport QToolButton {{ background: #2b2f35; color: #e4e7ea;
    border: 1px solid #3a3f46; border-radius: 4px; padding: 3px 7px;
    font-size: 14px; }}
QFrame#transport QToolButton:hover {{ background: #343941; }}
QFrame#transport QToolButton:checked {{ background: {_ACCENT};
    border-color: {_ACCENT}; color: white; }}
QSlider::groove:horizontal {{ height: 4px; background: #34383e;
    border-radius: 2px; }}
QSlider::handle:horizontal {{ background: {_ACCENT}; width: 12px;
    margin: -5px 0; border-radius: 6px; }}
QSlider::sub-page:horizontal {{ background: {_ACCENT}; border-radius: 2px; }}

QTabWidget::pane {{ border: 1px solid #303338; border-radius: 6px;
    background: #1d2024; top: -1px; }}
QTabBar::tab {{ background: transparent; color: #8a9099; padding: 8px 16px;
    margin-right: 2px; border-top-left-radius: 6px; border-top-right-radius: 6px;
    font-weight: bold; }}
QTabBar::tab:hover {{ color: #c4c8cf; }}
QTabBar::tab:selected {{ background: {_ACCENT}; color: white; }}

QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox, QTextEdit {{
    background: #24272c; color: #eceef1; border: 1px solid #34383e;
    border-radius: 5px; padding: 5px 7px;
    selection-background-color: {_ACCENT}; }}
QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus,
QTextEdit:focus {{ border: 1px solid {_ACCENT}; }}
QComboBox::drop-down {{ border: none; width: 20px; }}
QComboBox QAbstractItemView {{ background: #24272c; color: #eceef1;
    selection-background-color: {_ACCENT}; border: 1px solid #34383e; }}

QTableWidget {{ background: #1b1e22; alternate-background-color: #202329;
    color: #dfe2e6; border: 1px solid #303338; border-radius: 6px;
    gridline-color: #2b2f35; selection-background-color: {_ACCENT};
    selection-color: white; }}
QTableWidget {{ }}
QHeaderView::section {{ background: #24272c; color: #9aa0a8;
    border: none; border-right: 1px solid #303338;
    border-bottom: 1px solid #303338; padding: 5px 6px; font-weight: bold; }}
QTableCornerButton::section {{ background: #24272c; border: none; }}

QPushButton {{ background: #2b2f35; color: #e4e7ea; border: 1px solid #3a3f46;
    border-radius: 5px; padding: 6px 12px; }}
QPushButton:hover {{ background: #343941; border-color: #454b53; }}
QPushButton:pressed {{ background: #24272c; }}
QPushButton:checked {{ background: {_ACCENT}; border-color: {_ACCENT};
    color: white; }}
QPushButton#accept_btn {{ background: #2e9e4b; border-color: #2e9e4b;
    color: white; font-weight: bold; }}
QPushButton#accept_btn:hover {{ background: #37b357; }}
QPushButton#reject_btn {{ background: #d1453b; border-color: #d1453b;
    color: white; font-weight: bold; }}
QPushButton#reject_btn:hover {{ background: #e0554b; }}
QPushButton:disabled {{ background: #212429; color: #5a5f66;
    border-color: #2b2f35; }}

QProgressBar {{ border: 1px solid #303338; border-radius: 5px;
    text-align: center; background: #24272c; color: #c4c8cf; height: 16px; }}
QProgressBar::chunk {{ background: {_ACCENT}; border-radius: 4px; }}

QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border; subcontrol-position: top right; width: 18px;
    border-left: 1px solid #34383e; background: #2b2f35;
    border-top-right-radius: 5px; }}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border; subcontrol-position: bottom right; width: 18px;
    border-left: 1px solid #34383e; background: #2b2f35;
    border-bottom-right-radius: 5px; }}
QSpinBox::up-button:hover, QDoubleSpinBox::up-button:hover,
QSpinBox::down-button:hover, QDoubleSpinBox::down-button:hover {{
    background: #343941; }}
/* up/down arrow images are appended at runtime by _arrow_qss() */

QCheckBox {{ color: #c4c8cf; spacing: 6px; }}
QSplitter::handle {{ background: #303338; }}
QMenuBar {{ background: #1d2024; color: #c4c8cf; }}
QMenuBar::item:selected {{ background: {_ACCENT}; color: white; }}
QMenu {{ background: #24272c; color: #e4e7ea; border: 1px solid #34383e; }}
QMenu::item:selected {{ background: {_ACCENT}; color: white; }}

QScrollBar:vertical {{ background: #191b1f; width: 11px; margin: 0; }}
QScrollBar::handle:vertical {{ background: #3a3f46; border-radius: 5px;
    min-height: 24px; }}
QScrollBar::handle:vertical:hover {{ background: #474d55; }}
QScrollBar:horizontal {{ background: #191b1f; height: 11px; margin: 0; }}
QScrollBar::handle:horizontal {{ background: #3a3f46; border-radius: 5px;
    min-width: 24px; }}
QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
"""
