"""ABMA main window — the PyQt5 tab launched from the FNT 'WIFP' / ABMA tab.

Four-panel workflow: Arena → Population → Experiment → Run. The window collects
an :class:`ExperimentConfig` from the widgets and hands it to the headless
engine, streaming live agent positions back to the canvas during the first trial.
"""
from __future__ import annotations

import os
import subprocess
import sys
import time
import traceback

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QColor
from PyQt5.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QFormLayout,
    QTabWidget, QLabel, QPushButton, QDoubleSpinBox, QSpinBox, QComboBox,
    QLineEdit, QCheckBox, QTableWidget, QTableWidgetItem, QGroupBox,
    QTextEdit, QProgressBar, QFileDialog, QMessageBox, QHeaderView,
    QAbstractItemView, QAction, QSplitter, QStackedWidget, QScrollArea,
    QFrame, QDialog, QListWidget, QDialogButtonBox, QSizePolicy,
)

from ..core.config import (
    ExperimentConfig, ArenaConfig, AgentGroup, Genotype, Treatment,
    TraitProfile, ResourceObject, Intervention, Appearance, blank_experiment,
    default_vole_experiment,
)
from ..core.runner import run_experiment, grid_offsets
from ..core.sampling import parse_spec
from ..core.presets import PRESETS
from .abma_canvas import ArenaCanvas
from .agent_inspector import AgentInspector

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

    def run(self):
        try:
            res = run_experiment(
                self.config, self.project_dir,
                progress_cb=lambda f: self.progress.emit(float(f)),
                frame_cb=lambda fr: self.frame.emit(fr),
                log_cb=lambda m: self.log.emit(m),
                frame_interval_s=self._frame_interval,
                analyze=self.analyze,
                meta_cb=lambda meta: self.agents.emit(meta),
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
        self._run_t0 = None
        self._last_frame = None
        self._zones = []
        self.setStyleSheet(_DARK_QSS)
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

    # ------------------------------------------------------------------ #
    # Layout: left scrolling section column + right God's-eye preview
    # ------------------------------------------------------------------ #
    def _build_left_column(self):
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFrameShape(QFrame.NoFrame)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarAlwaysOff)
        scroll.setMinimumWidth(490)
        content = QWidget()
        col = QVBoxLayout(content)
        col.setContentsMargins(10, 10, 10, 10)
        col.setSpacing(10)
        for title, builder in [
            ("1 · Arena", self._build_arena_tab),
            ("2 · Build && Add Agents", self._build_population_tab),
            ("3 · Experiment", self._build_experiment_tab),
            ("4 · Run", self._build_run_tab),
        ]:
            gb = QGroupBox(title)
            gbl = QVBoxLayout(gb)
            gbl.setContentsMargins(8, 12, 8, 8)
            gbl.addWidget(builder())
            col.addWidget(gb)
        col.addStretch(1)
        scroll.setWidget(content)
        return scroll

    def _build_preview(self):
        right = QWidget()
        rlay = QVBoxLayout(right)
        rlay.setContentsMargins(0, 0, 0, 0)
        rlay.setSpacing(4)
        title_row = QHBoxLayout()
        self.preview_title = QLabel(
            "God's-eye preview  ·  drag to orbit, scroll to zoom")
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
        if self.view_3d is not None:
            self.view_3d.agent_hovered.connect(self._on_agent_hover)

        rlay.addWidget(self.view_stack, 1)   # preview takes the full width

        # inspector is a floating popup shown on hover / click (frees the preview)
        self.inspector = AgentInspector(self)
        self.inspector.setWindowFlags(
            Qt.FramelessWindowHint | Qt.Tool | Qt.WindowStaysOnTopHint)
        self.inspector.hide()
        self._inspector_pinned = False
        self._hover_idx = None
        return right

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
        if self._last_frame is not None:
            self._push_frame(self._active_view(), self._last_frame)

    def _open_preset_dialog(self):
        dlg = ArenaPresetDialog(self)
        if dlg.exec_() == QDialog.Accepted and dlg.chosen():
            p = dlg.chosen()
            self._load_config(p.factory())
            self.statusBar().showMessage(f"Loaded preset: {p.name}", 4000)

    # ------------------------------------------------------------------ #
    # Menu bar
    # ------------------------------------------------------------------ #
    def _build_menu(self):
        m = self.menuBar().addMenu("&File")
        preset_menu = m.addMenu("Load &Preset")
        for p in PRESETS:
            act = QAction(p.name, self)
            act.setStatusTip(p.description)
            act.triggered.connect(lambda _, pr=p: self._load_preset(pr))
            preset_menu.addAction(act)
        m.addSeparator()
        for text, shortcut, slot in [
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

    def _load_preset(self, preset):
        self._load_config(preset.factory())
        self.statusBar().showMessage(f"Loaded preset: {preset.name}", 4000)

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
        btn_preset = QPushButton("⬗  Load Preset Arena…")
        btn_preset.setObjectName("accept_btn")
        btn_preset.clicked.connect(self._open_preset_dialog)
        left.addWidget(btn_preset)
        form = QFormLayout()
        self.in_name = QLineEdit("vole_experiment")
        self.in_width = _dspin(0.2, 50, 2.2, 0.1, " m")
        self.in_height = _dspin(0.2, 50, 2.2, 0.1, " m")
        self.in_boundary = QComboBox()
        self.in_boundary.addItems(["reflective", "wrap", "absorbing"])
        for lab, wdg in [("Experiment name", self.in_name),
                         ("Arena width", self.in_width),
                         ("Arena height", self.in_height),
                         ("Boundary", self.in_boundary)]:
            form.addRow(lab, wdg)
        self.in_width.valueChanged.connect(self._refresh_arena)
        self.in_height.valueChanged.connect(self._refresh_arena)
        self.in_boundary.currentTextChanged.connect(self._refresh_arena)
        left.addLayout(form)

        # click-to-add controls
        add_box = QGroupBox("Add object (then click the arena)")
        ab = QHBoxLayout(add_box)
        for kind in ("nest", "food", "water"):
            b = QPushButton(f"+ {kind}")
            b.setCheckable(True)
            b.clicked.connect(lambda _, k=kind, btn=b: self._arm(k, btn))
            ab.addWidget(b)
            setattr(self, f"_arm_btn_{kind}", b)
        left.addWidget(add_box)

        self.obj_table = _table(OBJ_COLS)
        self.obj_table.setMinimumHeight(150)
        self.obj_table.itemChanged.connect(self._refresh_arena)
        left.addWidget(self.obj_table, 1)
        row = QHBoxLayout()
        b_del = QPushButton("Remove selected object")
        b_del.clicked.connect(lambda: self._del_row(self.obj_table))
        row.addWidget(b_del)
        left.addLayout(row)

        self.arena_summary = QLabel()
        self.arena_summary.setStyleSheet("color: #8fbfff; font-size: 11px;")
        left.addWidget(self.arena_summary)
        return w

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
        import copy
        self._agent_types.insert(i + 1, copy.deepcopy(self._agent_types[i]))
        self._render_agent_cards()

    def _remove_agent_type(self, i):
        del self._agent_types[i]
        self._render_agent_cards()

    # ------------------------------------------------------------------ #
    # Tab 3: Experiment
    # ------------------------------------------------------------------ #
    def _build_experiment_tab(self):
        w = QWidget()
        lay = QVBoxLayout(w)
        form = QFormLayout()
        self.in_days = _dspin(0.05, 365, 10.0, 0.5, " days")
        self.in_dt = _dspin(0.1, 60, 2.0, 0.5, " s")
        self.in_rec = _dspin(1, 3600, 10.0, 1.0, " s")
        self.in_trials = _ispin(1, 100, 3)
        self.in_seed = _ispin(0, 10 ** 6, 0)
        self.in_daystart = _dspin(0, 23.9, 6.0, 0.5, " h")
        self.in_dayact = _dspin(0, 3, 0.7, 0.1)
        self.in_nightact = _dspin(0, 3, 1.3, 0.1)
        self.in_start = QLineEdit("2025-11-07T18:00:00")
        self.in_variation = _dspin(0, 1, 0.0, 0.05)
        self.in_mortality = QCheckBox("Enable starvation mortality")
        self.in_analyze = QCheckBox("Compute socio-spatial analysis after run")
        self.in_analyze.setChecked(True)
        self.in_parallel = QCheckBox("Run trials in parallel (no live view)")
        self.in_workers = _ispin(1, 16, 2)
        for lab, wdg, tip in [
            ("Duration", self.in_days, "Simulated days per trial"),
            ("Timestep (dt)", self.in_dt, "Integration step in simulated seconds"),
            ("Record interval", self.in_rec, "Seconds between logged positions"),
            ("Trials (replicates)", self.in_trials, ""),
            ("Random seed", self.in_seed, "Base seed; trial i uses seed+i"),
            ("Day start hour", self.in_daystart, "Local hour the light phase starts"),
            ("Day activity ×", self.in_dayact, "Movement multiplier during day"),
            ("Night activity ×", self.in_nightact, "Movement multiplier at night"),
            ("Release datetime", self.in_start, "ISO timestamp for t=0"),
            ("Individual variation", self.in_variation,
             "Per-agent trait jitter SD (0 = identical clones)"),
            ("", self.in_mortality, "Agents can die of starvation if resources run out"),
            ("", self.in_analyze, ""),
            ("", self.in_parallel, ""),
            ("Parallel workers", self.in_workers, ""),
        ]:
            if tip:
                wdg.setToolTip(tip)
            form.addRow(lab, wdg)
        lay.addLayout(form)

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

    def _pick_outdir(self):
        d = QFileDialog.getExistingDirectory(self, "Choose project parent folder",
                                             self.in_outdir.text())
        if d:
            self.in_outdir.setText(d)

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
        self.btn_run = QPushButton("▶  Run experiment")
        self.btn_run.setObjectName("accept_btn")
        self.btn_run.clicked.connect(self._on_run)
        self.btn_stop = QPushButton("■  Stop")
        self.btn_stop.setObjectName("reject_btn")
        self.btn_stop.clicked.connect(self._on_stop)
        self.btn_stop.setEnabled(False)
        r = QHBoxLayout()
        r.addWidget(self.btn_run)
        r.addWidget(self.btn_stop)
        left.addLayout(r)
        self.status_label = QLabel("Idle.")
        self.status_label.setStyleSheet("color: #8fbfff; font-size: 12px;")
        left.addWidget(self.status_label)
        self.progress = QProgressBar()
        self.progress.setRange(0, 100)
        left.addWidget(self.progress)
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
    def _collect_config(self) -> ExperimentConfig:
        arena = self._arena_from_table()
        groups = [g for g in self._agent_types]
        if not groups:
            raise ValueError("Add at least one agent type "
                             "(Build & Add Agents → + Add Agent Type).")
        return ExperimentConfig(
            name=self.in_name.text().strip() or "experiment",
            arena=arena, groups=groups,
            interventions=self._interventions_from_table(),
            days=self.in_days.value(), dt=self.in_dt.value(),
            record_interval=self.in_rec.value(),
            n_trials=self.in_trials.value(), seed=self.in_seed.value(),
            day_start_hour=self.in_daystart.value(),
            day_activity=self.in_dayact.value(),
            night_activity=self.in_nightact.value(),
            start_datetime=self.in_start.text().strip(),
            individual_variation=self.in_variation.value(),
            enable_mortality=self.in_mortality.isChecked(),
            parallel=self.in_parallel.isChecked(),
            n_workers=self.in_workers.value(),
            trial_prefix="S",
        )

    def _load_config(self, cfg: ExperimentConfig):
        # arena attributes without a table cell ride along on the window
        self._zones = list(cfg.arena.zones)
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
        import copy
        self._agent_types = [copy.deepcopy(g) for g in cfg.groups]
        self._render_agent_cards()
        if hasattr(self, "iv_table"):
            self.iv_table.setRowCount(0)
            for iv in cfg.interventions:
                self._add_iv_row(iv)
        self.in_days.setValue(cfg.days)
        self.in_dt.setValue(cfg.dt)
        self.in_rec.setValue(cfg.record_interval)
        self.in_trials.setValue(cfg.n_trials)
        self.in_seed.setValue(cfg.seed)
        self.in_daystart.setValue(cfg.day_start_hour)
        self.in_dayact.setValue(cfg.day_activity)
        self.in_nightact.setValue(cfg.night_activity)
        self.in_start.setText(cfg.start_datetime)
        self.in_variation.setValue(cfg.individual_variation)
        self.in_mortality.setChecked(cfg.enable_mortality)
        self._refresh_arena()
        self._update_pop_summary()

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
            self.arena_summary.setText(
                f"Arena {arena.width:g}×{arena.height:g} {arena.units} · "
                f"{arena.boundary} · {n} nests, {f} food, {wt} water")

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
        self._last_frame = None
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
        self._last_frame = fr
        self._push_frame(self._active_view(), fr)   # only the visible view
        if self.inspector.isVisible():
            self._update_inspector_dynamic(fr)

    def _update_inspector_dynamic(self, fr):
        idx = self.inspector._idx
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
        QMessageBox.critical(self, "Simulation failed", tb.splitlines()[-1])

    def _on_stop(self):
        if self.worker and self.worker.isRunning():
            self.worker.terminate()
            self.worker.wait(2000)
            self._append_log("Stopped by user.")
        self.btn_run.setEnabled(True)
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
        lay = QVBoxLayout(self)
        lay.addWidget(QLabel("Select a preset to load into the preview:"))
        self.list = QListWidget()
        for p in PRESETS:
            self.list.addItem(p.name)
        self.list.setCurrentRow(0)
        self.list.currentRowChanged.connect(self._on_row)
        self.list.itemDoubleClicked.connect(lambda _: self.accept())
        lay.addWidget(self.list)
        self.desc = QLabel(PRESETS[0].description if PRESETS else "")
        self.desc.setWordWrap(True)
        self.desc.setStyleSheet("color:#8a9099; font-size:11px; padding:2px;")
        lay.addWidget(self.desc)
        bb = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
        bb.button(QDialogButtonBox.Ok).setText("Load")
        bb.accepted.connect(self.accept)
        bb.rejected.connect(self.reject)
        lay.addWidget(bb)

    def _on_row(self, r):
        if 0 <= r < len(PRESETS):
            self.desc.setText(PRESETS[r].description)

    def chosen(self):
        r = self.list.currentRow()
        return PRESETS[r] if 0 <= r < len(PRESETS) else None


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
