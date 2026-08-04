"""UWB Studio — explore and animate a preprocessing export folder.

Scaffold. Operates *solely* on the outputs of the UWB PreProcessing tool's
``<db>_FNT_analysis`` folder — the smoothed CSV (positions + sex/identity),
``fnt_config.json`` (identities, zones, anchors, scale, timezone, sitemap
reference) and the optional ``<db>_sitemap.png``. No SQLite, no re-processing.

Workflow: load a folder -> pick animals / time range / layers / render settings
-> scrub a live single-frame preview -> queue one or more animation jobs ->
render them all in the background. Rendering runs through the shared, GUI-free
``fnt.uwb.animation`` core.

Intentionally NOT built yet (future): GIF output, external-data overlays
(weather, etc.), multi-panel/side-by-side, per-animal colour overrides.
"""

import json
import os
import re

import numpy as np
import pandas as pd
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg
from matplotlib.figure import Figure

from PyQt5.QtCore import Qt, QThread, pyqtSignal
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel, QFileDialog,
    QGroupBox, QCheckBox, QComboBox, QSpinBox, QScrollArea, QSplitter,
    QSlider, QListWidget, QListWidgetItem, QMessageBox, QLineEdit,
    QDateTimeEdit, QProgressBar,
)

from fnt.uwb import animation as uwb_animation


DARK_STYLE = """
    QWidget { background-color: #2b2b2b; color: #cccccc; font-family: Arial; }
    QLabel { color: #cccccc; background-color: transparent; }
    QPushButton { background-color: #0078d4; color: white; border: none;
        padding: 7px 14px; border-radius: 4px; font-weight: bold; min-width: 90px; }
    QPushButton:hover { background-color: #106ebe; }
    QPushButton:disabled { background-color: #3f3f3f; color: #666666; }
    QGroupBox { background-color: #1e1e1e; border: 1px solid #3f3f3f;
        border-radius: 6px; margin-top: 12px; padding-top: 12px; font-weight: bold; }
    QGroupBox::title { subcontrol-origin: margin; subcontrol-position: top left;
        padding: 4px 8px; background-color: #2b2b2b; color: #0078d4; }
    QComboBox, QSpinBox, QDateTimeEdit, QLineEdit, QListWidget {
        background-color: #1e1e1e; color: #cccccc; border: 1px solid #3f3f3f;
        border-radius: 4px; padding: 3px 6px; }
    QScrollArea { border: none; }
"""


class RenderWorker(QThread):
    """Render a list of queued animation jobs sequentially, off the GUI thread."""
    log = pyqtSignal(str)
    frame_progress = pyqtSignal(int, int, int)   # job_index, frame, total
    job_finished = pyqtSignal(int, bool, str)     # job_index, success, path/msg
    all_finished = pyqtSignal()

    def __init__(self, positions, context, jobs, out_dir):
        super().__init__()
        self.positions = positions
        self.context = context          # arena_zones, anchors, bg_image, bg_extent, tag_identities
        self.jobs = jobs
        self.out_dir = out_dir
        self._cancel = False

    def cancel(self):
        self._cancel = True

    def run(self):
        os.makedirs(self.out_dir, exist_ok=True)
        for idx, job in enumerate(self.jobs):
            if self._cancel:
                break
            self.log.emit(f"Rendering job {idx + 1}/{len(self.jobs)}: {job['name']}")
            data = self._subset(job)
            if data is None or data.empty:
                self.job_finished.emit(idx, False, "no data for selection")
                continue
            out_path = os.path.join(self.out_dir, job['filename'])
            layers = job['layers']

            def _prog(i, total, _idx=idx):
                self.frame_progress.emit(_idx, i, total)

            res = uwb_animation.render_animation(
                data, out_path,
                frame_interval=uwb_animation.frame_interval_seconds(job['speed'], job['fps']),
                trailing_window=job['trail'], fps=job['fps'],
                dpi=uwb_animation.QUALITY_DPI.get(job['quality'], 100),
                speed_text=f"{job['speed']}x", title=job['name'],
                layers=layers,
                bg_image=self.context['bg_image'] if layers.get('background') else None,
                bg_extent=self.context['bg_extent'],
                arena_zones=self.context['arena_zones'],
                anchors=self.context['anchors'],
                tag_identities=self.context['tag_identities'], use_custom_identities=True,
                is_cancelled=lambda: self._cancel, progress=_prog, log=self.log.emit,
            )
            self.job_finished.emit(idx, res is not None, res or "render failed")
        self.all_finished.emit()

    def _subset(self, job):
        d = self.positions
        d = d[d['shortid'].isin(job['tags'])]
        d = d[(d['Timestamp'] >= job['t0']) & (d['Timestamp'] <= job['t1'])]
        if job['downsample_hz']:
            d = uwb_animation.downsample_to_hz(d, job['downsample_hz'])
        return d


class UWBStudioWindow(QWidget):
    SPEEDS = ["1x", "5x", "10x", "20x", "40x", "80x", "100x", "200x", "400x", "1000x"]
    FPSS = ["5", "10", "20", "30"]
    QUALITIES = ["Draft (Fast)", "Standard", "High Quality"]

    def __init__(self):
        super().__init__()
        # Loaded state
        self.folder = None
        self.positions = None            # full-resolution positions DataFrame
        self.preview_data = None         # downsampled + selected-tag cache for scrubbing
        self.tag_identities = {}         # {shortid: {'sex','identity'}}
        self.arena_zones = None          # DataFrame(zone, x, y) or None
        self.anchors = []                # [{'x','y',...}]
        self.bg_image = None
        self.bg_extent = None
        self.data_tz = None
        self.tag_checkboxes = {}
        self.jobs = []                   # queued job dicts
        self.render_worker = None
        self.initUI()

    # ------------------------------------------------------------------ #
    # UI
    # ------------------------------------------------------------------ #
    def initUI(self):
        self.setWindowTitle("UWB Studio")
        self.setGeometry(60, 60, 1400, 900)
        self.setStyleSheet(DARK_STYLE)

        splitter = QSplitter(Qt.Horizontal)
        splitter.setChildrenCollapsible(False)
        splitter.addWidget(self._build_controls())
        splitter.addWidget(self._build_preview())
        splitter.setStretchFactor(0, 0)
        splitter.setStretchFactor(1, 1)
        splitter.setSizes([460, 940])

        root = QVBoxLayout()
        root.addWidget(splitter)
        self.setLayout(root)

    def _build_controls(self):
        panel = QWidget()
        col = QVBoxLayout()

        # -- Folder --
        fgroup = QGroupBox("Export Folder")
        fl = QVBoxLayout()
        btn = QPushButton("Open Analysis Folder…")
        btn.setToolTip("Select a <db>_FNT_analysis folder produced by the UWB PreProcessing tool")
        btn.clicked.connect(self.open_folder)
        fl.addWidget(btn)
        self.lbl_folder = QLabel("No folder loaded")
        self.lbl_folder.setStyleSheet("color:#888; font-style:italic;")
        self.lbl_folder.setWordWrap(True)
        fl.addWidget(self.lbl_folder)
        fgroup.setLayout(fl)
        col.addWidget(fgroup)

        # -- Animals --
        self.tag_group = QGroupBox("Animals")
        self.tag_layout = QVBoxLayout()
        self.lbl_no_tags = QLabel("Load a folder to list animals")
        self.lbl_no_tags.setStyleSheet("color:#888; font-style:italic;")
        self.tag_layout.addWidget(self.lbl_no_tags)
        row = QHBoxLayout()
        b_all = QPushButton("All"); b_all.clicked.connect(lambda: self._set_all_tags(True))
        b_none = QPushButton("None"); b_none.clicked.connect(lambda: self._set_all_tags(False))
        row.addWidget(b_all); row.addWidget(b_none); row.addStretch()
        self.tag_layout.addLayout(row)
        self.tag_group.setLayout(self.tag_layout)
        col.addWidget(self.tag_group)

        # -- Time range --
        tgroup = QGroupBox("Time Range")
        tl = QVBoxLayout()
        r1 = QHBoxLayout(); r1.addWidget(QLabel("Start:"))
        self.dt_start = QDateTimeEdit(); self.dt_start.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.dt_start.dateTimeChanged.connect(self.update_preview)
        r1.addWidget(self.dt_start, 1); tl.addLayout(r1)
        r2 = QHBoxLayout(); r2.addWidget(QLabel("Stop: "))
        self.dt_stop = QDateTimeEdit(); self.dt_stop.setDisplayFormat("yyyy-MM-dd HH:mm:ss")
        self.dt_stop.dateTimeChanged.connect(self.update_preview)
        r2.addWidget(self.dt_stop, 1); tl.addLayout(r2)
        b_reset = QPushButton("Reset to full range"); b_reset.setStyleSheet("padding:4px; font-size:10px;")
        b_reset.clicked.connect(self._reset_time_range)
        tl.addWidget(b_reset)
        tgroup.setLayout(tl)
        col.addWidget(tgroup)

        # -- Layers --
        lgroup = QGroupBox("Layers")
        ll = QVBoxLayout()
        self.chk_bg = QCheckBox("Background image"); self.chk_bg.setEnabled(False)
        self.chk_zones = QCheckBox("Zones"); self.chk_zones.setEnabled(False)
        self.chk_anchors = QCheckBox("Anchors"); self.chk_anchors.setEnabled(False)
        for c in (self.chk_bg, self.chk_zones, self.chk_anchors):
            c.stateChanged.connect(self.update_preview)
            ll.addWidget(c)
        lgroup.setLayout(ll)
        col.addWidget(lgroup)

        # -- Render settings --
        rgroup = QGroupBox("Render Settings")
        rl = QVBoxLayout()
        rl.addLayout(self._combo_row("Speed:", "combo_speed", self.SPEEDS, "40x"))
        rl.addLayout(self._combo_row("FPS:", "combo_fps", self.FPSS, "20"))
        trail_row = QHBoxLayout(); trail_row.addWidget(QLabel("Trail (s):"))
        self.spin_trail = QSpinBox(); self.spin_trail.setRange(0, 3600); self.spin_trail.setValue(60)
        self.spin_trail.valueChanged.connect(self.update_preview)
        trail_row.addWidget(self.spin_trail); trail_row.addStretch(); rl.addLayout(trail_row)
        rl.addLayout(self._combo_row("Quality:", "combo_quality", self.QUALITIES, "Standard"))
        ds_row = QHBoxLayout()
        self.chk_downsample = QCheckBox("Downsample to"); self.chk_downsample.setChecked(True)
        ds_row.addWidget(self.chk_downsample)
        self.spin_downsample = QSpinBox(); self.spin_downsample.setRange(1, 10); self.spin_downsample.setValue(1)
        self.spin_downsample.setSuffix(" Hz"); self.spin_downsample.setFixedWidth(70)
        self.chk_downsample.toggled.connect(self.spin_downsample.setEnabled)
        ds_row.addWidget(self.spin_downsample); ds_row.addStretch(); rl.addLayout(ds_row)
        rgroup.setLayout(rl)
        col.addWidget(rgroup)

        # -- Queue --
        qgroup = QGroupBox("Render Queue")
        ql = QVBoxLayout()
        name_row = QHBoxLayout(); name_row.addWidget(QLabel("Name:"))
        self.edit_job_name = QLineEdit(); self.edit_job_name.setPlaceholderText("e.g. pair_248_276")
        name_row.addWidget(self.edit_job_name, 1); ql.addLayout(name_row)
        add_row = QHBoxLayout()
        self.btn_add = QPushButton("Add to Queue"); self.btn_add.clicked.connect(self.add_to_queue)
        self.btn_remove = QPushButton("Remove"); self.btn_remove.clicked.connect(self.remove_job)
        add_row.addWidget(self.btn_add); add_row.addWidget(self.btn_remove); ql.addLayout(add_row)
        self.queue_list = QListWidget(); self.queue_list.setMaximumHeight(120)
        ql.addWidget(self.queue_list)
        self.btn_render = QPushButton("Render All"); self.btn_render.clicked.connect(self.render_all)
        self.btn_render.setStyleSheet("background-color:#0a8f3c;")
        ql.addWidget(self.btn_render)
        self.btn_cancel = QPushButton("Cancel Render"); self.btn_cancel.clicked.connect(self.cancel_render)
        self.btn_cancel.setStyleSheet("background-color:#d41100;"); self.btn_cancel.setVisible(False)
        ql.addWidget(self.btn_cancel)
        self.progress = QProgressBar(); self.progress.setRange(0, 100); self.progress.setValue(0)
        ql.addWidget(self.progress)
        qgroup.setLayout(ql)
        col.addWidget(qgroup)

        col.addStretch()
        panel.setLayout(col)
        scroll = QScrollArea(); scroll.setWidget(panel); scroll.setWidgetResizable(True)
        scroll.setMinimumWidth(440)
        return scroll

    def _combo_row(self, label, attr, items, default):
        row = QHBoxLayout(); row.addWidget(QLabel(label))
        combo = QComboBox(); combo.addItems(items); combo.setCurrentText(default)
        combo.currentTextChanged.connect(self.update_preview)
        setattr(self, attr, combo)
        row.addWidget(combo, 1)
        return row

    def _build_preview(self):
        panel = QWidget()
        v = QVBoxLayout()
        self.fig = Figure(figsize=(7, 6))
        self.canvas = FigureCanvasQTAgg(self.fig)
        self.ax = self.fig.add_subplot(111)
        self._draw_placeholder()
        v.addWidget(self.canvas, 1)

        self.slider = QSlider(Qt.Horizontal); self.slider.setRange(0, 1000); self.slider.setValue(0)
        self.slider.setEnabled(False)
        self.slider.valueChanged.connect(self._on_scrub)
        v.addWidget(self.slider)
        self.lbl_time = QLabel("—"); self.lbl_time.setAlignment(Qt.AlignCenter)
        self.lbl_time.setStyleSheet("color:#888; font-family: Consolas, monospace;")
        v.addWidget(self.lbl_time)

        self.lbl_status = QLabel("Load an export folder to begin.")
        self.lbl_status.setStyleSheet("color:#888; font-style:italic;")
        self.lbl_status.setWordWrap(True)
        v.addWidget(self.lbl_status)
        panel.setLayout(v)
        return panel

    def _draw_placeholder(self):
        self.ax.clear()
        self.ax.text(0.5, 0.5, "UWB Studio\n\nOpen an analysis folder",
                     ha="center", va="center", color="#666", fontsize=14,
                     transform=self.ax.transAxes)
        self.ax.set_xticks([]); self.ax.set_yticks([])
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ #
    # Loading
    # ------------------------------------------------------------------ #
    def open_folder(self):
        folder = QFileDialog.getExistingDirectory(self, "Select <db>_FNT_analysis folder",
                                                  os.path.expanduser("~"))
        if not folder:
            return
        try:
            self._load_folder(folder)
        except Exception as e:
            QMessageBox.critical(self, "Load Failed", f"Could not load folder:\n{e}")

    def _load_folder(self, folder):
        # Find the smoothed CSV (exclude working '_plotdata_' files).
        csvs = [f for f in os.listdir(folder)
                if f.lower().endswith('_smoothed.csv') and 'plotdata' not in f.lower()]
        if not csvs:
            raise FileNotFoundError(
                "No '*_smoothed.csv' found. Re-run the preprocessing export with "
                "'Export Smoothed CSV' enabled.")
        csv_path = os.path.join(folder, sorted(csvs)[0])
        self.lbl_status.setText(f"Loading {os.path.basename(csv_path)} …")
        df = pd.read_csv(csv_path, low_memory=False)
        df['Timestamp'] = pd.to_datetime(df['Timestamp'], format='mixed')
        if 'smoothed_x' not in df.columns and 'location_x' in df.columns:
            df['smoothed_x'] = df['location_x']; df['smoothed_y'] = df['location_y']
        self.positions = df.sort_values(['shortid', 'Timestamp']).reset_index(drop=True)
        self.data_tz = self.positions['Timestamp'].dt.tz

        # Identities from the CSV (self-describing), falling back to config.
        self.tag_identities = {}
        for sid, g in self.positions.groupby('shortid'):
            first = g.iloc[0]
            self.tag_identities[int(sid)] = {
                'sex': str(first.get('sex', 'M')),
                'identity': str(first.get('identity', sid)),
            }

        # Config: zones, anchors, sitemap.
        self.arena_zones = None
        self.anchors = []
        self.bg_image = None
        self.bg_extent = None
        cfg_path = os.path.join(folder, 'fnt_config.json')
        if os.path.exists(cfg_path):
            with open(cfg_path) as f:
                cfg = json.load(f)
            if cfg.get('arena_zones'):
                self.arena_zones = pd.DataFrame(cfg['arena_zones'])
            xmlc = cfg.get('xml_config') or {}
            self.anchors = xmlc.get('anchor_positions_m', []) or []
            sm = cfg.get('sitemap')
            if sm and sm.get('filename'):
                img_path = os.path.join(folder, sm['filename'])
                if os.path.exists(img_path):
                    try:
                        import matplotlib.image as mpimg
                        self.bg_image = mpimg.imread(img_path)
                        self.bg_extent = sm.get('extent_m')
                    except Exception as e:
                        self.log_status(f"Could not load sitemap image: {e}")

        self.folder = folder
        self.lbl_folder.setText(folder)
        self._populate_after_load()

    def _populate_after_load(self):
        # Animal checkboxes
        for cb in self.tag_checkboxes.values():
            cb.setParent(None)
        self.tag_checkboxes = {}
        self.lbl_no_tags.setVisible(False)
        for sid in sorted(self.tag_identities):
            info = self.tag_identities[sid]
            cb = QCheckBox(f"{info['sex']}-{info['identity']}  (HexID {hex(sid).upper().replace('0X','')})")
            cb.setChecked(True)
            cb.stateChanged.connect(self._on_tag_changed)
            self.tag_checkboxes[sid] = cb
            self.tag_layout.insertWidget(self.tag_layout.count() - 1, cb)

        # Layer availability
        self.chk_bg.setEnabled(self.bg_image is not None); self.chk_bg.setChecked(self.bg_image is not None)
        has_zones = self.arena_zones is not None and not self.arena_zones.empty
        self.chk_zones.setEnabled(has_zones); self.chk_zones.setChecked(has_zones)
        self.chk_anchors.setEnabled(bool(self.anchors)); self.chk_anchors.setChecked(False)

        # Time range defaults
        self._reset_time_range()
        self.slider.setEnabled(True)

        n = len(self.positions)
        self.lbl_status.setText(
            f"Loaded {n:,} fixes · {len(self.tag_identities)} animals · "
            f"zones: {'yes' if has_zones else 'no'} · "
            f"background: {'yes' if self.bg_image is not None else 'no'}")
        self._rebuild_preview_cache()
        self.slider.setValue(0)
        self.update_preview()

    # ------------------------------------------------------------------ #
    # Selection / time helpers
    # ------------------------------------------------------------------ #
    def selected_tags(self):
        return [sid for sid, cb in self.tag_checkboxes.items() if cb.isChecked()]

    def _set_all_tags(self, state):
        for cb in self.tag_checkboxes.values():
            cb.setChecked(state)

    def _on_tag_changed(self, *_):
        self._rebuild_preview_cache()
        self.update_preview()

    def _reset_time_range(self):
        if self.positions is None or self.positions.empty:
            return
        t0 = self.positions['Timestamp'].min()
        t1 = self.positions['Timestamp'].max()
        for widget, ts in ((self.dt_start, t0), (self.dt_stop, t1)):
            widget.blockSignals(True)
            naive = ts.tz_localize(None) if ts.tzinfo else ts
            widget.setDateTime(naive.to_pydatetime())
            widget.blockSignals(False)
        self.update_preview()

    def _picker_ts(self, widget):
        """QDateTimeEdit (naive wall-clock) -> tz-aware Timestamp in the data tz."""
        ts = pd.Timestamp(widget.dateTime().toPyDateTime())
        if self.data_tz is not None:
            ts = ts.tz_localize(self.data_tz)
        return ts

    def _time_range(self):
        return self._picker_ts(self.dt_start), self._picker_ts(self.dt_stop)

    def _layers(self):
        return {'background': self.chk_bg.isChecked() and self.bg_image is not None,
                'zones': self.chk_zones.isChecked(),
                'anchors': self.chk_anchors.isChecked()}

    # ------------------------------------------------------------------ #
    # Preview
    # ------------------------------------------------------------------ #
    def _rebuild_preview_cache(self):
        """Downsample selected-tag data to 1 Hz for snappy scrubbing."""
        if self.positions is None:
            self.preview_data = None
            return
        tags = self.selected_tags()
        d = self.positions[self.positions['shortid'].isin(tags)] if tags else self.positions.iloc[0:0]
        self.preview_data = uwb_animation.downsample_to_hz(d, 1)

    def _on_scrub(self, _value):
        self.update_preview()

    def update_preview(self, *_):
        if self.positions is None or self.preview_data is None:
            return
        t0, t1 = self._time_range()
        if t1 <= t0:
            return
        frac = self.slider.value() / 1000.0
        current = t0 + (t1 - t0) * frac
        self.lbl_time.setText(current.strftime('%Y-%m-%d %H:%M:%S'))

        layers = self._layers()
        trail = self.spin_trail.value()
        window_start = current - pd.Timedelta(seconds=trail)

        self.ax.clear()
        uwb_animation.draw_static_context(
            self.ax, layers,
            bg_image=self.bg_image, bg_extent=self.bg_extent,
            arena_zones=self.arena_zones, anchors=self.anchors)

        data = self.preview_data
        xmin, xmax, ymin, ymax = uwb_animation.compute_axis_limits(
            data if not data.empty else self.positions, layers, self.bg_extent)
        yr = ymax - ymin
        styles = uwb_animation.build_tag_styles(self.selected_tags(), self.tag_identities, True)
        for sid in self.selected_tags():
            g = data[(data['shortid'] == sid) & (data['Timestamp'] >= window_start) &
                     (data['Timestamp'] <= current)]
            if g.empty:
                continue
            st = styles[sid]
            self.ax.plot(g['smoothed_x'], g['smoothed_y'], color=st['color'], alpha=0.5, linewidth=1)
            cx, cy = g['smoothed_x'].iloc[-1], g['smoothed_y'].iloc[-1]
            self.ax.plot(cx, cy, 'o', color=st['color'], markersize=9)
            self.ax.text(cx, cy + yr * 0.02, st['label'], ha='center', color=st['color'],
                         fontsize=9, fontweight='bold')
        self.ax.set_xlim(xmin, xmax); self.ax.set_ylim(ymin, ymax)
        self.ax.set_aspect('equal')
        self.ax.set_xlabel("X (m)"); self.ax.set_ylabel("Y (m)")
        self.ax.set_title(f"Preview · {current.strftime('%Y-%m-%d %H:%M:%S')}")
        self.canvas.draw_idle()

    # ------------------------------------------------------------------ #
    # Queue + render
    # ------------------------------------------------------------------ #
    def add_to_queue(self):
        if self.positions is None:
            QMessageBox.warning(self, "No Data", "Load an export folder first.")
            return
        tags = self.selected_tags()
        if not tags:
            QMessageBox.warning(self, "No Animals", "Select at least one animal.")
            return
        t0, t1 = self._time_range()
        speed = int(self.combo_speed.currentText().replace('x', ''))
        fps = int(self.combo_fps.currentText())
        name = self.edit_job_name.text().strip() or self._auto_job_name(tags)
        safe = re.sub(r'[^A-Za-z0-9_.-]+', '_', name)
        job = {
            'name': name,
            'tags': tags,
            't0': t0, 't1': t1,
            'layers': self._layers(),
            'speed': speed, 'fps': fps,
            'trail': self.spin_trail.value(),
            'quality': self.combo_quality.currentText(),
            'downsample_hz': self.spin_downsample.value() if self.chk_downsample.isChecked() else None,
            'filename': f"{safe}_{fps}fps_{speed}x.mp4",
        }
        self.jobs.append(job)
        ds = f"{job['downsample_hz']}Hz" if job['downsample_hz'] else "full"
        self.queue_list.addItem(QListWidgetItem(
            f"{name}  ·  {len(tags)} animals  ·  {speed}x@{fps}fps  ·  {ds}  ·  {job['filename']}"))
        self.edit_job_name.clear()

    def _auto_job_name(self, tags):
        ids = "_".join(self.tag_identities[t]['identity'] for t in tags[:4])
        return f"anim_{ids}" + ("_etc" if len(tags) > 4 else "")

    def remove_job(self):
        row = self.queue_list.currentRow()
        if row >= 0:
            self.queue_list.takeItem(row)
            del self.jobs[row]

    def render_all(self):
        if not self.jobs:
            QMessageBox.information(self, "Empty Queue", "Add at least one job to the queue.")
            return
        if self.render_worker and self.render_worker.isRunning():
            return
        out_dir = os.path.join(self.folder, 'animations')
        context = {'arena_zones': self.arena_zones, 'anchors': self.anchors,
                   'bg_image': self.bg_image, 'bg_extent': self.bg_extent,
                   'tag_identities': self.tag_identities}
        self.render_worker = RenderWorker(self.positions, context, list(self.jobs), out_dir)
        self.render_worker.log.connect(self.log_status)
        self.render_worker.frame_progress.connect(self._on_frame_progress)
        self.render_worker.job_finished.connect(self._on_job_finished)
        self.render_worker.all_finished.connect(self._on_all_finished)
        self._n_jobs = len(self.jobs)
        self.btn_render.setEnabled(False); self.btn_cancel.setVisible(True)
        self.render_worker.start()

    def cancel_render(self):
        if self.render_worker:
            self.render_worker.cancel()
            self.log_status("Cancelling…")

    def _on_frame_progress(self, job_idx, frame, total):
        per_job = 100.0 / max(1, self._n_jobs)
        pct = int(job_idx * per_job + (frame / max(1, total)) * per_job)
        self.progress.setValue(min(100, pct))

    def _on_job_finished(self, idx, ok, msg):
        self.log_status((f"✓ Rendered: {os.path.basename(msg)}" if ok
                         else f"✗ Job {idx + 1} failed: {msg}"))

    def _on_all_finished(self):
        self.progress.setValue(100)
        self.btn_render.setEnabled(True); self.btn_cancel.setVisible(False)
        self.log_status("Render queue complete.")
        QMessageBox.information(self, "Done", "Render queue complete.")

    def log_status(self, msg):
        self.lbl_status.setText(msg)

    def closeEvent(self, event):
        if self.render_worker and self.render_worker.isRunning():
            self.render_worker.cancel()
            self.render_worker.wait(3000)
        event.accept()
