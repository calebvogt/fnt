"""Pop-out 3D view: each masked call drawn as a path through feature space.

The spectrogram shows a call against time; this window plots its per-frame
features against each other — by default pitch (X), spectral entropy as timbre
(Y) and rate of pitch change as motion (Z, vertical) — so every call becomes a
path, and calls with different shapes separate visibly. See
:mod:`fnt.usv.usv_detector.mad_trajectory` for the numbers; this module only
gathers calls from the open recording and draws them.

Only masks the user would call *calls* are drawn: pending predictions and
accepted/confirmed labels. Rejected masks never appear — they are a record of
what a call is not. Everything is computed over each call's own pixels, which
is why it lives in MAD: broadband noise under a USV, or a second call
overlapping it in time, never bends another call's path.

The window opens slowly rotating about the vertical axis, which reads a 3D
shape far better than any single still angle. Dragging takes over the camera;
releasing hands it back to the rotation from wherever it was left.

Two modes, switched by the spectrogram's Play button. Stopped, it is a static
picture of the calls in view that follows scrolling. Playing, it becomes a
player: each call stays hidden until the playhead reaches it, then draws itself
frame by frame behind a marker at the current moment, and finished calls stay
as dim ghosts. The axes keep the static view's scale throughout, so pressing
Play or Stop never makes the box jump.

MAD imports this lazily, and a machine without a working OpenGL still gets the
window — it explains what is missing instead of failing to open.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from PyQt5.QtCore import QSettings, Qt, QTimer, pyqtSignal
from PyQt5.QtGui import QColor, QFont
from PyQt5.QtWidgets import (
    QCheckBox, QComboBox, QDialog, QHBoxLayout, QLabel, QPushButton, QSlider,
    QVBoxLayout,
)

from fnt.usv.usv_detector.mad_inference import call_frame_features
from fnt.usv.usv_detector.mad_trajectory import (
    DEFAULT_AXES, DEFAULT_SMOOTH_FRAMES, FEATURES, axis_ranges,
    call_trajectory, direction_alpha, finite_rows, label_for,
    playback_progress, to_unit_cube,
)

try:  # pragma: no cover - depends on the machine's OpenGL
    import pyqtgraph as pg
    import pyqtgraph.opengl as gl
    from pyqtgraph import Vector
    HAVE_GL = True
    GL_ERROR = ""
except Exception as _e:  # pragma: no cover
    HAVE_GL = False
    GL_ERROR = str(_e)


#: Which annotation statuses become which drawn class. Rejected is absent on
#: purpose: it is never drawn.
_STATUS_CLASS = {'prediction': 'pending', None: 'accepted', 'accepted': 'accepted'}

#: Orbit speed. Slow enough to read the shape as it turns (a full turn a
#: minute), fast enough that the motion itself carries the depth cue.
ROTATE_DEG_PER_S = 6.0
_TICK_MS = 33

#: Smoothing slider: steps of 0.5 ms, 0 (off) to 10 ms. Stored in ms rather
#: than frames so a setting means the same thing at any hop / sample rate.
_SMOOTH_STEP_MS = 0.5
_SMOOTH_MAX_STEPS = 20
_SMOOTH_SETTINGS_KEY = "mad/trajectory/smooth_ms"
#: Frame length assumed before a recording is open (MAD's default grid).
_FALLBACK_DT = 128 / 250_000

_BG = (17, 20, 24)
_BOX_COLOR = (0.55, 0.58, 0.62, 0.55)
_GRID_COLOR = (90, 96, 104, 70)
_TEXT_COLOR = (200, 206, 212)


# ----------------------------------------------------------------------
# Gathering calls from the open recording
# ----------------------------------------------------------------------
def _call_columns_db(sg, t0: int, t1: int) -> Optional[np.ndarray]:
    """Full-frequency dB columns for grid frames [t0, t1) — the same STFT, on
    the same sample offsets, the canvas and the label path use, so frame k here
    is frame t0 + k on screen."""
    audio = getattr(sg, 'audio_data', None)
    if audio is None or not getattr(sg, 'hop', None):
        return None
    nperseg = int(sg.nperseg)
    start = int(t0) * int(sg.hop)
    end = min(len(audio), (int(t1) - 1) * int(sg.hop) + nperseg)
    seg = audio[start:end]
    if len(seg) < nperseg:
        return None
    from scipy import signal as _signal
    _f, _t, Sxx = _signal.spectrogram(
        seg, fs=sg.sample_rate, nperseg=nperseg,
        noverlap=min(int(sg.noverlap or 0), nperseg - 1),
        nfft=int(sg.nfft), window='hann')
    return 10.0 * np.log10(Sxx + 1e-10)


def call_frames(sg, ann: Dict, db_min: float,
                db_max: float) -> Optional[Dict]:
    """One annotation's raw per-frame features (``call_frame_features``), or
    None if its mask is empty or runs off the recording.

    This is the expensive half — an STFT over the call — and it is what gets
    cached. Smoothing is applied afterwards by :func:`call_trajectory`, so
    changing it re-draws instantly instead of recomputing every call. Columns
    the recording ends before are trimmed from the mask rather than padded — a
    padded column would be invented evidence."""
    mask = ann.get('mask')
    if mask is None or not np.any(mask):
        return None
    t0, f0 = int(ann['t0']), int(ann['f0'])
    t1 = t0 + mask.shape[1]
    cols = _call_columns_db(sg, t0, t1)
    if cols is None:
        return None
    w = min(cols.shape[1], mask.shape[1])
    df = (sg.sample_rate / 2.0) / (int(sg.nfft) // 2)
    return call_frame_features(cols[:, :w], np.asarray(mask[:, :w], bool),
                               f0, df, db_min, db_max)


def call_path(sg, ann: Dict, db_min: float, db_max: float,
              smooth_frames: int = DEFAULT_SMOOTH_FRAMES) -> Optional[Dict]:
    """One annotation's trajectory, or None if its mask is empty or too short."""
    frames = call_frames(sg, ann, db_min, db_max)
    if frames is None:
        return None
    dt = sg.hop / float(sg.sample_rate)
    return call_trajectory(frames, dt, frame_offset=int(ann['t0']),
                           smooth_frames=smooth_frames)


def _cache_key(tag, ann: Dict):
    mask = np.asarray(ann.get('mask'))
    return (tag, ann.get('id'), int(ann['t0']), int(ann['f0']), mask.shape,
            hash(np.ascontiguousarray(mask, dtype=bool).tobytes()))


def collect_call_paths(
    sg, db_min: float, db_max: float, t_origin_s: float, *,
    include_pending: bool = True, include_accepted: bool = True,
    include_harmonics: bool = False,
    time_window: Optional[Tuple[float, float]] = None,
    cache: Optional[Dict] = None, cache_tag=None,
    smooth_frames: int = DEFAULT_SMOOTH_FRAMES,
) -> Tuple[List[Dict], Dict[str, int]]:
    """Every drawable call on ``sg`` as ``{'id', 'cls', 'ann_idx', 'traj'}``.

    ``cache`` holds each call's raw frame features (the STFT work); the
    trajectory is re-derived from them with ``smooth_frames`` on every call,
    which is cheap, so a smoothing change never recomputes a spectrogram.

    ``time_window`` (seconds) keeps only calls overlapping it. Harmonics
    (``harmonic_n`` > 1) are left out by default: they are scaled copies of the
    fundamental's contour, so each would draw a duplicate of the fundamental's
    path, displaced up the pitch axis. Returns the paths plus counts of what was
    left out and why, so the window can say so rather than silently show less.
    """
    paths: List[Dict] = []
    skipped = {'harmonics': 0, 'short': 0}
    sr = getattr(sg, 'sample_rate', None)
    if not sr or not getattr(sg, 'hop', None):
        return paths, skipped
    dt = sg.hop / float(sr)
    want = {'pending': include_pending, 'accepted': include_accepted}
    for idx, ann in enumerate(getattr(sg, 'annotations', []) or []):
        cls = _STATUS_CLASS.get(ann.get('status'))
        if cls is None or not want[cls]:
            continue                       # rejected, or filtered out
        if ann.get('mask') is None:
            continue
        if time_window is not None:
            a = t_origin_s + ann['t0'] * dt
            b = t_origin_s + ann['t1'] * dt
            if b < time_window[0] or a > time_window[1]:
                continue
        if not include_harmonics and int(ann.get('harmonic_n') or 1) > 1:
            skipped['harmonics'] += 1
            continue
        key = _cache_key(cache_tag, ann) if cache is not None else None
        if key is not None and key in cache:
            frames = cache[key]
        else:
            frames = call_frames(sg, ann, db_min, db_max)
            if key is not None:
                cache[key] = frames
        traj = (None if frames is None else
                call_trajectory(frames, dt, frame_offset=int(ann['t0']),
                                smooth_frames=smooth_frames))
        if traj is None:
            skipped['short'] += 1
            continue
        paths.append({'id': ann.get('id'), 'cls': cls, 'ann_idx': idx,
                      'traj': traj})
    return paths, skipped


# ----------------------------------------------------------------------
# The window
# ----------------------------------------------------------------------
if HAVE_GL:
    class _OrbitView(gl.GLViewWidget):
        """GLViewWidget that knows when the user is holding the camera, so the
        auto-rotation can yield instead of fighting the drag."""

        def __init__(self, parent=None):
            super().__init__(parent)
            self.dragging = False
            self.on_camera_moved = None     # set by the window

        def _moved(self):
            if self.on_camera_moved is not None:
                self.on_camera_moved()

        def mousePressEvent(self, ev):
            self.dragging = True
            super().mousePressEvent(ev)

        def mouseMoveEvent(self, ev):
            super().mouseMoveEvent(ev)
            if self.dragging:
                self._moved()

        def mouseReleaseEvent(self, ev):
            self.dragging = False
            super().mouseReleaseEvent(ev)


def _fmt(v: float) -> str:
    """Axis-end number: three significant figures, no scientific notation for
    the ranges these features actually take."""
    a = abs(v)
    if a >= 100:
        return f"{v:.0f}"
    if a >= 10:
        return f"{v:.1f}"
    if a >= 1:
        return f"{v:.2f}"
    return f"{v:.3f}"


def _rgba(rgb, alpha=1.0) -> Tuple[float, float, float, float]:
    return (rgb[0] / 255.0, rgb[1] / 255.0, rgb[2] / 255.0, alpha)


class MADTrajectoryWindow(QDialog):
    """Rotating 3D view of the open recording's pending + accepted calls."""

    #: Emitted on show/hide so a toggle elsewhere (MAD's header checkbox) can
    #: track the window — including when it is closed with its own X.
    visibility_changed = pyqtSignal(bool)

    def __init__(self, main, palette: Dict, parented: bool = False):
        # Un-parented, like MAD's other pop-outs, so it can sit behind the main
        # window or on another screen while the spectrogram is in use.
        super().__init__(main if parented else None)
        self._main = main
        self._pal = palette
        self._cache: Dict = {}
        self._cache_wav: Optional[str] = None
        self._paths: List[Dict] = []
        self._trail_items: Dict = {}       # ann id -> GLLinePlotItem
        self._dyn_items: List = []         # onset markers
        self._label_items: List = []       # axis text, moved with the camera
        self._label_corner: Optional[Tuple[int, int]] = None
        self._ranges: Optional[Dict[str, Tuple[float, float]]] = None
        self._axes: List[str] = list(DEFAULT_AXES)
        self._onset_item = None            # static onset dots
        self._head_item = None             # playback: the "now" markers
        self._t_org = 0.0                  # seconds at spectrogram frame 0
        self._dt = 0.0                     # seconds per frame
        self._status_args = None           # to restore the text after playback
        # Playback state, driven by MAD's playhead (see playback_*).
        self._playing = False
        self._play_range: Optional[Tuple[float, float]] = None
        self._play_pos: Optional[float] = None

        self.setWindowTitle("Call Trajectories (3D)")
        self.setModal(False)
        self.setWindowFlags(Qt.Window
                            | Qt.WindowMinimizeButtonHint
                            | Qt.WindowMaximizeButtonHint
                            | Qt.WindowCloseButtonHint)
        self.resize(820, 700)

        v = QVBoxLayout(self)
        v.setContentsMargins(8, 8, 8, 8)
        v.setSpacing(6)

        if not HAVE_GL:
            msg = QLabel(
                "The 3D view needs OpenGL, which could not be loaded:\n\n"
                f"{GL_ERROR}\n\nInstall PyOpenGL (pip install PyOpenGL) and "
                "reopen this window.")
            msg.setWordWrap(True)
            v.addWidget(msg)
            return

        v.addLayout(self._build_axis_row())
        v.addLayout(self._build_filter_row())
        v.addLayout(self._build_view_row())

        self.view = _OrbitView()
        self.view.setBackgroundColor(pg.mkColor(*_BG))
        self.view.setToolTip(
            "Drag to rotate · scroll to zoom · Ctrl/Cmd-drag to pan.\n"
            "Each line is one call; it brightens from onset to offset, and the "
            "dot marks where the call starts.")
        v.addWidget(self.view, 1)
        self.view.on_camera_moved = self._follow_camera
        self._build_static_scene()
        self.reset_view()

        self.lbl_status = QLabel("")
        self.lbl_status.setWordWrap(True)
        self.lbl_status.setStyleSheet("color:#aab4bd; font-size:10px;")
        v.addWidget(self.lbl_status)
        v.addWidget(self._legend_label())

        # Debounced refresh: scrolling the spectrogram fires a view change per
        # step, and every one of them would otherwise recompute the scene.
        self._refresh_timer = QTimer(self)
        self._refresh_timer.setSingleShot(True)
        self._refresh_timer.setInterval(150)
        self._refresh_timer.timeout.connect(self.refresh_now)

        self._rotate_timer = QTimer(self)
        self._rotate_timer.setInterval(_TICK_MS)
        self._rotate_timer.timeout.connect(self._rotate_tick)

    # -- controls -------------------------------------------------------- #
    def _build_axis_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(6)
        self._axis_combos: List[QComboBox] = []
        tips = ("Horizontal axis.", "Depth axis.",
                "Vertical axis — the one the view rotates around.")
        for i, (name, tip) in enumerate(zip("XYZ", tips)):
            row.addWidget(QLabel(f"{name}:"))
            combo = QComboBox()
            for key in FEATURES:
                combo.addItem(label_for(key), key)
            combo.setCurrentIndex(combo.findData(self._axes[i]))
            combo.setToolTip(tip)
            combo.currentIndexChanged.connect(self._on_axes_changed)
            self._axis_combos.append(combo)
            row.addWidget(combo, 1)
        return row

    def _build_filter_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(10)
        row.addWidget(QLabel("Show:"))
        self.chk_pending = QCheckBox("Pending")
        self.chk_pending.setChecked(True)
        self.chk_pending.setToolTip("Model predictions not yet reviewed.")
        self.chk_accepted = QCheckBox("Accepted")
        self.chk_accepted.setChecked(True)
        self.chk_accepted.setToolTip("Accepted predictions and hand labels.")
        self.chk_harmonics = QCheckBox("Harmonics")
        self.chk_harmonics.setChecked(False)
        self.chk_harmonics.setToolTip(
            "Include masks grouped as harmonics (H2, H3…). Off by default: a "
            "harmonic is a scaled copy of its fundamental's contour, so it "
            "draws the same path again, higher up the pitch axis.")
        for chk in (self.chk_pending, self.chk_accepted, self.chk_harmonics):
            chk.toggled.connect(self.refresh_now)
            row.addWidget(chk)

        row.addSpacing(12)
        row.addWidget(QLabel("Calls:"))
        self.combo_scope = QComboBox()
        self.combo_scope.addItem("In view", 'view')
        self.combo_scope.addItem("Whole file", 'file')
        self.combo_scope.setToolTip(
            "In view: only the calls on screen in the spectrogram, following "
            "it as you scroll. Whole file: every call in the recording.")
        self.combo_scope.currentIndexChanged.connect(self.refresh_now)
        row.addWidget(self.combo_scope)
        row.addStretch(1)
        return row

    def _build_view_row(self) -> QHBoxLayout:
        row = QHBoxLayout()
        row.setSpacing(8)
        row.addWidget(QLabel("Smoothing:"))
        self.slider_smooth = QSlider(Qt.Horizontal)
        self.slider_smooth.setRange(0, _SMOOTH_MAX_STEPS)
        default_ms = DEFAULT_SMOOTH_FRAMES * _FALLBACK_DT * 1000.0
        try:
            ms = float(QSettings("FNT", "MAD").value(
                _SMOOTH_SETTINGS_KEY, default_ms))
        except Exception:
            ms = default_ms
        self.slider_smooth.setValue(
            int(round(min(max(ms, 0.0), _SMOOTH_MAX_STEPS * _SMOOTH_STEP_MS)
                      / _SMOOTH_STEP_MS)))
        self.slider_smooth.setFixedWidth(150)
        self.slider_smooth.setFocusPolicy(Qt.NoFocus)
        self.slider_smooth.setToolTip(
            "Centred moving average applied to every feature, over this much "
            "time, before the rate of pitch change is taken.\n\n"
            "Pitch is measured to the nearest frequency bin (about 244 Hz at "
            "nfft 1024 / 250 kHz), so frame by frame it moves in steps — and "
            "its rate of change is mostly those steps. A few ms of smoothing "
            "shows the call's real contour; off shows every raw frame.\n\n"
            "Changes only this picture. The CSV metrics are always computed "
            "from the unsmoothed frames.")
        self.slider_smooth.valueChanged.connect(self._on_smoothing_changed)
        row.addWidget(self.slider_smooth)
        self.lbl_smooth = QLabel("")
        self.lbl_smooth.setMinimumWidth(120)
        self.lbl_smooth.setStyleSheet("color:#aab4bd;")
        row.addWidget(self.lbl_smooth)
        self._update_smooth_label()

        row.addStretch(1)
        self.chk_rotate = QCheckBox("Auto-rotate")
        self.chk_rotate.setChecked(True)
        self.chk_rotate.setToolTip(
            "Turn slowly about the vertical axis. Dragging the view takes over "
            "while you hold it.")
        self.chk_rotate.toggled.connect(self._on_rotate_toggled)
        row.addWidget(self.chk_rotate)
        btn_reset = QPushButton("Reset View")
        btn_reset.setFocusPolicy(Qt.NoFocus)
        btn_reset.clicked.connect(self.reset_view)
        row.addWidget(btn_reset)
        return row

    def _legend_label(self) -> QLabel:
        def dot(key, text):
            r, g, b = self._pal[key]
            return (f"<span style='color:rgb({r},{g},{b}); font-weight:bold;'>"
                    f"●</span> {text}")
        lbl = QLabel(
            dot('confirmed', "accepted") + " &nbsp; " +
            dot('pending', "pending") + " &nbsp; " +
            dot('selected', "selected in spectrogram") +
            " &nbsp;·&nbsp; <i>stopped: lines brighten from onset (dot) to "
            "offset · playing: each call draws as the playhead reaches it, "
            "then dims</i>")
        lbl.setWordWrap(True)
        lbl.setStyleSheet("font-size:10px; color:#8a949d;")
        return lbl

    # -- scene ----------------------------------------------------------- #
    def _build_static_scene(self):
        """The unit cube the calls are drawn in, plus a floor grid for depth."""
        c = [-1.0, 1.0]
        corners = np.array([[x, y, z] for x in c for y in c for z in c])
        edges = [(i, j) for i in range(8) for j in range(i + 1, 8)
                 if np.sum(corners[i] != corners[j]) == 1]
        seg = np.array([corners[k] for e in edges for k in e])
        box = gl.GLLinePlotItem(pos=seg, color=_BOX_COLOR, width=1,
                                mode='lines', antialias=True)
        box.setGLOptions('translucent')
        self.view.addItem(box)

        grid = gl.GLGridItem()
        grid.setSize(2, 2)
        grid.setSpacing(0.25, 0.25)
        grid.translate(0, 0, -1)
        grid.setColor(_GRID_COLOR)
        self.view.addItem(grid)

    def reset_view(self):
        self.view.setCameraPosition(pos=Vector(0, 0, 0), distance=5.4,
                                    elevation=20, azimuth=40)
        self._follow_camera()

    def _clear_dynamic(self):
        for it in list(self._trail_items.values()) + self._dyn_items:
            try:
                self.view.removeItem(it)
            except Exception:
                pass
        self._trail_items = {}
        self._dyn_items = []
        self._onset_item = None
        self._head_item = None

    def _add_label(self, pos, text, size=9, bold=False):
        font = QFont()
        font.setPointSize(size)
        font.setBold(bold)
        it = gl.GLTextItem(pos=tuple(float(p) for p in pos), text=text,
                           color=QColor(*_TEXT_COLOR), font=font)
        self.view.addItem(it)
        self._label_items.append(it)

    def _near_corner(self) -> Tuple[int, int]:
        """(sx, sy): the bottom corner of the box nearest the camera, in the
        box's ±1 coordinates. pyqtgraph puts the camera at azimuth ``a`` along
        (cos a, sin a), so the near corner is just their signs."""
        a = np.radians(self.view.opts['azimuth'])
        return (1 if np.cos(a) >= 0 else -1, 1 if np.sin(a) >= 0 else -1)

    def _draw_axis_labels(self, ranges: Dict[str, Tuple[float, float]]):
        """Axis titles, with each axis's real-unit range near its two ends,
        on the edges facing the camera.

        Labels fixed to one corner spend half of every rotation behind the
        box, floating over the trails; so, as matplotlib's 3D axes do, they
        sit on the near bottom edges (X and Y) and a side edge (Z), and move
        when the view turns past a quadrant — see :meth:`_follow_camera`. The
        end numbers are inset along their own axis so the low ends of two axes
        never meet at a shared corner.
        """
        for it in self._label_items:
            try:
                self.view.removeItem(it)
            except Exception:
                pass
        self._label_items = []
        self._ranges = ranges
        sx, sy = self._label_corner = self._near_corner()
        kx, ky, kz = self._axes
        off = 1.18          # out from the edge, so text clears the box
        e = 0.82            # in from the corner, along the axis
        ox, oy = sx * off, sy * off
        self._add_label((0.0, oy, -1.0), label_for(kx), 10, True)
        self._add_label((ox, 0.0, -1.0), label_for(ky), 10, True)
        self._add_label((ox, -oy, 0.0), label_for(kz), 10, True)
        for key, lo_pos, hi_pos in (
            (kx, (-e, oy, -1.0), (e, oy, -1.0)),
            (ky, (ox, -e, -1.0), (ox, e, -1.0)),
            (kz, (ox, -oy, -e), (ox, -oy, e)),
        ):
            lo, hi = ranges[key]
            self._add_label(lo_pos, _fmt(lo), 8)
            self._add_label(hi_pos, _fmt(hi), 8)

    def _follow_camera(self):
        """Move the axis labels if the camera has turned to a new quadrant.
        Cheap when it hasn't — which is all but four ticks per rotation."""
        if self._ranges is not None and self._near_corner() != self._label_corner:
            self._draw_axis_labels(self._ranges)

    def _selected_id(self):
        sg = getattr(self._main, 'spectrogram', None)
        idx = getattr(sg, '_selected_ann_idx', None)
        anns = getattr(sg, 'annotations', None) or []
        if idx is not None and 0 <= idx < len(anns):
            return anns[idx].get('id')
        return None

    def _line_style(self, cls: str, selected: bool, n: int):
        """Per-vertex colours (onset dim → offset bright) and a width."""
        rgb = self._pal['selected'] if selected else (
            self._pal['confirmed'] if cls == 'accepted' else self._pal['pending'])
        colors = np.empty((n, 4))
        colors[:, :3] = np.array(rgb[:3]) / 255.0
        colors[:, 3] = direction_alpha(n, lo=0.3 if not selected else 0.6)
        return colors, (4.0 if selected else 2.0)

    # -- refresh --------------------------------------------------------- #
    def schedule_refresh(self, view_changed: bool = False):
        """Something about the calls or the view changed; redraw shortly.

        A scroll only matters in *In view* mode — in *Whole file* the drawn set
        doesn't depend on the view, and redrawing on every scroll step would
        just churn the GL scene."""
        if not (HAVE_GL and self.isVisible()):
            return
        if view_changed and self.combo_scope.currentData() == 'file':
            return
        self._refresh_timer.start()

    def update_selection(self):
        """Re-style trails for a new spectrogram selection — no recompute."""
        if not HAVE_GL or not self.isVisible() or not self._paths:
            return
        if self._playing:
            # The static style would overwrite the partly drawn trails; mark
            # them stale and let the next playhead tick restyle with the new
            # selection.
            for p in self._paths:
                p['_state'] = None
            if self._play_pos is not None:
                self.playback_tick(self._play_pos)
            return
        sel = self._selected_id()
        for p in self._paths:
            it = self._trail_items.get(p['id'])
            if it is None:
                continue
            n = len(it.pos)
            colors, width = self._line_style(p['cls'], p['id'] == sel, n)
            it.setData(color=colors, width=width)

    def refresh_now(self):
        if not HAVE_GL:
            return
        main = self._main
        sg = getattr(main, 'spectrogram', None)
        wav = main._active_wav_path() if hasattr(main, '_active_wav_path') \
            else None
        if wav != self._cache_wav:
            self._cache = {}
            self._cache_wav = wav
        self.setWindowTitle(
            "Call Trajectories (3D)"
            + (f" — {os.path.basename(wav)}" if wav else ""))

        self._axes = [c.currentData() for c in self._axis_combos]
        scope = self.combo_scope.currentData()
        sp = main._spec_params() if hasattr(main, '_spec_params') else \
            {'db_min': -100.0, 'db_max': -20.0}
        t_org = main._frame_time_origin_s() \
            if getattr(main, 'sample_rate', None) else 0.0
        self._t_org = t_org
        if sg is not None and getattr(sg, 'hop', None) and \
                getattr(main, 'sample_rate', None):
            self._dt = sg.hop / float(main.sample_rate)
            self._update_smooth_label()     # frames depend on this file's hop
        window = None
        if scope == 'view' and sg is not None:
            window = (float(sg.view_start), float(sg.view_end))

        paths: List[Dict] = []
        skipped = {'harmonics': 0, 'short': 0}
        if sg is not None and getattr(main, 'audio_data', None) is not None:
            from PyQt5.QtWidgets import QApplication
            # Only a whole-file pass can take long enough to show a cursor for;
            # in-view refreshes are a handful of cached calls, and flashing
            # the wait cursor on every scroll step would just be noise.
            busy = scope == 'file'
            if busy:
                QApplication.setOverrideCursor(Qt.WaitCursor)
            try:
                paths, skipped = collect_call_paths(
                    sg, sp['db_min'], sp['db_max'], t_org,
                    include_pending=self.chk_pending.isChecked(),
                    include_accepted=self.chk_accepted.isChecked(),
                    include_harmonics=self.chk_harmonics.isChecked(),
                    time_window=window, cache=self._cache, cache_tag=wav,
                    smooth_frames=self._smooth_frames())
            finally:
                if busy:
                    QApplication.restoreOverrideCursor()
        self._paths = paths
        self._draw(paths)
        self._set_status(paths, skipped, scope, wav)
        # Calls or axes changed mid-playback (a scroll, an Accept, a new axis):
        # the static redraw above is replaced before anything is painted, so
        # the animation carries on from where the playhead is.
        if self._playing and self._play_range is not None:
            self._enter_playback(*self._play_range)

    def _draw(self, paths: Sequence[Dict]):
        self._clear_dynamic()
        keys = self._axes
        ranges = axis_ranges([p['traj'] for p in paths], keys)
        self._draw_axis_labels(ranges)
        if not paths:
            return
        sel = self._selected_id()
        onsets, onset_colors = [], []
        for p in paths:
            pts = to_unit_cube(p['traj'], keys, ranges)
            ok = finite_rows(pts)
            pts = pts[ok]
            if len(pts) < 2:
                continue
            # Each point's moment in the recording — what the playhead is
            # compared against.
            p['_pts'] = pts
            p['_t'] = self._t_org + p['traj']['frame'][ok] * self._dt
            colors, width = self._line_style(p['cls'], p['id'] == sel,
                                             len(pts))
            it = gl.GLLinePlotItem(pos=pts, color=colors, width=width,
                                   mode='line_strip', antialias=True)
            it.setGLOptions('translucent')
            self.view.addItem(it)
            self._trail_items[p['id']] = it
            onsets.append(pts[0])
            onset_colors.append(colors[-1])
        if onsets:
            dots = gl.GLScatterPlotItem(pos=np.array(onsets),
                                        color=np.array(onset_colors), size=7,
                                        pxMode=True)
            dots.setGLOptions('translucent')
            self.view.addItem(dots)
            self._dyn_items.append(dots)
            self._onset_item = dots

    # -- playback ------------------------------------------------------- #
    def playback_started(self, start_s: float, end_s: float):
        """MAD's Play pressed: switch from the static picture to the player."""
        if not HAVE_GL or not self.isVisible():
            return
        # A queued redraw would land mid-animation; do it now instead, so the
        # calls being played are the current ones.
        if self._refresh_timer.isActive():
            self._refresh_timer.stop()
            self.refresh_now()
        self._enter_playback(start_s, end_s)

    def _enter_playback(self, start_s: float, end_s: float):
        self._playing = True
        self._play_range = (float(start_s), float(end_s))
        if self._onset_item is not None:
            # Onset dots would reveal calls before the playhead reaches them.
            self._onset_item.setVisible(False)
        if self._head_item is None:
            head = gl.GLScatterPlotItem(pos=np.zeros((0, 3)), size=16,
                                        pxMode=True)
            head.setGLOptions('translucent')
            self.view.addItem(head)
            self._dyn_items.append(head)
            self._head_item = head
        a, b = self._play_range
        for p in self._paths:
            p['_state'] = None
            t = p.get('_t')
            # Calls outside the played stretch sit the whole run out.
            p['_in_play'] = t is not None and t[-1] >= a and t[0] <= b
        self.playback_tick(self._play_pos if self._play_pos is not None
                           and a <= self._play_pos <= b else a)

    def _play_style(self, cls: str, n: int, active: bool, selected: bool):
        """A call being played brightens toward the playhead; a finished one
        becomes a faint ghost, so the call under the playhead is the one that
        reads."""
        rgb = self._pal['selected'] if selected else (
            self._pal['confirmed'] if cls == 'accepted' else self._pal['pending'])
        colors = np.empty((n, 4))
        colors[:, :3] = np.array(rgb[:3]) / 255.0
        if active:
            colors[:, 3] = direction_alpha(n, lo=0.15, hi=1.0)
            return colors, 3.0
        colors[:, 3] = 0.3
        return colors, 1.5

    def playback_tick(self, pos_s: float):
        """The playhead moved: grow each call's trail up to ``pos_s``."""
        if not HAVE_GL or not self.isVisible():
            return
        if not self._playing:
            # Opened (or re-shown) mid-playback: pick the run up from MAD.
            main = self._main
            if not getattr(main, 'is_playing', False):
                return
            self._play_pos = float(pos_s)
            self._enter_playback(main._playback_start_s, main._playback_end_s)
            return
        self._play_pos = float(pos_s)
        sel = self._selected_id()
        heads, head_colors = [], []
        n_reached = n_play = 0
        for p in self._paths:
            it = self._trail_items.get(p['id'])
            t = p.get('_t')
            if it is None or t is None:
                continue
            if not p.get('_in_play'):
                if p.get('_state') != 'out':
                    it.setVisible(False)
                    p['_state'] = 'out'
                continue
            n_play += 1
            n, active = playback_progress(t, pos_s)
            if n:
                n_reached += 1
            state = (n, active, p['id'] == sel)
            if state != p.get('_state'):
                p['_state'] = state
                if n < 2:
                    it.setVisible(False)
                else:
                    colors, width = self._play_style(
                        p['cls'], n, active, p['id'] == sel)
                    it.setData(pos=p['_pts'][:n], color=colors, width=width)
                    it.setVisible(True)
            if active and n >= 1:
                heads.append(p['_pts'][n - 1])
                rgb = self._pal['selected'] if p['id'] == sel else (
                    self._pal['confirmed'] if p['cls'] == 'accepted'
                    else self._pal['pending'])
                head_colors.append(_rgba(rgb, 1.0))
        if self._head_item is not None:
            if heads:
                self._head_item.setData(pos=np.array(heads),
                                        color=np.array(head_colors))
            else:
                self._head_item.setData(pos=np.zeros((0, 3)))
        self.lbl_status.setText(
            f"▶ Playing {self._play_range[0]:.3f}–{self._play_range[1]:.3f} s"
            f" · now {pos_s:.3f} s · {n_reached} of {n_play} call(s) reached")

    def playback_stopped(self):
        """Playback ended or was stopped: back to the static picture."""
        if not self._playing:
            return
        self._playing = False
        self._play_range = None
        self._play_pos = None
        if not HAVE_GL:
            return
        self._draw(self._paths)
        if self._status_args is not None:
            self._set_status(*self._status_args)

    def _set_status(self, paths, skipped, scope, wav):
        self._status_args = (paths, skipped, scope, wav)
        if wav is None or getattr(self._main, 'audio_data', None) is None:
            self.lbl_status.setText("Open a recording to see its calls here.")
            return
        where = "in view" if scope == 'view' else "in this recording"
        if not paths:
            shown = [n for n, c in (("pending", self.chk_pending),
                                    ("accepted", self.chk_accepted))
                     if c.isChecked()]
            what = " or ".join(shown) if shown else "selected"
            self.lbl_status.setText(
                f"No {what} masks {where}. Label a call or run inference — "
                "only masked calls are drawn, so their paths come from the "
                "call's own pixels rather than the noise around it.")
            return
        n_frames = sum(len(p['traj']['frame']) for p in paths)
        n_pend = sum(1 for p in paths if p['cls'] == 'pending')
        n_acc = len(paths) - n_pend
        text = (f"{len(paths)} call(s) {where} — {n_acc} accepted, "
                f"{n_pend} pending · {n_frames:,} frames")
        notes = []
        if skipped.get('harmonics'):
            notes.append(f"{skipped['harmonics']} harmonic(s) hidden")
        if skipped.get('short'):
            notes.append(f"{skipped['short']} too short to trace")
        if notes:
            text += " · " + ", ".join(notes)
        self.lbl_status.setText(text)

    # -- smoothing ------------------------------------------------------ #
    def _smooth_frames(self) -> int:
        """The slider's time converted to frames at the open file's hop."""
        dt = self._dt or _FALLBACK_DT
        ms = self.slider_smooth.value() * _SMOOTH_STEP_MS
        return int(round(ms / (dt * 1000.0)))

    def _update_smooth_label(self):
        k = self._smooth_frames()
        k_eff = k if k % 2 else k - 1     # the moving average is centred: odd
        if k_eff < 3:
            self.lbl_smooth.setText("off — raw frames")
        else:
            ms = self.slider_smooth.value() * _SMOOTH_STEP_MS
            self.lbl_smooth.setText(f"{ms:g} ms · {k_eff} frames")

    def _on_smoothing_changed(self, _v=0):
        self._update_smooth_label()
        try:
            QSettings("FNT", "MAD").setValue(
                _SMOOTH_SETTINGS_KEY,
                self.slider_smooth.value() * _SMOOTH_STEP_MS)
        except Exception:
            pass
        # Debounced: a drag fires per step. Only re-smoothing and re-drawing —
        # the per-call spectrogram work is cached.
        if self.isVisible():
            self._refresh_timer.start()

    def _on_axes_changed(self, _i=0):
        # Features are cached per call; changing axes only re-projects.
        self.refresh_now()

    # -- rotation -------------------------------------------------------- #
    def _rotate_tick(self):
        if self.view.dragging:
            return
        self.view.orbit(-ROTATE_DEG_PER_S * _TICK_MS / 1000.0, 0)
        self._follow_camera()

    def _on_rotate_toggled(self, on: bool):
        if on and self.isVisible():
            self._rotate_timer.start()
        else:
            self._rotate_timer.stop()

    def showEvent(self, ev):
        super().showEvent(ev)
        if HAVE_GL:
            if self.chk_rotate.isChecked():
                self._rotate_timer.start()
            self.refresh_now()
        self.visibility_changed.emit(True)

    def hideEvent(self, ev):
        # A hidden window has no reason to spend a timer tick 30 times a second.
        if HAVE_GL:
            self._rotate_timer.stop()
            self._refresh_timer.stop()
        self._playing = False           # re-entered from MAD if re-shown mid-run
        self._play_range = None
        self._play_pos = None
        super().hideEvent(ev)
        # Spontaneous hides are the window system's (minimising); the window is
        # still open, so the toggle should stay on.
        if not ev.spontaneous():
            self.visibility_changed.emit(False)
