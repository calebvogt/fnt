"""GUI-free rendering core for UWB tracking animations.

Extracted from the preprocessing tool so the rendering is reusable and testable
independently of the GUI. Nothing here touches Qt: all UI concerns
(dialogs, progress bars, event pumping, cancellation) are handled by the caller
through the ``log``, ``progress`` and ``is_cancelled`` callbacks.

Contract for the input DataFrame: one row per fix with columns ``Timestamp``
(tz-aware datetime), ``shortid``/``ID``, and ``smoothed_x``/``smoothed_y`` (or
``location_x``/``location_y``, used as a fallback). ``prepare_animation_data``
normalises these.
"""

import gc
import os

import numpy as np
import pandas as pd
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Circle, Polygon as MplPolygon
from matplotlib.collections import LineCollection
import matplotlib.patheffects as pe

import cv2


# Video render resolution per the quality preset shown in the UI.
QUALITY_DPI = {"Draft (Fast)": 75, "Standard": 100, "High Quality": 150}


def frame_interval_seconds(speed_multiplier, fps):
    """Real seconds of tracking time represented by one video frame.

    ``speed_multiplier`` seconds of real time elapse per second of video, spread
    across ``fps`` frames — so each frame advances ``speed_multiplier / fps`` s.
    """
    return speed_multiplier / float(fps)


def prepare_animation_data(data, tag_identities=None):
    """Normalise a positions frame for rendering.

    Ensures an ``ID`` column, a ``sex`` column, and ``smoothed_x``/``smoothed_y``
    (falling back to ``location_x``/``location_y``), applies any configured
    identities, and sorts by (ID, Timestamp). Returns a copy.
    """
    data = data.copy()
    if 'ID' not in data.columns and 'shortid' in data.columns:
        data['ID'] = data['shortid']
    if 'sex' not in data.columns:
        data['sex'] = 'M'
    if tag_identities:
        for tag_id, info in tag_identities.items():
            mask = data['ID'] == tag_id
            data.loc[mask, 'sex'] = info.get('sex', 'M')
            data.loc[mask, 'custom_identity'] = info.get('identity', str(tag_id))
    if 'smoothed_x' not in data.columns and 'location_x' in data.columns:
        data['smoothed_x'] = data['location_x']
    if 'smoothed_y' not in data.columns and 'location_y' in data.columns:
        data['smoothed_y'] = data['location_y']
    if 'ID' in data.columns:
        data = data.sort_values(by=['ID', 'Timestamp'])
    else:
        data = data.sort_values(by=['Timestamp'])
    return data


def downsample_to_hz(data, hz):
    """Keep the first fix per 1/hz-second bin, per tag (resolution-safe floor).

    Returns ``data`` unchanged if ``hz`` is falsy or the frame is empty.
    """
    if not hz or data.empty:
        return data
    d = data.copy()
    d['_bin'] = d['Timestamp'].dt.floor(pd.Timedelta(seconds=1.0 / hz))
    id_col = 'shortid' if 'shortid' in d.columns else 'ID'
    d = d.groupby([id_col, '_bin'], as_index=False).first().drop(columns='_bin')
    return d


_NEUTRAL_COLOR = 'blue'


def build_tag_styles(tags, tag_identities=None, use_custom_identities=False,
                     color_by="None"):
    """Map each tag id to its {'label', 'color'} for the animation.

    Labels follow the configured identities (``sex-identity`` when available,
    otherwise a ``HexID`` label). Colour is governed by ``color_by``:

    * ``"None"`` (default) — every tag drawn in a single neutral colour.
    * ``"ID"``            — a distinct colour per tag (``tab20`` palette).
    * ``"sex"``           — blue = M, red = F.
    """
    styles = {}
    tag_identities = tag_identities or {}
    color_mode = (color_by or "None").lower()

    id_palette = None
    if color_mode == "id":
        import matplotlib.pyplot as plt
        cmap = plt.get_cmap('tab20')
        id_palette = {tag: cmap(idx % 20) for idx, tag in enumerate(tags)}

    for tag in tags:
        # Label: prefer configured identity, else HexID.
        if use_custom_identities and tag in tag_identities:
            info = tag_identities[tag]
            sex = info.get('sex', 'M')
            identity = info.get('identity', str(tag))
            label = f"{sex}-{identity}"
        else:
            info = tag_identities.get(tag, {})
            sex = info.get('sex', 'M')
            hex_id = hex(int(tag)).upper().replace('0X', '')
            label = f"HexID {hex_id}"

        if color_mode == "id":
            color = id_palette[tag]
        elif color_mode == "sex":
            color = 'blue' if sex == 'M' else 'red'
        else:  # "None"
            color = _NEUTRAL_COLOR

        styles[tag] = {'label': label, 'color': color}
    return styles


def draw_static_context(ax, layers, *, bg_image=None, bg_extent=None,
                        zones_xml=None, arena_zones=None, anchors=None,
                        rois=None, log=None):
    """Draw the background image, zone polygons (with labels) and anchors.

    Gated by ``layers`` (dict with 'background'/'zones'/'anchors' booleans).
    ``zones_xml`` is the preview's list of {name, color, points} polygons and is
    preferred, so the video shows the zones in the SAME authored colours the
    preview drew them in; ``arena_zones`` (a DataFrame with columns
    ('zone', 'x', 'y')) is the fallback for a config that carries only the flat
    table. ``anchors`` is a list of dicts with 'x'/'y'. Must be idempotent — the
    frame loop calls it after every ``ax.clear()``.
    """
    layers = layers or {}
    if layers.get('background') and bg_image is not None and bg_extent is not None:
        ax.imshow(bg_image, extent=list(bg_extent), origin='upper',
                  aspect='auto', alpha=0.6, zorder=0)
    if layers.get('zones'):
        try:
            if zones_xml:
                for z in zones_xml:
                    pts = np.asarray(z.get('points'), float)
                    if pts.ndim != 2 or len(pts) < 3:
                        continue
                    # "Arena" is the enclosure boundary, not a zone: outline
                    # only, so it never tints the floor it encloses.
                    is_bounds = (z.get('name', '') or '').strip().lower() == 'arena'
                    color = z.get('color', '#888888')
                    ax.add_patch(MplPolygon(
                        pts, closed=True,
                        facecolor='none' if is_bounds else color,
                        edgecolor=color,
                        alpha=1.0 if is_bounds else 0.28,
                        linewidth=2.0 if is_bounds else 1.2, zorder=1))
                    if not is_bounds:
                        cx, cy = pts[:, 0].mean(), pts[:, 1].mean()
                        ax.text(cx, cy, z.get('name', ''), fontsize=8,
                                color='#111111', fontweight='bold',
                                ha='center', va='center', zorder=1,
                                path_effects=[pe.withStroke(linewidth=2.2,
                                                            foreground='white')])
            elif arena_zones is not None and not arena_zones.empty:
                for zone_name in arena_zones['zone'].unique():
                    coords = arena_zones[arena_zones['zone'] == zone_name][['x', 'y']].values
                    if len(coords) >= 3:  # need >=3 points for a polygon
                        ax.add_patch(MplPolygon(coords, fill=False, edgecolor='black',
                                                linewidth=1.5, linestyle='--', zorder=1))
                        cx, cy = coords[:, 0].mean(), coords[:, 1].mean()
                        ax.text(cx, cy, zone_name, fontsize=8, ha='center', va='center',
                                bbox=dict(boxstyle='round,pad=0.3', facecolor='white', alpha=0.5),
                                zorder=1)
        except Exception as e:  # never let a zone glitch abort the render
            if log:
                log(f"Error drawing zones in animation: {e}")
    # The user's own regions (see fnt.uwb.uwb_roi): outline plus a faint
    # wash, so a track running through one stays readable.
    if layers.get('rois', True) and rois:
        from fnt.uwb.uwb_preview_canvas import label_halo
        for r in rois:
            pts = np.asarray(r.get('points'), dtype=float).reshape(-1, 2)
            if len(pts) < 3:
                continue
            col = r.get('color', '#e6194b')
            ax.add_patch(MplPolygon(pts, closed=True, facecolor=col,
                                    alpha=0.10, edgecolor='none', zorder=1.5))
            ax.add_patch(MplPolygon(pts, closed=True, facecolor='none',
                                    edgecolor=col,
                                    linewidth=float(r.get('linewidth', 2.0)),
                                    zorder=1.6))
            ax.annotate(r.get('name', ''),
                        (pts[:, 0].mean(), pts[:, 1].mean()), color=col,
                        fontsize=8, fontweight='bold', ha='center',
                        va='center', zorder=5,
                        path_effects=label_halo(col, 2.2))

    if layers.get('anchors') and anchors:
        ax.scatter([a['x'] for a in anchors], [a['y'] for a in anchors],
                   marker='^', s=40, c='#f2c24f', edgecolors='none', zorder=2)


def compute_axis_limits(data, layers=None, bg_extent=None, pad_frac=0.05):
    """(x_min, x_max, y_min, y_max) spanning the data (+ background) with padding."""
    x_min, x_max = data['smoothed_x'].min(), data['smoothed_x'].max()
    y_min, y_max = data['smoothed_y'].min(), data['smoothed_y'].max()
    if (layers or {}).get('background') and bg_extent is not None:
        x_min = min(x_min, bg_extent[0]); x_max = max(x_max, bg_extent[1])
        y_min = min(y_min, bg_extent[2]); y_max = max(y_max, bg_extent[3])
    x_range, y_range = x_max - x_min, y_max - y_min
    x_pad = x_range * pad_frac if x_range > 0 else 1
    y_pad = y_range * pad_frac if y_range > 0 else 1
    return x_min - x_pad, x_max + x_pad, y_min - y_pad, y_max + y_pad


def render_animation(data, output_path, *, frame_interval, trailing_window, fps,
                     dpi=100, speed_text="", title="UWB Tracking Animation",
                     layers=None, bg_image=None, bg_extent=None,
                     zones_xml=None, arena_zones=None, anchors=None,
                     rois=None,
                     tag_identities=None, use_custom_identities=False,
                     color_by="None", marker_size=10, show_battery=False,
                     show_speed=False, show_step=False, gap_s=60.0,
                     label_color=None, label_outline=True,
                     axis_limits=None, behavior=None, show_trail=True,
                     show_labels=True, time_range=None, data_view=None,
                     is_cancelled=None, progress=None, log=None):
    """Render tracking frames to an MP4 at ``output_path``.

    Each frame shows the current position at its timestamp plus the trailing
    ``trailing_window`` seconds behind it. Returns ``output_path`` on success,
    or None on failure/cancel.

    ``behavior`` puts the preview's behaviour overlays into the video, so an
    export looks like the preview it was tuned in. It is a dict of:

        radius       social radius in metres, or None for no circles
        circle_color edge colour for those circles
        grid         int64 ns timestamps of the classification frames
        tags         tag ids, in the column order of `loc`/`soc`
        soc          (frames, tags) social codes, or None
        links        (n, 4) int32 rows of (frame, i, j, kind), sorted by frame
        link_colors  {kind: colour}
        state_labels {"soc": {code: text}}
        state_colors {"soc": {code: colour}}

    The classification grid is independent of the video frame rate: each video
    frame takes the classification row nearest its own timestamp, so the
    overlay stays correct whatever speed the video is rendered at.

    Callbacks (all optional):
        is_cancelled() -> bool : checked before each frame; True aborts.
        progress(i, total)     : per-frame hook for UI update / event pump.
        log(str)               : status messages.
    """
    def _log(msg):
        if log:
            log(msg)

    layers = layers or {}

    # Callers rendering several clips of the same recording (the per-day
    # videos) pass one shared set of limits so every clip is drawn in the
    # same arena frame and can be concatenated without the view shifting.
    x_min, x_max, y_min, y_max = (
        axis_limits if axis_limits is not None
        else compute_axis_limits(data, layers, bg_extent))
    y_range = y_max - y_min  # for label offset

    # `time_range` lets a caller pass extra LEADING rows - to warm the
    # smoothing and fill the first frame's trail - without those rows
    # lengthening the video. Frames are generated over the requested span only;
    # every row in `data` is still available to draw them, so frame one opens
    # with a full trail instead of building one up.
    start, end = (time_range if time_range
                  else (data['Timestamp'].min(), data['Timestamp'].max()))
    total_seconds = (end - start).total_seconds()
    num_frames = int(total_seconds / frame_interval) + 1
    time_starts = [start + pd.Timedelta(seconds=i * frame_interval) for i in range(num_frames)]
    time_starts = [t for t in time_starts if t <= end]
    total_frames = len(time_starts)
    _log(f"Creating {total_frames} animation frames...")

    # Pre-slice each tag's trajectory + resolve its label/colour once.
    tags = list(data['shortid'].unique()) if 'shortid' in data.columns else list(data['ID'].unique())
    id_col = 'shortid' if 'shortid' in data.columns else 'ID'
    styles = build_tag_styles(tags, tag_identities, use_custom_identities, color_by)
    keep_cols = ['Timestamp', 'smoothed_x', 'smoothed_y']
    if show_battery and 'battery_voltage' in data.columns:
        keep_cols.append('battery_voltage')
    # Flat, sorted arrays rather than a DataFrame per tag. The frame loop
    # below takes the trailing window by binary search on `t_ns`; masking the
    # DataFrame instead cost ~15 ms a frame on one day of a 17-tag trial,
    # because every frame re-scanned every tag's ENTIRE series. Timestamps are
    # already sorted, so two searchsorted calls give the identical slice.
    tag_data_dict = {}
    for tag in tags:
        sub = data[data[id_col] == tag][keep_cols].sort_values('Timestamp')
        bat = (sub['battery_voltage'].to_numpy(dtype='float64')
               if 'battery_voltage' in sub.columns else None)
        t_ns = sub['Timestamp'].dt.as_unit('ns').astype('int64').to_numpy()
        xs = sub['smoothed_x'].to_numpy(dtype='float64')
        ys = sub['smoothed_y'].to_numpy(dtype='float64')
        step = speed = None
        if show_speed or show_step:
            # Measured here rather than taken from a column, so every caller
            # (clip, export, batch worker) gets the same numbers, on the same
            # coordinates the frame actually draws. A step spanning more than
            # gap_s is dropped, matching the preview's Time gap grouping.
            step = np.full(len(xs), np.nan)
            speed = np.full(len(xs), np.nan)
            if len(xs) > 1:
                dt = np.diff(t_ns) / 1e9
                d = np.hypot(np.diff(xs), np.diff(ys))
                ok = (dt > 0) & (dt <= gap_s)
                step[1:] = np.where(ok, d, np.nan)
                with np.errstate(divide='ignore', invalid='ignore'):
                    speed[1:] = np.where(ok, d / dt, np.nan)
        tag_data_dict[tag] = {
            # int64 nanoseconds: directly comparable to a Timestamp's own
            # .value, and free of any per-comparison timezone handling.
            't_ns': t_ns,
            'xs': xs,
            'ys': ys,
            'battery': bat,
            'step': step,
            'speed': speed,
            'label': styles[tag]['label'],
            'color': styles[tag]['color']}

    # GUI-free Agg figure/canvas (no pyplot global state).
    # The Data View widens the CANVAS rather than shrinking the arena: the
    # arena is the thing being watched, and squeezing it to make room for the
    # numbers would trade the more important half for the less.
    arena_w, fig_h = 10.0, 8.0
    panel = None
    panel_w = 0.0
    dv_fmt = None
    if data_view:
        from fnt.uwb import dataview_panel as DVP
        from fnt.uwb.dataview import human_duration as _hd
        dv_fmt = data_view.get('fmt') or _hd
        n_a = len(data_view.get('animals') or ())
        n_r = len(data_view.get('roi_names') or ())
        dyad_mode = data_view.get('dyad_mode', 'partner')
        dyad_rows = max(n_a - 1, 0) if dyad_mode == 'triangle' else n_a
        rows = (2 + n_a) + 2 + (2 + dyad_rows)
        fs = DVP._fit_fontsize(rows, fig_h)
        panel_w = DVP.panel_width_in(n_a, n_r, fs, dyad_mode)
    fig = Figure(figsize=(arena_w + panel_w, fig_h), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    frac = arena_w / (arena_w + panel_w)
    ax = fig.add_axes([0.08 * frac, 0.08, 0.86 * frac, 0.86])
    ax.grid(False)

    # Draw the unchanging scene (background image, zones, anchors, axes) ONCE and
    # cache it, then blit only the moving markers/trails each frame. The previous
    # implementation did ax.clear() + redraw the full-resolution background image
    # on *every* frame — very slow, and a heavy source of per-frame native memory
    # churn in the Agg/NumPy layer. Caching removes essentially all of that work.
    draw_static_context(ax, layers, bg_image=bg_image, bg_extent=bg_extent,
                        zones_xml=zones_xml, arena_zones=arena_zones,
                        anchors=anchors, rois=rois, log=log)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_xlabel("X Position (m)", fontsize=12)
    ax.set_ylabel("Y Position (m)", fontsize=12)

    # Built BEFORE the background is cached, so the panel's furniture (titles,
    # headers, row labels, frame) bakes into the cached scene and only the
    # numbers are redrawn per frame.
    if data_view:
        panel = DVP.DataViewPanel(
            fig, [frac + 0.012, 0.06, (1.0 - frac) - 0.024, 0.88],
            data_view['animals'], data_view.get('roi_names') or [],
            data_view.get('occupancy') or {},
            data_view.get('overlaps') or {},
            text_color=data_view.get('text_color', '#e6e6e6'),
            grid_color=data_view.get('grid_color', '#4a4a4a'),
            bg_color=data_view.get('bg_color', '#1b1b1b'),
            accent=data_view.get('accent', '#ffd166'),
            dyad_mode=dyad_mode)

    canvas.draw()
    width, height = canvas.get_width_height()
    static_bg = canvas.copy_from_bbox(fig.bbox)  # cached static scene

    # Persistent per-tag artists, updated in place each frame (set_data/set_text)
    # rather than created/destroyed — no artist-list growth, minimal allocation.
    # animated=True keeps them out of canvas.draw(), so they never bake into the
    # cached background; they are drawn only via draw_artist during blitting.
    # Per-tag label styling, inherited from the preview. The readouts used to
    # be hard-coded black, which vanished over a dark floorplan; the halo is
    # what makes any of this text survive an arbitrary background image.
    from fnt.uwb.uwb_preview_canvas import label_halo
    read_col = label_color or '#000000'
    read_fx = label_halo(read_col, 1.8) if label_outline else []
    # The behaviour state changes colour per frame, so its halo has to follow.
    # Memoised on the colour: the palette is a handful of entries and this sits
    # inside the per-frame loop of a render that can run for hours.
    _state_fx = {}

    def state_halo(col):
        if not label_outline:
            return []
        if col not in _state_fx:
            _state_fx[col] = label_halo(col, 1.8)
        return _state_fx[col]

    title_artist = ax.set_title("", fontsize=14, fontweight='bold')
    title_artist.set_animated(True)
    dyn = {}
    for tag, tag_info in tag_data_dict.items():
        color = tag_info['color']
        (trail_line,) = ax.plot([], [], color=color, alpha=0.5, linewidth=1,
                                animated=True)
        (marker,) = ax.plot([], [], 'o', color=color, markersize=marker_size,
                            animated=True)
        label = ax.text(0, 0, '', fontsize=10, ha='center', color=color,
                        fontweight='bold', animated=True,
                        path_effects=(label_halo(color, 2.0) if label_outline
                                      else []))
        # Battery voltage under the ID label (drawn only when show_battery is
        # on). va='top' so it hangs beneath the shared anchor.
        batt_label = ax.text(0, 0, '', fontsize=7, ha='center', va='top',
                             color=read_col, animated=True,
                             path_effects=read_fx)
        # Speed / step distance on one line under the voltage.
        readout_label = ax.text(0, 0, '', fontsize=7, ha='center', va='top',
                                color=read_col, animated=True,
                                path_effects=read_fx)
        dyn[tag] = (trail_line, marker, label, batt_label, readout_label)

    # Behaviour overlays get the same treatment as everything else here: one
    # persistent artist per tag, repositioned each frame rather than recreated,
    # so a multi-hour render allocates nothing per frame.
    beh = behavior or None
    beh_radius = float(beh['radius']) if beh and beh.get('radius') else None
    beh_grid = np.asarray(beh['grid'], dtype='int64') if beh else np.empty(0, 'int64')
    beh_links = np.asarray(beh['links'], dtype='int64').reshape(-1, 4) if beh else None
    beh_col_of = {t: i for i, t in enumerate(beh['tags'])} if beh else {}
    beh_circles, beh_states = {}, {}
    link_lc = None
    if beh:
        for tag in tag_data_dict:
            if beh_radius:
                circ = Circle((0, 0), beh_radius, fill=False,
                              linestyle=(0, (2, 2)), linewidth=1.2,
                              edgecolor=beh.get('circle_color') or '#9fb3c8',
                              alpha=0.9, zorder=4.5, animated=True)
                circ.set_visible(False)
                ax.add_patch(circ)
                beh_circles[tag] = circ
            beh_states[tag] = ax.text(0, 0, '', fontsize=7, ha='center',
                                      va='bottom', animated=True)
        # Dashed like the circles they join and at the same weight, so the
        # radius and the link it produced read as one thing; the longer dash is
        # what tells them apart. Matches the preview. The tag markers here are
        # Line2D at the default zorder 2, so 4.6 already puts the link across
        # the icons (the preview needs 5.5 because its markers are a scatter at
        # zorder 5); staying under 5 keeps ROI name labels readable.
        link_lc = LineCollection([], linewidths=1.2, alpha=0.95,
                                 linestyle=(0, (3, 2)), zorder=4.6,
                                 animated=True)
        ax.add_collection(link_lc)

    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not video_writer.isOpened():
        _log("✗ Could not open VideoWriter")
        return None
    _log(f"Video dimensions: {width}x{height}")

    for i, frame_start in enumerate(time_starts):
        if is_cancelled and is_cancelled():
            video_writer.release()
            _log("✗ Animation cancelled")
            return None
        if progress:
            progress(i, total_frames)

        # frame_start IS the current display time. Trail = the trailing_window
        # seconds behind it, so the window is [current - trail, current] and the
        # marker sits at `current`.
        window_start = frame_start - pd.Timedelta(seconds=trailing_window)
        # Same bounds as the mask this replaced: >= window_start ('left') and
        # <= frame_start ('right').
        win_ns = window_start.as_unit('ns').value
        now_ns = frame_start.as_unit('ns').value

        # Nearest classification row to this frame's display time. The grid is
        # regular, so this stays right no matter how the video speed compares
        # to the classification rate.
        beh_i = -1
        if beh is not None and len(beh_grid):
            beh_i = int(np.searchsorted(beh_grid, frame_start.value))
            beh_i = min(max(beh_i, 0), len(beh_grid) - 1)
        positions = {}

        canvas.restore_region(static_bg)  # wipe back to the cached scene

        title_text = title
        if speed_text:
            title_text += f" - Speed: {speed_text}"
        title_text += f"\nTime: {frame_start.strftime('%Y-%m-%d %H:%M:%S')}"
        title_artist.set_text(title_text)
        ax.draw_artist(title_artist)

        for tag, tag_info in tag_data_dict.items():
            trail_line, marker, label, batt_label, readout_label = dyn[tag]
            state_txt = beh_states.get(tag)
            circ = beh_circles.get(tag)
            t_ns = tag_info['t_ns']
            lo = int(np.searchsorted(t_ns, win_ns, 'left'))
            hi = int(np.searchsorted(t_ns, now_ns, 'right'))
            if hi <= lo:
                # Tag not reporting in this window (e.g. battery dead): hide it.
                trail_line.set_data([], [])
                marker.set_data([], [])
                label.set_text('')
                batt_label.set_text('')
                readout_label.set_text('')
                if state_txt is not None:
                    state_txt.set_text('')
                if circ is not None:
                    circ.set_visible(False)
            else:
                xs = tag_info['xs'][lo:hi]
                ys = tag_info['ys'][lo:hi]
                trail_line.set_data(xs, ys if show_trail else [])
                if not show_trail:
                    trail_line.set_data([], [])
                cx, cy = xs[-1], ys[-1]
                marker.set_data([cx], [cy])
                positions[tag] = (cx, cy)
                if circ is not None:
                    circ.set_center((cx, cy))
                    circ.set_visible(True)
                # The animal's social state, or nothing. An animal doing
                # nothing social is left unlabelled rather than carrying the
                # locomotor word that used to fill that space.
                state_text, state_col = '', '#cccccc'
                if state_txt is not None and beh_i >= 0:
                    j = beh_col_of.get(tag)
                    sc_ = beh.get('soc')
                    if j is not None and sc_ is not None:
                        code = int(sc_[beh_i, j])
                        txt = beh['state_labels']['soc'].get(code, '')
                        if txt:
                            state_text = txt
                            state_col = beh['state_colors']['soc'].get(
                                code, state_col)
                if state_txt is not None:
                    state_txt.set_text(state_text)
                    state_txt.set_color(state_col)
                    state_txt.set_path_effects(state_halo(state_col))
                    state_txt.set_position((cx, cy + y_range * 0.02))
                # The state line sits where the ID normally does, so lift the
                # ID clear of it when both are shown.
                lift = y_range * 0.028 if state_text else 0.0
                # Speed / step distance share one line, as in the preview.
                bits = []
                if show_speed and tag_info['speed'] is not None:
                    sv = tag_info['speed'][hi - 1]
                    if np.isfinite(sv):
                        bits.append(f"{sv:.3f} m/s")
                if show_step and tag_info['step'] is not None:
                    dv = tag_info['step'][hi - 1]
                    if np.isfinite(dv):
                        bits.append(f"{dv:.2f} m")
                readout_text = ' · '.join(bits)
                if show_battery:
                    # Lift the label and hang the voltage beneath it (shared
                    # anchor at cy+off: label grows up, voltage grows down).
                    off = y_range * 0.035 + lift
                    label.set_position((cx, cy + off))
                    label.set_verticalalignment('bottom')
                    label.set_text(tag_info['label'] if show_labels else '')
                    bv = (tag_info['battery'][hi - 1]
                          if tag_info['battery'] is not None else None)
                    batt_label.set_position((cx, cy + off))
                    batt_label.set_text(f"{bv:.2f} V" if bv is not None and pd.notna(bv) else '')
                else:
                    label.set_position((cx, cy + y_range * 0.02 + lift))
                    label.set_verticalalignment('baseline')
                    label.set_text(tag_info['label'] if show_labels else '')
                    batt_label.set_text('')
                # BELOW the marker, not appended to the ID/voltage stack above
                # it: that stack hangs downward and would put this line on top
                # of the dot. Clearance is a share of the view, so it holds at
                # any arena size.
                readout_label.set_position((cx, cy - y_range * 0.035))
                readout_label.set_text(readout_text)
            ax.draw_artist(trail_line)
            ax.draw_artist(marker)
            ax.draw_artist(label)
            ax.draw_artist(batt_label)
            ax.draw_artist(readout_label)
            if tag in beh_circles and beh_circles[tag].get_visible():
                ax.draw_artist(beh_circles[tag])
            if tag in beh_states:
                ax.draw_artist(beh_states[tag])

        # Links last: they join two markers, so every position has to be known
        # before any of them can be drawn.
        if link_lc is not None:
            segs, cols = [], []
            if beh_i >= 0 and beh_links is not None and len(beh_links):
                lo = int(np.searchsorted(beh_links[:, 0], beh_i, 'left'))
                hi = int(np.searchsorted(beh_links[:, 0], beh_i, 'right'))
                for _f, a, b, kind in beh_links[lo:hi]:
                    pa = positions.get(beh['tags'][a])
                    pb = positions.get(beh['tags'][b])
                    if pa is not None and pb is not None:
                        segs.append([pa, pb])
                        cols.append(beh['link_colors'].get(int(kind), '#9fb3c8'))
            link_lc.set_segments(segs)
            if cols:
                link_lc.set_color(cols)
            ax.draw_artist(link_lc)

        if panel is not None:
            # Read the cumulative curves at THIS frame's trial time. Nothing
            # is accumulated frame to frame, so the numbers are independent of
            # the video's speed and frame rate, and the final frame carries
            # the same totals the CSV does.
            panel.update(now_ns / 1e9, dv_fmt)
            panel.draw()

        canvas.blit(fig.bbox)
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
        video_writer.write(cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR))

    # No end card. The video is the tracking view and its counters, and
    # nothing else; a wall of totals appended to the end is not what anyone
    # scrubbing an animation is looking for. DataViewPanel.render_totals_card
    # is still there if that changes.

    video_writer.release()
    fig.clear()
    gc.collect()

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        _log(f"✓ Video compilation complete: {os.path.getsize(output_path):,} bytes")
        return output_path
    _log("✗ Video file was not created or is empty")
    return None
