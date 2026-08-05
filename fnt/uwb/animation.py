"""GUI-free rendering core for UWB tracking animations.

Extracted from the preprocessing tool so the same code drives both the current
export and the planned UWB Studio. Nothing here touches Qt: all UI concerns
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
from matplotlib.patches import Polygon as MplPolygon

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
                        arena_zones=None, anchors=None, log=None):
    """Draw the background image, zone polygons (with labels) and anchors.

    Gated by ``layers`` (dict with 'background'/'zones'/'anchors' booleans).
    ``arena_zones`` is a DataFrame with columns ('zone', 'x', 'y'); ``anchors``
    is a list of dicts with 'x'/'y'. Must be idempotent — the frame loop calls
    it after every ``ax.clear()``.
    """
    layers = layers or {}
    if layers.get('background') and bg_image is not None and bg_extent is not None:
        ax.imshow(bg_image, extent=list(bg_extent), origin='upper',
                  aspect='auto', alpha=0.6, zorder=0)
    if layers.get('zones') and arena_zones is not None and not arena_zones.empty:
        try:
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
                     arena_zones=None, anchors=None,
                     tag_identities=None, use_custom_identities=False,
                     color_by="None", marker_size=10, show_battery=False,
                     is_cancelled=None, progress=None, log=None):
    """Render tracking frames to an MP4 at ``output_path``.

    Each frame shows the current position at its timestamp plus the trailing
    ``trailing_window`` seconds behind it. Returns ``output_path`` on success,
    or None on failure/cancel.

    Callbacks (all optional):
        is_cancelled() -> bool : checked before each frame; True aborts.
        progress(i, total)     : per-frame hook for UI update / event pump.
        log(str)               : status messages.
    """
    def _log(msg):
        if log:
            log(msg)

    layers = layers or {}

    x_min, x_max, y_min, y_max = compute_axis_limits(data, layers, bg_extent)
    y_range = y_max - y_min  # for label offset

    start = data['Timestamp'].min()
    end = data['Timestamp'].max()
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
    tag_data_dict = {}
    for tag in tags:
        sub = data[data[id_col] == tag][keep_cols].sort_values('Timestamp')
        tag_data_dict[tag] = {'data': sub,
                              'label': styles[tag]['label'],
                              'color': styles[tag]['color']}

    # GUI-free Agg figure/canvas (no pyplot global state).
    fig = Figure(figsize=(10, 8), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    ax.grid(False)

    # Draw the unchanging scene (background image, zones, anchors, axes) ONCE and
    # cache it, then blit only the moving markers/trails each frame. The previous
    # implementation did ax.clear() + redraw the full-resolution background image
    # on *every* frame — very slow, and a heavy source of per-frame native memory
    # churn in the Agg/NumPy layer. Caching removes essentially all of that work.
    draw_static_context(ax, layers, bg_image=bg_image, bg_extent=bg_extent,
                        arena_zones=arena_zones, anchors=anchors, log=log)
    ax.set_xlim(x_min, x_max)
    ax.set_ylim(y_min, y_max)
    ax.set_aspect('equal')
    ax.set_xlabel("X Position (m)", fontsize=12)
    ax.set_ylabel("Y Position (m)", fontsize=12)

    canvas.draw()
    width, height = canvas.get_width_height()
    static_bg = canvas.copy_from_bbox(fig.bbox)  # cached static scene

    # Persistent per-tag artists, updated in place each frame (set_data/set_text)
    # rather than created/destroyed — no artist-list growth, minimal allocation.
    # animated=True keeps them out of canvas.draw(), so they never bake into the
    # cached background; they are drawn only via draw_artist during blitting.
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
                        fontweight='bold', animated=True)
        # Battery voltage under the ID label, small black font (drawn only when
        # show_battery is on). va='top' so it hangs beneath the shared anchor.
        batt_label = ax.text(0, 0, '', fontsize=7, ha='center', va='top',
                             color='#000000', animated=True)
        dyn[tag] = (trail_line, marker, label, batt_label)

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

        canvas.restore_region(static_bg)  # wipe back to the cached scene

        title_text = title
        if speed_text:
            title_text += f" - Speed: {speed_text}"
        title_text += f"\nTime: {frame_start.strftime('%Y-%m-%d %H:%M:%S')}"
        title_artist.set_text(title_text)
        ax.draw_artist(title_artist)

        for tag, tag_info in tag_data_dict.items():
            trail_line, marker, label, batt_label = dyn[tag]
            tag_df = tag_info['data']
            trailing = tag_df[(tag_df['Timestamp'] >= window_start) &
                              (tag_df['Timestamp'] <= frame_start)]
            if trailing.empty:
                # Tag not reporting in this window (e.g. battery dead): hide it.
                trail_line.set_data([], [])
                marker.set_data([], [])
                label.set_text('')
                batt_label.set_text('')
            else:
                xs = trailing['smoothed_x'].to_numpy()
                ys = trailing['smoothed_y'].to_numpy()
                trail_line.set_data(xs, ys)
                cx, cy = xs[-1], ys[-1]
                marker.set_data([cx], [cy])
                if show_battery:
                    # Lift the label and hang the voltage beneath it (shared
                    # anchor at cy+off: label grows up, voltage grows down).
                    off = y_range * 0.035
                    label.set_position((cx, cy + off))
                    label.set_verticalalignment('bottom')
                    label.set_text(tag_info['label'])
                    bv = (trailing['battery_voltage'].iloc[-1]
                          if 'battery_voltage' in trailing.columns else None)
                    batt_label.set_position((cx, cy + off))
                    batt_label.set_text(f"{bv:.2f} V" if bv is not None and pd.notna(bv) else '')
                else:
                    label.set_position((cx, cy + (y_range * 0.02)))
                    label.set_verticalalignment('baseline')
                    label.set_text(tag_info['label'])
                    batt_label.set_text('')
            ax.draw_artist(trail_line)
            ax.draw_artist(marker)
            ax.draw_artist(label)
            ax.draw_artist(batt_label)

        canvas.blit(fig.bbox)
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
        video_writer.write(cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR))

    video_writer.release()
    fig.clear()
    gc.collect()

    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        _log(f"✓ Video compilation complete: {os.path.getsize(output_path):,} bytes")
        return output_path
    _log("✗ Video file was not created or is empty")
    return None
