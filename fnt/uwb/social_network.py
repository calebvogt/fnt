"""
Social-network raw materials from UWB proximity data.

Turns the pairwise proximity output (see ``proximity_detection.py``) into the
same analysis products used in the LID_2020 RFID study, so the same downstream
R/asnipe/igraph workflows apply:

  * edge lists  — pairwise co-occurrence, event-based (bout count) and
    time-based (total overlap seconds), full-recording and per-day.
  * GBI matrix  — asnipe-style group-by-individual matrix where each row is a
    "flocking event": a maximal run during which a fixed set of animals formed
    one connected proximity cluster (chain rule). Carries m_sum/f_sum/mf_sum.
  * sliding-window networks — per-window weighted edges for the dynamic social
    network animation (see ``render_network_animation``).

All grouping uses the *chain rule*: at each sampled instant, animals within the
proximity threshold are linked and each connected component of size >= 2 is a
group (A-B and B-C put A, B, C in one flock even if A and C are far apart).
"""

import os

import numpy as np
import pandas as pd

# Sex -> node colour (shared by the animation).
SEX_COLORS = {'M': '#4477AA', 'F': '#EE6677', 'U': '#999999'}


# ── labels / sex ─────────────────────────────────────────────────────────────
def _label_sex_map(tag_identities):
    """label ('F9040') -> sex ('F'), matching proximity_detection's label scheme."""
    m = {}
    if tag_identities:
        for tag, info in tag_identities.items():
            sex = str(info.get('sex', 'M') or 'M')[0].upper()
            ident = info.get('identity', str(tag))
            m[f"{sex}{ident}"] = sex
    return m


def _sex_of(label, sexmap):
    if label in sexmap:
        return sexmap[label]
    return label[0] if label and label[0] in ('M', 'F') else 'U'


def _dyad_type(s1, s2):
    if s1 == s2:
        return s1 + s2          # MM / FF
    if 'U' in (s1, s2):
        return 'U?'
    return 'MF'                 # mixed


# ── edge lists ───────────────────────────────────────────────────────────────
def build_edgelists(proximity_bouts, tag_identities=None):
    """Aggregate pairwise proximity bouts into (edgelist_full, edgelist_daily).

    edgelist_full columns:
        animal1, animal2, sex1, sex2, dyad_type, n_events, total_duration_s,
        mean_distance, n_days
    edgelist_daily columns: same, plus Day (no n_days).
    ``n_events`` is the event-based weight; ``total_duration_s`` the time-based.
    """
    cols_full = ['animal1', 'animal2', 'sex1', 'sex2', 'dyad_type',
                 'n_events', 'total_duration_s', 'mean_distance', 'n_days']
    cols_daily = ['animal1', 'animal2', 'Day', 'sex1', 'sex2', 'dyad_type',
                  'n_events', 'total_duration_s', 'mean_distance']
    if proximity_bouts is None or proximity_bouts.empty:
        return pd.DataFrame(columns=cols_full), pd.DataFrame(columns=cols_daily)

    sexmap = _label_sex_map(tag_identities)
    b = proximity_bouts.copy()

    def _add_sex(df):
        df['sex1'] = df['animal1'].map(lambda a: _sex_of(a, sexmap))
        df['sex2'] = df['animal2'].map(lambda a: _sex_of(a, sexmap))
        df['dyad_type'] = [_dyad_type(a, c) for a, c in zip(df['sex1'], df['sex2'])]
        return df

    full = (b.groupby(['animal1', 'animal2'], sort=True)
              .agg(n_events=('duration_s', 'size'),
                   total_duration_s=('duration_s', 'sum'),
                   mean_distance=('mean_distance', 'mean'),
                   n_days=('Day', 'nunique'))
              .reset_index())
    full = _add_sex(full)[cols_full].sort_values(
        ['total_duration_s'], ascending=False).reset_index(drop=True)

    daily = (b.groupby(['animal1', 'animal2', 'Day'], sort=True)
               .agg(n_events=('duration_s', 'size'),
                    total_duration_s=('duration_s', 'sum'),
                    mean_distance=('mean_distance', 'mean'))
               .reset_index())
    daily = _add_sex(daily)[cols_daily].sort_values(
        ['Day', 'total_duration_s'], ascending=[True, False]).reset_index(drop=True)

    return full, daily


# ── chain-rule grouping (GBI) ────────────────────────────────────────────────
def _components_at(a1, a2):
    """Connected components (union-find) over the edges (a1[i], a2[i]).

    Returns a list of sets, one per component of the animals that appear.
    """
    parent = {}

    def find(x):
        parent.setdefault(x, x)
        root = x
        while parent[root] != root:
            root = parent[root]
        while parent[x] != root:      # path compression
            parent[x], x = root, parent[x]
        return root

    def union(x, y):
        rx, ry = find(x), find(y)
        if rx != ry:
            parent[rx] = ry

    for x, y in zip(a1, a2):
        union(x, y)
    comps = {}
    for node in list(parent):
        comps.setdefault(find(node), set()).add(node)
    return list(comps.values())


def build_gbi(events, tag_identities=None, gap_s=5, min_group=2):
    """Group-by-individual matrix of chain-rule proximity flocks.

    ``events`` is the per-second pairwise frame from detect_proximity_bouts
    (timestamp, animal1, animal2, distance, in_proximity, Day, ...). Each output
    row is a flocking event: a maximal contiguous run (gaps <= ``gap_s`` s) in
    which a fixed set of >= ``min_group`` animals formed one connected cluster.

    Columns: day, group_start, group_stop, duration_s, m_sum, f_sum, mf_sum,
    then one 0/1 column per animal (all animals seen in ``events``). Loads
    directly into asnipe::get_network(data_format="GBI") after dropping the
    leading metadata columns.
    """
    sexmap = _label_sex_map(tag_identities)
    meta_cols = ['day', 'group_start', 'group_stop', 'duration_s',
                 'm_sum', 'f_sum', 'mf_sum']
    if events is None or events.empty:
        return pd.DataFrame(columns=meta_cols)

    all_animals = sorted(set(events['animal1']) | set(events['animal2']))
    prox = events.loc[events['in_proximity'],
                      ['timestamp', 'animal1', 'animal2', 'Day']]
    if prox.empty:
        return pd.DataFrame(columns=meta_cols + all_animals)

    # Per timestamp: connected components -> one record per group (>= min_group).
    recs = []   # (timestamp, day, key, members-tuple)
    for ts, g in prox.groupby('timestamp', sort=True):
        day = g['Day'].iloc[0]
        for members in _components_at(g['animal1'].to_numpy(),
                                      g['animal2'].to_numpy()):
            if len(members) >= min_group:
                mt = tuple(sorted(members))
                recs.append((ts, day, '|'.join(mt), mt))
    if not recs:
        return pd.DataFrame(columns=meta_cols + all_animals)

    r = pd.DataFrame(recs, columns=['ts', 'day', 'key', 'members'])
    r = r.sort_values(['key', 'ts']).reset_index(drop=True)

    # Run-length encode each group signature into contiguous events.
    prev = r.groupby('key')['ts'].shift()
    gap = (r['ts'] - prev).dt.total_seconds()
    new_seg = gap.isna() | (gap > gap_s)
    r['seg'] = new_seg.groupby(r['key']).cumsum()

    ev = (r.groupby(['key', 'seg'], sort=False)
            .agg(group_start=('ts', 'min'), group_stop=('ts', 'max'),
                 day=('day', 'first'), members=('members', 'first'))
            .reset_index())
    ev['duration_s'] = ((ev['group_stop'] - ev['group_start'])
                        .dt.total_seconds().clip(lower=1))

    # 0/1 membership matrix + sex sums.
    mat = np.zeros((len(ev), len(all_animals)), dtype=int)
    a_index = {a: i for i, a in enumerate(all_animals)}
    m_sum = np.zeros(len(ev), int)
    f_sum = np.zeros(len(ev), int)
    for row_i, members in enumerate(ev['members']):
        for a in members:
            mat[row_i, a_index[a]] = 1
            s = _sex_of(a, sexmap)
            if s == 'M':
                m_sum[row_i] += 1
            elif s == 'F':
                f_sum[row_i] += 1

    out = pd.DataFrame(mat, columns=all_animals)
    out.insert(0, 'mf_sum', [len(m) for m in ev['members']])
    out.insert(0, 'f_sum', f_sum)
    out.insert(0, 'm_sum', m_sum)
    out.insert(0, 'duration_s', ev['duration_s'].to_numpy())
    out.insert(0, 'group_stop', ev['group_stop'].to_numpy())
    out.insert(0, 'group_start', ev['group_start'].to_numpy())
    out.insert(0, 'day', ev['day'].to_numpy())
    return out.sort_values('group_start').reset_index(drop=True)


# ── sliding-window dynamic network animation ─────────────────────────────────
def _ensure_dt(s):
    """Parse to UTC then drop tz, so numpy datetime64 comparisons are clean."""
    out = pd.to_datetime(s, utc=True, errors='coerce')
    try:
        return out.dt.tz_localize(None)
    except (AttributeError, TypeError):
        return out


def _naive(ts):
    ts = pd.Timestamp(ts)
    return ts.tz_localize(None) if ts.tz is not None else ts


def _window_weights(a1, a2, starts, stops, w0, w1, weighting):
    """Per-pair weight for the window [w0, w1].

    ``weighting='events'`` counts bouts overlapping the window; ``'time'`` sums
    the overlapping seconds. Returns {(animal1, animal2): weight}.
    """
    ov0 = np.maximum(starts, np.datetime64(w0))
    ov1 = np.minimum(stops, np.datetime64(w1))
    dur = (ov1 - ov0) / np.timedelta64(1, 's')
    mask = dur > 0
    if not mask.any():
        return {}
    df = pd.DataFrame({'a1': a1[mask], 'a2': a2[mask], 'dur': dur[mask]})
    if weighting == 'events':
        g = df.groupby(['a1', 'a2']).size()
    else:
        g = df.groupby(['a1', 'a2'])['dur'].sum()
    return {(i, j): float(v) for (i, j), v in g.items()}


# Node-size metrics offered in the UI -> igraph computation. All are the
# igraph metrics the R workflow used, computed on each window's weighted graph.
NODE_METRICS = ('none', 'degree', 'strength', 'betweenness',
                'closeness', 'eigenvector', 'pagerank')
EDGE_SCALES = ('weight', 'uniform')


# Networks are built with python-igraph (for consistency with the LID_2020
# R/asnipe/igraph workflow); rendering into the video stays on matplotlib.
def _ig_graph(ig, animals, wd):
    """igraph Graph over ``animals`` (fixed vertex order) with weighted edges
    from ``wd`` = {(a, b): weight}."""
    idx = {a: i for i, a in enumerate(animals)}
    g = ig.Graph()
    g.add_vertices(len(animals))
    if wd:
        g.add_edges([(idx[a], idx[b]) for (a, b) in wd])
        g.es['weight'] = [float(w) for w in wd.values()]
    return g


def _ig_metric(g, metric):
    """Per-vertex centrality (list, vertex order) on the weighted igraph graph.

    Tie weight = strength: strength/eigenvector/pagerank use it directly;
    betweenness/closeness use 1/weight as a distance (stronger = closer).
    """
    n = g.vcount()
    if metric in ('none', None) or g.ecount() == 0:
        return [0.0] * n
    has_w = 'weight' in g.es.attributes()
    w = g.es['weight'] if has_w else None
    if metric == 'degree':
        return g.degree()
    if metric == 'strength':
        return g.strength(weights=w)
    if metric in ('betweenness', 'closeness'):
        dist = [1.0 / max(x, 1e-9) for x in w] if has_w else None
        if metric == 'betweenness':
            return g.betweenness(weights=dist)
        vals = g.closeness(weights=dist)
        return [0.0 if (v is None or v != v) else v for v in vals]
    if metric == 'eigenvector':
        try:
            return g.eigenvector_centrality(weights=w)
        except Exception:
            return g.strength(weights=w)
    if metric == 'pagerank':
        return g.pagerank(weights=w)
    return [0.0] * n


def _rescale_list(vals, lo, hi):
    """Rescale a list to [lo, hi]; constant/empty -> midpoint (like scales::rescale)."""
    if not vals:
        return []
    vmin, vmax = min(vals), max(vals)
    if vmax - vmin < 1e-12:
        return [(lo + hi) / 2.0] * len(vals)
    return [lo + (hi - lo) * (v - vmin) / (vmax - vmin) for v in vals]


def _rescale(values, lo, hi):
    """Rescale an edge-weight dict to [lo, hi] per frame."""
    if not values:
        return {}
    vmin, vmax = min(values.values()), max(values.values())
    if vmax - vmin < 1e-12:
        return {k: (lo + hi) / 2.0 for k in values}
    return {k: lo + (hi - lo) * (v - vmin) / (vmax - vmin)
            for k, v in values.items()}


def _normalize_coords(coords):
    """Center + uniform-scale coordinates to ~[-1, 1] (preserves aspect ratio)."""
    a = np.asarray(coords, float)
    if len(a) == 0:
        return a
    mn, mx = a.min(0), a.max(0)
    half = max((mx - mn).max() / 2.0, 1e-9)
    return (a - (mn + mx) / 2.0) / half


def _ig_layout(ig, g, layout, seed=None):
    """One layout pass. When ``seed`` is given, evolve from it with a small
    temperature (niter=10, start_temp=0.05) so nodes DRIFT between frames rather
    than being pinned — replicating the R layout_with_fr(coords=old, ...) loop."""
    try:
        if layout == 'circular':
            return [list(p) for p in g.layout_circle()]
        if layout == 'kamada_kawai':
            return [list(p) for p in g.layout_kamada_kawai()]
        # Fruchterman-Reingold, UNWEIGHTED (as in the R layout_with_fr call) so
        # strongly-weighted ties don't yank nodes on top of each other — edge
        # weight still drives edge width and node metrics, not the geometry.
        if seed is not None:
            lay = g.layout_fruchterman_reingold(niter=10, start_temp=0.05, seed=seed)
        else:
            lay = g.layout_fruchterman_reingold(niter=500)
        return [list(p) for p in lay]
    except Exception:
        return seed if seed is not None else [list(p) for p in g.layout_circle()]


# Network edge-weight definitions (the "Network construction" choice) and the
# graph layouts offered in the UI.
WEIGHTINGS = ('time', 'events', 'sri', 'hwi')
WEIGHT_UNITS = {'time': 'seconds', 'events': 'bouts', 'sri': 'SRI', 'hwi': 'HWI'}
NETWORK_LAYOUTS = ('kamada_kawai', 'fruchterman_reingold', 'circular')


def _assoc_window(gbi_mat, gstart, animals, w0, w1, index):
    """Association-index edge weights over flocking events starting in [w0, w1).

    Treats each GBI flocking event as one gambit-of-the-group observation, then
    for every co-occurring dyad computes, from group-membership counts:
        x  = events with both A and B      ci/cj = events with A / with B
        SRI = x / (ci + cj - x)            (Jaccard-like; simple ratio index)
        HWI = 2x / (ci + cj)               (Dice-like; half-weight index)
    Returns {(a, b): weight} for pairs with x > 0.
    """
    mask = (gstart >= np.datetime64(w0)) & (gstart < np.datetime64(w1))
    if not mask.any():
        return {}
    M = gbi_mat[mask]                       # rows=events, cols=animals (0/1)
    counts = M.sum(0)
    co = M.T @ M                            # co-occurrence counts
    out = {}
    n = len(animals)
    for i in range(n):
        ci = counts[i]
        if ci == 0:
            continue
        for j in range(i + 1, n):
            x = co[i, j]
            if x <= 0:
                continue
            cj = counts[j]
            out[(animals[i], animals[j])] = (2.0 * x / (ci + cj) if index == 'hwi'
                                             else x / (ci + cj - x))
    return out


def _draw_network_frame(ax, pos, animals, node_colors, sizes, labels, wd,
                        edge_scale, title):
    """Draw one network frame onto ``ax``. ``pos`` = {animal: (x, y)} in the
    normalised [-1, 1] box; ``sizes`` = per-node marker areas."""
    ax.clear()
    ax.set_xlim(-1.2, 1.2)
    ax.set_ylim(-1.2, 1.2)
    ax.set_axis_off()
    if wd:
        ew = ({k: 2.5 for k in wd} if edge_scale == 'uniform'
              else _rescale(wd, 0.5, 12.0))
        for (i, j), w in wd.items():
            (xi, yi), (xj, yj) = pos[i], pos[j]
            ax.plot([xi, xj], [yi, yj], color='#666666', linewidth=ew[(i, j)],
                    alpha=0.6, zorder=1)
    xs = [pos[a][0] for a in animals]
    ys = [pos[a][1] for a in animals]
    ax.scatter(xs, ys, s=sizes, c=node_colors, edgecolors='#222222',
               linewidths=0.8, zorder=2)
    for a in animals:
        ax.annotate(labels[a], pos[a], textcoords='offset points', xytext=(0, 9),
                    fontsize=7, ha='center', va='bottom', color='#111111',
                    zorder=3, fontweight='bold')
    if title:
        ax.set_title(title, fontsize=12, fontweight='bold')


def _frame_positions_and_sizes(ig, animals, wd, node_metric, pos_prev, layout):
    """Evolve the layout one step (dynamic drift) and compute node sizes.
    Returns (pos_dict, raw_coords_for_next_seed, sizes_list)."""
    g = _ig_graph(ig, animals, wd)
    if layout == 'fruchterman_reingold':
        # Dynamic: evolve from the previous frame (nodes drift). When no seed
        # (popup single frame) this does a full FR pass instead.
        coords = _ig_layout(ig, g, layout, seed=pos_prev)
    elif pos_prev is not None:
        coords = pos_prev            # Kamada-Kawai / circular: fixed layout
    else:
        coords = _ig_layout(ig, g, layout, seed=None)   # popup: compute once
    norm = _normalize_coords(coords)
    pos = {animals[i]: (float(norm[i][0]), float(norm[i][1]))
           for i in range(len(animals))}
    if node_metric in ('none', None):
        sizes = [180] * len(animals)
    else:
        sizes = _rescale_list(_ig_metric(g, node_metric), 40.0, 900.0)
    return pos, coords, sizes


def render_network_animation(bouts, output_path, *, animals, sexes,
                             weighting='time', window_hours=24, step_hours=1,
                             node_metric='strength', edge_scale='weight',
                             layout='kamada_kawai', gbi=None,
                             fps=10, dpi=100, span=None, title_prefix='',
                             is_cancelled=None, progress=None, log=None):
    """Render a sliding-window dynamic social-network video to ``output_path``.

    Built with python-igraph. The layout EVOLVES each frame — a
    Fruchterman-Reingold pass seeded from the previous frame's positions with a
    small temperature — so nodes drift and move closer/farther as their ties
    change (matching the LID_2020 R animation), rather than sitting fixed. Node
    colour = sex; ``node_metric`` sets node size (rescaled per frame);
    ``edge_scale`` sets edge width. One frame per ``step_hours`` over the
    trailing ``window_hours``. Returns output_path on success.
    """
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import igraph as ig
    import cv2

    def _log(m):
        if log:
            log(m)

    b = bouts.copy()
    if b.empty or len(animals) < 2:
        _log("Network animation: not enough data")
        return None
    b['bout_start'] = _ensure_dt(b['bout_start'])
    b['bout_stop'] = _ensure_dt(b['bout_stop'])
    b = b.dropna(subset=['bout_start', 'bout_stop'])

    a1 = b['animal1'].to_numpy()
    a2 = b['animal2'].to_numpy()
    starts = b['bout_start'].to_numpy()
    stops = b['bout_stop'].to_numpy()

    t_start = _naive(span[0]) if span else b['bout_start'].min()
    t_end = _naive(span[1]) if span else b['bout_stop'].max()
    win = pd.Timedelta(hours=window_hours)
    step = pd.Timedelta(hours=step_hours)

    # SRI/HWI edges come from the GBI flocking events; time/events from bouts.
    gbi_mat = gstart = None
    gbi_animals = animals
    if weighting in ('sri', 'hwi'):
        if gbi is None or len(gbi) == 0:
            _log(f"Network animation: {weighting.upper()} needs the GBI — none available")
            return None
        gbi_animals = [a for a in animals if a in gbi.columns]
        gbi_mat = gbi[gbi_animals].to_numpy()
        gstart = (pd.to_datetime(gbi['group_start'], utc=True, errors='coerce')
                  .dt.tz_localize(None).to_numpy())

    def edges_for(w0, w1):
        if weighting in ('sri', 'hwi'):
            return _assoc_window(gbi_mat, gstart, gbi_animals, w0, w1, weighting)
        return _window_weights(a1, a2, starts, stops, w0, w1, weighting)

    # Frame end-times: step across the WHOLE span so short spans (e.g. a single
    # day, whose data is shorter than a 24 h window) still yield many frames —
    # not one. Each frame shows the trailing ``win`` hours, which is naturally
    # partial/growing near the start of the span.
    ends = []
    t = t_start + step
    while t < t_end:
        ends.append(t)
        t += step
    ends.append(t_end)
    ends = sorted(set(ends))

    # Initial layout from the aggregate network — the seed the per-frame layout
    # evolves from.
    agg = _window_weights(a1, a2, starts, stops, t_start, t_end, 'time')
    agg = {(i, j): w for (i, j), w in agg.items() if i in animals and j in animals}
    pos_prev = _ig_layout(ig, _ig_graph(ig, animals, agg), layout)

    # Per-window edges (edge width / node size / layout evolve per frame).
    frames = []
    for te in ends:
        wd = edges_for(te - win, te)
        wd = {(i, j): w for (i, j), w in wd.items() if i in animals and j in animals}
        frames.append((te, wd))

    fig = Figure(figsize=(10, 8), dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    node_colors = [SEX_COLORS.get(sexes.get(a, 'U'), '#999999') for a in animals]
    labels = {a: a for a in animals}   # full SexID label (e.g. 'M9627')

    canvas.draw()
    width, height = canvas.get_width_height()
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    vw = cv2.VideoWriter(output_path, fourcc, fps, (width, height))
    if not vw.isOpened():
        _log("✗ Could not open VideoWriter for network animation")
        return None

    unit = WEIGHT_UNITS.get(weighting, weighting)
    metric_label = 'uniform' if node_metric in ('none', None) else node_metric
    total = len(frames)
    for fi, (te, wd) in enumerate(frames):
        if is_cancelled and is_cancelled():
            vw.release()
            return None
        if progress:
            progress(fi, total)
        pos, pos_prev, sizes = _frame_positions_and_sizes(
            ig, animals, wd, node_metric, pos_prev, layout)
        day = int((te - t_start).total_seconds() // 86400) + 1
        title = (f"{title_prefix}Social network · edge {unit} · node {metric_label}"
                 f" · {window_hours:g} h window\n"
                 f"through {te:%Y-%m-%d %H:%M}  ·  day {day}")
        _draw_network_frame(ax, pos, animals, node_colors, sizes, labels, wd,
                            edge_scale, title)
        canvas.draw()
        buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
        buf = buf.reshape(canvas.get_width_height()[::-1] + (4,))
        vw.write(cv2.cvtColor(buf, cv2.COLOR_RGBA2BGR))

    vw.release()
    if os.path.exists(output_path) and os.path.getsize(output_path) > 0:
        _log(f"✓ Network animation: {os.path.basename(output_path)} "
             f"({total} frames, {os.path.getsize(output_path):,} bytes)")
        return output_path
    return None


def render_network_frame(bouts, *, animals, sexes, weighting='time',
                         window_hours=24, node_metric='none', edge_scale='weight',
                         layout='kamada_kawai', gbi=None, window_end=None,
                         title_prefix=''):
    """Render ONE network frame for the Preview Layout popup.

    Returns (rgb_uint8_array, window_end_timestamp). ``window_end=None`` picks a
    random window across the recording so the user can sample what the network
    looks like at different times.
    """
    import matplotlib
    matplotlib.use('Agg')
    from matplotlib.figure import Figure
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    import igraph as ig
    import random

    b = bouts.copy()
    if b.empty or len(animals) < 2:
        return None, None
    b['bout_start'] = _ensure_dt(b['bout_start'])
    b['bout_stop'] = _ensure_dt(b['bout_stop'])
    b = b.dropna(subset=['bout_start', 'bout_stop'])
    a1 = b['animal1'].to_numpy(); a2 = b['animal2'].to_numpy()
    starts = b['bout_start'].to_numpy(); stops = b['bout_stop'].to_numpy()
    t_start, t_end = b['bout_start'].min(), b['bout_stop'].max()
    win = pd.Timedelta(hours=window_hours)

    gbi_mat = gstart = None
    gbi_animals = animals
    if weighting in ('sri', 'hwi') and gbi is not None and len(gbi):
        gbi_animals = [a for a in animals if a in gbi.columns]
        gbi_mat = gbi[gbi_animals].to_numpy()
        gstart = (pd.to_datetime(gbi['group_start'], utc=True, errors='coerce')
                  .dt.tz_localize(None).to_numpy())

    def edges_for(w0, w1):
        if weighting in ('sri', 'hwi') and gbi_mat is not None:
            return _assoc_window(gbi_mat, gstart, gbi_animals, w0, w1, weighting)
        return _window_weights(a1, a2, starts, stops, w0, w1, weighting)

    lo = min(t_start + win, t_end)
    if window_end is None:
        span_s = max((t_end - lo).total_seconds(), 0.0)
        te = lo + pd.Timedelta(seconds=random.uniform(0, span_s)) if span_s > 0 else t_end
    else:
        te = _naive(window_end)
    wd = edges_for(te - win, te)
    wd = {(i, j): w for (i, j), w in wd.items() if i in animals and j in animals}

    fig = Figure(figsize=(8, 7), dpi=100)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_subplot(111)
    node_colors = [SEX_COLORS.get(sexes.get(a, 'U'), '#999999') for a in animals]
    labels = {a: a for a in animals}   # full SexID label (e.g. 'M9627')
    unit = WEIGHT_UNITS.get(weighting, weighting)
    ml = 'uniform' if node_metric in ('none', None) else node_metric
    day = int((te - t_start).total_seconds() // 86400) + 1
    title = (f"{title_prefix}edge {unit} · node {ml} · {window_hours:g} h window\n"
             f"through {te:%Y-%m-%d %H:%M} · day {day}")
    # Single frame: static layout of this window's own graph.
    pos, _seed, sizes = _frame_positions_and_sizes(
        ig, animals, wd, node_metric, None, layout)
    _draw_network_frame(ax, pos, animals, node_colors, sizes, labels, wd,
                        edge_scale, title)
    canvas.draw()
    buf = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8).reshape(
        canvas.get_width_height()[::-1] + (4,))
    return buf[:, :, :3].copy(), te
