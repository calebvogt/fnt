"""Home-range and zone-occupancy estimators for UWB tracks.

Pure functions over coordinate arrays: no Qt, no I/O, no plotting. The plot
worker in ``uwb_preprocessing_pyqt`` renders whatever these return.

Two families of home-range estimator are provided because they answer different
questions and disagree in informative ways:

* Kernel density (``kde_utilization`` + ``kde_isopleth``) estimates a
  utilization distribution — how *intensively* space is used — and its
  isopleths enclose a stated share of that usage. The 50% isopleth is the
  conventional "core area"; 95% is the conventional "home range". Because it is
  density-based it can produce several disjoint patches, which is usually the
  honest answer for an animal that shuttles between a nest and a feeder.

* Minimum convex polygon (``mcp``) is purely geometric: the smallest convex
  hull containing the points, with no notion of intensity. It is reported
  because it is the oldest and most comparable metric in the literature, but it
  is sensitive to single excursions and, being convex, necessarily includes
  space the animal never entered.
"""

import numpy as np

__all__ = [
    "kde_utilization", "kde_isopleth", "kde_isopleth_area",
    "mcp", "polygon_area", "assign_zones", "zone_occupancy",
    "zone_intervals", "zone_visits",
]


# ── kernel density utilization distribution ──────────────────────────────────

def kde_utilization(x, y, grid_size=256, bw_method=None, pad_frac=0.15):
    """Gaussian KDE of the fixes, evaluated on a regular grid.

    Returns ``(xg, yg, density, cell_area)`` where ``density`` is
    (grid_size, grid_size) with rows indexed by y, and ``cell_area`` is one
    grid cell in m^2. Returns ``(None, None, None, 0.0)`` when there are too
    few distinct fixes to estimate a density.

    Computed by BINNING then convolving rather than by evaluating a kernel at
    every fix: the fixes are histogrammed onto the grid and the histogram is
    convolved with the Gaussian kernel via FFT. Evaluating directly costs
    O(n_fixes x n_grid_cells), which for a 2.7 M-fix tag on a 256x256 grid is
    ~1.8e11 kernel evaluations - minutes per animal. Binning makes it
    O(n_fixes + n_grid log n_grid), independent of how long the recording is.
    This is the standard approach for large tracking datasets (the same one
    MASS::kde2d and the home-range packages use).

    The estimator is otherwise unchanged: the same full 2x2 bandwidth
    covariance as ``scipy.stats.gaussian_kde`` (data covariance scaled by
    Scott's factor n^(-1/(d+4)), i.e. n^(-1/6) in 2-D), including any
    correlation between x and y, so the kernel is tilted exactly as before.
    The only approximation is that each fix is placed at its grid cell rather
    than its exact coordinate - at 256 cells across a padded arena that is far
    finer than any sensible bandwidth.

    ``bw_method`` accepts a number (a direct multiplier on the covariance, as
    scipy's does) or None for Scott's rule.

    The grid is padded by ``pad_frac`` of the data range on each side so the
    kernel tails are not clipped, which would bias the enclosed-volume
    calculation.
    """
    from scipy.signal import fftconvolve

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    n = x.size
    if n < 10:
        return None, None, None, 0.0
    # A degenerate cloud (all fixes in one spot, or on a perfect line) has a
    # singular covariance and no estimable density.
    if np.ptp(x) <= 0 or np.ptp(y) <= 0:
        return None, None, None, 0.0

    # ---- bandwidth: exactly what gaussian_kde would use --------------------
    factor = float(bw_method) if bw_method is not None else n ** (-1.0 / 6.0)
    data_cov = np.cov(np.vstack([x, y]))
    bw = np.asarray(data_cov, dtype=float) * factor ** 2
    if not np.all(np.isfinite(bw)):
        return None, None, None, 0.0
    det = bw[0, 0] * bw[1, 1] - bw[0, 1] * bw[1, 0]
    if det <= 0:
        # Correlation of exactly +-1 (a perfectly straight track) leaves the
        # kernel singular; drop to the axis-aligned form rather than failing.
        bw = np.diag(np.diag(bw))
        det = bw[0, 0] * bw[1, 1]
        if det <= 0:
            return None, None, None, 0.0

    # ---- grid (cell centres, as before) ------------------------------------
    x_pad = np.ptp(x) * pad_frac
    y_pad = np.ptp(y) * pad_frac
    xg = np.linspace(x.min() - x_pad, x.max() + x_pad, grid_size)
    yg = np.linspace(y.min() - y_pad, y.max() + y_pad, grid_size)
    dx = float(xg[1] - xg[0])
    dy = float(yg[1] - yg[0])
    x_edges = np.append(xg - dx / 2.0, xg[-1] + dx / 2.0)
    y_edges = np.append(yg - dy / 2.0, yg[-1] + dy / 2.0)

    # (ny, nx) to match meshgrid orientation: rows are y, columns are x.
    counts = np.histogram2d(x, y, bins=[x_edges, y_edges])[0].T
    if counts.sum() <= 0:
        return None, None, None, 0.0

    # ---- kernel image on the same cell spacing -----------------------------
    # Out to 4 sigma on each axis; the marginal spreads bound the tilted
    # ellipse, so a correlated kernel is still fully covered.
    rx = max(1, int(np.ceil(4.0 * np.sqrt(bw[0, 0]) / dx)))
    ry = max(1, int(np.ceil(4.0 * np.sqrt(bw[1, 1]) / dy)))
    # A kernel wider than the grid means the bandwidth swamps the data; clamp
    # so the convolution stays cheap and well defined.
    rx = min(rx, grid_size)
    ry = min(ry, grid_size)
    ox = np.arange(-rx, rx + 1) * dx
    oy = np.arange(-ry, ry + 1) * dy
    OX, OY = np.meshgrid(ox, oy)
    inv = np.linalg.inv(bw)
    quad = (inv[0, 0] * OX * OX + (inv[0, 1] + inv[1, 0]) * OX * OY
            + inv[1, 1] * OY * OY)
    kernel = np.exp(-0.5 * quad)
    ksum = kernel.sum()
    if not np.isfinite(ksum) or ksum <= 0:
        return None, None, None, 0.0

    dens = fftconvolve(counts, kernel, mode="same")
    # Negative values of order 1e-18 come out of the FFT; they would sort to
    # the top of the descending isopleth walk as noise.
    np.clip(dens, 0.0, None, out=dens)

    cell_area = float(dx * dy)
    total = dens.sum() * cell_area
    if not np.isfinite(total) or total <= 0:
        return None, None, None, 0.0
    # Normalise numerically so the grid integrates to 1. The isopleths are
    # cumulative-mass cuts, so this is what they are defined against.
    dens /= total
    return xg, yg, dens, cell_area


def kde_isopleth(density, cell_area, percent):
    """Density level whose enclosed volume is ``percent``% of the total.

    The standard construction: sort every grid cell's density from highest to
    lowest, walk down accumulating probability mass (density x cell area), and
    return the density at which the running total first reaches ``percent``%.
    Contouring the grid at that level gives the isopleth enclosing that share
    of the utilization distribution.

    Returns None when the grid holds no usable mass.
    """
    if density is None or cell_area <= 0:
        return None
    flat = np.sort(density.ravel())[::-1]
    total = flat.sum() * cell_area
    if not np.isfinite(total) or total <= 0:
        return None
    cumulative = np.cumsum(flat) * cell_area
    idx = np.searchsorted(cumulative, total * (percent / 100.0))
    idx = min(int(idx), flat.size - 1)
    level = float(flat[idx])
    return level if level > 0 else None


def kde_isopleth_area(density, cell_area, level):
    """Area (m^2) of the region at or above ``level`` — the isopleth's area.

    Summing cells rather than measuring the drawn contour: the two agree to
    within one cell width, and cell-counting also handles isopleths that break
    into several disjoint patches, which contour measurement would not.
    """
    if density is None or level is None or cell_area <= 0:
        return float("nan")
    return float((density >= level).sum() * cell_area)


# ── minimum convex polygon ───────────────────────────────────────────────────

def polygon_area(pts):
    """Area of a simple polygon by the shoelace formula (m^2)."""
    pts = np.asarray(pts, dtype=float)
    if len(pts) < 3:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return float(0.5 * abs(np.dot(x, np.roll(y, 1)) - np.dot(y, np.roll(x, 1))))


def mcp(x, y, percent=100.0):
    """Minimum convex polygon over the fixes.

    Returns ``(vertices, area_m2)`` with vertices as an (N, 2) array in hull
    order, or ``(None, nan)`` if a hull cannot be formed.

    For ``percent`` < 100 the conventional peeling rule is applied: the
    (100 - percent)% of fixes furthest from the arithmetic centroid of all
    fixes are discarded, then the hull is taken over what remains. This is the
    standard way to stop one excursion from doubling the reported range, and it
    is why a 95% MCP is usually far smaller than a 100% MCP. Note the centroid
    is computed once, from the full set, and not recomputed after peeling.
    """
    from scipy.spatial import ConvexHull

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], y[ok]
    if x.size < 3:
        return None, float("nan")

    if percent < 100.0:
        cx, cy = x.mean(), y.mean()
        d = np.hypot(x - cx, y - cy)
        keep = int(np.ceil(x.size * percent / 100.0))
        keep = max(3, min(keep, x.size))
        order = np.argsort(d)[:keep]
        x, y = x[order], y[order]

    pts = np.column_stack([x, y])
    # Collinear or duplicate-only clouds have no 2-D hull.
    if np.ptp(x) <= 0 or np.ptp(y) <= 0:
        return None, float("nan")
    try:
        hull = ConvexHull(pts)
    except Exception:
        return None, float("nan")
    verts = pts[hull.vertices]
    return verts, polygon_area(verts)


# ── arena zones ──────────────────────────────────────────────────────────────

def assign_zones(x, y, zones):
    """Index of the zone containing each fix, or -1 for none.

    ``zones`` is the parsed site-XML list: dicts with ``name`` and ``points``,
    an (N, 2) polygon in metres. Uses matplotlib's even-odd point-in-polygon
    test, so concave zones and holes-by-winding behave as authored.

    Zones are tested in order and the FIRST match wins, so where two zones
    overlap in the XML a fix is attributed to the earlier one rather than being
    double counted.
    """
    from matplotlib.path import Path

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    out = np.full(x.shape, -1, dtype=np.int16)
    if not zones:
        return out
    pts = np.column_stack([x, y])
    finite = np.isfinite(x) & np.isfinite(y)
    for i, z in enumerate(zones):
        poly = np.asarray(z.get("points"), dtype=float)
        if poly is None or len(poly) < 3:
            continue
        unassigned = finite & (out < 0)
        if not unassigned.any():
            break
        inside = Path(poly).contains_points(pts[unassigned])
        idx = np.flatnonzero(unassigned)[inside]
        out[idx] = i
    return out


def zone_intervals(timestamps):
    """Seconds each fix is credited with, one per fix.

    Each fix is credited with the interval to the NEXT fix, so occupancy is
    measured in real time rather than in fix counts — a tag reporting twice as
    often in one zone does not appear to spend twice as long there. The final
    fix contributes the median interval, since its true dwell is unknown.

    A gap in the record is not dwell time, so each interval is capped at ten
    times the typical one: a dropout is not billed to whichever zone the tag
    was last seen in. Returned separately from ``zone_occupancy`` because the
    visit statistics have to weight the same intervals the same way, and two
    copies of this rule would eventually disagree.
    """
    t = np.asarray(timestamps, dtype="datetime64[ns]").astype("int64") / 1e9
    if t.size == 0:
        return np.zeros(0, dtype=float)
    dt = np.diff(t)
    if dt.size == 0:
        return np.zeros(1, dtype=float)
    typical = float(np.median(dt[dt > 0])) if np.any(dt > 0) else 0.0
    cap = max(typical * 10.0, 1.0)
    return np.append(np.clip(dt, 0.0, cap), typical)


def zone_occupancy(zone_idx, timestamps, n_zones):
    """Seconds spent in each zone, from a per-fix zone assignment.

    Returns an array of length ``n_zones + 1``; the last element is time spent
    outside every zone. See ``zone_intervals`` for how a fix is credited.
    """
    zone_idx = np.asarray(zone_idx)
    out = np.zeros(n_zones + 1, dtype=float)
    dt = zone_intervals(timestamps)
    if dt.size == 0:
        return out
    for i in range(n_zones):
        out[i] = dt[zone_idx == i].sum()
    out[n_zones] = dt[zone_idx < 0].sum()
    return out


def zone_visits(zone_idx, timestamps, n_zones):
    """Seconds, visit count and mean visit length per zone.

    A VISIT is a maximal run of consecutive fixes assigned to the same zone —
    the animal enters, stays, and leaves. Counting entries as well as total
    time distinguishes an animal that sat in the nest for six hours from one
    that passed through it sixty times, which a seconds total alone cannot.

    ``zone_idx`` must be in time order. Returns ``(seconds, visits,
    mean_visit_s)``, each of length ``n_zones + 1`` with the last element for
    time outside every zone. ``mean_visit_s`` is 0 where there were no visits.

    ``mean_visit_s`` is the seconds total divided by the visit count, so
    ``visits * mean_visit_s == seconds`` exactly. It is defined that way rather
    than as the first-fix-to-last-fix span of each run because anyone reading
    the two columns together will expect them to reconcile, and because a
    one-fix visit has no span at all yet plainly contributed time.

    A dropout mid-visit does not split a visit — the animal did not leave,
    the tag stopped reporting — but the seconds it spans are still capped
    out of the total, as everywhere else.
    """
    zone_idx = np.asarray(zone_idx)
    seconds = np.zeros(n_zones + 1, dtype=float)
    visits = np.zeros(n_zones + 1, dtype=np.int64)
    mean_s = np.zeros(n_zones + 1, dtype=float)
    dt = zone_intervals(timestamps)
    if dt.size == 0 or zone_idx.size == 0:
        return seconds, visits, mean_s

    # 'outside' is the last slot, so map -1 there and everything else stays put.
    slot = np.where(zone_idx < 0, n_zones, zone_idx).astype(np.int64)
    np.add.at(seconds, slot, dt[:slot.size])

    # Run boundaries: a fix starts a visit when its slot differs from the
    # previous fix's.
    starts = np.flatnonzero(np.r_[True, slot[1:] != slot[:-1]])
    np.add.at(visits, slot[starts], 1)
    nz = visits > 0
    mean_s[nz] = seconds[nz] / visits[nz]
    return seconds, visits, mean_s
