"""User-drawn Regions of Interest for the UWB preprocessing tool.

The Wiser software can author zones into the site XML, but they have to be
drawn against the raw fix stream as it arrives — which is noisy enough that
placing an accurate boundary is guesswork. These ROIs are drawn instead over
the preview, on top of the smoothed track and the site map, where the arena's
real geometry is visible. They live in the trial's ``fnt_config.json`` and are
never derived from, nor written back to, the XML.

Nothing here imports Qt or matplotlib at module level: the model, the
round-trip and the geometry are all testable on their own, and the canvas and
the dialog are the only pieces that need a GUI.

An ROI is deliberately shape-compatible with a parsed XML zone — ``name``,
``color`` and an (N, 2) ``points`` array in metres — so anything that already
consumes zones (the preview canvas, ``spatial_metrics.assign_zones``, the
plot context layers) can take an ROI without a translation step.
"""

import numpy as np

# Distinct, colour-blind-safe-ish defaults, handed out in order so several
# ROIs drawn in one sitting do not all come out the same colour.
ROI_PALETTE = (
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990",
)
DEFAULT_ROI_COLOR = ROI_PALETTE[0]
DEFAULT_ROI_LINEWIDTH = 2.0
MIN_ROI_POINTS = 3          # a polygon needs three corners to enclose anything


def next_roi_color(existing):
    """The first palette colour not already in use, else cycle by count."""
    used = {str(r.get("color", "")).lower() for r in (existing or [])}
    for c in ROI_PALETTE:
        if c.lower() not in used:
            return c
    return ROI_PALETTE[len(existing or []) % len(ROI_PALETTE)]


def unique_roi_name(existing, base="ROI"):
    """A name not already taken: ROI 1, ROI 2, ... skipping any in use."""
    taken = {str(r.get("name", "")).strip().lower() for r in (existing or [])}
    i = 1
    while f"{base} {i}".lower() in taken:
        i += 1
    return f"{base} {i}"


def make_roi(name, points, color=None, linewidth=DEFAULT_ROI_LINEWIDTH):
    """One ROI record, with its polygon as a float (N, 2) array in metres."""
    return {
        "name": str(name),
        "points": np.asarray(points, dtype=float).reshape(-1, 2),
        "color": color or DEFAULT_ROI_COLOR,
        "linewidth": float(linewidth),
    }


def roi_to_json(roi):
    """One ROI as plain JSON types, ready for fnt_config.json."""
    pts = np.asarray(roi.get("points"), dtype=float).reshape(-1, 2)
    return {
        "name": str(roi.get("name", "")),
        # Rounded to 0.1 mm: the arena is metres across and the config is meant
        # to be readable, so seventeen significant figures per corner is noise.
        "points": [[round(float(x), 4), round(float(y), 4)] for x, y in pts],
        "color": str(roi.get("color", DEFAULT_ROI_COLOR)),
        "linewidth": float(roi.get("linewidth", DEFAULT_ROI_LINEWIDTH)),
    }


def rois_to_json(rois):
    """The ROI list as it is stored in the config (``[]`` when there are none)."""
    return [roi_to_json(r) for r in (rois or [])]


def rois_from_json(raw):
    """Rebuild the ROI list from a config, skipping anything unusable.

    Written to be forgiving of a hand-edited config: a malformed entry is
    dropped rather than raising, because losing one ROI is better than
    refusing to open the trial.
    """
    out = []
    for item in (raw or []):
        if not isinstance(item, dict):
            continue
        try:
            pts = np.asarray(item.get("points"), dtype=float).reshape(-1, 2)
        except (TypeError, ValueError):
            continue
        if len(pts) < MIN_ROI_POINTS or not np.isfinite(pts).all():
            continue
        out.append(make_roi(
            item.get("name") or unique_roi_name(out),
            pts,
            item.get("color") or next_roi_color(out),
            item.get("linewidth", DEFAULT_ROI_LINEWIDTH)))
    return out


def polygon_area(points):
    """Absolute area of a simple polygon in m², by the shoelace formula."""
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) < MIN_ROI_POINTS:
        return 0.0
    x, y = pts[:, 0], pts[:, 1]
    return float(abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1))) / 2.0)


def polygon_centroid(points):
    """Centroid of a polygon, for placing its label.

    Falls back to the mean vertex for a degenerate (zero-area) polygon, where
    the area-weighted formula divides by zero.
    """
    pts = np.asarray(points, dtype=float).reshape(-1, 2)
    if len(pts) == 0:
        return (0.0, 0.0)
    if len(pts) < MIN_ROI_POINTS:
        return (float(pts[:, 0].mean()), float(pts[:, 1].mean()))
    x, y = pts[:, 0], pts[:, 1]
    cross = x * np.roll(y, -1) - np.roll(x, -1) * y
    a = cross.sum() / 2.0
    if abs(a) < 1e-12:
        return (float(x.mean()), float(y.mean()))
    cx = ((x + np.roll(x, -1)) * cross).sum() / (6.0 * a)
    cy = ((y + np.roll(y, -1)) * cross).sum() / (6.0 * a)
    return (float(cx), float(cy))


def polygons_overlap(a, b):
    """True if two ROI polygons share any area.

    Uses matplotlib's filled path intersection, which catches BOTH cases that
    matter: edges that cross, and one polygon sitting wholly inside another.
    An edge test alone would miss containment, which is the easier mistake to
    make when copying an ROI and dragging it only a little.

    Touching along an edge without enclosing area is not an overlap: fixes are
    points, and a shared boundary has no area for one to land in.
    """
    from matplotlib.path import Path

    pa = np.asarray(a, dtype=float).reshape(-1, 2)
    pb = np.asarray(b, dtype=float).reshape(-1, 2)
    if len(pa) < MIN_ROI_POINTS or len(pb) < MIN_ROI_POINTS:
        return False
    # Cheap rejection first: disjoint bounding boxes cannot overlap, and this
    # is the common case when several ROIs are laid out side by side.
    if (pa[:, 0].max() <= pb[:, 0].min() or pb[:, 0].max() <= pa[:, 0].min()
            or pa[:, 1].max() <= pb[:, 1].min() or pb[:, 1].max() <= pa[:, 1].min()):
        return False
    return bool(Path(pa).intersects_path(Path(pb), filled=True))


def overlapping_pairs(rois, only=None):
    """Index pairs (i, j), i < j, whose ROIs share area.

    ``only`` restricts the search to pairs involving that index - what the
    canvas wants after a single ROI is drawn or dragged, rather than re-testing
    every pair each time.
    """
    rois = list(rois or [])
    out = []
    for i in range(len(rois)):
        for j in range(i + 1, len(rois)):
            if only is not None and only not in (i, j):
                continue
            if polygons_overlap(rois[i].get("points"), rois[j].get("points")):
                out.append((i, j))
    return out


def set_precedence(rois, winner, loser):
    """Reorder so ``winner`` is tested before ``loser``, returning a new list.

    Precedence IS list order: assign_zones takes the first region containing a
    fix, so moving one ahead of another is the whole mechanism. Nothing else
    about the list changes - the other ROIs keep their relative order, so
    resolving one overlap cannot silently reshuffle an earlier decision.
    """
    rois = list(rois or [])
    if not (0 <= winner < len(rois)) or not (0 <= loser < len(rois)):
        return rois
    if winner < loser:
        return rois
    item = rois.pop(winner)
    rois.insert(loser, item)
    return rois


def points_in_roi(x, y, roi):
    """Boolean mask of which (x, y) fall inside one ROI.

    Uses matplotlib's even-odd test, the same one ``spatial_metrics`` uses for
    XML zones, so a fix cannot be judged inside by one code path and outside
    by the other.
    """
    from matplotlib.path import Path

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    poly = np.asarray(roi.get("points"), dtype=float).reshape(-1, 2)
    out = np.zeros(x.shape, dtype=bool)
    if len(poly) < MIN_ROI_POINTS:
        return out
    finite = np.isfinite(x) & np.isfinite(y)
    if not finite.any():
        return out
    out[finite] = Path(poly).contains_points(
        np.column_stack([x[finite], y[finite]]))
    return out


def nearest_vertex(rois, x, y, tol):
    """(roi index, vertex index) of the corner nearest (x, y) within ``tol``.

    Returns (None, None) when nothing is close enough. ``tol`` is in metres and
    the caller sets it from the current zoom, so the grab radius stays a
    constant size on screen however far in the user has zoomed.
    """
    best = (None, None)
    best_d = float(tol)
    for i, r in enumerate(rois or []):
        pts = np.asarray(r.get("points"), dtype=float).reshape(-1, 2)
        if not len(pts):
            continue
        d = np.hypot(pts[:, 0] - x, pts[:, 1] - y)
        j = int(np.argmin(d))
        if d[j] <= best_d:
            best_d = float(d[j])
            best = (i, j)
    return best


def describe_roi(roi):
    """One-line summary for a list row: name, corner count and area."""
    pts = np.asarray(roi.get("points"), dtype=float).reshape(-1, 2)
    return (f"{roi.get('name', '')}  "
            f"({len(pts)} pts, {polygon_area(pts):.2f} m²)")
