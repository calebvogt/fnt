"""Cumulative series behind the animation's Data View panel.

The panel exists to be CHECKED: you watch a number climb, scrub to the end,
and compare it with the exported summary CSV. That only works if the panel and
the CSV are the same arithmetic, so nothing here counts rendered frames.

A frame-driven counter cannot be made to agree. At 60x and 30 fps one frame
spans two seconds of trial time, so occupancy would quantise to two seconds
while the CSV measures every fix at ~7.6 Hz. Instead the same intervals the
CSV sums are turned into a cumulative curve once, and each frame simply reads
that curve at its own timestamp. The last frame then equals the CSV total
because it is the same number, not because the two happened to converge.

The curve is piecewise linear with knots at bout boundaries:

  * EXACT at the end of every bout, and therefore at the end of the trial.
  * INTERPOLATED inside a bout, so a cell ticks smoothly upward while an
    animal sits in a region rather than jumping when it leaves.

Storing knots per bout rather than per fix is what keeps this affordable. A
19-day trial holds ~25 M fixes; keeping two float64 arrays per (animal,
region) at fix resolution would cost hundreds of megabytes, which is the
territory that has already produced out-of-memory crashes here. Bouts are
orders of magnitude fewer.

No Qt and no matplotlib: the arithmetic is testable on its own, and the panel
is the only piece that needs a canvas.
"""

import numpy as np

# Below this, two bout boundaries are the same instant and the knot pair would
# be a vertical step. np.interp copes, but collapsing them keeps the arrays
# honest about how many distinct times they describe.
_TIME_EPS = 1e-9


def human_duration(seconds, width=5):
    """An accumulated duration at the precision a person reads mid-animation.

    A UWB trial runs for weeks, so one fixed unit cannot serve: seconds are
    unreadable at day scale and days are meaningless in the first minute. The
    unit therefore follows the value, which is what a Data View cell needs -
    it starts at 0.00s and walks up through m, h, d as overlap accumulates.

    Kept to ``width`` characters so a grid of them stays a grid. This is a
    DISPLAY form and it is lossy: the totals card reports the same numbers in
    the CSV's own units, which is what an exact check should be made against.
    """
    s = float(seconds or 0.0)
    if not np.isfinite(s):
        return "-".rjust(width)
    neg = "-" if s < 0 else ""
    s = abs(s)
    # One decimal in every unit. Two made the grid wider than it needed to be
    # for precision nobody reads off a moving video - the totals card and the
    # CSV are where exact figures come from. 59.9s is the five characters the
    # grid is built around.
    if s < 60.0:
        out = f"{neg}{s:.1f}s"
    elif s < 3600.0:
        out = f"{neg}{s / 60.0:.1f}m"
    elif s < 86400.0:
        out = f"{neg}{s / 3600.0:.1f}h"
    else:
        out = f"{neg}{s / 86400.0:.1f}d"
    # A value wide enough to break the column loses a decimal rather than the
    # magnitude: knowing it is 12.4h matters more than knowing it is 12.42h.
    if len(out) > width:
        unit, body = out[-1], out[:-1]
        try:
            out = f"{float(body):.1f}{unit}"
        except ValueError:
            pass
        if len(out) > width:
            out = f"{float(body):.0f}{unit}"
    return out.rjust(width)


def bouts_from_labels(labels, times, seconds, n_labels):
    """Group consecutively-labelled fixes into bouts, per label.

    ``labels`` is the per-fix region index (-1 for outside, as assign_zones
    returns), ``times`` the fix times as float epoch seconds, and ``seconds``
    the interval each fix is credited with (spatial_metrics.zone_intervals).

    Returns {label_index: [(t0, t1, secs), ...]}. A bout's ``secs`` is the SUM
    of its fixes' credited intervals, so summing a label's bouts reproduces the
    occupancy total exactly - the CSV's number, by the same route.
    """
    labels = np.asarray(labels)
    times = np.asarray(times, dtype=float)
    secs = np.asarray(seconds, dtype=float)
    out = {i: [] for i in range(int(n_labels))}
    if labels.size == 0:
        return out

    # Boundaries wherever the label changes; one pass, no Python loop per fix.
    change = np.flatnonzero(labels[1:] != labels[:-1]) + 1
    starts = np.concatenate([[0], change])
    ends = np.concatenate([change, [labels.size]])       # exclusive
    csum = np.concatenate([[0.0], np.cumsum(secs)])

    for s, e in zip(starts, ends):
        lab = int(labels[s])
        if lab < 0 or lab >= n_labels:
            continue                                      # outside every region
        total = float(csum[e] - csum[s])
        t0 = float(times[s])
        # The run's last fix is credited with an interval that extends BEYOND
        # its timestamp, so the bout ends where that credit ends. Using the
        # last timestamp instead would leave the curve short by one interval
        # and the final total would not match the CSV.
        t1 = float(times[e - 1]) + float(secs[e - 1])
        if t1 < t0:
            t1 = t0
        out[lab].append((t0, t1, total))
    return out


class CumulativeSeries:
    """A cumulative-seconds curve, read at any instant.

    Built once per measured thing (an animal in a region, or a pair
    overlapping) and then sampled per frame.
    """

    __slots__ = ("t", "c", "total")

    def __init__(self, t, c):
        self.t = np.asarray(t, dtype=float)
        self.c = np.asarray(c, dtype=float)
        self.total = float(self.c[-1]) if self.c.size else 0.0

    @classmethod
    def from_bouts(cls, bouts):
        """Build from [(t0, t1, seconds), ...]; unsorted input is fine."""
        bouts = sorted((b for b in (bouts or [])), key=lambda b: b[0])
        if not bouts:
            return cls([0.0], [0.0])
        t = [bouts[0][0]]
        c = [0.0]
        run = 0.0
        for t0, t1, secs in bouts:
            # Flat from the previous bout's end up to this one's start. Without
            # this knot the curve would ramp across the gap and report time in
            # a region the animal had already left.
            if t0 - t[-1] > _TIME_EPS:
                t.append(t0)
                c.append(run)
            run += float(secs)
            t.append(max(float(t1), t[-1]))
            c.append(run)
        return cls(t, c)

    def at(self, when):
        """Cumulative seconds at ``when`` (scalar or array of epoch seconds).

        Flat before the first bout and after the last, which is what np.interp
        does at the ends - so a frame before the animal was tracked reads 0.00
        and one after its last fix holds the final total.
        """
        if self.t.size == 0:
            return np.zeros_like(np.asarray(when, dtype=float))
        return np.interp(np.asarray(when, dtype=float), self.t, self.c)


def series_from_bouts(mapping):
    """{key: [(t0, t1, secs), ...]} -> {key: CumulativeSeries}."""
    return {k: CumulativeSeries.from_bouts(v) for k, v in (mapping or {}).items()}


def natural_key(name):
    """Sort key that orders embedded numbers numerically, not lexically.

    Animal labels are a sex letter and an ID ("M9657", "F9714"), so a plain
    string sort groups by sex and then by ID - which is what a reader scanning
    the panel expects. Digits are compared as numbers so that, should IDs ever
    differ in length, M999 sorts before M1000 rather than after it.
    """
    out, digits = [], ""
    for ch in str(name):
        if ch.isdigit():
            digits += ch
        else:
            if digits:
                out.append((1, int(digits), ""))
                digits = ""
            out.append((0, 0, ch.lower()))
    if digits:
        out.append((1, int(digits), ""))
    return out


def pair_key(a, b):
    """A dyad's canonical key, so (A, B) and (B, A) are the same cell."""
    return (a, b) if str(a) <= str(b) else (b, a)


def overlap_bouts_by_pair(rows):
    """Group social-overlap bout rows into {(a, b): [(t0, t1, secs), ...]}.

    ``rows`` is an iterable of (animal_a, animal_b, start_s, end_s) with times
    as float epoch seconds. Taking the exported bouts as the input - rather
    than recomputing proximity here - is deliberate: the panel and
    ``_SocialOverlapBouts.csv`` then cannot disagree, because there is only one
    computation.
    """
    out = {}
    for a, b, t0, t1 in rows or ():
        t0, t1 = float(t0), float(t1)
        if t1 < t0:
            t0, t1 = t1, t0
        out.setdefault(pair_key(a, b), []).append((t0, t1, t1 - t0))
    return out


def epoch_seconds(values, tz=None):
    """Datetimes as UTC epoch seconds, whatever timezone they arrive in.

    The occupancy curves are built from ``Timestamp.values``, which pandas
    hands back as UTC-based datetime64 regardless of the column's timezone. A
    bout read back from CSV has to land on the same scale or the two halves of
    the panel would be plotted against different clocks.
    """
    import pandas as pd

    s = pd.to_datetime(pd.Series(values), errors='coerce')
    if getattr(s.dtype, 'tz', None) is not None:
        s = s.dt.tz_convert('UTC').dt.tz_localize(None)
    elif tz is not None:
        s = (s.dt.tz_localize(tz, ambiguous=True, nonexistent='shift_forward')
              .dt.tz_convert('UTC').dt.tz_localize(None))
    return s.values.astype('datetime64[ns]').astype('int64') / 1e9


def overlap_bouts_from_frame(bouts, tz=None):
    """{pair: [(t0, t1, secs)]} from an exported proximity-bouts table.

    Takes the table's OWN ``duration_s`` rather than recomputing it from the
    two timestamps. That column is what a person summing the CSV would add up,
    so using it makes the panel's dyad totals identical to the file's by
    construction instead of merely close - two bout boundaries can round
    differently from the duration recorded between them.
    """
    if bouts is None or len(bouts) == 0:
        return {}
    cols = set(getattr(bouts, 'columns', ()))
    need = {'animal1', 'animal2', 'bout_start', 'bout_stop'}
    if not need <= cols:
        raise ValueError(f"bouts table missing {sorted(need - cols)}")

    t0 = epoch_seconds(bouts['bout_start'], tz)
    t1 = epoch_seconds(bouts['bout_stop'], tz)
    if 'duration_s' in cols:
        secs = np.asarray(bouts['duration_s'], dtype=float)
    else:
        secs = t1 - t0

    out = {}
    for a, b, s0, s1, d in zip(bouts['animal1'], bouts['animal2'], t0, t1, secs):
        if not (np.isfinite(s0) and np.isfinite(s1) and np.isfinite(d)):
            continue
        if s1 < s0:
            s0, s1 = s1, s0
        out.setdefault(pair_key(a, b), []).append((float(s0), float(s1), float(d)))
    return out


def triangle_cells(animals):
    """(row, col) index pairs of a lower-triangular dyad grid.

    Row i is animals[i], column j is animals[j], j < i - every unordered pair
    exactly once, which is what a dyad grid should show. 16 animals give 120
    cells in 15 rows, the widest holding 15.
    """
    n = len(animals)
    return [(i, j) for i in range(1, n) for j in range(i)]
