"""ROI occupancy as BOUTS rather than as summed dwell time.

A bout is a maximal run of consecutive fixes inside one region. The rule is
deliberately strict: a single fix outside the region, or inside a different
one, ends the bout, and the next fix back inside starts a new one. Nothing is
smoothed over, no gap is bridged, and no duration is rounded — the exported
numbers are the observed ones.

Two products come from the same runs:

``region_bouts``      the bouts themselves, one row per animal per visit.
``daily_summary``     per region per day, who held it and for how long.

Only the summary splits at midnight, because it is definitionally daily. The
bout list never does: splitting destroys the distinction between one bout that
crossed midnight and two that did not, and the summary can always re-derive
what it needs from whole bouts.

Nothing here imports Qt or matplotlib, so the segmentation and the ranking are
testable without a GUI.
"""

import numpy as np
import pandas as pd

# A lone fix inside a region is real evidence of a visit whose duration cannot
# be observed — its start and stop are the same instant. Crediting it zero
# would drop it from every total; crediting it the sampling interval is the
# smallest defensible non-zero answer.
SINGLE_FIX_S = 1.0


def region_bouts(zone_idx, times_ns, single_fix_s=SINGLE_FIX_S):
    """Runs of consecutive fixes in one region, as (region, t0, t1, seconds).

    ``zone_idx`` is the per-fix region index from ``spatial_metrics.assign_zones``
    (-1 for outside), ``times_ns`` the matching timestamps as int64 nanoseconds,
    both in fix order. A run of length one reports ``t0 == t1`` and
    ``single_fix_s`` seconds; every other run reports its true span, unrounded.
    """
    zi = np.asarray(zone_idx)
    t = np.asarray(times_ns, dtype='int64')
    n = len(zi)
    if n == 0 or len(t) != n:
        return []

    # A run break is any change of region index, outside included: leaving to
    # -1 and coming back is two visits, not one.
    brk = np.empty(n, dtype=bool)
    brk[0] = True
    brk[1:] = zi[1:] != zi[:-1]
    starts = np.flatnonzero(brk)
    stops = np.append(starts[1:], n) - 1

    out = []
    for s, e in zip(starts.tolist(), stops.tolist()):
        r = int(zi[s])
        if r < 0:                      # outside every region: not a visit
            continue
        t0, t1 = int(t[s]), int(t[e])
        if s == e:
            out.append((r, t0, t0, float(single_fix_s)))
        else:
            out.append((r, t0, t1, (t1 - t0) / 1e9))
    return out


def merge_animal_bouts(rows, single_fix_s=SINGLE_FIX_S):
    """Union the bouts of one animal carried by more than one tag.

    Statistics are keyed on the configured ANIMAL, not the tag, so a trial in
    which two tags map to one identity — a mid-trial replacement, or a spare
    running alongside — must not bill that animal twice for the same minute.
    Bouts within a single tag never overlap, so for the usual one-tag animal
    this returns its input unchanged.

    ``rows`` are dicts carrying at least SexID, ROI, bout_start, bout_stop and
    duration_s. Overlapping or touching intervals for the same (animal, region)
    are fused; the fused duration is recomputed from the union's own edges.
    """
    if not rows:
        return []
    by_key = {}
    for r in rows:
        by_key.setdefault((r['SexID'], r['ROI']), []).append(r)

    out = []
    for (_sex, _roi), group in by_key.items():
        group = sorted(group, key=lambda r: (r['bout_start'], r['bout_stop']))
        cur = dict(group[0])
        for nxt in group[1:]:
            if nxt['bout_start'] <= cur['bout_stop']:
                if nxt['bout_stop'] > cur['bout_stop']:
                    cur['bout_stop'] = nxt['bout_stop']
                    cur['duration_s'] = _span_seconds(
                        cur['bout_start'], cur['bout_stop'], single_fix_s)
            else:
                out.append(cur)
                cur = dict(nxt)
        out.append(cur)
    out.sort(key=lambda r: (natural_animal_key(r['SexID']), r['bout_start']))
    return out


def _span_seconds(t0, t1, single_fix_s=SINGLE_FIX_S):
    """Seconds between two timestamps, with the lone-fix convention applied."""
    secs = (t1 - t0).total_seconds()
    return float(single_fix_s) if secs <= 0 else float(secs)


def natural_animal_key(label):
    """Sort key that reads M9 before M10, and keeps the sexes together."""
    s = str(label)
    head = s[:1]
    digits = ''.join(ch for ch in s if ch.isdigit())
    return (head, int(digits) if digits else 0, s)


def split_by_day(t0, t1, duration_s, single_fix=False):
    """One bout's seconds attributed to the calendar days it covers.

    Yields ``(day, seconds)`` with ``day`` the normalised local midnight. Used
    only by the daily summary — the bout list itself is never split. A lone-fix
    bout has no span to divide, so its whole credited second lands on the day
    it was observed.
    """
    if single_fix or t1 <= t0:
        yield t0.normalize(), float(duration_s)
        return
    d0, d1 = t0.normalize(), t1.normalize()
    if d0 == d1:
        yield d0, float(duration_s)
        return
    edges = pd.date_range(d0, d1 + pd.Timedelta(days=1), freq='D')
    for i in range(len(edges) - 1):
        lo = max(t0, edges[i])
        hi = min(t1, edges[i + 1])
        if hi > lo:
            yield edges[i], (hi - lo).total_seconds()


def daily_summary(rows, single_fix_s=SINGLE_FIX_S):
    """Per region per day: who held it, for how long, and in what order.

    One row per animal per region per day, for animals that were actually in
    that region that day — an animal that never entered gets no row, matching
    the shape of the RFID zone-ownership table this mirrors.

    ``total_ROI_time_s`` is every animal's time in that region that day, so
    ``focal_ROI_perc_time`` is a share of the region's whole traffic and
    ``rank_order`` says who held it. Ties take the average rank, so two animals
    level on 2.5% are both 3.5 rather than arbitrarily 3 and 4.

    This is the only product that splits a bout at midnight. It has to: the
    file is per-day. Splitting here and nowhere else keeps the two files
    reconciling exactly — the summary's seconds sum to the bout list's.
    """
    if not rows:
        return []

    # (SexID, ROI, day) -> seconds
    acc = {}
    for r in rows:
        t0, t1 = r['bout_start'], r['bout_stop']
        lone = t1 <= t0
        for day, secs in split_by_day(t0, t1, r['duration_s'], single_fix=lone):
            key = (r['SexID'], r['ROI'], day)
            acc[key] = acc.get(key, 0.0) + secs
    if not acc:
        return []

    df = pd.DataFrame(
        [{'SexID': k[0], 'ROI': k[1], 'day': k[2], 'focal_ROI_time_s': v}
         for k, v in acc.items()])

    # Day 1 is the first calendar day any animal was in any region, matching
    # the Day convention every other export uses.
    day0 = df['day'].min()
    df['Day'] = (df['day'] - day0).dt.days + 1

    totals = df.groupby(['ROI', 'day'])['focal_ROI_time_s'].transform('sum')
    df['total_ROI_time_s'] = totals
    df['focal_ROI_perc_time'] = np.where(
        totals > 0, df['focal_ROI_time_s'] / totals * 100.0, 0.0)
    df['rank_order'] = (df.groupby(['ROI', 'day'])['focal_ROI_perc_time']
                          .rank(method='average', ascending=False))

    df = df.sort_values(['ROI', 'Day', 'rank_order', 'SexID'],
                        kind='mergesort').reset_index(drop=True)
    return df[['SexID', 'Day', 'ROI', 'total_ROI_time_s', 'focal_ROI_time_s',
               'focal_ROI_perc_time', 'rank_order']].to_dict('records')
