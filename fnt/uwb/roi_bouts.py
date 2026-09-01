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

# A lone read inside a region is real evidence of a visit whose duration
# cannot be observed — its start and stop are the same instant. It is reported
# as it happened: duration 0.0 s, with n_reads = 1.
#
# Crediting it the sampling interval instead was rejected. That would make
# duration_s mean two different things in one column — a measurement for most
# rows and an assumption for the rest — which no downstream total, mean or
# distribution could separate afterwards. Reporting 0 keeps
# `duration_s == bout_stop - bout_start` true with NO exceptions, makes total
# occupancy a strict lower bound on the truth, and loses nothing: the visit is
# still a row, and n_reads says exactly how much evidence stands behind it.
LONE_READ_S = 0.0


def region_bouts(zone_idx, times_ns, lone_read_s=LONE_READ_S):
    """Runs of consecutive reads in one region.

    Yields ``(region, t0, t1, seconds, n_reads)``.

    ``zone_idx`` is the per-fix region index from ``spatial_metrics.assign_zones``
    (-1 for outside), ``times_ns`` the matching timestamps as int64 nanoseconds,
    both in fix order. A run of length one reports ``t0 == t1`` and
    ``lone_read_s`` seconds; every other run reports its true span, unrounded.
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
        n_reads = e - s + 1
        if s == e:
            out.append((r, t0, t0, float(lone_read_s), n_reads))
        else:
            out.append((r, t0, t1, (t1 - t0) / 1e9, n_reads))
    return out


def merge_animal_bouts(rows, lone_read_s=LONE_READ_S):
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
                cur['n_reads'] = cur.get('n_reads', 0) + nxt.get('n_reads', 0)
                if nxt['bout_stop'] > cur['bout_stop']:
                    cur['bout_stop'] = nxt['bout_stop']
                    cur['duration_s'] = _span_seconds(
                        cur['bout_start'], cur['bout_stop'], lone_read_s)
            else:
                out.append(cur)
                cur = dict(nxt)
        out.append(cur)
    out.sort(key=lambda r: (natural_animal_key(r['SexID']), r['bout_start']))
    return out


def _span_seconds(t0, t1, lone_read_s=LONE_READ_S):
    """Seconds between two timestamps, with the lone-fix convention applied."""
    secs = (t1 - t0).total_seconds()
    return float(lone_read_s) if secs <= 0 else float(secs)


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


def daily_summary(rows, lone_read_s=LONE_READ_S):
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

    # (SexID, ROI, day) -> [seconds, reads]
    #
    # Reads are carried alongside the seconds because a lone-read visit
    # contributes 0 s. Without a count, such a row would say nothing at all -
    # and dropping it instead would hide a visit that genuinely happened.
    # A read is credited to the day of the bout it belongs to; only TIME is
    # divided at midnight, since a read is an instant and cannot be split.
    acc = {}
    for r in rows:
        t0, t1 = r['bout_start'], r['bout_stop']
        lone = t1 <= t0
        parts = list(split_by_day(t0, t1, r['duration_s'], single_fix=lone))
        total = sum(p[1] for p in parts)
        reads = int(r.get('n_reads', 0) or 0)
        for i, (day, secs) in enumerate(parts):
            key = (r['SexID'], r['ROI'], day)
            cur = acc.setdefault(key, [0.0, 0])
            cur[0] += secs
            # Whole reads, apportioned by each day's share of the span, with
            # any remainder on the last day so the total is preserved exactly.
            if i == len(parts) - 1:
                cur[1] += reads - sum(
                    int(reads * p[1] / total) for p in parts[:-1]) if total else reads
            else:
                cur[1] += int(reads * secs / total) if total else 0
    if not acc:
        return []

    df = pd.DataFrame(
        [{'SexID': k[0], 'ROI': k[1], 'day': k[2],
          'focal_ROI_time_s': v[0], 'n_reads': int(v[1])}
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
               'focal_ROI_perc_time', 'rank_order', 'n_reads']].to_dict('records')


# ── self-audit ───────────────────────────────────────────────────────────────
# Every exported product is checked against the invariants that must hold if it
# is safe to analyse, and the result goes in the session log. A number that is
# wrong in a way nobody notices is the expensive kind: these files are the input
# to statistics run days later, where a silent 1 % error looks like biology.
#
# Each function returns [(ok, message), ...]. The caller logs them; a False
# entry is a defect in the export, not a property of the animals.

def audit_bouts(rows):
    """Invariants for the ROI bout list.

    Takes the finished frame or the rows behind it. The frame is what the
    caller should pass: Day and the row order only exist once it is built,
    and those are among the things checked.
    """
    out = []
    df = rows if isinstance(rows, pd.DataFrame) else pd.DataFrame(rows)
    if df.empty:
        return [(True, "no ROI visits to check")]
    n = len(df)

    out.append((df.notna().all().all().item(), f"{n:,} bout(s), no missing values"))

    span = (df['bout_stop'] - df['bout_start']).dt.total_seconds()
    out.append(((span >= 0).all().item(), "bout_stop is never before bout_start"))
    out.append(((df['duration_s'] >= 0).all().item(),
                "no negative durations"))
    out.append(((df['n_reads'] >= 1).all().item(),
                "every visit is backed by at least one read"))

    # THE invariant this file is built on. No exceptions, lone reads included:
    # duration_s is always the observed span and never an assumption.
    bad = (span - df['duration_s']).abs() > 1e-6
    out.append((not bad.any().item(),
                "duration_s == bout_stop - bout_start for EVERY row"
                + ("" if not bad.any() else f" — {int(bad.sum())} disagree")))

    lone = df['n_reads'] == 1
    n_lone = int(lone.sum())
    out.append((bool((df.loc[lone, 'duration_s'] == 0.0).all()),
                f"{n_lone:,} lone-read visit(s), reported as 0 s — real visits "
                f"whose duration cannot be observed, so occupancy totals are a "
                f"lower bound"))

    # Regions take precedence over each other, so an animal is inside at most
    # one at a time. Overlapping bouts would mean double-counted occupancy.
    laps = 0
    for _a, g in df.groupby('SexID'):
        g = g.sort_values('bout_start')
        laps += int((g['bout_start'].values[1:] < g['bout_stop'].values[:-1]).sum())
    out.append((laps == 0,
                "no animal is in two regions at once"
                + ("" if laps == 0 else f" — {laps} overlapping pair(s)")))

    day0 = df['bout_start'].dt.normalize().min()
    want = (df['bout_start'].dt.normalize() - day0).dt.days + 1
    bad_day = int((want != df['Day']).sum())
    out.append((bad_day == 0,
                "Day is the calendar day each bout started"
                + ("" if bad_day == 0 else f" — {bad_day} row(s) disagree")))
    return out


def audit_daily(daily_rows, bout_rows):
    """Invariants for the daily summary, including the reconciliation."""
    out = []
    if not daily_rows:
        return [(True, "no daily rows to check")]
    df = pd.DataFrame(daily_rows)
    out.append((df.notna().all().all().item(),
                f"{len(df):,} row(s), no missing values"))
    # Zero-time rows are legitimate now: an animal whose only visits that day
    # were lone reads was genuinely there, for an unmeasurable duration.
    out.append(((df['focal_ROI_time_s'] >= 0).all().item(),
                "no negative times"))
    zero = int((df['focal_ROI_time_s'] == 0).sum())
    out.append(((df.loc[df['focal_ROI_time_s'] == 0, 'n_reads'] >= 1).all().item(),
                f"{zero:,} zero-time row(s), each backed by lone reads"))
    out.append(((df['focal_ROI_time_s']
                 <= df['total_ROI_time_s'] + 1e-6).all().item(),
                "focal time never exceeds the region's total"))

    bad_pct = bad_rank = n_zero_grp = 0
    for _k, g in df.groupby(['ROI', 'Day']):
        # A region-day whose every visit was a lone read totals 0 s. There is
        # no share of zero to take, so the percentages are 0 and CANNOT sum to
        # 100. Exempting it is not a loophole - asserting 100 there would make
        # the self-check report a failure on arithmetic that is correct.
        if g['total_ROI_time_s'].iloc[0] <= 0:
            n_zero_grp += 1
            continue
        if abs(g['focal_ROI_perc_time'].sum() - 100.0) > 1e-6:
            bad_pct += 1
        want = g['focal_ROI_perc_time'].rank(method='average', ascending=False)
        if not np.allclose(want.values, g['rank_order'].values):
            bad_rank += 1
    out.append((bad_pct == 0,
                "shares sum to 100% within every region-day"
                + (f" (excluding {n_zero_grp} whose visits were all lone "
                   f"reads, totalling 0 s)" if n_zero_grp else "")
                + ("" if bad_pct == 0 else f" — {bad_pct} group(s) do not")))
    out.append((bad_rank == 0,
                "rank_order is the average-tie rank of the share"
                + ("" if bad_rank == 0 else f" — {bad_rank} group(s) wrong")))

    # THE cross-file check. This file is the bout list split at midnight, so
    # its seconds must sum to the bout list's exactly. Any drift between the
    # two means one of them is measuring something the other is not.
    tot_d = float(df['focal_ROI_time_s'].sum())
    tot_b = float(sum(r['duration_s'] for r in bout_rows))
    err = abs(tot_d - tot_b)
    out.append((err < 1e-3,
                f"reconciles with the bout list ({tot_d:,.1f}s vs {tot_b:,.1f}s, "
                f"error {err:.2e}s)"))
    return out


def audit_social(bouts, threshold_m):
    """Invariants for the social overlap bouts (a pandas frame)."""
    out = []
    if bouts is None or len(bouts) == 0:
        return [(True, "no social bouts to check")]
    df = bouts
    out.append((df.notna().all().all().item(),
                f"{len(df):,} bout(s), no missing values"))
    out.append(((df['animal1'] != df['animal2']).all().item(), "no self-pairs"))
    out.append(((df['animal1'] < df['animal2']).all().item(),
                "dyads are canonically ordered"))
    span = (df['bout_stop'] - df['bout_start']).dt.total_seconds()
    out.append(((span >= 0).all().item(), "bout_stop is never before bout_start"))
    out.append(((span - df['duration_s']).abs().max() < 1e-6,
                "duration_s equals bout_stop - bout_start"))
    worst = float(df['mean_distance'].max())
    out.append((worst <= threshold_m + 1e-9,
                f"no bout exceeds the {threshold_m:g} m contact threshold "
                f"(worst mean {worst:.3f} m)"))
    laps = 0
    for _k, g in df.groupby(['animal1', 'animal2'], sort=False):
        g = g.sort_values('bout_start')
        laps += int((g['bout_start'].values[1:] < g['bout_stop'].values[:-1]).sum())
    out.append((laps == 0,
                "a dyad's bouts never overlap"
                + ("" if laps == 0 else f" — {laps} overlapping pair(s)")))
    n_zero = int((df['duration_s'] == 0).sum())
    if n_zero:
        out.append((True,
                    f"note: {n_zero:,} bout(s) span a single instant and are "
                    f"credited 0s, matching the ROI files' lone-read rule"))
    return out
