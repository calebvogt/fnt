"""
Proximity Detection for UWB Tracking Data

Ported from 2025_VoleCosm/Scripts/02_proximity_detection.R
Detects pairwise proximity events and bouts from smoothed UWB positional data.

Produces two outputs matching the VoleCosm schema:
  - proximity_events: per-timestamp pairwise distances with in_proximity flag
  - proximity_bouts: contiguous time periods where a pair is within threshold
"""

import numpy as np
import pandas as pd
from itertools import combinations


def detect_proximity_bouts(df, threshold=0.5, gap_s=5, tag_identities=None,
                           log_callback=None):
    """Detect pairwise proximity events and bouts from smoothed UWB data.

    Parameters
    ----------
    df : pd.DataFrame
        Smoothed full-resolution data with columns:
        Timestamp, shortid, smoothed_x/location_x, smoothed_y/location_y,
        and optionally sex, identity.
    threshold : float
        Distance in metres below which two tags are "in proximity".
    gap_s : float
        Maximum gap in seconds between consecutive proximity observations
        before a new bout is started.
    tag_identities : dict or None
        Mapping of shortid -> {'sex': str, 'identity': str, ...}.
        Used to build sexid-style animal labels (e.g. "F9040").
    log_callback : callable or None
        Function for progress messages.

    Returns
    -------
    (proximity_events, proximity_bouts) : tuple of pd.DataFrame
        proximity_events columns:
            timestamp, Day, Date, animal1, animal2, distance, in_proximity
        proximity_bouts columns:
            animal1, animal2, Day, Date, bout_start, bout_stop,
            duration_s, mean_distance, n_observations
    """
    def _log(msg):
        if log_callback:
            log_callback(msg)

    # ── Resolve coordinate columns ───────────────────────────────────────
    x_col = 'smoothed_x' if 'smoothed_x' in df.columns else 'location_x'
    y_col = 'smoothed_y' if 'smoothed_y' in df.columns else 'location_y'

    _log(f"Proximity detection: using {x_col}/{y_col}, "
         f"threshold={threshold} m, gap={gap_s} s")

    # ── Build animal labels ──────────────────────────────────────────────
    # Create a sexid-style label for each tag, matching VoleCosm convention
    if tag_identities:
        label_map = {}
        for tag, info in tag_identities.items():
            sex = info.get('sex', 'M')[0].upper()  # First letter
            identity = info.get('identity', str(tag))
            label_map[tag] = f"{sex}{identity}"
    else:
        label_map = None

    # ── Prepare working copy ─────────────────────────────────────────────
    work = df[['Timestamp', 'shortid', x_col, y_col]].copy()
    # Parse via UTC (robust to a recording that crosses a DST boundary and so
    # carries two offsets), then convert back to the recording's own timezone
    # so exported bout times read in local wall-clock like every other product.
    work['Timestamp'] = pd.to_datetime(work['Timestamp'], format='ISO8601',
                                       utc=True, errors='coerce')
    work = work.dropna(subset=['Timestamp', x_col, y_col])
    src_tz = getattr(df['Timestamp'].dt, 'tz', None) if 'Timestamp' in df else None
    if src_tz is not None:
        work['Timestamp'] = work['Timestamp'].dt.tz_convert(src_tz)

    # Bin to 1 s to PAIR the animals. Tags report asynchronously at well under
    # 1 Hz, so two animals essentially never share an exact millisecond: without
    # a binning key the pairwise join would match almost nothing. The bin is
    # only the join key — the true millisecond timestamps are carried alongside
    # (ts_first/ts_last below) so bout edges and durations keep full resolution.
    work['ts_rounded'] = work['Timestamp'].dt.floor('1s')

    # Derive Day and Date from timestamp.
    #
    # Kept as datetime64 normalised to midnight rather than Python
    # ``datetime.date`` objects. The old form built an object column (one Python
    # date per row) and then ran a per-row ``.apply()`` over the differences —
    # on a multi-million-row trial that is tens of millions of short-lived
    # object allocations before any distance is computed, which is slow and a
    # large amount of native allocator churn. The vectorised form below is
    # equivalent (both use the UTC calendar day) and allocates nothing per row.
    # Converted back to plain dates at the very end, on the much smaller bouts
    # table, so the exported CSV is byte-identical.
    day0 = work['ts_rounded'].dt.normalize()
    work['Date'] = day0
    work['Day'] = (day0 - day0.min()).dt.days + 1

    # Apply animal labels
    if label_map:
        work['animal'] = work['shortid'].map(
            lambda x: label_map.get(x, str(x)))
    else:
        work['animal'] = work['shortid'].astype(str)

    animals = sorted(work['animal'].unique())
    if len(animals) < 2:
        _log("Warning: fewer than 2 animals found — skipping proximity detection")
        empty_events = pd.DataFrame(columns=['timestamp', 'Day', 'Date',
                                             'animal1', 'animal2',
                                             'distance', 'in_proximity'])
        empty_bouts = pd.DataFrame(columns=['animal1', 'animal2', 'Day',
                                            'Date', 'bout_start', 'bout_stop',
                                            'duration_s', 'mean_distance',
                                            'n_observations'])
        return empty_events, empty_bouts

    _log(f"Computing pairwise distances for {len(animals)} animals "
         f"across {work['ts_rounded'].nunique()} unique timestamps...")

    # ── Pairwise distance calculation ────────────────────────────────────
    # Group by rounded timestamp, then compute all pair distances
    pairs = list(combinations(animals, 2))
    _log(f"  {len(pairs)} unique animal pairs")

    # Pivot so each timestamp has one row per animal with x,y
    grouped = work.groupby(['ts_rounded', 'animal']).agg(
        x=(x_col, 'mean'),
        y=(y_col, 'mean'),
        # True (millisecond) span of the fixes that fell in this bin, so a bout
        # can later report when contact actually started/ended rather than the
        # bin edge.
        ts_first=('Timestamp', 'min'),
        ts_last=('Timestamp', 'max'),
        Day=('Day', 'first'),
        Date=('Date', 'first')
    ).reset_index()

    # Build events via merge for each pair
    events_list = []
    for a1, a2 in pairs:
        d1 = grouped[grouped['animal'] == a1][
            ['ts_rounded', 'x', 'y', 'ts_first', 'ts_last', 'Day', 'Date']]
        d2 = grouped[grouped['animal'] == a2][
            ['ts_rounded', 'x', 'y', 'ts_first', 'ts_last']]
        merged = d1.merge(d2, on='ts_rounded', suffixes=('_1', '_2'))
        if merged.empty:
            continue
        merged['distance'] = np.sqrt(
            (merged['x_1'] - merged['x_2'])**2 +
            (merged['y_1'] - merged['y_2'])**2
        )
        merged['in_proximity'] = merged['distance'] <= threshold
        merged['animal1'] = a1
        merged['animal2'] = a2
        # Millisecond envelope of the evidence for this pair in this bin: the
        # earliest and latest real fix from either animal. Bout edges are taken
        # from these, so bout_start/bout_stop are true observation times rather
        # than 1 s bin edges.
        merged['ts_first'] = merged[['ts_first_1', 'ts_first_2']].min(axis=1)
        merged['ts_last'] = merged[['ts_last_1', 'ts_last_2']].max(axis=1)
        events_list.append(
            merged[['ts_rounded', 'ts_first', 'ts_last', 'Day', 'Date',
                    'animal1', 'animal2', 'distance', 'in_proximity']]
        )

    if not events_list:
        _log("Warning: no overlapping timestamps between animals")
        empty_events = pd.DataFrame(columns=['timestamp', 'Day', 'Date',
                                             'animal1', 'animal2',
                                             'distance', 'in_proximity'])
        empty_bouts = pd.DataFrame(columns=['animal1', 'animal2', 'Day',
                                            'Date', 'bout_start', 'bout_stop',
                                            'duration_s', 'mean_distance',
                                            'n_observations'])
        return empty_events, empty_bouts

    proximity_events = pd.concat(events_list, ignore_index=True)
    # 'timestamp' stays the 1 s pairing bin (that is what a per-bin distance
    # refers to); ts_first/ts_last carry the millisecond evidence window and are
    # consumed by the bout aggregation below.
    proximity_events = proximity_events.rename(columns={'ts_rounded': 'timestamp'})
    proximity_events = proximity_events.sort_values(
        ['animal1', 'animal2', 'timestamp']).reset_index(drop=True)

    n_in_prox = proximity_events['in_proximity'].sum()
    _log(f"  {len(proximity_events)} pairwise observations, "
         f"{n_in_prox} within threshold")

    # ── Bout detection ───────────────────────────────────────────────────
    _log("Detecting proximity bouts...")
    prox_only = proximity_events[proximity_events['in_proximity']].copy()

    if prox_only.empty:
        _log("No proximity events detected at this threshold")
        empty_bouts = pd.DataFrame(columns=['animal1', 'animal2', 'Day',
                                            'Date', 'bout_start', 'bout_stop',
                                            'duration_s', 'mean_distance',
                                            'n_observations'])
        return proximity_events, empty_bouts

    prox_only = prox_only.sort_values(['animal1', 'animal2', 'timestamp'])

    # Calculate time gaps within each pair
    prox_only['prev_ts'] = prox_only.groupby(
        ['animal1', 'animal2'])['timestamp'].shift(1)
    prox_only['time_gap'] = (
        prox_only['timestamp'] - prox_only['prev_ts']
    ).dt.total_seconds()

    # New bout when gap > gap_s or first observation for pair
    prox_only['new_bout'] = (
        prox_only['time_gap'].isna() | (prox_only['time_gap'] > gap_s)
    )
    prox_only['bout_id'] = prox_only.groupby(
        ['animal1', 'animal2'])['new_bout'].cumsum()

    # Summarise each bout
    proximity_bouts = prox_only.groupby(
        ['animal1', 'animal2', 'bout_id'], sort=False
    ).agg(
        # True millisecond edges: the first and last real fix supporting this
        # contact, NOT the 1 s bins used to pair the animals.
        bout_start=('ts_first', 'min'),
        bout_stop=('ts_last', 'max'),
        mean_distance=('distance', 'mean'),
        n_observations=('distance', 'count'),
        Day=('Day', 'first'),
        Date=('Date', 'first')
    ).reset_index()

    # Observed duration at full resolution. Deliberately NOT clipped to a 1 s
    # floor as the original R port was: a bout supported by a single pair of
    # near-simultaneous fixes genuinely spans only milliseconds, and rounding it
    # up to a second overstated contact time. Sum n_observations instead if a
    # sampling-interval-weighted budget is wanted.
    proximity_bouts['duration_s'] = (
        proximity_bouts['bout_stop'] - proximity_bouts['bout_start']
    ).dt.total_seconds()

    # Select and order columns to match VoleCosm schema
    proximity_bouts = proximity_bouts[
        ['animal1', 'animal2', 'Day', 'Date', 'bout_start', 'bout_stop',
         'duration_s', 'mean_distance', 'n_observations']
    ].sort_values(['animal1', 'animal2', 'bout_start']).reset_index(drop=True)

    # Present Date as a plain calendar date, matching the previous output.
    proximity_bouts['Date'] = proximity_bouts['Date'].dt.date

    _log(f"  {len(proximity_bouts)} proximity bouts detected")

    return proximity_events, proximity_bouts
