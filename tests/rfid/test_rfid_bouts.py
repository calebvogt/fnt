"""Bouts: gap, zone change, noon boundary, lone reads - and the legacy rule.

The default rule is the one the R reference uses: consecutive reads of one
animal in the same zone_id no more than the gap apart (<= 50 s) are one bout;
a zone change, a longer gap, or the noon day boundary ends it; a lone read is
a 1 s bout.
"""
import numpy as np
import pandas as pd
import pytest

from fnt.rfid.core.bout_detector import detect_bouts

GAP = 50.0
T0 = pd.Timestamp("2021-05-08 18:00:00")


def reads(rows, animal="Mars"):
    """rows of (seconds after T0, zone_id[, day])"""
    frame = pd.DataFrame({
        "animal": animal,
        "t": [T0 + pd.Timedelta(seconds=r[0]) for r in rows],
        "zone_id": [r[1] for r in rows],
        "day": [r[2] if len(r) > 2 else 2 for r in rows]})
    frame["paddock"] = frame["zone_id"].str[0]
    frame["zone"] = frame["zone_id"].str[1:]
    frame["away"] = False
    return frame


def test_close_reads_in_one_zone_are_one_bout():
    b = detect_bouts(reads([(0, "A3"), (5, "A3"), (20, "A3"), (30.5, "A3")]), GAP)
    assert len(b) == 1
    assert b["duration_s"].iloc[0] == 30.5 and b["n_reads"].iloc[0] == 4
    assert b["stop"].iloc[0] == T0 + pd.Timedelta(seconds=30.5)


def test_a_gap_of_exactly_the_threshold_continues_the_bout():
    b = detect_bouts(reads([(0, "A3"), (50, "A3"), (100.001, "A3")]), GAP)
    assert len(b) == 2
    assert b["n_reads"].tolist() == [2, 1]


def test_a_zone_change_ends_the_bout_even_without_a_gap():
    b = detect_bouts(reads([(0, "A3"), (1, "A3"), (2, "A4"), (3, "A3")]), GAP)
    assert b["zone_id"].tolist() == ["A3", "A4", "A3"]


def test_the_same_zone_number_in_another_paddock_is_another_zone():
    b = detect_bouts(reads([(0, "A3"), (1, "B3")]), GAP)
    assert b["zone_id"].tolist() == ["A3", "B3"]


def test_the_noon_boundary_ends_the_bout():
    b = detect_bouts(reads([(0, "A3", 1), (10, "A3", 1), (20, "A3", 2), (30, "A3", 2)]), GAP)
    assert len(b) == 2 and b["day"].tolist() == [1, 2]


def test_a_lone_read_is_a_one_second_bout():
    b = detect_bouts(reads([(0, "A3"), (500, "A3")]), GAP, min_bout_s=1.0)
    assert b["duration_s"].tolist() == [1.0, 1.0]
    assert (b["stop"] - b["start"]).dt.total_seconds().tolist() == [1.0, 1.0]


def test_animals_never_share_a_bout():
    two = pd.concat([reads([(0, "A3"), (5, "A3")], "Mars"),
                     reads([(6, "A3"), (8, "A3")], "Zeus")])
    two = two.sort_values(["animal", "t"]).reset_index(drop=True)
    b = detect_bouts(two, GAP)
    assert b["animal"].tolist() == ["Mars", "Zeus"]


def test_bouts_never_contain_a_long_gap_or_a_zone_change():
    rng = np.random.default_rng(0)
    offsets = np.cumsum(rng.integers(1, 120, size=500)).astype(float)
    zones = rng.choice(["A1", "A2", "B1"], size=500)
    r = reads(list(zip(offsets, zones)))
    b = detect_bouts(r, GAP)
    for start, stop, zone, n in zip(b["start"], b["stop"], b["zone_id"], b["n_reads"]):
        # a lone read's stop is start + 1 s by convention, not a read
        last = start if n == 1 else stop
        inside = r[(r["t"] >= start) & (r["t"] <= last)]
        assert (inside["zone_id"] == zone).all()
        if len(inside) > 1:
            assert (inside["t"].diff().dt.total_seconds().dropna() <= GAP).all()
    assert b["n_reads"].sum() == len(r)


def test_unknown_rule_is_rejected():
    with pytest.raises(ValueError):
        detect_bouts(reads([(0, "A3")]), GAP, rule="nonsense")


# --------------------------------------------------------------- legacy ----

def _legacy_loop(group_t, group_z, thresh):
    """The old per-animal-day loop from the previous bout_detector, verbatim in
    logic (a port of 2_create_ALLTRIAL_MOVEBOUT.R). The vectorised version must
    reproduce it exactly. Returns [(start_t, stop_t or None, kind)]."""
    t, z = np.asarray(group_t, float), np.asarray(group_z)
    n = len(t)
    status = [None] * n
    if n == 1:
        status[0] = "S"
    elif n == 2:
        if z[0] == z[1] and (t[1] - t[0]) <= thresh:
            status[0], status[1] = "START", "STOP"
        if (t[1] - t[0]) >= thresh:
            status[0] = status[1] = "S"
    else:
        if (t[1] - t[0]) >= thresh or z[1] != z[0]:
            status[0] = "S"
        if (t[-1] - t[-2]) >= thresh or z[-1] != z[-2]:
            status[-1] = "S"
        for i in range(1, n - 1):
            if ((t[i + 1] - t[i]) >= thresh and (t[i] - t[i - 1]) >= thresh) or \
                    (z[i - 1] != z[i] and z[i + 1] != z[i]):
                status[i] = "S"
    out = [(t[i], None, "single") for i in range(n) if status[i] == "S"]
    rest = [i for i in range(n) if status[i] != "S"]
    if not rest:
        return out
    marks = [None] * len(rest)
    marks[0], marks[-1] = "START", "STOP"
    rt, rz = t[rest], z[rest]
    for i in range(len(rest)):
        if marks[i] is not None:
            continue
        if rz[i + 1] == rz[i] and (rt[i + 1] - rt[i]) >= thresh:
            marks[i], marks[i + 1] = "STOP", "START"
        if rz[i + 1] != rz[i]:
            marks[i], marks[i + 1] = "STOP", "START"
    starts = [rt[i] for i in range(len(rest)) if marks[i] == "START"]
    stops = [rt[i] for i in range(len(rest)) if marks[i] == "STOP"]
    if len(starts) == len(stops) + 1:
        starts = starts[:-1]
    if len(stops) == len(starts) + 1:
        stops = stops[:-1]
    if len(starts) != len(stops) or not starts:
        return out
    return out + [(s, e, "paired") for s, e in zip(starts, stops)]


@pytest.mark.parametrize("seed", range(6))
def test_vectorised_legacy_rule_matches_the_old_loop(seed):
    rng = np.random.default_rng(seed)
    frames, expected = [], []
    for animal in ("Mars", "Zeus"):
        for day in (1, 2, 3):
            n = int(rng.integers(1, 40))
            offs = np.cumsum(rng.choice([1, 5, 20, 49, 50, 51, 300], size=n)).astype(float)
            zones = rng.choice(["A1", "A2", "A3"], size=n, p=[0.6, 0.3, 0.1])
            f = reads([(o + day * 100000, zz, day) for o, zz in zip(offs, zones)], animal)
            frames.append(f)
            secs = (f["t"] - T0).dt.total_seconds().to_numpy()
            for s, e, kind in _legacy_loop(secs, zones, GAP):
                expected.append((animal, day, round(s, 3),
                                 None if e is None else round(e, 3), kind))
    r = pd.concat(frames).sort_values(["animal", "t"], kind="mergesort").reset_index(drop=True)
    b = detect_bouts(r, GAP, rule="legacy_r")
    got = []
    for row in b.itertuples():
        s = round((row.start - T0).total_seconds(), 3)
        if pd.isna(row.n_reads):
            got.append((row.animal, row.day, s, round((row.stop - T0).total_seconds(), 3),
                        "paired"))
        else:
            got.append((row.animal, row.day, s, None, "single"))
    assert sorted(got, key=str) == sorted(expected, key=str)


def test_legacy_rule_can_pair_across_a_zone_change():
    """Why it is not the default: two reads in different zones less than the
    gap apart become one bout."""
    b = detect_bouts(reads([(0, "A1"), (10, "A2")]), GAP, rule="legacy_r")
    assert len(b) == 1 and b["duration_s"].iloc[0] == 10
