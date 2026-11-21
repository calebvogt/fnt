"""Summaries: co-occupancy, zone holders, and the per animal-day table."""
import itertools

import numpy as np
import pandas as pd

from _rfid_synth import lid_like_arena
from fnt.rfid.core.summaries import (animal_days, co_occupancy, overlapping_bouts,
                                     zone_holders)

T0 = pd.Timestamp("2021-05-08 22:00:00")


def bout(animal, zone_id, start_min, stop_min, day=2):
    return {"animal": animal, "zone_id": zone_id, "paddock": zone_id[0],
            "zone": zone_id[1:], "away": False, "day": day,
            "start": T0 + pd.Timedelta(minutes=start_min),
            "stop": T0 + pd.Timedelta(minutes=stop_min),
            "duration_s": (stop_min - start_min) * 60.0}


ANIMALS = pd.DataFrame({"animal": ["W", "X", "Y", "Z"], "name": ["W", "X", "Y", "Z"],
                        "trial": "T001", "sex": ["F", "M", "F", "M"], "phase": "",
                        "code": ["WWW", "XXX", "YYY", "ZZZ"]})


def test_co_occupancy_on_a_hand_made_example():
    """A1: X 0-10 and 15-16, Y 5-20, W 12-13. Z sits in A2 the whole time.

    X-Y overlap 5-10 (300 s) and 15-16 (60 s) = 360 s in two overlaps; W-Y
    12-13 = 60 s; W never meets X (X is out 10-15); Z is in another zone; X's
    two bouts do not count as X meeting X.
    """
    bouts = pd.DataFrame([bout("X", "A1", 0, 10), bout("Y", "A1", 5, 20),
                          bout("X", "A1", 15, 16), bout("W", "A1", 12, 13),
                          bout("Z", "A2", 0, 30)])
    co = co_occupancy(bouts, ANIMALS)
    got = {(a, b): (s, n) for a, b, s, n in
           zip(co["animal_a"], co["animal_b"], co["seconds"], co["n_overlaps"])}
    assert got == {("X", "Y"): (360.0, 2), ("W", "Y"): (60.0, 1)}
    assert co.set_index(["animal_a", "animal_b"]).loc[("X", "Y"), "zone_ids"] == "A1"
    assert co.set_index(["animal_a", "animal_b"]).loc[("X", "Y"), "sex_a"] == "M"


def test_a_long_bout_overlaps_every_short_one_inside_it():
    bouts = pd.DataFrame([bout("X", "B3", 0, 600)]
                         + [bout("Y", "B3", 10 * i, 10 * i + 2) for i in range(1, 30)])
    co = co_occupancy(bouts, ANIMALS)
    assert co["seconds"].sum() == 29 * 120 and co["n_overlaps"].sum() == 29


def test_touching_bouts_do_not_overlap():
    bouts = pd.DataFrame([bout("X", "A1", 0, 10), bout("Y", "A1", 10, 20)])
    assert co_occupancy(bouts, ANIMALS).empty


def test_sweep_matches_brute_force_on_random_bouts():
    rng = np.random.default_rng(3)
    rows = []
    for animal in "WXYZ":
        t = 0.0
        for _ in range(60):
            t += rng.exponential(20)
            d = rng.exponential(8) + 1 / 60
            rows.append(bout(animal, rng.choice(["A1", "A2", "B1"]), t, t + d))
            t += d
    bouts = pd.DataFrame(rows)
    fast = co_occupancy(bouts, ANIMALS).set_index(["animal_a", "animal_b"])["seconds"]
    brute = {}
    for i, j in itertools.combinations(range(len(bouts)), 2):
        a, b = bouts.iloc[i], bouts.iloc[j]
        if a["animal"] == b["animal"] or a["zone_id"] != b["zone_id"]:
            continue
        ov = (min(a["stop"], b["stop"]) - max(a["start"], b["start"])).total_seconds()
        if ov > 0:
            key = tuple(sorted((a["animal"], b["animal"])))
            brute[key] = brute.get(key, 0.0) + ov
    assert set(fast.index) == set(brute)
    for key, seconds in brute.items():
        assert abs(fast[key] - seconds) < 1e-3


def test_overlap_pairs_carry_the_day_of_the_later_bout():
    bouts = pd.DataFrame([bout("X", "A1", 0, 10, day=1), bout("Y", "A1", 5, 20, day=2)])
    pairs = overlapping_bouts(bouts)
    assert pairs["day"].tolist() == [2]


# ------------------------------------------------------------ holders ----

def reads_for(counts):
    """counts: {(animal, zone_id, day): n} -> a reads frame"""
    rows = []
    for (animal, zone_id, day), n in counts.items():
        rows += [{"animal": animal, "zone_id": zone_id, "paddock": zone_id[0],
                  "zone": zone_id[1:], "day": day, "away": False,
                  "t": T0 + pd.Timedelta(seconds=i)} for i in range(n)]
    return pd.DataFrame(rows)


def test_zone_holders_held_contested_and_female_reads_ignored():
    r = reads_for({("X", "A1", 2): 60, ("Z", "A1", 2): 40, ("Y", "A1", 2): 500,
                   ("X", "A2", 2): 50, ("Z", "A2", 2): 50,
                   ("Z", "B1", 2): 3})
    h = zone_holders(r, ANIMALS, share=0.5, sex="M").set_index(["zone_id", "day"])
    assert h.loc[("A1", 2), "holder"] == "X"
    assert h.loc[("A1", 2), "top_share"] == 0.6
    assert h.loc[("A1", 2), "sex_reads"] == 100              # Y's 500 are not male
    assert h.loc[("A1", 2), "runner_up"] == "Z"
    assert h.loc[("A2", 2), "status"] == "contested"         # 50/50 is not > 50%
    assert h.loc[("A2", 2), "holder"] == ""
    assert h.loc[("B1", 2), "holder"] == "Z"                  # a visitor counts where read
    assert h.loc[("A1", 2), "top_code"] == "XXX"


def test_holder_threshold_is_configurable():
    r = reads_for({("X", "A1", 2): 60, ("Z", "A1", 2): 40})
    assert zone_holders(r, ANIMALS, share=0.7)["status"].iloc[0] == "contested"


# ----------------------------------------------------------- animal-days ----

def test_minimum_distance_counts_moves_within_a_paddock_only():
    """X: A1 -> A2 -> B1 -> A1 on day 2; A1-A2 is 7.5 m apart (same row).
    A2 -> B1 crosses paddocks (nothing), B1 -> A1 too (nothing)."""
    bouts = pd.DataFrame([bout("X", "A1", 0, 1), bout("X", "A1", 2, 3),
                          bout("X", "A2", 4, 5), bout("X", "B1", 6, 7),
                          bout("X", "A1", 8, 9)])
    bouts["away"] = bouts["paddock"] == "B"
    r = reads_for({("X", "A1", 2): 3, ("X", "A2", 2): 1, ("X", "B1", 2): 1})
    r["away"] = r["paddock"] == "B"
    ad = animal_days(r, bouts, ANIMALS, lid_like_arena().zone_xy(), {"X": "A"})
    row = ad.set_index("day").loc[2]
    assert row["min_distance_m"] == 7.5
    assert row["zones_entered"] == 3 and row["zone_ids"] == "A1;A2;B1"
    assert row["reads"] == 5 and row["reads_away"] == 1
    assert row["time_in_zones_s"] == 300 and row["time_away_s"] == 60
    assert row["home_paddock"] == "A"


def test_days_without_reads_between_first_and_last_are_zero():
    bouts = pd.DataFrame([bout("X", "A1", 0, 1, day=2), bout("X", "A3", 0, 1, day=5)])
    r = reads_for({("X", "A1", 2): 2, ("X", "A3", 5): 2})
    ad = animal_days(r, bouts, ANIMALS, lid_like_arena().zone_xy())
    assert ad["day"].tolist() == [2, 3, 4, 5]
    assert ad["detected"].tolist() == [True, False, False, True]
    assert ad["reads"].tolist() == [2, 0, 0, 2]
    assert ad["zones_entered"].tolist() == [1, 0, 0, 1]
