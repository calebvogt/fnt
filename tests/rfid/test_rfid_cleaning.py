"""Cleaning rules: identity, place, cross-paddock policy, windows, lost tags,
found-dead cuts, and noon-to-noon days.

Each rule removes reads for a reason a person decided on; the tests pin both
what is removed and what is not.
"""
import numpy as np
import pandas as pd

from _rfid_synth import TAGS, animal_table, lid_like_arena, raw_reads
from fnt.rfid.config import RFIDConfig
from fnt.rfid.core.cleaning import clean_reads, day_index, elapsed_seconds

ANIMALS = animal_table([
    ("Mars", "T001", "M", [TAGS["M1a"], TAGS["M1b"]], "2021-05-07 17:48", "2021-05-20 09:20"),
    ("Flora", "T001", "F", [TAGS["F1a"], TAGS["F1b"]], "2021-05-07 17:40", None),
    ("Duke", "T002", "M", [TAGS["M2a"], TAGS["M2b"]], "2021-05-07 18:03", None),
])


def config(**kw):
    cfg = RFIDConfig(name="t", day_origin="2021-05-07 12:00:00")
    cfg.arena = lid_like_arena()
    for k, v in kw.items():
        setattr(cfg, k, v)
    return cfg


def at(minutes, day0="2021-05-08 20:00:00"):
    return pd.Timestamp(day0) + pd.Timedelta(minutes=minutes)


def test_unknown_tags_are_dropped_and_listed():
    raw = raw_reads([(at(0), "1", "3", TAGS["M1a"]), (at(1), "1", "3", TAGS["test"]),
                     (at(2), "2", "3", TAGS["test"]), (at(3), "1", "4", "999.1")])
    res = clean_reads(raw, ANIMALS, config())
    assert len(res.reads) == 1
    assert res.dropped["unknown_tag"] == 3
    u = res.unknown_tags.set_index("tag")
    assert u.loc[TAGS["test"], "reads"] == 2
    assert u.loc[TAGS["test"], "readers"] == "1,2"


def test_reader_and_antenna_together_decide_the_zone():
    raw = raw_reads([(at(0), "1", "3", TAGS["M1a"]), (at(1), "2", "3", TAGS["M2a"]),
                     (at(2), "2", "11", TAGS["M2a"])])
    reads = clean_reads(raw, ANIMALS, config()).reads
    got = dict(zip(zip(reads["animal"].astype(str), reads["antenna"].astype(str)),
                   reads["zone_id"].astype(str)))
    assert got[("Mars", "3")] == "A3"
    assert got[("Duke", "3")] == "B3"
    assert got[("Duke", "11")] == "B3"
    assert set(reads["location"].astype(str)) == {"wall", "floor"}


def test_unmapped_antennas_are_dropped_and_listed():
    raw = raw_reads([(at(0), "1", "3", TAGS["M1a"]), (at(1), "1", "40", TAGS["M1a"]),
                     (at(2), "9", "1", TAGS["M1a"])])
    res = clean_reads(raw, ANIMALS, config())
    assert len(res.reads) == 1 and res.dropped["unmapped_antenna"] == 2
    assert set(zip(res.unmapped["reader"], res.unmapped["antenna"])) == {("1", "40"), ("9", "1")}


def cross_raw():
    # Mars (home A) read twice by reader 2 (B5), once at home
    return raw_reads([(at(0), "1", "5", TAGS["M1a"]), (at(10), "2", "5", TAGS["M1a"]),
                      (at(11), "2", "13", TAGS["M1b"])])


def test_other_paddock_reads_are_kept_as_away_visits_by_default():
    reads = clean_reads(cross_raw(), ANIMALS, config()).reads
    assert reads["zone_id"].astype(str).tolist() == ["A5", "B5", "B5"]
    assert reads["away"].tolist() == [False, True, True]
    assert reads["paddock"].astype(str).tolist() == ["A", "B", "B"]


def test_other_paddock_reads_can_be_dropped():
    res = clean_reads(cross_raw(), ANIMALS, config(cross_paddock_policy="drop"))
    assert res.reads["zone_id"].astype(str).tolist() == ["A5"]
    assert res.dropped["cross_paddock"] == 2


def test_legacy_fold_relabels_into_the_home_paddock():
    reads = clean_reads(cross_raw(), ANIMALS, config(cross_paddock_policy="fold")).reads
    assert reads["zone_id"].astype(str).tolist() == ["A5", "A5", "A5"]
    assert reads["away"].tolist() == [False, True, True]    # still traceable


def test_lost_tag_drops_only_that_tag_inside_its_interval():
    raw = raw_reads([(at(m), "1", "3", TAGS["F1a"]) for m in range(0, 60, 10)]
                    + [(at(m), "1", "5", TAGS["F1b"]) for m in range(0, 60, 10)])
    cfg = config()
    cfg.animals["Flora"] = {"lost_tags": [{"tag": TAGS["F1a"], "start": str(at(20)),
                                           "stop": str(at(40))}]}
    res = clean_reads(raw, ANIMALS, cfg)
    f1a = res.reads[res.reads["tag"].astype(str) == TAGS["F1a"]]["t"].tolist()
    assert f1a == [at(0), at(10), at(40), at(50)]            # [start, stop)
    assert (res.reads["tag"].astype(str) == TAGS["F1b"]).sum() == 6
    assert res.dropped["lost_tag"] == 2


def test_open_ended_lost_tag_and_the_tag_as_a_spreadsheet_number():
    raw = raw_reads([(at(m), "1", "3", TAGS["F1a"]) for m in range(0, 30, 10)])
    cfg = config()
    cfg.animals["Flora"] = {"lost_tags": [{"tag": float(TAGS["F1a"]), "start": str(at(10))}]}
    assert clean_reads(raw, ANIMALS, cfg).reads["t"].tolist() == [at(0)]


def dead_raw():
    """Duke enters B5 twice, the second time for good; then only the floor
    antenna reads him, hour after hour - a body under the zone."""
    rows = [(at(0), "2", "5", TAGS["M2a"]),       # entry (first read)
            (at(0.5), "2", "13", TAGS["M2a"]),
            (at(30), "2", "5", TAGS["M2a"]),      # entry after a 29.5 min gap
            (at(30.2), "2", "13", TAGS["M2a"])]
    rows += [(at(60 * h), "2", "13", TAGS["M2a"]) for h in range(1, 30)]
    rows += [(at(60 * 12 + 5), "2", "5", TAGS["M2b"])]      # a later wall read, e.g. beetles
    return raw_reads(rows)


def test_found_dead_cuts_at_the_last_entry_plus_one_gap():
    cfg = config()
    cfg.animals["Duke"] = {"dead_mode": "last_entry"}
    res = clean_reads(dead_raw(), ANIMALS, cfg)
    le = res.last_entries.iloc[0]
    # the latest entry is the stray wall read at 12 h 05 min
    assert le["last_entry"] == at(60 * 12 + 5)
    assert le["cut_at"] == at(60 * 12 + 5) + pd.Timedelta(seconds=50)


def test_found_time_bounds_the_last_entry_search():
    """Entries after the body was found are not the animal's."""
    cfg = config()
    cfg.animals["Duke"] = {"dead_mode": "last_entry", "dead_time": str(at(45))}
    res = clean_reads(dead_raw(), ANIMALS, cfg)
    le = res.last_entries.iloc[0]
    assert le["last_entry"] == at(30) and le["last_entry_zone"] == "B5"
    kept = res.reads["t"].tolist()
    assert kept == [at(0), at(0.5), at(30), at(30.2)]      # within 50 s of the entry
    assert res.dropped["found_dead"] == 30


def test_found_dead_at_a_manual_time():
    cfg = config()
    cfg.animals["Duke"] = {"dead_mode": "manual", "dead_time": str(at(90))}
    res = clean_reads(dead_raw(), ANIMALS, cfg)
    assert res.reads["t"].max() == at(60)


def test_windows_trim_to_release_trap_and_manual_times():
    raw = raw_reads([(pd.Timestamp("2021-05-07 17:00"), "1", "3", TAGS["M1a"]),
                     (pd.Timestamp("2021-05-07 18:00"), "1", "3", TAGS["M1a"]),
                     (pd.Timestamp("2021-05-20 10:00"), "1", "3", TAGS["M1a"]),
                     (pd.Timestamp("2021-05-07 17:00"), "1", "4", TAGS["F1a"]),
                     (pd.Timestamp("2021-05-09 17:00"), "1", "4", TAGS["F1a"])])
    none = clean_reads(raw, ANIMALS, config())
    assert len(none.reads) == 5                             # off by default
    both = clean_reads(raw, ANIMALS, config(trim_to_release=True, trim_to_trap=True))
    assert both.dropped["before_start"] == 2 and both.dropped["after_stop"] == 1
    cfg = config()
    cfg.animals["Flora"] = {"start_mode": "manual", "start_time": "2021-05-08 00:00:00"}
    manual = clean_reads(raw, ANIMALS, cfg)
    assert manual.dropped["before_start"] == 1


def test_noon_to_noon_days():
    origin = pd.Timestamp("2021-05-07 12:00")
    t = pd.Series(pd.to_datetime(["2021-05-07 11:59:59", "2021-05-07 12:00:00",
                                  "2021-05-07 12:00:01", "2021-05-08 11:59:59",
                                  "2021-05-08 12:00:01", "2021-05-14 16:50:00"]))
    assert day_index(t, origin).tolist() == [0, 0, 1, 1, 2, 8]


def test_day_origin_defaults_to_noon_on_the_first_release_date():
    raw = raw_reads([(pd.Timestamp("2021-05-06 16:59"), "1", "3", TAGS["M1a"]),
                     (pd.Timestamp("2021-05-07 20:00"), "1", "3", TAGS["M1a"])])
    res = clean_reads(raw, ANIMALS, config(day_origin=""))
    assert res.origin == pd.Timestamp("2021-05-07 12:00")
    assert res.reads["day"].tolist() == [0, 1]


def test_gaps_are_measured_across_a_dst_change():
    t = pd.Series(pd.to_datetime(["2021-03-14 01:59:00", "2021-03-14 03:01:00"]))
    naive, _ = elapsed_seconds(t, "")
    local, _ = elapsed_seconds(t, "America/New_York")
    assert np.diff(naive)[0] == 62 * 60
    assert np.diff(local)[0] == 2 * 60                      # the clock jumped an hour


def test_reads_are_sorted_by_animal_then_time():
    raw = raw_reads([(at(5), "1", "3", TAGS["M1a"]), (at(1), "1", "4", TAGS["F1a"]),
                     (at(0), "1", "3", TAGS["M1b"])])
    reads = clean_reads(raw, ANIMALS, config()).reads
    assert reads["animal"].astype(str).tolist() == ["Flora", "Mars", "Mars"]
    assert reads["t"].tolist() == [at(1), at(0), at(5)]
