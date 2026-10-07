"""QC reports that point a person at lost tags and bodies."""
import pandas as pd

from _rfid_synth import TAGS, animal_table, lid_like_arena, raw_reads
from fnt.rfid.config import RFIDConfig
from fnt.rfid.core import qc
from fnt.rfid.core.cleaning import clean_reads

ANIMALS = animal_table([
    ("Calypso", "T002", "F", [TAGS["F1a"], TAGS["F1b"]], "2021-05-07 17:00", None),
    ("Duke", "T002", "M", [TAGS["M2a"], TAGS["M2b"]], "2021-05-07 17:00", None),
    ("Mars", "T001", "M", [TAGS["M1a"], TAGS["M1b"]], "2021-05-07 17:00", None),
])
# just after the start of day 3: a read exactly at noon is the last moment of day 2
DAY3 = pd.Timestamp("2021-05-09 12:00:30")


def cfg():
    c = RFIDConfig(name="qc", day_origin="2021-05-07 12:00:00")
    c.arena = lid_like_arena()
    return c


def lost_tag_scene():
    """Day 3: Calypso's tag a lies in the B3 entrance (wall antenna 3, read
    every ~20 min all day), tag b moves with her into B5. Duke's body lies
    under the B6 floor (antenna 14), read every hour, no entries. Mars moves
    normally between A1 and A2."""
    rows = []
    for k in range(70):                                   # ~23 h of B3 wall reads
        rows.append((DAY3 + pd.Timedelta(minutes=20 * k), "2", "3", TAGS["F1a"]))
    for h in range(0, 24, 3):
        rows.append((DAY3 + pd.Timedelta(hours=h, minutes=5), "2", "5", TAGS["F1b"]))
        rows.append((DAY3 + pd.Timedelta(hours=h, minutes=6), "2", "13", TAGS["F1b"]))
    for h in range(23):
        rows.append((DAY3 + pd.Timedelta(hours=h, minutes=30), "2", "14", TAGS["M2a"]))
    for h in range(0, 23, 2):
        rows.append((DAY3 + pd.Timedelta(hours=h, minutes=1), "1", "1", TAGS["M1a"]))
        rows.append((DAY3 + pd.Timedelta(hours=h + 1, minutes=1), "1", "2", TAGS["M1a"]))
    return raw_reads(rows)


def test_two_tags_in_two_places_is_flagged():
    res = clean_reads(lost_tag_scene(), ANIMALS, cfg())
    tp = qc.two_places(res.identified, share=0.8)
    assert tp["animal"].tolist() == ["Calypso"]
    row = tp.iloc[0]
    assert row["day"] == 3
    assert {row["zone_a"], row["zone_b"]} == {"B3", "B5"}
    assert row["cut_reads"] == 0


def test_a_configured_lost_tag_shows_as_covered():
    c = cfg()
    c.animals["Calypso"] = {"lost_tags": [{"tag": TAGS["F1a"], "start": str(DAY3)}]}
    res = clean_reads(lost_tag_scene(), ANIMALS, c)
    tp = qc.two_places(res.identified, share=0.8)
    assert tp.iloc[0]["cut_reads"] == 70
    # and the cleaned reads no longer hold the tag
    assert TAGS["F1a"] not in set(res.reads["tag"].astype(str))


def test_stationary_tags_find_the_body_and_the_tag_in_the_tube():
    res = clean_reads(lost_tag_scene(), ANIMALS, cfg())
    st = qc.stationary_tags(res.identified, ["wall"], 50.0, 0.5)
    got = {(a, z): s for a, z, s in zip(st["animal"], st["zone_id"], st["status"])}
    assert got[("Duke", "B6")].startswith("no entries")
    assert got[("Calypso", "B3")].startswith("entries on its only antenna")
    assert "Mars" not in set(st["animal"])                # two antennas: moving


def test_cross_paddock_report_and_roster():
    rows = [(DAY3, "1", "3", TAGS["M1a"]), (DAY3 + pd.Timedelta(hours=1), "2", "3", TAGS["M1a"]),
            (DAY3 + pd.Timedelta(hours=2), "2", "4", TAGS["M1b"])]
    res = clean_reads(raw_reads(rows), ANIMALS, cfg())
    cp = qc.cross_paddock(res.identified, "keep")
    assert cp.iloc[0]["animal"] == "Mars" and cp.iloc[0]["reads"] == 2
    assert cp.iloc[0]["zone_ids"] == "B3, B4"
    home = {"Mars": "A", "Duke": "B", "Calypso": "B"}
    roster = qc.animal_roster(ANIMALS.animals, res.identified, res.reads,
                              res.last_entries, home).set_index("animal")
    assert roster.loc["Mars", "reads"] == 3 and roster.loc["Mars", "away_reads"] == 2
    assert roster.loc["Duke", "never_read"] == True       # noqa: E712
    assert roster.loc["Mars", "reads_before_release"] == 0


def test_dropped_summary_lists_every_rule():
    res = clean_reads(lost_tag_scene(), ANIMALS, cfg())
    d = qc.dropped_summary(res.dropped, res.identified)
    assert d["rule"].tolist()[0] == "unknown_tag"
    assert set(d["rule"]) >= {"lost_tag", "found_dead", "cross_paddock"}
