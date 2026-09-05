"""GBI construction, the hand-off case, and config round-tripping.

The hand-off test is the one worth reading: when one animal's bout ends at the
same instant another's begins, the interval between those two endpoints has
zero length and BOTH animals are present in it. Those rows are the clearest
evidence of one animal replacing another in a zone, and an implementation that
probes half a second later loses every one of them.
"""
import pandas as pd
import pytest

from fnt.rfid.config import (Antenna, Arena, TrialConfig, Zone, ConfigManager,
                             get_default_config)
from fnt.rfid.core.gbi_generator import create_gbi, melt_gbi


def bout(name, sex, zone, start, stop, day=1):
    return {"trial": "T001", "name": name, "sex": sex, "zone": zone,
            "noon_day": day,
            "field_time": pd.Timestamp(start), "field_time_stop": pd.Timestamp(stop),
            "duration_s": (pd.Timestamp(stop) - pd.Timestamp(start)).total_seconds()}


SEX = {"A": "M", "B": "F", "C": "M"}


def test_two_overlapping_bouts_split_into_three_intervals():
    bouts = pd.DataFrame([
        bout("A", "M", 1, "2021-05-07 20:00:00", "2021-05-07 20:00:30"),
        bout("B", "F", 1, "2021-05-07 20:00:10", "2021-05-07 20:00:40")])
    gbi = create_gbi(bouts, ["A", "B"], SEX)
    assert len(gbi) == 3
    assert list(gbi["mf_sum"]) == [1, 2, 1]
    assert list(gbi["duration_s"]) == [10, 20, 10]


def test_a_handoff_records_both_animals():
    """B leaves exactly as A arrives: a zero-length interval holding both."""
    bouts = pd.DataFrame([
        bout("B", "F", 1, "2021-05-07 20:00:00", "2021-05-07 20:00:30"),
        bout("A", "M", 1, "2021-05-07 20:00:30", "2021-05-07 20:01:00")])
    gbi = create_gbi(bouts, ["A", "B"], SEX, min_duration_s=1.0)
    handoff = gbi[gbi["field_time_start"] == gbi["field_time_stop"]]
    assert len(handoff) == 1
    assert handoff["mf_sum"].iloc[0] == 2
    assert handoff["duration_s"].iloc[0] == 1.0


def test_intervals_with_nobody_present_are_dropped():
    bouts = pd.DataFrame([
        bout("A", "M", 1, "2021-05-07 20:00:00", "2021-05-07 20:00:10"),
        bout("A", "M", 1, "2021-05-07 21:00:00", "2021-05-07 21:00:10")])
    gbi = create_gbi(bouts, ["A"], SEX)
    assert len(gbi) == 2
    assert (gbi["mf_sum"] == 1).all()


def test_zones_do_not_leak_into_each_other():
    bouts = pd.DataFrame([
        bout("A", "M", 1, "2021-05-07 20:00:00", "2021-05-07 20:00:30"),
        bout("B", "F", 5, "2021-05-07 20:00:00", "2021-05-07 20:00:30")])
    gbi = create_gbi(bouts, ["A", "B"], SEX)
    assert set(gbi["zone"]) == {1, 5}
    assert (gbi["mf_sum"] == 1).all()


def test_melt_gives_one_row_per_animal_per_interval():
    bouts = pd.DataFrame([
        bout("A", "M", 1, "2021-05-07 20:00:00", "2021-05-07 20:00:30"),
        bout("B", "F", 1, "2021-05-07 20:00:10", "2021-05-07 20:00:40")])
    gbi = create_gbi(bouts, ["A", "B"], SEX)
    long = melt_gbi(gbi)
    assert len(long) == int(gbi["mf_sum"].sum()) == 4
    assert set(long["name"]) == {"A", "B"}


# --------------------------------------------------------------- config ----

def test_grid_arena_pairs_two_antennas_per_zone():
    arena = Arena.grid(cols=2, rows=4, dx=7.5, dy=7.6, x0=3.75, y0=7.6,
                       antennas_per_zone=2)
    assert len(arena.zones) == 8
    assert len(arena.antennas) == 16
    mapping = arena.antenna_zone_map()
    assert mapping[1] == 1 and mapping[9] == 1        # wall + floor, zone 1
    assert mapping[8] == 8 and mapping[16] == 8
    assert arena.antenna_location_map()[9] == "floor"


def test_config_round_trip_is_lossless(tmp_path):
    """A field that falls out of the round-trip silently changes a re-run."""
    cfg = get_default_config("8_zone_paddock")
    cfg.trial_id = "T001"
    cfg.raw_dir = str(tmp_path)
    cfg.metadata_path = str(tmp_path / "metadata.csv")
    cfg.reader_ids = [1]
    cfg.bout_threshold_s = 42.5
    cfg.time_resolution = "s"
    cfg.foreign_reader_policy = "keep"
    cfg.analysis_days = (2, 11)
    cfg.exports["sna"] = True

    path = tmp_path / "cfg.json"
    ConfigManager.save_to_file(cfg, str(path))
    back = ConfigManager.load_from_file(str(path))

    assert back.to_dict() == cfg.to_dict()
    assert back.analysis_days == (2, 11)
    assert back.arena.antenna_zone_map() == cfg.arena.antenna_zone_map()


def test_saved_config_carries_provenance(tmp_path):
    cfg = get_default_config("8_zone_paddock")
    ConfigManager.save(cfg, str(tmp_path))
    raw = ConfigManager.load_raw(str(tmp_path))
    assert "fnt_version" in raw and "run_timestamp" in raw
    # and the stamps must not come back as config fields
    assert ConfigManager.load(str(tmp_path)).to_dict() == cfg.to_dict()


def test_validation_catches_an_unreachable_zone():
    cfg = TrialConfig(arena=Arena(zones=[Zone(1, 0, 0), Zone(2, 1, 0)],
                                  antennas=[Antenna(1, zone=1)]))
    ok, problems = ConfigManager.validate(cfg)
    assert not ok
    assert any("no antenna" in p for p in problems)


def test_validation_catches_a_duplicate_antenna_id():
    cfg = TrialConfig(arena=Arena(zones=[Zone(1, 0, 0)],
                                  antennas=[Antenna(1, zone=1), Antenna(1, zone=1)]))
    ok, problems = ConfigManager.validate(cfg)
    assert not ok
    assert any("more than once" in p for p in problems)


def test_arena_reports_silent_and_unknown_antennas():
    """A zone whose only antenna is dead looks exactly like an unvisited zone."""
    arena = Arena.grid(cols=2, rows=1, dx=1, dy=1, antennas_per_zone=1)
    assert arena.silent_antennas([1]) == [2]
    assert arena.unmapped_antennas([1, 2, 99]) == [99]
