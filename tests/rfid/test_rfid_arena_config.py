"""The arena and the config: (reader, antenna) -> zone_id, presets, persistence.

The key change from the earlier module: an antenna number alone does not say
where a read happened. LID ran two paddocks side by side, reader 1 in Alpha
and reader 2 in Bravo, both numbering their antennas 1-16, so antenna 3 is A3
on reader 1 and B3 on reader 2.
"""
import json
import os

import pandas as pd

from _rfid_synth import lid_like_arena
from fnt.rfid.config import (Antenna, Arena, ConfigManager, Paddock, RFIDConfig,
                             Zone, default_output_dir, liddell_8_zone,
                             one_antenna_per_zone, read_antenna_csv,
                             write_antenna_csv, zones_for_antennas)
from fnt.rfid.config.config_manager import output_inside_downloads
from fnt.rfid.config.defaults import normalise_id


def antenna_map(arena):
    return {(r["reader"], r["antenna"]): r for r in arena.antenna_rows()}


def test_liddell_preset_wall_and_floor_share_a_zone():
    arena = liddell_8_zone(1)
    m = antenna_map(arena)
    assert len(arena.zones) == 8 and len(arena.antennas) == 16
    assert m[("1", "1")]["zone_id"] == "A1" and m[("1", "1")]["location"] == "wall"
    assert m[("1", "9")]["zone_id"] == "A1" and m[("1", "9")]["location"] == "floor"
    assert m[("1", "16")]["zone_id"] == "A8"
    assert (m[("1", "3")]["zone_x"], m[("1", "3")]["zone_y"]) == (3.75, 15.2)


def test_two_readers_with_the_same_antenna_numbers_map_to_different_paddocks():
    m = antenna_map(liddell_8_zone(2))
    assert m[("1", "3")]["zone_id"] == "A3"
    assert m[("2", "3")]["zone_id"] == "B3"
    assert m[("2", "11")]["zone_id"] == "B3"
    assert m[("2", "11")]["paddock"] == "B"


def test_an_antenna_can_override_its_readers_paddock():
    arena = liddell_8_zone(2)
    arena.antennas.append(Antenna("1", "17", "2", "tube", paddock_id="B"))
    assert antenna_map(arena)[("1", "17")]["zone_id"] == "B2"


def test_one_antenna_per_zone_preset():
    arena = one_antenna_per_zone(6, cols=3, width_m=2.44, length_m=2.44)
    m = antenna_map(arena)
    assert len(m) == 6
    assert all(m[("1", str(i))]["zone_id"] == f"A{i}" for i in range(1, 7))
    blocking, warnings = arena.problems()
    assert not blocking and not warnings


def test_ids_are_compared_without_zero_padding():
    assert normalise_id("001") == "1"
    assert normalise_id(1.0) == "1"
    assert normalise_id(" 13 ") == "13"
    assert normalise_id("A") == "A"


def test_home_paddock_matches_trial_id_or_name():
    arena = lid_like_arena()
    assert arena.home_paddock("T001") == "A"
    assert arena.home_paddock("t002") == "B"
    assert arena.home_paddock("B") == "B"
    assert arena.home_paddock("Alpha") == "A"
    assert arena.home_paddock("T009") == ""


def test_arena_problems():
    arena = Arena(paddocks=[Paddock("A", readers=["1"]), Paddock("B", readers=["1"])],
                  zones=[Zone("A", "1"), Zone("A", "2")],
                  antennas=[Antenna("1", "1", "1"), Antenna("1", "1", "1"),
                            Antenna("1", "2", "9"), Antenna("7", "1", "1")])
    blocking, warnings = arena.problems()
    text = " | ".join(blocking)
    assert "Reader 1 is listed in paddocks A and B" in text
    assert "more than once" in text
    assert "not in the zones table" in text and "A9" in text
    assert "belong to no paddock" in text
    assert any("no antenna" in w for w in warnings)
    assert any("without coordinates" in w for w in warnings)


def test_antenna_csv_round_trip_with_loose_headers(tmp_path):
    arena = lid_like_arena()
    path = tmp_path / "antennas.csv"
    write_antenna_csv(arena, path)
    back = read_antenna_csv(path)
    assert [(a.reader, a.antenna, a.zone, a.location) for a in back] == \
        [(a.reader, a.antenna, a.zone, a.location) for a in arena.antennas]
    typed = tmp_path / "typed.csv"
    typed.write_text("Reader ID,Antenna ID,Zone,Type\n001,003,3,wall\n2,5,5,floor\n")
    got = read_antenna_csv(typed)
    assert [(a.reader, a.antenna, a.zone, a.location) for a in got] == \
        [("1", "3", "3", "wall"), ("2", "5", "5", "floor")]


def test_zones_are_created_for_imported_antennas():
    arena = Arena(paddocks=[Paddock("A", readers=["1"])])
    new = zones_for_antennas(arena, [Antenna("1", "1", "1"), Antenna("1", "2", "1"),
                                     Antenna("1", "3", "2")])
    assert [z.zone_id for z in new] == ["A1", "A2"]


def test_config_round_trip_is_lossless(tmp_path):
    cfg = RFIDConfig(name="LID", download_dirs=["/data/T001"], timezone="America/New_York",
                     day_origin="2021-05-07 12:00:00", bout_gap_s=42.5,
                     cross_paddock_policy="fold", entrance_locations=["wall", "tube"])
    cfg.arena = lid_like_arena()
    cfg.metadata.path = "/data/meta.xlsx"
    cfg.metadata.columns = {"name": "name", "sex": "sex"}
    cfg.metadata.tag_columns = ["tag_1", "tag_2"]
    cfg.animals["Calypso"] = {"lost_tags": [{"tag": "985.113004548427",
                                             "start": "2021-05-16 12:00:00"}]}
    cfg.animals["Isac"] = {"dead_mode": "last_entry"}
    cfg.layers["gbi"] = True
    path = tmp_path / "cfg.json"
    ConfigManager.save_to_file(cfg, str(path))
    back = ConfigManager.load_from_file(str(path))
    assert back.to_dict() == cfg.to_dict()
    assert back.animal("Calypso")["lost_tags"][0]["tag"] == "985.113004548427"
    assert back.animal("Nobody")["dead_mode"] == "none"


def test_a_run_record_loads_back_as_a_config(tmp_path):
    cfg = RFIDConfig(name="X")
    cfg.arena = liddell_8_zone(1)
    ConfigManager.save(cfg, str(tmp_path), extra={"provenance": {"fnt_version": "t"}})
    raw = json.loads((tmp_path / "fnt_config.json").read_text())
    assert raw["provenance"]["fnt_version"] == "t" and "config" in raw
    assert ConfigManager.load(str(tmp_path)).to_dict() == cfg.to_dict()


def test_default_output_sits_beside_the_download_folders(tmp_path):
    a, b = tmp_path / "T001_Alpha", tmp_path / "T002_Bravo"
    a.mkdir()
    b.mkdir()
    cfg = RFIDConfig(download_dirs=[str(a), str(b)])
    out = default_output_dir(cfg)
    assert os.path.dirname(out) == str(tmp_path)
    assert os.path.basename(out) == "T001_Alpha+T002_Bravo_FNT_analysis"
    cfg.name = "LID_2021"
    assert os.path.basename(default_output_dir(cfg)) == "LID_2021_FNT_analysis"
    assert output_inside_downloads(str(a / "out"), [str(a)]) == str(a)
    assert output_inside_downloads(out, [str(a), str(b)]) is None


def test_config_problems_block_a_run(tmp_path):
    cfg = RFIDConfig()
    blocking, _ = cfg.problems()
    assert any("download folder" in p for p in blocking)
    assert any("metadata" in p.lower() for p in blocking)
    assert any("paddock" in p.lower() for p in blocking)
    cfg.timezone = "Mars/Olympus"
    assert any("timezone" in p for p in cfg.problems()[0])


def test_unknown_config_keys_are_ignored():
    cfg = RFIDConfig.from_dict({"name": "x", "bout_threshold_sec": 50, "num_zones": 8})
    assert cfg.name == "x" and cfg.bout_gap_s == 50.0
    assert isinstance(pd.Timestamp("2021-05-07"), pd.Timestamp)
