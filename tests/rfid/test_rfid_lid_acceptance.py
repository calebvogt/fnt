"""Acceptance: the 2021 LID dataset, against an independent R reimplementation.

Runs only on a machine that has the dataset (skipped otherwise). Outputs go to
a temporary folder, never into the dataset. Expected numbers come from
``Claude_Analysis/2021_LID_WDOB_TerritoryInvasion_CLAUDE.Rmd``:

* 1,523,161 rows in 32 .xlsx downloads -> 1,404,510 unique reads
* 64 animals, 128 tags; 1,643 reads of unknown tags
* 1,670 reads by the other paddock's reader, kept as away visits
* lost tag 985.113004548427 (Calypso) from 2021-05-16 12:00 -> 47,705 dropped
* found dead, cut at last entry: George 9,887, Glaucus 3,879, Isac 75,693
* 65,473 bouts and ~1,439 animal-hours in zones

The R document counts 26 unknown tags because it groups by (trial, tag): the
reader test tag 989.002011236362 was read in both paddocks. There are 25
distinct tags, on 26 (tag, reader) pairs.
"""
import os
from pathlib import Path

import pytest


def _find_lid():
    """(folder holding the metadata, folder holding T001_Alpha), or (None, None).

    OneDrive keeps the dataset under the home folder on the Mac and on
    Windows; FNT_LID_2021 can point at its root instead. Both the older
    Analysis/RFID/Data layout and the current Data/RFID one are accepted.
    """
    roots = [os.environ.get("FNT_LID_2021", "")]
    for onedrive in ("Library/CloudStorage/OneDrive-UCB-O365", "OneDrive - UCB-O365"):
        roots.append(str(Path.home() / onedrive / "Work" / "Manuscripts"
                         / "2021_LID_WDOBTerritoryInvasion"))
    for root in filter(None, roots):
        for base, rfid in (("Data", "RFID"), ("Analysis", os.path.join("RFID", "Data"))):
            meta_dir = os.path.join(root, base)
            if os.path.isdir(os.path.join(meta_dir, rfid, "T001_Alpha")):
                return meta_dir, os.path.join(meta_dir, rfid)
    return None, None


LID, LID_RFID = _find_lid()

pytestmark = pytest.mark.skipif(LID is None, reason="2021 LID dataset not on this machine")


@pytest.fixture(scope="module")
def result(tmp_path_factory):
    from fnt.rfid.config import MetadataSpec, RFIDConfig, liddell_8_zone
    from fnt.rfid.core.metadata import (detect_header_row, guess_mapping,
                                        header_names, read_grid)
    from fnt.rfid.pipeline import run

    out = tmp_path_factory.mktemp("lid") / "LID_2021_FNT_analysis"
    cfg = RFIDConfig(name="LID_2021", output_dir=str(out),
                     download_dirs=[os.path.join(LID_RFID, "T001_Alpha"),
                                    os.path.join(LID_RFID, "T002_Bravo")],
                     timezone="America/New_York", day_origin="2021-05-07 12:00:00")
    cfg.arena = liddell_8_zone(2)
    cfg.arena.paddocks[0].name, cfg.arena.paddocks[0].trials = "Alpha", ["T001"]
    cfg.arena.paddocks[1].name, cfg.arena.paddocks[1].trials = "Bravo", ["T002"]
    meta = os.path.join(LID, "LID_2021_metadata.xlsx")
    grid = read_grid(meta, "metadata")
    header = detect_header_row(grid)
    assert header == 1                                   # row 2, below the notes
    columns, tags = guess_mapping(header_names(grid, header))
    cfg.metadata = MetadataSpec(path=meta, sheet="metadata", columns=columns,
                                tag_columns=tags)
    cfg.animals["Calypso"] = {"lost_tags": [{"tag": "985.113004548427",
                                             "start": "2021-05-16 12:00:00"}]}
    for animal in ("George", "Glaucus", "Isac"):
        cfg.animals[animal] = {"dead_mode": "last_entry"}
    res = run(cfg)
    assert res.ok, res.error
    return res


def test_reads_and_identity(result):
    s = result.stats
    assert (s["files_used"], s["raw_rows"], s["unique_reads"]) == (32, 1_523_161, 1_404_510)
    assert (s["animals_in_metadata"], s["tags_in_metadata"]) == (64, 128)
    assert s["dropped"]["unknown_tag"] == 1_643
    unknown = result.tables["qc_unknown_tags"]
    assert len(unknown) == 25
    assert sum(len(r.split(",")) for r in unknown["readers"]) == 26
    assert s["dropped"]["unmapped_antenna"] == 0


def test_visits_to_the_other_paddock(result):
    cp = result.tables["qc_cross_paddock"].set_index("animal")["reads"].to_dict()
    assert cp == {"Circe": 927, "Joan": 219, "Basie": 522, "Jackson": 1, "Charlie": 1}
    assert result.stats["away_reads"] == 1_670
    zones = result.tables["qc_cross_paddock"].set_index("animal")["zone_ids"]
    assert zones["Circe"].startswith("B") and zones["Basie"].startswith("A")


def test_lost_tag_and_found_dead_cuts(result):
    d = result.stats["dropped"]
    assert d["lost_tag"] == 47_705
    fd = result.tables["qc_found_dead"].set_index("animal")
    assert fd["reads_dropped"].to_dict() == {"George": 9_887, "Glaucus": 3_879,
                                             "Isac": 75_693}
    last = fd["last_entry"].dt.strftime("%m/%d %H:%M").to_dict()
    assert last == {"George": "05/15 20:47", "Glaucus": "05/15 11:29",
                    "Isac": "05/16 02:18"}
    assert d["found_dead"] == 9_887 + 3_879 + 75_693


def test_bouts_and_time_in_zones(result):
    assert result.stats["bouts"] == 65_473
    assert abs(result.stats["animal_hours_in_zones"] - 1_439) < 1


def test_qc_finds_the_known_problems(result):
    tp = result.tables["qc_two_places"]
    assert set(tp.loc[tp["animal"] == "Calypso", "day"]) >= {10, 11, 12}
    st = result.tables["qc_stationary_tags"]
    bodies = set(st.loc[st["status"].str.startswith("no entries"), "animal"])
    assert {"Isac", "George"} <= bodies
    roster = result.tables["qc_animals"].set_index("animal")
    assert set(roster.index[roster["never_read"]]) == {"Bastet", "Eris", "Jackie"}


def test_outputs_went_to_the_temp_folder(result):
    assert not result.output_dir.startswith(LID)
    assert os.path.exists(os.path.join(result.output_dir, "fnt_config.json"))
