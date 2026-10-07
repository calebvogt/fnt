"""End to end: a small synthetic dataset through the pipeline, the worker thread,
and the GUI (offscreen).

The dataset has two paddocks side by side like LID: reader 1 in A (home of
T001), reader 2 in B (home of T002), a .txt twin that is truncated, a stray
non-download CSV, a test tag, and one visit through the shared wall.
"""
import json
import os
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd
import pytest

from _rfid_synth import (TAGS, lid_like_arena, record, write_biomark_xlsx, write_txt,
                         write_metadata_xlsx)
from fnt.rfid.config import MetadataSpec, RFIDConfig
from fnt.rfid.core.metadata import detect_header_row, guess_mapping, header_names, read_grid
from fnt.rfid.pipeline import run

NIGHT = pd.Timestamp("2021-05-08 21:00:00")        # day 2


def dataset(root):
    """Downloads + metadata on disk; returns the config for them."""
    a, b = root / "T001_Alpha", root / "T002_Bravo"
    a.mkdir()
    b.mkdir()
    s = lambda sec: NIGHT + pd.Timedelta(seconds=sec)          # noqa: E731
    alpha = ([record(s(i * 10), 1, 3, TAGS["M1a"]) for i in range(30)]      # Mars in A3
             + [record(s(i * 10 + 5), 1, 11, TAGS["F1a"]) for i in range(20)]  # Flora A3 floor
             + [record(s(4000), 1, 1, TAGS["test"])])                        # test tag
    bravo = ([record(s(i * 10), 2, 5, TAGS["M2a"]) for i in range(40)]      # Duke in B5
             + [record(s(1000), 2, 6, TAGS["M1b"])])                          # Mars visits B6
    write_biomark_xlsx(a / "DL_1.xlsx", alpha)
    write_txt(a / "DL_1.txt", alpha[:25], truncate_last=True)
    write_biomark_xlsx(b / "DL_1.xlsx", bravo)
    pd.DataFrame({"trial": ["T001"], "zone": [3]}).to_csv(a / "OLD_GBI.csv", index=False)
    meta = write_metadata_xlsx(root / "meta.xlsx")
    grid = read_grid(meta, "metadata")
    columns, tags = guess_mapping(header_names(grid, detect_header_row(grid)))
    cfg = RFIDConfig(name="SYN", download_dirs=[str(a), str(b)],
                     day_origin="2021-05-07 12:00:00", timezone="America/New_York")
    cfg.metadata = MetadataSpec(path=meta, sheet="metadata", columns=columns,
                                tag_columns=tags)
    cfg.arena = lid_like_arena()
    return cfg


def listing(folder):
    return sorted((f, os.path.getsize(os.path.join(folder, f)), os.path.getmtime(
        os.path.join(folder, f))) for f in os.listdir(folder))


def test_pipeline_writes_an_analysis_folder_beside_the_downloads(tmp_path):
    cfg = dataset(tmp_path)
    before = {d: listing(d) for d in cfg.download_dirs}
    res = run(cfg)
    assert res.ok, res.error
    out = tmp_path / "SYN_FNT_analysis"
    assert res.output_dir == str(out)
    for d in cfg.download_dirs:                              # raw folders untouched
        assert listing(d) == before[d]
    csvs = set(os.listdir(out / "csvs"))
    for table in ("reads", "bouts", "animal_days", "zone_holders", "co_occupancy",
                  "qc_files", "qc_animals", "qc_unknown_tags", "qc_cross_paddock",
                  "qc_two_places", "qc_stationary_tags", "qc_dropped_reads"):
        assert f"SYN_{table}.csv" in csvs
    assert (out / "SYN_messageLog.txt").exists()
    record_ = json.loads((out / "fnt_config.json").read_text())
    assert record_["config"]["name"] == "SYN"
    prov = record_["provenance"]
    assert prov["fnt_version"] and prov["run_time"]
    assert {os.path.basename(f["path"]) for f in prov["input_files"]} >= {"DL_1.xlsx", "DL_1.txt"}
    assert all("size_bytes" in f and "modified" in f for f in prov["input_files"])
    assert record_["run_stats"]["bouts"] == res.stats["bouts"]

    s = res.stats
    assert s["unique_reads"] == 92 and s["dropped"]["unknown_tag"] == 1
    assert s["away_reads"] == 1
    files = res.tables["qc_files"].set_index("file")
    assert files.loc["DL_1.txt", "status"] == "twin (not used)"
    assert files.loc["OLD_GBI.csv", "status"] == "not a download"
    bouts = res.tables["bouts"]
    assert set(bouts["zone_id"]) == {"A3", "B5", "B6"}
    assert bouts.loc[bouts["zone_id"] == "B6", "away"].all()
    # Mars and Flora share A3 for ~195 s
    co = res.tables["co_occupancy"]
    assert co.set_index(["animal_a", "animal_b"]).loc[("Flora", "Mars"), "seconds"] > 150
    reads_csv = pd.read_csv(out / "csvs" / "SYN_reads.csv", dtype={"tag": str}, nrows=2)
    assert list(reads_csv.columns) == ["animal", "tag", "reader", "paddock", "antenna",
                                       "location", "zone", "zone_id", "away", "day", "t"]
    assert reads_csv["tag"].iloc[0].startswith("985.1130045")


def test_a_config_that_cannot_run_writes_nothing(tmp_path):
    cfg = dataset(tmp_path)
    cfg.output_dir = os.path.join(cfg.download_dirs[0], "inside")
    res = run(cfg)
    assert res.error and "inside the download folder" in res.error
    assert not os.path.exists(cfg.output_dir)


def test_check_only_writes_nothing(tmp_path):
    cfg = dataset(tmp_path)
    res = run(cfg, write=False)
    assert res.ok and not res.written
    assert not (tmp_path / "SYN_FNT_analysis").exists()


def test_cancel_stops_before_writing(tmp_path):
    cfg = dataset(tmp_path)
    res = run(cfg, cancelled=lambda: True)
    assert res.cancelled and not res.written


# ------------------------------------------------------------------ Qt ----

@pytest.fixture(scope="module")
def app():
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


def wait_for(app, predicate, timeout=60):
    t0 = time.time()
    while not predicate() and time.time() - t0 < timeout:
        app.processEvents()
        time.sleep(0.02)
    app.processEvents()


def test_worker_runs_the_pipeline_in_a_thread(app, tmp_path):
    from fnt.rfid.rfid_worker import RFIDWorker
    cfg = dataset(tmp_path)
    worker = RFIDWorker(cfg, write=True)
    seen, done = [], []
    worker.progress.connect(lambda p, m: seen.append(p))
    worker.finished_run.connect(done.append)
    worker.start()
    wait_for(app, lambda: bool(done))
    worker.wait(5000)
    assert done and done[0].ok, done and done[0].error
    assert max(seen) == 100
    assert (tmp_path / "SYN_FNT_analysis" / "fnt_config.json").exists()


def test_window_opens_offscreen_and_runs_a_check(app, tmp_path):
    from fnt.rfid.rfid_preprocessing_pyqt import RESULT_TABS, RFIDPreprocessingWindow
    w = RFIDPreprocessingWindow()
    w.show()
    app.processEvents()
    assert w.windowTitle() == "RFID PreProcessing Tool"
    cfg = dataset(tmp_path)
    w.apply_config(cfg)
    assert "4 animals" in w.lbl_animals.text()
    assert "A (Alpha): reader 1" in w.lbl_arena.text()
    w.start_run(write=False)
    wait_for(app, lambda: w.worker is None)
    assert w.last_result is not None and w.last_result.ok, w.last_result.error
    assert w.models["bouts"].rowCount() == len(w.last_result.tables["bouts"])
    assert w.tabs.count() == 1 + len(RESULT_TABS)
    assert not (tmp_path / "SYN_FNT_analysis").exists()      # a check writes nothing

    # the QC shortcut adds a lost-tag interval from the start of the day
    w.mark_lost_tag("Flora", TAGS["F1a"], 2)
    lost = w.config.animals["Flora"]["lost_tags"][0]
    assert lost["tag"] == TAGS["F1a"] and lost["start"] == "2021-05-08 12:00:00"
    w.mark_found_dead("Duke")
    assert w.config.animals["Duke"]["dead_mode"] == "last_entry"

    # the collected config round-trips through the widgets unchanged
    before = w.collect_config().to_dict()
    w.apply_config(w.collect_config())
    assert w.collect_config().to_dict() == before
    w.close()


def test_dialogs_build_from_a_config(app, tmp_path):
    from fnt.rfid.core.metadata import load_animals
    from fnt.rfid.rfid_dialogs import (ArenaDialog, IdentityDialog, LostTagsDialog,
                                       MetadataMappingDialog)
    cfg = dataset(tmp_path)
    cfg.animals["Isac"] = {"dead_mode": "last_entry",
                           "lost_tags": [{"tag": TAGS["M3a"], "start": "2021-05-16 12:00:00"}]}
    arena = ArenaDialog(cfg.arena).arena()
    assert arena.to_dict() == cfg.arena.to_dict()
    spec = MetadataMappingDialog(cfg.metadata).result_spec()
    assert spec.columns == cfg.metadata.columns and spec.tag_columns == ["tag_1", "tag_2"]
    table = load_animals(cfg.metadata)
    settings = IdentityDialog(table.animals, cfg.animals).result_settings()
    assert settings["Isac"]["dead_mode"] == "last_entry"
    assert settings["Isac"]["lost_tags"][0]["tag"] == TAGS["M3a"]
    assert settings["Mars"]["start_mode"] == "auto"
    lost = LostTagsDialog("Isac", [TAGS["M3a"]], settings["Isac"]["lost_tags"]).intervals()
    assert lost[0]["start"].startswith("2021-05-16 12:00:00") and lost[0]["stop"] == ""
