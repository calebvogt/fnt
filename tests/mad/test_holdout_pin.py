"""Two runs are compared on the same held-out recordings, or not at all.

The recording-level split is a seeded draw over the recordings that have
examples, sized by tile count. Any change to the label set -- a session of
rejections on five files -- re-draws it: the 2026-10-11 retrain held out a
different 17 recordings from the run the night before, so the two validation
scores measured different things. ``--holdout-from RUN`` pins the split to an
earlier run's held-out recordings.

Runs under pytest, or directly.
"""
import json
import os
import tempfile

import numpy as np
import pytest

from fnt.usv.usv_detector import cli
from fnt.usv.usv_detector.mad_training import _split_tiles


def _groups(spec):
    """``spec``: [(recording, n_tiles, n_positive_tiles)] -> groups, positives"""
    groups, pos = [], []
    for rec, n, npos in spec:
        for i in range(n):
            groups.append((rec, f"{rec}_{i}"))
            pos.append(i < npos)
    return groups, np.array(pos)


SPEC = [("a.wav", 10, 5), ("b.wav", 10, 5), ("c.wav", 10, 5),
        ("d.wav", 10, 5), ("e.wav", 10, 0)]


def test_a_pinned_split_holds_out_exactly_those_recordings():
    g, pos = _groups(SPEC)
    s = _split_tiles(g, len(g), 0.2, positives=pos, mode="file",
                     holdout=["b.wav", "e.wav"])
    assert s["pinned"] and s["val_held_out"] and s["split_level"] == "file"
    assert s["val_groups"] == ["b.wav", "e.wav"]
    held = {g[i][0] for i in s["val_idx"]}
    trained = {g[i][0] for i in s["train_idx"]}
    assert held == {"b.wav", "e.wav"} and not held & trained


def test_the_pin_survives_a_changed_label_set():
    """The reason it exists: adding a recording must not move the split."""
    g1, p1 = _groups(SPEC)
    g2, p2 = _groups(SPEC + [("f.wav", 30, 10)])
    a = _split_tiles(g1, len(g1), 0.2, positives=p1, mode="file",
                     holdout=["c.wav"])
    b = _split_tiles(g2, len(g2), 0.2, positives=p2, mode="file",
                     holdout=["c.wav"])
    assert a["val_groups"] == b["val_groups"] == ["c.wav"]


def test_paths_and_basenames_match():
    g, pos = _groups(SPEC)
    s = _split_tiles(g, len(g), 0.2, positives=pos, mode="file",
                     holdout=[r"Z:\data\T005\a.wav"])
    assert s["val_groups"] == ["a.wav"]


def test_a_pinned_recording_with_no_examples_is_reported():
    g, pos = _groups(SPEC)
    s = _split_tiles(g, len(g), 0.2, positives=pos, mode="file",
                     holdout=["a.wav", "gone.wav"])
    assert s["val_groups"] == ["a.wav"]
    assert s["holdout_missing"] == ["gone.wav"]


def test_an_unusable_pin_fails_instead_of_substituting_a_split():
    g, pos = _groups(SPEC)
    with pytest.raises(RuntimeError):
        _split_tiles(g, len(g), 0.2, positives=pos, mode="file",
                     holdout=["e.wav"])          # no calls held out
    with pytest.raises(RuntimeError):
        _split_tiles(g, len(g), 0.2, positives=pos, mode="file",
                     holdout=["gone.wav"])       # nothing held out at all


def test_no_pin_keeps_the_seeded_draw():
    g, pos = _groups(SPEC)
    s = _split_tiles(g, len(g), 0.2, positives=pos, mode="file")
    assert not s.get("pinned")


def _project_with_run(d, groups):
    run = os.path.join(d, "models", "20261010_184905_unet_n=2255")
    os.makedirs(run)
    with open(os.path.join(run, "training_summary.json"), "w") as fh:
        json.dump({"val_groups": groups}, fh)
    return run


def test_resolve_holdout_from_a_folder_a_name_or_a_summary():
    with tempfile.TemporaryDirectory() as d:
        run = _project_with_run(d, ["x.wav", "y.wav"])
        name = os.path.basename(run)
        assert cli.resolve_holdout(run) == ["x.wav", "y.wav"]
        assert cli.resolve_holdout(name, d) == ["x.wav", "y.wav"]
        assert cli.resolve_holdout(
            os.path.join(run, "training_summary.json")) == ["x.wav", "y.wav"]
        with pytest.raises(FileNotFoundError):
            cli.resolve_holdout("nope", d)


def test_the_flag_reaches_the_training_config():
    import argparse
    with tempfile.TemporaryDirectory() as d:
        run = _project_with_run(d, ["x.wav"])
        p = argparse.ArgumentParser()
        p.add_argument("--project", default=d)
        cli.add_training_args(p)
        kw = cli.training_overrides(p.parse_args(
            ["--holdout-from", os.path.basename(run)]))
        assert kw["holdout_groups"] == ["x.wav"]
        assert "holdout_groups" not in cli.training_overrides(p.parse_args([]))


def test_a_gui_launch_without_the_flag_leaves_the_split_alone():
    import fnt.usv.mad_pyqt as M
    a = M._launch_parser().parse_args(["--project", "p", "--train"])
    assert "holdout_groups" not in cli.training_overrides(a)


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
