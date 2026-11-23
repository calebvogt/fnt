"""Reopening a finished trial: the run archive round-trips into the run view.

The claim being tested is that a record written by a run is enough on its own
to replay that run — positions, drives, condition and identity — without the
simulation, the config, or anything else from the session that produced it.
"""
from __future__ import annotations

import os

import numpy as np
import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtWidgets import QApplication          # noqa: E402

from fnt.abma.core.presets import vole_anosmia    # noqa: E402
from fnt.abma.core.record import RunRecord        # noqa: E402
from fnt.abma.core.runner import run_experiment   # noqa: E402
from fnt.abma.gui.science_panel import SciencePanel  # noqa: E402


@pytest.fixture(scope="module")
def finished_run(tmp_path_factory):
    """A short real run, so the archive under test is one the engine wrote."""
    cfg = vole_anosmia()
    cfg.days = 0.02
    cfg.n_trials = 1
    cfg.record_interval = 30.0
    project = tmp_path_factory.mktemp("run")
    run_experiment(cfg, str(project))
    return project


def test_a_run_writes_a_record_beside_its_csvs(finished_run):
    data = finished_run / "data"
    assert (data / "record_S001.npz").exists()
    assert (data / "uwb_S001_processed.csv").exists()   # the FNT contract
    assert (finished_run / "provenance.json").exists()


def test_the_record_carries_identity_and_drives(finished_run):
    rec = RunRecord.load(str(finished_run / "data" / "record_S001.npz"))
    assert len(rec) > 2
    assert rec.n_agents == 8
    assert {a["sexid"] for a in rec.agents}
    # scent marking is on in this preset, so these drives must be populated
    assert rec.series(0, "drive_wander").max() > 0
    assert rec.series(0, "drive_social").max() >= 0


def test_view_frame_rebuilds_what_the_arena_needs(finished_run):
    rec = RunRecord.load(str(finished_run / "data" / "record_S001.npz"))
    fr = rec.view_frame(1)
    for key in ("x", "y", "heading", "sex_m", "color", "size", "shape",
                "alive", "health"):
        assert key in fr, f"replay frame is missing {key}"
    assert np.asarray(fr["color"]).shape == (rec.n_agents, 4)
    assert set(np.unique(fr["sex_m"])) <= {0.0, 1.0}


def test_a_replayed_frame_drives_the_science_panel(finished_run):
    """The point of the whole archive: pick an animal from last week's run."""
    app = QApplication.instance() or QApplication([])
    rec = RunRecord.load(str(finished_run / "data" / "record_S001.npz"))
    panel = SciencePanel()
    panel.set_population(rec.agents)
    panel.select(0)
    for i in range(len(rec)):
        panel.update_frame(rec.view_frame(i))
    assert len(panel.roster.cards) == rec.n_agents
    assert panel.drives.values, "no drives recovered from the archive"
    assert len(panel.pages[0].plot.history["drive_wander"]) == len(rec)
    assert app is not None


def test_provenance_matches_the_config_that_produced_the_run(finished_run):
    from fnt.abma.core.config import ExperimentConfig
    from fnt.abma.core.provenance import verify

    cfg = ExperimentConfig.from_json(str(finished_run / "config.json"))
    ok, message = verify(str(finished_run), cfg)
    assert ok, message

    cfg.days = 99.0            # not the config that produced this data
    ok, message = verify(str(finished_run), cfg)
    assert not ok and "does not match" in message
