"""``mad analyze --project`` runs a whole project headlessly, like the GUI.

So a run can be dictated and left: every registered recording, logged to the
project's batch_runs (where the GUI's run summary looks), labels shielded from
the project's own store, and each recording's harmonics grouped as it lands.
"""
import json
import os
import tempfile

import numpy as np
import pytest
from scipy.io import wavfile

from fnt.usv.usv_detector import cli
from fnt.usv.usv_detector.mad_project import (
    MADProjectConfig, create_mad_project)
from fnt.usv.usv_detector.mad_registry import RegisteredFile


def project(tmp, n=3, missing=1):
    proj = os.path.join(tmp, "p")
    create_mad_project(proj)
    wavs = []
    for i in range(n):
        w = os.path.join(tmp, f"r{i}.wav")
        wavfile.write(w, 250_000, np.zeros(2500, np.int16))
        wavs.append(w)
    cfg = MADProjectConfig.load(proj)
    entries = [RegisteredFile.from_path(w) for w in wavs]
    for k in range(missing):
        entries.append(RegisteredFile(path=os.path.join(tmp, f"gone{k}.wav"),
                                      basename=f"gone{k}.wav"))
    cfg.set_audio_entries(entries)
    cfg.save()
    return proj, wavs


@pytest.fixture
def ran(monkeypatch):
    """Run analyze with inference and grouping replaced by recorders."""
    import fnt.usv.usv_detector.mad_inference as mi
    calls = {"wavs": None, "cfg": None, "grouped": []}

    def fake_run(cfg, wav_paths, on_file_done=None, **kw):
        calls["wavs"], calls["cfg"] = list(wav_paths), cfg
        out = []
        for w in wav_paths:
            s = {"wav_path": w, "n_blobs": 1, "timing": {}}
            if on_file_done:
                on_file_done(s)
            out.append(s)
        return out
    monkeypatch.setattr(mi, "run_inference_on_files", fake_run)
    monkeypatch.setattr(cli, "_group_harmonics_for",
                        lambda w: calls["grouped"].append(w))
    return calls


def test_every_registered_recording_is_analyzed(ran):
    with tempfile.TemporaryDirectory() as tmp:
        proj, wavs = project(tmp)
        rc = cli.main(["analyze", "--model", os.path.join(tmp, "m.pt"),
                       "--project", proj])
        assert rc == 0
        assert sorted(ran["wavs"]) == sorted(wavs), "a missing file is skipped"


def test_the_run_logs_where_the_gui_looks(ran):
    with tempfile.TemporaryDirectory() as tmp:
        proj, _ = project(tmp)
        cli.main(["analyze", "--model", os.path.join(tmp, "m.pt"),
                  "--project", proj])
        runs = os.listdir(os.path.join(proj, "batch_runs"))
        assert len(runs) == 1
        info = json.load(open(os.path.join(proj, "batch_runs", runs[0],
                                           "run_info.json")))
        assert info["n_files"] == 3


def test_labels_are_shielded_from_the_projects_store(ran):
    with tempfile.TemporaryDirectory() as tmp:
        proj, _ = project(tmp)
        cli.main(["analyze", "--model", os.path.join(tmp, "m.pt"),
                  "--project", proj])
        assert ran["cfg"].training_data_dir == \
            MADProjectConfig.load(proj).training_data_dir
        assert ran["cfg"].preserve_labels


def test_harmonics_are_grouped_per_file_unless_turned_off(ran):
    with tempfile.TemporaryDirectory() as tmp:
        proj, wavs = project(tmp)
        cli.main(["analyze", "--model", os.path.join(tmp, "m.pt"),
                  "--project", proj])
        assert sorted(ran["grouped"]) == sorted(wavs)
    ran["grouped"].clear()
    with tempfile.TemporaryDirectory() as tmp:
        proj, _ = project(tmp)
        cli.main(["analyze", "--model", os.path.join(tmp, "m.pt"),
                  "--project", proj, "--no-group-harmonics"])
        assert ran["grouped"] == []


def test_a_model_run_folder_stands_for_its_weights(ran):
    with tempfile.TemporaryDirectory() as tmp:
        proj, _ = project(tmp)
        run = os.path.join(tmp, "20261009_unet")
        os.makedirs(run)
        cli.main(["analyze", "--model", run, "--project", proj])
        assert ran["cfg"].model_path == os.path.join(run, "weights.pt")
