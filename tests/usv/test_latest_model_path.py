"""Which trained model counts as "the latest".

Run directories are usually timestamped, so sorting them by name looked
equivalent to sorting by date — until a run carried a ``--run-name``. Then the
sort is lexicographic, letters land after digits, and a directory called
``agent_headless_test`` outranks ``20260906_235419_unet_n=115``. A real project
spent a day prefilling its training settings from a throwaway test run because
of it, and the same call picks the model for inference.
"""
import os
import time

import pytest

pytest.importorskip("PyQt5")


@pytest.fixture
def latest(tmp_path):
    from PyQt5.QtWidgets import QApplication
    if QApplication.instance() is None:
        QApplication([])
    from fnt.usv.mad_pyqt import MADMainWindow

    class Proj:
        project_dir = str(tmp_path)

    class W:
        _project = Proj()
        _latest_model_path = MADMainWindow._latest_model_path

    return W(), tmp_path


def _run(root, name, when=None):
    """Create a run directory with weights, optionally back-dated."""
    d = root / "models" / name
    d.mkdir(parents=True, exist_ok=True)
    w = d / "weights.pt"
    w.write_bytes(b"x")
    if when is not None:
        os.utime(w, (when, when))
    return str(w)


def test_a_named_run_does_not_outrank_a_newer_timestamped_one(latest):
    """The exact failure: 'a' sorts after '2'."""
    win, root = latest
    now = time.time()
    _run(root, "agent_headless_test", now - 3600)
    newest = _run(root, "20260906_235419_unet_n=115", now)
    assert win._latest_model_path() == newest


def test_the_newest_wins_regardless_of_name_order(latest):
    win, root = latest
    now = time.time()
    _run(root, "zzz_last_alphabetically", now - 100)
    newest = _run(root, "aaa_first_alphabetically", now)
    assert win._latest_model_path() == newest


def test_timestamped_runs_still_order_correctly(latest):
    win, root = latest
    now = time.time()
    _run(root, "20260903_214659_unet_n=24", now - 7200)
    _run(root, "20260905_212613_unet_n=114", now - 3600)
    newest = _run(root, "20260906_235419_unet_n=115", now)
    assert win._latest_model_path() == newest


def test_a_directory_without_weights_is_ignored(latest):
    win, root = latest
    now = time.time()
    good = _run(root, "finished_run", now - 60)
    (root / "models" / "crashed_run").mkdir(parents=True)   # no weights.pt
    assert win._latest_model_path() == good


def test_no_models_yet_returns_none(latest):
    win, root = latest
    (root / "models").mkdir(parents=True)
    assert win._latest_model_path() is None


def test_no_models_directory_returns_none(latest):
    win, _root = latest
    assert win._latest_model_path() is None


def test_no_project_returns_none(latest):
    win, _root = latest
    win._project = None
    assert win._latest_model_path() is None


def test_ties_are_broken_deterministically(latest):
    """Two runs sharing a timestamp must not flip between calls."""
    win, root = latest
    now = time.time()
    _run(root, "run_a", now)
    _run(root, "run_b", now)
    assert win._latest_model_path() == win._latest_model_path()
    assert win._latest_model_path().endswith(os.path.join("run_b", "weights.pt"))
