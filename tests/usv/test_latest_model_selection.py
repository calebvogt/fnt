"""The auto-selected model must be the newest one, not the last alphabetically.

Model run directories are named ``<timestamp>_<arch>_n=<labels>``, so sorting
by name usually matches sorting by date — until a run is given a ``--run-name``.
Then letters land after digits and a one-off directory like
``agent_headless_test`` sorts last, which the combo treated as "newest".

The consequence was not cosmetic: Inference auto-selected a checkpoint a week
out of date, and the evaluation that runs against the selected model scored the
wrong one.

``_latest_model_path`` had already been fixed for exactly this; the combo's own
index chooser had not, and kept its own copy of the broken assumption.
"""
import os

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QComboBox  # noqa: E402

from fnt.usv.mad_pyqt import MADMainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _make_models(tmp_path, specs):
    """specs: [(dirname, mtime_offset_seconds)] -> project dir."""
    proj = tmp_path / "proj"
    models = proj / "models"
    models.mkdir(parents=True)
    for name, off in specs:
        d = models / name
        d.mkdir()
        w = d / "weights.pt"
        w.write_bytes(b"x")
        os.utime(w, (1_700_000_000 + off, 1_700_000_000 + off))
    return proj


@pytest.fixture
def win(qapp):
    class Proj:
        def __init__(self, d):
            self.project_dir = str(d)

    class W:
        _latest_project_model_index = MADMainWindow._latest_project_model_index
        _latest_model_path = MADMainWindow._latest_model_path

        def __init__(self, proj_dir):
            self._project = Proj(proj_dir)
            self.combo_deploy_model = QComboBox()

        def fill(self):
            """Populate the way the real code does: alphabetical by name."""
            root = os.path.join(self._project.project_dir, 'models')
            for name in sorted(os.listdir(root)):
                w = os.path.join(root, name, 'weights.pt')
                if os.path.isfile(w):
                    self.combo_deploy_model.addItem(name, w)
            return self

    return W


def test_a_run_named_directory_does_not_outrank_the_newest(win, tmp_path):
    """The reported bug, exactly."""
    proj = _make_models(tmp_path, [
        ("20260909_190553_unet_n=171", 0),
        ("20260913_150205_unet_n=413", 400),      # newest weights
        ("agent_headless_test", 100),             # sorts last by NAME
    ])
    w = win(proj).fill()
    assert w.combo_deploy_model.itemText(
        w.combo_deploy_model.count() - 1) == "agent_headless_test"
    i = w._latest_project_model_index()
    assert w.combo_deploy_model.itemText(i) == "20260913_150205_unet_n=413"


def test_timestamped_dirs_alone_still_pick_the_newest(win, tmp_path):
    proj = _make_models(tmp_path, [
        ("20260909_190553_unet_n=171", 0),
        ("20260911_091753_unet_n=289", 200),
        ("20260913_150205_unet_n=413", 400),
    ])
    w = win(proj).fill()
    i = w._latest_project_model_index()
    assert w.combo_deploy_model.itemText(i) == "20260913_150205_unet_n=413"


def test_a_newer_run_named_model_is_still_chosen(win, tmp_path):
    """The rule is recency, not a grudge against named runs."""
    proj = _make_models(tmp_path, [
        ("20260913_150205_unet_n=413", 0),
        ("my_best_model", 999),
    ])
    w = win(proj).fill()
    i = w._latest_project_model_index()
    assert w.combo_deploy_model.itemText(i) == "my_best_model"


def test_no_models_selects_nothing(win, tmp_path):
    proj = tmp_path / "proj"
    (proj / "models").mkdir(parents=True)
    w = win(proj).fill()
    assert w._latest_project_model_index() is None


def test_no_project_selects_nothing(win, tmp_path):
    proj = _make_models(tmp_path, [("20260913_150205_unet_n=413", 0)])
    w = win(proj).fill()
    w._project = None
    assert w._latest_project_model_index() is None


def test_a_model_from_another_project_is_not_chosen(win, tmp_path):
    """The combo can list models loaded from elsewhere; 'latest in THIS
    project' must not drift onto one of them."""
    proj = _make_models(tmp_path, [("20260913_150205_unet_n=413", 0)])
    other = tmp_path / "other" / "models" / "20270101_000000_unet_n=9"
    other.mkdir(parents=True)
    (other / "weights.pt").write_bytes(b"x")
    os.utime(other / "weights.pt", (1_800_000_000, 1_800_000_000))
    w = win(proj).fill()
    w.combo_deploy_model.addItem("[other] newer", str(other / "weights.pt"))
    i = w._latest_project_model_index()
    assert w.combo_deploy_model.itemText(i) == "20260913_150205_unet_n=413"


def test_a_directory_without_weights_is_ignored(win, tmp_path):
    proj = _make_models(tmp_path, [("20260913_150205_unet_n=413", 0)])
    (proj / "models" / "training_data").mkdir()
    w = win(proj).fill()
    i = w._latest_project_model_index()
    assert w.combo_deploy_model.itemText(i) == "20260913_150205_unet_n=413"


# ----------------------------------- the startup path, not just Refresh
"""``select_latest`` is only passed by the Refresh button. On project open the
combo falls through to a default, and that default was "last row" — which, in
an alphabetically filled list, is a run-named directory. So the fix to
``_latest_project_model_index`` never ran where it mattered most: the model
sitting selected when an 11,000-file batch is launched.
"""


def test_the_startup_fallback_prefers_the_newest_not_the_last_row():
    import inspect
    src = inspect.getsource(MADMainWindow._refresh_deploy_models)
    code = [ln for ln in src.splitlines() if not ln.strip().startswith("#")]
    joined = "\n".join(code)
    i_latest = joined.index("target = self._latest_project_model_index()\n"
                            "        if target is None and found:")
    i_last = joined.index("target = self.combo_deploy_model.count() - 1")
    assert i_latest < i_last, "newest must be tried before last-row"


def test_refresh_still_selects_latest():
    import inspect
    src = inspect.getsource(MADMainWindow)
    assert "self._refresh_deploy_models(select_latest=True)" in src
