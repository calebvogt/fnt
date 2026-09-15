"""Inference can target several folder trees at once.

``getExistingDirectory`` returns exactly one directory, so a run over fourteen
trial folders meant fourteen trips through the picker — or pointing at the
parent and taking every subfolder under it, wanted or not. Add Folder already
had a hand-built multi-select dialog for this; inference now uses the same one.

Each selection is walked to the bottom, so a trial folder brings in its
per-channel subfolders without being named. Selections are de-duplicated:
picking a folder and something inside it is easy to do with a multi-select, and
analysing one recording twice writes its detections twice.
"""
import os

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QCheckBox, QLabel  # noqa: E402

from fnt.usv.mad_pyqt import MADMainWindow  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _tree(root, spec):
    """spec: {relative dir: [wav names]}"""
    for rel, names in spec.items():
        d = root / rel
        d.mkdir(parents=True, exist_ok=True)
        for n in names:
            (d / n).write_bytes(b"RIFF")
    return root


@pytest.fixture
def win(qapp):
    class W:
        _pick_inference_folder = MADMainWindow._pick_inference_folder

        def __init__(self, picks):
            self._picks = picks
            self._infer_folder = None
            self._infer_folders = []
            self._infer_folder_wavs = []
            self.chk_scope_folder = QCheckBox()
            self.lbl_infer_folder = QLabel()
            self.logged = []

        def _pick_folders(self):
            return list(self._picks)

        def _log(self, m):
            self.logged.append(m)

        def _update_run_button(self):
            pass

    return W


# ------------------------------------------------------- multi-select
def test_several_folders_are_combined(win, tmp_path):
    _tree(tmp_path, {"T01": ["a.wav"], "T02": ["b.wav", "c.wav"]})
    w = win([str(tmp_path / "T01"), str(tmp_path / "T02")])
    w._pick_inference_folder()
    assert len(w._infer_folder_wavs) == 3
    assert w.chk_scope_folder.isChecked() is True
    assert "3 .wav file(s) under 2 folders" in w.lbl_infer_folder.text()


def test_each_folder_is_swept_to_the_bottom(win, tmp_path):
    """A trial folder should bring in its per-channel subfolders unnamed."""
    _tree(tmp_path, {
        "T01/ch1": ["a.wav"], "T01/ch2": ["b.wav"],
        "T01/ch3/deeper": ["c.wav"],
    })
    w = win([str(tmp_path / "T01")])
    w._pick_inference_folder()
    assert len(w._infer_folder_wavs) == 3


def test_overlapping_selections_do_not_double_count(win, tmp_path):
    """Selecting a parent and its child is the easy multi-select mistake, and
    analysing a recording twice writes its detections twice."""
    _tree(tmp_path, {"T01/ch1": ["a.wav"]})
    w = win([str(tmp_path / "T01"), str(tmp_path / "T01" / "ch1")])
    w._pick_inference_folder()
    assert len(w._infer_folder_wavs) == 1


def test_a_single_folder_still_names_it(win, tmp_path):
    _tree(tmp_path, {"T01": ["a.wav"]})
    target = str(tmp_path / "T01")
    w = win([target])
    w._pick_inference_folder()
    assert target in w.lbl_infer_folder.text()
    assert "folders" not in w.lbl_infer_folder.text()


def test_every_chosen_folder_is_listed_in_the_tooltip(win, tmp_path):
    """The label can only say '2 folders'; the tooltip says which."""
    _tree(tmp_path, {"T01": ["a.wav"], "T02": ["b.wav"]})
    picks = [str(tmp_path / "T01"), str(tmp_path / "T02")]
    w = win(picks)
    w._pick_inference_folder()
    for p in picks:
        assert p in w.lbl_infer_folder.toolTip()


# ---------------------------------------------------------- edge cases
def test_cancelling_changes_nothing(win, tmp_path):
    w = win([])
    w._infer_folder_wavs = ['previous.wav']
    w._pick_inference_folder()
    assert w._infer_folder_wavs == ['previous.wav']
    assert w._infer_folder is None


def test_folders_with_no_wavs_are_reported(win, tmp_path):
    (tmp_path / "empty").mkdir()
    w = win([str(tmp_path / "empty")])
    w._pick_inference_folder()
    assert w._infer_folder_wavs == []
    assert w.chk_scope_folder.isChecked() is False
    assert "No .wav files found" in w.lbl_infer_folder.text()


def test_a_representative_root_is_kept_for_the_run_record(win, tmp_path):
    """Without a project the run record is written under a folder, and only
    one path can serve as that root."""
    _tree(tmp_path, {"T01": ["a.wav"], "T02": ["b.wav"]})
    picks = [str(tmp_path / "T01"), str(tmp_path / "T02")]
    w = win(picks)
    w._pick_inference_folder()
    assert w._infer_folder == picks[0]
    assert w._infer_folders == picks
    assert os.path.isdir(w._infer_folder)


def test_the_button_offers_more_than_one(win):
    import inspect
    src = inspect.getsource(MADMainWindow)
    assert 'QPushButton("Choose Folder(s)…")' in src


def test_it_uses_the_shared_multiselect_dialog():
    """Not getExistingDirectory, which can only return one directory."""
    import inspect
    src = inspect.getsource(MADMainWindow._pick_inference_folder)
    assert "_pick_folders()" in src
    # Strip the docstring and comments: they explain what was replaced and
    # would otherwise read as the replaced call still being there.
    body = src.split('"""')[-1]
    code = [ln for ln in body.splitlines() if not ln.strip().startswith("#")]
    assert not any("getExistingDirectory" in ln for ln in code)
