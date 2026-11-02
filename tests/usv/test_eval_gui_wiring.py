"""The GUI side of region-scoped evaluation.

Three things worth pinning, because each is a place the meaning could quietly
drift from what the numbers actually support:

* the eval worker must route to the region-scoped function and persist the
  result beside the checkpoint, or the trend has nothing to read;
* the review-complete flag must be readable from a cache rather than from
  disk during painting, because the Audio list is on a network share and a
  per-row open is what made opening a project take 27 s; and
* the summary must stay silent when it has nothing to say.
"""
import os

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


# ------------------------------------------------------------- worker
class _Cfg:
    model_path = ""
    device = "cpu"
    tile_freq_bins = 512
    tile_time_frames = 256
    tile_overlap_fraction = 0.25
    batch_size = 8
    amp = False
    chunk_frames = 100000
    min_blob_pixels = 100
    nperseg = noverlap = nfft = db_min = db_max = db_norm = None


@pytest.fixture
def worker_env(qapp, tmp_path, monkeypatch):
    from fnt.usv.usv_detector import mad_eval as ME
    from fnt.usv.mad_pyqt import MADEvalWorker

    run_dir = tmp_path / "20260101_010101_unet_n=5"
    run_dir.mkdir()
    cfg = _Cfg()
    cfg.model_path = str(run_dir / "weights.pt")

    called = {}

    def fake_regions(wavs, c, **kw):
        called['fn'] = 'reviewed'
        return ME.EvalResult(thresholds=[0.5], scope='reviewed',
                             per_threshold=[{'threshold': 0.5, 'f1': 0.7,
                                             'precision': 0.7, 'recall': 0.7,
                                             'tp': 7, 'fp': 3, 'fn': 3,
                                             'n_pred': 10}],
                             n_labels=5, n_files=1)

    def fake_whole(wavs, c, **kw):
        called['fn'] = 'file'
        return ME.EvalResult(thresholds=[0.5], scope='file')

    monkeypatch.setattr(ME, 'evaluate_labeled_regions', fake_regions)
    monkeypatch.setattr(ME, 'evaluate_wavs', fake_whole)
    return MADEvalWorker, cfg, run_dir, called


def test_the_worker_routes_to_the_region_scoped_function(worker_env):
    W, cfg, _, called = worker_env
    w = W(cfg, ["a.wav"], scope='reviewed', save=False)
    w.run()
    assert called['fn'] == 'reviewed'


def test_whole_file_scope_still_reachable(worker_env):
    W, cfg, _, called = worker_env
    w = W(cfg, ["a.wav"], scope='file', save=False)
    w.run()
    assert called['fn'] == 'file'


def test_the_result_is_saved_beside_the_checkpoint(worker_env):
    """Without this the trend has nothing to read."""
    W, cfg, run_dir, _ = worker_env
    W(cfg, ["a.wav"], scope='reviewed', save=True).run()
    assert (run_dir / "eval.json").is_file()
    from fnt.usv.usv_detector.mad_metrics import load_eval
    assert load_eval(str(run_dir)).per_threshold[0]['f1'] == 0.7


def test_a_failed_save_does_not_lose_the_result(worker_env, monkeypatch):
    """The numbers are on screen either way; a network hiccup must not
    turn a finished evaluation into an error."""
    W, cfg, _, _ = worker_env
    import fnt.usv.usv_detector.mad_metrics as MM
    monkeypatch.setattr(MM, 'save_eval',
                        lambda *a, **k: (_ for _ in ()).throw(OSError("nope")))
    got = []
    w = W(cfg, ["a.wav"], scope='reviewed', save=True)
    w.finished_signal.connect(got.append)
    w.run()
    assert got and got[0].n_labels == 5


def test_a_stopped_run_is_not_saved(worker_env):
    """A partial sweep must not become the model's recorded score."""
    W, cfg, run_dir, _ = worker_env
    w = W(cfg, ["a.wav"], scope='reviewed', save=True)
    w.request_stop()
    w.run()
    assert not (run_dir / "eval.json").exists()


# ------------------------------------------------- review-done marker
def test_the_marker_is_read_from_the_cache_not_from_disk(qapp, monkeypatch):
    """Painting a row must never open a file on a network share."""
    from fnt.usv.mad_pyqt import MADMainWindow, _ROLE_REVIEW_DONE
    from PyQt5.QtWidgets import QListWidget, QListWidgetItem

    class W:
        _refresh_file_list_labels = MADMainWindow._refresh_file_list_labels

        def __init__(self):
            self.file_list = QListWidget()
            self.audio_files = ["/x/a.wav", "/x/b.wav"]
            for _ in self.audio_files:
                self.file_list.addItem(QListWidgetItem(""))
            self._review_done_cache = {"a.wav"}

    import fnt.usv.usv_detector.fnt_mask_store as MS
    monkeypatch.setattr(MS, 'is_review_complete', _boom)

    w = W()
    w._refresh_file_list_labels()
    assert w.file_list.item(0).data(_ROLE_REVIEW_DONE) is True
    assert w.file_list.item(1).data(_ROLE_REVIEW_DONE) is False


def _boom(*a, **k):
    raise AssertionError("painting must not touch the store")


def test_toggling_updates_the_cache_without_a_reread(qapp, tmp_path,
                                                     monkeypatch):
    pytest.importorskip("h5py")
    from fnt.usv.mad_pyqt import MADMainWindow
    from PyQt5.QtWidgets import QListWidget, QListWidgetItem
    import fnt.usv.usv_detector.fnt_mask_store as MS

    wav = str(tmp_path / "a.wav")
    monkeypatch.setattr(MS, 'masks_sibling_path',
                        lambda w: str(tmp_path / "a_FNT.mad"))

    class W:
        _set_review_complete = MADMainWindow._set_review_complete
        _refresh_file_list_labels = MADMainWindow._refresh_file_list_labels

        def __init__(self):
            self.file_list = QListWidget()
            self.file_list.addItem(QListWidgetItem(""))
            self.audio_files = [wav]
            self._review_done_cache = set()
            self.logged = []
            self.status_bar = type("S", (), {"showMessage": lambda s, m: None})()

        def _log(self, m):
            self.logged.append(m)

    w = W()
    w._set_review_complete([wav], True)
    assert "a.wav" in w._review_done_cache
    assert MS.is_review_complete(str(tmp_path / "a_FNT.mad")) is True
    w._set_review_complete([wav], False)
    assert "a.wav" not in w._review_done_cache


# --------------------------------------------------- summary lines
@pytest.fixture
def summary(qapp, monkeypatch, tmp_path):
    from fnt.usv.mad_pyqt import MADMainWindow

    class W:
        _stop_criterion_lines = MADMainWindow._stop_criterion_lines

        def __init__(self):
            self.audio_files = ["a.wav"]

        def _selected_deploy_model_path(self):
            return None

        def _default_model_path(self):
            return None

    return W()


def test_nothing_to_say_prints_nothing(summary, monkeypatch):
    """A block that always prints trains people to skip it."""
    import fnt.usv.usv_detector.mad_metrics as MM
    monkeypatch.setattr(MM, 'review_outcome_by_model', lambda w: {})
    assert summary._stop_criterion_lines() == []


def test_the_reject_rate_is_reported_when_there_is_enough_of_it(summary,
                                                                monkeypatch):
    import fnt.usv.usv_detector.mad_metrics as MM
    monkeypatch.setattr(MM, 'review_outcome_by_model', lambda w: {
        'm1': {'accepted': 90, 'rejected': 10, 'pending': 0, 'n_judged': 100,
               'accept_rate': 0.9, 'reject_rate': 0.1}})
    out = "\n".join(summary._stop_criterion_lines())
    assert "90% accepted" in out and "10% rejected" in out


def test_a_barely_reviewed_model_is_not_quoted(summary, monkeypatch):
    """Three judged detections is not a precision estimate."""
    import fnt.usv.usv_detector.mad_metrics as MM
    monkeypatch.setattr(MM, 'review_outcome_by_model', lambda w: {
        'm1': {'accepted': 2, 'rejected': 1, 'pending': 0, 'n_judged': 3,
               'accept_rate': 0.67, 'reject_rate': 0.33}})
    assert summary._stop_criterion_lines() == []
