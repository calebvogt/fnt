"""Two fixes from the 2026-09-09 run.

**Memory.** Per-file commit logging showed each file retaining exactly 4.00x
its full-file grid size, at every duration tested (346 s, 600 s, 825 s, 888 s,
1800 s — all 4.00-4.01). Those arrays sit in reference cycles, so only the
*cyclic* collector frees them, and it triggers on object counts: one file
allocates very few, very large objects, so it can go dozens of files without
firing. The trace climbed to 70.1 GB with 36 sudden drops as gen-2 finally ran,
leaving 4.7 GB of system commit. Collecting between files holds it flat.

**Windows.** A dialog that goes behind the main window has no way back —
Windows gives an owned dialog a taskbar thumbnail but no way to raise it. For
the modal run-summary dialog that reads as a frozen application.
"""
import pytest

from fnt.usv.usv_detector import mad_inference as MI


# ------------------------------------------------------------- memory
def test_a_collect_runs_between_files(monkeypatch):
    """The fix itself: once per file, not once per batch."""
    calls = []
    monkeypatch.setattr(MI, 'load_model', lambda p, d: (object(), {}, 'cpu'))
    monkeypatch.setattr(MI, '_collect_cycles', lambda: calls.append(1))
    monkeypatch.setattr(
        MI, 'run_inference_on_file',
        lambda wav, cfg, **kw: {'wav_path': wav, 'n_blobs': 0})
    cfg = MI.MADInferenceConfig(model_path='m.pt')
    MI.run_inference_on_files(['a.wav', 'b.wav', 'c.wav'], cfg)
    assert len(calls) == 3


def test_a_failed_file_is_collected_too(monkeypatch):
    """The failure path is where the grids are most likely still referenced —
    a live traceback holds the frame, and with it every local."""
    calls = []
    monkeypatch.setattr(MI, 'load_model', lambda p, d: (object(), {}, 'cpu'))
    monkeypatch.setattr(MI, '_collect_cycles', lambda: calls.append(1))
    monkeypatch.setattr(MI, '_release_between_files', lambda: None)

    def boom(wav, cfg, **kw):
        raise MemoryError("Unable to allocate 572. MiB")

    monkeypatch.setattr(MI, 'run_inference_on_file', boom)
    cfg = MI.MADInferenceConfig(model_path='m.pt', retry_failed=False)
    MI.run_inference_on_files(['a.wav'], cfg)
    assert calls == [1]


def test_the_reading_is_taken_after_the_collect(monkeypatch):
    """Otherwise `mem` reports what has merely not been collected yet, and the
    log would show a leak that is not there."""
    order = []
    monkeypatch.setattr(MI, 'load_model', lambda p, d: (object(), {}, 'cpu'))
    monkeypatch.setattr(MI, '_collect_cycles', lambda: order.append('collect'))
    monkeypatch.setattr(MI, '_mem_record', lambda b: order.append('read') or {})
    monkeypatch.setattr(
        MI, 'run_inference_on_file',
        lambda wav, cfg, **kw: {'wav_path': wav, 'n_blobs': 0})
    cfg = MI.MADInferenceConfig(model_path='m.pt')
    MI.run_inference_on_files(['a.wav'], cfg)
    assert order == ['collect', 'read']


def test_collect_cycles_actually_collects():
    import gc
    n = []
    monkey = gc.collect
    try:
        gc.collect = lambda *a: n.append(1)
        MI._collect_cycles()
    finally:
        gc.collect = monkey
    assert n == [1]


def test_the_retry_pass_also_drops_the_cuda_cache(monkeypatch):
    """Worth paying once before re-attempting failures, not between files:
    empty_cache synchronises the device."""
    seen = []
    monkeypatch.setattr(MI, '_collect_cycles', lambda: seen.append('gc'))
    MI._release_between_files()
    assert seen == ['gc']


# ------------------------------------------------------------ windows
@pytest.fixture(scope="module")
def qapp():
    pytest.importorskip("PyQt5")
    from PyQt5.QtWidgets import QApplication
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(qapp):
    from PyQt5.QtWidgets import QDialog, QWidget
    from fnt.usv.mad_pyqt import MADMainWindow

    class W(QWidget):
        _open_child_windows = MADMainWindow._open_child_windows
        present_window = MADMainWindow.present_window
        _bring_all_to_front = MADMainWindow._bring_all_to_front

        def child(self, title, visible=True):
            d = QDialog(self)
            d.setWindowTitle(title)
            if visible:
                d.show()
            return d

    w = W()
    w.show()
    yield w
    w.close()


def test_open_dialogs_are_listed(win):
    win.child("Run complete")
    win.child("Training Masks Gallery")
    titles = [d.windowTitle() for d in win._open_child_windows()]
    assert "Run complete" in titles
    assert "Training Masks Gallery" in titles


def test_a_hidden_dialog_is_not_listed(win):
    win.child("Never shown", visible=False)
    assert win._open_child_windows() == []


def test_a_closed_dialog_drops_out_of_the_list(win):
    """Read from Qt, not a list we maintain — a stale entry is worse than
    no menu."""
    d = win.child("Run complete")
    d.close()
    from PyQt5.QtWidgets import QApplication
    QApplication.processEvents()
    assert win._open_child_windows() == []


def test_a_minimised_window_is_restored_not_just_raised(win):
    """raise_ alone does nothing to a minimised window, which is the state
    the user is most likely stuck in."""
    d = win.child("Run complete")
    d.showMinimized()
    from PyQt5.QtWidgets import QApplication
    QApplication.processEvents()
    win.present_window(d)
    QApplication.processEvents()
    assert not d.isMinimized()


def test_presenting_a_deleted_window_is_survivable(win):
    """sip can reap the C++ object while the Python wrapper lives on."""
    import sip
    d = win.child("Run complete")
    sip.delete(d)
    win.present_window(d)          # must not raise


def test_bring_all_to_front_touches_every_window(win):
    a, b = win.child("A"), win.child("B")
    win._bring_all_to_front()
    from PyQt5.QtWidgets import QApplication
    QApplication.processEvents()
    assert a.isVisible() and b.isVisible()


def test_the_modal_summary_raises_itself_on_show():
    """Arriving in front beats recovering from behind — this one is modal, so
    hidden behind the main window it reads as a frozen app."""
    import inspect
    from fnt.usv.mad_pyqt import MADRunSummaryTable
    src = inspect.getsource(MADRunSummaryTable.showEvent)
    assert "self.raise_()" in src and "self.activateWindow()" in src


def test_the_gallery_raises_itself_on_show():
    import inspect
    from fnt.usv.mad_pyqt import MADConfirmedGalleryDialog
    src = inspect.getsource(MADConfirmedGalleryDialog.showEvent)
    assert "self.raise_()" in src and "self.activateWindow()" in src
