"""``mark_done`` runs its listeners before the caller's next line.

A chained Training + Inference run put up two completion dialogs — "Inference
complete" and "Run complete" — reporting the same run. The suppression was
already written and looked right::

    if not self._post_train_infer_wavs:
        self._show_inference_summary_dialog(...)

but the queue it tests is cleared by a slot on ``run_finished``, and the same
completion handler calls ``progress.mark_done()`` twenty-four lines earlier.
Qt delivers a direct connection synchronously, so by the time the check ran the
queue was empty and a chained run looked unchained.

The lesson is not about that one flag: any state a completion handler reads
*after* calling mark_done may already have been changed by a listener. The fix
was to capture the answer when the run started.
"""
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def panel(qapp):
    from fnt.usv.mad_pyqt import MADRunPanel
    return MADRunPanel()


def test_mark_done_delivers_run_finished_synchronously(panel):
    """The trap, stated directly."""
    seen = []
    panel.run_finished.connect(lambda ok: seen.append(ok))
    panel.mark_done(ok=True)
    assert seen == [True]          # already delivered, no event loop needed


def test_state_a_listener_clears_is_gone_by_the_next_line(panel):
    """Exactly the shape of the double-dialog bug."""
    state = {'queue': ['a.wav', 'b.wav']}
    panel.run_finished.connect(lambda _ok: state.update(queue=[]))

    # The buggy version: read the flag after mark_done.
    panel.mark_done(ok=True)
    read_after = bool(state['queue'])
    assert read_after is False     # looks unchained, though it was chained


def test_capturing_before_mark_done_survives_the_listener(panel):
    """The fix: decide while the answer is still true."""
    state = {'queue': ['a.wav', 'b.wav']}
    captured = bool(state['queue'])          # read at launch
    panel.run_finished.connect(lambda _ok: state.update(queue=[]))
    panel.mark_done(ok=True)
    assert captured is True                  # still knows it was chained


def test_mark_done_reports_failure_too(panel):
    seen = []
    panel.run_finished.connect(lambda ok: seen.append(ok))
    panel.mark_done(ok=False)
    assert seen == [False]


def test_mark_done_disables_the_stop_control(panel):
    panel.btn_stop.setEnabled(True)
    panel.mark_done(ok=True)
    assert not panel.btn_stop.isEnabled()
