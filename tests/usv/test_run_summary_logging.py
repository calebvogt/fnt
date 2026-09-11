"""A run's headline numbers must survive the dialog being dismissed.

Everything that answered "what did this 17-hour run actually do?" — labels
trained on, val_dice, the chosen threshold, the model path, how many files
failed — existed only inside a modal dialog that is shown once and cannot be
reopened. Meanwhile the per-file timing lines, which are the least interesting
part, were kept in the Session Logs forever.

Failures are named here specifically: a failed file writes no sidecar and no
timing line, so without this its name appears nowhere the user can return to.
"""
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(qapp):
    from fnt.usv.mad_pyqt import MADMainWindow

    class W:
        _log_run_summary = MADMainWindow._log_run_summary

        def __init__(self):
            self.logged = []

        def _log(self, msg):
            self.logged.append(msg)

        def text(self):
            return "\n".join(self.logged)

    return W()


def test_the_headline_and_summary_reach_the_log(win):
    win._log_run_summary(
        "Training and Inference complete!",
        ["Trained on 171 confirmed label(s).",
         "Best val_dice = 0.885  (val_loss 0.3451)",
         "Best threshold 0.70 — applied to Inference settings."],
        [])
    out = win.text()
    assert "Training and Inference complete!" in out
    assert "Trained on 171 confirmed label(s)." in out
    assert "Best val_dice = 0.885  (val_loss 0.3451)" in out
    assert "Best threshold 0.70" in out


def test_embedded_newlines_become_separate_entries(win):
    """Summary lines carry leading newlines for dialog spacing; a log entry
    holding a raw newline renders as one unreadable row."""
    win._log_run_summary("done", ["\nInference ran on 275 file(s).", "x"], [])
    assert "Inference ran on 275 file(s)." in win.logged
    assert not any("\n" in m for m in win.logged)


def test_blank_lines_are_not_logged(win):
    win._log_run_summary("done", ["", "\n", "real"], [])
    assert "real" in win.logged
    assert "" not in win.logged


def test_failures_are_named(win):
    """The point: a failed file leaves no other trace in the log."""
    win._log_run_summary("done", [], [
        {'wav_path': r"Z:\x\T006_C57_ch1_T0000380.wav",
         'error': 'Unable to allocate 572. MiB'},
        {'wav_path': r"Z:\x\ok.wav", 'n_blobs': 3},
    ])
    out = win.text()
    assert "T006_C57_ch1_T0000380.wav" in out
    assert "Unable to allocate 572. MiB" in out


def test_a_successful_file_is_not_named(win):
    """275 of them would bury the summary; they already have timing lines."""
    win._log_run_summary("done", [], [
        {'wav_path': r"Z:\x\ok.wav", 'n_blobs': 3}])
    assert "ok.wav" not in win.text()


def test_a_flood_of_failures_is_capped(win):
    """A run where everything failed must not push the session out of reach."""
    results = [{'wav_path': f"z{i}.wav", 'error': 'boom'} for i in range(120)]
    win._log_run_summary("done", [], results)
    named = [m for m in win.logged if m.startswith("  ✖")]
    assert len(named) == 40
    assert "and 80 more failure(s)" in win.text()


def test_no_cap_note_when_everything_fits(win):
    win._log_run_summary("done", [], [
        {'wav_path': "a.wav", 'error': 'boom'}])
    assert "more failure(s)" not in win.text()


def test_the_block_is_delimited(win):
    """So it is findable by eye in a log holding hundreds of timing lines."""
    win._log_run_summary("done", ["x"], [])
    assert win.logged[0].startswith("=") and win.logged[-1].startswith("=")


def test_it_is_logged_before_the_dialog_opens():
    """A crash building the dialog must still leave the record — and the
    dialog is modal, so logging after it would wait on the user."""
    import inspect
    from fnt.usv.mad_pyqt import MADMainWindow
    src = inspect.getsource(MADMainWindow._show_run_summary_dialog)
    assert src.index("_log_run_summary") < src.index("MADRunSummaryTable(")


def test_results_none_is_survivable(win):
    win._log_run_summary("done", ["x"], None)
    assert "done" in win.text()
