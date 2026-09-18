"""A batch inference run gets its own progress window — with Pause in it.

Two defects, one mechanism:

* A plain **Run Inference** showed its progress in the sidebar, under whatever
  else was open, for a run that can last days. Only the chained Training +
  Inference run floated the panel into a window of its own.
* When the chained run *did* float it, **Pause stayed behind**. The button was
  built straight into the sidebar layout rather than beside the panel, so
  floating the panel took Stop along and left Pause under the main window.

Pause and the panel now live in one box, and every run reported through that
box floats it — after the "analyzed before?" prompt, so cancelling there
leaves no empty window — and docks it back when the run ends.

Closing the window does not stop the run. It used to simply hide, taking the
bars, Pause and Stop into an invisible window for the rest of the run; now the
box returns to the sidebar and the user is told where it went.
"""
import inspect

import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import (  # noqa: E402
    QApplication, QDialog, QMainWindow, QPushButton, QVBoxLayout, QWidget,
)

from fnt.usv.mad_pyqt import (  # noqa: E402
    MADInferProgressWindow, MADMainWindow, MADRunPanel,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


class _Bar:
    def __init__(self):
        self.msgs = []

    def showMessage(self, m, *a):
        self.msgs.append(m)

    def clearMessage(self):
        pass


@pytest.fixture
def win(qapp):
    class W(QMainWindow):
        _show_infer_progress_dialog = \
            MADMainWindow._show_infer_progress_dialog
        _close_infer_progress_dialog = \
            MADMainWindow._close_infer_progress_dialog
        _on_infer_window_closed = MADMainWindow._on_infer_window_closed
        _open_child_windows = MADMainWindow._open_child_windows
        _start_inference = MADMainWindow._start_inference

        def __init__(self):
            super().__init__()
            self.status_bar = _Bar()
            self.logged = []
            self._infer_dialog = None
            self._train_dialog = None
            self._preview_dialog = None
            # Mirrors the sidebar construction in MADMainWindow; the wiring
            # itself is pinned against the real source further down.
            self.btn_infer_pause = QPushButton("Pause")
            self.infer_panel = MADRunPanel(show_plot=False)
            self.infer_panel.run_finished.connect(
                lambda ok: self._close_infer_progress_dialog())
            self._infer_progress_box = QWidget()
            pbox = QVBoxLayout(self._infer_progress_box)
            pbox.addWidget(self.btn_infer_pause)
            pbox.addWidget(self.infer_panel)
            self._sidebar = QWidget()
            self._infer_panel_home = QVBoxLayout(self._sidebar)
            self._infer_panel_home.addWidget(self._infer_progress_box)

        def _log(self, m):
            self.logged.append(m)

        def _update_infer_run_enabled(self):
            pass

        def running(self, on=True):
            self.btn_infer_pause.setEnabled(on)
            self.infer_panel.btn_stop.setEnabled(on)

        def in_window(self, w):
            p = w
            while p is not None:
                if p is self._infer_dialog:
                    return True
                p = p.parentWidget()
            return False

    w = W()
    yield w
    w._close_infer_progress_dialog()


def _settle(qapp):
    for _ in range(3):
        qapp.processEvents()


# ------------------------------------------------- floating
def test_the_run_gets_a_window_of_its_own(win):
    win._show_infer_progress_dialog()
    assert isinstance(win._infer_dialog, MADInferProgressWindow)
    assert win._infer_dialog.isVisible()
    assert win._infer_dialog.windowTitle() == "Running Inference"


def test_pause_goes_with_the_panel(win):
    """The defect: floating took Stop and left Pause in the sidebar."""
    win._show_infer_progress_dialog()
    assert win.in_window(win.btn_infer_pause), "Pause was left behind"
    assert win.in_window(win.infer_panel.btn_stop)
    assert win.in_window(win.infer_panel)


def test_pause_sits_above_the_progress_as_it_did_in_the_sidebar(win):
    """'Same visual, as a pop-up' — the order is the sidebar's order."""
    lay = win._infer_progress_box.layout()
    assert lay.indexOf(win.btn_infer_pause) < lay.indexOf(win.infer_panel)


def test_showing_twice_is_one_window(win):
    """The chained run floats from two places; there must be one window."""
    win._show_infer_progress_dialog()
    first = win._infer_dialog
    win._show_infer_progress_dialog()
    assert win._infer_dialog is first


def test_the_window_is_parentless(win):
    """So it minimises and moves on its own, like the training graph."""
    win._show_infer_progress_dialog()
    assert win._infer_dialog.parent() is None


# ------------------------------------------------- docking back
def test_the_end_of_the_run_docks_everything_back(win, qapp):
    win._show_infer_progress_dialog()
    win.infer_panel.mark_done(ok=True)
    _settle(qapp)
    assert win._infer_dialog is None
    assert win._infer_progress_box.parentWidget() is win._sidebar
    assert win.btn_infer_pause.parentWidget() is win._infer_progress_box


def test_docking_when_nothing_floated_is_harmless(win):
    win._close_infer_progress_dialog()
    win._close_infer_progress_dialog()
    assert win._infer_progress_box.parentWidget() is win._sidebar


# ------------------------------------------------- closing mid-run
def test_closing_the_window_does_not_stop_the_run(win, qapp):
    win.running(True)
    win._show_infer_progress_dialog()
    win._infer_dialog.close()
    _settle(qapp)
    assert win.infer_panel.btn_stop.isEnabled(), "closing stopped the run"
    assert win.btn_infer_pause.isEnabled()


def test_closing_mid_run_puts_the_controls_back_in_the_sidebar(win, qapp):
    """Not into a hidden window, where Pause and Stop would be unreachable
    for the rest of a multi-day run."""
    win.running(True)
    win._show_infer_progress_dialog()
    win._infer_dialog.close()
    _settle(qapp)
    assert win._infer_dialog is None
    assert win._infer_progress_box.parentWidget() is win._sidebar


def test_closing_mid_run_says_where_the_progress_went(win, qapp):
    win.running(True)
    win._show_infer_progress_dialog()
    win._infer_dialog.close()
    _settle(qapp)
    said = " ".join(win.status_bar.msgs + win.logged)
    assert "still running" in said and "Run Inference section" in said


def test_closing_after_the_run_says_nothing(win, qapp):
    win.running(False)
    win._show_infer_progress_dialog()
    win._infer_dialog.close()
    _settle(qapp)
    assert win.logged == []


def test_a_window_closed_by_the_app_does_not_call_back(win, qapp):
    """The app's own close (end of run) must not re-enter the user-close
    path and log 'still running' about a run that just ended."""
    win.running(True)
    win._show_infer_progress_dialog()
    win._close_infer_progress_dialog()
    _settle(qapp)
    assert win.logged == []


# ------------------------------------------------- the Window menu
def test_the_window_menu_can_find_it(win):
    """Parentless windows are invisible to findChildren — and this is the one
    most worth finding during a long run."""
    win._show_infer_progress_dialog()
    assert win._infer_dialog in win._open_child_windows()


def test_the_training_windows_are_findable_too(win, qapp):
    d = QDialog(None)
    d.show()
    win._train_dialog = d
    try:
        assert d in win._open_child_windows()
    finally:
        d.close()


def test_a_hidden_window_is_not_listed(win, qapp):
    d = QDialog(None)
    win._train_dialog = d                 # never shown
    assert d not in win._open_child_windows()


# ------------------------------------------------- when a run floats it
def test_a_cancelled_scope_prompt_leaves_no_window(win):
    """Nothing to analyze -> nothing to watch."""
    win._start_inference(object(), [], reporter=win.infer_panel,
                         skip_scope_prompt=True)
    assert win._infer_dialog is None


def _code(fn):
    src = inspect.getsource(fn)
    return "\n".join(ln for ln in src.splitlines()
                     if not ln.strip().startswith("#"))


def test_it_floats_after_the_scope_prompt_not_before():
    code = _code(MADMainWindow._start_inference)
    i_gate = code.index("if not wav_paths:")
    i_float = code.index("self._show_infer_progress_dialog()")
    assert i_gate < i_float


def test_only_sidebar_reported_runs_float():
    """The Project -> Run Inference dialog brings its own modal progress
    window; floating the sidebar panel as well would show two."""
    code = _code(MADMainWindow._start_inference)
    assert "if reporter is getattr(self, 'infer_panel', None):" in code


def test_plain_run_inference_reports_through_the_sidebar_panel():
    """Which is what makes the rule above cover it."""
    code = _code(MADMainWindow._on_deploy_infer)
    assert "reporter=self.infer_panel" in code


# ------------------------------------------------- the real construction
def _init_code():
    return _code(MADMainWindow)


def test_pause_is_built_into_the_travelling_box():
    code = _init_code()
    assert "pbox.addWidget(self.btn_infer_pause)" in code
    assert "pbox.addWidget(self.infer_panel)" in code
    assert "ibody.addWidget(self._infer_progress_box)" in code
    assert "ibody.addWidget(self.btn_infer_pause)" not in code


def test_every_floated_run_docks_on_finish():
    code = _init_code()
    assert ("self.infer_panel.run_finished.connect(\n"
            "            lambda ok: self._close_infer_progress_dialog())") in code
