"""Confirming a batch switches SAM and Paint off; the Eraser stays armed.

Both are drawing tools whose pending stroke is consumed by Enter, so leaving
either armed means the next click starts a mask the user did not ask for — SAM
proposes a segment, Paint lays down pixels. Disarming both makes starting the
next call deliberate, and makes the two behave alike.

The Eraser is not a drawing tool: it only removes pixels from a stroke in
progress, and there is nothing to remove until something is drawn. It persists.
"""
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import QApplication, QPushButton  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(qapp):
    from fnt.usv.mad_pyqt import MADMainWindow

    class Spec:
        def __init__(self):
            self.paint_mode = 'sam'
            self.prompts_cleared = 0

        def set_paint_mode(self, m):
            self.paint_mode = m

        def clear_sam_prompts(self):
            self.prompts_cleared += 1

    class W:
        _reset_labeling_tools_after_confirm = (
            MADMainWindow._reset_labeling_tools_after_confirm)

        def __init__(self):
            self.spectrogram = Spec()
            for n in ('btn_sam', 'btn_paint', 'btn_erase'):
                b = QPushButton()
                b.setCheckable(True)
                setattr(self, n, b)

    return W()


def test_confirming_switches_sam_off(win):
    """The requested change."""
    win.btn_sam.setChecked(True)
    assert win._reset_labeling_tools_after_confirm() == "SAM"
    assert win.btn_sam.isChecked() is False


def test_the_paint_mode_is_cleared_too(win):
    """setChecked does not emit `clicked`, so the button going up is not
    what turns SAM off — the widget would stay in SAM mode."""
    win.btn_sam.setChecked(True)
    win.spectrogram.paint_mode = 'sam'
    win._reset_labeling_tools_after_confirm()
    assert win.spectrogram.paint_mode is None


def test_confirming_switches_paint_off_too(win):
    """Mirrors SAM: Enter consumes the stroke, so the tool disarms with it."""
    win.btn_paint.setChecked(True)
    win.spectrogram.paint_mode = 'paint'
    assert win._reset_labeling_tools_after_confirm() == "Paint"
    assert win.btn_paint.isChecked() is False
    assert win.spectrogram.paint_mode is None


def test_eraser_stays_armed(win):
    win.btn_erase.setChecked(True)
    win.spectrogram.paint_mode = 'erase'
    win._reset_labeling_tools_after_confirm()
    assert win.btn_erase.isChecked() is True
    assert win.spectrogram.paint_mode == 'erase'


def test_prompts_are_always_dropped(win):
    """True whichever tool was up: a kept prompt would extend the saved call."""
    for setup in (lambda: win.btn_sam.setChecked(True),
                  lambda: win.btn_paint.setChecked(True),
                  lambda: None):
        win.spectrogram.prompts_cleared = 0
        setup()
        win._reset_labeling_tools_after_confirm()
        assert win.spectrogram.prompts_cleared == 1


def test_no_tool_armed_is_not_an_error(win):
    assert win._reset_labeling_tools_after_confirm() is None
    assert win.spectrogram.prompts_cleared == 1


def test_a_widget_without_clear_sam_prompts_is_survivable(win):
    """Older spectrogram widgets lack it; confirming must not fail."""
    del type(win.spectrogram).clear_sam_prompts
    win.btn_sam.setChecked(True)
    try:
        assert win._reset_labeling_tools_after_confirm() == "SAM"
    finally:
        type(win.spectrogram).clear_sam_prompts = (
            lambda self: setattr(self, 'prompts_cleared',
                                 self.prompts_cleared + 1))


def test_m_can_re_arm_it_afterwards():
    """The way back on, per the request: the shortcut drives the same path
    as the button and is not gated on anything confirm changes."""
    import inspect
    from fnt.usv.mad_pyqt import MADMainWindow
    src = inspect.getsource(MADMainWindow._shortcut_toggle_sam)
    assert "self.btn_sam.setChecked(new_state)" in src
    assert "self._on_sam_clicked(new_state)" in src
    binds = inspect.getsource(MADMainWindow._setup_shortcuts)
    assert "make(Qt.Key_M, self._shortcut_toggle_sam)" in binds


def test_confirm_tells_the_user_sam_went_off():
    """A tool that disarms silently reads as clicks that stopped working."""
    import inspect
    from fnt.usv.mad_pyqt import MADMainWindow
    src = inspect.getsource(MADMainWindow._confirm_pending)
    assert "tool_off = self._reset_labeling_tools_after_confirm()" in src
    assert "off — press" in src
