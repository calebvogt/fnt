"""Delete and Backspace remove a detection, but never while typing.

D alone is not what a Windows user reaches for. Adding Delete and Backspace is
trivial; the risk they bring is not. Every single-letter shortcut here is an
ApplicationShortcut, so it fires wherever focus happens to be — and Backspace
in the "Max epochs" box has to edit the number, not silently delete a mask.

The guard that stops that, ``_focus_is_edit``, previously matched only
QSpinBox / QDoubleSpinBox / QComboBox — so a plain QLineEdit or QTextEdit was
not covered, and Backspace in one would have removed a detection.
"""
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtWidgets import (  # noqa: E402
    QApplication, QComboBox, QDoubleSpinBox, QLineEdit, QPushButton, QSpinBox,
    QTextEdit, QWidget,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def guard(qapp):
    from fnt.usv.mad_pyqt import MADMainWindow

    class W:
        _focus_is_edit = MADMainWindow._focus_is_edit

    return W()


def _focus(w):
    w.show()
    w.setFocus()
    QApplication.processEvents()
    return w


@pytest.mark.parametrize("factory", [QSpinBox, QDoubleSpinBox, QComboBox,
                                     QLineEdit, QTextEdit])
def test_typing_into_a_field_blocks_the_shortcut(guard, factory):
    w = _focus(factory())
    try:
        assert guard._focus_is_edit() is True
    finally:
        w.close()


def test_a_spinbox_reports_itself_as_the_focus_widget(guard):
    """Qt hands back the QSpinBox, not the QLineEdit it draws inside itself.

    Worth pinning: it is the reason the original narrower check worked for
    spin boxes at all, and the reason widening it was about the plain text
    fields rather than about spin boxes.
    """
    sb = QSpinBox()
    sb.show()
    inner = sb.findChild(QLineEdit)
    assert inner is not None
    inner.setFocus()
    QApplication.processEvents()
    try:
        assert QApplication.focusWidget() is sb
        assert guard._focus_is_edit() is True
    finally:
        sb.close()


def test_a_non_editing_widget_allows_the_shortcut(guard):
    w = _focus(QPushButton("ok"))
    try:
        assert guard._focus_is_edit() is False
    finally:
        w.close()


def test_no_focus_allows_the_shortcut(guard):
    w = QWidget()
    w.show()
    QApplication.processEvents()
    for f in (QApplication.focusWidget(),):
        if f is not None:
            f.clearFocus()
    QApplication.processEvents()
    try:
        assert guard._focus_is_edit() is False
    finally:
        w.close()


def test_delete_and_backspace_are_bound_to_the_delete_action():
    """Read from the source: the binding block is not reachable without a
    fully constructed main window."""
    import inspect
    from fnt.usv.mad_pyqt import MADMainWindow
    src = inspect.getsource(MADMainWindow._setup_shortcuts)
    for key in ('make("D", self._delete_selected_annotation)',
                'make(Qt.Key_Delete, self._delete_selected_annotation)',
                'make(Qt.Key_Backspace, self._delete_selected_annotation)'):
        assert key in src, key


def test_the_delete_action_checks_the_guard_first():
    """Whatever key reaches it, it must refuse while a field has focus."""
    import inspect
    from fnt.usv.mad_pyqt import MADMainWindow
    src = inspect.getsource(MADMainWindow._delete_selected_annotation)
    body = src.split('"""')[-1]
    assert body.strip().startswith("if self._focus_is_edit():")
