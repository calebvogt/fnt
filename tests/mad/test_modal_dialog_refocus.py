"""A modal dialog buried behind the main window can be got back.

Reported against Add Folder: open the picker, click away to another window,
then come back to MAD from the taskbar. Windows raises the *main* window, which
lands on top of the dialog. The dialog is still modal, so every click on the
window now in front of it is refused -- and with no way to reach the thing
doing the refusing, the only way out is killing MAD and losing the session.

The fix watches for the app/window coming forward and pulls whatever is
blocking input back in front of it. It is deliberately about
``activeModalWidget`` rather than "a dialog MAD parented": what matters is what
is blocking the user, whoever owns it.

Runs under pytest, or directly.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

from PyQt5.QtCore import QEvent  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QApplication, QDialog, QFileDialog, QWidget)

import fnt.usv.mad_pyqt as M  # noqa: E402

raise_blocking = M.MADMainWindow._raise_blocking_dialog

_APP = None
_WINDOWS = []


def app():
    global _APP
    if _APP is None:
        _APP = QApplication.instance() or QApplication([])
    return _APP


def host():
    """A stand-in for the main window. ``_raise_blocking_dialog`` uses self only
    to skip itself and to find a screen, so the real window isn't needed."""
    a = app()
    w = QWidget()
    w.setGeometry(100, 100, 400, 300)
    w.show()
    a.processEvents()
    _WINDOWS.append(w)
    return w


def modal(parent, **kw):
    d = QDialog(parent, **kw)
    d.setModal(True)
    d.show()
    app().processEvents()
    _WINDOWS.append(d)
    return d


def close_all():
    for w in _WINDOWS:
        w.close()
    _WINDOWS.clear()
    app().processEvents()


# ------------------------------------------------------------- the fix
def test_an_open_modal_dialog_is_reported_and_raised():
    w = host()
    try:
        d = modal(w)
        assert raise_blocking(w) is True
        assert d.isVisible()
    finally:
        close_all()


def test_nothing_open_is_a_no_op():
    """The handler runs on every activation, so the common case has to be
    free and must not touch the window."""
    w = host()
    try:
        assert raise_blocking(w) is False
    finally:
        close_all()


def test_a_minimized_dialog_is_restored():
    """Minimizing MAD takes its dialogs down with it; raising a still-iconified
    window puts nothing on screen."""
    w = host()
    try:
        d = modal(w)
        d.showMinimized()
        app().processEvents()
        assert d.isMinimized()
        assert raise_blocking(w) is True
        assert not d.isMinimized()
    finally:
        close_all()


def test_a_dialog_on_an_unplugged_monitor_is_brought_back():
    """Off every screen, raising changes nothing visible -- the dialog is just
    as unreachable as before."""
    w = host()
    try:
        d = modal(w)
        d.move(-9999, -9999)
        app().processEvents()
        raise_blocking(w)
        app().processEvents()
        assert any(s.availableGeometry().intersects(d.frameGeometry())
                   for s in QApplication.screens()), d.frameGeometry()
    finally:
        close_all()


def test_a_dialog_hanging_off_the_edge_is_left_where_the_user_put_it():
    """Partly off-screen is a position someone chose. Only fully-lost windows
    get moved, or the fix becomes a dialog that jumps around."""
    w = host()
    try:
        d = modal(w)
        d.move(-40, 20)
        app().processEvents()
        before = d.pos()
        raise_blocking(w)
        app().processEvents()
        assert d.pos() == before
    finally:
        close_all()


def test_the_window_itself_is_never_the_target():
    """A modal dialog could in principle be the caller; raising self would be a
    no-op that reports success."""
    w = host()
    try:
        d = modal(w)
        assert raise_blocking(d) is False
    finally:
        close_all()


def test_a_file_picker_counts():
    """The reported case. QFileDialog shown non-natively is a plain modal
    QDialog, which is the whole reason it can end up behind the window."""
    w = host()
    try:
        d = QFileDialog(w, "Add folder(s) of .wav files")
        d.setOption(QFileDialog.DontUseNativeDialog, True)
        d.setModal(True)
        d.show()
        app().processEvents()
        _WINDOWS.append(d)
        assert QApplication.activeModalWidget() is d
        assert raise_blocking(w) is True
    finally:
        close_all()


# -------------------------------------------------------------- wiring
def test_the_events_we_watch_cover_taskbar_and_restore():
    ev = M.MADMainWindow._REFOCUS_EVENTS
    assert QEvent.WindowActivate in ev, "returning to the window"
    assert QEvent.WindowStateChange in ev, "restoring from minimized"
    assert None not in ev


def test_activation_pulls_the_dialog_back_on_the_real_window():
    """End to end through the installed application event filter: the window
    is told it became active, and the buried dialog comes back."""
    a = app()
    M.MADMainWindow._apply_dark_theme()
    win = M.MADMainWindow()
    _WINDOWS.append(win)
    try:
        win.show()
        a.processEvents()
        d = modal(win)
        d.showMinimized()
        a.processEvents()
        assert d.isMinimized()
        a.sendEvent(win, QEvent(QEvent.WindowActivate))
        for _ in range(10):          # the raise is deferred by a 0ms timer
            a.processEvents()
        assert not d.isMinimized(), "the dialog stayed buried"
    finally:
        close_all()


if __name__ == "__main__":
    import sys
    import traceback
    fails = 0
    for name, fn in sorted(globals().items()):
        if not name.startswith("test_") or not callable(fn):
            continue
        try:
            fn()
            print("  OK   " + name, flush=True)
        except Exception:
            fails += 1
            print("  FAIL " + name, flush=True)
            traceback.print_exc()
    print("")
    print("ALL OK" if not fails else str(fails) + " FAILURE(S)", flush=True)
    sys.stdout.flush()
    os._exit(1 if fails else 0)
