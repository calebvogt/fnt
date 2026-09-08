"""Touching a QThread that ``deleteLater`` has already reaped.

``finished.connect(deleteLater)`` stops worker objects piling up for the life of
a session, but it destroys the C++ object while the Python attribute still
refers to it. Asking that wrapper anything — even ``isRunning()`` — raises
RuntimeError, and PyQt turns an unhandled exception in a slot into an abort.

This is not hypothetical: one stale reference broke the confirmed-mask delete.
The masks came off disk, then ``_scan_all_file_counts`` raised on the dead
worker before the gallery could rebuild, so the tiles stayed on screen and a
completed deletion looked like a no-op.
"""
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import QThread  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(qapp):
    from fnt.usv.mad_pyqt import MADMainWindow

    class W:
        _live_worker = MADMainWindow._live_worker

    return W()


class _Idle(QThread):
    def run(self):
        return


class _Dead:
    """Stands in for a wrapper whose C++ object is gone."""

    def isRunning(self):
        raise RuntimeError(
            "wrapped C/C++ object of type _SidecarScanWorker has been deleted")


def test_a_reaped_worker_reads_as_absent(win):
    """The regression: this used to raise and abort the caller."""
    win._counts_worker = _Dead()
    assert win._live_worker('_counts_worker') is None


def test_a_reaped_worker_is_cleared_so_the_next_call_is_cheap(win):
    win._counts_worker = _Dead()
    win._live_worker('_counts_worker')
    assert win._counts_worker is None


def test_a_finished_worker_reads_as_absent(win, qapp):
    w = _Idle()
    w.start()
    w.wait(2000)
    win._counts_worker = w
    assert win._live_worker('_counts_worker') is None
    assert win._counts_worker is None


def test_a_missing_attribute_is_fine(win):
    assert win._live_worker('_never_set') is None


def test_none_is_fine(win):
    win._counts_worker = None
    assert win._live_worker('_counts_worker') is None


def test_a_running_worker_is_returned(win, qapp):
    class Busy(QThread):
        def run(self):
            self.msleep(400)

    w = Busy()
    w.start()
    try:
        assert win._live_worker('_counts_worker') is None  # not stored yet
        win._counts_worker = w
        assert win._live_worker('_counts_worker') is w
        assert win._counts_worker is w                     # not cleared
    finally:
        w.wait(3000)
