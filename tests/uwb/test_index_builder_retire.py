"""The indexed-copy builder must outlive its own signal.

PreviewIndexBuilder emits done/failed from inside its run(), so clearing the
last reference in the slot destroys a QThread that has not returned yet. Qt
aborts the process for that (SIGABRT, exit code 6) - which killed the UWB tool
on 2026-09-22 the moment a 6 GB index finished building. The release is
deferred until the thread reports finished; this guards that.

Runs under pytest, or directly (``python test_index_builder_retire.py``).
"""
import os
import sys
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from PyQt5.QtCore import QCoreApplication, QTimer  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

app = QApplication.instance() or QApplication(sys.argv)

from fnt.uwb.uwb_preprocessing_pyqt import UWBQuickVisualizationWindow  # noqa: E402

retire = UWBQuickVisualizationWindow._retire_index_builder


class _Thread:
    """Stands in for the builder: not finished until told otherwise."""

    def __init__(self):
        self.finished = False

    def isFinished(self):
        return self.finished


def _holder(thread):
    """A stand-in for the window: just the attribute and the bound method."""
    h = types.SimpleNamespace(preview_index_builder=thread)
    h._retire_index_builder = lambda: retire(h)   # what the retry re-calls
    return h


def _pump(ms=250):
    end = QTimer()
    end.setSingleShot(True)
    end.start(ms)
    while end.isActive():
        QCoreApplication.processEvents()


def test_a_running_builder_is_not_released():
    holder = _holder(_Thread())
    retire(holder)
    # Still running: the reference MUST survive, or Qt aborts the process.
    assert holder.preview_index_builder is not None
    _pump()
    assert holder.preview_index_builder is not None


def test_the_builder_is_released_once_it_finishes():
    holder = _holder(_Thread())
    retire(holder)
    assert holder.preview_index_builder is not None
    holder.preview_index_builder.finished = True
    _pump()          # the deferred retry runs on the event loop
    assert holder.preview_index_builder is None


def test_no_builder_is_a_no_op():
    holder = _holder(None)
    retire(holder)
    assert holder.preview_index_builder is None


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
