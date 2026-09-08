"""A recording inference FAILED on must not look like one never analyzed.

A 217-file overnight run lost 29 files to host-memory exhaustion. Those files
write no sidecar, and the Audio list derives its badge from the sidecar — so
they rendered as plain names, exactly like a file nobody had got to yet. The
completion dialog said "29 file(s) failed" and then elided the list, so there
was no way to find them afterwards. The run looked complete and 13% of it was
missing.

Three states, not two: analyzed with counts, analyzed and empty, and failed.
"""
import pytest

pytest.importorskip("PyQt5")

from PyQt5.QtCore import Qt  # noqa: E402
from PyQt5.QtWidgets import (  # noqa: E402
    QApplication, QListWidget, QListWidgetItem,
)


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


@pytest.fixture
def win(qapp):
    from fnt.usv.mad_pyqt import MADMainWindow

    class W:
        _apply_file_row = MADMainWindow._apply_file_row

        def __init__(self):
            self._file_errors = {}
            self._file_run_info = {}

    return W()


def _row(win, base, counts):
    item = QListWidgetItem(base)
    win._apply_file_row(item, base, counts)
    return item


def test_a_failed_file_is_marked_and_explained(win):
    from fnt.usv.mad_pyqt import _ROLE_FILE_ERROR
    win._file_errors = {"a.wav": "Unable to allocate 572. MiB"}
    it = _row(win, "a.wav", None)
    assert it.data(_ROLE_FILE_ERROR) == "Unable to allocate 572. MiB"
    tip = it.toolTip()
    assert "FAILED" in tip
    assert "572" in tip                       # the actual reason, not a shrug
    assert "Re-run inference" in tip


def test_a_never_analyzed_file_carries_no_error(win):
    from fnt.usv.mad_pyqt import _ROLE_FILE_ERROR
    it = _row(win, "fresh.wav", None)
    assert it.data(_ROLE_FILE_ERROR) is None


def test_an_analyzed_empty_file_is_not_a_failure(win):
    """(0, 0, 0) means inference ran and found nothing — a real result."""
    from fnt.usv.mad_pyqt import _ROLE_FILE_ERROR
    it = _row(win, "quiet.wav", (0, 0, 0))
    assert it.data(_ROLE_FILE_ERROR) is None
    assert "0 detections" in it.text()


def test_a_file_with_counts_is_not_a_failure(win):
    from fnt.usv.mad_pyqt import _ROLE_FILE_ERROR
    it = _row(win, "busy.wav", (3, 12, 1))
    assert it.data(_ROLE_FILE_ERROR) is None
    assert "(3, 12, 1)" in it.text()


def test_the_error_survives_a_file_that_also_has_stale_counts(win):
    """A file that failed THIS run may still carry counts from an earlier one.
    The failure is the newer fact and must win."""
    from fnt.usv.mad_pyqt import _ROLE_FILE_ERROR
    win._file_errors = {"b.wav": "boom"}
    it = _row(win, "b.wav", (1, 2, 3))
    assert it.data(_ROLE_FILE_ERROR) == "boom"
    assert "FAILED" in it.toolTip()


def test_clearing_the_error_map_restores_a_normal_row(win):
    from fnt.usv.mad_pyqt import _ROLE_FILE_ERROR
    win._file_errors = {"c.wav": "boom"}
    assert _row(win, "c.wav", None).data(_ROLE_FILE_ERROR) == "boom"
    win._file_errors = {}
    assert _row(win, "c.wav", None).data(_ROLE_FILE_ERROR) is None


def test_a_missing_error_map_is_not_an_error(win):
    """The attribute may not exist yet on a fresh window."""
    from fnt.usv.mad_pyqt import _ROLE_FILE_ERROR
    del win._file_errors
    it = _row(win, "d.wav", (1, 0, 0))
    assert it.data(_ROLE_FILE_ERROR) is None
