"""The test suite must never write the user's real settings.

See tests/conftest.py: a GUI test run once overwrote MAD's saved window layout
in the registry with an offscreen window's, which opened the next real session
with a dead band beside the spectrogram.
"""
import os

import pytest

pytest.importorskip("PyQt5")

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")


def test_qsettings_are_redirected_during_tests():
    from PyQt5.QtCore import QSettings
    s = QSettings("FNT", "MAD")
    assert s.format() == QSettings.IniFormat
    assert "fnt_test_settings_" in s.fileName(), s.fileName()


def test_mad_binds_the_redirected_class():
    """mad_pyqt imports QSettings by name; it must get the redirect."""
    import fnt.usv.mad_pyqt as M
    assert getattr(M.QSettings, "_fnt_isolated", False)


def test_a_mad_window_cannot_touch_the_registry():
    from PyQt5.QtWidgets import QApplication
    import fnt.usv.mad_pyqt as M
    app = QApplication.instance() or QApplication([])
    w = M.MADMainWindow()
    assert "fnt_test_settings_" in w._settings.fileName()
    w.deleteLater()
    app.processEvents()
