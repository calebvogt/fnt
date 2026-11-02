"""File > Export...: the dialog in front of mad_export.

Built offscreen without MAD's main window: the model list fills from a
background read of the chosen recordings (most common first, with counts), the
options become an ExportOptions, ticking images ticks labelled calls (and
unticking those unticks images), the choices are remembered, and an export
runs on a worker thread and lands its tables while the window stays live.

Runs under pytest, or directly.
"""
import os
import tempfile
import time

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pytest  # noqa: E402

pytest.importorskip("PyQt5")
pytest.importorskip("h5py")

from PyQt5.QtCore import QSettings  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from fnt.usv.mad_export_dialog import MADExportDialog  # noqa: E402
from fnt.usv.usv_detector import mad_export as mx  # noqa: E402

from test_mad_export import A1, A2, B1, TRIAL, _run_folder  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def _wait(cond, timeout=30.0):
    end = time.time() + timeout
    while not cond() and time.time() < end:
        QApplication.processEvents()
        time.sleep(0.02)
    QApplication.processEvents()
    return cond()


def _settings():
    return QSettings(os.path.join(tempfile.mkdtemp(), "mad.ini"), QSettings.IniFormat)


def test_the_model_list_and_the_options(qapp):
    root = _run_folder()
    wavs = [os.path.join(root, TRIAL, n) for n in (A1, A2, B1)]
    dlg = MADExportDialog(None, wavs=wavs, wavs_label="Project recordings",
                          settings=_settings())
    assert dlg.rb_list.isChecked() and "(3)" in dlg.rb_list.text()
    assert _wait(lambda: dlg.combo_model.count() == 3)
    assert dlg.combo_model.itemText(0) == "Most common: m_a (2)"
    assert dlg.combo_model.itemData(2) == "m_b"
    assert dlg.options() is None                     # no output folder yet
    dlg.edit_out.setText(os.path.join(root, "out"))
    dlg.combo_model.setCurrentIndex(2)
    opts = dlg.options()
    assert (opts.wavs, opts.folders, opts.model) == (wavs, [], "m_b")
    assert opts.detections and not opts.labels and not opts.harmonics
    assert opts.incremental and opts.settle_s == pytest.approx(600.0)
    dlg.chk_images.setChecked(True)
    assert dlg.chk_labels.isChecked()
    dlg.chk_labels.setChecked(False)
    assert not dlg.chk_images.isChecked()
    dlg.close()


def test_an_export_runs_on_a_worker_and_the_choices_are_kept(qapp):
    root = _run_folder()
    out = os.path.join(root, "out")
    s = _settings()
    dlg = MADExportDialog(None, settings=s)
    assert not dlg.rb_list.isEnabled() and dlg.rb_folders.isChecked()
    dlg.list_folders.addItem(root)
    dlg._inputs_changed()
    dlg.edit_out.setText(out)
    assert _wait(lambda: dlg.combo_model.count() == 3)
    dlg._run()
    assert dlg._export_worker is not None and not dlg.btn_export.isEnabled()
    assert _wait(lambda: dlg._export_worker is None)
    assert dlg.btn_export.isEnabled()
    assert "Exported 2 recording(s)" in dlg.lbl_status.text()
    assert os.path.isfile(os.path.join(out, mx.DETECTIONS_CSV))
    assert "Wrote" in dlg.log_view.toPlainText()
    dlg.close()
    again = MADExportDialog(None, settings=s)
    assert again.edit_out.text() == out
    assert [again.list_folders.item(i).text()
            for i in range(again.list_folders.count())] == [root]
    again.close()


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
