"""'Add every smoothing method to the Smoothed CSV'.

The export can write each smoothing method's coordinates beside
smoothed_x/smoothed_y (sg_, rollmean_, rollmedian_, ewma_ x/y), so an analysis
can compare smoothers without re-exporting. Guarded here: every column is the
same method run through the same code as the primary, the four are always all
written, and the setting and each column's parameters reach fnt_config.json.

Runs under pytest, or directly.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from fnt.uwb import uwb_preprocessing_pyqt as U  # noqa: E402

_APP = []


@pytest.fixture
def window():
    if not _APP:
        _APP.append(QApplication.instance() or QApplication([]))
    w = U.UWBQuickVisualizationWindow()
    w.db_path = os.path.join("C:", os.sep, "nowhere", "T000_fake.sqlite")
    yield w
    w.close()


def _tracks():
    rng = np.random.default_rng(1)
    frames = []
    for tag, n in ((7, 400), (9, 250)):
        ms = np.cumsum(rng.integers(300, 2500, n))
        frames.append(pd.DataFrame({
            'shortid': tag,
            'timestamp': 1730899200000 + ms,
            'location_x': np.cumsum(rng.normal(0, 0.05, n)),
            'location_y': np.cumsum(rng.normal(0, 0.05, n)),
        }))
    df = pd.concat(frames, ignore_index=True)
    df['Timestamp'] = pd.to_datetime(df['timestamp'], unit='ms', utc=True)
    return df


def _load(w, smoothing, window, units, all_smoothing=True):
    w._pending_config_override = {
        'timezone': 'US/Mountain', 'export_all_smoothing': all_smoothing,
        'preview': {'smoothing': smoothing, 'window': window,
                    'window_units': units}}
    w.load_config_if_exists()
    w._sync_export_from_preview()


@pytest.mark.parametrize("units", ["Samples", "Seconds"])
def test_each_column_is_that_method_run_as_primary(window, units):
    _load(window, "Rolling Average", 5, units)
    df = _tracks()
    out = window.add_alt_smoothing_columns(df.copy(), window=5, units=units)
    for method, prefix in U.ALT_SMOOTHING_METHODS:
        ref = window.apply_smoothing_to_data(df.copy(), method, window=5,
                                             units=units)
        np.testing.assert_allclose(out[f'{prefix}_x'], ref['smoothed_x'])
        np.testing.assert_allclose(out[f'{prefix}_y'], ref['smoothed_y'])
    # The inputs and the row order are untouched.
    pd.testing.assert_frame_equal(out[df.columns], df)


def test_rollmean_equals_primary_and_methods_differ(window):
    df = _tracks()
    prim = window.apply_smoothing_to_data(df.copy(), "Rolling Average",
                                          window=5, units="Samples")
    out = window.add_alt_smoothing_columns(prim.copy(), window=5,
                                           units="Samples")
    np.testing.assert_allclose(out['rollmean_x'], out['smoothed_x'])
    assert set(U.ALT_SMOOTHING_COLUMNS) <= set(out.columns)
    assert not np.allclose(out['sg_x'], out['rollmean_x'])
    assert not np.allclose(out['ewma_x'], out['rollmedian_x'])
    # Per tag: no smoother reaches across shortids.
    tag9 = out[out.shortid == 9]
    alone = window.add_alt_smoothing_columns(
        df[df.shortid == 9].copy(), window=5, units="Samples")
    np.testing.assert_allclose(tag9['sg_x'], alone['sg_x'])


def test_config_records_the_setting_and_parameters(window):
    _load(window, "Rolling Average", 5, "Samples")
    assert window.chk_export_all_smoothing.isChecked()
    cfg = window.get_config_dict()
    assert cfg['export_all_smoothing'] is True
    rec = cfg['all_smoothing_columns']
    # Every method uses the export's window and units now.
    assert rec['sg_x/sg_y']['window'] == window.spin_rolling_window.value()
    assert rec['sg_x/sg_y']['units'] == window.combo_window_units.currentText()
    assert rec['ewma_x/ewma_y']['span'] == window.spin_rolling_window.value()
    assert rec['rollmean_x/rollmean_y']['units'] == \
        window.combo_window_units.currentText()

    _load(window, "Rolling Average", 5, "Samples", all_smoothing=False)
    cfg = window.get_config_dict()
    assert cfg['export_all_smoothing'] is False
    assert cfg['all_smoothing_columns'] is None


def test_readers_skip_the_extra_columns():
    assert not U._not_alt_smoothing_column('sg_x')
    assert U._not_alt_smoothing_column('smoothed_x')
    assert set(U.ALT_SMOOTHING_COLUMNS) <= set(
        U.UWBQuickVisualizationWindow._SMOOTHED_NUMERIC_COLS)


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-q"]))
