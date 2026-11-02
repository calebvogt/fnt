"""The read-back check on analysis CSVs: missing values are not corruption.

A day with no moonrise (or moonset) is a NaT in the daylight table and an empty
cell on disk. The check compared the frame's text ('NaT') with the file's
(''), called the empty cell a value "never written", and aborted the export --
on every trial long enough to contain such a day. Real corruption must still
fail.

Runs under pytest, or directly.
"""
import os

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import pandas as pd  # noqa: E402
import pytest  # noqa: E402
from PyQt5.QtWidgets import QApplication  # noqa: E402

from fnt.uwb.uwb_preprocessing_pyqt import UWBQuickVisualizationWindow  # noqa: E402

_APP = []


@pytest.fixture
def window():
    if not _APP:
        _APP.append(QApplication.instance() or QApplication([]))
    w = UWBQuickVisualizationWindow()
    yield w
    w.close()


def _daylight(n=5):
    tz = "US/Mountain"
    day = pd.date_range("2026-09-15", periods=n, freq="D", tz=tz)
    rise = pd.Series(day + pd.Timedelta(hours=20, minutes=13))
    rise.iloc[2] = pd.NaT                       # no moonrise that day
    return pd.DataFrame({"Day": [d.strftime("%Y-%m-%d") for d in day],
                         "day_length_h": [12.1, 12.0, 11.9, 11.9, 11.8][:n],
                         "moonrise": rise,
                         "phase": ["waxing"] * n,
                         "note": ["a", None, "b", "a", "b"][:n]})


def test_a_day_without_moonrise_verifies(window, tmp_path):
    df = _daylight()
    p = tmp_path / "x_daylight.csv"
    df.to_csv(p, index=False)
    ok, problems = window.verify_written_frame(df, str(p), ("day_length_h",))
    assert ok, problems


def test_a_missing_time_in_a_long_column_verifies(window, tmp_path):
    t = pd.Series(pd.date_range("2026-09-15", periods=200, freq="min", tz="US/Mountain"))
    t.iloc[50] = pd.NaT
    df = pd.DataFrame({"timestamp": t, "v": range(200)})
    p = tmp_path / "x.csv"
    df.to_csv(p, index=False)
    ok, problems = window.verify_written_frame(df, str(p), ("v",))
    assert ok, problems


def test_a_stray_label_still_fails(window, tmp_path):
    df = _daylight()
    p = tmp_path / "x_daylight.csv"
    df.to_csv(p, index=False)
    p.write_text(p.read_text(encoding="utf-8").replace("waxing", "wax°ng", 1),
                 encoding="utf-8")
    ok, problems = window.verify_written_frame(df, str(p), ("day_length_h",))
    assert not ok and any("phase" in s for s in problems)


def test_a_value_where_none_was_written_still_fails(window, tmp_path):
    df = _daylight()
    p = tmp_path / "x_daylight.csv"
    df.to_csv(p, index=False)
    lines = p.read_text().splitlines()
    lines[2] = lines[2][:-1] + "pJ"            # 'note' was empty on that row
    p.write_text("\n".join(lines) + "\n")
    ok, _ = window.verify_written_frame(df, str(p), ("day_length_h",))
    assert not ok


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
