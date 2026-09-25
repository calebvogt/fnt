"""Off-animal intervals: holes inside a tag's deployment.

On VT-P002 Hestia's head cap came off in the Z7 nest at ~20:52 on 9/23 and was
re-glued at 10:38 the next morning. Her tag reported the whole time, so the
data held 13.5 h of a tag lying still under her name - and Start/Stop, which
bound ONE continuous deployment, had no way to carve it out. These intervals
do, everywhere Start/Stop already trims.

Runs under pytest, or directly (``python test_off_animal_intervals.py``).
"""
import os
import sqlite3
import sys
import tempfile
import types

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import pandas as pd  # noqa: E402
import pytz  # noqa: E402
from PyQt5.QtWidgets import QApplication, QMessageBox  # noqa: E402

app = QApplication.instance() or QApplication(sys.argv)

from fnt.uwb import identities as I  # noqa: E402
from fnt.uwb import uwb_preprocessing_pyqt as U  # noqa: E402

W = U.UWBQuickVisualizationWindow
TZ = pytz.timezone("America/Denver")
HESTIA = {
    'sex': 'F', 'identity': '9806', 'name': 'Hestia',
    'start_time': '2026-09-16 11:00:00.000', 'start_mode': 'manual',
    'stop_time': '2026-09-24 11:10:05.816', 'stop_mode': 'auto',
    'off_intervals': [{'start': '2026-09-23 20:52:00.000',
                       'stop': '2026-09-24 10:38:00.000',
                       'note': 'head cap off in Z7 nest'}],
}


def _holder(identities):
    h = types.SimpleNamespace(tag_identities=identities)
    for name in ('_tag_window', '_tag_off_intervals', '_trim_tag',
                 '_trim_to_tag_window'):
        setattr(h, name, getattr(W, name).__get__(h, types.SimpleNamespace))
    h._localize_wall_time = W._localize_wall_time
    return h


def _frame(times):
    ts = pd.to_datetime(pd.Series(times)).dt.tz_localize(TZ)
    return pd.DataFrame({'Timestamp': ts, 'x': range(len(ts))})


# -- pure helpers ---------------------------------------------------------- #

def test_parse_tolerates_old_and_malformed_records():
    assert I.off_intervals({}) == []
    assert I.off_intervals(None) == []
    assert I.off_intervals({'off_intervals': [
        {'start': 'a', 'stop': ''}, 'junk', {'start': 'a', 'stop': 'b'}]}) == \
        [{'start': 'a', 'stop': 'b', 'note': ''}]


def test_merge_unions_overlaps_and_drops_inverted():
    assert I.merge_intervals([(5, 9), (1, 3), (2, 4), (8, 12), (20, 20), (7, 6)]) \
        == [(1, 4), (5, 12)]


def test_inside_is_inclusive():
    m = I.inside_intervals([0, 1, 2, 3, 4], [(1, 3)])
    assert list(m) == [False, True, True, True, False]


# -- trimming --------------------------------------------------------------- #

def test_hestia_night_is_excluded_and_counted_separately():
    h = _holder({0x1F: HESTIA})
    df = _frame(['2026-09-23 20:51:59', '2026-09-23 20:52:00',   # edge: kept, dropped
                 '2026-09-24 03:00:00',                           # tag in the nest
                 '2026-09-24 10:38:00', '2026-09-24 10:38:01',   # edge: dropped, kept
                 '2026-09-16 10:59:59'])                          # before Start
    out, n_window, n_off = h._trim_tag(df, 0x1F, TZ)
    assert list(out['x']) == [0, 4]
    assert (n_window, n_off) == (1, 3)
    # The summed form every other caller uses.
    out2, n = h._trim_to_tag_window(df, 0x1F, TZ)
    assert n == 4 and list(out2['x']) == [0, 4]


def test_hole_outside_the_window_counts_as_window_trim():
    info = dict(HESTIA, off_intervals=[{'start': '2026-09-10 00:00:00.000',
                                        'stop': '2026-09-17 00:00:00.000'}])
    h = _holder({0x1F: info})
    df = _frame(['2026-09-16 09:00:00', '2026-09-16 12:00:00', '2026-09-18 00:00:00'])
    out, n_window, n_off = h._trim_tag(df, 0x1F, TZ)
    assert list(out['x']) == [2]
    assert (n_window, n_off) == (1, 1)


def test_tag_without_intervals_is_untouched():
    info = {k: v for k, v in HESTIA.items() if k != 'off_intervals'}
    info.update(start_mode='auto')
    h = _holder({0x1F: info})
    df = _frame(['2026-09-24 03:00:00'])
    out, n_window, n_off = h._trim_tag(df, 0x1F, TZ)
    assert len(out) == 1 and (n_window, n_off) == (0, 0)


# -- export scope count ------------------------------------------------------ #

def test_export_scope_subtracts_holes_once():
    tmp = tempfile.mkdtemp()
    db = os.path.join(tmp, 't_FNT_analysis', 't_indexed.sqlite')
    os.makedirs(os.path.dirname(db))
    ms = lambda s: int(TZ.localize(pd.Timestamp(s).to_pydatetime()).timestamp() * 1000)
    rows = [(0x1F, ms(t)) for t in pd.date_range('2026-09-23 20:00', periods=20, freq='1h')]
    con = sqlite3.connect(db)
    con.execute("CREATE TABLE T (shortid INTEGER, timestamp INTEGER)")
    con.executemany("INSERT INTO T VALUES (?, ?)", rows)
    con.commit(); con.close()

    # Two overlapping holes over 21:00..02:00 -> 6 hourly rows, counted once.
    info = {'sex': 'F', 'identity': '1', 'start_mode': 'auto', 'stop_mode': 'auto',
            'off_intervals': [{'start': '2026-09-23 21:00:00.000', 'stop': '2026-09-24 00:30:00.000'},
                              {'start': '2026-09-24 00:00:00.000', 'stop': '2026-09-24 02:00:00.000'}]}
    h = _holder({0x1F: info})
    h.tag_checkboxes = {0x1F: types.SimpleNamespace(isChecked=lambda: True)}
    h.table_name = 'T'
    h.db_path = os.path.join(tmp, 't.sqlite')
    h.combo_timezone = types.SimpleNamespace(currentText=lambda: 'America/Denver')
    h.current_indexed_db = lambda: db
    h.log_message = lambda m: None
    n_tags, total = W.export_scope_points.__get__(h, types.SimpleNamespace)()
    assert (n_tags, total) == (1, 20 - 6)


# -- dialogs ---------------------------------------------------------------- #

def test_identity_dialog_round_trips_intervals():
    ranges = {0x1F: {'start': '2026-09-16 11:00:00.000', 'end': '2026-09-24 16:50:25.000'},
              0x0B: {'start': '2026-09-15 17:07:00.000', 'end': '2026-09-24 16:50:27.000'}}
    dlg = U.IdentityAssignmentDialog([0x1F, 0x0B],
                                     {0x1F: HESTIA, 0x0B: {'sex': 'M', 'identity': '9831'}},
                                     ranges)
    assert dlg.off_buttons[0x1F].text().startswith("Off-animal (1)")
    assert dlg.off_buttons[0x0B].text().startswith("Off-animal")
    out = dlg.get_identities()
    assert out[0x1F]['off_intervals'] == HESTIA['off_intervals']
    # A tag with none keeps exactly the record it had before this feature.
    assert 'off_intervals' not in out[0x0B]


def test_interval_editor_rows_and_validation():
    dlg = U.OffAnimalIntervalsDialog("HexID 1F", HESTIA['off_intervals'],
                                     ('2026-09-16 11:00:00.000', None))
    assert dlg.intervals() == HESTIA['off_intervals']
    dlg._add_row()                         # starts where the last one ended
    new = dlg.intervals()[-1]
    assert new['start'] == HESTIA['off_intervals'][0]['stop']

    # An inverted row is refused, not saved.
    row = dlg.table.rowCount() - 1
    a = dlg.table.cellWidget(row, 0)
    dlg.table.cellWidget(row, 1).setDateTime(a.dateTime().addSecs(-60))
    warned = []
    orig = QMessageBox.warning
    QMessageBox.warning = staticmethod(lambda *a, **k: warned.append(a))
    try:
        dlg.accept()
    finally:
        QMessageBox.warning = orig
    assert warned and dlg.result() != U.QDialog.Accepted


if __name__ == "__main__":
    for name, fn in sorted(globals().items()):
        if name.startswith("test_"):
            fn()
            print(f"ok  {name}")
    print("all off-animal interval tests passed")
