"""Animation 'Day N' counts from the trial's first Start, not the database's
first date. VC-T010's database still held VC-T007's and VC-T008's dates
(3/11-3/20, 4/29), so its ten days were written as Day12..Day21."""
import pandas as pd

from fnt.uwb.uwb_preprocessing_pyqt import UWBQuickVisualizationWindow as W

DB_DAYS = ([f"2026-03-{d}" for d in range(11, 21)] + ["2026-04-29"]
           + [f"2026-06-{d:02d}" for d in range(6, 16)])


def test_days_before_the_first_start_are_not_trial_days():
    nums = W.trial_day_numbers(DB_DAYS, pd.Timestamp("2026-06-06 12:48"))
    assert nums["2026-06-06"] == 1 and nums["2026-06-15"] == 10
    assert "2026-03-11" not in nums and "2026-04-29" not in nums


def test_without_a_start_every_date_is_numbered():
    nums = W.trial_day_numbers(DB_DAYS)
    assert nums["2026-03-11"] == 1 and nums["2026-06-06"] == 12


def test_a_start_after_every_date_changes_nothing():
    assert W.trial_day_numbers(["2026-06-06"], "2026-07-01") == {"2026-06-06": 1}


class _Stub:
    _first_trial_start = W._first_trial_start

    def __init__(self, ids):
        self.tag_identities = ids


def test_first_start_is_the_earliest_and_needs_every_tag_limited():
    ids = {17: dict(start_mode="manual", start_time="2026-06-06 12:48:00"),
           12: dict(start_mode="manual", start_time="2026-06-11 11:08:00")}
    assert _Stub(ids)._first_trial_start() == pd.Timestamp("2026-06-06 12:48")
    ids[30] = dict(start_mode="auto", start_time="2026-06-06 12:52:00")
    assert _Stub(ids)._first_trial_start() is None
    assert _Stub({})._first_trial_start() is None
