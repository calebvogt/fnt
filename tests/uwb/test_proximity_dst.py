"""Proximity bouts across the end of daylight saving time.

VC-T014 ran across 2026-11-01, when 01:00-02:00 local happens twice. Flooring
the local timestamps to 1 s raised "Cannot infer dst time from 2026-11-01
01:00:00", and the export carried on without its SocialOverlapBouts CSV.

Runs under pytest, or directly.
"""
import numpy as np
import pandas as pd
import pytest

from fnt.uwb.proximity_detection import detect_proximity_bouts


def _two_tags_across_dst_end():
    # 06:30 UTC to 08:30 UTC = 00:30 MDT to 01:30 MST: the repeated hour inside
    utc = pd.date_range("2026-11-01 06:30", "2026-11-01 08:30", freq="1s", tz="UTC")
    local = utc.tz_convert("US/Mountain")
    rows = []
    for tag, dx in ((1, 0.0), (2, 0.1)):              # 0.1 m apart: always close
        rows.append(pd.DataFrame({"Timestamp": local, "shortid": tag,
                                  "smoothed_x": 1.0 + dx, "smoothed_y": 1.0}))
    return pd.concat(rows, ignore_index=True)


def test_bouts_across_the_repeated_hour():
    df = _two_tags_across_dst_end()
    _, bouts = detect_proximity_bouts(df, threshold=0.3, gap_s=5)
    assert len(bouts) >= 1
    # the pair is together for the whole two hours, not one hour less or more
    total = bouts["duration_s"].sum()
    assert abs(total - 2 * 3600) < 5, total


def test_the_bins_are_whole_utc_seconds():
    df = _two_tags_across_dst_end()
    df["Timestamp"] = df["Timestamp"] + pd.Timedelta(milliseconds=400)
    _, bouts = detect_proximity_bouts(df, threshold=0.3, gap_s=5)
    assert abs(bouts["duration_s"].sum() - 2 * 3600) < 5


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
