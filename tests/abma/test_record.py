"""The replayable run record: what it must keep, and what it must survive."""
from __future__ import annotations

import numpy as np
import pytest

from fnt.abma.core.record import (
    RunRecord, VALUE_FIELDS, STATE_FIELDS, RECORD_SCHEMA_VERSION, layout,
)


def _frame(t: float, n: int = 3, **extra) -> dict:
    frame = {
        "elapsed": t, "day": int(t // 86400) + 1, "hour": (t / 3600) % 24,
        "is_day": True,
        "x": np.full(n, t), "y": np.arange(n, dtype=float),
        "heading": np.zeros(n),
        "drive_social": np.full(n, 0.5), "drive_wander": np.full(n, 0.25),
        "energy": np.full(n, 80.0), "activity": np.arange(n),
        "alive": np.ones(n, bool),
    }
    frame.update(extra)
    return frame


def _filled(n_frames: int = 20, n: int = 3) -> RunRecord:
    rec = RunRecord(trial_id="S001", n_agents=n, frame_interval_s=300.0)
    for k in range(n_frames):
        rec.append(_frame(k * 300.0, n))
    return rec


def test_round_trips_through_disk(tmp_path):
    rec = _filled()
    path = rec.save(str(tmp_path / "record_S001.npz"))
    back = RunRecord.load(path)
    assert len(back) == len(rec)
    assert back.trial_id == "S001"
    assert np.allclose(back.series(0, "drive_social"),
                       rec.series(0, "drive_social"))
    assert np.allclose(back.times(), rec.times())


def test_keeps_the_drive_decomposition():
    """The whole point: a trajectory plus the motivation behind it."""
    rec = _filled()
    assert np.allclose(rec.series(1, "drive_social"), 0.5)
    assert np.allclose(rec.series(1, "drive_wander"), 0.25)
    # a drive this configuration never used is present and zero, so the record
    # has the same columns either way
    assert np.allclose(rec.series(1, "drive_territory"), 0.0)


def test_frame_reads_back_in_the_shape_the_view_consumes():
    fr = _filled().frame(4)
    assert fr["elapsed"] == pytest.approx(1200.0)
    assert len(fr["x"]) == 3
    assert set(VALUE_FIELDS + STATE_FIELDS) <= set(fr)


def test_unknown_field_is_rejected_loudly():
    with pytest.raises(KeyError):
        _filled().series(0, "not_a_field")


def test_decimation_keeps_the_whole_run_not_a_prefix():
    """A full archive must get coarser, not stop early — the end of an
    experiment is usually the part you care about."""
    rec = RunRecord(trial_id="S", n_agents=2, frame_interval_s=60.0,
                    cap_bytes=4096)
    for k in range(4000):
        rec.append(_frame(k * 60.0, 2))
    total = 3999 * 60.0
    assert rec.decimation > 1
    assert rec.sample_interval_s() == 60.0 * rec.decimation
    # spans the run to within one (coarsened) sample of the end, rather than
    # stopping partway through
    assert rec.times()[-1] >= total - rec.sample_interval_s()
    assert rec.times()[0] < rec.sample_interval_s()
    assert rec.nbytes() < 10 * 4096


def test_agent_axis_grows_when_the_roster_does():
    rec = RunRecord(trial_id="S", n_agents=2, frame_interval_s=300.0)
    rec.append(_frame(0.0, 2))
    rec.append(_frame(300.0, 5))          # a protocol event added three
    assert rec.n_agents == 5
    assert len(rec.frame(1)["x"]) == 5
    # the earlier frame keeps its shape, padded rather than corrupted
    assert np.allclose(rec.frame(0)["x"][2:], 0.0)


def test_old_archives_load_when_fields_are_added_later(tmp_path):
    """Fields are looked up by name, so a record written before a column
    existed still opens — that is what `self-describing` buys."""
    rec = _filled()
    path = rec.save(str(tmp_path / "old.npz"))
    with np.load(path) as z:
        values, states, frames, meta = (z["values"], z["states"], z["frames"],
                                        z["meta"])
    import json
    got = json.loads(str(meta))
    # pretend this archive predates the last two value fields
    keep = len(VALUE_FIELDS) - 2
    got["layout"]["value_fields"] = VALUE_FIELDS[:keep]
    np.savez_compressed(tmp_path / "trimmed.npz", values=values[:, :, :keep],
                        states=states, frames=frames,
                        meta=np.array(json.dumps(got)))
    back = RunRecord.load(str(tmp_path / "trimmed.npz"))
    assert np.allclose(back.series(0, "drive_social"),
                       rec.series(0, "drive_social"))
    assert np.allclose(back.series(0, VALUE_FIELDS[-1]), 0.0)


def test_a_newer_schema_refuses_rather_than_misreads(tmp_path):
    import json
    rec = _filled()
    path = rec.save(str(tmp_path / "future.npz"))
    with np.load(path) as z:
        values, states, frames, meta = (z["values"], z["states"], z["frames"],
                                        z["meta"])
    got = json.loads(str(meta))
    got["layout"]["schema_version"] = RECORD_SCHEMA_VERSION + 1
    np.savez_compressed(tmp_path / "future2.npz", values=values, states=states,
                        frames=frames, meta=np.array(json.dumps(got)))
    with pytest.raises(ValueError, match="newer than this build"):
        RunRecord.load(str(tmp_path / "future2.npz"))


def test_layout_is_self_describing():
    got = layout()
    assert got["schema_version"] == RECORD_SCHEMA_VERSION
    assert got["value_fields"] == VALUE_FIELDS
    assert got["state_fields"] == STATE_FIELDS
