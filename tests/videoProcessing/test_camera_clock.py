"""The burnt-in clock reader and the clock-following grid encode."""

import datetime

import cv2
import numpy as np
import pytest

from fnt.videoProcessing import camera_clock as cc
from fnt.videoProcessing.camera_grid_encode import (
    EncodeSettings, GridLayout, _corr_expr, _synced_runs, _shown_span,
    build_chunk_command)
from fnt.videoProcessing.camera_timeline import CameraTrack, Segment

SEARCH = (0, 0, 520, 60)


def render(text, bright_background=False):
    """A crop like the recorder's: white text, dark outline, fixed font."""
    img = np.full((60, 520), 220 if bright_background else 60, np.uint8)
    x = 20
    for ch in text:
        if ch == " ":
            x += 14
            continue
        cv2.putText(img, ch, (x, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 0, 6)
        cv2.putText(img, ch, (x, 45), cv2.FONT_HERSHEY_SIMPLEX, 1.0, 255, 2)
        (w, _), _ = cv2.getTextSize(ch, cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2)
        x += w + 6
    return img


def learned_reader():
    base = datetime.datetime(2026, 2, 7, 0, 0, 30)
    samples = []
    for h in range(24):
        t = base + datetime.timedelta(hours=h, minutes=7 * h)
        samples.append((render(t.strftime("%Y/%m/%d %H:%M:%S")), t.strftime("%Y/%m/%d %H:%M")))
    return cc.ClockReader.learn(samples, SEARCH)


def crop(reader, img):
    x, y, w, h = reader.box
    return img[y:y + h, x:x + w]


def test_reads_every_digit_after_learning_from_the_timeline():
    r = learned_reader()
    for text in ["2026/02/07 14:15:00", "2026/12/31 23:59:59", "2026/11/08 10:17:22"]:
        t, score = r.read(crop(r, render(text)))
        assert t == datetime.datetime.strptime(text, "%Y/%m/%d %H:%M:%S")
        assert score > 0.8


def test_reads_on_a_bright_background():
    r = learned_reader()
    t, _ = r.read(crop(r, render("2026/02/09 23:15:01", bright_background=True)))
    assert t == datetime.datetime(2026, 2, 9, 23, 15, 1)


def test_learning_needs_every_digit():
    samples = [(render("2026/02/07 00:00:30"), "2026/02/07 00:00")]
    with pytest.raises(ValueError):
        cc.ClockReader.learn(samples, SEARCH)


def test_reader_round_trips_through_json(tmp_path):
    r = learned_reader()
    r.save(tmp_path / "font.json")
    r2 = cc.ClockReader.load(tmp_path / "font.json")
    img = crop(r2, render("2026/02/07 14:15:09"))
    assert r2.read(img)[0] == datetime.datetime(2026, 2, 7, 14, 15, 9)


def _segments(n=3, dur=3600.0, start=datetime.datetime(2026, 2, 7)):
    segs = []
    for i in range(n):
        s = Segment(f"seg{i}.mp4", "20260207000000", i, dur)
        s.start = start + datetime.timedelta(seconds=i * dur)
        s.end = s.start + datetime.timedelta(seconds=dur)
        segs.append(s)
    return segs


def _probes(segs, f, every=60, bad=()):
    out = []
    for s in segs:
        for k in range(0, int(s.duration), every):
            t = s.start + datetime.timedelta(seconds=k)
            d = f(s, k)
            if (s.path, k) in bad:
                d += 1.0          # a misread whole second
            out.append({"path": s.path, "timeline": t, "delta": d})
    return out


def test_sync_fits_a_step_and_drift_per_file_and_ignores_misreads():
    segs = _segments()
    truth = {"seg0.mp4": (0.5, 1e-5), "seg1.mp4": (1.7, -2e-5), "seg2.mp4": (-0.3, 0.0)}
    f = lambda s, k: truth[s.path][0] + truth[s.path][1] * k
    sync = cc.ClockSync.fit(_probes(segs, f, bad={("seg1.mp4", 600)}), segs)
    for s in segs:
        for k in (0, 1800, 3500):
            when = s.start + datetime.timedelta(seconds=k)
            assert sync.delta(when) == pytest.approx(f(s, k), abs=0.01)


def test_sync_carries_a_neighbour_across_an_unread_file():
    segs = _segments()
    probes = [p for p in _probes(segs, lambda s, k: 1.0) if p["path"] != "seg1.mp4"]
    sync = cc.ClockSync.fit(probes, segs)
    assert sync.delta(segs[1].start + datetime.timedelta(seconds=10)) == pytest.approx(1.0)


def _track(sync=None):
    t = CameraTrack("cam1", "x", _segments())
    t.clock_sync = sync
    return t


def test_synced_runs_carry_the_correction_from_the_run_start():
    segs = _segments()
    sync = cc.ClockSync([(segs[0].start, segs[0].end, 1.0, 0.0),
                         (segs[1].start, segs[1].end, 2.0, 0.0),
                         (segs[2].start, segs[2].end, 3.0, 0.0)])
    ws = segs[0].start + datetime.timedelta(minutes=30)
    we = ws + datetime.timedelta(hours=1)
    runs = _synced_runs(_track(sync), sync, ws, we, pad=10)
    assert len(runs) == 1
    run = runs[0]
    assert run["offset"] == pytest.approx(-10)
    # the first piece begins at the run's first frame; the step is at seg1
    assert run["corr"][0][:3] == pytest.approx((0, 1810, 1.0))
    assert run["corr"][1][:3] == pytest.approx((1810, 3610 + 10, 2.0))
    t0, t1 = _shown_span(run)
    assert t0 == pytest.approx(0) and t1 == pytest.approx(-10 + run["duration"] + 2.0)


def test_a_file_joined_across_a_timeline_gap_carries_the_gap():
    """An NVR's first file of a day is stamped 00:00:00 whatever the file before midnight reached; the run plays the
    two back to back, so the second file's correction must carry the gap (it was missing: every frame after midnight
    sat 0.3-1.4 s off for the rest of the day)."""
    segs = _segments(n=2)
    gap = 0.4                                        # file 0 ends 0.4 s before file 1's stamp
    segs[0].duration -= gap
    segs[0].end -= datetime.timedelta(seconds=gap)
    sync = cc.ClockSync([(segs[0].start, segs[0].end, 1.0, 0.0), (segs[1].start, segs[1].end, 1.0, 0.0)])
    t = CameraTrack("cam1", "x", segs)
    ws = segs[0].start + datetime.timedelta(minutes=30)
    runs = _synced_runs(t, sync, ws, ws + datetime.timedelta(hours=1), pad=10)
    assert len(runs) == 1 and len(runs[0]["paths"]) == 2          # joined: the gap is under join_tol
    c = runs[0]["corr"]
    played0 = 1810 - gap                                          # file 0 plays this long
    assert c[0][:3] == pytest.approx((0, played0, 1.0))
    assert c[1][:3] == pytest.approx((played0, 3620 - gap, 1.0 + gap))
    # the frame showing clock C sits at C either side of the join: placement = run start + l + corr
    run_start = (ws - datetime.timedelta(seconds=10) - segs[0].start).total_seconds()
    # file 1's first frame: timeline segs[1].start, clock segs[1].start + 1.0 s
    l1 = played0
    placed = run_start + l1 + c[1][2]
    clock = (segs[1].start - segs[0].start).total_seconds() + 1.0
    assert placed == pytest.approx(clock)


def test_the_pad_covers_a_large_clock_error():
    """A camera whose clock runs two minutes behind its file timeline (VC-T010 cam1-2) or an hour (a DST day's 25-hour
    files): the window's footage must still be gathered, so the pad grows to the error."""
    segs = _segments()
    sync = cc.ClockSync([(s.start, s.end, -120.0, 0.0) for s in segs])
    ws = segs[0].start + datetime.timedelta(minutes=30)
    runs = _synced_runs(CameraTrack("cam1", "x", segs), sync, ws, ws + datetime.timedelta(hours=1), pad=10)
    t0, t1 = _shown_span(runs[0])
    assert t0 <= 0 and t1 >= 3600                    # the whole hour is covered


def test_a_25_hour_day_is_placed_by_its_clocks(monkeypatch):
    """Daylight saving ends: the day's files hold 25 h, so chained from the stamp the last one runs an hour into the
    next day. Its files are moved to where their clocks put them; a normal day is left alone."""
    segs = []
    for day, n in (("20261101000000", 5), ("20261102000000", 4)):
        base = datetime.datetime.strptime(day, "%Y%m%d%H%M%S")
        for i in range(n):
            s = Segment(f"{day}_{i}.mp4", day, i, 5 * 3600.0 if day.startswith("20261101") else 6 * 3600.0)
            s.start = base + datetime.timedelta(seconds=i * s.duration)
            s.end = s.start + datetime.timedelta(seconds=s.duration)
            segs.append(s)
    # the clock: 11/1's files from 01:00 on start an hour earlier than chained (the hour 01:00-02:00 came twice)
    true_start = {s.path: s.start - datetime.timedelta(hours=1 if (s.day.startswith("20261101") and s.seq > 0) else 0)
                  for s in segs}
    monkeypatch.setattr(cc, "find_tick", lambda reader, path, seek, fps=30.0: (
        seek + 0.5, true_start[path] + datetime.timedelta(seconds=seek + 0.5), 0.9))
    t = CameraTrack("cam1", "x", segs)
    moved = cc.anchor_overfull_days(None, t)
    assert len(moved) == 4                                          # 11/1's files after the first
    last = [s for s in t.segments if s.path == "20261101000000_4.mp4"][0]
    assert last.end == datetime.datetime(2026, 11, 2)               # ends at midnight, no longer at 01:00
    assert all(s.start == true_start[s.path] for s in t.segments if s.day.startswith("20261102"))


def test_corr_expression_evaluates_to_the_pieces():
    corr = [(0, 100, 1.0, 0.0), (100, 200, 2.0, 0.01)]
    expr = _corr_expr(corr)
    L = lambda v: expr.replace("(PTS-STARTPTS)*TB", str(v))
    env = {"gte": lambda a, b: float(a >= b), "lt": lambda a, b: float(a < b)}
    assert eval(L(50), env) == pytest.approx(1.0)
    assert eval(L(150), env) == pytest.approx(2.0 + 0.01 * 50)
    assert eval(L(-5), env) == pytest.approx(1.0)      # held past the ends
    assert eval(L(500), env) == pytest.approx(2.0 + 0.01 * 400)


def test_a_synced_track_moves_its_frames_in_the_command(tmp_path):
    segs = _segments()
    sync = cc.ClockSync([(s.start, s.end, 1.5, 0.0) for s in segs])
    layout = GridLayout(1, 1, {(0, 0): "cam1"})
    ws = segs[0].start + datetime.timedelta(minutes=30)
    cmd, n = build_chunk_command({"cam1": _track(sync)}, layout, EncodeSettings(),
                                 ws, ws + datetime.timedelta(hours=2),
                                 str(tmp_path / "o.mp4"), work_dir=str(tmp_path))
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "setpts='PTS-STARTPTS+(-10.0000+1.5000)/TB'" in graph
    assert "trim=start=0" in graph


def test_a_run_starting_late_in_a_file_gives_that_file_its_own_input():
    from fnt.videoProcessing.camera_grid_encode import _split_late_start
    run = {"paths": ["a.mp4", "b.mp4", "c.mp4"], "path": "a.mp4", "seek": 3000.0,
           "offset": -10.0, "duration": 1000.0 + 2 * 3600.0,
           "corr": [(0.0, 1000.0, 1.0, 0.0), (1000.0, 8200.0, 2.0, 1e-4)]}
    head, tail = _split_late_start(run, {"a.mp4": 4000.0, "b.mp4": 3600.0, "c.mp4": 3600.0})
    assert head["paths"] == ["a.mp4"] and head["seek"] == 3000.0 and head["duration"] == 1000.0
    assert tail["paths"] == ["b.mp4", "c.mp4"] and tail["seek"] == 0.0
    assert tail["offset"] == pytest.approx(990.0) and tail["duration"] == pytest.approx(7200.0)
    assert head["corr"] == [(0.0, 1000.0, 1.0, 0.0)]
    assert tail["corr"][0] == pytest.approx((0.0, 7200.0, 2.0, 1e-4))
    # a run starting near its first file's beginning stays one input
    assert len(_split_late_start(dict(run, seek=10.0), {"a.mp4": 4000.0})) == 1


# -- the day daylight saving ends (VoleCosm VC-T014, 2026-11-01) ---------------------------------------------------

def _dst_track():
    """Files as the names place them: 10/31 and 11/2 hold 24 h, 11/1 25 h (the repeated hour), 11/3 is the last."""
    from fnt.videoProcessing.camera_timeline import build_timeline
    segs = []
    for day, n, dur in (("20261031000000", 4, 6 * 3600.0), ("20261101000000", 5, 5 * 3600.0),
                        ("20261102000000", 4, 6 * 3600.0), ("20261103000000", 1, 6 * 3600.0)):
        segs += [Segment(f"{day}_{i}.mp4", day, i, dur) for i in range(n)]
    placed, gaps = build_timeline(segs)
    return CameraTrack("cam1", "x", placed, gaps)


def test_the_fallback_day_is_found_and_put_on_one_continuous_axis():
    t = _dst_track()
    assert cc.fallback_days(t) == {datetime.date(2026, 11, 1)}
    c = cc.continuous_track(t, datetime.date(2026, 11, 1))
    by = {s.path: s for s in c.segments}
    # 11/1 runs 25 h in recorded order, 11/2 follows on an hour later, earlier days stay put
    assert by["20261101000000_4.mp4"].end == datetime.datetime(2026, 11, 2, 1)
    assert by["20261102000000_0.mp4"].start == datetime.datetime(2026, 11, 2, 1)
    assert by["20261031000000_0.mp4"].start == datetime.datetime(2026, 10, 31)
    assert all(a.end == b.start for a, b in zip(c.segments, c.segments[1:]))      # no overlap, no hole
    # its own segments: anchoring the original track afterwards does not move them
    t.segments[5].start = datetime.datetime(2000, 1, 1)
    assert by["20261101000000_1.mp4"].start == datetime.datetime(2026, 11, 1, 5)


def _dst_probes(cam, path_of, delta_of, start=datetime.datetime(2026, 11, 1), hours=25):
    out = []
    for k in range(0, hours * 3600, 60):
        t = start + datetime.timedelta(seconds=k)
        d = delta_of(k)
        out.append({"camera": cam, "when": t, "path": path_of(t), "file_time": 0.0, "timeline": t,
                    "clock": t + datetime.timedelta(seconds=d), "delta": d, "score": 0.9})
    return out


def test_readings_after_a_fallback_move_an_hour_on_and_nothing_else_does():
    fall = 7198                                       # the clock goes 01:59:58 -> 01:00 at 02:00 on the axis
    cam1 = _dst_probes("cam1", lambda t: "a.mp4", lambda k: 0.4 if k < fall else 0.4 - 3600)
    cam1[700]["delta"] += 1.0                          # a misread second: left to the fit
    # cam2's clock runs two minutes behind (VC-T010 cam1-2) and one reading misreads its hour: no fallback
    cam2 = _dst_probes("cam2", lambda t: "b.mp4", lambda k: -3720.0 if k == 600 else -120.0)
    # cam3: a lone reading an hour ahead
    cam3 = _dst_probes("cam3", lambda t: "c.mp4", lambda k: 3600.2 if k == 6000 else 0.2)
    out, shifted = cc.unwrap_fallback(cam1 + cam2 + cam3)
    assert shifted == {"cam1": 25 * 60 - 120}
    after = [p for p in out[:len(cam1)] if p["timeline"] >= datetime.datetime(2026, 11, 1, 2)]
    assert all(abs(p["delta"] - 0.4) < 1.5 for p in after)
    p = out[300]                                       # 05:00 on the axis: the clock showed 04:00 (standard time)
    assert p["clock"] == datetime.datetime(2026, 11, 1, 5, 0, 0, 400000)
    assert cam1[300]["delta"] == pytest.approx(0.4 - 3600)                       # the input is not changed
    assert out[len(cam1):] == cam2 + cam3


def test_a_normal_day_is_left_alone():
    probes = _dst_probes("cam1", lambda t: "a.mp4", lambda k: 1.2, hours=24)
    out, shifted = cc.unwrap_fallback(probes)
    assert shifted == {} and out == probes


def test_the_fallback_days_grid_check_reads_the_second_hour_on_the_axis():
    start = datetime.datetime(2026, 11, 1)
    # 03:00 on the axis shows 02:00 standard time
    clock = datetime.datetime(2026, 11, 1, 2, 0, 0)
    assert cc.grid_error(clock, start, 3 * 3600 + 0.033, fallback=True) == pytest.approx(-0.033)
    assert cc.grid_error(clock, start, 3 * 3600 + 0.033) is None                 # another day: a misread
    assert cc.grid_error(clock, start, 2 * 3600 + 0.5) == pytest.approx(-0.5)


def test_the_fallback_day_runs_25_hours_and_fits_on_its_axis(tmp_path):
    """prepare_day on the continuous track: the day is 00:00-25:00 and the readings after the fallback, an hour
    behind the axis, fit to the camera's real error."""
    import argparse
    from fnt.videoProcessing import camera_grid_cli as cli
    c = cc.continuous_track(_dst_track(), datetime.date(2026, 11, 1))
    probes = _dst_probes("cam1", lambda t: cc.segment_at(c, t).path, lambda k: 0.4 if k < 7198 else 0.4 - 3600)
    clock = tmp_path / "clock"
    clock.mkdir()
    cli._write_csv(str(clock / "VC-T014_2026-11-01_clock_readings.csv"), probes, cli.PROBE_KEYS)
    args = argparse.Namespace(step=60.0, workers=1, keep_black=True, from_time=None, minutes=None)
    job = cli.prepare_day("VC-T014", {"cam1": c}, None, datetime.date(2026, 11, 1), str(tmp_path), args,
                          day_no=4, hours=25)
    assert job["end"] - job["start"] == datetime.timedelta(hours=25) and job["fallback"]
    sync = job["tracks"]["cam1"].clock_sync
    for h in (1.5, 3, 12, 24.5):
        assert sync.delta(datetime.datetime(2026, 11, 1) + datetime.timedelta(hours=h)) == pytest.approx(0.4)
    assert job["final"].endswith("VC-T014_2026-11-01_day4_grid.mp4")
    # the log writes the 25-hour span so the progress window reads it right
    from fnt.videoProcessing.camera_grid_progress import _span_s
    assert cli.hm(job["end"], job["midnight"]) == "25:00"
    assert _span_s("00:00", cli.hm(job["end"], job["midnight"])) == 25 * 3600
    assert cli.hm(datetime.datetime(2026, 11, 3), datetime.datetime(2026, 11, 2)) == "00:00"   # a normal day
