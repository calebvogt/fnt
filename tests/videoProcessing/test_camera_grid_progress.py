"""The grid progress window's reading of the grid CLI's logs (no Qt needed)."""
import datetime
import os
import time

from fnt.videoProcessing import camera_grid_progress as gp


def _job(tmp_path):
    clock = tmp_path / "VC-T005_x" / "grid" / "clock"
    clock.mkdir(parents=True)
    return gp.Job(str(tmp_path / "VC-T005_x"), "VC-T005", datetime.date(2026, 2, 8)), clock


LOG = """\
[2026-11-18 13:18:37] command: camera_grid_cli.py X --day 2026-02-08
[2026-11-18 13:18:40] VC-T005_2026-02-08: reading 5 clocks every 60 s
[2026-11-18 13:19:40]   VC-T005_2026-02-08: 1000/7200 readings (60 s)
[2026-11-18 13:19:41]   VC-T005_2026-02-09: 7000/7200 readings (60 s)
"""


def test_reading_clocks_then_encoding_then_done(tmp_path):
    job, clock = _job(tmp_path)
    gp.apply_log(job, LOG.splitlines())
    assert job.stage == "reading clocks" and job.readings == (1000, 7200)   # the other day's line is not ours
    more = LOG + (
        "[2026-11-18 13:30:00]   VC-T005_2026-02-08: footage 00:00 to 00:00; black before and after left out\n"
        "[2026-11-18 13:30:01] VC-T005_2026-02-08: encoding 00:00 to 00:00 (9 footage runs) -> Z:\\g\\VC-T005_2026-02-08_day2_grid.mp4\n")
    gp.apply_log(job, more.splitlines())
    assert job.stage == "encoding" and job.total_s == 86400
    assert job.final.endswith("VC-T005_2026-02-08_day2_grid.mp4")
    done = more + ("[2026-11-19 13:30:01] VC-T005_2026-02-08: encoded in 1440 min\n"
                   "[2026-11-19 13:40:01] VC-T005_2026-02-08_day2: clocks checked at 1440 times: 97.0% within 1 frame\n")
    gp.apply_log(job, done.splitlines())
    assert job.stage == "done" and "97.0% within 1 frame" in job.result and job.fraction == 1.0


def test_an_old_run_is_overridden_by_a_new_one(tmp_path):
    job, _ = _job(tmp_path)
    lines = ["[2026-11-17 18:40:54] VC-T005_2026-02-08: encoding (9 footage runs) -> Z:\\old.mp4",
             "[2026-11-18 04:17:53] VC-T005_2026-02-08: encoded in 577 min",
             "[2026-11-18 04:28:29] VC-T005_2026-02-08: clocks checked at 1440 times: 0.0% within 1 frame",
             "[2026-11-18 13:59:50] VC-T005_2026-02-08: encoding 00:00 to 00:00 (9 footage runs) -> Z:\\new.mp4"]
    gp.apply_log(job, lines)
    assert job.stage == "encoding" and job.final == "Z:\\new.mp4" and job.result == ""


def test_ffmpeg_progress_and_a_stalled_run(tmp_path):
    job, clock = _job(tmp_path)
    (clock / "VC-T005_grid_cli.log").write_text(
        "[2026-11-18 13:59:50] VC-T005_2026-02-08: encoding 11:18 to 00:00 (5 footage runs) -> Z:\\a.mp4\n")
    ff = clock / "VC-T005_2026-02-08_ffmpeg.log"
    ff.write_text("frame=1 fps=0 time=00:00:01.00 speed=0.5x\rframe=99 fps=20 time=01:02:03.50 bitrate=1 speed=0.62x\r")
    gp.refresh([job])
    assert job.stage == "encoding"
    assert abs(job.done_s - 3723.5) < 1e-6 and job.speed == 0.62
    assert job.total_s == (12 * 60 + 42) * 60
    assert "0.62x" in gp.detail(job) and "left" in gp.detail(job)
    gp.refresh([job], now=time.time() + 60)          # no progress for a minute: still encoding
    assert job.stage == "encoding"
    gp.refresh([job], now=time.time() + 3600)        # none for an hour: stopped
    assert job.stage == "stopped"


def test_a_failed_day(tmp_path):
    job, _ = _job(tmp_path)
    gp.apply_log(job, ["[2026-11-18 13:59:50] VC-T005_2026-02-08: failed: OSError('x')"])
    assert job.stage == "failed"


def test_queue_csv_finds_trial_folders(tmp_path):
    (tmp_path / "VC-T005_prairie").mkdir()
    (tmp_path / "VC-T006_prairie").mkdir()
    q = tmp_path / "q.csv"
    q.write_text("trial,date,day\nVC-T005,2026-02-08,2\nVC-T006,2026-02-26,7\nVC-T099,2026-01-01,1\n")
    jobs = gp.jobs_from_queue(str(q), str(tmp_path))
    assert [(j.trial, j.date.isoformat()) for j in jobs] == [("VC-T005", "2026-02-08"), ("VC-T006", "2026-02-26")]


def test_a_spliced_log_line_does_not_rename_the_day(tmp_path):
    """Several runs append to one log on a network share; two lines can run together."""
    job, _ = _job(tmp_path)
    gp.apply_log(job, ["[2026-11-18 14:01:05] VC-T005_2026-02-08: encoding (7 footage runs) -> Z:\g\VC-T005_2026-02-08_"
                       "[2026-11-18 14:0[2026-11-18 14:01:05]   VC-T005_2026-02-15 cam1 1440/1440 read, error +1.2 s"])
    assert job.stage == "encoding" and job.final is None


def test_the_cli_writes_each_days_lines_to_its_own_log(tmp_path, monkeypatch):
    from fnt.videoProcessing import camera_grid_cli as cli
    monkeypatch.setattr(cli, "_LOG", [])
    monkeypatch.setattr(cli, "_DAY_LOG_DIR", [str(tmp_path)])
    cli.log("VC-T005_2026-02-08: encoding 00:00 to 00:00 (9 footage runs) -> Z:\a.mp4")
    cli.log("  VC-T005_2026-02-08 cam1 1440/1440 read, error +0.5 to +0.9 s")
    cli.log("VC-T005_2026-02-08_day2: clocks checked at 1440 times: 99.0% within 1 frame")
    cli.log("all done")
    lines = (tmp_path / "VC-T005_2026-02-08_progress.log").read_text().splitlines()
    assert len(lines) == 3 and not (tmp_path / "all done_progress.log").exists()
