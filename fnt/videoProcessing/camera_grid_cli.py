"""Camera Grid without the window, every camera held to its burnt-in clock.

    python -m fnt.videoProcessing.camera_grid_cli TRIAL_FOLDER --day 2026-02-07
    python -m fnt.videoProcessing.camera_grid_cli TRIAL_FOLDER --all-days

For each day (midnight to midnight) of a trial folder holding one sub-folder
per camera, this:

1. learns the recorder's clock font from the footage (once per trial);
2. reads every camera's burnt-in clock every minute, to the frame
   (camera_clock.measure), and fits each segment file's clock error;
3. encodes the day's grid with every frame moved by its camera's clock error,
   so all cells show the same burnt-in time (black where a camera has no
   footage);
4. reads the five clocks back off the finished grid every `--check-every`
   seconds and reports how far apart they are, in frames.

Outputs go to ``TRIAL_FOLDER/grid/`` (``<trial>_<date>_grid.mp4``), the clock
readings, fitted errors and the check to ``grid/clock/``. A day already encoded
is skipped; readings already taken are reused.
"""

import argparse
import csv
import datetime
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time

from fnt.videoProcessing import camera_clock as cc
from fnt.videoProcessing.camera_grid_encode import (
    EncodeSettings, GridLayout, build_chunk_command)
from fnt.videoProcessing.camera_timeline import CameraTrack, VIDEO_EXTS

#: The mesocosm layout: cam2 top-left, cam3 top-right, cam5 centre, cam1
#: bottom-left, cam4 bottom-right, other cells black.
MESOCOSM_LAYOUT = "cam2,,cam3;,cam5,;cam1,,cam4"


_LOG = []     # the log file, once main() has opened it
_DAY_LOG_DIR = []   # the clock folder: each day's lines also go to <tag>_progress.log there
_DAY_TAG = re.compile(r"^\s*(\S+?_\d{4}-\d\d-\d\d)")


def log(msg):
    line = f"[{datetime.datetime.now():%Y-%m-%d %H:%M:%S}] {msg}"
    print(line, flush=True)
    for fh in _LOG:
        fh.write(line + "\n")
        fh.flush()
    # Several runs of one trial append to its log at once, and on a network
    # share their lines can interleave; a day's own file has a single writer
    # (the progress window reads it).
    m = _DAY_TAG.match(msg)
    if m and _DAY_LOG_DIR:
        try:
            with open(os.path.join(_DAY_LOG_DIR[0], f"{m.group(1)}_progress.log"), "a",
                      encoding="utf-8") as fh:
                fh.write(line + "\n")
        except OSError:
            pass


def parse_layout(text, centre_scale=1.0):
    rows = [r.split(",") for r in text.split(";")]
    layout = GridLayout(len(rows), max(len(r) for r in rows), centre_scale=centre_scale)
    for r, row in enumerate(rows):
        for c, name in enumerate(row):
            if name.strip():
                layout.place(r, c, name.strip())
    return layout


def camera_folders(trial):
    """{camN: folder} for every sub-folder of `trial` holding camera footage,
    named by the camera number in its files (cam1, RZ_1, overhead, ...)."""
    out = {}
    for sub in sorted(os.listdir(trial)):
        folder = os.path.join(trial, sub)
        if not os.path.isdir(folder) or sub.lower() in ("grid", "proc"):
            continue
        names = [f for f in os.listdir(folder) if f.lower().endswith(VIDEO_EXTS)]
        nums = {m.group(1) for f in names for m in [re.search(r"(?i)camera\s*(\d+)", f)] if m}
        if len(nums) == 1:
            out[f"cam{nums.pop()}"] = folder
        elif names:
            log(f"  skipped {sub}: files name {sorted(nums) or 'no'} camera numbers")
    return out


def _write_csv(path, rows, keys):
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, keys)
        w.writeheader()
        for row in rows:
            w.writerow({k: (v.isoformat(sep=" ") if isinstance(v, datetime.datetime) else v)
                        for k, v in ((k, row.get(k)) for k in keys)})


def _read_probes(path):
    rows = []
    with open(path, newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            for k in ("when", "timeline", "clock"):
                r[k] = datetime.datetime.fromisoformat(r[k]) if r[k] else None
            r["delta"] = float(r["delta"]) if r["delta"] else None
            rows.append(r)
    return rows


PROBE_KEYS = ["camera", "when", "path", "file_time", "timeline", "clock", "delta", "score"]


def grid_name(trial_name, day, day_no, from_time=None):
    """VC-T005_2026-02-07_day1_grid.mp4; a clip from a time of day adds it (..._day1_1405_grid.mp4)."""
    clip = f"_{from_time:%H%M}" if from_time else ""
    return f"{trial_name}_{day:%Y-%m-%d}_day{day_no}{clip}_grid.mp4"


def footage_window(probes, start, end, margin=60.0):
    """The part of [start, end) where some camera shows a readable clock.

    The NVR records "no video" black before a trial's cameras come on (and
    after they go off); its clock is not drawn there, so the first and last
    readings bound the footage. Readings come every minute, so `margin` either
    side keeps the frames between a reading and the last black one."""
    read = [p["timeline"] for p in probes if p.get("delta") is not None]
    if not read:
        return start, end
    pad = datetime.timedelta(seconds=margin)
    first = (min(read) - pad).replace(second=0, microsecond=0)
    last = (max(read) + 2 * pad).replace(second=0, microsecond=0)
    return max(start, first), min(end, last)


def prepare_day(trial_name, tracks, reader, day, out_dir, args, day_no=1):
    """Read the day's clocks (or reuse the readings) and fit each camera's
    error. Returns the day's job: copies of the tracks carrying the fit."""
    clock_dir = os.path.join(out_dir, "clock")
    os.makedirs(clock_dir, exist_ok=True)
    tag = f"{trial_name}_{day:%Y-%m-%d}"
    from_time = getattr(args, "from_time", None)
    final = os.path.join(out_dir, grid_name(trial_name, day, day_no, from_time))
    start = datetime.datetime.combine(day, datetime.time.min)
    end = start + datetime.timedelta(days=1)

    # 1. read every camera's clock every minute
    readings = os.path.join(clock_dir, f"{tag}_clock_readings.csv")
    if os.path.exists(readings):
        probes = _read_probes(readings)
        log(f"{tag}: {len(probes)} clock readings reused")
    else:
        log(f"{tag}: reading {len(tracks)} clocks every {args.step:.0f} s")
        t0 = time.time()

        def prog(d, n):
            if d % 1000 == 0 or d == n:
                log(f"  {tag}: {d}/{n} readings ({time.time() - t0:.0f} s)")
        probes = cc.measure(reader, list(tracks.values()), start, end,
                            step=args.step, workers=args.workers, progress=prog)
        _write_csv(readings, probes, PROBE_KEYS)

    # 2. fit each camera's clock error, segment by segment
    model_rows, summary, day_tracks = [], [], {}
    for name, tr in tracks.items():
        mine = [p for p in probes if p["camera"] == name]
        ok = [p for p in mine if p["delta"] is not None]
        tr = day_tracks[name] = tr.copy()   # each day its own fit, so days can run at once
        tr.clock_sync = cc.ClockSync.fit(ok, tr.segments)
        model_rows += tr.clock_sync.to_rows(name)
        if ok:
            d = sorted(p["delta"] for p in ok)
            summary.append(f"{name} {len(ok)}/{len(mine)} read, error "
                           f"{d[0]:+.2f} to {d[-1]:+.2f} s")
        else:
            summary.append(f"{name} no footage or no readable clock")
    _write_csv(os.path.join(clock_dir, f"{tag}_clock_model.csv"), model_rows,
               ["camera", "t0", "t1", "a", "b"])
    for s in summary:
        log(f"  {tag} {s}")

    # 3. the stretch to encode: the day without its black, or a clip
    if from_time:
        start = datetime.datetime.combine(day, from_time)
    elif not getattr(args, "keep_black", False):
        s0, e0 = footage_window(probes, start, end)
        if (s0, e0) != (start, end):
            log(f"  {tag}: footage {s0:%H:%M} to {e0:%H:%M}; black before and after left out")
        start, end = s0, e0
    if getattr(args, "minutes", None):
        end = min(end, start + datetime.timedelta(minutes=args.minutes))
    return {"tag": tag, "final": final, "start": start, "end": end,
            "tracks": day_tracks, "clock_dir": clock_dir}


def encode_day(job, layout, reader, args):
    """Encode one prepared day, then check its clocks. True if it worked."""
    tag, final, start, end = job["tag"], job["final"], job["start"], job["end"]
    tracks, clock_dir = job["tracks"], job["clock_dir"]
    # 3. encode
    if os.path.exists(final) and not args.overwrite:
        log(f"{tag}: grid exists, not re-encoded")
    else:
        settings = EncodeSettings(width=args.width, height=args.height, fps=30,
                                  codec=args.codec, crf=args.crf, preset=args.preset,
                                  show_camera_labels=args.labels, show_no_signal=False,
                                  show_clock=False)
        work = tempfile.mkdtemp(prefix="fnt_grid_")
        tmp_out = os.path.join(work, os.path.basename(final))
        cmd, n = build_chunk_command(tracks, layout, settings, start, end, tmp_out,
                                     work_dir=work)
        log(f"{tag}: encoding {start:%H:%M} to {end:%H:%M} ({n} footage runs) -> {final}")
        t0 = time.time()
        with open(os.path.join(clock_dir, f"{tag}_ffmpeg.log"), "w", encoding="utf-8") as lf:
            r = subprocess.run(cmd, stdout=lf, stderr=subprocess.STDOUT, stdin=subprocess.DEVNULL)
        if r.returncode:
            log(f"{tag}: ffmpeg failed ({r.returncode}); see {tag}_ffmpeg.log")
            return False
        shutil.move(tmp_out, final)
        shutil.rmtree(work, ignore_errors=True)
        log(f"{tag}: encoded in {(time.time() - t0) / 60:.0f} min")

    # 4. check the clocks in the finished grid (named after the video: a clip has its own)
    ctag = re.sub(r"(_grid)?\.mp4$", "", os.path.basename(final))
    done = os.path.exists(os.path.join(clock_dir, f"{ctag}_grid_check.csv"))
    if not args.no_check and not (done and not args.overwrite):
        check(ctag, final, tracks, layout, reader, clock_dir, args, start)
    return True


def check(tag, video, tracks, layout, reader, clock_dir, args, start):
    geo = EncodeSettings(width=args.width, height=args.height)
    cells = {cam: geo.cell_rect(layout, r, c) for (r, c), cam in layout.assignments.items()
             if cam in tracks}
    dur = float(subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of",
         "csv=p=0", video], capture_output=True, text=True).stdout.strip())
    times = [t for t in range(5, int(dur) - 2, int(args.check_every))]
    log(f"{tag}: checking the clocks in the grid at {len(times)} times")
    rows, spreads = [], []
    if True:
        for t, res in cc.check_grid_many(video, cells, reader, times, workers=args.workers):
            errs = {}
            for cam, hit in res.items():
                if hit is None:
                    continue
                clock, grid_t = hit
                # clock second minus grid time, both from the chunk's start
                err = (clock - start).total_seconds() - grid_t
                if abs(err) < 10:       # a further-out reading is a misread
                    errs[cam] = err
            row = {"grid_s": t, **{c: (round(e, 3) if c in errs else "") for c, e in
                                    ((c, errs.get(c)) for c in cells)}}
            if len(errs) >= 2:
                spread = max(errs.values()) - min(errs.values())
                row["spread_frames"] = round(spread * 30)
                spreads.append(spread)
            elif len(cells) == 1 and errs:
                # one camera alone (a one-cell layout): its clock against the video's own time
                spread = abs(next(iter(errs.values())))
                row["spread_frames"] = round(spread * 30)
                spreads.append(spread)
            rows.append(row)
    _write_csv(os.path.join(clock_dir, f"{tag}_grid_check.csv"), rows,
               ["grid_s"] + list(cells) + ["spread_frames"])
    if spreads:
        fr = [round(s * 30) for s in spreads]
        within1 = sum(f <= 1 for f in fr) / len(fr)
        log(f"{tag}: clocks checked at {len(fr)} times: {within1:.1%} within 1 frame, "
            f"{sum(f <= 3 for f in fr) / len(fr):.1%} within 3, worst {max(fr)} frames")
    else:
        log(f"{tag}: no clock could be read back from the grid")


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("trial", help="trial folder with one sub-folder per camera")
    ap.add_argument("--day", action="append", default=[],
                    help="YYYY-MM-DD, repeatable; or day numbers 1..N of the trial")
    ap.add_argument("--all-days", action="store_true")
    ap.add_argument("--out", help="output folder (default TRIAL/grid)")
    ap.add_argument("--layout", default=MESOCOSM_LAYOUT,
                    help="rows separated by ';', cells by ',' (default: %(default)s)")
    ap.add_argument("--centre-scale", type=float, default=1.2,
                    help="on a 3x3 layout, the centre cell's size against a third of "
                         "the frame; the corners shrink to fit (default %(default)s)")
    ap.add_argument("--labels", action="store_true", help="name each camera in its cell")
    ap.add_argument("--step", type=float, default=60.0, help="seconds between clock readings")
    ap.add_argument("--check-every", type=float, default=60.0,
                    help="seconds between clock checks in the finished grid")
    ap.add_argument("--no-check", action="store_true")
    ap.add_argument("--workers", type=int, default=12,
                    help="processes reading clocks (default %(default)s)")
    ap.add_argument("--parallel", type=int, default=1,
                    help="days encoded at once; each ffmpeg uses about 10 cores (default %(default)s)")
    ap.add_argument("--codec", default="libx264")
    ap.add_argument("--crf", type=int, default=23)
    ap.add_argument("--preset", default="veryfast")
    ap.add_argument("--width", type=int, default=1920)
    ap.add_argument("--height", type=int, default=1080)
    ap.add_argument("--minutes", type=float, help="encode only the first N minutes (tests)")
    ap.add_argument("--from", dest="from_time", type=lambda v: datetime.datetime.strptime(
                        v, "%H:%M:%S" if v.count(":") == 2 else "%H:%M").time(),
                    help="start at this time of day, HH:MM[:SS] (a clip; with --minutes)")
    ap.add_argument("--keep-black", action="store_true",
                    help="encode the whole day, the NVR's black before the cameras come on included")
    ap.add_argument("--clips", action="store_true",
                    help="also write each camera alone, cut by the same clock sync as the grid (frame n of a "
                         "clip is frame n of the grid): <grid name>_<camera>.mp4, for tracking each camera")
    ap.add_argument("--clip-size", default="1920x1080", help="clip frame size, WxH (default %(default)s)")
    ap.add_argument("--no-grid", action="store_true", help="with --clips: the clips only")
    ap.add_argument("--overwrite", action="store_true")
    win = ap.add_mutually_exclusive_group()
    win.add_argument("--window", dest="window", action="store_true", default=None,
                     help="show the progress window (camera_grid_progress); default when run from a console")
    win.add_argument("--no-window", dest="window", action="store_false",
                     help="no progress window (a queue that shows one window for all its runs)")
    args = ap.parse_args(argv)

    trial = os.path.abspath(args.trial)
    name = re.match(r"(VC-T\d+|T\d+)", os.path.basename(trial))
    trial_name = name.group(1) if name else os.path.basename(trial)
    out_dir = args.out or os.path.join(trial, "grid")
    os.makedirs(os.path.join(out_dir, "clock"), exist_ok=True)
    layout = parse_layout(args.layout, args.centre_scale)
    _DAY_LOG_DIR.append(os.path.join(out_dir, "clock"))
    _LOG.append(open(os.path.join(out_dir, "clock", f"{trial_name}_grid_cli.log"), "a",
                     encoding="utf-8"))
    log("command: " + " ".join(sys.argv))

    folders = camera_folders(trial)
    log(f"{trial_name}: cameras {', '.join(f'{k}={os.path.basename(v)}' for k, v in folders.items())}")
    tracks = {}
    for cam, folder in folders.items():
        if cam in layout.cameras:
            tracks[cam] = CameraTrack.from_folder(folder, name=cam)
            t = tracks[cam]
            log(f"  {cam}: {len(t.segments)} files, {t.start} to {t.end}, "
                f"{t.gap_seconds / 60:.0f} min missing")
    missing = [c for c in layout.cameras if c not in tracks]
    if missing:
        log(f"  no footage for {missing}: their cells stay black")

    font = os.path.join(out_dir, "clock", f"{trial_name}_clock_font.json")
    if os.path.exists(font):
        reader = cc.ClockReader.load(font)
    else:
        log("learning the clock font")
        reader = cc.learn_from_tracks(list(tracks.values()))
        reader.save(font)
    # a 25-hour day (daylight saving ends) runs into the next: place its files by their clocks
    for cam, t in tracks.items():
        for path, old, new in cc.anchor_overfull_days(reader, t):
            log(f"  {cam}: {os.path.basename(path)} placed by its clock at {new:%Y-%m-%d %H:%M:%S} "
                f"(file timeline said {old:%H:%M:%S})")

    first = min(t.start for t in tracks.values()).date()
    last = max(t.end for t in tracks.values()).date()
    all_days = [first + datetime.timedelta(days=i) for i in range((last - first).days + 1)]
    days = []
    for d in args.day:
        if d.isdigit():
            days.append(all_days[int(d) - 1])
        else:
            days.append(datetime.datetime.strptime(d, "%Y-%m-%d").date())
    if args.all_days:
        days = all_days
    if not days:
        ap.error("give --day or --all-days")
    show = args.window if args.window is not None else bool(sys.stdout and sys.stdout.isatty())
    if show:
        from fnt.videoProcessing import camera_grid_progress
        camera_grid_progress.launch(trial, days, watch_pid=os.getpid(),
                                    title=f"Camera grids â€” {trial_name}")
    # --parallel days at a time, each reading its clocks and then encoding,
    # so one day's encode runs while the next day's clocks are read
    def one(day):
        try:
            job = prepare_day(trial_name, tracks, reader, day, out_dir, args, day_no=(day - first).days + 1)
            ok = True if args.no_grid and args.clips else encode_day(job, layout, reader, args)
            if args.clips:
                cw, ch = (int(v) for v in args.clip_size.lower().split("x"))
                clip_args = argparse.Namespace(**{**vars(args), "width": cw, "height": ch, "labels": False})
                for cam in layout.cameras:
                    if cam in tracks:
                        clip = dict(job, final=job["final"][:-len("_grid.mp4")] + f"_{cam}.mp4")
                        ok = encode_day(clip, parse_layout(cam, 1.0), reader, clip_args) and ok
            return ok
        except Exception as exc:
            log(f"{trial_name}_{day:%Y-%m-%d}: failed: {exc!r}")
            return False
    from concurrent.futures import ThreadPoolExecutor
    with ThreadPoolExecutor(max_workers=max(1, args.parallel)) as ex:
        results = list(ex.map(one, days))
    failed = [f"{d:%Y-%m-%d}" for d, ok in zip(days, results) if not ok]
    log("all done" + (f"; failed: {', '.join(failed)}" if failed else ""))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
