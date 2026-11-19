"""ROI Analysis Tool from the command line: run saved ROI configurations (``*_roiConfig.json``) without the GUI.

Draw a video's tracking area, ROIs and scale bar once in the ROI Analysis Tool; the tool saves them as
``<video>_roiConfig.json`` next to its outputs. This runs those configurations again, headless:

    python -m fnt.sleapProcessing.roi_cli <config.json | folder> ... [--workers N] [--data-only | --visuals-only]

A folder is searched (recursively) for ``*_roiConfig.json`` (``--source mtt`` keeps Mask Tracker ones only;
folders named like backups or to-delete are skipped). Videos run in the order given (a folder's
configurations sorted by name). Each video's data (position coordinates, ROI occupancy, ROI summary)
is written for every video first, then the slow part, tracking plots and the tracked or data-view video,
for every video; ``--workers`` videos at once in each pass. Outputs go where the GUI put them: the Mask
Tracker / SimpleTracker folder when the configuration sits there, otherwise ``<video>_roiAnalysis[_SLEAP]/``.
Existing outputs are overwritten, never cleared first; the configuration itself is left as drawn.
"""
import argparse
import glob
import json
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed


SKIP_DIRS = re.compile(r"backup|delete", re.I)     # 0_Backups_delete, _TO_DELETE_tracking, ...


def find_configs(paths):
    out = []
    for p in paths:
        if os.path.isdir(p):
            found = sorted(glob.glob(os.path.join(p, "**", "*_roiConfig.json"), recursive=True))
            out += [c for c in found if not SKIP_DIRS.search(os.path.relpath(os.path.dirname(c), p))]
        elif os.path.isfile(p):
            out.append(p)
        else:
            raise SystemExit(f"not found: {p}")
    seen = set()
    return [c for c in out if not (c in seen or seen.add(c))]


def _track_csv(d, config_path):
    """The configuration's tracking CSV; when it has gone (a Mask Tracker re-track writes ``trajectories.csv`` and
    moves the old ``<video>_trajectories.csv`` aside), the track now beside the configuration."""
    p = d["csv_path"]
    if os.path.exists(p):
        return p
    here = os.path.dirname(os.path.abspath(config_path))
    stem = os.path.splitext(os.path.basename(d["video_path"]))[0]
    for name in (os.path.basename(p), f"{stem}_trajectories.csv", "trajectories.csv"):
        q = os.path.join(here, name)
        if os.path.exists(q):
            return q
    return p


def _load(config_path):
    from fnt.sleapProcessing.sleap_roi_tool_pyqt import VideoROIConfig
    with open(config_path, encoding="utf-8") as f:
        d = json.load(f)
    d["csv_path"] = _track_csv(d, config_path)
    c = VideoROIConfig(d["video_path"], d["csv_path"], d.get("data_source", "sleap"))
    c.from_config_dict(d)
    c.configured = True
    c.overwrite_files = False          # never clear the folder (it would delete this configuration in source mode)
    c.save_config = False              # the configuration is the input; leave it as drawn
    if c.data_source == "sleap":
        c.has_tracks = True            # as the GUI finds for SLEAP files with a track column
    # outputs where the GUI wrote them: beside the tracking CSV ("source") or in a new folder
    same = os.path.normcase(os.path.dirname(os.path.abspath(config_path))) == \
        os.path.normcase(os.path.dirname(os.path.abspath(d["csv_path"])))
    mode = "source" if (same and c.data_source in ("mtt", "st")) else "new"
    return c, mode


def run_one(config_path, part, log_dir=None):
    """Run one video's configuration; part = 'data', 'visuals' or 'all'. Returns (config_path, ok, message, seconds)."""
    t0 = time.time()
    try:
        from PyQt5.QtCore import QCoreApplication
        from fnt.sleapProcessing.sleap_roi_tool_pyqt import ROIProcessor
        app = QCoreApplication.instance() or QCoreApplication([])  # noqa: F841 (Qt objects need an application)
        c, mode = _load(config_path)
        if part == "data":
            c.save_tracking_plots = False
            c.create_tracked_video = False
        elif part == "visuals":
            c.save_position_coords = c.save_roi_occupancy = c.save_roi_summary = False
        lines, result = [], {}
        p = ROIProcessor([c], output_mode=mode)
        p.status.connect(lines.append)
        p.video_finished.connect(lambda i, ok, msg: result.update(ok=ok, msg=msg))
        p.run()                                   # in this process, not as a thread
        if log_dir:
            name = os.path.basename(config_path).replace("_roiConfig.json", f"_{part}.log")
            with open(os.path.join(log_dir, name), "w", encoding="utf-8") as f:
                f.write("\n".join(lines))
        return config_path, result.get("ok", False), result.get("msg", "no result"), time.time() - t0
    except Exception as e:                       # one bad video doesn't stop the others
        return config_path, False, f"Failed: {e}", time.time() - t0


def run_pass(configs, part, workers, log_dir):
    name = {"data": "data (CSVs)", "visuals": "plots and videos", "all": "everything"}[part]
    print(f"== {name}: {len(configs)} video(s), {workers} at once", flush=True)
    failed = []
    if workers <= 1:
        results = (run_one(c, part, log_dir) for c in configs)
    else:
        ex = ProcessPoolExecutor(max_workers=workers)
        futs = [ex.submit(run_one, c, part, log_dir) for c in configs]
        results = (f.result() for f in as_completed(futs))
    for i, (c, ok, msg, s) in enumerate(results, 1):
        stem = os.path.basename(c).replace("_roiConfig.json", "")
        print(f"[{i}/{len(configs)}] {'ok ' if ok else 'FAILED'} {stem} ({s:.0f} s){'' if ok else ': ' + msg}", flush=True)
        if not ok:
            failed.append(c)
    if workers > 1:
        ex.shutdown()
    return failed


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("paths", nargs="+", help="roiConfig.json files and/or folders to search for them")
    ap.add_argument("--workers", type=int, default=max(1, min(8, (os.cpu_count() or 2) // 2)),
                    help="videos processed at once (default: half the CPUs, up to 8)")
    g = ap.add_mutually_exclusive_group()
    g.add_argument("--data-only", action="store_true", help="only the CSVs and configuration, no plots or videos")
    g.add_argument("--visuals-only", action="store_true", help="only the tracking plots and videos")
    ap.add_argument("--source", choices=("mtt", "sleap", "st"),
                    help="only configurations for this tracking (Mask Tracker, SLEAP, SimpleTracker)")
    ap.add_argument("--log-dir", help="write each video's processing log here")
    a = ap.parse_args(argv)
    configs = find_configs(a.paths)
    if a.source:
        configs = [c for c in configs if json.load(open(c, encoding="utf-8")).get("data_source", "sleap") == a.source]
    if not configs:
        raise SystemExit("no *_roiConfig.json found")
    if a.log_dir:
        os.makedirs(a.log_dir, exist_ok=True)
    failed = []
    if not a.visuals_only:
        failed += run_pass(configs, "data", a.workers, a.log_dir)
    if not a.data_only:
        failed += run_pass([c for c in configs if c not in failed], "visuals", a.workers, a.log_dir)
    print(f"done: {len(configs) - len(set(failed))} of {len(configs)} video(s) without errors", flush=True)
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
