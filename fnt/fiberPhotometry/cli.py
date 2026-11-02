"""Fiber photometry from the command line.

    python -m fnt.fiberPhotometry.cli info  <recording.doric | recording.dat>
    python -m fnt.fiberPhotometry.cli process <folder or recordings...> [--pipeline guppy] [options]
    python -m fnt.fiberPhotometry.cli summary <run folder>
    python -m fnt.fiberPhotometry.cli gui [<folder or recordings...>]

``process`` writes a new dated run folder, ``<recordings folder>/FNT_FiberPhotometry/
<YYYY-MM-DD_HHMMSS>_<pipeline>/`` (see ``batch.py``); ``--output <folder>`` writes into that folder
instead and skips recordings already done there, so an interrupted run can be resumed.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .processing import PIPELINES


def _floats(text):
    return tuple(float(x) for x in text.replace(",", " ").split()) if text else ()


def add_settings_args(p):
    g = p.add_argument_group("processing")
    g.add_argument("--pipeline", choices=PIPELINES, default=PIPELINES[0],
                   help="guppy (default): robust fit, dF/F on the fit; matlab: Warden lab "
                        "linear fit, dF/F on mean signal; phat: biexponential + IQR reference fit")
    g.add_argument("--no-dropouts", action="store_true", help="do not remove wireless dropouts")
    g.add_argument("--notch", type=_floats, default=(), metavar="HZ",
                   help='frequencies to notch out of both channels, e.g. "3.32 9.94"')
    g.add_argument("--notch-q", type=float, default=30.0, help="notch quality: width = f / Q (30)")
    g.add_argument("--smooth", type=float, default=0.1, help="guppy/phat moving average, s (0.1)")
    g.add_argument("--no-fir", action="store_true", help="matlab: no FIR low-pass (fitcaldff.m)")
    g.add_argument("--fir-cutoff", type=float, default=20.0, help="matlab: FIR cutoff, Hz (20)")
    g.add_argument("--fir-order", type=int, default=100, help="matlab: FIR order (100)")
    g.add_argument("--biexp-r2", type=float, default=0.5, help="phat: minimum biexponential R^2")
    g.add_argument("--reference-fit", choices=("robust", "least squares"), default="robust")
    g.add_argument("--phat-bleaching", choices=("each channel", "control"), default="each channel",
                   help="phat: remove each channel's own bleaching (PhAT) or the control's from both")
    g.add_argument("--zscore", choices=("standard", "modified", "baseline"), default="standard")
    g.add_argument("--baseline", type=_floats, default=(), metavar="START END",
                   help='baseline window for --zscore baseline, in s, e.g. "0 60"')
    o = p.add_argument_group("outputs")
    o.add_argument("--no-video", action="store_true", help="skip the combined video")
    o.add_argument("--window", type=float, default=20.0, help="combined video trace window, s")
    o.add_argument("--output", help="write into this folder (resume) instead of a new run folder")


def settings_from(args):
    from .batch import Settings
    if args.zscore == "baseline" and len(args.baseline) != 2:
        raise SystemExit('--zscore baseline needs --baseline "START END"')
    return Settings(pipeline=args.pipeline, remove_dropouts=not args.no_dropouts,
                    notch_hz=args.notch, notch_q=args.notch_q, apply_filter=not args.no_fir,
                    fir_cutoff=args.fir_cutoff, fir_order=args.fir_order, smooth_s=args.smooth,
                    biexp_r2=args.biexp_r2, reference_fit=args.reference_fit,
                    phat_bleaching=args.phat_bleaching, zscore_method=args.zscore,
                    zscore_baseline=tuple(args.baseline) if args.zscore == "baseline" else None,
                    create_video=not args.no_video, video_window_s=args.window)


def cmd_info(args):
    from .batch import open_reader
    from .processing import find_dropouts
    from .sync import check_video_clock, recording_frame_times
    r = open_reader(args.recording)
    d = r.scan_file()
    print(f"{Path(args.recording).name}")
    for c in d.all_channels:
        role = ("signal" if d.signal_channel is c else "control" if d.isosbestic_channel is c else "")
        nm = f"{c.wavelength} nm, " if c.wavelength else ""
        print(f"  {c.name}: {nm}{c.sampling_rate:.1f} Hz, {c.n_samples:,} samples  {role}")
    d = r.load_data(d)
    print(f"  duration {d.signal_time[-1] - d.signal_time[0]:.1f} s")
    if d.isosbestic_data is not None:
        bad = find_dropouts(d.signal_data, d.isosbestic_data, d.signal_time, d.signal_channel.sampling_rate)
        print(f"  dropouts {100 * bad.mean():.1f}% of samples")
    v = d.video_info.absolute_path if d.video_info else None
    if v and Path(v).exists():
        ft, fps = recording_frame_times(d, v)
        c = check_video_clock(ft, d.video_timestamps)
        timing = "timed by the frame clock" if d.video_frame_clock else f"timed at {fps:g} fps"
        print(f"  video {Path(v).name}: {c['video_frames']:,} frames ({timing}); "
              f"the recording logged {c.get('logged_frames', 0):,} frame times")
    else:
        print("  no video" if not v else f"  video missing: {v}")
    return 0


def cmd_process(args):
    from .batch import find_recordings, run_batch
    recs = find_recordings(args.paths)
    if not recs:
        print("No recordings found.")
        return 1
    print(f"{len(recs)} recording(s), pipeline {args.pipeline}")
    out = run_batch(recs, settings_from(args), Path(args.output) if args.output else None,
                    log=lambda m: print(m, flush=True))
    print(f"Done: {out}")
    return 0


def cmd_summary(args):
    import csv
    from .batch import SUMMARY_FILE, write_summary
    path = write_summary(Path(args.run_folder))
    if path is None:
        print("Nothing processed in that folder.")
        return 1
    with open(path, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            print(f"{row['recording']}: {row['status']}, dropouts {row.get('dropout_pct')}%, "
                  f"dF/F robust SD {row.get('dff_robust_sd_pct')}%")
    print(f"({SUMMARY_FILE} rewritten)")
    return 0


def cmd_gui(args):
    from .fiber_photometry_pyqt import main as gui_main
    return gui_main(args.paths)


def main(argv=None):
    from fnt.usv.usv_detector.power import keep_full_speed
    keep_full_speed()
    p = argparse.ArgumentParser(prog="python -m fnt.fiberPhotometry.cli", description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("info", help="channels, duration, dropouts and video of one recording")
    s.add_argument("recording")
    s.set_defaults(func=cmd_info)
    s = sub.add_parser("process", help="process recordings into a run folder")
    s.add_argument("paths", nargs="+", help="recordings and/or folders of recordings")
    add_settings_args(s)
    s.set_defaults(func=cmd_process)
    s = sub.add_parser("summary", help="rebuild and print a run folder's summary")
    s.add_argument("run_folder")
    s.set_defaults(func=cmd_summary)
    s = sub.add_parser("gui", help="open the Fiber Photometry window, optionally with recordings")
    s.add_argument("paths", nargs="*")
    s.set_defaults(func=cmd_gui)
    args = p.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
