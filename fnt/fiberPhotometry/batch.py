"""Processing recordings end to end: read, dF/F and z-score, align to video, write outputs.

The GUI and the command line both use ``run_batch`` (or ``process_recording`` for one file).

Layout. The folder that holds the raw recordings is the user's; FNT never writes beside the raw
files. Each run gets its own dated folder under ``FNT_FiberPhotometry/`` in that folder::

    <recordings folder>/
        NC500_Acq_0001.doric, NC500_Acq_0001_Videos/, ...          raw (user's)
        FNT_FiberPhotometry/
            2026-11-02_154500_guppy/                                one run
                run_settings.json, run.log, FiberPhotometry_summary.csv
                NC500_Acq_0001_photometry.csv   one row per photometry sample
                NC500_Acq_0001_dff.csv          one row per video frame
                NC500_Acq_0001_combined.mp4     video with the traces underneath (optional)
                NC500_Acq_0001_settings.json    this recording's settings and fit results

So runs with different pipelines or settings sit side by side and never overwrite each other.
"""
from __future__ import annotations

import csv
import json
import platform
from dataclasses import asdict, dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

from .doric import DoricFileReader
from .labjack import LabJackFileReader, is_labjack_dat
from .processing import PIPELINES, process
from .sync import align_to_frames, check_video_clock, recording_frame_times

RUNS_FOLDER = "FNT_FiberPhotometry"
SUMMARY_FILE = "FiberPhotometry_summary.csv"
RECORDING_SUFFIXES = (".doric", ".dat")


def open_reader(recording):
    """The reader for ``recording``: Doric .doric or LabJack LJStreamUD .dat."""
    if Path(recording).suffix.lower() == ".dat":
        return LabJackFileReader(recording)
    return DoricFileReader(recording)


def _is_recording(p: Path) -> bool:
    if p.suffix.lower() == ".dat":
        return is_labjack_dat(p)
    return p.suffix.lower() in RECORDING_SUFFIXES


@dataclass
class Settings:
    pipeline: str = PIPELINES[0]
    remove_dropouts: bool = True
    dropout_fraction: float = 0.5
    dropout_pad_s: float = 0.1
    notch_hz: Tuple[float, ...] = ()
    notch_q: float = 30.0
    apply_filter: bool = True          # matlab: FIR low-pass
    fir_cutoff: float = 20.0
    fir_order: int = 100
    smooth_s: float = 0.1              # guppy / phat: moving average
    biexp_r2: float = 0.5              # phat
    reference_fit: str = "robust"      # phat
    phat_bleaching: str = "each channel"   # phat
    zscore_method: str = "standard"
    zscore_baseline: Optional[Tuple[float, float]] = None
    create_video: bool = True
    video_window_s: float = 20.0
    show_raw: bool = True
    signal_channel: Optional[str] = None
    control_channel: Optional[str] = None

    def processing_kwargs(self):
        keys = ("remove_dropouts", "dropout_fraction", "dropout_pad_s", "notch_hz", "notch_q",
                "apply_filter", "fir_cutoff", "fir_order", "smooth_s", "biexp_r2", "reference_fit",
                "phat_bleaching", "zscore_method", "zscore_baseline")
        return {k: getattr(self, k) for k in keys}


def output_paths(recording: Path, out_dir: Path):
    stem = Path(recording).stem
    out_dir = Path(out_dir)
    return {"photometry": out_dir / f"{stem}_photometry.csv", "frames": out_dir / f"{stem}_dff.csv",
            "video": out_dir / f"{stem}_combined.mp4", "settings": out_dir / f"{stem}_settings.json"}


def new_run_folder(recordings_folder: Path, pipeline: str, when: Optional[datetime] = None) -> Path:
    """``<recordings_folder>/FNT_FiberPhotometry/<YYYY-MM-DD_HHMMSS>_<pipeline>/`` (created)."""
    stamp = (when or datetime.now()).strftime("%Y-%m-%d_%H%M%S")
    path = Path(recordings_folder) / RUNS_FOLDER / f"{stamp}_{pipeline}"
    k = 1
    while path.exists():                       # two runs in the same second
        k += 1
        path = Path(recordings_folder) / RUNS_FOLDER / f"{stamp}_{pipeline}_{k}"
    path.mkdir(parents=True)
    return path


def find_recordings(items: Iterable[str]) -> List[Path]:
    """Recordings from files and folders (a folder's own files, not its subfolders, and never
    anything inside an ``FNT_FiberPhotometry`` folder)."""
    out = []
    for it in items:
        p = Path(it)
        if p.is_dir():
            out += sorted(f for f in p.iterdir() if f.is_file() and _is_recording(f))
        elif p.is_file() and _is_recording(p):
            out.append(p)
        else:
            raise FileNotFoundError(f"not a recording or folder: {it}")
    return [p for p in dict.fromkeys(out) if RUNS_FOLDER not in p.parts]


def _fnt_version():
    try:
        from importlib.metadata import version
        return version("fnt")
    except Exception:
        return "unknown"


def load_and_process(recording, settings: Settings):
    """(DoricFileData, PhotometryResult) for one recording: what the preview shows."""
    reader = open_reader(recording)
    data = reader.scan_file()
    for attr, override in (("signal_channel", settings.signal_channel),
                           ("isosbestic_channel", settings.control_channel)):
        if override:
            match = [c for c in data.all_channels if c.path == override]
            if not match:
                raise ValueError(f"channel {override} not in {Path(recording).name}")
            setattr(data, attr, match[0])
    if data.signal_channel is None:
        raise ValueError(f"no signal channel found in {Path(recording).name}")
    data = reader.load_data(data)
    kwargs = settings.processing_kwargs()
    if isinstance(reader, LabJackFileReader):
        kwargs["remove_dropouts"] = False        # wired rig: no radio link to drop out
    r = process(data.signal_data, data.isosbestic_data, data.signal_time,
                data.signal_channel.sampling_rate, settings.pipeline, **kwargs)
    return data, r


def process_recording(recording, settings: Settings, out_dir, log: Callable[[str], None] = print,
                      should_stop: Optional[Callable[[], bool]] = None, skip_existing: bool = False) -> dict:
    """Process one recording into ``out_dir``; returns its summary row."""
    recording = Path(recording)
    paths = output_paths(recording, out_dir)
    if skip_existing and paths["settings"].exists():
        log(f"Already done, skipped: {recording.name}")
        return {"recording": recording.name, "status": "skipped"}
    Path(out_dir).mkdir(parents=True, exist_ok=True)

    data, r = load_and_process(recording, settings)
    fs = data.signal_channel.sampling_rate
    duration = float(data.signal_time[-1] - data.signal_time[0])
    log(f"Signal {data.signal_channel.name} ({data.signal_channel.wavelength} nm), control "
        f"{data.isosbestic_channel.name if data.isosbestic_channel else 'none'}, {fs:.1f} Hz, "
        f"{duration:.1f} s")
    log(f"Pipeline {settings.pipeline}: dropouts {100 * r.info['dropout_fraction']:.1f}% of samples; "
        f"dF/F SD {np.nanstd(r.dff):.2f}%, range {np.nanmin(r.dff):.1f} to {np.nanmax(r.dff):.1f}%")

    pd.DataFrame({"Time_sec": r.time, "Signal_V": r.signal, "Control_V": r.control,
                  "FittedControl_V": r.fitted_control, "DeltaF_F_pct": r.dff, "ZScore": r.z,
                  "Dropout": r.dropout.astype(int)}).to_csv(paths["photometry"], index=False)

    video = data.video_info.absolute_path if data.video_info else None
    clock = None
    if video and Path(video).exists():
        ft, fps = recording_frame_times(data, video)
        clock = check_video_clock(ft, data.video_timestamps)
        if "logged_frames" in clock and clock["logged_frames"] != clock["video_frames"]:
            how = ("by the recording's frame clock, in order" if data.video_frame_clock
                   else f"at {fps:g} fps from the video")
            log(f"Video has {clock['video_frames']:,} frames; the recording logged "
                f"{clock['logged_frames']:,} frame times over {clock['logged_span_s']:.1f} s. "
                f"Frames are timed {how}.")
        frames = pd.DataFrame({"Frame": np.arange(1, len(ft) + 1), "Time_sec": ft,
                               "DeltaF_F": align_to_frames(r.time, r.dff, ft),
                               "ZScore": align_to_frames(r.time, r.z, ft)})
    else:
        if video:
            log(f"Video not found: {video}")
        video = None
        frames = pd.DataFrame({"Sample": np.arange(1, len(r.time) + 1), "Time_sec": r.time,
                               "DeltaF_F": r.dff, "ZScore": r.z})
    frames.to_csv(paths["frames"], index=False)

    made_video = False
    if settings.create_video and video:
        from .video_overlay import AMBER, BLUE, GREEN, VIOLET, Row, render_combined_video
        rows = []
        if settings.show_raw:
            wl = data.signal_channel.wavelength
            rows.append(Row(f"{wl} nm (V)" if wl else "Signal (V)",
                            r.time, np.where(r.dropout, np.nan, r.signal), GREEN,
                            second=np.where(r.dropout, np.nan, r.fitted_control),
                            second_colour=VIOLET, second_label="fitted control"))
        rows.append(Row("dF/F (%)", r.time, r.dff, BLUE, zero_line=True))
        rows.append(Row("z-score", r.time, r.z, AMBER, zero_line=True))
        log(f"Writing {paths['video'].name} ...")
        made_video = render_combined_video(
            video, str(paths["video"]), rows, window_s=settings.video_window_s,
            progress=lambda k, n: log(f"  frame {k:,} / {n:,}") if k % 3000 == 0 else None,
            should_stop=should_stop, frame_times=ft)

    row = {"recording": recording.name, "status": "done", "pipeline": settings.pipeline,
           "duration_s": round(duration, 2), "sampling_rate_hz": round(float(fs), 2),
           "dropout_pct": round(100 * r.info["dropout_fraction"], 2),
           "dff_sd_pct": round(float(np.nanstd(r.dff)), 3),
           "dff_robust_sd_pct": round(float(1.4826 * np.nanmedian(np.abs(r.dff - np.nanmedian(r.dff)))), 3),
           "fit_slope": r.info.get("slope"), "fit_intercept": r.info.get("intercept"),
           "video_frames": clock["video_frames"] if clock else None,
           "video_timing": ("frame clock" if data.video_frame_clock else "video fps") if clock else None,
           "logged_frames": clock.get("logged_frames") if clock else None,
           "combined_video": made_video}
    record = {"recording": str(recording), "settings": asdict(settings), "fit": r.info,
              "video": video, "video_clock": clock, "summary": row, "fnt_version": _fnt_version(),
              "host": platform.node(), "finished": datetime.now().isoformat(timespec="seconds")}
    paths["settings"].write_text(json.dumps(record, indent=2, default=str), encoding="utf-8")
    return row


def run_batch(recordings: List[Path], settings: Settings, out_dir: Optional[Path] = None,
              log: Callable[[str], None] = print, should_stop: Optional[Callable[[], bool]] = None,
              on_file: Optional[Callable[[int, dict], None]] = None) -> Path:
    """Process ``recordings`` into one run folder; returns it.

    ``out_dir`` None: a new dated run folder in the recordings' folder (all recordings must share
    one folder). Given: that folder, and recordings already done in it are skipped (resume).
    """
    recordings = [Path(r) for r in recordings]
    if not recordings:
        raise ValueError("no recordings to process")
    resume = out_dir is not None
    if out_dir is None:
        parents = {r.parent.resolve() for r in recordings}
        if len(parents) > 1:
            raise ValueError("recordings come from several folders; give an output folder")
        out_dir = new_run_folder(recordings[0].parent, settings.pipeline)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    lines: List[str] = []

    def say(msg):
        lines.append(f"{datetime.now():%H:%M:%S} {msg}")
        log(msg)

    (out_dir / "run_settings.json").write_text(json.dumps(
        {"settings": asdict(settings), "recordings": [str(r) for r in recordings],
         "started": datetime.now().isoformat(timespec="seconds"), "fnt_version": _fnt_version(),
         "host": platform.node()}, indent=2, default=str), encoding="utf-8")
    say(f"Run folder: {out_dir}")
    rows = []
    for k, rec in enumerate(recordings):
        if should_stop and should_stop():
            say("Stopped.")
            break
        say(f"[{k + 1}/{len(recordings)}] {rec.name}")
        try:
            row = process_recording(rec, settings, out_dir, log=say, should_stop=should_stop,
                                    skip_existing=resume)
        except Exception as e:
            say(f"Error: {e}")
            row = {"recording": rec.name, "status": f"failed: {e}"}
        rows.append(row)
        if on_file:
            on_file(k, row)
    with open(out_dir / "run.log", "a", encoding="utf-8") as f:
        f.write("\n".join(lines) + "\n")
    write_summary(out_dir)
    return out_dir


def write_summary(out_dir: Path) -> Optional[Path]:
    """``FiberPhotometry_summary.csv``: one row per recording processed in ``out_dir``."""
    rows = []
    for f in sorted(Path(out_dir).glob("*_settings.json")):
        try:
            rows.append(json.loads(f.read_text(encoding="utf-8"))["summary"])
        except Exception:
            continue
    if not rows:
        return None
    path = Path(out_dir) / SUMMARY_FILE
    cols = list(dict.fromkeys(k for r in rows for k in r))
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        w.writerows(rows)
    return path
