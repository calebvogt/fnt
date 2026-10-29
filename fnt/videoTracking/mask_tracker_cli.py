"""Mask Tracker from the command line, without the GUI.

``python -m fnt.videoTracking.mask_tracker_cli extract-frames --project P --frames F.csv``
cuts the listed frames into the project's ``training_frames/`` under the names
the GUI gives them (``{video stem}_frame_{index:06d}.png``) and records each
video in the project's source registry, so the GUI links every frame back to
its video. ``--focus-out`` writes the frame names as a list for
``python -m fnt.videoTracking.mask_tracker_gui --project P --focus LIST``.

The CSV needs columns ``video`` (path) and ``frame`` (0-based index). Frames
already in ``training_frames/`` are kept, not rewritten. Run it while the
project is not open in the GUI: both write ``project_config.json``.

``track --videos V... --model M`` tracks every frame of whole videos in the
Mask Tracker window (the Inference tab, several videos at once with
``--parallel auto``), with every Inference setting as a flag;
``--settings-from`` repeats an earlier run's settings from its output folder.
``track-summary --videos V...`` writes each video folder's
``MaskTracker_summary.csv`` (see :mod:`fnt.videoTracking.mtt_runs`).

``pack-session --project P --focus LIST --out DIR`` copies a focus list's frames
and labels into a portable session folder that opens as a project on any
machine with FNT; ``unpack-session --session S`` merges the labels made there
back into the project (see :mod:`fnt.videoTracking.mtt_portable`). Unpack while
the project is not open in the GUI: both write ``annotations.json``.
"""
from __future__ import annotations

import argparse
import concurrent.futures
import csv
import json
import os
import sys
from pathlib import Path
from typing import Dict, List, Tuple

#: Same as the GUI's extraction: lossless, fast to write.
PNG_LEVEL = 1


def resolve_project_config(project: str) -> str:
    """project_config.json for a project folder or the file itself."""
    path = project
    if os.path.isdir(path):
        path = os.path.join(path, "project_config.json")
    if not os.path.isfile(path):
        raise FileNotFoundError(f"no Mask Tracker project at {project}")
    return path


def project_of_model(model_dir: str) -> str:
    """The project a model run folder belongs to (``<project>/models/<run>``), or ""."""
    proj = os.path.dirname(os.path.dirname(os.path.normpath(model_dir)))
    return proj if os.path.isfile(os.path.join(proj, "project_config.json")) else ""


def _on_off(group, name: str, help_on: str, help_off: str) -> None:
    dest = name.replace("-", "_")
    group.add_argument(f"--{name}", dest=dest, action="store_true", default=None, help=help_on)
    group.add_argument(f"--no-{name}", dest=dest, action="store_false", help=help_off)


def add_inference_settings(g) -> None:
    """The Inference tab's settings as flags (shared by ``track`` and the
    window's ``--infer``). A flag not given keeps the tab's value, or the
    ``--settings-from`` run's."""
    g.add_argument("--settings-from", dest="settings_from", default=None, metavar="PATH",
                   help="Repeat an earlier run: a <video>_MaskTracker folder or its "
                        "run_settings.json (model and settings; flags given here win).")
    g.add_argument("--parallel", default=None, metavar="N",
                   help="Track several videos at once: N of them, or 'auto'.")
    g.add_argument("--max-objects", dest="max_objects", type=int, default=None,
                   help="Max objects per frame (0 = unlimited).")
    g.add_argument("--conf", type=float, default=None, help="Confidence threshold.")
    g.add_argument("--max-disappear", dest="max_disappear", default=None, metavar="N",
                   help="Frames a track may go undetected before it ends: 'never' or N.")
    g.add_argument("--iou", type=float, default=None, help="IoU threshold for matching.")
    g.add_argument("--matching", choices=["hungarian", "greedy"], default=None)
    g.add_argument("--resolution", default=None, metavar="PX",
                   help="Inference size: 'trained' or 256/384/512/640/800.")
    g.add_argument("--infer-device", dest="infer_device", default=None,
                   choices=["auto", "cpu", "cuda", "mps"], help="Device for inference.")
    _on_off(g, "masks", "Predict masks (centroids from the mask).", "Boxes only.")
    _on_off(g, "tracked-video", "Write the annotated video.", "No annotated video.")
    _on_off(g, "save-masks", "Save every frame's masks for proofreading.", "Don't.")
    _on_off(g, "preview", "Show the tracking preview.", "No preview.")
    g.add_argument("--classifier", default=None, metavar="DIR",
                   help="Behavior classifier folder ('none' to turn classification off).")
    g.add_argument("--nc-threshold", dest="nc_threshold", type=float, default=None)
    g.add_argument("--cls-window", dest="cls_window", type=int, default=None)
    g.add_argument("--min-bout", dest="min_bout", type=int, default=None)
    g.add_argument("--uncertain-gap", dest="uncertain_gap", type=float, default=None)
    g.add_argument("--quit-when-done", dest="quit_when_done", action="store_true",
                   help="Close the window when the run ends.")


#: (flag, dest) of add_inference_settings's value flags, for settings_argv.
_VALUE_FLAGS = [("--settings-from", "settings_from"), ("--parallel", "parallel"),
                ("--max-objects", "max_objects"), ("--conf", "conf"),
                ("--max-disappear", "max_disappear"), ("--iou", "iou"),
                ("--matching", "matching"), ("--resolution", "resolution"),
                ("--infer-device", "infer_device"), ("--classifier", "classifier"),
                ("--nc-threshold", "nc_threshold"), ("--cls-window", "cls_window"),
                ("--min-bout", "min_bout"), ("--uncertain-gap", "uncertain_gap")]
_ON_OFF_FLAGS = [("masks", "masks"), ("tracked-video", "tracked_video"),
                 ("save-masks", "save_masks"), ("preview", "preview")]


def settings_argv(args) -> List[str]:
    """The inference settings given in ``args``, as flags again."""
    out: List[str] = []
    for flag, dest in _VALUE_FLAGS:
        v = getattr(args, dest, None)
        if v is not None:
            out += [flag, str(v)]
    for name, dest in _ON_OFF_FLAGS:
        v = getattr(args, dest, None)
        if v is not None:
            out.append(f"--{name}" if v else f"--no-{name}")
    if getattr(args, "quit_when_done", False):
        out.append("--quit-when-done")
    return out


def inference_settings(args) -> Dict:
    """The settings asked for: ``--settings-from``'s, then the flags given.

    Keys follow MaskInferenceConfig, plus create_tracked_video, show_preview,
    parallel, classifier_dir (None: off) and its four settings, and model_dir
    (from --settings-from). Only keys with a value are present.
    """
    out: Dict = {}
    if getattr(args, "settings_from", None):
        from .mtt_runs import read_run_settings
        rec = read_run_settings(args.settings_from)
        out.update(rec.get("settings", {}))
        out.pop("model_path", None)
        out["create_tracked_video"] = rec.get("create_tracked_video", True)
        out["classifier_dir"] = rec.get("classifier")
        out.update(rec.get("classifier_settings") or {})
        out["model_dir"] = rec.get("model_dir")
    flags = {
        "max_objects": "max_detections", "conf": "confidence_threshold",
        "iou": "iou_match_threshold", "matching": "matching_algorithm",
        "infer_device": "device", "masks": "use_masks", "tracked_video": "create_tracked_video",
        "save_masks": "save_masks", "preview": "show_preview", "nc_threshold": "nc_threshold",
        "cls_window": "cls_window", "min_bout": "min_bout", "uncertain_gap": "uncertain_gap",
    }
    for a, k in flags.items():
        v = getattr(args, a, None)
        if v is not None:
            out[k] = v
    md = getattr(args, "max_disappear", None)
    if md is not None:
        out["max_disappeared_frames"] = 0 if str(md).lower() == "never" else int(md)
    res = getattr(args, "resolution", None)
    if res is not None:
        out["inference_size"] = 0 if str(res).lower() == "trained" else int(str(res).rstrip("px"))
    cls = getattr(args, "classifier", None)
    if cls is not None:
        out["classifier_dir"] = None if cls.lower() == "none" else cls
    par = getattr(args, "parallel", None)
    if par is not None:
        if par != "auto" and not str(par).isdigit():
            raise ValueError("--parallel takes a number or 'auto'")
        out["parallel"] = par
    return out


def frame_name(video: str, index: int) -> str:
    """The training-frame filename the GUI gives frame ``index`` of ``video``."""
    return f"{Path(video).stem}_frame_{int(index):06d}.png"


def read_frame_list(path: str) -> List[Tuple[str, int]]:
    """(video, frame) pairs from a CSV with columns ``video`` and ``frame``."""
    with open(path, newline="", encoding="utf-8") as fh:
        rows = list(csv.DictReader(fh))
    if rows and not {"video", "frame"} <= set(rows[0]):
        raise ValueError(f"{path}: needs columns 'video' and 'frame'")
    return [(r["video"], int(float(r["frame"]))) for r in rows]


def _extract_video(video: str, indices: List[int], out_dir: str) -> Tuple[List[str], List[str]]:
    """Write the listed frames of one video; returns (frame names, problems)."""
    import cv2
    names, problems = [], []
    todo = []
    for idx in sorted(set(indices)):
        name = frame_name(video, idx)
        if os.path.exists(os.path.join(out_dir, name)):
            names.append(name)
        else:
            todo.append((idx, name))
    if not todo:
        return names, problems
    cap = cv2.VideoCapture(video)
    if not cap.isOpened():
        return names, problems + [f"cannot open {video}"]
    try:
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        for idx, name in todo:
            if total and idx >= total:
                problems.append(f"{os.path.basename(video)}: frame {idx} beyond {total}")
                continue
            cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
            ok, frame = cap.read()
            if not ok:
                problems.append(f"{os.path.basename(video)}: frame {idx} unreadable")
                continue
            cv2.imwrite(os.path.join(out_dir, name), frame,
                        [cv2.IMWRITE_PNG_COMPRESSION, PNG_LEVEL])
            names.append(name)
    finally:
        cap.release()
    return names, problems


def register_source_videos(config_path: str, videos: List[str]) -> int:
    """Append videos to the project's ``source_videos`` registry (no duplicates)."""
    with open(config_path, encoding="utf-8") as fh:
        cfg = json.load(fh)
    reg = list(cfg.get("source_videos", []))
    have = set(reg)
    added = [v for v in dict.fromkeys(videos) if v not in have]
    if added:
        cfg["source_videos"] = reg + added
        tmp = config_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as fh:
            json.dump(cfg, fh, indent=2)
        os.replace(tmp, config_path)
    return len(added)


def extract_frames(project: str, pairs: List[Tuple[str, int]], workers: int = 4) -> Dict:
    """Cut ``pairs`` into ``project``/training_frames; register their videos.

    Returns {"frames": names in input order, "problems": [...], "registered": n}.
    """
    config_path = resolve_project_config(project)
    out_dir = os.path.join(os.path.dirname(config_path), "training_frames")
    os.makedirs(out_dir, exist_ok=True)
    by_video: Dict[str, List[int]] = {}
    for video, idx in pairs:
        by_video.setdefault(video, []).append(idx)
    written, problems = set(), []
    with concurrent.futures.ThreadPoolExecutor(max_workers=max(1, workers)) as ex:
        futs = [ex.submit(_extract_video, v, ix, out_dir) for v, ix in by_video.items()]
        for fut in concurrent.futures.as_completed(futs):
            names, probs = fut.result()
            written.update(names)
            problems.extend(probs)
    ok_videos = [v for v in by_video
                 if any(frame_name(v, i) in written for i in by_video[v])]
    registered = register_source_videos(config_path, ok_videos)
    ordered = [frame_name(v, i) for v, i in pairs if frame_name(v, i) in written]
    return {"frames": list(dict.fromkeys(ordered)), "problems": problems,
            "registered": registered}


def _parser():
    p = argparse.ArgumentParser(prog="mask-tracker-cli",
                                description="FNT Mask Tracker without the GUI.")
    sub = p.add_subparsers(dest="cmd", required=True)
    e = sub.add_parser("extract-frames",
                       help="Cut listed video frames into a project's training_frames/.")
    e.add_argument("--project", required=True,
                   help="Project folder or its project_config.json.")
    e.add_argument("--frames", required=True,
                   help="CSV with columns video (path) and frame (0-based index).")
    e.add_argument("--focus-out", default="",
                   help="Write the frame names here, one per line, for --focus.")
    e.add_argument("--workers", type=int, default=4,
                   help="Videos read in parallel (default 4).")
    b = sub.add_parser("infer-videos",
                       help="Segment every Nth frame of whole videos with a trained model; "
                            "one <video>_masks.h5 per video (resumes where it stopped).")
    b.add_argument("--model", required=True, help="Model run folder (or a .pt file).")
    b.add_argument("--videos", nargs="+", required=True,
                   help="Video files, folders (searched recursively) and/or .txt lists.")
    b.add_argument("--step", type=int, default=250,
                   help="Every Nth frame (default 250: one keyframe per 16.7 s at 15 fps).")
    b.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default 0.25).")
    b.add_argument("--imgsz", type=int, default=None,
                   help="Input size (default: the model's training size).")
    b.add_argument("--device", default="", help="e.g. 0 or cpu (default: auto).")
    b.add_argument("--order", choices=["coarse", "name"], default="coarse",
                   help="coarse (default): spread over every folder and camera from the "
                        "start, so a stopped run is an even sample; name: alphabetical.")
    b.add_argument("--no-window", action="store_true",
                   help="Run without the progress window (log to the console).")
    b.add_argument("--gate", action="store_true",
                   help="Choose frames by activity instead of --step: every frame where something "
                        "moves, a few a second where it may, keyframes elsewhere (MP4/MOV; read "
                        "from each file's frame sizes, see frame_activity).")
    b.add_argument("--gate-hi", type=float, default=None,
                   help="Moving: a second's frames this many times the camera's still level (default 2).")
    b.add_argument("--gate-lo", type=float, default=None,
                   help="Uncertain above this ratio (default 1.3).")
    b.add_argument("--gate-pad", type=float, default=None,
                   help="Seconds added before and after each moving second (default 1).")
    b.add_argument("--gate-low-fps", type=float, default=None,
                   help="Frames a second in uncertain seconds (default 2).")
    b.add_argument("--gate-busy-fps", type=float, default=None,
                   help="Frames a second in moving seconds (default: every frame).")
    q = sub.add_parser("predict-frames",
                       help="Predict a project's unreviewed frames with a trained model and save the "
                            "masks as pending predictions for review (the window's Predict, headless).")
    q.add_argument("--project", required=True, help="Project folder or its project_config.json.")
    q.add_argument("--model", required=True, help="Model run folder.")
    q.add_argument("--focus", default=None,
                   help="Only these frames (a focus list, one name per line); default all.")
    q.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default 0.25).")
    q.add_argument("--max-det", type=int, default=100, help="At most this many masks per frame.")
    q.add_argument("--device", default="auto",
                   help="auto (default), cpu (leaves the GPU to another job) or cuda.")
    w = sub.add_parser("behavior-window",
                       help="Masks on every frame of a stretch of video, tracks, mask-first pose and "
                            "candidate behaviour bouts (tracks.h5, tracks/pairs/bouts CSVs).")
    w.add_argument("--model", required=True, help="Model run folder (or a .pt file).")
    w.add_argument("--video", required=True)
    w.add_argument("--start", type=int, default=0, help="First frame (default 0).")
    w.add_argument("--seconds", type=float, default=60.0, help="Length (default 60 s).")
    w.add_argument("--out", required=True, help="Output folder.")
    w.add_argument("--conf", type=float, default=0.25, help="Confidence threshold (default 0.25).")
    w.add_argument("--imgsz", type=int, default=None,
                   help="Input size (default: the model's training size).")
    w.add_argument("--device", default="", help="e.g. 0 or cpu (default: auto).")
    w.add_argument("--crops", type=int, default=0,
                   help="Also save nose-up body crops of this size (px) for embeddings.")
    s_ = sub.add_parser("behavior-session",
                        help="Pool windows' candidate bouts into one behavior_review session CSV.")
    s_.add_argument("--bouts", nargs="+", required=True, help="*_bouts.csv files.")
    s_.add_argument("--out", required=True, help="Session CSV to write.")
    s_.add_argument("--per-kind", type=int, default=0,
                    help="Keep at most this many bouts of each kind, taken in turn from the windows.")
    r_ = sub.add_parser("behavior-rebout",
                        help="Rebuild windows' candidate bouts with the current rules from their saved "
                             "tracks and pairs CSVs (no detection).")
    r_.add_argument("--bouts", nargs="+", required=True, help="*_bouts.csv files to rewrite.")
    k_ = sub.add_parser("pack-session",
                        help="Copy a focus list's frames and labels into a portable session folder "
                             "(e.g. on OneDrive) that opens as a project on any machine with FNT.")
    k_.add_argument("--project", required=True, help="Project folder or its project_config.json.")
    k_.add_argument("--focus", required=True, help="Focus list: frame names, one per line.")
    k_.add_argument("--out", required=True, help="Folder the session folder goes into.")
    k_.add_argument("--name", default=None, help="Session folder name (default: project + time).")
    k_.add_argument("--note", default="", help="A line for the README (what to do).")
    u_ = sub.add_parser("unpack-session",
                        help="Merge a session's labels back into its project, frame by frame.")
    u_.add_argument("--session", required=True, help="The session folder.")
    u_.add_argument("--project", default=None,
                    help="Merge into this project instead of the one it was packed from.")
    u_.add_argument("--force", action="store_true",
                    help="Overwrite frames whose labels changed in the project since packing.")
    u_.add_argument("--dry-run", action="store_true", help="Report what would merge; write nothing.")
    t_ = sub.add_parser("track",
                        help="Track every frame of whole videos in the Mask Tracker window (the "
                             "Inference tab): trajectories, annotated video, masks; several at "
                             "once with --parallel auto.")
    t_.add_argument("--videos", nargs="+", required=True,
                    help="Video files, folders (searched recursively; output folders are "
                         "skipped) and/or .txt lists.")
    t_.add_argument("--model", default=None,
                    help="Model run folder (default: --settings-from's model).")
    t_.add_argument("--project", default=None,
                    help="Project to open (default: the one the model folder is in).")
    t_.add_argument("--redo", action="store_true",
                    help="Track videos that already have output too (their output folder is "
                         "moved to _TO_DELETE_tracking/ first, not deleted).")
    add_inference_settings(t_.add_argument_group("settings"))
    ts = sub.add_parser("track-summary",
                        help="Write MaskTracker_summary.csv in each video folder (one row per "
                             "tracked video: frames, frames without a detection, tracks, speed, "
                             "settings) and print it.")
    ts.add_argument("--videos", nargs="+", required=True,
                    help="Video folders (searched recursively), files and/or .txt lists.")
    x_ = sub.add_parser("exits",
                        help="Draw each camera's exits (e.g. an entrance tube's mouth) on its "
                             "median frame; saved as exits.json in each video folder.")
    x_.add_argument("--videos", nargs="+", required=True,
                    help="Video files and/or folders (searched recursively).")
    x_.add_argument("--focus", default=None,
                    help="Text file of folder/camera names (e.g. T001_PAH/CAM14), one per line, "
                         "shown first among the unchecked.")
    return p


def summary_folders(items) -> List[str]:
    """Every folder holding videos named by ``items`` (folders searched recursively)."""
    from .mtt_runs import OUTPUT_SUFFIX, find_track_videos
    folders = []
    for it in items:
        if os.path.isdir(it):
            for root, dirs, _ in os.walk(it):
                dirs[:] = [d for d in dirs if not d.endswith(OUTPUT_SUFFIX)]
                folders.append(root)
    folders += [os.path.dirname(v) for v in
                find_track_videos([i for i in items if not os.path.isdir(i)])]
    seen, out = set(), []
    for f in folders:
        k = os.path.normcase(os.path.abspath(f))
        if k not in seen:
            seen.add(k)
            out.append(f)
    return out


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
    if args.cmd == "track":
        from .mask_tracker_gui import track_main
        return track_main(args)
    if args.cmd == "track-summary":
        from .mtt_runs import folder_summary, print_summary, write_folder_summary
        for f in summary_folders(args.videos):
            path = write_folder_summary(f)
            if path:
                print(f"== {path}")
                print_summary(folder_summary(f))
        return 0
    if args.cmd == "extract-frames":
        res = extract_frames(args.project, read_frame_list(args.frames), args.workers)
        if args.focus_out:
            with open(args.focus_out, "w", encoding="utf-8") as fh:
                fh.write("\n".join(res["frames"]) + "\n")
        for msg in res["problems"]:
            print(f"  {msg}", file=sys.stderr)
        print(f"{len(res['frames'])} frame(s) in training_frames/, "
              f"{res['registered']} video(s) newly registered, "
              f"{len(res['problems'])} problem(s)", flush=True)
        return 1 if res["problems"] and not res["frames"] else 0
    if args.cmd == "infer-videos":
        from .mask_tracker_batch import find_videos, order_coarse_to_fine, run_batch
        videos = find_videos(args.videos)
        if args.order == "coarse":
            videos = order_coarse_to_fine(videos)
        if not videos:
            print("no videos found", file=sys.stderr)
            return 1
        gate = None
        if args.gate:
            gate = {k: v for k, v in (("hi", args.gate_hi), ("lo", args.gate_lo),
                                      ("pad_s", args.gate_pad), ("low_fps", args.gate_low_fps),
                                      ("busy_fps", args.gate_busy_fps)) if v is not None}
        if args.no_window:
            res = run_batch(args.model, videos, args.step, args.conf, args.imgsz, args.device,
                            gate=gate)
            return 0 if res["done"] == res["videos"] else 1
        from .mask_tracker_batch_window import run_with_window
        return run_with_window(args.model, videos, args.step, args.conf, args.imgsz, args.device,
                               gate)
    if args.cmd == "predict-frames":
        from .mtt_portable import read_focus
        from .mtt_predict import predict_frames
        if args.device == "cpu":
            os.environ.setdefault("CUDA_VISIBLE_DEVICES", "-1")
        project = os.path.dirname(args.project) if args.project.endswith(".json") else args.project
        predict_frames(project, args.model, read_focus(args.focus) if args.focus else None,
                       args.conf, args.max_det, args.device)
        return 0
    if args.cmd == "behavior-window":
        import cv2
        from .behavior_features import run_window, yolo_detector
        cap = cv2.VideoCapture(args.video)
        fps = cap.get(cv2.CAP_PROP_FPS) or 15.0
        cap.release()
        n = int(round(args.seconds * fps))

        def progress(i, total):
            if i % 150 == 0 or i == total:
                print(f"  {i}/{total} frames", flush=True)

        paths = run_window(args.video, args.start, n, args.out,
                           yolo_detector(args.model, args.conf, args.imgsz, args.device),
                           os.path.basename(os.path.normpath(args.model)), args.crops, progress)
        for k, v in paths.items():
            print(f"{k}: {v}")
        return 0
    if args.cmd == "behavior-session":
        from .behavior_features import build_session
        print(f"{build_session(args.bouts, args.out, args.per_kind)} bouts -> {args.out}")
        return 0
    if args.cmd == "behavior-rebout":
        from .behavior_features import rebuild_bouts
        for p in args.bouts:
            print(f"{rebuild_bouts(p)} bouts: {os.path.basename(p)}", flush=True)
        return 0
    if args.cmd == "pack-session":
        from .mtt_portable import pack_session, read_focus
        path = pack_session(args.project, read_focus(args.focus), args.out, args.name, args.note)
        print(f"session: {path}")
        return 0
    if args.cmd == "unpack-session":
        from .mtt_portable import unpack_session
        try:
            res = unpack_session(args.session, args.project, args.force, args.dry_run)
        except RuntimeError as e:
            print(str(e), file=sys.stderr)
            return 1
        print(f"{'would merge' if args.dry_run else 'merged'} {res['frames']} frame(s)"
              f"{'; overwrote ' + str(len(res['conflicts'])) + ' changed frame(s)' if res['conflicts'] else ''}"
              f"{'; backup ' + res['backup'] if res['backup'] else ''}")
        return 0
    if args.cmd == "exits":
        from PyQt5.QtWidgets import QApplication
        from .mtt_exits import cameras_in
        from .mtt_exits_editor import ExitsWindow
        focus = []
        if args.focus:
            with open(args.focus, encoding="utf-8") as fh:
                focus = [ln.strip() for ln in fh if ln.strip()]
        cams = cameras_in(args.videos)
        if not cams:
            print("no videos found", file=sys.stderr)
            return 1
        app = QApplication.instance() or QApplication(sys.argv)
        w = ExitsWindow(cams, focus)
        w.show()
        return app.exec_()
    return 2


if __name__ == "__main__":
    sys.exit(main())
