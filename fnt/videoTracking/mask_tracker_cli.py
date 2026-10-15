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
    return p


def main(argv=None) -> int:
    args = _parser().parse_args(argv)
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
    return 2


if __name__ == "__main__":
    sys.exit(main())
