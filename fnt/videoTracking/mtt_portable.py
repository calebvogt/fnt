"""Portable Mask Tracker labelling sessions.

A focus list can be labelled anywhere FNT is installed, not only on the machine
that holds the project: ``pack_session`` copies the list's frames and their
labels into a self-contained session folder (e.g. on OneDrive), which opens as
an ordinary Mask Tracker project; ``unpack_session`` merges the labels made
there back into the project, frame by frame.

A session folder::

    <session>/
        project_config.json      # a project of its own: categories, no videos
        annotations/annotations.json   # the frames' labels, predictions and flags
        training_frames/*.png    # the frames (copies)
        focus.txt                # the frames, in the list's order
        fnt_session.json         # where it came from, when, and each frame's
                                 # fingerprint at packing (the merge's conflict check)
        README.txt               # how to open it, and how to hand it back

Merging replaces each session frame's labels in the project with the session's
(empty and skipped flags too). A frame whose labels changed in the project
after packing is a conflict: the merge stops and names it, unless forced.
"""
from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
from datetime import datetime
from typing import Dict, List, Optional

SESSION_FILE = "fnt_session.json"
_FLAGS = ("empty", "skipped")


def _read_json(path: str) -> Dict:
    with open(path, encoding="utf-8") as fh:
        return json.load(fh)


def _write_json(path: str, data: Dict) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    os.replace(tmp, path)


def _project_paths(project: str):
    cfg = project if project.endswith(".json") else os.path.join(project, "project_config.json")
    if not os.path.isfile(cfg):
        raise FileNotFoundError(f"no Mask Tracker project at {project}")
    d = os.path.dirname(os.path.abspath(cfg))
    return d, cfg, os.path.join(d, "annotations", "annotations.json"), os.path.join(d, "training_frames")


def _load_coco(path: str) -> Dict:
    if os.path.isfile(path):
        c = _read_json(path)
    else:
        c = {}
    for k in ("images", "annotations", "categories"):
        c.setdefault(k, [])
    return c


def _frame_state(coco: Dict, image: Optional[Dict]) -> Dict:
    """A frame's labels and flags, without ids (comparable across files)."""
    if image is None:
        return {"flags": {}, "anns": []}
    cats = {c["id"]: c["name"] for c in coco["categories"]}
    anns = []
    for a in coco["annotations"]:
        if a.get("image_id") == image["id"]:
            b = {k: v for k, v in a.items() if k not in ("id", "image_id")}
            b["category"] = cats.get(b.pop("category_id", None))
            anns.append(b)
    anns.sort(key=lambda b: json.dumps(b, sort_keys=True))
    return {"flags": {k: image[k] for k in _FLAGS if k in image}, "anns": anns}


def _fingerprint(state: Dict) -> str:
    return hashlib.sha1(json.dumps(state, sort_keys=True).encode()).hexdigest()


def _images_by_name(coco: Dict) -> Dict[str, Dict]:
    return {os.path.basename(i["file_name"]): i for i in coco["images"]}


def read_focus(path: str) -> List[str]:
    with open(path, encoding="utf-8") as fh:
        return [ln.strip() for ln in fh if ln.strip()]


def pack_session(project: str, frames: List[str], out_root: str, name: Optional[str] = None,
                 note: str = "") -> str:
    """Copy ``frames`` (names in the project's training_frames/) and their labels
    into a new session folder under ``out_root``; returns its path."""
    pdir, cfg_path, ann_path, frames_dir = _project_paths(project)
    cfg = _read_json(cfg_path)
    coco = _load_coco(ann_path)
    name = name or f"{os.path.basename(pdir)}_{datetime.now():%Y%m%d_%H%M%S}"
    sdir = os.path.join(out_root, name)
    if os.path.exists(sdir):
        raise FileExistsError(f"{sdir} already exists")
    missing = [f for f in frames if not os.path.isfile(os.path.join(frames_dir, f))]
    if missing:
        raise FileNotFoundError(f"{len(missing)} frame(s) not in {frames_dir}: {missing[:3]}")
    os.makedirs(os.path.join(sdir, "training_frames"))
    os.makedirs(os.path.join(sdir, "annotations"))
    for f in frames:
        shutil.copy2(os.path.join(frames_dir, f), os.path.join(sdir, "training_frames", f))
    by_name = _images_by_name(coco)
    keep_ids = {by_name[f]["id"] for f in frames if f in by_name}
    sub = {"images": [i for i in coco["images"] if i["id"] in keep_ids],
           "annotations": [a for a in coco["annotations"] if a.get("image_id") in keep_ids],
           "categories": coco["categories"]}
    _write_json(os.path.join(sdir, "annotations", "annotations.json"), sub)
    _write_json(os.path.join(sdir, "project_config.json"), {
        "project_dir": ".", "video_paths": [], "source_videos": [],
        "categories": cfg.get("categories", [c["name"] for c in coco["categories"]]),
        "sam2_model_path": None})
    with open(os.path.join(sdir, "focus.txt"), "w", encoding="utf-8") as fh:
        fh.write("\n".join(frames) + "\n")
    _write_json(os.path.join(sdir, SESSION_FILE), {
        "kind": "mask_tracker",
        "status": "packed",
        "origin_project": pdir,
        "origin_host": platform.node(),
        "packed_at": datetime.now().isoformat(timespec="seconds"),
        "note": note,
        "frames": frames,
        "fingerprints": {f: _fingerprint(_frame_state(coco, by_name.get(f))) for f in frames},
    })
    with open(os.path.join(sdir, "README.txt"), "w", encoding="utf-8") as fh:
        fh.write(
            f"Mask Tracker labelling session: {len(frames)} frame(s) from {pdir}\n"
            f"packed {datetime.now():%Y-%m-%d %H:%M} on {platform.node()}. {note}\n\n"
            "Open it (any machine with FNT; update FNT first):\n"
            "  python -m fnt.videoTracking.mask_tracker_gui --project <this folder> "
            "--focus <this folder>/focus.txt\n\n"
            "Label as usual; everything saves into this folder. When you are done, set\n"
            '"status" in fnt_session.json to "done" (or tell Claude), and close Mask Tracker.\n'
            "The home PC merges the labels back into the project and clears this folder.\n")
    return sdir


def unpack_session(session: str, project: Optional[str] = None, force: bool = False,
                   dry_run: bool = False) -> Dict:
    """Merge a session's labels back into its project (or ``project``).

    Returns {"frames": n merged, "conflicts": [...], "added_categories": [...],
    "backup": path}. Refuses (raises RuntimeError) when a frame changed in the
    project since packing, unless ``force``."""
    meta = _read_json(os.path.join(session, SESSION_FILE))
    pdir, _, ann_path, _ = _project_paths(project or meta["origin_project"])
    proj = _load_coco(ann_path)
    sess = _load_coco(os.path.join(session, "annotations", "annotations.json"))
    p_by, s_by = _images_by_name(proj), _images_by_name(sess)
    frames = meta.get("frames", [])
    conflicts = [f for f in frames
                 if _fingerprint(_frame_state(proj, p_by.get(f))) != meta["fingerprints"].get(f)]
    if conflicts and not force:
        raise RuntimeError(f"{len(conflicts)} frame(s) changed in the project since packing "
                           f"(e.g. {conflicts[:3]}); merge with force=True to overwrite them")
    out = {"frames": 0, "conflicts": conflicts, "added_categories": [], "backup": None}
    # categories by name
    p_cat = {c["name"]: c["id"] for c in proj["categories"]}
    s_cat = {c["id"]: c["name"] for c in sess["categories"]}
    for cname in s_cat.values():
        if cname not in p_cat:
            cid = max([c["id"] for c in proj["categories"]] + [0]) + 1
            proj["categories"].append({"id": cid, "name": cname, "supercategory": ""})
            p_cat[cname] = cid
            out["added_categories"].append(cname)
    next_img = max([i["id"] for i in proj["images"]] + [0]) + 1
    next_ann = max([a["id"] for a in proj["annotations"]] + [0]) + 1
    for f in frames:
        si = s_by.get(f)
        pi = p_by.get(f)
        if pi is None and si is None:
            continue                                   # untouched, and not in the project either
        if pi is None:
            pi = {"id": next_img, "file_name": f, "width": si["width"], "height": si["height"]}
            next_img += 1
            proj["images"].append(pi)
        for k in _FLAGS:
            pi.pop(k, None)
            if si is not None and k in si:
                pi[k] = si[k]
        proj["annotations"] = [a for a in proj["annotations"] if a.get("image_id") != pi["id"]]
        if si is not None:
            for a in sess["annotations"]:
                if a.get("image_id") == si["id"]:
                    b = dict(a)
                    b["id"] = next_ann
                    next_ann += 1
                    b["image_id"] = pi["id"]
                    b["category_id"] = p_cat[s_cat[a["category_id"]]]
                    proj["annotations"].append(b)
        out["frames"] += 1
    if dry_run:
        return out
    if os.path.isfile(ann_path):
        out["backup"] = ann_path + f".bak-{datetime.now():%Y%m%d_%H%M%S}-session"
        shutil.copy2(ann_path, out["backup"])
    os.makedirs(os.path.dirname(ann_path), exist_ok=True)
    _write_json(ann_path, proj)
    meta["status"] = "merged"
    meta["merged_at"] = datetime.now().isoformat(timespec="seconds")
    meta["merged_into"] = pdir
    _write_json(os.path.join(session, SESSION_FILE), meta)
    return out
