"""HDF5 storage for per-frame tracked masks (MTT inference output).

The **CSV remains the canonical tabular output** — ``trajectories.csv`` holds
centroids, boxes, areas and velocities exactly as before. This store holds the
*pixel data* the CSV cannot: the silhouette of every detection on every frame,
as COCO RLE.

Without it, anything that needs the masks again — repairing an identity swap,
re-running behavior classification at a different window size, recomputing
posture metrics — means re-running detection over the whole video. With it,
those become cheap offline passes over a file that sits next to the CSV::

    <video>_MaskTracker/<video>_tracks.h5

        /                   attrs: schema_version, video_name, video_path,
                                   fps, frame_count, width, height,
                                   model_dir, has_masks, created
        /rle_counts         int32  [total]   every RLE run concatenated
        /rle_start          int64  [N]       offset of row i into rle_counts
        /rle_len            int64  [N]       run count for row i (0 = no mask)
        /frame              int32  [N]
        /object_id          int32  [N]       editable: proofreading rewrites it
        /label              int32  [N]
        /score              float32[N]
        /bbox               float32[N, 4]    x1, y1, x2, y2
        /centroid           float32[N, 2]
        /area               int32  [N]
        /major_axis_px      float32[N]   equivalent-ellipse posture
        /minor_axis_px      float32[N]   descriptors, schema 2 onward;
        /elongation         float32[N]   NaN on box-only runs
        /orientation_deg    float32[N]
        /eccentricity       float32[N]
        /perimeter_px       float32[N]
        /solidity           float32[N]

One row per (frame, detection). Rows are appended in frame order, so a frame's
detections are contiguous, and ``/frame`` is non-decreasing.

``object_id`` is deliberately a plain dataset rather than part of a compound
type: track proofreading rewrites it in place, and the masks themselves never
move when identities are reassigned.
"""
from __future__ import annotations

import os
from datetime import datetime
from pathlib import Path
from typing import Dict, Iterator, List, Optional, Tuple

import numpy as np

try:
    import h5py
    _HAS_H5 = True
except Exception:  # pragma: no cover
    _HAS_H5 = False

from .mask_tracker_annotator import mask_to_rle, rle_to_mask
from .mask_tracker_inference import SHAPE_FIELDS

# 2 added the per-detection shape descriptors. Stores written by schema 1
# still open; they simply carry no shape columns into the rebuilt CSV.
SCHEMA_VERSION = 2
TRACKS_SUFFIX = "_tracks.h5"

# Rows buffered before a flush. Keeps peak memory flat on hour-long videos
# while still writing in chunks big enough that HDF5 resizing isn't the
# bottleneck.
_FLUSH_EVERY = 2000


def _require_h5():
    if not _HAS_H5:
        raise RuntimeError(
            "h5py is required for MTT track storage. Install with: pip install h5py"
        )


def tracks_path_for(video_path: str, output_dir: Optional[str] = None) -> str:
    """Path of the track store belonging to a video."""
    stem = Path(video_path).stem
    if output_dir is None:
        output_dir = os.path.join(
            str(Path(video_path).parent), f"{stem}_MaskTracker"
        )
    return os.path.join(output_dir, f"{stem}{TRACKS_SUFFIX}")


class TrackMaskWriter:
    """Buffered writer for per-frame track masks.

    Used as a context manager so a crashed or cancelled run still closes the
    file with whatever frames completed::

        with TrackMaskWriter(path, video_path=..., fps=30) as w:
            w.add_frame(frame_idx, matched)
    """

    _SCALARS = (
        ("rle_start", np.int64), ("rle_len", np.int64),
        ("frame", np.int32), ("object_id", np.int32), ("label", np.int32),
        ("score", np.float32), ("area", np.int32),
    )

    # Posture descriptors, stored beside the geometry rather than recomputed.
    # The proofreader rebuilds the CSV from this file, and decoding several
    # hundred thousand masks to recover shape would make saving a correction
    # take minutes. NaN on box-only runs.
    _SHAPE = tuple((name, np.float32) for name in SHAPE_FIELDS)

    def __init__(self, path: str, video_path: str = "", fps: float = 30.0,
                 frame_count: int = 0, width: int = 0, height: int = 0,
                 model_dir: str = "", has_masks: bool = True):
        _require_h5()
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        self.path = path
        self._f = h5py.File(path, "w")
        self._f.attrs["schema_version"] = SCHEMA_VERSION
        self._f.attrs["video_name"] = os.path.basename(video_path)
        self._f.attrs["video_path"] = video_path
        self._f.attrs["fps"] = float(fps)
        self._f.attrs["frame_count"] = int(frame_count)
        self._f.attrs["width"] = int(width)
        self._f.attrs["height"] = int(height)
        self._f.attrs["model_dir"] = model_dir
        self._f.attrs["has_masks"] = bool(has_masks)
        self._f.attrs["created"] = datetime.now().isoformat(timespec="seconds")

        self._f.create_dataset(
            "rle_counts", shape=(0,), maxshape=(None,), dtype=np.int32,
            chunks=(65536,), compression="gzip", compression_opts=4,
        )
        for name, dt in self._SCALARS + self._SHAPE:
            self._f.create_dataset(
                name, shape=(0,), maxshape=(None,), dtype=dt,
                chunks=(4096,), compression="gzip", compression_opts=4,
            )
        for name, cols in (("bbox", 4), ("centroid", 2)):
            self._f.create_dataset(
                name, shape=(0, cols), maxshape=(None, cols), dtype=np.float32,
                chunks=(4096, cols), compression="gzip", compression_opts=4,
            )

        self._buf: Dict[str, list] = {
            n: [] for n, _ in self._SCALARS + self._SHAPE
        }
        self._buf["bbox"] = []
        self._buf["centroid"] = []
        self._counts_buf: List[np.ndarray] = []
        self._counts_total = 0
        self.n_rows = 0

    def add_frame(self, frame_idx: int, matched: Dict[int, Dict]):
        """Record every detection matched on one frame.

        ``matched`` is the tracker's ``{object_id: detection}`` mapping, so
        what lands in the store is exactly what the tracker believed at that
        frame — including the identity assignment that may later be corrected.
        """
        for obj_id, det in matched.items():
            mask = det.get("mask")
            if mask is not None:
                counts = np.asarray(
                    mask_to_rle(np.asarray(mask).astype(bool))["counts"],
                    dtype=np.int32,
                )
            else:
                counts = np.zeros((0,), dtype=np.int32)

            self._counts_buf.append(counts)
            self._buf["rle_start"].append(self._counts_total)
            self._buf["rle_len"].append(int(counts.size))
            self._counts_total += int(counts.size)

            box = det.get("bbox")
            box = ([float(b) for b in box] if box is not None
                   else [0.0, 0.0, 0.0, 0.0])
            cx, cy = det.get("centroid", (0.0, 0.0))

            self._buf["frame"].append(int(frame_idx))
            self._buf["object_id"].append(int(obj_id))
            self._buf["label"].append(int(det.get("label", 0)))
            self._buf["score"].append(float(det.get("score", 0.0)))
            self._buf["area"].append(int(det.get("area", 0)))
            self._buf["bbox"].append(box)
            self._buf["centroid"].append([float(cx), float(cy)])

            shape = det.get("shape") or {}
            for name, _ in self._SHAPE:
                value = shape.get(name)
                self._buf[name].append(
                    float("nan") if value is None else float(value)
                )
            self.n_rows += 1

        if len(self._buf["frame"]) >= _FLUSH_EVERY:
            self.flush()

    def flush(self):
        if not self._buf["frame"]:
            return
        n_new = len(self._buf["frame"])

        if self._counts_buf:
            joined = np.concatenate(self._counts_buf) if len(self._counts_buf) > 1 \
                else self._counts_buf[0]
            ds = self._f["rle_counts"]
            old = ds.shape[0]
            ds.resize((old + joined.size,))
            if joined.size:
                ds[old:] = joined
            self._counts_buf = []

        for name, dt in self._SCALARS + self._SHAPE:
            ds = self._f[name]
            old = ds.shape[0]
            ds.resize((old + n_new,))
            ds[old:] = np.asarray(self._buf[name], dtype=dt)
            self._buf[name] = []

        for name, cols in (("bbox", 4), ("centroid", 2)):
            ds = self._f[name]
            old = ds.shape[0]
            ds.resize((old + n_new, cols))
            ds[old:] = np.asarray(self._buf[name], dtype=np.float32).reshape(-1, cols)
            self._buf[name] = []

    def close(self):
        if self._f is None:
            return
        try:
            self.flush()
            self._f.attrs["n_rows"] = int(self.n_rows)
        finally:
            self._f.close()
            self._f = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


class TrackMaskReader:
    """Read-only access to a track store, with in-place identity edits.

    Metadata columns are small enough to load eagerly (a 30-min two-animal
    video is ~100k rows); only the RLE runs stay on disk and decode on demand.
    """

    def __init__(self, path: str):
        _require_h5()
        if not os.path.exists(path):
            raise FileNotFoundError(f"No track store at {path}")
        self.path = path
        self._f = h5py.File(path, "r")
        a = self._f.attrs
        self.schema_version = int(a.get("schema_version", 0))
        self.video_name = str(a.get("video_name", ""))
        self.video_path = str(a.get("video_path", ""))
        self.fps = float(a.get("fps", 30.0))
        self.frame_count = int(a.get("frame_count", 0))
        self.width = int(a.get("width", 0))
        self.height = int(a.get("height", 0))
        self.model_dir = str(a.get("model_dir", ""))
        self.has_masks = bool(a.get("has_masks", True))

        self.frame = self._f["frame"][:]
        self.object_id = self._f["object_id"][:]
        self.label = self._f["label"][:]
        self.score = self._f["score"][:]
        self.area = self._f["area"][:]
        self.bbox = self._f["bbox"][:]
        self.centroid = self._f["centroid"][:]
        self._rle_start = self._f["rle_start"][:]
        self._rle_len = self._f["rle_len"][:]

        # Absent from schema 1 stores. Left empty rather than backfilled,
        # since recovering it would mean decoding every mask in the file.
        self.shape_features: Dict[str, np.ndarray] = {
            name: self._f[name][:] for name in SHAPE_FIELDS if name in self._f
        }

        # frame -> row slice. Rows are written in frame order, so each frame's
        # detections form one contiguous block.
        self._rows_by_frame: Dict[int, Tuple[int, int]] = {}
        if self.frame.size:
            boundaries = np.flatnonzero(np.diff(self.frame)) + 1
            starts = np.concatenate(([0], boundaries))
            ends = np.concatenate((boundaries, [self.frame.size]))
            for s, e in zip(starts, ends):
                self._rows_by_frame[int(self.frame[s])] = (int(s), int(e))

    # -- queries --------------------------------------------------------
    @property
    def n_rows(self) -> int:
        return int(self.frame.size)

    def object_ids(self) -> List[int]:
        return sorted({int(v) for v in self.object_id})

    def frames_present(self) -> List[int]:
        return sorted(self._rows_by_frame.keys())

    def rows_for_frame(self, frame_idx: int) -> List[int]:
        span = self._rows_by_frame.get(int(frame_idx))
        return list(range(span[0], span[1])) if span else []

    def frame_range_for_object(self, obj_id: int) -> Optional[Tuple[int, int]]:
        sel = self.frame[self.object_id == int(obj_id)]
        return (int(sel.min()), int(sel.max())) if sel.size else None

    def mask_for_row(self, row: int) -> Optional[np.ndarray]:
        """Decode one detection's mask, or None if the run was box-only."""
        length = int(self._rle_len[row])
        if length == 0:
            return None
        start = int(self._rle_start[row])
        counts = self._f["rle_counts"][start:start + length]
        return rle_to_mask(
            {"counts": counts.tolist(), "size": [self.height, self.width]}
        )

    def detections_for_frame(self, frame_idx: int) -> List[Dict]:
        """All detections on a frame: object id, box, score and decoded mask."""
        out = []
        for row in self.rows_for_frame(frame_idx):
            out.append({
                "row": row,
                "object_id": int(self.object_id[row]),
                "label": int(self.label[row]),
                "score": float(self.score[row]),
                "area": int(self.area[row]),
                "bbox": self.bbox[row].tolist(),
                "centroid": self.centroid[row].tolist(),
                "mask": self.mask_for_row(row),
            })
        return out

    def occupancy(self, n_bins: int = 600) -> Tuple[List[int], np.ndarray]:
        """Track-presence matrix for the timeline: (object_ids, [n_obj, n_bins]).

        Each bin covers a *span* of frames and is lit if the track appears
        anywhere in it. Binning by nearest frame instead would punch holes into
        continuous tracks whenever the timeline is wider than the video is
        long — gaps the user would read as the animal being lost.
        """
        ids = self.object_ids()
        n_bins = max(1, int(n_bins))
        grid = np.zeros((len(ids), n_bins), dtype=bool)
        if not ids or self.frame.size == 0:
            return ids, grid

        n_frames = max(int(self.frame.max()) + 1, self.frame_count, 1)
        b = np.arange(n_bins, dtype=np.int64)
        lo = b * n_frames // n_bins
        hi = np.maximum((b + 1) * n_frames // n_bins, lo + 1)

        for i, oid in enumerate(ids):
            present = np.zeros(n_frames + 1, dtype=np.int32)
            present[self.frame[self.object_id == oid]] = 1
            cum = np.concatenate(([0], np.cumsum(present)))
            grid[i] = (cum[np.minimum(hi, n_frames)] - cum[lo]) > 0
        return ids, grid

    # -- edits ----------------------------------------------------------
    def write_object_ids(self, new_object_ids: np.ndarray):
        """Commit corrected identities, handling the open read handle.

        HDF5 refuses to open a file for writing while this reader holds it
        open read-only, so the handle is dropped for the write and restored
        afterwards. Going through the reader keeps that dance in one place
        rather than leaving every caller to remember it.
        """
        new_object_ids = np.asarray(new_object_ids, dtype=np.int32)
        self.close()
        try:
            apply_identity_edits(self.path, new_object_ids)
        finally:
            self.reopen()
        self.object_id = new_object_ids.copy()

    def reopen(self):
        """Re-acquire the read handle after it was dropped for a write."""
        if self._f is None:
            self._f = h5py.File(self.path, "r")

    def close(self):
        if self._f is not None:
            self._f.close()
            self._f = None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


def trajectories_dataframe(
    reader: "TrackMaskReader",
    object_ids: Optional[np.ndarray] = None,
    categories: Optional[Dict] = None,
):
    """Rebuild the trajectories table from a track store.

    Column-for-column identical to what inference writes, so a proofread CSV
    is a drop-in replacement for the original. ``object_ids`` overrides the
    stored identities, letting the proofreader preview corrections before
    committing them. Ids <= 0 mark deleted detections and are dropped.
    """
    import pandas as pd

    oid = reader.object_id if object_ids is None else np.asarray(object_ids)
    keep = oid > 0
    if not keep.any():
        return pd.DataFrame()

    df = pd.DataFrame({
        "frame": reader.frame[keep].astype(int),
        "time_s": np.round(reader.frame[keep] / max(reader.fps, 1e-6), 4),
        "object_id": oid[keep].astype(int),
        "label": reader.label[keep].astype(int),
        "x": np.round(reader.centroid[keep, 0], 2),
        "y": np.round(reader.centroid[keep, 1], 2),
        "bbox_x1": np.round(reader.bbox[keep, 0], 2),
        "bbox_y1": np.round(reader.bbox[keep, 1], 2),
        "bbox_x2": np.round(reader.bbox[keep, 2], 2),
        "bbox_y2": np.round(reader.bbox[keep, 3], 2),
        "mask_area": reader.area[keep].astype(int),
        "confidence": np.round(reader.score[keep], 4),
    })
    for name, values in reader.shape_features.items():
        df[name] = values[keep]
    df = df.sort_values(["object_id", "frame"]).reset_index(drop=True)

    if categories:
        cat_map = {int(k): v for k, v in categories.items()}
        counters: Dict[str, int] = {}
        names: Dict[int, str] = {}
        for o in sorted(df["object_id"].unique()):
            lab = int(df.loc[df["object_id"] == o, "label"].iloc[0])
            cname = cat_map.get(lab, f"class{lab}")
            counters[cname] = counters.get(cname, 0) + 1
            names[o] = f"{cname}_{counters[cname]}"
        df["object_name"] = df["object_id"].map(names)

    for o in df["object_id"].unique():
        m = df["object_id"] == o
        dt = df.loc[m, "time_s"].diff().replace(0, np.nan)
        dx = df.loc[m, "x"].diff()
        dy = df.loc[m, "y"].diff()
        df.loc[m, "vx"] = (dx / dt).round(2)
        df.loc[m, "vy"] = (dy / dt).round(2)
        df.loc[m, "speed"] = (np.sqrt(dx**2 + dy**2) / dt).round(2)

    order = ["frame", "time_s", "object_id", "object_name", "label",
             "x", "y", "bbox_x1", "bbox_y1", "bbox_x2", "bbox_y2",
             "mask_area", *SHAPE_FIELDS,
             "confidence", "vx", "vy", "speed"]
    return df[[c for c in order if c in df.columns]]


def apply_identity_edits(path: str, new_object_ids: np.ndarray):
    """Persist a corrected /object_id column, leaving mask pixels untouched."""
    _require_h5()
    with h5py.File(path, "a") as f:
        ds = f["object_id"]
        if ds.shape[0] != new_object_ids.shape[0]:
            raise ValueError(
                f"object_id length mismatch: store has {ds.shape[0]}, "
                f"got {new_object_ids.shape[0]}"
            )
        ds[:] = np.asarray(new_object_ids, dtype=np.int32)
        f.attrs["proofread"] = True
        f.attrs["proofread_at"] = datetime.now().isoformat(timespec="seconds")
