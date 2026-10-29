"""Which stretches of a fixed camera's video have something moving, read from the
compressed size of every frame -- without decoding a frame.

A video encoder spends almost nothing on a frame that matches the one before it, so
in a still, empty scene the inter frames are tiny (~150 B on the 2021_8x8 resource-zone
cameras) and a moving animal multiplies that (400-800+ B). MP4/MOV files list every
frame's size in their index (moov/stbl: stsz sizes, stss keyframes, stts/ctts timing),
which reads in ~20 ms per 30-minute clip over the network, against minutes to decode it.

Per second of video, its *level* is the median size of its non-key frames, and its
*baseline* is a low quantile of the levels around it on the same camera: the clips of
one camera (same folder, same name prefix, e.g. ``CAM13``) chained in name (= time)
order, so the baseline follows lighting changes. ``ratio = level / baseline``.

Measured on the 2021_8x8 zone cameras (17 trials, 2026-10-28), against the mouse model:
ratio > 2 marks ~21% of the video and 94% of those seconds have a mouse in view; 45% of
the remaining (still) seconds hold a resting mouse -- no motion, so only a look at the
frame finds it. Hence :func:`plan_frames`: every frame of busy seconds (ratio > ``hi``,
padded), ``low_fps`` frames a second where the ratio is uncertain (``lo`` < ratio <=
``hi``), and only the keyframes elsewhere (one decode each to reach, ~every 17 s there).
"""
from __future__ import annotations

import os
import struct
from dataclasses import dataclass
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np

#: Tiers of a planned frame, as stored with each sampled frame.
TIER_KEY, TIER_LOW, TIER_BUSY = 0, 1, 2

# pad_s 1: busy seconds are 21% of the zone video, 25% padded by 1 s, 28% by 2 s.
DEFAULT_GATE = {"hi": 2.0, "lo": 1.3, "pad_s": 1.0, "low_fps": 2.0, "busy_fps": 0.0,
                "baseline_s": 7200.0, "quantile": 0.10}


@dataclass
class FrameSizes:
    sizes: np.ndarray        # bytes per frame, display order
    keys: np.ndarray         # display indices of the keyframes
    fps: float


# ------------------------------------------------------------------ MP4 index
def _boxes(f, start: int, end: int):
    pos = start
    while pos + 8 <= end:
        f.seek(pos)
        size, kind = struct.unpack(">I4s", f.read(8))
        hdr = 8
        if size == 1:
            size = struct.unpack(">Q", f.read(8))[0]
            hdr = 16
        elif size == 0:
            size = end - pos
        if size < hdr:
            return
        yield kind, pos + hdr, pos + size
        pos += size


def _find(f, start: int, end: int, path: Sequence[bytes]):
    for kind, a, b in _boxes(f, start, end):
        if kind == path[0]:
            if len(path) == 1:
                yield a, b
            else:
                yield from _find(f, a, b, path[1:])


def _table(f, box, fields: int, signed: bool = False) -> np.ndarray:
    """A full box's entry table: (n, fields) of 32-bit values after version/flags + count."""
    a, _ = box
    f.seek(a)
    version = f.read(1)[0]
    f.seek(a + 4)
    n = struct.unpack(">I", f.read(4))[0]
    dt = ">i4" if (signed and version == 1) else ">u4"
    return np.frombuffer(f.read(4 * fields * n), dt).reshape(n, fields).astype(np.int64)


def mp4_frame_sizes(path: str) -> Optional[FrameSizes]:
    """The video track's frame sizes and keyframes in display order, or None when the file
    has no readable MP4 index (another container, or a recording cut off before its index
    was written)."""
    try:
        with open(path, "rb") as f:
            f.seek(0, 2)
            end = f.tell()
            moov = next(_find(f, 0, end, [b"moov"]), None)
            if moov is None:
                return None
            for ta, tb in _find(f, *moov, [b"trak"]):
                hd = next(_find(f, ta, tb, [b"mdia", b"hdlr"]), None)
                if hd is None:
                    continue
                f.seek(hd[0] + 8)
                if f.read(4) != b"vide":
                    continue
                mdhd = next(_find(f, ta, tb, [b"mdia", b"mdhd"]))
                f.seek(mdhd[0])
                version = f.read(1)[0]
                f.seek(mdhd[0] + (20 if version == 1 else 12))
                timescale = struct.unpack(">I", f.read(4))[0]
                stbl = next(_find(f, ta, tb, [b"mdia", b"minf", b"stbl"]))
                boxes = {k: (a, b) for k, a, b in _boxes(f, *stbl)}
                a, _ = boxes[b"stsz"]
                f.seek(a + 4)
                fixed, n = struct.unpack(">II", f.read(8))
                sizes = (np.full(n, fixed, np.uint32) if fixed
                         else np.frombuffer(f.read(4 * n), ">u4").astype(np.uint32))
                keys = (_table(f, boxes[b"stss"], 1)[:, 0] - 1 if b"stss" in boxes
                        else np.arange(n))
                stts = _table(f, boxes[b"stts"], 2)
                deltas = np.repeat(stts[:, 1], stts[:, 0])[:n]
                dts = np.concatenate(([0], np.cumsum(deltas)[:-1]))
                pts = dts
                if b"ctts" in boxes:
                    ctts = _table(f, boxes[b"ctts"], 2, signed=True)
                    pts = dts + np.repeat(ctts[:, 1], ctts[:, 0])[:n]
                order = np.argsort(pts, kind="stable")        # decode index of each shown frame
                rank = np.empty(n, np.int64)
                rank[order] = np.arange(n)
                med = float(np.median(deltas)) if len(deltas) else 0.0
                fps = timescale / med if med else 0.0
                return FrameSizes(sizes[order], np.sort(rank[keys]), fps)
    except (OSError, StopIteration, KeyError, struct.error, ValueError):
        return None
    return None


# ------------------------------------------------------------------ activity
def second_levels(fs: FrameSizes, fps: Optional[float] = None) -> np.ndarray:
    """Median size of each second's non-key frames (NaN for a second with none)."""
    per = max(1, int(round(fps or fs.fps or 15.0)))
    s = fs.sizes.astype(float)
    s[fs.keys] = np.nan
    n = len(s) // per
    tail = len(s) - n * per
    out = np.full(n + (1 if tail else 0), np.nan)
    with np.errstate(all="ignore"):
        import warnings
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN seconds
            if n:
                out[:n] = np.nanmedian(s[: n * per].reshape(n, per), axis=1)
            if tail:
                out[n] = np.nanmedian(s[n * per:])
    return out


def camera_key(video: str) -> Tuple[str, str]:
    """Videos of one camera: the same folder and the same name prefix (before the first
    underscore) -- as order_coarse_to_fine groups them."""
    return os.path.dirname(os.path.abspath(video)), os.path.basename(video).split("_")[0]


def activity_ratios(videos: Sequence[str], baseline_s: float = DEFAULT_GATE["baseline_s"],
                    quantile: float = DEFAULT_GATE["quantile"],
                    log: Optional[Callable[[str], None]] = None,
                    whole_camera: bool = True) -> Dict[str, Optional[Tuple[FrameSizes, np.ndarray]]]:
    """video -> (its frame sizes, the ratio of each of its seconds), or None for a video
    without a readable index. The baseline of a second is the ``quantile`` of the levels
    within ``baseline_s`` around it, over the camera's clips chained in name order --
    with ``whole_camera``, all of that camera's clips in the folder, not only those given
    (an hour of busy clips on their own would set a busy baseline)."""
    import pandas as pd

    groups: Dict[Tuple[str, str], List[str]] = {}
    for v in videos:
        groups.setdefault(camera_key(v), []).append(v)
    out: Dict[str, Optional[Tuple[FrameSizes, np.ndarray]]] = {}
    for (folder, cam), given in groups.items():
        vs = set(given)
        if whole_camera:
            ext = {os.path.splitext(v)[1].lower() for v in given}
            vs |= {os.path.join(folder, f) for f in os.listdir(folder)
                   if f.split("_")[0] == cam and os.path.splitext(f)[1].lower() in ext}
        keep = {os.path.normcase(os.path.abspath(v)): v for v in given}
        vs = sorted({os.path.normcase(os.path.abspath(v)): v for v in vs}.items(),
                    key=lambda kv: os.path.basename(kv[1]))
        vs = [keep.get(k, v) for k, v in vs]
        given = set(given)
        parts, spans = [], []
        for v in vs:
            fs = mp4_frame_sizes(v)
            if fs is None:
                if v in given:
                    out[v] = None
                if log and v in given:
                    log(f"[gate] {os.path.basename(v)}: no readable index, every frame will be read")
                continue
            lv = second_levels(fs)
            spans.append((v, fs, sum(len(p) for p in parts), len(lv)))
            parts.append(lv)
        if not parts:
            continue
        levels = pd.Series(np.concatenate(parts))
        win = max(3, int(baseline_s))
        base = levels.rolling(win, center=True, min_periods=min(win, 60)).quantile(quantile)
        base = base.bfill().ffill()
        if base.isna().all():                       # a camera with too few seconds
            base[:] = levels.quantile(quantile)
        ratio = (levels / base).to_numpy()
        for v, fs, a, n in spans:
            if v in given:
                out[v] = (fs, ratio[a:a + n])
    return out


def plan_frames(fs: FrameSizes, ratio: np.ndarray, n_frames: Optional[int] = None,
                fps: Optional[float] = None, hi: float = DEFAULT_GATE["hi"],
                lo: float = DEFAULT_GATE["lo"], pad_s: float = DEFAULT_GATE["pad_s"],
                low_fps: float = DEFAULT_GATE["low_fps"], busy_fps: float = DEFAULT_GATE["busy_fps"],
                **_) -> Tuple[np.ndarray, np.ndarray]:
    """(frame indices, tiers) to look at in one video: every frame of a busy second
    (ratio > ``hi``, widened by ``pad_s`` on each side; ``busy_fps`` frames a second
    instead when set), ``low_fps`` frames a second of an uncertain one (``lo`` < ratio <=
    ``hi``), and the keyframes everywhere else."""
    n = int(n_frames if n_frames is not None else len(fs.sizes))
    per = max(1, int(round(fps or fs.fps or 15.0)))
    r = np.nan_to_num(np.asarray(ratio, float), nan=np.inf)       # unknown counts as busy
    busy = r > hi
    pad = int(np.ceil(pad_s))
    if pad and busy.any():
        idx = np.flatnonzero(busy)
        widened = np.zeros_like(busy)
        for d in range(-pad, pad + 1):
            j = np.clip(idx + d, 0, len(busy) - 1)
            widened[j] = True
        busy = widened
    low = (r > lo) & ~busy
    tier = np.full(n, -1, np.int8)
    sec = np.minimum(np.arange(n) // per, len(r) - 1)
    within = np.arange(n) % per

    def every(rate):
        stride = max(1, int(round(per / rate)))
        return within % stride == stride // 2

    tier[busy[sec] & (every(busy_fps) if busy_fps > 0 else True)] = TIER_BUSY
    if low_fps > 0:
        tier[low[sec] & every(low_fps) & (tier < 0)] = TIER_LOW
    keys = fs.keys[fs.keys < n]
    tier[keys[tier[keys] < 0]] = TIER_KEY
    frames = np.flatnonzero(tier >= 0)
    return frames, tier[frames]
