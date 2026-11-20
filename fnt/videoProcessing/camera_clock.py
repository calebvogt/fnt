"""Read the clock an NVR burns into its footage, to the frame.

Why this exists
---------------
The Camera Grid places every camera on one wall clock from the file names and
durations (camera_timeline.py). That lands each camera within a second or two,
but the error wanders over a trial: on VC-T005 the five burnt-in clocks
disagreed by 0-2 s, and which camera led changed from day to day. A single
offset per camera cannot fix that, so the grid can instead measure every
camera's own clock every minute and follow it.

How a clock is read
-------------------
The overlay is one fixed bitmap font in one fixed place, ``YYYY/MM/DD
HH:MM:SS``, white with a dark outline. So it is read by template matching, not
general OCR (EasyOCR misread it badly):

* The templates are LEARNED from the footage itself. The file-based timeline
  is right to a few seconds, so at a timeline time whose seconds are well away
  from a minute boundary the date, hour and minute on screen are known. Over a
  day the hour digits run through 0-9, which is every template needed.
  Learning uses only frames whose characters separate cleanly (a dark
  background behind the text); reading then works on any background because
  the templates carry the dark outline.
* Reading walks the string left to right. Every slot of the format is known
  to be a digit or a separator, so only the digits are classified, and each
  digit's own width (a 1 is narrower) moves the cursor on.

How a clock is read to the frame
--------------------------------
The clock shows whole seconds, but the frame on which the seconds digit
changes marks the true second boundary. ``find_tick`` decodes a little over a
second of footage and returns the file time of that frame and the second it
starts, which pins the camera's clock to about one frame (33 ms at 30 fps).

Kept free of Qt; ffmpeg does the decoding.
"""

import datetime
import json
import os
import subprocess

import numpy as np

FORMAT = "DDDD/DD/DD DD:DD:DD"
DIGIT_SLOTS = [i for i, ch in enumerate(FORMAT) if ch == "D"]
#: Slots whose characters are drawn (the space between date and time is not).
DRAWN_SLOTS = [i for i, ch in enumerate(FORMAT) if ch != " "]
#: The date, hour and minute: what the file timeline can label for learning.
PREFIX_LEN = 16     # "YYYY/MM/DD HH:MM"

#: Search area for the clock on a 1920x1080 frame: the bottom-right corner.
#: (x, y, w, h); scaled for other frame sizes.
DEFAULT_SEARCH = (1190, 993, 730, 87)
_WHITE = 190        # grey level above which a pixel counts as clock text
_PAD = 2            # template margin around a glyph, so it carries the outline


def _no_window():
    if os.name == "nt":
        return {"creationflags": getattr(subprocess, "CREATE_NO_WINDOW", 0x08000000)}
    return {}


def grab_gray(path, seek, n_frames, box, fps=None):
    """`n_frames` greyscale crops of `box` (x, y, w, h) from `seek` seconds.

    Input seeking is frame-accurate when transcoding, so frame k is at
    ``seek + k / fps``. Returns a list (possibly shorter near the file's end).
    """
    x, y, w, h = box
    cmd = ["ffmpeg", "-v", "error", "-nostdin", "-ss", f"{max(0.0, seek):.3f}", "-i", path,
           "-an", "-frames:v", str(int(n_frames)),
           "-vf", f"crop={w}:{h}:{x}:{y},format=gray",
           "-f", "rawvideo", "-pix_fmt", "gray", "-"]
    try:
        r = subprocess.run(cmd, capture_output=True, timeout=300, **_no_window())
    except (OSError, subprocess.SubprocessError):
        return []
    buf = np.frombuffer(r.stdout, np.uint8)
    n = buf.size // (w * h)
    return [buf[i * w * h:(i + 1) * w * h].reshape(h, w) for i in range(n)]


def _boxes(gray):
    """Character boxes in a clock crop, left to right, colons' dots merged.

    Returns [(x, y, w, h)] or [] when the text does not separate cleanly into
    the 18 drawn characters (a bright background bleeding into the text).
    """
    import cv2
    m = (gray > _WHITE).astype(np.uint8)
    n, _, st, _ = cv2.connectedComponentsWithStats(m, connectivity=8)
    comps = [tuple(int(v) for v in s[:4]) for s in st[1:] if s[4] >= 8]
    if not comps:
        return []
    # the text band: components as tall as the tallest digit
    tall = [c for c in comps if c[3] >= 0.7 * max(c[3] for c in comps)]
    top = int(np.median([c[1] for c in tall]))
    bottom = int(np.median([c[1] + c[3] for c in tall]))
    comps = sorted((c for c in comps if c[1] >= top - 4 and c[1] + c[3] <= bottom + 4),
                   key=lambda c: c[0])
    merged = []
    for c in comps:     # a colon is two dots stacked at the same x
        if merged and abs(c[0] - merged[-1][0]) <= 2 and c[3] < 0.5 * (bottom - top):
            x0 = min(c[0], merged[-1][0])
            y0 = min(c[1], merged[-1][1])
            x1 = max(c[0] + c[2], merged[-1][0] + merged[-1][2])
            y1 = max(c[1] + c[3], merged[-1][1] + merged[-1][3])
            merged[-1] = (x0, y0, x1 - x0, y1 - y0)
        else:
            merged.append(c)
    if len(merged) != len(DRAWN_SLOTS):
        return []
    if any(b[3] > 1.4 * (bottom - top) for b in merged):
        return []
    return merged


def _ncc(a, b):
    a = a.astype(np.float32) - a.mean()
    b = b.astype(np.float32) - b.mean()
    d = np.sqrt((a * a).sum() * (b * b).sum())
    return float((a * b).sum() / d) if d > 0 else 0.0


class ClockReader:
    """Reads ``YYYY/MM/DD HH:MM:SS`` from a greyscale crop of `box`."""

    def __init__(self, box, top, height, x0, templates, widths, gaps):
        self.box = tuple(int(v) for v in box)      # (x, y, w, h) in the frame
        self.top = int(top)                         # text band inside the crop
        self.height = int(height)
        self.x0 = int(x0)                           # first character's x in the crop
        self.templates = {k: np.asarray(v, np.float32) for k, v in templates.items()}
        self.widths = {k: int(v) for k, v in widths.items()}
        self.gaps = [int(g) for g in gaps]          # per drawn slot: digit: next x - (x + width); separator: next x - x

    # -- learning ------------------------------------------------------------

    @classmethod
    def learn(cls, samples, search=DEFAULT_SEARCH):
        """Learn the font from [(grey crop of `search`, known prefix)].

        The prefix is ``YYYY/MM/DD HH:MM`` (from the file timeline, at a time
        well inside a minute). Samples whose text does not separate cleanly
        are skipped. Raises ValueError if any digit 0-9 was never seen.
        """
        glyphs, widths, gaps, tops, bottoms, x0s = {}, {}, {}, [], [], []
        for gray, prefix in samples:
            bx = _boxes(gray)
            if not bx:
                continue
            labels = [prefix[s] if s < PREFIX_LEN else None for s in DRAWN_SLOTS]
            tops.append(min(b[1] for b in bx))
            bottoms.append(max(b[1] + b[3] for b in bx))
            x0s.append(bx[0][0])
            for i, (b, lab) in enumerate(zip(bx, labels)):
                if i + 1 < len(bx):
                    # after a digit: the space past its own width (a 1 is
                    # narrower); after a separator: the whole step
                    step = bx[i + 1][0] - b[0]
                    is_digit = FORMAT[DRAWN_SLOTS[i]] == "D"
                    gaps.setdefault(i, []).append(step - b[2] if is_digit else step)
                if lab is None or not lab.isdigit():
                    continue
                widths.setdefault(lab, []).append(b[2])
                glyphs.setdefault(lab, []).append((gray, b))
        missing = [d for d in "0123456789" if d not in glyphs]
        if missing:
            raise ValueError(f"clock digits never seen while learning: {missing}")
        top, bottom = int(np.median(tops)), int(np.median(bottoms))
        h = bottom - top
        tw = {d: int(np.median(widths[d])) for d in glyphs}
        templates = {}
        for d, items in glyphs.items():
            stack = []
            for gray, b in items:
                if b[2] != tw[d]:
                    continue
                y0, x0 = top - _PAD, b[0] - _PAD
                patch = gray[y0:y0 + h + 2 * _PAD, x0:x0 + tw[d] + 2 * _PAD]
                if patch.shape == (h + 2 * _PAD, tw[d] + 2 * _PAD):
                    stack.append(patch.astype(np.float32))
            templates[d] = np.median(np.stack(stack), axis=0)
        gap_list = [int(np.median(gaps.get(i, [0]))) for i in range(len(DRAWN_SLOTS))]
        # crop tightly around the text for reading
        x_lo = int(np.median(x0s)) - 8
        span = sum(max(tw.values()) for _ in DIGIT_SLOTS) + sum(gap_list) + 60
        sx, sy, sw, sh = search
        box = (sx + x_lo, sy + top - 6, min(span, sw - x_lo), h + 12)
        return cls(box, 6, h, 8, templates, tw, gap_list)

    # -- reading -------------------------------------------------------------

    def read(self, gray, search=3):
        """(datetime, score) from a crop of ``self.box``, or (None, score).

        `score` is the worst digit's match (normalised correlation, 1 =
        perfect); a reading whose digits do not form a valid date is None.
        """
        x = self.x0
        out, worst = [], 1.0
        h = self.height + 2 * _PAD
        y0 = self.top - _PAD
        widths = self.widths
        for i, slot in enumerate(DRAWN_SLOTS):
            ch = FORMAT[slot]
            if ch == "D":
                best, best_d, best_dx = -2.0, None, 0
                for d, tpl in self.templates.items():
                    tw = tpl.shape[1]
                    for dx in range(-search, search + 1):
                        xs = x + dx - _PAD
                        if xs < 0 or xs + tw > gray.shape[1]:
                            continue
                        s = _ncc(gray[y0:y0 + h, xs:xs + tw], tpl)
                        if s > best:
                            best, best_d, best_dx = s, d, dx
                if best_d is None:
                    return None, -1.0
                out.append(best_d)
                worst = min(worst, best)
                x = x + best_dx + widths[best_d] + self.gaps[i]
            else:
                out.append(ch)
                x = x + self.gaps[i]
            if slot + 1 < len(FORMAT) and FORMAT[slot + 1] == " ":
                out.append(" ")
        text = "".join(out)
        try:
            return datetime.datetime.strptime(text, "%Y/%m/%d %H:%M:%S"), worst
        except ValueError:
            return None, worst

    # -- persistence ---------------------------------------------------------

    def to_dict(self):
        return {"box": self.box, "top": self.top, "height": self.height, "x0": self.x0,
                "templates": {k: v.round(1).tolist() for k, v in self.templates.items()},
                "widths": self.widths, "gaps": self.gaps}

    @classmethod
    def from_dict(cls, d):
        return cls(d["box"], d["top"], d["height"], d["x0"], d["templates"],
                   d["widths"], d["gaps"])

    def save(self, path):
        with open(path, "w", encoding="utf-8") as fh:
            json.dump(self.to_dict(), fh)

    @classmethod
    def load(cls, path):
        with open(path, encoding="utf-8") as fh:
            return cls.from_dict(json.load(fh))


def learn_from_tracks(tracks, n_per_camera=48, search=DEFAULT_SEARCH):
    """Learn the clock font from a trial's cameras via their file timelines.

    Samples each camera at `n_per_camera` times spread over its footage, each
    at :30 seconds on the timeline so the date, hour and minute shown are
    known even when the camera's clock is several seconds off.
    """
    samples = []
    for t in tracks:
        if not t.segments:
            continue
        span = (t.end - t.start).total_seconds()
        for i in range(n_per_camera):
            when = t.start + datetime.timedelta(seconds=span * (i + 0.5) / n_per_camera)
            when = when.replace(second=30, microsecond=0)
            path, off = t.locate(when)
            if path is None:
                continue
            fr = grab_gray(path, off, 1, search)
            if fr:
                samples.append((fr[0], when.strftime("%Y/%m/%d %H:%M")))
    return ClockReader.learn(samples, search)


def find_tick(reader, path, seek, fps=30.0, seconds=1.2, min_score=0.6):
    """Where this file's clock next changes second, from `seek` on.

    Returns ``(file_time, clock_time, score)``: `file_time` is the time in the
    file (seconds) of the first frame showing the new second, `clock_time`
    that second as a datetime. None when the clock could not be read or did
    not advance by exactly one second.

    Only frames whose clock pixels changed are read in full, so a probe costs
    about two readings rather than one per frame.
    """
    frames = grab_gray(path, seek, int(round(seconds * fps)) + 1, reader.box)
    if len(frames) < 2:
        return None
    prev_t, prev_s = reader.read(frames[0])
    if prev_t is None or prev_s < min_score:
        return None
    prev_mask = frames[0] > _WHITE
    for k in range(1, len(frames)):
        mask = frames[k] > _WHITE
        changed = np.count_nonzero(mask != prev_mask)
        prev_mask = mask
        if changed < 20:
            continue
        t, s = reader.read(frames[k])
        if t is None or s < min_score:
            return None
        if t == prev_t:
            continue
        if (t - prev_t).total_seconds() == 1:
            return seek + k / fps, t, min(s, prev_s)
        return None
    return None


def anchor_overfull_days(reader, track, slack=60.0, fps=30.0):
    """Place the files of an over-long day bucket by their burnt-in clocks.

    The file timeline chains a day's files from its stamp. On the day
    daylight saving ends the NVR records 25 hours into that day (the clock
    repeats an hour), so the chain runs an hour into the next day and its last
    hour lies on top of the next day's first (VoleCosm VC-T014, 2026-11-01).
    Each file of such a bucket is moved to where its clock says it starts.
    Returns ``[(path, old_start, new_start)]`` for the files moved.

    That places the next day right, but the over-full day itself would have its
    repeated hour's files on top of each other; the grid CLI encodes that day
    on the continuous axis of `continuous_track` instead.
    """
    from .camera_timeline import _expected_bucket_seconds
    by_day = {}
    for s in track.segments:
        by_day.setdefault(s.day, []).append(s)
    days = sorted(by_day)
    moved = []
    for i, day in enumerate(days):
        segs = by_day[day]
        expected = _expected_bucket_seconds(day, days[i + 1] if i + 1 < len(days) else None)
        if sum(s.duration for s in segs) <= expected + slack:
            continue
        for s in segs:
            hit = None
            for seek in (2.0, 30.0, 120.0, max(s.duration - 60.0, 0.0)):
                if seek < s.duration:
                    hit = find_tick(reader, s.path, seek, fps=fps)
                if hit:
                    break
            if not hit:
                continue
            file_t, clock, _ = hit
            new = clock - datetime.timedelta(seconds=file_t)
            if abs((new - s.start).total_seconds()) > 2.0:
                moved.append((s.path, s.start, new))
                s.start = new
                s.end = new + datetime.timedelta(seconds=s.duration)
    if moved:
        track.segments.sort(key=lambda s: s.start)
    return moved


#: How far an NVR's clock falls back when daylight saving ends.
FALLBACK_S = 3600.0


def fallback_days(track, slack=600.0):
    """The dates whose day bucket holds about 25 h of footage: daylight saving ended that night and the NVR recorded
    the repeated hour into the same day (VoleCosm VC-T014, 2026-11-01). Judged from the file durations, on the
    timeline as the file names built it (before `anchor_overfull_days`). A trial's last bucket is never one: it ends
    whenever recording stopped."""
    from .camera_timeline import _expected_bucket_seconds
    by_day = {}
    for s in track.segments:
        by_day.setdefault(s.day, []).append(s)
    days = sorted(by_day)
    out = set()
    for i, day in enumerate(days[:-1]):
        over = sum(s.duration for s in by_day[day]) - _expected_bucket_seconds(day, days[i + 1])
        if abs(over - FALLBACK_S) <= slack:
            out.add(datetime.datetime.strptime(day, "%Y%m%d%H%M%S").date())
    return out


def continuous_track(track, day):
    """A copy of `track` on one continuous time axis through `day`, the day daylight saving ends.

    `track` must be as the file names built it (not yet anchored by clocks): `day`'s files are then chained from its
    stamp in the order recorded, so they run 25 h, to 01:00 the next day, with both 01:xx hours in order. Every later
    day's files (and gaps) move an hour later to follow on. On this axis the day is [00:00, 25:00) and, from its
    fallback on, a camera's burnt-in clock reads an hour behind the axis (see `unwrap_fallback`). The segments are
    new objects, so anchoring the original track later does not move them.
    """
    from .camera_timeline import CameraTrack, Gap, Segment
    stamp = f"{day:%Y%m%d}"
    hour = datetime.timedelta(seconds=FALLBACK_S)
    segs = []
    for s in track.segments:
        c = Segment(s.path, s.day, s.seq, s.duration)
        later = s.day[:8] > stamp
        c.start = s.start + hour if later else s.start
        c.end = s.end + hour if later else s.end
        segs.append(c)
    later = [s.start for s in track.segments if s.day[:8] > stamp]
    first_later = min(later) if later else None
    gaps = [Gap(g.start + (hour if first_later is not None and g.start >= first_later else datetime.timedelta(0)),
                g.duration, seq=g.seq, unlocated=g.unlocated) for g in track.gaps]
    clone = CameraTrack(track.name, track.folder, sorted(segs, key=lambda s: s.start), gaps)
    clone.clock_offset = getattr(track, "clock_offset", 0.0)
    return clone


def unwrap_fallback(probes, tol=60.0, min_readings=3):
    """Clock readings put on the continuous axis of the day daylight saving ends (see `continuous_track`).

    There a camera's clock reads about 3600 s behind the axis from its fallback on (inside a file or at a file
    change). Per camera: when its readings sit on two levels 3600 +- `tol` s apart, the higher one first, every
    reading on the lower level from the first one on gets 3600 s added to its clock and its error. A lone reading an
    hour off (a misread hour digit) is not a fallback: each level needs `min_readings`.

    Returns ``(readings, {camera: number shifted})``; readings of other days come back as they were.
    """
    hour = datetime.timedelta(seconds=FALLBACK_S)
    out = list(probes)
    by_cam = {}
    for i, p in enumerate(probes):
        if p and p.get("delta") is not None:
            by_cam.setdefault(p.get("camera"), []).append(i)
    shifted = {}
    for cam, idx in by_cam.items():
        idx = sorted(idx, key=lambda i: probes[i]["timeline"])
        d = np.array([probes[i]["delta"] for i in idx])
        if d.max() - d.min() < FALLBACK_S - tol:
            continue
        mid = (d.max() + d.min()) / 2
        hi, lo = d > mid, d <= mid
        if hi.sum() < min_readings or lo.sum() < min_readings:
            continue
        if abs(np.median(d[hi]) - np.median(d[lo]) - FALLBACK_S) > tol:
            continue
        first = int(np.argmax(lo))
        if hi[first:].sum() > 0.1 * hi.sum():      # not a step down: the levels are mixed in time
            continue
        n = 0
        for k in range(first, len(idx)):
            if lo[k] and abs(d[k] + FALLBACK_S - np.median(d[hi])) <= tol:
                p = dict(probes[idx[k]])
                p["delta"] = p["delta"] + FALLBACK_S
                if p.get("clock") is not None:
                    p["clock"] = p["clock"] + hour
                out[idx[k]] = p
                n += 1
        if n:
            shifted[cam] = n
    return out, shifted


def grid_error(clock, start, grid_t, fallback=False, misread=10.0):
    """A cell's clock error in a finished grid: the second its clock ticks to, from the chunk's `start`, minus the
    grid time of the frame it ticks on; None for a misread (`misread` s or more off). On the day daylight saving ends
    (`fallback`) the grid runs on the continuous axis, so a clock reading an hour behind it is that day's second
    01:xx hour and is moved back onto the axis."""
    err = (clock - start).total_seconds() - grid_t
    if fallback and abs(err + FALLBACK_S) < misread:
        err += FALLBACK_S
    return err if abs(err) < misread else None


def segment_at(track, when):
    """The segment holding wall-clock `when` (no clock offset applied)."""
    lo, hi = 0, len(track.segments) - 1
    while lo <= hi:
        mid = (lo + hi) // 2
        s = track.segments[mid]
        if when < s.start:
            hi = mid - 1
        elif when >= s.end:
            lo = mid + 1
        else:
            return s
    return None


def probe(reader, track, when, fps=30.0):
    """Measure one camera's clock against its file timeline at `when`.

    Returns a dict with ``delta`` = burnt-in clock minus file timeline, in
    seconds, accurate to about a frame, or None where there is no footage or
    the clock could not be read.
    """
    seg = segment_at(track, when)
    if seg is None:
        return None
    off = (when - seg.start).total_seconds()
    hit = find_tick(reader, seg.path, off, fps=fps)
    if hit is None:
        return {"camera": track.name, "when": when, "path": seg.path,
                "file_time": off, "delta": None, "score": None}
    file_time, clock, score = hit
    timeline = seg.start + datetime.timedelta(seconds=file_time)
    return {"camera": track.name, "when": when, "path": seg.path,
            "file_time": file_time, "timeline": timeline, "clock": clock,
            "delta": (clock - timeline).total_seconds(), "score": score}


def _probe_job(args):
    reader_dict, track, when, fps = args
    return probe(ClockReader.from_dict(reader_dict), track, when, fps)


def measure(reader, tracks, start, end, step=60.0, fps=30.0, workers=8, progress=None):
    """Probe every camera every `step` seconds from `start` to `end`.

    Returns a list of probe dicts (see `probe`), in time order. Runs the
    probes in `workers` processes; `progress(done, total)` is called as they
    finish.
    """
    from concurrent.futures import ProcessPoolExecutor
    times = []
    t = start
    while t < end:
        times.append(t)
        t += datetime.timedelta(seconds=step)
    rd = reader.to_dict()
    jobs = [(rd, tr, w, fps) for w in times for tr in tracks]
    out = []
    with ProcessPoolExecutor(max_workers=workers) as ex:
        for i, r in enumerate(ex.map(_probe_job, jobs, chunksize=4)):
            if r is not None:
                out.append(r)
            if progress:
                progress(i + 1, len(jobs))
    return out


class ClockSync:
    """One camera's burnt-in clock minus its file timeline, over time.

    Stored as pieces ``(t0, t1, a, b)`` in file-timeline time: inside
    ``[t0, t1)`` the clock reads ``a + b * (T - t0)`` seconds ahead of the
    timeline. Fitted one piece per segment file, because the error is set
    where a file starts (the timeline sums file durations) and then drifts
    slowly inside it as the recorder's frame clock and wall clock part.
    """

    def __init__(self, pieces):
        self.pieces = sorted(pieces, key=lambda p: p[0])

    def delta(self, when):
        """Seconds the clock reads ahead of the timeline at `when`."""
        if not self.pieces:
            return 0.0
        for t0, t1, a, b in self.pieces:
            if t0 <= when < t1:
                return a + b * (when - t0).total_seconds()
        # outside every piece: hold the nearest piece's end value
        first, last = self.pieces[0], self.pieces[-1]
        if when < first[0]:
            return first[2]
        t0, t1, a, b = min(self.pieces, key=lambda p: abs((when - p[1]).total_seconds()))
        return a + b * (t1 - t0).total_seconds()

    def span(self, t_start, t_end):
        """The pieces overlapping [t_start, t_end), clipped to it, nearest
        pieces carried across stretches with no piece."""
        out = []
        for t0, t1, a, b in self.pieces:
            if t1 <= t_start or t0 >= t_end:
                continue
            c0 = max(t0, t_start)
            out.append((c0, min(t1, t_end), a + b * (c0 - t0).total_seconds(), b))
        if not out:
            d = self.delta(t_start)
            out = [(t_start, t_end, d, 0.0)]
        return out

    @classmethod
    def fit(cls, probes, segments, max_resid=0.07):
        """Fit from `probe` results for one camera and its segments.

        Per segment: a straight line through its readings, refitted without
        readings more than `max_resid` s off it (a misread or a stalled
        frame); a constant when it has one or two readings. Segments without
        a reading take their neighbours' values.
        """
        by_path = {}
        for p in probes:
            if p and p.get("delta") is not None:
                by_path.setdefault(p["path"], []).append(p)
        pieces, missing = [], []
        for seg in segments:
            ps = by_path.get(seg.path, [])
            if not ps:
                missing.append(seg)
                continue
            x = np.array([(p["timeline"] - seg.start).total_seconds() for p in ps])
            y = np.array([p["delta"] for p in ps])
            a, b = _robust_line(x, y, max_resid)
            pieces.append((seg.start, seg.end, a, b))
        sync = cls(pieces)
        for seg in missing:
            sync.pieces.append((seg.start, seg.end, sync.delta(seg.start), 0.0))
        sync.pieces.sort(key=lambda p: p[0])
        return sync

    def to_rows(self, camera):
        return [{"camera": camera, "t0": t0.isoformat(), "t1": t1.isoformat(),
                 "a": round(a, 4), "b": b} for t0, t1, a, b in self.pieces]


def _robust_line(x, y, max_resid):
    if len(x) < 3:
        return float(np.median(y)), 0.0
    keep = np.ones(len(x), bool)
    for _ in range(3):
        b, a = np.polyfit(x[keep], y[keep], 1)
        r = np.abs(y - (a + b * x))
        new = r <= max(max_resid, 0.0)
        if new.sum() < 3:
            # readings disagree too much for a line: fall back to the median
            return float(np.median(y)), 0.0
        if (new == keep).all():
            break
        keep = new
    return float(a), float(b)


def _cell_clock(frame, cell, reader, src_size=(1920, 1080)):
    """The clock crop of one grid cell, scaled back up to the reader's size."""
    import cv2
    cx, cy, cw, ch = cell
    sx, sy = cw / src_size[0], ch / src_size[1]
    bx, by, bw, bh = reader.box
    x0, y0 = int(round(cx + bx * sx)), int(round(cy + by * sy))
    x1, y1 = int(round(cx + (bx + bw) * sx)), int(round(cy + (by + bh) * sy))
    crop = frame[y0:y1, x0:x1]
    if crop.size == 0:
        return None
    return cv2.resize(crop, (bw, bh), interpolation=cv2.INTER_CUBIC)


def check_grid(video, cells, reader, when, fps=30.0, seconds=1.2,
               src_size=(1920, 1080), min_score=0.3):
    """Read every cell's burnt-in clock in a finished grid at `when` seconds.

    `cells` maps camera -> (x, y, w, h) in the grid. For each cell, returns
    the clock's error against the grid's own time to the frame: the second
    the cell's clock ticks to, minus the grid time of the frame it ticks on.
    Only differences between cells matter (the grid's time origin is the
    chunk start). None for a cell whose clock could not be followed.

    The match threshold is lower than for the cameras' own footage: a clock
    shrunk into a cell over a bright background (the mesocosm's centre
    camera) matches at about 0.4. A reading still has to advance by exactly
    one second between neighbouring frames to count.
    """
    probe = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0", "-show_entries",
         "stream=width,height", "-of", "csv=p=0", video],
        capture_output=True, text=True, **_no_window())
    W, H = (int(v) for v in probe.stdout.strip().split(",")[:2])
    frames = grab_gray(video, when, int(round(seconds * fps)) + 1, (0, 0, W, H))
    out = {}
    for cam, cell in cells.items():
        out[cam] = None
        crops = [_cell_clock(fr, cell, reader, src_size) for fr in frames]
        if not crops or crops[0] is None:
            continue
        prev_t, prev_s = reader.read(crops[0])
        if prev_t is None or prev_s < min_score:
            continue
        prev_mask = crops[0] > _WHITE
        for k in range(1, len(crops)):
            mask = crops[k] > _WHITE
            changed = np.count_nonzero(mask != prev_mask)
            prev_mask = mask
            if changed < 20:
                continue
            t, sc = reader.read(crops[k])
            if t is None or sc < min_score or t == prev_t:
                continue
            if (t - prev_t).total_seconds() == 1:
                out[cam] = (t, when + k / fps)
            break
    return out


def _check_job(args):
    video, cells, reader_dict, when, fps = args
    return when, check_grid(video, cells, ClockReader.from_dict(reader_dict), when, fps)


def check_grid_many(video, cells, reader, times, fps=30.0, workers=8):
    """`check_grid` at each of `times`, in `workers` processes; yields
    (when, result) in order."""
    from concurrent.futures import ProcessPoolExecutor
    rd = reader.to_dict()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        yield from ex.map(_check_job, [(video, cells, rd, float(t), fps) for t in times])
