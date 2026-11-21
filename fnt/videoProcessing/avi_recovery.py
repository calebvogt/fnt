"""Recover a damaged AVI with its real timing.

Security recorders (Viewtron/iDVR) write plain AVI: one chunk per video frame
(``00dc``, I-frames as ``00db``) interleaved with audio (``01wb``) and an
``idx1`` table at the end that lists every chunk in order. When blocks of the
file are lost (zeroed, or overwritten by another file), FFmpeg either cannot
open the file at all (header gone) or encodes it with the lost frames simply
left out, so the video runs ahead of the camera clock after every break.

:func:`map_avi` finds every chunk that survives. The data is walked from every
plausible chunk header; each chain of chunks is placed in time by any member
the ``idx1`` table still lists (exact). A chain the table no longer lists (its
entries were lost too) is placed between the nearest listed chunks: the
chain's own length is exact and the entries left over are shared among the lost
byte ranges in proportion to their size (VT-P001 08-09: 0-3 s off, against the
camera's burned-in clock). A lost file header is filled in from an intact
recording of the same recorder (:func:`find_reference`).

:func:`recover_avi` decodes each run of consecutive frames from its first
keyframe and writes black wherever no frame survives, so the output has the
recording's real length and frame *n* is at time *n / fps*. Each stretch is
reported (recovered or black, and whether its place came from the index or was
estimated) so analyses can leave the black stretches out.

H.264 and HEVC in Annex-B form (what these recorders store) are supported;
OpenDML (``AVIX`` / ``ix00``-only) files are not.
"""
import bisect
import csv
import os
import re
import struct
import subprocess
import threading
from collections import Counter
from dataclasses import dataclass, field

import numpy as np

AVIIF_KEYFRAME = 0x10
_CHUNK_RE = re.compile(rb'\d\d(?:dc|db|wb)')
_START4 = b'\x00\x00\x00\x01'
_START3 = b'\x00\x00\x01'
_TAIL = 64 << 20          # the index is looked for in the file's last 64 MB


@dataclass
class AviMap:
    """What survives of an AVI, frame by frame."""
    frames: dict            # frame number -> (offset of chunk header, payload size, is_keyframe)
    total_frames: int       # frames the recording had (header, reference or index)
    fps: float              # recorded frame rate, or None if unknown
    codec: str              # 'hevc' or 'h264' (FFmpeg raw demuxer name)
    width: int = None
    height: int = None
    estimated: set = field(default_factory=set)   # frame numbers placed by estimate
    info: dict = field(default_factory=dict)


def _u32(d, p):
    return struct.unpack_from('<I', d, p)[0]


def _chunk_ids(kinds=(b'dc', b'db', b'wb')):
    return {struct.unpack('<I', b'0' + str(s).encode() + k)[0]: k for s in range(10) for k in kinds}


_VID = {k for k, v in _chunk_ids().items() if v in (b'dc', b'db')}
_AUD = {k for k, v in _chunk_ids().items() if v == b'wb'}
_ALL = np.array(sorted(_VID | _AUD), dtype='<u4')


# --- bitstream -------------------------------------------------------------

def _nal_types(payload, codec, limit=6):
    """NAL unit types at the start of an Annex-B access unit."""
    out, i, n = [], 0, len(payload)
    while len(out) < limit:
        j = payload.find(_START3, i)
        if j < 0 or j + 3 >= n:
            break
        b = payload[j + 3]
        out.append((b >> 1) & 0x3F if codec == 'hevc' else b & 0x1F)
        i = j + 3
    return out


def _guess_codec(payload):
    """'hevc' or 'h264' from the first NAL of an Annex-B frame, or None."""
    j = payload.find(_START3)
    if j < 0 or j + 4 >= len(payload):
        return None
    b0, b1 = payload[j + 3], payload[j + 4]
    if b0 & 0x80:
        return None
    # HEVC: 2-byte header, layer id 0, temporal id + 1 >= 1
    if (b0 & 1) == 0 and (b1 >> 3) == 0 and (b1 & 0x07) >= 1 and ((b0 >> 1) & 0x3F) <= 40:
        return 'hevc'
    if (b0 & 0x1F) in (1, 5, 6, 7, 8, 9):
        return 'h264'
    return None


_ZEROS = bytes(16)


def _damaged(payload):
    """Coded H.264/HEVC data never holds three zero bytes in a row (emulation
    prevention), so a run of zeros means part of the frame is lost."""
    return payload.find(_ZEROS, 4) >= 0


def _is_key(payload, codec):
    t = _nal_types(payload, codec)
    if codec == 'hevc':
        return any(16 <= x <= 21 for x in t)
    return 5 in t


# --- container ---------------------------------------------------------------

def _parse_header(d):
    """(movi fourcc position, fps, total video frames, width, height, codec) from an
    intact header; each None when it is lost."""
    head = d[:min(len(d), 1 << 20)]
    movi = None
    m = re.search(rb'LIST....movi', head, re.S)
    if m:
        movi = m.start() + 8
    fps = total = w = h = codec = None
    for m in re.finditer(rb'strh', head):
        p = m.start() + 8
        if p + 40 > len(head) or head[p:p + 4] != b'vids':
            continue
        fcc = head[p + 4:p + 8].lower()
        scale, rate, length = _u32(head, p + 20), _u32(head, p + 24), _u32(head, p + 32)
        if scale and rate:
            fps = rate / scale
        total = length or None
        codec = ('hevc' if fcc in (b'hevc', b'hev1', b'hvc1', b'h265')
                 else 'h264' if fcc in (b'h264', b'avc1', b'x264') else None)
        f = head.find(b'strf', p)
        if 0 <= f < p + 200 and f + 20 < len(head):
            w, h = _u32(head, f + 12), abs(struct.unpack_from('<i', head, f + 16)[0])
        break
    return movi, fps, total, w, h, codec


class _Chunks:
    """Validates chunk headers in the file data."""

    def __init__(self, d, end):
        self.d, self.end = d, end

    def at(self, pos):
        """('v' or 'a', payload size) of a plausible chunk at pos, or None."""
        d = self.d
        if pos < 0 or pos + 12 > self.end:
            return None
        cid = d[pos:pos + 4]
        if not (48 <= cid[0] <= 57 and 48 <= cid[1] <= 57):
            return None
        kind = cid[2:4]
        sz = _u32(d, pos + 4)
        if pos + 8 + sz > self.end or sz > 8_000_000:
            return None
        if kind in (b'dc', b'db'):
            pl = d[pos + 8:pos + 12]
            if sz < 5 or not (pl == _START4 or pl[:3] == _START3):
                return None
            return 'v', sz
        if kind == b'wb':
            return ('a', sz) if 0 < sz <= 1_000_000 else None
        return None

    def next_pos(self, pos, sz):
        """Position of the chunk after one at pos (AVI pads chunks to even sizes;
        some recorders don't)."""
        q = pos + 8 + sz
        if sz & 1 and self.at(q) is None and self.at(q + 1) is not None:
            return q + 1
        return q


def _tail_rows(d, nbytes=_TAIL):
    m = min(len(d), nbytes) // 16
    return np.frombuffer(d, dtype='<u4', count=m * 4, offset=len(d) - m * 16).reshape(-1, 4)


def _increasing(values):
    """Indices of a longest strictly increasing subsequence of ``values``."""
    tails, tails_i, prev = [], [], [-1] * len(values)
    for i, v in enumerate(values):
        j = bisect.bisect_left(tails, v)
        if j == len(tails):
            tails.append(v)
            tails_i.append(i)
        else:
            tails[j] = v
            tails_i[j] = i
        prev[i] = tails_i[j - 1] if j else -1
    out, i = [], (tails_i[-1] if tails_i else -1)
    while i >= 0:
        out.append(i)
        i = prev[i]
    return out[::-1]


def _rows_look_like_index(d, pos, movi, count=64):
    """Whether most of the ``count`` 16-byte rows at ``pos`` are index entries."""
    L = len(d)
    count = min(count, (L - pos) // 16)
    if count < 8:
        return False
    rows = np.frombuffer(d, dtype='<u4', count=4 * count, offset=pos).reshape(-1, 4)
    ok = np.isin(rows[:, 0], _ALL) & (rows[:, 2].astype(np.int64) + movi < L)
    return ok.mean() >= 0.8


def _find_index(d, movi, n_hint=None, data_end=None):
    """(index entries as an (n, 4) uint32 array or None, table position, how found).

    With its ``idx1`` header intact the table gives its own size. Otherwise it
    is taken to end the file (as recorders write it) and its length must be
    known exactly, or the frames could not be put in their place: from
    ``n_hint`` (an intact recording of the same recorder, :func:`read_reference`),
    from its first row surviving (the first chunk sits at offset 4), or from it
    starting right after ``data_end``, the end of the last chunk of data.
    None when none of these holds."""
    L = len(d)
    lo = max(0, L - _TAIL)
    p = d.rfind(b'idx1', lo)
    while p >= 0:
        size = _u32(d, p + 4) if p + 8 <= L else 0
        if size and size % 16 == 0 and p + 8 + size <= L:
            n = size // 16
            return np.frombuffer(d, dtype='<u4', count=4 * n, offset=p + 8).reshape(n, 4), p, 'idx1'
        p = d.rfind(b'idx1', lo, p)
    if movi is None:
        return None, None, None
    n = how = None
    if n_hint:
        n, how = int(n_hint), 'end-aligned (reference)'
    else:
        tail = _tail_rows(d)
        base = L - len(tail) * 16
        for r in np.nonzero((tail[:, 2] == 4) & np.isin(tail[:, 0], _ALL))[0][::-1]:
            if _rows_look_like_index(d, base + 16 * int(r), movi):
                n, how = len(tail) - int(r), 'end-aligned (first row)'
                break
        if n is None and data_end:
            q = data_end + ((L - 8 - data_end) % 16)
            if q + 8 < L and _rows_look_like_index(d, q + 8, movi):
                n, how = (L - q - 8) // 16, 'end-aligned (after the data)'
    if not n:
        return None, None, None
    p = L - 8 - 16 * n
    if p < 0:
        return None, None, None
    return np.frombuffer(d, dtype='<u4', count=4 * n, offset=p + 8).reshape(n, 4), p, how


def read_reference(path):
    """Header facts of an intact recording from the same recorder, for files whose
    own header is lost: {'entries', 'fps', 'total_frames', 'width', 'height',
    'codec', 'movi', 'path'}, or None if ``path`` is not intact."""
    try:
        size = os.path.getsize(path)
        with open(path, 'rb') as f:
            head = f.read(1 << 20)
            f.seek(max(0, size - _TAIL))
            tail = f.read()
    except OSError:
        return None
    movi, fps, total, w, h, codec = _parse_header(head)
    p = tail.rfind(b'idx1')
    if movi is None or p < 0 or p + 8 > len(tail):
        return None
    n = _u32(tail, p + 4) // 16
    if not n or p + 8 + 16 * n != len(tail):       # the table must end the file
        return None
    return dict(entries=n, fps=fps, total_frames=total, width=w, height=h, codec=codec,
                movi=movi, path=path)


def find_reference(path, max_tries=12):
    """:func:`read_reference` of the nearest-named intact AVI beside ``path`` of
    about the same size (same recorder, same segment length), or None."""
    folder = os.path.dirname(os.path.abspath(path))
    me = os.path.basename(path)
    try:
        size = os.path.getsize(path)
        cands = sorted(n for n in os.listdir(folder)
                       if n.lower().endswith('.avi') and n != me
                       and abs(os.path.getsize(os.path.join(folder, n)) - size) <= 0.02 * size)
    except OSError:
        return None
    i = bisect.bisect_left(cands, me)
    for j in sorted(range(len(cands)), key=lambda j: abs(j - i))[:max_tries]:
        ref = read_reference(os.path.join(folder, cands[j]))
        if ref:
            return ref
    return None


def _walk_chains(d, end):
    ck = _Chunks(d, end)
    seen, chains = set(), []
    for m in _CHUNK_RE.finditer(d, 0, end):
        s = m.start()
        if s in seen or ck.at(s) is None:
            continue
        ch, q = [], s
        while True:
            r = ck.at(q)
            if r is None or q in seen:
                break
            seen.add(q)
            ch.append((q, r[0], r[1]))
            q = ck.next_pos(q, r[1])
        if len(ch) >= 2 or (ch and ch[0][1] == 'v'):
            chains.append(ch)
    return ck, chains


def map_avi(path, data=None, reference=None):
    """Map the surviving chunks of an AVI to frame numbers. Returns an
    :class:`AviMap`, or None when nothing can be placed in time.

    ``reference`` (:func:`read_reference` of an intact recording from the same
    recorder) fills in what a lost header would have said."""
    d = data if data is not None else open(path, 'rb').read()
    L = len(d)
    movi, fps, header_total, w, h, codec = _parse_header(d)
    header_ok = movi is not None
    ref = (reference or {}) if not header_ok else {}
    fps = fps or ref.get('fps')
    header_total = header_total or ref.get('total_frames')
    w, h = w or ref.get('width'), h or ref.get('height')
    codec = codec or ref.get('codec')

    # 1. chains: walk the data from every plausible chunk header
    t = d.rfind(b'idx1', max(0, L - _TAIL))
    ck, chains = _walk_chains(d, t if t > 0 else L)
    if not chains:
        return None

    # 2. the index, and the offset base its entries count from
    if movi is None:
        # vote: chunk position - listed offset, over rows whose (id, size) is
        # found exactly once in the data
        by_sig = {}
        for ch in chains:
            for q, _, sz in ch:
                by_sig.setdefault((d[q:q + 4], sz), []).append(q)
        a, _, _ = _find_index(d, None)
        rows = a if a is not None else _tail_rows(d, 8 << 20)
        cand = Counter()
        for r in rows[::max(1, len(rows) // 4000)]:
            hits = by_sig.get((struct.pack('<I', int(r[0])), int(r[3])))
            if hits and len(hits) == 1:
                cand[hits[0] - int(r[2])] += 1
        if not cand:
            return None
        movi = cand.most_common(1)[0][0]
    data_end = max(c[-1][0] + 8 + c[-1][2] for c in chains)
    a, end, how = _find_index(d, movi, ref.get('entries'), data_end)
    if a is None:
        return None
    n = len(a)
    chains = [c for c in chains if c[0][0] < end]

    kind = np.full(n, -1, np.int8)        # 0 video, 1 audio, -1 unknown
    off = np.full(n, -1, np.int64)        # chunk position where its data survives
    listed = np.full(n, -1, np.int64)     # chunk position the index gives
    keyflag = np.zeros(n, bool)
    for k in range(n):
        cid = int(a[k, 0])
        o = int(a[k, 2]) + movi
        if (cid not in _VID and cid not in _AUD) or not (0 <= o < end):
            continue
        kind[k] = 0 if cid in _VID else 1
        listed[k] = o
        keyflag[k] = bool(int(a[k, 1]) & AVIIF_KEYFRAME)
    # rows whose offsets do not rise with the rest belong to another file (a
    # block overwritten by it) or are stray data: drop them
    lk = np.nonzero(listed >= 0)[0]
    keep = np.zeros(n, bool)
    keep[lk[_increasing([int(listed[k]) for k in lk])]] = True
    drop = (listed >= 0) & ~keep
    listed[drop] = -1
    kind[drop] = -1
    pos2k = {}
    for k in np.nonzero(listed >= 0)[0]:
        o = int(listed[k])
        if o + 8 <= end and _u32(d, o) == int(a[k, 0]) and _u32(d, o + 4) == int(a[k, 3]) and ck.at(o):
            off[k] = o
            pos2k[o] = int(k)

    # 3. chains the index still lists are placed exactly
    floating = []
    for ch in chains:
        anchor = next(((i, pos2k[q]) for i, (q, _, _) in enumerate(ch) if q in pos2k), None)
        if anchor is None:
            floating.append(ch)
            continue
        i0, k0 = anchor
        for i, (q, t_, sz) in enumerate(ch):
            k = k0 + i - i0
            tt = 0 if t_ == 'v' else 1
            if 0 <= k < n and kind[k] in (-1, tt) and off[k] < 0:
                kind[k] = tt
                off[k] = q
    # ... the rest between the nearest fixed points: chain lengths are exact,
    # leftover entries are shared among the lost byte ranges by size
    fixed_k = np.nonzero((off >= 0) | (listed >= 0))[0]
    fixed_o = [int(off[k]) if off[k] >= 0 else int(listed[k]) for k in fixed_k]
    groups = {}
    for ch in floating:
        j = bisect.bisect_left(fixed_o, ch[0][0])
        k1, o1 = (int(fixed_k[j - 1]), fixed_o[j - 1]) if j > 0 else (-1, movi + 4)
        k2, o2 = (int(fixed_k[j]), fixed_o[j]) if j < len(fixed_k) else (n, end)
        groups.setdefault((k1, o1, k2, o2), []).append(ch)
    est_k = set()
    for (k1, o1, k2, o2), chs in groups.items():
        chs.sort(key=lambda c: c[0][0])
        left = (k2 - k1 - 1) - sum(len(c) for c in chs)
        if left < 0:
            continue
        gaps, cur = [], o1
        for c in chs:
            gaps.append(max(0, c[0][0] - cur))
            cur = c[-1][0] + 8 + c[-1][2]
        gaps.append(max(0, o2 - cur))
        tot = sum(gaps) or 1
        kk, acc = k1 + 1, 0.0
        for c, g in zip(chs, gaps):
            acc += left * g / tot
            k0 = kk + int(round(acc))
            for i, (q, t_, sz) in enumerate(c):
                k = k0 + i
                if 0 <= k < n and kind[k] == -1 and off[k] < 0:
                    kind[k] = 0 if t_ == 'v' else 1
                    off[k] = q
                    est_k.add(k)
            kk += len(c)

    # 4. frame number of each video entry: the video entries before it, with
    # unknown entries counted at the file's share of video; when the total is
    # known, counted back from the end instead where fewer entries are unknown
    # on that side (a table that lost its first rows is still exact)
    known = kind >= 0
    share = float((kind == 0).sum()) / max(1, int(known.sum()))
    weight = np.where(known, (kind == 0).astype(float), share)
    vnum = np.concatenate([[0.0], np.cumsum(weight)])
    unk = np.concatenate([[0], np.cumsum(~known)])
    total = header_total or int(round(vnum[-1]))

    def frame_number(k):
        if header_total and unk[-1] - unk[k + 1] < unk[k]:
            return total - 1 - int(round(vnum[-1] - vnum[k + 1]))
        return int(round(vnum[k]))

    vid_k = np.nonzero((kind == 0) & (off >= 0))[0]
    if codec is None:
        votes = Counter(_guess_codec(d[int(off[k]) + 8:int(off[k]) + 8 + 64])
                        for k in vid_k[::max(1, len(vid_k) // 200)])
        votes.pop(None, None)
        codec = votes.most_common(1)[0][0] if votes else None
    frames, estimated, damaged = {}, set(), 0
    for k in vid_k:
        fn = frame_number(int(k))
        if fn < 0 or fn >= total or fn in frames:
            continue
        o = int(off[k])
        sz = _u32(d, o + 4)
        if _damaged(d[o + 8:o + 8 + sz]):
            damaged += 1
            continue
        listed_key = bool(keyflag[k]) and listed[k] == o
        frames[fn] = (o, sz, listed_key or d[o + 2:o + 4] == b'db')
        if int(k) in est_k:
            estimated.add(fn)
    if codec is None or not frames:
        return None
    for fn, (o, sz, key) in frames.items():
        if not key:
            frames[fn] = (o, sz, _is_key(d[o + 8:o + 8 + min(sz, 4096)], codec))
    info = dict(index=how, entries=n, unknown_entries=int((kind < 0).sum()),
                frames_with_data=len(frames), damaged_frames=damaged,
                estimated_frames=len(estimated),
                header=header_ok, reference=ref.get('path'))
    return AviMap(frames, total, fps, codec, w, h, estimated, info)


def frame_runs(amap, total=None):
    """Runs of consecutive frame numbers, each starting at its first keyframe
    (frames before it cannot be decoded)."""
    total = total or amap.total_frames
    runs, cur = [], []
    for fn in sorted(f for f in amap.frames if f < total):
        if cur and fn != cur[-1] + 1:
            runs.append(cur)
            cur = []
        if not cur and not amap.frames[fn][2]:
            continue
        cur.append(fn)
    if cur:
        runs.append(cur)
    return runs


def _probe_size(payload, codec):
    r = subprocess.run(['ffprobe', '-v', 'error', '-f', codec, '-i', 'pipe:',
                        '-show_entries', 'stream=width,height', '-of', 'csv=p=0'],
                       input=payload, capture_output=True)
    try:
        w, h = r.stdout.decode().strip().splitlines()[0].split(',')[:2]
        return int(w), int(h)
    except (ValueError, IndexError):
        return None


def _keep_full_speed(pid):
    try:
        from fnt.usv.usv_detector.power import keep_full_speed
        keep_full_speed(pid)
    except Exception:
        pass


# --- recovery ------------------------------------------------------------------

@dataclass
class RecoveryResult:
    output: str
    total_frames: int
    recovered: int
    black: int
    rows: list              # (first_frame, last_frame, 'recovered'|'black', 'index'|'estimated'|'')
    info: dict


def recover_avi(path, output, encoder_cmd, gray=True, fps=None, progress=None,
                should_stop=None, data=None, amap=None, reference='auto'):
    """Write ``output`` from the frames of ``path`` that survive, black where none
    does, at the recording's real length.

    ``encoder_cmd(width, height, pix_fmt, fps, output)`` returns the FFmpeg command
    that reads raw frames of that size and format from stdin and writes
    ``output``; ``gray`` picks the raw format (``gray`` or ``yuv420p``). ``fps`` is
    used only when the recording's frame rate is unknown. ``reference``: 'auto'
    looks for an intact AVI beside ``path`` when this file's header is lost, or
    pass :func:`read_reference` output, or None. ``should_stop()`` is polled
    between runs. Returns a :class:`RecoveryResult`, or None when the file
    cannot be mapped, is stopped, or the encoder fails.
    """
    d = data if data is not None else open(path, 'rb').read()
    if amap is None:
        if reference == 'auto':
            reference = None if _parse_header(d)[0] is not None else find_reference(path)
        amap = map_avi(path, d, reference)
    if amap is None:
        return None
    rate = amap.fps or fps or 30.0
    runs = frame_runs(amap)
    if not runs:
        return None
    w, h = amap.width, amap.height
    if not (w and h):
        o, sz, _ = amap.frames[runs[0][0]]
        wh = _probe_size(d[o + 8:o + 8 + sz], amap.codec)
        if wh is None:
            return None
        w, h = wh
    pix = 'gray' if gray else 'yuv420p'
    fsize = w * h if gray else w * h * 3 // 2
    black = bytes(fsize) if gray else bytes(w * h) + b'\x80' * (fsize - w * h)
    total = amap.total_frames
    enc = subprocess.Popen(encoder_cmd(w, h, pix, rate, output), stdin=subprocess.PIPE,
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    _keep_full_speed(enc.pid)
    rows, written, stopped = [], 0, False

    def black_until(n):
        nonlocal written
        if n > written:
            rows.append((written, n - 1, 'black', ''))
            while written < n:
                enc.stdin.write(black)
                written += 1

    try:
        for r in runs:
            if should_stop and should_stop():
                stopped = True
                break
            black_until(r[0])
            dec = subprocess.Popen(
                ['ffmpeg', '-nostdin', '-v', 'quiet', '-f', amap.codec, '-i', 'pipe:', '-vf',
                 'setparams=colorspace=bt709:color_primaries=bt709:color_trc=bt709,'
                 f'scale={w}:{h},format={pix}',
                 '-fps_mode', 'passthrough', '-f', 'rawvideo', '-pix_fmt', pix, 'pipe:'],
                stdin=subprocess.PIPE, stdout=subprocess.PIPE, stderr=subprocess.DEVNULL)
            _keep_full_speed(dec.pid)

            def feed(r=r, dec=dec):
                try:
                    for fn in r:
                        o, sz, _ = amap.frames[fn]
                        dec.stdin.write(d[o + 8:o + 8 + sz])
                    dec.stdin.close()
                except OSError:
                    pass

            th = threading.Thread(target=feed, daemon=True)
            th.start()
            got, start = 0, written
            while got < len(r):
                buf = dec.stdout.read(fsize)
                if len(buf) < fsize:
                    break
                enc.stdin.write(buf)
                got += 1
                written += 1
            dec.stdout.close()
            dec.wait()
            th.join()
            if got:
                how = 'estimated' if any(fn in amap.estimated for fn in r[:got]) else 'index'
                rows.append((start, start + got - 1, 'recovered', how))
            black_until(r[0] + len(r))      # frames of the run that did not decode
            if progress:
                progress(written, total)
        if not stopped:
            black_until(total)
        enc.stdin.close()
    except OSError:
        stopped = True
    rc = enc.wait()
    if stopped or rc != 0:
        return None
    rec = sum(b - a + 1 for a, b, k, _ in rows if k == 'recovered')
    return RecoveryResult(output, total, rec, total - rec, rows, dict(amap.info, fps=rate))


def write_frames_csv(result, path):
    """One row per stretch of the output: recovered frames or black."""
    fps = result.info['fps']
    with open(path, 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['first_frame', 'last_frame', 'start_s', 'end_s', 'content', 'placement'])
        for a, b, k, how in result.rows:
            w.writerow([a, b, f'{a / fps:.3f}', f'{(b + 1) / fps:.3f}', k, how])


def default_encoder_cmd(crf=20, preset='ultrafast', codec='libx265', width=1920, height=1080):
    """FFmpeg command builder for :func:`recover_avi` with Video PreProcessing's
    defaults (1080p pad, grayscale if the frames are, no audio, mp4)."""
    def build(w, h, pix, fps, output):
        vf = [f'scale={width}:{height}:force_original_aspect_ratio=decrease',
              f'pad={width}:{height}:-1:-1:color=black']
        if pix == 'gray':
            vf.append('format=gray')
        cmd = ['ffmpeg', '-y', '-nostdin', '-v', 'error', '-f', 'rawvideo', '-pix_fmt', pix,
               '-s', f'{w}x{h}', '-r', f'{fps:g}', '-i', 'pipe:', '-vf', ','.join(vf),
               '-c:v', codec, '-preset', preset, '-crf', str(crf)]
        if codec == 'libx265':
            cmd += ['-x265-params', 'log-level=error']
        return cmd + ['-pix_fmt', 'yuv420p', '-an', '-movflags', '+faststart', '-f', 'mp4', output]
    return build


def main(argv=None):
    import argparse
    ap = argparse.ArgumentParser(
        prog='python -m fnt.videoProcessing.avi_recovery',
        description='Recover damaged AVIs at their real length: surviving frames in '
                    'place, black where none survives. Writes <out>/<name>.mp4 and '
                    '<name>_frames.csv (Video PreProcessing default encode settings).')
    ap.add_argument('avi', nargs='+')
    ap.add_argument('--out', default=None, help="output folder (default: 'proc' beside each AVI)")
    ap.add_argument('--reference', default=None,
                    help='intact AVI from the same recorder, for files whose header is lost '
                         '(default: the nearest intact AVI of about the same size beside it)')
    ap.add_argument('--fps', type=float, default=30.0,
                    help='frame rate if the recording does not say (default 30)')
    ap.add_argument('--color', action='store_true', help='keep colour (default grayscale)')
    ap.add_argument('--crf', type=int, default=20)
    ap.add_argument('--preset', default='ultrafast')
    ap.add_argument('--map-only', action='store_true', help='report what survives; write nothing')
    a = ap.parse_args(argv)
    rc = 0
    for path in a.avi:
        name = os.path.splitext(os.path.basename(path))[0]
        with open(path, 'rb') as f:
            d = f.read()
        ref = None
        if _parse_header(d)[0] is None:
            ref = read_reference(a.reference) if a.reference else find_reference(path)
        amap = map_avi(path, d, ref)
        if amap is None:
            print(f'{name}: nothing recoverable')
            rc = 1
            continue
        usable = sum(len(r) for r in frame_runs(amap))
        print(f'{name}: {usable}/{amap.total_frames} frames decodable '
              f'({usable / max(1, amap.total_frames):.1%}); {amap.info}', flush=True)
        if a.map_only:
            continue
        out_dir = a.out or os.path.join(os.path.dirname(os.path.abspath(path)), 'proc')
        os.makedirs(out_dir, exist_ok=True)
        out = os.path.join(out_dir, name + '.mp4')
        part = out + '.partial'
        last = [0]

        def prog(done, total):
            if done - last[0] >= max(1, total // 20):
                last[0] = done
                print(f'  {done}/{total} frames', flush=True)

        res = recover_avi(path, part, default_encoder_cmd(a.crf, a.preset), gray=not a.color,
                          fps=a.fps, progress=prog, data=d, amap=amap)
        if res is None:
            print(f'{name}: recovery failed')
            rc = 1
            if os.path.exists(part):
                os.remove(part)
            continue
        os.replace(part, out)
        write_frames_csv(res, os.path.join(out_dir, name + '_frames.csv'))
        print(f'{name}: {res.recovered}/{res.total_frames} frames recovered '
              f'({res.recovered / res.total_frames:.1%}), {res.black} black -> {out}', flush=True)
    return rc


if __name__ == '__main__':
    raise SystemExit(main())
