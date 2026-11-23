"""USV lane for the Fusion View: the microphones' spectrograms under the camera grid, MAD's detected calls drawn
on them, and call bouts to step through.

Audio and detections come from a trial's USV folder (``chN_*`` subfolders, or ``chN...wav`` files, of Avisoft
recordings named ``chN<YYYY-MM-DD_HH-MM-SS>_<seq>.wav`` in the recording PC's local clock), the calls from each
recording's MAD store (``_FNT.mad``) through MAD's own readers. Mic N sits at resource zone N, so it belongs to the
zone camera whose ``zone`` is ``ZN``.

Time: the PC's clock (audio, UWB) = the video's clock + ``offset_s`` (the recorder ran 121 s ahead in VoleCosm, so
-121, the same as ``uwb_offset_s``). Bouts are read from a bout table (``mic``, ``start_utc_s``, ``end_utc_s`` in
PC time, as ``usv_bouts/<trial>_usv_bouts.csv`` holds them) or, without one, built from the calls: a mic's
fundamental calls no more than ``gap_s`` apart.

identity.json may say where things are (otherwise the folders are guessed from the trial folder the cameras sit in)::

    "usv": {"dir": ".../UltrasonicVocalizations_proc/VC-T005_prairie_anosmic-saline", "offset_s": -121,
            "bouts": ".../usv_bouts/VC-T005_usv_bouts.csv", "tz": "America/Denver"}
"""
from __future__ import annotations

import glob
import os
import re
import struct
import subprocess
import threading
from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

WAV_RE = re.compile(r"^ch(\d+)(\d{4}-\d\d-\d\d_\d\d-\d\d-\d\d)_\d+\.wav$", re.IGNORECASE)
TILE_S = 4.0                # spectrogram tiles: seconds each
PX_PER_S = 160              # tile columns per second
ROWS = 112                  # tile rows (frequency)
F_LO, F_HI = 15_000.0, 125_000.0
DB_FLOOR, DB_RANGE = 6.0, 30.0   # dB above each row's noise floor drawn black / white
GRID = dict(nperseg=512, noverlap=384, nfft=1024)     # MAD's grid (only for drawing; calls carry their own)


# ---------------------------------------------------------------------- audio
def _wav_layout(path: str) -> Dict:
    """Format, rate, block size and data offset of a WAV file, from its header."""
    with open(path, "rb") as f:
        head = f.read(65536)
    if head[:4] != b"RIFF" or head[8:12] != b"WAVE":
        raise ValueError(f"not a WAV file: {path}")
    i, out = 12, {}
    while i + 8 <= len(head):
        cid, size = head[i:i + 4], struct.unpack("<I", head[i + 4:i + 8])[0]
        if cid == b"fmt ":
            tag, ch, sr, _bps, align, bits = struct.unpack("<HHIIHH", head[i + 8:i + 24])
            if tag == 0xFFFE and size >= 40:                 # WAVE_FORMAT_EXTENSIBLE: the sub-format's first 2 bytes
                tag = struct.unpack("<H", head[i + 32:i + 34])[0]
            out.update(tag=tag, channels=ch, sr=sr, align=align, bits=bits, fmt=head[i:i + 8 + size])
        elif cid == b"data":
            out.update(data_at=i + 8, data_len=size)
            break
        i += 8 + size + (size & 1)
    if "tag" not in out or "data_at" not in out:
        raise ValueError(f"no fmt/data chunk in {path}")
    return out


def read_samples(path: str, start: int, n: int) -> Tuple[np.ndarray, int]:
    """``n`` samples (mono float32 in [-1, 1]) from sample ``start`` of a WAV, and its rate. IMA ADPCM (Avisoft's
    compressed files) is cut by whole blocks, which decode on their own, and decoded by ffmpeg, so the samples are
    exact without decoding the file from its start (ffmpeg's own seek lands tens of ms off in these files)."""
    lay = _wav_layout(path)
    sr, ch = lay["sr"], max(1, lay["channels"])
    start = max(0, int(start))
    if lay["tag"] == 1 and lay["bits"] == 16:                # plain PCM
        with open(path, "rb") as f:
            f.seek(lay["data_at"] + start * 2 * ch)
            x = np.frombuffer(f.read(int(n) * 2 * ch), "<i2").reshape(-1, ch)[:, 0]
        return x.astype(np.float32) / 32768.0, sr
    if lay["tag"] == 0x11:                                   # IMA ADPCM
        align = lay["align"]
        spb = (align - 4 * ch) * 8 // (4 * ch) + 1           # samples per block
        b0 = start // spb
        nb = (start - b0 * spb + int(n)) // spb + 1
        with open(path, "rb") as f:
            f.seek(lay["data_at"] + b0 * align)
            data = f.read(nb * align)
        data = data[:len(data) // align * align]
        fmt = lay["fmt"]
        body = b"WAVE" + fmt + (b"\0" if len(fmt) & 1 else b"") + b"data" + struct.pack("<I", len(data)) + data
        wav = b"RIFF" + struct.pack("<I", len(body)) + body
        r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-f", "wav", "-i", "pipe:0", "-f", "s16le",
                            "-ac", "1", "pipe:1"], input=wav, capture_output=True, check=True)
        x = np.frombuffer(r.stdout, "<i2").astype(np.float32) / 32768.0
        skip = start - b0 * spb
        return x[skip:skip + int(n)], sr
    # anything else: decode with ffmpeg from the start, seeking after -i (exact, slower)
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", path, "-ss", f"{start / sr:.6f}",
                        "-t", f"{n / sr:.6f}", "-f", "s16le", "-ac", "1", "pipe:1"], capture_output=True, check=True)
    return np.frombuffer(r.stdout, "<i2").astype(np.float32) / 32768.0, sr


def spectrogram_tile(x: np.ndarray, sr: int, width: int, rows: int = ROWS) -> np.ndarray:
    """A uint8 picture (rows x width, low frequencies at the bottom) of audio ``x``: MAD's spectrogram, F_LO-F_HI,
    each row's median taken as its noise floor, DB_FLOOR above it black and DB_RANGE above it white."""
    from fnt.usv.usv_detector.spectrogram import compute_spectrogram
    if len(x) < GRID["nperseg"]:
        return np.zeros((rows, width), np.uint8)
    f, _t, s = compute_spectrogram(x, sr, min_freq=F_LO, max_freq=F_HI, **GRID)
    s = s - np.median(s, axis=1, keepdims=True)
    s = np.clip((s - DB_FLOOR) / (DB_RANGE - DB_FLOOR), 0, 1)
    # pool to the picture's size, keeping the loudest (calls are thin)
    nf, nt = s.shape
    ti = np.linspace(0, nt, width + 1).astype(int)
    fi = np.linspace(0, nf, rows + 1).astype(int)
    cols = np.stack([s[:, a:max(b, a + 1)].max(1) for a, b in zip(ti[:-1], ti[1:])], 1) if nt else np.zeros((nf, width))
    img = np.stack([cols[a:max(b, a + 1)].max(0) for a, b in zip(fi[:-1], fi[1:])], 0)
    return (img[::-1] * 255).astype(np.uint8)


# ---------------------------------------------------------------------- recordings, calls and bouts
@dataclass
class Recording:
    mic: int
    wav: str
    t0: float               # PC clock, epoch seconds
    dur: float = 600.0
    mad: Optional[str] = None


@dataclass
class UsvSource:
    """A trial's recordings, their MAD calls and bouts, on the PC clock (epoch seconds)."""
    usv_dir: str
    tz: object
    recordings: Dict[int, List[Recording]] = field(default_factory=dict)
    bouts_csv: Optional[str] = None

    @classmethod
    def scan(cls, usv_dir: str, tz, bouts_csv: Optional[str] = None) -> "UsvSource":
        recs: Dict[int, List[Recording]] = {}
        for wav in glob.glob(os.path.join(usv_dir, "*.wav")) + glob.glob(os.path.join(usv_dir, "ch*", "*.wav")):
            m = WAV_RE.match(os.path.basename(wav))
            if not m:
                continue
            local = datetime.strptime(m.group(2), "%Y-%m-%d_%H-%M-%S")
            t0 = _aware(local, tz).timestamp()
            stem = wav[:-4]
            mad = next((p for p in (stem + "_FNT.mad", stem + ".mad", stem + "_FNT_masks.h5") if os.path.exists(p)),
                       None)
            recs.setdefault(int(m.group(1)), []).append(Recording(int(m.group(1)), wav, t0, mad=mad))
        for mic, rs in recs.items():
            rs.sort(key=lambda r: r.t0)
            for a, b in zip(rs, rs[1:]):                     # each runs until the next starts (Avisoft: 10 min)
                a.dur = min(a.dur, b.t0 - a.t0) if b.t0 > a.t0 else a.dur
        return cls(usv_dir, tz, recs, bouts_csv if bouts_csv and os.path.exists(bouts_csv) else None)

    @property
    def mics(self) -> List[int]:
        return sorted(self.recordings)

    def local(self, t: float) -> datetime:
        """PC time t as the PC's local clock."""
        tz = self.tz
        if isinstance(tz, str):
            from zoneinfo import ZoneInfo
            tz = ZoneInfo(tz)
        return datetime.fromtimestamp(t, tz or timezone.utc)

    def recording_at(self, mic: int, t: float) -> Optional[Recording]:
        for r in self.recordings.get(mic, ()):
            if r.t0 <= t < r.t0 + r.dur:
                return r
        return None

    def audio(self, mic: int, t0: float, t1: float) -> Tuple[np.ndarray, int]:
        """Mic ``mic``'s audio from t0 to t1 (PC clock), zeros where no recording covers it."""
        rs = [r for r in self.recordings.get(mic, ()) if r.t0 < t1 and r.t0 + r.dur > t0]
        sr = None
        parts = []
        for r in rs:
            a, b = max(t0, r.t0), min(t1, r.t0 + r.dur)
            lay_sr = _wav_layout(r.wav)["sr"]
            sr = sr or lay_sr
            x, _ = read_samples(r.wav, int(round((a - r.t0) * lay_sr)), int(round((b - a) * lay_sr)))
            parts.append((a, x))
        sr = sr or 250_000
        out = np.zeros(int(round((t1 - t0) * sr)), np.float32)
        for a, x in parts:
            i = int(round((a - t0) * sr))
            k = min(len(x), len(out) - i)
            if k > 0:
                out[i:i + k] = x[:k]
        return out, sr

    def calls(self, mic: int, t0: float, t1: float) -> pd.DataFrame:
        """MAD's detections on mic ``mic`` from t0 to t1 (PC clock): t0, t1 (s), f_lo, f_hi (Hz), score,
        fundamental (not a harmonic), from each recording's store through MAD's readers."""
        from fnt.usv.usv_detector import fnt_mask_store as S
        from fnt.usv.usv_detector.mad_inference import frames_to_seconds
        rows = []
        for r in self.recordings.get(mic, ()):
            if r.mad is None or r.t0 >= t1 or r.t0 + r.dur <= t0:
                continue
            try:
                s = S.read_store_summary(r.mad, preds=True)
            except OSError:
                continue
            g = s["grid"]
            try:
                sr, nps, nov, nfft = int(g["sample_rate"]), int(g["nperseg"]), int(g["noverlap"]), int(g["nfft"])
            except KeyError:
                continue
            for key, c in (s.get("preds") or {}).items():
                nf, nt = c["mask"].shape
                a = r.t0 + float(frames_to_seconds(c["t_off"], nps, nov, sr))
                b = r.t0 + float(frames_to_seconds(c["t_off"] + nt, nps, nov, sr))
                hn = c.get("harmonic_n")
                try:
                    fund = hn in (None, "") or int(float(hn)) <= 1
                except (TypeError, ValueError):
                    fund = True
                rows.append(dict(mic=mic, t0=a, t1=b, f_lo=c["f_off"] * sr / nfft, f_hi=(c["f_off"] + nf) * sr / nfft,
                                 score=float(c.get("score", np.nan)), fundamental=fund, blob=str(key)))
        df = pd.DataFrame(rows, columns=["mic", "t0", "t1", "f_lo", "f_hi", "score", "fundamental", "blob"])
        return df[(df.t1 > t0) & (df.t0 < t1)].sort_values("t0").reset_index(drop=True)

    def bouts(self, calls: pd.DataFrame, t0: float, t1: float, gap_s: float = 2.0) -> Tuple[pd.DataFrame, str]:
        """Bouts (mic, t0, t1, n_calls; PC clock) between t0 and t1: from the bout table when there is one, else
        from ``calls`` (fundamentals no more than gap_s apart, per mic). Returns (bouts, where from)."""
        if self.bouts_csv:
            b = pd.read_csv(self.bouts_csv)
            if {"mic", "start_utc_s", "end_utc_s"} <= set(b.columns):
                b = pd.DataFrame(dict(mic=b.mic.astype(int), t0=b.start_utc_s.astype(float),
                                      t1=b.end_utc_s.astype(float),
                                      n_calls=b.n_calls if "n_calls" in b else np.nan))
                b = b[(b.t1 > t0) & (b.t0 < t1)].sort_values("t0").reset_index(drop=True)
                return b, os.path.basename(self.bouts_csv)
        return bouts_from_calls(calls, gap_s), f"calls {gap_s:g} s apart"


def bouts_from_calls(calls: pd.DataFrame, gap_s: float = 2.0) -> pd.DataFrame:
    """A mic's fundamental calls no more than gap_s apart (end to start) make one bout."""
    out = []
    c = calls[calls.fundamental.astype(bool)] if len(calls) else calls
    for mic, g in c.sort_values("t0").groupby("mic"):
        a, b, n = None, None, 0
        for r in g.itertuples():
            if a is not None and r.t0 - b <= gap_s:
                b, n = max(b, r.t1), n + 1
                continue
            if a is not None:
                out.append((mic, a, b, n))
            a, b, n = r.t0, r.t1, 1
        if a is not None:
            out.append((mic, a, b, n))
    return pd.DataFrame(out, columns=["mic", "t0", "t1", "n_calls"]).sort_values("t0").reset_index(drop=True)


def _aware(local: datetime, tz) -> datetime:
    if isinstance(tz, str):
        from zoneinfo import ZoneInfo
        tz = ZoneInfo(tz)
    return local.replace(tzinfo=tz or timezone.utc)


def mic_of_zone(zone) -> Optional[int]:
    """Z3 -> 3."""
    m = re.search(r"(\d+)", str(zone or ""))
    return int(m.group(1)) if m else None


def locate_usv(cfg: Dict, cams: Dict) -> Tuple[Optional[str], Optional[str]]:
    """(USV folder, bout table) of a run: identity.json's ``usv`` entry, else guessed from the trial folder the
    cameras' files sit in (``.../Video_MesocosmCameras/<trial>/...`` -> ``.../UltrasonicVocalizations_proc/
    <trial>`` and ``.../Video_MesocosmCameras/<trial>/usv_bouts/*_usv_bouts.csv``)."""
    u = cfg.get("usv") or {}
    usv_dir, bouts = u.get("dir"), u.get("bouts")
    if usv_dir and bouts:
        return usv_dir, bouts
    paths = [str(v) for c in cams.values() for v in (c or {}).values() if isinstance(v, str)]
    paths += [str(v) for c in (cfg.get("cameras") or {}).values() for v in (c or {}).values() if isinstance(v, str)]
    for p in paths:
        m = re.search(r"^(.*?)[\\/]Video_MesocosmCameras[\\/]([^\\/]+)", p)
        if not m:
            continue
        root, trial = m.group(1), m.group(2)
        guess = os.path.join(root, "UltrasonicVocalizations_proc", trial)
        if not usv_dir and os.path.isdir(guess):
            usv_dir = guess
        if not bouts:
            hits = sorted(glob.glob(os.path.join(root, "Video_MesocosmCameras", trial, "usv_bouts", "*_usv_bouts.csv")))
            bouts = hits[0] if hits else None
        if usv_dir:
            break
    return usv_dir, bouts


# ---------------------------------------------------------------------- the tiles, made off the GUI thread
class TileCache:
    """Spectrogram tiles (mic, k) = TILE_S seconds from k * TILE_S (PC clock), made by worker threads and kept
    (most recent ~200). ``on_ready(key)`` is called from the worker thread when one is done."""

    def __init__(self, source: UsvSource, on_ready, workers: int = 2, keep: int = 200):
        self.source, self.on_ready, self.keep = source, on_ready, keep
        self.tiles: "OrderedDict[tuple, np.ndarray]" = OrderedDict()
        self.pending = set()
        self.lock = threading.Lock()
        self.closed = False
        self.pool = ThreadPoolExecutor(workers, thread_name_prefix="usv-tile")

    def get(self, mic: int, k: int) -> Optional[np.ndarray]:
        key = (mic, k)
        if self.closed:
            return None
        with self.lock:
            t = self.tiles.get(key)
            if t is not None:
                self.tiles.move_to_end(key)
                return t
            if key in self.pending:
                return None
            self.pending.add(key)
        self.pool.submit(self._make, key)
        return None

    def _make(self, key):
        mic, k = key
        try:
            x, sr = self.source.audio(mic, k * TILE_S, (k + 1) * TILE_S)
            img = spectrogram_tile(x, sr, int(TILE_S * PX_PER_S))
        except Exception:                                    # unreadable audio: an empty tile, not a crash
            img = np.zeros((ROWS, int(TILE_S * PX_PER_S)), np.uint8)
        with self.lock:
            self.tiles[key] = img
            self.pending.discard(key)
            while len(self.tiles) > self.keep:
                self.tiles.popitem(last=False)
        if not self.closed:
            self.on_ready(key)

    def close(self):
        """Stop making tiles: queued ones are dropped, the ones being made finish without telling anyone (their
        receiver may be gone)."""
        self.closed = True
        self.pool.shutdown(wait=True, cancel_futures=True)
