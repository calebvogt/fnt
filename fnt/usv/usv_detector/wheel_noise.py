"""Running-wheel squeak detection from the audio alone.

A running wheel near a microphone makes two sounds at once:

* **thumps** in the low register (2-15 kHz), one per stride or rung, at a
  steady beat (about 7 Hz in the 2021_8x8 cages) while a mouse runs;
* short, narrow **squeaks** in the upper register (often 100-112 kHz) that
  the USV model can take for calls. They come back at the thump beat.

A mouse can call while it runs, so running alone rejects nothing. A
detection is flagged as a squeak when, over the seconds around it,

* the low band beats cleanly (its autocorrelation peaks at 1, 2 and 3 beats
  and dips between them: a "comb"),
* the energy in **the detection's own frequency band** combs at that same
  beat, and
* the detection is shorter than most of a beat.

A call that lasts several beats, or a bout with its own rhythm, does not comb
at the wheel's beat and is kept. Requiring the low band's own comb is what
keeps a dense call bout (about 8 calls/s, near the wheel's rhythm) over the
mouse's own movement noise: that noise is rhythmic only loosely.

Calibrated on the 2021_8x8 MAD labels (2026-11-03): 1 of 1,515 accepted calls
flagged, 40 % of the rejected C57 detections and 75 % of the squeak train
hand-checked in T003 C57 ch2 _1733 (144-160 s).

The results go to ``/wheel`` in the recording's ``.mad`` (see
:func:`write_wheel`); nothing else in the file is touched.
"""
from __future__ import annotations

from typing import Dict, List, Optional

import numpy as np

LOW_BAND_HZ = (2_000.0, 15_000.0)
#: Beats searched for in the low band (Hz).
BEAT_RANGE_HZ = (2.0, 20.0)
#: A lag counts as the fundamental if its autocorrelation is within this
#: fraction of the best peak; the shortest such lag wins, so a 137-ms thump
#: train is not read as its 411-ms subharmonic.
SUBHARMONIC_TOL = 0.8
#: Seconds of audio either side of a detection used to measure the beat.
HALF_WINDOW_S = 2.0
SMOOTH_FRAMES = 9          # ~4.6 ms at the MAD hop

# The squeak rule.
COMB_MIN = 0.25            # the detection band's comb at the wheel beat
LOW_COMB_MIN = 0.15        # the low band's own comb (a clean wheel beat)
LOW_ACF_MIN = 0.20         # low band periodic at all (wheel turning)
MAX_DURATION_BEATS = 0.6

#: File-level running track (4-s windows every 0.5 s of the low band).
TRACK_WINDOW_S = 4.0
TRACK_STEP_S = 0.5
RUNNING_ACF_MIN = 0.30

#: Bumped whenever the rule or its inputs change; a ``/wheel`` group with an
#: older version is recomputed.
WHEEL_VERSION = 2
WHEEL_GROUP = "wheel"


def _smooth(v: np.ndarray, n: int = SMOOTH_FRAMES) -> np.ndarray:
    from scipy.ndimage import uniform_filter1d
    return uniform_filter1d(np.asarray(v, np.float64), n, mode="nearest") if n > 1 else np.asarray(v, float)


def acf(v: np.ndarray) -> np.ndarray:
    """Normalised autocorrelation (lag 0 = 1)."""
    v = np.asarray(v, np.float64)
    v = v - v.mean()
    n = len(v)
    f = np.fft.rfft(v, 2 * n)
    a = np.fft.irfft(f * np.conj(f))[:n]
    return a / (a[0] + 1e-12)


def beat_period(env: np.ndarray, hop_s: float, beat_range=BEAT_RANGE_HZ,
                tol=SUBHARMONIC_TOL):
    """``(period_frames, acf_at_period)`` of an envelope, preferring the
    fundamental over its subharmonics; ``(0, 0.0)`` without a peak in range."""
    a = acf(env)
    lo = max(2, int(round(1.0 / beat_range[1] / hop_s)))
    hi = min(len(a) - 2, int(round(1.0 / beat_range[0] / hop_s)))
    if hi <= lo + 2:
        return 0, 0.0
    seg = a[lo - 1:hi + 2]
    pk = np.flatnonzero((seg[1:-1] > seg[:-2]) & (seg[1:-1] >= seg[2:])) + lo
    pk = pk[a[pk] > 0]
    if not len(pk):
        return 0, 0.0
    i = int(pk[np.flatnonzero(a[pk] >= tol * a[pk].max())[0]])
    return i, float(a[i])


def comb_score(a: np.ndarray, period: float, kmax: int = 3, tol: float = 0.08) -> float:
    """How much an autocorrelation peaks at whole beats: mean over k = 1..3 of
    (best value within 8 % of k beats) minus (value at k - 1/2 beats)."""
    on, off = [], []
    for k in range(1, kmax + 1):
        c = int(round(k * period))
        w = max(1, int(round(tol * k * period)))
        if c + w >= len(a):
            break
        on.append(a[c - w:c + w + 1].max())
        off.append(a[int(round((k - 0.5) * period))])
    return float(np.mean(on) - np.mean(off)) if on else float("nan")


_EMPTY = dict(period_s=0.0, low_acf=0.0, comb=float("nan"), low_comb=float("nan"),
              dur_beats=float("nan"), squeak=False)
_FEATURES = ("comb", "low_comb", "period_s", "low_acf", "dur_beats")


def detection_features(low_db: np.ndarray, band_db: np.ndarray, det_t0: int,
                       det_t1: int, hop_s: float) -> Dict:
    """Wheel features of one detection.

    ``low_db`` and ``band_db`` are per-frame mean dB of the low band and of
    the detection's own frequency rows, over a window centred on it
    (``HALF_WINDOW_S`` each side); ``det_t0/1`` are its frames within it.
    """
    low, band = _smooth(low_db), _smooth(band_db)
    p, r = beat_period(low, hop_s)
    out = dict(_EMPTY, period_s=p * hop_s, low_acf=r)
    if not p:
        return out
    out["comb"] = comb_score(acf(band), p)
    out["low_comb"] = comb_score(acf(low), p)
    out["dur_beats"] = (det_t1 - det_t0) / p
    out["squeak"] = bool(out["comb"] >= COMB_MIN and out["low_comb"] >= LOW_COMB_MIN
                         and r >= LOW_ACF_MIN and out["dur_beats"] <= MAX_DURATION_BEATS)
    return out


def _frames_fft(x: np.ndarray, a: int, b: int, hop: int, nperseg: int, nfft: int):
    """Complex spectrum (frames ``a..b-1`` x bins) on the MAD grid: frame ``i``
    covers samples ``i*hop .. i*hop + nperseg`` (Hann window)."""
    from numpy.lib.stride_tricks import sliding_window_view
    from scipy.fft import rfft
    from scipy.signal import get_window
    seg = np.asarray(x[a * hop:(b - 1) * hop + nperseg], np.float32)
    fr = sliding_window_view(seg, nperseg)[::hop]
    w = get_window("hann", nperseg).astype(np.float32)
    return rfft(fr * w, nfft, axis=1)          # float32 in, complex64 out


def _rows_db(z: np.ndarray, rows: np.ndarray) -> np.ndarray:
    """Mean over ``rows`` of each frame's dB."""
    q = z[:, rows]
    return (10.0 * np.log10(q.real ** 2 + q.imag ** 2 + 1e-20)).mean(1)


#: Detections whose windows overlap share one spectrum, up to this span.
_MAX_SPAN_S = 20.0


def score_detections(x: np.ndarray, sr: float, dets: List[Dict], nperseg: int = 512,
                     noverlap: int = 384, nfft: int = 1024) -> List[Dict]:
    """Features for each detection ``{t0, t1}`` (frames) ``{f_lo, f_hi}`` (Hz),
    in the order given."""
    hop = nperseg - noverlap
    hop_s = hop / sr
    n_frames = (len(x) - nperseg) // hop + 1
    h = int(round(HALF_WINDOW_S / hop_s))
    span = int(round(_MAX_SPAN_S / hop_s))
    f = np.arange(nfft // 2 + 1) * sr / nfft
    low_rows = np.flatnonzero((f >= LOW_BAND_HZ[0]) & (f < LOW_BAND_HZ[1]))
    win = []
    for d in dets:
        c = (int(d["t0"]) + int(d["t1"])) // 2
        win.append((max(0, c - h), min(n_frames, c + h)))
    out: List[Optional[Dict]] = [None] * len(dets)
    order = sorted(range(len(dets)), key=lambda i: win[i][0])
    i = 0
    while i < len(order):
        a, b = win[order[i]]
        j = i + 1
        while j < len(order) and win[order[j]][0] < b and win[order[j]][1] - a <= span:
            b = max(b, win[order[j]][1])
            j += 1
        if b - a >= 4:
            z = _frames_fft(x, a, b, hop, nperseg, nfft)
            low = _rows_db(z, low_rows)
        for k in order[i:j]:
            wa, wb = win[k]
            if wb - wa < 4 or b - a < 4:
                out[k] = dict(_EMPTY)
                continue
            d = dets[k]
            sel = np.flatnonzero((f >= d["f_lo"]) & (f <= d["f_hi"]))
            if not len(sel):
                sel = np.array([np.argmin(np.abs(f - (d["f_lo"] + d["f_hi"]) / 2))])
            out[k] = detection_features(low[wa - a:wb - a], _rows_db(z[wa - a:wb - a], sel),
                                        int(d["t0"]) - wa, int(d["t1"]) - wa, hop_s)
        i = j
    return out


def low_band_envelope_db(x: np.ndarray, sr: float, hop: int = 128) -> np.ndarray:
    """Low-band energy (dB) per MAD frame, from a band-pass filter (cheap
    enough for a whole recording)."""
    from scipy.signal import butter, sosfilt
    sos = butter(4, list(LOW_BAND_HZ), "bandpass", fs=sr, output="sos")
    y = sosfilt(sos, np.asarray(x, np.float32)).astype(np.float32)
    n = len(y) // hop
    p = (y[:n * hop].reshape(n, hop) ** 2).mean(1, dtype=np.float64)
    return 10.0 * np.log10(p + 1e-20)


def running_track(env_db: np.ndarray, hop_s: float) -> Dict[str, np.ndarray]:
    """Beat and running flag in 4-s windows every 0.5 s: ``t_s`` (centre),
    ``period_s``, ``acf``, ``comb`` (the envelope's own comb at the beat) and
    ``running`` (both above their thresholds)."""
    env = _smooth(env_db)
    w = int(round(TRACK_WINDOW_S / hop_s))
    s = int(round(TRACK_STEP_S / hop_s))
    t, per, ac, cb = [], [], [], []
    for a0 in range(0, max(1, len(env) - w + 1), s):
        seg = env[a0:a0 + w]
        p, r = beat_period(seg, hop_s)
        t.append((a0 + len(seg) / 2) * hop_s)
        per.append(p * hop_s)
        ac.append(r)
        cb.append(comb_score(acf(seg), p) if p else float("nan"))
    ac, cb = np.asarray(ac), np.asarray(cb)
    return dict(t_s=np.asarray(t), period_s=np.asarray(per), acf=ac, comb=cb,
                running=(ac >= RUNNING_ACF_MIN) & (np.nan_to_num(cb) >= LOW_COMB_MIN))


# ------------------------------------------------------------------ store
def read_detections(h5_path: str) -> List[Dict]:
    """Every prediction crop's id, frames and band from a ``.mad``."""
    import h5py
    from .fnt_mask_store import PRED_GROUP
    with h5py.File(h5_path, "r") as f:
        g = f.get(PRED_GROUP)
        if g is None:
            return []
        sr = float(f.attrs.get("sample_rate", 250_000))
        nfft = int(f.attrs.get("nfft", 1024))
        df = sr / nfft
        out = []
        for k in g:
            ds = g[k]
            fo, to = int(ds.attrs["f_off"]), int(ds.attrs["t_off"])
            hh, ww = ds.shape
            out.append(dict(blob_id=str(k), t0=to, t1=to + ww, f_lo=fo * df, f_hi=(fo + hh) * df))
    return out


def _infer_stamp(f) -> str:
    """The inference run a file's detections came from (model + time)."""
    a = f.attrs
    return f"{a.get('last_infer_model', '')}@{a.get('last_infer_at', '')}"


def write_wheel(h5_path: str, dets: List[Dict], feats: List[Dict],
                track: Optional[Dict] = None) -> None:
    """Replace ``/wheel`` in a ``.mad``: one row per detection (``blob_id``,
    ``squeak`` 0/1, ``comb``, ``low_comb``, ``period_s``, ``low_acf``,
    ``dur_beats``) and, when given, the running track (``track_t_s``,
    ``track_period_s``, ``track_acf``, ``track_comb``, ``track_running``). Attributes: ``wheel_version``, the
    thresholds, and ``infer_run`` (the inference run the ids belong to)."""
    import h5py
    with h5py.File(h5_path, "a") as f:
        if WHEEL_GROUP in f:
            del f[WHEEL_GROUP]
        g = f.create_group(WHEEL_GROUP)
        g.create_dataset("blob_id", data=np.array([d["blob_id"] for d in dets], dtype=h5py.string_dtype()))
        g.create_dataset("squeak", data=np.array([int(q["squeak"]) for q in feats], np.int8))
        for k in _FEATURES:
            g.create_dataset(k, data=np.array([q[k] for q in feats], np.float32))
        if track is not None:
            for k in ("t_s", "period_s", "acf", "comb"):
                g.create_dataset("track_" + k, data=np.asarray(track[k], np.float32))
            g.create_dataset("track_running", data=np.asarray(track["running"], np.int8))
        g.attrs.update(wheel_version=WHEEL_VERSION, comb_min=COMB_MIN, low_comb_min=LOW_COMB_MIN,
                       low_acf_min=LOW_ACF_MIN,
                       max_duration_beats=MAX_DURATION_BEATS, running_acf_min=RUNNING_ACF_MIN,
                       infer_run=_infer_stamp(f))


def wheel_status(h5_path: str) -> Dict:
    """``{done, version, n}``: ``done`` when ``/wheel`` is current, was made
    from the file's latest inference run, and covers exactly its crops."""
    import h5py
    from .fnt_mask_store import PRED_GROUP
    with h5py.File(h5_path, "r") as f:
        g = f.get(WHEEL_GROUP)
        preds = set(f[PRED_GROUP].keys()) if PRED_GROUP in f else set()
        if g is None:
            return dict(done=False, version=0, n=0)
        v = int(g.attrs.get("wheel_version", 0))
        ids = {s.decode() if isinstance(s, bytes) else str(s) for s in g["blob_id"][()]}
        same_run = str(g.attrs.get("infer_run", "")) == _infer_stamp(f)
        return dict(done=v >= WHEEL_VERSION and same_run and ids == preds, version=v, n=len(ids))


def read_wheel(h5_path: str) -> Dict[str, Dict]:
    """``{blob_id: {squeak, comb, low_comb, period_s, low_acf, dur_beats}}``."""
    import h5py
    with h5py.File(h5_path, "r") as f:
        g = f.get(WHEEL_GROUP)
        if g is None:
            return {}
        ids = [s.decode() if isinstance(s, bytes) else str(s) for s in g["blob_id"][()]]
        cols = {k: g[k][()] for k in ("squeak",) + _FEATURES}
    return {i: {k: (bool(v[j]) if k == "squeak" else float(v[j])) for k, v in cols.items()}
            for j, i in enumerate(ids)}


def run_recording(wav_path: str, h5_path: Optional[str] = None, overwrite: bool = False,
                  model: str = "") -> Dict:
    """Flag wheel squeaks among one recording's detections and store them.
    ``model``: only if the file's last inference used a model whose name
    starts with it. Returns ``{status, n, squeaks, running_s, duration_s}``."""
    import os
    import h5py
    from .fnt_mask_store import masks_sibling_path
    from .spectrogram import load_audio
    h5 = h5_path or masks_sibling_path(wav_path)
    if not os.path.isfile(h5):
        return dict(status="no_mad", n=0, squeaks=0)
    with h5py.File(h5, "r") as f:
        grid = {k: int(f.attrs[k]) for k in ("nperseg", "noverlap", "nfft") if k in f.attrs}
        last = str(f.attrs.get("last_infer_model", ""))
    if model and not last.startswith(model):
        return dict(status="other_model", n=0, squeaks=0)
    if not overwrite and wheel_status(h5)["done"]:
        return dict(status="done", n=0, squeaks=0)
    dets = read_detections(h5)
    x, sr = load_audio(wav_path)
    feats = score_detections(x, sr, dets, **grid)
    hop = grid.get("nperseg", 512) - grid.get("noverlap", 384)
    track = running_track(low_band_envelope_db(x, sr, hop), hop / sr)
    write_wheel(h5, dets, feats, track)
    return dict(status="ok", n=len(dets), squeaks=int(sum(q["squeak"] for q in feats)),
                running_s=float(track["running"].sum() * TRACK_STEP_S), duration_s=len(x) / sr)
