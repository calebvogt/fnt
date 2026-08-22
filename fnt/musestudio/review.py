"""Offline review of a finished MuseStudio recording.

Loads a session folder and answers the question the protocol was designed to
ask: *did the measure move during stimulation, relative to baseline?*

Everything here is pure pandas/NumPy so it can be tested (and reused for batch
analysis) without a GUI.

All timelines are expressed in **seconds since the start of the recording**, so
signals, protocol phases and stimulus events share one axis even though they
were written by different subsystems.
"""

import json
import os
from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from fnt.musestudio.dsp import BAND_ORDER, BANDS, psd

SPEC_MAX_COLUMNS = 1500     # cap so long sessions stay responsive
WINDOW_SEC = 4.0


@dataclass
class SessionData:
    root: str
    name: str = ""
    config: dict = field(default_factory=dict)
    eeg: pd.DataFrame = None            # lsl_timestamp + one column per channel
    eeg_channels: list = field(default_factory=list)
    fs: float = 256.0
    synchrony: pd.DataFrame = None
    audio: pd.DataFrame = None
    events: pd.DataFrame = None
    t0: float = 0.0
    duration: float = 0.0
    notes: list = field(default_factory=list)   # non-fatal load problems

    def has_eeg(self):
        return self.eeg is not None and len(self.eeg) > 1


def _read_csv(path):
    try:
        if os.path.exists(path):
            df = pd.read_csv(path)
            return df if len(df) else None
    except Exception:
        return None
    return None


def _find_stream_csv(muse_dir, keyword):
    if not os.path.isdir(muse_dir):
        return None
    for fn in sorted(os.listdir(muse_dir)):
        if fn.lower().endswith(".csv") and keyword in fn.upper():
            return os.path.join(muse_dir, fn)
    return None


def load_session(root):
    """Load a ``*_FNT_MuseStudio_recording`` folder into a :class:`SessionData`."""
    data = SessionData(root=root, name=os.path.basename(os.path.normpath(root)))

    cfg_path = os.path.join(root, "recording_config.json")
    if os.path.exists(cfg_path):
        try:
            with open(cfg_path, encoding="utf-8") as f:
                data.config = json.load(f)
        except Exception:
            data.notes.append("recording_config.json could not be parsed")

    paths = data.config.get("paths", {})
    muse_dir = paths.get("muse") or os.path.join(root, "Data", "Muse")
    analysis_dir = paths.get("analysis") or os.path.join(root, "Data", "Analysis")
    audio_dir = paths.get("audio") or os.path.join(root, "Data", "Audio")
    events_dir = paths.get("events") or os.path.join(root, "Data", "Events")
    # Folders move if the recording is copied elsewhere — always prefer local.
    for name, fallback in (("muse", muse_dir), ("analysis", analysis_dir),
                           ("audio", audio_dir), ("events", events_dir)):
        local = os.path.join(root, "Data", name.capitalize())
        if os.path.isdir(local):
            if name == "muse":
                muse_dir = local
            elif name == "analysis":
                analysis_dir = local
            elif name == "audio":
                audio_dir = local
            else:
                events_dir = local

    eeg_path = _find_stream_csv(muse_dir, "EEG")
    if eeg_path:
        data.eeg = _read_csv(eeg_path)
        if data.eeg is not None:
            data.eeg_channels = [c for c in data.eeg.columns if c != "lsl_timestamp"]
    else:
        data.notes.append("no EEG CSV found in Data/Muse")

    data.synchrony = _read_csv(os.path.join(analysis_dir, "synchrony.csv"))
    data.audio = _read_csv(os.path.join(audio_dir, "audio_events.csv"))
    data.events = _read_csv(os.path.join(events_dir, "events.csv"))

    # Establish a common origin across every file.
    starts = []
    for df in (data.eeg, data.synchrony, data.audio, data.events):
        if df is not None and "lsl_timestamp" in df:
            starts.append(float(df["lsl_timestamp"].iloc[0]))
    if starts:
        data.t0 = min(starts)

    if data.has_eeg():
        ts = data.eeg["lsl_timestamp"].to_numpy(dtype=float)
        data.duration = float(ts[-1] - ts[0])
        dt = np.diff(ts)
        dt = dt[(dt > 0) & (dt < 1.0)]
        if len(dt):
            data.fs = float(1.0 / np.median(dt))
        cfg_rates = data.config.get("sample_rates") or {}
        for stream, rate in cfg_rates.items():
            if rate and "EEG" in str(stream).upper():
                data.fs = float(rate)
                break
    elif data.synchrony is not None:
        ts = data.synchrony["lsl_timestamp"].to_numpy(dtype=float)
        data.duration = float(ts[-1] - ts[0])
    return data


def phase_intervals(data):
    """``[(label, t_start, t_end)]`` in seconds from recording start."""
    if data.events is None or "kind" not in data.events:
        return []
    ev = data.events[data.events["kind"] == "phase"]
    if not len(ev):
        return []
    times = ev["lsl_timestamp"].to_numpy(dtype=float) - data.t0
    labels = ev["label"].astype(str).tolist()
    end = data.duration if data.duration > 0 else float(times[-1])
    out = []
    for i, label in enumerate(labels):
        stop = times[i + 1] if i + 1 < len(times) else max(end, times[i])
        out.append((label, float(times[i]), float(stop)))
    return out


def audio_intervals(data):
    """Spans where a tone was audible, as ``[(t_start, t_end, beat_hz)]``."""
    if data.audio is None or "event" not in data.audio:
        return []
    df = data.audio
    t = df["lsl_timestamp"].to_numpy(dtype=float) - data.t0
    events = df["event"].astype(str).tolist()
    beats = (df["beat_hz"].to_numpy(dtype=float)
             if "beat_hz" in df else np.zeros(len(df)))
    spans, start, beat = [], None, 0.0
    for i, e in enumerate(events):
        if e.startswith("play"):
            start, beat = t[i], beats[i]
        elif e.startswith(("stop", "fade_out", "record_stop")) and start is not None:
            spans.append((start, t[i], beat))
            start = None
    if start is not None:
        spans.append((start, data.duration, beat))
    return spans


def analyze_eeg(data, channel_index=0, window_sec=WINDOW_SEC,
                max_columns=SPEC_MAX_COLUMNS):
    """One sliding-window pass producing both the spectrogram and band series.

    Returns ``(times, freqs, spec_db, band_series)`` where ``spec_db`` is
    ``(n_freqs, n_times)`` and ``band_series`` maps band -> relative power.
    """
    empty = (np.array([]), np.array([]), None, {})
    if not data.has_eeg() or not data.eeg_channels:
        return empty
    idx = min(max(channel_index, 0), len(data.eeg_channels) - 1)
    x = data.eeg[data.eeg_channels[idx]].to_numpy(dtype=float)
    ts = data.eeg["lsl_timestamp"].to_numpy(dtype=float) - data.t0
    fs = data.fs
    win = int(window_sec * fs)
    if len(x) < win or win < 32:
        return empty

    n_possible = len(x) - win
    hop = max(int(fs * 0.5), int(np.ceil(n_possible / max_columns)))
    starts = np.arange(0, n_possible + 1, hop, dtype=int)

    cols, times, band_rows = [], [], []
    for s in starts:
        seg = x[s:s + win]
        f, p = psd(seg, fs)
        if len(f) == 0:
            continue
        mask = f <= 50.0
        f, p = f[mask], p[mask]
        cols.append(10.0 * np.log10(np.maximum(p, 1e-12)))
        times.append(ts[s + win // 2])
        total = 0.0
        row = {}
        for name, (lo, hi) in BANDS.items():
            m = (f >= lo) & (f < hi)
            val = float(np.trapezoid(p[m], f[m])) if m.any() else 0.0
            row[name] = val
            total += val
        band_rows.append({k: (v / total if total > 0 else 0.0)
                          for k, v in row.items()})

    if not cols:
        return empty
    spec = np.array(cols).T
    times = np.array(times, dtype=float)
    series = {b: np.array([r[b] for r in band_rows], dtype=float)
              for b in BAND_ORDER}
    return times, f, spec, series


def _mean_in(times, values, t0, t1):
    if times is None or len(times) == 0:
        return float("nan")
    m = (times >= t0) & (times < t1)
    return float(np.nanmean(values[m])) if m.any() else float("nan")


def summarize_phases(data, band_times=None, band_series=None):
    """Per-phase means — the table that answers 'did it move?'.

    Each row carries mean PLV, mean relative alpha, mean drift and the fraction
    of samples with good electrode contact (so a phase built on bad signal is
    visibly untrustworthy rather than silently averaged in).
    """
    phases = phase_intervals(data)
    if not phases:
        if data.duration > 0:
            phases = [("Whole recording", 0.0, data.duration)]
        else:
            return []

    sync = data.synchrony
    if sync is not None and "lsl_timestamp" in sync:
        s_t = sync["lsl_timestamp"].to_numpy(dtype=float) - data.t0
    else:
        s_t = None

    rows = []
    for label, t0, t1 in phases:
        row = {"phase": label, "start": t0, "duration": max(t1 - t0, 0.0)}
        if s_t is not None:
            for key, col in (("plv", "plv_combined"), ("level", "level"),
                             ("drift", "drift_hz")):
                row[key] = (_mean_in(s_t, sync[col].to_numpy(dtype=float), t0, t1)
                            if col in sync else float("nan"))
            row["contact"] = (_mean_in(s_t, sync["contact_ok"].to_numpy(dtype=float),
                                       t0, t1) if "contact_ok" in sync else float("nan"))
        else:
            row.update({"plv": float("nan"), "level": float("nan"),
                        "drift": float("nan"), "contact": float("nan")})
        row["alpha"] = (_mean_in(band_times, band_series["alpha"], t0, t1)
                        if band_series and "alpha" in band_series else float("nan"))
        rows.append(row)
    return rows


def compare_to_baseline(rows, baseline_hint="baseline"):
    """Deltas of each phase against the baseline phase, if one is present."""
    if not rows:
        return None, []
    base = next((r for r in rows if baseline_hint in r["phase"].lower()), None)
    if base is None:
        return None, []
    deltas = []
    for r in rows:
        if r is base:
            continue
        deltas.append({
            "phase": r["phase"],
            "d_plv": r.get("plv", float("nan")) - base.get("plv", float("nan")),
            "d_alpha": r.get("alpha", float("nan")) - base.get("alpha", float("nan")),
        })
    return base, deltas
