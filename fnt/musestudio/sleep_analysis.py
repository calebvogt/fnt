"""Overnight recording report.

    python -m fnt.musestudio.sleep_analysis LocalData/Muse/<recording>

Deliberately *not* a sleep stager. Clinical staging scores 30-second epochs
against EEG **plus** EOG and chin EMG, and Muse has none of the latter two.
What four frontal/temporal electrodes genuinely support is the layer beneath
staging: how the spectral content of the night evolved, when you moved, and —
first and most important — how much of the night was actually recorded.

So the report leads with coverage. An overnight recording that lost three hours
to a flat battery looks superficially fine in a plot; the numbers here make
that impossible to miss.
"""

import os
import sys

import numpy as np
import pandas as pd

from fnt.musestudio.dsp import (
    BAND_ORDER, band_powers, contact_quality, relative_band_powers,
    scalp_channels,
)
from fnt.musestudio.review import load_session
from fnt.musestudio.sleep import fmt_duration

EPOCH_SEC = 30.0        # the conventional sleep epoch


def _find_stream(root, keyword):
    muse = os.path.join(root, "Data", "Muse")
    if not os.path.isdir(muse):
        return None
    for fn in sorted(os.listdir(muse)):
        if fn.lower().endswith(".csv") and keyword in fn.upper():
            return os.path.join(muse, fn)
    return None


def epoch_metrics(data, epoch_sec=EPOCH_SEC):
    """Per-epoch band powers and contact fraction across the night."""
    if not data.has_eeg():
        return None
    channels = scalp_channels(data.eeg_channels)
    if not channels:
        return None
    fs = data.fs
    ts = data.eeg["lsl_timestamp"].to_numpy(dtype=float) - data.t0
    sig = {c: data.eeg[c].to_numpy(dtype=float) for c in channels}
    n = int(epoch_sec * fs)
    rows = []
    for start in range(0, len(ts) - n + 1, n):
        stop = start + n
        chunk = {c: sig[c][start:stop] for c in channels}
        good = [contact_quality(v, fs) for v in chunk.values()]
        row = {"t": float(ts[start]), "contact": float(np.mean(good))}
        if any(good):
            rel = []
            for c, v in chunk.items():
                if contact_quality(v, fs):
                    rel.append(relative_band_powers(band_powers(v, fs)))
            for b in BAND_ORDER:
                row[b] = float(np.mean([r[b] for r in rel]))
        else:
            for b in BAND_ORDER:
                row[b] = float("nan")
        rows.append(row)
    return pd.DataFrame(rows)


def movement_series(root, t0, epoch_sec=EPOCH_SEC):
    """Per-epoch accelerometer variability — a proxy for gross body movement."""
    path = _find_stream(root, "ACCGYRO") or _find_stream(root, "ACC")
    if path is None:
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    acc_cols = [c for c in df.columns if "ACC" in str(c).upper()]
    if not acc_cols or "lsl_timestamp" not in df:
        return None
    ts = df["lsl_timestamp"].to_numpy(dtype=float) - t0
    mag = np.linalg.norm(df[acc_cols].to_numpy(dtype=float), axis=1)
    rows = []
    end = ts[-1] if len(ts) else 0.0
    edges = np.arange(0, end, epoch_sec)
    for a in edges:
        m = (ts >= a) & (ts < a + epoch_sec)
        rows.append({"t": float(a),
                     "motion": float(np.std(mag[m])) if m.any() else float("nan")})
    return pd.DataFrame(rows)


def battery_series(root, t0):
    path = _find_stream(root, "BATTERY")
    if path is None:
        return None
    try:
        df = pd.read_csv(path)
    except Exception:
        return None
    value_cols = [c for c in df.columns if c != "lsl_timestamp"]
    if not value_cols:
        return None
    vals = df[value_cols[0]].to_numpy(dtype=float)
    if np.nanmax(vals) <= 1.0:
        vals = vals * 100.0
    return pd.DataFrame({"t": df["lsl_timestamp"].to_numpy(dtype=float) - t0,
                         "pct": vals})


def report(root):
    data = load_session(root)
    lines = []
    add = lines.append
    add("=" * 72)
    add(f"OVERNIGHT REPORT   ·   {os.path.basename(os.path.normpath(root))}")
    add("=" * 72)

    if not data.has_eeg():
        add("No EEG data in this recording — nothing to report.")
        return "\n".join(lines)

    subj = (data.config.get("subject") or {})
    if subj.get("id"):
        label = subj["id"]
        if subj.get("handedness") and subj["handedness"] != "unspecified":
            label += f" ({subj['handedness']}-handed)"
        if subj.get("session_label"):
            label += f" · {subj['session_label']}"
        add(f"subject  : {label}")
    dur = data.duration
    add(f"recorded : {fmt_duration(dur)}   ({data.fs:.0f} Hz, "
        f"{len(scalp_channels(data.eeg_channels))} scalp channels)")

    # --- 1. coverage: did we actually capture the night? -----------------
    add("")
    add("1. COVERAGE")
    ts = data.eeg["lsl_timestamp"].to_numpy(dtype=float)
    gaps = np.diff(ts)
    expected = 1.0 / data.fs
    lost = float(np.sum(gaps[gaps > expected * 5]))
    n_gaps = int(np.sum(gaps > expected * 5))
    add(f"   wall-clock span   {fmt_duration(dur)}")
    add(f"   gaps              {n_gaps}  totalling {fmt_duration(lost)}")
    coverage = 100.0 * (1 - lost / dur) if dur > 0 else 0.0
    add(f"   coverage          {coverage:.1f}% of the span has samples")
    if n_gaps and lost > 300:
        add("   >> Significant data loss. Most likely the headband battery ran")
        add("      out or it slipped out of Bluetooth range.")

    bat = battery_series(root, data.t0)
    if bat is not None and len(bat) > 1:
        add(f"   battery           {bat['pct'].iloc[0]:.0f}% → "
            f"{bat['pct'].iloc[-1]:.0f}%")
        drop = bat["pct"].iloc[0] - bat["pct"].iloc[-1]
        hours = dur / 3600.0
        if drop > 0 and hours > 0.5:
            rate = drop / hours
            add(f"   drain             {rate:.1f}%/hour → "
                f"~{100.0 / rate:.1f}h from full")

    # --- 2. signal quality over the night ---------------------------------
    epochs = epoch_metrics(data)
    add("")
    add("2. SIGNAL QUALITY")
    if epochs is None or epochs.empty:
        add("   (not enough data to epoch)")
        return "\n".join(lines)
    usable = float((epochs["contact"] > 0.5).mean())
    add(f"   epochs            {len(epochs)} × {EPOCH_SEC:.0f}s")
    add(f"   usable epochs     {usable * 100:.1f}%")
    if usable < 0.5:
        add("   >> Under half the night has usable contact — the headband")
        add("      probably shifted. Spectral results below are unreliable.")

    # --- 3. how the night's spectrum evolved -------------------------------
    add("")
    add("3. SPECTRAL COURSE  (relative power per band, by third of the night)")
    # Split by row index rather than np.array_split on the frame itself, which
    # routes through a deprecated pandas path.
    bounds = np.linspace(0, len(epochs), 4).astype(int)
    thirds = [epochs.iloc[a:b] for a, b in zip(bounds, bounds[1:])]
    header = "   " + "third".ljust(8) + "".join(b.rjust(9) for b in BAND_ORDER)
    add(header + "   contact")
    for i, part in enumerate(thirds, 1):
        if part.empty:
            continue
        vals = "".join(f"{np.nanmean(part[b]):9.3f}" for b in BAND_ORDER)
        add(f"   {i}/3".ljust(11) + vals +
            f"   {np.nanmean(part['contact']) * 100:5.0f}%")
    add("")
    add("   Deep sleep shows as high delta; REM and wake look flatter with")
    add("   relatively more beta. Muse cannot separate REM from wake reliably")
    add("   without EOG/EMG, so read these as trends, not stages.")

    # --- 4. movement --------------------------------------------------------
    move = movement_series(root, data.t0)
    add("")
    add("4. MOVEMENT")
    if move is None or move["motion"].dropna().empty:
        add("   (no accelerometer data)")
    else:
        m = move["motion"].to_numpy(dtype=float)
        thresh = np.nanpercentile(m, 90)
        restless = float(np.nanmean(m > thresh))
        add(f"   epochs with high movement: {restless * 100:.1f}%")
        quiet = np.where(m <= np.nanpercentile(m, 50))[0]
        if len(quiet):
            add(f"   quietest stretch begins ~{fmt_duration(move['t'].iloc[quiet[0]])} in")

    add("")
    add("=" * 72)
    return "\n".join(lines)


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(__doc__)
        return 1
    print(report(argv[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
