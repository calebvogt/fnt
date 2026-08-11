"""Assessment of a controlled hemi-sync probe.

Run it on a recording folder::

    python -m fnt.musestudio.probe_analysis LocalData/Muse/<recording>

The report is ordered so that it fails fast. Data quality and the eyes-closed
positive control come first: if those don't pass there is no point reading the
intervention results, because a null (or a spurious positive) would be
uninterpretable.

Every contrast is reported with an effect size (Cohen's d over per-window
values) rather than a bare mean difference, and PLV is always shown next to
imaginary coherence so a shared-reference artifact can't masquerade as
interhemispheric coupling.
"""

import json
import os
import sys

import numpy as np

from fnt.musestudio.dsp import (
    BANDS, band_connectivity, band_powers, contact_quality,
    relative_band_powers, scalp_channels,
)
from fnt.musestudio.review import load_session, phase_intervals

WINDOW_SEC = 4.0
STEP_SEC = 2.0
PAIRS = [("frontal", "AF7", "AF8"), ("temporal", "TP9", "TP10")]


# ----------------------------------------------------------------- helpers
def _find(channels, tag):
    return next((c for c in channels if tag in str(c).upper()), None)


def _windows(n_samples, fs):
    win = int(WINDOW_SEC * fs)
    step = int(STEP_SEC * fs)
    return [(s, s + win) for s in range(0, max(0, n_samples - win) + 1, step)]


def _cohens_d(a, b):
    a = np.asarray([v for v in a if np.isfinite(v)], dtype=float)
    b = np.asarray([v for v in b if np.isfinite(v)], dtype=float)
    if len(a) < 2 or len(b) < 2:
        return float("nan")
    va, vb = np.var(a, ddof=1), np.var(b, ddof=1)
    pooled = np.sqrt(((len(a) - 1) * va + (len(b) - 1) * vb) /
                     (len(a) + len(b) - 2))
    if pooled <= 0:
        return float("nan")
    return float((np.mean(a) - np.mean(b)) / pooled)


def _d_label(d):
    if not np.isfinite(d):
        return "n/a"
    a = abs(d)
    size = ("negligible" if a < 0.2 else "small" if a < 0.5
            else "moderate" if a < 0.8 else "large")
    return f"{size} ({'+' if d > 0 else '-'})"


def _mean(values):
    vals = [v for v in values if np.isfinite(v)]
    return float(np.mean(vals)) if vals else float("nan")


# ----------------------------------------------------------------- analysis
def analyse(root, band="alpha"):
    data = load_session(root)
    if not data.has_eeg():
        return {"error": "no EEG data in this recording"}

    fs = data.fs
    channels = scalp_channels(data.eeg_channels)
    ts = data.eeg["lsl_timestamp"].to_numpy(dtype=float) - data.t0
    signals = {c: data.eeg[c].to_numpy(dtype=float) for c in channels}
    blocks = phase_intervals(data)
    lo, hi = BANDS[band]

    per_block = {}
    for label, t0, t1 in blocks:
        mask = (ts >= t0) & (ts < t1)
        idx = np.flatnonzero(mask)
        if len(idx) < int(WINDOW_SEC * fs):
            continue
        seg = {c: signals[c][idx] for c in channels}
        rows = {"alpha_rel": [], "theta_rel": [], "alpha_abs": [],
                "plv": [], "imag": [], "good": []}
        for a, b in _windows(len(idx), fs):
            chunk = {c: seg[c][a:b] for c in channels}
            good = all(contact_quality(v, fs) for v in chunk.values())
            rows["good"].append(1.0 if good else 0.0)
            if not good:
                continue
            rel, absol = [], []
            for v in chunk.values():
                ab = band_powers(v, fs)
                rel.append(relative_band_powers(ab))
                absol.append(ab)
            rows["alpha_rel"].append(np.mean([r["alpha"] for r in rel]))
            rows["theta_rel"].append(np.mean([r["theta"] for r in rel]))
            rows["alpha_abs"].append(np.mean([x["alpha"] for x in absol]))
            plvs, imags = [], []
            for _, left, right in PAIRS:
                lc, rc = _find(channels, left), _find(channels, right)
                if lc and rc:
                    p, i = band_connectivity(chunk[lc], chunk[rc], fs, (lo, hi))
                    plvs.append(p)
                    imags.append(i)
            rows["plv"].append(_mean(plvs))
            rows["imag"].append(_mean(imags))
        per_block[label] = {k: np.array(v, dtype=float) for k, v in rows.items()}
        per_block[label]["duration"] = t1 - t0

    return {"data": data, "blocks": per_block, "band": band,
            "channels": channels, "fs": fs}


def _pick(blocks, *needles):
    for name in blocks:
        low = name.lower()
        if all(n in low for n in needles):
            return name
    return None


def report(root, band="alpha"):
    out = analyse(root, band)
    if "error" in out:
        return f"ERROR: {out['error']}"
    data, blocks = out["data"], out["blocks"]
    lines = []
    add = lines.append

    add("=" * 72)
    add(f"HEMI-SYNC PROBE REPORT   ·   {os.path.basename(os.path.normpath(root))}")
    add("=" * 72)
    add(f"protocol : {data.config.get('protocol') or data.config.get('mode')}")
    add(f"duration : {data.duration / 60:.1f} min    "
        f"EEG {len(out['channels'])} ch @ {out['fs']:.0f} Hz    band: {band}")
    add("")

    # --- 1. data quality (fail fast) -----------------------------------
    add("1. DATA QUALITY")
    if not blocks:
        add("   !! no phase blocks found — cannot analyse")
        return "\n".join(lines)
    overall = []
    for name, b in blocks.items():
        frac = float(np.mean(b["good"])) if len(b["good"]) else 0.0
        overall.append(frac)
        flag = "ok" if frac >= 0.7 else "POOR"
        add(f"   {name:<20} {b['duration']:5.0f}s   clean windows {frac*100:5.1f}%  {flag}")
    usable = float(np.mean(overall))
    add(f"   overall clean: {usable*100:.1f}%")
    if usable < 0.5:
        add("   >> VERDICT: recording too noisy to interpret. Refit the headband.")
        return "\n".join(lines)
    add("")

    # --- 2. positive control -------------------------------------------
    add("2. POSITIVE CONTROL  (eyes closed should raise alpha vs eyes open)")
    eo = _pick(blocks, "eyes", "open")
    ec = _pick(blocks, "closed") or _pick(blocks, "rest")
    if eo and ec:
        d = _cohens_d(blocks[ec]["alpha_rel"], blocks[eo]["alpha_rel"])
        m_eo, m_ec = _mean(blocks[eo]["alpha_rel"]), _mean(blocks[ec]["alpha_rel"])
        add(f"   alpha(rel)  eyes-open {m_eo:.3f}  ->  eyes-closed {m_ec:.3f}")
        add(f"   Cohen's d = {d:+.2f}   {_d_label(d)}")
        if np.isfinite(d) and d > 0.5:
            add("   >> PASS — the pipeline detects a known real effect.")
        else:
            add("   >> FAIL — no reliable alpha rise on eye closure.")
            add("      Electrode contact, placement or analysis is suspect;")
            add("      treat everything below as uninterpretable.")
    else:
        add("   (no eyes-open / eyes-closed blocks in this protocol)")
    add("")

    # --- 3. primary contrast -------------------------------------------
    add("3. PRIMARY CONTRAST  (binaural beat vs matched control tone)")
    binaural = _pick(blocks, "binaural")
    control = _pick(blocks, "control")
    if binaural and control:
        for key, title in (("plv", "PLV        "),
                           ("imag", "imag-coh   "),
                           ("alpha_rel", "alpha (rel)")):
            d = _cohens_d(blocks[binaural][key], blocks[control][key])
            add(f"   {title}  control {_mean(blocks[control][key]):.3f}"
                f"  ->  binaural {_mean(blocks[binaural][key]):.3f}"
                f"   d={d:+.2f}  {_d_label(d)}")
        d_plv = _cohens_d(blocks[binaural]["plv"], blocks[control]["plv"])
        d_img = _cohens_d(blocks[binaural]["imag"], blocks[control]["imag"])
        add("")
        if np.isfinite(d_plv) and d_plv > 0.5 and np.isfinite(d_img) and d_img > 0.3:
            add("   >> Both PLV and imaginary coherence rose: consistent with")
            add("      genuine interhemispheric coupling, not just common mode.")
        elif np.isfinite(d_plv) and d_plv > 0.5:
            add("   >> PLV rose but imaginary coherence did not. That pattern is")
            add("      what a shared-reference / common-mode artifact looks like —")
            add("      treat this as weak evidence at best.")
        else:
            add("   >> No clear beat-specific effect at this dose.")
    else:
        add("   (protocol lacks a binaural/control pair)")
    add("")

    # --- 4. time-drift control ------------------------------------------
    add("4. TIME-DRIFT CONTROL  (early rest vs late rest, no stimulus in either)")
    rests = [n for n in blocks if "rest" in n.lower() or "closed" in n.lower()]
    if len(rests) >= 2:
        first, last = rests[0], rests[-1]
        d = _cohens_d(blocks[last]["plv"], blocks[first]["plv"])
        add(f"   PLV  {first} {_mean(blocks[first]['plv']):.3f}"
            f"  ->  {last} {_mean(blocks[last]['plv']):.3f}   d={d:+.2f}")
        if np.isfinite(d) and abs(d) > 0.5:
            add("   >> Rest itself drifted. Any 'effect' above may be time-on-task")
            add("      or arousal change rather than the stimulus.")
        else:
            add("   >> Rest was stable — good, contrasts above are more trustworthy.")
    else:
        add("   (need two rest blocks)")
    add("")

    # --- 5. drowsiness --------------------------------------------------
    add("5. DROWSINESS CHECK  (theta/alpha rising = sliding toward sleep)")
    ordered = list(blocks)
    for name in ordered:
        b = blocks[name]
        a, th = _mean(b["alpha_rel"]), _mean(b["theta_rel"])
        ratio = th / a if a > 0 else float("nan")
        add(f"   {name:<20} theta/alpha = {ratio:.2f}")
    add("")

    # --- 6. subjective --------------------------------------------------
    add("6. SUBJECTIVE")
    qpath = os.path.join(root, "questionnaire.json")
    if os.path.exists(qpath):
        with open(qpath, encoding="utf-8") as f:
            q = json.load(f)
        for k, v in q.get("ratings", {}).items():
            add(f"   {k:<12} {v}/10")
        if q.get("most_distinct_block"):
            add(f"   most distinct: {q['most_distinct_block']}")
        if q.get("notes"):
            add(f"   notes: {q['notes']}")
    else:
        add("   (no questionnaire.json — subjective context missing)")

    add("")
    add("=" * 72)
    return "\n".join(lines)


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        print(__doc__)
        return 1
    band = argv[1] if len(argv) > 1 else "alpha"
    print(report(argv[0], band))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
