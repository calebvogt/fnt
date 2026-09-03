"""Turn a Flight Calibration recording into the flight control law's constants.

    python -m fnt.musestudio.flight.calibration <recording_dir>

Every number in :mod:`fnt.musestudio.flight.pipeline` currently derives from a
single early session in which both ear electrodes failed within two minutes.
This module replaces those with per-subject measurements, and -- more
importantly -- answers the question that decides what Flight Mode actually is:

    Can the pilot voluntarily push alpha ABOVE their own eyes-closed rest?

If yes, Flight Mode is genuine continuous neurofeedback and the craft can be
flown. If no, then closing the eyes is a one-shot switch, the craft climbs once
and saturates, and the honest response is to redesign around a single discrete
transition rather than to dress a switch up as a flight yoke. The report says
which, in those terms, rather than leaving it to be inferred from a table.
"""

import glob
import json
import os
import sys

import numpy as np
import pandas as pd
from scipy.signal import butter, sosfiltfilt, welch

from fnt.musestudio.flight.pipeline import (
    ARTIFACT_BAND, CONTROL_BAND, EPOCH_MAX_UV, EMG_RATIO_MAX, TOTAL_BAND,
    UPDATE_HZ, WINDOW_SEC,
)

CH = ["TP9", "AF7", "AF8", "TP10"]

# Phase-label -> role. Labels come from protocol._flight_calibration; the
# fallbacks let this also read the older probe sessions, which share the first
# two blocks even though they were recorded for a different purpose.
ROLES = {
    "Eyes open": "baseline",
    "Eyes closed rest": "rest",
    "Deepen": "deepen",
    "Alert": "alert",
    "Artifact: eye movement": "art_eye",
    "Artifact: blinks": "art_blink",
    "Artifact: jaw clench": "art_clench",
    "Artifact: head motion": "art_motion",
}


def _audio_context(root):
    """What gear this session was recorded on, from recording_config.json."""
    try:
        with open(os.path.join(root, "recording_config.json"), encoding="utf-8") as f:
            return (json.load(f) or {}).get("audio") or {}
    except Exception:  # noqa: BLE001
        return {}


def _load(root):
    eeg_files = glob.glob(os.path.join(root, "Data", "Muse", "Muse-EEG*.csv"))
    if not eeg_files:
        raise SystemExit(f"no Muse-EEG CSV under {root}/Data/Muse")
    eeg = pd.read_csv(eeg_files[0])
    ev_path = os.path.join(root, "Data", "Events", "events.csv")
    ev = pd.read_csv(ev_path) if os.path.exists(ev_path) else pd.DataFrame()
    return eeg, ev


def _windows(eeg, ev):
    """Slide the analysis window and compute per-electrode features."""
    t0 = eeg.lsl_timestamp.iloc[0]
    span = eeg.lsl_timestamp.iloc[-1] - t0
    fs = len(eeg) / span if span > 0 else 256.0
    sos = butter(4, list(TOTAL_BAND), btype="band", fs=fs, output="sos")
    W = int(WINDOW_SEC * fs)
    HOP = max(1, int(fs / UPDATE_HZ))

    marks = [(r.lsl_timestamp - t0, r.label) for r in ev.itertuples()
             if getattr(r, "kind", "") == "phase"] if len(ev) else []

    def phase_at(t):
        lab = "(pre)"
        for pt, pl in marks:
            if t >= pt:
                lab = pl
        return lab

    cols = {c: f"EEG_{c}" for c in CH if f"EEG_{c}" in eeg.columns}
    arrs = {c: eeg[v].to_numpy(float) for c, v in cols.items()}
    rows = []
    for k in range((len(eeg) - W) // HOP):
        a = k * HOP
        b = a + W
        t = b / fs
        ph = phase_at(t)
        for c, arr in arrs.items():
            x = arr[a:b]
            x = x - x.mean()
            y = sosfiltfilt(sos, x)
            amp = float(np.percentile(np.abs(y), 95))
            f, pxx = welch(x, fs=fs, nperseg=int(fs))

            def bp(lo, hi):
                m = (f >= lo) & (f < hi)
                return float(np.trapezoid(pxx[m], f[m])) if m.sum() > 1 else 0.0

            tot = bp(*TOTAL_BAND)
            rows.append(dict(
                t=t, ch=c, phase=ph, role=ROLES.get(ph, ""),
                amp=amp,
                alpha_rel=bp(*CONTROL_BAND) / tot if tot > 0 else 0.0,
                emg=bp(*ARTIFACT_BAND) / tot if tot > 0 else 0.0,
            ))
    d = pd.DataFrame(rows)
    d["clean"] = (d.amp <= EPOCH_MAX_UV) & (d.emg <= EMG_RATIO_MAX)
    return d, fs


def _cohen_d(a, b):
    if len(a) < 5 or len(b) < 5:
        return float("nan")
    pooled = np.sqrt((np.var(a, ddof=1) + np.var(b, ddof=1)) / 2)
    return float((np.mean(a) - np.mean(b)) / pooled) if pooled > 0 else float("nan")


def _hdr(title):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)


def report(root):
    eeg, ev = _load(root)
    d, fs = _windows(eeg, ev)
    present = set(d.role) - {""}

    audio = _audio_context(root)
    print(f"\nFlight calibration report — {os.path.basename(root.rstrip('/'))}")
    if audio:
        print(f"Audio: {audio.get('device','?')} ({audio.get('transport','?')}) — "
              f"tier={audio.get('tier','?')}, stereo={audio.get('stereo_separation')}, "
              f"latency~{audio.get('latency_ms','?')} ms")
        if audio.get("tier") == "speakers":
            print("  NOTE: recorded on speakers. The eyes-closed blocks were open to "
                  "room noise,\n        and audio could carry no left/right "
                  "information. Do not pool these\n        blocks with a headphone "
                  "session without saying so.")
        elif audio.get("tier") == "headphones" and audio.get("latency_ms", 0) >= 150:
            print(f"  NOTE: Bluetooth output added ~{audio['latency_ms']} ms one-way "
                  "latency. Irrelevant to\n        these measurements (no closed loop "
                  "here), but it comes straight out of\n        the budget in Flight "
                  "Mode.")
    else:
        print("Audio: not recorded (session predates audio-capability logging).")
    print(f"EEG {len(eeg)} samples @ {fs:.1f} Hz "
          f"({(eeg.lsl_timestamp.iloc[-1]-eeg.lsl_timestamp.iloc[0])/60:.1f} min), "
          f"{len(d)//max(len(set(d.ch)),1)} windows/electrode")

    # ---------------------------------------------------------------- 1. yield
    _hdr("1. SIGNAL QUALITY — clean-window yield per electrode per block")
    piv = d.pivot_table(index="ch", columns="phase", values="clean", aggfunc="mean")
    print((100 * piv).round(0).astype("Int64").to_string())
    print("\n(% of windows passing both gates: amp <= "
          f"{EPOCH_MAX_UV:.0f} uV and EMG ratio <= {EMG_RATIO_MAX})")
    dead = [c for c in sorted(set(d.ch)) if d[d.ch == c].clean.mean() < 0.10]
    if dead:
        print(f"\n  UNUSABLE THIS SESSION: {', '.join(dead)} "
              "— under 10% of windows survived gating.")
    best = d.groupby("ch").clean.mean().sort_values(ascending=False)
    print(f"  Best electrode: {best.index[0]} ({100*best.iloc[0]:.0f}% clean)")

    # ------------------------------------------------------------- 2. baseline
    _hdr("2. BASELINE — can a usable eyes-open reference be built, and how fast?")
    base = {}
    if "baseline" in present:
        bl = d[(d.role == "baseline") & d.clean]
        need = 100      # pipeline's ControlConfig.baseline_sec * UPDATE_HZ
        print(f"{'electrode':<12}{'clean n':>9}{'need':>7}{'center':>10}{'scale':>10}   status")
        for c in sorted(set(d.ch)):
            s = bl[bl.ch == c].alpha_rel
            if len(s) >= 10:
                m = float(np.median(s))
                scale = max(float(np.median(np.abs(s - m))) * 1.4826, 0.05 * abs(m), 1e-6)
                base[c] = (m, scale)
                ok = "OK" if len(s) >= need else "SHORT — preflight must run longer"
                print(f"{c:<12}{len(s):>9}{need:>7}{m:>10.4f}{scale:>10.4f}   {ok}")
            else:
                print(f"{c:<12}{len(s):>9}{need:>7}{'--':>10}{'--':>10}   too few clean windows")
        dur = d[d.role == "baseline"].t.max() - d[d.role == "baseline"].t.min()
        rate = len(bl) / max(dur, 1e-6) / max(len(set(d.ch)), 1)
        if rate > 0:
            print(f"\n  Clean windows per electrode per second: {rate:.2f}")
            print(f"  => preflight needs ~{need/max(rate,1e-6):.0f} s to build a baseline "
                  f"on the best electrode.")
    else:
        print("  No eyes-open block in this recording — cannot build a baseline.")

    if not base:
        print("\nNo baseline: the remaining sections need one. Stopping here.")
        return

    d["z"] = [((r.alpha_rel - base[r.ch][0]) / base[r.ch][1]) if r.ch in base else np.nan
              for r in d.itertuples()]

    # --------------------------------------------------- 3. the decisive block
    _hdr("3. THE DECISIVE QUESTION — is continuous control possible?")
    g = d[d.clean & d.z.notna()]
    rest = g[g.role == "rest"].z
    deep = g[g.role == "deepen"].z
    alert = g[g.role == "alert"].z

    def line(name, s):
        if len(s) < 5:
            print(f"  {name:<22} — not present")
            return
        print(f"  {name:<22} n={len(s):<5} median z={s.median():+6.2f}  "
              f"mean={s.mean():+6.2f}  IQR {s.quantile(.25):+.2f}..{s.quantile(.75):+.2f}")

    line("eyes-open (baseline)", g[g.role == "baseline"].z)
    line("eyes-closed rest", rest)
    line("deepen", deep)
    line("alert (suppress)", alert)

    verdict = []
    if len(deep) >= 5 and len(rest) >= 5:
        dd = _cohen_d(deep, rest)
        print(f"\n  DEEPEN vs eyes-closed REST:  Cohen's d = {dd:+.2f}")
        if dd >= 0.5:
            verdict.append(
                "CONTINUOUS CONTROL IS VIABLE. The pilot raised alpha above their own\n"
                "  eyes-closed rest, so climb can be commanded voluntarily in flight and\n"
                "  the craft is genuinely flown rather than merely switched on.")
        elif dd >= 0.2:
            verdict.append(
                "MARGINAL. There is a push above eyes-closed rest but it is small; expect\n"
                "  a sluggish craft and consider a longer window or per-subject training\n"
                "  before treating altitude as a controlled variable.")
        else:
            verdict.append(
                "CONTINUOUS CONTROL IS NOT SUPPORTED BY THIS SESSION. The pilot could not\n"
                "  raise alpha above their own eyes-closed rest, so eye closure is acting\n"
                "  as a one-shot switch. The honest redesign is a single discrete\n"
                "  transition (launch on eye closure, then autopilot) rather than a\n"
                "  continuously flown altitude axis.")
    if len(alert) >= 5 and len(rest) >= 5:
        da = _cohen_d(alert, rest)
        print(f"  ALERT  vs eyes-closed REST:  Cohen's d = {da:+.2f}"
              + ("   (suppression works — descent can be commanded)"
                 if da <= -0.5 else "   (weak — descent must rely on gravity)"))

    # -------------------------------------------------- 4. artifact rejection
    _hdr("4. PURITY — can any artifact actually RAISE the control signal?")
    print("  The question is not whether an artifact passes the gate — plenty of")
    print("  clean-looking windows do. It is whether it moves the control signal UP,")
    print("  because only that lets a pilot cheat. An artifact that passes the gate")
    print("  but drives z DOWN can stop the craft; it can never fly it.\n")
    arts = [("art_eye", "eye movement", "baseline"),
            ("art_blink", "blinks", "rest"),
            ("art_clench", "jaw clench", "rest"),
            ("art_motion", "head motion", "rest")]
    fused = (g[g.clean].groupby("t").z.mean().rename("z").to_frame()
             .join(g.groupby("t").role.first()))
    refs = {"baseline": fused[fused.role == "baseline"].z,
            "rest": fused[fused.role == "rest"].z}
    if not any(r in present for r, _, _ in arts):
        print("  No artifact blocks in this recording — rejection is unverified.")
    else:
        print(f"{'artifact':<16}{'n':>5}{'median z':>10}{'vs its eye-state':>18}   verdict")
        cheats = []
        for role, label, refname in arts:
            s_ = fused[fused.role == role].z
            ref = refs.get(refname)
            if len(s_) < 5 or ref is None or len(ref) < 5:
                continue
            delta = float(s_.median() - ref.median())
            # Compared against the SAME eye state. Blinking happens with the eyes
            # shut, where alpha is high anyway, so comparing it to the eyes-open
            # baseline would credit the artifact with the Berger effect.
            bad = delta > 0.5
            if bad:
                cheats.append(label)
            print(f"{label:<16}{len(s_):>5}{s_.median():>10.2f}{delta:>+18.2f}   "
                  + ("COULD CHEAT — tighten the gates" if bad
                     else "cannot raise thrust — safe"))
        if cheats:
            print(f"\n  ACTION REQUIRED: {', '.join(cheats)} raised the control signal.")
        else:
            print("\n  PURITY HOLDS: every artifact left the control signal at or below")
            print("  its own eye-state reference. Muscle and movement can stop the")
            print("  craft, which is intended, but cannot fly it.")

    # ------------------------------------------------- 5. recommended constants
    _hdr("5. RECOMMENDED CONSTANTS for ControlConfig")
    if len(rest) >= 20:
        # Dead zone should sit above the eyes-open noise floor so rest is quiet,
        # and below the flight state's median so the craft can actually lift.
        open_z = g[g.role == "baseline"].z
        dz = float(np.nanpercentile(open_z, 90)) if len(open_z) >= 20 else 0.8
        flight = deep if len(deep) >= 20 else rest
        fs_z = float(np.nanpercentile(flight, 90))
        dz = float(np.clip(dz, 0.3, 2.0))
        fs_z = float(np.clip(max(fs_z, dz + 1.0), dz + 1.0, 8.0))
        print(f"  dead_zone      = {dz:.2f}    (90th pct of eyes-open z — rest stays quiet)")
        print(f"  full_scale_z   = {fs_z:.2f}    (90th pct of flight-state z — reachable, "
              "not saturated)")
        hover = dz + 0.2 * (fs_z - dz)
        print(f"\n  break-even hover lands at z = {hover:+.2f}")
        med = float(flight.median())
        print(f"  flight-state median z = {med:+.2f}  => "
              + ("climbs on average — good" if med > hover else
                 "SINKS on average; lower dead_zone or reduce CraftConfig.gravity"))
    for v in verdict:
        print(f"\n  VERDICT: {v}")
    print()


def main():
    if len(sys.argv) < 2:
        raise SystemExit(__doc__)
    report(sys.argv[1])


if __name__ == "__main__":
    main()
