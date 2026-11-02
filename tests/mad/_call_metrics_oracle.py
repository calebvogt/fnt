"""Frozen copy of ``compute_call_metrics`` from before the per-frame refactor.

The refactor pulled the per-frame loops out into ``call_frame_features`` so a
3D trajectory view and the metric CSV share one computation. That is only safe
if the CSV numbers did not move, and the only way to show that is to keep the
old function verbatim and compare against it. Do not "fix" or modernise this
file — its value is that it is the old behaviour, exactly.
"""
from typing import Dict

import numpy as np

from fnt.usv.usv_detector.mad_inference import (
    _FREQ_JUMP_HZ, _TONALITY_HALF_BINS,
)


def legacy_compute_call_metrics(
    spec_db_cols: np.ndarray, mask: np.ndarray, f_low: int,
    df: float, dt: float, db_min: float, db_max: float,
) -> Dict:
    """Quantify one call.

    ``spec_db_cols`` is the **full-frequency** spectrogram (dB) for the call's
    time columns, shape ``(F_full, W)``; ``mask`` is the call's bounding-box
    pixel mask, shape ``(H, W)``; ``f_low`` is the global frequency-bin index of
    the mask's top row (so the band crop is ``spec_db_cols[f_low:f_low+H]``).
    dB is clipped to [db_min, db_max] so predictions (clipped spec) and
    hand-labels (raw dB) compute on the same scale. Per-frame spectral entropy
    and tonality use the full column (matching CAD's `dsp_detector`). Returns a
    dict keyed by :data:`CALL_METRIC_KEYS`; degenerate metrics are omitted.
    """
    m: Dict = {}
    if mask is None or mask.size == 0 or not mask.any():
        return m
    H, W = mask.shape
    full = np.clip(np.asarray(spec_db_cols, dtype=np.float64), db_min, db_max)
    F_full = full.shape[0]
    f_hi = f_low + H
    if f_hi > F_full or full.shape[1] != W:
        return m
    bb = full[f_low:f_hi, :]                  # call-band crop (H, W), dB
    Pbb = np.power(10.0, bb / 10.0)           # linear power in the band
    ys, xs = np.where(mask)
    vals_db = bb[mask]

    # --- power / energy (over the call's pixels) ---
    m['max_power_db'] = round(float(vals_db.max()), 2)
    m['mean_power_db'] = round(float(vals_db.mean()), 2)
    m['total_energy_db'] = round(float(10.0 * np.log10(Pbb[mask].sum() + 1e-12)), 2)
    peak_pix = int(np.argmax(vals_db))        # loudest call pixel → peak freq
    m['peak_freq_hz'] = round(float((f_low + ys[peak_pix]) * df), 2)

    # --- frequency contour: peak-power freq per masked time column ---
    cols, cfreq = [], []
    for t in range(W):
        rows_t = np.where(mask[:, t])[0]
        if rows_t.size == 0:
            continue
        peak_row = rows_t[int(np.argmax(bb[rows_t, t]))]
        cols.append(t)
        cfreq.append((f_low + peak_row) * df)
    if cfreq:
        cols_a = np.asarray(cols, dtype=np.float64)
        cf = np.asarray(cfreq, dtype=np.float64)
        m['start_freq_hz'] = round(float(cf[0]), 2)
        m['end_freq_hz'] = round(float(cf[-1]), 2)
        m['mean_freq_hz'] = round(float(cf.mean()), 2)
        m['freq_std_hz'] = round(float(cf.std()), 2)
        t_s = cols_a * dt
        if cf.size >= 2 and np.ptp(t_s) > 0:
            slope = float(np.polyfit(t_s, cf, 1)[0])
        else:
            slope = 0.0
        m['freq_slope_hz_per_s'] = round(slope, 2)
        dcf = np.abs(np.diff(cf))
        m['freq_excursion_hz'] = round(float(dcf.sum()), 2)
        m['num_freq_jumps'] = int((dcf > _FREQ_JUMP_HZ).sum())
        # sinuosity: contour path length / chord length, in (frame, bin) space
        fb = cf / df
        seg = np.hypot(np.diff(cols_a), np.diff(fb))
        chord = float(np.hypot(cols_a[-1] - cols_a[0], fb[-1] - fb[0]))
        m['sinuosity'] = round(float(seg.sum()) / chord, 3) if chord > 1e-6 else 1.0

    # --- frequency bandwidth from mask extent ---
    m['freq_bandwidth_hz'] = round(float((ys.max() - ys.min() + 1) * df), 2)

    # --- spectral centroid: power-weighted mean freq over the call's pixels ---
    Pm = np.where(mask, Pbb, 0.0)
    row_power = Pm.sum(axis=1)                 # power per band freq bin (masked)
    tot = float(row_power.sum())
    if tot > 0:
        freqs = (f_low + np.arange(H)) * df
        m['spectral_centroid_hz'] = round(float((freqs * row_power).sum() / tot), 2)

    # --- per-frame spectral entropy + tonality over the FULL column (CAD) ---
    Pfull = np.power(10.0, full / 10.0)
    max_ent = np.log2(F_full) if F_full > 1 else 1.0
    ton = np.zeros(W)
    ent = np.zeros(W)
    half = _TONALITY_HALF_BINS
    for t in range(W):
        col = Pfull[:, t]
        s = float(col.sum())
        if s <= 0:
            ent[t] = 1.0       # empty column → maximally "noisy"
            continue
        pk = int(np.argmax(col))
        lo, hi = max(0, pk - half), min(F_full, pk + half + 1)
        ton[t] = float(col[lo:hi].sum()) / s
        p = col / s
        p = p[p > 0]
        ent[t] = float(-(p * np.log2(p)).sum()) / max_ent
    m['tonality'] = round(float(ton.mean()), 4)
    m['spectral_entropy'] = round(float(ent.mean()), 4)

    # --- amplitude envelope over time (masked band energy per frame) ---
    col_energy = Pm.sum(axis=0)
    if W > 1:
        m['peak_time_frac'] = round(int(np.argmax(col_energy)) / float(W - 1), 3)
    env = col_energy[col_energy > 0]
    if env.size:
        emax, emin = float(env.max()), float(env.min())
        denom = emax + emin
        m['amplitude_modulation'] = round((emax - emin) / denom, 3) if denom > 0 else 0.0

    # --- morphology ---
    area = int(mask.sum())
    m['fill_ratio'] = round(area / float(H * W), 3) if H * W else 0.0
    m['aspect_ratio'] = round(W / float(H), 3) if H else 0.0

    # --- SNR: peak call power minus the local noise floor (CAD: max − floor).
    # Floor = median dB of the out-of-band rows at the call's time columns (the
    # background spectrum flanking the call), always available; fall back to the
    # off-mask pixels inside the bbox if the call spans the whole band. ---
    band = np.zeros(F_full, dtype=bool)
    band[f_low:f_hi] = True
    if (~band).any():
        noise = float(np.median(full[~band, :]))
    else:
        off = ~mask
        noise = float(np.median(bb[off])) if off.any() else db_min
    m['snr_db'] = round(float(m['max_power_db'] - noise), 2)
    return m
