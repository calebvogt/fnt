"""Running-wheel squeaks are flagged from the audio alone; calls are kept.

A synthetic recording: low-register thumps at 7.3 Hz (a running wheel), a
narrow 107 kHz squeak a fixed time after every thump, a frequency-modulated
call train at its own irregular rhythm, and one long call spanning several
beats. Only the squeaks should be flagged.

Runs under pytest, or directly.
"""
import os
import tempfile

import numpy as np

from fnt.usv.usv_detector import fnt_mask_store as MS
from fnt.usv.usv_detector import wheel_noise as WN

SR, NPERSEG, NOVERLAP, NFFT = 250_000, 512, 384, 1024
HOP = NPERSEG - NOVERLAP
BEAT = 0.137
DUR = 12.0


def _tone(n, f0, f1=None):
    t = np.arange(n) / SR
    f1 = f0 if f1 is None else f1
    ph = 2 * np.pi * (f0 * t + (f1 - f0) * t ** 2 / (2 * t[-1]))
    return np.sin(ph) * np.hanning(n)


def _recording(seed=0):
    rng = np.random.default_rng(seed)
    x = 0.002 * rng.standard_normal(int(DUR * SR))
    squeaks, calls = [], []
    for k, t in enumerate(np.arange(0.2, DUR - 0.2, BEAT)):
        i = int(t * SR)
        thump = rng.standard_normal(int(0.012 * SR)) * np.hanning(int(0.012 * SR))
        from scipy.signal import butter, sosfilt
        x[i:i + len(thump)] += 0.2 * sosfilt(butter(4, [3e3, 10e3], "bandpass", fs=SR, output="sos"), thump)
        if 2.0 < t < 6.0:
            j = i + int(0.030 * SR)
            n = int(0.008 * SR)
            x[j:j + n] += 0.05 * _tone(n, 107e3)
            squeaks.append((j, j + n, 105e3, 109e3))
    # an FM call train with its own rhythm (intervals 90-260 ms), 6-10 s
    t = 6.3
    while t < 9.8:
        i, n = int(t * SR), int(0.040 * SR)
        x[i:i + n] += 0.05 * _tone(n, 60e3, 80e3)
        calls.append((i, i + n, 58e3, 82e3))
        t += rng.uniform(0.09, 0.26)
    # one long call over several beats
    i, n = int(10.2 * SR), int(0.45 * SR)
    x[i:i + n] += 0.05 * _tone(n, 70e3, 75e3)
    calls.append((i, i + n, 68e3, 77e3))
    return x.astype(np.float32), squeaks, calls


def _dets(spans):
    return [dict(t0=a // HOP, t1=b // HOP, f_lo=lo, f_hi=hi) for a, b, lo, hi in spans]


def test_beat_period_prefers_the_fundamental():
    hop_s = HOP / SR
    env = np.zeros(int(4.0 / hop_s))
    env[(np.arange(0, 4.0, BEAT) / hop_s).astype(int)] = 1.0
    p, r = WN.beat_period(WN._smooth(env), hop_s)
    assert abs(p * hop_s - BEAT) < 0.003
    assert r > 0.5


def test_squeaks_are_flagged_and_calls_kept():
    x, squeaks, calls = _recording()
    fs = WN.score_detections(x, SR, _dets(squeaks))
    fc = WN.score_detections(x, SR, _dets(calls))
    assert np.mean([q["squeak"] for q in fs]) > 0.9
    assert not any(q["squeak"] for q in fc)
    assert abs(np.median([q["period_s"] for q in fs]) - BEAT) < 0.005


def test_no_wheel_no_flags():
    x, squeaks, _ = _recording()
    # same squeak train, no thumps: nothing in the low register to lock to
    from scipy.signal import butter, sosfiltfilt
    y = sosfiltfilt(butter(6, 20e3, "highpass", fs=SR, output="sos"), x).astype(np.float32)
    y += 0.002 * np.random.default_rng(1).standard_normal(len(y)).astype(np.float32)
    fs = WN.score_detections(y, SR, _dets(squeaks))
    assert not any(q["squeak"] for q in fs)


def test_running_track_marks_the_wheel():
    x, _, _ = _recording()
    tr = WN.running_track(WN.low_band_envelope_db(x, SR, HOP), HOP / SR)
    assert tr["running"].mean() > 0.9
    assert abs(np.median(tr["period_s"]) - BEAT) < 0.005


def test_run_recording_writes_and_skips_when_done():
    import soundfile as sf
    x, squeaks, calls = _recording()
    d = tempfile.mkdtemp(prefix="mad_wheel_")
    wav = os.path.join(d, "rec.wav")
    sf.write(wav, x, SR, subtype="PCM_16")
    h5 = MS.masks_sibling_path(wav)
    MS.set_grid_attrs(h5, sample_rate=SR, nperseg=NPERSEG, noverlap=NOVERLAP,
                      nfft=NFFT, n_freq_bins=513, n_time_frames=len(x) // HOP)
    crops = []
    for k, (a, b, lo, hi) in enumerate(squeaks + calls):
        fo, fh = int(lo / (SR / NFFT)), int(hi / (SR / NFFT))
        crops.append({"blob_id": k, "mask": np.ones((fh - fo, (b - a) // HOP), np.uint8),
                      "f_off": fo, "t_off": a // HOP, "score": 0.9})
    MS.write_pred_masks(h5, crops)
    r = WN.run_recording(wav)
    assert r["status"] == "ok" and r["n"] == len(crops)
    got = WN.read_wheel(h5)
    sq = sum(got[str(k)]["squeak"] for k in range(len(squeaks)))
    assert sq >= 0.9 * len(squeaks)
    assert not any(got[str(k)]["squeak"] for k in range(len(squeaks), len(crops)))
    assert WN.wheel_status(h5)["done"]
    assert WN.run_recording(wav)["status"] == "done"
    # a new inference run is stale even if it reuses the same blob ids
    MS.set_infer_run_attrs(h5, last_infer_model="m2", last_infer_at="2026-11-03T10:00:00")
    assert not WN.wheel_status(h5)["done"]
    assert WN.run_recording(wav)["status"] == "ok"
    assert WN.run_recording(wav, overwrite=False, model="m1")["status"] == "other_model"
    # ...and so is one with different crops
    MS.delete_pred_mask(h5, 0)
    assert not WN.wheel_status(h5)["done"]


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_"):
            fn()
            print("ok", name)
