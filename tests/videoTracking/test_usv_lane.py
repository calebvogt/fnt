"""Fusion View's USV lane: exact audio from Avisoft's IMA ADPCM wavs without decoding from the start, MAD's
detections placed on the PC clock, bouts, and the window stepping from bout to bout with the lane on the bout's mic.

A synthetic trial: mic 1's folder (ch1_RZ1) holds one 20-s recording named in the PC's clock, with a 40 kHz tone
and a MAD store whose detections sit on it; the Fusion View run of test_fusion_view is pointed at it.
"""
import json
import os
import shutil
import subprocess

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import pytest  # noqa: E402

SR = 250_000
pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")


def _adpcm_wav(path, seconds=20.0):
    t = np.arange(int(seconds * SR)) / SR
    x = 0.01 * np.random.default_rng(0).standard_normal(len(t))
    for a in (5.0, 6.0, 12.0):                                    # 40 kHz bursts, 50 ms
        m = (t >= a) & (t < a + 0.05)
        x[m] += 0.5 * np.sin(2 * np.pi * 40_000 * t[m])
    raw = path + ".raw"
    (np.clip(x, -1, 1) * 32767).astype("<i2").tofile(raw)
    subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-y", "-f", "s16le", "-ar", str(SR), "-ac", "1", "-i", raw,
                    "-c:a", "adpcm_ima_wav", path], check=True)
    os.remove(raw)


def _trial(tmp_path):
    """A USV folder with mic 1's recording (PC clock 998-1018 s) and its MAD store: calls at 5, 6 and 12 s in."""
    from fnt.usv.usv_detector import fnt_mask_store as S
    from fnt.usv.usv_detector.mad_inference import seconds_to_frames
    d = tmp_path / "usv" / "ch1_RZ1"
    d.mkdir(parents=True)
    wav = str(d / "ch11970-01-01_00-16-38_00001.wav")              # 998 s since the epoch, UTC
    _adpcm_wav(wav)
    mad = wav[:-4] + "_FNT.mad"
    S.set_grid_attrs(mad, sample_rate=SR, nperseg=512, noverlap=384, nfft=1024, n_freq_bins=513,
                     n_time_frames=int(20 * SR / 128))
    crops = []
    for i, a in enumerate((5.0, 6.0, 12.0)):
        t0 = int(seconds_to_frames(a, 512, 384, SR))
        nt = int(0.05 * SR / 128)
        crops.append(dict(blob_id=i, mask=np.ones((8, nt), bool), f_off=160, t_off=t0, score=0.9,
                          harmonic_n=1))
    S.write_pred_masks(mad, crops)
    return str(tmp_path / "usv"), wav


def test_adpcm_samples_are_exact_without_decoding_from_the_start(tmp_path):
    from fnt.videoTracking.usv_lane import read_samples
    _, wav = _trial(tmp_path)
    r = subprocess.run(["ffmpeg", "-nostdin", "-v", "error", "-i", wav, "-f", "s16le", "-ac", "1", "pipe:1"],
                       capture_output=True, check=True)
    full = np.frombuffer(r.stdout, "<i2").astype(np.float32) / 32768.0
    for start in (0, 2040, 2041, 1_234_567):
        x, sr = read_samples(wav, start, 100_000)
        assert sr == SR and np.array_equal(x, full[start:start + 100_000])


def test_calls_land_on_the_pc_clock_and_make_bouts(tmp_path):
    from fnt.videoTracking.usv_lane import UsvSource, bouts_from_calls, spectrogram_tile
    usv_dir, _ = _trial(tmp_path)
    src = UsvSource.scan(usv_dir, "UTC")
    assert src.mics == [1] and src.recordings[1][0].t0 == 998.0
    c = src.calls(1, 990, 1030)
    assert np.allclose(c.t0, [1003.0, 1004.0, 1010.0], atol=0.003)
    assert np.allclose(c.f_lo, 160 * SR / 1024)                   # 39 kHz
    b = bouts_from_calls(c, 2.0)
    assert len(b) == 2 and list(b.n_calls) == [2, 1]
    assert len(bouts_from_calls(c, 0.5)) == 3
    x, sr = src.audio(1, 1002.5, 1003.5)                          # the 40 kHz burst shows bright at 1003 s
    img = spectrogram_tile(x, sr, 160)
    assert img[:, 80:90].max() == 255 and img[:, 10:60].mean() < 60
    x, _ = src.audio(1, 1017.0, 1019.0)                           # past the recording's end: zeros
    assert x[-SR // 2:].max() == 0


def test_the_window_steps_bout_to_bout_on_the_bouts_mic(tmp_path):
    from PyQt5.QtWidgets import QApplication
    from test_fusion_view import _run
    from fnt.videoTracking.fusion_view import MIC_ALL, USV_JUMP, ReviewWindow
    app = QApplication.instance() or QApplication([])
    p = _run(tmp_path)
    usv_dir, _ = _trial(tmp_path)
    cfg = json.load(open(p, encoding="utf-8"))
    cfg["usv"] = {"dir": usv_dir, "offset_s": 0, "tz": "UTC"}
    json.dump(cfg, open(p, "w", encoding="utf-8"))
    w = ReviewWindow(p)
    assert w.usv is not None and w.lane.mics == [1]
    w.loader.wait()
    app.processEvents()
    assert len(w.calls) == 3 and len(w.bout_table) == 2
    fps = w.scene.fps
    assert w.targets[USV_JUMP] == [int(round(3.0 * fps)), int(round(10.0 * fps))]
    w.show_frame(0)
    w.jump_bout(1)
    assert w.n == int(round(3.0 * fps)) and abs(w.lane.t - 1003.0) < 1 / fps
    w.jump_bout(1)
    assert w.n == int(round(10.0 * fps))
    w.jump_bout(-1)
    assert w.n == int(round(3.0 * fps))
    w.offset_spin.setValue(-1.0)                                  # audio 1 s behind the video: bouts 1 s later
    assert w.targets[USV_JUMP] == [int(round(4.0 * fps)), int(round(11.0 * fps))]
    w.mic_mode.setCurrentText(MIC_ALL)
    assert w.lane.mics == [1]
    img = w.lane.grab()                                            # draws without the tiles too
    assert not img.isNull()
    w.close()
