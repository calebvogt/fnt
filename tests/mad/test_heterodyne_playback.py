"""Heterodyne playback: hear the window at real-time speed.

Slowed-down playback keeps the whole spectrum but stretches a bout until its
rhythm is gone. Heterodyning mixes the ultrasound down by a local oscillator
(LO), so a call at LO + 5 kHz is heard at 5 kHz as it happens.

The DSP is checked for what a listener would notice: the call lands where the
arithmetic says, things far from the LO stay out, a silent window does not
turn into roaring hiss, and the cut edges do not click. The GUI is checked for
plumbing: its own button and shortcut, a remembered LO, 48 kHz output, and a
playhead that moves at real time whatever the Speed slider says.

Runs under pytest, or directly.
"""
import os
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402

from fnt.usv.usv_heterodyne import (  # noqa: E402
    HETERODYNE_OUT_SR, heterodyne_for_playback)

SR = 250_000


def tone(khz, seconds=0.5, amp=0.05, noise=0.0, seed=0):
    t = np.arange(int(seconds * SR)) / SR
    x = amp * np.sin(2 * np.pi * khz * 1000 * t)
    if noise:
        x = x + np.random.default_rng(seed).normal(0, noise, t.size)
    return x.astype(np.float32)


def peak_hz(y, sr=HETERODYNE_OUT_SR):
    spec = np.abs(np.fft.rfft(y * np.hanning(y.size)))
    return np.fft.rfftfreq(y.size, 1 / sr)[int(np.argmax(spec))]


def rms(y):
    return float(np.sqrt(np.mean(np.square(y, dtype=np.float64))))


# ---------------------------------------------------------------------- DSP
def test_a_call_above_the_lo_is_heard_at_the_difference():
    y = heterodyne_for_playback(tone(45), SR, 40_000)
    assert abs(peak_hz(y) - 5_000) < 50, peak_hz(y)


def test_both_sides_of_the_lo_land_on_the_same_pitch():
    """How a heterodyne works, and why the tooltip says so."""
    a = heterodyne_for_playback(tone(45), SR, 40_000)
    b = heterodyne_for_playback(tone(35), SR, 40_000)
    assert abs(peak_hz(a) - peak_hz(b)) < 50


def test_playback_runs_at_real_time():
    y = heterodyne_for_playback(tone(45, seconds=1.5), SR, 40_000)
    assert abs(y.size - 1.5 * HETERODYNE_OUT_SR) <= 2


def test_a_call_far_from_the_lo_stays_out():
    """30 kHz away is outside the 12 kHz audible band: it must not leak in
    as an aliased whistle."""
    near = heterodyne_for_playback(tone(45) + tone(70, amp=0.0), SR, 40_000,
                                   max_gain=1.0)
    far = heterodyne_for_playback(tone(70), SR, 40_000, max_gain=1.0)
    assert rms(far) < 0.01 * rms(near), (rms(far), rms(near))


def test_a_silent_window_plays_quietly():
    """The gain cap: normalising hiss to full scale is a blast of noise."""
    quiet = heterodyne_for_playback(tone(45, amp=0.0, noise=1e-5), SR, 40_000)
    loud = heterodyne_for_playback(tone(45), SR, 40_000)
    assert rms(quiet) < 0.05 * rms(loud)


def test_quiet_calls_are_brought_up_to_a_listenable_level():
    y = heterodyne_for_playback(tone(45, amp=0.002), SR, 40_000)
    assert 0.3 < np.percentile(np.abs(y), 99.9) <= 1.0


def test_the_edges_do_not_click():
    y = heterodyne_for_playback(tone(45), SR, 40_000)
    assert abs(y[0]) < 1e-3 and abs(y[-1]) < 1e-3


def test_one_click_does_not_set_the_gain():
    x = tone(45, amp=0.01)
    x[1000] = 1.0                     # a single knock in the window
    y = heterodyne_for_playback(x, SR, 40_000)
    assert rms(y) > 0.05, "the click made the calls inaudible"


def test_an_lo_above_nyquist_is_refused():
    with pytest.raises(ValueError):
        heterodyne_for_playback(tone(45), 96_000, 60_000)


def test_nothing_in_nothing_out():
    assert heterodyne_for_playback(np.zeros(3), SR).size == 0


# ---------------------------------------------------------------------- GUI
class FakeSD:
    def __init__(self):
        self.played = []
        self.stopped = 0

    def play(self, data, sr):
        self.played.append((np.asarray(data), sr))

    def stop(self):
        self.stopped += 1

    def get_stream(self):
        return None


_KEEP = []


@pytest.fixture
def win(monkeypatch):
    from PyQt5.QtWidgets import QApplication
    from scipy.io import wavfile
    import fnt.usv.mad_pyqt as M
    from fnt.usv.usv_detector.mad_project import (
        MADProjectConfig, create_mad_project)
    fake = FakeSD()
    monkeypatch.setattr(M, "sd", fake, raising=False)
    monkeypatch.setattr(M, "HAS_SOUNDDEVICE", True)
    root = tempfile.mkdtemp(prefix="mad_het_")
    wav = os.path.join(root, "ch12026-02-07_11-17-43_00001.wav")
    wavfile.write(wav, SR, (tone(45, seconds=3.0) * 32767).astype(np.int16))
    proj = os.path.join(root, "p")
    create_mad_project(proj)
    app = QApplication.instance() or QApplication([])
    w = M.MADMainWindow()
    _KEEP.append((app, w))
    w._activate_project(MADProjectConfig.load(proj))
    w._register_audio_files([wav])
    w._rescan_project_wavs()
    w._wait_for_audio_load()
    w._update_playback_buttons_enabled()
    w._fake_sd = fake
    yield w
    w._stop_playback()


def test_het_plays_at_48k_and_real_time(win):
    win.slider_speed.setValue(0)              # slowest; must not apply
    win._toggle_het_playback()
    data, sr = win._fake_sd.played[-1]
    assert sr == HETERODYNE_OUT_SR
    span = win._playback_end_s - win._playback_start_s
    assert abs(data.size / sr - span) < 0.01, "not real time"
    assert win._playback_rate == 1.0
    assert win.btn_het.text() == "Stop" and win.btn_play.text() == "Play"


def test_the_call_is_heard_where_the_lo_puts_it(win):
    win.spin_het_lo.setValue(41)              # 45 kHz call -> 4 kHz
    win._toggle_het_playback()
    data, _ = win._fake_sd.played[-1]
    assert abs(peak_hz(data) - 4_000) < 50


def test_normal_play_still_follows_the_speed_slider(win):
    idx = 0
    win.slider_speed.setValue(idx)
    win._toggle_playback()
    assert win._playback_rate == win._speed_values[idx]
    assert win._fake_sd.played[-1][1] == 44100


def test_either_button_stops_whatever_is_playing(win):
    win._toggle_playback()
    win._toggle_het_playback()                # stop, not a second stream
    assert not win.is_playing and len(win._fake_sd.played) == 1
    assert win.btn_play.text() == "Play" and win.btn_het.text() == "Het"


def test_the_lo_is_remembered(win):
    import fnt.usv.mad_pyqt as M
    win.spin_het_lo.setValue(33)
    assert int(win._settings.value(M.HET_LO_KEY)) == 33


def test_shift_space_is_the_shortcut(win):
    from PyQt5.QtWidgets import QShortcut
    keys = {sc.key().toString() for sc in win.findChildren(QShortcut)}
    assert "Shift+Space" in keys and "Space" in keys


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
