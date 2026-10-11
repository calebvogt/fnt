"""Render Clip asks how the clip should sound: slowed, heterodyne, or silent.

Before heterodyne playback existed a clip's audio was always the view slowed
by the Speed slider. Now there are three honest choices and which is right
depends on the audience, so the render asks -- defaulting to what the user is
listening with, so Enter renders what they just heard.

Each mode must make audio exactly as long as the video (the playhead sweep is
timed to the clip), heterodyne must be real time, and a job written before
modes existed must still mean "slowed".

Runs under pytest, or directly.
"""
import os
import shutil
import subprocess
import tempfile

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import numpy as np  # noqa: E402
import pytest  # noqa: E402
from PyQt5.QtWidgets import QApplication, QDialog  # noqa: E402

import fnt.usv.mad_pyqt as M  # noqa: E402

SR = 250_000


def tone(khz, seconds=1.0, amp=0.05):
    t = np.arange(int(seconds * SR)) / SR
    return (amp * np.sin(2 * np.pi * khz * 1000 * t)).astype(np.float32)


def peak_hz(y, sr):
    spec = np.abs(np.fft.rfft(y * np.hanning(y.size)))
    return np.fft.rfftfreq(y.size, 1 / sr)[int(np.argmax(spec))]


_APP = []


def app():
    # Held at module level: a QApplication nothing references is collected
    # at once, and the next widget built takes the process down.
    if not _APP:
        _APP.append(QApplication.instance() or QApplication([]))
    return _APP[0]


# ------------------------------------------------------------ the options
def dialog(**kw):
    app()
    base = dict(view_dur=0.5, speeds=[0.01, 0.1, 0.5, 1.0], speed=0.1,
                lo_khz=40, mode='slow')
    base.update(kw)
    return M.MADClipOptionsDialog(None, **base)


def test_the_defaults_are_the_current_controls():
    d = dialog(speed=0.1, lo_khz=33, mode='slow')
    assert d.rb_slow.isChecked()
    assert d.cmb_speed.currentData() == 0.1
    assert d.spin_lo.value() == 33


def test_each_mode_reports_its_settings():
    d = dialog()
    d.rb_het.setChecked(True)
    assert d.options() == {'mode': 'het', 'lo_hz': 40_000.0, 'speed': 1.0}
    d.rb_silent.setChecked(True)
    assert d.options() == {'mode': 'silent', 'speed': 1.0}
    d.rb_slow.setChecked(True)
    assert d.options() == {'mode': 'slow', 'speed': 0.1}


def test_the_clip_length_is_shown_before_rendering():
    d = dialog(view_dur=0.5, speed=0.1)
    assert "5.0 s" in d.lbl_len.text()          # 0.5 s at 0.1x
    d.rb_het.setChecked(True)
    assert "0.5 s" in d.lbl_len.text()          # real time


def test_only_the_relevant_control_is_live():
    d = dialog()
    d.rb_het.setChecked(True)
    assert d.spin_lo.isEnabled() and not d.cmb_speed.isEnabled()


def test_what_you_are_listening_with_wins():
    pick = M.MADClipOptionsDialog.default_mode
    assert pick('het', 'slow') == 'het'          # just played with Het
    assert pick(None, 'silent') == 'silent'      # nothing played: last render
    assert pick(None, None) == 'slow'
    assert pick('bogus', 'nonsense') == 'slow'


# -------------------------------------------------------------- the audio
def job(mode=None, seconds=0.5, speed=1.0, out_sr=44100, lo=40_000.0):
    j = {'segment': tone(45, seconds), 'n_out': int(seconds / speed * out_sr),
         'out_sr': out_sr, 'sr': SR, 'lo_hz': lo}
    if mode:
        j['mode'] = mode
    return j


def test_heterodyne_audio_is_real_time_and_lands_at_the_difference():
    j = job('het', out_sr=48000)
    y = M.MADClipWorker.clip_audio(j)
    assert y.size == j['n_out']
    assert abs(peak_hz(y, 48000) - 5000) < 60


def test_slowed_audio_is_stretched_and_audible():
    j = job('slow', speed=0.1)
    y = M.MADClipWorker.clip_audio(j)
    assert y.size == j['n_out']                       # 10x longer
    assert abs(peak_hz(y, 44100) - 4500) < 60         # 45 kHz / 10
    assert 0.9 < np.abs(y).max() <= 1.0


def test_silent_is_silent_and_still_the_right_length():
    j = job('silent')
    y = M.MADClipWorker.clip_audio(j)
    assert y.size == j['n_out'] and not y.any()


def test_a_job_from_before_modes_still_means_slowed():
    j = job(None, speed=0.1)
    y = M.MADClipWorker.clip_audio(j)
    assert abs(peak_hz(y, 44100) - 4500) < 60


# --------------------------------------------------------- the render path
_KEEP = []


@pytest.fixture
def win(monkeypatch):
    from scipy.io import wavfile
    from fnt.usv.usv_detector.mad_project import (
        MADProjectConfig, create_mad_project)
    root = tempfile.mkdtemp(prefix="mad_clipmode_")
    wav = os.path.join(root, "ch12026-02-07_11-17-43_00001.wav")
    wavfile.write(wav, SR, (tone(45, 3.0) * 32767).astype(np.int16))
    proj = os.path.join(root, "p")
    create_mad_project(proj)
    a = app()
    w = M.MADMainWindow()
    _KEEP.append((a, w))
    w._activate_project(MADProjectConfig.load(proj))
    w._register_audio_files([wav])
    w._rescan_project_wavs()
    w._wait_for_audio_load()

    jobs = []

    class FakeWorker:
        def __init__(self, job, parent=None):
            jobs.append(job)
            self.finished_signal = self.error_signal = self

        def connect(self, *_):
            pass

        def start(self):
            pass
    monkeypatch.setattr(M, "MADClipWorker", FakeWorker)
    monkeypatch.setattr(M.shutil, "which", lambda name: "ffmpeg")
    w._jobs = jobs
    yield w
    w.btn_render_clip.setEnabled(True)


def t_of(job):
    cmd = job['cmd']
    return float(cmd[cmd.index("-t") + 1])


def test_a_heterodyne_clip_runs_at_real_time(win):
    view = win.spectrogram.view_end - win.spectrogram.view_start
    win.render_spectrogram_clip({'mode': 'het', 'lo_hz': 38_000.0})
    j = win._jobs[-1]
    assert j['mode'] == 'het' and j['lo_hz'] == 38_000.0
    assert abs(t_of(j) - view) < 1e-3
    assert j['out_sr'] == 48000
    assert "het38k" in os.path.basename(j['out'])


def test_a_slowed_clip_is_stretched_by_its_own_speed(win):
    view = win.spectrogram.view_end - win.spectrogram.view_start
    win.slider_speed.setValue(win.slider_speed.maximum())   # slider says 1x
    win.render_spectrogram_clip({'mode': 'slow', 'speed': 0.1})
    j = win._jobs[-1]
    assert abs(t_of(j) - view / 0.1) < 1e-3, "used the slider, not the choice"
    assert "0.1x" in os.path.basename(j['out'])


def test_the_choice_is_remembered(win):
    win.render_spectrogram_clip({'mode': 'silent'})
    assert win._settings.value(M.CLIP_AUDIO_MODE_KEY) == 'silent'


def test_the_button_asks_and_cancelling_renders_nothing(win, monkeypatch):
    seen = {}

    def fake_exec(self):
        seen['mode'] = self.options()['mode']
        return QDialog.Rejected
    monkeypatch.setattr(M.MADClipOptionsDialog, "exec_", fake_exec)
    win._last_playback_mode = 'het'
    win.btn_render_clip.click()
    assert seen == {'mode': 'het'}, "did not default to what was playing"
    assert win._jobs == []


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")
def test_a_real_heterodyne_render_has_a_real_time_soundtrack():
    """End to end through ffmpeg: the mp4's audio is as long as the window."""
    j = job('het', seconds=0.8, out_sr=48000)
    with tempfile.TemporaryDirectory() as d:
        png = os.path.join(d, "f.png")
        bar = os.path.join(d, "b.png")
        from PyQt5.QtGui import QColor, QImage
        app()
        for path, w in ((png, 64), (bar, 2)):
            img = QImage(w, 48, QImage.Format_RGB888)
            img.fill(QColor(20, 20, 20))
            img.save(path, "PNG")
        j['wav'] = os.path.join(d, "a.wav")
        j['out'] = os.path.join(d, "c.mp4")
        ff = shutil.which("ffmpeg")
        j['cmd'] = [ff, "-y", "-loglevel", "error", "-loop", "1",
                    "-framerate", "10", "-i", png, "-loop", "1", "-i", bar,
                    "-i", j['wav'], "-filter_complex",
                    "[0:v][1:v]overlay=x=0:y=0,format=yuv420p[v]",
                    "-map", "[v]", "-map", "2:a", "-c:v", "libx264",
                    "-c:a", "aac", "-t", "0.8", "-shortest", j['out']]
        got = []
        w = M.MADClipWorker(j)
        w.finished_signal.connect(got.append)
        w.run()
        assert got and os.path.isfile(j['out'])
        probe = shutil.which("ffprobe")
        if probe:
            dur = float(subprocess.run(
                [probe, "-v", "error", "-select_streams", "a:0",
                 "-show_entries", "stream=duration", "-of", "csv=p=0",
                 j['out']], capture_output=True, text=True).stdout.strip())
            assert abs(dur - 0.8) < 0.1, dur


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
