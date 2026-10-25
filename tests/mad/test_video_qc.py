"""Video QC: the video filmed alongside a stretch of a recording.

A project's video_qc.json says how to get from a recording to its camera
(mad_video); MAD's Video QC button opens the frames for the view, synced with
the playhead (mad_video_qc). Here on synthetic files: two 8-s videos at 15 fps,
back to back, each frame a flat grey whose level gives its number, and a
recording that starts 3 s into the first video.

Runs under pytest, or directly.
"""
import json
import os
import tempfile
import time
from datetime import datetime

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")

import cv2
import numpy as np
import pytest
from scipy.io import wavfile

from fnt.usv.usv_detector import mad_video as MV

SR = 250_000
FPS = 15
N = 120                       # frames per video file (8 s)
WAV_NAME = "T004_OB_ch1_T2021-10-25_02-53-33_0000377.wav"
_APP = None
_WINDOWS = []


def _level(file_i, k):
    """Grey level of frame k of video file i (0 or 1)."""
    return 10 + 2 * k if file_i == 0 else 249 - 2 * k


def _write_video(path, file_i):
    vw = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), FPS, (64, 48))
    for k in range(N):
        vw.write(np.full((48, 64, 3), _level(file_i, k), np.uint8))
    vw.release()


def _dataset():
    """(root, wav, cfg dict): one recording of T004 ch1 and its CAM16 video."""
    root = tempfile.mkdtemp(prefix="mad_video_qc_")
    ch = os.path.join(root, "audio", "T004_WD", "ch1")
    os.makedirs(ch)
    wav = os.path.join(ch, WAV_NAME)
    wavfile.write(wav, SR, (np.random.default_rng(0).normal(0, .05, SR * 10)
                            * 32767).astype(np.int16))
    vid = os.path.join(root, "video", "T004_WD")
    os.makedirs(vid)
    _write_video(os.path.join(vid, "CAM16_2021-10-25_02_53_30_000.mp4"), 0)
    _write_video(os.path.join(vid, "CAM16_2021-10-25_02_53_38_000.mp4"), 1)
    cfg = {
        "trial_regex": r"/(T\d{3})_",
        "channel_regex": r"_(ch\d)",
        "audio_start": [{"regex": r"T(\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2})_",
                         "format": "%Y-%m-%d_%H-%M-%S"}, {"avisoft_log": True}],
        "cameras": {"ch1": "CAM16", "ch2": "CAM15"},
        "sources": [{"root": os.path.join(root, "video"), "cameras": ["CAM13", "CAM16"]}],
        "video_start": {"regex": r"_(\d{4}-\d{2}-\d{2}_\d{2}_\d{2}_\d{2}_\d{3})\.mp4$",
                        "format": "%Y-%m-%d_%H_%M_%S_%f"},
    }
    return root, wav, cfg


def _levels(clip):
    return [int(round(float(f.mean()))) for f in clip.frames]


def test_a_recording_names_its_trial_channel_start_and_cameras():
    _, wav, d = _dataset()
    cfg = MV.VideoQCConfig.from_dict(d)
    assert MV.recording_ids(wav, cfg) == ("T004", "ch1")
    assert MV.recording_start(wav, cfg) == datetime(2021, 10, 25, 2, 53, 33)
    assert MV.cameras_for(wav, cfg) == ["CAM16", "CAM13"]   # its own camera first


def test_the_avisoft_log_gives_a_sequence_numbered_files_start():
    root = tempfile.mkdtemp(prefix="mad_video_qc_log_")
    ch = os.path.join(root, "T009_MupKO", "ch2")
    os.makedirs(ch)
    wav = os.path.join(ch, "T009_KOOBko_ch2_T0000130.wav")
    open(wav, "wb").close()
    with open(os.path.join(ch, "T009_KOOBko_ch2.LOG"), "w") as f:
        f.write("12/08/21\t13:15:02\tT0000129.wav\t600\n12/08/21\t13:25:02\tT0000130.wav\t600\n")
    cfg = MV.VideoQCConfig.from_dict({"audio_start": [{"avisoft_log": True}]})
    assert MV.recording_start(wav, cfg) == datetime(2021, 12, 8, 13, 25, 2)
    assert MV.recording_ids(wav, cfg) == ("T009", "ch2")


def test_the_frames_line_up_with_the_recording():
    _, wav, d = _dataset()
    cfg = MV.VideoQCConfig.from_dict(d)
    clip = MV.read_clip(wav, 1.0, 2.0, "CAM16", cfg)
    # 1 s into the recording = 4 s into the first video = its frame 60
    i = clip.index_at(1.0)
    assert abs(clip.times_s[i] - 1.0) < 1e-6
    assert abs(_levels(clip)[i] - _level(0, 60)) <= 3
    name, k, when = clip.frame_info(i)
    assert (name, k) == ("CAM16_2021-10-25_02_53_30_000.mp4", 60)
    assert when == datetime(2021, 10, 25, 2, 53, 34)


def test_a_stretch_across_two_files_takes_frames_from_both():
    _, wav, d = _dataset()
    cfg = MV.VideoQCConfig.from_dict(d)
    clip = MV.read_clip(wav, 4.0, 6.0, "CAM16", cfg)
    assert len(clip.spans) == 2
    assert np.all(np.diff(clip.times_s) > 0)
    lv = _levels(clip)
    assert abs(lv[clip.index_at(4.5)] - _level(0, int(7.5 * FPS))) <= 3   # first file, 7.5 s in
    assert abs(lv[clip.index_at(5.5)] - _level(1, int(0.5 * FPS))) <= 3   # second file, 0.5 s in


def test_a_long_stretch_keeps_every_kth_frame():
    _, wav, d = _dataset()
    cfg = MV.VideoQCConfig.from_dict(d)
    clip = MV.read_clip(wav, 0.0, 10.0, "CAM16", cfg, max_frames=30)
    assert len(clip.frames) <= 30
    assert np.all(np.diff(clip.times_s) > 0)


def test_a_clock_offset_shifts_the_frames():
    _, wav, d = _dataset()
    d["offsets_s"] = {"T004": 2.0}          # the video clock runs 2 s ahead
    cfg = MV.VideoQCConfig.from_dict(d)
    clip = MV.read_clip(wav, 1.0, 2.0, "CAM16", cfg)
    assert clip.frame_info(clip.index_at(1.0))[1] == 90


def test_no_video_says_why():
    _, wav, d = _dataset()
    cfg = MV.VideoQCConfig.from_dict(d)
    with pytest.raises(MV.VideoQCError, match="No CAM13 video for T004"):
        MV.read_clip(wav, 1.0, 2.0, "CAM13", cfg)
    with pytest.raises(MV.VideoQCError, match="not recording then"):
        MV.read_clip(wav, 300.0, 301.0, "CAM16", cfg)


# ----------------------------------------------------------------------
# MAD's button and window
# ----------------------------------------------------------------------
def _window(with_video=True):
    global _APP
    from PyQt5.QtWidgets import QApplication
    import fnt.usv.mad_pyqt as M
    from fnt.usv.usv_detector.mad_project import MADProjectConfig, create_mad_project
    _APP = QApplication.instance() or QApplication([])
    root, wav, d = _dataset()
    proj = os.path.join(root, "p")
    create_mad_project(proj)
    if with_video:
        with open(os.path.join(proj, MV.VIDEO_QC_FILENAME), "w") as f:
            json.dump(d, f)
    M.MADMainWindow._apply_dark_theme()
    w = M.MADMainWindow()
    w._activate_project(MADProjectConfig.load(proj))
    w._register_audio_files([wav])
    w._append_audio_paths([wav])
    w._wait_for_audio_load()
    _WINDOWS.append(w)
    return w


def _wait_loaded(dlg, timeout=30):
    t0 = time.time()
    while (dlg._worker is not None or dlg._clip is None) and time.time() - t0 < timeout:
        _APP.processEvents()
        time.sleep(0.01)
    return dlg._clip


def test_the_button_is_off_without_a_video_source():
    w = _window(with_video=False)
    assert not w.btn_video_qc.isEnabled()


def test_video_qc_opens_the_channels_camera_on_the_view_and_follows_the_playhead():
    w = _window()
    assert w.btn_video_qc.isEnabled()
    sg = w.spectrogram
    sg.view_start, sg.view_end = 2.0, 3.0
    w._show_video_qc()
    dlg = w._video_qc_window()
    clip = _wait_loaded(dlg)
    assert clip is not None and clip.camera == "CAM16"
    assert dlg.combo_camera.currentData() == "CAM16"
    assert clip.times_s[0] <= 1.0 + 1 / FPS and clip.times_s[-1] >= 4.0 - 1 / FPS   # view +- 1 s
    # MAD's playhead drives the frame
    dlg.playback_started(2.0, 3.0)
    dlg.playback_tick(2.5)
    assert abs(float(clip.times_s[dlg._index]) - 2.5) < 1 / FPS + 1e-9
    dlg.playback_stopped()
    # the slider moves the playhead line on the spectrogram
    dlg.slider.setValue(clip.index_at(3.2))
    assert abs(sg.playback_position - 3.2) < 1 / FPS + 1e-9
    # closing hides it and takes the line away
    dlg.close()
    assert not dlg.isVisible() and sg.playback_position is None


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
