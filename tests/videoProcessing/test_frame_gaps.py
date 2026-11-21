"""Lost frames in videos with their own timestamps (fnt.videoProcessing.frame_gaps):
Video PreProcessing keeps the real length and draws the lost frames black, instead of
repeating the last frame before them.

The test videos (mp4 and mkv, H.264) are made by FFmpeg with frames 100-199 of 300 left
out and their timestamps kept, as a recorder that dropped frames writes them; every frame
carries its number as nine black/white blocks."""
import shutil
import subprocess

import numpy as np
import pytest

from fnt.videoProcessing import frame_gaps as G

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")

W, H, N, FPS = 160, 96, 300, 30
LOST = (100, 199)


def _frame(i):
    f = np.full((H, W), 128, np.uint8)
    f[0:32, 0:32] = 255                                  # marker: a real frame
    for b in range(9):
        f[40:72, 16 * b:16 * b + 16] = 255 if (i >> b) & 1 else 0
    return f


def _read_number(f):
    if f[4:28, 4:28].mean() < 64:
        return None                                      # black
    return sum(1 << b for b in range(9) if f[44:68, 16 * b + 3:16 * b + 13].mean() > 128)


def _make(path, lost=LOST):
    sel = f"select='not(between(n,{lost[0]},{lost[1]}))'" if lost else "null"
    p = subprocess.Popen(["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray",
                          "-s", f"{W}x{H}", "-r", str(FPS), "-i", "pipe:", "-vf", sel,
                          "-fps_mode", "passthrough", "-c:v", "libx264", "-preset", "ultrafast",
                          "-crf", "8", "-g", "30", "-pix_fmt", "yuv420p", path],
                         stdin=subprocess.PIPE)
    for i in range(N):
        p.stdin.write(_frame(i).tobytes())
    p.stdin.close()
    assert p.wait() == 0
    return path


def _decode(path, vf="null"):
    """Frames as W x H gray; vf undoes the encode's 720p scale and pad (_ENC)."""
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", path, "-vf", vf,
                        "-f", "rawvideo", "-pix_fmt", "gray", "-fps_mode", "passthrough", "pipe:"],
                       capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.uint8).reshape(-1, H, W)


_ENC = f"crop=1200:720:40:0,scale={W}:{H}"             # 160x96 -> 1200x720 + 40 px each side


@pytest.mark.parametrize("ext", ["mp4", "mkv"])
def test_the_lost_stretch_is_found(tmp_path, ext):
    fg = G.probe(_make(str(tmp_path / f"src.{ext}")))
    assert len(fg.gaps) == 1 and fg.frames == N - 100
    a, b = fg.gaps[0]
    assert a == pytest.approx(99 / FPS, abs=1e-3) and b == pytest.approx(200 / FPS, abs=1e-3)
    assert fg.lost_s == pytest.approx(100 / FPS, abs=1e-3)


def test_an_intact_video_has_no_gaps(tmp_path):
    fg = G.probe(_make(str(tmp_path / "ok.mp4"), lost=None))
    assert fg.gaps == [] and G.blackout_filter(fg, FPS) is None


def test_frame_rows_cover_the_video(tmp_path):
    fg = G.probe(_make(str(tmp_path / "src.mp4")))
    rows, n, black = G.frame_rows(fg, FPS)
    assert n == N and black == 100
    assert rows == [(0, 99, "recovered", "timestamps"), (100, 199, "black", "timestamps"),
                    (200, 299, "recovered", "timestamps")]


def test_jitter_is_not_a_gap():
    t = np.arange(600) / FPS + np.random.default_rng(1).uniform(-0.4, 0.4, 600) / FPS
    assert G.find_gaps(np.sort(t)).gaps == []


def test_many_gaps_are_merged_to_a_short_filter():
    t = np.delete(np.arange(20000) / FPS, np.arange(5, 20000, 20))   # 1,000 one-frame gaps
    t = np.delete(t, np.arange(5, len(t), 19))                       # ... two frames each
    fg = G.find_gaps(t)
    assert 0 < len(fg.gaps) <= G.MAX_GAPS


# --- Video PreProcessing uses it --------------------------------------------------

def _worker(tmp_path, recover=True):
    from PyQt5.QtWidgets import QApplication
    from fnt.videoProcessing import videoProcessing as V
    QApplication.instance() or QApplication([])
    w = V.VideoProcessorWorker([str(tmp_path)], [], FPS, True, False, True, "mp4", 20,
                               "720p", "libx264", "ultrafast", recover_damaged=recover)
    msgs = []
    w.progress_update.connect(msgs.append)
    return w, msgs


@pytest.mark.parametrize("ext", ["mp4", "mkv"])
def test_preprocessing_draws_lost_frames_black(tmp_path, ext):
    src = _make(str(tmp_path / f"cam1_seg.{ext}"))
    out_dir = tmp_path / "proc"
    out_dir.mkdir()
    w, msgs = _worker(tmp_path)
    assert w.process_single_file(src, str(out_dir), 1)
    frames = _decode(str(out_dir / "cam1_seg.mp4"), _ENC)
    assert len(frames) == N
    nums = [_read_number(f) for f in frames]
    assert all(nums[k] is None for k in range(LOST[0], LOST[1] + 1))
    assert all(nums[k] == k for k in range(N) if not LOST[0] <= k <= LOST[1])
    csv = (out_dir / "recovered_frames" / "cam1_seg_frames.csv").read_text().splitlines()
    assert csv[2].startswith("100,199,") and ",black,timestamps" in csv[2]
    assert any("lost frames" in m for m in msgs)


def test_without_recovery_the_last_frame_is_repeated(tmp_path):
    src = _make(str(tmp_path / "cam1_seg.mp4"))
    out_dir = tmp_path / "proc"
    out_dir.mkdir()
    w, _ = _worker(tmp_path, recover=False)
    assert w.process_single_file(src, str(out_dir), 1)
    nums = [_read_number(f) for f in _decode(str(out_dir / "cam1_seg.mp4"), _ENC)]
    assert len(nums) == N and nums[150] == 99           # frozen on the last real frame
    assert not (out_dir / "recovered_frames").exists()
