"""Damaged-AVI recovery (fnt.videoProcessing.avi_recovery): surviving frames go
back at their real time, black where frames are lost, so the video keeps the
recording's length and stays in step with the camera clock.

The test AVI is made by FFmpeg (H.264 + PCM audio, like a security recorder's)
and every frame carries its own number as ten black/white blocks, so each frame
of a rebuilt video can be checked against the slot it was put in."""
import os
import shutil
import struct
import subprocess

import numpy as np
import pytest

from fnt.videoProcessing import avi_recovery as R

pytestmark = pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="needs ffmpeg")

W, H, N, FPS = 160, 96, 600, 30


def _frame(i):
    f = np.full((H, W), 128, np.uint8)
    f[0:32, 0:32] = 255                                  # marker: a real frame
    for b in range(10):                                  # frame number, 10 bits
        f[40:72, 16 * b:16 * b + 16] = 255 if (i >> b) & 1 else 0
    f[76:96, (i * 3) % 140:(i * 3) % 140 + 20] = 200     # something that moves
    return f


def _read_number(f):
    if f[4:28, 4:28].mean() < 128:
        return None                                      # black: no frame here
    return sum(1 << b for b in range(10) if f[44:68, 16 * b + 3:16 * b + 13].mean() > 128)


@pytest.fixture(scope="module")
def intact(tmp_path_factory):
    path = str(tmp_path_factory.mktemp("avi") / "cam1_seg.avi")
    p = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "gray", "-s", f"{W}x{H}",
         "-r", str(FPS), "-i", "pipe:", "-f", "lavfi", "-i", "sine=frequency=440:sample_rate=8000",
         "-t", str(N / FPS), "-c:v", "libx264", "-preset", "ultrafast", "-crf", "8", "-g", "30",
         "-bf", "0", "-pix_fmt", "yuv420p", "-bsf:v", "dump_extra=freq=keyframe",  # SPS/PPS in
         "-c:a", "pcm_s16le", "-shortest", path],                                  # each I-frame
        stdin=subprocess.PIPE)
    for i in range(N):
        p.stdin.write(_frame(i).tobytes())
    p.stdin.close()
    assert p.wait() == 0
    return path


def _copy(intact, tmp_path, name="cam1_seg_damaged.avi"):
    out = str(tmp_path / name)
    shutil.copy(intact, out)
    return out


def _layout(path):
    d = open(path, "rb").read()
    movi = R._parse_header(d)[0]
    idx = d.rfind(b"idx1")
    return d, movi, idx


def _zero(path, start, length):
    with open(path, "r+b") as f:
        f.seek(start)
        f.write(bytes(length))


def _decode(mp4):
    r = subprocess.run(["ffmpeg", "-v", "error", "-i", mp4, "-f", "rawvideo", "-pix_fmt", "gray",
                        "-fps_mode", "passthrough", "pipe:"], capture_output=True, check=True)
    return np.frombuffer(r.stdout, np.uint8).reshape(-1, H, W)


def _recover(path, tmp_path, **kw):
    out = str(tmp_path / "out.mp4")
    res = R.recover_avi(path, out, R.default_encoder_cmd(crf=10, codec="libx264", width=W, height=H),
                        gray=True, **kw)
    return res, (_decode(out) if res else None)


def _check_in_place(frames, slack=0):
    """Every real frame sits at its own number (within ``slack``); returns the count."""
    real = 0
    for slot, f in enumerate(frames):
        num = _read_number(f)
        if num is not None:
            real += 1
            assert abs(num - slot) <= slack, f"frame {num} written at slot {slot}"
    return real


def test_an_intact_avi_maps_every_frame(intact):
    amap = R.map_avi(intact)
    assert amap.total_frames == N and len(amap.frames) == N
    assert amap.codec == "h264" and amap.fps == FPS and not amap.estimated
    assert sum(len(r) for r in R.frame_runs(amap)) == N


def test_a_lost_block_keeps_the_real_length(intact, tmp_path):
    path = _copy(intact, tmp_path)
    d, movi, idx = _layout(path)
    _zero(path, movi + (idx - movi) * 30 // 100, (idx - movi) * 20 // 100)
    res, frames = _recover(path, tmp_path)
    assert res.total_frames == N and len(frames) == N
    real = _check_in_place(frames)
    assert res.black > 60 and real == res.recovered and real > N // 2
    # the rows cover the video end to end, in order
    edges = [(a, b) for a, b, _, _ in res.rows]
    assert edges[0][0] == 0 and edges[-1][1] == N - 1
    assert all(b + 1 == a2 for (_, b), (a2, _) in zip(edges, edges[1:]))


def test_a_lost_header_borrows_from_an_intact_recording(intact, tmp_path):
    path = _copy(intact, tmp_path)
    d, movi, idx = _layout(path)
    _zero(path, 0, movi + 4096)                  # header and the first chunks
    _zero(path, idx, 8)                          # and the index's own header
    _zero(path, idx + 8, 16 * 40)                # and its first rows
    _zero(path, idx - 3000, 3000)                # and the data just before it
    assert R.map_avi(path) is None               # its length can't be pinned alone
    ref = R.read_reference(intact)
    assert ref["entries"] > N and ref["fps"] == FPS
    res, frames = _recover(path, tmp_path, reference=ref)
    assert res.total_frames == N and len(frames) == N
    assert _check_in_place(frames) == res.recovered > N * 3 // 4


def test_a_reference_is_found_beside_the_file(intact, tmp_path):
    shutil.copy(intact, tmp_path / "cam1_seg_ok.avi")
    path = _copy(intact, tmp_path)
    _zero(path, 0, 4096)
    ref = R.find_reference(path)
    assert ref and os.path.basename(ref["path"]) == "cam1_seg_ok.avi"


def test_without_its_header_the_index_still_gives_the_length(intact, tmp_path):
    path = _copy(intact, tmp_path)
    d, movi, idx = _layout(path)
    _zero(path, 0, movi + 4096)
    _zero(path, idx, 8)                          # the first row still says offset 4
    res, frames = _recover(path, tmp_path, reference=None, fps=FPS)
    assert res.info["index"] == "end-aligned (first row)"
    assert len(frames) == N and _check_in_place(frames) == res.recovered


def test_frames_the_index_lost_are_placed_by_estimate(intact, tmp_path):
    path = _copy(intact, tmp_path)
    d, movi, idx = _layout(path)
    n = (len(d) - idx - 8) // 16
    rows = np.frombuffer(d, "<u4", count=4 * n, offset=idx + 8).reshape(n, 4)
    vid = [k for k in range(n) if rows[k, 0] == struct.unpack("<I", b"00dc")[0]]
    k1, k2 = vid[200], vid[400]                  # rows of frames 200-400 lost
    _zero(path, idx + 8 + 16 * k1, 16 * (k2 - k1))
    for f in (260, 340):                         # and two small blocks of their data
        o = int(rows[vid[f], 2]) + movi
        _zero(path, o, 2000)
    amap = R.map_avi(path)
    assert amap.estimated and amap.total_frames == N
    res, frames = _recover(path, tmp_path, amap=amap)
    assert len(frames) == N
    assert _check_in_place(frames, slack=10) == res.recovered
    assert any(h == "estimated" for _, _, _, h in res.rows)


def test_the_frames_csv_lists_every_stretch(intact, tmp_path):
    path = _copy(intact, tmp_path)
    d, movi, idx = _layout(path)
    _zero(path, movi + (idx - movi) // 2, 20000)
    res, _ = _recover(path, tmp_path)
    csv_path = str(tmp_path / "frames.csv")
    R.write_frames_csv(res, csv_path)
    lines = open(csv_path).read().splitlines()
    assert lines[0] == "first_frame,last_frame,start_s,end_s,content,placement"
    assert {ln.split(",")[4] for ln in lines[1:]} == {"recovered", "black"}
    assert lines[-1].split(",")[3] == f"{N / FPS:.3f}"


# --- Video PreProcessing uses it --------------------------------------------------

def _worker(tmp_path):
    from PyQt5.QtWidgets import QApplication
    from fnt.videoProcessing import videoProcessing as V
    QApplication.instance() or QApplication([])
    w = V.VideoProcessorWorker([str(tmp_path)], [], FPS, True, False, True, "mp4", 20,
                               "720p", "libx264", "ultrafast")
    msgs = []
    w.progress_update.connect(msgs.append)
    return w, msgs


def test_the_raw_encoder_command_uses_the_run_settings(tmp_path):
    w, _ = _worker(tmp_path)
    cmd = w.build_raw_encoder_command(1280, 720, "gray", 25, "o.mp4.partial")
    assert cmd[-3:] == ["-f", "mp4", "o.mp4.partial"]
    vf = cmd[cmd.index("-vf") + 1]
    assert vf.startswith("fps=30,scale=1280:720") and vf.endswith("format=gray")
    assert cmd[cmd.index("-r") + 1] == "25" and "-an" in cmd and "libx264" in cmd


def test_preprocessing_rebuilds_an_avi_whose_header_is_lost(intact, tmp_path):
    shutil.copy(intact, tmp_path / "cam1_seg_ok.avi")
    path = _copy(intact, tmp_path)
    d, movi, idx = _layout(path)
    _zero(path, 0, movi + 4096)
    out_dir = tmp_path / "proc"
    out_dir.mkdir()
    w, msgs = _worker(tmp_path)
    assert w.process_single_file(path, str(out_dir), 1)
    out = out_dir / "cam1_seg_damaged.mp4"
    assert out.exists() and abs(w._media_duration(str(out)) - N / FPS) < 0.2
    assert (out_dir / "recovered_frames" / "cam1_seg_damaged_frames.csv").exists()
    assert any("frames recovered" in m for m in msgs)


def test_preprocessing_rebuilds_an_encode_that_came_out_short(intact, tmp_path):
    path = _copy(intact, tmp_path)
    d, movi, idx = _layout(path)
    _zero(path, movi + (idx - movi) * 30 // 100, (idx - movi) * 40 // 100)
    out_dir = tmp_path / "proc"
    out_dir.mkdir()
    w, msgs = _worker(tmp_path)
    assert w.process_single_file(path, str(out_dir), 1)
    out = str(out_dir / "cam1_seg_damaged.mp4")
    assert abs(w._media_duration(out) - N / FPS) < 0.2
    assert any("shorter than the recording" in m for m in msgs)


def test_recovery_can_be_turned_off(intact, tmp_path):
    path = _copy(intact, tmp_path)
    d, movi, idx = _layout(path)
    _zero(path, 0, movi + 4096)
    out_dir = tmp_path / "proc"
    out_dir.mkdir()
    w, msgs = _worker(tmp_path)
    w.recover_damaged = False
    w.process_single_file(path, str(out_dir), 1)        # whatever FFmpeg makes of it
    assert not (out_dir / "recovered_frames").exists()
    assert not any("🩹" in m for m in msgs)
