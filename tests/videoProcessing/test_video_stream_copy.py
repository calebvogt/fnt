"""Video PreProcessing's stream copy: a source that already is what the encode would make
(H.264/H.265 at the target size and frame rate, gray, no lost frames) is copied into the
output without re-encoding; anything else is encoded as before."""
import json
import os
import subprocess

import pytest
from PyQt5.QtWidgets import QApplication

from fnt.videoProcessing import videoProcessing as V

_APP = []


def _worker(tmp_path, copy=True):
    if not _APP:
        _APP.append(QApplication.instance() or QApplication([]))
    w = V.VideoProcessorWorker([str(tmp_path)], [], 30, True, False, True, "mp4", 20,
                               "720p", "libx265", "ultrafast", stream_copy=copy)
    msgs = []
    w.progress_update.connect(msgs.append)
    return w, msgs


def _make(path, size="1280x720", rate=30, colour=False, codec="libx265", lost=None):
    src = f"testsrc2=s={size}:r={rate}:d=3"
    vf = "null" if colour else "format=gray"
    if lost:
        vf += f",select='not(between(n,{lost[0]},{lost[1]}))'"
    cmd = ["ffmpeg", "-y", "-v", "error", "-f", "lavfi", "-i", src, "-vf", vf,
           "-c:v", codec, "-preset", "ultrafast", "-pix_fmt", "yuv420p"]
    if codec == "libx265":
        cmd += ["-x265-params", "log-level=error"]
        if path.endswith(".avi"):
            cmd += ["-tag:v", "HEVC"]                    # as the NVR's AVIs carry it
    if lost:
        cmd += ["-fps_mode", "passthrough"]
    subprocess.run(cmd + [path], check=True)
    return path


def _probe(path):
    r = subprocess.run(["ffprobe", "-v", "error", "-select_streams", "v:0", "-count_frames",
                        "-show_entries", "stream=codec_name,codec_tag_string,nb_read_frames,width,height",
                        "-of", "json", path], capture_output=True, text=True, check=True)
    return json.loads(r.stdout)["streams"][0]


def _run(tmp_path, src, copy=True):
    out_dir = tmp_path / "proc"
    out_dir.mkdir(exist_ok=True)
    w, msgs = _worker(tmp_path, copy)
    assert w.process_single_file(src, str(out_dir), 1)
    out = out_dir / (os.path.splitext(os.path.basename(src))[0] + ".mp4")
    assert out.exists() and not os.path.exists(str(out) + ".partial")
    return str(out), msgs


@pytest.mark.parametrize("ext", ["mp4", "avi", "mkv"])
def test_a_ready_gray_video_is_copied_not_reencoded(tmp_path, ext):
    src = _make(str(tmp_path / f"cam1_seg.{ext}"))
    out, msgs = _run(tmp_path, src)
    assert any("stream copy" in m for m in msgs)
    p = _probe(out)
    assert p["codec_name"] == "hevc" and p["codec_tag_string"] == "hvc1"
    assert int(p["nb_read_frames"]) == 90
    assert (p["width"], p["height"]) == (1280, 720)


def test_h264_is_copied_too(tmp_path):
    out, msgs = _run(tmp_path, _make(str(tmp_path / "cam1.mp4"), codec="libx264"))
    assert any("stream copy" in m for m in msgs)
    assert _probe(out)["codec_name"] == "h264"


@pytest.mark.parametrize("kw,why", [(dict(colour=True), "has colour"),
                                    (dict(rate=25), "frame rate"),
                                    (dict(size="640x360"), "640x360"),
                                    # lost frames also lower the average frame rate
                                    (dict(lost=(30, 44)), ("lost frames", "frame rate"))])
def test_anything_else_is_reencoded(tmp_path, kw, why):
    out, msgs = _run(tmp_path, _make(str(tmp_path / "cam1.mp4"), **kw))
    assert not any("stream copy" in m for m in msgs)
    why = (why,) if isinstance(why, str) else why
    assert any(m.strip().startswith("re-encoding") and any(y in m for y in why) for m in msgs), msgs
    p = _probe(out)
    assert p["codec_name"] == "hevc" and (p["width"], p["height"]) == (1280, 720)
    assert int(p["nb_read_frames"]) == 90                # at the real length, 30 fps


def test_switched_off_everything_is_reencoded(tmp_path):
    out, msgs = _run(tmp_path, _make(str(tmp_path / "cam1.mp4")), copy=False)
    assert not any("stream copy" in m or "re-encoding" in m for m in msgs)
    assert _probe(out)["codec_name"] == "hevc"
