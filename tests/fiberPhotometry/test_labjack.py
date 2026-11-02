"""LabJack LJStreamUD .dat recordings: reading, the camera-TTL frame clock and the batch run."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from fnt.fiberPhotometry.batch import Settings, find_recordings, open_reader, process_recording
from fnt.fiberPhotometry.labjack import LabJackFileReader, is_labjack_dat, ttl_frame_samples
from fnt.fiberPhotometry.sync import recording_frame_times

FS = 250.0


def _synthetic_dat(path: Path, seconds=8.0, cam_start=2.0, cam_fps=30.3, n_frames=None):
    """A .dat like the 2019 rig's: v0 473 nm, v1 405 nm, v2 camera TTL (3 V, low 4 ms per frame)."""
    t = np.arange(int(seconds * FS)) / FS
    rng = np.random.default_rng(0)
    iso = 6.0 + 0.002 * rng.standard_normal(len(t))
    sig = 6.3 + 0.9 * (iso - 6.0) + 0.05 * np.exp(-((t - 5) / 0.3) ** 2) + 0.002 * rng.standard_normal(len(t))
    ttl = np.full(len(t), 3.2)
    frames = np.arange(cam_start, seconds - 0.05, 1 / cam_fps)
    if n_frames is not None:
        frames = frames[:n_frames]
    for f in frames:
        ttl[int(np.ceil(f * FS))] = 0.03            # one low sample per frame
    cols = {"Time": t, "v0": sig, "v1": iso, "v2": ttl}
    for k in range(3, 14):
        cols[f"v{k}"] = np.zeros(len(t)) - k
    pd.DataFrame(cols).to_csv(path, sep="\t", index=False, float_format="%.6f")
    return frames


def _video(path: Path, n, fps=30):
    import cv2
    w = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (64, 48))
    for k in range(n):
        w.write(np.full((48, 64, 3), k % 255, np.uint8))
    w.release()


def test_the_header_identifies_a_labjack_file(tmp_path):
    _synthetic_dat(tmp_path / "a.dat")
    (tmp_path / "other.dat").write_text("x,y\n1,2\n")
    assert is_labjack_dat(tmp_path / "a.dat")
    assert not is_labjack_dat(tmp_path / "other.dat")
    assert [p.name for p in find_recordings([tmp_path])] == ["a.dat"]
    assert isinstance(open_reader(tmp_path / "a.dat"), LabJackFileReader)


def test_ttl_marks_are_the_samples_matlab_marks():
    ttl = np.array([3, 3, 0, 3, 3, 3, 0, 0, 3], float)
    # MATLAB: ttl1 = ttl < 1; find(diff(ttl1) == 1) -> 1-based 2 and 6 -> 0-based 1 and 5
    assert list(ttl_frame_samples(ttl)) == [1, 5]


def test_reader_channels_and_frame_clock(tmp_path):
    frames = _synthetic_dat(tmp_path / "rec.dat")
    _video(tmp_path / "rec.mp4", len(frames))
    r = LabJackFileReader(tmp_path / "rec.dat")
    d = r.scan_file()
    assert d.signal_channel.path == "v0" and d.signal_channel.wavelength == 473
    assert d.isosbestic_channel.path == "v1" and d.isosbestic_channel.wavelength == 405
    assert d.signal_channel.sampling_rate == pytest.approx(FS, rel=1e-3)
    assert d.video_info.absolute_path.endswith("rec.mp4")
    d = r.load_data(d)
    assert d.video_frame_clock and len(d.video_timestamps) == len(frames)
    # each mark is the sample just before the low one, within one sample of the true frame time
    assert np.abs(d.video_timestamps - frames).max() <= 1 / FS + 1e-9
    ft, _ = recording_frame_times(d, d.video_info.absolute_path)
    assert ft[0] == pytest.approx(2.0, abs=0.01)    # the camera started 2 s in, not at 0


def test_surplus_ttls_are_dropped_at_the_end_and_missing_ones_extrapolated(tmp_path):
    frames = _synthetic_dat(tmp_path / "rec.dat")
    _video(tmp_path / "rec.mp4", len(frames) - 5)
    d = LabJackFileReader(tmp_path / "rec.dat")
    d = d.load_data(d.scan_file())
    ft, _ = recording_frame_times(d, str(tmp_path / "rec.mp4"))
    assert len(ft) == len(frames) - 5 and np.allclose(ft, d.video_timestamps[:len(ft)])
    _video(tmp_path / "rec.mp4", len(frames) + 3)
    ft, _ = recording_frame_times(d, str(tmp_path / "rec.mp4"))
    assert len(ft) == len(frames) + 3
    assert np.all(np.diff(ft) > 0) and ft[-1] - ft[len(frames) - 1] == pytest.approx(3 / 30.3, abs=0.005)


def test_the_video_closest_in_frame_count_is_paired(tmp_path):
    frames = _synthetic_dat(tmp_path / "rec.dat")
    _video(tmp_path / "a_short.mp4", 40)
    _video(tmp_path / "b_match.mp4", len(frames))
    d = LabJackFileReader(tmp_path / "rec.dat").scan_file()
    assert Path(d.video_info.absolute_path).name == "b_match.mp4"


def test_batch_times_frames_by_the_ttl(tmp_path):
    frames = _synthetic_dat(tmp_path / "rec.dat")
    _video(tmp_path / "rec.mp4", len(frames))
    out = tmp_path / "out"
    row = process_recording(tmp_path / "rec.dat", Settings(pipeline="matlab", create_video=False), out)
    assert row["status"] == "done" and row["video_timing"] == "frame clock"
    f = pd.read_csv(out / "rec_dff.csv")
    assert len(f) == len(frames) and f["Time_sec"].iloc[0] == pytest.approx(2.0, abs=0.01)
    peak = f.loc[f["DeltaF_F"].idxmax(), "Time_sec"]
    assert peak == pytest.approx(5.0, abs=0.1)     # the transient lands on its own frames
