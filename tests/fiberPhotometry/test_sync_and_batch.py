"""Video alignment, the combined video and the batch outputs."""
import json
from pathlib import Path

import numpy as np
import pytest

from fnt.fiberPhotometry.sync import align_to_frames, check_video_clock


def test_each_frame_gets_the_mean_of_its_samples():
    t = np.arange(0, 10, 0.01)                   # 100 Hz
    v = t.copy()
    ft = np.arange(0, 9.9, 1 / 30)               # 30 fps
    out = align_to_frames(t, v, ft)
    # frame k spans [k/30, (k+1)/30): mean time inside it
    for k in (0, 10, 200):
        sel = (t >= ft[k]) & (t < ft[k] + 1 / 30)
        assert out[k] == pytest.approx(v[sel].mean())


def test_nan_samples_are_ignored_and_long_gaps_stay_blank():
    t = np.arange(0, 10, 0.01)
    v = np.ones_like(t)
    v[300:600] = np.nan                          # 3 s gap
    ft = np.arange(0, 9.9, 1 / 30)
    out = align_to_frames(t, v, ft)
    assert np.isnan(out[(ft > 3.1) & (ft < 5.9)]).all()
    assert np.allclose(out[ft < 2.9], 1)


def test_frames_are_timed_by_the_video_not_by_the_logged_count():
    """A recorder that logs 84% of frame times over the same span: index pairing would drift."""
    ft = np.arange(102773) / 30.0
    logged = np.linspace(0, ft[-1], 85727)
    c = check_video_clock(ft, logged)
    assert c["video_frames"] == 102773 and c["logged_frames"] == 85727
    assert c["frame_count_ratio"] == pytest.approx(1.199, abs=0.001)
    assert abs(c["video_span_s"] - c["logged_span_s"]) < 0.05
    drift_at_end = ft[85726] - logged[85726]     # what pairing by index would do
    assert abs(drift_at_end) > 500


def _synthetic_doric(path: Path, seconds=6.0, video=None):
    """A minimal .doric (HDF5) laid out like the NC500 files."""
    import h5py
    fs = 100.0
    t = np.arange(int(seconds * fs)) / fs
    sig = 0.064 + 0.002 * np.sin(t)
    iso = 0.055 + 0.0017 * np.sin(t)
    with h5py.File(path, "w") as f:
        base = "DataAcquisition/NC500/Signals/Series0001"
        for k, (x, nm) in enumerate(((sig, 470), (iso, 415)), start=1):
            g = f.create_group(f"{base}/Headstage01LockIn0{k}")
            g.create_dataset(f"Headstage01LockIn0{k}", data=x)
            g.create_dataset("Time", data=t)
        f.create_group("Configurations/NC500/LED1").attrs["WaveLength"] = 470
        f.create_group("Configurations/NC500/LED2").attrs["WaveLength"] = 415
    return t


def test_a_run_writes_a_dated_folder_beside_the_recordings(tmp_path):
    from fnt.fiberPhotometry.batch import RUNS_FOLDER, SUMMARY_FILE, Settings, run_batch
    for name in ("a", "b"):
        _synthetic_doric(tmp_path / f"{name}.doric")
    recs = sorted(tmp_path.glob("*.doric"))
    out = run_batch(recs, Settings(create_video=False), log=lambda m: None)
    assert out.parent == tmp_path / RUNS_FOLDER and out.name.endswith("_guppy")
    for name in ("a", "b"):
        for suffix in ("_photometry.csv", "_dff.csv", "_settings.json"):
            assert (out / f"{name}{suffix}").exists()
    assert (out / "run_settings.json").exists() and (out / "run.log").exists()
    import pandas as pd
    summary = pd.read_csv(out / SUMMARY_FILE)
    assert list(summary.recording) == ["a.doric", "b.doric"] and set(summary.status) == {"done"}
    df = pd.read_csv(out / "a_photometry.csv")
    assert list(df.columns) == ["Time_sec", "Signal_V", "Control_V", "FittedControl_V",
                                "DeltaF_F_pct", "ZScore", "Dropout"]
    # nothing written beside the raw files
    assert sorted(p.name for p in tmp_path.iterdir()) == ["FNT_FiberPhotometry", "a.doric", "b.doric"]
    # a second run gets its own folder; resuming into a folder skips what is done
    out2 = run_batch(recs, Settings(pipeline="matlab", create_video=False), log=lambda m: None)
    assert out2 != out and out2.name.endswith("_matlab")
    rows = []
    run_batch(recs, Settings(create_video=False), out, log=lambda m: None,
              on_file=lambda k, row: rows.append(row["status"]))
    assert rows == ["skipped", "skipped"]


def test_find_recordings_ignores_run_folders(tmp_path):
    from fnt.fiberPhotometry.batch import find_recordings
    _synthetic_doric(tmp_path / "a.doric")
    (tmp_path / "FNT_FiberPhotometry").mkdir()
    _synthetic_doric(tmp_path / "FNT_FiberPhotometry" / "copy.doric")
    assert [p.name for p in find_recordings([tmp_path])] == ["a.doric"]


def test_cli_info_and_process(tmp_path, capsys):
    from fnt.fiberPhotometry import cli
    rec = tmp_path / "rec.doric"
    _synthetic_doric(rec)
    assert cli.main(["info", str(rec)]) == 0
    assert "470 nm" in capsys.readouterr().out
    assert cli.main(["process", str(tmp_path), "--pipeline", "matlab", "--no-video",
                     "--notch", "3.32 9.94", "--zscore", "modified"]) == 0
    run = next((tmp_path / "FNT_FiberPhotometry").iterdir())
    rec_json = json.loads((run / "rec_settings.json").read_text())
    assert rec_json["settings"]["pipeline"] == "matlab"
    assert rec_json["settings"]["notch_hz"] == [3.32, 9.94]
    assert rec_json["settings"]["zscore_method"] == "modified"
    assert cli.main(["summary", str(run)]) == 0


def test_the_combined_video_has_one_output_frame_per_input_frame(tmp_path):
    cv2 = pytest.importorskip("cv2")
    import shutil
    if shutil.which("ffmpeg") is None:
        pytest.skip("ffmpeg not installed")
    from fnt.fiberPhotometry.video_overlay import Row, render_combined_video
    src = tmp_path / "in.mp4"
    w = cv2.VideoWriter(str(src), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 240))
    for k in range(45):
        w.write(np.full((240, 320, 3), k * 5 % 255, np.uint8))
    w.release()
    t = np.arange(0, 2, 0.01)
    rows = [Row("dF/F (%)", t, np.sin(t), (200, 100, 50), zero_line=True)]
    out = tmp_path / "out.mp4"
    assert render_combined_video(str(src), str(out), rows, window_s=2, row_h=80)
    cap = cv2.VideoCapture(str(out))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) == 45
    assert int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) == 240 + 80
    cap.release()
    assert not Path(str(out) + ".partial").exists()
