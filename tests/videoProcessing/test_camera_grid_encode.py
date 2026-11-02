"""Tests for mosaic command construction and the size/time estimates.

No ffmpeg is executed here -- these check the command we would emit. Runtime
behaviour of the filter graph was verified separately against real footage.
"""

import datetime

import pytest

from fnt.videoProcessing.camera_timeline import CameraTrack, Segment, build_timeline
from fnt.videoProcessing.camera_grid_encode import (
    EncodeSettings, GridLayout, best_grid, build_chunk_command, chunk_filename,
    collect_runs, estimate, human_size, human_time,
)

MID = datetime.datetime(2026, 2, 24, 0, 0, 0)


def track(name, n=12, missing=(), day="20260224000000"):
    each = 86400.0 / n
    segs = [Segment(f"/v/{name}_{day}({i}).mp4", day, i, each)
            for i in range(n) if i not in missing]
    return CameraTrack(name, f"/v/{name}", *build_timeline(segs))


# --- layout ----------------------------------------------------------------

def test_filled_places_cameras_left_to_right():
    g = GridLayout.filled(["cam1", "cam2", "cam3", "cam4"], 2, 2)
    assert g.assignments == {(0, 0): "cam1", (0, 1): "cam2",
                             (1, 0): "cam3", (1, 1): "cam4"}


def test_filled_drops_cameras_that_do_not_fit():
    """Choosing a smaller grid is how you export only some cameras."""
    g = GridLayout.filled(["cam1", "cam2", "cam3", "cam4", "cam5"], 2, 2)
    assert list(g.assignments.values()) == ["cam1", "cam2", "cam3", "cam4"]


def test_filled_leaves_spare_cells_empty():
    g = GridLayout.filled(["cam1", "cam2"], 2, 2)
    assert len(g.assignments) == 2
    assert (1, 1) not in g.assignments


@pytest.mark.parametrize("n,expected", [
    (1, (1, 1)), (2, (1, 2)), (4, (2, 2)), (5, (2, 3)),
    (6, (2, 3)), (9, (3, 3)), (12, (3, 4)), (16, (4, 4)), (25, (5, 5)),
])
def test_best_grid_picks_the_tightest_fit(n, expected):
    assert best_grid(n) == expected


def test_best_grid_caps_at_the_largest_shape():
    assert best_grid(100) == (5, 5)


def test_placing_a_camera_moves_it_rather_than_duplicating():
    g = GridLayout(2, 2, {(0, 0): "cam1"})
    g.place(1, 1, "cam1")
    assert g.assignments == {(1, 1): "cam1"}


def test_resizing_drops_cells_that_no_longer_exist():
    g = GridLayout(3, 3, {(2, 2): "cam4", (0, 0): "cam1"})
    g.resize(2, 2)
    assert g.assignments == {(0, 0): "cam1"}


def test_cell_size_is_even_for_yuv420p():
    g = GridLayout(3, 3)
    w, h = EncodeSettings(width=1920, height=1080).cell_size(g)
    assert w % 2 == 0 and h % 2 == 0
    assert (w, h) == (640, 360)


# --- run collection --------------------------------------------------------

def test_runs_cover_the_window_for_a_continuous_camera():
    t = track("cam1")
    runs = collect_runs(t, MID, MID + datetime.timedelta(hours=1))
    assert sum(r["duration"] for r in runs) == pytest.approx(3600, abs=1)
    assert runs[0]["offset"] == pytest.approx(0)


def test_run_offsets_are_relative_to_the_window_not_the_file():
    t = track("cam1")
    w0 = MID + datetime.timedelta(seconds=7200 + 100)
    runs = collect_runs(t, w0, w0 + datetime.timedelta(seconds=60))
    assert runs[0]["offset"] == pytest.approx(0)
    assert runs[0]["seek"] == pytest.approx(100)      # into that segment


def test_a_dropout_splits_the_window_into_two_runs():
    t = track("cam3", missing={6})
    runs = collect_runs(t, MID, MID + datetime.timedelta(hours=24))
    gap_at = 6 * 7200
    before = [r for r in runs if r["offset"] < gap_at]
    after = [r for r in runs if r["offset"] >= gap_at]
    assert before and after
    # nothing is scheduled inside the gap
    assert all(r["offset"] + r["duration"] <= gap_at + 1 for r in before)
    assert all(r["offset"] >= gap_at + 7200 - 1 for r in after)


def test_window_entirely_inside_a_gap_yields_no_runs():
    t = track("cam3", missing={6})
    w0 = MID + datetime.timedelta(seconds=6 * 7200 + 600)
    assert collect_runs(t, w0, w0 + datetime.timedelta(seconds=60)) == []


# --- command construction --------------------------------------------------

def build(tracks, layout, settings=None, minutes=10):
    settings = settings or EncodeSettings()
    w0 = MID
    w1 = w0 + datetime.timedelta(minutes=minutes)
    return build_chunk_command(tracks, layout, settings, w0, w1, "/out/x.mp4")


def test_command_has_one_input_per_run():
    tracks = {f"cam{i}": track(f"cam{i}") for i in range(1, 6)}
    layout = GridLayout.filled(list(tracks), 3, 3)
    cmd, n = build(tracks, layout)
    assert n == 5
    assert cmd.count("-i") == 5


def test_canvas_matches_requested_output_size():
    tracks = {"cam1": track("cam1")}
    cmd, _ = build(tracks, GridLayout(2, 2, {(0, 0): "cam1"}),
                   EncodeSettings(width=1280, height=720))
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "color=black:s=1280x720" in graph


def test_each_run_is_placed_at_its_cell_origin():
    tracks = {"cam4": track("cam4")}
    layout = GridLayout(3, 3, {(2, 2): "cam4"})
    cmd, _ = build(tracks, layout)
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "overlay=1280:720" in graph          # col 2 * 640, row 2 * 360


def test_overlay_is_gated_to_the_runs_time_range():
    """This gating is what makes a dropout show black: outside the range
    nothing is drawn, so the canvas shows through."""
    tracks = {"cam1": track("cam1")}
    cmd, _ = build(tracks, GridLayout(1, 1, {(0, 0): "cam1"}), minutes=5)
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "enable='between(t," in graph


def test_no_filler_inputs_are_generated_for_gaps():
    """Gaps are absence of overlay, not black files fed through the graph."""
    tracks = {"cam3": track("cam3", missing={6})}
    layout = GridLayout(1, 1, {(0, 0): "cam3"})
    w0 = MID + datetime.timedelta(seconds=6 * 7200 - 60)
    cmd, n = build_chunk_command(tracks, layout, EncodeSettings(),
                                 w0, w0 + datetime.timedelta(seconds=120),
                                 "/out/x.mp4")
    assert n == 1                                   # only the real footage
    assert "color=black" in cmd[cmd.index("-filter_complex") + 1]


def test_camera_with_no_footage_in_window_contributes_nothing():
    tracks = {"cam1": track("cam1"), "cam3": track("cam3", missing={6})}
    layout = GridLayout(2, 2, {(0, 0): "cam1", (0, 1): "cam3"})
    w0 = MID + datetime.timedelta(seconds=6 * 7200 + 600)
    cmd, n = build_chunk_command(tracks, layout, EncodeSettings(),
                                 w0, w0 + datetime.timedelta(seconds=60),
                                 "/out/x.mp4")
    assert n == 1                                   # cam1 only


def test_unassigned_camera_is_not_rendered():
    tracks = {"cam1": track("cam1"), "cam2": track("cam2")}
    _, n = build(tracks, GridLayout(2, 2, {(0, 0): "cam1"}))
    assert n == 1


def test_seek_uses_fast_then_precise_trim():
    """A bare -ss lands on a keyframe and would desync the cell."""
    tracks = {"cam1": track("cam1")}
    w0 = MID + datetime.timedelta(seconds=7200 + 500)
    cmd, _ = build_chunk_command(tracks, GridLayout(1, 1, {(0, 0): "cam1"}),
                                 EncodeSettings(), w0,
                                 w0 + datetime.timedelta(seconds=60), "/o.mp4")
    assert "-ss" in cmd
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "trim=start=" in graph


# --- overlay toggles -------------------------------------------------------

def test_overlay_toggles_are_independent():
    tracks = {"cam3": track("cam3", missing={6})}
    layout = GridLayout(1, 1, {(0, 0): "cam3"})

    def graph_for(**kw):
        opts = {"show_camera_labels": False, "show_no_signal": False,
                "show_clock": False}
        opts.update(kw)
        cmd, _ = build_chunk_command(tracks, layout, EncodeSettings(**opts), MID,
                                     MID + datetime.timedelta(hours=24), "/o.mp4")
        return cmd[cmd.index("-filter_complex") + 1]

    assert "drawtext" not in graph_for()
    assert "cam3" in graph_for(show_camera_labels=True)
    assert "NO SIGNAL" in graph_for(show_no_signal=True)
    assert "NO SIGNAL" not in graph_for(show_camera_labels=True)


def test_no_signal_only_drawn_for_cameras_that_actually_drop():
    tracks = {"cam1": track("cam1")}                 # no gaps
    cmd, _ = build_chunk_command(
        tracks, GridLayout(1, 1, {(0, 0): "cam1"}),
        EncodeSettings(show_no_signal=True, show_camera_labels=False),
        MID, MID + datetime.timedelta(hours=24), "/o.mp4")
    assert "NO SIGNAL" not in cmd[cmd.index("-filter_complex") + 1]


def test_label_text_is_escaped():
    tracks = {"cam:1": track("cam:1")}
    cmd, _ = build_chunk_command(
        tracks, GridLayout(1, 1, {(0, 0): "cam:1"}),
        EncodeSettings(show_camera_labels=True, show_no_signal=False),
        MID, MID + datetime.timedelta(minutes=1), "/o.mp4")
    assert "cam\\:1" in cmd[cmd.index("-filter_complex") + 1]


# --- encoder args ----------------------------------------------------------

def test_nvenc_uses_cq_not_crf():
    tracks = {"cam1": track("cam1")}
    cmd, _ = build(tracks, GridLayout(1, 1, {(0, 0): "cam1"}),
                   EncodeSettings(codec="h264_nvenc", crf=28))
    assert "-cq" in cmd and "28" in cmd
    assert "-crf" not in cmd


def test_x264_uses_crf_and_preset():
    tracks = {"cam1": track("cam1")}
    cmd, _ = build(tracks, GridLayout(1, 1, {(0, 0): "cam1"}),
                   EncodeSettings(codec="libx264", crf=20, preset="veryfast"))
    assert cmd[cmd.index("-crf") + 1] == "20"
    assert cmd[cmd.index("-preset") + 1] == "veryfast"


# --- estimates -------------------------------------------------------------

def test_estimate_reproduces_the_measured_reference():
    """1080p30 x264 crf23 five cameras measured ~0.19 Mbps."""
    e = estimate(EncodeSettings(), GridLayout(3, 3), 3600, 5)
    assert e["bitrate_mbps"] == pytest.approx(0.19, rel=0.15)


def test_lower_fps_reduces_size_and_time():
    layout = GridLayout(3, 3)
    hi = estimate(EncodeSettings(fps=30), layout, 3600, 5)
    lo = estimate(EncodeSettings(fps=15), layout, 3600, 5)
    assert lo["size_bytes"] < hi["size_bytes"]
    assert lo["encode_seconds"] < hi["encode_seconds"]


def test_higher_crf_shrinks_output():
    layout = GridLayout(3, 3)
    assert (estimate(EncodeSettings(crf=28), layout, 3600, 5)["size_bytes"]
            < estimate(EncodeSettings(crf=23), layout, 3600, 5)["size_bytes"])


def test_estimate_scales_with_duration():
    layout = GridLayout(3, 3)
    one = estimate(EncodeSettings(), layout, 3600, 5)
    ten = estimate(EncodeSettings(), layout, 36000, 5)
    assert ten["size_bytes"] == pytest.approx(one["size_bytes"] * 10, rel=0.01)


def test_full_trial_estimate_is_in_the_measured_ballpark():
    """243h at the measured settings came to roughly 21 GB."""
    e = estimate(EncodeSettings(), GridLayout(3, 3), 243 * 3600, 5)
    assert 12e9 < e["size_bytes"] < 35e9


# --- formatting ------------------------------------------------------------

@pytest.mark.parametrize("n,expect", [(512, "512.0 B"), (2048, "2.0 KB"),
                                      (5 * 1024**3, "5.0 GB")])
def test_human_size(n, expect):
    assert human_size(n) == expect


@pytest.mark.parametrize("s,expect", [(90, "1m"), (3700, "1h 1m"), (200000, "2d 7h")])
def test_human_time(s, expect):
    assert human_time(s) == expect


def test_daily_chunks_are_named_by_date():
    assert chunk_filename("T006", MID, "daily", 0, 11) == "T006_20260224_grid.mp4"


def test_continuous_single_chunk_has_no_index():
    assert chunk_filename("T006", MID, "continuous", 0, 1) == "T006_grid.mp4"


# --- audio and grayscale --------------------------------------------------

def test_audio_removed_by_default():
    tracks = {"cam1": track("cam1")}
    cmd, _ = build(tracks, GridLayout(1, 1, {(0, 0): "cam1"}))
    assert "-an" in cmd
    assert "-c:a" not in cmd


def test_grayscale_applied_once_to_the_composite():
    tracks = {f"cam{i}": track(f"cam{i}") for i in range(1, 6)}
    layout = GridLayout.filled(list(tracks), 3, 3)
    cmd, _ = build(tracks, layout, EncodeSettings(grayscale=True))
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert graph.count("format=gray") == 1     # not once per camera


def test_no_grayscale_filter_when_disabled():
    tracks = {"cam1": track("cam1")}
    cmd, _ = build(tracks, GridLayout(1, 1, {(0, 0): "cam1"}),
                   EncodeSettings(grayscale=False))
    assert "format=gray" not in cmd[cmd.index("-filter_complex") + 1]


def test_audio_chain_delays_each_run_to_its_offset(monkeypatch):
    """Audio must track the video, including silence across a dropout."""
    import fnt.videoProcessing.camera_grid_encode as enc
    monkeypatch.setattr(enc, "has_audio", lambda p: True)
    tracks = {"cam3": track("cam3", missing={6})}
    cmd, _ = enc.build_chunk_command(
        tracks, GridLayout(1, 1, {(0, 0): "cam3"}),
        EncodeSettings(remove_audio=False, audio_source="cam3"),
        MID, MID + datetime.timedelta(hours=24), "/o.mp4")
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert "adelay=delays=0:all=1" in graph            # first run at t=0
    assert f"adelay=delays={7 * 7200 * 1000}:all=1" in graph   # after the gap
    # contiguous segments merge, so a single dropout yields exactly two runs
    assert "amix=inputs=2" in graph
    assert "-c:a" in cmd and "-an" not in cmd


def test_audio_takes_only_the_chosen_camera(monkeypatch):
    import fnt.videoProcessing.camera_grid_encode as enc
    monkeypatch.setattr(enc, "has_audio", lambda p: True)
    tracks = {"cam1": track("cam1"), "cam2": track("cam2")}
    layout = GridLayout(2, 2, {(0, 0): "cam1", (0, 1): "cam2"})
    cmd, _ = enc.build_chunk_command(
        tracks, layout, EncodeSettings(remove_audio=False, audio_source="cam2"),
        MID, MID + datetime.timedelta(minutes=30), "/o.mp4")
    graph = cmd[cmd.index("-filter_complex") + 1]
    assert graph.count("atrim") == 1                  # cam2 only


def test_silent_source_falls_back_to_no_audio(monkeypatch):
    """A missing audio stream would abort the encode, so it is detected."""
    import fnt.videoProcessing.camera_grid_encode as enc
    monkeypatch.setattr(enc, "has_audio", lambda p: False)
    tracks = {"cam1": track("cam1")}
    cmd, _ = enc.build_chunk_command(
        tracks, GridLayout(1, 1, {(0, 0): "cam1"}),
        EncodeSettings(remove_audio=False, audio_source="cam1"),
        MID, MID + datetime.timedelta(minutes=10), "/o.mp4")
    assert "-an" in cmd
    assert "atrim" not in cmd[cmd.index("-filter_complex") + 1]


# --- segment merging ------------------------------------------------------

def test_contiguous_segments_merge_into_one_run():
    """A full day of 12 files must become ONE ffmpeg input, not twelve."""
    runs = collect_runs(track("cam1", n=12), MID, MID + datetime.timedelta(hours=24))
    assert len(runs) == 1
    assert len(runs[0]["paths"]) == 12


def test_dropout_splits_the_merge():
    runs = collect_runs(track("cam3", n=12, missing={6}), MID,
                        MID + datetime.timedelta(hours=24))
    assert len(runs) == 2
    assert [len(r["paths"]) for r in runs] == [6, 5]
    assert runs[1]["offset"] == pytest.approx(7 * 7200, abs=1)


def test_merged_run_keeps_seek_into_its_first_file():
    w0 = MID + datetime.timedelta(seconds=7200 + 100)
    runs = collect_runs(track("cam1", n=12), w0, w0 + datetime.timedelta(hours=5))
    assert runs[0]["seek"] == pytest.approx(100)
    assert runs[0]["paths"][0].endswith("(1).mp4")


def test_full_day_five_cameras_is_a_handful_of_inputs():
    """The whole point of merging: ~64 inputs measured 2.1x realtime."""
    tracks = {f"cam{i}": track(f"cam{i}", n=12) for i in range(1, 6)}
    tracks["cam3"] = track("cam3", n=21, missing={6})
    layout = GridLayout.filled(sorted(tracks), 3, 3)
    _, n = build_chunk_command(tracks, layout, EncodeSettings(), MID,
                               MID + datetime.timedelta(hours=24), "/o.mp4")
    assert n <= 7          # 4 contiguous cameras + cam3 split in two


def test_multi_file_run_uses_the_concat_demuxer(tmp_path):
    tracks = {"cam1": track("cam1", n=12)}
    cmd, _ = build_chunk_command(
        tracks, GridLayout(1, 1, {(0, 0): "cam1"}), EncodeSettings(), MID,
        MID + datetime.timedelta(hours=24), "/o.mp4", work_dir=str(tmp_path))
    assert "-f" in cmd and "concat" in cmd and "-safe" in cmd
    listed = list(tmp_path.glob("run*.txt"))
    assert len(listed) == 1
    assert listed[0].read_text().count("file ") == 12


def test_without_work_dir_falls_back_to_the_first_file(tmp_path):
    """Previews and samples sit inside one segment, so no list file is needed."""
    tracks = {"cam1": track("cam1", n=12)}
    cmd, _ = build_chunk_command(
        tracks, GridLayout(1, 1, {(0, 0): "cam1"}), EncodeSettings(), MID,
        MID + datetime.timedelta(hours=24), "/o.mp4")
    assert "concat" not in cmd


def test_concat_list_escapes_quotes(tmp_path):
    from fnt.videoProcessing.camera_grid_encode import write_concat_list
    p = write_concat_list(["C:/it's/a.mp4"], str(tmp_path / "l.txt"))
    # a literal quote must reach the demuxer as  '\''  inside the quoted path
    q, bs = chr(39), chr(92)
    expected = "file " + q + "C:/it" + q + bs + q + q + "s/a.mp4" + q + "\n"
    assert open(p, encoding="utf-8").read() == expected


def test_estimate_matches_the_measured_full_day_encode():
    """A real 24h five-camera chunk measured 8.14x realtime and 0.183 Mbps."""
    e = estimate(EncodeSettings(), GridLayout(3, 3), 86400, 5)
    assert e["realtime_factor"] == pytest.approx(8.14, rel=0.15)
    assert e["bitrate_mbps"] == pytest.approx(0.183, rel=0.25)
    # 24h chunk took just under 3 hours
    assert 2.5 * 3600 < e["encode_seconds"] < 3.6 * 3600


# --- queued jobs capture their settings ------------------------------------

def test_layout_copy_is_independent():
    a = GridLayout.filled(["cam1", "cam2"], 2, 2)
    b = a.copy()
    a.place(1, 1, "cam3")
    assert "cam3" not in b.assignments


def test_settings_copy_is_independent():
    a = EncodeSettings(fps=30, grayscale=False)
    b = a.copy()
    a.fps, a.grayscale = 15, True
    assert (b.fps, b.grayscale) == (30, False)


def test_settings_copy_carries_every_field():
    a = EncodeSettings(fps=15, grayscale=True, remove_audio=False,
                       audio_source="cam2", chunk_mode="continuous", crf=28)
    b = a.copy()
    assert b.__dict__ == a.__dict__


def test_calibration_offset_reaches_the_encode_path():
    """Preview and export must shift together, or the sample lies."""
    t = track("cam1")
    w0 = MID + datetime.timedelta(seconds=7200 + 100)
    w1 = w0 + datetime.timedelta(seconds=60)
    plain = collect_runs(t, w0, w1)[0]["seek"]
    t.clock_offset = 1.5
    shifted = collect_runs(t, w0, w1)[0]["seek"]
    # positive offset = later footage, matching CameraTrack.locate
    assert shifted == pytest.approx(plain + 1.5)


def test_encode_and_preview_agree_on_direction():
    t = track("cam1")
    when = MID + datetime.timedelta(seconds=7200 + 100)
    t.clock_offset = 0.8
    _, preview_seek = t.locate(when)
    run = collect_runs(t, when, when + datetime.timedelta(seconds=30))[0]
    assert run["seek"] == pytest.approx(preview_seek)


# --- local output staging --------------------------------------------------

def test_staging_is_on_by_default():
    assert EncodeSettings().stage_output_locally is True


def test_staging_setting_survives_a_copy():
    s = EncodeSettings(stage_output_locally=False)
    assert s.copy().stage_output_locally is False


# --- "both": daily files plus a joined trial video -------------------------

def test_both_mode_names_its_daily_files_by_date():
    assert chunk_filename("T005", MID, "both", 0, 11) == "T005_20260224_grid.mp4"


def test_joined_file_uses_the_plain_trial_name():
    """The joined video must not collide with any daily file."""
    daily = chunk_filename("T005", MID, "both", 0, 11)
    joined = chunk_filename("T005", MID, "continuous", 0, 1)
    assert joined == "T005_grid.mp4"
    assert joined != daily


def test_concat_command_stream_copies(tmp_path):
    """Joining must not re-encode -- that is the whole point of 'both'."""
    from fnt.videoProcessing.camera_grid_encode import build_concat_command
    cmd = build_concat_command(["/a/d1.mp4", "/a/d2.mp4"], "/a/all.mp4",
                               str(tmp_path / "join.txt"))
    assert cmd[cmd.index("-c") + 1] == "copy"
    assert "-crf" not in cmd and "libx264" not in cmd
    assert "concat" in cmd


def test_concat_list_holds_the_chunks_in_order(tmp_path):
    from fnt.videoProcessing.camera_grid_encode import build_concat_command
    listing = tmp_path / "join.txt"
    build_concat_command(["/a/d1.mp4", "/a/d2.mp4", "/a/d3.mp4"],
                         "/a/all.mp4", str(listing))
    lines = [l for l in listing.read_text().splitlines() if l.strip()]
    assert [l.split("/")[-1].rstrip("'") for l in lines] == \
        ["d1.mp4", "d2.mp4", "d3.mp4"]


def test_both_mode_estimates_two_copies_of_the_footage():
    layout = GridLayout(3, 3)
    one = estimate(EncodeSettings(chunk_mode="daily"), layout, 86400, 5)
    two = estimate(EncodeSettings(chunk_mode="both"), layout, 86400, 5)
    assert two["size_bytes"] == pytest.approx(one["size_bytes"] * 2, rel=0.01)


def test_both_mode_costs_little_extra_time():
    """Joining is a copy, so it must not double the estimate."""
    layout = GridLayout(3, 3)
    one = estimate(EncodeSettings(chunk_mode="daily"), layout, 86400, 5)
    two = estimate(EncodeSettings(chunk_mode="both"), layout, 86400, 5)
    assert two["encode_seconds"] < one["encode_seconds"] * 1.3
