"""Tests for the camera-grid synchronisation engine.

These build Segments directly rather than touching real media, so the whole
suite runs without ffmpeg, Qt, or the network share. The scenarios mirror
patterns measured in real trial data (see camera_timeline's module docstring).
"""

import datetime

import pytest

from fnt.videoProcessing.camera_timeline import (
    CameraTrack, Gap, Segment, TrialTimeline, build_timeline,
    infer_camera_name, parse_segment_name,
)

MID = datetime.datetime(2026, 2, 24, 0, 0, 0)


def seg(day, seq, duration, name=None):
    return Segment(name or f"/v/Cam_{day}({seq}).mp4", day, seq, duration)


def full_day(day="20260224000000", n=12, missing=()):
    """A day of `n` equal segments summing to 86400s, minus any `missing` seqs."""
    each = 86400.0 / n
    return [seg(day, i, each) for i in range(n) if i not in missing]


# --- filename parsing ------------------------------------------------------

@pytest.mark.parametrize("name,expected", [
    ("VoleCosm_Camera3_{G}_20260224000000(007)_processed.mp4", ("20260224000000", 7)),
    ("VoleCosm_Camera3_{G}_20260224000000_processed.mp4", ("20260224000000", 0)),
    ("VoleCosm_Camera1_{G}_20260220120000(001).avi", ("20260220120000", 1)),
    ("VoleCosm_Camera1_{G}_20260220120000.avi", ("20260220120000", 0)),
])
def test_parses_stamp_and_sequence(name, expected):
    assert parse_segment_name(name) == expected


def test_unsuffixed_file_is_first_segment_of_day():
    """The file with no (seq) sorts before (001) -- not alphabetically after."""
    assert parse_segment_name("x_20260224000000_processed.mp4")[1] == 0
    assert parse_segment_name("x_20260224000000(001)_processed.mp4")[1] == 1


@pytest.mark.parametrize("name", [
    "notes.txt", "concatenated_output.mp4", "Camera3_summary.mp4",
])
def test_ignores_non_segment_files(name):
    assert parse_segment_name(name) is None


# --- the contiguous case ---------------------------------------------------

def test_contiguous_day_starts_at_midnight_and_has_no_gaps():
    placed, gaps = build_timeline(full_day())
    assert gaps == []
    assert placed[0].start == MID
    assert placed[-1].end == MID + datetime.timedelta(seconds=86400)


def test_segment_start_is_cumulative_duration():
    placed, _ = build_timeline(full_day(n=4))     # 21600s each
    assert [p.start for p in placed] == [
        MID + datetime.timedelta(seconds=k * 21600) for k in range(4)]


def test_uneven_segment_lengths_still_chain_exactly():
    """Real NVR splits are size-based, so durations vary wildly."""
    durs = [7622.0, 7920.0, 7174.0, 7730.0, 7868.0, 7812.0,
            7812.0, 7962.0, 7764.0, 7862.0, 7790.0, 1084.0]
    placed, gaps = build_timeline(
        [seg("20260224000000", i, d) for i, d in enumerate(durs)])
    assert gaps == []
    running = 0.0
    for p, d in zip(placed, durs):
        assert p.start == MID + datetime.timedelta(seconds=running)
        running += d


# --- dropouts --------------------------------------------------------------

def test_missing_sequence_becomes_a_gap_of_the_days_shortfall():
    """cam3's real pattern: seq 6 absent, day short by that segment's length."""
    segs = full_day(n=12, missing={6})       # 11 x 7200s = 79200s, short 7200s
    placed, gaps = build_timeline(segs)
    assert len(gaps) == 1
    assert gaps[0].seq == 6
    assert gaps[0].duration == pytest.approx(7200, abs=1)


def test_footage_after_a_gap_keeps_its_true_wall_clock_time():
    """The whole point: later segments must not slide earlier."""
    placed, _ = build_timeline(full_day(n=12, missing={6}))
    after = [p for p in placed if p.seq == 7][0]
    # 7 segments' worth of time has genuinely elapsed before seq 7 begins
    assert after.start == MID + datetime.timedelta(seconds=7 * 7200)


def test_gap_occupies_the_hole_between_neighbours():
    placed, gaps = build_timeline(full_day(n=12, missing={6}))
    before = [p for p in placed if p.seq == 5][0]
    after = [p for p in placed if p.seq == 7][0]
    assert gaps[0].start == before.end
    assert gaps[0].end == after.start


def test_multiple_missing_sequences_split_the_deficit():
    placed, gaps = build_timeline(full_day(n=12, missing={4, 8}))
    assert [g.seq for g in gaps] == [4, 8]
    assert sum(g.duration for g in gaps) == pytest.approx(14400, abs=2)


def test_short_interior_day_with_no_missing_sequence_is_flagged_unlocated():
    """Observed once in real data: ~19 min lost with every file present."""
    day1 = full_day("20260223000000", n=12)
    day1[-1].duration -= 1132                      # short, but seqs complete
    segs = day1 + full_day("20260224000000", n=12)
    _, gaps = build_timeline(segs)
    assert len(gaps) == 1
    assert gaps[0].unlocated is True
    assert gaps[0].duration == pytest.approx(1132, abs=1)


def test_short_final_day_is_not_reported_as_loss():
    """On the last day, 'stopped recording' and 'lost footage' are
    indistinguishable, so no gap is invented."""
    segs = full_day("20260223000000", n=12) + full_day("20260224000000", n=12)
    segs[-1].duration -= 1132
    _, gaps = build_timeline(segs)
    assert gaps == []


def test_small_shortfall_is_tolerated_as_rounding():
    """Real days land ~1-2s under 86400; that must not become a gap."""
    segs = full_day(n=12)
    segs[-1].duration -= 2
    _, gaps = build_timeline(segs)
    assert gaps == []


# --- partial days ----------------------------------------------------------

def test_midday_start_expects_only_the_rest_of_the_day():
    """A 12:00 trial start must not read as a 12-hour gap."""
    day = "20260220120000"
    segs = [seg(day, i, 43200.0 / 6) for i in range(6)]   # 12h total
    placed, gaps = build_timeline(segs)
    assert gaps == []
    assert placed[0].start == datetime.datetime(2026, 2, 20, 12, 0, 0)


def test_missing_sequence_on_a_midday_start_day_is_still_caught():
    day = "20260220120000"
    segs = [seg(day, i, 43200.0 / 6) for i in range(6) if i != 3]
    _, gaps = build_timeline(segs)
    assert [g.seq for g in gaps] == [3]
    assert gaps[0].duration == pytest.approx(7200, abs=1)


def test_final_day_shortfall_is_not_treated_as_a_gap():
    """Recording simply stopped; that isn't lost footage."""
    segs = full_day("20260224000000") + [
        seg("20260225000000", i, 3600.0) for i in range(5)]   # only 5h on day 2
    _, gaps = build_timeline(segs)
    assert gaps == []


# --- locate ----------------------------------------------------------------

def test_locate_returns_offset_into_the_right_segment():
    track = CameraTrack("cam1", "/v", *build_timeline(full_day(n=12)))
    path, off = track.locate(MID + datetime.timedelta(seconds=7200 + 42))
    assert "(1)" in path
    assert off == pytest.approx(42)


def test_locate_inside_a_gap_reports_no_footage():
    track = CameraTrack("cam3", "/v", *build_timeline(full_day(n=12, missing={6})))
    inside = MID + datetime.timedelta(seconds=6 * 7200 + 100)
    assert track.locate(inside) == (None, None)


def test_locate_outside_the_recording_reports_no_footage():
    track = CameraTrack("cam1", "/v", *build_timeline(full_day(n=12)))
    assert track.locate(MID - datetime.timedelta(hours=1)) == (None, None)
    assert track.locate(MID + datetime.timedelta(days=2)) == (None, None)


def test_locate_is_exact_at_segment_boundaries():
    track = CameraTrack("cam1", "/v", *build_timeline(full_day(n=12)))
    path, off = track.locate(MID + datetime.timedelta(seconds=7200))
    assert off == pytest.approx(0)          # start of seq 1, not end of seq 0


# --- track + trial aggregates ----------------------------------------------

def test_track_reports_footage_span_and_gap_totals():
    track = CameraTrack("cam3", "/v", *build_timeline(full_day(n=12, missing={6})))
    assert track.footage_seconds == pytest.approx(79200, abs=1)
    assert track.span_seconds == pytest.approx(86400, abs=1)
    assert track.gap_seconds == pytest.approx(7200, abs=1)


def test_trial_window_spans_all_cameras():
    a = CameraTrack("cam1", "/a", *build_timeline(full_day()))
    b = CameraTrack("cam2", "/b", *build_timeline(
        [seg("20260225000000", i, 7200.0) for i in range(12)]))
    trial = TrialTimeline([a, b])
    assert trial.start == MID
    assert trial.duration_seconds == pytest.approx(2 * 86400, abs=2)


def test_cameras_with_no_segments_are_dropped_from_the_trial():
    good = CameraTrack("cam1", "/a", *build_timeline(full_day()))
    empty = CameraTrack("cam9", "/b", [], [])
    assert TrialTimeline([good, empty]).tracks == [good]


# --- chunking --------------------------------------------------------------

def test_daily_chunks_break_on_midnight():
    track = CameraTrack("cam1", "/v", *build_timeline(
        full_day("20260220120000", n=6) +          # starts 12:00
        [seg("20260221000000", i, 7200.0) for i in range(12)]))
    chunks = TrialTimeline([track]).chunks("daily")
    assert chunks[0][0] == datetime.datetime(2026, 2, 20, 12, 0)
    assert chunks[0][1] == datetime.datetime(2026, 2, 21, 0, 0)
    assert chunks[1][0] == datetime.datetime(2026, 2, 21, 0, 0)


def test_continuous_mode_is_a_single_chunk():
    track = CameraTrack("cam1", "/v", *build_timeline(full_day()))
    chunks = TrialTimeline([track]).chunks("continuous")
    assert len(chunks) == 1
    assert chunks[0] == (track.start, track.end)


def test_daily_chunks_tile_the_window_without_overlap_or_hole():
    track = CameraTrack("cam1", "/v", *build_timeline(
        full_day("20260220120000", n=6) +
        [seg(f"2026022{d}000000", i, 7200.0) for d in (1, 2) for i in range(12)]))
    chunks = TrialTimeline([track]).chunks("daily")
    for (_, prev_end), (next_start, _) in zip(chunks, chunks[1:]):
        assert prev_end == next_start
    assert chunks[0][0] == track.start
    assert chunks[-1][1] == track.end


# --- camera naming ---------------------------------------------------------

def test_camera_name_from_folder_when_specific():
    assert infer_camera_name("/trials/T006/cam3") == "cam3"
    assert infer_camera_name("/trials/T001/cam7") == "cam7"


def test_camera_name_falls_back_to_filename_for_flat_folders():
    """T011 keeps segments loose in the trial folder, with no cam* directory."""
    segs = [seg("20260617114000", 0, 60.0,
                name="/t/T011/VoleCosm_Camera3_{G}_20260617114000.mp4")]
    assert infer_camera_name("/t/T011_prairie_anosmic_saline", segs) == "Camera3"


def test_coverage_marks_gap_buckets_empty():
    track = CameraTrack("cam3", "/v", *build_timeline(full_day(n=12, missing={6})))
    cov = track.coverage(track.start, track.end, buckets=12)
    assert cov[0] == pytest.approx(1.0, abs=0.05)
    assert cov[6] == pytest.approx(0.0, abs=0.05)     # the missing segment


# --- offload windows other than midnight ----------------------------------

def test_six_am_offload_window_invents_no_gap():
    """Midnight-to-midnight is the operator's habit, not a recorder property.

    A 06:00-to-06:00 offload gives buckets that each hold a full 24h. Judging
    them against the remainder of the calendar day would fabricate an 18-hour
    gap in every bucket.
    """
    segs = ([seg("20260220060000", i, 86400.0 / 12) for i in range(12)] +
            [seg("20260221060000", i, 86400.0 / 12) for i in range(12)])
    placed, gaps = build_timeline(segs)
    assert gaps == []
    assert placed[0].start == datetime.datetime(2026, 2, 20, 6, 0)
    assert placed[11].end == datetime.datetime(2026, 2, 21, 6, 0)


def test_bucket_length_comes_from_the_next_bucket_stamp():
    """A 12:00 start followed by a midnight bucket spans 12h, not 24h."""
    segs = ([seg("20260220120000", i, 43200.0 / 6) for i in range(6)] +
            [seg("20260221000000", i, 86400.0 / 12) for i in range(12)])
    _, gaps = build_timeline(segs)
    assert gaps == []


def test_dropout_detected_inside_a_non_midnight_window():
    segs = ([seg("20260220060000", i, 86400.0 / 12) for i in range(12) if i != 5] +
            [seg("20260221060000", i, 86400.0 / 12) for i in range(12)])
    _, gaps = build_timeline(segs)
    assert [g.seq for g in gaps] == [5]
    assert gaps[0].duration == pytest.approx(7200, abs=2)


# --- profiles --------------------------------------------------------------

def test_alphabetical_profile_chains_without_inventing_gaps():
    """For recorders we do not model: continuous playback, no gap detection."""
    segs = full_day(n=12, missing={6})
    placed, gaps = build_timeline(segs, profile="alphabetical")
    assert gaps == []
    assert len(placed) == 11
    # laid end to end, so the missing file simply isn't represented
    assert placed[-1].end == MID + datetime.timedelta(seconds=11 * 7200)


def test_alphabetical_profile_preserves_order():
    segs = full_day(n=5)
    placed, _ = build_timeline(segs, profile="alphabetical")
    assert [p.seq for p in placed] == [0, 1, 2, 3, 4]


def test_viewtron_profile_is_the_default():
    from fnt.videoProcessing.camera_timeline import DEFAULT_PROFILE, PROFILES
    assert DEFAULT_PROFILE == "viewtron"
    assert set(PROFILES) == {"viewtron", "alphabetical"}


def test_profiles_differ_only_where_footage_is_missing():
    """With no dropouts both profiles must agree exactly."""
    a, _ = build_timeline(full_day(n=12), profile="viewtron")
    b, _ = build_timeline(full_day(n=12), profile="alphabetical")
    assert [s.start for s in a] == [s.start for s in b]


# --- camera auto-detection -------------------------------------------------

from fnt.videoProcessing.camera_timeline import camera_token, discover_cameras


@pytest.mark.parametrize("name,expected", [
    ("VoleCosm_Camera3_{G}_20260224000000(007)_processed.mp4", "Camera3"),
    ("VoleCosm_Camera12_{G}_20260224000000.avi", "Camera12"),
    ("rig_cam_7_20260224000000.mp4", "cam7"),
    ("Trial_CAM04_20260224000000.mp4", "CAM04"),
])
def test_camera_token_extracted_from_filename(name, expected):
    assert camera_token(name) == expected


def test_camera_token_absent_when_unlabelled():
    assert camera_token("recording_20260224000000.mp4") is None


def _write(folder, names):
    import os
    os.makedirs(folder, exist_ok=True)
    for n in names:
        open(os.path.join(folder, n), "wb").close()


def test_discovers_cameras_from_per_camera_subfolders(tmp_path):
    for i in (1, 2):
        _write(str(tmp_path / f"cam{i}"),
               [f"V_Camera{i}_{{G}}_20260224000000.mp4",
                f"V_Camera{i}_{{G}}_20260224000000(001).mp4"])
    groups = discover_cameras(str(tmp_path))
    assert sorted(groups) == ["Camera1", "Camera2"]
    assert all(len(v) == 2 for v in groups.values())


def test_discovers_cameras_from_a_flat_folder(tmp_path):
    """The layout that would otherwise force sorting files by hand."""
    _write(str(tmp_path), [
        "V_Camera3_{G}_20260224000000.mp4",
        "V_Camera3_{G}_20260224000000(001).mp4",
        "V_Camera4_{G}_20260224000000.mp4",
    ])
    groups = discover_cameras(str(tmp_path))
    assert sorted(groups) == ["Camera3", "Camera4"]
    assert len(groups["Camera3"]) == 2


def test_discovers_a_mixed_layout(tmp_path):
    """Real trial: some cameras in subfolders, others loose, plus empty dirs."""
    _write(str(tmp_path / "cam1"), ["V_Camera1_{G}_20260224000000.mp4"])
    _write(str(tmp_path / "cam9"), [])                 # empty leftover folder
    _write(str(tmp_path), ["V_Camera3_{G}_20260224000000.mp4"])
    groups = discover_cameras(str(tmp_path))
    assert sorted(groups) == ["Camera1", "Camera3"]    # empty dir contributes nothing


def test_grouping_follows_the_filename_not_the_folder(tmp_path):
    """A misfiled camera must not be mislabelled by its folder."""
    _write(str(tmp_path / "cam1"), ["V_Camera7_{G}_20260224000000.mp4"])
    assert list(discover_cameras(str(tmp_path))) == ["Camera7"]


def test_unlabelled_files_fall_back_to_their_folder(tmp_path):
    _write(str(tmp_path / "leftcam"), ["recording_20260224000000.mp4"])
    assert list(discover_cameras(str(tmp_path))) == ["leftcam"]


def test_non_segment_files_are_ignored(tmp_path):
    _write(str(tmp_path), ["V_Camera1_{G}_20260224000000.mp4",
                           "notes.txt", "Camera1_summary.mp4"])
    groups = discover_cameras(str(tmp_path))
    assert len(groups["Camera1"]) == 1


def test_discovery_does_not_descend_past_camera_folders(tmp_path):
    _write(str(tmp_path / "cam1"), ["V_Camera1_{G}_20260224000000.mp4"])
    _write(str(tmp_path / "cam1" / "proc"),
           ["V_Camera1_{G}_20260224000000_processed.mp4"])
    groups = discover_cameras(str(tmp_path), max_depth=1)
    assert len(groups["Camera1"]) == 1      # the proc/ copy is not swept in


# --- manual clock calibration ---------------------------------------------

def _track(n=12):
    return CameraTrack("cam1", "/v", *build_timeline(full_day(n=n)))


def test_positive_offset_shows_later_footage():
    """'>' in the calibration dialog must advance the burnt-in clock.

    The two halves of this once disagreed -- nudge added to the offset while
    locate subtracted it -- so stepping back moved the timestamp forward.
    """
    t = _track()
    when = MID + datetime.timedelta(seconds=7200 + 100)
    _, before = t.locate(when)
    t.clock_offset = 1.0
    _, after = t.locate(when)
    assert after == pytest.approx(before + 1.0)


def test_negative_offset_shows_earlier_footage():
    t = _track()
    when = MID + datetime.timedelta(seconds=7200 + 100)
    _, before = t.locate(when)
    t.clock_offset = -1.0
    _, after = t.locate(when)
    assert after == pytest.approx(before - 1.0)


def test_zero_offset_changes_nothing():
    a = _track(); b = _track(); b.clock_offset = 0.0
    when = MID + datetime.timedelta(seconds=5000)
    assert a.locate(when) == b.locate(when)


def test_offset_survives_a_copy_but_is_independent():
    t = _track(); t.clock_offset = 0.75
    c = t.copy()
    assert c.clock_offset == 0.75
    t.clock_offset = -2.0
    assert c.clock_offset == 0.75          # a queued job keeps its calibration


def test_offset_can_move_across_a_segment_boundary():
    """A nudge near a boundary must cross into the neighbouring file."""
    t = _track()
    just_after = MID + datetime.timedelta(seconds=7200, milliseconds=100)
    path_a, _ = t.locate(just_after)
    t.clock_offset = -1.0                  # step back over the boundary
    path_b, _ = t.locate(just_after)
    assert path_a != path_b
