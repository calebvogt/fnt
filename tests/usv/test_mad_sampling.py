"""Tests for the recording sampler.

The properties worth pinning down are the ones that fail *quietly*: a draw that
returns the right count but clusters in one channel, or one whose "20 more"
pass re-offers the same twenty files. Counting alone would pass all of those,
so the assertions here are about spread and disjointness, not just totals.
"""
import os

import pytest

from fnt.usv.usv_detector.mad_sampling import (
    SampleSpec, allocate, group_paths, natural_key, parse_channel,
    sample_paths, stride_pick,
)


def _tree(folder, channels, per_channel, start=1):
    """Filenames shaped like the real sets: ``T001_PAH_ch2_T0000042.wav``."""
    return [os.path.join(folder, f"{os.path.basename(folder)}_{ch}_T{i:07d}.wav")
            for ch in channels
            for i in range(start, start + per_channel)]


# ---------------------------------------------------------------- channels
@pytest.mark.parametrize("name, expect", [
    ("T001_PAH_ch1T2021-09-24_15-03-11_0000045.wav", "ch1"),
    ("T004_OB_ch1_T2021-10-22_12-14-01_0000001.wav", "ch1"),
    ("T005_OB_ch4_T0000952.wav", "ch4"),
    ("T0016_PAH_ch2_T0000002.wav", "ch2"),
    ("rig_CH03_T0000002.wav", "ch3"),        # case and zero padding normalize
    ("mouse_ch12_T0000002.wav", "ch12"),
    ("single_mic_recording.wav", None),
])
def test_parse_channel(name, expect):
    assert parse_channel(name) == expect


def test_channel_token_is_not_matched_inside_a_word():
    """A bare ``ch\\d+`` would claim the ch4 in a genotype name."""
    assert parse_channel("T020_Arch4_T0000001.wav") is None
    assert parse_channel("T020_Arch4_ch2_T0000001.wav") == "ch2"


def test_natural_key_orders_unpadded_sequences():
    names = ["f_9.wav", "f_10.wav", "f_1.wav"]
    assert [os.path.basename(p) for p in sorted(names, key=natural_key)] == [
        "f_1.wav", "f_9.wav", "f_10.wav"]


# ---------------------------------------------------------------- allocate
def test_allocate_is_proportional_and_conserves_the_total():
    take = allocate([847, 847, 845, 847], 20)
    assert sum(take) == 20
    assert take == [5, 5, 5, 5]


def test_allocate_handles_unequal_groups():
    """T014's real shape: 610 and 548, which must not both round to 10."""
    take = allocate([610, 548], 20)
    assert sum(take) == 20
    assert take == [11, 9]


def test_allocate_never_exceeds_a_groups_size():
    take = allocate([3, 1000], 5)
    assert take[0] <= 3
    assert sum(take) == 5


def test_allocate_represents_a_tiny_group_rather_than_starving_it():
    """Strict proportionality gives a 3-file trial zero picks beside a
    1000-file one, which defeats the point of sampling across trials."""
    take = allocate([3, 1000], 100)
    assert take[0] >= 1
    assert sum(take) == 100


def test_allocate_keeps_proportionality_when_nobody_is_starved():
    """The floor must not distort the ordinary case: 610/548 stays 11/9."""
    assert allocate([610, 548], 20) == [11, 9]
    assert allocate([847, 847, 845, 847], 20) == [5, 5, 5, 5]


def test_allocate_falls_back_to_proportional_when_picks_are_scarcer_than_groups():
    take = allocate([100, 100, 100], 2)
    assert sum(take) == 2        # a floor of 1 each is simply unreachable


def test_allocate_returns_everything_when_asking_for_more_than_exists():
    assert allocate([5, 7], 100) == [5, 7]


def test_allocate_is_order_stable_for_ties():
    assert allocate([100, 100], 3) == allocate([100, 100], 3)
    assert sum(allocate([100, 100], 3)) == 3


# ---------------------------------------------------------------- stride
def test_stride_covers_both_endpoints():
    items = [f"{i}" for i in range(100)]
    got = stride_pick(items, 5)
    assert got[0] == "0" and got[-1] == "99"
    assert len(got) == 5


def test_stride_of_one_takes_the_middle_not_the_start():
    """The first file of a trial is its least representative one."""
    assert stride_pick([f"{i}" for i in range(101)], 1) == ["50"]


def test_stride_is_deterministic():
    items = [f"{i}" for i in range(1000)]
    assert stride_pick(items, 17) == stride_pick(items, 17)


# ---------------------------------------------------------------- sampling
def test_per_folder_budget_is_split_across_channels_not_multiplied():
    """20 per folder with 4 mics means 5 each — not 80."""
    paths = _tree("/data/T001_PAH", ["ch1", "ch2", "ch3", "ch4"], 847)
    res = sample_paths(paths, SampleSpec(per="folder", n=20))
    assert len(res) == 20
    by_ch = {}
    for p in res.paths:
        by_ch[parse_channel(p)] = by_ch.get(parse_channel(p), 0) + 1
    assert by_ch == {"ch1": 5, "ch2": 5, "ch3": 5, "ch4": 5}


def test_lopsided_channels_still_sum_to_the_budget():
    """T014: 610 on ch1, 548 on ch2, and no channel may be starved."""
    paths = (_tree("/data/T014", ["ch1"], 610) + _tree("/data/T014", ["ch2"], 548))
    res = sample_paths(paths, SampleSpec(per="folder", n=20))
    assert len(res) == 20
    by_ch = {}
    for p in res.paths:
        by_ch[parse_channel(p)] = by_ch.get(parse_channel(p), 0) + 1
    assert by_ch == {"ch1": 11, "ch2": 9}


def test_pooling_channels_reproduces_the_naive_flat_stride():
    """The old behaviour is still reachable, and is measurably worse."""
    paths = (_tree("/data/T014", ["ch1"], 610) + _tree("/data/T014", ["ch2"], 548))
    res = sample_paths(paths, SampleSpec(per="folder", n=20, channel_mode="pool"))
    assert len(res) == 20
    assert len(res.rows) == 1              # one group, not two


def test_only_one_channel_keeps_just_that_mic():
    paths = _tree("/data/T001", ["ch1", "ch2", "ch3", "ch4"], 100)
    res = sample_paths(paths, SampleSpec(
        per="folder", n=20, channel_mode="only", channels=("ch1",)))
    assert len(res) == 20
    assert {parse_channel(p) for p in res.paths} == {"ch1"}


def test_total_mode_apportions_across_folders_by_size():
    paths = (_tree("/data/T001", ["ch1"], 800)
             + _tree("/data/T005", ["ch1"], 200))
    res = sample_paths(paths, SampleSpec(per="total", n=100))
    assert len(res) == 100
    per_folder = {}
    for p in res.paths:
        per_folder[os.path.basename(os.path.dirname(p))] = \
            per_folder.get(os.path.basename(os.path.dirname(p)), 0) + 1
    assert per_folder == {"T001": 80, "T005": 20}


def test_all_mode_takes_everything():
    paths = _tree("/data/T001", ["ch1", "ch2"], 50)
    res = sample_paths(paths, SampleSpec(per="all"))
    assert len(res) == 100


def test_exclusion_yields_genuinely_new_files_on_a_second_pass():
    """The label / correct / retrain loop depends on this."""
    paths = _tree("/data/T001", ["ch1", "ch2", "ch3", "ch4"], 200)
    spec = SampleSpec(per="folder", n=20)
    first = sample_paths(paths, spec)
    second = sample_paths(paths, spec, exclude=first.paths)
    assert len(second) == 20
    assert not (set(first.paths) & set(second.paths))
    assert second.n_excluded == 20


def test_exclusion_spreads_over_the_remainder_rather_than_leaving_holes():
    """A second pass must still tile the series, not cluster in the gaps."""
    paths = _tree("/data/T001", ["ch1"], 400)
    spec = SampleSpec(per="folder", n=10)
    first = sample_paths(paths, spec)
    second = sample_paths(paths, spec, exclude=first.paths)
    idx = sorted(paths.index(p) for p in second.paths)
    gaps = [b - a for a, b in zip(idx, idx[1:])]
    # Evenly spread means no gap is wildly larger than the mean.
    assert max(gaps) < 2 * (sum(gaps) / len(gaps))


def test_random_is_reproducible_from_its_seed():
    paths = _tree("/data/T001", ["ch1", "ch2"], 300)
    spec = SampleSpec(per="folder", n=20, spacing="random", seed=12345)
    assert sample_paths(paths, spec).paths == sample_paths(paths, spec).paths


def test_random_with_a_different_seed_draws_differently():
    paths = _tree("/data/T001", ["ch1", "ch2"], 300)
    a = sample_paths(paths, SampleSpec(per="folder", n=20, spacing="random", seed=1))
    b = sample_paths(paths, SampleSpec(per="folder", n=20, spacing="random", seed=2))
    assert a.paths != b.paths


def test_asking_for_more_than_a_folder_holds_takes_all_of_it():
    paths = _tree("/data/T001", ["ch1"], 7)
    res = sample_paths(paths, SampleSpec(per="folder", n=50))
    assert len(res) == 7


def test_files_without_a_channel_form_their_own_group_and_are_not_dropped():
    paths = (_tree("/data/T001", ["ch1"], 10)
             + [os.path.join("/data/T001", "loose_recording.wav")])
    res = sample_paths(paths, SampleSpec(per="all"))
    assert len(res) == 11
    assert any(r["channel"] == "" for r in res.rows)


def test_rows_report_every_group_including_ones_that_got_nothing():
    paths = (_tree("/data/T001", ["ch1"], 100)
             + _tree("/data/T002", ["ch1"], 100))
    res = sample_paths(paths, SampleSpec(per="total", n=2))
    assert len(res.rows) == 2
    assert sum(r["picked"] for r in res.rows) == 2


def test_result_is_sorted_and_free_of_duplicates():
    paths = _tree("/data/T001", ["ch1", "ch2"], 100)
    res = sample_paths(paths, SampleSpec(per="folder", n=20))
    assert len(set(res.paths)) == len(res.paths)
    assert res.paths == sorted(
        res.paths, key=lambda p: (os.path.dirname(os.path.abspath(p)),
                                  natural_key(p)))


def test_spec_round_trips_through_a_dict():
    spec = SampleSpec(per="total", n=250, spacing="random", seed=7,
                      channel_mode="only", channels=("ch1", "ch3"))
    again = SampleSpec.from_dict(spec.to_dict())
    assert again.to_dict() == spec.to_dict()
    assert "random (seed 7)" in again.describe()


@pytest.mark.parametrize("kwargs", [
    {"per": "nonsense"}, {"spacing": "nonsense"}, {"channel_mode": "nonsense"},
])
def test_spec_rejects_unknown_modes(kwargs):
    with pytest.raises(ValueError):
        SampleSpec(**kwargs)


def test_grouping_is_stable_regardless_of_input_order():
    paths = _tree("/data/T001", ["ch1", "ch2"], 50)
    assert list(group_paths(paths)) == list(group_paths(list(reversed(paths))))
