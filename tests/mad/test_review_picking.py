"""What a targeted QC batch is picked with (mad_sampling).

The held-out clip audit these were written for was dropped on 2026-10-12 in
favour of focused review of whole recordings; the activity screen and the
recording-name parsing are what picking those recordings still needs.

Runs under pytest, or directly.
"""
from datetime import datetime

import numpy as np
import pytest

from fnt.usv.usv_detector import mad_sampling as S

SR = 250_000


def tone(seconds=1.0, khz=40.0, start=0.2, dur=0.3, noise=0.01, seed=0):
    rng = np.random.default_rng(seed)
    t = np.arange(int(seconds * SR)) / SR
    x = rng.normal(0, noise, t.size)
    on = (t >= start) & (t < start + dur)
    x[on] += 0.3 * np.sin(2 * np.pi * khz * 1000 * t[on])
    return x.astype(np.float32)


def test_a_tonal_call_scores_and_silence_does_not():
    act = S.activity_per_window(tone(2.0, start=0.2, dur=0.4), SR, 1.0)
    assert act[0] > 50, act
    assert act[1] < 5, act


def test_a_broadband_click_is_not_mistaken_for_a_call():
    """Knocks and cage noise light the whole band at once; a USV is a ridge."""
    x = tone(1.0, dur=0.0)
    for c in range(20, SR, SR // 40):
        x[c:c + 30] += np.random.default_rng(c).normal(0, 0.5, 30)
    assert S.activity_per_window(x, SR, 1.0)[0] < 5


def test_an_intermittent_tone_counts_unless_the_band_starts_above_it():
    """The ~21 kHz electrical tone switches on and off, so it reads as
    activity -- which is why batches are screened from 25 kHz."""
    x = tone(1.0, khz=21.5, start=0.2, dur=0.3)
    assert S.activity_per_window(x, SR, 1.0)[0] > 50
    assert S.activity_per_window(x, SR, 1.0, band_hz=(25_000, 100_000))[0] < 5


def test_a_tone_that_never_stops_is_its_own_floor():
    x = tone(1.0, khz=21.5, start=0.0, dur=1.0)
    assert S.activity_per_window(x, SR, 1.0)[0] < 5


def test_an_avisoft_name_gives_mic_and_start():
    r = S.parse_recording(r"Z:\T009\ch42026-05-20_12-17-17_00007.wav",
                          "T009", "saline")
    assert r.mic == "4" and r.start == datetime(2026, 5, 20, 12, 17, 17)
    assert S.parse_recording("notes.wav", "T009", "saline") is None


@pytest.mark.parametrize("hour, phase", [(6, "night"), (7, "day"),
                                         (18, "day"), (19, "night")])
def test_day_and_night_by_clock_hour(hour, phase):
    assert S.phase_of(datetime(2026, 5, 20, hour)) == phase


def test_trial_and_condition_from_the_folder():
    assert S.trial_of(r"Z:\Data\T009_prairie_anosmic-saline") == "T009"
    assert S.condition_of("T010_prairie_anosmic-mmz", ["mmz", "saline"]) == "mmz"
    assert S.condition_of("T001_Prairie_2M2F", ["mmz", "saline"]) == ""


if __name__ == "__main__":
    import sys
    sys.exit(pytest.main([__file__, "-q"]))
