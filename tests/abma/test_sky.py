"""Sun and moon over a real site.

Solar geometry has known right answers, so these check against them rather than
against the implementation: the sun's noon elevation at a solstice is
``90 - latitude ± 23.44``, it is due south at local noon in the northern
hemisphere, and Boulder's day length runs from about 9h 20m to about 15h. If
the model can hit those it can be trusted for "is it dark", which is all the
behaviour model asks of it.
"""
from __future__ import annotations

from datetime import datetime

import numpy as np
import pytest

from fnt.abma.core.compose import design
from fnt.abma.core.config import SkyParams, ExperimentConfig
from fnt.abma.core.simulation import Simulation
from fnt.abma.core.sky import (
    sky_state, sun_position, moon_position, day_length_hours, season_start,
    season_of, growth_factor, SEASON_DATES,
)

BOULDER = SkyParams(enabled=True, latitude=40.0150, longitude=-105.2705,
                    timezone_hours=-7.0)


# --------------------------------------------------------------------------- #
# Solar geometry, against known answers
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("month,day,expected", [
    (6, 21, 90 - 40.015 + 23.44),      # summer solstice
    (12, 21, 90 - 40.015 - 23.44),     # winter solstice
    (3, 20, 90 - 40.015),              # equinox
])
def test_noon_sun_elevation_matches_the_geometry(month, day, expected):
    """Peak elevation over the day, which is solar noon by definition."""
    best = max(sun_position(datetime(2026, month, day, h, m), BOULDER.latitude,
                            BOULDER.longitude, BOULDER.timezone_hours)[0]
               for h in range(10, 15) for m in (0, 15, 30, 45))
    assert best == pytest.approx(expected, abs=1.0)


def test_the_sun_is_due_south_at_solar_noon():
    elevations = [(sun_position(datetime(2026, 6, 21, h, m), BOULDER.latitude,
                                BOULDER.longitude, BOULDER.timezone_hours), h, m)
                  for h in range(10, 15) for m in (0, 15, 30, 45)]
    (elev, azim), h, m = max(elevations, key=lambda t: t[0][0])
    assert azim == pytest.approx(180.0, abs=3.0)


def test_the_sun_rises_in_the_east_and_sets_in_the_west():
    lat, lon, tz = BOULDER.latitude, BOULDER.longitude, BOULDER.timezone_hours
    dawn = sun_position(datetime(2026, 6, 21, 5, 30), lat, lon, tz)
    dusk = sun_position(datetime(2026, 6, 21, 19, 30), lat, lon, tz)
    assert 30 < dawn[1] < 130, f"dawn azimuth {dawn[1]} is not easterly"
    assert 230 < dusk[1] < 330, f"dusk azimuth {dusk[1]} is not westerly"


def test_day_length_swings_with_the_season_at_this_latitude():
    summer = day_length_hours(datetime(2026, 6, 21), BOULDER)
    winter = day_length_hours(datetime(2026, 12, 21), BOULDER)
    equinox = day_length_hours(datetime(2026, 3, 20), BOULDER)
    assert summer == pytest.approx(14.9, abs=0.5)
    assert winter == pytest.approx(9.3, abs=0.5)
    assert equinox == pytest.approx(12.0, abs=0.3)
    assert summer > equinox > winter


def test_at_the_equator_the_day_barely_changes():
    equator = SkyParams(latitude=0.0, longitude=0.0, timezone_hours=0.0)
    lengths = [day_length_hours(datetime(2026, m, 15), equator)
               for m in range(1, 13)]
    assert max(lengths) - min(lengths) < 0.2


def test_night_is_night_and_noon_is_not():
    assert not sky_state(datetime(2026, 6, 21, 1, 0), BOULDER).is_day
    assert sky_state(datetime(2026, 6, 21, 12, 0), BOULDER).is_day


def test_twilight_is_a_ramp_not_a_switch():
    """A crepuscular animal needs a dawn to be active in."""
    values = [sky_state(datetime(2026, 6, 21, 4, 0), BOULDER).daylight,
              sky_state(datetime(2026, 6, 21, 5, 0), BOULDER).daylight,
              sky_state(datetime(2026, 6, 21, 6, 0), BOULDER).daylight]
    assert values == sorted(values)
    assert 0.0 < values[1] < 1.0, "no intermediate light level exists"


# --------------------------------------------------------------------------- #
# Moon
# --------------------------------------------------------------------------- #
def test_the_moon_cycles_through_its_phases_in_about_a_month():
    phases = [moon_position(datetime(2026, 1, d), BOULDER.latitude,
                            BOULDER.longitude, BOULDER.timezone_hours)[2]
              for d in range(1, 30)]
    assert min(phases) < 0.15 and max(phases) > 0.85


def test_illumination_peaks_at_full_moon():
    """Phase 0.5 is opposition, which is when the disc is fully lit."""
    lit = []
    for d in range(1, 30):
        state = sky_state(datetime(2026, 1, d, 23, 0), BOULDER)
        lit.append((abs(state.moon_phase - 0.5), state.moon_illumination))
    nearest_full = min(lit)[1]
    nearest_new = max(lit)[1]
    assert nearest_full > 0.9 and nearest_new < 0.2


def test_moonlight_only_counts_at_night_and_when_the_moon_is_up():
    noon = sky_state(datetime(2026, 6, 21, 12, 0), BOULDER)
    assert noon.night_light == 0.0
    nights = [sky_state(datetime(2026, 1, d, 23, 0), BOULDER)
              for d in range(1, 30)]
    assert any(s.night_light > 0 for s in nights)
    assert all(s.night_light == 0 or s.moon_elevation > 0 for s in nights)


# --------------------------------------------------------------------------- #
# Seasons
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("season", sorted(set(SEASON_DATES)))
def test_every_season_resolves_to_a_date_in_that_season(season):
    when = datetime.fromisoformat(season_start(season, 2026))
    assert season_of(when) == ("autumn" if season == "autumn"
                               else season).replace("autumn", "fall")


def test_an_unknown_season_is_refused():
    with pytest.raises(ValueError, match="unknown season"):
        season_start("monsoon")


def test_grass_grows_in_spring_and_barely_in_winter():
    assert growth_factor(datetime(2026, 5, 20)) > 0.9
    assert growth_factor(datetime(2026, 12, 21)) < 0.15


# --------------------------------------------------------------------------- #
# In the engine
# --------------------------------------------------------------------------- #
def test_the_engine_takes_day_night_from_the_sun_when_the_sky_is_on():
    cfg = design(preset="voleterra", males=1, females=1, days=1,
                 season="summer")
    sim = Simulation(cfg, 0)
    assert sim.sky_p is not None
    # released at 18:00 in June it is still light; by 02:00 it is not
    assert sim._is_day(0.0)
    assert not sim._is_day(8 * 3600.0)


def test_activity_falls_through_dusk_rather_than_stepping():
    cfg = design(preset="voleterra", males=1, females=1, days=1,
                 season="summer")
    sim = Simulation(cfg, 0)
    levels = [sim._activity(h * 3600.0) for h in range(0, 8)]
    assert len(set(np.round(levels, 3))) > 2, "activity is a step function"


def test_season_changes_how_much_of_the_day_is_dark():
    dark = {}
    for season in ("summer", "winter"):
        cfg = design(preset="voleterra", males=1, females=1, days=1,
                     season=season)
        sim = Simulation(cfg, 0)
        dark[season] = sum(not sim._is_day(h * 900.0) for h in range(96))
    assert dark["winter"] > dark["summer"]


def test_the_sky_can_be_switched_off_and_the_clock_takes_over():
    cfg = ExperimentConfig()
    cfg.groups = design(preset="voleterra", males=1, females=1).groups
    sim = Simulation(cfg, 0)
    assert sim.sky_p is None
    assert sim.sky_at(0.0) is None
    # the legacy fixed window still applies
    assert sim._activity(0.0) in (cfg.day_activity, cfg.night_activity)
