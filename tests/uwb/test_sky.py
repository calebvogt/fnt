"""The sky condition behind the preview's and the video's sky icon
(fnt/uwb/weather.py: WeatherTimeline.sky and friends), offline.

The rule under test is "measured beats modelled": a station's rain gauge
decides whether it is raining wherever the station has a record, measured
sunlight decides daytime cloud, and the model is used only in the gaps or
for what nothing measures (night cloud, thunder, fog) - and says so. On
VT-P002 the model missed a 15-minute shower the station recorded, which is
the failure these tests exist to prevent.

The site is NOAA SURFRAD's Desert Rock station (public coordinates), so the
sun is high at local noon and down at 01:00.

Runs under pytest, or directly (``python test_sky.py``).
"""
import json
import os
import sys

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fnt.uwb import weather as W  # noqa: E402

LAT, LON, TZ = 36.624, -116.019, "US/Pacific"
NOON = int(pd.Timestamp("2025-06-21 19:00", tz="UTC").value)   # 12:00 PDT
NIGHT = int(pd.Timestamp("2025-06-22 08:00", tz="UTC").value)  # 01:00 PDT
MIN = W.NS_PER_MIN
MEASURED = {"kind": "measured"}
MODEL = {"kind": "model"}


def _frame(t0, step_min, n, **cols):
    """A canonical frame of ``n`` records, ``step_min`` apart, first ending
    at ``t0 + step``. Column values are scalars or length-n sequences."""
    ends = t0 + np.arange(1, n + 1, dtype="int64") * step_min * MIN
    df = pd.DataFrame({"time_ns": ends, "interval_s": step_min * 60.0})
    for k, v in cols.items():
        df[k] = v
    return W._finish_frame(df)


def _sky_model(t0, hours, code=3, cover=100.0):
    ends = t0 + np.arange(1, hours + 1, dtype="int64") * 60 * MIN
    return pd.DataFrame({"time_ns": ends, "interval_s": 3600.0,
                         "weather_code": np.broadcast_to(np.asarray(code, float), hours),
                         "cloud_cover_pct": np.broadcast_to(np.asarray(cover, float),
                                                            hours)})


def _env(start, end, frames=None, meta=None, roles=None, sky=None):
    env = W.Environment(settings=W.SiteSettings(latitude=LAT, longitude=LON),
                        tz=TZ, start_ns=start, end_ns=end,
                        frames=frames or {}, meta=meta or {}, roles=roles or {})
    if sky is not None:
        env.extra["open_meteo_sky"] = sky
        env.meta["open_meteo_sky"] = MODEL
        env.roles["sky_model"] = "open_meteo_sky"
    return env


def _station(t0, n, precip, drop=()):
    """5-min gauge records (mm per record), with some records missing."""
    fr = _frame(t0, 5, n, precip_mm=precip, temp_c=20.0, rh_pct=80.0)
    return fr.drop(index=list(drop)).reset_index(drop=True)


def test_the_gauge_decides_wherever_it_has_a_record():
    t0 = NOON - 60 * MIN
    rain = np.zeros(24)
    rain[14] = 0.254                     # one tip, record ending NOON+15
    station = _station(t0, 24, rain, drop=range(18, 22))   # gap NOON+30..+50
    model_p = np.zeros(8)
    model_p[1] = 0.5                     # model rain NOON-30..-15: gauge dry
    model_p[6] = 0.5                     # model rain NOON+30..+45: no gauge
    model = _frame(t0, 15, 8, precip_mm=model_p, rain_mm=model_p,
                   snowfall_cm=0.0)
    env = _env(t0, t0 + 120 * MIN,
               {"weatherlink": station, "open_meteo": model},
               {"weatherlink": MEASURED, "open_meteo": MODEL},
               {"weather": "weatherlink", "supplement": "open_meteo"},
               sky=_sky_model(t0 - 60 * MIN, 4))
    tl = env.timeline()
    at = lambda m: tl.sky(NOON + m * MIN)
    wet = at(13)
    assert wet["raining"] and wet["state"] == "light_rain"
    assert wet["basis"] == "measured" and wet["source"] == "weatherlink"
    dry = at(-20)                        # the model's rain, the gauge's no
    assert not dry["raining"] and dry["state"] == "cloudy"
    assert dry["basis"] == "model"       # the overcast is the model's
    gap = at(40)                         # no gauge record: model, labelled
    assert gap["raining"] and gap["basis"] == "model"
    assert gap["source"] == "open_meteo"
    assert W.format_sky(wet) == "Light rain · measured"
    assert W.format_sky(gap).endswith("· model")


def test_thunder_is_the_models_but_only_with_rain_at_the_gauge():
    t0 = NOON - 60 * MIN
    rain = np.zeros(24)
    rain[14] = 0.254
    station = _station(t0, 24, rain)
    env = _env(t0, t0 + 120 * MIN, {"weatherlink": station},
               {"weatherlink": MEASURED}, {"weather": "weatherlink"},
               sky=_sky_model(t0 - 60 * MIN, 4, code=95))
    tl = env.timeline()
    storm = tl.sky(NOON + 13 * MIN)
    assert storm["state"] == "thunderstorm" and storm["basis"] == "model"
    assert storm["raining"]
    quiet = tl.sky(NOON - 30 * MIN)      # model thunder, dry gauge
    assert quiet["state"] == "cloudy" and not quiet["raining"]


def test_rain_intensity_is_judged_over_a_window_and_gaps_are_bridged():
    t0 = NOON - 60 * MIN
    rain = np.zeros(36)
    rain[[12, 15]] = 0.254               # tips at NOON+5, +20: one spell
    rain[20] = 0.254                     # NOON+45, after 20 dry minutes
    rain[26:32] = 1.0                    # 12 mm/h, NOON+70 to +100
    env = _env(t0, t0 + 180 * MIN, {"weatherlink": _station(t0, 36, rain)},
               {"weatherlink": MEASURED}, {"weather": "weatherlink"})
    tl = env.timeline()
    at = lambda m: tl.sky(NOON + m * MIN)
    # A single tip reads 3 mm/h in its own record; spread over the window it
    # is the light rain it was.
    assert at(3)["state"] == "light_rain"
    assert at(8)["raining"]              # between the two tips: bridged
    assert not at(30)["raining"]         # 20 dry minutes: a real break
    assert at(43)["state"] == "light_rain"
    assert at(85)["state"] == "heavy_rain"
    g = tl.sky_grid()
    spell = g["raining"] & (g["time_ns"] > NOON + 60 * MIN)
    assert spell.sum() == 30             # never lengthened or shortened
    # No weather at all: day with an unknown sky, never "clear".
    none = _env(t0, t0 + 60 * MIN).timeline().sky(NOON)
    assert none["state"] == "day" and none["basis"] == ""


def test_daytime_cloud_comes_from_measured_sunlight():
    t0 = NOON - 90 * MIN
    t = t0 + np.arange(1, 181, dtype="int64") * MIN
    el, _ = W.solar_position(t, LAT, LON)
    clear = W.clear_sky_ghi(el)
    factor = np.ones(180)
    factor[60:120] = 0.2                                   # overcast hour
    factor[120:180] = np.where(np.arange(60) % 4 < 2, 1.0, 0.25)  # broken
    surfrad = _frame(t0, 1, 180, ghi_wm2=clear * factor)
    env = _env(t0, t0 + 180 * MIN, {"surfrad": surfrad},
               {"surfrad": MEASURED}, {"solar": "surfrad"},
               # the model disagrees everywhere; the measurement must win
               sky=_sky_model(t0 - 60 * MIN, 6, code=0, cover=0.0))
    tl = env.timeline()
    for minute, state in ((30, "clear_day"), (90, "cloudy"),
                          (150, "partly_cloudy_day")):
        sky = tl.sky(t0 + minute * MIN)
        assert sky["state"] == state, (minute, sky["state"])
        assert sky["basis"] == "measured" and sky["source"] == "surfrad"
    ci = tl.sky(t0 + 30 * MIN)["clear_sky_index"]
    assert 0.95 < ci < 1.05


def test_night_cloud_and_fog_are_the_models_and_say_so():
    t0 = NIGHT - 60 * MIN
    cover = [90.0, 90.0, 10.0, 10.0]
    env = _env(t0, t0 + 180 * MIN, sky=_sky_model(t0 - 60 * MIN, 4, code=3,
                                                    cover=cover))
    tl = env.timeline()
    over = tl.sky(NIGHT - 30 * MIN)      # hour ending NIGHT: 90 %
    assert over["state"] == "cloudy" and over["basis"] == "model"
    assert tl.sky(NIGHT + 90 * MIN)["state"] == "clear_night"
    assert W.format_sky(over) == "Overcast · model"
    # Fog needs the station's humidity to agree.
    for rh, want in ((99.0, "fog"), (60.0, "cloudy")):
        station = _frame(t0, 5, 36, precip_mm=0.0, temp_c=8.0, rh_pct=rh)
        env = _env(t0, t0 + 180 * MIN, {"weatherlink": station},
                   {"weatherlink": MEASURED}, {"weather": "weatherlink"},
                   sky=_sky_model(t0 - 60 * MIN, 4, code=45, cover=100.0))
        assert env.timeline().sky(NIGHT)["state"] == want, rh
    # Measurements only (no model): the sky at night is unknown.
    bare = _env(t0, t0 + 60 * MIN).timeline().sky(NIGHT)
    assert bare["state"] == "night" and W.format_sky(bare) == "Night, sky unknown"
    # A sun-only timeline (no fetch yet) still knows day from night.
    sun_only = _env(0, 0).timeline()
    assert sun_only.sky(NIGHT)["state"] == "night"
    assert sun_only.sky(NOON)["state"] == "day"


def test_short_cloud_changes_are_held_but_long_ones_kept():
    a = np.array([1] * 20 + [2] * 3 + [1] * 20, dtype=np.int8)
    assert (W._hold_runs(a, (), 10)[0] == 1).all()
    b = np.array([2] * 3 + [1] * 20, dtype=np.int8)
    assert (W._hold_runs(b, (), 10)[0] == 1).all()
    c = np.array([1] * 20 + [3] * 12 + [1] * 20, dtype=np.int8)
    assert (W._hold_runs(c, (), 10)[0] == c).all()
    tags = np.array(["x"] * 20 + ["y"] * 3 + ["x"] * 20, dtype=object)
    held, carried = W._hold_runs(a, (tags,), 10)
    assert (carried == "x").all()
    wet = np.array([1, 0, 0, 1, 0, 0, 0, 0, 1], dtype=bool)
    assert list(W._bridge(wet, 2)) == [0, 1, 1, 0, 0, 0, 0, 0, 0]


def test_every_state_has_an_icon_inside_its_box():
    for state in W.SKY_STATES:
        shapes = W.sky_icon_shapes(state)
        assert shapes, state
        for shape in shapes:
            assert shape[0] in ("circle", "poly", "line"), shape
            if shape[0] == "circle":
                pts = [shape[1]]
            else:
                pts = shape[1]
            for x, y in pts:
                assert -0.05 <= x <= 1.05 and -0.05 <= y <= 1.05, (state, x, y)
    assert W.sky_icon_shapes("bogus") == []
    assert set(W.SKY_LABELS) == set(W.SKY_STATES)


def test_sky_model_parse_and_the_supplement_setting():
    payload = {"hourly": {"time": ["2025-06-21T00:00", "2025-06-21T01:00",
                                   "2025-06-21T02:00"],
                          "weather_code": [3, 95, None],
                          "cloud_cover": [100, 80, None]}}
    now = pd.Timestamp("2025-06-21 01:30", tz="UTC").value
    fr = W.parse_open_meteo_sky(payload, now_ns=now)
    assert list(fr["weather_code"]) == [3.0, 95.0]
    assert (fr["interval_s"] == 3600.0).all()
    assert W.parse_open_meteo_sky({}).empty
    assert W.SiteSettings().model_supplement is True
    old = W.SiteSettings.from_dict({"latitude": 1, "longitude": 2})
    assert old.model_supplement is True          # profiles from before
    off = W.SiteSettings.from_dict({"model_supplement": False})
    again = W.SiteSettings.from_dict(json.loads(json.dumps(off.to_dict())))
    assert again.model_supplement is False


def test_sky_export_is_one_row_per_minute_with_its_evidence():
    t0 = NOON - 60 * MIN
    rain = np.zeros(24)
    rain[14] = 0.254
    env = _env(t0, t0 + 120 * MIN, {"weatherlink": _station(t0, 24, rain)},
               {"weatherlink": MEASURED}, {"weather": "weatherlink"},
               sky=_sky_model(t0 - 60 * MIN, 4))
    out = W.sky_frame(env)
    assert list(out.columns) == list(W.SKY_EXPORT_COLUMNS)
    assert len(out) == 121
    assert (out["timestamp"].diff().dropna() == 60_000).all()
    assert str(out["Timestamp"].dt.tz) == TZ
    wet = out[out["raining"]]
    assert len(wet) == 5 and (wet["sky_basis"] == "measured").all()
    assert (wet["precip_source"] == "weatherlink").all()
    assert (out.loc[~out["raining"], "sky_state"] == "cloudy").all()
    assert set(out["sky_label"]) <= set(W.SKY_LABELS.values())
    assert list(W.sky_frame(_env(0, 0)).columns) == list(W.SKY_EXPORT_COLUMNS)


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
