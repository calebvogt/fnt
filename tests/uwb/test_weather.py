"""Weather, sunlight and sun position (fnt/uwb/weather.py), offline.

Guards the parts a trial's results would silently depend on:

* the sun's position, checked against the zenith angle NOAA's SURFRAD
  station reports for its own minutes, and sunrise/sunset against a
  published almanac time;
* the SURFRAD and WeatherLink parsers - column mapping, QC and missing
  values, unit conversion, and the DST fall-back hour a station logs once;
* the "record covering this moment" lookup the preview and video use, with
  its staleness rule;
* the download cache's re-fetch rule, without touching the network;
* the export schema.

Runs under pytest, or directly (``python test_weather.py``).
"""
import io
import json
import os
import sys
import tempfile
import urllib.request
from datetime import datetime, timedelta, timezone

import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from fnt.uwb import weather as W  # noqa: E402

# Seven minutes of NOAA SURFRAD Desert Rock (public data), 2025-06-21 UTC.
SURFRAD_SAMPLE = """ Desert Rock
   36.62 -116.02 1007 m version 1
 2025 172  6 21 12 40 12.667  88.13    10.1 0     2.2 0    -0.6 0    13.8 0   303.6 0   293.946 0   293.748 0   407.7 0   293.587 0   293.617 0     0.8 0     7.3 0    11.5 0  -104.1 0   -92.6 0    21.4 0    14.7 0     3.6 0   183.0 0   892.7 0
 2025 172  6 21 12 41 12.683  87.97    10.7 0     2.5 0    -0.6 0    14.4 0   303.2 0   293.960 0   293.734 0   407.3 0   293.587 0   293.602 0     0.8 0     7.7 0    11.9 0  -104.2 0   -92.3 0    21.4 0    14.7 0     2.9 0   191.9 0   892.7 0
 2025 172  6 21 17  0 17.000  38.02   834.5 0   170.7 0   939.8 0    92.4 0   321.7 0   301.719 0   301.283 0   515.9 0   299.834 0   300.254 0   161.7 0   353.0 0   662.1 0  -194.1 0   468.0 0    26.1 0    13.7 0    11.9 0   230.0 0   892.4 0
 2025 172  6 21 17  1 17.017  37.82   837.2 0   172.5 0   941.1 0    92.3 0   321.6 0   301.756 0   301.332 0   514.2 0   299.882 0   300.187 0   162.9 0   354.1 0   663.2 0  -192.6 0   470.7 0    25.9 0    13.9 0    11.8 0   220.8 0   892.4 0
 2025 172  6 21 17  2 17.033  37.62   840.5 0   173.4 0   942.9 0    92.1 0   320.7 0   301.802 0   301.372 0   513.8 0   299.890 0   300.202 0   164.2 0   355.4 0   665.5 0  -193.1 0   472.4 0    25.8 0    14.0 0    12.8 0   230.7 0   892.3 0
 2025 172  6 21 17  3 17.050  37.42   844.7 0   175.0 0   944.7 1    91.6 0   320.7 0   301.823 0   301.230 0   511.8 0   299.855 0   300.100 0   165.6 0   356.8 0   667.0 0  -191.1 0   475.9 0    25.4 0    14.3 0    14.0 0   235.6 0   892.3 0
 2025 172  6 21 17  4 17.067  37.22   846.6 0   175.4 0   945.1 0    91.7 0   321.1 0   301.834 0   301.345 0   512.0 0   299.795 0   300.078 0   166.6 0 -9999.9 1   668.9 0  -190.9 0   478.0 0    25.3 0    14.3 0    12.6 0   231.7 0   892.3 0
"""

# A made-up station in the standard WeatherLink export layout, US units, on
# the US fall-back day: 01:00-01:55 happens twice but is logged once.
WL_HEADER = (
    "                  Temp     Hi    Low   Out    Dew  Wind  Wind   Wind    Hi    Hi   Wind   Heat    THW                Rain    Heat    Cool    In     In    In     In     In   In Air  Wind  Wind    ISS   Arc.\n"
    "  Date    Time     Out   Temp   Temp   Hum    Pt. Speed   Dir    Run Speed   Dir  Chill  Index  Index   Bar    Rain  Rate    D-D     D-D    Temp   Hum    Dew   Heat    EMC Density  Samp   Tx   Recept  Int.\n"
    "---------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------\n"
)


def _wl_row(date, tm, temp, hum, wind, wdir, gust, bar, rain):
    return (f"{date:>8s} {tm:>6s}  {temp:5.1f}  {temp:5.1f}  {temp:5.1f}  {hum:4d}  "
            f"{temp - 10:5.1f}  {wind:4.1f}  {wdir:>4s}   0.25  {gust:4.1f}  {wdir:>4s}  "
            f"{temp:5.1f}  {temp:5.1f}  {temp:5.1f}  {bar:6.1f}  {rain:4.2f}  0.00   0.000   "
            f"0.000   68.4    36   40.4   65.6   7.22  .0617    116    1    100.0    5 \n")


WL_SAMPLE = WL_HEADER + "".join([
    _wl_row("11/03/24", "12:55a", 50.0, 40, 5.0, "NW", 9.0, 840.0, 0.00),
    _wl_row("11/03/24", "1:00a", 50.0, 40, 5.0, "NW", 9.0, 840.0, 0.00),
    _wl_row("11/03/24", "1:05a", 41.0, 90, 0.0, "---", 0.0, 840.0, 0.10),
    _wl_row("11/03/24", "2:00a", 32.0, 55, 10.0, "S", 12.0, 840.0, 0.00),
])


def test_solar_position_matches_surfrad_zenith():
    _info, fr = W.parse_surfrad(SURFRAD_SAMPLE)
    # SURFRAD's zenith is apparent (refracted), at the centre of the minute.
    el, _az = W.solar_position(fr["time_ns"].to_numpy() - 30 * W.NS_PER_S,
                               36.624, -116.019, refraction=True)
    err = np.abs((90.0 - el) - fr_zenith(SURFRAD_SAMPLE))
    assert err.max() < 0.03, err


def fr_zenith(text):
    return np.array([float(ln.split()[7]) for ln in text.splitlines()[2:]])


def test_sunrise_sunset_against_almanac():
    # Royal Observatory Greenwich, 2024-06-21: sunrise 04:43, sunset 21:21 BST.
    t0 = pd.Timestamp("2024-06-21 12:00", tz="Europe/London").value
    dl = W.daylight_table(51.4769, -0.0005, "Europe/London", t0, t0)
    row = dl.iloc[0]
    for col, want in (("sunrise", "04:43"), ("sunset", "21:21")):
        got = row[col]
        ref = pd.Timestamp(f"2024-06-21 {want}", tz="Europe/London")
        assert abs((got - ref).total_seconds()) < 90, (col, got)
    assert 16.5 < row["day_length_h"] < 16.7
    assert row["civil_dawn"] < row["sunrise"] < row["solar_noon"] < row["sunset"] < row["civil_dusk"]
    assert row["Day"] == 1


def test_daylight_table_spans_a_dst_change():
    tz = "US/Mountain"
    t0 = pd.Timestamp("2024-11-02 12:00", tz=tz).value
    t1 = pd.Timestamp("2024-11-04 12:00", tz=tz).value
    dl = W.daylight_table(36.624, -116.019, tz, t0, t1)
    assert list(dl["Day"]) == [1, 2, 3]
    # Sunrise jumps an hour earlier on the clock across the change.
    r = [dl["sunrise"].iloc[i].hour + dl["sunrise"].iloc[i].minute / 60 for i in (0, 2)]
    assert 0.9 < r[0] - r[1] < 1.1


def test_sun_phase_and_light_level():
    ph = W.sun_phase(np.array([10.0, -0.5, -3.0, -9.0, -15.0, -30.0]))
    assert list(ph) == ["day", "day", "civil", "nautical", "astronomical", "night"]
    lv = W.light_level(np.array([45.0, 45.0, -3.0, -30.0]),
                       np.array([900.0, 50.0, np.nan, np.nan]))
    assert lv[0] > lv[1] > lv[2] > lv[3] == 0.0
    # Clear sky stands in where no measurement exists.
    assert W.light_level(np.array([45.0]))[0] > 0.8


def test_parse_surfrad_qc_and_components():
    info, fr = W.parse_surfrad(SURFRAD_SAMPLE)
    assert info["name"] == "Desert Rock" and info["elevation_m"] == 1007.0
    assert len(fr) == 7 and (fr["interval_s"] == 60.0).all()
    first_day = fr.iloc[2]
    # direct*cos(zenith) + diffuse, preferred over the global pyranometer.
    want = 939.8 * np.cos(np.radians(38.02)) + 92.4
    assert abs(first_day["ghi_wm2"] - want) < 1e-6
    assert first_day["temp_c"] == 26.1 and first_day["pressure_hpa"] == 892.4
    # QC 1 on the direct beam: fall back to the global pyranometer.
    assert fr.iloc[5]["ghi_wm2"] == 844.7 and np.isnan(fr.iloc[5]["dni_wm2"])
    # -9999.9 is missing.
    assert np.isnan(fr.iloc[6]["par_wm2"])
    assert pd.Timestamp(int(fr.iloc[2]["time_ns"]), tz="UTC") == \
        pd.Timestamp("2025-06-21 17:00", tz="UTC")


def test_weatherlink_columns_and_units():
    names = W.weatherlink_columns(*WL_HEADER.splitlines()[:2])
    for want in ("Temp Out", "Out Hum", "Dew Pt.", "Wind Speed", "Wind Dir",
                 "Hi Speed", "Bar", "Rain", "Rain Rate", "In Air Density",
                 "Arc. Int."):
        assert want in names, (want, names)
    fr = W.parse_weatherlink(WL_SAMPLE, "US/Mountain",
                             {"temp": "F", "wind": "mph", "rain": "in"})
    fr = W._finish_frame(fr)
    assert len(fr) == 4
    r = fr.iloc[0]
    assert abs(r["temp_c"] - 10.0) < 1e-9
    assert abs(r["wind_speed_ms"] - 5 * 0.44704) < 1e-9
    assert r["wind_dir_deg"] == 315.0 and r["pressure_hpa"] == 840.0
    assert r["interval_s"] == 300.0
    calm = fr.iloc[2]
    assert np.isnan(calm["wind_dir_deg"])
    assert abs(calm["precip_mm"] - 2.54) < 1e-9


def _physical_rows(unit):
    """Rows whose dew point really follows from temperature and humidity."""
    rows = []
    for i, (tc, rh) in enumerate([(22.4, 30), (18.0, 45), (10.0, 70), (4.0, 90)]):
        dc = float(W.dewpoint_c(tc, rh))
        t, d = ((tc * 9 / 5 + 32, dc * 9 / 5 + 32) if unit == "F" else (tc, dc))
        row = _wl_row("9/15/26", f"{i + 1}:00p", t, rh, 5.0, "N", 8.0, 840.0, 0.0)
        # _wl_row writes temp-10 as the dew point; put the real one in.
        cells = row.split()
        cells[6] = f"{d:.1f}"
        rows.append("  ".join(cells) + "\n")
    return rows


def test_weatherlink_temperature_unit_is_detected():
    assert W.detect_temp_unit([72.4], [39.2], [30]) is None     # too few rows
    for unit, want_c in (("F", 22.4), ("C", 22.4)):
        text = WL_HEADER + "".join(_physical_rows(unit))
        fr = W.parse_weatherlink(text, "US/Mountain", {"temp": "auto"})
        assert fr.attrs["temp_unit_detected"] == unit
        assert fr.attrs["temp_unit_used"] == unit
        assert abs(fr["temp_c"].iloc[0] - want_c) < 0.06
    # An explicit wrong choice is honoured but reported by the fetch.
    fr = W.parse_weatherlink(WL_HEADER + "".join(_physical_rows("F")),
                             "US/Mountain", {"temp": "C"})
    assert fr.attrs["temp_unit_used"] == "C"
    assert fr.attrs["temp_unit_detected"] == "F"


def test_weatherlink_fall_back_hour_is_flagged():
    fr = W._finish_frame(W.parse_weatherlink(WL_SAMPLE, "US/Mountain"))
    flags = dict(zip(pd.to_datetime(fr["time_ns"], utc=True)
                     .dt.tz_convert("US/Mountain").dt.strftime("%H:%M"),
                     fr["time_flag"]))
    assert flags["00:55"] == "" and flags["02:00"] == ""
    assert flags["01:00"] == "ambiguous_dst" and flags["01:05"] == "ambiguous_dst"


def _env(frames, roles, lat=36.624, lon=-116.019, meta=None):
    s = W.SiteSettings(latitude=lat, longitude=lon)
    env = W.Environment(settings=s, tz="US/Pacific", start_ns=0, end_ns=0,
                        frames=frames, roles=roles, meta=meta or {})
    return env


def _frame(ends_s, interval_s, **cols):
    df = pd.DataFrame({"time_ns": np.array(ends_s, dtype="int64") * W.NS_PER_S,
                       "interval_s": float(interval_s)})
    for k, v in cols.items():
        df[k] = v
    return W._finish_frame(df)


def test_timeline_uses_the_covering_record_and_flags_stale():
    base = 1_750_000_000
    fr = _frame([base + 300, base + 600, base + 3600], 300,
                temp_c=[1.0, 2.0, 3.0])
    tl = _env({"weatherlink": fr}, {"weather": "weatherlink"}).timeline()
    at = lambda s: tl.at((base + s) * W.NS_PER_S)["weather"]
    assert at(1)["values"]["temp_c"] == 1.0 and at(1)["age_s"] == 0
    assert at(300)["values"]["temp_c"] == 1.0          # end is inclusive
    assert at(301)["values"]["temp_c"] == 2.0
    gap = at(3000)                                      # 40 min after 600
    assert gap["values"]["temp_c"] == 2.0 and gap["stale"]
    assert at(0) is None                                # before the first
    assert at(3600 + 4 * 3600) is None                  # too old to show
    text = W.format_weather(tl.at((base + 3000) * W.NS_PER_S))
    assert "2.0 °C" in text and "old" in text
    us = W.format_weather(tl.at((base + 1) * W.NS_PER_S), "us")
    assert "33.8 °F" in us


def test_light_uses_measured_irradiance_when_fresh():
    t = int(pd.Timestamp("2025-06-21 17:00", tz="UTC").value // W.NS_PER_S)
    fr = _frame([t, t + 60], 60, ghi_wm2=[5.0, 5.0])
    tl = _env({"surfrad": fr}, {"solar": "surfrad"},
              meta={"surfrad": {"kind": "measured"}}).timeline()
    st = tl.at(t * W.NS_PER_S)
    assert st["ghi"] == 5.0 and st["ghi_kind"] == "measured"
    assert st["phase"] == "day" and st["sun_elevation"] > 45
    clear = _env({}, {}).timeline().at(t * W.NS_PER_S)
    assert np.isnan(clear["ghi"]) and clear["light"] > st["light"]
    times, levels, moon = tl.light_strip(t * W.NS_PER_S - 3600 * W.NS_PER_S,
                                         t * W.NS_PER_S + 3600 * W.NS_PER_S, 13)
    assert len(times) == 13 and np.isfinite(levels).all()
    assert len(moon) == 13 and ((moon >= 0) & (moon <= 1)).all()
    assert "moon" in W.format_light(st)


# US Naval Observatory: principal lunar phases, November 2024 (UTC).
USNO_PHASES = (("new moon", "2024-11-01 12:47", 0.0),
               ("first quarter", "2024-11-09 05:55", 0.5),
               ("full moon", "2024-11-15 21:28", 1.0),
               ("last quarter", "2024-11-23 01:28", 0.5))


def test_moon_phase_against_usno():
    for name, when, lit in USNO_PHASES:
        k, d, wax, names = W.moon_phase(
            np.array([pd.Timestamp(when, tz="UTC").value]))
        assert abs(k[0] - lit) < 0.01, (name, k[0])
        assert names[0] == name
    k, d, wax, _ = W.moon_phase(np.array([
        pd.Timestamp("2024-11-05", tz="UTC").value,
        pd.Timestamp("2024-11-20", tz="UTC").value]))
    assert list(wax) == [True, False]
    assert 0 < k[0] < 0.5 < k[1] < 1


def test_moonrise_against_usno():
    # Greenwich, 2024-11-10: moonrise 14:15 UTC, no moonset that date.
    t0 = pd.Timestamp("2024-11-10 12:00", tz="UTC").value
    row = W.daylight_table(51.48, 0.0, "UTC", t0, t0).iloc[0]
    ref = pd.Timestamp("2024-11-10 14:15", tz="UTC")
    assert abs((row["moonrise"] - ref).total_seconds()) < 180, row["moonrise"]
    assert pd.isna(row["moonset"])
    assert row["moon_phase_midnight"] == "waxing gibbous"
    assert 0.5 < row["moon_illumination_midnight"] < 0.8
    assert 0 < row["moonlit_dark_h"] <= row["dark_h"] < 24


def test_moon_disc_polygon_area_is_the_lit_fraction():
    def area(p):
        x, y = p[:, 0], p[:, 1]
        return 0.5 * abs(np.dot(x, np.roll(y, -1)) - np.dot(y, np.roll(x, -1)))
    for k in (0.0, 0.1, 0.5, 0.8, 1.0):
        poly = W.moon_disc_polygon(k, True, n=400)
        assert abs(area(poly) / np.pi - k) < 0.01, k
    # Waxing is lit on the right in the north, on the left in the south.
    assert W.moon_disc_polygon(0.3, True)[:, 0].mean() > 0
    assert W.moon_disc_polygon(0.3, False)[:, 0].mean() < 0
    assert W.moon_disc_polygon(0.3, True, southern=True)[:, 0].mean() < 0
    # Moonlight only matters when the moon is up and lit.
    lv = W.moonlight_level(np.array([45.0, -5.0, 45.0]), np.array([1.0, 1.0, 0.0]))
    assert lv[0] > 0.5 and lv[1] == 0 and lv[2] == 0
    assert W.light_rgb(0.0, 1.0) != W.light_rgb(0.0, 0.0)
    assert W.light_rgb(1.0, 1.0) == W.light_rgb(1.0, 0.0)


def test_download_cache_refetches_only_unsettled_files(monkeypatch=None):
    calls = []

    class _Resp(io.BytesIO):
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

    def fake_urlopen(req, timeout=0):
        calls.append(req.full_url)
        return _Resp(b"payload %d" % len(calls))

    real = urllib.request.urlopen
    urllib.request.urlopen = fake_urlopen
    try:
        with tempfile.TemporaryDirectory() as root:
            cache = W.DownloadCache(root)
            past = datetime.now(timezone.utc) - timedelta(days=3)
            p = cache.get("https://example.org/a.txt", "x/a.txt", past)
            assert open(p, "rb").read() == b"payload 1"
            # Fetched after its period settled: reused.
            assert cache.get("https://example.org/a.txt", "x/a.txt", past) == p
            assert len(calls) == 1
            # Fetched while its period was still running, and long enough
            # ago: fetched again.
            future = datetime.now(timezone.utc) + timedelta(hours=1)
            cache.get("https://example.org/b.txt", "x/b.txt", future)
            rec = cache.manifest["x/b.txt"]
            rec["retrieved_utc"] = (datetime.now(timezone.utc)
                                    - timedelta(hours=2)).isoformat()
            cache.get("https://example.org/b.txt", "x/b.txt", future)
            assert len(calls) == 3
            man = json.load(open(os.path.join(root, "manifest.json")))
            assert set(man) == {"x/a.txt", "x/b.txt"}
            assert len(man["x/a.txt"]["sha256"]) == 64
            # Offline: cache only, never the network.
            off = W.DownloadCache(root, offline=True)
            assert off.get("https://example.org/c.txt", "x/c.txt", past) is None
            assert len(calls) == 3
    finally:
        urllib.request.urlopen = real


def test_wet_bulb_matches_stull():
    # Stull (2011): 20 C at 50% RH has a wet-bulb temperature of 13.7 C.
    assert abs(float(W.wet_bulb_c(20.0, 50.0)) - 13.7) < 0.1


def test_precipitation_kind_and_rate():
    base = 1_750_000_000
    # Reported split (a model): rain, snow, both, and a dry interval.
    rep = _frame([base + 900 * i for i in range(1, 5)], 900,
                 precip_mm=[1.0, 0.5, 0.8, 0.0], rain_mm=[1.0, 0.0, 0.3, 0.0],
                 snowfall_cm=[0.0, 0.35, 0.3, 0.0], temp_c=[5.0, -3.0, 0.5, 5.0])
    assert list(rep["precip_type"]) == ["rain", "snow", "mixed", ""]
    assert list(rep["precip_basis"]) == ["reported"] * 3 + [""]
    assert rep["precip_rate_mmh"].iloc[0] == 4.0          # 1 mm in 15 min
    # A station (no split): kind from wet-bulb temperature, marked estimated.
    stn = _frame([base + 300 * i for i in range(1, 4)], 300,
                 precip_mm=[0.2, 0.2, 0.2], temp_c=[-4.0, 1.5, 8.0],
                 rh_pct=[90.0, 95.0, 60.0])
    assert list(stn["precip_type"]) == ["snow", "mixed", "rain"]
    assert set(stn["precip_basis"]) == {"wet-bulb"}
    dry_air = _frame([base + 300], 300, precip_mm=[0.2], temp_c=[1.0])
    assert dry_air["precip_type"].iloc[0] == "mixed"
    assert dry_air["precip_basis"].iloc[0] == "air temperature"

    tl = _env({"open_meteo": rep}, {"weather": "open_meteo"}).timeline()
    snow = tl.at((base + 1500) * W.NS_PER_S)
    assert W.format_precip(snow) == "SNOW 1.4 cm/h"
    assert W.format_weather(snow).startswith("SNOW 1.4 cm/h · -3.0 °C")
    rain = tl.at((base + 600) * W.NS_PER_S)
    assert W.format_precip(rain) == "RAIN 4.0 mm/h"
    assert W.format_precip(rain, "us") == "RAIN 0.16 in/h"
    assert W.format_precip(tl.at((base + 3500) * W.NS_PER_S)) == ""
    stl = _env({"weatherlink": stn}, {"weather": "weatherlink"}).timeline()
    assert W.format_precip(stl.at((base + 100) * W.NS_PER_S)) == \
        "SNOW (est.) 2.4 mm/h water"
    rate, kind = tl.precip_strip(base * W.NS_PER_S, (base + 3600) * W.NS_PER_S, 5)
    assert list(kind) == ["", "rain", "snow", "mixed", ""]
    assert rate[1] == 4.0
    assert W.precip_rgba(0.0, "") is None
    heavy, light = W.precip_rgba(20.0, "rain"), W.precip_rgba(0.1, "rain")
    assert heavy[3] == 1.0 and light[3] == 0.35


def test_open_meteo_drops_forecast_rows():
    payload = {"minutely_15": {
        "time": ["2025-06-21T00:00", "2025-06-21T00:15", "2025-06-21T00:30"],
        "temperature_2m": [20.0, 21.0, 22.0],
        "shortwave_radiation": [0.0, 0.0, None]}}
    now = pd.Timestamp("2025-06-21 00:20", tz="UTC").value
    fr = W.parse_open_meteo(payload, "minutely_15", now_ns=now)
    assert list(fr["temp_c"]) == [20.0, 21.0]
    assert (fr["interval_s"] == 900.0).all()


def test_export_frame_schema_and_local_times():
    t = int(pd.Timestamp("2025-06-21 17:00", tz="UTC").value // W.NS_PER_S)
    fr = _frame([t, t + 60], 60, temp_c=[26.1, 25.9])
    env = _env({"surfrad": fr}, {"weather": "surfrad"},
               meta={"surfrad": {"station": "Desert Rock, NV", "kind": "measured"}})
    env.start_ns, env.end_ns = t * W.NS_PER_S, (t + 60) * W.NS_PER_S
    out = W.export_frame(env)
    assert list(out.columns) == list(W.EXPORT_COLUMNS)
    assert len(out) == 2 and set(out["source"]) == {"surfrad"}
    assert str(out["Timestamp"].dt.tz) == "US/Pacific"
    assert out["timestamp"].iloc[0] == t * 1000
    assert out["sun_phase"].iloc[0] == "day"
    assert out["moon_phase"].iloc[0] in W.MOON_PHASES
    assert 0 <= out["moon_illumination"].iloc[0] <= 1


def test_site_settings_round_trip_and_validation():
    s = W.SiteSettings.from_dict({"latitude": "12.5", "longitude": -3,
                                  "weather_source": "bogus",
                                  "solar_source": "surfrad",
                                  "weatherlink_units": {"temp": "K"}})
    assert s.has_location and s.weather_source == "none"
    assert s.weatherlink_units["temp"] == "auto"
    again = W.SiteSettings.from_dict(json.loads(json.dumps(s.to_dict())))
    assert again.to_dict() == s.to_dict()
    assert not W.SiteSettings().has_location
    assert W.SiteSettings().to_dict()["latitude"] is None
    assert not W.valid_coordinates(91, 0) and not W.valid_coordinates("x", 0)
    assert W.validate_url_template("https://h/x{date:%Y%m%d}.txt") is None
    assert W.validate_url_template("https://h/x.txt")
    assert W.validate_url_template("ftp://h/{date:%Y}")
    assert W.nearest_surfrad(36.0, -116.5)[0] == "dra"
    assert W.compass(350) == "N" and W.compass(95) == "E"


if __name__ == "__main__":
    for name, fn in list(globals().items()):
        if name.startswith("test_") and callable(fn):
            fn()
            print("ok", name)
