"""Weather, sunlight and sun position for a tracking trial.

The site is whatever latitude/longitude the user types (or loads from a site
profile kept next to their data). Nothing here knows any particular site.

Three kinds of source, each downloaded once and cached beside the analysis
so a re-run is reproducible and works offline:

``surfrad``      NOAA SURFRAD network, nearest station to the site. MEASURED
                 1-minute solar radiation (global, direct, diffuse, PAR, UV-B)
                 plus 10 m air temperature, humidity, wind and pressure.
``open_meteo``   Open-Meteo weather model at the site itself. 15-minute values
                 (hourly ERA5 reanalysis when 15-minute data are unavailable).
                 Modelled, not measured.
``weatherlink``  A Davis WeatherLink text archive at a URL the user supplies,
                 one file per day - e.g. a station a university publishes.

and ``computed``: the sun's position from the NOAA solar equations, and the
moon's position, phase and illumination from the Astronomical Almanac's
low-precision lunar series - both exact enough for behaviour work at any
instant, and needing no network.

Time conventions
----------------
Every record is stamped at the END of the interval it averages (the SURFRAD,
WeatherLink and Open-Meteo radiation convention alike) and carries that
interval in ``interval_s``. A moment ``t`` is described by the record whose
interval contains it. All times are handled as UTC int64 nanoseconds and only
converted to the trial's timezone for display and export.

Nothing here imports Qt, so the parsing and the arithmetic are testable
headless.
"""
from __future__ import annotations

import hashlib
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from dataclasses import dataclass, field
from datetime import date, datetime, timedelta, timezone

import numpy as np
import pandas as pd

NS_PER_S = 1_000_000_000
NS_PER_MIN = 60 * NS_PER_S
NS_PER_DAY = 86_400 * NS_PER_S

USER_AGENT = "FNT-UWB-weather/1.0 (+https://github.com/calebvogt/fnt)"

WEATHER_SOURCES = ("none", "surfrad", "open_meteo", "weatherlink")
SOLAR_SOURCES = ("computed", "surfrad", "open_meteo")

SOURCE_LABELS = {
    "none": "None",
    "computed": "Computed sun position only (offline)",
    "surfrad": "NOAA SURFRAD - nearest station (measured, 1 min)",
    "open_meteo": "Open-Meteo - weather model at site (15 min)",
    "weatherlink": "WeatherLink text archive at a URL (station, 5 min typ.)",
}

#: Canonical measurement columns, in export order. Every source frame carries
#: all of them (NaN where the source does not measure it), so the export has
#: one stable schema whichever sources were chosen.
FIELDS = (
    "temp_c", "rh_pct", "dewpoint_c", "wind_speed_ms", "wind_dir_deg",
    "wind_gust_ms", "pressure_hpa", "precip_mm", "precip_rate_mmh", "rain_mm",
    "snowfall_cm", "ghi_wm2", "dni_wm2", "dhi_wm2", "par_wm2", "uvb_mwm2",
    "uv_index",
)

#: Precipitation kinds, and the colour each is drawn in (r, g, b, 0-1).
PRECIP_COLORS = {"rain": (0.25, 0.56, 1.0), "snow": (0.93, 0.96, 1.0),
                 "mixed": (0.72, 0.62, 1.0)}
#: Rate (mm/h, water equivalent) drawn at full strength: heavy precipitation.
PRECIP_FULL_MMH = 8.0

#: SURFRAD stations (public NOAA network metadata): code -> name, lat, lon, m.
SURFRAD_STATIONS = {
    "bon": ("Bondville, IL", 40.05, -88.37, 213),
    "dra": ("Desert Rock, NV", 36.624, -116.019, 1007),
    "fpk": ("Fort Peck, MT", 48.31, -105.10, 634),
    "gwn": ("Goodwin Creek, MS", 34.25, -89.87, 98),
    "psu": ("Penn State, PA", 40.72, -77.93, 376),
    "sxf": ("Sioux Falls, SD", 43.73, -96.62, 473),
    "tbl": ("Table Mountain, CO", 40.125, -105.237, 1689),
}
SURFRAD_URL = ("https://gml.noaa.gov/aftp/data/radiation/surfrad/"
               "{code}/{year}/{code}{yy:02d}{doy:03d}.dat")

OPEN_METEO_15MIN_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
OPEN_METEO_HOURLY_URL = "https://archive-api.open-meteo.com/v1/archive"
OPEN_METEO_VARS = {
    "temperature_2m": "temp_c",
    "relative_humidity_2m": "rh_pct",
    "dew_point_2m": "dewpoint_c",
    "wind_speed_10m": "wind_speed_ms",
    "wind_direction_10m": "wind_dir_deg",
    "wind_gusts_10m": "wind_gust_ms",
    "surface_pressure": "pressure_hpa",
    "precipitation": "precip_mm",
    "rain": "rain_mm",
    "snowfall": "snowfall_cm",
    "shortwave_radiation": "ghi_wm2",
    "direct_normal_irradiance": "dni_wm2",
    "diffuse_radiation": "dhi_wm2",
}

#: Sun-elevation thresholds (degrees, geometric). -0.833 is the standard
#: sunrise/sunset: the upper limb on the horizon, refraction included.
SUNRISE_ELEV = -0.833
TWILIGHTS = (("civil", -6.0), ("nautical", -12.0), ("astronomical", -18.0))

#: Moon phase names by elongation bin (45 degrees each, centred on the
#: principal phases).
MOON_PHASES = ("new moon", "waxing crescent", "first quarter", "waxing gibbous",
               "full moon", "waning gibbous", "last quarter", "waning crescent")

COMPASS = ("N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
           "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW")


# --------------------------------------------------------------------------- #
# Sun
# --------------------------------------------------------------------------- #
def _julian(t_ns):
    """(Julian day, Julian centuries since J2000) for int64-ns UTC times."""
    jd = np.asarray(t_ns, dtype="int64") / NS_PER_DAY + 2440587.5
    return jd, (jd - 2451545.0) / 36525.0


def _sun_apparent_longitude(jc):
    """Apparent ecliptic longitude of the sun, degrees (NOAA/Meeus)."""
    mean_long = 280.46646 + jc * (36000.76983 + jc * 0.0003032)
    m = np.radians(357.52911 + jc * (35999.05029 - 0.0001537 * jc))
    centre = (np.sin(m) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
              + np.sin(2 * m) * (0.019993 - 0.000101 * jc)
              + np.sin(3 * m) * 0.000289)
    omega = np.radians(125.04 - 1934.136 * jc)
    return np.mod(mean_long + centre - 0.00569 - 0.00478 * np.sin(omega), 360.0)


def refraction_deg(elevation):
    """NOAA's atmospheric refraction correction (degrees) for a geometric
    elevation. Adds about 0.5 degree at the horizon and nothing overhead."""
    h = np.asarray(elevation, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        te = np.tan(np.radians(h))
        r = np.select(
            [h > 85.0, h > 5.0, h > -0.575],
            [0.0,
             58.1 / te - 0.07 / te ** 3 + 0.000086 / te ** 5,
             1735.0 + h * (-518.2 + h * (103.4 + h * (-12.79 + h * 0.711)))],
            default=-20.772 / te)
    return r / 3600.0


def solar_position(times_ns, lat, lon, refraction=False):
    """(elevation, azimuth) in degrees for UTC int64-ns times. Vectorised.

    NOAA solar-calculator equations (Meeus), good to about 0.01 degree over
    1900-2100. Geometric by default; ``refraction=True`` gives the APPARENT
    elevation (what SURFRAD reports). Sunrise/sunset use the geometric value
    against -0.833, which already allows for refraction.
    """
    t = np.asarray(times_ns, dtype="int64")
    jd, jc = _julian(t)
    mean_long = np.mod(280.46646 + jc * (36000.76983 + jc * 0.0003032), 360.0)
    mean_anom = 357.52911 + jc * (35999.05029 - 0.0001537 * jc)
    ecc = 0.016708634 - jc * (0.000042037 + 0.0000001267 * jc)
    m = np.radians(mean_anom)
    centre = (np.sin(m) * (1.914602 - jc * (0.004817 + 0.000014 * jc))
              + np.sin(2 * m) * (0.019993 - 0.000101 * jc)
              + np.sin(3 * m) * 0.000289)
    omega = np.radians(125.04 - 1934.136 * jc)
    app_long = np.radians(mean_long + centre - 0.00569 - 0.00478 * np.sin(omega))
    obliq = (23.0 + (26.0 + (21.448 - jc * (46.815 + jc * (0.00059 - jc * 0.001813)))
                     / 60.0) / 60.0) + 0.00256 * np.cos(omega)
    obliq = np.radians(obliq)
    decl = np.arcsin(np.sin(obliq) * np.sin(app_long))
    y = np.tan(obliq / 2.0) ** 2
    l0 = np.radians(mean_long)
    eq_time = 4.0 * np.degrees(
        y * np.sin(2 * l0) - 2 * ecc * np.sin(m)
        + 4 * ecc * y * np.sin(m) * np.cos(2 * l0)
        - 0.5 * y * y * np.sin(4 * l0) - 1.25 * ecc * ecc * np.sin(2 * m))
    minutes = np.mod(t, NS_PER_DAY) / NS_PER_MIN
    true_solar = np.mod(minutes + eq_time + 4.0 * lon, 1440.0)
    hour_angle = np.radians(true_solar / 4.0 - 180.0)
    phi = math.radians(lat)
    cos_zen = (math.sin(phi) * np.sin(decl)
               + math.cos(phi) * np.cos(decl) * np.cos(hour_angle))
    zen = np.arccos(np.clip(cos_zen, -1.0, 1.0))
    elevation = 90.0 - np.degrees(zen)
    if refraction:
        elevation = elevation + refraction_deg(elevation)
    with np.errstate(invalid="ignore", divide="ignore"):
        cos_az = ((math.sin(phi) * np.cos(zen) - np.sin(decl))
                  / (math.cos(phi) * np.sin(zen)))
    az = np.degrees(np.arccos(np.clip(cos_az, -1.0, 1.0)))
    azimuth = np.where(hour_angle > 0, np.mod(az + 180.0, 360.0),
                       np.mod(540.0 - az, 360.0))
    return elevation, azimuth


# --------------------------------------------------------------------------- #
# Moon
# --------------------------------------------------------------------------- #
def _moon_ecliptic(jc):
    """(longitude, latitude, horizontal parallax), degrees.

    The Astronomical Almanac's low-precision series: about 0.3 degree in
    longitude and 0.2 in latitude, i.e. moonrise to a couple of minutes.
    """
    def s(a, b):
        return np.sin(np.radians(a + b * jc))

    def c(a, b):
        return np.cos(np.radians(a + b * jc))

    lam = (218.32 + 481267.881 * jc
           + 6.29 * s(135.0, 477198.87) - 1.27 * s(259.3, -413335.36)
           + 0.66 * s(235.7, 890534.22) + 0.21 * s(269.9, 954397.74)
           - 0.19 * s(357.5, 35999.05) - 0.11 * s(186.5, 966404.03))
    beta = (5.13 * s(93.3, 483202.02) + 0.28 * s(228.2, 960400.89)
            - 0.28 * s(318.3, 6003.15) - 0.17 * s(217.6, -407332.21))
    par = (0.9508 + 0.0518 * c(135.0, 477198.87) + 0.0095 * c(259.3, -413335.36)
           + 0.0078 * c(235.7, 890534.22) + 0.0028 * c(269.9, 954397.74))
    return np.mod(lam, 360.0), beta, par


def moon_position(times_ns, lat, lon):
    """Topocentric (elevation, azimuth) of the moon's centre, degrees.

    Parallax is applied - up to a degree near the horizon, which is what
    moves moonrise by several minutes - and refraction is not; moonrise uses
    ``moon_horizon`` which allows for both.
    """
    t = np.asarray(times_ns, dtype="int64")
    jd, jc = _julian(t)
    lam, beta, par = _moon_ecliptic(jc)
    eps = np.radians(23.439291 - 0.0130042 * jc)
    lr, br = np.radians(lam), np.radians(beta)
    ra = np.arctan2(np.sin(lr) * np.cos(eps) - np.tan(br) * np.sin(eps), np.cos(lr))
    dec = np.arcsin(np.sin(br) * np.cos(eps)
                    + np.cos(br) * np.sin(eps) * np.sin(lr))
    gmst = np.mod(280.46061837 + 360.98564736629 * (jd - 2451545.0)
                  + 0.000387933 * jc * jc, 360.0)
    ha = np.radians(gmst + lon) - ra
    phi = math.radians(lat)
    alt = np.arcsin(np.clip(math.sin(phi) * np.sin(dec)
                            + math.cos(phi) * np.cos(dec) * np.cos(ha), -1, 1))
    az = np.degrees(np.arctan2(np.sin(ha), np.cos(ha) * math.sin(phi)
                               - np.tan(dec) * math.cos(phi))) + 180.0
    elev = np.degrees(alt) - par * np.cos(alt)
    return elev, np.mod(az, 360.0)


def moon_horizon(times_ns):
    """Elevation of the moon's centre at rise/set: -(refraction + radius)."""
    _jd, jc = _julian(times_ns)
    _lam, _beta, par = _moon_ecliptic(jc)
    return -(0.5667 + 0.2725 * par)


def moon_phase(times_ns):
    """(illuminated fraction 0-1, elongation angle 0-360, waxing, name).

    The angle is the moon's ecliptic longitude minus the sun's: 0 new,
    90 first quarter, 180 full, 270 last quarter. Names are 45-degree bins
    centred on those four.
    """
    t = np.asarray(times_ns, dtype="int64")
    _jd, jc = _julian(t)
    lam, beta, _par = _moon_ecliptic(jc)
    d = np.mod(lam - _sun_apparent_longitude(jc), 360.0)
    cos_e = np.cos(np.radians(beta)) * np.cos(np.radians(d))
    frac = (1.0 - cos_e) / 2.0
    names = np.array(MOON_PHASES, dtype=object)[
        (np.floor((d + 22.5) / 45.0).astype(int)) % 8]
    return frac, d, d < 180.0, names


def moonlight_level(moon_elevation, illumination):
    """0-1 moonlight index: brightness of the lit disc, dimmed low in the sky."""
    e = np.clip(np.asarray(moon_elevation, dtype=float), 0.0, 90.0)
    k = np.clip(np.asarray(illumination, dtype=float), 0.0, 1.0)
    return (k ** 1.5) * np.sqrt(np.sin(np.radians(e)))


def moon_disc_polygon(illumination, waxing, southern=False, n=48):
    """Outline of the LIT part of a unit moon disc, as an (m, 2) array.

    Waxing moons are lit on the right as seen from the northern hemisphere
    (mirrored for the southern). The terminator is a half-ellipse whose
    half-width is |1 - 2k| of the radius.
    """
    k = float(np.clip(illumination, 0.0, 1.0))
    ang = np.linspace(-np.pi / 2, np.pi / 2, n)
    limb = np.column_stack([np.cos(ang), np.sin(ang)])
    back = ang[::-1]
    term = np.column_stack([(1.0 - 2.0 * k) * np.cos(back), np.sin(back)])
    pts = np.vstack([limb, term])
    if (not waxing) != bool(southern):
        pts[:, 0] = -pts[:, 0]
    return pts


def sun_phase(elevation):
    """'day' / 'civil' / 'nautical' / 'astronomical' / 'night' per elevation."""
    e = np.asarray(elevation, dtype=float)
    out = np.full(e.shape, "night", dtype=object)
    for name, lim in reversed(TWILIGHTS):
        out[e > lim] = name
    out[e > SUNRISE_ELEV] = "day"
    return out


def clear_sky_ghi(elevation):
    """Haurwitz clear-sky global irradiance, W/m^2, from sun elevation."""
    cz = np.sin(np.radians(np.asarray(elevation, dtype=float)))
    with np.errstate(divide="ignore", invalid="ignore", over="ignore"):
        g = 1098.0 * cz * np.exp(-0.057 / cz)
    return np.where(cz > 0, g, 0.0)


def light_level(elevation, ghi=None):
    """0-1 brightness for the light bar.

    Measured (or modelled) irradiance where given, clear-sky irradiance where
    not, on a square-root scale so dawn is visible; plus a dim twilight glow
    from the sun's depth below the horizon, which irradiance cannot show.
    """
    e = np.asarray(elevation, dtype=float)
    ref = clear_sky_ghi(e)
    if ghi is not None:
        g = np.asarray(ghi, dtype=float)
        ref = np.where(np.isfinite(g), g, ref)
    day = np.sqrt(np.clip(ref / 1000.0, 0.0, 1.0))
    glow = np.clip((e + 18.0) / 18.0, 0.0, 1.0) ** 2 * 0.15
    return np.maximum(day, glow)


def light_rgb(level, moon=0.0):
    """Night navy -> daylight cream, for a 0-1 light level. (r, g, b) 0-1.

    ``moon`` (0-1 moonlight index) silvers the dark end, so a moonlit night
    reads differently from a moonless one without competing with twilight.
    """
    night = np.array([0.043, 0.063, 0.149])
    dusk = np.array([0.93, 0.48, 0.25])
    day = np.array([1.0, 0.957, 0.76])
    lv = float(np.clip(level, 0.0, 1.0))
    if lv < 0.15:
        f = lv / 0.15
        rgb = night + (dusk - night) * f * 0.55
    else:
        f = (lv - 0.15) / 0.85
        base = night + (dusk - night) * 0.55
        rgb = base + (day - base) * min(1.0, f * 1.4)
    m = float(np.clip(moon if np.isfinite(moon) else 0.0, 0.0, 1.0))
    if m > 0:
        dark = max(0.0, 1.0 - lv / 0.15)
        silver = np.array([0.36, 0.43, 0.62])
        rgb = rgb + (silver - rgb) * m * 0.75 * dark
    return tuple(float(c) for c in np.clip(rgb, 0, 1))


def _crossings(t_ns, values, level):
    """(rising, falling) crossing times of ``values`` through ``level``."""
    v = np.asarray(values, dtype=float) - level
    s = np.sign(v)
    idx = np.flatnonzero((s[:-1] <= 0) & (s[1:] > 0))
    rise = [_interp_cross(t_ns, v, i) for i in idx]
    idx = np.flatnonzero((s[:-1] > 0) & (s[1:] <= 0))
    fall = [_interp_cross(t_ns, v, i) for i in idx]
    return rise, fall


def _interp_cross(t_ns, v, i):
    a, b = v[i], v[i + 1]
    frac = 0.0 if b == a else -a / (b - a)
    return int(t_ns[i] + frac * (t_ns[i + 1] - t_ns[i]))


def daylight_table(lat, lon, tz, start_ns, end_ns):
    """One row per local calendar day of the trial: sunrise, sunset, twilights.

    Scans each local day (23 or 25 h on a DST change) at 1-minute steps and
    interpolates the crossings, so times are good to a few seconds. A day with
    no crossing (polar day/night) leaves those cells empty.
    """
    d0 = pd.Timestamp(int(start_ns), tz="UTC").tz_convert(tz).normalize()
    d1 = pd.Timestamp(int(end_ns), tz="UTC").tz_convert(tz).normalize()
    rows = []
    for i, day in enumerate(pd.date_range(d0.tz_localize(None),
                                          d1.tz_localize(None), freq="D")):
        lo = pd.Timestamp(day).tz_localize(tz, ambiguous=True,
                                           nonexistent="shift_forward")
        hi = (pd.Timestamp(day) + pd.Timedelta(days=1)).tz_localize(
            tz, ambiguous=True, nonexistent="shift_forward")
        t = np.arange(lo.value, hi.value + NS_PER_MIN, NS_PER_MIN, dtype="int64")
        el, _az = solar_position(t, lat, lon)

        def _local(ns):
            return pd.Timestamp(int(ns), tz="UTC").tz_convert(tz)

        def _first(xs):
            return _local(xs[0]) if xs else pd.NaT

        def _last(xs):
            return _local(xs[-1]) if xs else pd.NaT

        rise, fall = _crossings(t, el, SUNRISE_ELEV)
        k = int(np.argmax(el))
        row = {
            "Date": day.date(),
            "Day": i + 1,
            "sunrise": _first(rise),
            "sunset": _last(fall),
            "solar_noon": _local(t[k]),
            "max_elevation_deg": round(float(el[k]), 3),
        }
        if rise and fall and fall[-1] > rise[0]:
            row["day_length_h"] = round((fall[-1] - rise[0]) / NS_PER_S / 3600, 4)
        else:
            row["day_length_h"] = (24.0 if el.min() > SUNRISE_ELEV else 0.0)
        for name, lim in TWILIGHTS:
            r, f = _crossings(t, el, lim)
            row[f"{name}_dawn"] = _first(r)
            row[f"{name}_dusk"] = _last(f)
        mel, _maz = moon_position(t, lat, lon)
        mh = mel - moon_horizon(t)
        mrise, mset = _crossings(t, mh, 0.0)
        row["moonrise"] = _first(mrise)
        row["moonset"] = _first(mset)
        # The night that FOLLOWS this date's sunset, judged at the midnight
        # that ends the date - the moon a nocturnal animal met that night.
        k, _d, _wax, name = moon_phase(np.array([t[-1]]))
        row["moon_illumination_midnight"] = round(float(k[0]), 4)
        row["moon_phase_midnight"] = str(name[0])
        dark = el < -6.0
        row["moonlit_dark_h"] = round(float((dark & (mh > 0)).sum()) / 60.0, 3)
        row["dark_h"] = round(float(dark.sum()) / 60.0, 3)
        rows.append(row)
    cols = ["Date", "Day", "sunrise", "sunset", "solar_noon", "day_length_h",
            "max_elevation_deg", "civil_dawn", "civil_dusk", "nautical_dawn",
            "nautical_dusk", "astronomical_dawn", "astronomical_dusk",
            "moonrise", "moonset", "moon_illumination_midnight",
            "moon_phase_midnight", "dark_h", "moonlit_dark_h"]
    return pd.DataFrame(rows, columns=cols)


# --------------------------------------------------------------------------- #
# Geometry / units helpers
# --------------------------------------------------------------------------- #
def haversine_km(lat1, lon1, lat2, lon2):
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dp, dl = p2 - p1, math.radians(lon2 - lon1)
    a = math.sin(dp / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dl / 2) ** 2
    return 2 * 6371.0 * math.asin(math.sqrt(a))


def nearest_surfrad(lat, lon):
    """(code, name, distance_km) of the SURFRAD station nearest the site."""
    best = min(SURFRAD_STATIONS.items(),
               key=lambda kv: haversine_km(lat, lon, kv[1][1], kv[1][2]))
    code, (name, slat, slon, _elev) = best
    return code, name, haversine_km(lat, lon, slat, slon)


def compass(deg):
    if deg is None or not np.isfinite(deg):
        return ""
    return COMPASS[int((float(deg) % 360.0) / 22.5 + 0.5) % 16]


def compass_to_deg(token):
    t = str(token).strip().upper()
    return COMPASS.index(t) * 22.5 if t in COMPASS else np.nan


def valid_coordinates(lat, lon):
    try:
        lat, lon = float(lat), float(lon)
    except (TypeError, ValueError):
        return False
    return (np.isfinite(lat) and np.isfinite(lon)
            and -90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0)


def _empty_frame():
    df = pd.DataFrame({"time_ns": pd.Series(dtype="int64"),
                       "interval_s": pd.Series(dtype="float64")})
    for f in FIELDS:
        df[f] = pd.Series(dtype="float64")
    df["precip_type"] = pd.Series(dtype="object")
    df["precip_basis"] = pd.Series(dtype="object")
    df["time_flag"] = pd.Series(dtype="object")
    return df


def wet_bulb_c(temp_c, rh_pct):
    """Wet-bulb temperature (Stull 2011), degrees C. Good to ~1 degree for
    RH 5-99% and -20..50 C, which is all a rain/snow split needs."""
    t = np.asarray(temp_c, dtype=float)
    rh = np.clip(np.asarray(rh_pct, dtype=float), 1.0, 100.0)
    return (t * np.arctan(0.151977 * np.sqrt(rh + 8.313659))
            + np.arctan(t + rh) - np.arctan(rh - 1.676331)
            + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh) - 4.686035)


def classify_precip(df):
    """(precip_type, precip_basis) arrays for a canonical frame.

    Where the source splits rain from snow (Open-Meteo) that split is used.
    Otherwise the kind is inferred from the wet-bulb temperature - snow below
    0.5 C, mixed to 1.5 C, rain above - or from air temperature (0 / 2 C)
    when there is no humidity. A tipping-bucket gauge records snow only as it
    melts, so station snow can appear late and at above-freezing temperatures.
    """
    n = len(df)
    kind = np.full(n, "", dtype=object)
    basis = np.full(n, "", dtype=object)
    p = df["precip_mm"].to_numpy(dtype=float)
    wet = np.isfinite(p) & (p > 0)
    rain = df["rain_mm"].to_numpy(dtype=float)
    snow = df["snowfall_cm"].to_numpy(dtype=float)
    split = wet & np.isfinite(rain) & np.isfinite(snow)
    kind[split & (snow > 0) & (rain > 0)] = "mixed"
    kind[split & (snow > 0) & ~(rain > 0)] = "snow"
    kind[split & ~(snow > 0)] = "rain"
    basis[split] = "reported"
    rest = wet & ~split
    if rest.any():
        t = df["temp_c"].to_numpy(dtype=float)
        rh = df["rh_pct"].to_numpy(dtype=float)
        use_tw = np.isfinite(t) & np.isfinite(rh)
        tw = np.where(use_tw, wet_bulb_c(np.where(np.isfinite(t), t, 0.0),
                                         np.where(np.isfinite(rh), rh, 50.0)), t)
        lo = np.where(use_tw, 0.5, 0.0)
        hi = np.where(use_tw, 1.5, 2.0)
        ok = rest & np.isfinite(tw)
        kind[ok & (tw < lo)] = "snow"
        kind[ok & (tw >= lo) & (tw <= hi)] = "mixed"
        kind[ok & (tw > hi)] = "rain"
        kind[rest & ~np.isfinite(tw)] = "rain"
        basis[ok & use_tw] = "wet-bulb"
        basis[ok & ~use_tw] = "air temperature"
        basis[rest & ~np.isfinite(tw)] = "assumed"
    return kind, basis


def _finish_frame(df):
    """Canonical column set and order, sorted, one row per time."""
    for f in FIELDS:
        if f not in df.columns:
            df[f] = np.nan
    if "time_flag" not in df.columns:
        df["time_flag"] = ""
    df["time_flag"] = df["time_flag"].fillna("")
    df = df.copy()
    df["precip_rate_mmh"] = (df["precip_mm"].astype(float) * 3600.0
                             / df["interval_s"].astype(float))
    df["precip_type"], df["precip_basis"] = classify_precip(df)
    df = df[["time_ns", "interval_s", *FIELDS, "precip_type", "precip_basis",
             "time_flag"]]
    df = (df.sort_values("time_ns", kind="stable")
            .drop_duplicates("time_ns", keep="last")
            .reset_index(drop=True))
    df["time_ns"] = df["time_ns"].astype("int64")
    return df


# --------------------------------------------------------------------------- #
# Download cache
# --------------------------------------------------------------------------- #
class DownloadCache:
    """Raw downloads under ``root``, with a manifest of where each came from.

    A file is re-fetched only when it may have been incomplete: it was saved
    before the period it covers had finished (plus ``settle_h`` for the
    publisher to post the rest), and it is older than ``min_age_s``.
    """

    def __init__(self, root, offline=False, log=None, timeout=30,
                 settle_h=6.0, min_age_s=1800):
        self.root = root
        self.offline = offline
        self.log = log or (lambda msg: None)
        self.timeout = timeout
        self.settle_h = settle_h
        self.min_age_s = min_age_s
        self.manifest_path = os.path.join(root, "manifest.json")
        self.downloaded = 0
        self.reused = 0
        self.failed = []
        try:
            with open(self.manifest_path) as f:
                self.manifest = json.load(f)
        except (OSError, ValueError):
            self.manifest = {}

    def _stale(self, rel, period_end_utc):
        rec = self.manifest.get(rel)
        if rec is None:
            return True
        try:
            got = datetime.fromisoformat(rec["retrieved_utc"])
        except (KeyError, ValueError):
            return True
        settled = period_end_utc + timedelta(hours=self.settle_h)
        if got >= settled:
            return False
        age = (datetime.now(timezone.utc) - got).total_seconds()
        return age > self.min_age_s

    def get(self, url, rel, period_end_utc):
        """Local path for ``url`` (downloading if needed), or None."""
        path = os.path.join(self.root, rel)
        have = os.path.exists(path)
        if have and not self._stale(rel, period_end_utc):
            self.reused += 1
            return path
        if self.offline:
            if have:
                self.reused += 1
            return path if have else None
        try:
            req = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                body = r.read()
        except urllib.error.HTTPError as e:
            if e.code == 404:
                # Not published (yet): a day beyond the archive, or a gap.
                self.failed.append((url, "not published (404)"))
            else:
                self.failed.append((url, f"HTTP {e.code}"))
            return path if have else None
        except (urllib.error.URLError, OSError, TimeoutError) as e:
            self.failed.append((url, f"network: {getattr(e, 'reason', e)}"))
            return path if have else None
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".part"
        with open(tmp, "wb") as f:
            f.write(body)
        os.replace(tmp, path)
        self.manifest[rel] = {
            "url": url,
            "retrieved_utc": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "bytes": len(body),
            "sha256": hashlib.sha256(body).hexdigest(),
        }
        self._save_manifest()
        self.downloaded += 1
        return path

    def _save_manifest(self):
        os.makedirs(self.root, exist_ok=True)
        tmp = self.manifest_path + ".part"
        with open(tmp, "w") as f:
            json.dump(self.manifest, f, indent=1, sort_keys=True)
        os.replace(tmp, self.manifest_path)

    def provenance(self, rels):
        return [dict(file=r, **self.manifest[r]) for r in rels if r in self.manifest]


# --------------------------------------------------------------------------- #
# SURFRAD
# --------------------------------------------------------------------------- #
SURFRAD_COLUMNS = (
    "dw_solar", "uw_solar", "direct_n", "diffuse", "dw_ir", "dw_casetemp",
    "dw_dometemp", "uw_ir", "uw_casetemp", "uw_dometemp", "uvb", "par",
    "netsolar", "netir", "totalnet", "temp", "rh", "windspd", "winddir",
    "pressure")


def parse_surfrad(text):
    """(station_info, frame) from one SURFRAD daily file."""
    lines = text.splitlines()
    info = {"name": lines[0].strip() if lines else ""}
    if len(lines) > 1:
        parts = lines[1].split()
        try:
            info.update(latitude=float(parts[0]), longitude=float(parts[1]),
                        elevation_m=float(parts[2]))
        except (IndexError, ValueError):
            pass
    rows = [ln.split() for ln in lines[2:] if ln.strip()]
    rows = [r for r in rows if len(r) == 8 + 2 * len(SURFRAD_COLUMNS)]
    if not rows:
        return info, _empty_frame()
    a = np.array(rows, dtype=float)
    stamp = pd.to_datetime(dict(year=a[:, 0], month=a[:, 2], day=a[:, 3],
                                hour=a[:, 4], minute=a[:, 5]), utc=True)
    out = pd.DataFrame({"time_ns": stamp.dt.as_unit("ns").astype("int64").to_numpy()})
    zen = a[:, 7]
    vals = {}
    for k, name in enumerate(SURFRAD_COLUMNS):
        v = a[:, 8 + 2 * k].copy()
        qc = a[:, 9 + 2 * k]
        # -9999.9 is missing; QC 1 is a knowingly bad value. Higher flags are
        # "use with scrutiny" and are kept, as SURFRAD recommends.
        v[(v <= -9999.0) | (qc == 1)] = np.nan
        vals[name] = v
    # The thermopile global pyranometer carries an infrared offset SURFRAD
    # says makes it the fallback; direct*cos(zenith) + diffuse is preferred.
    comp = vals["direct_n"] * np.clip(np.cos(np.radians(zen)), 0, None) + vals["diffuse"]
    out["ghi_wm2"] = np.where(np.isfinite(comp), comp, vals["dw_solar"])
    out["dni_wm2"] = vals["direct_n"]
    out["dhi_wm2"] = vals["diffuse"]
    out["par_wm2"] = vals["par"]
    out["uvb_mwm2"] = vals["uvb"]
    out["temp_c"] = vals["temp"]
    out["rh_pct"] = vals["rh"]
    out["wind_speed_ms"] = vals["windspd"]
    out["wind_dir_deg"] = vals["winddir"]
    out["pressure_hpa"] = vals["pressure"]
    step = np.median(np.diff(out["time_ns"])) / NS_PER_S if len(out) > 2 else 60.0
    out["interval_s"] = 180.0 if step > 120 else 60.0
    out["zenith_deg"] = zen
    return info, out


def fetch_surfrad(lat, lon, start_ns, end_ns, cache, log=None):
    log = log or (lambda m: None)
    code, name, dist = nearest_surfrad(lat, lon)
    frames, rels, info = [], [], {}
    day = pd.Timestamp(int(start_ns), tz="UTC").normalize()
    # Records are stamped at the END of their minute and a file starts with
    # its 00:00 record, so the last minute of a day lives in the NEXT file.
    last = pd.Timestamp(int(end_ns) + NS_PER_MIN, tz="UTC").normalize()
    now = pd.Timestamp(time.time_ns(), tz="UTC")
    while day <= min(last, now.normalize()):
        url = SURFRAD_URL.format(code=code, year=day.year, yy=day.year % 100,
                                 doy=day.dayofyear)
        rel = f"surfrad/{code}/{os.path.basename(url)}"
        path = cache.get(url, rel, (day + pd.Timedelta(days=1)).to_pydatetime())
        if path:
            with open(path, encoding="latin-1") as f:
                info, fr = parse_surfrad(f.read())
            frames.append(fr)
            rels.append(rel)
        day += pd.Timedelta(days=1)
    df = pd.concat(frames, ignore_index=True) if frames else _empty_frame()
    df = df[(df["time_ns"] > start_ns - 180 * NS_PER_S)
            & (df["time_ns"] <= end_ns + 180 * NS_PER_S)]
    meta = {
        "source": "surfrad", "kind": "measured", "station_code": code,
        "station": name, "distance_km": round(dist, 2),
        "station_latitude": info.get("latitude"),
        "station_longitude": info.get("longitude"),
        "station_elevation_m": info.get("elevation_m"),
        "interval_s": float(df["interval_s"].median()) if len(df) else 60.0,
        "files": cache.provenance(rels),
    }
    return _finish_frame(df.drop(columns=["zenith_deg"], errors="ignore")), meta


# --------------------------------------------------------------------------- #
# Open-Meteo
# --------------------------------------------------------------------------- #
def parse_open_meteo(payload, key, now_ns=None):
    block = payload.get(key) or {}
    times = block.get("time") or []
    if not times:
        return _empty_frame()
    t = pd.to_datetime(pd.Series(times), utc=True, format="ISO8601")
    df = pd.DataFrame({"time_ns": t.dt.as_unit("ns").astype("int64").to_numpy()})
    for var, col in OPEN_METEO_VARS.items():
        df[col] = pd.to_numeric(pd.Series(block.get(var, [None] * len(times))),
                                errors="coerce").to_numpy()
    df["interval_s"] = 900.0 if key == "minutely_15" else 3600.0
    have = df[list(OPEN_METEO_VARS.values())].notna().any(axis=1)
    df = df[have]
    if now_ns is not None:
        # The historical-forecast endpoint continues into the FUTURE with
        # forecast values; only what has already happened is kept.
        df = df[df["time_ns"] <= now_ns]
    return df


def fetch_open_meteo(lat, lon, start_ns, end_ns, cache, log=None, now_ns=None):
    now_ns = now_ns if now_ns is not None else time.time_ns()
    d0 = pd.Timestamp(int(start_ns), tz="UTC").date()
    d1 = pd.Timestamp(int(end_ns), tz="UTC").date()
    frames, rels, used = [], [], set()
    chunk = d0
    tag = f"{lat:.4f}_{lon:.4f}"
    while chunk <= d1:
        stop = min(d1, chunk + timedelta(days=30))
        common = (f"latitude={lat:.5f}&longitude={lon:.5f}"
                  f"&start_date={chunk}&end_date={stop}&timezone=GMT"
                  f"&wind_speed_unit=ms")
        vars_ = ",".join(OPEN_METEO_VARS)
        end_utc = datetime.combine(stop + timedelta(days=1), datetime.min.time(),
                                   tzinfo=timezone.utc)
        fr = None
        url = f"{OPEN_METEO_15MIN_URL}?{common}&minutely_15={vars_}"
        rel = f"open_meteo/{tag}_{chunk}_{stop}_15min.json"
        path = cache.get(url, rel, end_utc)
        if path:
            try:
                with open(path) as f:
                    fr = parse_open_meteo(json.load(f), "minutely_15", now_ns)
                rels.append(rel)
                used.add("15 min model")
            except (OSError, ValueError):
                fr = None
        if fr is None or fr.empty:
            url = f"{OPEN_METEO_HOURLY_URL}?{common}&hourly={vars_}"
            rel = f"open_meteo/{tag}_{chunk}_{stop}_hourly.json"
            path = cache.get(url, rel, end_utc + timedelta(days=6))
            if path:
                try:
                    with open(path) as f:
                        fr = parse_open_meteo(json.load(f), "hourly", now_ns)
                    rels.append(rel)
                    used.add("hourly ERA5 reanalysis")
                except (OSError, ValueError):
                    fr = None
        if fr is not None and len(fr):
            frames.append(fr)
        chunk = stop + timedelta(days=1)
    df = pd.concat(frames, ignore_index=True) if frames else _empty_frame()
    pad = 3600 * NS_PER_S
    df = df[(df["time_ns"] > start_ns - pad) & (df["time_ns"] <= end_ns + pad)]
    meta = {
        "source": "open_meteo", "kind": "model", "station": "model grid at site",
        "distance_km": 0.0, "resolution": ", ".join(sorted(used)) or "none",
        "interval_s": float(df["interval_s"].median()) if len(df) else 900.0,
        "files": cache.provenance(rels),
    }
    return _finish_frame(df), meta


# --------------------------------------------------------------------------- #
# Davis WeatherLink text archive
# --------------------------------------------------------------------------- #
WEATHERLINK_MAP = {
    "Temp Out": "temp", "Out Hum": "rh", "Dew Pt.": "dewpoint",
    "Wind Speed": "wind_speed", "Wind Dir": "wind_dir", "Hi Speed": "wind_gust",
    "Bar": "pressure", "Rain": "precip", "Solar Rad.": "ghi",
    "UV Index": "uv_index", "Arc. Int.": "interval_min",
}
TEMP_UNITS = ("auto", "F", "C")
WIND_UNITS = ("mph", "m/s", "km/h", "knots")
RAIN_UNITS = ("in", "mm")
_WIND_TO_MS = {"mph": 0.44704, "m/s": 1.0, "km/h": 1 / 3.6, "knots": 0.514444}


def dewpoint_c(temp_c, rh_pct):
    """Dew point (Magnus, Alduchov & Eskridge 1996), degrees C."""
    t = np.asarray(temp_c, dtype=float)
    rh = np.clip(np.asarray(rh_pct, dtype=float), 1.0, 100.0)
    g = np.log(rh / 100.0) + 17.625 * t / (243.04 + t)
    return 243.04 * g / (17.625 - g)


def detect_temp_unit(temp, dewpoint, rh):
    """'F' or 'C': whichever makes the file's own dew point agree with its
    temperature and humidity. None when the file cannot tell (too few rows,
    no dew point, or readings where the two scales agree).

    WeatherLink files do not state their units, and a wrong guess is not
    obviously wrong - 22 °C read as °F is a cold day, not an error - so the
    choice is checked against physics rather than against plausibility.
    """
    t, d, h = (np.asarray(a, dtype=float) for a in (temp, dewpoint, rh))
    ok = np.isfinite(t) & np.isfinite(d) & np.isfinite(h) & (h > 5) & (h <= 100)
    if ok.sum() < 3:
        return None
    t, d, h = t[ok], d[ok], h[ok]

    def err(tc, dc):
        with np.errstate(invalid="ignore"):
            return float(np.nanmedian(np.abs(dewpoint_c(tc, h) - dc)))

    e_f = err((t - 32.0) * 5.0 / 9.0, (d - 32.0) * 5.0 / 9.0)
    e_c = err(t, d)
    if not (np.isfinite(e_f) and np.isfinite(e_c)):
        return None
    if min(e_f, e_c) > 3.0 or abs(e_f - e_c) < 1.0:
        return None
    return "F" if e_f < e_c else "C"


def weatherlink_columns(header1, header2):
    """Column names from WeatherLink's two right-aligned header lines."""
    top = [(m.start(), m.end(), m.group()) for m in re.finditer(r"\S+", header1)]
    cols = [[m.start(), m.end(), m.group(), []]
            for m in re.finditer(r"\S+", header2)]
    for s, e, word in top:
        overlap = [max(0, min(e, ce) - max(s, cs)) for cs, ce, _w, _t in cols]
        k = int(np.argmax(overlap))
        if overlap[k] > 0:
            cols[k][3].append(word)
    return [" ".join(t + [w]) for _s, _e, w, t in cols]


def parse_weatherlink(text, tz, units=None):
    """Canonical frame from one WeatherLink daily text export.

    Times are the logger's local wall clock. A spring-forward gap cannot
    occur in them; the repeated hour at fall-back is logged only once, so
    those rows are ambiguous and flagged ``ambiguous_dst``.
    """
    units = units or {}
    lines = text.splitlines()
    if len(lines) < 4:
        return _empty_frame()
    names = weatherlink_columns(lines[0], lines[1])
    rows = [ln.split() for ln in lines[2:]]
    rows = [r for r in rows if len(r) == len(names)
            and re.match(r"\d{1,2}/\d{1,2}/\d{2,4}$", r[0])]
    if not rows:
        return _empty_frame()
    raw = pd.DataFrame(rows, columns=[f"c{i}" for i in range(len(names))])
    stamp = pd.to_datetime(raw["c0"] + " " + raw["c1"].str.upper() + "M",
                           format="%m/%d/%y %I:%M%p", errors="coerce")
    ok = stamp.notna()
    raw, stamp = raw[ok], stamp[ok]
    loc = stamp.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT")
    flag = pd.Series("", index=raw.index, dtype=object)
    amb = loc.isna()
    if amb.any():
        first = stamp[amb].dt.tz_localize(tz, ambiguous=True,
                                          nonexistent="shift_forward")
        # Tell the two apart: a time that exists twice vs one that never did.
        twice = stamp[amb].dt.tz_localize(tz, ambiguous=False,
                                          nonexistent="shift_forward")
        flag[amb] = np.where(first.values == twice.values,
                             "nonexistent_dst", "ambiguous_dst")
        loc = loc.copy()
        loc[amb] = first
    df = pd.DataFrame({"time_ns": loc.dt.tz_convert("UTC").dt.as_unit("ns")
                       .astype("int64").to_numpy()})
    df["time_flag"] = flag.to_numpy()

    def col(key):
        for i, n in enumerate(names):
            if WEATHERLINK_MAP.get(n) == key:
                return raw[f"c{i}"]
        return None

    def num(key):
        s = col(key)
        return (pd.to_numeric(s, errors="coerce").to_numpy()
                if s is not None else np.full(len(raw), np.nan))

    t_unit = units.get("temp", "auto")
    w_unit = units.get("wind", "mph")
    r_unit = units.get("rain", "in")
    detected = detect_temp_unit(num("temp"), num("dewpoint"), num("rh"))
    if t_unit not in ("F", "C"):
        # Auto: trust the file's own dew point; US stations default to °F.
        t_unit = detected or "F"
    df.attrs["temp_unit_used"] = t_unit
    df.attrs["temp_unit_detected"] = detected

    def temp(v):
        return (v - 32.0) * 5.0 / 9.0 if t_unit == "F" else v

    df["temp_c"] = temp(num("temp"))
    df["dewpoint_c"] = temp(num("dewpoint"))
    df["rh_pct"] = num("rh")
    df["wind_speed_ms"] = num("wind_speed") * _WIND_TO_MS.get(w_unit, 1.0)
    df["wind_gust_ms"] = num("wind_gust") * _WIND_TO_MS.get(w_unit, 1.0)
    d = col("wind_dir")
    df["wind_dir_deg"] = (d.map(compass_to_deg).to_numpy(dtype=float)
                          if d is not None else np.nan)
    p = num("pressure")
    med = np.nanmedian(p) if np.isfinite(p).any() else np.nan
    if np.isfinite(med) and med < 40:          # inches of mercury
        p = p * 33.8639
    df["pressure_hpa"] = p
    df["precip_mm"] = num("precip") * (25.4 if r_unit == "in" else 1.0)
    df["ghi_wm2"] = num("ghi")
    df["uv_index"] = num("uv_index")
    iv = num("interval_min")
    if np.isfinite(iv).any():
        df["interval_s"] = np.where(np.isfinite(iv), iv * 60.0, np.nanmedian(iv) * 60.0)
    else:
        step = (np.median(np.diff(np.sort(df["time_ns"]))) / NS_PER_S
                if len(df) > 2 else 300.0)
        df["interval_s"] = float(step)
    return df


def validate_url_template(template):
    """Error text for a WeatherLink URL template, or None if usable."""
    t = (template or "").strip()
    if not t:
        return "enter the archive URL"
    if not re.match(r"https?://", t):
        return "the URL must start with http:// or https://"
    if "{date" not in t:
        return "put {date:%Y%m%d} where the date goes in the file name"
    try:
        t.format(date=date(2024, 1, 2))
    except (KeyError, IndexError, ValueError) as e:
        return f"cannot fill in the date: {e}"
    return None


def fetch_weatherlink(template, tz, units, start_ns, end_ns, cache, log=None):
    err = validate_url_template(template)
    if err:
        raise ValueError(f"WeatherLink URL: {err}")
    key = hashlib.sha1(template.encode()).hexdigest()[:10]
    d0 = pd.Timestamp(int(start_ns), tz="UTC").tz_convert(tz).date()
    d1 = pd.Timestamp(int(end_ns), tz="UTC").tz_convert(tz).date()
    frames, rels = [], []
    used, detected = set(), set()
    today = pd.Timestamp(time.time_ns(), tz="UTC").tz_convert(tz).date()
    day = d0
    # One day past the end: an archive file can carry the tail of the day
    # before its name. Never ask for a day that has not started.
    while day <= min(d1 + timedelta(days=1), today):
        url = template.format(date=day)
        rel = f"weatherlink/{key}/{day:%Y%m%d}_{os.path.basename(url.split('?')[0]) or 'day.txt'}"
        end_local = pd.Timestamp(day + timedelta(days=1)).tz_localize(
            tz, ambiguous=True, nonexistent="shift_forward")
        path = cache.get(url, rel, end_local.tz_convert("UTC").to_pydatetime())
        if path:
            with open(path, encoding="latin-1") as f:
                fr = parse_weatherlink(f.read(), tz, units)
            if len(fr):
                used.add(fr.attrs.get("temp_unit_used"))
                if fr.attrs.get("temp_unit_detected"):
                    detected.add(fr.attrs["temp_unit_detected"])
                frames.append(fr)
            rels.append(rel)
        day += timedelta(days=1)
    df = pd.concat(frames, ignore_index=True) if frames else _empty_frame()
    pad = 3600 * NS_PER_S
    df = df[(df["time_ns"] > start_ns - pad) & (df["time_ns"] <= end_ns + pad)]
    meta = {
        "source": "weatherlink", "kind": "measured",
        "station": "WeatherLink archive", "url_template": template,
        "units": dict(units or {}),
        "temp_unit_used": "/".join(sorted(u for u in used if u)) or None,
        "temp_unit_detected": "/".join(sorted(detected)) or None,
        "interval_s": float(df["interval_s"].median()) if len(df) else 300.0,
        "files": cache.provenance(rels),
    }
    return _finish_frame(df), meta


# --------------------------------------------------------------------------- #
# Site settings, fetching, lookup
# --------------------------------------------------------------------------- #
@dataclass
class SiteSettings:
    latitude: float = float("nan")
    longitude: float = float("nan")
    weather_source: str = "none"
    solar_source: str = "computed"
    weatherlink_url: str = ""
    weatherlink_units: dict = field(default_factory=lambda: {
        "temp": "auto", "wind": "mph", "rain": "in"})
    name: str = ""

    @property
    def has_location(self):
        return valid_coordinates(self.latitude, self.longitude)

    def to_dict(self):
        return {
            "name": self.name,
            "latitude": None if not np.isfinite(self.latitude) else self.latitude,
            "longitude": None if not np.isfinite(self.longitude) else self.longitude,
            "weather_source": self.weather_source,
            "solar_source": self.solar_source,
            "weatherlink_url": self.weatherlink_url,
            "weatherlink_units": dict(self.weatherlink_units),
        }

    @classmethod
    def from_dict(cls, d):
        d = d or {}

        def _f(v):
            try:
                return float(v)
            except (TypeError, ValueError):
                return float("nan")

        s = cls(latitude=_f(d.get("latitude")), longitude=_f(d.get("longitude")),
                name=str(d.get("name") or ""))
        if d.get("weather_source") in WEATHER_SOURCES:
            s.weather_source = d["weather_source"]
        if d.get("solar_source") in SOLAR_SOURCES:
            s.solar_source = d["solar_source"]
        s.weatherlink_url = str(d.get("weatherlink_url") or "")
        u = d.get("weatherlink_units") or {}
        s.weatherlink_units = {
            "temp": u.get("temp") if u.get("temp") in TEMP_UNITS else "auto",
            "wind": u.get("wind") if u.get("wind") in WIND_UNITS else "mph",
            "rain": u.get("rain") if u.get("rain") in RAIN_UNITS else "in",
        }
        return s


@dataclass
class Environment:
    """Fetched records for one trial: {role: frame} plus provenance."""
    settings: SiteSettings
    tz: str
    start_ns: int
    end_ns: int
    frames: dict = field(default_factory=dict)     # source -> frame
    meta: dict = field(default_factory=dict)       # source -> meta
    roles: dict = field(default_factory=dict)      # 'weather'/'solar' -> source
    warnings: list = field(default_factory=list)

    def summary(self):
        out = {"site": self.settings.to_dict(), "roles": dict(self.roles),
               "sources": {}, "warnings": list(self.warnings)}
        for src, meta in self.meta.items():
            fr = self.frames.get(src)
            m = dict(meta)
            m["records"] = int(len(fr)) if fr is not None else 0
            out["sources"][src] = m
        return out

    def timeline(self):
        return WeatherTimeline(self)


def fetch_environment(settings, tz, start_ns, end_ns, cache_root, log=None,
                      offline=False, now_ns=None):
    """Download (or reuse) everything ``settings`` asks for over the span."""
    log = log or (lambda m: None)
    env = Environment(settings=settings, tz=tz, start_ns=int(start_ns),
                      end_ns=int(end_ns))
    if not settings.has_location:
        env.warnings.append("no site latitude/longitude set")
        return env
    cache = DownloadCache(cache_root, offline=offline, log=log)
    wanted = []
    if settings.weather_source != "none":
        wanted.append(settings.weather_source)
        env.roles["weather"] = settings.weather_source
    if settings.solar_source != "computed":
        if settings.solar_source not in wanted:
            wanted.append(settings.solar_source)
        env.roles["solar"] = settings.solar_source
    lat, lon = settings.latitude, settings.longitude
    for src in wanted:
        try:
            if src == "surfrad":
                fr, meta = fetch_surfrad(lat, lon, start_ns, end_ns, cache, log)
                if meta["distance_km"] > 100:
                    env.warnings.append(
                        f"nearest SURFRAD station ({meta['station']}) is "
                        f"{meta['distance_km']:.0f} km from the site")
            elif src == "open_meteo":
                fr, meta = fetch_open_meteo(lat, lon, start_ns, end_ns, cache,
                                            log, now_ns=now_ns)
            elif src == "weatherlink":
                fr, meta = fetch_weatherlink(settings.weatherlink_url, tz,
                                             settings.weatherlink_units,
                                             start_ns, end_ns, cache, log)
                chosen = settings.weatherlink_units.get("temp", "auto")
                seen = meta.get("temp_unit_detected")
                if chosen in ("F", "C") and seen and seen != chosen:
                    env.warnings.insert(0, (
                        f"UNIT MISMATCH: the WeatherLink file's temperatures "
                        f"read as °{seen} (its dew point only agrees with its "
                        f"temperature and humidity in °{seen}), but File units "
                        f"say °{chosen}. Set Temp to Auto or °{seen}."))
            else:
                continue
        except ValueError as e:
            env.warnings.append(str(e))
            continue
        env.frames[src] = fr
        env.meta[src] = meta
        if fr.empty:
            env.warnings.append(f"{SOURCE_LABELS[src]}: no records for this span")
        else:
            first, last = int(fr["time_ns"].iloc[0]), int(fr["time_ns"].iloc[-1])
            iv = int(meta["interval_s"] * NS_PER_S)
            if first - iv > start_ns or last < end_ns:
                env.warnings.append(
                    f"{SOURCE_LABELS[src]}: covers "
                    f"{_fmt_local(first - iv, tz)} to {_fmt_local(last, tz)}, "
                    f"not the whole trial")
    missing = [u for u, why in cache.failed if why.startswith("not published")]
    if missing:
        env.warnings.append(
            f"{len(missing)} file(s) not published (yet), e.g. {missing[-1]}")
    for url, why in [f for f in cache.failed if not f[1].startswith("not published")][:5]:
        env.warnings.append(f"download failed ({why}): {url}")
    log(f"Weather: {cache.downloaded} file(s) downloaded, "
        f"{cache.reused} reused from cache")
    return env


def _fmt_local(ns, tz):
    return pd.Timestamp(int(ns), tz="UTC").tz_convert(tz).strftime("%Y-%m-%d %H:%M")


class WeatherTimeline:
    """Fast "what was it like at t" lookups for the preview and the video."""

    #: A record older than this (beyond its own interval) is not shown.
    MAX_AGE_S = 3 * 3600

    def __init__(self, env):
        self.env = env
        self.lat = env.settings.latitude
        self.lon = env.settings.longitude
        self.has_location = env.settings.has_location
        self._arrays = {}
        for src, fr in env.frames.items():
            if fr is None or fr.empty:
                continue
            self._arrays[src] = (
                fr["time_ns"].to_numpy(dtype="int64"),
                (fr["interval_s"].to_numpy(dtype=float) * NS_PER_S).astype("int64"),
                {f: fr[f].to_numpy(dtype=float) for f in FIELDS},
                fr["precip_type"].to_numpy(dtype=object),
                fr["precip_basis"].to_numpy(dtype=object),
            )

    def _record(self, src, t_ns):
        arr = self._arrays.get(src)
        if arr is None:
            return None
        ends, ivs, vals, ptype, pbasis = arr
        i = int(np.searchsorted(ends, t_ns, "left"))
        # A record covers (end - interval, end].
        if i < len(ends) and ends[i] - ivs[i] < t_ns:
            k, age = i, 0.0
        elif i > 0:
            k = i - 1
            age = (t_ns - ends[k]) / NS_PER_S
            if age > self.MAX_AGE_S:
                return None
        else:
            return None
        stale = age > max(2.0 * ivs[k] / NS_PER_S, 600.0)
        return {"values": {f: vals[f][k] for f in FIELDS}, "age_s": age,
                "stale": stale, "interval_s": ivs[k] / NS_PER_S,
                "end_ns": int(ends[k]), "precip_type": ptype[k] or "",
                "precip_basis": pbasis[k] or ""}

    def at(self, t_ns):
        t_ns = int(t_ns)
        out = {"weather": None, "solar": None, "sun_elevation": np.nan,
               "sun_azimuth": np.nan, "phase": "", "light": np.nan,
               "ghi": np.nan, "ghi_kind": "", "moon_elevation": np.nan,
               "moon_illumination": np.nan, "moon_phase": "",
               "moon_waxing": True, "moonlight": 0.0,
               "southern": bool(self.has_location and self.lat < 0)}
        wsrc = self.env.roles.get("weather")
        if wsrc:
            out["weather"] = self._record(wsrc, t_ns)
            out["weather_source"] = wsrc
        if not self.has_location:
            return out
        el, az = solar_position(np.array([t_ns]), self.lat, self.lon)
        out["sun_elevation"], out["sun_azimuth"] = float(el[0]), float(az[0])
        out["phase"] = str(sun_phase(el)[0])
        mel, maz = moon_position(np.array([t_ns]), self.lat, self.lon)
        k, _d, wax, name = moon_phase(np.array([t_ns]))
        out.update(moon_elevation=float(mel[0]), moon_azimuth=float(maz[0]),
                   moon_illumination=float(k[0]), moon_waxing=bool(wax[0]),
                   moon_phase=str(name[0]),
                   moon_up=bool(mel[0] > moon_horizon(np.array([t_ns]))[0]),
                   moonlight=float(moonlight_level(mel, k)[0]))
        ssrc = self.env.roles.get("solar")
        ghi, kind = np.nan, "clear-sky"
        if ssrc:
            rec = self._record(ssrc, t_ns)
            out["solar"] = rec
            if rec is not None and not rec["stale"] and np.isfinite(rec["values"]["ghi_wm2"]):
                ghi = rec["values"]["ghi_wm2"]
                kind = self.env.meta.get(ssrc, {}).get("kind", "")
        out["ghi"] = ghi
        out["ghi_kind"] = kind
        out["light"] = float(light_level(el, np.array([ghi]))[0])
        return out

    def precip_strip(self, t0_ns, t1_ns, n=288):
        """(rate mm/h, kind) across [t0, t1] from the weather source.

        Each sample takes the record whose interval contains it, so a 5-min
        shower is as wide on the strip as it was in time.
        """
        t = np.linspace(t0_ns, t1_ns, n).astype("int64")
        rate = np.zeros(n)
        kind = np.full(n, "", dtype=object)
        wsrc = self.env.roles.get("weather")
        arr = self._arrays.get(wsrc) if wsrc else None
        if arr is None:
            return rate, kind
        ends, ivs, vals, ptype, _pb = arr
        i = np.searchsorted(ends, t, "left")
        j = np.minimum(i, len(ends) - 1)
        ok = (i < len(ends)) & (ends[j] - ivs[j] < t)
        r = vals["precip_rate_mmh"][j]
        good = ok & np.isfinite(r) & (r > 0)
        rate[good] = r[good]
        kind[good] = ptype[j][good]
        return rate, kind

    def light_strip(self, t0_ns, t1_ns, n=288):
        """(times_ns, light levels, moonlight) across [t0, t1] for the strip."""
        t = np.linspace(t0_ns, t1_ns, n).astype("int64")
        if not self.has_location:
            return t, np.full(n, np.nan), np.zeros(n)
        el, _ = solar_position(t, self.lat, self.lon)
        mel, _ = moon_position(t, self.lat, self.lon)
        moon = moonlight_level(mel, moon_phase(t)[0])
        ghi = np.full(n, np.nan)
        ssrc = self.env.roles.get("solar")
        arr = self._arrays.get(ssrc) if ssrc else None
        if arr is not None:
            ends, ivs, vals, _pt, _pb = arr
            i = np.searchsorted(ends, t, "left")
            ok = i < len(ends)
            j = np.minimum(i, len(ends) - 1)
            ok &= ends[j] - ivs[j] < t
            # Allow the nearest record within two intervals so a coarse
            # source still fills the strip between its stamps.
            near = ~ok & (i > 0)
            jp = np.maximum(i - 1, 0)
            near &= (t - ends[jp]) <= 2 * ivs[jp]
            g = vals["ghi_wm2"]
            ghi = np.where(ok, g[j], np.where(near, g[jp], np.nan))
        return t, light_level(el, ghi), moon


# --------------------------------------------------------------------------- #
# Display formatting
# --------------------------------------------------------------------------- #
def _fmt_age(s):
    if s < 90:
        return f"{s:.0f} s"
    if s < 5400:
        return f"{s / 60:.0f} min"
    return f"{s / 3600:.1f} h"


def format_weather(state, units="metric"):
    """One line for the preview / video, or '' when there is nothing to say."""
    rec = (state or {}).get("weather")
    if not rec:
        return ""
    v = rec["values"]
    us = units == "us"
    parts = []
    if np.isfinite(v["temp_c"]):
        parts.append(f"{v['temp_c'] * 9 / 5 + 32:.1f} °F" if us
                     else f"{v['temp_c']:.1f} °C")
    if np.isfinite(v["rh_pct"]):
        parts.append(f"RH {v['rh_pct']:.0f}%")
    if np.isfinite(v["wind_speed_ms"]):
        spd = (f"{v['wind_speed_ms'] / 0.44704:.1f} mph" if us
               else f"{v['wind_speed_ms']:.1f} m/s")
        d = compass(v["wind_dir_deg"])
        w = f"wind {spd}{' ' + d if d else ''}"
        if np.isfinite(v["wind_gust_ms"]) and v["wind_gust_ms"] > v["wind_speed_ms"]:
            g = (v["wind_gust_ms"] / 0.44704) if us else v["wind_gust_ms"]
            w += f" (gust {g:.0f})"
        parts.append(w)
    if np.isfinite(v["pressure_hpa"]):
        parts.append(f"{v['pressure_hpa'] * 0.02953:.2f} inHg" if us
                     else f"{v['pressure_hpa']:.0f} hPa")
    if not parts:
        return ""
    pr = format_precip(state, units)
    if pr:
        parts.insert(0, pr)
    line = " · ".join(parts)
    if rec["stale"]:
        line += f"  (last record {_fmt_age(rec['age_s'])} old)"
    return line


def format_precip(state, units="metric"):
    """'rain 2.4 mm/h' / 'snow 0.6 cm/h' / ... for the covering record, or ''."""
    rec = (state or {}).get("weather")
    if not rec:
        return ""
    v = rec["values"]
    rate = v.get("precip_rate_mmh", np.nan)
    if not np.isfinite(rate) or rate <= 0:
        return ""
    kind = rec.get("precip_type") or "rain"
    est = " (est.)" if rec.get("precip_basis") in ("wet-bulb", "air temperature") else ""
    us = units == "us"
    snow_cm = v.get("snowfall_cm", np.nan)
    if kind == "snow" and np.isfinite(snow_cm) and snow_cm > 0:
        cmh = snow_cm * 3600.0 / rec["interval_s"]
        amount = f"{cmh / 2.54:.2f} in/h" if us else f"{cmh:.1f} cm/h"
    else:
        amount = f"{rate / 25.4:.2f} in/h" if us else f"{rate:.1f} mm/h"
        if kind != "rain":
            amount += " water"
    name = {"rain": "RAIN", "snow": "SNOW", "mixed": "RAIN/SNOW"}.get(kind, "PRECIP")
    return f"{name}{est} {amount}"


def precip_rgba(rate, kind):
    """Colour for a precipitation sample, alpha scaled with intensity."""
    if not kind or not np.isfinite(rate) or rate <= 0:
        return None
    r, g, b = PRECIP_COLORS.get(kind, PRECIP_COLORS["rain"])
    a = float(np.clip(np.sqrt(rate / PRECIP_FULL_MMH), 0.35, 1.0))
    return (r, g, b, a)


PHASE_LABELS = {"day": "day", "civil": "civil twilight",
                "nautical": "nautical twilight",
                "astronomical": "astronomical twilight", "night": "night"}


def format_light(state):
    s = state or {}
    if not np.isfinite(s.get("sun_elevation", np.nan)):
        return ""
    txt = f"{PHASE_LABELS.get(s['phase'], s['phase'])} · sun {s['sun_elevation']:+.1f}°"
    if np.isfinite(s.get("ghi", np.nan)):
        txt = f"{max(s['ghi'], 0):.0f} W/m² ({s['ghi_kind']}) · " + txt
    if np.isfinite(s.get("moon_illumination", np.nan)):
        where = (f"up {s['moon_elevation']:.0f}°" if s.get("moon_up")
                 else "below horizon")
        txt += (f" · moon {s['moon_illumination'] * 100:.0f}% "
                f"{s['moon_phase']}, {where}")
    return txt


# --------------------------------------------------------------------------- #
# Export
# --------------------------------------------------------------------------- #
EXPORT_COLUMNS = ("source", "station", "Timestamp", "timestamp", "interval_s",
                  *FIELDS, "precip_type", "precip_basis",
                  "sun_elevation_deg", "sun_azimuth_deg", "sun_phase",
                  "moon_elevation_deg", "moon_illumination", "moon_phase",
                  "time_flag")


def export_frame(env):
    """Every fetched record over the trial, one stable schema, local times."""
    parts = []
    for src, fr in env.frames.items():
        if fr is None or fr.empty:
            continue
        sub = fr[(fr["time_ns"] > env.start_ns - (fr["interval_s"] * NS_PER_S).astype("int64"))
                 & (fr["time_ns"] <= env.end_ns + (fr["interval_s"] * NS_PER_S).astype("int64"))].copy()
        if sub.empty:
            continue
        meta = env.meta.get(src, {})
        sub.insert(0, "source", src)
        sub.insert(1, "station", meta.get("station", ""))
        el, az = solar_position(sub["time_ns"].to_numpy(), env.settings.latitude,
                                env.settings.longitude)
        sub["sun_elevation_deg"] = np.round(el, 4)
        sub["sun_azimuth_deg"] = np.round(az, 4)
        sub["sun_phase"] = sun_phase(el)
        mel, _maz = moon_position(sub["time_ns"].to_numpy(), env.settings.latitude,
                                  env.settings.longitude)
        k, _d, _wax, names = moon_phase(sub["time_ns"].to_numpy())
        sub["moon_elevation_deg"] = np.round(mel, 3)
        sub["moon_illumination"] = np.round(k, 4)
        sub["moon_phase"] = names
        parts.append(sub)
    if not parts:
        return pd.DataFrame(columns=list(EXPORT_COLUMNS))
    df = pd.concat(parts, ignore_index=True).sort_values(
        ["time_ns", "source"], kind="stable")
    df["Timestamp"] = pd.to_datetime(df["time_ns"], utc=True).dt.tz_convert(env.tz)
    df["timestamp"] = (df["time_ns"] // 1_000_000).astype("int64")
    return df[list(EXPORT_COLUMNS)].reset_index(drop=True)


def read_site_profile(path):
    with open(path) as f:
        return SiteSettings.from_dict(json.load(f))


def write_site_profile(path, settings):
    tmp = path + ".part"
    with open(tmp, "w") as f:
        json.dump(settings.to_dict(), f, indent=2)
    os.replace(tmp, path)


#: File name auto-detected beside a database.
SITE_PROFILE_NAME = "fnt_site.json"
