"""Sun and moon for a real place on a real date.

ABMA's original day/night was a switch: before ``day_start_hour`` it is night,
after it is day, every day of the year, everywhere. That is a light-cycle room,
and it is the right model for one. It is the wrong model for a field enclosure,
where the question worth asking — does this cohort space itself differently in
December than in June — needs day length, twilight and moonlight to be things
the world produces rather than constants somebody typed in.

Given a latitude, a longitude and a timestamp this gives:

  * **solar elevation and azimuth**, so day, night and the long twilights that
    actually bracket a crepuscular animal's activity all fall out of geometry;
  * **day length**, which is what "season" really means to an animal;
  * **moon phase, position and illumination**, because lunar-phobic foraging is
    one of the better-described behaviours in small mammals.

Accuracy and its limits
-----------------------
The solar position follows the standard NOAA low-precision algorithm: better
than a tenth of a degree over the years an experiment spans, and far beyond
what a behaviour model can use. The lunar position is a truncated ELP series —
good to roughly a degree, which is ample for "how bright is it tonight" and not
intended for anything that cares where the moon *is* to arcminutes.

Refraction, elevation above sea level and atmospheric extinction are ignored.
They shift sunrise by about a minute; nothing here resolves that.
"""
from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import datetime, timedelta

#: Representative dates for each season in the northern hemisphere — the
#: solstices and equinoxes. Choosing a season *is* choosing a date; this is the
#: lookup that makes "run it in winter" a thing you can say.
SEASON_DATES: dict[str, tuple[int, int]] = {
    "spring": (3, 20),
    "summer": (6, 21),
    "fall": (9, 22),
    "autumn": (9, 22),
    "winter": (12, 21),
}

#: Northern-hemisphere sward growth by season, 0-1. Free parameters: they set
#: how fast a clipped trail closes over, which is the whole reason season and
#: grass interact at all.
SEASON_GROWTH: dict[str, float] = {
    "spring": 1.0, "summer": 0.75, "fall": 0.35, "autumn": 0.35, "winter": 0.05,
}


def season_start(season: str, year: int = 2026, hour: int = 18) -> str:
    """ISO timestamp for the start of a run in ``season``.

    Releases default to early evening, which is when an enclosure release
    usually happens — it gives nocturnal and crepuscular animals a night to
    settle before their first full day.
    """
    key = (season or "").strip().lower()
    if key not in SEASON_DATES:
        raise ValueError(
            f"unknown season {season!r}; choose one of "
            f"{sorted(set(SEASON_DATES))}")
    month, day = SEASON_DATES[key]
    return datetime(year, month, day, hour, 0, 0).isoformat()


def season_of(when: datetime) -> str:
    """Which season a date falls in (northern hemisphere, meteorological)."""
    m = when.month
    if m in (3, 4, 5):
        return "spring"
    if m in (6, 7, 8):
        return "summer"
    if m in (9, 10, 11):
        return "fall"
    return "winter"


def growth_factor(when: datetime) -> float:
    """Sward growth multiplier for the time of year, smoothly varying.

    A cosine through the year rather than four steps, so a run that crosses a
    season boundary does not see the grass change gear overnight. Peaks in late
    spring and bottoms in midwinter.
    """
    doy = when.timetuple().tm_yday
    # peak growth near day 140 (late May at temperate latitudes)
    phase = 2.0 * math.pi * (doy - 140) / 365.25
    return float(max(0.02, 0.5 + 0.5 * math.cos(phase)))


# --------------------------------------------------------------------------- #
# Solar position (NOAA low-precision)
# --------------------------------------------------------------------------- #
def _julian_day(when: datetime, tz_hours: float) -> float:
    """Julian day for a local timestamp with a fixed UTC offset."""
    utc = when - timedelta(hours=tz_hours)
    y, m = utc.year, utc.month
    d = (utc.day + (utc.hour + (utc.minute + utc.second / 60.0) / 60.0) / 24.0)
    if m <= 2:
        y, m = y - 1, m + 12
    a = y // 100
    b = 2 - a + a // 4
    return (math.floor(365.25 * (y + 4716)) + math.floor(30.6001 * (m + 1))
            + d + b - 1524.5)


@dataclass
class SkyState:
    """Everything the engine and the views need to know about the sky."""
    sun_elevation: float      # degrees above the horizon (negative = below)
    sun_azimuth: float        # degrees clockwise from north
    moon_elevation: float
    moon_azimuth: float
    moon_phase: float         # 0 = new, 0.5 = full, 1 = new again
    moon_illumination: float  # 0-1 lit fraction of the disc
    is_day: bool
    daylight: float           # 0-1 smooth light level from the sun alone
    night_light: float        # 0-1 moonlight when the sun is down

    @property
    def moon_up(self) -> bool:
        return self.moon_elevation > 0.0


def sun_position(when: datetime, lat: float, lon: float,
                 tz_hours: float) -> tuple[float, float]:
    """(elevation, azimuth) of the sun in degrees, for a local timestamp."""
    jd = _julian_day(when, tz_hours)
    n = jd - 2451545.0
    # mean longitude and anomaly
    L = math.radians((280.460 + 0.9856474 * n) % 360.0)
    g = math.radians((357.528 + 0.9856003 * n) % 360.0)
    # ecliptic longitude, obliquity
    lam = L + math.radians(1.915) * math.sin(g) + math.radians(0.020) * math.sin(2 * g)
    eps = math.radians(23.439 - 0.0000004 * n)
    # equatorial
    ra = math.atan2(math.cos(eps) * math.sin(lam), math.cos(lam))
    dec = math.asin(math.sin(eps) * math.sin(lam))
    # local sidereal time -> hour angle
    gmst = (18.697374558 + 24.06570982441908 * n) % 24.0
    lst = math.radians((gmst * 15.0 + lon) % 360.0)
    ha = lst - ra
    return _to_horizon(ha, dec, lat)


def _to_horizon(ha: float, dec: float, lat: float) -> tuple[float, float]:
    """Hour angle + declination -> (elevation, azimuth) in degrees."""
    phi = math.radians(lat)
    sin_alt = (math.sin(dec) * math.sin(phi)
               + math.cos(dec) * math.cos(phi) * math.cos(ha))
    alt = math.asin(max(-1.0, min(1.0, sin_alt)))
    az = math.atan2(-math.sin(ha) * math.cos(dec),
                    math.cos(phi) * math.sin(dec)
                    - math.sin(phi) * math.cos(dec) * math.cos(ha))
    return math.degrees(alt), math.degrees(az) % 360.0


def moon_position(when: datetime, lat: float, lon: float,
                  tz_hours: float) -> tuple[float, float, float]:
    """(elevation, azimuth, phase) of the moon; phase 0=new, 0.5=full.

    A truncated lunar series — the leading evection and variation terms only.
    Good to about a degree, which is far more than "is it up and how bright".
    """
    jd = _julian_day(when, tz_hours)
    n = jd - 2451545.0
    # mean elements (degrees)
    Lm = (218.316 + 13.176396 * n) % 360.0        # mean longitude
    Mm = (134.963 + 13.064993 * n) % 360.0        # mean anomaly
    Fm = (93.272 + 13.229350 * n) % 360.0         # argument of latitude
    Ms = (357.529 + 0.98560028 * n) % 360.0       # sun's mean anomaly
    lam = math.radians(Lm + 6.289 * math.sin(math.radians(Mm)))
    beta = math.radians(5.128 * math.sin(math.radians(Fm)))
    eps = math.radians(23.439 - 0.0000004 * n)
    # ecliptic -> equatorial
    ra = math.atan2(math.sin(lam) * math.cos(eps)
                    - math.tan(beta) * math.sin(eps), math.cos(lam))
    dec = math.asin(math.sin(beta) * math.cos(eps)
                    + math.cos(beta) * math.sin(eps) * math.sin(lam))
    gmst = (18.697374558 + 24.06570982441908 * n) % 24.0
    lst = math.radians((gmst * 15.0 + lon) % 360.0)
    alt, az = _to_horizon(lst - ra, dec, lat)
    # phase from the sun-moon elongation
    sun_lon = math.radians((280.460 + 0.9856474 * n
                            + 1.915 * math.sin(math.radians(Ms))) % 360.0)
    elong = (math.degrees(lam - sun_lon)) % 360.0
    return alt, az, elong / 360.0


def sky_state(when: datetime, params) -> SkyState:
    """The full sky for one instant, ready for the engine and the views."""
    lat, lon, tz = params.latitude, params.longitude, params.timezone_hours
    sun_alt, sun_az = sun_position(when, lat, lon, tz)
    moon_alt, moon_az, phase = moon_position(when, lat, lon, tz)
    # lit fraction of the disc: 0 at new, 1 at full
    illum = (1.0 - math.cos(2.0 * math.pi * phase)) / 2.0
    night_at = float(params.night_elevation_deg)
    is_day = sun_alt > night_at
    # a smooth ramp through twilight rather than a switch, so a crepuscular
    # animal has a dawn and a dusk to be active in
    daylight = max(0.0, min(1.0, (sun_alt - night_at) / (12.0 - night_at)))
    night_light = (illum * max(0.0, min(1.0, moon_alt / 30.0))
                   if sun_alt <= night_at and moon_alt > 0 else 0.0)
    return SkyState(sun_elevation=sun_alt, sun_azimuth=sun_az,
                    moon_elevation=moon_alt, moon_azimuth=moon_az,
                    moon_phase=phase, moon_illumination=illum,
                    is_day=is_day, daylight=daylight, night_light=night_light)


def day_length_hours(when: datetime, params) -> float:
    """Hours between sunrise and sunset — what "season" means to an animal."""
    lat = math.radians(params.latitude)
    doy = when.timetuple().tm_yday
    dec = math.radians(23.44) * math.sin(2 * math.pi * (doy - 81) / 365.25)
    cos_ha = -math.tan(lat) * math.tan(dec)
    if cos_ha <= -1.0:
        return 24.0            # midnight sun
    if cos_ha >= 1.0:
        return 0.0             # polar night
    return 2.0 * math.degrees(math.acos(cos_ha)) / 15.0
