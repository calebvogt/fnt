"""A FED3's state in words: the same in Setup's rows and on a run's cards."""
from . import theme

READINGS = (("batt", "V"), ("temp", "°C"), ("hum", "% RH"))


def health(device):
    """What's ready and what isn't: ``(parts, colour)``, the worst colour winning."""
    if not device or not device["connected"]:
        return ["not connected"], theme.BAD
    status = device["status"]
    if not status:
        return ["reading its status…"], theme.MUTED
    parts, colour = [], theme.GOOD
    if status.get("clock") == "ok":
        drift = device.get("drift")
        parts.append("clock ok" if drift is None or abs(drift) < 5 else f"clock {drift:+.0f}s")
    else:
        parts.append("clock not set")
        colour = theme.WARN
    if status.get("sd") == "ok":
        parts.append("SD ok")
    else:
        parts.append("SD error")
        colour = theme.BAD
    if status.get("jam") == "1":
        parts.append("jammed")
        colour = theme.BAD
    return parts, colour


def readings(status):
    """Battery, and temperature and humidity where there's a sensor: a FED3
    without one (most) reports nan."""
    return [f"{status[key]} {unit}" for key, unit in READINGS
            if status.get(key, "nan") not in ("", "nan")]
