"""Overnight recording support.

A 9-hour night is not "a long probe" — the failure modes are different, and
most of them are silent:

* **The Mac goes to sleep and the recording stops.** Idle sleep after ~10
  minutes would end the night before it starts. ``SleepGuard`` holds a
  ``caffeinate`` wake-lock that keeps the *system* awake while still letting
  the *display* sleep, so the screen goes dark but Bluetooth keeps streaming.
* **The app wakes you up.** The dropout watchdog normally plays an alert tone
  and speaks "Signal lost". At 3 a.m. that is worse than the dropout it is
  reporting, so sleep mode silences every audible channel.
* **Rendering all night costs fan noise and battery.** Redrawing plots 30
  times a second for nine hours spins up the fans next to your head. Sleep
  mode pauses live rendering; the data path is untouched.
* **Disk fills up.** ~1.5 GB per night at full precision. Checked up front
  rather than discovered at 4 a.m.

What this cannot do is change the headband's own power behaviour. Muse's app
reaches 9+ hours using a firmware low-power mode that we cannot reach through
OpenMuse's streaming interface, so **the Athena's battery, not this software,
is the limit on how long a night can be recorded.** Sleep mode logs the battery
trajectory so you can find out empirically what your device manages.
"""

import os
import shutil
import subprocess
import sys

# Per-second bytes, measured from the CSV writer (see musestudio storage math).
_BYTES_PER_SEC = {
    "eeg": 8 * 7 + 18,          # 8 ch at 2-decimal precision + timestamp
    "optics": 16 * 7 + 18,
    "accgyro": 6 * 7 + 18,
}
_RATES = {"eeg": 256, "optics": 64, "accgyro": 52}


def estimate_night_bytes(hours=9.0, include_optics=True):
    """Rough disk cost of a night at sleep-mode precision."""
    total = 0.0
    for name, rate in _RATES.items():
        if name == "optics" and not include_optics:
            continue
        total += _BYTES_PER_SEC[name] * rate * 3600.0 * hours
    return total


def free_bytes(path):
    try:
        return shutil.disk_usage(path).free
    except Exception:
        return None


def storage_check(path, hours=9.0, include_optics=True):
    """``(ok, message)`` — is there room for the night, with margin?"""
    need = estimate_night_bytes(hours, include_optics)
    free = free_bytes(path)
    need_gb = need / 1e9
    if free is None:
        return True, f"~{need_gb:.1f} GB needed (could not read free space)"
    free_gb = free / 1e9
    # Keep 2 GB of headroom so the machine doesn't hit zero overnight.
    ok = free > need + 2e9
    msg = f"~{need_gb:.1f} GB needed · {free_gb:.1f} GB free"
    if not ok:
        msg += "  — not enough room for a full night"
    return ok, msg


class SleepGuard:
    """Holds a system wake-lock for the duration of an overnight recording."""

    def __init__(self):
        self._proc = None

    def available(self):
        return sys.platform == "darwin" and shutil.which("caffeinate") is not None

    def start(self):
        """Prevent idle *system* sleep; the display may still switch off.

        ``-i`` (idle) plus ``-m`` (disk) keeps the machine and its disks awake
        without ``-d``, which would force the screen to stay lit all night.
        """
        if self._proc is not None and self._proc.poll() is None:
            return True
        if not self.available():
            return False
        try:
            self._proc = subprocess.Popen(
                ["caffeinate", "-i", "-m", "-w", str(os.getpid())],
                stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            return True
        except Exception:
            self._proc = None
            return False

    def active(self):
        return self._proc is not None and self._proc.poll() is None

    def stop(self):
        if self._proc is None:
            return
        try:
            self._proc.terminate()
            self._proc.wait(timeout=3)
        except Exception:
            try:
                self._proc.kill()
            except Exception:
                pass
        finally:
            self._proc = None


def fmt_duration(seconds):
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h {m:02d}m"
    return f"{m}m {s:02d}s"
