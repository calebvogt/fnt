"""Signal processing for MuseStudio — display filtering, spectral estimation
and fNIRS hemodynamics.

Pure NumPy/SciPy (no Qt) so every function here is directly testable.

Two important separations of concern:

* **Display filtering never touches recorded data.** Recording writes the raw
  stream straight from the reader thread; filters here only shape what you see.
* **Filtering for the scrolling view is causal and stateful** (``StreamingFilter``)
  so each sample is filtered exactly once as it arrives. Zero-phase filtfilt on
  a sliding window would make the newest samples wobble as the window moves.
"""

import numpy as np
from scipy.signal import butter, hilbert, iirnotch, sosfilt, tf2sos, welch

# Canonical EEG bands (Hz). Order matters for display.
BANDS = {
    "delta": (1.0, 4.0),
    "theta": (4.0, 8.0),
    "alpha": (8.0, 12.0),
    "beta": (13.0, 30.0),
    "gamma": (30.0, 50.0),
}
BAND_ORDER = ["delta", "theta", "alpha", "beta", "gamma"]

# --------------------------------------------------------------------------
# Channel semantics (verified against a real Muse S Athena)
# --------------------------------------------------------------------------
# The four scalp electrodes. Everything else on the EEG stream (AUX_1..AUX_4)
# is an auxiliary input for an external electrode; with nothing plugged into
# the accessory port those pins float and read as noise, so they must be kept
# out of band power, contact checks and synchrony.
EEG_ELECTRODES = ("TP9", "AF7", "AF8", "TP10")

# Optics channels are named OPTICS_<side><depth>_<wavelength>, e.g.
# OPTICS_LI_NIR = left, inner detector, near-infrared.
OPTICS_SIDES = {"L": "left", "R": "right"}
OPTICS_DEPTHS = {"I": "inner", "O": "outer"}
OPTICS_WAVELENGTHS = {
    "NIR": "near-infrared (~850 nm)",
    "IR": "infrared (~730 nm)",
    "RED": "red (~660 nm)",
    "AMB": "ambient (LEDs off)",
}


def is_scalp_eeg(name):
    """True for the four real electrodes, False for AUX and anything else."""
    upper = str(name).upper()
    return any(e in upper for e in EEG_ELECTRODES)


def scalp_channels(names):
    """Subset of ``names`` that are real scalp electrodes, in the given order."""
    return [n for n in names if is_scalp_eeg(n)]


def parse_optics_channel(name):
    """``OPTICS_LI_NIR`` -> ``{'side': 'left', 'depth': 'inner', 'wavelength': 'NIR'}``.

    Returns ``None`` when the name doesn't follow the Athena convention.
    """
    parts = str(name).upper().split("_")
    if len(parts) < 3 or parts[0] != "OPTICS":
        return None
    code, wavelength = parts[1], parts[2]
    if len(code) != 2 or wavelength not in OPTICS_WAVELENGTHS:
        return None
    side = OPTICS_SIDES.get(code[0])
    depth = OPTICS_DEPTHS.get(code[1])
    if side is None or depth is None:
        return None
    return {"side": side, "depth": depth, "wavelength": wavelength}


def describe_optics_channel(name):
    """Human-readable explanation for tooltips."""
    info = parse_optics_channel(name)
    if info is None:
        return str(name)
    depth_note = ("short separation — mostly scalp/skin blood flow"
                  if info["depth"] == "inner" else
                  "long separation — reaches cortex")
    return (f"{info['side']} {info['depth']} detector, "
            f"{OPTICS_WAVELENGTHS[info['wavelength']]} · {depth_note}")


def curated_channels(stream_name, channel_names):
    """The channels worth watching by default, per stream.

    Rationale:
    * **EEG** — the four scalp electrodes. AUX_1..AUX_4 are unconnected
      auxiliary inputs and read as noise.
    * **ACC/GYRO** — accelerometer only. It shows head movement, which is the
      main source of EEG artifact; gyro rarely adds anything when seated.
    * **Optics** — outer (long-separation) detectors at the two wavelengths
      used for hemodynamics. Inner detectors mostly see scalp blood flow and
      ambient channels measure background light, not tissue.
    """
    upper = str(stream_name).upper()
    names = [str(n) for n in channel_names]
    if "EEG" in upper:
        return scalp_channels(names) or names
    if "ACC" in upper or "GYRO" in upper:
        acc = [n for n in names if "ACC" in n.upper()]
        return acc or names
    if "OPTIC" in upper or "NIRS" in upper:
        keep = []
        for n in names:
            info = parse_optics_channel(n)
            if info and info["depth"] == "outer" and info["wavelength"] in ("NIR", "RED"):
                keep.append(n)
        return keep or names
    return names


def contact_quality(x, fs, p2p_limit=250.0, min_std=0.5):
    """Is this electrode making usable contact?

    The check runs on a high-passed copy: raw Muse EEG carries a large DC
    offset and slow drift, so a peak-to-peak test on the raw signal fails even
    on a perfectly good electrode.
    """
    x = np.asarray(x, dtype=float)
    if len(x) < 16:
        return False
    sos = design_display_sos(fs, low=1.0, high=None, notch=None)
    y = sosfilt(sos, x - np.mean(x)) if sos is not None else x - np.mean(x)
    y = y[len(y) // 4:]           # drop the filter start-up transient
    if len(y) == 0:
        return False
    return bool(np.ptp(y) < p2p_limit and np.std(y) > min_std)


# --------------------------------------------------------------------------
# Display filtering
# --------------------------------------------------------------------------
def design_display_sos(fs, low=1.0, high=40.0, notch=60.0):
    """Second-order sections for a display filter: high-pass (drift removal),
    low-pass (anti-noise) and an optional mains notch."""
    nyq = fs / 2.0
    sections = []
    if low and 0 < low < nyq:
        sections.append(butter(2, low / nyq, btype="high", output="sos"))
    if high and 0 < high < nyq * 0.98:
        sections.append(butter(4, high / nyq, btype="low", output="sos"))
    if notch and 0 < notch < nyq * 0.95:
        b, a = iirnotch(notch, Q=30.0, fs=fs)
        sections.append(tf2sos(b, a))
    if not sections:
        return None
    return np.vstack(sections)


class StreamingFilter:
    """Causal SOS filter that carries state across chunks (one filter bank,
    independent state per channel)."""

    def __init__(self, fs, n_channels, low=1.0, high=40.0, notch=60.0):
        self.fs = float(fs)
        self.n_channels = int(n_channels)
        self.sos = design_display_sos(fs, low, high, notch)
        self._zi = None
        self.reset()

    def reset(self):
        if self.sos is None:
            self._zi = None
        else:
            # sosfilt wants x.shape with axis-0 replaced by 2, prefixed by
            # the section count -> (n_sections, 2, n_channels).
            self._zi = np.zeros((self.sos.shape[0], 2, self.n_channels))

    def process(self, chunk):
        """Filter a ``(n_samples, n_channels)`` chunk, returning the same shape."""
        if self.sos is None or chunk is None or len(chunk) == 0:
            return chunk
        x = np.asarray(chunk, dtype=float)
        if x.ndim == 1:
            x = x[:, None]
        if x.shape[1] != self.n_channels:   # channel count changed -> re-init
            self.n_channels = x.shape[1]
            self.reset()
        y, self._zi = sosfilt(self.sos, x, axis=0, zi=self._zi)
        return y


# --------------------------------------------------------------------------
# Spectral estimation
# --------------------------------------------------------------------------
def psd(x, fs, nperseg=None):
    """Welch power spectral density. Returns ``(freqs, power)``."""
    x = np.asarray(x, dtype=float)
    if len(x) < 16:
        return np.array([]), np.array([])
    if nperseg is None:
        nperseg = int(min(len(x), fs * 2))       # ~0.5 Hz resolution at fs*2
    nperseg = max(16, min(nperseg, len(x)))
    return welch(x - np.mean(x), fs=fs, nperseg=nperseg, noverlap=nperseg // 2)


def band_powers(x, fs, bands=None):
    """Absolute band power (integrated PSD) per band for one channel."""
    bands = bands or BANDS
    f, p = psd(x, fs)
    out = {name: 0.0 for name in bands}
    if len(f) == 0:
        return out
    for name, (lo, hi) in bands.items():
        mask = (f >= lo) & (f < hi)
        out[name] = float(np.trapezoid(p[mask], f[mask])) if mask.any() else 0.0
    return out


def relative_band_powers(absolute):
    """Normalize band powers to fractions of total power (sums to 1)."""
    total = sum(absolute.values())
    if total <= 0:
        return {k: 0.0 for k in absolute}
    return {k: v / total for k, v in absolute.items()}


def band_connectivity(x, y, fs, band, nperseg=None):
    """Coupling between two channels in ``band``: ``(plv, imag_coh)``.

    Both are reported because they fail in different ways:

    * **PLV** is sensitive but, on a headband where every electrode shares one
      reference (Fpz), common-mode signal — reference noise, blinks, a single
      source seen by both sensors — inflates it. A high PLV alone is weak
      evidence of genuine interhemispheric coupling.
    * **Imaginary coherence** keeps only the part of the cross-spectrum with a
      non-zero phase lag, so zero-lag common-mode contributions cancel. It is
      conservative (it also discards genuine zero-lag coupling), but a change
      in imaginary coherence is much harder to explain away as artifact.

    If an intervention moves PLV but not imaginary coherence, suspect artifact.
    """
    from scipy.signal import csd

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    n = min(len(x), len(y))
    if n < 64:
        return float("nan"), float("nan")
    x, y = x[:n] - np.mean(x[:n]), y[:n] - np.mean(y[:n])
    lo, hi = band

    # --- PLV via the analytic signal of the band-passed pair ---
    sos = butter(4, [lo, hi], btype="band", fs=fs, output="sos")
    px = np.angle(hilbert(sosfilt(sos, x)))
    py = np.angle(hilbert(sosfilt(sos, y)))
    skip = min(len(px) // 4, int(fs))          # drop filter start-up
    dphi = px[skip:] - py[skip:]
    plv = float(np.abs(np.mean(np.exp(1j * dphi)))) if len(dphi) else float("nan")

    # --- imaginary coherence via cross-spectral density ---
    if nperseg is None:
        nperseg = int(min(n, fs * 4))
    nperseg = max(64, min(nperseg, n))
    f, pxy = csd(x, y, fs=fs, nperseg=nperseg)
    _, pxx = welch(x, fs=fs, nperseg=nperseg)
    _, pyy = welch(y, fs=fs, nperseg=nperseg)
    mask = (f >= lo) & (f < hi)
    if not mask.any():
        return plv, float("nan")
    denom = np.sqrt(pxx[mask] * pyy[mask])
    denom[denom <= 0] = np.nan
    coherency = pxy[mask] / denom
    imag = float(np.nanmean(np.abs(np.imag(coherency))))
    return plv, imag


def alpha_asymmetry(alpha_left, alpha_right):
    """Frontal alpha asymmetry: ``ln(right) - ln(left)``.

    Alpha power is *inversely* related to cortical activation, so a positive
    value means more right-hemisphere alpha (i.e. relatively **less** right
    activation / relatively more left activation).
    """
    eps = 1e-12
    return float(np.log(max(alpha_right, eps)) - np.log(max(alpha_left, eps)))


class RollingSpectrogram:
    """Fixed-size time x frequency image built one PSD column at a time."""

    def __init__(self, n_cols=240, fmax=50.0, seconds_per_col=0.5):
        self.n_cols = int(n_cols)
        self.fmax = float(fmax)
        self.seconds_per_col = float(seconds_per_col)
        self.freqs = None
        self.image = None       # (n_freqs, n_cols), dB
        self._filled = 0

    def push(self, x, fs):
        """Append one column computed from the current window."""
        f, p = psd(x, fs)
        if len(f) == 0:
            return
        mask = f <= self.fmax
        f, p = f[mask], p[mask]
        col = 10.0 * np.log10(np.maximum(p, 1e-12))
        if self.image is None or self.image.shape[0] != len(col):
            self.freqs = f
            self.image = np.full((len(col), self.n_cols), col.min())
            self._filled = 0
        self.image = np.roll(self.image, -1, axis=1)
        self.image[:, -1] = col
        self._filled = min(self._filled + 1, self.n_cols)

    def ready(self):
        return self.image is not None and self._filled > 1

    def filled_image(self):
        """Only the columns actually written — avoids rendering empty padding."""
        if not self.ready():
            return None
        return self.image[:, -self._filled:]

    def span_seconds(self):
        return self._filled * self.seconds_per_col

    def levels(self, lo_pct=5, hi_pct=99):
        """Robust colour limits from the filled portion of the image."""
        if not self.ready():
            return (0.0, 1.0)
        data = self.image[:, -self._filled:]
        return (float(np.percentile(data, lo_pct)),
                float(np.percentile(data, hi_pct)))

    def reset(self):
        self.image = None
        self.freqs = None
        self._filled = 0


# --------------------------------------------------------------------------
# fNIRS / optics
# --------------------------------------------------------------------------
def optical_density(intensity, baseline):
    """Change in optical density, ``-log10(I / I0)``.

    Positive ΔOD means *less* light reached the detector — i.e. more absorption,
    which for near-infrared light tracks increased blood volume in the tissue.
    """
    intensity = np.asarray(intensity, dtype=float)
    baseline = float(baseline)
    if baseline <= 0:
        return np.zeros_like(intensity)
    ratio = np.maximum(intensity, 1e-9) / baseline
    return -np.log10(ratio)


def split_lateral(channel_names, prefer_outer=True, exclude_ambient=True):
    """Split optics channels into ``(left_idx, right_idx)``.

    Uses the Athena's ``OPTICS_<side><depth>_<wavelength>`` naming when present.
    By default this keeps only the **outer** (long-separation) detectors, which
    are the ones that actually reach cortex — inner/short channels mostly see
    scalp blood flow — and drops **ambient** channels, which measure background
    light with the LEDs off rather than tissue.

    Falls back to loose name hints, then to splitting the list in half, for
    devices or firmwares that name things differently.
    """
    left, right = [], []
    parsed_any = False
    for i, raw in enumerate(channel_names):
        info = parse_optics_channel(raw)
        if info is None:
            continue
        parsed_any = True
        if exclude_ambient and info["wavelength"] == "AMB":
            continue
        if prefer_outer and info["depth"] != "outer":
            continue
        (left if info["side"] == "left" else right).append(i)
    if left and right:
        return left, right
    if parsed_any and prefer_outer:
        # No outer channels present — retry including the inner ones.
        return split_lateral(channel_names, prefer_outer=False,
                             exclude_ambient=exclude_ambient)

    for i, raw in enumerate(channel_names):
        name = str(raw).upper()
        if any(tag in name for tag in ("_L", "L_", "LEFT", "-L")):
            left.append(i)
        elif any(tag in name for tag in ("_R", "R_", "RIGHT", "-R")):
            right.append(i)
    if left and right:
        return left, right
    half = max(1, len(channel_names) // 2)
    return list(range(half)), list(range(half, len(channel_names)))
