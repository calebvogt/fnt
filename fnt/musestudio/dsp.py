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
from scipy.signal import butter, iirnotch, sosfilt, tf2sos, welch

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


def split_lateral(channel_names):
    """Best-effort split of optics channels into (left_idx, right_idx).

    The Athena's optode layout is not published, so this uses name hints
    (``L``/``R``, ``1``/``2``) and otherwise falls back to splitting the channel
    list in half. Treat laterality from optics as provisional.
    """
    left, right = [], []
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
