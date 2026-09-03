"""Deep-meditation biofeedback: a depth index and its audio/visual mapping.

WHY THIS IS NOT THE HEMI-SYNC MEASURE
-------------------------------------
Meditative depth and interhemispheric synchrony are different phenomena with
different signatures, and conflating them is the standard error in this area.
Depth is a POWER measure at a site; synchrony is a PHASE relationship between
two sites. A person can go deep with no change in coherence, or synchronize
without going deep. This module measures depth only, and says so.

WHAT THE LITERATURE SUPPORTS
----------------------------
The two most consistently replicated EEG correlates of focused-attention and
mindfulness practice are:

* **Frontal midline theta (4-8 Hz) increases** with sustained attentional
  engagement — generated in anterior cingulate / medial prefrontal cortex, and
  the closest thing to a signature of "absorbed but not drowsy".
* **Alpha (8-12 Hz) increases** with relaxed, internally-directed attention.

Both are visible at AF7/AF8 on this montage, which is what makes this target
tractable here when hemi-sync is not.

THE DROWSINESS PROBLEM, WHICH IS THE WHOLE DIFFICULTY
-----------------------------------------------------
Falling asleep also raises theta. A naive "more theta = deeper" index therefore
rewards dozing off, and would happily train a subject to nap. Two guards:

* Depth requires theta AND alpha to rise together. Sleep onset raises theta
  while alpha *falls* (alpha drops out at stage N1), so the conjunction
  separates absorbed from drowsy where either alone does not.
* A rising delta (1-4 Hz) fraction vetoes the index outright — delta is the
  clearest scalp marker that the subject is descending into sleep rather than
  into meditation.

Everything is expressed relative to the subject's own eyes-closed baseline, as
in the flight pipeline, because absolute band power is not comparable between
people, sessions, or electrode placements.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import welch

THETA = (4.0, 8.0)
ALPHA = (8.0, 12.0)
DELTA = (1.0, 4.0)
TOTAL = (1.0, 20.0)

WINDOW_SEC = 4.0        # longer than flight: depth is a slow variable
UPDATE_HZ = 2.0
BASELINE_SEC = 45.0     # eyes-open rest, against which depth is measured

# Delta fraction above this vetoes the reading as sleep onset rather than depth.
# Delta fraction above which a window is treated as sleep onset rather than
# depth. CALIBRATED ON REAL DATA: delta legitimately holds a median 36% of
# 1-20 Hz power in ordinary waking EEG (CV01, p10 0.20 / p90 0.69), so the
# first guess of 0.55 vetoed 32% of perfectly normal windows — which did not
# just bias the index, it stalled calibration, stretching a 60 s baseline to
# roughly 115 s of real time. 0.80 vetoes 1.2% of waking windows, which is the
# right rate for something that should only fire when the subject is actually
# going under.
DROWSY_DELTA_FRAC = 0.80

# How far above baseline BOTH rhythms must sit before any depth is credited.
DEPTH_MARGIN_Z = 1.5


@dataclass
class DepthMetrics:
    t: float = 0.0
    theta_rel: float = 0.0
    alpha_rel: float = 0.0
    delta_rel: float = 0.0
    theta_z: float = 0.0
    alpha_z: float = 0.0
    depth: float = 0.0          # 0..1, the feedback variable
    drowsy: bool = False
    calibrated: bool = False
    contributing: int = 0
    per_channel: dict = field(default_factory=dict)

    def as_row(self):
        return {"t": round(self.t, 3), "theta_rel": round(self.theta_rel, 6),
                "alpha_rel": round(self.alpha_rel, 6),
                "delta_rel": round(self.delta_rel, 6),
                "theta_z": round(self.theta_z, 4), "alpha_z": round(self.alpha_z, 4),
                "depth": round(self.depth, 4), "drowsy": int(self.drowsy),
                "calibrated": int(self.calibrated), "contributing": self.contributing}


class MeditationIndex:
    """Rolling depth index from theta+alpha, vetoed by delta.

    Pure NumPy/SciPy. Feed with :meth:`push`, read with :meth:`tick`.
    """

    # The baseline must span this much wall clock, not merely accumulate this
    # many ticks. Windows overlap, so a tick count is not a count of independent
    # samples, and a spread estimated from a handful of them is biased low —
    # which inflates every later z. The identical defect in the flight pipeline
    # made plain eyes-closed rest score 0.97 on this index, because theta_z read
    # +3.6 in a condition with no theta change at all.
    BASELINE_MIN_SPAN_S = 45.0

    def __init__(self, channels, fs=256.0, baseline_sec=BASELINE_SEC,
                 baseline_min_span_s=None):
        self.channels = list(channels)
        self.min_span = (self.BASELINE_MIN_SPAN_S if baseline_min_span_s is None
                         else float(baseline_min_span_s))
        self._t0 = None
        self._t = 0.0
        self.fs = float(fs)
        self._n = int(WINDOW_SEC * self.fs)
        self._buf = {c: np.zeros(0) for c in self.channels}
        self._hist = {"theta": [], "alpha": []}
        self._target = max(8, int(baseline_sec * UPDATE_HZ))
        self._base = {}
        self._depth = 0.0

    # ------------------------------------------------------------- ingestion
    def push(self, name, samples):
        if name not in self._buf:
            return
        x = np.asarray(samples, dtype=float).ravel()
        if x.size:
            buf = np.concatenate([self._buf[name], x])
            self._buf[name] = buf[-self._n:] if buf.size > self._n else buf

    def ready(self):
        return all(self._buf[c].size >= int(self._n * 0.6) for c in self.channels)

    def calibrated(self):
        if self._t0 is None or (self._t - self._t0) < self.min_span:
            return False
        return len(self._hist["theta"]) >= self._target

    def progress(self):
        """Toward BOTH requirements, so the bar cannot sit at 100% and wait."""
        by_count = min(1.0, len(self._hist["theta"]) / float(self._target))
        span = (self._t - self._t0) if self._t0 is not None else 0.0
        by_span = min(1.0, span / max(self.min_span, 1e-6))
        return min(by_count, by_span)

    # -------------------------------------------------------------- analysis
    def _bands(self, x):
        x = x - np.mean(x)
        f, p = welch(x, fs=self.fs, nperseg=min(len(x), int(self.fs * 2)))

        def bp(lo, hi):
            m = (f >= lo) & (f < hi)
            return float(np.trapezoid(p[m], f[m])) if m.sum() > 1 else 0.0

        total = bp(*TOTAL)
        if total <= 0:
            return None
        # Theta and alpha are kept ABSOLUTE, delta relative.
        #
        # Relative band powers are compositional — they sum to one — so raising
        # theta mechanically lowers relative alpha. A conjunction built on
        # relative powers therefore fights itself: a genuine deep state that
        # raises both rhythms shows theta up and alpha DOWN, and scores zero.
        # Measured on synthetic data with both bands boosted, relative alpha_z
        # came out at -0.76. Absolute power has no such constraint, and the
        # per-subject baseline z-scoring already removes the scale differences
        # that relative power is normally used to handle.
        #
        # Delta stays relative because its job is different: it is a *fraction*
        # test for sleep onset, where what matters is delta's share of the
        # spectrum rather than its absolute size.
        return {"theta": bp(*THETA), "alpha": bp(*ALPHA),
                "delta": bp(*DELTA) / total}

    def raw_calm(self):
        """Scale-free calm with NO baseline: (theta+alpha) as a share of 1-30 Hz.

        Bounded 0..1 by construction, so it needs no per-session calibration and
        can be compared directly against a recording of the same person on the
        same headband. That is what Mindball actually needs: the opponent is
        your own past trace, so both sides are already on one scale, and the
        game reads a DIFFERENCE which cancels what remains.

        The calibrated depth index is the right tool for training, where the
        question is "deeper than YOUR normal". It is the wrong tool for a match:
        it costs 45 s of sitting still first, and it assumes the player is at
        resting calm during that window — which is precisely when someone about
        to play a competitive game is not.

        Returns (calm, drowsy). Drowsy still vetoes: dozing must not win.
        """
        vals = []
        delta = []
        for c in self.channels:
            x = self._buf[c]
            if x.size < int(self._n * 0.6):
                continue
            xx = x - np.mean(x)
            f, p = welch(xx, fs=self.fs, nperseg=min(len(xx), int(self.fs * 2)))

            def bp(lo, hi):
                m = (f >= lo) & (f < hi)
                return float(np.trapezoid(p[m], f[m])) if m.sum() > 1 else 0.0

            wide = bp(1.0, 30.0)
            if wide <= 0:
                continue
            vals.append((bp(*THETA) + bp(*ALPHA)) / wide)
            delta.append(bp(*DELTA) / wide)
        if not vals:
            return None, False
        return float(np.mean(vals)), bool(np.mean(delta) > DROWSY_DELTA_FRAC)

    def tick(self, t=0.0):
        if self._t0 is None:
            self._t0 = t
        self._t = t
        m = DepthMetrics(t=t)
        vals = []
        for c in self.channels:
            x = self._buf[c]
            if x.size < int(self._n * 0.6):
                continue
            b = self._bands(x)
            if b is None:
                continue
            m.per_channel[c] = b
            vals.append(b)
        m.contributing = len(vals)
        if not vals:
            m.depth = self._depth
            return m

        m.theta_rel = float(np.mean([v["theta"] for v in vals]))
        m.alpha_rel = float(np.mean([v["alpha"] for v in vals]))
        m.delta_rel = float(np.mean([v["delta"] for v in vals]))

        # Sleep-onset veto. Depth must never be rewarded for dozing.
        m.drowsy = m.delta_rel > DROWSY_DELTA_FRAC

        if not self.calibrated():
            if not m.drowsy:
                self._hist["theta"].append(m.theta_rel)
                self._hist["alpha"].append(m.alpha_rel)
            m.depth = 0.0
            return m

        m.calibrated = True
        for key, value in (("theta", m.theta_rel), ("alpha", m.alpha_rel)):
            if key not in self._base:
                h = np.asarray(self._hist[key], dtype=float)
                c0 = float(np.median(h))
                s0 = float(np.median(np.abs(h - c0))) * 1.4826
                self._base[key] = (c0, max(s0, 0.05 * abs(c0), 1e-6))
        m.theta_z = (m.theta_rel - self._base["theta"][0]) / self._base["theta"][1]
        m.alpha_z = (m.alpha_rel - self._base["alpha"][0]) / self._base["alpha"][1]

        # CONJUNCTION, not sum. Depth counts only where BOTH rhythms are above
        # the subject's own rest: theta alone is the drowsiness confound, alpha
        # alone is ordinary eyes-closed rest. Taking the smaller of the two
        # means a high score cannot be bought with either one by itself.
        # Both rhythms must clear a real margin above the subject's own rest,
        # not merely be positive. Requiring only "> 0" let ordinary baseline
        # noise in one band satisfy the conjunction.
        joint = min(m.theta_z, m.alpha_z) - DEPTH_MARGIN_Z
        target = 0.0 if m.drowsy else float(np.clip(joint / 2.5, 0.0, 1.0))
        # Slow smoothing: depth is a state, not an event, and a twitchy index
        # would invite chasing the feedback rather than settling.
        self._depth = 0.85 * self._depth + 0.15 * target
        m.depth = float(np.clip(self._depth, 0.0, 1.0))
        return m
