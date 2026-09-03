"""Cortical EEG -> one control signal, with every intermediate step preserved.

WHAT FLIES THE CRAFT
--------------------
Relative alpha power (8-12 Hz), fused across whichever scalp electrodes are
currently clean, normalized against the pilot's own rolling baseline, and mapped
through a dead zone to a thrust command in [-1, +1].

That is the entire control law, and the shortness is the point. Four electrodes
at TP9/AF7/AF8/TP10 cannot support the sensorimotor mu/beta scheme every
published BCI-drone demo uses -- that signal is generated at C3/C4, over
sensorimotor cortex, and this montage has no central electrodes at all. What it
*can* measure, verified on this exact rig at Cohen's d = +2.38, is the Berger
effect: alpha rises when the eyes close and attention turns inward. So that is
what we fly on, and Flight Mode is therefore an eyes-closed instrument flight.

PURITY MANDATE
--------------
Blinks, saccades, jaw clench, frontalis/temporalis EMG and head motion are
NOISE TO BE REJECTED here, never signal to be exploited. This is a deliberate
constraint from the project owner, and it is also good science: the reason
consumer "mind control" demos look impressive is that most of them are reading
facial muscle. Two consequences run through this module:

* **Nothing above 20 Hz reaches the control signal.** Whitham et al. (2007)
  paralysed subjects and showed that scalp EEG above ~20 Hz is substantially
  muscle. Both the control band and the relative-power denominator stop at 20 Hz.
* **20-45 Hz is used only as an accusation.** High power there means the
  temporalis or frontalis is active, so the tick is vetoed -- the control signal
  freezes rather than responding. Muscle can stop the craft; it can never fly it.

WHY RELATIVE, NOT ABSOLUTE, ALPHA
---------------------------------
Absolute band power tracks electrode impedance as much as it tracks cortex, and
impedance is exactly what drifts during a session -- subject M01's TP10 went from
87 to 491 uV in two minutes as hair worked between the sensor and the skin.
Dividing alpha by total 1-20 Hz power in the same channel cancels most of that
gain term, so a slowly-loosening electrode does not read as a slowly-climbing
craft.

NORMALIZATION IS PER-ELECTRODE
------------------------------
Each electrode is normalized against *its own* rolling baseline, and the control
signal is the mean of those per-electrode z-scores over whichever electrodes are
currently clean.

Averaging raw relative alpha across the accepted set instead -- the obvious first
implementation -- has a subtle and serious bug, found by replaying M01's real
session: the number of accepted electrodes climbed from 0.4 to 1.8 over nine
minutes as the subject settled and frontal EMG fell. Every electrode sits at a
different resting alpha, so the fused average moved simply because its membership
changed. The pilot would have been climbing on the arrival of an electrode.
Normalizing first makes every contributor mean-zero at rest, so composition can
change without the control signal jumping.

NORMALIZATION
-------------
Raw relative alpha has no meaningful absolute scale -- 0.3 is high for one person
and low for another, and high for the same person in the morning and low after
lunch. Following BCI2000's Normalizer, the pipeline subtracts a running centre
and divides by a running spread, so the control signal is "how far above your own
recent baseline are you right now". Median and MAD rather than mean and SD,
because a single swallowed artifact should not redefine the scale.

The baseline deliberately keeps updating during flight. A fixed baseline is the
documented failure mode of long neurofeedback sessions: the pilot's resting alpha
drifts within minutes and a fixed threshold silently becomes unreachable, or
trivially reachable. The cost is that sustaining a state forever eventually
re-centres it -- you cannot park at full thrust -- which is honest, and is why
the craft is designed around climbing rather than hovering.
"""

from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, sosfiltfilt, welch

# --- bands ------------------------------------------------------------------
CONTROL_BAND = (8.0, 12.0)      # alpha: the one verified large effect on this rig
TOTAL_BAND = (1.0, 20.0)        # relative-power denominator; hard-stops below EMG
ARTIFACT_BAND = (20.0, 45.0)    # muscle accusation band -- never a control input

# --- timing -----------------------------------------------------------------
# 2 s is 20 cycles of a 10 Hz rhythm, which is enough for a band-power estimate
# whose variance is smaller than the effect being measured. Shorter windows are
# tempting for responsiveness and produce an estimate dominated by its own noise.
WINDOW_SEC = 2.0
UPDATE_HZ = 10.0                # windows overlap; only statistics need independence

# --- gates ------------------------------------------------------------------
# Amplitude ceiling for a usable epoch. dsp.CONTACT_MAX_UV (150) judges whether an
# electrode is seated at all over many seconds; this is the per-tick version and
# sits lower, because a 2 s window containing a blink is not a window we want to
# fly on even though the electrode itself is fine.
# Also swept against M01: 60 gave the best contrast (d = +1.01) at the same yield
# as 100 or 150, and it correctly excludes the electrode that actually failed
# (TP10 ran a 99 uV median that session, against AF8's 7.9).
EPOCH_MAX_UV = 60.0
EPOCH_MIN_UV = 0.5
# Fraction of in-band power allowed to sit in the muscle band before the tick is
# vetoed. CALIBRATED ON REAL DATA (subject M01, 2026-08-16, 9-minute session).
#
# Measured median EMG ratios per electrode were AF8 0.47, TP10 0.92, AF7 0.99,
# TP9 1.40 -- far above the 0.45 originally guessed here, which vetoed 48% of all
# ticks outright and left the craft unflyable. Sweeping the threshold against
# that session showed 0.8 to be the joint optimum: it keeps 84% of ticks usable
# (vs 52% at 0.45) AND maximises the eyes-open -> eyes-closed alpha contrast the
# whole control law rests on (Cohen's d = +1.01, vs +0.91 at 0.45). Looser is not
# safer here and stricter is not purer -- both directions cost effect size.
#
# The same sweep showed eyes-OPEN EMG ratios run 2-3x eyes-closed (AF7 3.28 vs
# 1.27; TP9 3.20 vs 1.34), which is independent support for flying this with the
# eyes shut: it is the quieter signal regime, not merely the higher-alpha one.
EMG_RATIO_MAX = 0.8

# Percentile of each electrode's own baseline 20-45 Hz power used as its
# muscle ceiling. 90 keeps 90% of good windows while rejecting 46-80% of
# labelled artifacts, measured on CV01's artifact blocks.
EMG_LIMIT_PCT = 90.0

MIN_CHANNELS = 1                # fly on a single good electrode rather than refuse


@dataclass
class ChannelTrace:
    """Everything the pipeline concluded about one electrode on one tick."""
    name: str
    amplitude_uv: float = 0.0
    alpha_abs: float = 0.0
    alpha_rel: float = 0.0
    emg_ratio: float = 0.0
    emg_abs: float = 0.0          # absolute 20-45 Hz power — the real gate
    emg_limit: float = 0.0        # this electrode's own learned ceiling
    center: float = 0.0           # this electrode's own rolling baseline
    scale: float = 1.0
    z: float = 0.0                # this electrode's own normalized deviation
    accepted: bool = False
    reject_reason: str = ""


@dataclass
class PipelineTrace:
    """One tick, every stage. This dataclass *is* the reviewability guarantee.

    Flight Mode is flown with the eyes closed, so the pilot cannot watch the
    craft while flying it. Everything they will later want to ask -- why did it
    climb there, was that me or an artifact, which electrode was carrying it --
    has to be answerable from what we recorded at the time. So each stage is kept
    rather than collapsed into a final number.
    """
    t: float = 0.0
    channels: dict = field(default_factory=dict)     # {name: ChannelTrace}
    n_accepted: int = 0
    fused: float = 0.0            # mean relative alpha over accepted channels
    center: float = 0.0           # rolling baseline centre (median)
    scale: float = 1.0            # rolling baseline spread (MAD -> sigma)
    z: float = 0.0                # normalized deviation from own baseline
    thrust: float = 0.0           # final command, -1..+1
    vetoed: bool = False
    veto_reason: str = ""
    baseline_ready: bool = False

    def as_row(self):
        """Flat dict for the telemetry CSV."""
        row = {
            "t": round(self.t, 6),
            "n_accepted": self.n_accepted,
            "fused": round(self.fused, 6),
            "center": round(self.center, 6),
            "scale": round(self.scale, 6),
            "z": round(self.z, 4),
            "thrust": round(self.thrust, 4),
            "vetoed": int(self.vetoed),
            "veto_reason": self.veto_reason,
            "baseline_ready": int(self.baseline_ready),
        }
        for name, c in self.channels.items():
            row[f"{name}_amp_uv"] = round(c.amplitude_uv, 2)
            row[f"{name}_alpha_rel"] = round(c.alpha_rel, 6)
            row[f"{name}_emg"] = round(c.emg_ratio, 4)
            row[f"{name}_emg_abs"] = round(c.emg_abs, 3)
            row[f"{name}_emg_lim"] = round(c.emg_limit, 3)
            row[f"{name}_z"] = round(c.z, 4)
            row[f"{name}_ok"] = int(c.accepted)
        return row


@dataclass
class ControlConfig:
    fs: float = 256.0
    window_sec: float = WINDOW_SEC
    # Counted in ACCEPTED ticks, not wall-clock: eyes-open is the EMG-heaviest
    # block, and M01 yielded only 241 clean windows across a 60 s eyes-open
    # phase (AF8 119, AF7 44, TP10 47, TP9 31). Asking for 30 s x 10 Hz = 300
    # clean ticks per electrode would never have completed. 10 s worth (100) is
    # reachable by the best electrode inside a 60 s block -- AF8 produced 119.
    #
    # That margin (119 vs 100) is uncomfortably thin, and it is why the preflight
    # phase must run UNTIL baseline_ready() rather than for a fixed duration. A
    # fixed 60 s block that happens to yield 99 clean windows silently leaves the
    # baseline incomplete, and it then finishes mid-flight -- absorbing the
    # pilot's eyes-closed state as "normal" and flattening the control signal to
    # nothing. That exact failure appeared in replay before this was pinned down.
    baseline_sec: float = 10.0
    # The baseline must also SPAN this much wall clock before it counts.
    #
    # Tick count alone is not enough and this defect made the craft unflyable.
    # 100 accepted ticks of 2 s windows at 10 Hz is ~10 s of signal containing
    # only ~5 INDEPENDENT windows, and a MAD estimated from 5 samples is biased
    # badly low — which divides into every later z and inflates it. Measured on
    # the 172258 flight: median flight z was +4.56 against a full-scale of 4.0,
    # thrust saturated in 31% of windows, and the craft pinned at the ceiling in
    # block 3, making the remaining four cued blocks meaningless.
    #
    # 40 s gives ~20 non-overlapping windows, which is enough for a stable
    # spread. The settle phase already waits for baseline_ready(), so this
    # extends it automatically rather than needing a longer fixed block.
    baseline_min_span_s: float = 40.0
    adapt_sec: float = 240.0       # rolling memory, if adaptation is left on
    # Both constants are measured, not chosen. Replaying M01's session with the
    # baseline fixed on the eyes-open block gives, for eyes-closed rest:
    # median z = +0.90, IQR +0.10 to +2.34, mean +1.66; and for eyes-open
    # itself: median 0.00, std 1.21.
    #
    # So the two states overlap heavily and alpha arrives in bursts rather than
    # as a plateau -- which is what alpha actually does. A dead zone of 1.2 (the
    # first guess here) sits above the eyes-closed median and the craft never
    # leaves the ground. 0.8 clears the eyes-open noise floor while leaving the
    # bursts room to lift, and 3.0 puts the eyes-closed upper quartile near 70%
    # thrust with the median close to neutral.
    #
    # Combined with sim.py's gravity, break-even hover lands at z = +1.24:
    # eyes-closed (mean +1.66) climbs on average, eyes-open (mean +0.13) sinks,
    # and burst-to-burst variation is what the pilot feels as control.
    # RE-MEASURED on subject CV01, 2026-09-02 (clean session, AF7 100% clean):
    # eyes-open z p50 +0.02 / p90 +1.74; eyes-closed z p50 +2.66 / p90 +4.22,
    # Cohen's d = +1.89 with non-overlapping IQRs. With gravity/climb giving a
    # break-even hover at thrust 0.20, these put eyes-open at 0.00 (sinks) and
    # eyes-closed at 0.46 (climbs +1.6 units/s) — a usable margin either side.
    dead_zone: float = 1.5
    full_scale_z: float = 4.0
    smoothing: float = 0.25        # EMA on thrust, 0..1 (lower = smoother)
    epoch_max_uv: float = EPOCH_MAX_UV
    emg_ratio_max: float = EMG_RATIO_MAX

    # Freeze the baseline when preflight ends, rather than letting it keep
    # adapting. Adaptation guards against session-long drift, but it costs two
    # things this feature cannot afford: the pilot's sustained success slowly
    # redefines itself as normal and the craft sinks for invisible reasons, and
    # -- decisively -- a moving reference makes the post-flight review nearly
    # impossible to read, since the same alpha value means different things at
    # different times. A frozen baseline makes every later frame comparable.
    # Drift over a 5-10 minute flight is the accepted cost, and the review view
    # plots the baseline so it stays visible rather than assumed.
    freeze_after_calibration: bool = True

    # NOTE: both gates are absolute and calibrated on a single subject. If a
    # second subject's clean resting EMG ratio sits above 0.8, this rejects them
    # wholesale. Making the veto relative to each pilot's own calibration-block
    # EMG is the obvious next step; it is deliberately not done yet, because one
    # subject is not enough data to fit a per-subject model against.


class ControlPipeline:
    """Stateful translator: raw multi-channel EEG windows -> thrust in [-1, 1].

    Push samples with :meth:`push`, call :meth:`tick` at ``UPDATE_HZ``. Pure
    NumPy/SciPy -- no Qt, no LSL -- so it runs identically on live hardware,
    on a replayed recording, and on synthetic test signals.
    """

    def __init__(self, channels, config=None):
        self.cfg = config or ControlConfig()
        self.channels = list(channels)
        self._n = max(1, int(self.cfg.window_sec * self.cfg.fs))
        self._buf = {c: np.zeros(0) for c in self.channels}
        self._history = {c: [] for c in self.channels}   # per-electrode baselines
        self._emg_hist = {c: [] for c in self.channels}  # per-electrode muscle floor
        self._emg_limit = {}                             # learned ceilings
        self._hist_max = max(8, int(self.cfg.adapt_sec * UPDATE_HZ))
        self._baseline_target = max(4, int(self.cfg.baseline_sec * UPDATE_HZ))
        self._thrust = 0.0
        self._frozen = False
        self._t_first_sample = None
        self._t_last = 0.0
        self._settle_z = []
        self._settle_raw = []
        self._sos_band = butter(4, list(TOTAL_BAND), btype="band",
                                fs=self.cfg.fs, output="sos")
        self._frozen_z = 0.0

    # ------------------------------------------------------------ ingestion
    def push(self, name, samples):
        """Append samples for one channel, keeping only the analysis window."""
        if name not in self._buf:
            return
        x = np.asarray(samples, dtype=float).ravel()
        if x.size == 0:
            return
        buf = np.concatenate([self._buf[name], x])
        self._buf[name] = buf[-self._n:] if buf.size > self._n else buf

    def ready(self):
        """True once every channel holds at least 60% of a window."""
        need = int(self._n * 0.6)
        return all(self._buf[c].size >= need for c in self.channels)

    def baseline_progress(self):
        """Progress toward BOTH requirements: enough ticks and enough span."""
        best = max((len(h) for h in self._history.values()), default=0)
        by_count = min(1.0, best / float(self._baseline_target))
        span = (self._t_last - self._t_first_sample) if self._t_first_sample is not None else 0.0
        by_span = min(1.0, span / max(self.cfg.baseline_min_span_s, 1e-6))
        return min(by_count, by_span)

    def baseline_ready(self):
        if self._t_first_sample is None:
            return False
        if (self._t_last - self._t_first_sample) < self.cfg.baseline_min_span_s:
            return False
        return any(len(h) >= self._baseline_target for h in self._history.values())

    def calibrate_thresholds(self, floor=1.0):
        """Set the thrust mapping from the pilot's OWN eyes-open z spread.

        A dead zone fixed in z units cannot work, because the z scale is not
        comparable between sessions: replaying four real flights, the median
        flight z ranged from +0.52 to +7.59 against a full-scale of 4.0, so the
        same constant meant "barely moving" in one session and "saturated 65% of
        the time" in another. That is why one flight pinned at the ceiling.

        Once the baseline is frozen, z measured over the REMAINDER of the
        eyes-open block is a genuine sample of this pilot's resting spread on
        this session's scale. Putting the dead zone at its 90th percentile makes
        rest quiet by construction, and full-scale a fixed step above that keeps
        the flight state reachable without saturating. This is exactly the rule
        flight/calibration.py already recommends offline; it just has to run
        live too.

        Returns (dead_zone, full_scale_z) actually applied.
        """
        # Freeze now, then score the ENTIRE settle block against the final
        # baseline. Scoring only the post-freeze tail against an early baseline
        # measured a mismatch rather than the pilot's resting spread.
        self._frozen = True
        z = []
        for sample in self._settle_raw:
            per = []
            for name, ar in sample.items():
                hist = self._history.get(name) or []
                if len(hist) < self._baseline_target:
                    continue
                c0, s0 = self._baseline(hist)
                if s0 > 0:
                    per.append((ar - c0) / s0)
            if per:
                z.append(float(np.mean(per)))
        z = [v for v in z if np.isfinite(v)]
        if len(z) >= 20:
            dz = float(np.percentile(z, 90))
            dz = float(np.clip(dz, floor, 12.0))
            self.cfg.dead_zone = dz
            # +2.0 rather than +2.5: on CV01's measured distributions
            # (eyes-open z p90 = 1.74, eyes-closed median = +2.66) this puts the
            # flight state at thrust 0.46 against a 0.20 hover threshold — a
            # clear margin either side rather than a knife-edge.
            self.cfg.full_scale_z = dz + 2.0
        return self.cfg.dead_zone, self.cfg.full_scale_z

    def baseline_frozen(self):
        return self._frozen

    def baseline_summary(self):
        """Per-electrode (n, center, scale) — recorded so a flight can be read back."""
        return {c: (len(h), *self._baseline(h)) for c, h in self._history.items()}

    # ------------------------------------------------------------- analysis
    def _channel_trace(self, name):
        x = self._buf[name]
        tr = ChannelTrace(name=name)
        if x.size < int(self._n * 0.6):
            tr.reject_reason = "short buffer"
            return tr

        x = x - np.mean(x)
        # Zero-phase filter is legitimate here: we work on a completed window,
        # not a running stream, so there is no causality violation -- the window
        # is already 2 s in the past by definition.
        y = sosfiltfilt(self._sos_band, x)
        tr.amplitude_uv = float(np.percentile(np.abs(y), 95))

        nperseg = min(len(x), int(self.cfg.fs))
        freqs, pxx = welch(x, fs=self.cfg.fs, nperseg=nperseg)

        def band_power(lo, hi):
            m = (freqs >= lo) & (freqs < hi)
            return float(np.trapezoid(pxx[m], freqs[m])) if m.sum() > 1 else 0.0

        total = band_power(*TOTAL_BAND)
        tr.alpha_abs = band_power(*CONTROL_BAND)
        tr.alpha_rel = tr.alpha_abs / total if total > 0 else 0.0
        emg = band_power(*ARTIFACT_BAND)
        tr.emg_abs = emg
        tr.emg_ratio = emg / total if total > 0 else 0.0

        # Muscle gate: ABSOLUTE 20-45 Hz power against this electrode's own
        # learned ceiling — not the ratio, and not a shared constant.
        #
        # The ratio was backwards. Measured on the labelled artifact blocks, a
        # blink scores LOWER on emg_ratio than eyes-closed rest (AF7 d = -1.81,
        # AF8 -1.09, TP10 -1.64) because it dumps enormous energy into the
        # 1-20 Hz denominator faster than into the numerator. A ratio gate
        # therefore passed artifacts and rejected clean data. Absolute 20-45 Hz
        # power separates every artifact on every channel in the right
        # direction (+0.62 to +1.88).
        #
        # The ceiling is per-electrode and learned, because the floor differs
        # 17x across sensors on the same head (CV01: AF7 1.3, TP10 21.8) --
        # the Fpz reference sits between AF7 and AF8 and suppresses them. A
        # single shared number rejects at best 41% of artifacts; each
        # electrode's own 90th percentile rejects 46-80% while keeping 90% of
        # good windows. Learned from the pilot's own baseline so no subject's
        # values are ever baked in -- the previous constants came from a
        # different subject in a different regime and were wrong here.
        limit = self._emg_limit.get(tr.name)
        tr.emg_limit = limit or 0.0
        if tr.amplitude_uv > self.cfg.epoch_max_uv:
            tr.reject_reason = f"amplitude {tr.amplitude_uv:.0f}uV"
        elif tr.amplitude_uv < EPOCH_MIN_UV:
            tr.reject_reason = "flat/disconnected"
        elif limit and tr.emg_abs > limit:
            tr.reject_reason = f"muscle {tr.emg_abs:.0f}>{limit:.0f}"
        elif limit is None and tr.emg_ratio > self.cfg.emg_ratio_max:
            # Before the ceiling is learned, fall back to the old ratio gate so
            # the baseline block itself is not admitted wholesale.
            tr.reject_reason = f"muscle(ratio) {tr.emg_ratio:.2f}"
        else:
            tr.accepted = True
        return tr

    def tick(self, t=0.0):
        """Run one full pipeline pass and return its complete trace."""
        if self._t_first_sample is None:
            self._t_first_sample = t
        self._t_last = t
        trace = PipelineTrace(t=t)
        trace.channels = {c: self._channel_trace(c) for c in self.channels}
        self._learn_emg_limits(trace)
        if not self._frozen:
            # Keep the raw fused inputs from the settle block so the thrust
            # mapping can be scored retrospectively against the FINAL baseline.
            good_now = [c for c in trace.channels.values() if c.accepted]
            if good_now:
                self._settle_raw.append({c.name: c.alpha_rel for c in good_now})
        good = [c for c in trace.channels.values() if c.accepted]
        trace.n_accepted = len(good)

        if len(good) < MIN_CHANNELS:
            # Freeze rather than fall. A dropped thrust command on contact loss
            # would read to the pilot as "I lost it", teaching them to distrust
            # a control signal that was actually fine -- the electrode failed,
            # not the brain. Hold the last command and say so.
            trace.vetoed = True
            reasons = {c.reject_reason for c in trace.channels.values() if c.reject_reason}
            trace.veto_reason = "; ".join(sorted(reasons)) or "no usable channel"
            trace.z = self._frozen_z
            trace.thrust = self._thrust
            trace.baseline_ready = self.baseline_ready()
            return trace

        # Normalize each electrode against itself, then fuse the z-scores.
        # Freeze EACH channel as it completes, not all of them when the first
        # one does.
        #
        # The global freeze was a serious defect, confirmed in every recorded
        # flight: baseline_ready() is any(), so the moment ONE electrode reached
        # its target the global _frozen latched, at_rest went permanently False,
        # and every other channel's history stopped growing — so they never
        # reached _baseline_target, never passed the guard below, and never
        # contributed a z again. Measured across three flights, exactly one
        # electrode ever contributed (TP9, AF7, TP9) while the other three
        # contributed in 0.0% of windows despite being ACCEPTED 45-94% of the
        # time. The craft was flying on a single electrode picked by whichever
        # won the race to calibrate.
        # NOTE: no early freeze. The baseline keeps building through the whole
        # settle block and is frozen explicitly by calibrate_thresholds() at the
        # transition to the first cue. Freezing as soon as the minimum was met
        # left the rest of settle being scored against a baseline that did not
        # represent it: measured on flight 173350, resting z p90 came out at
        # 3.91 while eyes-closed climb reached only 4.04 — an excess of +0.13,
        # thrust 0.039, and a craft that could not leave the ground.
        globally_settled = self.baseline_ready() and abs(self._frozen_z) >= self.cfg.dead_zone
        zs = []
        for c in good:
            hist = self._history[c.name]
            # Adapt on rest, not on effort. Drift is the documented failure mode
            # of long neurofeedback sessions, so the baseline has to keep moving
            # -- but one that also absorbs the pilot's successful climbs
            # re-centres the very state they are being rewarded for, and the
            # achievement decays under them for reasons they cannot see.
            # This channel keeps building until IT has enough, regardless of
            # what the others have done. Once complete it stops, so a finished
            # baseline is still frozen against the pilot's own effort.
            channel_done = len(hist) >= self._baseline_target
            if not channel_done and not globally_settled:
                hist.append(c.alpha_rel)
                if len(hist) > self._hist_max:
                    hist.pop(0)
            c.center, c.scale = self._baseline(hist)
            if len(hist) >= self._baseline_target and c.scale > 0:
                c.z = (c.alpha_rel - c.center) / c.scale
                zs.append(c.z)

        trace.fused = float(np.mean([c.alpha_rel for c in good]))
        trace.center = float(np.mean([c.center for c in good]))
        trace.scale = float(np.mean([c.scale for c in good]))
        trace.baseline_ready = self.baseline_ready()
        if not zs:
            # Channels were ACCEPTED but none has a finished baseline yet, so
            # there is no z to report. Emitting 0.0 with vetoed=False (the old
            # behaviour) wrote a fake "perfectly average" sample into telemetry.
            # Those zeros are cue-correlated — they cluster where windows are
            # rejected, which is not evenly spread across cues — so they
            # manufactured climb-vs-descend contrast that was not in the EEG.
            # The recorded d of +0.93 was substantially this artifact.
            trace.vetoed = True
            trace.veto_reason = "baseline not ready"
            trace.z = self._frozen_z
            trace.thrust = self._thrust
            return trace
        trace.z = float(np.mean(zs))
        self._frozen_z = trace.z
        if self._frozen:
            self._settle_z.append(trace.z)

        target = self._to_thrust(trace.z)
        a = float(np.clip(self.cfg.smoothing, 0.01, 1.0))
        self._thrust = (1 - a) * self._thrust + a * target
        trace.thrust = float(np.clip(self._thrust, -1.0, 1.0))
        return trace

    # ------------------------------------------------------------- internals
    def _learn_emg_limits(self, trace):
        """Learn each electrode's muscle ceiling during the baseline block."""
        for name, c in trace.channels.items():
            if name in self._emg_limit:
                continue
            # Only windows that pass the amplitude gate; a blink-laden window
            # would otherwise raise the ceiling it is supposed to trip.
            if EPOCH_MIN_UV < c.amplitude_uv <= self.cfg.epoch_max_uv:
                self._emg_hist[name].append(c.emg_abs)
            if len(self._emg_hist[name]) >= self._baseline_target:
                self._emg_limit[name] = float(
                    np.percentile(self._emg_hist[name], EMG_LIMIT_PCT))

    def _baseline(self, hist):
        if not hist:
            return 0.0, 1.0
        h = np.asarray(hist, dtype=float)
        center = float(np.median(h))
        # MAD -> sigma via the normal-consistency constant, so `full_scale_z`
        # keeps its plain reading as "standard deviations above your own rest".
        mad = float(np.median(np.abs(h - center)))
        scale = mad * 1.4826
        # Floor the spread relative to the centre. A quiet stretch can drive the
        # MAD arbitrarily close to zero, and dividing by it turns a meaningless
        # wobble into a saturated command -- the craft flying on rounding error.
        floor = max(0.05 * abs(center), 1e-6)
        if scale < floor:
            scale = max(float(np.std(h)), floor)
        return center, scale

    def _to_thrust(self, z):
        """Dead zone, then linear ramp to saturation.

        The dead zone is the fix for the Midas-touch problem: without it every
        moment is a command, ordinary drift becomes constant twitching, and the
        pilot cannot tell their intent from the noise floor.
        """
        dz = self.cfg.dead_zone
        span = max(self.cfg.full_scale_z - dz, 1e-6)
        # One-sided on purpose. Lift is commanded; descent is simply the absence
        # of lift, and gravity in sim.py supplies it. Giving alpha *below*
        # baseline its own active-dive command would double the surface on which
        # ordinary drift can be mistaken for intent, and buy nothing: the pilot
        # has no use for diving faster than falling.
        if z <= dz:
            return 0.0
        return float(min((z - dz) / span, 1.0))
