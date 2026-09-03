"""Flight Mode control law: does it fly on cortex, and refuse everything else?

The purity tests here are the point of the file. Flight Mode's whole premise is
that the craft responds to cortical alpha and to nothing else -- not blinks, not
jaw clench, not head motion. That is a deliberate constraint from the project
owner, and it is also what separates this from the consumer "mind control" demos
that are quietly reading facial muscle. If ``test_muscle_cannot_fly`` or
``test_blinks_cannot_fly`` ever go green-to-red, the feature has stopped being
what it claims to be, regardless of how well it demos.
"""

import numpy as np
import pytest
from scipy.signal import butter, sosfilt

from fnt.musestudio.flight.pipeline import (
    ControlConfig, ControlPipeline, UPDATE_HZ,
)
from fnt.musestudio.flight.sim import CraftSim, FlightPhase

FS = 256.0
CH = ["TP9", "AF7", "AF8", "TP10"]
HOP = 1.0 / UPDATE_HZ
NHOP = int(FS * HOP)


def synth_eeg(dur, alpha_gain=1.0, emg=0.0, blinks=None, dead=False, seed=0):
    """1/f background with an alpha bump, built in the frequency domain.

    ``alpha_gain`` 1.0 is eyes-open resting alpha; 3.4 reproduces the
    eyes-closed z of about +2.7 measured on subject CV01 with a correctly
    estimated baseline spread.

    The value used to be 2.4, chosen when the baseline MAD was estimated from
    ~5 overlapping windows and every z was therefore inflated. With the span
    requirement in place the scale is honest and the old gain no longer clears
    the hover threshold — the test was calibrated against a broken instrument.
    """
    r = np.random.default_rng(seed)
    n = int(dur * FS)
    if dead:                       # railed / disconnected electrode
        return r.standard_normal(n) * 400.0
    f = np.fft.rfftfreq(n, 1 / FS)
    amp = np.zeros_like(f)
    nz = f > 0
    amp[nz] = 1.0 / (f[nz] ** 0.9)
    bump = np.exp(-0.5 * ((f - 10.0) / 1.1) ** 2)
    amp = amp + alpha_gain * 0.42 * bump * amp.max() * 0.06
    x = np.fft.irfft(amp * np.exp(1j * r.uniform(0, 2 * np.pi, len(f))), n=n)
    x = x / (np.std(x) + 1e-12) * 12.0          # ~12 uV RMS
    if emg:
        sos = butter(4, [20, 110], btype="band", fs=FS, output="sos")
        x = x + emg * sosfilt(sos, r.standard_normal(n))
    if blinks is not None:
        w = int(0.3 * FS)
        for bt in np.atleast_1d(blinks):
            i = int(bt * FS)
            if 0 <= i and i + w < n:
                x[i:i + w] += 200.0 * np.hanning(w)
    return x


class Rig:
    """Drives the pipeline and craft with a continuous clock, as a session does."""

    def __init__(self, cfg=None):
        self.pipe = ControlPipeline(CH, cfg or ControlConfig(fs=FS, baseline_sec=25.0))
        self.craft = CraftSim()
        self.t = 0.0

    def block(self, dur, per_ch=None, **kw):
        sig = per_ch or {c: synth_eeg(dur, seed=i, **kw) for i, c in enumerate(CH)}
        out = []
        for k in range(int(dur / HOP)):
            for c in CH:
                self.pipe.push(c, sig[c][k * NHOP:(k + 1) * NHOP])
            self.t += HOP
            if not self.pipe.ready():
                continue
            tr = self.pipe.tick(t=self.t)
            self.craft.step(tr.thrust, HOP, t=self.t)
            out.append(tr)
        return out


def _calibrated_rig():
    # 50 s, not 35: the baseline now also requires a minimum wall-clock SPAN
    # (ControlConfig.baseline_min_span_s = 40 s) so the MAD is estimated from
    # ~20 independent windows rather than ~5 overlapping ones. A short block
    # produces an under-estimated spread, which inflates every later z — that
    # defect saturated thrust and pinned a real flight at the ceiling.
    r = Rig()
    rest = r.block(50, alpha_gain=1.0)
    assert r.pipe.baseline_ready(), "baseline never completed on clean rest data"
    return r, rest


def test_rest_is_quiet_and_the_craft_stays_down():
    """Resting alpha must not command thrust, or the pilot cannot tell intent
    from noise -- and the craft must not launch itself during calibration."""
    r, rest = _calibrated_rig()
    assert np.mean([abs(t.thrust) for t in rest[-150:]]) < 0.05
    assert r.craft.state.phase is FlightPhase.GROUNDED


def test_alpha_surge_takes_off_and_climbs():
    r, _ = _calibrated_rig()
    r.block(60, alpha_gain=3.4)
    assert r.craft.state.phase is FlightPhase.AIRBORNE
    assert r.craft.state.peak_altitude > 5
    assert any(e[1] == "takeoff" for e in r.craft.state.events)


def test_muscle_cannot_fly():
    """PURITY. A jaw clench with no alpha change must not lift the craft."""
    r, _ = _calibrated_rig()
    emg = r.block(60, alpha_gain=1.0, emg=30.0)
    assert np.mean([t.vetoed for t in emg]) > 0.9, "muscle was not rejected"
    assert r.craft.state.peak_altitude < 1.0, "MUSCLE FLEW THE CRAFT"


def test_blinks_cannot_fly():
    """PURITY. Rhythmic blinking is the single most reliable control primitive
    on this hardware, and is exactly what must not work here."""
    r, _ = _calibrated_rig()
    r.block(60, alpha_gain=1.0, blinks=np.arange(0.4, 60, 1.1))
    assert r.craft.state.peak_altitude < 1.0, "BLINKING FLEW THE CRAFT"


def test_flies_on_the_frontal_pair_alone():
    """Graceful degradation. Subject M01 lost both ear electrodes to long hair
    within two minutes while AF7/AF8 stayed clean all session, so frontal-only
    is the common case rather than an edge case."""
    r = Rig()

    def mixed(dur, gain):
        return {c: (synth_eeg(dur, dead=True, seed=i) if c in ("TP9", "TP10")
                    else synth_eeg(dur, alpha_gain=gain, seed=i))
                for i, c in enumerate(CH)}

    a = r.block(50, per_ch=mixed(50, 1.0))
    assert a[-1].n_accepted == 2, "expected exactly the frontal pair to survive"
    r.block(60, per_ch=mixed(60, 3.4))
    assert r.craft.state.peak_altitude > 5, "could not fly on AF7/AF8 alone"


def test_contact_loss_freezes_rather_than_drops():
    """Losing every electrode must hold the last command, not command zero.

    A craft that falls out of the sky on contact loss teaches the pilot to
    distrust a control signal that was fine -- the electrode failed, not them.
    """
    r, _ = _calibrated_rig()
    r.block(20, alpha_gain=3.4)
    airborne_thrust = r.pipe._thrust
    assert airborne_thrust > 0
    dead = r.block(5, per_ch={c: synth_eeg(5, dead=True, seed=i)
                              for i, c in enumerate(CH)})
    assert all(t.vetoed for t in dead)
    assert dead[-1].thrust == pytest.approx(airborne_thrust, abs=1e-9)


def test_takeoff_gate_needs_sustained_lift():
    """A single threshold crossing is noise; holding it for seconds is an act."""
    sim = CraftSim()
    for _ in range(int(sim.cfg.arm_seconds / 0.1) - 5):
        sim.step(1.0, 0.1)
    assert sim.state.phase is FlightPhase.ARMING
    assert sim.state.altitude == 0.0
    for _ in range(10):
        sim.step(1.0, 0.1)
    assert sim.state.phase is FlightPhase.AIRBORNE


def test_arming_decays_rather_than_resets():
    """One vetoed tick must not wipe out seconds of sustained state."""
    sim = CraftSim()
    for _ in range(10):
        sim.step(1.0, 0.1)
    held = sim.state.arm_progress
    sim.step(0.0, 0.1)              # a single dropout
    assert 0 < sim.state.arm_progress < held
    assert sim.state.phase is FlightPhase.ARMING


def test_landing_is_not_terminal():
    """After touching down the craft must be able to take off again.

    LANDED used to be a dead end: step() handled only GROUNDED/ARMING/AIRBORNE,
    so the first descent in a cued climb/descend test left the craft inert for
    the rest of the run while the timer carried on counting. The pilot was told
    to descend, did so successfully, and was punished with a craft that never
    responded again.
    """
    sim = CraftSim()
    for _ in range(40):                       # arm and climb
        sim.step(1.0, 0.1)
    assert sim.state.phase is FlightPhase.AIRBORNE
    for _ in range(400):                      # let it sink and land
        sim.step(0.0, 0.1)
    assert sim.state.phase is FlightPhase.LANDED
    for _ in range(40):                       # ask for lift again
        sim.step(1.0, 0.1)
    assert sim.state.phase is FlightPhase.AIRBORNE, "could not re-launch after landing"
    assert sum(1 for e in sim.state.events if e[1] == "takeoff") >= 2


# --- imaginary coherence: the artifact discriminator -----------------------

def _pair(kind, fs=256.0, n=512, seed=0):
    rng = np.random.default_rng(seed)
    if kind == "independent":
        return rng.standard_normal(n), rng.standard_normal(n)
    if kind == "common":                       # identical source, zero lag
        s = np.sin(2 * np.pi * 10 * np.arange(n) / fs)
        return s + 0.1 * rng.standard_normal(n), s + 0.1 * rng.standard_normal(n)
    t = np.arange(n + 200) / fs                # genuine lagged coupling
    s = np.sin(2 * np.pi * 10 * t) + 0.5 * rng.standard_normal(len(t))
    k = int(0.025 * fs)
    return s[200:200 + n], s[200 - k:200 - k + n]


def test_imaginary_coherence_rejects_zero_lag_common_mode():
    """The whole point of imagCoh on this montage.

    Every Muse electrode shares an Fpz reference, so a common signal appears in
    both channels at zero lag and inflates PLV. Imaginary coherence must reject
    that while keeping genuinely lagged coupling. It silently did NOT: the
    default nperseg equalled the window length, leaving one Welch segment, and
    single-segment coherence is identically 1 with an imaginary part of
    |sin(random phase)| ~ 0.64. It scored 0.398 on pure common-mode.
    """
    from fnt.musestudio.dsp import band_connectivity
    fs = 256.0
    ind = np.mean([band_connectivity(*_pair("independent", seed=s), fs, (8, 12))[1]
                   for s in range(12)])
    com = np.mean([band_connectivity(*_pair("common", seed=s), fs, (8, 12))[1]
                   for s in range(12)])
    lag = np.mean([band_connectivity(*_pair("lagged", seed=s), fs, (8, 12))[1]
                   for s in range(12)])
    assert com < 0.10, f"zero-lag common-mode not rejected (imagCoh={com:.3f})"
    assert ind < 0.30, f"independent noise should be near zero (got {ind:.3f})"
    assert lag > 0.70, f"genuine lagged coupling must survive (got {lag:.3f})"
    assert lag > com * 5, "discriminator has no dynamic range"


def test_every_clean_channel_eventually_contributes():
    """A late-settling electrode must still join the control signal.

    The global baseline freeze locked out every channel that had not finished
    calibrating at the instant the FIRST one did: baseline_ready() is any(), it
    latched _frozen, at_rest went permanently False, and the remaining channels'
    histories stopped growing so they never reached the target. Measured across
    three real flights, exactly ONE electrode ever contributed a z while the
    other three contributed in 0.0% of windows despite being accepted 45-94% of
    the time — the craft flew on whichever electrode won the race to calibrate.
    """
    cfg = ControlConfig(fs=FS, baseline_sec=8.0, baseline_min_span_s=15.0)
    pipe = ControlPipeline(CH, cfg)
    hop = int(FS / UPDATE_HZ)
    trace = None
    for block in range(2):
        dur = 20
        sig = {}
        for i, c in enumerate(CH):
            dirty = (block == 0 and c != "TP9")   # only TP9 is clean at first
            sig[c] = synth_eeg(dur, alpha_gain=1.0,
                               emg=40.0 if dirty else 0.0, seed=i)
        for k in range(int(dur / 0.1)):
            for c in CH:
                pipe.push(c, sig[c][k * hop:(k + 1) * hop])
            if pipe.ready():
                trace = pipe.tick(t=k * 0.1)

    done = [c for c in CH if len(pipe._history[c]) >= pipe._baseline_target]
    assert len(done) == 4, f"only {done} completed a baseline; the rest were locked out"
    contributing = [c for c, t in trace.channels.items() if t.z != 0.0]
    assert len(contributing) >= 3, (
        f"only {contributing} contributed a z — the craft is flying on too few "
        "electrodes")
