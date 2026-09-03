"""Craft physics for Flight Mode. Pure, deterministic, no Qt.

WHAT THE PILOT ACTUALLY CONTROLS
--------------------------------
Altitude. That is all, and it is a deliberate choice rather than a first
milestone to be outgrown.

The gold-standard BCI quadcopter result (LaFleur & He, 2013) auto-piloted
forward motion -- the brain never controlled thrust -- and that was with a full
64-electrode cap over sensorimotor cortex. With four electrodes, none of them
central, honest continuous control tops out near 1-2 bits/min. Spending that
budget on one axis the pilot can actually learn beats spreading it across three
they cannot.

So the craft flies forward at a constant speed, as a real quadcopter running a
course does, and the single cortical axis raises and lowers it. "Directional
control", the third item in the goal sequence, is reached later as a discrete
branch choice at a gate -- steering with the same one axis rather than adding a
second one -- and is not implemented in this version.

FLIGHT PHASES
-------------
GROUNDED -> ARMING -> AIRBORNE, with LANDED as the terminal state.

ARMING is the takeoff gate: sustained positive thrust for ``arm_seconds``. It is
borrowed from ``protocol.Phase.gate``, which already implements "advance when a
neural condition holds for N seconds", and it exists because a single threshold
crossing is noise, while holding a state for several seconds is a real and
learnable act. It is also the pilot's first unambiguous success -- the moment
they find out that the thing responds to them at all.

SIGN CONVENTION
---------------
Positive thrust is *up*, and positive thrust corresponds to alpha above the
pilot's own baseline -- i.e. eyes closed, attention inward, relaxed. Climbing is
therefore the relaxed state and sinking is the alert one, which is the correct
way round for a meditation instrument: the pilot is rewarded for settling, not
for straining. Straining raises frontal EMG, which the pipeline vetoes anyway.
"""

from dataclasses import dataclass, field
from enum import Enum


class FlightPhase(str, Enum):
    GROUNDED = "grounded"
    ARMING = "arming"
    AIRBORNE = "airborne"
    LANDED = "landed"


@dataclass
class CraftConfig:
    forward_speed: float = 12.0     # world units/s, constant -- never brain-driven
    climb_rate: float = 6.0         # world units/s at full positive thrust
    sink_rate: float = 4.0          # world units/s at full negative thrust
    # Passive descent with zero command. Without it, neutral means "hang here
    # forever" and the pilot gets no feedback from doing nothing; with it,
    # holding altitude is itself an achievement that requires sustained state.
    gravity: float = 1.2
    # Generous, so a good flight is never truncated. At 120 the craft reached
    # the roof in the second cued block of the 172258 flight and the remaining
    # four blocks recorded no altitude change at all — the test measured the
    # ceiling rather than the pilot.
    ceiling: float = 400.0
    # Arming at exactly the hover threshold: if the pilot can hold the craft
    # level, they have earned the lift-off. 0.25 was set before any real data
    # and required z >= 2.13, which sits above CV01's eyes-closed lower quartile
    # (+1.75) — so a genuine, sustained eyes-closed state would still have
    # failed to arm roughly a quarter of the time, for no reason the pilot could
    # perceive. 2 s rather than 3 keeps the first success within a breath or two.
    arm_thrust: float = 0.20
    arm_seconds: float = 2.0
    ground_grace: float = 1.5       # seconds below ground before it counts as landed
    # --- steering (head tilt) ------------------------------------------------
    # Explicitly NOT cortical, and labelled as such everywhere it appears. The
    # purity mandate applies to the LIFT axis, which is what the neurofeedback
    # is training; steering is a deliberate ergonomic concession so the craft is
    # enjoyable to fly. Keeping them on separate modalities is what makes that
    # honest — head motion cannot leak into altitude, because head motion is
    # exactly what the EEG artifact gate rejects.
    turn_gain: float = 1.2          # rad/s at full tilt
    turn_deadzone: float = 0.12     # normalised tilt below this does nothing
    max_turn: float = 1.0           # rad/s ceiling


@dataclass
class CraftState:
    t: float = 0.0
    heading: float = 0.0            # radians; 0 = straight ahead
    turn_rate: float = 0.0          # rad/s, from head tilt
    lateral: float = 0.0            # accumulated left/right offset
    distance: float = 0.0           # forward progress
    altitude: float = 0.0
    vertical_speed: float = 0.0
    thrust: float = 0.0
    phase: FlightPhase = FlightPhase.GROUNDED
    arm_progress: float = 0.0       # 0..1 toward takeoff
    airborne_seconds: float = 0.0
    peak_altitude: float = 0.0
    events: list = field(default_factory=list)   # (t, name, detail)

    def as_row(self):
        return {
            "distance": round(self.distance, 3),
            "altitude": round(self.altitude, 3),
            "vertical_speed": round(self.vertical_speed, 3),
            "phase": self.phase.value,
            "arm_progress": round(self.arm_progress, 3),
        }


class CraftSim:
    """Integrates thrust commands into craft motion.

    Fixed-``dt`` stepping driven by the control tick, so a replayed session
    reproduces the original flight exactly from the recorded thrust column --
    which is what makes the review mode trustworthy rather than a re-enactment.
    """

    def __init__(self, config=None):
        self.cfg = config or CraftConfig()
        self.state = CraftState()
        self._arm_held = 0.0
        self._below_ground = 0.0

    def reset(self):
        self.state = CraftState()
        self._arm_held = 0.0
        self._below_ground = 0.0

    def step(self, thrust, dt, t=None, tilt=0.0):
        s = self.state
        cfg = self.cfg
        thrust = float(max(-1.0, min(1.0, thrust)))
        s.thrust = thrust
        s.t = t if t is not None else s.t + dt

        # Steering from head tilt. Applied whenever airborne; a dead zone keeps
        # ordinary postural sway from turning the craft.
        tilt = float(max(-1.0, min(1.0, tilt)))
        if abs(tilt) < cfg.turn_deadzone:
            tilt = 0.0
        else:
            tilt = (abs(tilt) - cfg.turn_deadzone) / (1.0 - cfg.turn_deadzone) * (1 if tilt > 0 else -1)
        s.turn_rate = float(max(-cfg.max_turn, min(cfg.max_turn, tilt * cfg.turn_gain)))
        if s.phase is FlightPhase.AIRBORNE:
            s.heading += s.turn_rate * dt
            s.lateral += s.turn_rate * cfg.forward_speed * dt * 0.5

        if s.phase in (FlightPhase.GROUNDED, FlightPhase.ARMING,
                       FlightPhase.LANDED):
            # LANDED is NOT terminal. Touching down used to leave the craft
            # inert for the rest of the run, so a cued climb/descend test died
            # the first time the pilot let it sink — the timer kept counting
            # while nothing on screen could respond to them again. Landing is
            # just being on the ground; lift again and you take off again.
            if s.phase is FlightPhase.LANDED and thrust >= self.cfg.arm_thrust:
                s.phase = FlightPhase.GROUNDED
                self._below_ground = 0.0
            self._step_ground(thrust, dt)
        elif s.phase is FlightPhase.AIRBORNE:
            self._step_air(thrust, dt)
        return s

    # ---------------------------------------------------------------- phases
    def _step_ground(self, thrust, dt):
        s, cfg = self.state, self.cfg
        if thrust >= cfg.arm_thrust:
            self._arm_held += dt
            if s.phase is FlightPhase.GROUNDED:
                s.phase = FlightPhase.ARMING
                s.events.append((s.t, "arming", "sustained lift detected"))
        else:
            # Decay rather than reset. A single vetoed tick -- one swallow, one
            # electrode blip -- should not wipe out three seconds of held state
            # and tell the pilot they failed when they did not.
            self._arm_held = max(0.0, self._arm_held - dt * 2.0)
            if self._arm_held <= 0.0 and s.phase is FlightPhase.ARMING:
                s.phase = FlightPhase.GROUNDED
                s.events.append((s.t, "arm_lost", "lift not sustained"))

        s.arm_progress = min(1.0, self._arm_held / max(cfg.arm_seconds, 1e-6))
        if s.arm_progress >= 1.0:
            s.phase = FlightPhase.AIRBORNE
            s.arm_progress = 1.0
            s.events.append((s.t, "takeoff", f"held {cfg.arm_seconds:.0f}s"))

    def _step_air(self, thrust, dt):
        s, cfg = self.state, self.cfg
        rate = cfg.climb_rate if thrust >= 0 else cfg.sink_rate
        s.vertical_speed = thrust * rate - cfg.gravity
        s.altitude += s.vertical_speed * dt
        s.distance += cfg.forward_speed * dt
        s.airborne_seconds += dt
        s.peak_altitude = max(s.peak_altitude, s.altitude)

        if s.altitude >= cfg.ceiling:
            s.altitude = cfg.ceiling
            s.vertical_speed = min(s.vertical_speed, 0.0)

        if s.altitude <= 0.0:
            s.altitude = 0.0
            self._below_ground += dt
            if self._below_ground >= cfg.ground_grace:
                s.phase = FlightPhase.LANDED
                s.vertical_speed = 0.0
                s.events.append((s.t, "landed",
                                 f"{s.distance:.0f} units, peak {s.peak_altitude:.0f}"))
        else:
            self._below_ground = 0.0
