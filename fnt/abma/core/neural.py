"""The policy that puts a learned brain in charge of an animal's movement.

:class:`NeuralPolicy` satisfies the same contract as the rule-based policy —
``decide(sim, elapsed_s, dt)`` returns a desired heading per animal plus the
intermediates the engine needs — so the engine does not know or care which is
driving. What differs is everything inside: there are no drives, no weights on
them and no knowledge of where anything is. There is a row of sensory input
(:mod:`.senses`), a recurrent network (:mod:`.brain`), and a turn and a speed.

Two things the brain controls that the rule-based policy never did:

* **Speed.** The engine used to slow an animal to a rest when it was fed,
  watered and on familiar ground — a behaviour it was simply given. A brain
  emits its own speed, so resting is something the animal does or does not do.
  The engine still owns how fast it *can* go.
* **Its own state.** Each animal carries a recurrent state between ticks, plus a
  record of where it last ate and drank. Both are per-animal arrays that grow
  when a protocol event adds animals mid-run.

What the body is trying to reduce
---------------------------------
:func:`discomfort` is the one scalar a learned animal is rewarded for lowering:
a convex sum of hunger, thirst, stress and injury, all read off the body's own
bars. Reward during learning is the *drop* in it from one tick to the next
(homeostatic reinforcement, after Keramati & Gutkin 2014), so eating is worth a
lot to a starving animal and nothing to a fed one. No behaviour is rewarded
directly — nothing in the reward says "approach food" or "keep away from
rivals" — only what the behaviour did to the body.
"""
from __future__ import annotations

import numpy as np

from .brain import Brain, find_brain
from .policy import Policy
from .rng import CH_ACTION, CH_BITE
from .senses import PlaceMemory, sense


def discomfort(sim) -> np.ndarray:
    """How far each animal's body is from where it wants to be (0 = content).

    Convex in each deficit, so the same meal relieves a starving animal far
    more than a peckish one. Injury is weighted most heavily: it is the one
    deficit that does not come back on its own.
    """
    d = ((sim.hunger / 100.0) ** 2
         + (sim.thirst / 100.0) ** 2
         + 0.5 * (sim.stress / 100.0) ** 2
         + 2.0 * ((100.0 - sim.health) / 100.0) ** 2)
    # tiredness (zero unless ContestParams is on) is the mildest of them: it
    # is what makes rest worth something
    fatigue = getattr(sim, "fatigue", None)
    if fatigue is not None:
        d = d + 0.25 * (fatigue / 100.0) ** 2
    # being tired, cold or apart from the animal it is attached to (zero
    # unless SocialParams is on)
    soc = getattr(sim, "soc", None)
    if soc is not None:
        soc.resize(len(d))
        d = d + soc.discomfort()
    return d


class NeuralPolicy(Policy):
    """Movement decided by a :class:`~fnt.abma.core.brain.Brain`."""

    #: below this fraction of its available speed an animal counts as resting
    REST_FRAC = 0.12

    def __init__(self, brain: Brain, stochastic: bool = True,
                 noise_scale: float = 1.0):
        self.brain = brain
        self.stochastic = bool(stochastic)
        self.noise_scale = float(noise_scale)
        self.params = brain.senses
        self.memory: PlaceMemory | None = None
        self.h = None                 # (n, n_rnn) recurrent state
        self.u = None                 # (n, 2) last pre-squash motor command
        self.last_speed = None
        self.last_turn = None
        #: what went in and what came out on the last tick, for recording and
        #: for a learner that wants to score the action
        self.last_obs = None
        self.last_sensory = None
        self.last_motor = None
        self.ticked = False
        self._clock = np.inf          # seconds since the brain last ticked
        #: extra perturbations, as `Brain.forward` takes them; set directly by
        #: a twin experiment. Config-scheduled ones are added on top each tick.
        self.stim: list = []
        #: per-sensor gain, (n_sensors,) or (n, n_sensors) — a sensory lesion
        self.sensor_gain = None

    @staticmethod
    def from_config(cfg) -> "NeuralPolicy":
        bp = cfg.brain
        return NeuralPolicy(Brain.load(find_brain(bp.name)),
                            stochastic=bp.stochastic,
                            noise_scale=bp.noise_scale)

    # ---- per-animal state ----------------------------------------------- #
    def _ensure(self, n: int) -> None:
        if self.h is None:
            self.h = self.brain.initial_state(n)
            self.u = np.zeros((n, self.brain.n_out), np.float32)
            self.last_speed = np.zeros(n)
            self.last_turn = np.zeros(n)
            self.memory = PlaceMemory(n)
            return
        have = len(self.h)
        if n > have:                              # the roster grew mid-run
            pad = n - have
            self.h = np.vstack([self.h, self.brain.initial_state(pad)])
            self.u = np.vstack([self.u, np.zeros((pad, self.brain.n_out),
                                                 np.float32)])
            self.last_speed = np.concatenate([self.last_speed, np.zeros(pad)])
            self.last_turn = np.concatenate([self.last_turn, np.zeros(pad)])
            self.memory.resize(n)

    def _scheduled_stim(self, sim, elapsed_s: float) -> list:
        """Config-scheduled unit perturbations that are active right now."""
        out = []
        for st in getattr(sim.cfg, "stimulations", []) or []:
            t0 = st.at_day * 86400.0
            if not (t0 <= elapsed_s < t0 + st.duration_s):
                continue
            rows = np.array([a.index for a in sim.agents
                             if sim._match_target(a, st.target)], int)
            if len(rows):
                out.append((st.layer, int(st.unit), st.op, float(st.value),
                            rows))
        return out

    # ---- the contract --------------------------------------------------- #
    def decide(self, sim, elapsed_s: float, dt: float):
        n = sim.n
        self._ensure(n)
        brain = self.brain
        obs, perc = sense(sim, elapsed_s, self.memory, self.params,
                          self.last_speed, self.last_turn)
        if self.sensor_gain is not None:
            obs = obs * np.asarray(self.sensor_gain, np.float32)

        # The brain runs on its own clock. A simulation stepping finer than the
        # brain's tick holds the last command until the next tick is due, so a
        # brain behaves the same at any integration step no coarser than that.
        self._clock += dt
        self.ticked = self._clock >= brain.tick_s - 1e-9
        if self.ticked:
            self._clock = 0.0
            stim = list(self.stim) + self._scheduled_stim(sim, elapsed_s)
            motor, self.h, sensory = brain.forward(obs, self.h, stim or None)
            noise = None
            if self.stochastic:
                noise = sim.arand.normal_2d(sim.uid, sim._step_k, CH_ACTION)
                if brain.n_out > 2:       # the bite command has its own stream
                    noise = np.column_stack(
                        [noise, sim.arand.normal(sim.uid, sim._step_k, CH_BITE)])
            _, _, self.u = brain.act(motor, noise, self.noise_scale)
            self.last_obs, self.last_sensory = obs, sensory
            self.last_motor = motor
        turn, speed = brain.squash(self.u)
        # a turn is per tick; at a finer step it is spread across the tick
        turn_now = turn * min(1.0, dt / brain.tick_s)
        heading = sim.H + turn_now
        desired = np.stack([np.cos(heading), np.sin(heading)], axis=1)
        self.last_speed = speed.astype(float)
        self.last_turn = (turn / brain.max_turn).astype(float)

        perc.update(
            speed_frac=np.where(sim.alive, speed, 0.0),
            resting=speed < self.REST_FRAC,
            drives={}, want_chew=np.zeros(n, bool),
        )
        if brain.n_out > 2:
            # issued once, on the tick it was decided; the engine holds the
            # refractory period
            perc["bite"] = brain.bite(self.u) & self.ticked & sim.alive
        return desired, perc

    # ---- recording ------------------------------------------------------ #
    def frame_extra(self, n: int) -> dict:
        """Neural activity for the live view and the run record."""
        if self.h is None or self.last_obs is None:
            return {}
        turn, speed = self.brain.squash(self.u)
        return {
            "brain_rnn": self.h[:n].copy(),
            "brain_sensory": self.last_sensory[:n].copy(),
            "brain_obs": self.last_obs[:n].copy(),
            "brain_turn": (turn / self.brain.max_turn)[:n].copy(),
            "brain_speed": speed[:n].copy(),
        }


def make_policy(cfg) -> Policy:
    """The policy a config asks for: a learned brain, or the rules."""
    bp = getattr(cfg, "brain", None)
    if bp is not None and bp.enabled and getattr(bp, "anatomy", ""):
        from .anatomy import AnatomicalPolicy
        return AnatomicalPolicy.from_config(cfg)
    if bp is not None and bp.enabled:
        return NeuralPolicy.from_config(cfg)
    from .policy import RuleBasedPolicy
    return RuleBasedPolicy()
