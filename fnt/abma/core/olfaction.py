"""A mechanistic nose, replacing ``recognition = smell_ability x identity_signal``.

Why the scalar version was not enough
-------------------------------------
ABMA's original olfaction gate multiplied two numbers: how good the perceiver's
nose is, and how distinctive the target's scent is. That reproduces the headline
result — take the nose away and territorial spacing collapses — but it can only
ever return the assumption it was given. In particular it cannot produce:

  * **selective anosmia.** Methimazole ablates olfactory epithelium; it does not
    turn a global gain knob. Two animals at the same dose lose *different*
    receptor channels, so they end up confused about different individuals.
  * **graded confusability.** Under a scalar gate every target is equally
    recognisable. Real recognition failures are structured: animals whose
    signatures differ on a surviving channel stay discriminable, and the ones
    that differ only on a lost channel become interchangeable.
  * **presence without identity.** Detecting that *somebody* marked a spot and
    knowing *who* are separate readouts that fail separately.

Those are the phenomena a habituation-dishabituation experiment measures, so a
model that cannot express them cannot be compared against one.

The three layers
----------------
**Emission.** Each animal emits an odour profile over ``n_channels`` chemical
channels — a point on the simplex, so it is a mixture, not a magnitude. A
private profile is drawn from the animal's own stable random stream. The
``identity_signal`` trait is *distinctiveness*: the emitted profile is a blend
between the animal's private profile and the flat population-average profile::

    emitted = (1 - identity) * flat + identity * private

At ``identity_signal = 0`` — a MUP knockout — every animal emits the population
average. The marks are still real (a nose detects that someone passed) but they
carry no individual information at all. That falls out of the geometry rather
than being special-cased.

**Reception.** Each animal has a per-channel receptor gain vector. Anosmia is
channel loss: at ``smell_ability = g`` the animal keeps a ``g`` fraction of its
receptor mass, and *which* channels survive is drawn from that animal's own
stream. The construction is exact rather than sampled (top-ranked channels at
full gain, one fractional channel, the rest dead), so mean receptor gain equals
``smell_ability`` for every animal at every dose. That is deliberate: it makes
this model reduce to the scalar one in the mean, so the previously validated
methimazole dose-response is preserved and the only new content is structure.

``ablation_selectivity`` interpolates between the two readings of a dose:
``0`` is a uniform gain reduction on every channel (the old model), ``1`` is
whole channels dying. The default is 1 because that is what an epithelial
lesion actually does.

**Readout.** Under perceiver *i*'s receptors, target *j*'s emitted profile is
perceived as ``r_i * e_j``. Two quantities come out of it:

  ``detect[i, j]``   fraction of j's odour mass that i's surviving receptors
                     capture at all — the "can I smell anything" channel.
  ``sep[i, j]``      how far j's perceived profile sits from the nearest *other*
                     animal's perceived profile — the "is this someone in
                     particular" channel.

Separability is normalised against a reference cohort with intact receptors and
full distinctiveness, so an intact wild-type animal reads ``recognition ~= 1``
and the old scalar model is recovered. Reduced distinctiveness lowers the
numerator only; lost receptors lower it in a target-specific way.

Cost and caching
----------------
The recognition matrix depends only on signatures and receptors, both of which
change just when biology changes — at release, at drug onset, at an
intervention, when the roster changes. It is therefore computed on a dirty flag,
not per step. The build is O(n^2 * channels) for detection and O(n^3 * channels)
for separability, which is nothing at the cohort sizes ABMA runs (8-100).

Honesty
-------
Channel count, discrimination slope and confusion threshold are **free**
parameters in the sense of :data:`fnt.abma.core.project.SOURCES` — they are not
measured from any animal. What is claimed here is structural: that recognition
should degrade selectively rather than uniformly. The magnitudes are chosen so
the cohort aggregate matches the model this replaces.
"""
from __future__ import annotations

import numpy as np

from .rng import CH_OLFACTION

#: stable per-agent draw tags (constants, not time steps — see AgentRandom)
_TAG_SIGNATURE = 0x5163
_TAG_RECEPTOR = 0x5164

#: separability below this is treated as identical. Guards the 0/0 that a
#: MUP-KO cohort produces, where the reference profiles are flat too.
_SEP_EPS = 1e-6


def _softmax_rows(x: np.ndarray) -> np.ndarray:
    """Map gaussian rows onto the simplex — a positive, normalised mixture."""
    if x.size == 0:
        return x
    z = np.exp(x - x.max(axis=1, keepdims=True))
    return z / np.clip(z.sum(axis=1, keepdims=True), 1e-12, None)


def emitted_profiles(private: np.ndarray, identity: np.ndarray) -> np.ndarray:
    """Blend each animal's private profile toward the flat population profile.

    ``identity`` is the ``identity_signal`` trait in 0..1. At 0 the animal emits
    the flat profile and is individually anonymous; at 1 it emits its own.
    """
    n, d = private.shape
    flat = np.full((1, d), 1.0 / max(1, d))
    c = np.clip(identity, 0.0, 1.0)[:, None]
    out = (1.0 - c) * flat + c * private
    return out / np.clip(out.sum(axis=1, keepdims=True), 1e-12, None)


def receptor_gains(rank_key: np.ndarray, smell: np.ndarray,
                   selectivity: float) -> np.ndarray:
    """Per-channel receptor gains whose row mean is exactly ``smell``.

    ``rank_key`` is a stable per-agent random key per channel; the channels an
    animal keeps are the ones ranked highest by its own key, so two animals
    given the same dose lose different channels. The construction is exact
    rather than sampled (whole channels at full gain, one fractional channel,
    the rest dead), which makes the row mean equal ``smell_ability`` instead of
    binomially noisy — the property that lets this model reproduce the scalar
    model's dose-response.

    ``selectivity`` blends between a uniform gain reduction (0, the old model)
    and whole-channel loss (1, an epithelial lesion).
    """
    n, d = rank_key.shape
    g = np.clip(np.asarray(smell, float), 0.0, 1.0)[:, None]
    # rank 0 = the channel this animal preserves first
    order = np.argsort(rank_key, axis=1)[:, ::-1]
    rank = np.empty_like(order)
    np.put_along_axis(rank, order,
                      np.broadcast_to(np.arange(d), (n, d)).copy(), axis=1)
    budget = g * d                      # channel-equivalents of surviving mass
    selective = np.clip(budget - rank, 0.0, 1.0)
    s = float(np.clip(selectivity, 0.0, 1.0))
    return s * selective + (1.0 - s) * np.broadcast_to(g, (n, d))


def _separability(receptors: np.ndarray, emitted: np.ndarray) -> np.ndarray:
    """``sep[i, j]``: distance from j's perceived profile to the nearest other.

    Perceived profile of j under i's receptors is ``r_i * e_j``, compared by
    cosine similarity because what carries identity is the *pattern* across
    channels, not the overall intensity — that is what ``detect`` measures.
    """
    n = emitted.shape[0]
    if n == 0:
        return np.zeros((0, 0))
    q = receptors[:, None, :] * emitted[None, :, :]        # (i, j, d)
    norm = np.linalg.norm(q, axis=2, keepdims=True)
    qhat = q / np.clip(norm, 1e-12, None)
    cos = np.einsum("ijd,ikd->ijk", qhat, qhat)
    if n > 1:
        cos = np.where(np.eye(n, dtype=bool)[None, :, :], -np.inf, cos)
        nearest = cos.max(axis=2)
    else:
        nearest = np.zeros((n, 1))
    sep = np.clip(1.0 - nearest, 0.0, 2.0)
    # a target the perceiver cannot smell at all is not "maximally distinct"
    return np.where(norm[:, :, 0] <= 1e-12, 0.0, sep)


class OlfactorySystem:
    """Signatures, receptors and the recognition matrix they imply.

    Rebuilt only when biology changes (drug onset, intervention, roster
    change), never per step — see the module docstring.
    """

    def __init__(self, params, arand, uids):
        self.p = params
        self.arand = arand
        self.d = max(2, int(getattr(params, "n_channels", 8)))
        self._recog = None
        self._detect = None
        self.set_roster(uids)

    # ---- stable per-agent draws ---------------------------------------- #
    def set_roster(self, uids) -> None:
        """Assign the per-agent signature and receptor keys for this roster.

        Both come from the agent's own counter-based stream keyed by its stable
        uid, so an animal's smell and its nose are properties of that animal
        rather than of its row position or of when it joined the run.
        """
        self.uids = np.asarray(uids, dtype=np.int64)
        n = len(self.uids)
        if n == 0:
            self.private = np.zeros((0, self.d))
            self.rank_key = np.zeros((0, self.d))
        else:
            self.private = _softmax_rows(
                self.arand.vector(self.uids, _TAG_SIGNATURE, CH_OLFACTION,
                                  self.d))
            self.rank_key = self.arand.vector(
                self.uids, _TAG_RECEPTOR, CH_OLFACTION, self.d)
        self._recog = None
        self._detect = None

    # ---- the readout ---------------------------------------------------- #
    def rebuild(self, smell, identity) -> None:
        """Recompute detection and recognition for the current biology."""
        n = len(self.uids)
        if n == 0:
            self._recog = np.zeros((0, 0))
            self._detect = np.zeros((0, 0))
            self.emitted = np.zeros((0, self.d))
            self.receptors = np.zeros((0, self.d))
            return
        smell = np.asarray(smell, float)[:n]
        identity = np.asarray(identity, float)[:n]
        self.emitted = emitted_profiles(self.private, identity)
        self.receptors = receptor_gains(
            self.rank_key, smell,
            float(getattr(self.p, "ablation_selectivity", 1.0)))

        # detection: fraction of the target's odour mass this nose captures
        mass = np.clip(self.emitted.sum(axis=1), 1e-12, None)
        self._detect = np.clip(
            (self.receptors @ self.emitted.T) / mass[None, :], 0.0, 1.0)

        # separability, normalised against an intact, fully distinctive
        # reference cohort so a wild-type animal reads ~1 and the scalar model
        # is recovered. Reduced distinctiveness lowers only the numerator.
        sep = _separability(self.receptors, self.emitted)
        ref = _separability(np.ones_like(self.receptors),
                            emitted_profiles(self.private, np.ones(n)))
        scale = np.where(ref > _SEP_EPS,
                         sep / np.clip(ref, _SEP_EPS, None), 0.0)

        slope = float(getattr(self.p, "discrimination", 6.0))
        thr = float(getattr(self.p, "confusion_threshold", 0.15))
        # psychometric readout: how reliably this pattern is told apart, given
        # a perceptual threshold below which two odours are indistinguishable
        conf = 1.0 / (1.0 + np.exp(-slope * (scale - thr)))
        # anchored at both ends so an intact animal reaches 1 and a fully
        # confused one reaches 0, rather than the sigmoid's asymptotes
        lo = 1.0 / (1.0 + np.exp(slope * thr))
        hi = 1.0 / (1.0 + np.exp(-slope * (1.0 - thr)))
        conf = np.clip((conf - lo) / max(1e-9, hi - lo), 0.0, 1.0)
        self._recog = np.clip(self._detect * conf, 0.0, 1.0)

    # ---- accessors ------------------------------------------------------ #
    @property
    def recognition(self) -> np.ndarray:
        """``R[i, j]``: how well i can tell that a scent belongs to j (0..1)."""
        if self._recog is None:
            raise RuntimeError("OlfactorySystem.rebuild() has not been called")
        return self._recog

    @property
    def detection(self) -> np.ndarray:
        """``D[i, j]``: how much of j's odour i smells at all (presence)."""
        if self._detect is None:
            raise RuntimeError("OlfactorySystem.rebuild() has not been called")
        return self._detect

    def acuity(self) -> np.ndarray:
        """Per-animal mean receptor gain — the scalar ``smell_ability`` again.

        Kept so callers that only need "how good is this nose" (mark
        deposition, display, the legacy path) need not know about channels.
        """
        return (self.receptors.mean(axis=1) if len(self.uids)
                else np.zeros(0))

    def channel_loss(self) -> np.ndarray:
        """Fraction of channels each animal has effectively lost (for display)."""
        if not len(self.uids):
            return np.zeros(0)
        return (self.receptors < 0.5).mean(axis=1)
