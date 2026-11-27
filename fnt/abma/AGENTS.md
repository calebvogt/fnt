# ABMA: instructions for AI agents

For agents working in `fnt/abma` (Animal Behavior Modeling Arena). Read the
repository's top-level `AGENTS.md` first: it covers the licence, attribution
and citation terms, which apply here too. User documentation is
`docs/tools/abma.md`; the design history is `fnt/abma/README.md`.

## What ABMA is for

In-silico experiments on small-mammal social behaviour: put model animals in
an enclosure, change one thing (a drug, a gene, a lesion, a connection), and
read the result with the same measures used on real tracking data. The owner's
direction (Caleb C. Vogt, 2026-11): a biologically informed nervous system
whose wiring and simple rewards produce the behaviour, experiments that can be
run like the real ones (dose, timing, cohort), and a tool that is engaging to
use and watch.

## Layout

| Path | What is there |
|---|---|
| `core/simulation.py`, `config.py`, `biology.py` | The engine: bodies, needs, scent, mating, fights, treatments. |
| `core/senses.py` | Every sensor channel a brain can get (`SENSORS`). |
| `core/social.py` | Rest, warmth, attachment (`SocialParams`); off unless enabled. |
| `core/brain.py`, `neural.py` | The generic learned brain (one sensory layer, one recurrent layer). |
| `core/anatomy.py` | A brain wired from an anatomy spec (`brains/anatomy/*.json`): areas, projections, neuromodulators, lesions, plastic connections. |
| `learn/` | Training: `nursery.py` (worlds), `ppo.py` and `anatomy_ppo.py` (reward), `settle.py`, `imitate.py`. |
| `core/assays.py`, `run_headless.py` | Bench assays and the command line. |
| `gui/` | The window and the brain map. |
| `tests/abma/` (repo root) | Tests. Run this folder after any change here. |

## Rules that are easy to break

- **`SENSORS` is append-only.** Weights are matched to sensors by name; add new
  channels at the end, never rename or reorder. A group in `GROUPS` must hold
  each sensor exactly once (a test checks it).
- **A spec's unit counts are part of its weights.** Changing an area's `n`
  invalidates every weights file raised on that spec. Nursery workers reload
  the spec file on every rollout, so do not edit a spec while a training run
  on it is going.
- **`vno_contact` and `vno_mup_*` belong to the house-mouse work.** Do not
  change them for another species; add channels instead.
- **New engine behaviour is off by default** and must leave existing configs
  and shipped brains cell for cell as they were.
- **Keep each file's line endings** (most are CRLF) and keep
  `docs/tools/abma.md` in step with behaviour.
- **Say what is assumed and what emerges.** If the engine computes something a
  brain is supposed to work out (who a partner is, where food was), say so in
  the code and in any result. If a brain was started by imitation, say so.

## Raising an anatomy brain: what works (2026-11)

Reward alone did not teach a brain wired from anatomy to forage in the ticks
we can afford. The route that works:

1. `python -m fnt.abma.learn.settle --anatomy SPEC`: resting levels, so no
   area starts silent or pinned on (a silent cell passes no gradient, and a
   silent VTA or PVN means no dopamine or oxytocin).
2. `python -m fnt.abma.learn.imitate --name N --start SPEC_settled --teacher
   vole_doorways --own 0.5`: taught by a generic brain; `--own` matters.
3. `python -m fnt.abma.learn --anatomy SPEC --resume N --critic-warmup 8`:
   reward alone from there holds and improves it.

Things already found the hard way:

- At a tick of about a second the all-at-once update oscillates (an area's
  inhibition arrives a tick late) and delays a signal by one projection per
  tick. Use `"dynamics": {"sweep": true}` in the spec; the trainer's gradient
  goes through each area's balance point (implicit derivative).
- The muscles' exploration noise has a ceiling (`log_std_max`); without it
  the network stopped commanding the muscles and let the noise do the moving.
- A plastic connection is only as specific as the activity of its source
  cells. In a random recurrent network that is the sense organ's own cells.
- A reward that moves a few points in a 17-minute nursery life teaches
  nothing. Check the arithmetic of a new need before training on it.
- Training is CPU-bound. Use `--torch-threads` and below-normal priority on a
  computer that is also doing other work.

## Who is working on what

Two computers push to `dev`: one on house-mouse dominance and territory
(`c57_male`, contest physics, the skeleton), one on the prairie vole and the
anosmia experiments (`prairie_vole`, `social.py`, plasticity, settling and
imitation). Pull before starting, run `tests/abma` before pushing, and write
commit messages that say what changed for the animal, not only for the code.
