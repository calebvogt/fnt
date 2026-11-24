# ABMA — Animal Behavior Modeling Arena

A GUI-driven agent-based platform for running *in silico* animal-behaviour
experiments inside the FieldNeuroToolbox (FNT). Design an arena,
populate it with genetically and pharmacologically manipulated agents, run
replicate trials, and export tracking data in FNT's canonical schema for
post-hoc analysis in R — the same pipeline you use for real Ultra-Wideband data.

Launch from the FNT main GUI: **ABMA** tab → **Open ABMA Designer**.

---

## The core design principle

**The simulator emits the exact schema FNT produces for real UWB tracking.**

Each trial writes `data/uwb_<trial>_processed.csv` with columns:

```
Trial, Species, sex, sexid, shortid, Date, Day, Timestamp, time_sec,
location_x, location_y, smoothed_x, smoothed_y, Meso1Start
```

Because this is byte-for-byte the output of the UWB PreProcessing tool, every
downstream analysis — proximity detection, daily edgelists, group-by-individual
matrices, network metrics, your R scripts — runs on simulated animals with **zero
changes**. Validation is a direct apples-to-apples comparison between real and
simulated socio-spatial metrics.

---

## Architecture

```
fnt/fnt/abma/
  core/                     headless engine — no GUI, no Qt
    config.py               ExperimentConfig & sub-dataclasses; JSON round-trip
    presets.py              ready-made paradigms (Blank, Open Field Test, vole)
    sampling.py             attribute distribution specs — N(33,3) etc.
    biology.py              gene + drug registries -> trait modifications
    rng.py                  per-agent counter-based random streams
    olfaction.py            receptor/signature nose (selective anosmia)
    policy.py               rule-based policy + the drive decomposition
    senses.py               what an animal can sense, from where it stands
    brain.py                a learned brain: numpy network, files, stimulation
    neural.py               NeuralPolicy (a brain in charge) + discomfort
    twin.py, twin_plot.py   one life lived twice: counterfactual experiments
    assays.py               bench tests: approach, choice, search, vitals
    simulation.py           Simulation: physics, physiology, combat (dice-roll
                            contests, or bites / sprint / fatigue / cover), events
    scent.py                decaying grid of identity-carrying marks
    sward.py                living grass: trails they wear and share
    sky.py                  sun, moon and season for a real site
    physiology.py           energy/water budget in real units
    compose.py              describe a run in one line -> ExperimentConfig
    analysis.py             built-in socio-spatial + dominance analysis
    structure.py            hierarchy or territories: readouts from zone visits,
                            the same for real RFID data and simulated runs
    recorder.py             streaming writers for the canonical CSV schema
    record.py               replayable per-frame archive (drives + sensing)
    provenance.py           config hash + where every number came from
    study.py                conditions x replicates, paired seeds, comparison
    runner.py               project folders + multi-trial orchestration
    run_headless.py         CLI: python -m fnt.abma config.json --out DIR
  learn/                    raising a brain (the only part that needs PyTorch)
    nursery.py              the randomised worlds a brain grows up in
    ppo.py                  recurrent PPO; python -m fnt.abma.learn
  brains/                   brains shipped with ABMA (.npz)
  gui/                      thin PyQt5 wrapper (needs a QApplication)
    abma_canvas.py          2D matplotlib canvas + territory-map overlay
    pg_canvas.py            pyqtgraph OpenGL 3D live run view (PyOpenGL)
    agent_inspector.py      hover stat card
    science_panel.py        roster, drive bars, traces, brain raster
    twin_dialog.py          run a twin experiment from the window
    launch.py               open the window on a config and run it
    abma_main_pyqt.py       ABMAWindow: Arena / Population / Experiment / Run
```

ABMA is a **general-purpose ABM sandbox**, not a vole tool: it opens on a blank
slate (empty arena, one generic agent type). The prairie-vole/anosmia setup is
just **File → Load Example**. You (a) build an arena, (b) define an agent type and
stamp N seeded copies, (c) script duration/speed and a **timed intervention
schedule**, (d) run and watch it in the 3D view.

**Decision policy.** Movement rules live in `policy.py` as `RuleBasedPolicy` (all
`k_*` weights). The engine owns physics/physiology/combat; the policy owns "what
the agent wants to do". A learned brain (`NeuralPolicy`, below) plugs into the
same seam with no engine change; the rules stay the default.

**Intervention schedule.** Interventions are `(at_day, target, attribute, op,
value)` applied at run time — e.g. induce anosmia on day 3 with
`3 · all · smell_ability · scale · 0`. Editable in the Experiment tab.

**Layout.** Left is a column of **collapsible section steps** (1·Arena,
2·Build & Add Agents, 3·Experiment, 4·Run log) in the FNT house style (cf. Mask
Tracker / MAD), with a **persistent Run bar** pinned at the bottom so the primary
action + progress are always visible. The Experiment step shows only the
essentials (Duration, Replicates, a Resolution preset that sets dt/record); the
rest lives under a collapsed **Advanced**. Right is one **interactive preview**
used for *both* setup and running — drag to orbit, scroll to zoom — with a
**3D/2D toggle** and a **transport bar** (play/pause · timeline scrubber · speed ·
reset-camera · top-down · follow-selected). During editing a **live preview**
animates the agents (unsaved); after a run the frame buffer is retained so you can
**scrub/replay** it. The agent inspector is not docked — **hover** an agent to pop
its stat card near the cursor, **click** to pin. The Arena step has a
**Load Preset Arena…** button (dialog picker).

**Live views (2D ⇄ 3D toggle).** The preview is either a 2D top-down canvas or a
pyqtgraph OpenGL 3D scene. The 3D scene renders the arena as a **solid object**
(floor slab + true walls, e.g. the 50×50×50 cm OFT box) and draws each agent as an
oriented **body + head** so heading is visible; the 2D view shows heading via
ticks. Both support trails, day/night, and click-to-inspect; the 3D view also
supports click-to-place (floor ray-cast). 3D needs PyOpenGL; without it the view
stays 2D. Agents' `heading` is streamed each frame, so position *and* facing show
(not just an xy point).

**Replicates side-by-side.** Set **Trials (replicates)** > 1 and the run steps all
replicates in lockstep, laying their chambers out in a grid in the preview so you
watch them simultaneously (each seeded differently; each writes its own trial
CSVs). Big headless parameter sweeps still use the parallel path.

**Presets & zones.** `core/presets.py` is a registry of ready-made paradigms
loaded from **File → Load Preset** — e.g. the **Open Field Test** (50×50 cm empty
box, one subject, 10 min). Arenas carry **zones** (`ArenaConfig.zones`): named
rectangular regions rendered in both the 2D and 3D views and measured by the
analysis. The OFT centre zone (inner 50%) yields `center_time_pct` in the summary
plus a `zone_occupancy_<trial>.csv` — the classic thigmotaxis / anxiety readout.
Add a paradigm by writing a factory and appending a `Preset` to `PRESETS`.

**VoleCosm** (`--preset volecosm`) is the indoor prairie-vole mesocosm: a
2.81 × 2.61 m floor, a walled 0.51 m resource box in each corner (chow, a water
tower and a hut inside; two floor-level doorways, one in each wall that faces
the room), a free-standing water tower in the middle, 3 M + 3 F, ten days,
14:10 light. The floor and the boxes are the measured ones; doorway width, box
height, the central tower's size and the antenna mounts are approximate (see
the factory's docstring). Building it needed three things the engine did not
have, all now general:

* **A zone can have more than one doorway.** `ResourceZone.entrance` takes one
  wall or several (`"E"`, `"NE"`). An animal inside leaves by the nearer one.
* **A zone can hold water** (`ResourceZone.water`). Drinking, like feeding,
  then happens anywhere inside the box, and from outside the way to it is a
  doorway. A tower *object* placed inside the walls would be drinkable through
  them.
* **A light cycle that is not 12:12.** `ExperimentConfig.day_length_hours`
  (**Light hours** in the Experiment tab) with `day_start_hour`; ignored when
  the sky is on.

It also exposed two faults. A rule-based animal inside a box was still pulled
toward the doorway it had come through, so it hovered on the threshold and fed
or drank at about half the rate; it now settles toward the middle of a box that
holds what it came for. And a box wall that ended exactly at the room's wall
could be walked around on the outside: a step overshoots the boundary before it
is reflected, passes beyond the end of the wall, and is bounced back in on the
other side. Walls that meet the boundary are now carried on past it. That leak
affected every preset with a box against a wall.

The engine is fully importable and scriptable without the GUI:

```python
from fnt.abma.core.config import default_vole_experiment
from fnt.abma.core.runner import run_experiment

cfg = default_vole_experiment()
cfg.days = 10; cfg.n_trials = 3
run_experiment(cfg, "/path/to/ABMA_projects/my_run")
```

A project folder is self-describing and re-runnable:

```
<project>/
  config.json              full ExperimentConfig
  provenance.json          config hash + where every number came from
  README.txt               how it was generated
  data/
    uwb_S001_processed.csv  trajectory (FNT schema)
    events_S001.csv         mating / aggression events
    condition_S001.csv      condition-bar time series
    agents_S001.csv         per-agent metadata (genotype, treatment, traits)
    record_S001.npz         replayable frames: drives + sensing + condition
    ... one set per trial
```

---

## Behaviour model (v1)

Each agent integrates a weighted blend of drives every timestep (all gains are
transparent constants on `Simulation`, so they can be tuned):

| Drive | What it does |
|-------|--------------|
| Home-range spring (`k_home`) | pulls an agent toward its **emergent** home centre; gives site fidelity |
| Resource seeking (`k_resource`) | heads to nearest food/water when hungry/thirsty |
| Social forces (`k_social`) | pairwise: opposite-sex attraction (amplified by female estrus), female–female affiliation, male–male avoidance |
| **Territory avoidance** (`k_territory`) | same-sex agents avoid others' scent-marked home ranges |
| Random walk (`k_random`) | correlated exploratory noise |

Animals are released together near the arena centre and **self-organise**: each
agent's home centre slowly tracks its occupied position (`_settle_tau_s`), so
territories emerge rather than being prescribed.

### The olfaction gate (the point of the whole thing)

Social affiliation and territory avoidance are **gated by recognition**:

```
recognition(i sees j) = smell_ability[i] × identity_signal[j]
```

(That is the default. `OlfactionParams.enabled` swaps it for a receptor and
signature model whose cohort mean tracks the same curve but which fails
*selectively* — see **The mechanistic nose** below.)

- **Methimazole → anosmia** scales `smell_ability` toward 0 (dose-dependent).
- **MUP knockout** sets `identity_signal` to 0 (no individual scent signature).

Either collapses recognition, so agents stop respecting territories and the
socio-spatial structure restructures. Validated dose-response (3-day trials,
mean male–male spacing):

```
methimazole dose  0.00 → 1.21 m   (intact: clean territories)
                  0.50 → 1.14 m
                  0.75 → 0.97 m
                  1.00 → 0.43 m   (anosmic: territories collapse)
```

---

## Agent model: attributes vs. condition

Each agent has two layers (a "stat block" plus a live "condition"):

**Attributes** (static, set at creation): `species`, `sex`, `mass` (g),
`aggression`, `boldness`, `sociability`, `exploration`, `metabolism`,
`smell_ability`, `identity_signal`, `base_speed`, `home_range_r`, plus
`genotype` and `treatment` (which modify the rest).

**Seeding from domain knowledge.** Any attribute can be a fixed value *or* a
distribution each founder samples individually, so a cohort varies realistically
instead of being identical clones. In the Population table an attribute cell
accepts `N(mean,sd)`, `N(mean,sd)[min,max]` (truncated), or `U(min,max)` — e.g.
mass `N(33,3)` seeds 8 males around 33 g. (Under the hood this is `AgentGroup.dists`
and `core/sampling.py`.)

**Condition** (dynamic 0–100 bars, updated every step): **Health**, **Energy**,
**Hunger**, **Thirst**, **Stress**, plus body **mass** (drifts with energy
balance within a physiological band). Health/energy are decoupled — an animal can
be exhausted but healthy, or injured but rested.

Coupled dynamics: locomotion costs energy ∝ mass × speed (so mass is a
dominance-vs-foraging tradeoff); feeding restores energy; crowding raises stress,
solitude lowers it; health heals when fed and calm, erodes when starving or
chronically stressed; death occurs at Health 0 (if `enable_mortality`).

**Combat & dominance.** Same-sex animals in *active patrol* contest with
probability ∝ both animals' aggression; the winner is drawn from resource-holding
potential `aggression × √mass × health × (1+boldness) × (1−0.3·stress)`. The loser
takes health damage, spends energy, gains stress, and flees. Repeated losses can
be fatal, and the win/loss record yields a **dominance hierarchy** (David's score
in `dominance_<trial>.csv`).

**Live inspector.** Click any agent in the run canvas to open a docked stat card:
identity + genotype/treatment/status badges, innate stats, five live condition
bars, a health/energy sparkline, and counters (distance today, fights W/L,
matings, current activity). A yellow ring marks the selected agent.

New per-trial outputs: `condition_<trial>.csv` (condition time-series),
`dominance_<trial>.csv` (win/loss + David's score). The analysis summary adds
fight counts, final mass, mean health, and mean stress.

## Genetics & pharmacology (how to extend)

Add a gene in `biology.py::GENE_EFFECTS` as `status → [(trait, op, value), …]`:

```python
"OXTR": {"KO": [("sociability", "scale", 0.4)]},
```

Add a drug in `biology.py::DRUG_EFFECTS` as `dose → [(trait, op, value), …]`:

```python
def _methimazole(dose):  # ablates olfactory epithelium
    return [("smell_ability", "scale", max(0.0, 1.0 - dose))]
```

`op` is `set` | `scale` | `add`; effects are clamped to sane ranges. Genotype is
applied first (developmental), then treatment (acute).

---

## Also included

- **Energy & mortality** (`enable_mortality`) — energy tracks nutritional state;
  agents starve and die if food/water is removed or unreachable. Off by default.
- **Timed treatment onset** — `Treatment.day_offset > 0` delivers a drug *during*
  the run (e.g. anosmia induced on day 3); the effect switches on at that time.
- **Individual variation** (`individual_variation`) — per-agent multiplicative
  trait jitter so a cohort isn't identical clones.
- **Built-in analysis** (`core/analysis.py`) — after a run, derive daily social
  edgelists, per-agent space use (home-range area, path length), and an
  experiment-level `analysis_summary.csv` (mean F-F/M-M/F-M distances, network
  density, event counts). Exposed in the GUI ("analyse after run" + "Analyze
  existing project…").
- **Headless CLI** — `python -m fnt.abma config.json --out DIR --analyze`
  (also `--write-default`, `--trials`, `--days`, `--parallel`) for batch/cluster
  runs and reproducibility.
- **GUI niceties** — File menu (save/load/reset config), pre-run validation
  summary, live trial/day/ETA status, day–night tinting with fading trajectory
  trails, and auto-updating arena/population summaries.

---

## Reading a run back: why, not just where

A trajectory tells you where an animal went. The question an *in silico*
experiment exists to ask is why it went there — and the engine already knows,
because the policy blends a set of named, competing drives every step and then
sums them.

**The drive decomposition is kept** (`policy.py`). `perception["drives"]`
carries each component — home fidelity, memory, foraging, social attraction,
territory avoidance, exploration — as its own vector, alongside what the animal
was sensing when it decided. It costs no extra maths: the parts were always
computed, they were just being added into an accumulator and discarded.

**A replayable archive** (`core/record.py`). Every run writes
`data/record_<trial>.npz` beside its CSVs: per animal, per frame, its position
and heading, every drive magnitude, what it smelled, and its condition bars.
The archive ships its own field names and schema version, so a record written
today still opens after new fields are added — a reader looks columns up by
name. It is bounded by construction: when it fills it *decimates* (drops every
other frame, thinning from the newest backwards) rather than truncating, so it
always spans the whole run and simply gets coarser.

**File → Open Run Record…** reopens any finished trial and scrubs it through
the same view and inspection panel that drew it live. An experiment from last
week can be interrogated animal by animal without re-running it.

The CSVs are untouched by all of this — they remain the contract with FNT's
real UWB pipeline.

## The inspection column

The right-hand panel (`gui/science_panel.py`) is where a run becomes legible:

* **Roster** — one card per animal: sex colour, live health bar, what it is
  doing right now, and badges for anosmic / estrus / dead. Click to select it
  everywhere. Clicking an animal in the arena selects its card, and vice versa.
* **Why it is moving that way** — the selected animal's competing drives as
  bars on a shared scale. The longest bar is what is steering it. A drive this
  configuration cannot produce is never listed; one that is in play but happens
  to be zero keeps its row, so "this drive fell to zero" is visible rather than
  silent.
* **Traces** — the same drives, the condition bars, or what the animal senses,
  over the last stretch of the run, with a clickable legend.
* **Condition dynamics** — the `config.dynamics` table drawn as the graph it
  already is. Each row is literally an edge (`source` drives `target` with some
  gain); nodes fill with the live bar value and edge colour carries the sign.

**Territory map (🐾).** The scent field rendered under the arena — colour is
whose marks dominate each patch, opacity is how fresh. Territory is the thing
this model exists to produce and it used to be invisible until you opened a
CSV. Only rasterised while the toggle is on.

## Reproducibility: per-agent random streams

`core/rng.py` replaces the single shared generator with **counter-based streams
keyed by stable agent identity**: a draw is a pure function of
`(seed, agent uid, step, channel)`. Nothing is consumed, so

* agent 5's noise at step 900 is identical whether or not agent 3 exists,
* ablating one animal perturbs only that animal,
* a protocol event that adds an animal mid-run does not re-roll the residents.

This is what makes `seed_policy="paired"` actually paired. Measured on a
four-animal isolated cohort: under the old shared generator, adding one animal
mid-run displaced the untouched residents by up to **4.14 m**. It is now exactly
zero (`tests/abma/test_rng_streams.py`).

Dyadic events (contests, matings) draw from a per-pair stream for the same
reason, so the order in which pairs are resolved cannot change an outcome.

## The mechanistic nose

`core/olfaction.py`, off by default (`OlfactionParams.enabled`).

The scalar gate `recognition = smell_ability × identity_signal` reproduces the
headline result but can only return the assumption it was given. It cannot
express selective anosmia (methimazole ablates *epithelium*, it does not turn a
global gain knob), graded confusability, or detecting a mark without
identifying it. Those are exactly what a habituation–dishabituation assay
measures.

Three layers replace it:

* **Emission** — each animal emits an odour profile over `n_channels` chemical
  channels. `identity_signal` is *distinctiveness*: the emitted profile is a
  blend between the animal's private profile and the flat population average.
  At 0 — a MUP knockout — everyone emits the average, so marks are real but
  carry no individual information. That falls out of the geometry.
* **Reception** — anosmia is channel loss. At `smell_ability = g` an animal
  keeps a `g` fraction of its receptor mass, and *which* channels survive comes
  from that animal's own stream. The construction is exact, not sampled, so
  mean receptor gain equals `smell_ability` for every animal at every dose.
* **Readout** — detection (how much odour is captured at all) and separability
  (how far a target's perceived profile sits from the nearest other animal's)
  are computed separately, then combined through a psychometric function.

The design constraint was that it must **reduce to the scalar model in the
mean**, so the previously validated dose-response survives and the only new
content is structure. It does, to RMSE 0.033:

| condition | scalar gate | mechanistic |
|---|---|---|
| intact WT | 1.00 | 1.000 |
| methimazole 0.25 | 0.75 | 0.719 |
| methimazole 0.50 | 0.50 | 0.462 |
| methimazole 0.75 | 0.25 | 0.190 |
| methimazole 1.00 | 0.00 | 0.000 |
| MUP HET | 0.50 | 0.524 |
| MUP KO | 0.00 | 0.000 |

The residual is a prediction, not an error: losing half your receptors costs
more than half your discrimination, because discrimination needs the channels
that carry the distinguishing information. What the scalar model could not say
at all: at one uniform dose, animals differ in *which* cage-mates they can still
tell apart, and a MUP-KO cohort is detected (0.9+) while being unidentifiable
(<1e-6).

`ablation_selectivity=0` recovers the old uniform-gain behaviour exactly, which
is how the two are compared.

## Running the anosmia experiment

`study.anosmia_study()` builds the design ABMA was written for:

```python
from fnt.abma.core.study import anosmia_study, run_study

study = anosmia_study(doses=(0.0, 0.5, 0.75, 1.0), replicates=4, days=3.0)
run_study(study, "/path/to/studies/anosmia")
```

Nothing in the config says how far apart animals should sit, so mean male–male
spacing is a *result*. Measured (8 animals, 2 days, 2 replicates, paired seeds,
mechanistic nose):

| metric | saline | methimazole 1.0 | Cohen's d |
|---|---|---|---|
| mean male–male distance | 0.62 m | 0.15 m | −8.7 |
| mean home-range area | 1.15 m² | 0.41 m² | −4.9 |
| fights | 71 | 129 | +6.4 |
| mean female–male distance | 0.40 m | 0.16 m | −6.0 |

Territorial spacing collapses, home ranges shrink, and contests rise — animals
that cannot read a boundary blunder into each other. None of that was
prescribed.

## Describing a run, and watching it

A whole experiment is usually one sentence — "VoleTerra, six males and six
females, five days" — but turning that into a config used to mean knowing which
preset factory to call, that a species card is stamped onto a group rather than
typed into it, and that mass is a distribution spec rather than a number.
`core/compose.py` is that sentence:

```python
from fnt.abma.core.compose import design, summary

cfg = design(preset="voleterra", males=6, females=6, days=5)
print(summary(cfg))
```

Preset and species names match loosely ("voleterra", "prairie"), and an
ambiguous fragment is an error rather than a silent pick — "vole" matches three
species cards and guessing which one an experiment meant is not the composer's
job. Bodies come from the species card and personality from its suggested
spread, so founders vary the way a cohort does. Home-range size is never set:
it is what the run produces.

Anything left unspecified keeps the preset's value, so `design(preset=…)` alone
is just the preset. `check()` refuses a design that cannot produce a meaningful
run (no animals, zero days, a sampling interval finer than the timestep) before
anything reaches disk.

### From the command line

```bash
# what would this run?
python -m fnt.abma --preset voleterra --males 6 --females 6 --days 5 --dry-run

# run it headless
python -m fnt.abma --preset voleterra --males 6 --females 6 --days 5 \
    --out ~/ABMA_Projects --analyze

# open the window and run it on screen, no clicking
python -m fnt.abma --preset voleterra --males 6 --females 6 --days 5 \
    --out ~/ABMA_Projects --watch

python -m fnt.abma --list-presets
```

`--watch` (`gui/launch.py`) builds the real `ABMAWindow` — same views, same
inspection column — loads the design, switches on the territory map, selects an
animal so the drive panel has something to show from the first frame, and
starts the run once the window is actually on screen. It is *not* a simplified
preview: what you watch is the run being written to disk, and closing the
window leaves a normal project folder behind. `--no-autostart` opens it loaded
but idle, for reviewing a design before committing the compute.

`--save-config PATH` writes the composed config, so a design that looked right
on screen can be re-run headless, checked into version control, or handed to a
study as its base.

A 5-day, 12-animal VoleTerra run is ~216,000 steps and takes about **3 minutes**
— fast enough to watch end to end.

## The sward: an environment they wear down and share

`core/sward.py`, on by default in the VoleTerra preset.

Everything else an ABMA animal changes, it changes for itself — its energy, its
marks, its memory of home. Grass is the first part of the world that is
genuinely **common**. An animal that walks a route flattens it, and the
flattened route is then cheaper and faster for *every* animal, including ones
that never contributed. Runway networks are one of the most conspicuous things
voles actually do in the field, and they are not explicable from an individual
optimum.

Three couplings:

* **Height slows movement.** The speed multiplier is anchored at
  `speed_ref_cm` (factor 1) and runs from 1.35× on a worn trail down to 0.45×
  in a full-height sward.
* **Height costs energy.** Pushing through grass is charged per metre *and* per
  centimetre, on top of flat-ground locomotion. A trail is cheaper as well as
  quicker, which is what makes maintaining one rational.
* **Movement and clipping lower it.** Walking wears it slowly; clipping is 4×
  faster — but the animal must **stop**, for a bout proportional to the sward
  height. Clipping yields **no energy**: the cost of trail maintenance is the
  foraging time it displaces and nothing else. (Real voles do eat what they
  clip; separating the two is a deliberate choice so trail-building and feeding
  can be told apart in a result.)

Regrowth is what makes it a *maintenance* problem rather than a one-off
demolition — without it the first week flattens the enclosure permanently and
there is no ongoing decision to study. Growth scales with the season.

The clipping decision reads the scent field: an animal clips where the grass is
worth cutting, nothing more urgent is pressing, **and its own marks are already
on the ground**. So trails form along routes an animal already uses, and belong
to settled animals — a cohort still scrambling for food never builds any.

Measured on VoleTerra, 12 animals, 5 days, summer: **11.6% of the enclosure
worn into runway**, whole-enclosure mean sward steady at 7.9 cm (regrowth
balancing wear while local trails deepen), and clipping effort ranging from 1.5
to 28 minutes per animal per day — some animals invest in trails and some free-ride.

Watch it with the 🌱 toolbar toggle; dark ground is worn runway.

## Sun, moon and season

`core/sky.py`. The original day/night was a switch on the clock — right for a
light-cycle room, wrong for a field enclosure, where "does this cohort space
itself differently in December than in June" needs day length and twilight to
be produced rather than typed in.

Given a latitude, longitude and date it computes solar elevation and azimuth
(standard NOAA algorithm), moon position and phase (a truncated lunar series),
day length, and a smooth twilight ramp so a crepuscular animal gets a dawn and
a dusk instead of a step change. Moonlight suppresses night activity, which is
one of the better-described behaviours in nocturnal small mammals.

The VoleTerra preset turns the sky on at a synthetic mid-latitude reference
(40° N, 0° E, UTC+0); set `sky.latitude`, `sky.longitude` and
`sky.timezone_hours` to put a run at your own site. Verified against known
geometry at that latitude: solar noon elevation 73.4° at the summer solstice
and 26.5° at the winter one, due south at noon, day length 14.9 h vs 9.2 h.

Season is how you say all of that at once:

```bash
python -m fnt.abma --preset voleterra --males 6 --females 6 --days 5 \
    --season winter --watch
```

Choosing a season sets the release date, and at a real latitude day length,
solar angle and grass growth then all change together — which is the point of
asking for a season rather than a date.

## Release: where animals are put down

`ExperimentConfig.release_mode` — `auto` (default), `point`, `scatter`, `nests`.

A fixed 15 cm scatter was fine in a 2.2 m cage and disastrous in a 22.9 m
enclosure: twelve animals set down inside one body-length of each other fuse
into a clump that social attraction then holds together, and no territory ever
forms. `auto` keeps the point release for arenas under 3 m and disperses above
it. Measured: mean release spacing 0.18 m in the cage (unchanged), 10.8 m in
VoleTerra.

## Scale, and the two bugs it exposed

Running the cage-tuned model in a 523 m² enclosure broke it in ways that were
silent rather than loud. Both fixes are in the **Field enclosure** physiology
preset and the VoleTerra scent settings:

* **A scent mark is a droplet, not a void.** `mark_volume_ul` was 22 µL — a
  full urination. A 0.54 mL bladder therefore bought 25 marks, and marking was
  hard-limited by hydration. At 4 µL a full bladder buys ~135, which is the
  order a scent-marking rodent actually deposits.
* **The scent grid has to scale with the enclosure.** A 10 cm cell is right for
  a cage (484 cells); across 75 ft it is 52,000 cells, and a cohort's entire
  daily marking budget covered about 1% of them — so no territory could form
  however well the animals behaved. Half-metre cells put it back within reach.

Together these took emergent territory from **0.2 m² to 89 m²** over five days
and moved the cohort from "dead by day six" to a stable energy plateau at full
health. Note what the failure looked like: nothing errored, and the mechanism
the enclosure model rests on had simply switched itself off.

## Making them interact — two more silent failures

Once the cohort stopped starving and started holding territory, it stopped
*interacting*: 3 fights and stress 0.2 over five days, and the social network
went sparse. Two independent causes, both silent, both found by instrumenting
the funnel rather than by turning knobs.

**Contests were structurally impossible for settled animals.** A fight required
at least one animal to be ROAMING. Once animals settle successfully they rest
~84% of the time, and only **2.1% of 4,345 same-sex contacts** had anyone
roaming — so a resident could never defend its patch against an intruder
walking through it, and `dominance_<trial>.csv` had nothing in it. The rule was
trying to stop affiliative huddling from registering as aggression, which is
right, but a huddle is *both* animals settled together — so that is now what it
excludes. Fights went **0 → 48 per three days**.

**`perception_r` was the lever, not `k_social`.** Social attraction only acts
within the perception radius; at 0.6 m — a quarter of a 2.2 m cage — it is
effectively blind across 523 m², and opposite-sex pairs never touched at all.
Swept over 3-day runs (one trial per cell, so counts bounce):

| `perception_r` | fights | matings | territory |
|---|---|---|---|
| 0.6 m (cage default) | 48 | **0** | 66 m² |
| 1.2 m | 31 | 49 | 55 m² |
| **2.0 m (VoleTerra)** | **63** | **55** | **57 m²** |
| 3.0 m | 86 | 79 | 48 m² |

Raising `k_social` instead reached only 16–17 matings even at 2.5–4.0× while
costing territory and condition: it scales a force that almost never activates.
Widening the radius that force acts within is the honest lever, so VoleTerra
sets `perception_r = 2.0` and leaves the gain alone. The cage default is
unchanged.

`tests/abma/test_interaction.py` now pins floors — contests happen, matings
happen, a hierarchy can form, and territory survives all of it. Nothing had
asserted that any of those were *reachable*, which is exactly how both bugs
reached a finished run without going red.

## Reading the animals: tracks, not dot clouds

Both views used to draw recent history as a **scatter** — the last 10 frames
(3D) or 8 (2D), pooled across the whole cohort into one item. Around each animal
that reads as a smear of identical dots: no direction, no usable history, and
nothing telling you which mark is *now*.

Each animal now gets its own **fading polyline**: a line strip whose tail
dissolves toward transparent and whose head is the animal's own colour at full
opacity, plus a bright dot on the current position so the head of the track is
unambiguous even when the body mesh is small on screen. A line has direction; a
cloud of dots does not.

The 〰 toolbar button cycles track length — off / 100 / **1000** / 5000 recorded
steps. Hiding tracks does not forget them, so toggling is free.

## Provenance

`core/provenance.py` writes `provenance.json` into every run folder:

* **A SHA-256 of the exact config**, so a run folder can prove what produced it
  and `verify()` detects a config edited after the fact.
* **A source class per parameter**, from `project.SOURCES` — `measured`,
  `literature`, `estimated`, `free`, `default` — plus the free parameters
  listed by name. For the vole preset that is 21 free parameters out of 112.

The point is not to make free parameters go away; a behaviour model needs some.
It is that a reader can see, from the run folder alone, how much of a result
rests on them.

## Learned brains: animals that are not told what to do

The rule-based animal is steered: when it is hungry it is handed the bearing
of the nearest food and walks there. That is fine for asking where animals end
up, and useless for asking *how* — take its nose away and it finds food exactly
as fast as before (`--assay approach` shows this), because the nose was never
involved.

A **learned brain** replaces the rules with a small recurrent neural network
that receives only what an animal standing at that spot could sense, and
outputs only a turn and a speed. Nobody writes its behaviour. It is raised by
reinforcement in randomised worlds, where the single thing scored is the
body's own discomfort.

```
 senses (55 channels, egocentric)                       motor
 ─────────────────────────────────                      ─────
 smell: food, water, own marks, others'  ┐
   marks, male, female, estrous          │   sensory     recurrent
   (strength + ahead/behind + left/right)├─► layer  ──►  layer    ──► turn
 whiskers: wall ahead / left / right     │   (64)        (64 GRU)     speed
 others nearby, taste, grass underfoot   │     ▲
 place memory: last food, last water     │     │ gain
 the body: hunger, thirst, bladder,      ┘     │
   stress, pain, light, time of day ───────────┘  neuromodulators
```

* **Senses** (`core/senses.py`) are egocentric: food is "ahead and to the
  left", never "north-east". Every olfactory channel is scaled by the animal's
  `smell_ability`, so anosmia removes exactly those inputs and nothing else.
  There is no map and no bearing-to-food; a smell is a strength and a
  direction of increase that fades with distance.
* **Neuromodulators.** Hunger, thirst, stress, pain, receptivity, sex, light
  and the personality traits do not only arrive as inputs; they set the *gain*
  of each sensory unit, the way a hormone changes how strongly a circuit
  responds without carrying any message of its own. Each unit starts with its
  own random sensitivity to each modulator; which sensitivity ends up used for
  what is learned.
* **The brain owns speed.** With a learned brain the engine no longer decides
  when an animal rests or forages. Physiology still applies: a starving animal
  is slower whatever its brain asks for.
* **A brain is a file** (`.npz`, in `fnt/abma/brains/` or `~/ABMA_brains/`) and
  runs in plain numpy. PyTorch is needed only to train one. Sensors are matched
  by name, so a brain trained before a sense existed still loads, and one that
  needs a sense the body has lost refuses to.

### How a brain is raised

```bash
python -m fnt.abma.learn --name my_vole --minutes 20
python -m fnt.abma.learn --name my_vole --resume my_vole --minutes 10
```

Training (`learn/`) runs the **real engine** — the same `Simulation`, scent
marks, contests and walls an experiment uses — in hundreds of small randomised
worlds (the *nursery*): different enclosure sizes, food and water in different
places, different numbers of companions, animals arriving at every level of
hunger and thirst. The reward is the drop in

    discomfort = (hunger/100)² + (thirst/100)² + ½(stress/100)² + 2(injury)²

plus a small bonus for mating. Nothing rewards approaching food, following a
smell, or avoiding a rival. Eating when hungry lowers discomfort; that is the
whole lesson, and the animal has to work out the rest.

The log prints, per iteration, how much time hungry and sated animals spend at
food, thirsty and sated animals at water, and `right`: among animals with one
need much stronger than the other, in a world offering both, the share of
their time at a resource spent at the one they needed (0.5 is chance).

Two things about the nursery are there because training failed without them:

* **Some worlds hold only food, or only water.** When every world had both,
  animals learned to find food, sat on it, and from there never met water
  often enough to learn what it was for — time at water stayed at chance for
  eight million animal-ticks, while a world with nothing but water teaches it
  in about one million.
* **Needs must have a say from birth.** With neuromodulator weights starting
  at zero the gains barely moved in training, and a thirsty animal chose food
  as often as a hungry one. Random initial sensitivities fixed that.

Four more are there so that a brain can live in an enclosure with structures
in it, and can lose its nose without losing everything:

* **Walled boxes with doorways** (`--boxes`, default half the worlds that have
  food). Food is inside one to four boxes — in a corner, against a wall or
  free-standing — each with one or two doorways and sometimes water. A brain
  raised only in open worlds meets its first wall in an experiment.
* **Animals with no nose** (`--anosmic`, default a quarter; a tenth more with
  a weak one). While every animal can smell, the smell of food always leads
  to the doorway and nothing else about finding one is ever learned.
* **Starts at a doorway, and at the edge of a resource** (`--born-at-door`,
  `--born-at-resource`). A nose-less animal wandering the open floor almost
  never comes across a 7 cm doorway; putting some down beside one makes "an
  opening in a wall is worth going through" a common experience, and what is
  learned there spreads outward to "walls are worth following". Likewise an
  animal put down at the edge of a water source's reach, with no use for
  water, can find out what lies beyond it.
* **Water towers** (`--towers`). Some open water is a solid cylinder to drink
  beside rather than a patch to walk over. To an animal's whiskers a tower is
  a wall.

### Putting a brain in an experiment

In the window: **Run tab → Nervous system → Learned brain**. From the command
line, add `--brain NAME` to any run:

```bash
python -m fnt.abma --list-brains
python -m fnt.abma --preset anosmia --brain vole_forager --days 2 --watch
```

Everything else — arenas, protocol events, treatments, the run record, the
UWB-schema output — is unchanged. With a learned brain the inspection column
shows **what is reaching its brain** (smell, touch, taste, memory, body state)
in place of the drive bars, and two more pages: **Brain activity**, a raster of
every unit over time for the selected animal, and **Brain output**, the turn
and speed it asked for.

### Showing a run to someone else: `--movie`

```bash
# the first half hour of a run, as a GIF
python -m fnt.abma --preset anosmia --males 3 --females 3 --brain vole_forager --movie

# side by side: the cohort as described, and the same animals on methimazole,
# released hungry and thirsty so foraging starts at once
python -m fnt.abma --preset anosmia --males 3 --females 3 --brain vole_forager \
    --movie --movie-hours 0.33 --movie-deprived 60 --movie-vs-mmz
```

`core/movie.py` films trial 0 of the run — the enclosure from above, each
animal (males blue, females red) with a fading trail, and each animal's hunger
and thirst underneath — and writes `<out>/<name>/movie.gif` (`--movie mp4`
with ffmpeg installed). The film is about 300 frames whatever its length, so a
longer one plays faster rather than growing. It works for the rule-based
animal too; leave `--brain` off.

### A whole comparison in one line

The intact-versus-methimazole study takes the same description an ordinary run
does, and a brain:

```bash
# 3 males + 3 females in the VoleCosm, 5 days; 2 replicates intact, 2 on methimazole
python -m fnt.abma --preset volecosm --males 3 --females 3 --days 5 \
    --anosmia-study --doses 0,1 --replicates 2 --brain vole_forager --name mmz_3m3f
```

Add `--dry-run` to see what would run. Replicates are paired: replicate *r*
starts from the same release and the same founders in both arms. Results land
in `<out>/mmz_3m3f/results/` (`metrics_long.csv` for R, `comparison.csv`).

### Bench assays: does it behave like an animal?

Before trusting a brain in an enclosure, test it the way you would a new line:

```bash
python -m fnt.abma --assay all --brain vole_forager     # a learned brain
python -m fnt.abma --assay all                          # the rule-based animal
python -m fnt.abma --preset anosmia --brain vole_forager --days 2 --assay vitals
```

| Assay | What it asks |
|-------|--------------|
| `approach` | One animal, one food source out of reach. Run hungry, sated and hungry-but-anosmic: does motivation gate it, and does it need its nose? |
| `choice` | Food one side, water the other, the animal midway. Hungry vs thirsty: does a need pick its own goal? |
| `search` | It eats for 40 s and the food is removed. Does it stay and search where food was (area-restricted search), compared with the same spot never baited? |
| `doorway` | The chow is in a walled box in a corner, behind two small doorways. With and without a nose: does it get in, and how long does it take? |
| `vitals` | Your config, your cohort: does everyone eat and drink, how much, how far do they travel, what do they do to each other? |

#### What the shipped brain does (`vole_forager`, 22M animal-ticks, ~25 min of training)

Forty animals per condition; the rule-based animal alongside.

| | rule-based | `vole_forager` |
|---|---|---|
| **Find the food**, hungry | 100% in 16 s, path directness 0.93 | 100% in 34 s, directness 0.81 |
| … hungry, **no sense of smell** | 100% in 16 s, directness 0.93 — *unchanged* | **55% in 224 s, directness 0.21** |
| … sated | 28% | 100% — it goes and stays anyway (see below) |
| **Food or water**, hungry → food first | 100% | 100% |
| … thirsty → water first | 100% | 100% |
| … need of only 20/100 | — | 88% (hungry), 72% (thirsty) |
| … no need at all | — | 68% food, 32% water |
| **After food is lost**, distance from the spot 3 min later (vs never fed there) | 0.91 m vs 0.97 m | **1.26 m vs 1.94 m**, turning more (0.52 vs 0.45 rad per 2 s) |
| **Vole cage, 8 animals, 2 days**: lowest energy / highest thirst anyone reached | 34 / 46 | **84 / 27** |
| … food and water per animal per day | 7.3 g, 1.9 mL | 10.7 g, 4.3 mL |
| … fights and matings per animal per day | 15.6, 30.5 | 23.9, 40.8 |

What that says:

* **It smells its way to food.** Anosmia halves how often it gets there and
  turns a direct path into a wander. The rule-based animal cannot show this at
  all. This is the result the anosmia experiments needed.
* **A need picks its own goal, and by degree.** Nothing told it water answers
  thirst; a slightly thirsty animal usually goes to water, a very thirsty one
  always does.
* **It lingers where food was** (area-restricted search), where the rule-based
  animal walks on.
* **A cohort keeps itself fed and watered** in an enclosure it was not raised
  in, better than the rule-based cohort does.
* **In the VoleCosm, where food is behind doorways, the nose matters far
  more.** Released hungry, all six intact animals are inside a box within five
  minutes and fed within thirty; of six on methimazole, none finds a doorway
  in the first half hour (they drink at the central tower). Over two days the
  anosmic cohort does get in and everyone eats — lowest energy 64 against 94.
  This brain was never raised with walls or doorways, so how it finds a
  doorway without smell is chance, not skill; `vole_doorways` (below) was
  raised with them.

And what it does *not* show, which matters as much:

* **Hunger does not decide whether it visits a lone food source.** A sated
  animal still walks there and camps. Nothing in its world makes anywhere else
  better, so sitting on the food is what the reward taught.
* **It lives on the line between water and food.** In the cage the animals
  shuttle between the two water sources through the feeder and never use a
  nest. They rest 15% of the time (rule-based: 61%). In the VoleCosm, where a
  box holds both, they walk into one and stay: 99% of day one inside a box.
* **Its social behaviour is not to be trusted.** Fights and matings are
  saturated in the small cage under *both* nervous systems (the engine lets
  every dyad contest once an hour), and a brain that gathers animals at food
  and water makes it worse. It has not learned to keep away from a rival:
  half-hour nursery episodes never let stress build far enough to matter.

#### A second brain: `vole_doorways`

`vole_forager` raised on for another 28M animal-ticks (about 40 minutes) in
worlds with boxes, doorways and nose-less animals: first mixed, then a "door
school" (every food world boxed, most animals nose-less, a third starting at a
doorway), then a mixed, social consolidation. Same senses, same network; no
sense was added.

| | `vole_forager` | `vole_doorways` |
|---|---|---|
| **Find the box**, with a nose | 100% in 41 s | 100% in 42 s |
| … **without a nose** | 32% in 370 s | **95% in 214 s** |
| Find food in the open, with a nose | 100% in 34 s | 100% in 36 s |
| … without a nose | 55% in 224 s | **5%** |
| Food or water, by need | 100% / 100% | 100% / 100% |
| After food is lost, distance 3 min later (vs never fed) | 1.26 vs 1.94 m | 0.06 vs 2.06 m |
| **VoleCosm, methimazole, released by a wall**: in a box within 10 min | 0 of 24 | **24 of 24** (median 3.4 min) |
| **VoleCosm, methimazole, released at the centre** (the default): in a box within 60 min | 13 of 24 | **3 of 24** |
| Open vole cage, methimazole, one day: lowest energy reached | 15 | 53 |

What it learned is to follow walls when it cannot smell: it goes to the
room's wall, runs along it, and falls into the first doorway. That takes about
three and a half minutes wherever it starts, which is why a nose is still six
times quicker. With a nose nothing changed.

Two things it got worse at, both consequences of the same strategy:

* **Food lying in the open, without a nose: 5%.** A wall-follower never
  crosses the middle of the room. The open-world brain wandered and found it
  about half the time.
* **It cannot leave a water tower.** Beside a tower a nose-less animal keeps
  turning on the spot and pushing at the cylinder — the behaviour it uses at
  any resource — and never steps out of the water's reach, so training never
  shows it that leaving pays. Three targeted stages (animals started at
  water, towers in the worlds, starts at the edge of reach, more exploration
  noise) did not move this. It matters because **VoleCosm releases its
  animals in the middle, beside the tower**: from there the methimazole
  cohort mostly stays put for hours. Released anywhere else, all of them are
  in a box within ten minutes. The gap is recorded as an expected failure in
  `tests/abma/test_shipped_brain.py`.

So for a VoleCosm comparison with this brain, either release the animals away
from the tower (`release_mode="scatter"`) or treat the first hours of the
methimazole arm as dominated by that artefact. The underlying limit is the
sensorium, not the training: a nose-less animal here is also blind, with three
whisker rays. A real anosmic vole sees the boxes.

`vole_forager` stays the default; ask for this one with `--brain
vole_doorways`.

### Twin experiments: one life, lived twice

Every random draw an animal makes is a pure function of (seed, animal, step),
so a world can be copied mid-run and both copies will live the *same* future
to the last decimal — unless something is changed in one of them. That makes
a counterfactual exact: run to a chosen moment, fork, change one thing in the
copy, and whatever differs afterwards was caused by that change.

```bash
# animal 0 is 30 points less hungry at hour 6; follow both lives for 6 h
python -m fnt.abma --preset anosmia --twin "hunger-30@0"

# all males lose their sense of smell
python -m fnt.abma --preset anosmia --twin "smell_ability=0@males"

# drive one unit of animal 0's brain hard for 10 seconds (needs --brain)
python -m fnt.abma --preset anosmia --brain vole_forager --twin "rnn:12=1@0~10"

# silence a whole sense for ten minutes
python -m fnt.abma --preset anosmia --brain vole_forager --twin "olfaction*0@0~600"
```

In the window: **File → Twin Experiment**, or the ⑂ button in the Nervous
system box. The result (`twin.npz`, `twin.png`) gives the two paths (black =
as it was, red dashed = with the change), when the lives first parted, how far
apart they ended, and how many *other* animals' lives changed because of it.
A twin with no change diverges by exactly 0.0 m; that is tested.

Two examples on the shipped brain, eight voles, forked six hours in. Making
one animal 40 points hungrier: it is 0.9 m from where it would have been a
minute later, has eaten its way back to its old hunger within half an hour,
and is still living a different life six hours on — as are all seven of its
cage-mates, to whom nothing was done. Driving one unit (36, one synapse from
the turn output) for ten seconds: the two lives part within 8 s and are 1.4 m
apart two hours later; again all seven others end up somewhere else.

`python -m fnt.abma --list-brains` prints, for each brain, the units wired
most directly to turning and to speed — the first ones to try. Hovering a row
of the Brain activity raster names the same thing for that unit.

Unit stimulation can also be scheduled inside an ordinary experiment
(`ExperimentConfig.stimulations`: when, how long, which animals, which unit,
set / add / scale) — the in-silico version of an optogenetics protocol.

### What a learned brain is not (yet)

* It is a **phenomenological** brain, not a model of vole neuroanatomy. Units
  are not cell types. What it offers is a closed sensory → neural → motor loop
  whose behaviour was not written by hand, which can be lesioned, driven and
  recorded completely.
* It was raised in half-hour episodes. Nothing in the body makes rest pay, so
  it has no sleep and no daily rhythm of its own beyond what light and the
  clock inputs teach it. The same short episodes are why it has learned
  nothing about rivals (above).
* One diet. Protein, carbohydrate and lipid as separate needs, predators, and
  a PufferLib adapter for much faster training are not built.

## Contest physics: the means to fight, chase and get away

Off by default (`ExperimentConfig.contest`, `ContestParams`). Until now a
same-sex contest was a dice roll the engine made when two animals touched: the
engine decided there was an attack, picked a winner and moved the loser twelve
centimetres. Nobody chose to fight, nobody was chased, cornered or escaped, so
the size of the enclosure could only matter through how often animals bumped
into each other. That is the wrong place for the engine to be making decisions
if the question is why the same animals form a hierarchy in a small arena and
territories in a large one.

With `contest.enabled` the dice roll is gone and nothing replaces it but a
body:

* **A bite is something an animal does.** A learned brain raised with it has a
  third motor output beside turn and speed. A bite lands on the nearest animal
  within reach and inside a cone ahead of the biter. It costs the biter energy
  and a refractory moment; it costs the bitten animal health, in proportion to
  the biter's mass and condition, and it stings (`hurt`, with a direction).
  Bites are written to the event table (`bite`: actor, target, damage).
* **Sprinting and fatigue.** A brain's speed output now spans rest to
  `sprint_factor` times the easy pace. Fatigue fills above that pace, drains
  below it, and takes the sprint away as it fills. A chase therefore ends, and
  who gives up first is an outcome. Fatigue is felt, and counts (mildly) in
  the discomfort a brain is raised to reduce, which gives rest a value.
* **Cover.** A hut shields its occupant from bites from outside, and holds
  only `refuge_capacity` animals (one by default): a full hut cannot be
  entered, so its occupant cannot simply be followed in. Two animals under one
  hut can bite each other. A wall or a pole between two animals stops a bite.
* **Death** from injury, when `enable_mortality` is on.

New senses, silent unless contest physics is on: `hurt` (+ where from),
`fatigue`, `in_cover`, the nearest hut (`cover` + bearing), whether the
nearest animal is coming at it (`other_closing`) and whether it is bigger or
smaller (`other_size`). `hurt` and `fatigue` also act as neuromodulators.

Nothing rewards a bite, a retreat or a win. Whether anyone attacks, flees,
hides or holds its ground has to come from what those acts do to the body.

```bash
# raise a brain in worlds where animals can hurt each other
python -m fnt.abma.learn --name my_mouse --contest --minutes 60
# an older brain can be raised on: it starts out exactly as it was, not biting
python -m fnt.abma.learn --name forager_plus --resume vole_forager --contest
```

Brains raised without contest physics keep working everywhere: they have no
bite output and never bite. The rule-based animal does not bite either, so
with contest physics on it does not fight at all; it is a foraging baseline
there, nothing more.

`python -m fnt.abma --assay chase` checks the body itself, with behaviour
held fixed: a scripted chaser and a scripted fleer with identical bodies, in
square arenas from 0.6 to 38 m, on an open floor and with one hut to reach
(`--structure-out table.csv` writes the table). Measured with the default
values: on an open floor the fleer takes about 29 bites a minute at 0.6 m, 2.6
at 1.2 m, almost none at 2.4 m and none beyond; with a hut it takes under one a
minute at every size. So for equal bodies a chase by itself stops mattering at
a couple of metres, and one refuge nearly removes its cost. Whatever makes a
small arena different has to come from what an animal must come back for.

Known limits. A bite is resolved along the paths of the step, so a coarse
step does not let animals run through each other, but a brain still decides
only once per tick (2 s by default), which is slow for a fight. How far an animal notices another is still
`policy.perception_r`, set by hand per arena. Animals cannot yet tell one
individual from another through the brain's senses, and nothing is limited
enough to be worth excluding a rival from except a receptive female.

## A brain wired from anatomy

`fnt/abma/core/anatomy.py` builds a network from an *anatomy spec* instead of
two anonymous layers: one block of units per brain area, connected only where
the spec lists a projection, with the sign the spec gives it. The first spec is
`fnt/abma/brains/anatomy/c57_male.json`, a reduced male C57BL/6J house mouse
(44 areas, 311 units, 114 projections, 7 neuromodulators). A spec is plain
JSON:

- `regions`: each area's id, name, class (sense organ, sensory area, internal
  area, motor area, muscle) and number of units.
- `projections`: `[from, to, sign, strength 1-3, [evidence keys]]`. The sign is
  `+`, `-` or `±` (mixed cell types: each source cell is one or the other).
- `modulators`: where each neuromodulator is released and a receptor level
  0-3 per area. This table is where strains and species are meant to differ.
- `entry`: which engine sensor channels enter at which sense organ or area.
- `evidence`: what each projection or area rests on, either a paper (with its
  PubMed id) or an idea (with its author and date). A projection with no
  evidence is one that still needs verifying.

**It is untrained.** The weights are random draws inside the wiring
(`--anatomy-seed` picks the draw), so the behaviour means nothing yet: a given
draw usually turns one way, because its two turn units are not balanced. What
is real is the route activity takes, and what changes when one projection or
one receptor level is changed. Training it is the open problem.

How it meets the body: every engine sensor channel enters at the organ the
spec names (wall contact at the whiskers, hunger at the viscera, every odour
the engine has at the main olfactory epithelium). The hand-set disposition
channels (`aggression`, `boldness`, `sociability`, `exploration`) are not
wired at all; this brain is meant to replace them. The outputs are muscles:
neck and trunk (a left and a right unit, read as their difference), limbs
(speed), jaw (bite), and, for urination, the bladder wall and the urethral
sphincter.

**Urine and the nose.** When scent marking is on and a policy hands the engine
those two muscles, urination is no longer drawn at random: urine leaves when
pressure in the bladder (the wall contracting, plus stretch from filling)
beats the closure of the urethra (the sphincter, plus a passive resistance),
and a void that has started runs until the bladder is empty (`ScentParams`:
`void_stretch`, `void_resistance`, `void_flow_s`). Urine lands as *spots*
(`sim.urine_spots`), finer than the scent grid, each carrying its owner. Every
animal has a MUP barcode (`biology.mup_barcode`, 12 isoforms): its own
combination for a wild-type animal, one shared barcode for genotype
`MUP: INBRED` (an inbred strain such as C57BL/6), none for `MUP: KO`. Thirteen
sensor channels (`vno_contact`, `vno_mup_0..11`) carry the barcode of whatever
the nose is on, a urine spot within 2.5 cm or another animal's body, and
nothing otherwise; they enter this brain at the vomeronasal organ. Volatile
odour at a distance (`olf_*`) still enters at the main olfactory epithelium.
Nothing tells the animal whose barcode it is reading.

```
python -m fnt.abma --list-anatomies
python -m fnt.abma --anatomy c57_male --brain-recording 5
python -m fnt.abma --anatomy c57_male --anatomy-seed 3 --strength "SC>TURN=0" --receptor "OT@LS=3" --brain-recording 5
python -m fnt.abma --anatomy c57_male --watch
python -m fnt.abma --preset volecosm --anatomy c57_male --movie
```

`--anatomy` works wherever `--brain` does, because both go through
`make_policy(cfg)`: a headless run, the live window, a movie. With no
`--preset` it uses the open field test. `--strength A>B=X` multiplies one
projection (0 cuts it) and `--receptor MOD@AREA=N` sets one receptor level;
both are applied on top of the same draw, so two runs that differ only in an
edit are otherwise identical cell for cell. `--brain-recording MINUTES` runs
one trial and keeps every unit at every engine step
(`brain_recording.npz`: positions, heading, muscle commands, and each unit's
firing rate as a fraction of its maximum) with a table of mean firing per area
(`brain_areas.csv`). The ordinary run record also carries the activity, at its
own coarser sampling, under the same names a learned brain uses, so the
inspection column's activity strip shows it.

### The brain map: `--brain-map`

```
python -m fnt.abma --brain-map
python -m fnt.abma --anatomy c57_male --anatomy-seed 3 --brain-map browser
```

Opens a page that draws the anatomy (as ribbons, as a left-to-right network or
as a free network of model neurons), with the animal run live by the engine:
every model neuron lights with its firing, a click on a neuron traces its
firing rate and lists what it listens to and talks to, a click on an area
lists its inputs and outputs with the evidence for each, and `-` / `+` on a
projection or a receptor level changes the running brain. A jointed body walks
the open field beside it.

The page is only the display (`fnt/abma/gui/brain_map_page/`, built by
`core/brainmap.py`). The engine side is `gui/brain_map.py`: a bridge object
sends one frame per engine step and takes the controls back, so the one network
that runs is the Python one. The page appears **inside an FNT window** when
Qt's web engine is installed (`pip install fieldneurotoolbox[brainmap]`, about
130 MB), and **in your own browser** otherwise, talking to the engine over a
WebSocket on this computer only; `--brain-map window` or `--brain-map browser`
chooses. Nothing is fetched from the internet: the 3D library (three.js, MIT)
ships with FNT. What you change or verify on the page is kept in
`~/ABMA_brains/anatomy/<name>.notes.json`; the spec is never rewritten.

The map follows the first animal of whatever design it is given, in that
design's arena (drawn as its outline; the other animals as plain bodies). The
🧠 button in the ABMA window opens it on the current design, as its own run.

**Replay.** `--brain-recording MINUTES` also writes `brain_recording.html`:
the map with that run packed into it, which the page plays back by itself
(play, speed, a slider through the run). It is one file and needs neither FNT
nor the internet, so a run made on one computer can be watched on another.
`--brain-replay FILE.npz` opens an existing recording the same way.

Known limits: only the first animal's brain is shown; arena furniture (zones,
huts, food) is not drawn; Qt 5's web engine is an old browser and redraws about
15-30 times a second, less smoothly than a current browser.

## Roadmap / not yet done

1. **Absorbing boundary** — currently treated as reflective.
2. **Live view for parallel trials** — parallel mode runs headless; sequential
   mode streams the first trial to the canvas.
3. **Per-agent trait editor** — traits are set per group cohort (plus jitter).
4. **Territory map in the 3D view** — currently 2D only.
5. **Marks remember the signature they were laid with** — under the mechanistic
   nose, identity is read live, so changing `identity_signal` mid-run
   retroactively changes how that animal's old marks read.
6. **A connectome-backed species** — the framework's sensory → neural → motor →
   environment loop validated against a published circuit. The loop itself now
   exists (learned brains, above) with a phenomenological brain in it.
7. **A reason to rest** — a fatigue/sleep need in the physiology, so activity
   bouts and a daily rhythm can emerge in a learned brain instead of being
   absent.
8. **Macronutrients and predators** — separate protein / carbohydrate / lipid
   needs met by different foods; a predator with a brain of its own.
9. **Faster training** — a vectorised (PufferLib-style) environment; training
   currently runs the full engine in worker processes, about 10,000 animal-ticks
   per second.

See `biology.py`, `olfaction.py`, `policy.py` (the `k_*` weights), and
`config.py` for the extension points.
