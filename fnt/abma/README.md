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
    policy.py               decision policies + the drive decomposition
    simulation.py           Simulation: physics, physiology, combat, events
    scent.py                decaying grid of identity-carrying marks
    physiology.py           energy/water budget in real units
    compose.py              describe a run in one line -> ExperimentConfig
    analysis.py             built-in socio-spatial + dominance analysis
    recorder.py             streaming writers for the canonical CSV schema
    record.py               replayable per-frame archive (drives + sensing)
    provenance.py           config hash + where every number came from
    study.py                conditions x replicates, paired seeds, comparison
    runner.py               project folders + multi-trial orchestration
    run_headless.py         CLI: python -m fnt.abma config.json --out DIR
  gui/                      thin PyQt5 wrapper (needs a QApplication)
    abma_canvas.py          2D matplotlib canvas + territory-map overlay
    pg_canvas.py            pyqtgraph OpenGL 3D live run view (PyOpenGL)
    agent_inspector.py      hover stat card
    science_panel.py        roster, drive bars, traces, coupling diagram
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
the agent wants to do". Swapping in a learned `RLPolicy` later needs no engine
change.

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
   environment loop validated against a published circuit, before trusting the
   same loop shape for a phenomenological vole brain.

See `biology.py`, `olfaction.py`, `policy.py` (the `k_*` weights), and
`config.py` for the extension points.
