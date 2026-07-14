# ABMA — Animal Behavior Modeling Arena

A GUI-driven agent-based platform for running *in silico* animal-behaviour
experiments inside the Field Neuroethology Toolbox (FNT). Design an arena,
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
    policy.py               decision policies (RuleBasedPolicy; RL-ready seam)
    simulation.py           Simulation: physics, physiology, combat, events
    analysis.py             built-in socio-spatial + dominance analysis
    recorder.py             streaming writers for the canonical CSV schema
    runner.py               project folders + multi-trial orchestration
    run_headless.py         CLI: python -m fnt.abma config.json --out DIR
  gui/                      thin PyQt5 wrapper (needs a QApplication)
    abma_canvas.py          2D matplotlib canvas for arena design
    pg_canvas.py            pyqtgraph OpenGL 3D live run view (PyOpenGL)
    agent_inspector.py      live per-agent stat card
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
  README.txt               provenance
  data/
    uwb_S001_processed.csv  trajectory (FNT schema)
    events_S001.csv         mating / aggression events
    agents_S001.csv         per-agent metadata (genotype, treatment, traits)
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

## Roadmap / not yet done

1. **Absorbing boundary** — currently treated as reflective.
2. **Explicit scent-mark field** — a decaying spatial grid of marks per identity,
   so avoidance responds to *where marks were laid* rather than to live home
   centres (mark-then-leave territory dynamics).
3. **Live view for parallel trials** — parallel mode runs headless; sequential
   mode streams the first trial to the canvas.
4. **Per-agent trait editor** — traits are set per group cohort (plus jitter).
5. **Species presets** — bundle prairie/meadow vole and house-mouse defaults.

See `biology.py`, `simulation.py` (the `k_*` constants), and `config.py` for the
extension points.
