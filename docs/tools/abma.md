# ABMA — Animal Behavior Modeling Arena

!!! warning "Experimental"
    ABMA is a research tool under active development. Its animals are models:
    what they do is a consequence of the rules or the training they were
    given, and every result should be read that way. The known limits of each
    model are listed below and in `fnt/abma/README.md`.

ABMA runs enclosure experiments on simulated animals. You build an arena,
populate it with cohorts tuned to the species you study, set a protocol
(duration, light cycle, treatments, animals or resources added and removed
mid-run), run replicate trials and export the tracks.

Launch from the FNT main GUI **ABMA** tab → **ABMA Designer**, or from the
command line (below).

## Why the output looks like UWB data

Each trial writes `data/uwb_<trial>_processed.csv` with the same columns the
[UWB PreProcessing tool](uwb.md) exports (`Trial`, `sex`, `sexid`, `Timestamp`,
`location_x`, `location_y`, `smoothed_x`, `smoothed_y`, …). Every downstream
analysis — contact bouts, edge lists, networks, your R scripts — runs on
simulated animals unchanged, so a simulation can be compared with a real trial
metric for metric.

## The window

| Tab | What you set |
|-----|--------------|
| **Arena** | size, boundary, nests, food, water, walled resource zones with doorways, water towers, huts, poles |
| **Population** | cohorts from a species card (body facts), sex, count, genotype, treatment |
| **Experiment** | days, time step, light cycle (start hour and hours of light), replicates, seed, protocol timeline |
| **Run** | nervous system (rules or a learned brain), the live 2D/3D view, per-animal drives and senses |

**Presets** (File → Load Preset): blank arena, open field test, prairie-vole
cage (saline vs methimazole), **VoleCosm** mesocosm (2.81 × 2.61 m, a walled
food-and-water box in each corner, a tower in the middle, 3 M + 3 F, 14:10
light), **VoleTerra** field enclosure, **Liddell Echo** mouse room.

## What the animals have

- **Bodies with budgets.** Energy comes only from eating and water only from
  drinking; hunger and thirst are what the stores are missing. Moving,
  fighting and mating cost energy. Every condition bar runs 0–100.
- **Scent marks** carrying the marker's identity, paid for from the bladder
  and fading with a half-life you set. Territory is not prescribed; it is what
  the marks and the animals' responses to them produce.
- **A nose** that methimazole (or a genotype) can take away, in part or whole.
- Optionally a **sky** (real sunrise, sunset and moon for a latitude) and a
  **sward** the animals wear trails into.

## Nervous systems: rules or a learned brain

**Rules** (the default) steer each animal by a weighted blend of named drives:
home, food, water, others, rival scent, exploration. Legible, and fast — but
an animal that is hungry is handed the direction of the food, so a rule-based
animal's foraging does not depend on its nose at all.

A **learned brain** is a small recurrent neural network that receives only what
an animal at that spot could sense (smell gradients ahead/left/right, three
whisker rays, taste, a memory of the last food and water, its own body state)
and outputs only a turn and a speed. Hunger, thirst and stress also set the
*gain* of its sensory layer, as neuromodulators would. Nothing about its
behaviour is written by hand: it was trained by reinforcement in randomised
worlds where the only reward is a drop in its own discomfort.

Two brains ship with ABMA:

| Brain | Raised in | Good at | Known limits |
|-------|-----------|---------|--------------|
| `vole_forager` (default) | open worlds, alone then in groups | finds food by smell; a need picks its own goal (thirsty → water); searches where food was lost; a cohort keeps itself fed | hunger does not stop it camping at a lone food source; little rest; no learned avoidance of rivals; without a nose it rarely finds food behind a doorway |
| `vole_doorways` | the same, then worlds with walled boxes, doorways and nose-less animals | without a nose it follows walls into boxes (95% in about 3.5 min, against 32%) | without a nose it rarely finds food lying in the open, and it does not leave a water tower — which is where VoleCosm releases its animals |

Choose one under **Run → Nervous system**. With a learned brain the inspection
column shows what is reaching the selected animal's brain and a raster of every
unit's activity.

## Command line

Run from the FNT environment. Every run can be described in one line:

```bash
# what would this run?
python -m fnt.abma --preset volecosm --males 3 --females 3 --days 5 --dry-run

# run it, headless, with a learned brain
python -m fnt.abma --preset volecosm --males 3 --females 3 --days 5 --brain vole_forager --out ~/ABMA_Projects

# open the window and run it on screen
python -m fnt.abma --preset volecosm --males 3 --females 3 --days 5 --watch

# intact vs methimazole: 2 paired replicates each, results as a tidy table for R
python -m fnt.abma --preset volecosm --males 3 --females 3 --days 5 --anosmia-study --doses 0,1 --replicates 2 --name mmz_3m3f

# a GIF of the first half hour, intact and methimazole side by side
python -m fnt.abma --preset volecosm --brain vole_forager --movie --movie-vs-mmz --movie-deprived 60

python -m fnt.abma --list-presets
python -m fnt.abma --list-brains
```

### Bench assays

Before trusting a brain in an enclosure, test it the way you would a new line:

```bash
python -m fnt.abma --assay all --brain vole_doorways   # a learned brain
python -m fnt.abma --assay all                          # the rule-based animal
```

| Assay | Question |
|-------|----------|
| `approach` | One food source out of reach; hungry, sated and anosmic. Does motivation gate it, and does it need its nose? |
| `choice` | Food one side, water the other; hungry vs thirsty. Does a need pick its own goal? |
| `search` | Food is removed after 40 s of eating. Does it search where the food was? |
| `doorway` | Chow in a walled box behind two doorways; with and without a nose. |
| `vitals` | Your config: does every animal eat and drink, and what do they do to each other? |

### Twin experiments: one life, lived twice

Every random draw an animal makes is fixed by (seed, animal, time step), so a
run can be copied mid-way and both copies live the same future — unless
something is changed in one. Whatever differs afterwards was caused by that
change.

```bash
# animal 0 is 30 points less hungry at hour 6; follow both lives for 6 h
python -m fnt.abma --preset anosmia --twin "hunger-30@0"

# drive one unit of animal 0's brain for 10 seconds
python -m fnt.abma --preset anosmia --brain vole_forager --twin "rnn:36=1@0~10"
```

Each writes `twin.png` (the two paths, when they parted, how far apart they
ended, how many other animals' lives changed) and `twin.npz`.

### Training a brain

Training needs PyTorch (installed with FNT) and runs the full simulation in
worker processes:

```bash
python -m fnt.abma.learn --name my_vole --minutes 20
python -m fnt.abma.learn --name my_vole --resume my_vole --minutes 10 --boxes 0.6 --anosmic 0.3
```

Brains are written to `~/ABMA_brains/` and appear in the window and in
`--list-brains`. The training log reports, per iteration, how much time hungry
and thirsty animals spend at food and water and how often a need picks the
right goal.

## Contest physics (experimental)

By default a same-sex contest is a dice roll made by the engine. Setting
`contest.enabled` in a config replaces it with a body that can do the things a
contest is made of, and leaves the rest to the animals:

- **Bite**: a third output of a learned brain. It lands on the nearest animal
  in reach and ahead, unless a wall is in the way or the target is under a hut.
  It costs the biter energy and the bitten animal health.
- **Sprint and fatigue**: speed runs from rest to `sprint_factor` times the
  easy pace; running faster than that pace tires the animal and takes the
  sprint away, so chases end.
- **Cover**: an animal under a hut cannot be bitten from outside, and a full hut
  (`refuge_capacity`, one by default) cannot be entered.

Raise a brain with `python -m fnt.abma.learn --contest`. Brains raised without
it, and the rule-based animal, never bite. Every value in `ContestParams` is a
free parameter. See the ABMA README for the senses that come with it and for
the known limits.

## A brain wired from anatomy (experimental)

`--anatomy c57_male` gives every animal a brain built from an anatomy spec: one
block of model neurons per brain area, connected only where the spec lists a
projection. The shipped spec is a reduced male C57BL/6J house mouse (44 areas,
311 model neurons). **It is untrained**: the weights are random draws inside
the wiring, so the behaviour means nothing yet. It is for seeing where
activity goes and what changes when one connection or one receptor level is
changed.

```
python -m fnt.abma --list-anatomies
python -m fnt.abma --anatomy c57_male --brain-recording 5
python -m fnt.abma --anatomy c57_male --strength "SC>TURN=0" --receptor "OT@LS=3" --brain-recording 5
python -m fnt.abma --anatomy c57_male --watch
```

| Option | What it does |
|---|---|
| `--anatomy NAME` | Use this spec (a name from `--list-anatomies`, or a path to a `.json`). With no `--preset`, the open field test is used. |
| `--anatomy-seed N` | Which random draw of the weights. A draw usually turns one way; another draw turns the other. |
| `--strength A>B=X` | Multiply the strength of one projection (0 cuts it). Repeatable. |
| `--receptor MOD@AREA=N` | Set one area's receptor level for a neuromodulator, 0 to 3. Repeatable. |
| `--brain-recording MINUTES` | Run one trial and keep every model neuron at every step. |
| `--brain-map [window\|browser]` | Open the brain map (below). |

A brain recording writes two files into the run folder: `brain_recording.npz`
(position, heading, the commands to the muscles, and every model neuron's
firing rate as a fraction of its maximum) and `brain_areas.csv` (mean, spread
and peak firing of each area, in percent of maximum).

### The brain map

`python -m fnt.abma --brain-map` opens a live picture of the brain: every area
and model neuron, lit by its firing while one animal walks an open field.

- **Layouts:** ribbons (projection strength as width), the network left to
  right, or the network arranged freely.
- **Click an area** for its inputs and outputs, each one's share and the
  evidence behind it. `-` and `+` make a projection weaker or stronger, or
  change a receptor level, in the running brain.
- **Click a neuron** to trace its firing rate (percent of its maximum) and to
  see what it listens to and talks to.
- **Verify:** a projection with no paper behind it says "needs verification";
  the button takes a paper or a note.
- **New random wiring** draws the weights again.

The map opens inside FNT if Qt's web engine is installed
(`pip install PyQtWebEngine`), and in your browser if it is not. Either way the
animal is run by FNT on your computer and nothing is sent anywhere. Your
changes and notes are saved in `~/ABMA_brains/anatomy/`.

The spec is a JSON file you can copy into `~/ABMA_brains/anatomy/` and edit:
areas, projections with a sign and a strength, where each neuromodulator is
released and each area's receptor level, where each sense enters, and the
evidence each connection rests on (a paper, or an idea with its author and
date).

## Output

A run folder holds `config.json` (the exact experiment), `provenance.json`
(a hash of the config and the source class of every parameter: measured,
literature, estimated or free), `data/` with the UWB-format tracks, condition
and event tables per trial, and, if asked, an analysis summary.

## Hierarchy or territories: the same readouts for real and simulated animals

`fnt/abma/core/structure.py` measures the social structure of one sex from a
table of **zone visits** (`animal`, `zone`, `start`, `stop` in seconds, and
optionally `day`). An RFID antenna at a resource zone gives that table for
real animals; `bouts_from_run(project_dir)` derives it from a finished ABMA
trial. Both then go through the same functions.

```python
from fnt.abma.core import structure as st

bouts = st.bouts_from_run("~/ABMA_Projects/my_run")     # or a real RFID visit table
males = bouts[bouts.sex == "M"]
st.structure_summary(males, animals=roster, zones=zone_list)
```

From the command line, on a visit table or on a finished run folder (one row
of readouts per group, written next to the input unless `--structure-out` says
otherwise):

```bash
python -m fnt.abma --structure visits.csv --structure-by arena,trial,sex --structure-days 1-5
python -m fnt.abma --structure ~/ABMA_Projects/my_run
```

| Readout | Meaning |
|---------|---------|
| `owner_share` | Share of the time at each zone taken by that zone's top animal (1 = exclusive). |
| `poles` | Effective number of animals that own zones: 1 = one animal owns every zone, Z = each of Z zones has a different owner. |
| `landless` | Share of animals that own no zone. |
| `owner_stays` | How often a zone has the same owner the next day. |
| `shared` | Share of the occupied time with two or more animals present. |
| `access_gini` | How unequal the animals' time at the zones is. |
| `landau_h`, `ttri`, `dci` | How linear, transitive and one-sided the displacement matrix is. |
| `resident_win`, `owner_win` | How often the animal that was there first, or yesterday's owner, stays. |
| `rank_transfer` | How often an animal's record at the other zones predicts who stays at this one. |

A **displacement** is read from the visits, not scored from a fight: one
animal is alone at a zone, a second joins it, then one of the two is alone
again; the one that stays is the winner (the sequence used for the field mice
in Vogt et al. 2024). With RFID visits this mostly records animals passing
through, so treat the displacement readouts as weak evidence of dominance and
prefer scored fights where they exist. Ownership is decided per day, and
`owner_win` uses the previous day's owner.
