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
| `partner` | The partner preference test: a tethered partner at one end of a long box, a tethered stranger at the other, 30 minutes. Run attached with a nose, without one, without a vomeronasal organ and unattached. Needs a brain raised with `--life`. With an anatomy brain that keeps its own attachments (`--anatomy NAME --anatomy-weights W`), each test begins with ten minutes beside the partner-to-be, and a fifth condition loses the sense of smell after that pairing. |

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

| Option | What it does |
|---|---|
| `--life` | The animals also tire, get cold and become attached (below). The log then adds how much they rest, touch and stay with a partner. |
| `--attached X` | With `--life`: share of animals that arrive attached to another animal of their world (default 0.5). |
| `--bond-in-brain` | With `--life` and an anatomy spec that has plastic connections: attachments are written by the brain itself. Each life begins with the dealt pairs held together for three minutes while the body reports a mating. |
| `--social-param NAME=VALUE` | With `--life`: override one value of the settings below. Repeatable. |
| `--torch-threads N` | Threads the weight update may use. Set it to a few on a computer busy with other work. |

## Rest, warmth and attachment (experimental)

Off unless a config turns it on (`social.enabled`, or `--life` in training).
More needs, each uncomfortable and nothing else:

- **Tired.** Rises while awake, faster while moving; falls at rest.
- **Cold.** Falls in a box or under a hut, and for each body it is touching.
  Warmth needs no nose.
- **Separation distress.** Rises in an animal that has an attachment while the
  partner's odour is not close by; falls while it is.

A fourth need is off until `libido_rise_h` is set above 0: **the pull toward
the other sex**. It rises by itself and falls while the animal takes in the
odour of an animal of the other sex, through the air by main olfaction or on
contact through the vomeronasal organ; a mating takes a step off it. An animal
with no olfactory epithelium can quiet it only by contact.

An attachment forms between animals in contact and steps up at a mating, as
far as the animal can take in the other's odour. Which nose that needs is a
setting, because it is not settled for the prairie vole: `bond_needs_moe`
(main olfaction, the default), `bond_needs_vno` (the vomeronasal organ),
`read_by_vno`. An animal's vomeronasal organ is its own trait, `vno_ability`,
separate from `smell_ability`; methimazole acts on `smell_ability` only.

By default the engine keeps the attachments in a table and hands the brain a
ready-made "partner odour" and "stranger odour". With `bond_in_brain` the
engine hands over neither: the brain gets the mixed odour signatures of the
animals nearby (`olf_id_0`..`olf_id_23`) and the fact of a mating (`mating`),
and separation distress follows what the brain's own plastic connections hold
(see "Connections that experience changes").

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
projection. The shipped specs are a reduced male C57BL/6J house mouse
(`c57_male`: 44 areas, 311 model neurons) and a draft prairie vole
(`prairie_vole`: 45 areas, 371 model neurons, either sex). **Both are untrained as shipped**: the weights are random draws inside
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
| `--brain-replay FILE.npz` | Play a brain recording back on the brain map. |

A brain recording writes three files into the run folder:
`brain_recording.npz` (position, heading, the commands to the muscles, bladder,
urine spots, and every model neuron's firing rate as a fraction of its
maximum), `brain_areas.csv` (mean, spread and peak firing of each area, in
percent of maximum) and `brain_recording.html`, the brain map playing that run
back. The HTML is a single file: copy it to another computer and open it in a
browser.

With scent marking on, this brain urinates by muscle (bladder wall against
urethral sphincter) and reads urine with its vomeronasal organ: only when its
nose is on a spot, and then the spot's MUP barcode. Give a group the genotype
`MUP: INBRED` to make every animal's barcode the same, as in an inbred strain.

### Training it

```
python -m fnt.abma.learn --anatomy c57_male --name c57_forager --workers 2 --minutes 600
python -m fnt.abma --anatomy c57_male --anatomy-weights c57_forager --brain-map
```

Training changes only the strength of connections the spec already has, and
never a projection's sign. The animal is scored on nothing but its own bodily
needs. Weights go to `~/ABMA_brains/anatomy/`; `--resume NAME` continues a
run. Expect hours, not minutes.

### What a spec may say beyond areas and projections

These keys are optional; a spec without them runs as before.

| Key | What it does |
|---|---|
| `"species"` | The animal the spec describes (`"Prairie vole"`); training then raises that species. |
| `"dynamics": {"tick_s", "tau_s", "sweep"}` | How often the network ticks and how fast a model neuron follows its drive. With `"sweep": true` the areas are updated in order from the senses inward within a tick and each area's own inhibition is balanced inside the tick. Use it for a tick of about a second; without it a coarse tick makes areas flip on and off and delays a signal by one projection per tick. |
| `"lesion": {"MOE": "smell_ability", "VNO": "vno_ability"}` | The cells of a sense organ die with a sense of the body. An animal with half its sense of smell has half of that organ's cells, permanently silent; which cells go is fixed for each animal. |
| `"entry_gain": {"PVN": {"mating": 3.0}}` | Every cell of an area answers a channel with this weight (the entry table gives a channel one cell). |
| `"plasticity": [...]` | Projections whose strength an animal's own experience changes (below). |

### Connections that experience changes

A projection listed under `"plasticity"` gets, in every animal, extra weights
that start at zero and are written during its life:

```
"plasticity": [{"projection": "MOE>MOB", "gate": ["OT", "DA"], "pre_channels": "olf_id_",
                "rate_per_s": 0.02, "gate_threshold": 0.5, "pre_threshold": 0.35,
                "cap": 1.0, "decay_days": 30}]
```

A weight grows while its source cell fires above `pre_threshold`, its target
cell is active and every `gate` neuromodulator is above `gate_threshold`. The
weights onto one target cell share a budget (`cap`), and fade over
`decay_days`. `pre_channels` limits writing to the source cells whose own
sensor channel starts with that text. Training does not touch these weights;
it replays them.

The shipped `prairie_vole` spec (a draft) uses this for a partner memory: a
mating reaches the PVN and the VTA, oxytocin and dopamine rise, and the odour
signature the nose is carrying is written into the first olfactory synapse.
Nothing tells the brain who anyone is. Three things about it are assumptions,
and the spec says so: that outbred voles each have their own odour signature
(three of 24 components per animal), that the memory sits at the first
olfactory synapse, and every number of the rule.

Methimazole is given with a group's `treatment` (`drug: methimazole`, `dose`
0 to 1, `day_offset` in days from release; a positive offset gives it during
the run). With a spec that has a `lesion` table, the dose kills that share of
the olfactory sensory cells, and whatever was written onto their synapses goes
with them.

### Starting from a brain that already behaves

Reward alone has not so far taught an anatomy brain to forage in the time we
can run. Two steps before training help:

- **Settling.** In the untrained network some areas are silent for good and
  others pinned on. A silent cell passes no learning signal.
  `python -m fnt.abma.learn.settle --anatomy prairie_vole` moves each cell's
  resting drive until its mean rate over nursery lives sits at a target;
  weights are not touched.
- **Teaching by example.** `python -m fnt.abma.learn.imitate --name NAME
  --teacher vole_doorways --start prairie_vole_settled --own 0.5` shows the
  anatomy brain what a generic learned brain sensed and moves its weights (same wiring and sign limits)
  until its own muscles command what that brain commanded. A share of the
  lives are the student's own, with the teacher saying what it would have done
  there.

Reward-only training then continues from the taught brain: add
`--critic-warmup 8` to `python -m fnt.abma.learn --anatomy ... --resume NAME`
so the critic catches up before the brain is touched. A brain started this
way has to be described as such.

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

The 🧠 button in the ABMA window opens the map on the design you are building.

The map opens inside FNT if Qt's web engine is installed
(`pip install PyQtWebEngine`), and in your browser if it is not. Either way the
animal is run by FNT on your computer and nothing is sent anywhere. Your
changes and notes are saved in `~/ABMA_brains/anatomy/`.

The spec is a JSON file you can copy into `~/ABMA_brains/anatomy/` and edit:
areas, projections with a sign and a strength, where each neuromodulator is
released and each area's receptor level, where each sense enters, and the
evidence each connection rests on (a paper, or an idea with its author and
date).

### The skeleton

The **Skeleton** button beside the 3D views swaps the schematic body for a
jointed skeleton built from published measurements:

    python -m fnt.abma --skeleton
    python -m fnt.abma --skeleton --skeleton-plot skeleton.png

The first command prints every bone length with the paper it comes from and
how much of each printed joint range a walk uses. Ten of the fifteen lengths
are printed in a paper and one is worked out from printed numbers. The other
four (neck, chest, loin, sacrum) have no printed length, so the printed body
length is split along the spine by our own guess. Posture and stepping are
also our own. It is geometry only: the network sets speed and turning, and
the legs step in a fixed trot inside the printed joint ranges.

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
