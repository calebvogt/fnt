# Running FED3 experiments

> [!WARNING]
> **Alpha.** This works end to end against simulated devices and the
> firmware's logic built for a PC. It has not yet been run on FED3 hardware;
> see the bench checklist in the `fnt-fed3` README first.

The FED3s run the experiment themselves. FNT writes the protocol, starts the
run, keeps a copy of the data, and shows you what each device is doing. If the
PC crashes, reboots or is unplugged, the devices carry on; reopen FED3 in FNT
and it picks the run back up.

## Before the first run

1. **Flash every FED3 with firmware v3** (`fnt-fed3`; 3.1 adds **Lights**).
   - Devices on older firmware show up in FED3 with a warning and aren't used.
   - On 3.0 everything works except Lights.
2. **Give each one its own ID.** A FED3's ID is its one name: on its screen,
   in your protocols, in its data file and on FNT's screens, e.g. `FED007`.
   A newly flashed FED3 is FED001. Set it either way; it's kept on the SD card,
   so it's a one-time job:
   - **On the FED3:** power it on, and while the screen shows `FED3 ID`, hold
     both pokes for a second. Left poke −1, right poke +1 (hold to go faster).
     It saves after 4 seconds without a poke. (Not offered during a run.)
   - **In FNT:** **FNT → FED → FED3**, then type the ID in the **ID** box on
     that FED3's row in Setup and press Enter: `7`, `007` or `FED007`, any
     number from 1 to 999. **Lights** shows which FED3 a row is.

## Writing a protocol

A protocol is a short YAML file: the FED3s it uses, by ID (`devices: [FED001,
FED002]`), and phases, each with a rule and a way to end.

- **Start from a template.** In Setup, **New…** opens the editor; **New from
  template** has:
  - **Classic FED3 modes**, one protocol each: Free feeding, FR1, FR3, FR5,
    Progressive ratio, Extinction, FR1 and PR reversed (right poke),
    Optogenetic stimulation and its reversed version, Timed feeding, and FNT's
    FR with timeout and Random ratio. They keep the classic 4 kHz tone and blue
    light on a reward, and the click on pokes that don't earn one. Light
    tracking isn't available: it needs a poke light that stays on.
  - **Examples**: FR1 on both pokes, stepping the ratio up, a four-patch
    competition.
- **Add phase…** writes a phase from a form: where it goes, when it ends, which
  poke is active, the ratio, timeout, quota, free feeding, whether a reward
  delivers a pellet, a tone, flash or BNC pulses on rewards, and clicks.
  - The form switches off whatever can't apply. Free feeding means no active
    poke, so the ratio, reward and cues go grey; with no active poke a phase
    can't end on rewards. A click needs what it marks: a timeout, a quota, or a
    ratio above 1.
  - The check catches the same things in text: a phase that only ends on
    rewards no poke can earn is an error, and free feeding with an active poke
    is a warning.
- **Edit phase** reads a phase back into the same form and rewrites it, and
  **Remove phase** deletes one. A phase that uses something the form doesn't
  have (a named rule, a rotation, daily windows, per-device rules) is edited in
  the text: the form says what it can't show and takes you to the phase rather
  than drop it. Comments inside a phase the form rewrites are lost.
- **The check on the right** updates as you type and points at the line of any
  problem.
- **The timeline below it** shows when each phase will start, and marks the
  phases that need FNT running to end.
- **The full format** is in [design.md](design.md), Part 4. The command line
  checks files too:

  ```bash
  fnt-fed check protocol.yaml
  fnt-fed preview protocol.yaml --days 21
  fnt-fed compile protocol.yaml        # the text each device receives
  ```

**Times of day are in standard time all year**, the time your vivarium's
lights keep. In summer, `"19:00"` is 20:00 on a wall clock that follows
daylight saving. FNT labels every time it shows with its zone.

## Starting a run

Setup has the protocol and its timeline on the left, and every FED3 plugged
in on the right, one row each in port order: Port, ID, Status, and Tests
(Lights, Tone, Feed). A row always shows the same FED3. The bar across the top
says how many of the protocol's FED3s are plugged in.

1. **Choose the protocol** (**Open…**).
2. **Plug in the FED3s it names.** Setup lists them (`This protocol uses
   FED001, FED002`) and shows their IDs in the colour each has on the cards
   and plot. A FED3 the protocol doesn't name shows `not in this protocol`
   and isn't used.
   - If one is missing, Setup says so (`FED004 isn't plugged in`). Plug it in,
     or type that ID in another FED3's **ID** box.
   - Typing an ID another FED3 has swaps the two, and Setup says so. Two
     FED3s can't share an ID, because their data files are named by it.
   - A FED3 can't be renumbered during a run. After Stop it can: the stopped
     run's copy keeps following it under its new number, and `run.json`
     records the change.
   - **Lights** lights a FED3's bar and pokes white until you press it again, so
     you can see which is which. They go dark when the run starts.
   - The status column shows whether the device is ready: clock, SD card and
     firmware.
   - **Tone** plays the protocol's reward tone, and **Feed** dispenses a pellet.
     With an empty hopper a dispense sounds like rhythmic beeping with bursts:
     that's the motor working through its jam-clearing turns for about a
     minute, then a `jam` row and a retry every 10 minutes.
3. **Press Start run.**
   - FNT checks every device first: connected, clock set, SD card working,
     schedule uploaded.
   - If any device fails, nothing starts and the reason shows on that device's
     row.
   - Otherwise every device starts at the same moment.
   - **A FED3 still in a run that was never stopped** (its row says `still in
     run …`) isn't started over without asking. This happens when FNT couldn't
     pick that run back up, for example because it was saved in another
     folder. Start names the FED3s and the run and asks: **Cancel**, then
     **Change folder…** to that run's folder, to keep it going; **End it and
     start** ends it on those FED3s first.
   - FED3s still recording into a stopped run are listed under the table;
     starting a new run on them ends that.

Runs are saved in `Documents/FED3_Runs` unless you choose another folder. Runs
in folders used before are still followed: a stopped run keeps copying, and an
unfinished one is picked back up. If FNT can't read or load one, it says so in
Setup instead of failing to open.

## During a run

The bar across the top of the window shows where you are: **Setup › Running ›
Stopped**, with a note on the current step.

- **Each device's card** shows:
  - the FED3 drawn with its left pokes, pellets and right pokes on it, each
    flashing as it goes up (dimmed while the device isn't connected);
  - beside it, its current phase and rule, rewards this phase, when the phase
    ends, its last event, and its port, battery and temperature;
  - any problem: a jam, offline, a watchdog restart, a dead SD card, a clock
    that's wrong, no pellets for too long, or the wrong schedule.

  A problem stays on the card, outlined in red, until you press
  **Acknowledge**, even if it has cleared, so an overnight jam is still visible
  in the morning.
- **Feed** on a card dispenses a pellet by hand (recorded as manual: a pellet,
  not a reward). **More ▾** has **Play a cue…** and **Set its clock now…**,
  which sets the clock even across a big jump, after asking.
- **The counts on the drawing** are left pokes, pellets and right pokes, as
  labelled under it.
- **The plot** runs across the top at full width: cumulative pellets per
  device, with phase changes dotted and Stop dashed. The menu in its title
  draws the time axis in standard time (as the devices keep it), local time
  with daylight saving, or UTC. The cards flow below it in as many columns as
  fit.
- **Temperature and humidity** show only on a FED3 fitted with an AHT20
  sensor; without one the device reports `nan` and the card leaves them out.
- **The log** runs along the bottom of the window, newest last: problems in red,
  phase changes in blue, run starts and stops in green, and things people did in
  purple. **▼ Log** collapses it to its header, which then counts what's new.
- **Edit protocol…** changes the protocol mid-run.
  - Phases already over can't change.
  - The current phase can, and the change applies at once.
  - The editor says as you type if the running devices can't take an edit,
    and won't apply it. If one moves on to the next phase while you're
    editing and the edit no longer fits, FNT says so and reopens the editor
    with your changes.
  - A FED3 that's unplugged when you apply gets the edit when it's back.
  - Each version is kept in the run folder.
- **Phases FNT moves** (counts across devices, or by hand): a FED3 that's
  unplugged at that moment is moved on as soon as it's back.
- **Advance phase** appears only when the current phase ends by hand.
- **Closing the window** doesn't stop the run. The devices keep running and
  recording to their SD cards; when you reopen FED3, FNT copies what it
  missed.

## Stopping

**Stop run** ends the run: phases stop changing. Each device keeps running its
current phase, including any daily feeding window, until the next run starts,
so stopping never cuts the animals off.

The devices also keep recording. Their rows after Stop go to the same file on
the card, and FNT keeps copying them into the run's folder (after a `run_end`
row) until the device starts another run: in the Stopped view, after **New
run**, and after FNT is reopened. A device that was unplugged when you pressed
Stop is stopped as soon as it's back.

To separate the experiment from what came after it:
- **In each data file**, the `run_end` row is the stop, stamped by the device:
  every row after it (by `seq`) was recorded after Stop.
- **In `run.json`**, `stopped_at` is when Stop was pressed (UTC), and `ended`
  gives each device's `run_end` row: its `time` and `seq`. A device that was
  unplugged at Stop ends later than `stopped_at`, and its entry shows when. If a
  device's clock was lost at that moment its `time` is empty, and `seq` still
  marks the point.

## What's in a run folder

| File | |
| --- | --- |
| `run.json` | The run: protocol hash, which FED3 played which device, start time, rotation seed, edits, phase changes FNT made (with any device that missed one, under `missed`, until it's caught up), when Stop was pressed (`stopped_at`), where each device's data after Stop begins (`ended`), and any FED3 renumbered after Stop (`renumbered`). |
| `protocol.yaml` | The protocol exactly as started, plus `protocol.v2.yaml` and so on after edits. |
| `sched/FED<nnn>.txt` | What each FED3 was sent. |
| `FED<nnn>.csv` | A byte-identical copy of that FED3's data file for the run, including what it recorded after Stop. |
| `log.csv` | Everything FNT did and noticed, in UTC. |

The data columns are described in [design.md](design.md), Part 3.

## Trying it without hardware

```bash
python -m fnt.fed3.ui.window --demo 4 --speed 60
```

This opens FED3 with four simulated devices and a simulated animal poking at
each, with a minute passing every second. It uses a separate settings store and
saves runs under `FED3_Runs/demo`.
