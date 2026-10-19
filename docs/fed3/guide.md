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

1. **Flash every FED3 with firmware v3** (the `v3` branch of `fnt-fed3`).
   - Devices on older firmware show up in FED3 with a warning and aren't used.
2. **Give each one a number.**
   - Plug them in one at a time.
   - Open **FNT → FED → FED3** and use **Renumber a FED3…** in Setup.
   - The number is stored on the device's SD card, and labels its data files.

## Writing a protocol

A protocol is a short YAML file: phases, each with a rule and a way to end.

- **Start from a template.** In Setup, **New…** opens the editor with a template
  (FR1 on both pokes, stepping the ratio up, timed feeding, a four-patch
  competition).
- **Add phase…** writes a phase from a form: when it ends, which poke is active,
  the ratio, timeout, quota, free feeding, and a tone or flash on rewards.
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

1. **Choose the protocol** (**Open…**).
2. **For each device the protocol names, choose a FED3.**
   - The status column shows whether the device is ready: clock, SD card and
     firmware.
   - **Feed** and **Test cue** check a device on the bench.
3. **Press Start run.**
   - FNT checks every device first: connected, clock set, SD card working,
     schedule uploaded.
   - If any device fails, nothing starts and the reason shows on that device's
     row.
   - Otherwise every device starts at the same moment.

Runs are saved in `Documents/FED3_Runs` unless you choose another folder.

## During a run

- **Each device's card** shows:
  - its current phase and rule;
  - its counts;
  - its last event;
  - any problem: a jam, offline, a watchdog restart, a dead SD card, a clock
    that's wrong, no pellets for too long, or the wrong schedule.

  A problem stays on the card, outlined in red, until you press
  **Acknowledge**, even if it has cleared, so an overnight jam is still visible
  in the morning.
- **The ⋯ menu** on a card feeds a pellet by hand, plays a test cue, or sets the
  device's clock.
- **Edit protocol…** changes the protocol mid-run.
  - Phases already over can't change.
  - The current phase can, and the change applies at once.
  - Each version is kept in the run folder.
- **Advance phase** appears only when the current phase ends by hand.
- **Closing the window** doesn't stop the run. The devices keep running and
  recording to their SD cards; when you reopen FED3, FNT copies what it
  missed.

## Stopping

**Stop run** ends recording. Each device keeps running its current phase,
including any daily feeding window, until the next run starts, so stopping
never cuts the animals off.

## What's in a run folder

| File | |
| --- | --- |
| `run.json` | The run: protocol hash, which FED3 played which device, start time, rotation seed, edits, phase changes FNT made. |
| `protocol.yaml` | The protocol exactly as started, plus `protocol.v2.yaml` and so on after edits. |
| `sched/<device>.txt` | What each device was sent. |
| `<device>_FED<nnn>.csv` | A byte-identical copy of that device's data file for the run. |
| `log.csv` | Everything FNT did and noticed, in UTC. |

The data columns are described in [design.md](design.md), Part 3.

## Trying it without hardware

```bash
python -m fnt.fed3.ui.window --demo 4 --speed 60
```

This opens FED3 with four simulated devices and a simulated animal poking at
each, with a minute passing every second. It uses a separate settings store and
saves runs under `FED3_Runs/demo`.
