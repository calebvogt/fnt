# FED3 v3: device rule, schedules and protocol files

> [!WARNING]
> **Draft.** Nothing here is implemented yet. This document is the contract
> between three pieces that will be built against it: the FED3 firmware
> (`fnt-fed3`, v3), the FNT protocol compiler and run monitor, and the FNT FED
> tab. There is no compatibility with the legacy FED3 library's modes, commands
> or file format.

## Goals

Run FED3 devices unattended for weeks, in single-device and multi-device
(competitive, social) arrangements, with behaviour defined in a file rather than
in firmware.

1. **No new firmware for new combinations.** A "mode" is a set of values for
   one configurable rule. Firmware changes only to add a new building block.
2. **The device runs the whole experiment.** At Start, FNT compiles the protocol
   into a schedule for each device and uploads it. The device then makes:
   - every per-poke decision;
   - every phase change, whether time-, clock- or count-based;
   - every dispense and every cue.

   It keeps its schedule and its place in it on the SD card. A crashed,
   rebooted or unplugged PC changes nothing.
3. **The PC watches, records and edits.** FNT shows what each device is doing,
   keeps a live copy of the data, keeps the clocks set, and uploads a new
   schedule when the protocol is edited mid-run. It also checks the few
   conditions that depend on more than one device.
4. **The SD card is the record.** Every row the device writes is also sent live,
   numbered, and can be re-read after a gap. FNT's copy is byte-identical.
5. **Every state can be rebuilt from what is on the card.** After a power cut,
   watchdog reset or clock loss, the device works out where it should be, in
   the same way every time.

| Concern | Owner | If the PC is down |
| --- | --- | --- |
| Poke decisions, ratio, timeout, quota | Device | Unaffected |
| Dispensing, jam recovery, cues | Device | Unaffected |
| Phase changes on time, clock time, rewards or pokes | Device | Unaffected |
| Schedule and place in it | Device, on SD | Survive reboot and watchdog reset |
| Phase changes that depend on several devices; manual advance | FNT | Wait until FNT returns, or take the phase's device-local fallback |
| Run record | SD; FNT copies it | Rows are re-read on reconnect; nothing is lost |
| Problems (jams, resets, offline devices) | FNT shows them on screen | Seen at the daily check once FNT is back; the rows are still on the card |

### Terms

| Term | Meaning |
| --- | --- |
| **Reward** | A met ratio. It delivers a pellet unless `reward=none`. |
| **Pellet** | A pellet that physically reached the well: earned, free-fed or dispensed by hand. |
| **Poke** | Any beam break of either poke, whatever its outcome. |

---

## Part 1: the rule

Each phase sets every key below. Any key a phase doesn't mention takes its
default, not its value from the previous phase, so a phase always means the
same thing wherever it is in the schedule.

### Keys

| Key | Values | Default | Meaning |
| --- | --- | --- | --- |
| `active` | `left` `right` `both` `none` | `left` | Which poke can earn a reward. |
| `count` | `shared` `per_port` | `shared` | With `active=both`: `shared` means pokes on either side add to one ratio counter; `per_port` gives each side its own. |
| `ratio` | 1–999 | `1` | Pokes per reward, for `fixed` and `random`. |
| `ratio_kind` | `fixed` `random` `progressive` | `fixed` | `random`: each counted poke is rewarded with probability 1/`ratio`. `progressive`: the requirement after *k* rewards this phase is round(5·e^(0.2·(k+1)) − 5), giving 1, 2, 4, 6, 9, 12, 15, 20, 25, 32 and so on. `ratio` is ignored. |
| `timeout` | seconds, 0–3600 | `0` | After a reward, pokes are logged with outcome `timeout` and do not count. |
| `min_poke_ms` | 0–5000 | `0` | Pokes shorter than this are logged with outcome `short` and do not count. |
| `reward` | `pellet` `none` | `pellet` | `none` means a met ratio is logged and cued but delivers nothing. Use it for extinction, and with a `bnc` cue for opto self-stimulation. |
| `quota` | 0–9999 | `0` | The most rewards per quota period (0 means unlimited). Once the quota is used up the patch is **depleted**: pokes are logged with outcome `depleted` and `cue.depleted` plays. |
| `quota_period` | seconds | `0` | 0 means once per phase. Otherwise the quota refills every period, counted from when the phase started. |
| `free_feed` | `0` `1` | `0` | Keep a pellet in the well regardless of pokes. Pokes are still logged. |
| `refill_delay` | seconds | `5` | With `free_feed`: the wait after a retrieval before the next dispense. |

**How a poke is decided**
- A poke is decided the moment the beam breaks, so cues and dispenses have no
  delay. With `min_poke_ms`, it is decided once the beam has stayed broken
  that long, or as `short` if it is released sooner.
- Its row is written when the beam is released, stamped with the time it
  broke and carrying its duration. A dispense that lands while the animal is
  still in the poke can therefore appear before the poke's row; `time` and
  `ms` keep the true order. A beam held longer than 60 s is logged as `held`
  at 60 s.
- The first of these that applies is the outcome:
  1. `short`: released before `min_poke_ms`;
  2. `pellet_in_well`: a pellet is in the well or on its way;
  3. `inactive`: not the active poke (with `active=none`, every poke);
  4. `timeout`;
  5. `depleted`;
  6. `reward` if this poke meets the ratio, otherwise `counted`.
- Every poke counts toward `phase_left`/`phase_right` and the `pokes` end
  condition, whatever its outcome.

**Counting rules**
- A ratio counter resets to zero when it earns a reward (with `per_port`, only
  that side's counter), and all counters reset at a phase change.
- Daily windows (Part 2) change keys but never reset counters.

**Dispensing and jams**
- A dispense takes about a second, until the pellet breaks the well beam,
  which writes a `dispense` row. Taking it writes a `retrieve` row.
- If the motor can't deliver a pellet, a `jam` row is written and the same
  dispense is retried every 10 minutes until it succeeds. At most one
  dispense is ever owed, so rewards earned while jammed don't pile up.

**A blank or corrupt card** boots into a one-phase schedule of the defaults:
FR1, left poke, no cues.

### Cues

**The lights are off unless a cue flashes them.** A cue is up to 3 actions
joined by `+`, played at once and without blocking. A new action on the same
output replaces the one already playing. Every cue is triggered by a poke.

| Key | Plays when |
| --- | --- |
| `cue.reward` | A ratio is met (just before the dispense). |
| `cue.counted` | A counted poke has not yet met the ratio. |
| `cue.inactive` | The poke that cannot earn a reward is used. |
| `cue.timeout` | A poke is made during the timeout. |
| `cue.depleted` | A poke is made while the patch is depleted. |

| Action | Syntax | Notes |
| --- | --- | --- |
| Tone | `tone:<hz>:<ms>` | A piezo square wave. One tone at a time, and there is no volume control. |
| Click | `click` | Short for `tone:800:8`. |
| Noise | `noise:<ms>` | A burst of random 50–250 Hz tones. It doesn't block. |
| Flash | `flash:<target>:<r:g:b:w>:<ms>` | Target is `bar` (the 8-LED strip), `poke_l`, `poke_r`, `pokes`, or `poke`, meaning the poke that triggered the cue. The LEDs go dark again afterwards. |
| BNC pulse train | `bnc:<width_ms>:<hz>:<count>` | For external speakers, LEDs, opto or sync. |

`none` (the default for every cue) plays nothing.

### What replaces the old modes

| Old mode | Rule |
| --- | --- |
| FR *n* | `ratio=n` |
| PR | `ratio_kind=progressive` |
| RR *n* | `ratio_kind=random ratio=n` |
| FRTO *n*, *t* s | `ratio=n timeout=t` |
| Free feeding | `active=none free_feed=1` |
| Extinction | `reward=none` |
| Opto | `reward=none cue.reward=bnc:25:20:20` |
| Any `_R` variant | the same, plus `active=right` |
| Timed feeding | A phase with a daily window that sets `free_feed=1` (example 3). |
| Light tracking | Dropped. It needs a poke light that stays on. |

Each old mode's tone and light becomes a `cue.reward` choice.

---

## Part 2: device schedules

FNT compiles the protocol file (Part 4) into one schedule per device. It is
plain text, one directive per line. The device keeps only the current phase in
memory and reads the next phase from the card when it moves on, so a schedule's
length costs SD space, not RAM.

**Limits:** at most 256 phases, 8 daily windows per phase, 6 `until` lines per
phase, and 64 KB per schedule.

```
NAME Four-patch competition
PHASE Habituation
UNTIL at=19:00 day=3
SET active=none free_feed=1
PHASE Training
UNTIL goto
UNTIL after=345600
SET active=both cue.reward=tone:4000:200+flash:bar:0:30:0:0:1000 cue.depleted=click
PHASE Competition 1
UNTIL at=19:00
SET active=both quota=30 quota_period=3600 cue.reward=tone:4000:200+flash:bar:0:30:0:0:1000
SET cue.depleted=click
…
```

- `PHASE <name>` starts a phase. Names can be up to 24 characters, and phases
  are numbered from 0 in file order.
- `SET <k=v …>` gives the phase's rule and cues. One phase can have several
  `SET` lines. The compiler writes only keys that differ from their defaults,
  in the order of Part 1's tables. It starts a new `SET` line before any line
  passes 100 characters, so every upload line fits the 128-byte input limit.
  The same protocol therefore always compiles to the same text and the same
  crc.
- `WINDOW <hh:mm>-<hh:mm> <k=v …>` overrides keys between two clock times
  every day. A window may cross midnight. Several `WINDOW` lines with the same
  times form one window, so long windows split the same way as `SET`.
- `UNTIL <condition>` ends the phase. If there are several, the first one met
  ends it. A phase with no `UNTIL` runs until the run ends.

**Every device in a run gets the same list of phases**: the same count, the
same names and the same order. Only the rules differ between devices. That
keeps phase numbers meaningful across devices, for `GOTO` and for display.

| Condition | Ends the phase |
| --- | --- |
| `after=<s>` | That many seconds after the phase started. |
| `at=<hh:mm>` | At the first such clock time **strictly after** the phase started, so a phase that begins at 19:00 runs to 19:00 the next day. |
| `at=<hh:mm> day=<n>` | At that time on day *n* of the run. Day 1 is the calendar day of t0. |
| `rewards=<n>` | After *n* rewards in this phase. |
| `pokes=<n> side=<left\|right\|any>` | After *n* pokes of any outcome in this phase. |
| `goto` | Only when FNT sends `GOTO`. Used for manual advance and conditions that need FNT. |

### Phase start times

Every later time condition counts from a phase's start time, so that time is
defined exactly:

| The phase was entered by | Its start time is |
| --- | --- |
| A time condition (`after`, `at`) | **The time the condition came due**, not the time the device noticed. |
| A count condition | The time of the poke that met it. |
| `GOTO` | The time `GOTO` arrived. |
| `RUN:BEGIN` | t0. |

Counting from the due time is what makes catching up exact. If the device is
off when a time condition comes due, it boots and finds the condition already
past, so:
1. it ends the phase at the due time and starts the next phase at that same
   time;
2. it checks the next phase the same way, and repeats until it reaches the
   present;
3. it logs each change as a `phase` row. The row is stamped when it is
   written, so timestamps never run backwards, and its detail records the
   phase's real start: `Rot 2;catch-up;start=2026-10-22T19:00:00-07:00`.

The device ends up in the same phase, with the same start time, as if it had
never been off. Devices on a daily rotation therefore stay in step through
outages.

Count conditions can't come due while a device is off, and `goto` waits for
FNT.

### Clock

The device clock runs on **standard time all year, with no daylight-saving
changes.** Every `at` time, window and day boundary uses it, and every row's
timestamp carries its UTC offset (for example `2026-11-03T19:00:04-07:00`).

That way:
- a 19:00 phase change stays at the same point in a vivarium light cycle that
  ignores daylight saving;
- the data never repeats or skips an hour.

FNT shows clock times, plots and the timeline in the same standard time,
labelled with the zone. A vivarium whose lights do follow daylight saving would
need daylight-saving rules in the firmware; that isn't planned until someone
needs it.

**Protection against clock problems**

| Problem | What the device does |
| --- | --- |
| The clock lost power (the RTC reports `lostPower()`) | Status reports `clock=unset`. The device keeps the current phase's rule but ignores windows and time conditions until the clock is set. FNT sets it automatically as soon as it sees `clock=unset`, then the device catches up. |
| A time change of more than 10 minutes during a run | Refused with `ERR,TIME_JUMP`. FNT asks you to confirm and then sends `TIME!:`. This stops a mistyped clock from catching up through a week of phases. |
| Small corrections from automatic sync | Applied, and logged as a `clock` row showing the before and after times. |

### State on the card, and recovery after a reboot

| File | Contents | Written |
| --- | --- | --- |
| `DEVICE.TXT` | Device number and the clock's UTC offset. | On `ID:` and `TIME:`. |
| `SCHED.TXT` | The schedule in force. | On upload during a run, and at `RUN:BEGIN`. |
| `NEXT.TXT` | A schedule uploaded outside a run, waiting for `RUN:BEGIN`. | On upload outside a run. |
| `RUN.TXT` | Run id, t0, data file, ended flag, current phase number and phase start time. | At `RUN:BEGIN` and `RUN:END`, and at each phase change. |
| `<run id>.CSV` | The run's rows. Before the first run ever, rows go to `IDLE.CSV`. | Every event. |

Every file except the data file is written to a temporary name and then
renamed, so a power cut leaves either the old version or the new one, never a
partial file.

**On boot, the device rebuilds its state like this:**

| What | Where it comes from |
| --- | --- |
| Phase and its start time | `RUN.TXT` |
| Totals and phase counts | the last row of the data file |
| Quota used in the current period | `phase_rewards` if `quota_period=0`; otherwise a scan back through rows to the start of the period, which is bounded by the period |
| Partial progress towards a ratio | Not kept: the ratio counter restarts at 0, and the `boot` row says so |

It then catches up as described above.

**After `RUN:END`** the device stays in its current phase, including its daily
windows, so timed feeding keeps working. No more phase changes happen, and the
device keeps logging to the same data file until the next `RUN:BEGIN`.

---

## Part 3: serial commands (firmware v3)

These are newline-terminated ASCII lines at 115200 baud.
- Input lines can be up to 128 bytes.
- Every device line is `TYPE,` followed by comma-separated `key=value` fields,
  except `ROW`, which carries an SD row verbatim.
- `ERR,<code>,<detail>`: the detail is the rest of the line and may contain
  commas.
- A command that writes rows sends them before its reply, so `ROW` lines can
  arrive between a command and its answer.
- Every command that changes state is safe to repeat, except `FEED`: sending
  it twice has the same effect as sending it once, so FNT can always retry
  after a timeout. A repeated `RUN:BEGIN` for the run already in progress
  changes nothing.

### Host → device

| Command | Reply | Purpose |
| --- | --- | --- |
| `PING` | `PONG,id=<n>,fw=3.0,proto=3` | Discovery. FNT refuses `proto` ≠ 3. |
| `STATUS` | `STATUS,…` (below) | Full state. FNT polls it every 30 s. |
| `TIME:<iso with offset>` | `TIME,before=<iso>,after=<iso>` | Set the clock and its UTC offset. During a run a jump over 10 min is refused with `ERR,TIME_JUMP`; `TIME!:` overrides that. |
| `ID:<n>` | `ID,id=<n>` | Set and save the device number. This replaces the boot-time ID menu. |
| `SCHED:BEGIN` | `OK` | Start receiving a schedule. |
| `S:<line>` | `OK` or `ERR,SCHED,<line no>,<reason>` | One schedule line, checked as it arrives. |
| `SCHED:COMMIT,<crc>[,<phase>]` | `SCHED,crc=<hex>,phases=<n>,slot=<now\|next>` | Commit the upload (see below). |
| `RUN:BEGIN,<run_id>,<t0 iso>,<crc>` | `RUN,run=<id>,file=<name>,phase=0` | Start a run (see below). |
| `RUN:END` | `RUN,run=<id>,ended=1` | Log `run_end`. Phase changes stop; the current phase and its windows continue (Part 2). |
| `GOTO:<from>,<to>` | `PHASE,n=<to>,name=<name>` | Move from phase `from` to phase `to`. Ignored with `ERR,GOTO_STALE` if the device is no longer in `from`, so a delayed or repeated `GOTO` can't skip a phase or move a device backwards. |
| `READ:<run_id>,<from_seq>` | `ROW,…` lines, then `READ_END,run=<id>,last=<seq>` | Re-send rows from `from_seq` onward. These interleave safely with live rows. |
| `FEED` | `FEED,result=<started\|pellet_in_well\|jam>` | Dispense once by hand. The outcome follows as a `dispense` (`detail=manual`) or `jam` row. Counts as a pellet but not a reward. |
| `TEST:<cue>` | `OK` | Play a cue with the same syntax as `cue.*`. |
| `REBOOT` | (the device restarts) | Remote recovery; a `BOOT` line follows. |

**`SCHED:COMMIT`**
1. The device checks the crc. It refuses with `ERR,SCHED_REJECTED` if the crc
   is wrong or any line failed, and nothing changes.
2. **Outside a run,** the schedule goes to `NEXT.TXT` and takes effect at the
   next `RUN:BEGIN`. The device keeps running what it was running.
3. **During a run,** the schedule replaces `SCHED.TXT` straight away. The device
   keeps its phase number and phase start time, and re-checks the `UNTIL`
   conditions immediately.
   - `<phase>` is the phase FNT believes the device is in. If the device has
     moved on in the meantime, the commit is refused with `ERR,SCHED_STALE`;
     FNT re-checks the edit and sends it again.
4. A partial upload is discarded 10 s after its last line.

**`RUN:BEGIN`**
1. The device refuses with `ERR,RUN_SCHED` unless its pending schedule (or, if
   none is pending, its schedule in force) has crc `<crc>`. That guarantees the
   run starts on the schedule FNT expects.
2. If a run is already active, the device ends it first (`run_end` in the old
   file).
3. It promotes `NEXT.TXT`, zeroes all counters, opens `<run id>.CSV`, logs
   `run_start`, and enters phase 0 with t0 as its start time.

Run ids are 8 characters, so file names stay valid on any FAT card.

### Device → host

| Line | When |
| --- | --- |
| `ROW,<SD row>` | Every row written to the card, the moment it is written, and again in reply to `READ`. |
| `BOOT,fw=…,reset=<power\|watchdog\|software\|external>,clock=<ok\|unset>,sched=<crc\|default>,run=<id\|->` | At startup. |
| `STATUS,id=,fw=,time=,clock=ok\|unset,sd=ok\|err,run=,ended=0\|1,sched=,next_sched=,phase=,phase_name=,phase_start=,next=,file=,seq=,l=,r=,pellets=,phase_rewards=,phase_left=,phase_right=,depleted=0\|1,well=0\|1,jam=0\|1,uptime=,batt=,temp=,hum=` | In reply to `STATUS`. `sched` and `next_sched` are crcs (`-` if none). `next` describes the soonest `UNTIL`, for example `at=2026-10-20T19:00-07:00` or `rewards=37/100`. |
| `ERR,<code>,<detail>` | Bad command or bad value, or an SD write failure (`ERR,SD,…`). |

`crc` is CRC-32 (the `zlib.crc32` polynomial) of the schedule's lines, each
ending in `\n`. FNT compiles the same text, so a single number shows whether a
device holds the schedule FNT expects.

### Data file

One file per run, with this header:

```
time,ms,seq,phase,event,side,outcome,left,right,pellets,phase_rewards,phase_left,phase_right,retrieval_ms,poke_ms,detail
```

| Column | Meaning |
| --- | --- |
| `time` | ISO 8601 with UTC offset, to the second (RTC resolution). Blank while `clock=unset`. |
| `ms` | `millis()`, for ordering and intervals within a second. It wraps after 49.7 days, so only differences between nearby rows mean anything. |
| `seq` | Monotonic within a run, starting at 1. |
| `phase` | The phase number when the row was written. |
| `event` | `poke`, `dispense` (a pellet reached the well), `retrieve`, `jam`, `phase`, `schedule`, `run_start`, `run_end`, `boot`, `clock`, `status`, `error`. |
| `side`, `outcome` | For pokes. `side` is `L` or `R`. `outcome` is `reward`, `counted`, `inactive`, `timeout`, `short`, `pellet_in_well` or `depleted`. |
| `left`, `right`, `pellets` | Totals for the run. |
| `phase_rewards`, `phase_left`, `phase_right` | Totals for the phase, which `UNTIL` and the quota use. |
| `retrieval_ms` | On `retrieve` rows: the time from dispense to retrieval, however long it was. Nothing times out; the row is written whenever the well clears. |
| `poke_ms` | On `poke` rows: how long the beam was broken. |
| `detail` | `;`-separated, never commas. `phase`: the name, plus `;goto` or `;catch-up;start=<iso>`. `dispense`: `free`, `manual` or `retry` (blank for a reward). `jam`: what was being dispensed. `boot`: `reset=<cause>;ratio counters reset`. `clock`: `from=<iso>;to=<iso>`. `run_start`: `t0=<iso>;<first phase>`. `status` (hourly): `batt=…;temp=…;hum=…`. `poke`: `held` if the beam stayed broken past 60 s. |

### Firmware changes needed to keep these promises

- **Hardware watchdog**, with every wait bounded well below its period. The
  library's `logLeftPoke`, `logRightPoke` and `RotateDisk` contain unbounded
  `while (beam broken)` loops and are replaced. The watchdog is also fed during
  the boot-time back-scan.
- **Jam handling:** a bounded number of attempts, then a `jam` row, then a
  retry every 10 minutes rather than giving up for good.
- **SD failure:** if the card can't be read, the device stays in the current
  phase (it can't read the next one), keeps running, reports `sd=err`, and
  retries.
- **Output:** whole lines go out with `Serial.write(buf, len)`, with no
  `if (Serial)` checks, which cost 10 ms each.
- **SD:** the card is initialised once at boot instead of on every row, and
  there is no 100 ms LED blink per row.
- **Display:** shows the phase name, the counts and the next change, and is
  redrawn on change only.
- **LEDs:** turned off and the motor-driver enable line released after every
  flash.
- **Removed:** the 14 modes, the `MODE`/`NEW_TRIAL`/`SYNC`/`LIGHTS`/file-transfer
  commands, the event queue, the pending-command slot, the boot ID menu and
  `start.csv`/`stop.csv`.

---

## Part 4: protocol files

A protocol is a YAML file. It is what the FNT editor writes, what a methods
section can quote, and what the tests run without hardware. FNT compiles it
into one schedule per device and stores the file, with its hash, in every run
folder.

### Structure

```yaml
format: fnt-fed3-protocol/1
name: Short name
description: Free text.
devices: [A, B]          # logical names; bound to physical FEDs at Start

colors:                  # optional; added to the built-in names
  amber: [30, 10, 0, 0]

cues:                    # optional named cues
  reward_flash:
    - tone: {hz: 4000, ms: 200}
    - flash: {target: bar, color: blue, ms: 500}

rules:                   # optional named rules
  base:
    active: both
    cue: {reward: reward_flash}
  rich: {use: base, quota: 30, quota_period: 1h}

phases:                  # run in order; the last one runs until Stop
  - name: …
    until: …             # when this phase ends (omit on the last phase)
    all: …               # rule for every device
    devices: {A: …}      # per-device rule
    rotate: …            # reassign rules between devices periodically
    daily: […]           # clock-time windows inside this phase

checks:                  # optional; what the GUI flags as a problem
  no_pellets: 8h
```

### Rules in the file

A rule is either a mapping of the keys from Part 1, or the name of an entry in
`rules`. `use: <name>` inside a mapping starts from that named rule; the keys
written beside it override it.

`cue.*` keys are written as a nested `cue:` mapping. A cue is a named cue, or a
list of actions where each action is either a one-key mapping or `click`:

```yaml
cue:
  reward:
    - tone: {hz: 4000, ms: 200}
    - flash: {target: poke, color: green, ms: 300}
  counted: [click]
  timeout:
    - noise: {ms: 300}
  depleted:
    - bnc: {width: 10, hz: 1, count: 1}
```

**Built-in colours:** `white` (0:0:0:40), `red` (30:0:0:0), `green` (0:30:0:0)
and `blue` (0:0:30:0). Colours can also be written as `[r, g, b, w]`.

### Durations, clock times and conditions

- **Durations** are `90s`, `15m`, `48h`, `10d`, or combinations such as
  `1d12h`. The rule keys `timeout`, `quota_period` and `refill_delay` also
  accept a bare number of seconds. `after` and `every` need a unit, so
  `after: 48` can't silently mean 48 seconds.
- **Clock times** are `"19:00"`, quoted because YAML would otherwise read
  `19:00` as the number 1140. They are in standard time (see
  [Clock](#clock)).

`until` takes one condition, or a list of conditions where the first one met
ends the phase:

| Condition | Ends the phase | Checked by |
| --- | --- | --- |
| `{after: 48h}` | 48 h after the phase started. | Device |
| `{at: "19:00"}` | At the first 19:00 after the phase started. | Device |
| `{at: "19:00", day: 3}` | At 19:00 on day 3 of the run. | Device |
| `{rewards: 200}` | When this device has delivered 200 rewards in this phase. | Device |
| `{pokes: 500, side: any}` | After 500 pokes on this device in this phase (`side` can be `left`, `right` or `any`). | Device |
| `{manual: true}` | When someone presses **Advance phase**. | FNT |
| `{rewards: 200, devices: all}` | When every device (or `any` device, or a list such as `[A, B]`) has delivered 200 rewards in this phase. | FNT |

### Conditions that need FNT

A device only knows its own counts. A condition that depends on other devices,
or on a person pressing **Advance phase**, has to be checked by FNT.

- **How FNT does it:** it watches `phase_rewards`, `phase_left` and `phase_right` in each
  device's live rows. When the condition is met, it sends `GOTO:<n>,<n+1>` to
  every device still in phase *n*. Devices that already left by a local
  fallback are untouched.
- **If FNT is down,** such a phase doesn't end until FNT is back, unless it
  also has a device-local condition. `until: [{rewards: 100, devices: all},
  {after: 4d}]` ends at whichever comes first, and the 4-day limit still
  applies with FNT down.
- **The checker and editor** mark every phase that needs FNT, and warn when it
  has no device-local fallback.

### Rotation

`rotate` reassigns named rules between devices, one period at a time, which
covers patch swaps and rich/poor reshuffles:

```yaml
rotate:
  every: 1d
  at: "19:00"          # optional, and only with every: 1d
  mode: cycle          # cycle: each device takes the next one's assignment
                       # shuffle: random permutation, seeded per run
  assign: {A: rich, B: rich, C: poor, D: poor}   # the first period
```

FNT expands a rotation into consecutive phases (`Competition 1`,
`Competition 2`, …), one per period, with each device given its own
assignment. The device needs no rotation logic of its own. The shuffle seed is
saved in `run.json`, so the sequence can be reproduced.

**A rotating phase lasts a whole number of periods.** Its `until` must be
`{after: …}` and a multiple of `every`: `{after: 14d}` with `every: 1d` means 14
periods. Each expanded phase ends at `at=<time>`, or at `after=<every>` if
there's no `at`. Counting periods matters because a rotating phase usually
starts when an earlier condition is met, so its start time isn't known when the
schedule is compiled. With `at`, the first period runs only from the phase
start to the next boundary.

### Daily windows

```yaml
daily:
  - {from: "07:00", to: "19:00", all: {free_feed: 1}}
  - {from: "20:00", to: "22:00", devices: {A: {ratio: 3}}}
```

A window overrides keys from the phase's rule between two clock times every
day, for every device (`all`), for some (`devices`), or both. A window may cross midnight. Windows within a phase may not overlap; this
is checked when the file is loaded.

### Merge order

Each phase's rule for a device is built by merging, in this order (later wins):
1. the built-in defaults;
2. the phase's `all`;
3. the phase's rotation assignment for that device;
4. the phase's `devices` entry for that device.

`daily` windows become `WINDOW` lines.

### Checks

`checks.no_pellets` flags a device that hasn't dispensed a pellet for that
long, while its current rule can dispense (`free_feed=1`, or `reward=pellet`
with `active` other than `none`). The other problems the GUI flags are always
on (Part 5).

### Command-line tools

The compiler that produces the schedules is also available from the command
line:

- `fnt-fed check protocol.yaml` finds errors with line numbers:
  - unknown keys;
  - out-of-range values;
  - overlapping windows;
  - a phase that can never end;
  - a device name that isn't declared;
  - a schedule over the device limits.

  It also lists the phases that need FNT.
- `fnt-fed compile protocol.yaml` prints each device's schedule.
- `fnt-fed preview protocol.yaml --days 21` prints the timeline. Count
  conditions are shown as "after 200 rewards" rather than given times.

---

## Part 5: runs

```
Setup ──Start──▶ Pre-flight ──all pass──▶ Running ──Stop──▶ Stopped
                     │ fail
                     └──▶ back to Setup, with the reason shown on each device
```

**Setup**
- Choose the protocol and output folder, and bind each logical device to a
  physical FED by ID.
- Manual `FEED` and `TEST` work here for bench checks.

**Pre-flight**, for each bound device:
- it is connected, with `proto=3`;
- `clock=ok`, and the drift is recorded after setting the clock;
- `sd=ok`;
- its compiled schedule uploads to `NEXT.TXT`, and the crc matches;
- optionally, it plays a test cue.

A device that fails blocks Start unless you remove it from the run explicitly.

**Start** happens at a single instant, t0:
- FNT sends `RUN:BEGIN,<id>,<t0>,<crc>` to every device;
- FNT waits for every `RUN` reply before showing *Running*;
- counters, plot, log and the elapsed clock all start at t0.

**Running**
- FNT polls `STATUS`, keeps a live copy of the rows and fills any gap with
  `READ`.
- FNT checks the conditions that need it.
- FNT flags problems on screen.
- Manual actions are logged with their source.

**Editing mid-run**
1. Edit the protocol in FNT.
2. FNT checks the edit:
   - phases before each device's current phase must be unchanged;
   - the current phase must keep its position, but its rule and conditions may
     change.
3. FNT recompiles and commits to each device, with the device's current phase
   (`ERR,SCHED_STALE` means "re-check and resend").
4. The change takes effect immediately; elapsed time and counts carry over.

Each version is kept in the run folder (`protocol.v2.yaml`, …), and `run.json`
records when it took effect.

**Stop**
- FNT sends `RUN:END`, pulls the remaining rows and writes a summary.
- The devices keep running their current phase until the next run.

**Crash recovery:** if FNT finds an unfinished run at launch, it resumes it:
- it rebinds the devices;
- it compares the schedule crcs;
- it `READ`s the missing rows.

The devices were never waiting for it.

### What the GUI flags

There are no external notifications: rooms are checked daily, so a problem
only has to be impossible to miss on screen. Each device card shows a problem
state, and the run header shows a count of open problems.

**A device card flags:**
- no reply for 2 minutes;
- `jam`;
- watchdog reset;
- `sd=err`;
- `clock=unset`;
- schedule crc mismatch;
- clock drift over 60 s;
- the device being in a different run;
- `checks.no_pellets`.

A problem stays flagged until you acknowledge it, even if it has cleared, so
an overnight jam is still visible at the morning check.

### Run folder

```
2026-10-20_1900_four-patch/
  run.json          # protocol hash, binding, t0 (UTC), clock offset, seed,
                    # edits, FNT-driven phase changes, firmware versions
  protocol.yaml     # exact copy (plus protocol.v2.yaml … after edits)
  sched/P1.txt      # each device's compiled schedule
  log.csv           # FNT actions, uploads, problems, disconnects (UTC)
  P1_FED003.csv     # byte-identical copy of the device's data file
  P2_FED007.csv
  …
```

FNT writes each device's copy in `seq` order, holding rows that arrive early
until `READ` fills the gap. The live view uses rows as they arrive.

---

## Part 6: the GUI

The GUI has one job per run state. Everything the devices and FNT can do on
their own (clock sync, data copying, port scanning, recovery) has no button.

### Code layout

```
fnt/fed3/
  schedule.py     # the device schedule format, limits and timing rules
  protocol.py     # parse, check, compile, preview. Pure Python, no Qt.
  device_sim.py   # a FED3 on firmware v3, simulated: the firmware's spec
  link.py         # one serial port and the v3 line protocol (no threads)
  run.py          # devices, runs, data copies, FNT conditions, problems (no Qt)
  demo.py         # simulated devices for trying the window without hardware
  cli.py          # fnt-fed check | compile | preview
  ui/             # window, setup and running views, editor, plot
```

- **`run.py` has no widgets and no threads of its own.** Everything happens in
  `Engine.step()`; the window runs it on one background thread, sends it
  commands through a queue and reads snapshots. The tests drive `step()`
  directly against simulated devices on a simulated clock.
- **Separate process:** the FED3 window is a launcher tool like the others
  (`fnt.tool_launcher`, key `fed3`), so a crash takes down only the window, and
  the devices don't notice either way.
- **Demo mode:** `python -m fnt.fed3.ui.window --demo 4 --speed 60` opens the
  window with four simulated FED3s and an animal poking at each, a minute per
  second.

### Setup

```
┌ FED ─ Setup ───────────────────────────────────────────────────────────────────┐
│ Protocol: four-patch.yaml  [Open] [New] [Edit]     ✓ checks pass · 17 phases  │
│ ┌ Timeline ────────────────────────────────────────────────────────────────┐   │
│ │ Habituation ▸ day 3 19:00 │ Training ▸ 100 rewards (all) or 4d │ Comp…   │   │
│ └──────────────────────────────────────────────────────────────────────────┘   │
│ Devices          FED            status                         bench          │
│  P1   [FED003 ▾] ● connected · clock ok · SD ok · fw 3.0   [Feed] [Test cue] │
│  P2   [FED007 ▾] ● connected · clock ok · SD ok · fw 3.0   [Feed] [Test cue] │
│  P3   [FED002 ▾] ▲ SD error                                 [Feed] [Test cue] │
│  P4   [FED011 ▾] ● connected · clock ok · SD ok · fw 3.0   [Feed] [Test cue] │
│ Save to: ~/FED3_Runs                                         [ Start run ▶ ]   │
└────────────────────────────────────────────────────────────────────────────────┘
```

- Unbound FEDs appear in the drop-downs automatically. A FED can be renumbered
  from its entry (`ID:`).
- **Start** runs pre-flight and reports per row, so a failure is shown on the
  row that caused it.

### Protocol editor

The file is the protocol, so the editor edits the file:
- **The YAML text** on the left. Nothing is converted to and from a form, so
  comments and layout survive.
- **A live check** on the right, re-run as you type. It uses the same code as
  `fnt-fed check`, and clicking a problem jumps to its line.
- **The timeline** below it: when each phase starts and ends, each device's
  rule, and which phases need FNT.
- **New from template:** FR1 on both pokes with a cue; stepping the ratio up by
  rewards; timed feeding; a four-patch competition.
- **Add phase…:** a form for the common settings (end condition, active poke,
  ratio, timeout, quota, free feeding, reward tone and flash). It writes the
  YAML for you, so a new phase never needs typing.
- **During a run** the editor opens on the running protocol and **Apply to
  running devices** sends it. The engine refuses an edit that changes a
  finished phase or moves a device's current phase.

A form for every construct (rotation, windows, per-device overrides), or a
Blockly or node-graph editor, could be added later as other ways to write the
same file. Nothing else would change.

### Running

```
┌ Four-patch competition ── ● RUNNING  6d 03:12 ── 3 problems ── [Edit] [Stop] ┐
│ Training ▸ Competition 3/14 ▸ Recovery     next: Competition 4 at 19:00 (2h47m) │
├───────────────────────────────┬────────────────────────────────────────────────┤
│ P1 FED003  ● Competition 3    │ cumulative rewards, per device                 │
│  rich · FR1 · quota 12/30 /h  │ (standard time; dark phase shaded)             │
│  L 912  R 340  rewards 801    │                                                │
│  last: retrieve 40 s ago      │                                                │
│ P3 FED002  ▲ jam 02:14 [ack]  │                                                │
│  poor · FR5 · depleted        │                                                │
│ …                             │                                                │
├───────────────────────────────┴────────────────────────────────────────────────┤
│ 19:00 all → Competition 3 · 02:14 P3 jam · 08:02 you: Feed P3 · 08:03 ack jam  │
└────────────────────────────────────────────────────────────────────────────────┘
```

- **Header:** state, elapsed time since t0, open problems, **Edit** and
  **Stop**. **Advance phase** appears only when the current phase has a manual
  condition.
- **Device cards:**
  - the current phase;
  - a summary of the rule generated from its keys;
  - counts since t0;
  - the last event;
  - problems with an **Acknowledge** button.

  **Feed** and **Test cue** are in each card's menu.
- **Plot:** cumulative pellets per device against standard time, with phase
  changes marked.
- **Log:** phase changes, problems, uploads and manual actions.

**Stopped** shows when the run stopped, pellets per device, **Open folder** and
**New run**.

---

## Examples

### 1. Both pokes, stepping up the ratio by count

```yaml
format: fnt-fed3-protocol/1
name: FR1 to FR5, both pokes
devices: [FED]

cues:
  reward:
    - tone: {hz: 4000, ms: 200}
    - flash: {target: poke, color: blue, ms: 500}

rules:
  base: {active: both, cue: {reward: reward}}

phases:
  - name: FR1
    until: {rewards: 50}
    all: {use: base, ratio: 1}
  - name: FR3
    until: {rewards: 100}
    all: {use: base, ratio: 3}
  - name: FR5
    all: {use: base, ratio: 5}
```

This compiles to:

```
NAME FR1 to FR5, both pokes
PHASE FR1
UNTIL rewards=50
SET active=both cue.reward=tone:4000:200+flash:poke:0:0:30:0:500
PHASE FR3
UNTIL rewards=100
SET active=both ratio=3 cue.reward=tone:4000:200+flash:poke:0:0:30:0:500
PHASE FR5
SET active=both ratio=5 cue.reward=tone:4000:200+flash:poke:0:0:30:0:500
```

### 2. Four-patch competition with depletion and daily reshuffle

```yaml
format: fnt-fed3-protocol/1
name: Four-patch competition
description: >
  Two rich and two poor patches that each refill hourly. Which patches are
  rich is reshuffled every day at lights-off.
devices: [P1, P2, P3, P4]

cues:
  reward:
    - tone: {hz: 4000, ms: 200}
    - flash: {target: bar, color: green, ms: 1000}

rules:
  patch:
    active: both
    cue: {reward: reward, depleted: [click]}
  rich: {use: patch, ratio: 1, quota: 30, quota_period: 1h}
  poor: {use: patch, ratio: 5, quota: 10, quota_period: 1h}

phases:
  - name: Habituation
    until: {at: "19:00", day: 3}
    all: {active: none, free_feed: 1}

  - name: Training
    until: [{rewards: 100, devices: all}, {after: 4d}]
    all: patch

  - name: Competition
    until: {after: 14d}
    rotate:
      every: 1d
      at: "19:00"
      mode: shuffle
      assign: {P1: rich, P2: rich, P3: poor, P4: poor}

  - name: Recovery
    all: {active: none, free_feed: 1}

checks:
  no_pellets: 8h
```

Training is the only phase that needs FNT, and it falls back to 4 days on its
own. Each device's schedule has 17 phases: Habituation, Training, Competition
1–14 and Recovery. Every Competition phase ends at the next 19:00, so the first
one is short unless Training happens to end at 19:00.

### 3. Timed feeding

```yaml
format: fnt-fed3-protocol/1
name: Timed feeding 19:00-21:00
devices: [FED]
phases:
  - name: Restricted
    all: {active: none}
    daily:
      - {from: "19:00", to: "21:00", all: {free_feed: 1}}
```

The window runs on the device, so it opens and closes on time with the PC off.
It also keeps working after Stop.

---

## Decisions

| Question | Decision |
| --- | --- |
| Legacy FED3 compatibility | None: new modes, commands and file format. |
| Lights | Off unless a cue flashes them. |
| Both pokes | One shared counter by default; `count=per_port` is a setting. |
| Clock | Standard time all year, matching a vivarium that ignores daylight saving. |
| Blank card | FR1, left poke. |
| After Stop | The device keeps running its current phase, including daily windows. |
| Where the schedule runs | On the device; FNT only checks conditions that span devices. |
| Notifications | None. The GUI flags problems until they're acknowledged. |
| `status` rows | Hourly. |
| Limits | 256 phases, 8 daily windows and 6 `until` lines per phase. |
| Mid-run edits to the current phase | Take effect immediately. |
