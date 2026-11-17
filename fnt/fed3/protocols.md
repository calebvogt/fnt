# FED3 protocols with several FED3s

This file lives beside the code for now. Once the format settles it belongs in
`docs/fed3/` with the [user guide](../../docs/fed3/guide.md) and the
[format reference](../../docs/fed3/design.md) (Part 4).

The first half explains how a protocol runs several FED3s, in the editor and in
the file. The second half is a review of YAML as the protocol format: what
works, what gets in the way, and what to change.

## How a protocol runs several FED3s

Written in format 3 (`format: fnt-fed3-protocol/3`). Formats 1 and 2 still
load; format 2 called `per_device` `only`.

Five words carry it:

| Word | Where | Means |
| --- | --- | --- |
| `devices` | top of the file | The FED3s in the run. Every one runs every phase. |
| `every_phase` | top of the file, optional | What every phase starts from, so shared settings are written once. |
| `rule` | in a phase | What every FED3 does in that phase, on top of `every_phase`. |
| `per_device` | in a phase, beside `rule` | What some FED3s do differently. Keys it doesn't mention come from `rule`. |
| `of` | in an `until` condition | Whose counts end the phase: `each` FED3 on its own, `all` of them, `any` one, or a list. |

A phase's rule for one FED3 is the built-in defaults, then `every_phase`, then
the phase's `rule`, then (with a rotation) its assignment, then its
`per_device` entry. Nothing carries over from the previous phase.

Ratios are written `FR5`, `RR3` or `PR`, and the classic FED3 reward cue
(4 kHz tone, blue light bar) is called `classic`.

### In the editor

1. **Setup → New…** opens the editor with the FED3s that are plugged in. To
   change them, click **Devices: …** in the toolbar and type IDs or ranges
   (`1-4`, `FED001, FED007`).
2. **New from template** fills in a template for those FED3s.
3. **Add phase…** (or **Edit phase**) has a **Per device** table, one row per
   FED3. A row left on `same` runs the settings above it. Set a row's active
   poke, ratio or reward to make that FED3 differ; **Runs as** shows the
   result (`FR1 · both`, `RR2 · both`). With an `every_phase`, a new phase
   starts from it and the form writes only what differs.
4. The timeline on the right lists each FED3's rule when they differ, and
   **What each FED3 is sent** shows the exact schedule.

The table covers the active poke, the ratio and the reward. Anything else that
differs per FED3 (a timeout, a quota, a cue, free feeding) is written in the
text, and the form says so instead of dropping it.

### In the text

FR1 on one FED3 and random ratios on three others, all with the same tone and
flash on a reward:

```yaml
format: fnt-fed3-protocol/3
name: FR1, RR2, RR3 and RR4, both pokes
devices: [FED001, FED002, FED003, FED004]
phases:
  - name: FR1 and RR
    rule:
      active: both
      cue:
        reward:
          - tone: {hz: 4000, ms: 200}
          - flash: {target: poke, color: blue, ms: 500}
    per_device:
      FED002: {ratio: RR2}
      FED003: {ratio: RR3}
      FED004: {ratio: RR4}
```

FED3s with the same change can share a key: `FED003, FED004: {ratio: FR5}`.
Inside `{ }` such a key needs quotes, `{"FED003, FED004": {ratio: FR5}}`,
because there a comma separates entries.

With several phases, put what they share in `every_phase`:

```yaml
every_phase:
  active: both
  cue: {reward: classic}
phases:
  - name: Training
    until: {rewards: 50, of: each}
    rule: {ratio: FR1}
  - name: Test
    per_device:
      FED001: {ratio: FR1}
      FED002, FED003, FED004: {ratio: RR3}
```

Ending a phase on counts needs `of` once there are several FED3s:

```yaml
until: {rewards: 50, of: each}     # each FED3 moves on alone at 50
until: {rewards: 50, of: all}      # all move on together once each has 50 (needs FNT running)
until: [{rewards: 50, of: all}, {after: 4d}]   # ...or after 4 days, FNT or not
```

### Mistakes and what the checker says

| Written | The checker says |
| --- | --- |
| `per_device:` indented under `rule:` | per_device belongs to the phase, not inside its rule: move it left to line up with rule |
| `only:` in a format 3 file | unknown key 'only' in a phase; in fnt-fed3-protocol/3 it is called 'per_device' |
| `per_device: {FED001, FED002: {ratio: 2}}` | FED001 has no rule; inside { }, a key naming several FED3s needs quotes |
| `until: {rewards: 10}` with several FED3s | (warning) they can end up in different phases; write of: each or of: all |
| `until: rewards: 10` | not valid YAML: … two keys on one line need braces (until: {rewards: 100}) … |
| `at: 19:00` | write clock times in quotes, e.g. "19:00" |
| `active: no` | active can't be true/false (YAML reads yes, no, on and off that way); for no active poke write none |
| A tab for indent | not valid YAML: a tab; indent with spaces |
| `ratio:3` | 'ratio:3' needs a space after the colon: ratio: 3 |
| `ratio: VR3` | ratio is a whole number, or FR, RR or PR as in FR5, RR3 or PR |
| `per_device` with no `rule` and no `every_phase` | (warning) the FED3s it leaves out run the defaults (FR1 on the left poke) |

## Review: is YAML the right protocol format?

This review was written against format 2, and its examples use format 2's
`only`. Format 3 acts on recommendations 2 to 6 below. What each change
addresses is marked under **What gets in the way**.

### Verdict

Keep the YAML file as the protocol, and keep the reasons it was chosen: it's
plain text, a methods section can quote it, git can diff it, every run folder
stores it with its hash, and the tests run it without hardware. Nothing else
on the table does all of that.

But the file shouldn't be how most people write a protocol. Someone who
has never edited YAML should be able to set up a common multi-FED3
experiment entirely in the editor, and read the file afterwards as a record.
Today the format asks for a few things that trip people up, mostly where
several FED3s meet. Those are fixable without leaving YAML.

### What works

- **The checker.** Most mistakes get a message in the protocol's own terms,
  with a line number and often a "did you mean" (see the table above). A
  clock time without quotes or a duration without a unit is caught, not
  silently misread.
- **It reads as the experiment.** `until: {after: 48h}`, `active: both`,
  `quota: 30` and `quota_period: 1h` say what they do.
- **Every phase states its whole rule.** A phase means the same thing
  wherever it sits, and a phase can be read on its own.
- **The editor already writes most of it.** Templates, the phase form and the
  live timeline mean a common protocol needs little typing.
- **Many researchers have seen YAML.** Anyone who has edited a DeepLabCut
  `config.yaml` will recognise the look.

### What gets in the way

1. **Several FED3s add three concepts at once.** `devices`, `only` and `of`
   all arrive with the second FED3, plus `rotate.assign` and `daily[].only`
   for the same idea in other places. Before this branch the form couldn't
   write `only` at all, so anything per-FED3 meant writing YAML by hand.
   *Now: the editor's Devices control and per-device table.*
2. **`only` reads the wrong way.** `only: {FED002: {ratio: 2}}` can be read
   as "only FED002 runs this phase", not "FED002 differs like this".
   Format 1's `devices:` had the opposite problem: it clashed with the
   top-level list. *Now: `per_device` in format 3.*
3. **The field's names for schedules take two keys.** People say FR5, RR3 and
   PR, and FNT prints them that way in the timeline, but the file wants
   `ratio_kind: random` plus `ratio: 3`. Getting one key wrong gives a
   different schedule that still passes the checker (`ratio: 3` without the
   kind is FR3). *Now: `ratio: RR3`, `FR5`, `PR`, which the editor writes.*
4. **Cues nest four levels deep.** `cue:` → `reward:` → a list → a one-key
   mapping → its parameters. It's the most-copied block and the hardest to
   type correctly. The classic 4 kHz tone and blue light has no short name.
   *Now: `cue: {reward: classic}`. Custom cues still nest.*
5. **Every phase restates the cue.** Since phases don't inherit, a
   five-phase protocol repeats the same cue five times unless the writer
   knows about `rules:` and `use:`. Leaving it out of one phase silently
   removes the cue there. *Now: `every_phase`.*
6. **YAML's own traps.** Indentation decides meaning (`only` one level too
   deep belongs to `rule`), tabs are invalid, `no`/`off` are booleans,
   clock times need quotes, and a key with a comma needs quotes inside `{ }`
   but not outside. FNT reports the first two as raw PyYAML errors
   (`expected <block end>, but found '<block mapping start>'`). *Now: those,
   `ratio:3`, a key indented into `rule` and `active: no` get plain messages.
   The traps themselves remain.*
7. **One mistake can produce two errors.** A bad `until` value is reported,
   and so is "'a' never ends", which comes first in the list and points at
   the wrong problem. *Now: only the `until` error is reported.*
8. **Defaults aren't obvious.** A phase that leaves out `active` gets `left`.
   The classic templates use the left poke, the examples use both, and
   nothing in the file says which one applies. *Partly: the form always
   writes `active`, and `every_phase` can state it once. The default is
   still `left`.*

### Is it approachable for other researchers?

As a record to read: yes. Someone who knows FED3 can read a protocol and say
what it does.

As something to write from scratch: not yet for multi-FED3 work. A one-FED3
FR or PR protocol is a template and a couple of edits. The four-FED3 FR1/RR
protocol above needed `only`, `ratio_kind` and the cue block by hand before
this branch. Now it can be made in the editor, and by hand it's
`per_device` with `ratio: RR2`. Whether a first-time user gets it right
without the reference open is what the test below would show.

For comparison, the usual alternatives are Arduino code (classic FED3
modes), MED-PC's MSN language, or MATLAB/Python state machines (Bpod). A
checked YAML file is easier than any of those. The bar is the mode menu on
the FED3 itself, where choosing FR1 takes a few button presses.

### Other formats considered

- **TOML** removes indentation, but the nested cues and conditions become
  `[[phases.until]]` tables, which are harder to read. Not worth a migration.
- **JSON** is worse to type by hand, with no comments.
- **A spreadsheet** (a row per phase, a column per FED3) is the most familiar
  shape and fits "which FED3 runs what, when". It can't hold cues,
  conditions or windows, though. It works as a view or an import, but not as
  the format.
- **A Python script** gives the most power and is the least approachable.
- **The GUI alone, with no file**, loses the record, the diffs and the
  hash.

### Recommendations

In order of value for effort. 1 to 6 are done in this branch.

1. **Done: per-FED3 settings in the editor.** A Devices control, templates
   that take the protocol's FED3s, and a per-device table in the phase form
   that reads and writes `per_device` (`only` in a format 2 file).
2. **Done: the field's shorthand.** `ratio: RR3`, `ratio: FR5` and `ratio: PR`
   work in every format alongside the long form, and the editor writes the
   short one. Writing it beside `ratio_kind` is an error.
3. **Done: the classic cue.** `classic` is built in, and the classic
   templates use it. (`click` was already built in.) Format 3 files can't
   redefine it; older files that defined their own keep theirs.
4. **Done: plain YAML errors.** Tabs, a line out of line, two keys on one
   line, a `{ }` missing a comma, an unclosed quote, `ratio:3`, a key
   indented into `rule`, and `active: no` each get a message saying what to
   do. A bad `until` no longer also reports "never ends".
5. **Done: `every_phase`.** A top-level rule each phase starts from, merged
   before the phase's `rule`. The form starts new phases from it and writes
   only what differs, including `cue: {reward: none}` to drop an inherited
   cue.
6. **Done: `per_device`.** Format 3 renames `only`. Formats 1 and 2 still
   load, and the checker names the right key when a file uses another
   format's.
7. **A phases × FED3s grid view.** Show the protocol as a table, one row per
   phase and one column per FED3, with cells like `RR2 · both`, and allow
   editing cells. That's the shape people sketch on paper, and the timeline
   already has the data.
8. **A methods paragraph.** `fnt-fed describe protocol.yaml` would turn the
   compiled protocol into prose for a methods section, using the same
   summaries the timeline uses.

### How to find out

None of the above has been tested with users, format 3 included. Ask two or
three people in the lab who haven't used it to build the four-FED3 FR1/RR
protocol twice: once in the editor and once in the text, with only the guide
to hand. Note where they stall and what the checker told them. That shows
whether format 3 helped, and whether 7 and 8 are worth building, better than
this review can.
