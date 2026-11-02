# Wiser SQLite schema — reference and measured behaviour

What the Wiser tracking server writes, what it means, and what FNT does with
it. Two kinds of statement appear below and they are kept apart deliberately:

- **Documented** — from Wiser Systems' own GitbookDocs database page.
- **Measured** — established from a real recording, with the evidence given.
  Where a measurement contradicts an assumption, the measurement wins.
- **Inferred** — a reading consistent with the measurements but not confirmed
  by Wiser. Flagged as such every time.

Reference recording for every measurement here: `VT001_LossExperiment.sqlite`
(2026 VoleTerra loss pilot) — **24,680,309 rows, 17 tags, 18.88 days**, single
table `VoleTerra`. Server-side thresholding was **disabled** for this
recording, so the distributions below are the complete raw output of the
solver with nothing pre-filtered away. That matters: any quality distribution
measured from a server that *was* thresholding is truncated and not comparable
to this.

---

## Column reference

| Column | Type | Wiser's description | What FNT does |
|---|---|---|---|
| `reportid` | INT8 | unique per report | ignored |
| `shortid` | INT | decimal encoding of the hexadecimal Tag ID | the tag identity; every grouping key |
| `arenaid` | INT | internal ID for `arenaname`; "not for external use" | ignored |
| `arenaname` | TEXT | name of arena the tag is tracked by | ignored |
| `calculation_error` | REAL | "arbitrary scale, smaller is more accurate" | **not used** — see below |
| `location_x` | REAL | **inches** | × 0.0254 → metres |
| `location_y` | REAL | **inches** | × 0.0254 → metres |
| `location_z` | REAL | **inches** | ignored (tracking is 2-D) |
| `anchors_used` | INT | number of antennas | **not used** — see below |
| `timestamp` | INT8 | ms from Unix epoch | the time base; see below |
| `battery_voltage` | REAL | — | optional per-tag readout |
| `zones` | TEXT | — | ignored; FNT's regions come from the site XML or the ROI tool |
| `tag_type` | INT | — | ignored |
| `using_geostake`, `geopoint_lat/long/alt` | | GPS georeferencing | ignored (unused indoors; all NULL here) |
| `anchors_list` | TEXT | comma-delimited anchor IDs | ignored |
| `alias`, `alternateid`, `groupnames` | TEXT | user metadata | ignored; identities come from FNT's own tag configuration |

The unit conversion on `location_x/y` is the one silent transformation in the
pipeline: the database is in inches, everything FNT computes and exports is in
metres.

---

## `calculation_error`

Wiser documents this only as *"arbitrary scale, smaller is more accurate"*,
with a link to a server thresholds page that returns 404. The number itself is
more forthcoming.

### Measured: it is an exact rational function of the anchor count

```
calculation_error == k / (4 * anchors_used),    k a non-negative integer
```

Exact for **100.00%** of 4,000,000 rows tested, and `k <= 4 * anchors_used` in
every row — so the value is a **bounded ratio in [0, 1)**. The whole table
contains only **531 distinct values**, and every one is a simple fraction of
this form. The maximum observed across all 24.7 M rows is 0.8947 = 68/76.

Examples, straight from the file:

| value | `anchors_used` | fraction |
|---|---|---|
| 0.013157894736842146 | 19 | 1/76 |
| 0.012499999999999956 | 20 | 1/80 |
| 0.014705882352941124 | 17 | 1/68 |
| 0.026315789473684181 | 19 | 2/76 |

**42.3% of all rows are exactly zero.**

### Inferred: a normalised mean of a small per-anchor score

Rearranging, `error = (k / N) / 4`. That reads as: each contributing anchor
carries an integer disagreement score from 0 to 4, and `calculation_error` is
their mean expressed as a fraction of the worst case. In 99.3% of rows the
average anchor scores below 1.

What the integer physically counts — range residual in quarter-units, a count
of anchors flagged inconsistent, something else — is **not established**. Only
the arithmetic form is.

### Measured: it is NOT anchor-count-neutral, and zero does not mean good

All 24,680,309 rows, not a sample:

| `anchors_used` | rows | median error | fraction exactly 0 |
|---|---|---|---|
| 3–5 | 716,473 | 0.00000 | **71.1%** |
| 6–8 | 1,181,626 | 0.00000 | 59.3% |
| 9–11 | 1,772,243 | 0.00000 | 50.3% |
| 12–14 | 2,752,197 | 0.01786 | 45.8% |
| 15–17 | 3,951,877 | 0.01471 | 44.9% |
| 18–20 | 5,550,067 | 0.01316 | 41.1% |
| 21–24 | 8,755,826 | 0.01136 | 34.6% |

The relationship is cleanly monotone: the more anchors contributed, the less
often the solver reports a perfect score.

This runs backwards from intuition, and the structure above explains why: the
metric measures *disagreement between anchors*, so a solve with almost no
redundancy has almost nothing to disagree. A fix computed from 3–5 anchors —
the weakest geometry in the file — reports a perfect 0 **71%** of the time,
twice as often as a fix built from 21 or more. `calculation_error == 0` substantially means *"too few anchors to detect
a problem"*, not *"accurate"*.

Checked against genuinely bad fixes (out-and-back ghosts on tag 11, defined as
a fix more than 1 m from both neighbours while those neighbours sit within
1 m of each other):

| | median error | median anchors | error exactly 0 |
|---|---|---|---|
| ghost fixes | 0.0938 | 6 | **32.2%** |
| everything else | 0.0119 | 19 | 43.9% |

A third of the genuinely bad fixes report a flawless error, and they cluster
at the anchor counts where the metric is blindest. **The dangerous combination
is low `anchors_used` together with zero `calculation_error`**: it reads as a
clean fix and is the least constrained one in the file.

### Why FNT does not filter on it

Thresholding at `error <= 0.05` discards 17% of fixes to catch 34% of the
steps over 1 m — a poor trade, and one that is worst exactly where it is
needed. FNT's spatial-outlier (Hampel) threshold judges a fix against the
surrounding track instead, which is independent of how many anchors happened
to contribute. See `hampel_keep` in `uwb_preprocessing_pyqt.py`.

If this column is ever used for anything, it **must** be conditioned on
`anchors_used`: a cutoff that is meaningful at 20 anchors is inert at 5.

---

## `anchors_used`

Range 3–24, median 17 in the reference recording; 0.77% of fixes used fewer
than 4 anchors and 2.11% fewer than 6.

It is a genuinely **strong** signal for bad fixes — median 6 on ghost fixes
against 19 elsewhere, a much cleaner separation than `calculation_error`
manages. It is nonetheless **not** used as a filter, for a specific reason:
gating at `>= 8` costs 4.5% of fixes and can make the track *worse*. Deleting
a run of low-anchor fixes opens a gap, and the step across that gap is longer
than the steps it replaced. On tag 10 the largest surviving step grew from
9.9 m to 13.8 m when the gate was added on top of the outlier threshold.
Across six tags the gate changed outlier-filtered quality by nothing much
either way while costing 3% of the data.

It is a good **diagnostic** — worth looking at when a stretch of track is
suspect — and a poor **gate**.

---

## `timestamp`

Milliseconds since the Unix epoch (1970-01-01 00:00:00 UTC), 13 digits.

The integer denotes an **instant**, not a local time, and carries no timezone
because it does not need one. Selecting a timezone in FNT does not move the
data; it chooses how that instant is rendered as a wall clock. That rendering
does, however, decide calendar-day boundaries, so it drives the `Date` column,
per-day plot facets, hourly bins and the `day`-scope rows in the occupancy
CSV. Picking the wrong zone cannot corrupt positions or intervals; it will
silently shift every day boundary.

### Measured characteristics

- **Millisecond resolution is real.** Only 24,888 of 24.68 M values land on a
  whole second — about 0.1%, i.e. chance.
- **Reporting is irregular and far faster than 1 Hz.** Median interval
  0.132 s (~7.6 Hz); 1st percentile 0.001 s, 99th 1.075 s. This is why the
  velocity threshold needs a minimum time base: dividing a few centimetres of
  positional noise by a millisecond yields tens of m/s, so an unfloored
  velocity threshold measures how closely spaced the reports were rather than
  how fast the animal moved.
- **Rows are NOT stored in time order.** Ordering tag 11 by `rowid` gives
  4,435 backwards steps in 200,000 rows. FNT always sorts by timestamp; any
  direct query of this database must do the same.
- **Duplicate timestamps occur within a tag** — 12,209 of tag 11's first
  200,000 rows repeat the previous row's millisecond. Naive interval
  arithmetic on this data will produce zeros and divide-by-zero infinities.
- **One clock, shared across tags.** Different tags carry identical
  millisecond values, so positions are stamped by a single system clock rather
  than free-running per-tag clocks. **Relative** timing between animals is
  therefore internally consistent, which is what proximity, contact bouts, GBI
  and chase detection depend on.

### Resolution is not accuracy

Nothing in the schema records whether the Wiser server's clock is
NTP-disciplined or how far it drifts from true UTC. Absolute accuracy is
unknown and cannot be established from the database. It only matters when
aligning UWB to something external — video, a lighting controller, another
instrument — and in that case it should be verified empirically against a
known shared event rather than assumed.

---

## What FNT exports

The smoothed CSV carries both representations of time, deliberately:

| column | example | notes |
|---|---|---|
| `timestamp` | `1785364311664` | the original integer, unmodified |
| `Timestamp` | `2026-07-29 16:31:51.664000-06:00` | local rendering with an explicit UTC offset |

Downstream code — including FNT's own — should read the **integer**. Parsing
24 M date strings costs minutes where the integer conversion is near-instant,
and a recording that crosses a daylight-saving boundary carries *two different
offsets* in the string column, which fixed-format parsers mangle silently. The
integer has neither problem. See `smoothed_csv_timestamps`.

Rows are ordered by tag, then by time within each tag.

---

## Open questions for Wiser

1. **What does the integer numerator of `calculation_error` count?** The form
   is established as `k / (4 * anchors_used)`; only the meaning of `k` is
   missing.
2. **Is the 0–4 per-anchor ceiling deliberate**, and does a score of 4 mean a
   specific failure the solver can name?
3. The documented link to `installation/server/tracking-settings.md#thresholds`
   is dead. If a server-side `calculation_error` threshold is enabled, the
   database is **pre-filtered** and the distributions above do not apply.
