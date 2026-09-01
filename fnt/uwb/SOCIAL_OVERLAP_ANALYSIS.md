# Analysing `_SocialOverlapBouts.csv`

Reference for downstream R work on the FNT UWB tool's social output. Paste this
into a fresh context window before starting an analysis session.

---

## What this file is

One row per **dyad** per contiguous contact bout. Two animals are "in contact"
when their centres are within twice the social radius set in the preview — the
dotted circles overlapping. Contact is measured on the filtered and smoothed
track at full temporal resolution.

This is the **only** social product the tool exports, deliberately. The GBI,
the windowed edge list and a per-animal tracking summary were all removed
because each is derivable from this file, and each committed the pipeline to a
definition better owned by the analysis.

| Column | Holds |
|---|---|
| `animal1` | Sex+identity label, e.g. `F9002`. Built as `sex[0].upper() + identity` |
| `animal2` | The other member of the dyad |
| `Day` | 1-based day index, **of the day the bout started** |
| `Date` | Calendar date of the start |
| `bout_start` | Timestamp of the first real fix supporting the contact |
| `bout_stop` | Last real fix |
| `duration_s` | `bout_stop − bout_start`, full millisecond resolution |
| `mean_distance` | Mean centre-to-centre distance over the bout, metres |
| `n_observations` | Number of 1 s pairing bins supporting the bout |

---

## Five properties that govern how you use it

**1. Dyads do not fragment when a third animal joins.** Each pair is detected
independently. If A and B are together 0–20 s and C joins for 5–15 s, you get
three rows: A–B for the full 20 s, A–C for 10 s, B–C for 10 s. The A–B bout is
*not* cut at C's arrival, because the A–B relationship did not change.

**2. It is lossless for group structure.** The dyad list determines the contact
graph at every instant, so chain-rule groups are always recoverable (recipe 2
below). The reverse is false — given an "ABC group" you cannot recover whether
A and C were touching or three metres apart at opposite ends of a chain. Always
reconstruct from this file rather than from any pre-aggregated group product.

**3. Nothing is split at midnight.** A bout spanning midnight is one row, and
`Day` is the day it *started*. Splitting destroys information (you cannot tell
a split bout from two real ones); leaving it whole means you can split downstream
whenever a question needs it. Splitting is two lines — see recipe 4.

**4. The social radius is baked in.** Contact was thresholded at export time.
Changing the radius means re-exporting; you cannot re-threshold from this file,
which carries only `mean_distance` per bout, not per-instant distances.

**5. `duration_s` is exact, not floored.** A bout supported by a single pair of
near-simultaneous fixes genuinely spans milliseconds. If you want a
sampling-interval-weighted budget instead, sum `n_observations`.

---

## Loading it

Only `data.table` and `lubridate` are needed. The group reconstruction below
does its own connected components, so `igraph` is not a dependency — you may
still want it for the network analysis itself.

```r
library(data.table)
library(lubridate)

options(digits.secs = 3)   # REQUIRED: durations are millisecond-precise and
                           # R prints whole seconds without this

TZ <- "US/Mountain"        # whatever the trial's timezone column was set to

bouts <- fread("VT001_SocialOverlapBouts.csv")
bouts[, bout_start := with_tz(ymd_hms(bout_start), TZ)]
bouts[, bout_stop  := with_tz(ymd_hms(bout_stop),  TZ)]
```

`ymd_hms` handles the ISO-8601 offset pandas writes; `as.POSIXct` with a
format string chokes on the colon in `-07:00` on some platforms.

---

## Recipe 1 — time spent with each partner

The file is dyad-canonical (`animal1` sorts before `animal2`), so mirror it to
long form to aggregate per focal animal.

```r
long <- rbind(
  bouts[, .(focal = animal1, partner = animal2, duration_s, bout_start)],
  bouts[, .(focal = animal2, partner = animal1, duration_s, bout_start)]
)

partner_time <- long[, .(time_s      = sum(duration_s),
                         n_bouts     = .N,
                         median_s    = median(duration_s),
                         longest_s   = max(duration_s)),
                     by = .(focal, partner)][order(focal, -time_s)]
```

Report `n_bouts` alongside `time_s`. One long huddle and forty brief passes give
identical total time and mean very different things.

---

## Recipe 2 — reconstructing groups

The contact graph only changes at a bout boundary. So: pool every start and
stop, and between consecutive boundaries the graph is constant. Take connected
components on each constant interval, then merge adjacent intervals where an
animal's group membership did not change.

Run on the A/B/C example this yields AB(5 s) → ABC(10 s) → AB(5 s), correctly
segmented and non-overlapping.

```r
# Connected components by min-label propagation. The graph is at most one node
# per animal, so this beats pulling in igraph for it.
connected_labels <- function(a, b) {
  nodes <- sort(unique(c(a, b)))
  lab <- seq_along(nodes)
  ia <- match(a, nodes); ib <- match(b, nodes)
  repeat {
    old <- lab
    for (k in seq_along(ia)) {
      m <- min(lab[ia[k]], lab[ib[k]])
      lab[ia[k]] <- m; lab[ib[k]] <- m
    }
    if (identical(lab, old)) break
  }
  setNames(lab, nodes)
}

group_bouts <- function(bouts) {
  edges <- unique(bouts[, .(a = animal1, b = animal2,
                            t0 = bout_start, t1 = bout_stop)])
  bounds <- sort(unique(c(edges$t0, edges$t1)))
  n_int  <- length(bounds) - 1L
  if (n_int < 1L) return(data.table())

  res <- vector("list", n_int)
  for (k in seq_len(n_int)) {
    s <- bounds[k]; e <- bounds[k + 1L]
    # A bout either spans this interval entirely or not at all, because every
    # start and stop is itself a boundary. No partial overlap is possible.
    act <- edges[t0 <= s & t1 >= e]
    if (nrow(act) == 0L) next          # nobody in contact; all singletons
    lab <- connected_labels(act$a, act$b)
    m <- data.table(SexID = names(lab), cid = as.integer(lab))
    m[, group_size := .N, by = cid]
    m[, members := paste(sort(SexID), collapse = "|"), by = cid]
    m[, `:=`(seg_start = s, seg_stop = e)]
    res[[k]] <- m
  }
  segs <- rbindlist(res)
  if (nrow(segs) == 0L) return(data.table())

  # Merge consecutive intervals in which this animal's group is unchanged AND
  # time is continuous. The continuity test matters: an animal that leaves a
  # group and later re-forms the SAME group must get two rows, not one row
  # spanning the gap.
  setorder(segs, SexID, seg_start)
  segs[, new_run := is.na(shift(members)) |
                    members != shift(members) |
                    seg_start != shift(seg_stop), by = SexID]
  segs[, run := cumsum(new_run), by = SexID]

  out <- segs[, .(bout_start = min(seg_start),
                  bout_stop  = max(seg_stop),
                  members    = members[1],
                  group_size = group_size[1]),
              by = .(SexID, run)]
  out[, duration_s := as.numeric(difftime(bout_stop, bout_start, units = "secs"))]
  out[, run := NULL]
  setorder(out, SexID, bout_start)
  out[]
}

gb <- group_bouts(bouts)
```

Output is one row **per animal per group bout** — `SexID`, `bout_start`,
`bout_stop`, `duration_s`, `group_size`, `members` (pipe-delimited).
`group_id` if you want it: `gb[, group_id := .GRP, by = .(bout_start, members)]`.

Two invariants worth asserting on your own data, both of which hold on the
test fixture: an animal's group bouts **never overlap each other**, and the same
group re-forming after a gap yields **two rows, not one spanning the gap**.

```r
chk <- gb[order(SexID, bout_start)]
chk[, ok := is.na(shift(bout_stop)) | bout_start >= shift(bout_stop), by = SexID]
stopifnot(all(chk$ok))
```

**Two caveats.** The loop runs once per boundary interval; with tens of
thousands of bouts expect tens of seconds. If it drags, run it per day
(`bouts[Day == d]`) and `rbindlist` the results — groups never span a stretch in
which nobody is in contact, so day-chunking is safe unless a group is genuinely
active across midnight. Second, a zero-duration bout (`bout_start == bout_stop`,
possible when two fixes share a timestamp) contributes no interval and is
silently dropped; check `bouts[duration_s == 0, .N]` if that matters to you.

---

## Recipe 3 — participation in group sizes

```r
gb[, .(time_s = sum(duration_s), n_bouts = .N),
   by = .(SexID, group_size)][order(SexID, group_size)]
```

As a share of each animal's social time:

```r
gb[, .(time_s = sum(duration_s)), by = .(SexID, group_size)
 ][, perc := 100 * time_s / sum(time_s), by = SexID][order(SexID, group_size)]
```

**There is no `group_size == 1`.** This file knows only about contacts, so time
alone is time it cannot see. To add solo time you need each animal's tracked
duration, which is a definition you own — derive it from the smoothed CSV, where
every fix carries a timestamp:

```r
sm <- fread("VT001_smoothed.csv", select = c("Timestamp", "sex", "identity"))
sm[, SexID := paste0(substr(toupper(sex), 1, 1), identity)]
sm[, Timestamp := with_tz(ymd_hms(Timestamp), TZ)]
# your own rule for what counts as tracked goes here — e.g. sum the gaps
# between consecutive fixes, capping any gap longer than some dropout threshold
```

The tool deliberately does not pre-compute this. Whether a 30-minute dead tag
voids that window for one animal or for all of them is an analytical choice, and
baking one answer into an export would hide it.

---

## Recipe 4 — splitting at day boundaries

Only when a question actually needs it:

```r
split_at_midnight <- function(b, tz = TZ) {
  b <- copy(b)
  edges <- seq(floor_date(min(b$bout_start), "day"),
               ceiling_date(max(b$bout_stop), "day"), by = "day")
  out <- lapply(seq_len(length(edges) - 1L), function(i) {
    lo <- edges[i]; hi <- edges[i + 1L]
    s <- b[bout_stop > lo & bout_start < hi]
    if (!nrow(s)) return(NULL)
    s[, `:=`(bout_start = pmax(bout_start, lo),
             bout_stop  = pmin(bout_stop,  hi),
             Date       = as.Date(lo, tz = tz))]
    s[, duration_s := as.numeric(difftime(bout_stop, bout_start, units = "secs"))]
    s
  })
  rbindlist(out)[duration_s > 0]
}
```

---

## Recipe 5 — network edges, and the trap

Edge weight should be **time-based**, not a count of bouts:

```r
edge <- partner_time[focal < partner, .(a = focal, b = partner, time_s, n_bouts)]
```

For a normalized index, the time-domain analogue of the Simple Ratio Index is
`T_AB / (T_A + T_B − T_AB)`, where `T_A` is how long A was tracked — from your
own definition above. Do **not** feed this data to `asnipe::get_network`: SRI
counts rows, and your rows have wildly unequal durations, so a six-hour huddle
and a two-second brush weigh the same.

**The trap.** Raw edge weights confound social attraction with shared space use
and enclosure geometry. Two animals that both prefer the nest will co-occur
there with zero social preference, and in a bounded arena the geometry term is
not small. Before interpreting any network, build a null:

> **Circular time-shift permutation.** Shift each animal's entire trajectory by
> a random temporal offset, wrapping around. This preserves each individual's
> spatial distribution and its own movement autocorrelation exactly, while
> destroying temporal alignment between animals. Recompute contacts, repeat a
> few thousand times, and report observed-versus-expected rather than the raw
> weight.

Note this needs the **smoothed CSV**, not this file — you are shifting
trajectories and re-detecting contact, which the bouts cannot give you. Budget
for that when planning the analysis.

Relevant background: the `spatsoc` R package (Robitaille, Webber & Vander Wal)
is built for exactly this data shape; Holme & Saramäki's *Temporal Networks*
review is the formal framework; SocioPatterns (Cattuto, Barrat et al.) is the
human proximity-badge analogue. Verify citations before using them.

---

## What this file cannot tell you

- **Time alone** — needs tracked duration (your definition).
- **Distance below the contact threshold** — thresholded at export.
- **Who was near whom inside a chain-rule group** — actually it *can*, and that
  is the point; a group product cannot.
- **Anything at a different social radius** — re-export.
