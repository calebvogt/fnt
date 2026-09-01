# Multiple position solutions per timestamp in Wiser UWB exports

**Observed in:** `VT001_LossExperiment.sqlite`, table `VoleTerra`
**Recording:** 2026-07-29 16:24 → 2026-08-17 13:35 (18.9 days), 17 tags, 25 anchors
**Prepared:** 2026-08-26

---

## Summary

The exported table contains, for some tags, **several rows sharing an identical
`shortid` and `timestamp` but carrying different `location_x` / `location_y`
values**. They are not repeated rows: each has its own `reportid` and its own
`anchors_list`, i.e. each is an independent position solution computed from a
different subset of anchors for the same instant.

Across the trial this accounts for **6,171,423 of 24,680,309 rows (25.0%)**.

Any consumer that reads the table as a time series — as we did — treats these
co-temporal solutions as consecutive movement, because nothing in the row
marks them as alternatives. The result is a track that appears to move between
positions in zero elapsed time.

We are not reporting this as a defect, because we cannot tell from the export
whether it is intended. We are reporting it because a downstream consumer has
no way to discover it except by noticing that a tag apparently moved 7 metres
in 0.45 seconds.

---

## What the rows look like

A single tag at a single millisecond:

```sql
SELECT reportid, shortid, timestamp, location_x, location_y,
       anchors_used, anchors_list, calculation_error
FROM VoleTerra
WHERE shortid = 7 AND timestamp = 1785421441248;
```

```
 reportid  shortid     timestamp  location_x  location_y  anchors_used       anchors_list  calculation_error
  5671018        7 1785421441248  998.871040  178.528303             5      ,5,8,9,13,14,                0.0
  5671019        7 1785421441248  740.476244  300.310598             3      ,3,8,9,13,14,                0.0
  5671022        7 1785421441248  996.045971  173.880414             6   ,5,8,9,13,18,19,                0.0
  5671025        7 1785421441248  987.740218  166.776977             7 ,3,5,8,9,13,14,19,                0.0
  5671028        7 1785421441248  997.126972  176.818857             6   ,5,8,9,13,14,18,                0.0
  5671029        7 1785421441248 1000.995182  181.023569             5      ,5,8,9,13,14,                0.0
  5671030        7 1785421441248  992.217763  171.875221             6    ,3,5,8,9,13,19,                0.0
  5671032        7 1785421441248  983.762385  164.851858             6    ,3,5,8,9,13,14,                0.0
```

In metres (`location_x/y` × 0.0254):

| anchors | position (m) | anchors_list |
|--------:|--------------|--------------|
| 7 | (25.09, 4.24) | `,3,5,8,9,13,14,19,` |
| 6 | (24.99, 4.19) | `,3,5,8,9,13,14,` |
| 6 | (25.20, 4.37) | `,3,5,8,9,13,19,` |
| 6 | (25.30, 4.42) | `,5,8,9,13,18,19,` |
| 6 | (25.33, 4.49) | `,5,8,9,13,14,18,` |
| 5 | (25.37, 4.53) | `,5,8,9,13,14,` |
| 5 | (25.43, 4.60) | `,5,8,9,13,14,` |
| 3 | (18.81, 7.63) | `,3,8,9,13,14,` |

Seven solutions agree within ~0.4 m. The eighth, computed from the minimum
three anchors, is **7.0 m away**.

Note that `calculation_error` is **0.0 on all eight**, including the outlier.

---

## Scale

```sql
SELECT COUNT(*) FROM VoleTerra;                                    -- 24,680,309
SELECT COUNT(*) FROM (SELECT 1 FROM VoleTerra
                      GROUP BY shortid, timestamp);                -- 18,508,886
```

| | rows |
|---|---:|
| total rows in table | 24,680,309 |
| distinct (`shortid`, `timestamp`) | 18,508,886 |
| **surplus rows** | **6,171,423 (25.0%)** |

Instants carrying more than one row: **3,974,096**.
Of those, instants where every row has **identical** `location_x`/`location_y`:
**8**. So essentially none are duplicate records — they are distinct solutions.

---

## Distribution: a property of the tag, not of a time period

| shortid | instants | multi-solution instants | % | max rows/instant | rows per instant |
|--------:|---------:|------------------------:|----:|---:|---:|
| 2  |   418,571 |        26 |  0.01% | 2  | 1.00 |
| **3**  |   848,148 |   411,465 | **48.51%** | 9  | 1.90 |
| 4  | 3,732,040 |     6,274 |  0.17% | 2  | 1.00 |
| **6**  | 2,937,142 | 1,817,306 | **61.87%** | 11 | 2.08 |
| **7**  |   894,390 |   450,299 | **50.35%** | 11 | 2.04 |
| 10 | 1,875,489 |    11,534 |  0.61% | 2  | 1.01 |
| **11** | 5,230,121 | 1,199,888 | **22.94%** | 4  | 1.23 |
| **16** |   392,436 |    30,285 |  **7.72%** | 3  | 1.08 |
| **21** |   449,714 |    46,718 | **10.39%** | 3  | 1.11 |
| 24 |    52,911 |       169 |  0.32% | 3  | 1.00 |
| 32 |    54,349 |         0 |  0.00% | 1  | 1.00 |
| 39 |   129,455 |        15 |  0.01% | 2  | 1.00 |
| 42 |   109,155 |        16 |  0.01% | 2  | 1.00 |
| 44 |   117,252 |        70 |  0.06% | 2  | 1.00 |
| 47 |   105,300 |        27 |  0.03% | 2  | 1.00 |
| 48 | 1,132,038 |         1 |  0.00% | 2  | 1.00 |
| 49 |    30,375 |         3 |  0.01% | 2  | 1.00 |

Six tags (3, 6, 7, 11, 16, 21) are heavily affected; the other eleven are
essentially clean.

**The behaviour is stable for a tag's whole deployment.** Rows-per-instant, by
calendar day:

```
date     tag3  tag6  tag7  tag11   tag4  tag10  tag48
07-29    1.96  1.92  1.90   1.08   1.00    -     1.00
07-30    1.90  2.00  2.05   1.12   1.00    -     1.00
07-31     -    2.17   -     1.11   1.00    -     1.00
08-01     -    2.08   -     1.08   1.00    -     1.00
08-02     -    2.22   -     1.29   1.00    -     1.00
08-08     -     -     -     1.30   1.00   1.01   1.00
08-17     -     -     -      -     1.01    -     1.00
```

It is not a server-wide event, a firmware window, or a period of poor
conditions — an affected tag is affected from its first day to its last, and a
clean tag stays clean for twenty days alongside it.

**It does not track data volume.** Tag 4 has 3.7 M rows and is clean (0.17%);
tag 3 has 1.6 M rows and is 48.5% multi-solution.

**No column in the export distinguishes the two groups.** `tag_type` (0),
`arenaid` (1) and `using_geostake` (0) are identical for all 17 tags.

---

## How far apart are the co-temporal solutions?

Sample: `shortid` 7, first 2 hours, 41,653 multi-solution instants. Maximum
pairwise distance between solutions at the same instant:

| percentile | spread |
|---|---:|
| 50th | 0.10 m |
| 75th | 0.15 m |
| 90th | 0.21 m |
| 95th | 0.26 m |
| 99th | 0.93 m |
| max  | **15.20 m** |

Instants where solutions differ by more than 1 m: **1.0%**.

So the majority are near-agreeing solutions ~10 cm apart, and a 1% tail
diverges severely.

Anchor counts among the tied solutions: best median 19, worst median 17
(range 3–24). At **21.2%** of multi-solution instants, every solution shares
the same `anchors_used`, so anchor count alone does not always identify a
preferred row.

---

## Downstream impact

Read as a time series, the surplus rows produce two separate problems.

**1. Inflated path length.** Consecutive co-temporal solutions ~10 cm apart,
with zero elapsed time between them, are indistinguishable from movement.
Over 6 hours of tag 7:

| | path length |
|---|---:|
| every row | 58.0 km |
| one solution per instant | 28.7 km |

58 km over 6 h is 2.7 m/s sustained, for a 40 g vole.

**2. Apparent teleports.** The 1% tail seeds discontinuities that survive
speed and step-distance filtering. Worked example, tag 7 (`M9682`),
2026-07-30 08:24:08 — the track moves **7.22 m in 0.45 s** (~16 m/s):

```
 08:24:08.374   (25.15, 4.26)   anchors 5
 08:24:08.506   (21.32, 8.40)   anchors 3
 08:24:08.776   (18.64, 7.80)   anchors 3     <- three solutions, same instant
 08:24:08.776   (18.74, 7.65)   anchors 4
 08:24:08.776   (21.17, 8.51)   anchors 3
 08:24:08.822   (18.75, 7.61)   anchors 3
```

The tag does not return: subsequent fixes continue near (18.7, 7.7). So this
is not a transient spike but a sustained relocation of the reported position.

---

## What we could not determine from the export

- Whether emitting several solutions per instant is intended behaviour.
- Why six tags do it consistently and eleven do not, given identical
  `tag_type`, `arenaid` and `using_geostake`.
- Whether one row per instant is meant to be authoritative. Nothing in the
  schema marks a preferred solution — `reportid` is sequential across all tags,
  and `calculation_error` is 0.0 even on badly divergent rows.
- Whether the differing `anchors_list` values reflect distinct ranging cycles
  or repeated solving of one cycle.

---

## What we do about it

We collapse to one row per (`shortid`, `timestamp`) at read time, before any
filtering: the solution with the highest `anchors_used`, and where several tie
at the top count, the coordinate-wise median of the tied rows. Tested against
temporal continuity on 6 h of tag 7 (490,585 rows):

| rule | median step | p95 | steps > 1 m | path |
|---|---:|---:|---:|---:|
| every row (no collapse) | 0.078 m | 0.22 m | 0.70% | 58.0 km |
| most anchors, tie → first row | 0.079 m | 0.21 m | 0.56% | 29.8 km |
| **most anchors, tie → median** | **0.075 m** | **0.21 m** | **0.56%** | **28.7 km** |
| median of all solutions | 0.068 m | 0.19 m | 0.73% | 27.6 km |
| first row at each instant | 0.081 m | 0.23 m | 0.97% | 34.3 km |

Taking the median of *all* solutions gives the smoothest track overall but a
worse >1 m tail, because a minimum-anchor outlier drags the median when only
two or three solutions exist. Restricting to the best-constrained solutions
first avoids that.

---

## Reproducing

```sql
-- surplus rows
SELECT COUNT(*) AS rows_,
       (SELECT COUNT(*) FROM (SELECT 1 FROM VoleTerra
                              GROUP BY shortid, timestamp)) AS instants
FROM VoleTerra;

-- per-tag breakdown
SELECT shortid, COUNT(*) AS instants,
       SUM(CASE WHEN n > 1 THEN 1 ELSE 0 END) AS multi, MAX(n) AS worst
FROM (SELECT shortid, timestamp, COUNT(*) AS n
      FROM VoleTerra GROUP BY shortid, timestamp)
GROUP BY shortid ORDER BY shortid;

-- instants that are genuinely identical rows (result: 8)
SELECT COUNT(*) FROM (
  SELECT shortid, timestamp FROM VoleTerra
  GROUP BY shortid, timestamp
  HAVING COUNT(*) > 1
     AND COUNT(DISTINCT location_x) = 1
     AND COUNT(DISTINCT location_y) = 1);
```

See also [`WISER_SCHEMA.md`](WISER_SCHEMA.md) for the full column reference and
our interpretation of each field.
