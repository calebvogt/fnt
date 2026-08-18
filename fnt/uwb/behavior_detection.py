"""Behavior detection from UWB (x, y) tracks — zone-free, preview-oriented.

Everything here is a pure function over the arrays the live preview already
builds: a regular time grid (1/2/5 Hz) with one x/y column per tag. That grid is
the per-animal-per-second table behavior detection needs, so nothing has to be
re-read from the database to classify a preview window.

Design notes that matter scientifically:

* Angles are always wrapped before comparison. Headings of +3.0 and -3.0 rad are
  ~0.28 rad apart, not 6.0; comparing raw differences silently rejects real
  events.
* Thresholds are parameters, not constants. The defaults below are starting
  points, not validated optima — they are coupled to how heavily the track was
  smoothed and to the grid rate, so they should be re-derived (e.g. from the
  speed percentiles this module reports) whenever those change.
* Detection inherits whatever smoothing the caller applied. Smoothing is a
  low-pass filter, so it attenuates exactly the speed signal chases depend on;
  running detection on the same track the user is looking at makes that
  trade-off visible instead of hidden.
"""

from dataclasses import dataclass, asdict, fields

import numpy as np

# Locomotor state (mutually exclusive, one per animal per frame).
# Deliberately bimodal: UWB centroid tracking of small animals does not resolve
# a trustworthy third "running" tier - the speed that would separate it sits in
# the same range as position noise once the track is smoothed at all.
LOC_NODATA, LOC_INACTIVE, LOC_MOVING = 0, 1, 2
LOCOMOTOR_LABELS = {LOC_NODATA: "no data", LOC_INACTIVE: "inactive",
                    LOC_MOVING: "moving"}

# Social state (at most one per animal per frame; independent of locomotor)
# Ordered by precedence: a higher code overwrites a lower one when an animal
# qualifies for several at once (being displaced outranks merely touching).
SOC_NONE, SOC_CONTACT, SOC_DISPLACED, SOC_DISPLACING, SOC_CHASED, SOC_CHASING = 0, 1, 2, 3, 4, 5
SOCIAL_LABELS = {SOC_NONE: "", SOC_CONTACT: "contact",
                 SOC_DISPLACED: "displaced", SOC_DISPLACING: "displacing",
                 SOC_CHASED: "chased", SOC_CHASING: "chasing"}

# Display colours, shared by the canvas overlay and the state panel.
STATE_COLORS = {
    LOC_INACTIVE: "#7f8c99",
    LOC_MOVING: "#4da3ff",
    LOC_NODATA: "#4a4a4a",
}
SOCIAL_COLORS = {
    SOC_CONTACT: "#2ea043",
    SOC_DISPLACING: "#c77dff",
    SOC_DISPLACED: "#9d8df1",
    SOC_CHASING: "#ff4d4d",
    SOC_CHASED: "#ff9e4d",
}


@dataclass
class BehaviorParams:
    """Tunable detection thresholds. Defaults reproduce current proximity."""

    # Social overlap: each animal carries a circle of this radius, and a social
    # overlap is when two circles intersect (centres within 2 * radius).
    # 0.25 m reproduces the existing 0.5 m centre-to-centre proximity exactly.
    social_radius: float = 0.25

    # Locomotor speed cut (m/s): at or below is inactive, above is moving.
    still_speed: float = 0.06

    # Chase: close, both moving, a heading at b, b heading away from a.
    chase_distance: float = 0.50
    chase_speed: float = 0.20
    chase_angle_deg: float = 45.0
    min_chase_s: float = 1.0

    # Heading is taken over a lag so it is not dominated by per-frame jitter.
    heading_lag_s: float = 3.0

    # Displacement (supplant): a moves in on a stationary b, and b is the one
    # that leaves. Unlike chasing this needs no sprint from either animal, so it
    # still works on a heavily smoothed track - which is what makes it the
    # dependable dominance measure here.
    displace_distance: float = 0.30       # how close the arrival must get
    displace_loser_speed: float = 0.10    # b counts as settled below this
    displace_winner_speed: float = 0.15   # a must actually be moving in
    displace_leave_distance: float = 0.75 # separation that counts as resolved
    displace_window_s: float = 5.0        # how long to wait for that resolution

    # A tag with no fix for longer than this is treated as absent rather than
    # parked at its last position. Without it a tag whose battery died would
    # sit in permanent "contact" with whatever it was near.
    max_stale_s: float = 30.0

    def contact_distance(self):
        """Centre-to-centre separation at which the two circles touch."""
        return 2.0 * self.social_radius

    def to_dict(self):
        return asdict(self)

    @classmethod
    def from_dict(cls, d):
        known = {f.name for f in fields(cls)}
        return cls(**{k: v for k, v in (d or {}).items() if k in known})


def _wrap(angle):
    """Wrap radians to [-pi, pi] — required before any angular comparison."""
    return np.arctan2(np.sin(angle), np.cos(angle))


def grid_dt_seconds(times):
    """Seconds per frame of the preview grid (median, robust to gaps)."""
    if times is None or len(times) < 2:
        return 1.0
    try:
        stamps = np.asarray(times, dtype="datetime64[ns]")
        deltas = np.diff(stamps).astype("timedelta64[ns]").astype(np.float64) / 1e9
    except Exception:
        return 1.0
    deltas = deltas[np.isfinite(deltas) & (deltas > 0)]
    return float(np.median(deltas)) if len(deltas) else 1.0


def compute_kinematics(times, x, y, params):
    """Per-tag speed and heading, measured between real fixes only.

    x, y are (frames, tags) on the preview grid with NaN wherever a tag had no
    fix in that bin. Speed is the displacement between *consecutive real fixes*
    divided by the true time between them, then held until the next fix.

    Doing it that way matters twice over: differencing the forward-filled track
    would read zero through every gap (deflating the speed distribution), and
    then post a spike the moment the tag reappears — a tag silent for 10 s that
    returns 0.5 m away would read 0.5 m/s and could fabricate a chase.
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dt = grid_dt_seconds(times)
    n_frames, n_tags = x.shape

    velocity = np.full(x.shape, np.nan)
    heading = np.full(x.shape, np.nan)
    lag_frames = max(1, int(round(params.heading_lag_s / dt)))

    for t in range(n_tags):
        idx = np.flatnonzero(np.isfinite(x[:, t]) & np.isfinite(y[:, t]))
        if len(idx) < 2:
            continue
        gaps = np.diff(idx) * dt
        step = np.hypot(np.diff(x[idx, t]), np.diff(y[idx, t]))
        with np.errstate(divide="ignore", invalid="ignore"):
            velocity[idx[1:], t] = step / gaps

        # Heading over roughly heading_lag_s of real elapsed time, comparing
        # each fix with the most recent fix at least that far back.
        ref = np.searchsorted(idx, idx - lag_frames, side="right") - 1
        valid = ref >= 0
        src = idx[ref[valid]]
        dst = idx[valid]
        moved = (x[dst, t] != x[src, t]) | (y[dst, t] != y[src, t])
        dst, src = dst[moved], src[moved]
        heading[dst, t] = np.arctan2(y[dst, t] - y[src, t], x[dst, t] - x[src, t])

    return velocity, heading, dt


def hold_last(values, have, dt, max_stale_s):
    """Carry each column's last real value forward, up to max_stale_s.

    Mirrors what the preview does with marker positions, so the classification
    describes the tag the user can actually see. Beyond the limit the tag is
    treated as absent instead of parked where it was last seen.
    """
    values = np.asarray(values, dtype=float)
    have = np.asarray(have, dtype=bool)
    out = np.full(values.shape, np.nan)
    max_frames = max(0, int(round(max_stale_s / dt))) if dt > 0 else 0
    n_frames, n_cols = values.shape
    for c in range(n_cols):
        last_val = np.nan
        age = 0
        for f in range(n_frames):
            if have[f, c]:
                last_val = values[f, c]
                age = 0
            else:
                age += 1
            if np.isfinite(last_val) and age <= max_frames:
                out[f, c] = last_val
    return out


def compute_pairs(x, y, velocity, heading):
    """Pairwise geometry for every ordered pair, per frame.

    Returns a dict of (frames, tags, tags) arrays indexed [t, a, b]:
      dist     separation
      closing  negative when the pair is converging
      toward   angle between a's heading and the a->b bearing (small: a at b)
      away     angle between b's heading and the a->b bearing (small: b fleeing)
    """
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    dx = x[:, None, :] - x[:, :, None]          # [t, a, b] = x_b - x_a
    dy = y[:, None, :] - y[:, :, None]
    dist = np.hypot(dx, dy)

    closing = np.full(dist.shape, np.nan)
    if dist.shape[0] >= 2:
        closing[1:] = np.diff(dist, axis=0)

    bearing = np.arctan2(dy, dx)                # direction a -> b
    toward = np.abs(_wrap(heading[:, :, None] - bearing))
    away = np.abs(_wrap(heading[:, None, :] - bearing))
    return {"dist": dist, "closing": closing, "toward": toward, "away": away}


def _min_run_frames(mask, min_frames):
    """Zero out True-runs shorter than min_frames along the time axis."""
    if min_frames <= 1:
        return mask
    n_frames = mask.shape[0]
    flat = mask.reshape(n_frames, -1)
    out = np.zeros_like(flat)
    for col in range(flat.shape[1]):
        series = flat[:, col]
        if not series.any():
            continue
        start = None
        for t in range(n_frames + 1):
            on = t < n_frames and bool(series[t])
            if on and start is None:
                start = t
            elif not on and start is not None:
                if t - start >= min_frames:
                    out[start:t, col] = True
                start = None
    return out.reshape(mask.shape)


def detect_displacements(dist, velocity, x, y, params, dt):
    """Find supplants: a arrives on a settled b, and b is the one who leaves.

    A continuous-space analogue of the RFID supplant index, and the only
    detector here that does not hinge on instantaneous speed. Returns a list of
    (onset_frame, resolved_frame, winner, loser).

    Two stages, per the arrival/resolution logic:
      1. onset  - the pair closes to within displace_distance for the first
         time, with b slow and a moving in;
      2. resolve - within displace_window_s the pair separates past
         displace_leave_distance, and b has travelled further from the onset
         position than a has. That last test is what makes it directional: both
         animals moving apart is not a supplant unless the arriver held ground.
    """
    n_frames, n_tags = velocity.shape
    win = max(1, int(round(params.displace_window_s / dt)))
    events = []
    for a in range(n_tags):
        for b in range(n_tags):
            if a == b:
                continue
            d = dist[:, a, b]
            close = np.isfinite(d) & (d <= params.displace_distance)
            if not close.any():
                continue
            prev_close = np.concatenate(([False], close[:-1]))
            onset = (close & ~prev_close
                     & np.isfinite(velocity[:, b])
                     & (velocity[:, b] <= params.displace_loser_speed)
                     & np.isfinite(velocity[:, a])
                     & (velocity[:, a] >= params.displace_winner_speed))
            for i in np.flatnonzero(onset):
                for j in range(i + 1, min(n_frames, i + win + 1)):
                    if not (np.isfinite(d[j]) and d[j] >= params.displace_leave_distance):
                        continue
                    moved_a = np.hypot(x[j, a] - x[i, a], y[j, a] - y[i, a])
                    moved_b = np.hypot(x[j, b] - x[i, b], y[j, b] - y[i, b])
                    if np.isfinite(moved_a) and np.isfinite(moved_b) and moved_b > moved_a:
                        events.append((int(i), int(j), int(a), int(b)))
                    break      # first resolution settles this arrival
    return events


def classify(times, x, y, params=None):
    """Classify every tag in every preview frame.

    Returns a dict with:
      velocity, heading   (frames, tags)
      pairs               dict of (frames, tags, tags) geometry
      locomotor           (frames, tags) LOC_* codes
      social              (frames, tags) SOC_* codes
      partner             (frames, tags) index of the social partner, else -1
      contact             (frames, tags, tags) bool, circles overlapping
      chase               (frames, tags, tags) bool, a chasing b
      dt                  seconds per frame
    """
    params = params or BehaviorParams()
    velocity, heading, dt = compute_kinematics(times, x, y, params)

    # The preview draws each tag at its last known position so markers do not
    # blink between the tags' sparse fixes. Classify that same held track, or
    # the overlay would disagree with the marker the user is looking at —
    # a contact line flickering off while both circles still visibly overlap.
    have_fix = np.isfinite(np.asarray(x, dtype=float)) & np.isfinite(
        np.asarray(y, dtype=float))
    xh = hold_last(x, have_fix, dt, params.max_stale_s)
    yh = hold_last(y, have_fix, dt, params.max_stale_s)
    velocity = hold_last(velocity, have_fix & np.isfinite(velocity), dt,
                         params.max_stale_s)
    heading = hold_last(heading, have_fix & np.isfinite(heading), dt,
                        params.max_stale_s)

    pairs = compute_pairs(xh, yh, velocity, heading)

    n_frames, n_tags = velocity.shape
    have = np.isfinite(xh)

    # --- locomotor --------------------------------------------------------
    locomotor = np.full((n_frames, n_tags), LOC_NODATA, dtype=np.int8)
    v_ok = np.isfinite(velocity)
    locomotor[have & v_ok & (velocity <= params.still_speed)] = LOC_INACTIVE
    locomotor[have & v_ok & (velocity > params.still_speed)] = LOC_MOVING

    # --- social overlap ---------------------------------------------------
    dist = pairs["dist"]
    eye = np.eye(n_tags, dtype=bool)[None, :, :]
    contact = np.isfinite(dist) & (dist <= params.contact_distance()) & ~eye

    # --- chase ------------------------------------------------------------
    ang = np.deg2rad(params.chase_angle_deg)
    v_a = velocity[:, :, None]
    v_b = velocity[:, None, :]
    chase = (
        np.isfinite(dist) & (dist <= params.chase_distance) & ~eye
        & np.isfinite(v_a) & (v_a >= params.chase_speed)
        & np.isfinite(v_b) & (v_b >= params.chase_speed)
        & np.isfinite(pairs["toward"]) & (pairs["toward"] <= ang)
        & np.isfinite(pairs["away"]) & (pairs["away"] <= ang)
    )
    chase = _min_run_frames(chase, max(1, int(round(params.min_chase_s / dt))))

    # --- displacement -----------------------------------------------------
    displacements = detect_displacements(dist, velocity, xh, yh, params, dt)
    displacing = np.zeros((n_frames, n_tags, n_tags), dtype=bool)
    for i, j, a_idx, b_idx in displacements:
        displacing[i:j + 1, a_idx, b_idx] = True

    # --- collapse to one social state per animal --------------------------
    social = np.full((n_frames, n_tags), SOC_NONE, dtype=np.int8)
    partner = np.full((n_frames, n_tags), -1, dtype=np.int16)
    finite_dist = np.where(np.isfinite(dist), dist, np.inf)

    def _assign(mask_ab, code_for_a):
        """mask_ab[t, a, b] -> set a's state, remembering the nearest b."""
        any_b = mask_ab.any(axis=2)
        if not any_b.any():
            return
        candidates = np.where(mask_ab, finite_dist, np.inf)
        best = np.argmin(candidates, axis=2)
        upgrade = any_b & (social < code_for_a)
        social[upgrade] = code_for_a
        partner[upgrade] = best[upgrade].astype(np.int16)

    # Applied in increasing precedence: a later, stronger state overwrites.
    _assign(contact, SOC_CONTACT)
    _assign(np.swapaxes(displacing, 1, 2), SOC_DISPLACED)  # b is pushed off
    _assign(displacing, SOC_DISPLACING)
    _assign(np.swapaxes(chase, 1, 2), SOC_CHASED)   # b is the one being chased
    _assign(chase, SOC_CHASING)

    return {"velocity": velocity, "heading": heading, "pairs": pairs,
            "locomotor": locomotor, "social": social, "partner": partner,
            "contact": contact, "chase": chase, "dt": dt,
            "displacing": displacing, "displacements": displacements,
            "have_fix": have_fix, "present": have, "x": xh, "y": yh}


def speed_summary(velocity, have_fix, params):
    """Plain-language read on whether the track can support the thresholds.

    Returns (percentiles, message). Percentiles are taken only over frames where
    the tag actually had a fix, so values repeated through a hold do not skew
    them. The message says outright when a threshold sits above essentially all
    observed movement, which is the failure mode that silently yields zero
    detections.
    """
    v = np.asarray(velocity, dtype=float)
    mask = np.asarray(have_fix, dtype=bool) & np.isfinite(v)
    vals = v[mask]
    if not len(vals):
        return {}, "No movement measured in this window."
    pct = {q: float(np.percentile(vals, q)) for q in (50, 90, 99)}
    msg = ("Speed of the shown track: half of fixes below {:.3f} m/s, "
           "fastest 1% above {:.3f} m/s.".format(pct[50], pct[99]))
    over = [name for name, thr in (("moving", params.still_speed),
                                   ("chasing", params.chase_speed))
            if thr > pct[99]]
    if over:
        msg += ("  Nothing can register as " + " or ".join(over) +
                ": that threshold is above the fastest 1% of movement here. "
                "Either lower it or smooth the track less.")
    return pct, msg


def speed_percentiles(velocity, qs=(50, 75, 90, 95, 99)):
    """Speed distribution — the anchor for choosing velocity thresholds.

    Low percentiles index the noise floor (a resting animal reading a large p50
    is jitter, not motion); the top percentile indexes real locomotion. Useful
    thresholds sit between the two, and if the two collapse together the track
    is too heavily smoothed to separate movement from noise.
    """
    v = np.asarray(velocity, dtype=float)
    v = v[np.isfinite(v)]
    if not len(v):
        return {q: float("nan") for q in qs}
    return {q: float(np.percentile(v, q)) for q in qs}


def state_label(loc_code, soc_code):
    """Single display label: social state when present, else locomotor."""
    if soc_code and soc_code != SOC_NONE:
        return SOCIAL_LABELS.get(soc_code, "")
    return LOCOMOTOR_LABELS.get(loc_code, "")


def state_color(loc_code, soc_code):
    if soc_code and soc_code != SOC_NONE:
        return SOCIAL_COLORS.get(soc_code, "#cccccc")
    return STATE_COLORS.get(loc_code, "#cccccc")
