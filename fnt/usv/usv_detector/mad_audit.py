"""Held-out audit of a MAD model: how many real calls it finds, per trial.

Training's validation numbers cannot answer the question a full batch run
needs answered. They are pixel Dice on tiles; with the default call-level split
they come from the same recordings the model trained on; and nobody has ever
marked a recording as exhaustively reviewed, so no recording in a training
project can say what the model *missed*. A study comparing call rates between
conditions needs exactly that number, per trial -- a detector that misses more
calls in one condition manufactures the effect being tested for.

So the audit is a separate set of short clips, cut from recordings no training
project has touched, labelled exhaustively, and scored at the level a person
works at: one call, found or missed.

**Drawing** (:func:`draw_audit`) samples per trial, spread over microphones and
day/night, from two strata recorded on every clip:

* ``random`` -- a uniformly random window. Unbiased, but a random minute of a
  nine-day recording is usually silent, so these mostly test false positives.
* ``active`` -- a window chosen by a crude *model-independent* tonal-energy
  screen, so it is likely to hold calls. Model-independent on purpose: picking
  windows where the model already fired would select exactly the places it
  works and report a flattering recall.

Every trial gets the same mix, so a between-condition comparison of recall is
like for like. The two strata are reported separately as well as pooled.

The clips are written into their own MAD project, never the training one, and
the scorer checks the model's training registry for any overlap with them.

**Scoring** (:func:`score_audit`) runs the model over every clip marked review
complete, through the same extract -> merge path a real run uses, at each
threshold separately (not by re-filtering one pass, which is not what a run at
that threshold would store), and matches detections to labels with
:func:`mad_eval.match_boxes`. Confidence intervals bootstrap over clips within
a trial, and over trials then clips within a condition, since clips from one
trial share its animals and its room.
"""
from __future__ import annotations

import csv
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Callable, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

AUDIT_MANIFEST = "audit_manifest.csv"
CLIP_DIR = "clips"
RESULTS_DIR = "results"
README_NAME = "README_audit.txt"

#: Avisoft multichannel naming: ch<mic><YYYY-MM-DD>_<HH-MM-SS>_<seq>.wav
_NAME = re.compile(
    r"^ch(\d)(\d{4}-\d{2}-\d{2})_(\d{2}-\d{2}-\d{2})_(\d+)\.wav$", re.I)

MANIFEST_FIELDS = [
    "clip", "trial", "condition", "mic", "phase", "stratum", "source_wav",
    "offset_s", "offset_samples", "sample_rate", "window_s", "activity",
    "seed", "drawn_at",
]

ProgressFn = Callable[[str], None]


# ----------------------------------------------------------------------
# Recordings
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class Recording:
    path: str
    trial: str
    condition: str
    mic: str
    start: datetime


def parse_recording(path: str, trial: str, condition: str
                    ) -> Optional[Recording]:
    m = _NAME.match(os.path.basename(path))
    if not m:
        return None
    try:
        start = datetime.strptime(f"{m.group(2)} {m.group(3)}",
                                  "%Y-%m-%d %H-%M-%S")
    except ValueError:
        return None
    return Recording(path=path, trial=trial, condition=condition,
                     mic=m.group(1), start=start)


def phase_of(t: datetime, day_start: int = 7, day_end: int = 19) -> str:
    """'day' or 'night' by clock hour. Voles are most vocal around dark, so
    an audit drawn only from daytime would miss the bulk of the calling."""
    return "day" if day_start <= t.hour < day_end else "night"


def condition_of(folder: str, keywords: Sequence[str]) -> str:
    low = os.path.basename(os.path.normpath(folder)).lower()
    for k in keywords:
        if k.lower() in low:
            return k.lower()
    return ""


def trial_of(folder: str) -> str:
    """'T005' from 'T005_prairie_anosmic-saline'; the folder name otherwise."""
    name = os.path.basename(os.path.normpath(folder))
    m = re.match(r"^(T\d+)", name, re.I)
    return m.group(1).upper() if m else name


def project_recordings(project_dirs: Iterable[str]) -> set:
    """Normalised basenames of every recording registered in these projects.

    Basenames, not paths: a recording copied elsewhere (a sample folder, a
    packed project) is still the same recording, and its labels would still be
    training data.
    """
    from .mad_project import MADProjectConfig
    out = set()
    for d in project_dirs:
        if not d or not os.path.isdir(d):
            continue
        try:
            cfg = MADProjectConfig.load(d)
        except Exception:
            continue
        for e in cfg.audio_entries():
            out.add(os.path.basename(e.path).lower())
    return out


def labelled_recordings(folder: str) -> set:
    """Lower-cased basenames in ``folder`` whose sidecar holds any example
    (label or rejection).

    A labelled recording can join a training project at any time, so it is
    never a safe audit source. One listing, then only the sidecars that exist
    are opened -- probing each of five thousand recordings for every legacy
    sidecar name costs tens of thousands of round trips on a network share.
    """
    from .fnt_mask_store import LEGACY_MAD_SUFFIXES, MAD_SUFFIX, td_count
    suffixes = tuple(x.lower() for x in (MAD_SUFFIX,) + LEGACY_MAD_SUFFIXES)
    out = set()
    try:
        names = os.listdir(folder)
    except OSError:
        return out
    for n in names:
        low = n.lower()
        suf = next((x for x in suffixes if low.endswith(x)), None)
        if suf is None:
            continue
        try:
            labelled = td_count(os.path.join(folder, n)) > 0
        except Exception:
            labelled = True     # unreadable: unsafe rather than a leak
        if labelled:
            out.add(low[: -len(suf)] + ".wav")
    return out


# ----------------------------------------------------------------------
# Model-independent activity screen
# ----------------------------------------------------------------------
def activity_per_window(audio: np.ndarray, sr: int, window_s: float, *,
                        band_hz: Tuple[float, float] = (18_000, 100_000),
                        peak_db: float = 12.0, max_width: float = 0.15,
                        nfft: int = 512) -> np.ndarray:
    """Tonal-frame count in each consecutive ``window_s`` window.

    A frame counts when its strongest in-band bin stands ``peak_db`` above that
    bin's median (the window's noise floor) while under ``max_width`` of the
    band is lit -- a narrow ridge, as a USV is, rather than a broadband click
    or knock. Deliberately crude and deliberately unlike the model: it only
    has to tell "something tonal happened here" from "nothing did", without
    sharing the model's blind spots.
    """
    x = np.asarray(audio, dtype=np.float32)
    win = int(round(window_s * sr))
    if win <= 0 or x.size < win:
        return np.zeros(0)
    n_win = x.size // win
    freqs = np.fft.rfftfreq(nfft, 1.0 / sr)
    band = (freqs >= band_hz[0]) & (freqs <= band_hz[1])
    if not band.any():
        return np.zeros(n_win)
    taper = np.hanning(nfft).astype(np.float32)
    out = np.zeros(n_win)
    for w in range(n_win):
        seg = x[w * win:(w + 1) * win]
        n_fr = seg.size // nfft
        if n_fr < 4:
            continue
        frames = seg[:n_fr * nfft].reshape(n_fr, nfft) * taper
        p = np.abs(np.fft.rfft(frames, axis=1))[:, band] ** 2
        db = 10.0 * np.log10(p + 1e-20)
        excess = db - np.median(db, axis=0, keepdims=True)
        peak = excess.max(axis=1)
        width = (excess > 6.0).mean(axis=1)
        out[w] = float(np.count_nonzero((peak > peak_db) & (width < max_width)))
    return out


# ----------------------------------------------------------------------
# Drawing
# ----------------------------------------------------------------------
def _round_robin(rng, strata: List[Tuple[str, str]], n: int):
    """``n`` strata, cycling through a seeded shuffle so every mic and both
    phases are represented before any repeats."""
    order = list(strata)
    rng.shuffle(order)
    return [order[i % len(order)] for i in range(n)] if order else []


def draw_audit(source_dirs: Sequence[str], audit_dir: str, *,
               per_trial_active: int = 6, per_trial_random: int = 4,
               window_s: float = 60.0, seed: int = 20261009,
               exclude_projects: Sequence[str] = (),
               condition_keywords: Sequence[str] = ("mmz", "saline"),
               edge_hours: float = 2.0, screen_per_stratum: int = 2,
               min_activity: float = 20.0,
               progress: Optional[ProgressFn] = None) -> List[Dict]:
    """Cut audit clips from ``source_dirs`` (one folder per trial) into a new
    MAD project at ``audit_dir``. Returns the manifest rows.

    A seed is right here even though import sampling dropped its own: this is
    the measuring instrument, and two people scoring two models against it
    must be scoring against the same clips.
    """
    import soundfile as sf
    from .mad_project import MADProjectConfig, create_mad_project
    from .mad_registry import RegisteredFile
    from .spectrogram import load_audio

    say = progress or (lambda m: None)
    if os.path.isfile(os.path.join(audit_dir, AUDIT_MANIFEST)):
        raise RuntimeError(
            f"{audit_dir} already holds an audit. Drawing again would replace "
            "clips someone may already have labelled -- use a new folder.")
    rng = np.random.default_rng(seed)
    excluded = project_recordings(exclude_projects)
    clip_dir = os.path.join(audit_dir, CLIP_DIR)
    os.makedirs(clip_dir, exist_ok=True)
    drawn_at = datetime.now().isoformat(timespec="seconds")
    rows: List[Dict] = []

    for src in source_dirs:
        trial = trial_of(src)
        cond = condition_of(src, condition_keywords)
        recs = [r for r in (parse_recording(os.path.join(src, n), trial, cond)
                            for n in sorted(os.listdir(src))
                            if n.lower().endswith(".wav"))
                if r is not None]
        if not recs:
            say(f"{trial}: no recordings matched the naming pattern; skipped")
            continue
        t0 = min(r.start for r in recs)
        t1 = max(r.start for r in recs) + timedelta(minutes=10)
        edge = timedelta(hours=edge_hours)
        pool = [r for r in recs
                if t0 + edge <= r.start <= t1 - edge - timedelta(minutes=10)
                and os.path.basename(r.path).lower() not in excluded]
        labelled = labelled_recordings(src)
        pool = [r for r in pool
                if os.path.basename(r.path).lower() not in labelled]
        strata: Dict[Tuple[str, str], List[Recording]] = {}
        for r in pool:
            strata.setdefault((r.mic, phase_of(r.start)), []).append(r)
        keys = sorted(strata)
        say(f"{trial} ({cond or '?'}): {len(pool)} eligible of {len(recs)} "
            f"recordings, {len(keys)} mic x day/night strata")
        used = set()

        def _clip(rec: Recording, offset: float, stratum: str,
                  activity: float, audio=None, sr=None):
            if audio is None:
                audio, sr = load_audio(rec.path)
            # The sample is the unit of record: a clip's labels map back to
            # its source by this offset, and at 250 kHz one rounded
            # millisecond is 250 samples of misregistration.
            a = int(round(offset * sr))
            offset = a / float(sr)
            seg = audio[a:a + int(round(window_s * sr))]
            stem = os.path.splitext(os.path.basename(rec.path))[0]
            name = f"{trial}__{stem}__{offset:07.1f}s.wav"
            out = os.path.join(clip_dir, name)
            # Bit-exact: the sources decode to 16-bit integers, which this
            # writes back unchanged, so the model sees the same spectrogram
            # here as it would in the full recording.
            sf.write(out, np.clip(np.round(seg * 32768.0), -32768, 32767)
                     .astype(np.int16), sr, subtype="PCM_16")
            rows.append({
                "clip": name, "trial": trial, "condition": cond,
                "mic": rec.mic, "phase": phase_of(rec.start),
                "stratum": stratum, "source_wav": rec.path,
                "offset_s": round(offset, 6), "offset_samples": a,
                "sample_rate": int(sr), "window_s": window_s,
                "activity": round(float(activity), 1), "seed": seed,
                "drawn_at": drawn_at,
            })
            used.add((rec.path, round(offset, 1)))

        # -- active: screen a few recordings per stratum, sample a window in
        # proportion to its tonal activity.
        screened: Dict[Tuple[str, str], List[Tuple[float, Recording, float]]] = {}
        for k in keys:
            cands = list(strata[k])
            rng.shuffle(cands)
            for rec in cands[:screen_per_stratum]:
                try:
                    audio, sr = load_audio(rec.path)
                except Exception as ex:
                    say(f"  could not read {os.path.basename(rec.path)}: {ex}")
                    continue
                act = activity_per_window(audio, sr, window_s)
                for w, v in enumerate(act):
                    if v >= min_activity:
                        screened.setdefault(k, []).append((float(v), rec,
                                                           w * window_s))
        for k in _round_robin(rng, keys, per_trial_active):
            pool_k = screened.get(k) or [x for v in screened.values()
                                         for x in v]
            pool_k = [x for x in pool_k
                      if (x[1].path, round(x[2], 1)) not in used]
            if not pool_k:
                say(f"  {trial}: no active window left to draw")
                break
            wts = np.array([x[0] for x in pool_k], dtype=float)
            v, rec, off = pool_k[int(rng.choice(len(pool_k), p=wts / wts.sum()))]
            _clip(rec, off, "active", v)

        # -- random: a uniform window in a random recording of each stratum.
        for k in _round_robin(rng, keys, per_trial_random):
            rec = strata[k][int(rng.integers(len(strata[k])))]
            try:
                audio, sr = load_audio(rec.path)
            except Exception as ex:
                say(f"  could not read {os.path.basename(rec.path)}: {ex}")
                continue
            dur = len(audio) / float(sr)
            off = float(rng.uniform(0.0, max(0.0, dur - window_s)))
            act = activity_per_window(
                audio[int(off * sr):int((off + window_s) * sr)], sr, window_s)
            _clip(rec, off, "random", act[0] if act.size else 0.0, audio, sr)
        say(f"  {trial}: {sum(1 for r in rows if r['trial'] == trial)} clips")

    with open(os.path.join(audit_dir, AUDIT_MANIFEST), "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=MANIFEST_FIELDS)
        w.writeheader()
        w.writerows(rows)

    if not os.path.isfile(os.path.join(audit_dir, "mad_project_info.json")):
        create_mad_project(audit_dir)
    cfg = MADProjectConfig.load(audit_dir)
    cfg.set_audio_entries([
        RegisteredFile.from_path(os.path.join(clip_dir, r["clip"]),
                                 embedded=True) for r in rows])
    cfg.save()
    _write_readme(audit_dir, rows, window_s)
    return rows


def _write_readme(audit_dir: str, rows: List[Dict], window_s: float) -> None:
    n = len(rows)
    with open(os.path.join(audit_dir, README_NAME), "w") as f:
        f.write(
            f"MAD audit set -- {n} clips of {window_s:g} s\n\n"
            "This project measures a model. Never train on it, and never add\n"
            "its source recordings to a training project.\n\n"
            "For each clip:\n"
            "  1. Label EVERY call, the same way as in training: each element\n"
            "     (a fundamental and each harmonic) as its own mask. Faint\n"
            "     calls too -- they are what this is here to find.\n"
            "  2. Don't run inference on these clips first. Labelling blind\n"
            "     keeps the reference from leaning toward what a model saw.\n"
            "  3. When nothing is left: right-click the clip in the Audio list\n"
            "     and mark it review complete. Only complete clips are scored;\n"
            "     an empty clip marked complete is a real result (any\n"
            "     detection in it is a false positive).\n\n"
            "Score a model with:\n"
            "  mad audit-score --audit <this folder> --model <run folder>\n")


# ----------------------------------------------------------------------
# Scoring
# ----------------------------------------------------------------------
def read_manifest(audit_dir: str) -> List[Dict]:
    with open(os.path.join(audit_dir, AUDIT_MANIFEST), newline="") as f:
        return list(csv.DictReader(f))


def _prf(tp: int, fp: int, fn: int) -> Dict:
    p = tp / (tp + fp) if tp + fp else None
    r = tp / (tp + fn) if tp + fn else None
    f1 = (2 * p * r / (p + r)) if p and r else (0.0 if p is not None
                                                 and r is not None else None)
    return {"tp": tp, "fp": fp, "fn": fn,
            "precision": None if p is None else round(p, 4),
            "recall": None if r is None else round(r, 4),
            "f1": None if f1 is None else round(f1, 4)}


def _ratio(num, den):
    return num / den if den else np.nan


def _boot_clips(counts: np.ndarray, rng, n_boot: int):
    """(precision, recall) bootstrap draws resampling clips. ``counts`` is
    (n_clips, 3) of tp/fp/fn."""
    n = len(counts)
    if n == 0:
        return np.full(n_boot, np.nan), np.full(n_boot, np.nan)
    idx = rng.integers(0, n, size=(n_boot, n))
    s = counts[idx].sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        p = s[:, 0] / (s[:, 0] + s[:, 1])
        r = s[:, 0] / (s[:, 0] + s[:, 2])
    return p, r


def _boot_nested(by_trial: Dict[str, np.ndarray], rng, n_boot: int):
    """Resample trials, then clips within each drawn trial. With three or
    four trials per condition the trial is the unit that varies most, and a
    clip-only bootstrap would pretend sixty clips were sixty independent
    observations."""
    trials = [t for t, c in by_trial.items() if len(c)]
    if not trials:
        return np.full(n_boot, np.nan), np.full(n_boot, np.nan)
    ps, rs = np.empty(n_boot), np.empty(n_boot)
    for b in range(n_boot):
        tot = np.zeros(3)
        for t in rng.choice(trials, size=len(trials), replace=True):
            c = by_trial[t]
            tot += c[rng.integers(0, len(c), size=len(c))].sum(axis=0)
        ps[b] = _ratio(tot[0], tot[0] + tot[1])
        rs[b] = _ratio(tot[0], tot[0] + tot[2])
    return ps, rs


def _ci(draws) -> Optional[List[float]]:
    d = np.asarray(draws, dtype=float)
    d = d[np.isfinite(d)]
    if d.size < 10:
        return None
    return [round(float(np.percentile(d, 2.5)), 4),
            round(float(np.percentile(d, 97.5)), 4)]


def training_overlap(model_path: str, manifest: List[Dict]) -> List[str]:
    """Audit sources (or clips) the model's training project registered.

    Read from the run's own training_summary, so it checks the project the
    model was actually trained from.
    """
    run_dir = os.path.dirname(os.path.abspath(model_path))
    summ = os.path.join(run_dir, "training_summary.json")
    try:
        with open(summ) as f:
            proj = (json.load(f).get("config") or {}).get("project_dir", "")
    except Exception:
        return []
    registered = project_recordings([proj])
    hits = []
    for r in manifest:
        for name in (os.path.basename(r["source_wav"]), r["clip"]):
            if name.lower() in registered:
                hits.append(name)
    return sorted(set(hits))


def score_audit(audit_dir: str, model_path: str, *,
                thresholds: Sequence[float] = (0.3, 0.4, 0.5, 0.6, 0.7),
                min_blob_pixels: int = 50, merge: bool = True,
                merge_gap_s: float = 0.01, iou_min: float = 0.3,
                device: str = "auto", n_boot: int = 1000, seed: int = 0,
                progress: Optional[ProgressFn] = None,
                write: bool = True) -> Dict:
    """Score ``model_path`` on the audit's review-complete clips."""
    from .fnt_mask_store import is_review_complete, masks_sibling_path
    from .mad_csv_rebuild import rows_for_wav
    from .mad_dataset import db_range_for
    from .mad_eval import match_boxes, rows_to_boxes
    from .mad_inference import (
        MADInferenceConfig, blobs_to_rows, compute_full_spec_image,
        extract_blobs, infer_probability_mask, load_model,
        merge_consecutive_blobs, model_label)
    from .spectrogram import load_audio

    say = progress or (lambda m: None)
    if os.path.isdir(model_path):
        model_path = os.path.join(model_path, "weights.pt")
    manifest = read_manifest(audit_dir)
    thresholds = sorted({round(float(t), 4) for t in thresholds})
    icfg = MADInferenceConfig(model_path=model_path)
    model, ckpt, dev = load_model(model_path, device)
    nperseg = int(ckpt.get("nperseg", 512))
    noverlap = int(ckpt.get("noverlap", 384))
    nfft = int(ckpt.get("nfft", 1024))
    db_min = float(ckpt.get("db_min", -100.0))
    db_max = float(ckpt.get("db_max", -20.0))
    db_norm = str(ckpt.get("db_norm", "fixed"))
    tile_f = int(ckpt.get("tile_freq_bins", 512))
    tile_t = int(ckpt.get("tile_time_frames", 256))

    clips_out: List[Dict] = []
    n_incomplete = 0
    for i, r in enumerate(manifest):
        path = os.path.join(audit_dir, CLIP_DIR, r["clip"])
        h5 = masks_sibling_path(path)
        if not (os.path.isfile(h5) and is_review_complete(h5)):
            n_incomplete += 1
            continue
        say(f"[{i + 1}/{len(manifest)}] {r['clip']}")
        truth = rows_to_boxes(rows_for_wav(path), source="truth")
        audio, sr = load_audio(path)
        f_lo, f_hi = db_range_for(db_norm, db_min, db_max, audio=audio,
                                  sample_rate=sr, nperseg=nperseg,
                                  noverlap=noverlap, nfft=nfft)
        spec = compute_full_spec_image(audio.astype(np.float32), sr,
                                       nperseg=nperseg, noverlap=noverlap,
                                       nfft=nfft, db_min=f_lo, db_max=f_hi)
        prob = infer_probability_mask(
            model, spec, tile_freq_bins=tile_f, tile_time_frames=tile_t,
            overlap_fraction=icfg.tile_overlap_fraction, device=dev,
            batch_size=icfg.batch_size, use_amp=icfg.amp)
        dt = (nperseg - noverlap) / float(sr)
        per_t = {}
        for t in thresholds:
            blobs = extract_blobs(prob, threshold=t,
                                  min_blob_pixels=min_blob_pixels,
                                  include_mask=True, spec=spec)
            if merge:
                blobs = merge_consecutive_blobs(
                    blobs, max_gap_frames=int(round(merge_gap_s / dt)),
                    require_freq_overlap=icfg.merge_require_freq_overlap,
                    max_freq_gap_bins=icfg.merge_max_freq_gap_bins,
                    concurrent_fraction=icfg.merge_concurrent_fraction)
            preds = rows_to_boxes(blobs_to_rows(
                blobs, nperseg=nperseg, noverlap=noverlap, nfft=nfft, sr=sr))
            c, _ = match_boxes(preds, truth, iou_min=iou_min)
            per_t[f"{t:g}"] = [c.tp, c.fp, c.fn]
        clips_out.append({**{k: r[k] for k in ("clip", "trial", "condition",
                                               "mic", "phase", "stratum")},
                          "n_truth": len(truth), "counts": per_t})

    rng = np.random.default_rng(seed)
    per_threshold = []
    for t in thresholds:
        key = f"{t:g}"
        arr = lambda cl: np.array([c["counts"][key] for c in cl],
                                  dtype=float).reshape(-1, 3)

        def block(cl, nested=False):
            a = arr(cl)
            out = _prf(*[int(v) for v in a.sum(axis=0)]) if len(a) else \
                _prf(0, 0, 0)
            out["n_clips"] = len(cl)
            if nested:
                bt = {}
                for c in cl:
                    bt.setdefault(c["trial"], []).append(c["counts"][key])
                p, rr = _boot_nested(
                    {k: np.array(v, dtype=float) for k, v in bt.items()},
                    rng, n_boot)
            else:
                p, rr = _boot_clips(a, rng, n_boot)
            out["precision_ci"], out["recall_ci"] = _ci(p), _ci(rr)
            return out, rr

        overall, _ = block(clips_out, nested=True)
        by_trial = {k: block([c for c in clips_out if c["trial"] == k])[0]
                    for k in sorted({c["trial"] for c in clips_out})}
        by_stratum = {k: block([c for c in clips_out if c["stratum"] == k],
                               nested=True)[0]
                      for k in sorted({c["stratum"] for c in clips_out})}
        conds = sorted({c["condition"] for c in clips_out if c["condition"]})
        by_condition, rdraws = {}, {}
        for k in conds:
            by_condition[k], rdraws[k] = block(
                [c for c in clips_out if c["condition"] == k], nested=True)
        diff = None
        if len(conds) == 2:
            a, b = conds
            d = rdraws[a] - rdraws[b]
            ra, rb = by_condition[a]["recall"], by_condition[b]["recall"]
            diff = {"contrast": f"{a} - {b}",
                    "recall_diff": (None if ra is None or rb is None
                                    else round(ra - rb, 4)),
                    "ci": _ci(d)}
        per_threshold.append({
            "threshold": t, "overall": overall, "by_condition": by_condition,
            "recall_diff": diff, "by_trial": by_trial,
            "by_stratum": by_stratum})

    result = {
        "model": model_label(model_path), "model_path": model_path,
        "audit_dir": audit_dir,
        "scored_at": datetime.now().isoformat(timespec="seconds"),
        "settings": {"min_blob_pixels": min_blob_pixels, "merge": merge,
                     "merge_gap_s": merge_gap_s, "iou_min": iou_min,
                     "n_boot": n_boot},
        "n_clips": len(manifest), "n_scored": len(clips_out),
        "n_incomplete": n_incomplete,
        "n_truth_calls": int(sum(c["n_truth"] for c in clips_out)),
        "training_overlap": training_overlap(model_path, manifest),
        "per_threshold": per_threshold, "clips": clips_out,
    }
    if write:
        write_results(result)
    return result


def write_results(result: Dict) -> Tuple[str, str]:
    """JSON + flat CSV into the audit's results folder; the JSON is copied
    beside the model's weights too, so a run folder carries its own audit."""
    res_dir = os.path.join(result["audit_dir"], RESULTS_DIR)
    os.makedirs(res_dir, exist_ok=True)
    base = os.path.join(res_dir, f"audit_{result['model']}")
    with open(base + ".json", "w") as f:
        json.dump(result, f, indent=2)
    with open(base + ".csv", "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["threshold", "group_type", "group", "n_clips", "tp", "fp",
                    "fn", "precision", "precision_lo", "precision_hi",
                    "recall", "recall_lo", "recall_hi", "f1"])
        for row in result["per_threshold"]:
            groups = [("overall", "all", row["overall"])]
            groups += [("condition", k, v)
                       for k, v in row["by_condition"].items()]
            groups += [("trial", k, v) for k, v in row["by_trial"].items()]
            groups += [("stratum", k, v)
                       for k, v in row["by_stratum"].items()]
            for gt, g, v in groups:
                pci = v.get("precision_ci") or [None, None]
                rci = v.get("recall_ci") or [None, None]
                w.writerow([row["threshold"], gt, g, v["n_clips"], v["tp"],
                            v["fp"], v["fn"], v["precision"], pci[0], pci[1],
                            v["recall"], rci[0], rci[1], v["f1"]])
    try:
        run_dir = os.path.dirname(os.path.abspath(result["model_path"]))
        name = os.path.basename(os.path.normpath(result["audit_dir"]))
        with open(os.path.join(run_dir, f"audit_{name}.json"), "w") as f:
            json.dump({k: v for k, v in result.items() if k != "clips"}, f,
                      indent=2)
    except Exception:
        pass
    return base + ".json", base + ".csv"


def format_table(result: Dict, threshold: Optional[float] = None) -> str:
    """Plain-text summary at one threshold (best overall F1 if not given)."""
    rows = result["per_threshold"]
    if not rows:
        return "nothing scored"
    if threshold is None:
        row = max(rows, key=lambda r: r["overall"].get("f1") or 0.0)
    else:
        row = min(rows, key=lambda r: abs(r["threshold"] - threshold))

    def fmt(v, ci):
        if v is None:
            return "   —          "
        s = f"{v:.3f}"
        return s + (f" [{ci[0]:.2f}-{ci[1]:.2f}]" if ci else "          ")

    out = [f"model {result['model']}  threshold {row['threshold']:g}  "
           f"({result['n_scored']}/{result['n_clips']} clips scored, "
           f"{result['n_truth_calls']} labelled calls)"]
    if result.get("training_overlap"):
        out.append("!! LEAK: the training project includes audit sources: "
                   + ", ".join(result["training_overlap"][:5]))
    out.append(f"{'':12s} {'clips':>5s} {'calls':>6s}  "
               f"{'recall [95% CI]':22s} {'precision [95% CI]':22s}")

    def line(name, v):
        return (f"{name:12s} {v['n_clips']:5d} {v['tp'] + v['fn']:6d}  "
                f"{fmt(v['recall'], v.get('recall_ci')):22s} "
                f"{fmt(v['precision'], v.get('precision_ci')):22s}")
    out.append(line("ALL", row["overall"]))
    for k, v in row["by_condition"].items():
        out.append(line(k, v))
    if row.get("recall_diff"):
        d = row["recall_diff"]
        ci = d.get("ci")
        out.append(f"recall {d['contrast']}: "
                   f"{d['recall_diff'] if d['recall_diff'] is not None else '—'}"
                   + (f"  [{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else ""))
    for k, v in row["by_stratum"].items():
        out.append(line(k, v))
    for k, v in row["by_trial"].items():
        out.append(line(k, v))
    return "\n".join(out)
