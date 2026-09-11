"""Call-level evaluation of a MAD checkpoint, swept across thresholds.

Training already reports validation Dice, but Dice is a **pixel** score on
**tiles**. It answers "how well does the mask overlap?", not the question that
decides whether a model is ready for a 3,000-file run: *how many real calls will
it find, and how much junk will I have to reject?*

So this evaluates at the level the user actually works at — one call, accepted or
rejected — by matching predicted blobs against the calls the user confirmed
(hand-drawn labels *and* accepted predictions) and reporting precision / recall /
F1. Because every prediction carries a probability score,
one inference pass yields the whole threshold curve: run the model once at a
permissive threshold, then re-score the same detections at each candidate cutoff.
That turns "pick a threshold" from a guess into a table.

Matching rule: a prediction matches a label when their time spans overlap and
their frequency spans overlap, with IoU over the (time x frequency) boxes at or
above ``iou_min``. Greedy, highest-scoring prediction first, one-to-one — the
same accounting a human does when reviewing.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, List, Optional, Sequence, Tuple

import numpy as np


@dataclass
class Box:
    """A call as a (time, frequency) rectangle, plus its score."""
    t0: float
    t1: float
    f0: float
    f1: float
    score: float = 1.0

    def area(self) -> float:
        return max(0.0, self.t1 - self.t0) * max(0.0, self.f1 - self.f0)


def iou(a: Box, b: Box) -> float:
    """Intersection-over-union of two time/frequency boxes."""
    it = min(a.t1, b.t1) - max(a.t0, b.t0)
    if it <= 0:
        return 0.0
    jf = min(a.f1, b.f1) - max(a.f0, b.f0)
    if jf <= 0:
        return 0.0
    inter = it * jf
    union = a.area() + b.area() - inter
    return inter / union if union > 0 else 0.0


@dataclass
class Counts:
    tp: int = 0
    fp: int = 0
    fn: int = 0

    @property
    def precision(self) -> float:
        d = self.tp + self.fp
        return self.tp / d if d else 0.0

    @property
    def recall(self) -> float:
        d = self.tp + self.fn
        return self.tp / d if d else 0.0

    @property
    def f1(self) -> float:
        p, r = self.precision, self.recall
        return (2 * p * r / (p + r)) if (p + r) else 0.0

    @property
    def n_pred(self) -> int:
        return self.tp + self.fp

    def as_dict(self) -> Dict:
        # ``n_pred`` disambiguates the degenerate row. Precision is 0/0 when a
        # threshold is so high that nothing is detected; reporting it as 0.0
        # (needed so f1 stays 0 and such a threshold is never chosen as "best")
        # otherwise reads as "the model got everything wrong" rather than "the
        # model returned nothing". Callers show "—" when n_pred is 0.
        return {'tp': self.tp, 'fp': self.fp, 'fn': self.fn,
                'n_pred': self.n_pred,
                'precision': round(self.precision, 4),
                'recall': round(self.recall, 4),
                'f1': round(self.f1, 4)}


def match_boxes(preds: Sequence[Box], labels: Sequence[Box],
                iou_min: float = 0.3) -> Tuple[Counts, List[Tuple[int, int]]]:
    """Greedy one-to-one matching, best-scoring prediction first.

    Greedy-by-score (rather than globally optimal assignment) is deliberate: it
    mirrors review order, where the most confident detection claims a call first,
    and it never inflates the score relative to an optimal matcher by more than a
    pathological tie.
    """
    order = sorted(range(len(preds)), key=lambda i: -preds[i].score)
    taken = set()
    pairs: List[Tuple[int, int]] = []
    for pi in order:
        best_j, best_iou = -1, iou_min
        for j, lb in enumerate(labels):
            if j in taken:
                continue
            v = iou(preds[pi], lb)
            if v >= best_iou:
                best_j, best_iou = j, v
        if best_j >= 0:
            taken.add(best_j)
            pairs.append((pi, best_j))
    c = Counts(tp=len(pairs),
               fp=len(preds) - len(pairs),
               fn=len(labels) - len(pairs))
    return c, pairs


def _to_box(r: Dict) -> Optional[Box]:
    """One CSV row as a time/frequency box, or None if its numbers are unusable."""
    try:
        return Box(
            t0=float(r.get('start_s', 0.0)),
            t1=float(r.get('stop_s', 0.0)),
            f0=float(r.get('min_freq_hz', 0.0)),
            f1=float(r.get('max_freq_hz', 0.0)),
            score=float(r.get('score', 1.0) or 1.0),
        )
    except (TypeError, ValueError):
        return None


def _is_prediction(r: Dict) -> bool:
    """Model output, as opposed to a hand-drawn label. The unified CSV
    distinguishes them by blob_id type: int for predictions, string id for
    hand-labels."""
    return isinstance(r.get('blob_id'), int)


def _status(r: Dict) -> str:
    return (r.get('status') or 'pending')


def rows_to_boxes(rows: Sequence[Dict], source: Optional[str] = None
                  ) -> List[Box]:
    """Convert CSV rows to boxes, optionally filtered by role.

    ``source='prediction'`` selects model output. ``source='truth'`` selects
    **every call a human affirmed is real** — that is hand-drawn labels *and*
    predictions the user accepted.

    Including accepted predictions is the whole point. In MAD's workflow most
    confirmed calls arrive by accepting a prediction, not by painting one, and
    accepting keeps the row's int blob_id while only flipping status. Ground
    truth built from hand-labels alone therefore omits the majority of the
    user's confirmed calls, and the model's correct re-detections of them get
    scored as false positives — reporting near-zero precision for a model that
    was right about every call.

    Rejected rows are excluded: a rejection is a recorded "not a call", so
    re-detecting one is genuinely a false positive. Pending predictions are
    excluded too — they are unjudged, not confirmed; see :func:`count_unreviewed`
    for why that has to be surfaced to the reader.
    """
    out: List[Box] = []
    for r in rows:
        is_pred = _is_prediction(r)
        if source == 'prediction' and not is_pred:
            continue
        if source == 'truth':
            status = _status(r)
            if status == 'rejected':
                continue
            # A hand-label is affirmed by existing; a prediction has to have
            # been accepted.
            if is_pred and status != 'accepted':
                continue
        box = _to_box(r)
        if box is not None:
            out.append(box)
    return out


def count_unreviewed(rows: Sequence[Dict]) -> int:
    """Predictions on this file that the user has neither accepted nor rejected.

    These are the one remaining way to get a misleading precision number: an
    unreviewed prediction is not ground truth, so if the model re-detects it the
    match is scored as a false positive even though nobody has said whether it
    is one. Evaluating a partially-reviewed file understates precision by
    roughly this count, so every consumer reports it rather than hiding it.
    """
    return sum(1 for r in rows
               if _is_prediction(r) and _status(r) == 'pending')


DEFAULT_THRESHOLDS = (0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90)


# ----------------------------------------------------------------------
# Reviewed regions — where the truth is actually known
# ----------------------------------------------------------------------
#: Audio kept either side of a judged span when running the model, so a call at
#: a window edge is seen with the context the model was trained on rather than
#: against a hard cut. Scoring uses the unpadded span; this is only about what
#: the network gets to look at.
DEFAULT_CONTEXT_S = 1.0

#: Judged spans closer than this are merged into one inference window. Two tile
#: widths' worth: below it, running them separately costs more in per-window
#: overhead than the audio saved.
DEFAULT_MERGE_GAP_S = 0.5


def reviewed_windows(rows: Sequence[Dict],
                     merge_gap_s: float = DEFAULT_MERGE_GAP_S
                     ) -> List[Tuple[float, float]]:
    """Time spans of a recording whose content the user has actually judged.

    Scoring a whole file punishes the model for detections in regions nobody
    has looked at. Those are *unjudged*, not wrong — and on a partially
    reviewed recording they are most of what it finds, which is why whole-file
    precision reads far worse than the model deserves.

    A span is judged when it holds a confirmed call (hand-drawn or accepted) or
    a **rejected** one. Rejections matter as much as confirmations here: a
    rejection is a recorded "there is no call here", which is exactly the
    evidence a false positive has to be measured against. Pending predictions
    contribute nothing — nobody has said what they are.

    Within a judged span the truth is taken to be complete, which is the same
    assumption training makes: ``supervision_weight`` marks a whole time column
    supervised if any pixel in it is labelled, so the model was already taught
    that unlabelled pixels in a labelled column are background. Scoring on the
    same footing keeps the metric and the training signal talking about the
    same thing.
    """
    spans: List[Tuple[float, float]] = []
    for r in rows:
        status = _status(r)
        is_pred = _is_prediction(r)
        judged = (status == 'rejected'
                  or (not is_pred and status != 'rejected')
                  or (is_pred and status == 'accepted'))
        if not judged:
            continue
        box = _to_box(r)
        if box is None or box.t1 <= box.t0:
            continue
        spans.append((box.t0, box.t1))
    return _merge_spans(spans, merge_gap_s)


def _merge_spans(spans: Sequence[Tuple[float, float]], gap: float
                 ) -> List[Tuple[float, float]]:
    """Union of spans, coalescing any pair closer than ``gap``."""
    if not spans:
        return []
    out: List[Tuple[float, float]] = []
    for t0, t1 in sorted(spans):
        if out and t0 - out[-1][1] <= gap:
            if t1 > out[-1][1]:
                out[-1] = (out[-1][0], t1)
        else:
            out.append((t0, t1))
    return out


def expand_windows(windows: Sequence[Tuple[float, float]],
                   context_s: float = DEFAULT_CONTEXT_S,
                   duration_s: Optional[float] = None,
                   merge_gap_s: float = DEFAULT_MERGE_GAP_S
                   ) -> List[Tuple[float, float]]:
    """Judged spans grown by ``context_s`` — what to actually run the model on.

    Kept separate from the scoring spans on purpose. The extra audio exists so
    the network sees a call in context; counting detections that land in it
    would be counting them in a region nobody judged.
    """
    grown = [(max(0.0, t0 - context_s),
              (t1 + context_s if duration_s is None
               else min(duration_s, t1 + context_s)))
             for t0, t1 in windows]
    return _merge_spans(grown, merge_gap_s)


def overlaps_any(box: Box, windows: Sequence[Tuple[float, float]]) -> bool:
    """Whether a call falls in any judged span.

    Overlap, not containment: a detection straddling the edge of a judged span
    is one the user would have been shown, so it counts.
    """
    for t0, t1 in windows:
        if box.t1 > t0 and box.t0 < t1:
            return True
    return False


def windows_duration(windows: Sequence[Tuple[float, float]]) -> float:
    return float(sum(max(0.0, t1 - t0) for t0, t1 in windows))


def score_in_windows(preds: Sequence[Box], labels: Sequence[Box],
                     windows: Optional[Sequence[Tuple[float, float]]],
                     iou_min: float = 0.3) -> Tuple[Counts, List[Tuple[int, int]]]:
    """Match, counting only calls inside judged spans.

    ``windows=None`` scores everything, which is what an exhaustively reviewed
    recording wants.
    """
    if windows is None:
        return match_boxes(preds, labels, iou_min=iou_min)
    p = [b for b in preds if overlaps_any(b, windows)]
    lb = [b for b in labels if overlaps_any(b, windows)]
    return match_boxes(p, lb, iou_min=iou_min)


@dataclass
class EvalResult:
    thresholds: List[float] = field(default_factory=list)
    per_threshold: List[Dict] = field(default_factory=list)
    n_labels: int = 0
    n_files: int = 0
    files: List[Dict] = field(default_factory=list)
    model_name: str = ""
    iou_min: float = 0.3
    # Pending predictions across the scored files. Nonzero means precision is
    # understated: those calls are unjudged, so re-detecting one scores as a
    # false positive. Consumers must show this, not bury it.
    n_unreviewed: int = 0
    #: 'file' scores whole recordings; 'reviewed' scores only judged spans.
    scope: str = 'file'
    #: Counts from exhaustively-reviewed recordings alone, same threshold rows.
    #: This is the only cohort whose RECALL can be believed — elsewhere a call
    #: nobody labelled is invisible, so missing it is never counted. Empty until
    #: at least one recording is flagged complete.
    per_threshold_exhaustive: List[Dict] = field(default_factory=list)
    n_exhaustive_files: int = 0
    n_exhaustive_labels: int = 0
    #: Seconds of audio actually scored, against the total in the scored files.
    #: The ratio is how much of the recording the numbers speak for.
    scored_s: float = 0.0
    total_s: float = 0.0

    def best_exhaustive(self, metric: str = 'f1') -> Optional[Dict]:
        if not self.per_threshold_exhaustive:
            return None
        return max(self.per_threshold_exhaustive,
                   key=lambda d: d.get(metric, 0.0))

    def best(self, metric: str = 'f1') -> Optional[Dict]:
        if not self.per_threshold:
            return None
        return max(self.per_threshold, key=lambda d: d.get(metric, 0.0))

    def as_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'iou_min': self.iou_min,
            'scope': self.scope,
            'n_files': self.n_files,
            'n_labels': self.n_labels,
            'n_unreviewed': self.n_unreviewed,
            'per_threshold': self.per_threshold,
            'per_threshold_exhaustive': self.per_threshold_exhaustive,
            'n_exhaustive_files': self.n_exhaustive_files,
            'n_exhaustive_labels': self.n_exhaustive_labels,
            'scored_s': round(self.scored_s, 2),
            'total_s': round(self.total_s, 2),
            'files': self.files,
        }

    @classmethod
    def from_dict(cls, d: Dict) -> "EvalResult":
        """Rebuild a stored result. Unknown/missing keys fall back to defaults
        so an eval.json written by an older build still loads for the trend."""
        r = cls()
        for k, v in (d or {}).items():
            if hasattr(r, k):
                setattr(r, k, v)
        r.thresholds = [float(p['threshold']) for p in r.per_threshold
                        if 'threshold' in p]
        return r


def evaluate_wavs(
    wav_paths: Sequence[str],
    cfg,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    iou_min: float = 0.3,
    progress: Optional[Callable[[int, int, str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> EvalResult:
    """Run ``cfg``'s model over held-out recordings and score it per call.

    The model runs **once per file** at the lowest threshold in the sweep; every
    higher threshold is then evaluated by filtering the same detections by score.
    That is what makes a full curve affordable — the expensive part (the tiled
    forward pass) is not repeated per threshold.

    Ground truth is every call the user affirmed on that recording — hand-drawn
    labels **and** accepted predictions (see :func:`rows_to_boxes`). Files with
    no confirmed calls are skipped and reported, since scoring against an empty
    ground truth would make precision look catastrophic for no reason. Files that
    still hold unreviewed predictions are scored but counted in
    ``n_unreviewed``, because those understate precision.
    """
    from .mad_inference import (
        blobs_to_rows, compute_full_spec_image, extract_blobs, load_model,
        infer_probability_mask, read_blob_csv,
    )
    from .mad_labels import pred_csv_sibling_path
    from .spectrogram import load_audio

    thresholds = sorted(set(round(float(t), 4) for t in thresholds))
    if not thresholds:
        thresholds = list(DEFAULT_THRESHOLDS)
    base_thr = thresholds[0]

    model, ckpt, device = load_model(cfg.model_path, cfg.device)
    nperseg = int(cfg.nperseg if cfg.nperseg is not None else ckpt.get('nperseg', 512))
    noverlap = int(cfg.noverlap if cfg.noverlap is not None else ckpt.get('noverlap', 384))
    nfft = int(cfg.nfft if cfg.nfft is not None else ckpt.get('nfft', 1024))
    db_min = float(cfg.db_min if cfg.db_min is not None else ckpt.get('db_min', -100.0))
    db_max = float(cfg.db_max if cfg.db_max is not None else ckpt.get('db_max', -20.0))
    db_norm = str(cfg.db_norm if getattr(cfg, 'db_norm', None) is not None
                  else ckpt.get('db_norm', 'fixed'))
    tile_f = int(ckpt.get('tile_freq_bins', cfg.tile_freq_bins))
    tile_t = int(ckpt.get('tile_time_frames', cfg.tile_time_frames))

    totals = {t: Counts() for t in thresholds}
    files_out: List[Dict] = []
    n_labels_total = 0
    n_unreviewed_total = 0
    n_scored = 0

    for i, wav in enumerate(wav_paths):
        if should_stop is not None and should_stop():
            break
        name = Path(wav).name
        if progress is not None:
            progress(i, len(wav_paths), name)

        # Ground truth comes from the store, which is where a review decision
        # is actually recorded; the CSV is an export and may not exist at all.
        # A legacy CSV is still read for recordings labelled before the store
        # was authoritative.
        try:
            from .mad_csv_rebuild import rows_for_wav
            rows = rows_for_wav(wav)
        except Exception:
            rows = []
        if not rows:
            csv_path = pred_csv_sibling_path(wav)
            try:
                rows = read_blob_csv(csv_path) if Path(csv_path).is_file() else []
            except Exception:
                rows = []
        labels = rows_to_boxes(rows, source='truth')
        if not labels:
            files_out.append({'name': name, 'skipped': 'no confirmed calls'})
            continue
        n_pending = count_unreviewed(rows)

        audio, sr = load_audio(wav)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        from .mad_dataset import db_range_for
        f_lo, f_hi = db_range_for(
            db_norm, db_min, db_max, audio=audio, sample_rate=sr,
            nperseg=nperseg, noverlap=noverlap, nfft=nfft)
        spec = compute_full_spec_image(
            audio.astype(np.float32), sr, nperseg=nperseg, noverlap=noverlap,
            nfft=nfft, db_min=f_lo, db_max=f_hi)
        prob = infer_probability_mask(
            model, spec, tile_freq_bins=tile_f, tile_time_frames=tile_t,
            overlap_fraction=cfg.tile_overlap_fraction, device=device,
            batch_size=cfg.batch_size, use_amp=cfg.amp,
            chunk_frames=cfg.chunk_frames)
        blobs = extract_blobs(prob, threshold=base_thr,
                              min_blob_pixels=cfg.min_blob_pixels,
                              include_mask=True, spec=spec)
        pred_rows = blobs_to_rows(blobs, nperseg=nperseg, noverlap=noverlap,
                                  nfft=nfft, sr=sr, db_min=db_min,
                                  db_max=db_max, spec=spec)
        preds_all = rows_to_boxes(pred_rows)

        n_labels_total += len(labels)
        n_unreviewed_total += n_pending
        n_scored += 1
        per_file = {'name': name, 'n_labels': len(labels),
                    'n_unreviewed': n_pending}
        for t in thresholds:
            # Re-scoring by the stored per-call score is what makes the sweep
            # cheap; only calls whose mean probability clears the cutoff survive.
            preds = [p for p in preds_all if p.score >= t]
            c, _ = match_boxes(preds, labels, iou_min=iou_min)
            totals[t].tp += c.tp
            totals[t].fp += c.fp
            totals[t].fn += c.fn
            per_file[f"{t:g}"] = c.as_dict()
        files_out.append(per_file)

    per_threshold = []
    for t in thresholds:
        d = {'threshold': t}
        d.update(totals[t].as_dict())
        per_threshold.append(d)

    return EvalResult(
        thresholds=list(thresholds),
        per_threshold=per_threshold,
        n_labels=n_labels_total,
        n_unreviewed=n_unreviewed_total,
        n_files=n_scored,
        files=files_out,
        model_name=Path(cfg.model_path).stem if cfg.model_path else "",
        iou_min=iou_min,
    )


# ----------------------------------------------------------------------
# Region-scoped evaluation
# ----------------------------------------------------------------------
def _rows_for(wav: str) -> List[Dict]:
    """Every stored row for a recording; the store first, a legacy CSV after."""
    from .mad_labels import pred_csv_sibling_path
    from .mad_inference import read_blob_csv
    try:
        from .mad_csv_rebuild import rows_for_wav
        rows = rows_for_wav(wav)
    except Exception:
        rows = []
    if rows:
        return rows
    csv_path = pred_csv_sibling_path(wav)
    try:
        return read_blob_csv(csv_path) if Path(csv_path).is_file() else []
    except Exception:
        return []


def evaluate_labeled_regions(
    wav_paths: Sequence[str],
    cfg,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    iou_min: float = 0.3,
    context_s: float = DEFAULT_CONTEXT_S,
    progress: Optional[Callable[[int, int, str], None]] = None,
    should_stop: Optional[Callable[[], bool]] = None,
) -> EvalResult:
    """Score a model only where the user has actually judged the audio.

    Restricting to judged spans fixes a correctness problem and a cost problem
    at the same time.

    *Correctness*: on a partially reviewed recording most detections sit where
    nobody has looked. They are unjudged, not wrong — but whole-file scoring
    counts every one as a false positive, so precision reads far below what the
    model deserves, and reads worse the more unreviewed audio there is.

    *Cost*: judged spans are a few percent of a recording, and the tiled
    forward pass is the great majority of a file's runtime. Running the model
    on only those windows takes an eval over every labelled recording from
    hours to minutes, which is what makes it affordable after every training
    run rather than only when someone remembers to ask.

    Recordings flagged by :func:`~.fnt_mask_store.set_review_complete` are
    scored **whole** and totalled separately: they are the only ones whose
    recall means anything, for the reason given there.
    """
    import numpy as np
    from .fnt_mask_store import is_review_complete, masks_sibling_path
    from .mad_dataset import db_range_for
    from .mad_inference import (
        blobs_to_rows, compute_full_spec_image, extract_blobs,
        infer_probability_mask, load_model,
    )
    from .spectrogram import load_audio

    thresholds = sorted(set(round(float(t), 4) for t in thresholds))
    if not thresholds:
        thresholds = list(DEFAULT_THRESHOLDS)
    base_thr = thresholds[0]

    model, ckpt, device = load_model(cfg.model_path, cfg.device)

    def _param(name, default):
        v = getattr(cfg, name, None)
        return ckpt.get(name, default) if v is None else v

    nperseg = int(_param('nperseg', 512))
    noverlap = int(_param('noverlap', 384))
    nfft = int(_param('nfft', 1024))
    db_min = float(_param('db_min', -100.0))
    db_max = float(_param('db_max', -20.0))
    db_norm = str(_param('db_norm', 'fixed'))
    tile_f = int(ckpt.get('tile_freq_bins', cfg.tile_freq_bins))
    tile_t = int(ckpt.get('tile_time_frames', cfg.tile_time_frames))

    totals = {t: Counts() for t in thresholds}
    ex_totals = {t: Counts() for t in thresholds}
    files_out: List[Dict] = []
    n_labels_total = 0
    n_unreviewed_total = 0
    n_scored = 0
    n_ex_files = 0
    n_ex_labels = 0
    scored_s = 0.0
    total_s = 0.0

    for i, wav in enumerate(wav_paths):
        if should_stop is not None and should_stop():
            break
        name = Path(wav).name
        if progress is not None:
            progress(i, len(wav_paths), name)

        rows = _rows_for(wav)
        labels = rows_to_boxes(rows, source='truth')
        judged = reviewed_windows(rows)
        if not labels or not judged:
            files_out.append({'name': name, 'skipped': 'no confirmed calls'})
            continue
        try:
            exhaustive = is_review_complete(masks_sibling_path(wav))
        except Exception:
            exhaustive = False

        audio, sr = load_audio(wav)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        dur = len(audio) / float(sr) if sr else 0.0
        # Exhaustively reviewed means every second of it is judged.
        score_windows = None if exhaustive else judged
        infer_wins = ([(0.0, dur)] if exhaustive
                      else expand_windows(judged, context_s, dur))
        f_lo, f_hi = db_range_for(db_norm, db_min, db_max, audio=audio,
                                  sample_rate=sr, nperseg=nperseg,
                                  noverlap=noverlap, nfft=nfft)

        preds_all: List[Box] = []
        for w0, w1 in infer_wins:
            s0 = int(max(0.0, w0) * sr)
            s1 = int(min(dur, w1) * sr)
            if s1 - s0 < nperseg * 2:
                continue                      # too short to make a tile
            spec = compute_full_spec_image(
                np.asarray(audio[s0:s1], dtype=np.float32), sr,
                nperseg=nperseg, noverlap=noverlap, nfft=nfft,
                db_min=f_lo, db_max=f_hi, as_uint8=True)
            prob = infer_probability_mask(
                model, spec, tile_freq_bins=tile_f, tile_time_frames=tile_t,
                overlap_fraction=cfg.tile_overlap_fraction, device=device,
                batch_size=cfg.batch_size, use_amp=cfg.amp,
                chunk_frames=cfg.chunk_frames)
            blobs = extract_blobs(prob, threshold=base_thr,
                                  min_blob_pixels=cfg.min_blob_pixels,
                                  include_mask=True, spec=spec)
            win_rows = blobs_to_rows(blobs, nperseg=nperseg, noverlap=noverlap,
                                     nfft=nfft, sr=sr, db_min=f_lo,
                                     db_max=f_hi, spec=spec)
            # Window-local seconds back to file seconds. blobs_to_rows takes no
            # offset, and adding one would touch the write path every real run
            # uses for the sake of the eval path.
            off = s0 / float(sr)
            for b in rows_to_boxes(win_rows):
                preds_all.append(
                    Box(b.t0 + off, b.t1 + off, b.f0, b.f1, b.score))
            del spec, prob
        del audio

        n_pending = count_unreviewed(rows)
        n_labels_total += len(labels)
        n_unreviewed_total += n_pending
        n_scored += 1
        total_s += dur
        win_s = dur if exhaustive else windows_duration(judged)
        scored_s += win_s
        per_file = {'name': name, 'n_labels': len(labels),
                    'n_unreviewed': n_pending, 'exhaustive': exhaustive,
                    'scored_s': round(win_s, 2), 'total_s': round(dur, 2)}
        if exhaustive:
            n_ex_files += 1
            n_ex_labels += len(labels)
        for t in thresholds:
            preds = [p for p in preds_all if p.score >= t]
            c, _ = score_in_windows(preds, labels, score_windows,
                                    iou_min=iou_min)
            totals[t].tp += c.tp
            totals[t].fp += c.fp
            totals[t].fn += c.fn
            if exhaustive:
                ex_totals[t].tp += c.tp
                ex_totals[t].fp += c.fp
                ex_totals[t].fn += c.fn
            per_file[f"{t:g}"] = c.as_dict()
        files_out.append(per_file)

    def _threshold_rows(src):
        out = []
        for t in thresholds:
            d = {'threshold': t}
            d.update(src[t].as_dict())
            out.append(d)
        return out

    return EvalResult(
        thresholds=list(thresholds),
        per_threshold=_threshold_rows(totals),
        per_threshold_exhaustive=(_threshold_rows(ex_totals)
                                  if n_ex_files else []),
        n_labels=n_labels_total,
        n_unreviewed=n_unreviewed_total,
        n_files=n_scored,
        n_exhaustive_files=n_ex_files,
        n_exhaustive_labels=n_ex_labels,
        scored_s=scored_s,
        total_s=total_s,
        files=files_out,
        model_name=Path(cfg.model_path).stem if cfg.model_path else "",
        iou_min=iou_min,
        scope='reviewed',
    )
