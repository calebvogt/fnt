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

    def best(self, metric: str = 'f1') -> Optional[Dict]:
        if not self.per_threshold:
            return None
        return max(self.per_threshold, key=lambda d: d.get(metric, 0.0))

    def as_dict(self) -> Dict:
        return {
            'model_name': self.model_name,
            'iou_min': self.iou_min,
            'n_files': self.n_files,
            'n_labels': self.n_labels,
            'n_unreviewed': self.n_unreviewed,
            'per_threshold': self.per_threshold,
            'files': self.files,
        }


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
