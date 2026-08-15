"""Call-level evaluation of a MAD checkpoint, swept across thresholds.

Training already reports validation Dice, but Dice is a **pixel** score on
**tiles**. It answers "how well does the mask overlap?", not the question that
decides whether a model is ready for a 3,000-file run: *how many real calls will
it find, and how much junk will I have to reject?*

So this evaluates at the level the user actually works at — one call, accepted or
rejected — by matching predicted blobs against hand-labeled calls and reporting
precision / recall / F1. Because every prediction carries a probability score,
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


def rows_to_boxes(rows: Sequence[Dict], source: Optional[str] = None
                  ) -> List[Box]:
    """Convert CSV rows to boxes, optionally filtered to predictions or labels.

    ``source='prediction'`` selects model output (int blob_id), ``'label'``
    selects hand-labels (string blob_id). Rejected rows are dropped from labels —
    a rejection is a recorded "not a call", so it is not ground truth.
    """
    out: List[Box] = []
    for r in rows:
        is_pred = isinstance(r.get('blob_id'), int)
        if source == 'prediction' and not is_pred:
            continue
        if source == 'label':
            if is_pred:
                continue
            if (r.get('status') or '') == 'rejected':
                continue
        try:
            out.append(Box(
                t0=float(r.get('start_s', 0.0)),
                t1=float(r.get('stop_s', 0.0)),
                f0=float(r.get('min_freq_hz', 0.0)),
                f1=float(r.get('max_freq_hz', 0.0)),
                score=float(r.get('score', 1.0) or 1.0),
            ))
        except (TypeError, ValueError):
            continue
    return out


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

    Ground truth is each recording's hand-labels, read from its sibling CSV. Files
    with no hand-labels are skipped and reported, since scoring against an empty
    ground truth would make precision look catastrophic for no reason.
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
    tile_f = int(ckpt.get('tile_freq_bins', cfg.tile_freq_bins))
    tile_t = int(ckpt.get('tile_time_frames', cfg.tile_time_frames))

    totals = {t: Counts() for t in thresholds}
    files_out: List[Dict] = []
    n_labels_total = 0
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
        labels = rows_to_boxes(rows, source='label')
        if not labels:
            files_out.append({'name': name, 'skipped': 'no hand-labels'})
            continue

        audio, sr = load_audio(wav)
        if audio.ndim > 1:
            audio = audio.mean(axis=1)
        spec = compute_full_spec_image(
            audio.astype(np.float32), sr, nperseg=nperseg, noverlap=noverlap,
            nfft=nfft, db_min=db_min, db_max=db_max)
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
        n_scored += 1
        per_file = {'name': name, 'n_labels': len(labels)}
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
        n_files=n_scored,
        files=files_out,
        model_name=Path(cfg.model_path).stem if cfg.model_path else "",
        iou_min=iou_min,
    )
