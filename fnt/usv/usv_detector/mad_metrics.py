"""Storing model metrics, and reading them back as a trend.

The question this exists to answer is not "is the model good?" but *"is another
round of labelling still worth it?"* — which is a question about the
**difference** between rounds, not about any single number.

Three sources, deliberately separate because they fail in different ways:

* ``eval.json`` beside each checkpoint — call-level precision/recall/F1 from
  :mod:`.mad_eval`, scored inside judged spans. One file per trained model, so
  the trend is just reading them in run order.
* the exhaustive cohort inside that file — the only recall anyone should quote.
* :func:`review_outcome_by_model` — needs no stored state at all, and is the
  only one of the three measured on audio the model had never seen.
"""
from __future__ import annotations

import json
import os
import re
from pathlib import Path
from typing import Dict, List, Optional, Sequence

EVAL_NAME = "eval.json"

#: Run directories are named ``<timestamp>_<arch>_n=<n labels>``; the label
#: count is the x-axis of the trend, since it is the thing the user spends
#: effort on between rounds.
_RUN_RE = re.compile(r"^(?P<ts>\d{8}_\d{6})_(?P<arch>.+?)_n=(?P<n>\d+)$")


# ----------------------------------------------------------------------
# Persistence
# ----------------------------------------------------------------------
def eval_path(run_dir: str) -> str:
    return os.path.join(str(run_dir), EVAL_NAME)


def save_eval(run_dir: str, result) -> str:
    """Write a run's evaluation beside its weights.

    Its own file rather than a key in ``training_summary.json``: an eval can be
    re-run later, at a different IoU or over more recordings, and rewriting the
    training record to store it would put a mutable number inside the immutable
    account of what was trained.
    """
    os.makedirs(str(run_dir), exist_ok=True)
    p = eval_path(run_dir)
    payload = result.as_dict() if hasattr(result, 'as_dict') else dict(result)
    with open(p, 'w', encoding='utf-8') as f:
        json.dump(payload, f, indent=2)
    return p


def load_eval(run_dir: str):
    """The stored :class:`~.mad_eval.EvalResult`, or None."""
    p = eval_path(run_dir)
    if not os.path.isfile(p):
        return None
    try:
        with open(p, encoding='utf-8') as f:
            d = json.load(f)
    except Exception:
        return None
    from .mad_eval import EvalResult
    try:
        return EvalResult.from_dict(d)
    except Exception:
        return None


def parse_run_name(name: str) -> Dict:
    """``20260908_203506_unet_n=161`` -> ts / arch / n_labels. ``{}`` if it does
    not parse, so a hand-renamed directory is skipped rather than crashing."""
    m = _RUN_RE.match(os.path.basename(str(name).rstrip("/\\")))
    if not m:
        return {}
    return {'timestamp': m.group('ts'), 'arch': m.group('arch'),
            'n_labels': int(m.group('n'))}


def model_run_dirs(models_dir: str) -> List[str]:
    """Every run directory under ``models_dir``, oldest first.

    Sorted by the timestamp in the name, not mtime: re-running an eval touches
    the directory and would otherwise reorder the history under the user.
    """
    root = Path(str(models_dir))
    if not root.is_dir():
        return []
    out = []
    for p in root.iterdir():
        if p.is_dir() and parse_run_name(p.name):
            out.append(p)
    return [str(p) for p in sorted(out, key=lambda q: q.name)]


# ----------------------------------------------------------------------
# Trend
# ----------------------------------------------------------------------
def _at_best_f1(rows: Sequence[Dict]) -> Optional[Dict]:
    return max(rows, key=lambda d: d.get('f1', 0.0)) if rows else None


def eval_trend(models_dir: str) -> List[Dict]:
    """One row per evaluated model, oldest first.

    Each row is that model's best-F1 threshold row, which is the fairest
    single-number comparison: comparing two models at one fixed cutoff
    penalises whichever one happens to be calibrated differently, and the
    cutoff is a setting the user can change for free.

    Runs with no ``eval.json`` are omitted rather than shown as zeroes — never
    evaluated and scored zero are not the same claim.
    """
    out: List[Dict] = []
    for run in model_run_dirs(models_dir):
        res = load_eval(run)
        if res is None:
            continue
        meta = parse_run_name(run)
        best = _at_best_f1(res.per_threshold)
        row = {
            'run': os.path.basename(run),
            'timestamp': meta.get('timestamp', ''),
            'n_labels': meta.get('n_labels'),
            'n_files': res.n_files,
            'n_unreviewed': res.n_unreviewed,
            'scope': res.scope,
            'threshold': best.get('threshold') if best else None,
            'precision': best.get('precision') if best else None,
            'recall': best.get('recall') if best else None,
            'f1': best.get('f1') if best else None,
            'tp': best.get('tp') if best else None,
            'fp': best.get('fp') if best else None,
            'fn': best.get('fn') if best else None,
            'n_exhaustive_files': res.n_exhaustive_files,
        }
        ex = _at_best_f1(res.per_threshold_exhaustive)
        if ex:
            row['exhaustive_recall'] = ex.get('recall')
            row['exhaustive_precision'] = ex.get('precision')
            row['exhaustive_f1'] = ex.get('f1')
        out.append(row)
    return out


def trend_delta(trend: Sequence[Dict], metric: str = 'f1') -> Optional[float]:
    """Change in ``metric`` between the last two evaluated runs, or None.

    This is the actual stop signal. A high F1 does not say whether to keep
    labelling; a *flat* F1 across a round that added labels does.
    """
    vals = [r.get(metric) for r in trend if r.get(metric) is not None]
    if len(vals) < 2:
        return None
    return float(vals[-1]) - float(vals[-2])


# ----------------------------------------------------------------------
# Review outcome — the free, unbiased one
# ----------------------------------------------------------------------
def review_outcome_by_model(wav_paths: Sequence[str]) -> Dict[str, Dict]:
    """Accept/reject tallies for each model's detections, from the store.

    The one metric here measured on audio the model had never seen, and it
    needs no eval pass and no new state: reviewing already records, per
    detection, whether it was real. Of the detections a model proposed, the
    fraction accepted *is* its precision on fresh data.

    It is also the honest counterweight to :func:`eval_trend`, whose precision
    is measured partly on recordings the model trained on. Watching the
    rejection rate fall between rounds — 40% rejected down to 5% — is the
    cheapest evidence that labelling is still paying off, and the first thing
    to distrust when eval precision improves but review does not get easier.

    Pending detections are counted but kept out of the ratio: unjudged is not
    the same as wrong.
    """
    from .fnt_mask_store import masks_sibling_path
    from .mad_csv_rebuild import rows_for_wav

    out: Dict[str, Dict] = {}
    for wav in wav_paths:
        try:
            if not os.path.isfile(masks_sibling_path(wav)):
                continue
            rows = rows_for_wav(wav)
        except Exception:
            continue
        for r in rows:
            if not isinstance(r.get('blob_id'), int):
                continue                       # hand-drawn, not a proposal
            model = str(r.get('model_name') or '') or '(unknown)'
            d = out.setdefault(model, {'accepted': 0, 'rejected': 0,
                                       'pending': 0})
            status = r.get('status') or 'pending'
            if status in d:
                d[status] += 1
    for d in out.values():
        judged = d['accepted'] + d['rejected']
        d['n_judged'] = judged
        d['accept_rate'] = (d['accepted'] / judged) if judged else None
        d['reject_rate'] = (d['rejected'] / judged) if judged else None
    return out
