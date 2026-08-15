"""Resumable batch inference runs for MAD.

A 3,000-file run is hours of compute. Without a record of what finished, a crash,
a reboot, or a deliberate stop at file 2,800 throws all of it away — so batch
runs write a **manifest** as they go and can skip work that is already done.

Two independent mechanisms, because they answer different questions:

* **The manifest** (``batch_runs/<timestamp>/manifest.jsonl``) is the run's own
  append-only log — one JSON line per file, flushed immediately, so an
  interrupted run is readable exactly up to the last completed file. It is also
  what the run-summary view reads.

* **Provenance skipping** asks a different question: "does this recording already
  carry detections from *this* model at *these* settings?" That answer lives in
  each recording's own CSV (``model_name``, ``threshold``, ``min_blob_pixels``
  columns, already written by ``mad_inference``), so it survives losing the
  manifest entirely, and it works across runs — pointing a second run at an
  overlapping folder re-does nothing.

Both are deliberately cheap: skipping reads only the CSV header rows, never the
h5 mask crops or the audio.
"""
from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Dict, Iterable, List, Optional

from .mad_labels import pred_csv_sibling_path


MANIFEST_NAME = "manifest.jsonl"
RUN_INFO_NAME = "run_info.json"


def batch_runs_dir(project_dir: str) -> str:
    return os.path.join(project_dir, "batch_runs")


def new_run_dir(project_dir: str, label: str = "") -> str:
    """Create and return a fresh timestamped run directory."""
    stamp = time.strftime("%Y%m%d_%H%M%S")
    name = f"{stamp}_{label}" if label else stamp
    d = os.path.join(batch_runs_dir(project_dir), name)
    os.makedirs(d, exist_ok=True)
    return d


def list_runs(project_dir: str) -> List[str]:
    """Existing run directories, newest first."""
    root = batch_runs_dir(project_dir)
    if not os.path.isdir(root):
        return []
    dirs = [os.path.join(root, d) for d in os.listdir(root)
            if os.path.isdir(os.path.join(root, d))]
    return sorted(dirs, reverse=True)


# ----------------------------------------------------------------------
# Provenance — "is this file already done at these settings?"
# ----------------------------------------------------------------------
@dataclass(frozen=True)
class RunSettings:
    """The inference settings that determine whether a result is reusable.

    Only settings that change the *detections* belong here. batch_size, amp and
    device deliberately do not: they alter speed, not output, so a run should
    never redo work just because it is using a bigger batch this time.
    """
    model_name: str
    threshold: float
    min_blob_pixels: int

    @classmethod
    def from_config(cls, cfg) -> 'RunSettings':
        return cls(
            model_name=Path(cfg.model_path).stem if cfg.model_path else "",
            threshold=round(float(cfg.threshold), 6),
            min_blob_pixels=int(cfg.min_blob_pixels),
        )

    def matches_row(self, row: Dict) -> bool:
        try:
            return (str(row.get('model_name', '')) == self.model_name
                    and abs(float(row.get('threshold', -1))
                            - self.threshold) < 1e-6
                    and int(float(row.get('min_blob_pixels', -1)))
                    == self.min_blob_pixels)
        except (TypeError, ValueError):
            return False


def file_already_done(wav_path: str, settings: RunSettings) -> bool:
    """True when ``wav_path``'s sibling CSV already holds predictions from this
    model at these settings.

    A file with *no* prediction rows is treated as done when the CSV exists and
    was written by this model — a recording containing zero calls is a legitimate
    result, and re-running it every time would defeat resuming on exactly the
    quiet files that dominate 24/7 recordings. Presence of the provenance is
    established from any prediction row; the empty case falls back to the run
    manifest, which records zero-detection files explicitly.
    """
    csv_path = pred_csv_sibling_path(wav_path)
    if not os.path.isfile(csv_path):
        return False
    try:
        from .mad_inference import read_blob_csv
        rows = read_blob_csv(csv_path)
    except Exception:
        return False
    preds = [r for r in rows if isinstance(r.get('blob_id'), int)]
    if not preds:
        return False
    return all(settings.matches_row(r) for r in preds)


def partition_done(wav_paths: Iterable[str], settings: RunSettings,
                   manifest_done: Optional[set] = None,
                   progress: Optional[Callable[[int, int], None]] = None):
    """Split paths into ``(todo, done)`` for a resumed run.

    ``manifest_done`` is the set of absolute paths a previous manifest recorded
    as complete; it covers the zero-detection files that provenance alone cannot
    confirm.
    """
    paths = list(wav_paths)
    todo: List[str] = []
    done: List[str] = []
    md = manifest_done or set()
    n = len(paths)
    for i, p in enumerate(paths):
        if progress is not None and (i % 50 == 0 or i == n - 1):
            progress(i + 1, n)
        if os.path.normcase(os.path.abspath(p)) in md or \
                file_already_done(p, settings):
            done.append(p)
        else:
            todo.append(p)
    return todo, done


# ----------------------------------------------------------------------
# Manifest
# ----------------------------------------------------------------------
class RunManifest:
    """Append-only JSONL log of a batch run.

    Every record is flushed as it is written, so a killed process leaves a
    manifest that is valid up to its last completed file — the whole point.
    """

    def __init__(self, run_dir: str):
        self.run_dir = run_dir
        os.makedirs(run_dir, exist_ok=True)
        self.path = os.path.join(run_dir, MANIFEST_NAME)
        self._fh = None

    # -- writing --------------------------------------------------------
    def open(self):
        if self._fh is None:
            self._fh = open(self.path, 'a', encoding='utf-8')
        return self

    def close(self):
        if self._fh is not None:
            try:
                self._fh.close()
            finally:
                self._fh = None

    def __enter__(self):
        return self.open()

    def __exit__(self, *exc):
        self.close()
        return False

    def write_info(self, info: Dict) -> None:
        with open(os.path.join(self.run_dir, RUN_INFO_NAME), 'w',
                  encoding='utf-8') as f:
            json.dump(info, f, indent=2)

    def read_info(self) -> Dict:
        p = os.path.join(self.run_dir, RUN_INFO_NAME)
        if not os.path.isfile(p):
            return {}
        try:
            with open(p, encoding='utf-8') as f:
                return json.load(f)
        except Exception:
            return {}

    def record(self, summary: Dict, status: str = "") -> None:
        """Append one file's result. ``summary`` is a run_inference_on_file dict."""
        self.open()
        wav = summary.get('wav_path', '')
        timing = summary.get('timing', {}) or {}
        rec = {
            'wav_path': os.path.abspath(wav) if wav else '',
            'name': os.path.basename(wav),
            'status': status or ('error' if 'error' in summary else 'ok'),
            'n_detections': int(summary.get('n_blobs', 0) or 0),
            'csv_path': summary.get('csv_path'),
            'audio_dur_s': timing.get('audio_dur_s'),
            't_total': timing.get('t_total'),
            'realtime_factor': timing.get('realtime_factor'),
            'device': timing.get('device'),
            'error': summary.get('error'),
            'ts': time.time(),
        }
        assert self._fh is not None
        self._fh.write(json.dumps(rec) + "\n")
        self._fh.flush()

    def record_skipped(self, wav_path: str) -> None:
        self.open()
        assert self._fh is not None
        self._fh.write(json.dumps({
            'wav_path': os.path.abspath(wav_path),
            'name': os.path.basename(wav_path),
            'status': 'skipped',
            'ts': time.time(),
        }) + "\n")
        self._fh.flush()

    # -- reading --------------------------------------------------------
    def records(self) -> List[Dict]:
        """Every record in the manifest. Truncated trailing lines are ignored,
        which is what makes a manifest from a killed process still usable."""
        if not os.path.isfile(self.path):
            return []
        out: List[Dict] = []
        with open(self.path, encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    out.append(json.loads(line))
                except json.JSONDecodeError:
                    continue          # partial final write — skip it
        return out

    def completed_paths(self) -> set:
        """Absolute paths this run already finished (ok or deliberately empty)."""
        return {r['wav_path'] for r in self.records()
                if r.get('status') in ('ok', 'skipped') and r.get('wav_path')}


def summarize(records: List[Dict]) -> Dict:
    """Roll a manifest up into run-level totals for the summary view."""
    ok = [r for r in records if r.get('status') == 'ok']
    errs = [r for r in records if r.get('status') == 'error']
    skipped = [r for r in records if r.get('status') == 'skipped']
    dets = sum(int(r.get('n_detections') or 0) for r in ok)
    audio = sum(float(r.get('audio_dur_s') or 0.0) for r in ok)
    wall = sum(float(r.get('t_total') or 0.0) for r in ok)
    with_calls = sum(1 for r in ok if (r.get('n_detections') or 0) > 0)
    return {
        'n_files': len(ok) + len(errs) + len(skipped),
        'n_ok': len(ok),
        'n_error': len(errs),
        'n_skipped': len(skipped),
        'n_detections': dets,
        'n_files_with_calls': with_calls,
        'audio_hours': audio / 3600.0,
        'wall_hours': wall / 3600.0,
        'realtime_factor': (audio / wall) if wall > 0 else 0.0,
    }
