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

Both are scoped by :class:`RunSettings`. A manifest is only trusted when its
``run_info`` attests to the *same* settings — otherwise retraining a model and
re-running the same folder would find every file marked done by the old model's
run and quietly analyze nothing. Use :func:`completed_by_settings` to gather
prior completions; both the GUI and the CLI go through it so they cannot drift.

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

    Two sources can attest that a file is done, and they can verify different
    amounts:

    * a recording's own CSV carries ``model_name`` / ``threshold`` /
      ``min_blob_pixels`` as columns, so :meth:`matches_row` can check only
      those three;
    * a run manifest's ``run_info`` records the whole settings block, so
      :meth:`matches_info` also checks the merge options, which change how
      blobs are combined into calls and therefore change the output.

    The manifest check is the stricter one. That asymmetry is deliberate: the
    failure it prevents (skipping a file that was never analyzed at these
    settings) is far worse than the one it risks (redoing work).
    """
    model_name: str
    threshold: float
    min_blob_pixels: int
    merge_consecutive: bool = False
    merge_max_gap_s: float = 0.01
    merge_require_freq_overlap: bool = True

    @classmethod
    def from_config(cls, cfg) -> 'RunSettings':
        return cls(
            model_name=Path(cfg.model_path).stem if cfg.model_path else "",
            threshold=round(float(cfg.threshold), 6),
            min_blob_pixels=int(cfg.min_blob_pixels),
            merge_consecutive=bool(getattr(cfg, 'merge_consecutive', False)),
            merge_max_gap_s=round(
                float(getattr(cfg, 'merge_max_gap_s', 0.01)), 6),
            merge_require_freq_overlap=bool(
                getattr(cfg, 'merge_require_freq_overlap', True)),
        )

    def to_info(self) -> Dict:
        """The settings block to record in a run's ``run_info.json``, so a later
        run can tell whether that run's results are reusable."""
        return {
            'model_name': self.model_name,
            'threshold': self.threshold,
            'min_blob_pixels': self.min_blob_pixels,
            'merge_consecutive': self.merge_consecutive,
            'merge_max_gap_s': self.merge_max_gap_s,
            'merge_require_freq_overlap': self.merge_require_freq_overlap,
        }

    def matches_row(self, row: Dict) -> bool:
        """Whether a CSV detection row was produced by these settings (the three
        provenance columns the CSV actually carries)."""
        try:
            return (str(row.get('model_name', '')) == self.model_name
                    and abs(float(row.get('threshold', -1))
                            - self.threshold) < 1e-6
                    and int(float(row.get('min_blob_pixels', -1)))
                    == self.min_blob_pixels)
        except (TypeError, ValueError):
            return False

    def matches_info(self, info: Optional[Dict]) -> bool:
        """Whether a run's recorded ``run_info`` was produced by these settings.

        A run whose info is missing, unreadable, or predates a field is treated
        as NOT matching. Its files then get re-analyzed, which costs time; the
        alternative — trusting an unverifiable claim — silently skips work that
        may never have happened.
        """
        if not info:
            return False
        try:
            if str(info.get('model_name', '')) != self.model_name:
                return False
            if abs(float(info.get('threshold', -1)) - self.threshold) > 1e-6:
                return False
            if int(float(info.get('min_blob_pixels', -1))) != self.min_blob_pixels:
                return False
            for key, mine in (
                ('merge_consecutive', self.merge_consecutive),
                ('merge_require_freq_overlap', self.merge_require_freq_overlap),
            ):
                if key not in info or bool(info.get(key)) != bool(mine):
                    return False
            if 'merge_max_gap_s' not in info:
                return False
            if abs(float(info.get('merge_max_gap_s', -1))
                    - self.merge_max_gap_s) > 1e-6:
                return False
        except (TypeError, ValueError):
            return False
        return True


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
    # Streamed, and only the four provenance fields are touched. read_blob_csv
    # would parse and float-convert ~40 columns of every detection on the file;
    # at 3,000 previously-analyzed recordings that is minutes of work to answer
    # a yes/no question. Bails at the first mismatching row.
    import csv as _csv
    n_pred = 0
    try:
        with open(csv_path, newline='') as f:
            for row in _csv.DictReader(f):
                raw = row.get('call_id', row.get('blob_id'))
                if raw is None:
                    continue
                try:                       # int call_id == model prediction
                    int(str(raw).strip())
                except (TypeError, ValueError):
                    continue               # string id == hand-label, not ours
                n_pred += 1
                if not settings.matches_row(row):
                    return False
    except (OSError, _csv.Error):
        return False
    return n_pred > 0


def completed_by_settings(roots, settings: RunSettings) -> set:
    """Normcased absolute paths that earlier runs finished **at these settings**.

    This is what makes resume safe across a retrain. Unioning every prior
    manifest regardless of settings would mean: analyze a folder with model A,
    retrain to model B, re-run the same folder, and every file counts as already
    done — model B silently runs on nothing while the run reports success.

    Runs whose ``run_info`` does not attest to these exact settings are ignored
    (see :meth:`RunSettings.matches_info`), so an unverifiable manifest can only
    ever cost a redo, never a wrongly skipped file.
    """
    if isinstance(roots, (str, bytes)):
        roots = [roots]
    out: set = set()
    seen_dirs: set = set()
    for root in roots:
        if not root or not os.path.isdir(root):
            continue
        for run_dir in list_runs(root):
            key = os.path.normcase(os.path.abspath(run_dir))
            if key in seen_dirs:
                continue
            seen_dirs.add(key)
            m = RunManifest(run_dir)
            if not settings.matches_info(m.read_info()):
                continue
            for path in m.completed_paths():
                out.add(os.path.normcase(os.path.abspath(path)))
    return out


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
