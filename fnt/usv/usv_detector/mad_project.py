"""MAD (Mask Audio Detector) project configuration and directory layout.

A MAD project is a directory that *references* recordings where they already
live (SLEAP's model: the project points at videos, it does not ingest them).
Pixel-level labels live in a ``_FNT.mad`` sidecar beside each recording --
that is the master copy. ``training_data/`` holds a consolidated
``training_data.h5`` rebuilt from those sidecars at the start of every run;
it is a cache, and it sits beside ``models/`` rather than inside it because
it is an input to models rather than one of them.
"""
from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, field
from typing import Dict, List, Optional


PROJECT_INFO_FILENAME = 'mad_project_info.json'


@dataclass
class MADProjectConfig:
    """Configuration for a MAD (Mask Audio Detector) project.

    Layout on disk::

        <project_dir>/
            mad_project_info.json     # this config: audio registry + params
            training_data/
                training_data.h5      # cache rebuilt from the .mad sidecars
            models/
                <run>/weights.pt      # per-run checkpoint + its provenance
            batch_runs/<run>/         # inference run logs
            recordings/               # ONLY when Pack Project embeds audio

    Labels live next to each recording, not inside the project, in a
    ``<base>_FNT.mad`` HDF5 sidecar (see ``fnt_mask_store``):

      * ``/examples/<id>`` — confirmed calls and hard negatives, each a
        self-contained spec patch + mask with its own metadata.
      * ``/pred_calls/<id>`` — prediction crops awaiting review.
      * root attrs — the spectrogram grid every mask was computed on, plus
        provenance (FNT version, created/updated, last inference run).

    That is what makes a recording portable: hand someone the .wav and its
    .mad and they have the labels, with no project at all.
    """
    project_dir: str = ""
    project_name: str = ""
    source_folders: List[str] = field(default_factory=list)
    last_opened_file: Optional[str] = None

    # The project's audio — every recording it knows about, referenced by path
    # rather than copied into recordings/ (SLEAP-style: a project points at
    # videos where they live). Serialized RegisteredFile dicts; see
    # fnt.usv.usv_detector.mad_registry for why referencing is safe here
    # (training reads training_data.h5, never the wav) and how missing files
    # are re-resolved. Legacy projects with recordings/ copies are adopted into
    # this list on open, with embedded=True so they stay project-owned.
    #
    # There is one list, not two: anything in the project is labelable and
    # trains the model. Batch inference over recordings you are NOT curating
    # runs through the Run Inference "Folder" target, which never touches this.
    audio_files: List[Dict] = field(default_factory=list)

    # Every sampling draw that fed this project, appended in order.
    #
    # A subset drawn from a 34,000-file tree is a methods-section fact: "20 per
    # trial, evenly spaced, spread across four microphones" is the difference
    # between a training set someone can reproduce and one they cannot. It is
    # also what lets a later pass ask for twenty MORE — each draw excludes the
    # files already imported, so the history is the record of what was taken.
    # Entries are SampleSpec dicts plus root/n_added/at; see mad_sampling.
    sample_history: List[Dict] = field(default_factory=list)

    # Spectrogram parameters — must match between label, train, and inference.
    nperseg: int = 512
    noverlap: int = 384
    nfft: int = 1024
    db_min: float = -100.0
    db_max: float = -20.0
    # How the spectrogram is normalized before the model sees it.
    #   'fixed'    — db_min…db_max for every recording (default; what every
    #                model trained before this option used).
    #   'per_file' — each recording's own percentile range, so a call looks the
    #                same to the model regardless of that file's gain and noise
    #                floor. Better cross-trial generalization, but examples
    #                labeled under one setting must not be mixed with the other
    #                (the patch is normalized and quantized when it is saved).
    db_norm: str = 'fixed'
    colormap: str = 'viridis'

    # Model architecture — user-selectable per training run.
    #   'unet'     : segmentation_models_pytorch U-Net (default)
    #   'yolo_seg' : ultralytics YOLOv11-seg (polygonized from raster masks)
    model_arch: str = 'unet'

    # Training parameters (shared across archs where sensible).
    tile_time_window_s: float = 0.5
    tile_overlap_fraction: float = 0.25
    val_fraction: float = 0.20

    # Inference.
    mask_threshold: float = 0.5

    # Call-type classes the user has confirmed (metadata on each saved
    # training example; the segmentation model itself stays binary). The
    # class dialog defaults to ``last_class`` so repeat-Enter reuses it.
    classes: List[str] = field(default_factory=lambda: ["USV"])
    last_class: str = "USV"

    # Model history: list of {name, arch, n_positive_pixels, n_negative_pixels, path, date}.
    models: List[Dict] = field(default_factory=list)

    schema_version: int = 1

    # ------------------------------------------------------------------
    #: Where the training cache lived before 2026-09-08. Read forever so an
    #: existing project keeps working; migrated on open by
    #: :func:`migrate_training_data_dir`.
    LEGACY_TRAINING_DATA_DIR = ('models', 'training_data')

    @property
    def training_data_dir(self) -> str:
        """Per-call example cache, rebuilt from the ``.mad`` sidecars each run.

        Sits beside ``models/`` rather than inside it because it is an *input*
        to models, not one of them: each ``models/<run>/`` is one frozen
        training run, while this corpus accumulates across all of them. Filing
        it under ``models/`` read as "a model called training_data".

        A project created before the move keeps its old location — see
        :func:`migrate_training_data_dir`.
        """
        new = os.path.join(self.project_dir, 'training_data')
        if os.path.isdir(new):
            return new
        old = os.path.join(self.project_dir, *self.LEGACY_TRAINING_DATA_DIR)
        return old if os.path.isdir(old) else new

    def migrate_training_data_dir(self) -> Optional[str]:
        """Move ``models/training_data/`` up to ``training_data/``.

        Safe to call on every open: it does nothing when the new location
        already exists, and nothing when the old one does not. Returns the new
        path if it moved, else None.

        The contents are a cache rebuilt from the ``.mad`` sidecars at the
        start of every run, so even a failed move costs nothing but a rebuild —
        which is why this can just log and carry on rather than block opening a
        project.
        """
        import shutil
        if not self.project_dir:
            return None
        new = os.path.join(self.project_dir, 'training_data')
        old = os.path.join(self.project_dir, *self.LEGACY_TRAINING_DATA_DIR)
        if os.path.isdir(new) or not os.path.isdir(old):
            return None
        try:
            shutil.move(old, new)
        except Exception:
            return None
        return new

    @property
    def recordings_dir(self) -> str:
        """Project-owned audio: legacy copies, and anything embedded by
        "Pack project". Added files are referenced in place instead — see
        ``audio_files``."""
        return os.path.join(self.project_dir, 'recordings')

    # ------------------------------------------------------------------
    # Audio-file registry
    # ------------------------------------------------------------------
    def audio_entries(self):
        """The project's audio as :class:`RegisteredFile` objects."""
        from .mad_registry import entries_from_dicts
        return entries_from_dicts(self.audio_files)

    def set_audio_entries(self, entries) -> None:
        from .mad_registry import entries_to_dicts
        self.audio_files = entries_to_dicts(entries)

    def save(self, path: Optional[str] = None) -> None:
        """Save config to ``<project_dir>/mad_project_info.json``."""
        if path is None:
            path = os.path.join(self.project_dir, PROJECT_INFO_FILENAME)
        if self.project_dir and not self.project_name:
            self.project_name = os.path.basename(os.path.normpath(self.project_dir))
        data = asdict(self)
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)

    @classmethod
    def load(cls, path: str) -> 'MADProjectConfig':
        """Load config; ``path`` may be the JSON file or the project dir."""
        if os.path.isdir(path):
            path = os.path.join(path, PROJECT_INFO_FILENAME)
        with open(path) as f:
            data = json.load(f)
        data = _migrate_audio_files(data)
        known = {f.name for f in cls.__dataclass_fields__.values()}  # type: ignore[attr-defined]
        data = {k: v for k, v in data.items() if k in known}
        cfg = cls(**data)
        if not cfg.project_dir:
            cfg.project_dir = os.path.dirname(os.path.abspath(path))
        if not cfg.project_name and cfg.project_dir:
            cfg.project_name = os.path.basename(os.path.normpath(cfg.project_dir))
        return cfg


def _migrate_audio_files(data: Dict) -> Dict:
    """Fold a pre-merge project's two file lists into the single ``audio_files``
    registry, in place on the raw JSON dict.

    Older projects kept ``training_files`` (RegisteredFile dicts — the curated
    training set) separate from ``audio_files`` (plain path strings — the
    working session list). Both are the same thing now, so registered entries
    come first and any session path not already registered is appended as a
    plain reference. Nothing is dropped, and re-loading a migrated project is a
    no-op because ``audio_files`` is already a list of dicts.
    """
    registered = data.get('training_files') or []
    session = data.get('audio_files') or []
    if not registered and all(isinstance(e, dict) for e in session):
        return data  # already merged (or an empty project)
    entries = [e for e in registered if isinstance(e, dict)]
    entries.extend(e for e in session if isinstance(e, dict))
    known = {os.path.normcase(os.path.abspath(str(e.get('path', ''))))
             for e in entries}
    for p in session:
        if not isinstance(p, str) or not p:
            continue
        ap = os.path.abspath(p)
        if os.path.normcase(ap) in known:
            continue
        known.add(os.path.normcase(ap))
        entries.append({'path': ap, 'basename': os.path.basename(ap),
                        'embedded': False})
    data['audio_files'] = entries
    data.pop('training_files', None)
    return data


def create_mad_project(
    project_dir: str,
    config: Optional[MADProjectConfig] = None,
    source_folders: Optional[List[str]] = None,
) -> MADProjectConfig:
    """Create a new MAD project directory and write its config."""
    os.makedirs(project_dir, exist_ok=True)
    os.makedirs(os.path.join(project_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(project_dir, 'training_data'), exist_ok=True)
    # No 'datasets/' — it held exported tile/mask files for a training path
    # that no longer exists (training reads training_data/*.h5), so it
    # was created empty in every project and never written to.
    #
    # No 'recordings/' either: only Pack Project puts anything there, and it
    # creates the folder itself. An empty one in every project just invites
    # "what is this for?".

    if config is None:
        config = MADProjectConfig()
    config.project_dir = project_dir
    config.project_name = os.path.basename(os.path.normpath(project_dir))
    if source_folders:
        for folder in source_folders:
            if folder and folder not in config.source_folders:
                config.source_folders.append(folder)
    config.save()
    return config
