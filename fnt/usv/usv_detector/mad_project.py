"""MAD (Mask Audio Detector) project configuration and directory layout.

A MAD project is a directory that *references* recordings where they already
live (SLEAP's model: the project points at videos, it does not ingest them).
Pixel-level labels and training examples live in per-wav ``_FNT_masks.h5``
siblings and a consolidated ``training_data.h5`` store under the project's
``models/training_data/``.
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
            mad_project_info.json
            datasets/           # exported tiles + mask tiles (regenerated each train)
            models/             # per-run segmentation model checkpoints

    Sibling label format (lives next to each .wav, NOT inside the project):
      * ``<base>_FNT_MAD_labels.png`` — 8-bit PNG, spectrogram-pixel grid.
        Values: 0 = unlabeled, 1 = painted positive, 2 = certified negative.
      * ``<base>_FNT_MAD_labels.json`` — sidecar with committed-column ranges,
        spectrogram params hash, and paint-tool metadata.
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

    # Spectrogram parameters — must match between label, train, and inference.
    nperseg: int = 512
    noverlap: int = 384
    nfft: int = 1024
    db_min: float = -100.0
    db_max: float = -20.0
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
    @property
    def training_data_dir(self) -> str:
        """Self-contained per-call example store, shared across model runs."""
        return os.path.join(self.project_dir, 'models', 'training_data')

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
    os.makedirs(os.path.join(project_dir, 'models', 'training_data'), exist_ok=True)
    os.makedirs(os.path.join(project_dir, 'datasets'), exist_ok=True)
    os.makedirs(os.path.join(project_dir, 'recordings'), exist_ok=True)

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
