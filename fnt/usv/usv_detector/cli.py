"""Command-line interface for the Mask Audio Detector (MAD).

Exposes MAD's train / analyze / embeddings pipeline as a headless CLI so it can
be scripted into batch and HPC workflows without the GUI. Installed as the
``mad`` console script (see ``pyproject.toml``)::

    mad analyze    --model weights.pt --input recordings/
    mad train      --project my_project/ --split-mode file
    mad embeddings --model weights.pt --input recordings/ --out emb.npz

Heavy deps (torch, segmentation-models-pytorch) are imported lazily by the
underlying pipeline, so ``mad --help`` stays fast.
"""
from __future__ import annotations

import argparse
import glob
import os
import sys
from typing import List, Optional


_SKIP_DIRS = {'models', 'datasets', 'batch_runs', '.scratch', 'legacy_pre_h5'}


def _walk_wavs(root: str) -> List[str]:
    """Every ``.wav`` under ``root``, skipping dot-dirs and project internals."""
    out: List[str] = []
    for dirpath, dirnames, filenames in os.walk(root):
        dirnames[:] = sorted(d for d in dirnames
                             if not d.startswith('.')
                             and d.lower() not in _SKIP_DIRS)
        for fn in sorted(filenames):
            if fn.lower().endswith('.wav') and not fn.startswith('.'):
                out.append(os.path.join(dirpath, fn))
    return out


def _expand_inputs(inputs: List[str], recursive: bool = True) -> List[str]:
    """Expand a mix of .wav files and folders into a sorted, de-duped wav list.

    Folders are walked recursively by default — 24/7 multi-mic sets are nested
    (experiment / mic / day), so a flat scan finds nothing at the level a user
    naturally points at. Pass ``recursive=False`` for the old flat behavior.
    """
    out: List[str] = []
    seen = set()
    for item in inputs:
        if os.path.isdir(item):
            if recursive:
                paths = _walk_wavs(item)
            else:
                paths = sorted(glob.glob(os.path.join(item, '*.wav'))
                               + glob.glob(os.path.join(item, '*.WAV')))
        elif os.path.isfile(item):
            paths = [item]
        else:
            paths = sorted(glob.glob(item))  # allow shell-style globs
        for p in paths:
            ap = os.path.abspath(p)
            if ap not in seen:
                seen.add(ap)
                out.append(p)
    return out


def _add_sampling_args(p: argparse.ArgumentParser) -> None:
    """Sampling flags, identical in meaning to the GUI's import dialog.

    Analyzing a whole 24/7 multi-microphone set is not a thing anyone can
    afford: 33,996 ten-minute recordings at roughly 3x realtime is weeks of
    GPU. A subset that tiles every trial, microphone and time of day is an
    overnight run, and it is what the GUI offers on import — so the CLI has to
    offer it too, or a headless run silently analyzes something different from
    what the user set up interactively.
    """
    g = p.add_argument_group('sampling')
    g.add_argument('--sample-per-folder', type=int, default=0, metavar='N',
                   help="Analyze only N recordings from each folder, chosen to "
                        "span the whole series. With --sample-channel-mode "
                        "spread (the default) N is the folder's total budget, "
                        "split across its microphones — not N per microphone.")
    g.add_argument('--sample-total', type=int, default=0, metavar='N',
                   help="Analyze N recordings across the whole input, "
                        "apportioned by folder size.")
    g.add_argument('--sample-spacing', default='stride',
                   choices=['stride', 'random'],
                   help="'stride' spreads picks evenly over each group "
                        "(default, reproducible without a seed); 'random' "
                        "draws uniformly.")
    g.add_argument('--sample-seed', type=int, default=12345,
                   help="Seed for --sample-spacing random (default 12345).")
    g.add_argument('--sample-channel-mode', default='spread',
                   choices=['spread', 'only', 'pool'],
                   help="'spread' splits each folder's budget across its "
                        "microphones; 'only' keeps just --sample-channels; "
                        "'pool' ignores channels (right for single-mic sets, "
                        "wrong for multi-mic ones).")
    g.add_argument('--sample-channels', nargs='+', default=(), metavar='CH',
                   help="Channels to keep with --sample-channel-mode only, "
                        "e.g. ch1 ch3.")


def _apply_sampling(wavs, args) -> list:
    """Narrow ``wavs`` per the --sample-* flags, reporting what was drawn.

    Printed rather than silent because the count is the whole point: a run that
    was meant to sample 20 per folder and actually took 33,996 files is a
    three-week mistake that should be visible in the first line of the log.
    """
    per_folder = getattr(args, 'sample_per_folder', 0) or 0
    total = getattr(args, 'sample_total', 0) or 0
    if per_folder and total:
        raise SystemExit(
            "Use --sample-per-folder or --sample-total, not both.")
    if not per_folder and not total:
        return wavs

    from .mad_sampling import SampleSpec, sample_paths
    spec = SampleSpec(
        per='folder' if per_folder else 'total',
        n=per_folder or total,
        spacing=args.sample_spacing,
        seed=(args.sample_seed if args.sample_spacing == 'random' else None),
        channel_mode=args.sample_channel_mode,
        channels=tuple(args.sample_channels or ()))
    res = sample_paths(wavs, spec)
    print(f"Sampling {len(res)} of {len(wavs)} file(s) — {spec.describe()}",
          flush=True)
    for row in res.rows:
        if row['picked']:
            label = os.path.basename(row['folder']) or row['folder']
            if row['channel']:
                label += f" {row['channel']}"
            print(f"    {label}: {row['picked']} of {row['available']}",
                  flush=True)
    return list(res.paths)


# ----------------------------------------------------------------------
# analyze
# ----------------------------------------------------------------------
def _group_harmonics_for(wav_path: str) -> None:
    """Group one recording's detections into calls, as the GUI does after
    each file of a run. A failure is reported and never fails the file: the
    detections are already written and correct."""
    from .fnt_mask_store import masks_sibling_path
    from .mad_harmonics import group_recording
    try:
        store = masks_sibling_path(wav_path)
        if os.path.isfile(store):
            group_recording(store, os.path.basename(wav_path))
    except Exception as e:
        print(f"  harmonic grouping failed for "
              f"{os.path.basename(wav_path)}: {e}", file=sys.stderr, flush=True)


def _cmd_analyze(args: argparse.Namespace) -> int:
    from .mad_inference import MADInferenceConfig, run_inference_on_files

    if os.path.isdir(args.model):
        args.model = os.path.join(args.model, 'weights.pt')
    wavs = _expand_inputs(args.input) if args.input else []
    if args.project:
        from .mad_project import MADProjectConfig
        try:
            proj = MADProjectConfig.load(args.project)
        except Exception as e:
            print(f"Could not open project {args.project}: {e}",
                  file=sys.stderr, flush=True)
            return 2
        seen = {os.path.normcase(os.path.abspath(w)) for w in wavs}
        missing = 0
        for e in proj.audio_entries():
            key = os.path.normcase(os.path.abspath(e.path))
            if key in seen:
                continue
            if not e.exists():
                missing += 1      # a moved recording is soft; say so, go on
                continue
            seen.add(key)
            wavs.append(e.path)
        if missing:
            print(f"  {missing} registered recording(s) not found on disk "
                  "— skipped.", file=sys.stderr, flush=True)
        # What a GUI run on this project would use, unless told otherwise.
        if not args.log_root and not args.run_dir:
            args.log_root = proj.project_dir
        if not args.training_data_dir:
            args.training_data_dir = proj.training_data_dir
    if not wavs:
        print("No .wav files found in the given input(s).", file=sys.stderr, flush=True)
        return 2
    wavs = _apply_sampling(wavs, args)
    print(f"Analyzing {len(wavs)} file(s) with {os.path.basename(args.model)}", flush=True)

    cfg = MADInferenceConfig(
        model_path=args.model,
        threshold=args.threshold,
        min_blob_pixels=args.min_blob_pixels,
        device=args.device,
        preserve_labels=not args.no_preserve_labels,
        training_data_dir=args.training_data_dir or "",
        merge_consecutive=args.merge_consecutive,
        merge_max_gap_s=args.merge_gap_s,
        merge_require_freq_overlap=not args.merge_ignore_freq,
        batch_size=args.batch_size,
        amp=not args.no_amp,
        prefetch_next=not args.no_prefetch,
        gpu_spectrogram=not args.cpu_spectrogram,
    )

    from .mad_batch import (
        RunManifest, RunSettings, completed_by_settings, new_run_dir,
        partition_done)

    settings = RunSettings.from_config(cfg)
    # Runs log under --log-root (beside the model by default), and resume reads
    # that same place back, so repeated invocations against one model
    # accumulate a history they can actually use.
    log_root = args.log_root or os.path.dirname(os.path.abspath(args.model))
    read_roots = [log_root]
    if args.run_dir:
        read_roots.append(os.path.dirname(os.path.abspath(args.run_dir)))

    # Resume: skip recordings already analyzed at these settings. Two sources,
    # because neither alone is enough:
    #   * each file's own CSV carries the model/threshold/min-blob provenance,
    #     which survives losing the manifests entirely;
    #   * the manifests cover files that produced ZERO detections. Those write
    #     no prediction rows, so the CSV cannot prove they ran — and on a 24/7
    #     set the silent files are the overwhelming majority, so without this
    #     a resume re-does most of the work it was supposed to skip.
    # "Re-detect from scratch" means redo everything, so it disables resume.
    if not args.no_resume and not args.no_preserve_labels:
        manifest_done = completed_by_settings(read_roots, settings)
        todo, done = partition_done(wavs, settings, manifest_done)
        if done:
            print(f"  Resuming — {len(done)} file(s) already analyzed with "
                  f"these settings, {len(todo)} to go.", flush=True)
        wavs = todo
        if not wavs:
            print("Nothing to do — every file is already analyzed.", flush=True)
            return 0

    run_dir = args.run_dir or new_run_dir(log_root)
    manifest = RunManifest(run_dir).open()
    # The settings block comes from RunSettings so a later resume checks
    # exactly what this run recorded.
    info = dict(settings.to_info())
    info.update({
        'model_path': args.model,
        'n_files': len(wavs), 'device': cfg.device,
        'batch_size': cfg.batch_size, 'amp': cfg.amp,
        'preserve_labels': cfg.preserve_labels,
    })
    manifest.write_info(info)
    print(f"  Run log: {run_dir}", flush=True)

    def _on_done(summary: dict):
        # Grouped before the manifest records the file, so a run killed in
        # between redoes the file rather than leaving it ungrouped and marked
        # done.
        if 'error' not in summary and args.group_harmonics:
            _group_harmonics_for(summary.get('wav_path', ''))
        # Flushed per file, so a killed run resumes from exactly here.
        try:
            manifest.record(summary)
        except Exception:
            pass
        wav = os.path.basename(summary.get('wav_path', '?'))
        if 'error' in summary:
            print(f"  [FAIL] {wav}: {summary['error']}", file=sys.stderr, flush=True)
            return
        t = summary.get('timing', {})
        print(f"  [ok]   {wav}: {summary.get('n_blobs', 0)} detection(s) "
              f"in {t.get('t_total', '?')}s ({t.get('device', '?')})")

    try:
        results = run_inference_on_files(cfg=cfg, wav_paths=wavs,
                                         on_file_done=_on_done)
    finally:
        manifest.close()

    n_fail = sum(1 for r in results if 'error' in r)
    total = sum(r.get('n_blobs', 0) for r in results if 'error' not in r)
    print(f"Done — {total} detection(s) across {len(results) - n_fail} file(s)"
          + (f", {n_fail} failed" if n_fail else ""), flush=True)
    return 1 if n_fail else 0


# ----------------------------------------------------------------------
# train
# ----------------------------------------------------------------------
def add_training_args(p: argparse.ArgumentParser) -> None:
    """Every training setting, shared by ``mad train`` and the GUI's
    ``--train`` launch so the two cannot drift apart.

    Defaults match the GUI's training panel, so "train a new model" means the
    same thing whichever way it is started. A flag left at its default is
    reported by :func:`training_overrides` as *not given*, which is how a GUI
    launch knows to keep the value already in its panel.
    """
    g = p.add_argument_group('training')
    g.add_argument('--epochs', type=int, default=1000,
                   help="Maximum epochs (default: 1000; early stopping ends "
                        "the run well before).")
    g.add_argument('--patience', type=int, default=40,
                   help="Early-stop patience in epochs (default: 40; 0 "
                        "disables).")
    g.add_argument('--batch-size', type=int, default=16)
    g.add_argument('--lr', type=float, default=1e-3,
                   help="Starting learning rate (default: 1e-3).")
    g.add_argument('--lr-schedule', default='plateau',
                   choices=['plateau', 'none'],
                   help="'plateau' halves the rate when val loss stalls "
                        "(default); 'none' keeps it fixed.")
    g.add_argument('--lr-patience', type=int, default=8,
                   help="Epochs without improvement before the rate is "
                        "halved (default: 8).")
    g.add_argument('--arch', default='',
                   help="unet | unetpp | manet | hrnet (default: project's).")
    g.add_argument('--encoder', default='resnet18')
    g.add_argument('--device', default='auto',
                   choices=['auto', 'cuda', 'mps', 'cpu'])
    g.add_argument('--loss', default='bce_dice',
                   choices=['bce_dice', 'focal_tversky'],
                   help="Segmentation loss (default: bce_dice).")
    g.add_argument('--split-mode', default='call',
                   choices=['call', 'auto', 'file'],
                   help="How validation is held out: 'call' (default) holds "
                        "out individual calls, so validation shares "
                        "recordings with training and reads optimistic; "
                        "'file' holds out whole recordings -- the honest "
                        "measure of a recording the model has never seen.")
    g.add_argument('--split-seed', type=int, default=42,
                   help="Seed for the train/validation split (default: 42).")
    g.add_argument('--val-fraction', type=float, default=None,
                   help="Override the project's validation fraction.")
    g.add_argument('--no-augment', dest='augment', action='store_false',
                   help="Turn off all training augmentation.")
    g.add_argument('--no-faint-aug', dest='faint_aug', action='store_false',
                   help="Turn off faint-call (distance) augmentation.")
    g.add_argument('--faint-p', type=float, default=0.3,
                   help="Fraction of training tiles made fainter "
                        "(default: 0.3).")
    g.add_argument('--faint-db', type=float, nargs=2, default=(3.0, 18.0),
                   metavar=('MIN', 'MAX'),
                   help="Attenuation range in dB (default: 3 18).")
    g.add_argument('--max-per-recording', type=int, default=150,
                   help="Cap on examples drawn from any one recording, "
                        "labels and negatives counted separately (default: "
                        "150; 0 = no cap).")
    g.add_argument('--tile-placements', type=int, default=3,
                   help="Times each call is placed into a training tile "
                        "(default: 3).")
    g.add_argument('--run-name', default='',
                   help="Name for the model run dir (default: timestamped).")
    g.add_argument('--run-suffix', default='',
                   help="Tag appended to the timestamped name, keeping the "
                        "label count: <date>_<time>_unet_n=<labels>_<tag> "
                        "(e.g. ft_r7).")
    g.add_argument('--holdout-from', default=None, metavar='RUN',
                   help="Hold out the same recordings as an earlier run (its "
                        "folder, name, or training_summary.json), so the two "
                        "are scored on one validation set.")
    g.add_argument('--init-from', default='', metavar='WEIGHTS',
                   help="Fine-tune: start from this weights.pt (or its run "
                        "folder) instead of ImageNet. With --split-mode file "
                        "the parent's held-out recordings stay held out. "
                        "Pair with a lower --lr (e.g. 3e-4) and --patience.")
    g.add_argument('--context-ms', type=float, default=None, metavar='MS',
                   help="Context trained on each side of a call, in ms "
                        "(default: the project's context_ms; unset = each "
                        "example as saved, ~65 ms at 250 kHz / hop 128).")


#: argparse dest -> UNetTrainingConfig field, for every training flag.
_TRAIN_FIELD = {
    'epochs': 'n_epochs', 'patience': 'early_stop_patience',
    'batch_size': 'batch_size', 'lr': 'learning_rate',
    'lr_schedule': 'lr_schedule', 'lr_patience': 'lr_plateau_patience',
    'arch': 'model_arch', 'encoder': 'encoder_name', 'device': 'device',
    'loss': 'loss', 'split_mode': 'split_mode', 'split_seed': 'split_seed',
    'val_fraction': 'val_fraction', 'augment': 'augment',
    'faint_aug': 'augment_faint', 'faint_p': 'faint_p',
    'faint_db': 'faint_db_range',
    'max_per_recording': 'max_examples_per_recording',
    'tile_placements': 'tile_placements', 'run_name': 'run_name',
    'run_suffix': 'run_suffix',
    'holdout_from': 'holdout_groups',
    'init_from': 'init_from', 'context_ms': 'context_ms',
}


def resolve_holdout(spec: str, project_dir: str = "") -> List[str]:
    """The recordings an earlier run held out, from its folder, its name
    (under ``<project>/models``) or its ``training_summary.json``."""
    import json
    cands = [spec, os.path.join(spec, 'training_summary.json')]
    if project_dir:
        run = os.path.join(project_dir, 'models', spec)
        cands += [os.path.join(run, 'training_summary.json')]
    for c in cands:
        if os.path.isfile(c) and c.endswith('.json'):
            with open(c) as fh:
                groups = json.load(fh).get('val_groups') or []
            if not groups:
                raise ValueError(f"{c} records no held-out recordings")
            return [str(g) for g in groups]
    raise FileNotFoundError(f"No training_summary.json found for '{spec}'")


def suppress_training_defaults(p: argparse.ArgumentParser) -> None:
    """Make every training flag absent from the namespace unless given.

    For the GUI launch, which applies only what was asked for on top of its
    own panel. Comparing against the defaults instead would silently drop an
    explicit ``--split-mode call`` whenever the panel happened to say 'file'.
    """
    # On the actions themselves: set_defaults() would also seed the namespace
    # with the SUPPRESS sentinel string, which argparse then tries to convert
    # with each flag's type.
    for action in p._actions:
        if action.dest in _TRAIN_FIELD:
            action.default = argparse.SUPPRESS


def training_overrides(args: argparse.Namespace) -> dict:
    """``UNetTrainingConfig`` keyword arguments from parsed training flags.

    Only flags present on ``args`` are returned, so a parser built with
    :func:`suppress_training_defaults` yields just what was given.
    """
    out = {}
    for dest, field_name in _TRAIN_FIELD.items():
        if not hasattr(args, dest):
            continue
        val = getattr(args, dest)
        if dest == 'arch' and not val:
            continue
        if dest == 'val_fraction' and val is None:
            continue
        if dest == 'context_ms' and val is None:
            continue
        if dest == 'faint_db':
            val = (float(val[0]), float(val[1]))
        if dest == 'holdout_from':
            if not val:
                continue
            val = resolve_holdout(val, getattr(args, 'project', '') or '')
        out[field_name] = val
    return out


def _cmd_train(args: argparse.Namespace) -> int:
    from .mad_project import MADProjectConfig
    from .mad_training import UNetTrainingConfig, train_unet

    if not os.path.isdir(args.project):
        print(f"Project directory not found: {args.project}", file=sys.stderr, flush=True)
        return 2
    proj = MADProjectConfig.load(args.project)

    # Rebuild the consolidated store from the per-recording .mad sidecars
    # first. Without this the run trains on whatever the last GUI session left
    # behind — a headless run could silently fit a stale label set and still
    # report success, which is exactly what an unattended workflow cannot
    # tolerate.
    from .mad_examples import rebuild_training_store
    wavs = [e.path for e in proj.audio_entries()]
    try:
        n_lab = rebuild_training_store(proj.training_data_dir, wavs)
    except Exception as e:
        print(f"Could not rebuild the training store: {e}", file=sys.stderr, flush=True)
        print("Training was NOT started — the model must never be fitted on a "
              "partial label set.", file=sys.stderr, flush=True)
        return 1
    print(f"Training store rebuilt from {len(wavs)} recording(s): "
          f"{n_lab} example(s)", flush=True)

    kw = training_overrides(args)
    kw.setdefault('model_arch', proj.model_arch or "unet")
    kw.setdefault('val_fraction', proj.val_fraction)
    kw.setdefault('context_ms', getattr(proj, 'context_ms', None))
    cfg = UNetTrainingConfig(
        project_dir=proj.project_dir,
        nperseg=proj.nperseg, noverlap=proj.noverlap, nfft=proj.nfft,
        db_min=proj.db_min, db_max=proj.db_max,
        db_norm=getattr(proj, 'db_norm', 'fixed') or 'fixed',
        training_data_dir=proj.training_data_dir,
        **kw,
    )

    def _progress(epoch: int, n_epochs: int, info: dict):
        status = info.get('status', '')
        if status == 'collecting_tiles':
            fn = info.get('file_name')
            if fn:
                print(f"  collecting tiles: {info.get('file_i', '?')}/"
                      f"{info.get('file_n', '?')} {fn}      ", end='\r', flush=True)
            return
        if status == 'device':
            print(f"  device: {info.get('device', '?')}", flush=True)
        elif status == 'split':
            level = info.get('split_level', '?')
            print(f"  split: {level}-level — "
                  f"{info.get('n_val_groups', 0)}/{info.get('n_groups', 0)} "
                  f"group(s) held out"
                  f"{' (pinned)' if info.get('pinned') else ''}, "
                  f"{info.get('n_train_tiles', 0)} train / "
                  f"{info.get('n_val_tiles', 0)} val tiles", flush=True)
            miss = info.get('holdout_missing') or []
            if miss:
                print(f"  pinned hold-out: {len(miss)} recording(s) have no "
                      f"examples now: {', '.join(miss[:5])}", flush=True)
            if info.get('inherited'):
                print("  (held-out recordings inherited from the parent model)",
                      flush=True)
            if not info.get('val_held_out', True):
                print("  WARNING: validation is not held out at the recording "
                      "level — val scores will flatter the model.",
                      file=sys.stderr, flush=True)
        elif status == 'training':
            tl, vl = info.get('train_loss'), info.get('val_loss')
            msg = f"  epoch {epoch}/{n_epochs}"
            if tl is not None:
                msg += f"  train_loss={tl:.4f}"
            if vl is not None:
                msg += f"  val_loss={vl:.4f}"
            vd = info.get('val_dice')
            if vd is not None:
                msg += f"  val_dice={vd:.3f}"
            lr = info.get('lr')
            if lr is not None:
                msg += f"  lr={lr:.1e}"
            print(msg, flush=True)
        elif status == 'capped':
            for rec, kinds in sorted((info.get('capped') or {}).items()):
                parts = ", ".join(f"{k} {kept}/{tot}"
                                  for k, (kept, tot) in sorted(kinds.items()))
                print(f"  capped {rec}: {parts}", flush=True)
        elif status == 'early_stop':
            print(f"  early stop at epoch {epoch}", flush=True)
        # 'batch' / 'epoch_preview' / 'done' are intentionally not printed.

    print(f"Training {cfg.model_arch} on project '{proj.project_name}' "
          f"(max {cfg.n_epochs} epochs, patience {cfg.early_stop_patience}, "
          f"encoder={cfg.encoder_name}, split={cfg.split_mode}, "
          f"lr={cfg.learning_rate:g} {cfg.lr_schedule}, augment="
          f"{'on' if cfg.augment else 'off'}"
          f"{'+faint' if cfg.augment and cfg.augment_faint else ''}, "
          f"cap={cfg.max_examples_per_recording or 'off'})", flush=True)
    if cfg.init_from:
        print(f"  fine-tuning from {cfg.init_from}", flush=True)
    if cfg.context_ms is not None:
        print(f"  context: {cfg.context_ms:g} ms each side of a call", flush=True)
    try:
        summary = train_unet(cfg, progress=_progress)
    except RuntimeError as e:
        print(f"Training failed: {e}", file=sys.stderr, flush=True)
        return 1
    dice = summary.get('best_val_dice')
    dice_str = f"{dice:.3f}" if isinstance(dice, (int, float)) else "?"
    print(f"Done — model saved to {summary.get('model_path', '?')}", flush=True)
    print(f"  val_dice={dice_str} "
          f"({summary.get('split_level', '?')}-level split"
          f"{'' if summary.get('val_held_out') else ', NOT held out'})")
    return 0


# ----------------------------------------------------------------------
# embeddings
# ----------------------------------------------------------------------
def _cmd_embeddings(args: argparse.Namespace) -> int:
    from .mad_inference import (
        MADInferenceConfig, load_model, embed_file, write_embeddings_npz)

    wavs = _expand_inputs(args.input)
    if not wavs:
        print("No .wav files found in the given input(s).", file=sys.stderr, flush=True)
        return 2
    cfg = MADInferenceConfig(model_path=args.model, device=args.device)
    model, ckpt, device = load_model(cfg.model_path, cfg.device)
    print(f"Embedding detections from {len(wavs)} file(s) on {device}", flush=True)

    results = []
    for wav in wavs:
        try:
            res = embed_file(wav, cfg, model=model, ckpt=ckpt, device=device)
        except RuntimeError as e:
            print(f"  [skip] {os.path.basename(wav)}: {e}", file=sys.stderr, flush=True)
            continue
        n = res['embeddings'].shape[0]
        print(f"  [ok]   {os.path.basename(wav)}: {n} detection(s)", flush=True)
        results.append(res)

    n = write_embeddings_npz(args.out, results)
    print(f"Wrote {n} embedding(s) → {args.out}", flush=True)
    return 0


# ----------------------------------------------------------------------
# Parser
# ----------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog='mad',
        description="Mask Audio Detector — headless train / analyze / "
                    "embeddings pipeline.")
    sub = p.add_subparsers(dest='command', required=True)

    # analyze
    pa = sub.add_parser('analyze', help="Run a trained model over wav files.")
    pa.add_argument('--model', required=True,
                    help="Path to weights.pt, or the model run folder.")
    pa.add_argument('--input', nargs='+', default=[],
                    help="Wav files and/or folders to analyze.")
    pa.add_argument('--project', default='',
                    help="Analyze every recording registered in this MAD "
                         "project (with or instead of --input). Also sets the "
                         "defaults the GUI would use: runs log to the "
                         "project's batch_runs, and its labels are shielded.")
    pa.add_argument('--no-group-harmonics', dest='group_harmonics',
                    action='store_false', default=True,
                    help="Skip grouping each recording's harmonics into calls "
                         "after its detections are written (on by default, "
                         "matching the GUI).")
    pa.add_argument('--threshold', type=float, default=0.5,
                    help="Probability threshold (default 0.5).")
    pa.add_argument('--min-blob-pixels', type=int, default=8,
                    help="Drop blobs smaller than this (default 8).")
    pa.add_argument('--device', default='auto',
                    choices=['auto', 'cuda', 'mps', 'cpu'])
    pa.add_argument('--no-merge-consecutive', dest='merge_consecutive',
                    action='store_false',
                    help="Do NOT merge fragments of one call. Merging is on "
                         "by default, matching the GUI.")
    pa.add_argument('--merge-consecutive', dest='merge_consecutive',
                    action='store_true', default=True,
                    help="Merge consecutive blobs of one call into a single "
                         "detection.")
    pa.add_argument('--merge-gap-s', type=float, default=0.01,
                    help="Max time gap (s) bridged when merging (default "
                         "0.01).")
    pa.add_argument('--merge-ignore-freq', action='store_true',
                    help="When merging, don't require frequency overlap.")
    pa.add_argument('--batch-size', type=int, default=8,
                    help="Tiles per forward pass (default 8). Higher is faster "
                         "on a GPU with spare VRAM; does not change results.")
    pa.add_argument('--no-amp', action='store_true',
                    help="Disable fp16 mixed precision (CUDA only).")
    pa.add_argument('--no-prefetch', action='store_true',
                    help="Build each spectrogram only when its file "
                         "comes up, instead of during the previous "
                         "file's scan (slower; same output).")
    pa.add_argument('--cpu-spectrogram', action='store_true',
                    help="Build spectrograms on the CPU even on a CUDA "
                         "GPU (the GPU one matches it to one grey "
                         "level and is ~12x faster).")
    pa.add_argument('--no-resume', action='store_true',
                    help="Re-analyze files that already carry detections from "
                         "this model at these settings (default is to skip "
                         "them, so an interrupted run resumes). Implied by "
                         "--no-preserve-labels.")
    pa.add_argument('--run-dir', default='',
                    help="Directory for this run's manifest.jsonl (default: a "
                         "timestamped folder under --log-root).")
    pa.add_argument('--log-root', default='',
                    help="Where timestamped run folders are created (default: "
                         "beside the model).")
    pa.add_argument('--training-data-dir', default='',
                    help="Example store, to preserve confirmed labels.")
    pa.add_argument('--no-preserve-labels', action='store_true',
                    help="Re-detect from scratch (ignore prior decisions).")
    _add_sampling_args(pa)
    pa.set_defaults(func=_cmd_analyze)

    # train
    pt = sub.add_parser('train', help="Train a model from a MAD project.")
    pt.add_argument('--project', required=True, help="MAD project directory.")
    add_training_args(pt)
    pt.set_defaults(func=_cmd_train)

    # embeddings
    pe = sub.add_parser('embeddings',
                        help="Export per-detection encoder embeddings.")
    pe.add_argument('--model', required=True, help="Path to weights.pt.")
    pe.add_argument('--input', required=True, nargs='+',
                    help="Wav files/folders (must already have detections).")
    pe.add_argument('--out', default='mad_embeddings.npz',
                    help="Output .npz (default mad_embeddings.npz).")
    pe.add_argument('--device', default='auto',
                    choices=['auto', 'cuda', 'mps', 'cpu'])
    pe.set_defaults(func=_cmd_embeddings)

    # add-specs
    ps = sub.add_parser('add-specs',
                        help="Add the spectrogram under each stored detection "
                             "to .mad files made before MAD saved it.")
    ps.add_argument('--input', required=True, nargs='+',
                    help="Wav files/folders whose .mad files hold detections.")
    ps.add_argument('--db-min', type=float, default=-100.0,
                    help="The model's dB floor (default -100).")
    ps.add_argument('--db-max', type=float, default=-20.0,
                    help="The model's dB ceiling (default -20).")
    ps.add_argument('--db-norm', default='fixed',
                    help="The model's dB normalisation (default fixed).")
    ps.add_argument('--overwrite', action='store_true',
                    help="Recompute crops that already have one.")
    ps.set_defaults(func=_cmd_add_specs)

    # add-ridge
    pr = sub.add_parser('add-ridge',
                        help="Add the brightness ridge (turns, rate, depth, trill) of each "
                             "stored detection and labelled call; fills missing detection "
                             "spectrograms from the audio first.")
    pr.add_argument('--input', required=True, nargs='+',
                    help="Wav files/folders whose .mad files hold detections or labels, "
                         "and/or training stores (training_data.h5) to fill directly.")
    pr.add_argument('--db-min', type=float, default=-100.0,
                    help="The model's dB floor, for missing spectrograms (default -100).")
    pr.add_argument('--db-max', type=float, default=-20.0,
                    help="The model's dB ceiling, for missing spectrograms (default -20).")
    pr.add_argument('--db-norm', default='fixed',
                    help="The model's dB normalisation (default fixed).")
    pr.add_argument('--overwrite', action='store_true',
                    help="Recompute ridges that are already stored.")
    pr.set_defaults(func=_cmd_add_ridge)

    return p


def _cmd_add_specs(args) -> int:
    """Backfill /pred_specs for every given recording with stored detections."""
    from .fnt_mask_store import masks_sibling_path, has_pred_masks
    from .mad_inference import backfill_pred_specs
    wavs = [w for w in _expand_inputs(args.input)
            if has_pred_masks(masks_sibling_path(w))]
    n_files = n_crops = 0
    for i, w in enumerate(wavs, 1):
        try:
            k = backfill_pred_specs(w, db_min=args.db_min, db_max=args.db_max,
                                    db_norm=args.db_norm,
                                    overwrite=args.overwrite)
        except Exception as e:
            print(f"[{i}/{len(wavs)}] {os.path.basename(w)}: failed ({e})",
                  flush=True)
            continue
        n_files += bool(k)
        n_crops += k
        print(f"[{i}/{len(wavs)}] {os.path.basename(w)}: {k} detection(s)",
              flush=True)
    print(f"added spectrograms to {n_crops} detection(s) in {n_files} file(s)")
    return 0


def _cmd_add_ridge(args) -> int:
    """Store the brightness ridge (mad_ridge) of every detection and every labelled
    call in the given recordings' .mad files, and of every example in any training
    store (.h5) given directly. Detections recorded before MAD kept their spectrogram
    get it from the audio first (as add-specs). Never run on files MAD is writing."""
    from .fnt_mask_store import (add_example_ridges, add_pred_ridges, has_pred_masks,
                                 masks_sibling_path, read_all_pred_masks, td_count)
    from .mad_inference import backfill_pred_specs
    stores = [p for p in args.input if os.path.isfile(p)
              and p.lower().endswith(('.h5', '.mad'))]
    wavs = [w for w in _expand_inputs([p for p in args.input if p not in stores])
            if os.path.isfile(masks_sibling_path(w))]
    n_files = n_calls = n_labels = 0
    for i, w in enumerate(wavs, 1):
        h5 = masks_sibling_path(w)
        k = e = 0
        try:
            if has_pred_masks(h5):
                crops = read_all_pred_masks(h5, include_spec=True)
                if any('spec' not in c for c in crops.values()):
                    backfill_pred_specs(w, db_min=args.db_min, db_max=args.db_max,
                                        db_norm=args.db_norm)
                k = add_pred_ridges(h5, overwrite=args.overwrite)
            if td_count(h5):
                e = add_example_ridges(h5, overwrite=args.overwrite)
        except Exception as ex:
            print(f"[{i}/{len(wavs)}] {os.path.basename(w)}: failed ({ex})", flush=True)
            continue
        n_files += bool(k or e)
        n_calls += k
        n_labels += e
        print(f"[{i}/{len(wavs)}] {os.path.basename(w)}: {k} detection(s), "
              f"{e} label(s)", flush=True)
    for s in stores:
        try:
            k = add_pred_ridges(s, overwrite=args.overwrite)
            e = add_example_ridges(s, overwrite=args.overwrite)
        except Exception as ex:
            print(f"{os.path.basename(s)}: failed ({ex})", flush=True)
            continue
        n_files += bool(k or e)
        n_calls += k
        n_labels += e
        print(f"{os.path.basename(s)}: {k} detection(s), {e} label(s)", flush=True)
    print(f"added ridges to {n_calls} detection(s) and {n_labels} label(s) "
          f"in {n_files} file(s)")
    return 0


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    # Headless runs have no window, so Windows treats them as background work
    # and throttles them: 47 s per recording instead of 27 (see power.py).
    from .power import keep_full_speed
    keep_full_speed()
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
