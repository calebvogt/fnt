"""Command-line interface for the Mask Audio Detector (MAD).

Exposes MAD's train / analyze / embeddings pipeline as a headless CLI so it can
be scripted into batch and HPC workflows without the GUI. Installed as the
``mad`` console script (see ``pyproject.toml``)::

    mad analyze    --model weights.pt --input recordings/
    mad train      --project my_project/ --epochs 30
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
def _cmd_analyze(args: argparse.Namespace) -> int:
    from .mad_inference import MADInferenceConfig, run_inference_on_files

    wavs = _expand_inputs(args.input)
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

    cfg = UNetTrainingConfig(
        project_dir=proj.project_dir,
        run_name=args.run_name or "",
        model_arch=args.arch or proj.model_arch or "unet",
        encoder_name=args.encoder,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        val_fraction=(args.val_fraction if args.val_fraction is not None
                      else proj.val_fraction),
        loss=args.loss,
        split_mode=args.split_mode,
        early_stop_patience=args.patience,
        nperseg=proj.nperseg, noverlap=proj.noverlap, nfft=proj.nfft,
        db_min=proj.db_min, db_max=proj.db_max,
        training_data_dir=proj.training_data_dir,
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
                  f"group(s) held out, "
                  f"{info.get('n_train_tiles', 0)} train / "
                  f"{info.get('n_val_tiles', 0)} val tiles", flush=True)
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
            print(msg, flush=True)
        elif status == 'early_stop':
            print(f"  early stop at epoch {epoch}", flush=True)
        # 'batch' / 'epoch_preview' / 'done' are intentionally not printed.

    print(f"Training {cfg.model_arch} on project '{proj.project_name}' "
          f"({cfg.n_epochs} epochs, encoder={cfg.encoder_name})", flush=True)
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
    pa.add_argument('--model', required=True, help="Path to weights.pt.")
    pa.add_argument('--input', required=True, nargs='+',
                    help="Wav files and/or folders to analyze.")
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
    pt.add_argument('--epochs', type=int, default=30)
    pt.add_argument('--arch', default='',
                    help="unet | unetpp | manet | hrnet (default: project's).")
    pt.add_argument('--encoder', default='resnet18')
    pt.add_argument('--batch-size', type=int, default=8)
    pt.add_argument('--lr', type=float, default=1e-3)
    pt.add_argument('--device', default='auto',
                    choices=['auto', 'cuda', 'mps', 'cpu'])
    pt.add_argument('--loss', default='bce_dice',
                    choices=['bce_dice', 'focal_tversky'],
                    help="Segmentation loss (default: bce_dice).")
    pt.add_argument('--split-mode', default='call',
                    choices=['call', 'auto', 'file'],
                    help="How validation is held out (default: call).")
    pt.add_argument('--val-fraction', type=float, default=None,
                    help="Override the project's validation fraction.")
    pt.add_argument('--patience', type=int, default=10,
                    help="Early-stop patience in epochs (0 disables).")
    pt.add_argument('--run-name', default='',
                    help="Name for the model run dir (default: timestamped).")
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

    return p


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == '__main__':
    raise SystemExit(main())
