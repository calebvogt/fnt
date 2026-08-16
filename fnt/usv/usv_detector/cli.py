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


# ----------------------------------------------------------------------
# analyze
# ----------------------------------------------------------------------
def _cmd_analyze(args: argparse.Namespace) -> int:
    from .mad_inference import MADInferenceConfig, run_inference_on_files

    wavs = _expand_inputs(args.input)
    if not wavs:
        print("No .wav files found in the given input(s).", file=sys.stderr)
        return 2
    print(f"Analyzing {len(wavs)} file(s) with {os.path.basename(args.model)}")

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

    # Resume: skip recordings this model already analyzed at these settings.
    # Provenance lives in each file's own CSV, so this works across runs and
    # survives losing the manifest.
    from .mad_batch import RunManifest, RunSettings, new_run_dir, partition_done
    manifest = None
    if not args.no_resume:
        settings = RunSettings.from_config(cfg)
        todo, done = partition_done(wavs, settings)
        if done:
            print(f"  Resuming — {len(done)} file(s) already analyzed with "
                  f"these settings, {len(todo)} to go.")
        wavs = todo
        if not wavs:
            print("Nothing to do — every file is already analyzed.")
            return 0

    run_dir = args.run_dir or new_run_dir(
        args.log_root or os.path.dirname(os.path.abspath(args.model)))
    manifest = RunManifest(run_dir).open()
    manifest.write_info({
        'model_path': args.model, 'model_name': os.path.splitext(
            os.path.basename(args.model))[0],
        'threshold': cfg.threshold, 'min_blob_pixels': cfg.min_blob_pixels,
        'n_files': len(wavs), 'device': cfg.device,
        'batch_size': cfg.batch_size, 'amp': cfg.amp,
    })
    print(f"  Run log: {run_dir}")

    def _on_done(summary: dict):
        # Flushed per file, so a killed run resumes from exactly here.
        try:
            manifest.record(summary)
        except Exception:
            pass
        wav = os.path.basename(summary.get('wav_path', '?'))
        if 'error' in summary:
            print(f"  [FAIL] {wav}: {summary['error']}", file=sys.stderr)
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
          + (f", {n_fail} failed" if n_fail else ""))
    return 1 if n_fail else 0


# ----------------------------------------------------------------------
# train
# ----------------------------------------------------------------------
def _cmd_train(args: argparse.Namespace) -> int:
    from .mad_project import MADProjectConfig
    from .mad_training import UNetTrainingConfig, train_unet

    if not os.path.isdir(args.project):
        print(f"Project directory not found: {args.project}", file=sys.stderr)
        return 2
    proj = MADProjectConfig.load(args.project)

    cfg = UNetTrainingConfig(
        project_dir=proj.project_dir,
        run_name=args.run_name or "",
        model_arch=args.arch or proj.model_arch or "unet",
        encoder_name=args.encoder,
        n_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        val_fraction=proj.val_fraction,
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
                      f"{info.get('file_n', '?')} {fn}      ", end='\r')
            return
        if status == 'device':
            print(f"  device: {info.get('device', '?')}")
        elif status == 'split':
            level = info.get('split_level', '?')
            print(f"  split: {level}-level — "
                  f"{info.get('n_val_groups', 0)}/{info.get('n_groups', 0)} "
                  f"group(s) held out, "
                  f"{info.get('n_train_tiles', 0)} train / "
                  f"{info.get('n_val_tiles', 0)} val tiles")
            if not info.get('val_held_out', True):
                print("  WARNING: validation is not held out at the recording "
                      "level — val scores will flatter the model.",
                      file=sys.stderr)
        elif status == 'training':
            tl, vl = info.get('train_loss'), info.get('val_loss')
            msg = f"  epoch {epoch}/{n_epochs}"
            if tl is not None:
                msg += f"  train_loss={tl:.4f}"
            if vl is not None:
                msg += f"  val_loss={vl:.4f}"
            print(msg)
        elif status == 'early_stop':
            print(f"  early stop at epoch {epoch}")
        # 'batch' / 'epoch_preview' / 'done' are intentionally not printed.

    print(f"Training {cfg.model_arch} on project '{proj.project_name}' "
          f"({cfg.n_epochs} epochs, encoder={cfg.encoder_name})")
    try:
        summary = train_unet(cfg, progress=_progress)
    except RuntimeError as e:
        print(f"Training failed: {e}", file=sys.stderr)
        return 1
    dice = summary.get('best_val_dice')
    dice_str = f"{dice:.3f}" if isinstance(dice, (int, float)) else "?"
    print(f"Done — model saved to {summary.get('model_path', '?')}")
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
        print("No .wav files found in the given input(s).", file=sys.stderr)
        return 2
    cfg = MADInferenceConfig(model_path=args.model, device=args.device)
    model, ckpt, device = load_model(cfg.model_path, cfg.device)
    print(f"Embedding detections from {len(wavs)} file(s) on {device}")

    results = []
    for wav in wavs:
        try:
            res = embed_file(wav, cfg, model=model, ckpt=ckpt, device=device)
        except RuntimeError as e:
            print(f"  [skip] {os.path.basename(wav)}: {e}", file=sys.stderr)
            continue
        n = res['embeddings'].shape[0]
        print(f"  [ok]   {os.path.basename(wav)}: {n} detection(s)")
        results.append(res)

    n = write_embeddings_npz(args.out, results)
    print(f"Wrote {n} embedding(s) → {args.out}")
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
    pa.add_argument('--merge-consecutive', action='store_true',
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
                         "them, so an interrupted run resumes).")
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
