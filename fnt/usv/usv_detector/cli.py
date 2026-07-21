"""Command-line interface for the Mask Audio Detector (MAD).

Exposes MAD's train / analyze / embeddings pipeline as a headless CLI so it can
be scripted into batch and HPC workflows without the GUI. Installed as the
``mad`` console script (see ``pyproject.toml``)::

    mad analyze    --model weights.pt --input recordings/ --export raven
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


def _expand_inputs(inputs: List[str]) -> List[str]:
    """Expand a mix of .wav files and folders into a sorted, de-duped wav list.
    Folders are scanned non-recursively for ``*.wav`` (case-insensitive)."""
    out: List[str] = []
    seen = set()
    for item in inputs:
        if os.path.isdir(item):
            hits = (glob.glob(os.path.join(item, '*.wav'))
                    + glob.glob(os.path.join(item, '*.WAV')))
            paths = sorted(hits)
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
    from .mad_inference import (
        MADInferenceConfig, run_inference_on_files, read_blob_csv,
        write_raven_selection_table, write_audacity_labels)

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
    )

    def _on_done(summary: dict):
        wav = os.path.basename(summary.get('wav_path', '?'))
        if 'error' in summary:
            print(f"  [FAIL] {wav}: {summary['error']}", file=sys.stderr)
            return
        t = summary.get('timing', {})
        print(f"  [ok]   {wav}: {summary.get('n_blobs', 0)} detection(s) "
              f"in {t.get('t_total', '?')}s ({t.get('device', '?')})")

    results = run_inference_on_files(cfg=cfg, wav_paths=wavs,
                                     on_file_done=_on_done)

    n_export = 0
    if args.export:
        for res in results:
            csv_path = res.get('csv_path')
            if not csv_path or not os.path.isfile(csv_path):
                continue
            rows = read_blob_csv(csv_path)
            stem = os.path.splitext(res['wav_path'])[0]
            if args.export == 'raven':
                write_raven_selection_table(
                    f"{stem}.Table.1.selections.txt", rows)
            else:
                write_audacity_labels(f"{stem}.labels.txt", rows)
            n_export += 1
        print(f"Exported {n_export} file(s) to {args.export} format.")

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
    print(f"Done — model saved to {summary.get('model_path', '?')}")
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
    pa.add_argument('--export', choices=['raven', 'audacity'],
                    help="Also write an interchange table per file.")
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
