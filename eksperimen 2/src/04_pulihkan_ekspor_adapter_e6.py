from __future__ import annotations
import argparse
import datetime as dt
import json
from pathlib import Path
import sys


def need(cond, reason):
    if not cond:
        raise RuntimeError(reason)


def valid_adapter(directory: Path):
    if not directory.is_dir():
        return False, 'directory absent'
    cfg = directory / 'adapter_config.json'
    weights = directory / 'adapter_model.safetensors'
    if not cfg.is_file() or not weights.is_file():
        return False, 'adapter config or safetensors absent'
    if not weights.stat().st_size:
        return False, 'empty safetensors'
    try:
        settings = json.loads(cfg.read_text(encoding='utf-8'))
        if settings.get('r') != 8 or settings.get('lora_alpha') != 16:
            return False, 'LoRA settings differ from locked protocol'
        from safetensors import safe_open
        param_count = 0
        with safe_open(str(weights), framework='pt', device='cpu') as reader:
            keys = list(reader.keys())
            for key in keys:
                need('lora_' in key, f'Unexpected adapter key: {key}')
                shape = reader.get_slice(key).get_shape()
                size = 1
                for dimension in shape:
                    size *= dimension
                param_count += size
        if param_count != 98304:
            return False, f'Unexpected LoRA tensor parameter count: {param_count}'
        return True, f'valid adapter ({len(keys)} tensors; {param_count} params)'
    except Exception as exc:
        return False, f'invalid adapter: {type(exc).__name__}: {exc}'


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--fold', type=int, required=True, choices=[1, 2, 3, 4, 5])
    ap.add_argument('--variant', required=True, choices=['raw', 'clahe'])
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument('--check', action='store_true')
    mode.add_argument('--repair', action='store_true')
    args = ap.parse_args()

    root = Path(__file__).resolve().parents[1]
    out = root / 'output' / 'training' / f'lora_{args.variant}_fold_{args.fold}'
    target = out / 'best_adapter'
    stage = out / 'best_adapter_staging'
    backup = out / 'best_adapter_backup'
    progress = out / 'latest_training_state.pt'
    summary = out / 'run_summary.json'

    print(f'LO-RA EXPORT RECOVERY | {out}', flush=True)
    need(out.is_dir(), 'Output fold directory not found')
    need(progress.is_file(), 'No latest_training_state.pt: cannot verify resume safety')
    need(not summary.exists(), 'Fold already COMPLETE; refusing to alter best adapter')

    print(f'checkpoint: {progress.name}, {progress.stat().st_size:,} bytes')
    need(progress.stat().st_size > 0, 'Latest checkpoint empty')
    for p in (target, stage, backup):
        ok, detail = valid_adapter(p)
        print(f'{p.name:<22}: {"present" if p.exists() else "absent":<8} | {detail}')

    present = {p.name for p in (target, stage, backup) if p.exists()}
    need(present, 'No adapter directory exists; nothing to recover')
    need(stage.exists() or backup.exists(), 'No stale staging/backup exists; export error cause differs')

    complete_best_artifacts = all((out / name).is_file() for name in (
        'best_classifier.pt', 'best_validation_metrics.json', 'best_validation_predictions.csv'
    ))
    need(complete_best_artifacts,
         'Best classifier/metrics/predictions incomplete; cannot guarantee adapter pairing. '
         'Do NOT delete/rename the folders. Provide a folder listing for manual recovery.')

    target_ok, target_msg = valid_adapter(target)
    backup_ok, backup_msg = valid_adapter(backup)
    if backup.exists():
        need(backup_ok, 'best_adapter_backup invalid; cannot restore trustworthy previous best: '+backup_msg)
        # The backup predates export of new adapter, and is the only one consistent
        # with previously saved best_classifier.pt (written AFTER adapter export).
        strategy = 'RESTORE_PREVIOUS_BEST_FROM_BACKUP'
    else:
        need(target_ok, 'best_adapter missing/invalid and no good backup available: '+target_msg)
        strategy = 'KEEP_PREVIOUS_BEST_ARCHIVE_STAGING'
    print(f'SAFE PLAN: {strategy}')

    if args.check:
        print('DRY CHECK PASS | no changes performed. Use --repair to apply the safe plan.')
        return

    timestamp = dt.datetime.now().strftime('%Y%m%d_%H%M%S_%f')
    archive = out / f'_adapter_export_recovery_{timestamp}'
    need(not archive.exists(), 'Recovery archive collision')
    archive.mkdir()
    if stage.exists():
        stage.rename(archive / 'best_adapter_staging_preserved')
        print('PRESERVED: stale staging directory')
    if backup.exists():
        if target.exists():
            target.rename(archive / 'best_adapter_candidate_preserved')
            print('PRESERVED: current candidate best_adapter (may be newer but unmatched to head)')
        backup.rename(target)
        print('RESTORED: previous best adapter from backup')
    ok, msg = valid_adapter(target)
    need(ok, 'Recovery did not produce valid adapter: '+msg)
    need(not stage.exists() and not backup.exists(), 'Stale folders remain')
    print(f'PASS_E6_LORA_EXPORT_RECOVERY | archive={archive.name}')
    print('NEXT: python 04_train_lora_5fold_e6.py '
          f'--fold {args.fold} --variant {args.variant} --resume')
    print('NOTE: last interrupted epoch repeats from latest_training_state.pt.')


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        print(f'RECOVERY BLOCKED: {type(exc).__name__}: {exc}', file=sys.stderr)
        sys.exit(1)
